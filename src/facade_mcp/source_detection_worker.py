"""Local detection entry points from FacadeFixer, for an isolated worker.

Reuse the standalone source adapters' loaders without importing the legacy
backend singleton. Decode stored pixels once and preserve every checkpoint
label, including labels outside the downstream mask taxonomy.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
from unittest.mock import patch


WEIGHTS = {
    "yolo12m": "model_weights/detection/yolo12m_building.pt",
    "yolo11m": "model_weights/detection/yolo11m_building.pt",
    "rtdetr": "model_weights/rtdetr-l-bst.pt",
    "faster_rcnn": "model_weights/detection/Faster_R-CNN.pth",
}
PRIMARY_PROFILES = ("yolo12m", "yolo11m", "rtdetr")
FRCNN_CLASSES = (
    "Concrete_Crack", "Concrete_Delamination", "Concrete_Spalling",
    "Rust_Stain", "Vegeterian", "Degraded_Plaster", "Craquelure",
    "Tile_Crack", "Tile_spalling", "Water_Stain", "Bulging", "Contaminants",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_file(repo: Path, relative_path: str) -> Path:
    path = (repo / relative_path).resolve()
    if not path.is_relative_to(repo) or not path.is_file():
        raise FileNotFoundError("Required local model/source file is unavailable.")
    return path


@contextmanager
def _offline():
    """The calling process is isolated; local models may never fetch assets."""
    environment = {
        "YOLO_AUTOINSTALL": "false", "YOLO_OFFLINE": "true",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "WANDB_DISABLED": "true", "COMET_MODE": "DISABLED",
    }

    def denied(*args, **kwargs):
        raise RuntimeError("Network access is disabled in local detection workers.")

    with patch.dict(os.environ, environment), patch.object(socket.socket, "connect", denied), \
            patch.object(socket.socket, "connect_ex", denied), \
            patch.object(socket, "create_connection", denied):
        yield


def _source_adapter(repo: Path, profile: str, checkpoint: Path):
    kind = "rtdetr" if profile == "rtdetr" else "yolo"
    source = _local_file(repo, f"model_runner/adapters/detection/{kind}.py")
    module_name = f"model_runner.adapters.detection.{kind}"
    # Every invocation normally has a fresh process. Refuse a different source
    # tree accidentally cached by a caller that reuses the interpreter.
    if module_name in sys.modules:
        loaded = Path(sys.modules[module_name].__file__).resolve()
        if loaded != source:
            raise RuntimeError("A different FacadeFixer source tree is already imported.")
    sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module(module_name)
        if Path(module.__file__).resolve() != source:
            raise RuntimeError("Source adapter import did not resolve to the configured repository.")
        adapter = module.RTDetrDetectionAdapter if kind == "rtdetr" else module.YoloDetectionAdapter
        return adapter(checkpoint)._model(), source
    finally:
        sys.path.remove(str(repo))


def _finite(value, name: str) -> float:
    value = float(value.item() if hasattr(value, "item") else value)
    if not math.isfinite(value):
        raise ValueError(f"Detector returned a non-finite {name}.")
    return value


def _prediction(label, class_id, score, coordinates, width, height, model):
    confidence = _finite(score, "confidence")
    if not 0 <= confidence <= 1:
        raise ValueError("Detector confidence lies outside [0, 1].")
    values = coordinates.tolist() if hasattr(coordinates, "tolist") else list(coordinates)
    if len(values) != 4:
        raise ValueError("Detector bbox must have four coordinates.")
    values = [max(0., min(_finite(v, "bbox coordinate"), edge))
              for v, edge in zip(values, (width, height, width, height))]
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return {"class_name": str(label), "class_id": int(class_id),
            "bbox_xyxy": values, "confidence": confidence, "source_model": model}


def _run_single(repo, profile, raster_bgr, threshold, device):
    height, width = raster_bgr.shape[:2]
    checkpoint = _local_file(repo, WEIGHTS[profile])
    checkpoint_sha256 = _sha256(checkpoint)
    source_files = {}
    if profile == "faster_rcnn":
        # Fixed local counterpart of backend/detection_agent.py's
        # _ensure_faster_rcnn_model / _run_faster_rcnn_inference. Importing that
        # module would eagerly load unrelated models and provider clients.
        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
        except ImportError as exc:
            raise RuntimeError("Faster R-CNN requires detectron2 in the configured model runtime.") from exc
        config = get_cfg()
        config.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        config.MODEL.ROI_HEADS.NUM_CLASSES = len(FRCNN_CLASSES)
        config.MODEL.WEIGHTS = str(checkpoint)
        config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        config.MODEL.ROI_HEADS.NMS_THRESH_TEST = .5
        config.TEST.DETECTIONS_PER_IMAGE = 300
        config.MODEL.DEVICE = "cuda:" + device if device.isdigit() else device
        instances = DefaultPredictor(config)(raster_bgr)["instances"].to("cpu")
        entries = zip(instances.pred_boxes.tensor, instances.scores, instances.pred_classes)
        predictions = []
        for coordinates, score, class_id in entries:
            class_id = int(class_id.item())
            label = FRCNN_CLASSES[class_id] if 0 <= class_id < len(FRCNN_CLASSES) else f"unmapped_model_class_{class_id}"
            predictions.append(_prediction(label, class_id, score, coordinates, width, height, profile))
        source_path = _local_file(repo, "backend/detection_agent.py")
        source_files[source_path.relative_to(repo).as_posix()] = _sha256(source_path)
        library = "detectron2"
    else:
        model, source_path = _source_adapter(repo, profile, checkpoint)
        results = model.predict(source=raster_bgr, conf=threshold, iou=.5,
                                max_det=300, device=device, verbose=False,
                                save=False, save_txt=False, save_crop=False)
        predictions = []
        for result in results:
            if result.boxes is None:
                continue
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = names.get(class_id, f"unmapped_model_class_{class_id}") if isinstance(names, dict) else (
                    names[class_id] if 0 <= class_id < len(names) else f"unmapped_model_class_{class_id}")
                predictions.append(_prediction(label, class_id, box.conf[0], box.xyxy[0], width, height, profile))
        source_files[source_path.relative_to(repo).as_posix()] = _sha256(source_path)
        library = "ultralytics"
    from importlib.metadata import PackageNotFoundError, version
    try:
        library_version = version(library)
    except PackageNotFoundError:
        library_version = "unknown"
    if _sha256(checkpoint) != checkpoint_sha256:
        raise RuntimeError("The checkpoint changed during detection.")
    provenance = {"profile": profile, "library": library, "library_version": library_version,
                  "weights_relative_path": WEIGHTS[profile], "weights_sha256": checkpoint_sha256,
                  "source_files_sha256": source_files}
    if profile == "faster_rcnn":
        provenance["source_integration"] = "adapted_fixed_detectron2_configuration"
    else:
        provenance["source_integration"] = "original_model_loader_with_label_preserving_postprocess"
    return [p for p in predictions if p is not None and p["confidence"] >= threshold], provenance


def _overlap(first, second):
    # FacadeFixer legacy merging uses max(IoU, intersection/smaller-area).
    # Keep that geometric rule; count distinct source models for consensus.
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0., x2 - x1) * max(0., y2 - y1)
    area_a = (first[2] - first[0]) * (first[3] - first[1])
    area_b = (second[2] - second[0]) * (second[3] - second[1])
    return max(intersection / (area_a + area_b - intersection), intersection / min(area_a, area_b))


def _aggregate(predictions: list[dict], profile: str) -> list[dict]:
    # The merge geometry follows backend/detection_agent.py, while this
    # implementation fixes duplicate boxes from one model counting as votes.
    groups = []
    for item in sorted(predictions, key=lambda p: p["confidence"], reverse=True):
        key = re.sub(r"[^a-z0-9]", "", item["class_name"].casefold()).replace("vegeterian", "vegetation")
        for group in groups:
            if group["key"] == key and _overlap(item["bbox_xyxy"], group["anchor"]) >= .5:
                group["items"].append(item)
                break
        else:
            groups.append({"key": key, "anchor": item["bbox_xyxy"], "items": [item]})
    combined = []
    for group in groups:
        # One contribution per model prevents duplicate detections dominating
        # either the consensus or the arithmetic mean.
        contributors = {}
        for item in group["items"]:
            contributors.setdefault(item["source_model"], item)
        if profile == "intersection" and len(contributors) < 2:
            continue
        items = list(contributors.values())
        value = dict(items[0])
        value["bbox_xyxy"] = [sum(p["bbox_xyxy"][i] for p in items) / len(items) for i in range(4)]
        value["confidence"] = sum(p["confidence"] for p in items) / len(items)
        value["source_model"] = profile
        value["contributing_models"] = sorted(contributors)
        value["class_id"] = None  # Checkpoint class indices are model-specific.
        combined.append(value)
    return combined


def run_detection(repo: Path, profile: str, image_path: Path, output_dir: Path,
                  threshold: float, device: str) -> dict:
    """Run a named local detector. Exceptions mean failure, never no defects.

    ensemble/intersection deliberately use exactly YOLO12m, YOLO11m and RT-DETR;
    all three must execute successfully. The provenance declares this membership.
    """
    if profile not in {*WEIGHTS, "ensemble", "intersection"}:
        raise ValueError("Unsupported source detection profile.")
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite number in [0, 1].")
    if not isinstance(device, str) or not re.fullmatch(r"cpu|mps|cuda(?::\d+)?|\d+", device):
        raise ValueError("device must name a local CPU, MPS or CUDA device.")
    repo, image_path, output_dir = Path(repo).resolve(), Path(image_path).resolve(), Path(output_dir).resolve()
    if not image_path.is_file():
        raise FileNotFoundError("Detection image does not exist.")
    source_image_sha256 = _sha256(image_path)
    profiles = PRIMARY_PROFILES if profile in {"ensemble", "intersection"} else (profile,)
    for member in profiles:
        _local_file(repo, WEIGHTS[member])
    from PIL import Image
    import numpy as np
    with Image.open(image_path) as image:
        raster = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    predictions, models = [], []
    # Prevent model libraries writing settings/cache to the user's profile.
    runtime_dir = output_dir / "runtime"
    directories = {"YOLO_CONFIG_DIR": runtime_dir / "ultralytics", "MPLCONFIGDIR": runtime_dir / "matplotlib"}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    with patch.dict(os.environ, {key: str(path) for key, path in directories.items()}), _offline():
        for member in profiles:
            found, provenance = _run_single(repo, member, raster, threshold, device)
            predictions.extend(found)
            models.append(provenance)
    if profile in {"ensemble", "intersection"}:
        predictions = _aggregate(predictions, profile)
    if _sha256(image_path) != source_image_sha256:
        raise RuntimeError("The image changed during detection.")
    for index, prediction in enumerate(predictions, 1):
        prediction["instance_id"] = str(index)
        if prediction.get("class_id") is None:
            prediction.pop("class_id", None)
    provenance = {
        "profile": profile, "device": device, "confidence_threshold": threshold,
        "iou_threshold": .5, "max_detections_per_model": 300,
        "pixel_convention": "stored_pixels", "color_space": "BGR",
        "source_image_sha256": source_image_sha256,
        "worker_sha256": _sha256(Path(__file__)), "models": models,
        "automatic_weight_download": False, "network_access": False,
        "expert_validated": False,
    }
    if len(models) == 1:
        provenance.update({key: models[0][key] for key in ("weights_sha256", "source_files_sha256", "library", "library_version")})
    else:
        provenance.update({"aggregation": "mean_of_one_prediction_per_model", "overlap_rule": "max_iou_ioa_small>=0.5",
                           "required_models": list(PRIMARY_PROFILES), "minimum_distinct_model_votes": 2 if profile == "intersection" else 1,
                           "model_failures_allowed": False, "source_integration": "adapted_consensus_not_legacy_equivalent"})
        source = _local_file(repo, "backend/detection_agent.py")
        provenance["source_files_sha256"] = {"backend/detection_agent.py": _sha256(source)}
    result = {"status": "completed", "defects": predictions,
              "width": int(raster.shape[1]), "height": int(raster.shape[0]),
              "coordinate_space": "original_pixels", "model_provenance": provenance}
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "source_detection.json"
    temporary = raw_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, raw_path)
    return result | {"raw_result_path": str(raw_path)}
