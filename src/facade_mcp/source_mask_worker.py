"""Local FacadeFixer segmentation adapters for the isolated source worker.

The SAM path adapts ``backend/run_sam_bbox_baseline.py`` without importing the
legacy web application. The U-Net path reuses the repository's UNet16 class and
the preprocessing in ``backend/run_unet_crack_direct_baseline.py``, while loading
an existing checkpoint on the requested device. No model download is performed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
from pathlib import Path


_PROFILES = {
    "sam3": ("model_weights/sam3.pt", "backend/run_sam_bbox_baseline.py"),
    "mobile_sam": ("model_weights/mobile_sam.pt", "backend/run_sam_bbox_baseline.py"),
    "unet_crack": ("model_weights/crack/model_unet_vgg_16_best.pt",
                   "backend/run_unet_crack_direct_baseline.py"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(repo: Path, relative: str) -> Path:
    path = (repo / relative).resolve()
    if not path.is_relative_to(repo) or not path.is_file():
        raise FileNotFoundError("Required local FacadeFixer source or checkpoint is unavailable")
    return path


def _validate_defects(defects, width, height, profile):
    if not isinstance(defects, list):
        raise ValueError("defects must be a list")
    seen = set()
    for defect in defects:
        if not isinstance(defect, dict):
            raise ValueError("Each defect must be an object")
        identifier = defect.get("instance_id", defect.get("id"))
        if (isinstance(identifier, bool) or not isinstance(identifier, (int, str))
                or not str(identifier).strip() or str(identifier) in seen):
            raise ValueError("Each defect requires a unique instance_id")
        seen.add(str(identifier))
        box = defect.get("bbox_xyxy")
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or int(value) != value for value in box)):
            raise ValueError("bbox_xyxy must contain four integer stored-pixel coordinates")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("bbox_xyxy must be inside the original stored image")
        if profile == "unet_crack" and defect.get("class_code") not in {"C", "M"}:
            raise ValueError("unet_crack supports only crack class codes C and M")


def _validate_points(points, width, height):
    if points is None:
        return []
    if not isinstance(points, list):
        raise ValueError("Point prompts must be lists of original-pixel [x, y] pairs")
    result = []
    for point in points:
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in point)
                or not (0 <= point[0] < width and 0 <= point[1] < height)):
            raise ValueError("Point prompt is outside original stored-image bounds")
        result.append([float(point[0]), float(point[1])])
    return result


def _make_sam(weights):
    import ultralytics
    from ultralytics import SAM
    return SAM(str(weights)), ultralytics.__version__


def _make_text_sam(weights, device):
    import ultralytics
    from ultralytics.models.sam import SAM3SemanticPredictor
    return SAM3SemanticPredictor(overrides={"model": str(weights), "device": device,
        "task": "segment", "mode": "predict", "verbose": False, "save": False}), ultralytics.__version__


def _as_array(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _sam_mask(model, raster, box, device, positive, negative, prompt_mode="bbox", text_prompt=None):
    import cv2
    import numpy as np
    kwargs = {"source": raster, "bboxes": [box], "device": device,
              "verbose": False, "save": False, "save_txt": False, "save_crop": False}
    if positive or negative:
        # The outer dimension groups all clicks into this one prompted object.
        kwargs.update(points=[positive + negative], labels=[[1] * len(positive) + [0] * len(negative)])
    if prompt_mode == "point":
        kwargs.pop("bboxes")
        kwargs.update(points=[[(box[0]+box[2])/2, (box[1]+box[3])/2]], labels=[1])
    if prompt_mode == "text":
        import numpy as np
        model.set_image(np.asarray(raster))
        results = model(text=[text_prompt])
    else:
        results = model.predict(**kwargs)
    candidates = []
    for result in results:
        if result.masks is None:
            continue
        arrays = _as_array(result.masks.data)
        if arrays.ndim == 2:
            arrays = arrays[None]
        if arrays.ndim != 3:
            raise ValueError("SAM returned an invalid mask array")
        scores = None
        boxes = getattr(result, "boxes", None)
        if boxes is not None and getattr(boxes, "conf", None) is not None:
            scores = _as_array(boxes.conf).reshape(-1)
        for index, array in enumerate(arrays):
            if not np.isfinite(array).all():
                raise ValueError("SAM returned a nonfinite mask")
            score = None
            if scores is not None and index < len(scores):
                value = float(scores[index])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("SAM returned an invalid model score")
                score = value
            if array.shape != (raster.height, raster.width):
                array = cv2.resize(array.astype(np.float32), raster.size, interpolation=cv2.INTER_NEAREST)
            candidates.append((array > 0.5, score))
    if not candidates:
        return np.zeros((raster.height, raster.width), dtype=bool), None
    # The interactive predictor normally returns one candidate. If several are
    # returned, use the highest model score; never merge unrelated alternatives.
    return max(candidates, key=lambda pair: -1.0 if pair[1] is None else pair[1])


def _load_unet(repo, weights, device):
    import torch
    source = _source_path(repo, "training/crack_segmentation/unet/unet_transfer.py")
    spec = importlib.util.spec_from_file_location("_facade_source_unet_transfer", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the local UNet16 architecture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Do not call the source utils loader: it forces .cuda() even on CPU.
    model = module.UNet16(pretrained=False)
    checkpoint = torch.load(str(weights), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported local U-Net checkpoint format")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint.get("check_point")))
    if state is None:
        raise ValueError("Local U-Net checkpoint does not contain a state dict")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, tuple(module.input_size), torch.__version__


def _unet_mask(model, input_size, raster, box, device):
    import cv2
    import numpy as np
    import torch
    x1, y1, x2, y2 = box
    crop = np.asarray(raster)[y1:y2, x1:x2]
    resized = cv2.resize(crop, input_size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    normalized = (resized - np.asarray([.485, .456, .406], dtype=np.float32)) / np.asarray([.229, .224, .225], dtype=np.float32)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probability = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("U-Net returned an invalid probability mask")
    probability = cv2.resize(probability, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
    mask = np.zeros((raster.height, raster.width), dtype=bool)
    # Keep the source direct-baseline default, also used by CrackService.
    mask[y1:y2, x1:x2] = probability >= .2
    return mask


def run_segmentation(repo: Path, profile: str, image_path: Path, output_dir: Path,
                     defects: list[dict], device: str, positive_points: list | None = None,
                     negative_points: list | None = None, prompt_mode: str = "bbox",
                     text_prompts: dict | None = None) -> dict:
    """Write one full-resolution, bbox-clipped binary PNG for each real candidate.

    Inputs are pre-authorized local paths from the parent worker. Coordinates are
    stored pixels with exclusive maximum coordinates. A zero-area mask stays an
    explicitly flagged empty model result; it is never replaced by a rectangle.
    """
    import numpy as np
    from PIL import Image

    if profile not in _PROFILES:
        raise ValueError("Supported segmentation profiles are sam3, mobile_sam and unet_crack; COCO YOLO-seg is not a facade-defect model")
    if prompt_mode not in {"bbox", "point", "text"}:
        raise ValueError("Unsupported prompt_mode")
    if prompt_mode == "text" and profile != "sam3":
        raise ValueError("Text prompts require sam3; no automatic fallback")
    if profile == "unet_crack" and prompt_mode != "bbox":
        raise ValueError("U-Net supports only bbox crop inference")
    configured_device = device
    # Ultralytics accepts GPU index strings, while torch requires cuda:N.
    if isinstance(device, str) and device.isdecimal():
        device = "cuda:{}".format(int(device))
    repo, image_path, output_dir = Path(repo).resolve(), Path(image_path).resolve(), Path(output_dir).resolve()
    weights_relative, source_relative = _PROFILES[profile]
    weights = _source_path(repo, weights_relative)
    source = _source_path(repo, source_relative)
    input_sha256 = _sha256(image_path)
    with Image.open(image_path) as image:
        # Avoid EXIF transposition and metadata-dependent library rotation.
        raster = Image.new("RGB", image.size)
        raster.paste(image.convert("RGB"))
    _validate_defects(defects, raster.width, raster.height, profile)
    positive = _validate_points(positive_points, raster.width, raster.height)
    negative = _validate_points(negative_points, raster.width, raster.height)
    if (positive or negative) and len(defects) != 1:
        raise ValueError("Point prompts require exactly one defect instance")
    if profile == "unet_crack" and (positive or negative):
        raise ValueError("U-Net does not support interactive point prompts")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["YOLO_CONFIG_DIR"] = str(output_dir / "runtime-config")
    os.environ["MPLCONFIGDIR"] = str(output_dir / "runtime-config" / "matplotlib")
    source_files = {source_relative: _sha256(source)}
    if profile == "unet_crack":
        architecture_relative = "training/crack_segmentation/unet/unet_transfer.py"
        source_files[architecture_relative] = _sha256(_source_path(repo, architecture_relative))
    provenance = {"profile": profile, "weights_relative_path": weights_relative,
                  "weights_sha256": _sha256(weights), "source_files_sha256": source_files,
                  "source_image_sha256": input_sha256,
                  "adapter_file_sha256": _sha256(Path(__file__)), "device": device,
                  "configured_device": configured_device, "prompt_mode": prompt_mode, "text_prompts": text_prompts,
                  "automatic_weight_download": False, "automatic_dependency_install": False,
                  "pixel_convention": "stored_pixels", "bbox_constraint": "xyxy_max_exclusive",
                  "adapter": "unet16_cpu_safe_checkpoint_loader" if profile == "unet_crack" else "ultralytics_sam_prompt_adapter"}
    model = input_size = None
    if defects:
        if profile == "unet_crack":
            model, input_size, version = _load_unet(repo, weights, device)
            provenance.update(library="torch", library_version=str(version), probability_threshold=.2,
                              preprocessing="RGB crop, source input_size, ImageNet normalization, sigmoid, INTER_AREA restore")
        else:
            if prompt_mode == "text":
                model, version = _make_text_sam(weights, device)
            else:
                model, version = _make_sam(weights)
            provenance.update(library="ultralytics", library_version=str(version))
    instances = []
    for index, defect in enumerate(defects):
        box = [int(value) for value in defect["bbox_xyxy"]]
        if profile == "unet_crack":
            mask, score = _unet_mask(model, input_size, raster, box, device), None
        else:
            text_prompt = (text_prompts or {}).get(str(defect["instance_id"]))
            if prompt_mode == "text" and not text_prompt:
                raise ValueError("Text mode needs one nonempty prompt per instance")
            mask, score = _sam_mask(model, raster, box, device, positive, negative, prompt_mode, text_prompt)
        constrained = np.zeros((raster.height, raster.width), dtype=bool)
        x1, y1, x2, y2 = box
        constrained[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        mask_path = output_dir / "instance-{:04d}.png".format(index)
        raw_mask_path = output_dir / "instance-{:04d}-raw.png".format(index)
        if mask_path.exists() or raw_mask_path.exists():
            raise ValueError("Segmentation output already exists; use a new job directory")
        # Keep the full-resolution prediction before the detection bbox policy.
        # This is evidence of the model candidate, not an approved segmentation.
        Image.fromarray(mask.astype(np.uint8) * 255).save(raw_mask_path)
        Image.fromarray(constrained.astype(np.uint8) * 255).save(mask_path)
        clipped_pixels = int(np.count_nonzero(mask) - np.count_nonzero(constrained))
        flags = ["candidate_requires_review", "bbox_clipped_pixels:{}".format(clipped_pixels)]
        if not constrained.any():
            flags.append("empty_mask")
        if score is None:
            flags.append("model_score_unavailable")
        else:
            flags.append("model_score_is_not_accuracy")
        instances.append({"instance_id": defect.get("instance_id", defect.get("id")),
                          "mask_path": str(mask_path), "raw_mask_path": str(raw_mask_path),
                          "bbox_clipped_pixels": clipped_pixels, "score": score, "quality_flags": flags,
                          "model": {"profile": profile, "weights_sha256": provenance["weights_sha256"]}})
    if _sha256(image_path) != input_sha256:
        raise ValueError("Input image changed during segmentation; result cannot be accepted")
    if _sha256(weights) != provenance["weights_sha256"]:
        raise ValueError("Local model weights changed during segmentation; result cannot be accepted")
    provenance.update(source_image_unchanged=True, weights_unchanged=True)
    return {"status": "completed", "instances": instances, "model_provenance": provenance}
