"""Managed, isolated access to the local FacadeFixer detection and mask models."""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .detection_service import DetectionService
from .mask_service import MaskService, _rle
from .storage import ArtifactStore, sha256


# Paths and implementations are server owned. Tool callers select a profile only.
PROFILES = {
    "yolo12m": {"operation": "detection", "weights": ["model_weights/detection/yolo12m_building.pt"], "packages": ["torch", "ultralytics"], "source": "model_runner/adapters/detection/yolo.py"},
    "yolo11m": {"operation": "detection", "weights": ["model_weights/detection/yolo11m_building.pt"], "packages": ["torch", "ultralytics"], "source": "model_runner/adapters/detection/yolo.py"},
    "rtdetr": {"operation": "detection", "weights": ["model_weights/rtdetr-l-bst.pt"], "packages": ["torch", "ultralytics"], "source": "model_runner/adapters/detection/rtdetr.py"},
    "faster_rcnn": {"operation": "detection", "weights": ["model_weights/detection/Faster_R-CNN.pth"], "packages": ["torch", "detectron2"], "source": "backend/detection_agent.py"},
    "ensemble": {"operation": "detection", "weights": ["model_weights/detection/yolo12m_building.pt", "model_weights/detection/yolo11m_building.pt", "model_weights/rtdetr-l-bst.pt"], "packages": ["torch", "ultralytics"], "source": "backend/detection_agent.py"},
    "intersection": {"operation": "detection", "weights": ["model_weights/detection/yolo12m_building.pt", "model_weights/detection/yolo11m_building.pt", "model_weights/rtdetr-l-bst.pt"], "packages": ["torch", "ultralytics"], "source": "backend/detection_agent.py"},
    "mobile_sam": {"operation": "segmentation", "weights": ["model_weights/mobile_sam.pt"], "packages": ["torch", "torchvision", "ultralytics"], "source": "backend/run_sam_bbox_baseline.py", "interactive": True},
    "sam3": {"operation": "segmentation", "weights": ["model_weights/sam3.pt"], "packages": ["torch", "torchvision", "ultralytics"], "source": "backend/run_sam_bbox_baseline.py", "interactive": True},
    "unet_crack": {"operation": "segmentation", "weights": ["model_weights/crack/model_unet_vgg_16_best.pt"], "packages": ["torch", "torchvision", "cv2"], "source": "backend/run_unet_crack_direct_baseline.py", "interactive": False, "class_codes": ["C"]},
}


class ModelError(ValueError):
    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code, self.details = code, details


class ModelService:
    def __init__(self, store: ArtifactStore, repo_path: Path | None = None,
                 python_path: Path | None = None, device: str | None = None,
                 timeout_seconds: float | None = None):
        self.store = store
        default_repo = Path(__file__).resolve().parents[3] / "项目代码" / "FacadeFixer-main"
        self.repo = Path(repo_path or os.environ.get("FACADE_SOURCE_REPO", default_repo)).expanduser().resolve()
        default_python = self.repo / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self.python = Path(python_path or os.environ.get("FACADE_SOURCE_PYTHON", default_python)).expanduser().resolve()
        self.device = device or os.environ.get("FACADE_SOURCE_DEVICE", "cpu")
        self.timeout = timeout_seconds if timeout_seconds is not None else os.environ.get("FACADE_SOURCE_TIMEOUT_SECONDS", "600")
        self.masks = MaskService(store)
        self._slot = threading.BoundedSemaphore(1)

    def _config(self, profile: str | None = None, operation: str | None = None):
        if not (self.repo / "model_runner").is_dir() or not (self.repo / "backend").is_dir():
            raise ModelError("SOURCE_REPO_NOT_FOUND", "Set FACADE_SOURCE_REPO to the FacadeFixer-main checkout.")
        if not self.python.is_file():
            raise ModelError("SOURCE_PYTHON_NOT_FOUND", "Set FACADE_SOURCE_PYTHON to an existing model runtime Python.")
        if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?|[0-9]+", self.device):
            raise ModelError("INVALID_DEVICE", "Device must be cpu, cuda, cuda:N, or a GPU index.")
        if self.device.isdigit():
            self.device = "cuda:" + self.device
        try:
            self.timeout = float(self.timeout)
        except (ValueError, TypeError):
            raise ModelError("INVALID_TIMEOUT", "Model timeout must be between 1 and 7200 seconds.") from None
        if not math.isfinite(self.timeout) or not 1 <= self.timeout <= 7200:
            raise ModelError("INVALID_TIMEOUT", "Model timeout must be between 1 and 7200 seconds.")
        if profile is not None:
            if profile not in PROFILES or PROFILES[profile]["operation"] != operation:
                raise ModelError("UNKNOWN_MODEL_PROFILE", "Select an approved model profile for this operation.",
                                 allowed=[key for key, value in PROFILES.items() if value["operation"] == operation])
            missing = [name for name in PROFILES[profile]["weights"] if not (self.repo / name).is_file()]
            if missing:
                raise ModelError("WEIGHTS_NOT_FOUND", "Local model weights are missing; no automatic download is performed.", missing=missing)

    def _run(self, directory: Path, action: str, **arguments) -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        request, output = directory / "worker_input.json", directory / "worker_output.json"
        self.store.write_json(request, {"action": action, "repo": str(self.repo), "device": self.device,
                                       "output_dir": str(directory), **arguments})
        # Do not give the model subprocess API credentials or load source .env files.
        names = {"SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "PATHEXT", "COMSPEC",
                 "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "LOCALAPPDATA", "APPDATA", "USERPROFILE"}
        env = {key: value for key, value in os.environ.items() if key.upper() in names}
        env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONNOUSERSITE="1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true", DO_NOT_TRACK="1",
                   YOLO_CONFIG_DIR=str(directory / "runtime_config"), TORCH_HOME=str(directory / "runtime_cache"),
                   MPLCONFIGDIR=str(directory / "runtime_config" / "matplotlib"))
        if not self._slot.acquire(timeout=float(self.timeout)):
            raise ModelError("MODEL_BUSY", "Another local model process is still running; retry after its job finishes.")
        try:
            process = subprocess.run([str(self.python), str(Path(__file__).with_name("model_worker.py")),
                                      "--request", str(request), "--output", str(output)],
                                     cwd=directory, env=env, capture_output=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise ModelError("MODEL_TIMEOUT", "Model process exceeded its configured deadline; no automatic retry.") from None
        except OSError as error:
            raise ModelError("MODEL_PROCESS_FAILED", str(error)) from None
        finally:
            self._slot.release()
        if process.returncode != 0 or not output.is_file():
            raise ModelError("MODEL_PROCESS_FAILED", "Model runtime exited without a structured result.", exit_code=process.returncode)
        try:
            result = json.loads(output.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raise ModelError("INVALID_MODEL_OUTPUT", "Model runtime returned invalid JSON.") from None
        if not isinstance(result, dict) or result.get("status") != "completed":
            error = result.get("error", {}) if isinstance(result, dict) else {}
            raise ModelError("MODEL_EXECUTION_FAILED", "Local model execution failed.", worker_error=error)
        return result

    def _fail(self, job_id: str, error: Exception) -> dict:
        failure = {"code": getattr(error, "code", "INVALID_INPUT"), "message": str(error),
                   "details": getattr(error, "details", {})}
        self.store.fail_job(job_id, failure)
        return {"job_id": job_id, "status": "failed", "error": failure}

    def list_tools(self) -> dict:
        job_id, directory = self.store.create_job("model_capabilities", {})
        try:
            self._config()
            runtime = self._run(directory, "probe")
            rows = []
            for name, profile in PROFILES.items():
                missing_weights = [value for value in profile["weights"] if not (self.repo / value).is_file()]
                missing_packages = [value for value in profile["packages"] if not runtime["packages"].get(value)]
                rows.append({"model_profile": name, **profile,
                             "availability": "unavailable" if missing_weights or missing_packages else "available_unverified",
                             "missing_weights": missing_weights, "missing_packages": missing_packages,
                             "model_load_verified": False})
            result = {"schema_version": "facade-model-catalog/v1", "status": "configured", "job_id": job_id,
                      "source_repository": str(self.repo), "python": str(self.python), "device": self.device,
                      "runtime": runtime, "models": rows, "paid_api_requests": False,
                      "note": "Dependency discovery does not prove successful model loading or inference. Each execution records its own result."}
            self.store.complete_job(job_id, result)
            return result
        except Exception as error:
            return self._fail(job_id, error)

    def _snapshot(self, image_path: str, directory: Path) -> tuple[Path, dict]:
        source = self.store.resolve_input(image_path)
        digest = sha256(source)
        with Image.open(source) as image:
            image.load()
            width, height = image.size
            orientation = image.getexif().get(274, 1)
        target = directory / "input" / ("source" + source.suffix.lower())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256(target) != digest:
            raise ModelError("SOURCE_CHANGED", "Input changed while creating its immutable job snapshot.")
        return target, {"width": width, "height": height, "source_sha256": digest,
                        "pixel_convention": "stored_pixels", "exif_orientation": orientation}

    def detect(self, image_paths: list[str], model_profile: str = "yolo12m", threshold: float = .25) -> dict:
        job_id, directory = self.store.create_job("source_detection", {"image_paths": image_paths, "model_profile": model_profile,
                                               "threshold": str(threshold)})
        try:
            self._config(model_profile, "detection")
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ModelError("INVALID_THRESHOLD", "threshold must be a finite number between 0 and 1.")
            if not isinstance(image_paths, list) or not image_paths or len(image_paths) > 100:
                raise ModelError("INVALID_IMAGES", "Supply between 1 and 100 image paths or artifact URIs.")
            paths = [self.store.resolve_input(value) for value in image_paths]
            if len(set(paths)) != len(paths):
                raise ModelError("DUPLICATE_IMAGE", "Input images must not repeat.")
            images, warnings, worker_artifacts = [], [], []
            normalizer = DetectionService(self.store)
            for index, path in enumerate(paths):
                current = directory / f"image_{index:04d}"
                snapshot, metadata = self._snapshot(str(path), current)
                raw = self._run(current, "detect", image_path=str(snapshot), model_profile=model_profile, threshold=threshold)
                if sha256(snapshot) != metadata["source_sha256"]:
                    raise ModelError("SOURCE_CHANGED", "Detection input snapshot changed during inference.")
                declared = raw.get("model_provenance", {}).get("source_image_sha256")
                if declared is not None and declared != metadata["source_sha256"]:
                    raise ModelError("MODEL_SOURCE_MISMATCH", "Model result belongs to a different source image.")
                image_id = "image_" + metadata["source_sha256"][:20]
                defects, unmapped = normalizer._normalize(raw, metadata["width"], metadata["height"], threshold, image_id, warnings)
                if metadata["exif_orientation"] not in (None, 1):
                    warnings.append(f"{image_id}: stored pixels used without EXIF rotation.")
                images.append({"index": index, "image_id": image_id, "image_path": str(snapshot), **metadata,
                               "defects": defects, "unmapped_defects": unmapped,
                               "original_image": self.store.artifact(snapshot, "original_image", job_id),
                               "model_provenance": raw.get("model_provenance", {})})
                worker_artifacts.append(self.store.artifact(current / "worker_output.json", "model_worker_result", job_id))
            result = {"status": "completed", "job_id": job_id, "schema": "bbox_class_only_v1",
                      "coordinate_space": "original_pixels", "bbox_convention": "zero_based_xyxy_max_exclusive",
                      "images": images, "source": {"method": "source", "model_profile": model_profile,
                      "threshold": threshold, "device": self.device}, "provenance": {"worker_outputs": worker_artifacts,
                      "source_repository": str(self.repo), "reference_mask_used": False},
                      "review_required": True, "expert_validated": False, "warnings": list(dict.fromkeys(warnings))}
            manifest = directory / "detections.json"
            self.store.write_json(manifest, result)
            result["manifest_artifact"] = self.store.artifact(manifest, "detection_manifest", job_id)
            self.store.complete_job(job_id, result)
            return result
        except Exception as error:
            return self._fail(job_id, error) | {"images": []}

    @staticmethod
    def _points(points: list, width: int, height: int) -> list:
        if not isinstance(points, list) or len(points) > 1000:
            raise ModelError("INVALID_POINTS", "Points must be a list of at most 1000 original-pixel XY pairs.")
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point):
                raise ModelError("INVALID_POINTS", "Each point must contain two finite numbers.")
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ModelError("INVALID_POINTS", "Points must lie inside the original stored image.")
        return points

    def segment(self, image_path: str, defects: list[dict], model_profile: str = "mobile_sam") -> dict:
        return self._segment(image_path, defects, model_profile)

    def refine(self, image_path: str, annotation_manifest_path: str, instance_id: str,
               positive_points: list[list[float]], negative_points: list[list[float]],
               model_profile: str = "mobile_sam") -> dict:
        return self._segment(image_path, None, model_profile, annotation_manifest_path,
                             str(instance_id), positive_points, negative_points)

    def _read_mask(self, instance: dict, shape: tuple) -> tuple[np.ndarray, dict]:
        artifact = instance.get("mask_artifact") or instance.get("mask")
        if not isinstance(artifact, dict) or not artifact.get("sha256"):
            raise ModelError("INVALID_PARENT_MASK", "Parent masks must carry immutable artifact hashes.")
        path = self.store.resolve_input(artifact.get("uri") or artifact.get("path", ""))
        if sha256(path) != artifact["sha256"]:
            raise ModelError("PARENT_MASK_CHANGED", "Parent mask hash does not match its manifest.")
        with Image.open(path) as image:
            array = np.asarray(image)
        if array.shape != shape or not set(np.unique(array)).issubset({0, 255}):
            raise ModelError("INVALID_PARENT_MASK", "Parent mask must be binary and match the source raster dimensions.")
        return array > 0, artifact

    def _segment(self, image_path, defects, profile, parent_path=None, instance_id=None,
                 positive_points=None, negative_points=None):
        started = time.perf_counter()
        job_id, directory = self.store.create_job("source_mask_refine" if parent_path else "source_mask", {
            "image_path": image_path, "defects": defects, "model_profile": profile,
            "parent_manifest": parent_path, "instance_id": instance_id,
            "positive_points": positive_points, "negative_points": negative_points})
        try:
            self._config(profile, "segmentation")
            parent, parent_artifact, preserved = None, None, {}
            if parent_path:
                if not PROFILES[profile].get("interactive"):
                    raise ModelError("INTERACTIVE_NOT_SUPPORTED", "Point refinement requires mobile_sam or sam3.")
                parent_file = self.store.resolve_input(parent_path)
                parent_bytes = parent_file.read_bytes()
                parent = json.loads(parent_bytes.decode("utf-8"))
                if parent.get("schema_version") != "facade-mask/v1":
                    raise ModelError("INVALID_PARENT_MANIFEST", "Expected a facade-mask/v1 manifest.")
                parent_artifact = {"path": str(parent_file), "sha256": hashlib.sha256(parent_bytes).hexdigest(), "job_id": parent.get("job_id")}
                entries = parent.get("instances", [])
                defects = [{key: value for key, value in row.items() if key in ("instance_id", "class_code", "class_name", "class_subtype", "bbox_xyxy", "original_class_code", "original_class_name")} for row in entries]
                if sum(str(row.get("instance_id")) == instance_id for row in entries) != 1:
                    raise ModelError("INSTANCE_NOT_FOUND", "Select exactly one instance from the parent manifest.")
            validated = self.masks._validated(image_path, defects)
            snapshot, metadata = self._snapshot(validated["image_path"], directory)
            if metadata["source_sha256"] != validated["source_sha256"]:
                raise ModelError("SOURCE_CHANGED", "Source changed since validation.")
            if parent:
                if parent.get("source_sha256") != metadata["source_sha256"]:
                    raise ModelError("SOURCE_HASH_MISMATCH", "Refinement requires the exact original image of the parent mask.")
                positive_points = self._points(positive_points or [], metadata["width"], metadata["height"])
                negative_points = self._points(negative_points or [], metadata["width"], metadata["height"])
                if not positive_points and not negative_points:
                    raise ModelError("POINTS_REQUIRED", "Supply at least one positive or negative point.")
                for entry in parent["instances"]:
                    preserved[str(entry["instance_id"])]=self._read_mask(entry, (metadata["height"], metadata["width"]))
            selected = [row for row in validated["defects"] if not parent or str(row["instance_id"]) == instance_id]
            if profile == "unet_crack" and any(row["class_code"] != "C" for row in selected):
                raise ModelError("UNSUPPORTED_CLASS", "unet_crack only supports canonical C crack instances (legacy M is normalized); choose SAM for other classes.")
            if selected:
                raw = self._run(directory / "model", "segment", image_path=str(snapshot), model_profile=profile,
                                defects=selected, positive_points=positive_points, negative_points=negative_points)
            else:
                raw = {"instances": [], "model_provenance": {"model_profile": profile, "inference_skipped": "no_defects_provided"}}
            if sha256(snapshot) != metadata["source_sha256"]:
                raise ModelError("SOURCE_CHANGED", "Segmentation input snapshot changed during inference.")
            declared = raw.get("model_provenance", {}).get("source_image_sha256")
            if declared is not None and declared != metadata["source_sha256"]:
                raise ModelError("MODEL_SOURCE_MISMATCH", "Model mask belongs to a different source image.")
            records = raw.get("instances")
            if not isinstance(records, list) or len(records) != len(selected):
                raise ModelError("INSTANCE_COUNT_MISMATCH", "Model must return exactly one mask for every requested defect, including empty masks.")
            predicted = {str(row["instance_id"]): row for row in records}
            if len(predicted) != len(selected) or set(predicted) != {str(row["instance_id"]) for row in selected}:
                raise ModelError("INSTANCE_ID_MISMATCH", "Model returned unexpected or repeated instance identifiers.")
            with Image.open(snapshot) as image:
                rgb = np.asarray(image.convert("RGB"))
            arrays, instances = [], []
            for index, defect in enumerate(validated["defects"]):
                key = str(defect["instance_id"])
                record = predicted.get(key)
                if record is not None:
                    mask_path = Path(record["mask_path"]).resolve()
                    if not mask_path.is_relative_to((directory / "model").resolve()):
                        raise ModelError("MODEL_OUTPUT_ESCAPE", "Model mask must be inside this job's model directory.")
                    with Image.open(mask_path) as image:
                        data = np.asarray(image)
                    if data.shape != rgb.shape[:2] or not set(np.unique(data)).issubset({0, 255}):
                        raise ModelError("INVALID_MASK", "Model mask must be an original-size binary PNG.")
                    array = data > 0
                    flags = list(record.get("quality_flags", []))
                else:
                    array, _ = preserved[key]
                    flags = ["unchanged_from_parent_revision"]
                if not array.any():
                    flags.append("empty_or_tiny_foreground")
                mask_file = directory / "instance_masks" / f"instance_{index:05d}.png"
                mask_file.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(array.astype(np.uint8) * 255).save(mask_file)
                artifact = self.store.artifact(mask_file, "instance_mask", job_id)
                item = {**defect, "mask": artifact, "mask_artifact": artifact, "mask_path": str(mask_file),
                        "area_pixels": int(array.sum()), "segmentation_rle": _rle(array), "expert_validated": False,
                        "method": "source_model", "algorithm": profile if record is not None else "preserved_parent_mask",
                        "quality_flags": flags, "model": raw.get("model_provenance") if record is not None else None}
                if record and record.get("score") is not None:
                    score = record["score"]
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                        raise ModelError("INVALID_MODEL_SCORE", "Model score must be a finite number.")
                    item["model_score"] = score
                    item["model_score_meaning"] = "model candidate score; not measured accuracy"
                if record and record.get("raw_mask_path"):
                    raw_path = Path(record["raw_mask_path"]).resolve()
                    if not raw_path.is_relative_to((directory / "model").resolve()):
                        raise ModelError("MODEL_OUTPUT_ESCAPE", "Raw model mask must be inside this job's model directory.")
                    with Image.open(raw_path) as image:
                        raw_array = np.asarray(image)
                    if raw_array.shape != rgb.shape[:2] or not set(np.unique(raw_array)).issubset({0, 255}):
                        raise ModelError("INVALID_MASK", "Raw model mask must be an original-size binary PNG.")
                    item["raw_model_mask"] = self.store.artifact(raw_path, "raw_model_mask", job_id)
                    item["bbox_clipped_pixels"] = record.get("bbox_clipped_pixels", 0)
                if parent:
                    item["parent_mask"] = preserved[key][1]
                arrays.append(array)
                instances.append(item)
            original = self.store.artifact(snapshot, "original_image", job_id)
            result = self.masks._export(job_id, directory, rgb, instances, arrays, original,
                                        "image_" + metadata["source_sha256"][:20], snapshot,
                                        metadata["source_sha256"], validated["quality_flags"])
            result["revision"] = int(parent.get("revision", 1)) + 1 if parent else 1
            result["provenance"] = {"method": "source_model", "model_profile": profile,
                                    "source_repository": str(self.repo), "model": raw.get("model_provenance", {}),
                                    "reference_mask_used": False, "human_guides_used": bool(parent),
                                    "model_mask_used_as_input": False,
                                    "refinement_semantics": "bbox_and_points_regeneration_preserving_parent" if parent else None,
                                    "parent_manifest": parent_artifact,
                                    "events": {"instance_id": instance_id, "positive_points": positive_points,
                                               "negative_points": negative_points} if parent else None}
            if parent:
                encoded_events = json.dumps(result["provenance"]["events"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
                result["provenance"]["events_sha256"] = hashlib.sha256(encoded_events).hexdigest()
            if selected:
                result["provenance"]["worker_output"] = self.store.artifact(directory / "model" / "worker_output.json", "model_worker_result", job_id)
            result["metrics"] = {"seconds": time.perf_counter() - started, "api_requests": 0, "actual_charge": 0,
                                 "cost_scope": "no API requests; local compute and human review excluded"}
            result = self.masks._manifest(job_id, directory, result)
            self.store.complete_job(job_id, result)
            return result
        except Exception as error:
            return self._fail(job_id, error)

    def get_job(self, job_id: str) -> dict:
        return self.store.job_status(job_id)
