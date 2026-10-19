"""Normalize supplied annotations, local YOLO, or FacadeFixer detector HTTP.

This adapter does not import the heavyweight backend or invent bounding boxes.
The HTTP branch calls only POST /api/detect, explicitly disabling paid MLLM
justification. Its threshold is a postfilter: the backend can have an additional
model-side threshold that this adapter cannot lower.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import socket
import subprocess
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image


CLASS_NAMES = {
    "C": "Linearcrack", "M": "Meshcrack", "P": "Peeling",
    "S": "Spalling", "T": "RustStain", "R": "Corrosion",
    "D": "Dirt", "V": "Vegetation", "L": "Leakage",
}
# Numeric model class IDs are intentionally not mapped: IDs vary by checkpoint.
LABEL_CODES = {
    "concretecrack": "C", "tilecrack": "C", "linearcrack": "C",
    "crack": "C", "裂缝": "C", "混凝土裂缝": "C",
    "craquelure": "M", "meshcrack": "M", "网裂": "M", "龟裂": "M",
    "concretedelamination": "P", "degradedplaster": "P",
    "peeling": "P", "delamination": "P", "剥离": "P",
    "concretespalling": "S", "tilespalling": "S", "spalling": "S", "剥落": "S",
    "ruststain": "T", "锈渍": "T", "corrosion": "R", "锈蚀": "R",
    "contaminants": "D", "dirt": "D", "wallstain": "D", "groundlitter": "D",
    "污染物": "D", "污渍": "D", "vegetation": "V", "vegeterian": "V", "植被": "V",
    "waterstain": "L", "leakage": "L", "水渍": "L", "渗漏": "L",
}
LOCAL_MODELS = {"yolo12m", "yolo11m", "faster_rcnn", "rtdetr", "ensemble", "intersection"}


class DetectionError(ValueError):
    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code, self.details = code, details

    def as_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "details": self.details}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise DetectionError("INVALID_GEOMETRY", f"{field} must be a finite number.")
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise DetectionError("INVALID_GEOMETRY", f"{field} must be a finite number.") from None
    if not math.isfinite(result):
        raise DetectionError("INVALID_GEOMETRY", f"{field} must be a finite number.")
    return result


def _label_key(label: str) -> str:
    return "".join(c for c in label.casefold() if c not in " _-./")


class DetectionService:
    def __init__(self, store, backend_url: str | None = None, token: str | None = None,
                 model: str | None = None, timeout_seconds: float | None = None):
        self.store = store
        self.backend_url = backend_url or os.environ.get("FACADE_DETECTOR_URL")
        self.token = token or os.environ.get("FACADE_DETECTOR_TOKEN")
        self.model = model or os.environ.get("FACADE_DETECTOR_MODEL", "yolo12m")
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else float(os.environ.get("FACADE_DETECTOR_TIMEOUT_SECONDS", "180"))
        self.local_python = os.environ.get("FACADE_DETECTOR_PYTHON")
        self.local_weights = os.environ.get("FACADE_DETECTOR_WEIGHTS")
        self.local_device = os.environ.get("FACADE_DETECTOR_DEVICE", "cpu")

    def detect(self, image_paths: list[str], method: str = "annotations",
               annotation_path: str | None = None, threshold: float = .25) -> dict:
        """Return original-image XYXY boxes and a traceable detection manifest.

        Annotation imports require an explicit manifest. An empty supplied
        defects list is valid; a missing annotation is an error, not an empty
        successful detection. Unsupported classes remain in unmapped_defects.
        """
        job_id, job_dir = self.store.create_job("detection", {
            "image_paths": image_paths, "method": method,
            "annotation_path": annotation_path,
            "threshold": threshold if isinstance(threshold, (int, float)) and math.isfinite(threshold) else str(threshold),
        })
        try:
            if method not in {"annotations", "http", "local_yolo"}:
                raise DetectionError("UNKNOWN_METHOD", "Use annotations, http, or local_yolo.")
            threshold = _number(threshold, "threshold")
            if not 0 <= threshold <= 1:
                raise DetectionError("INVALID_THRESHOLD", "threshold must be between 0 and 1.")
            if not isinstance(image_paths, list) or not image_paths:
                raise DetectionError("IMAGES_REQUIRED", "Supply at least one image path or image artifact ID.")
            paths = [self.store.resolve_input(value) for value in image_paths]
            if len(set(paths)) != len(paths):
                raise DetectionError("DUPLICATE_IMAGE", "The same image path was supplied more than once.")
            records, annotation_artifact = self._read_annotations(annotation_path, job_id) if method == "annotations" else (None, None)
            warnings: list[str] = []
            images = []
            local_artifacts = []
            for index, path in enumerate(paths):
                with Image.open(path) as image:
                    width, height = image.size
                    orientation = image.getexif().get(274, 1)
                    image.load()
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                image_id = "image_" + sha[:20]
                if orientation not in (None, 1):
                    warnings.append(f"{image_id}: stored image pixels are used without EXIF rotation; all boxes use this same pixel convention.")
                if records is not None:
                    raw = self._find_annotation(records, path, sha)
                elif method == "local_yolo":
                    raw, artifact = self._run_local_yolo(path, image_id, sha, job_id, index, threshold)
                    local_artifacts.append(artifact)
                else:
                    raw = self._request(path, image_id, job_id)
                if isinstance(raw.get("error"), (str, dict)):
                    raise DetectionError("BACKEND_DETECTION_FAILED", "FacadeFixer returned an error instead of detections.",
                                         setup="Start the backend with its configured local detector weights.", model=self.model)
                defects, unmapped = self._normalize(raw, width, height, threshold, image_id, warnings)
                images.append({"index": index, "image_path": str(path), "image_id": image_id,
                               "source_sha256": sha, "width": width, "height": height,
                               "pixel_convention": "stored_pixels", "exif_orientation": orientation,
                               "defects": defects, "unmapped_defects": unmapped})
            supported_count = sum(len(image["defects"]) for image in images)
            unmapped_count = sum(len(image["unmapped_defects"]) for image in images)
            if method == "http":
                warnings.append("threshold is applied after inference; it cannot recover boxes removed by the detector's own threshold.")
            result = {"job_id": job_id, "status": "needs_review" if unmapped_count and not supported_count else "completed",
                      "schema": "bbox_class_only_v1", "coordinate_space": "original_pixels", "images": images,
                      "source": {"method": method, "threshold": threshold},
                      "provenance": {"adapter": "facade_mcp.detection_service", "annotations": annotation_artifact},
                      "review_required": bool(unmapped_count), "warnings": list(dict.fromkeys(warnings))}
            if method == "http":
                result["source"].update({"endpoint": self.backend_url.rstrip("/") + "/api/detect", "model": self.model,
                                         "enable_doubao_justification": False,
                                         "upload_representation": "png_from_stored_pixels_no_exif_rotation"})
            elif method == "local_yolo":
                result["source"].update({"model": "ultralytics_yolo", "device": self.local_device,
                                         "threshold_applied_at_inference": True,
                                         "weights_sha256": raw["model_provenance"]["weights_sha256"]})
                result["provenance"]["local_worker_outputs"] = local_artifacts
            manifest_path = Path(job_dir) / "detections.json"
            self.store.write_json(manifest_path, result)
            result["manifest_artifact"] = self.store.artifact(manifest_path, "detection_manifest", job_id)
            self.store.complete_job(job_id, result)
            return result
        except DetectionError as error:
            failure = error.as_dict()
        except (OSError, ValueError, TypeError, KeyError) as error:
            failure = {"code": "INVALID_INPUT", "message": str(error), "details": {}}
        self.store.fail_job(job_id, failure)
        return {"job_id": job_id, "status": "failed", "error": failure, "images": []}

    def _run_local_yolo(self, path: Path, image_id: str, image_sha: str,
                        job_id: str, index: int, threshold: float) -> tuple[dict, dict]:
        if not self.local_python or not self.local_weights:
            raise DetectionError("LOCAL_DETECTOR_NOT_CONFIGURED",
                                 "Set FACADE_DETECTOR_PYTHON and FACADE_DETECTOR_WEIGHTS for local YOLO.",
                                 required=["FACADE_DETECTOR_PYTHON", "FACADE_DETECTOR_WEIGHTS"])
        python_path = Path(self.local_python).expanduser().resolve()
        weights_path = Path(self.local_weights).expanduser().resolve()
        if not python_path.is_file():
            raise DetectionError("LOCAL_PYTHON_NOT_FOUND", "Configured local detector Python executable does not exist.")
        if not weights_path.is_file():
            raise DetectionError("LOCAL_WEIGHTS_NOT_FOUND", "Configured local detector weights do not exist; automatic download is disabled.")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise DetectionError("INVALID_TIMEOUT", "Detector timeout must be a finite positive number.")
        job_dir = self.store.job_dir(job_id)
        output = job_dir / f"local_detection_{index:04d}.json"
        # A fixed script and server-configured runtime are used, never shell text.
        command = [str(python_path), str(Path(__file__).with_name("yolo_worker.py")),
                   "--image", str(path), "--weights", str(weights_path),
                   "--output", str(output), "--threshold", str(threshold), "--device", self.local_device]
        environment = dict(os.environ)
        environment.update({"YOLO_AUTOINSTALL": "false", "YOLO_CONFIG_DIR": str(job_dir / "ultralytics_config"),
                            "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        try:
            completed = subprocess.run(command, shell=False, cwd=str(job_dir), env=environment,
                                       capture_output=True, timeout=self.timeout_seconds, check=False)
        except subprocess.TimeoutExpired:
            raise DetectionError("LOCAL_DETECTOR_TIMEOUT", "Local YOLO inference exceeded the configured timeout.",
                                 timeout_seconds=self.timeout_seconds) from None
        except OSError:
            raise DetectionError("LOCAL_DETECTOR_LAUNCH_FAILED", "Could not launch the configured detector Python runtime.") from None
        # stdout/stderr may contain library logs. They are captured and never
        # included in tool observations or persisted as possible secret output.
        if completed.returncode:
            details = {"returncode": completed.returncode,
                       "required": ["ultralytics and torch in configured runtime", "compatible local detector weights"]}
            if output.is_file():
                try:
                    error = json.loads(output.read_text(encoding="utf-8")).get("error", {})
                    details["worker_error_code"] = str(error.get("code", "INFERENCE_FAILED"))
                    details["worker_error_type"] = str(error.get("type", "UnknownError"))
                except (OSError, ValueError, TypeError, AttributeError):
                    pass
            raise DetectionError("LOCAL_DETECTOR_FAILED", "Local YOLO worker failed. Check its runtime and weight compatibility.", **details)
        if not output.is_file():
            raise DetectionError("LOCAL_DETECTOR_MISSING_OUTPUT", "Local YOLO worker did not produce its output JSON.")
        try:
            raw = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise DetectionError("INVALID_LOCAL_DETECTOR_OUTPUT", "Local YOLO worker output is not valid JSON.") from None
        if not isinstance(raw, dict) or not isinstance(raw.get("defects"), list) or not isinstance(raw.get("model_provenance"), dict):
            raise DetectionError("INVALID_LOCAL_DETECTOR_OUTPUT", "Local YOLO worker output is missing required fields.")
        if raw.get("source_sha256") != image_sha:
            raise DetectionError("LOCAL_DETECTOR_IMAGE_MISMATCH", "Local worker output does not match the input image hash.", image_id=image_id)
        return raw, self.store.artifact(output, "local_detector_output", job_id)

    def _read_annotations(self, value: str | None, job_id: str):
        if not value:
            raise DetectionError("ANNOTATIONS_REQUIRED", "annotations mode requires annotation_path; no detection is fabricated.")
        path = self.store.resolve_input(value)
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict) and isinstance(data.get("images"), list):
            records = data["images"]
        elif isinstance(data, list):
            records = data
        elif isinstance(data, dict) and ("defects" in data or "annotations_in_crop" in data):
            records = [data]
        else:
            raise DetectionError("INVALID_ANNOTATIONS", "Expected an images manifest or a single image annotation object.")
        resolved = []
        for record in records:
            if not isinstance(record, dict):
                raise DetectionError("INVALID_ANNOTATIONS", "Every image annotation must be an object.")
            record = dict(record)
            if isinstance(data, dict) and "coordinate_space" in data:
                record.setdefault("coordinate_space", data["coordinate_space"])
            ref = record.get("image_path")
            if ref:
                candidate = Path(ref)
                if not candidate.is_absolute() and not str(ref).startswith("artifact:"):
                    candidate = path.parent / candidate
                resolved_path = self.store.resolve_input(str(candidate))
                record["_resolved_path"] = str(resolved_path)
                record["_image_sha256"] = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
            elif not record.get("source_sha256", record.get("sha256")) and not record.get("crop_filename"):
                raise DetectionError("IMAGE_BINDING_REQUIRED", "Every annotation needs image_path, source_sha256, or crop_filename.")
            resolved.append(record)
        snapshot = self.store.job_dir(job_id) / "annotation_input.json"
        snapshot.write_bytes(path.read_bytes())
        return resolved, self.store.artifact(snapshot, "annotation_input", job_id)

    def _find_annotation(self, records: list[dict], path: Path, sha: str) -> dict:
        exact = [record for record in records if record.get("_resolved_path") == str(path)]
        if not exact:
            exact = [record for record in records if record.get("source_sha256", record.get("sha256")) == sha]
        if not exact:
            exact = [record for record in records if record.get("_image_sha256") == sha]
        if not exact:
            exact = [record for record in records if not record.get("_resolved_path") and record.get("crop_filename") == path.name]
        if len(exact) != 1:
            raise DetectionError("ANNOTATION_IMAGE_MISMATCH", "Each input image must match exactly one annotation record.", image=path.name, matches=len(exact))
        raw = exact[0]
        declared_sha = raw.get("source_sha256", raw.get("sha256"))
        if declared_sha and declared_sha != sha:
            raise DetectionError("STALE_ANNOTATION", "Annotation image hash does not match the current file.", image=path.name)
        if "defects" not in raw and "annotations_in_crop" not in raw:
            raise DetectionError("INVALID_ANNOTATIONS", "Matched annotation has no defects list.", image=path.name)
        return raw

    def _request(self, path: Path, image_id: str, job_id: str) -> dict:
        if not self.backend_url:
            raise DetectionError("BACKEND_NOT_CONFIGURED", "Set FACADE_DETECTOR_URL to the existing FacadeFixer backend URL.",
                                 required=["backend process", "local detector weights", "FACADE_DETECTOR_URL"])
        parsed = urlsplit(self.backend_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise DetectionError("INVALID_BACKEND_URL", "Backend URL must be HTTP(S), without credentials, query, or fragment.")
        if self.model not in LOCAL_MODELS:
            raise DetectionError("UNSUPPORTED_DETECTOR_MODEL", "Choose a supported local detector model.", allowed=sorted(LOCAL_MODELS))
        # The name is unique to this job; no GET /api/detection_result is used.
        # cv2.imdecode may rotate JPEGs according to EXIF. A PNG re-encoding of
        # the raw stored raster makes the detector and downstream mask geometry
        # agree, without resizing or transposing the image.
        encoded = io.BytesIO()
        with Image.open(path) as image:
            image.convert("RGB").save(encoded, format="PNG")
        payload = {"image_data": base64.b64encode(encoded.getvalue()).decode("ascii"),
                   "filename": f"{job_id}_{image_id}.png", "model": self.model,
                   "text_prompt": "Detect facade defects in the supplied image.", "enable_doubao_justification": False}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = Request(self.backend_url.rstrip("/") + "/api/detect", json.dumps(payload).encode("utf-8"), headers, method="POST")
        try:
            with build_opener(_NoRedirect()).open(request, timeout=self.timeout_seconds) as response:
                data = json.load(response)
        except HTTPError as error:
            raise DetectionError("BACKEND_HTTP_ERROR", "FacadeFixer detector HTTP request failed.",
                                 http_status=error.code, retryable=error.code >= 500) from None
        except (URLError, TimeoutError, socket.timeout):
            raise DetectionError("BACKEND_UNREACHABLE", "Cannot reach the FacadeFixer detector backend within the configured timeout.",
                                 required=["running backend", "network reachability"]) from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise DetectionError("INVALID_BACKEND_RESPONSE", "The detector did not return a JSON object.") from None
        if not isinstance(data, dict):
            raise DetectionError("INVALID_BACKEND_RESPONSE", "The detector did not return a JSON object.")
        if not data.get("error") and "annotations_in_crop" not in data:
            raise DetectionError("INVALID_BACKEND_RESPONSE", "The detector response is missing annotations_in_crop.")
        return data

    def _normalize(self, raw: dict, width: int, height: int, threshold: float,
                   image_id: str, warnings: list[str]):
        entries = raw.get("defects", raw.get("annotations_in_crop"))
        if not isinstance(entries, list):
            raise DetectionError("INVALID_ANNOTATIONS", "defects or annotations_in_crop must be a list.")
        if raw.get("coordinate_space", "original_pixels") not in {"original_pixels", "crop_pixels", "normalized_crop"}:
            raise DetectionError("UNKNOWN_COORDINATE_SPACE", "Unsupported annotation coordinate space.")
        for field, actual in (("width", width), ("height", height)):
            if field in raw and _number(raw[field], field) != actual:
                raise DetectionError("IMAGE_SIZE_MISMATCH", "Annotation image dimensions do not match the input image.", field=field)
        origin_x, origin_y, crop_w, crop_h = 0., 0., float(width), float(height)
        if "crop_coordinates_pixels" in raw:
            crop = raw["crop_coordinates_pixels"]
            if not isinstance(crop, dict):
                raise DetectionError("INVALID_GEOMETRY", "crop_coordinates_pixels must be an object.")
            origin_x, origin_y, end_x, end_y = [_number(crop.get(key), key) for key in ("x1", "y1", "x2", "y2")]
            if not (0 <= origin_x < end_x <= width and 0 <= origin_y < end_y <= height):
                raise DetectionError("INVALID_GEOMETRY", "Crop bounds must lie within the original image.")
            crop_w, crop_h = end_x - origin_x, end_y - origin_y
        for field, actual in (("crop_width_pixels", crop_w), ("crop_height_pixels", crop_h)):
            if field in raw and _number(raw[field], field) != actual:
                raise DetectionError("CROP_SIZE_MISMATCH", "Crop dimensions disagree with the declared crop coordinates.", field=field)
        defects, unmapped, seen_ids = [], [], set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise DetectionError("INVALID_ANNOTATIONS", "Every defect annotation must be an object.")
            confidence = entry.get("confidence")
            if confidence is not None:
                confidence = _number(confidence, "confidence")
                if not 0 <= confidence <= 1:
                    raise DetectionError("INVALID_CONFIDENCE", "confidence must be between 0 and 1.")
                if confidence < threshold:
                    continue
            supplied_id = entry.get("instance_id", entry.get("id", index + 1))
            instance_id = str(supplied_id)
            if instance_id in seen_ids:
                raise DetectionError("DUPLICATE_INSTANCE_ID", "Instance IDs must be unique within each image.")
            seen_ids.add(instance_id)
            bbox_float = self._bbox(entry, raw.get("coordinate_space", "original_pixels"), origin_x, origin_y, crop_w, crop_h, width, height)
            # Enclose the detector's continuous box in integer stored-image pixels.
            # Preserve its original floating coordinates for audit.
            bbox = [max(0,math.floor(bbox_float[0])),max(0,math.floor(bbox_float[1])),
                    min(width,math.ceil(bbox_float[2])),min(height,math.ceil(bbox_float[3]))]
            label = str(entry.get("original_class_name", entry.get("class_name", entry.get("label", ""))))
            supplied_code = entry.get("class_code")
            code = str(supplied_code).upper() if supplied_code is not None else LABEL_CODES.get(_label_key(label))
            item = {"instance_id": instance_id, "id": supplied_id, "bbox_xyxy": bbox,
                    "bbox_xyxy_float":bbox_float,"bbox_rounding":"floor_min_ceil_max", "original_class_name": label}
            if "class_id" in entry:
                item["original_class_id"] = entry["class_id"]
            if confidence is not None:
                item["confidence"] = confidence
            if code not in CLASS_NAMES:
                item["reason"] = "unsupported_class"
                unmapped.append(item)
                warnings.append(f"{image_id}: unsupported class {label or supplied_code!r} preserved for review; it will not be segmented.")
                continue
            item.update({"class_code": code, "class_name": CLASS_NAMES[code]})
            if entry.get("class_subtype"):
                item["class_subtype"] = str(entry["class_subtype"])
            defects.append(item)
            if _label_key(label) == "waterstain":
                warnings.append("Water_Stain mapped to L as a moisture indicator; the presence and cause of leakage require review.")
        return defects, unmapped

    @staticmethod
    def _bbox(entry: dict, coordinate_space: str, ox: float, oy: float,
              cw: float, ch: float, width: int, height: int) -> list[float]:
        if coordinate_space not in {"original_pixels","crop_pixels","normalized_crop"}:
            raise DetectionError("INVALID_GEOMETRY", "Unsupported coordinate_space.")
        if "bbox_xyxy" in entry:
            box = entry["bbox_xyxy"]
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise DetectionError("INVALID_GEOMETRY", "bbox_xyxy must contain four numbers.")
            x1, y1, x2, y2 = [_number(v, "bbox_xyxy") for v in box]
            if coordinate_space == "crop_pixels":
                if not (0 <= x1 < x2 <= cw and 0 <= y1 < y2 <= ch):
                    raise DetectionError("INVALID_GEOMETRY", "Crop-pixel bbox must lie within its crop.")
                x1, x2, y1, y2 = x1 + ox, x2 + ox, y1 + oy, y2 + oy
            elif coordinate_space == "normalized_crop":
                if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                    raise DetectionError("INVALID_GEOMETRY", "Normalized bbox must lie within [0, 1].")
                x1, x2, y1, y2 = ox + x1 * cw, ox + x2 * cw, oy + y1 * ch, oy + y2 * ch
        else:
            keys = ("x_center_norm", "y_center_norm", "width_norm", "height_norm")
            if not all(key in entry for key in keys):
                raise DetectionError("INVALID_GEOMETRY", "Defect needs bbox_xyxy or all four normalized center fields.")
            cx, cy, bw, bh = [_number(entry[key], key) for key in keys]
            epsilon = 1e-6
            if not (bw > 0 and bh > 0 and -epsilon <= cx - bw / 2 < cx + bw / 2 <= 1 + epsilon and
                    -epsilon <= cy - bh / 2 < cy + bh / 2 <= 1 + epsilon):
                raise DetectionError("INVALID_GEOMETRY", "Normalized center bbox must lie within its crop.")
            x1, y1 = ox + (cx - bw / 2) * cw, oy + (cy - bh / 2) * ch
            x2, y2 = ox + (cx + bw / 2) * cw, oy + (cy + bh / 2) * ch
        epsilon = 1e-6
        if not (-epsilon <= x1 < x2 <= width + epsilon and -epsilon <= y1 < y2 <= height + epsilon):
            raise DetectionError("INVALID_GEOMETRY", "Defect bbox must have positive area and lie within the original image.", bbox=[x1, y1, x2, y2])
        return [round(max(0., x1), 6), round(max(0., y1), 6), round(min(float(width), x2), 6), round(min(float(height), y2), 6)]
