"""Isolated, auditable adapters for the FacadeFixer mask repository.

Coordinates always refer to stored image pixels: xyxy, zero based, max exclusive.
The service imports algorithms by absolute file path without modifying sys.path,
never reuses old experiment folders and never substitutes a bbox for a failed mask.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .storage import ArtifactStore
from .taxonomy import CLASS_NAMES, normalize_class, opencv_code

LOG = logging.getLogger(__name__)
_ALGORITHM_LOCK = threading.RLock()
_COLORS = [(255, 60, 60), (50, 220, 80), (50, 150, 255),
           (255, 190, 30), (220, 60, 255), (40, 230, 225)]
_DEFINITIONS = {
    "C": "visible linear or network cracks at actual width; keep thin branches; exclude joints and shadows",
    "P": "peeling coating/plaster including light curled flakes",
    "S": "material loss including exposed mortar, brick, aggregate and cavities",
    "T": "rust stain on the surface", "R": "visible corrosion/rust",
    "D": "surface contaminants; distinguish ground litter from wall stains using subtype",
    "V": "visible plant including fine leaves",
    "L": "visible water stains and damp areas; exclude white mineral efflorescence",
    "E": "visible white salt/mineral deposits (efflorescence); exclude paint, glare and damp stains",
}


def _sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _rle(mask: np.ndarray) -> dict[str, Any]:
    flat = mask.astype(np.uint8).ravel(order="F")
    cuts = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.r_[0, cuts, flat.size]).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)
    return {"size": list(mask.shape), "order": "F", "counts": counts}


def mask_tool_result(result: dict) -> dict:
    """Keep pixel RLE in artifact JSON rather than filling an agent's context window."""
    def compact(value):
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items()
                    if key not in ("segmentation_rle", "defects")}
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value
    return compact(result)


class MaskService:
    """A service owned configuration selects providers; tool arguments cannot set URLs."""

    def __init__(self, store: ArtifactStore, repo_path: Path | None = None,
                 model_config_path: Path | None = None):
        self.store = store
        configured_repo = repo_path or os.getenv("FACADE_MASK_REPO")
        self.repo_path = Path(configured_repo).expanduser().resolve() if configured_repo else None
        configured_models = model_config_path or os.getenv("FACADE_MASK_MODELS_CONFIG")
        self.model_config_path = Path(configured_models).expanduser().resolve() if configured_models else None
        self._modules: dict[str, Any] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="facade-mask-api")

    def close(self) -> None:
        """Finish outstanding API jobs before shutting down a process."""
        self._executor.shutdown(wait=True)

    def _module(self, relative_path: str) -> Any:
        if not self.repo_path:
            raise ValueError("Set FACADE_MASK_REPO to the FacadeFixer_mask_api checkout")
        path = (self.repo_path / relative_path).resolve()
        if not path.is_relative_to(self.repo_path) or not path.is_file():
            raise ValueError(f"Mask repository is missing {relative_path}")
        with _ALGORITHM_LOCK:
            if relative_path not in self._modules:
                name = "_facade_mask_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
                spec = importlib.util.spec_from_file_location(name, path)
                if spec is None or spec.loader is None:
                    raise ValueError("Cannot load mask algorithm")
                module = importlib.util.module_from_spec(spec)
                with redirect_stdout(sys.stderr):
                    spec.loader.exec_module(module)
                self._modules[relative_path] = module
            return self._modules[relative_path]

    def validate_input(self, image_path: str, defects: list[dict]) -> dict:
        errors: list[str] = []
        normalized: list[dict] = []
        width = height = None
        resolved = None
        orientation = None
        source_sha256 = None
        try:
            resolved = self.store.resolve_input(image_path)
            if not resolved.is_file():
                raise ValueError("Image does not exist")
            with Image.open(resolved) as image:
                width, height = image.size
                orientation = image.getexif().get(274, 1)
                image.load()
            if width < 1 or height < 1:
                raise ValueError("Image has no pixels")
            source_sha256 = _sha(resolved)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"image: {exc}")
        if not isinstance(defects, list):
            errors.append("defects must be a list")
        else:
            seen = set()
            for index, defect in enumerate(defects):
                prefix = f"defects[{index}]"
                if not isinstance(defect, dict):
                    errors.append(f"{prefix} must be an object")
                    continue
                defect = normalize_class(defect)
                instance_id = defect.get("instance_id", defect.get("id"))
                if (isinstance(instance_id, bool) or not isinstance(instance_id, (int, str))
                        or not str(instance_id).strip()):
                    errors.append(f"{prefix}.instance_id must be a nonempty string or integer")
                    continue
                if str(instance_id) in seen:
                    errors.append(f"{prefix}.instance_id must be unique")
                    continue
                seen.add(str(instance_id))
                code = defect.get("class_code")
                if not isinstance(code, str) or code not in CLASS_NAMES:
                    errors.append(f"{prefix}.class_code must be one of {', '.join(CLASS_NAMES)}")
                    continue
                box = defect.get("bbox_xyxy")
                if (not isinstance(box, (list, tuple)) or len(box) != 4
                        or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) or int(v) != v for v in box)):
                    errors.append(f"{prefix}.bbox_xyxy must contain four integer pixel coordinates")
                    continue
                box = [int(v) for v in box]
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 and 0 <= y1 < y2
                        and (width is None or x2 <= width) and (height is None or y2 <= height)):
                    errors.append(f"{prefix}.bbox_xyxy must be inside the original image")
                    continue
                # Explicitly reject geometric priors in the automatic bbox/class interface.
                if any(key in defect for key in ("mask_path", "mask", "segmentation_rle",
                                                 "contour", "polygon", "prior_mask")):
                    errors.append(f"{prefix}: mask/contour priors belong to mask_refine")
                    continue
                subtype = defect.get("class_subtype", "")
                if subtype and (code != "D" or subtype not in ("GroundLitter", "WallStain")):
                    errors.append(f"{prefix}.class_subtype must be GroundLitter or WallStain for D")
                    continue
                normalized.append({"instance_id": instance_id, "class_code": code,
                                   "class_name": str(defect.get("class_name") or CLASS_NAMES[code]),
                                   "class_subtype": subtype, "bbox_xyxy": box,
                                   **{k: defect[k] for k in ("original_class_code", "original_class_name") if k in defect}})
        flags = ["stored_pixels_used_without_exif_rotation"] if orientation not in (None, 1) else []
        return {"valid": not errors, "errors": errors, "image_path": str(resolved) if resolved else None,
                "width": width, "height": height, "source_sha256": source_sha256, "defects": normalized,
                "bbox_convention": "zero_based_xyxy_max_exclusive", "quality_flags": flags,
                "estimated_api_requests": len(normalized), "supported_classes": CLASS_NAMES}

    def _validated(self, image_path: str, defects: list[dict]) -> dict:
        result = self.validate_input(image_path, defects)
        if not result["valid"]:
            raise ValueError("; ".join(result["errors"]))
        return result

    def segment(self, image_path: str, defects: list[dict], method: str = "opencv",
                model_profile: str | None = None) -> dict:
        """Run a new job. API requests occur only when method is explicitly api."""
        if method not in ("opencv", "api"):
            raise ValueError("method must be opencv or api")
        validated = self._validated(image_path, defects)
        if method == "opencv" and model_profile:
            raise ValueError("model_profile is only used with method=api")
        profile = self._profile(model_profile) if method == "api" else None
        job_id, job_dir = self.store.create_job("mask_" + method, {
            "image_path": validated["image_path"], "defects": validated["defects"],
            "method": method, "model_profile": model_profile})
        try:
            result = self._run(job_id, job_dir, validated, method, profile)
            self.store.complete_job(job_id, result)
            return result
        except Exception as exc:
            # Provider errors may include request headers: persist only a safe error summary.
            error = self._safe_error(exc, method)
            self.store.fail_job(job_id, error)
            raise RuntimeError(f"Mask job {job_id} failed: {error}") from None

    def start_api_job(self, image_path: str, defects: list[dict], model_profile: str) -> dict:
        validated = self._validated(image_path, defects)
        profile = self._profile(model_profile)
        job_id, job_dir = self.store.create_job("mask_api", {
            "image_path": validated["image_path"], "defects": validated["defects"],
            "method": "api", "model_profile": model_profile})

        def run() -> None:
            try:
                result = self._run(job_id, job_dir, validated, "api", profile)
                self.store.complete_job(job_id, result)
            except Exception as exc:
                self.store.fail_job(job_id, self._safe_error(exc, "api"))
                LOG.error("Mask API job %s failed (%s)", job_id, type(exc).__name__)

        self._executor.submit(run)
        return {"job_id": job_id, "status": "submitted", "method": "api",
                "estimated_api_requests": len(validated["defects"])}

    def get_job(self, job_id: str) -> dict:
        return self.store.job_status(job_id)

    def get_artifact(self, annotation_manifest_path: str) -> dict:
        path = self.store.resolve_input(annotation_manifest_path)
        result = json.loads(path.read_text(encoding="utf-8-sig"))
        if result.get("schema_version") != "facade-mask/v1":
            raise ValueError("Expected a facade-mask/v1 annotation manifest")
        return result

    @staticmethod
    def _safe_error(exc: Exception, method: str) -> str:
        if method == "api":
            status = getattr(exc, "status", None)
            return f"{type(exc).__name__}" + (f" HTTP {status}" if status is not None else "")
        return str(exc)[:1000]

    def _profile(self, model_profile: str | None) -> dict:
        if not isinstance(model_profile, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", model_profile):
            raise ValueError("Choose a configured model_profile name; URLs are not tool arguments")
        if not self.model_config_path or not self.model_config_path.is_file():
            raise ValueError("Configure FACADE_MASK_MODELS_CONFIG before requesting API masks")
        config = json.loads(self.model_config_path.read_text(encoding="utf-8-sig"))
        rows = config.get("models", [])
        matches = [row for row in rows if row.get("id") == model_profile and row.get("enabled", True)]
        if len(matches) != 1:
            raise ValueError("model_profile must identify one enabled server configuration")
        model = dict(matches[0])
        parsed = urlsplit(model.get("base_url", ""))
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Configured provider must have an HTTPS URL without embedded credentials")
        if model.get("api_key") or model.get("key_file") or model.get("headers"):
            raise ValueError("Provider secrets must use api_key_env; inline keys and headers are unsupported")
        env_name = model.get("api_key_env")
        if not isinstance(env_name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
            raise ValueError("Provider config requires api_key_env")
        key = os.getenv(env_name)
        if not key or "\n" in key or "\r" in key:
            raise ValueError("Configured API key environment variable is unset or invalid")
        if not isinstance(model.get("model"), str) or not model["model"].strip():
            raise ValueError("Provider config requires a model name")
        if model.get("transport", "multipart") not in ("multipart", "json", "dashscope"):
            raise ValueError("Unsupported provider transport")
        max_side = model.get("max_canvas_side", 1024)
        if isinstance(max_side, bool) or not isinstance(max_side, int) or not 256 <= max_side <= 8192:
            raise ValueError("max_canvas_side must be an integer from 256 to 8192")
        model["max_canvas_side"] = max_side
        return model

    def _run(self, job_id: str, job_dir: Path, validated: dict, method: str,
             profile: dict | None = None) -> dict:
        started = time.perf_counter()
        source = Path(validated["image_path"])
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        snapshot = input_dir / ("source" + source.suffix.lower())
        shutil.copyfile(source, snapshot)
        with Image.open(snapshot) as image:
            rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        if (width, height) != (validated["width"], validated["height"]):
            raise ValueError("Source dimensions changed after validation")
        sha = _sha(snapshot)
        if sha != validated["source_sha256"]:
            raise ValueError("Source image changed after validation")
        image_id = "image_" + sha[:20]
        instances = []
        masks = []
        usage = []
        fees = []
        for index, defect in enumerate(validated["defects"]):
            tick = time.perf_counter()
            box = defect["bbox_xyxy"]
            if method == "opencv":
                module = self._module("skills/facade-opencv-mask/scripts/segment.py")
                with _ALGORITHM_LOCK, redirect_stdout(sys.stderr):
                    cv2.setRNGSeed(20260928)
                    patch, diagnostics = module.segment(rgb, box, opencv_code(defect),
                                                        defect.get("class_subtype", ""), 1000)
                if defect["class_code"] == "E":
                    diagnostics.setdefault("parameters", {}).update(legacy_algorithm_class="L", canonical_class="E")
                    diagnostics.setdefault("quality_flags", []).append("generic_deposit_baseline_requires_efflorescence_review")
                x1, y1, x2, y2 = box
                if patch.shape != (y2 - y1, x2 - x1):
                    raise ValueError("Segmenter returned incorrect ROI dimensions")
                mask = np.zeros((height, width), dtype=bool)
                mask[y1:y2, x1:x2] = patch > 0
                model_details = None
            else:
                mask, diagnostics, model_details = self._api_instance(snapshot, rgb, defect, profile,
                                                                       job_dir, index)
                usage.append(model_details["usage"])
                fees.append(model_details["estimated_cost"])
            flags = list(diagnostics.get("quality_flags", []))
            if not mask.any() and "empty_or_tiny_foreground" not in flags:
                flags.append("empty_or_tiny_foreground")
            if defect["class_code"] == "D" and not defect.get("class_subtype"):
                flags.append("contamination_subtype_unspecified_default_wall_stain")
            masks.append(mask)
            dest = job_dir / "instance_masks" / f"instance_{index:05d}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(dest)
            artifact = self.store.artifact(dest, "instance_mask", job_id)
            instances.append({**defect, "image_id": image_id, "mask": artifact,
                              "mask_artifact": artifact, "mask_path": str(dest),
                              "area_pixels": int(mask.sum()), "segmentation_rle": _rle(mask),
                              "method": method, "algorithm": diagnostics.get("method"),
                              "parameters": diagnostics.get("parameters", {}),
                              "roi_scale": diagnostics.get("roi_scale", 1.0),
                              "quality_flags": flags, "expert_validated": False,
                              "seconds": time.perf_counter() - tick, "model": model_details})
        original_artifact = self.store.artifact(snapshot, "original_image", job_id)
        result = self._export(job_id, job_dir, rgb, instances, masks, original_artifact,
                              image_id, source, sha, validated["quality_flags"])
        algorithm_path = ("skills/facade-opencv-mask/scripts/segment.py" if method == "opencv"
                          else "scripts/api_adapter.py")
        algorithm_file = self.repo_path / algorithm_path if self.repo_path else None
        result["provenance"] = {"method": method, "source_repository": "https://github.com/mo0407/FacadeFixer_mask_api",
                                "algorithm_path": algorithm_path,
                                "algorithm_sha256": _sha(algorithm_file) if algorithm_file else None,
                                "opencv_version": cv2.__version__, "model_profile": profile.get("id") if profile else None,
                                "model_name": profile.get("model") if profile else None,
                                "reference_mask_used": False, "human_guides_used": False,
                                "model_mask_used_as_input": False, "reused_previous_results": False}
        result["metrics"] = {"seconds": time.perf_counter() - started,
                             "api_requests": len(instances) if method == "api" else 0,
                             "api_usage": usage if method == "api" else [],
                             "estimated_cost": (sum(fees) if fees and all(v is not None for v in fees) else None)
                             if method == "api" else 0,
                             "actual_charge": None if method == "api" else 0,
                             "currency": (profile.get("pricing") or {}).get("currency") if profile else None,
                             "cost_scope": "API usage only; excludes local compute and human review"}
        return self._manifest(job_id, job_dir, result)

    def _api_instance(self, source: Path, rgb: np.ndarray, defect: dict, profile: dict,
                      job_dir: Path, index: int) -> tuple[np.ndarray, dict, dict]:
        adapter = self._module("scripts/api_adapter.py")
        original, png, geometry = adapter.prepare(source, None, profile["max_canvas_side"])
        width, height = original.size
        canvas_width, canvas_height = geometry["canvas_size"]
        submitted_width, submitted_height = geometry["input_size"]
        ox, oy, _, _ = geometry["crop"]
        prompt = ("Return ONLY one binary segmentation mask. White is the specified defect instance; "
                  "black is all other pixels. Preserve full image geometry without crop, zoom, labels "
                  f"or redraw. Original {width}x{height}; canvas {submitted_width}x{submitted_height}. "
                  f"Original point maps as x_input=(x+{ox})*{submitted_width/canvas_width}, "
                  f"y_input=(y+{oy})*{submitted_height/canvas_height}. Magenta padding is background. "
                  "Target in original pixel coordinates: " + json.dumps(defect, ensure_ascii=False)
                  + ". Definition: " + _DEFINITIONS[defect["class_code"]]
                  + ". Segment only this target inside the supplied zero-based xyxy bbox, right/bottom "
                  "exclusive. Keep irregular boundaries, thin cracks and holes. If no visible target, return black.")
        raw_dir = job_dir / "api_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"submitted_{index:05d}.png").write_bytes(png)
        self.store.write_json(raw_dir / f"geometry_{index:05d}.json", {**geometry, "prompt": prompt})
        minimum_interval = float(profile.get("min_interval_seconds", 0))
        if minimum_interval < 0 or minimum_interval > 60:
            raise ValueError("Configured min_interval_seconds must be between 0 and 60")
        # Do not persist keys, provider URLs, proxies, headers, or complete provider responses.
        with _ALGORITHM_LOCK, redirect_stdout(sys.stderr):
            if index and minimum_interval:
                time.sleep(minimum_interval)
            asset, api_usage, _ = adapter.call_api(profile, os.environ[profile["api_key_env"]], png, prompt)
            blob = adapter.asset_bytes(asset, profile.get("proxy"))
            (raw_dir / f"returned_{index:05d}.bin").write_bytes(blob)
            raw_mask = raw_dir / f"unclipped_{index:05d}.png"
            adapter.write_images(original, blob, geometry, raw_mask, raw_dir / f"overlay_{index:05d}.png")
        with Image.open(raw_mask) as image:
            generated = np.asarray(image.convert("L")) > 0
        if generated.shape != rgb.shape[:2]:
            raise ValueError("API output mask dimensions do not match original image")
        x1, y1, x2, y2 = defect["bbox_xyxy"]
        clipped = np.zeros_like(generated)
        clipped[y1:y2, x1:x2] = generated[y1:y2, x1:x2]
        info = {"method": "api_per_instance_bbox", "quality_flags": ["generated_mask_may_shift",
                "clipped_to_frozen_bbox"], "parameters": {"binary_threshold": adapter.THRESHOLD,
                "outside_bbox_pixels": int(generated.sum() - clipped.sum()),
                "input_size": geometry["input_size"], "canvas_size": geometry["canvas_size"]}}
        # Numeric usage metadata only, deliberately omit opaque provider request ids.
        clean_usage = {key: value for key, value in adapter.usage_fields(api_usage).items()
                       if value is None or isinstance(value, (int, float))}
        model = {"profile": profile["id"], "name": profile["model"], "usage": clean_usage,
                 "estimated_cost": adapter.cost(api_usage, profile.get("pricing")), "actual_charge": None}
        return clipped, info, model

    def _export(self, job_id: str, job_dir: Path, rgb: np.ndarray, instances: list,
                masks: list, original_artifact: dict, image_id: str, source: Path,
                source_sha: str, flags: list) -> dict:
        height, width = rgb.shape[:2]
        union = np.zeros((height, width), dtype=bool)
        overlay = rgb.copy()
        bbox_image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(bbox_image)
        class_masks: dict[str, np.ndarray] = {}
        output_dir = job_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, (record, mask) in enumerate(zip(instances, masks)):
            if mask.shape != union.shape:
                raise ValueError("Mask dimensions do not match original image")
            x1, y1, x2, y2 = record["bbox_xyxy"]
            inside = np.zeros_like(union)
            inside[y1:y2, x1:x2] = True
            if (mask & ~inside).any():
                raise ValueError("Instance mask escapes its frozen bbox")
            color = _COLORS[index % len(_COLORS)]
            union |= mask
            class_masks.setdefault(record["class_code"], np.zeros_like(union))
            class_masks[record["class_code"]] |= mask
            overlay[mask] = np.rint(rgb[mask] * .55 + np.array(color) * .45).astype(np.uint8)
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=max(1, min(width, height) // 250))
            draw.text((x1, max(0, y1 - 13)), f"{record['instance_id']} {record['class_code']}", fill=color)
            record["overlay_color_rgb"] = list(color)
            crop_path = output_dir / f"instance_{index+1:04d}_crop.png"
            Image.fromarray(rgb[y1:y2,x1:x2]).save(crop_path)
            record["crop"] = self.store.artifact(crop_path,"crop",job_id)
            record["crop_origin_xy"] = [x1,y1]
            local_overlay = rgb.copy()
            local_overlay[mask] = np.rint(rgb[mask]*.55+np.array(color)*.45).astype(np.uint8)
            instance_overlay = output_dir / f"instance_{index+1:04d}_overlay.png"
            Image.fromarray(local_overlay).save(instance_overlay)
            record["overlay"] = self.store.artifact(instance_overlay,"overlay",job_id)
        union_path, overlay_path, bbox_path = output_dir / "union_mask.png", output_dir / "overlay.png", output_dir / "bbox_overlay.png"
        Image.fromarray(union.astype(np.uint8) * 255).save(union_path)
        Image.fromarray(overlay).save(overlay_path)
        bbox_image.save(bbox_path)
        class_artifacts = {}
        for code, mask in class_masks.items():
            class_path = output_dir / f"class_{code}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(class_path)
            class_artifacts[code] = self.store.artifact(class_path, "class_mask", job_id)
        union_artifact = self.store.artifact(union_path, "union_mask", job_id)
        overlay_artifact = self.store.artifact(overlay_path, "overlay", job_id)
        return {"schema_version": "facade-mask/v1", "job_id": job_id,
                "status": "candidate_requires_review" if instances else "no_defects_provided",
                "image_id": image_id, "image_path": str(source), "source_path": str(source),
                "source_sha256": source_sha, "width": width, "height": height,
                "original_image": original_artifact, "instances": instances, "defects": instances,
                "union_mask": union_artifact, "overlay": overlay_artifact,
                "bbox_overlay": self.store.artifact(bbox_path, "bbox_overlay", job_id),
                "class_masks": class_artifacts, "mask_path": str(union_path), "overlay_path": str(overlay_path),
                "area_pixels": int(union.sum()), "foreground_pixels": int(union.sum()),
                "foreground_fraction": float(union.mean()), "segmentation_rle": _rle(union),
                "bbox_convention": "zero_based_xyxy_max_exclusive", "expert_validated": False,
                "physical_area_m2": None, "quality_flags": list(flags) + ["requires_visual_or_expert_review"],
                "review": {"expert_validated": False, "pixel_accuracy_verified": False}}

    def _manifest(self, job_id: str, job_dir: Path, result: dict) -> dict:
        manifest = job_dir / "annotation_manifest.json"
        self.store.write_json(manifest, result)
        result["annotation_manifest"] = self.store.artifact(manifest, "annotation_manifest", job_id)
        result["annotation_manifest_path"] = str(manifest)
        return result

    def refine(self, image_path: str, annotation_manifest_path: str) -> dict:
        """Offline hybrid refinement of explicit candidate masks, preserving raw outputs."""
        manifest_path = self.store.resolve_input(annotation_manifest_path)
        source_manifest = self.get_artifact(str(manifest_path))
        source = self.store.resolve_input(image_path)
        if _sha(source) != source_manifest.get("source_sha256"):
            raise ValueError("Refinement source image hash does not match the annotation manifest")
        rows = source_manifest["instances"]
        defects = [{key: row[key] for key in ("instance_id", "class_code", "class_name", "bbox_xyxy")}
                   | {"class_subtype": row.get("class_subtype", "")}
                   | {k: row[k] for k in ("original_class_code", "original_class_name") if k in row} for row in rows]
        validated = self._validated(str(source), defects)
        module = self._module("scripts/refine_api_masks.py")
        job_id, job_dir = self.store.create_job("mask_refine", {
            "image_path": str(source), "annotation_manifest_path": str(manifest_path)})
        try:
            started = time.perf_counter()
            input_dir = job_dir / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            snapshot = input_dir / ("source" + source.suffix.lower())
            shutil.copyfile(source, snapshot)
            rgb = np.asarray(Image.open(snapshot).convert("RGB"))
            masks, instances = [], []
            for index, (row, defect) in enumerate(zip(rows, validated["defects"])):
                raw_path = self.store.resolve_input(row.get("mask_path") or "artifact:" + row["mask"]["artifact_id"])
                raw = module.binary(raw_path)
                if row.get("mask", {}).get("sha256") and _sha(raw_path) != row["mask"]["sha256"]:
                    raise ValueError("Input mask hash has changed")
                with _ALGORITHM_LOCK, redirect_stdout(sys.stderr):
                    selected, candidate, diagnostics = module.refine(rgb, raw, defect["bbox_xyxy"], opencv_code(defect))
                if defect["class_code"] == "E":
                    diagnostics.update(legacy_algorithm_class="L", canonical_class="E")
                    diagnostics.setdefault("flags", []).append("generic_deposit_baseline_requires_efflorescence_review")
                paths = {}
                for folder, array in (("raw_masks", raw), ("candidate_masks", candidate), ("instance_masks", selected)):
                    dest = job_dir / folder / f"instance_{index:05d}.png"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(array.astype(np.uint8) * 255).save(dest)
                    paths[folder] = self.store.artifact(dest, folder, job_id)
                masks.append(selected)
                instances.append({**defect, "image_id": source_manifest["image_id"],
                                  "mask": paths["instance_masks"], "mask_path": paths["instance_masks"]["path"],
                                  "raw_mask": paths["raw_masks"], "candidate_mask": paths["candidate_masks"],
                                  "area_pixels": int(selected.sum()), "segmentation_rle": _rle(selected),
                                  "method": "hybrid_opencv_refinement", "refinement": diagnostics,
                                  "quality_flags": diagnostics.get("flags", []), "expert_validated": False})
            original = self.store.artifact(snapshot, "original_image", job_id)
            result = self._export(job_id, job_dir, rgb, instances, masks, original,
                                  source_manifest["image_id"], source, _sha(snapshot), validated["quality_flags"])
            result["provenance"] = {"method": "hybrid_opencv_refinement",
                                    "source_annotation_sha256": _sha(manifest_path),
                                    "algorithm_sha256": _sha(self.repo_path / "scripts/refine_api_masks.py"),
                                    "parameters": module.PARAMS, "reference_mask_used": False,
                                    "model_mask_used_as_input": source_manifest.get("provenance", {}).get("method") in ("api", "hybrid_opencv_refinement"),
                                    "candidate_mask_used_as_input": True, "human_guides_used": False,
                                    "source_job_id": source_manifest["job_id"]}
            result["metrics"] = {"seconds": time.perf_counter() - started,
                                 "additional_api_requests": 0, "additional_api_cost": 0}
            result = self._manifest(job_id, job_dir, result)
            self.store.complete_job(job_id, result)
            return result
        except Exception as exc:
            self.store.fail_job(job_id, str(exc)[:1000])
            raise RuntimeError(f"Refinement job {job_id} failed: {exc}") from None
