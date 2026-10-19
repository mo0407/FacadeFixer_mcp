"""Fixed isolated worker for an already installed Ultralytics runtime.

Executed by detection_service with explicit local weights. This file deliberately
does not depend on facade_mcp or import the original FastAPI application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(image_path: Path, weights_path: Path, threshold: float, device: str) -> dict:
    if not weights_path.is_file():
        raise FileNotFoundError("Configured weights do not exist; automatic download is disabled.")
    if not image_path.is_file():
        raise FileNotFoundError("Input image does not exist.")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from PIL import Image
    import ultralytics
    from ultralytics import YOLO

    # Create a metadata-free RGB raster. Ultralytics receives the same stored
    # pixel orientation and dimensions as annotation imports and mask generation.
    with Image.open(image_path) as original:
        raster = Image.new("RGB", original.size)
        raster.paste(original.convert("RGB"))
    source_sha = file_sha256(image_path)
    weights_sha = file_sha256(weights_path)
    model = YOLO(str(weights_path), task="detect")
    results = model.predict(source=raster, conf=threshold, device=device, verbose=False,
                            save=False, save_txt=False, save_crop=False)
    defects = []
    names = model.names
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            if isinstance(names, dict):
                label = names.get(class_id)
            else:
                label = names[class_id] if 0 <= class_id < len(names) else None
            if label is None:
                label = f"unmapped_model_class_{class_id}"
            defects.append({"id": len(defects) + 1, "class_id": class_id,
                            "class_name": str(label), "confidence": float(box.conf[0].item()),
                            "bbox_xyxy": [float(value) for value in box.xyxy[0].tolist()]})
    return {"coordinate_space": "original_pixels", "pixel_convention": "stored_pixels",
            "width": raster.width, "height": raster.height, "source_sha256": source_sha,
            "defects": defects, "model_provenance": {
                "library": "ultralytics", "library_version": ultralytics.__version__,
                "weights_sha256": weights_sha, "device": device, "confidence_threshold": threshold,
                "automatic_weight_download": False, "automatic_dependency_install": False}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        result = run(args.image.resolve(), args.weights.resolve(), args.threshold, args.device)
        returncode = 0
    except Exception as error:
        # No env values, provider keys, or captured library logs are printed.
        result = {"error": {"code": "LOCAL_YOLO_INFERENCE_FAILED", "type": type(error).__name__}}
        returncode = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, args.output)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
