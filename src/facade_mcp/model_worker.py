"""Private subprocess boundary for the approved FacadeFixer local models.

This entry point is not an MCP tool and accepts only server-generated requests.
It imports neither the legacy backend singleton nor the MCP/LLM dependencies.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
from pathlib import Path


def probe() -> dict:
    packages = ("torch", "torchvision", "ultralytics", "detectron2", "cv2", "PIL", "timm")
    available = {name: importlib.util.find_spec(name) is not None for name in packages}
    runtime = {"status": "completed", "python_version": os.sys.version.split()[0],
               "packages": available, "cuda_available": False,
               "model_load_verified": False}
    if available["torch"]:
        try:
            import torch
            runtime.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available())
        except Exception as error:
            runtime.update(status="failed", error={"code": "TORCH_IMPORT_FAILED", "message": str(error)})
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def offline(*arguments, **keywords):
        raise RuntimeError("Network access is disabled in local model workers.")
    socket.socket.connect = offline
    socket.socket.connect_ex = offline
    socket.create_connection = offline
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        action = request["action"]
        if action == "probe":
            result = probe()
        elif action == "detect":
            from source_detection_worker import run_detection
            result = run_detection(Path(request["repo"]), request["model_profile"],
                                   Path(request["image_path"]), Path(request["output_dir"]),
                                   request["threshold"], request["device"])
        elif action == "segment":
            from source_mask_worker import run_segmentation
            result = run_segmentation(Path(request["repo"]), request["model_profile"],
                                      Path(request["image_path"]), Path(request["output_dir"]),
                                      request["defects"], request["device"],
                                      request.get("positive_points"), request.get("negative_points"))
        else:
            raise ValueError("Unsupported worker action")
    except Exception as error:
        result = {"status": "failed", "error": {"code": "MODEL_EXECUTION_FAILED",
                  "type": type(error).__name__, "message": str(error)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
