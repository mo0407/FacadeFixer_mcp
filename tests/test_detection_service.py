import hashlib
import base64
import io
import json
import shutil
import threading
import subprocess
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

from facade_mcp.detection_service import DetectionService
from facade_mcp.storage import ArtifactStore


@pytest.fixture(autouse=True)
def clean_detector_configuration(monkeypatch):
    for name in ("FACADE_DETECTOR_URL", "FACADE_DETECTOR_TOKEN", "FACADE_DETECTOR_MODEL"):
        monkeypatch.delenv(name, raising=False)
    for name in ("FACADE_DETECTOR_PYTHON", "FACADE_DETECTOR_WEIGHTS", "FACADE_DETECTOR_DEVICE", "FACADE_DETECTOR_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "facade.png"
    Image.new("RGB", (200, 120), (180, 180, 180)).save(path)
    return path


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "workspace", [tmp_path])


@pytest.fixture
def detector_server():
    state = {"status": 200, "response": {"annotations_in_crop": []}, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append({"path": self.path, "body": data, "authorization": self.headers.get("Authorization")})
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = state["response"]
            self.wfile.write(response if isinstance(response, bytes) else json.dumps(response).encode())

        def do_GET(self):
            state["requests"].append({"path": self.path, "method": "GET"})
            self.send_response(404)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def write_manifest(tmp_path: Path, image_path: Path, defects: list, **image_fields):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps({"schema": "bbox_class_only_v1", "images": [
        {"image_path": str(image_path), "defects": defects, **image_fields}
    ]}), encoding="utf-8")
    return path


def annotation(code="C", confidence=None, **overrides):
    result = {"id": 7, "class_code": code, "bbox_xyxy": [10, 20, 60, 90]}
    if confidence is not None:
        result["confidence"] = confidence
    return {**result, **overrides}


def test_import_preserves_real_boxes_and_publishes_job_artifact(tmp_path, image_path, store):
    manifest = write_manifest(tmp_path, image_path, [annotation()])
    result = DetectionService(store).detect([str(image_path)], annotation_path=str(manifest))
    assert result["status"] == "completed"
    defect = result["images"][0]["defects"][0]
    assert defect["bbox_xyxy"] == [10, 20, 60, 90]
    assert defect["id"] == 7 and defect["instance_id"] == "7"
    assert result["images"][0]["source_sha256"] == hashlib.sha256(image_path.read_bytes()).hexdigest()
    artifact = result["manifest_artifact"]
    assert store.resolve_input(artifact["uri"]) == Path(artifact["path"])
    assert store.job_status(result["job_id"])["result"]["manifest_artifact"] == artifact
    assert Path(result["provenance"]["annotations"]["path"]).is_relative_to(store.job_dir(result["job_id"]))


def test_annotations_are_required_and_empty_supplied_list_is_valid(tmp_path, image_path, store):
    missing = DetectionService(store).detect([str(image_path)])
    assert missing["status"] == "failed" and missing["error"]["code"] == "ANNOTATIONS_REQUIRED"
    manifest = write_manifest(tmp_path, image_path, [])
    empty = DetectionService(store).detect([str(image_path)], annotation_path=str(manifest))
    assert empty["status"] == "completed" and empty["images"][0]["defects"] == []


def test_archival_copy_binds_annotations_by_actual_image_hash(tmp_path, image_path, store):
    manifest = write_manifest(tmp_path, image_path, [annotation()])
    copied_path = tmp_path / "archived_f08a.png"
    shutil.copyfile(image_path, copied_path)
    result = DetectionService(store).detect([str(copied_path)], annotation_path=str(manifest))
    assert result["status"] == "completed"
    assert result["images"][0]["image_path"] == str(copied_path)


def test_stale_annotations_are_rejected(tmp_path, image_path, store):
    manifest = write_manifest(tmp_path, image_path, [annotation()], source_sha256="0" * 64)
    result = DetectionService(store).detect([str(image_path)], annotation_path=str(manifest))
    assert result["status"] == "failed" and result["error"]["code"] == "STALE_ANNOTATION"
    assert "manifest_artifact" not in result


def test_cropped_normalized_backend_boxes_are_returned_in_original_pixels(image_path, store, detector_server):
    url, state = detector_server
    state["response"] = {
        "crop_coordinates_pixels": {"x1": 20, "y1": 30, "x2": 120, "y2": 90},
        "crop_width_pixels": 100, "crop_height_pixels": 60,
        "annotations_in_crop": [
            {"class_name": "Concrete_Crack", "confidence": .9,
             "x_center_norm": .5, "y_center_norm": .5, "width_norm": .4, "height_norm": .5},
            {"class_name": "Contaminants", "confidence": .1,
             "x_center_norm": .5, "y_center_norm": .5, "width_norm": .4, "height_norm": .5},
        ],
    }
    result = DetectionService(store, url).detect([str(image_path)], method="http", threshold=.5)
    assert result["status"] == "completed"
    defects = result["images"][0]["defects"]
    assert len(defects) == 1
    assert defects[0]["bbox_xyxy"] == [50, 45, 90, 75]
    assert defects[0]["class_code"] == "C" and defects[0]["original_class_name"] == "Concrete_Crack"
    request = state["requests"][0]
    assert request["path"] == "/api/detect"
    assert request["body"]["enable_doubao_justification"] is False
    assert request["body"]["model"] == "yolo12m"
    assert result["job_id"] in request["body"]["filename"]
    assert len(state["requests"]) == 1  # no stale-result GET or auto retry


def test_unknown_classes_are_preserved_unsegmented_for_review(tmp_path, image_path, store):
    manifest = write_manifest(tmp_path, image_path, [annotation(code="unknown", class_name="Bulging")])
    result = DetectionService(store).detect([str(image_path)], annotation_path=str(manifest))
    assert result["status"] == "needs_review" and result["review_required"]
    assert result["images"][0]["defects"] == []
    unknown = result["images"][0]["unmapped_defects"][0]
    assert unknown["original_class_name"] == "Bulging"
    assert unknown["reason"] == "unsupported_class" and "class_code" not in unknown
    assert unknown["bbox_xyxy"] == [10, 20, 60, 90]


def test_water_stain_mapping_retains_diagnostic_uncertainty(image_path, store, detector_server):
    url, state = detector_server
    state["response"] = {"annotations_in_crop": [
        {"class_name": "Water_Stain", "confidence": .9,
         "x_center_norm": .5, "y_center_norm": .5, "width_norm": .2, "height_norm": .2}
    ]}
    result = DetectionService(store, url).detect([str(image_path)], method="http")
    assert result["images"][0]["defects"][0]["class_code"] == "L"
    assert any("cause of leakage require review" in warning for warning in result["warnings"])


def test_backend_http_failure_is_structured_and_does_not_leak_token(image_path, store, detector_server):
    url, state = detector_server
    state["status"] = 500
    state["response"] = {"detail": "private-test-token-123"}
    result = DetectionService(store, url, token="private-test-token-123").detect([str(image_path)], method="http")
    assert result["status"] == "failed" and result["error"]["code"] == "BACKEND_HTTP_ERROR"
    assert result["error"]["details"]["http_status"] == 500
    assert "private-test-token-123" not in json.dumps(result)
    assert "private-test-token-123" not in json.dumps(store.job_status(result["job_id"]))
    assert state["requests"][0]["authorization"] == "Bearer private-test-token-123"


def test_missing_backend_configuration_is_actionable(image_path, store):
    result = DetectionService(store).detect([str(image_path)], method="http")
    assert result["status"] == "failed" and result["error"]["code"] == "BACKEND_NOT_CONFIGURED"
    assert "FACADE_DETECTOR_URL" in result["error"]["details"]["required"]


@pytest.mark.parametrize("response,code", [
    ({"error": "YOLO model not loaded"}, "BACKEND_DETECTION_FAILED"),
    ({"other": []}, "INVALID_BACKEND_RESPONSE"),
    (b"<html>invalid response</html>", "INVALID_BACKEND_RESPONSE"),
])
def test_backend_missing_models_and_malformed_response_are_not_empty_success(image_path, store, detector_server, response, code):
    url, state = detector_server
    state["response"] = response
    result = DetectionService(store, url).detect([str(image_path)], method="http")
    assert result["status"] == "failed" and result["error"]["code"] == code


def test_geometry_and_threshold_validation(tmp_path, image_path, store):
    manifest = write_manifest(tmp_path, image_path, [annotation(bbox_xyxy=[-1, 20, 60, 90])])
    invalid_box = DetectionService(store).detect([str(image_path)], annotation_path=str(manifest))
    assert invalid_box["error"]["code"] == "INVALID_GEOMETRY"
    invalid_threshold = DetectionService(store).detect([str(image_path)], annotation_path=str(manifest), threshold=float("nan"))
    assert invalid_threshold["status"] == "failed" and invalid_threshold["error"]["code"] == "INVALID_GEOMETRY"


def test_crop_pixel_annotation_conversion(tmp_path, image_path, store):
    manifest = write_manifest(tmp_path, image_path, [annotation(bbox_xyxy=[10, 5, 50, 40])],
                              coordinate_space="crop_pixels",
                              crop_coordinates_pixels={"x1": 20, "y1": 30, "x2": 120, "y2": 90})
    result = DetectionService(store).detect([str(image_path)], annotation_path=str(manifest))
    assert result["images"][0]["defects"][0]["bbox_xyxy"] == [30, 35, 70, 70]


def test_exif_orientation_does_not_change_detector_coordinate_system(tmp_path, store, detector_server):
    url, state = detector_server
    path = tmp_path / "camera_photo.jpg"
    exif = Image.Exif()
    exif[274] = 6  # Display-time clockwise rotation; stored pixels are 200 x 120.
    Image.new("RGB", (200, 120), (150, 150, 150)).save(path, exif=exif)
    state["response"] = {
        "crop_coordinates_pixels": {"x1": 0, "y1": 0, "x2": 200, "y2": 120},
        "crop_width_pixels": 200, "crop_height_pixels": 120,
        "annotations_in_crop": [
            {"class_name": "Concrete_Crack", "confidence": .9,
             "x_center_norm": .5, "y_center_norm": .5, "width_norm": .2, "height_norm": .2}
        ],
    }
    result = DetectionService(store, url).detect([str(path)], method="http")
    assert result["status"] == "completed"
    upload_bytes = base64.b64decode(state["requests"][0]["body"]["image_data"])
    with Image.open(io.BytesIO(upload_bytes)) as upload:
        assert upload.format == "PNG" and upload.size == (200, 120)
        assert upload.getexif().get(274, 1) == 1
    record = result["images"][0]
    assert record["exif_orientation"] == 6
    assert record["pixel_convention"] == "stored_pixels"
    assert record["defects"][0]["bbox_xyxy"] == [80, 48, 120, 72]
    assert record["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def configure_local_runtime(tmp_path, monkeypatch):
    executable = tmp_path / "trusted_model_python.exe"
    executable.write_bytes(b"mock runtime")
    weights = tmp_path / "existing_detector.pt"
    weights.write_bytes(b"mock weights")
    monkeypatch.setenv("FACADE_DETECTOR_PYTHON", str(executable))
    monkeypatch.setenv("FACADE_DETECTOR_WEIGHTS", str(weights))
    monkeypatch.setenv("FACADE_DETECTOR_DEVICE", "cpu")
    return executable, weights


def test_local_yolo_runs_fixed_worker_without_shell_and_preserves_output(tmp_path, image_path, store, monkeypatch):
    executable, weights = configure_local_runtime(tmp_path, monkeypatch)
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(command=command, **kwargs)
        output = Path(command[command.index("--output") + 1])
        image = Path(command[command.index("--image") + 1])
        output.write_text(json.dumps({"width": 200, "height": 120,
            "coordinate_space": "original_pixels", "source_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "defects": [{"id": 1, "class_name": "Concrete_Crack", "class_id": 0, "confidence": .9,
                         "bbox_xyxy": [10.3, 20.8, 60.1, 90.2]}],
            "model_provenance": {"weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout=b"private-library-log", stderr=b"private-library-log")

    monkeypatch.setattr("facade_mcp.detection_service.subprocess.run", fake_run)
    result = DetectionService(store, timeout_seconds=22).detect([str(image_path)], method="local_yolo", threshold=.4)
    assert result["status"] == "completed"
    assert captured["command"][0] == str(executable)
    assert captured["command"][1].endswith("yolo_worker.py")
    assert captured["command"][captured["command"].index("--threshold") + 1] == "0.4"
    assert captured["shell"] is False and captured["capture_output"] is True and captured["timeout"] == 22
    assert captured["env"]["YOLO_AUTOINSTALL"] == "false"
    assert Path(captured["env"]["YOLO_CONFIG_DIR"]).is_relative_to(store.job_dir(result["job_id"]))
    defect = result["images"][0]["defects"][0]
    assert defect["bbox_xyxy"] == [10, 20, 61, 91]
    assert defect["bbox_xyxy_float"] == [10.3, 20.8, 60.1, 90.2]
    assert result["source"]["threshold_applied_at_inference"] is True
    artifact = result["provenance"]["local_worker_outputs"][0]
    assert store.resolve_input(artifact["uri"]).is_file()
    assert "private-library-log" not in json.dumps(result)
    assert "private-library-log" not in json.dumps(store.job_status(result["job_id"]))


def test_local_yolo_requires_explicit_existing_runtime_and_weights(tmp_path, image_path, store, monkeypatch):
    result = DetectionService(store).detect([str(image_path)], method="local_yolo")
    assert result["error"]["code"] == "LOCAL_DETECTOR_NOT_CONFIGURED"
    _, weights = configure_local_runtime(tmp_path, monkeypatch)
    weights.unlink()
    monkeypatch.setattr("facade_mcp.detection_service.subprocess.run", lambda *a, **k: pytest.fail("Must not launch with missing weights"))
    result = DetectionService(store).detect([str(image_path)], method="local_yolo")
    assert result["error"]["code"] == "LOCAL_WEIGHTS_NOT_FOUND"


def test_local_yolo_timeout_is_structured(tmp_path, image_path, store, monkeypatch):
    configure_local_runtime(tmp_path, monkeypatch)

    def timed_out(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=b"private-library-log")

    monkeypatch.setattr("facade_mcp.detection_service.subprocess.run", timed_out)
    result = DetectionService(store, timeout_seconds=10).detect([str(image_path)], method="local_yolo")
    assert result["error"]["code"] == "LOCAL_DETECTOR_TIMEOUT"
    assert result["error"]["details"]["timeout_seconds"] == 10
    assert "private-library-log" not in json.dumps(result)


@pytest.mark.parametrize("scenario,code", [
    ("missing", "LOCAL_DETECTOR_MISSING_OUTPUT"),
    ("invalid_json", "INVALID_LOCAL_DETECTOR_OUTPUT"),
    ("wrong_hash", "LOCAL_DETECTOR_IMAGE_MISMATCH"),
    ("worker_failed", "LOCAL_DETECTOR_FAILED"),
])
def test_local_yolo_rejects_missing_malformed_stale_or_failed_worker_output(tmp_path, image_path, store, monkeypatch, scenario, code):
    configure_local_runtime(tmp_path, monkeypatch)

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        if scenario == "invalid_json":
            output.write_text("broken-json", encoding="utf-8")
        elif scenario == "wrong_hash":
            output.write_text(json.dumps({"defects": [], "model_provenance": {}, "source_sha256": "0" * 64}), encoding="utf-8")
        elif scenario == "worker_failed":
            output.write_text(json.dumps({"error": {"code": "LOCAL_YOLO_INFERENCE_FAILED", "type": "ImportError"}}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 1 if scenario == "worker_failed" else 0, stdout=b"", stderr=b"sensitive log")

    monkeypatch.setattr("facade_mcp.detection_service.subprocess.run", fake_run)
    result = DetectionService(store).detect([str(image_path)], method="local_yolo")
    assert result["status"] == "failed" and result["error"]["code"] == code
    assert "sensitive log" not in json.dumps(result)


def test_yolo_worker_uses_native_threshold_model_labels_and_raw_pixels(tmp_path, monkeypatch):
    from facade_mcp import yolo_worker
    image = tmp_path / "exif_photo.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (200, 120)).save(image, exif=exif)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"existing local weights")
    captured = {}
    box = types.SimpleNamespace(cls=[types.SimpleNamespace(item=lambda: 0)],
                                conf=[types.SimpleNamespace(item=lambda: .8)],
                                xyxy=[types.SimpleNamespace(tolist=lambda: [10.5, 20.5, 60.5, 90.5])])

    class FakeYOLO:
        names = {0: "Concrete_Crack"}

        def __init__(self, path, task):
            captured.update(weights_path=path, task=task)

        def predict(self, **kwargs):
            captured.update(kwargs)
            return [types.SimpleNamespace(boxes=[box])]

    monkeypatch.setitem(__import__("sys").modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO, __version__="test"))
    result = yolo_worker.run(image, weights, .4, "cpu")
    assert captured["conf"] == .4 and captured["verbose"] is False and captured["device"] == "cpu"
    assert captured["source"].size == (200, 120)
    assert captured["source"].getexif().get(274, 1) == 1
    assert result["defects"][0]["class_name"] == "Concrete_Crack"
    assert result["defects"][0]["bbox_xyxy"] == [10.5, 20.5, 60.5, 90.5]
    assert result["model_provenance"]["weights_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert result["model_provenance"]["automatic_weight_download"] is False


def test_yolo_worker_checks_weights_before_importing_or_downloading(tmp_path):
    from facade_mcp import yolo_worker
    with pytest.raises(FileNotFoundError, match="automatic download is disabled"):
        yolo_worker.run(tmp_path / "image.png", tmp_path / "nonexistent.pt", .25, "cpu")
