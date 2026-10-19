import hashlib
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from facade_mcp import source_detection_worker as worker


@pytest.fixture
def local_input(tmp_path):
    repo = tmp_path / "source"
    for relative in worker.WEIGHTS.values():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    image = tmp_path / "camera.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (30, 20), (220, 10, 30)).save(image, exif=exif)
    return repo, image, tmp_path / "output"


def test_stored_pixels_bgr_labels_and_provenance_are_preserved(local_input, monkeypatch):
    repo, image, output = local_input
    adapter_source = repo / "model_runner/adapters/detection/yolo.py"
    adapter_source.parent.mkdir(parents=True)
    adapter_source.write_text("# source loader\n")
    calls = []

    class Model:
        def predict(self, **kwargs):
            calls.append(kwargs)
            boxes = [SimpleNamespace(cls=np.array([10]), conf=np.array([.86]), xyxy=np.array([[-1., 2., 10.5, 18.]])),
                     SimpleNamespace(cls=np.array([23]), conf=np.array([.73]), xyxy=np.array([[11., 2., 29., 18.]]))]
            return [SimpleNamespace(names={10: "Bulging", 23: "new_custom_class"}, boxes=boxes)]

    monkeypatch.setattr(worker, "_source_adapter", lambda *args: (Model(), adapter_source))
    result = worker.run_detection(repo, "yolo12m", image, output, .25, "cpu")
    assert (result["width"], result["height"]) == (30, 20)  # EXIF never rotates pixels.
    assert calls[0]["source"].shape == (20, 30, 3)
    assert calls[0]["source"][0, 0, 2] > calls[0]["source"][0, 0, 0]  # RGB red -> BGR red.
    assert calls[0]["device"] == "cpu" and calls[0]["save"] is False
    assert [d["class_name"] for d in result["defects"]] == ["Bulging", "new_custom_class"]
    assert result["defects"][0]["bbox_xyxy"] == [0., 2., 10.5, 18.]
    provenance = result["model_provenance"]
    assert provenance["source_image_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert provenance["weights_sha256"] == hashlib.sha256((repo / worker.WEIGHTS["yolo12m"]).read_bytes()).hexdigest()
    assert provenance["source_files_sha256"][adapter_source.relative_to(repo).as_posix()] == hashlib.sha256(adapter_source.read_bytes()).hexdigest()
    assert json.loads(Path(result["raw_result_path"]).read_text())["defects"] == result["defects"]


def test_missing_weights_fail_before_any_loader_and_no_output(local_input, monkeypatch):
    repo, image, output = local_input
    (repo / worker.WEIGHTS["yolo12m"]).unlink()
    monkeypatch.setattr(worker, "_source_adapter", lambda *args: pytest.fail("Cannot download absent weights."))
    with pytest.raises(FileNotFoundError):
        worker.run_detection(repo, "yolo12m", image, output, .25, "cpu")
    assert not output.exists()


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -.1, 1.1, True])
def test_invalid_thresholds_are_rejected(local_input, threshold):
    with pytest.raises(ValueError):
        worker.run_detection(*local_input[:1], "yolo12m", *local_input[1:], threshold, "cpu")


def test_offline_worker_blocks_network_and_restores_socket():
    original = socket.socket.connect
    with worker._offline():
        with socket.socket() as client, pytest.raises(RuntimeError, match="Network access"):
            client.connect(("127.0.0.1", 1))
    assert socket.socket.connect is original


def test_intersection_requires_different_models_and_preserves_empty_result():
    def item(model, box, confidence=.9):
        return {"class_name": "Concrete_Crack", "class_id": 0, "bbox_xyxy": box,
                "confidence": confidence, "source_model": model}
    duplicate = [item("yolo12m", [1, 1, 20, 20]), item("yolo12m", [2, 2, 20, 20], .8)]
    assert worker._aggregate(duplicate, "intersection") == []
    combined = worker._aggregate(duplicate + [item("rtdetr", [3, 3, 20, 20])], "intersection")
    assert combined[0]["contributing_models"] == ["rtdetr", "yolo12m"]
    assert combined[0]["bbox_xyxy"] == [2, 2, 20, 20]
    assert worker._aggregate([], "ensemble") == []


def test_ensemble_does_not_silently_drop_failed_models(local_input, monkeypatch):
    repo, image, output = local_input
    called = []

    def run(repo, profile, *args):
        called.append(profile)
        if profile == "yolo11m":
            raise RuntimeError("inference failed")
        return [], {}

    monkeypatch.setattr(worker, "_run_single", run)
    with pytest.raises(RuntimeError, match="inference failed"):
        worker.run_detection(repo, "ensemble", image, output, .25, "cpu")
    assert called == ["yolo12m", "yolo11m"] and not (output / "source_detection.json").exists()


def test_intersection_records_adapted_membership_and_actual_votes(local_input, monkeypatch):
    repo, image, output = local_input
    source = repo / "backend/detection_agent.py"
    source.parent.mkdir()
    source.write_text("# original pure aggregation logic\n")

    def run(repo, profile, *args):
        predictions = [] if profile == "rtdetr" else [
            {"class_name": "Bulging", "class_id": 10, "bbox_xyxy": [1, 1, 20, 15],
             "confidence": .8, "source_model": profile}]
        return predictions, {"profile": profile}

    monkeypatch.setattr(worker, "_run_single", run)
    result = worker.run_detection(repo, "intersection", image, output, .25, "cpu")
    assert result["defects"][0]["class_name"] == "Bulging"
    assert result["defects"][0]["contributing_models"] == ["yolo11m", "yolo12m"]
    assert "class_id" not in result["defects"][0]
    provenance = result["model_provenance"]
    assert provenance["required_models"] == list(worker.PRIMARY_PROFILES)
    assert provenance["source_integration"] == "adapted_consensus_not_legacy_equivalent"
    assert provenance["model_failures_allowed"] is False


def test_frcnn_missing_dependency_is_an_explicit_failure(local_input, monkeypatch):
    repo, image, output = local_input
    monkeypatch.setitem(sys.modules, "detectron2", None)
    with pytest.raises(RuntimeError, match="requires detectron2"):
        worker.run_detection(repo, "faster_rcnn", image, output, .25, "cpu")
    assert not (output / "source_detection.json").exists()


def test_input_change_during_inference_is_rejected(local_input, monkeypatch):
    repo, image, output = local_input

    def run(*args):
        image.write_bytes(b"replaced while inference ran")
        return [], {"weights_sha256": "unused", "source_files_sha256": {}, "library": "test", "library_version": "1"}

    monkeypatch.setattr(worker, "_run_single", run)
    with pytest.raises(RuntimeError, match="image changed"):
        worker.run_detection(repo, "yolo12m", image, output, .25, "cpu")
    assert not (output / "source_detection.json").exists()


def test_non_finite_model_output_cannot_be_reported_as_no_defects():
    with pytest.raises(ValueError, match="non-finite"):
        worker._prediction("Concrete_Crack", 0, .9, [1, 2, float("nan"), 10], 30, 20, "yolo12m")


def test_source_files_cannot_escape_repository(local_input, tmp_path):
    repo, *_ = local_input
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    with pytest.raises(FileNotFoundError):
        worker._local_file(repo, "../outside.pt")


def test_source_import_refuses_cached_other_repository(local_input, monkeypatch):
    repo, *_ = local_input
    adapter_source = repo / "model_runner/adapters/detection/yolo.py"
    adapter_source.parent.mkdir(parents=True)
    adapter_source.write_text("# source loader\n")
    monkeypatch.setitem(sys.modules, "model_runner.adapters.detection.yolo", SimpleNamespace(__file__=str(repo.parent / "other.py")))
    with pytest.raises(RuntimeError, match="different FacadeFixer"):
        worker._source_adapter(repo, "yolo12m", repo / worker.WEIGHTS["yolo12m"])
