"""Contract, artifact and failure boundaries of managed source-model jobs."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from facade_mcp.model_service import ModelService, PROFILES
from facade_mcp.storage import ArtifactStore


@pytest.fixture
def setup(tmp_path):
    repo = tmp_path / "source_repo"
    (repo / "model_runner").mkdir(parents=True)
    (repo / "backend").mkdir()
    for profile in PROFILES.values():
        for name in profile["weights"]:
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture checkpoint, not a model")
    image = tmp_path / "facade.png"
    Image.new("RGB", (80, 60), (175, 175, 175)).save(image)
    service = ModelService(ArtifactStore(tmp_path / "workspace", [tmp_path]), repo, Path(sys.executable))
    yield service, image
    service.masks.close()


def simulated_worker(monkeypatch, service, *, escape=False, empty=False):
    """Only model inference is stubbed; real PNG/artifact/export logic is used."""
    def run(directory, action, **arguments):
        directory.mkdir(parents=True, exist_ok=True)
        if action == "detect":
            result = {"status": "completed", "defects": [
                {"instance_id": "a", "class_name": "Rust_Stain", "bbox_xyxy": [5.2, 6.2, 25.4, 36.4], "confidence": .9},
                {"instance_id": "b", "class_name": "Bulging", "bbox_xyxy": [40, 10, 70, 35], "confidence": .8}],
                "model_provenance": {"weights_sha256": "fake-for-test-only"}}
        else:
            entries = []
            for index, row in enumerate(arguments["defects"]):
                mask = np.zeros((60, 80), dtype=np.uint8)
                x1, y1, x2, y2 = row["bbox_xyxy"]
                if not empty:
                    mask[y1+1:y2-1, x1+1:x2-1] = 255
                path = (directory.parent if escape else directory) / f"mask_{index}.png"
                Image.fromarray(mask).save(path)
                entries.append({"instance_id": row["instance_id"], "mask_path": str(path), "score": .8})
            result = {"status": "completed", "instances": entries, "model_provenance": {"profile": arguments["model_profile"]}}
        service.store.write_json(directory / "worker_output.json", result)
        return result
    monkeypatch.setattr(service, "_run", run)


def defects():
    return [{"instance_id": "a", "class_code": "C", "bbox_xyxy": [5, 6, 25, 36]},
            {"instance_id": "b", "class_code": "T", "bbox_xyxy": [40, 10, 70, 35]}]


def test_prompt_comparison_keeps_candidates_and_requires_paid_consent(setup, monkeypatch):
    service, image = setup
    simulated_worker(monkeypatch, service)
    rejected = service.segment(str(image), defects(), "sam3", prompt_mode="doubao")
    assert rejected["status"] == "failed" and "allow_paid_api" in rejected["error"]["message"]
    result = service.segment(str(image), defects(), "sam3", prompt_mode="compare")
    assert result["status"] == "needs_selection"
    assert set(result["candidates"]) == {"bbox", "point", "text"}
    assert result["selection"]["api_requests"] == 0
    text = json.loads(Path(result["candidates"]["text"]["manifest"]["path"]).read_text(encoding="utf-8"))
    assert text["provenance"]["text_prompts"] == {"a": "crack", "b": "rust stain"}


def test_selector_exports_selected_manifest_and_preserves_candidates(setup, monkeypatch):
    from facade_mcp import prompt_selection
    service, image = setup
    simulated_worker(monkeypatch, service)
    monkeypatch.setattr(prompt_selection, "selector_config", lambda: ("https://example.invalid", "test", "test"))
    monkeypatch.setattr(prompt_selection, "judge", lambda *args: {"selected_mode": "point", "reason": "fixture", "api_requests": 1})
    result = service.segment(str(image), defects(), "sam3", prompt_mode="doubao", allow_paid_api=True)
    assert "instances" in result
    assert result["provenance"]["prompt_mode"] == "point"
    assert result["provenance"]["prompt_selection"]["selection"]["selected_mode"] == "point"
    assert result["metrics"]["actual_charge"] is None
    manifest = json.loads(Path(result["annotation_manifest"]["path"]).read_text(encoding="utf-8"))
    assert manifest["job_id"] == result["job_id"]


def test_text_custom_prompts_and_unsupported_mode(setup, monkeypatch):
    service, image = setup
    simulated_worker(monkeypatch, service)
    result = service.segment(str(image), defects(), "sam3", prompt_mode="text", text_prompts={"a": "thin crack"})
    assert result["provenance"]["text_prompts"]["a"] == "thin crack"
    assert service.segment(str(image), defects(), "mobile_sam", prompt_mode="text")["status"] == "failed"
    assert service.segment(str(image), defects(), "sam3", prompt_mode="text", text_prompts={"missing": "x"})["status"] == "failed"


@pytest.mark.parametrize("code,profile,expected", [("M", "unet_crack", "C"), ("E", "mobile_sam", "E")])
def test_model_masks_use_canonical_classes(setup, monkeypatch, code, profile, expected):
    service, image = setup
    simulated_worker(monkeypatch, service)
    item = {**defects()[0], "class_code": code}
    result = service.segment(str(image), [item], profile)
    assert result["instances"][0]["class_code"] == expected
    assert item["class_code"] == code


def test_detect_preserves_unknowns_rounding_snapshot_and_provenance(setup, monkeypatch):
    service, image = setup
    simulated_worker(monkeypatch, service)
    result = service.detect([str(image)])
    assert result["status"] == "completed"
    row = result["images"][0]
    assert row["defects"][0]["bbox_xyxy"] == [5, 6, 26, 37]
    assert row["defects"][0]["class_code"] == "T"
    assert row["unmapped_defects"][0]["original_class_name"] == "Bulging"
    assert row["source_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert Path(row["image_path"]) != image
    assert service.get_job(result["job_id"])["result"]["manifest_artifact"] == result["manifest_artifact"]
    assert result["expert_validated"] is False


def test_real_export_png_rle_union_empty_and_pending(setup, monkeypatch):
    service, image = setup
    simulated_worker(monkeypatch, service)
    result = service.segment(str(image), defects())
    assert result["status"] == "candidate_requires_review"
    arrays = [np.asarray(Image.open(row["mask_path"])) > 0 for row in result["instances"]]
    union = np.logical_or.reduce(arrays)
    rle = result["segmentation_rle"]
    decoded = np.repeat(np.arange(len(rle["counts"])) % 2, rle["counts"]).reshape(rle["size"], order="F") > 0
    assert np.array_equal(decoded, union)
    assert result["area_pixels"] == int(union.sum())
    assert result["physical_area_m2"] is None and result["expert_validated"] is False
    assert result["metrics"]["api_requests"] == 0
    simulated_worker(monkeypatch, service, empty=True)
    empty = service.segment(str(image), defects())
    assert empty["area_pixels"] == 0 and len(empty["instances"]) == 2
    assert all("empty_or_tiny_foreground" in row["quality_flags"] for row in empty["instances"])


def test_refinement_preserves_other_instances_and_immutable_parent(setup, monkeypatch):
    service, image = setup
    simulated_worker(monkeypatch, service)
    first = service.segment(str(image), defects())
    parent_path = Path(first["annotation_manifest_path"])
    parent_bytes = parent_path.read_bytes()
    refined = service.refine(str(image), first["annotation_manifest"]["uri"], "a", [[10, 15]], [[20, 25]])
    assert refined["status"] == "candidate_requires_review", refined
    assert refined["revision"] == 2
    assert parent_path.read_bytes() == parent_bytes
    assert first["instances"][1]["mask"]["sha256"] == refined["instances"][1]["mask"]["sha256"]
    assert refined["provenance"]["model_mask_used_as_input"] is False
    assert refined["provenance"]["events"]["positive_points"] == [[10, 15]]
    assert refined["provenance"]["parent_manifest"]["sha256"] == hashlib.sha256(parent_bytes).hexdigest()


@pytest.mark.parametrize("mutation,code", [("different_image", "SOURCE_HASH_MISMATCH"), ("changed_mask", "Artifact content changed"), ("outside_points", "INVALID_POINTS")])
def test_refinement_rejects_stale_geometry_and_hashes(setup, monkeypatch, mutation, code):
    service, image = setup
    simulated_worker(monkeypatch, service)
    first = service.segment(str(image), defects())
    positive = [[10, 15]]
    if mutation == "different_image":
        Image.new("RGB", (80, 60), "black").save(image)
    elif mutation == "changed_mask":
        Image.new("L", (80, 60), 0).save(first["instances"][0]["mask_path"])
    else:
        positive = [[80, 60]]
    result = service.refine(str(image), first["annotation_manifest"]["uri"], "a", positive, [])
    assert result["status"] == "failed"
    assert code in result["error"]["code"] or code in result["error"]["message"]


def test_worker_cannot_return_a_mask_outside_job(setup, monkeypatch):
    service, image = setup
    simulated_worker(monkeypatch, service, escape=True)
    result = service.segment(str(image), defects())
    assert result["status"] == "failed" and result["error"]["code"] == "MODEL_OUTPUT_ESCAPE"


def test_model_failure_is_not_no_defects(setup, monkeypatch):
    service, image = setup
    def failed(*args, **kwargs):
        raise RuntimeError("model failed to load")
    monkeypatch.setattr(service, "_run", failed)
    result = service.detect([str(image)])
    assert result["status"] == "failed" and result["images"] == []
    assert service.get_job(result["job_id"])["status"] == "failed"


def test_profiles_class_scope_and_input_allowlist(setup):
    service, image = setup
    for threshold in (float("nan"), True, 1.5):
        assert service.detect([str(image)], threshold=threshold)["error"]["code"] == "INVALID_THRESHOLD"
    assert service.detect([str(image)], "../../arbitrary.py")["error"]["code"] == "UNKNOWN_MODEL_PROFILE"
    assert service.segment(str(image), defects(), "unet_crack")["error"]["code"] == "UNSUPPORTED_CLASS"
    assert service.detect([str(Path(sys.executable))])["status"] == "failed"


def test_subprocess_excludes_credentials_and_timeout_does_not_retry(setup, monkeypatch):
    service, image = setup
    monkeypatch.setenv("OPENAI_API_KEY", "secret-fixture")
    monkeypatch.setenv("FACADE_LLM_API_KEY", "another-secret-fixture")
    calls = []
    def timeout(command, **kwargs):
        calls.append(kwargs)
        assert "OPENAI_API_KEY" not in kwargs["env"] and "FACADE_LLM_API_KEY" not in kwargs["env"]
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(subprocess, "run", timeout)
    result = service.detect([str(image)])
    assert len(calls) == 1
    assert result["error"]["code"] == "MODEL_TIMEOUT"


def test_real_probe_does_not_claim_model_load(setup):
    service, _ = setup
    result = service.list_tools()
    assert result["status"] == "configured", result
    assert len(result["models"]) == 9
    assert all(row["model_load_verified"] is False for row in result["models"])
    frcnn = next(row for row in result["models"] if row["model_profile"] == "faster_rcnn")
    assert frcnn["availability"] == "unavailable" and "detectron2" in frcnn["missing_packages"]


def test_bad_timeout_is_a_job_failure_and_does_not_prevent_server_start(setup, monkeypatch):
    service, image = setup
    monkeypatch.setenv("FACADE_SOURCE_TIMEOUT_SECONDS", "abc")
    configured = ModelService(service.store, service.repo, service.python)
    try:
        assert configured.list_tools()["error"]["code"] == "INVALID_TIMEOUT"
        assert configured.detect([str(image)])["error"]["code"] == "INVALID_TIMEOUT"
    finally:
        configured.masks.close()


@pytest.mark.parametrize("operation", ["detect", "segment"])
def test_mutating_inference_snapshot_is_rejected(setup, monkeypatch, operation):
    service, image = setup
    def mutate(directory, action, **arguments):
        Image.new("RGB", (80, 60), "black").save(arguments["image_path"])
        return {"status": "completed", "defects": [], "instances": []}
    monkeypatch.setattr(service, "_run", mutate)
    result = service.detect([str(image)]) if operation == "detect" else service.segment(str(image), defects())
    assert result["error"]["code"] == "SOURCE_CHANGED"
