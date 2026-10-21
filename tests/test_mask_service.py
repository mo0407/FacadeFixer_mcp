"""Offline integration tests: real OpenCV, geometry/RLE, isolation and failure handling."""
from __future__ import annotations

import hashlib
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import pytest

from facade_mcp.mask_service import MaskService
from facade_mcp.storage import ArtifactStore


@pytest.fixture
def service(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    configured = os.environ.get("FACADE_MASK_REPO")
    vendor_repo = project_root / "vendor" / "FacadeFixer_mask_api"
    default_repo = vendor_repo if vendor_repo.is_dir() else project_root.parent / ".task_tmp" / "mask-mcp-feasibility"
    repo = Path(configured) if configured else default_repo
    if not (repo / "skills/facade-opencv-mask/scripts/segment.py").is_file():
        pytest.skip("Set FACADE_MASK_REPO to exercise the real upstream mask algorithm")
    result = MaskService(ArtifactStore(tmp_path / "workspace", [tmp_path]), repo_path=repo)
    yield result
    result.close()


@pytest.fixture
def cracked_image(tmp_path):
    path = tmp_path / "source.png"
    image = Image.new("RGB", (128, 96), (205, 205, 205))
    draw = ImageDraw.Draw(image)
    draw.line([(30, 15), (45, 40), (60, 70), (80, 80)], fill=(20, 20, 20), width=2)
    draw.line([(45, 40), (70, 35)], fill=(20, 20, 20), width=2)
    image.save(path)
    return path


def decode(rle):
    return np.repeat(np.arange(len(rle["counts"])) % 2, rle["counts"]).reshape(rle["size"], order="F") > 0


def defect(instance_id="crack-a", box=None, code="C"):
    return {"instance_id": instance_id, "class_code": code, "bbox_xyxy": box or [15, 5, 100, 90]}


def test_legacy_mesh_and_efflorescence_round_trip(service, cracked_image):
    inputs = [defect(code="M"), {**defect("deposit", code="L"), "class_name": "白华"}]
    result = service.segment(str(cracked_image), inputs)
    rows = result["instances"]
    assert [row["class_code"] for row in rows] == ["C", "E"]
    assert [row["original_class_code"] for row in rows] == ["M", "L"]
    assert "generic_deposit_baseline_requires_efflorescence_review" in rows[1]["quality_flags"]
    assert inputs[0]["class_code"] == "M"  # original inputs are immutable
    refined = service.refine(str(cracked_image), result["annotation_manifest"]["path"])
    assert [row["class_code"] for row in refined["instances"]] == ["C", "E"]



def test_real_opencv_geometry_rle_artifacts_and_fresh_jobs(service, cracked_image):
    result = service.segment(str(cracked_image), [defect(), defect("overlap", [40, 20, 100, 90])])
    assert result["schema_version"] == "facade-mask/v1"
    assert result["expert_validated"] is False
    assert result["metrics"]["api_requests"] == 0
    assert result["metrics"]["estimated_cost"] == 0
    assert result["physical_area_m2"] is None
    assert result["source_sha256"] == hashlib.sha256(cracked_image.read_bytes()).hexdigest()
    masks = []
    for row in result["instances"]:
        array = np.asarray(Image.open(row["mask_path"]))
        assert array.shape == (96, 128)
        assert set(np.unique(array)) <= {0, 255}
        selected = array > 0
        assert selected.any(), "synthetic dark crack should have visible foreground"
        assert row["area_pixels"] == int(selected.sum())
        assert np.array_equal(selected, decode(row["segmentation_rle"]))
        inside = np.zeros_like(selected)
        x1, y1, x2, y2 = row["bbox_xyxy"]
        inside[y1:y2, x1:x2] = True
        assert not (selected & ~inside).any()
        assert service.store.resolve_input(row["mask"]["uri"]) == Path(row["mask_path"])
        masks.append(selected)
    union = np.logical_or.reduce(masks)
    assert np.array_equal(union, np.asarray(Image.open(result["union_mask"]["path"])) > 0)
    assert np.array_equal(union, decode(result["segmentation_rle"]))
    assert result["area_pixels"] == int(union.sum())
    source = np.asarray(Image.open(cracked_image))
    overlay = np.asarray(Image.open(result["overlay"]["path"]))
    assert np.array_equal(source[~union], overlay[~union])
    state = service.get_job(result["job_id"])
    assert state["status"] == "completed"
    assert state["result"]["annotation_manifest"]["sha256"] == result["annotation_manifest"]["sha256"]
    manifest = service.get_artifact(result["annotation_manifest"]["uri"])
    assert manifest["image_id"] == result["image_id"]
    second = service.segment(str(cracked_image), [defect()])
    assert second["job_id"] != result["job_id"]
    assert second["provenance"]["reused_previous_results"] is False
    assert second["instances"][0]["area_pixels"] == result["instances"][0]["area_pixels"]


@pytest.mark.parametrize("rows", [
    [defect(box=[0, 0, 129, 96])],
    [defect(box=[0, 0, 1.5, 96])],
    [defect(box=[0, 0, 0, 96])],
    [defect(code="unsupported")],
    [defect(), defect()],
    [dict(defect(), prior_mask="not-read.png")],
])
def test_invalid_input_rejected_before_job_creation(service, cracked_image, rows):
    validation = service.validate_input(str(cracked_image), rows)
    assert not validation["valid"] and validation["errors"]
    with pytest.raises(ValueError):
        service.segment(str(cracked_image), rows)
    assert not list((service.store.root / "jobs").iterdir())


def test_input_outside_allowed_roots_is_rejected(service, tmp_path):
    unrelated = str(tmp_path.parent / "outside-image.png")
    assert not service.validate_input(unrelated, [])['valid']


def test_algorithm_failure_never_fabricates_bbox_mask(service, cracked_image, monkeypatch):
    def fail(*args):
        raise ValueError("real segmentation failure")
    monkeypatch.setattr(service, "_module", lambda *args: SimpleNamespace(segment=fail))
    with pytest.raises(RuntimeError, match="real segmentation failure"):
        service.segment(str(cracked_image), [defect()])
    job = next((service.store.root / "jobs").iterdir())
    state = service.get_job(job.name)
    assert state["status"] == "failed"
    assert not (job / "annotation_manifest.json").exists()
    assert not (job / "outputs" / "union_mask.png").exists()


def test_empty_detection_set_does_not_invent_defects(service, cracked_image):
    result = service.segment(str(cracked_image), [])
    assert result["status"] == "no_defects_provided"
    assert result["instances"] == [] and result["area_pixels"] == 0
    assert not np.asarray(Image.open(result["union_mask"]["path"])).any()


def test_api_profile_is_explicit_and_cannot_be_url(service, cracked_image):
    with pytest.raises(ValueError, match="configured model_profile"):
        service.start_api_job(str(cracked_image), [defect()], "https://provider.example/api")
    with pytest.raises(ValueError, match="FACADE_MASK_MODELS_CONFIG"):
        service.segment(str(cracked_image), [defect()], method="api", model_profile="configured")
    assert not list((service.store.root / "jobs").iterdir())


def test_inline_secret_config_rejected(service, cracked_image, tmp_path):
    config = tmp_path / "models.json"
    config.write_text(json.dumps({"models": [{"id": "unsafe", "model": "x", "base_url": "https://example.test",
                                             "api_key": "secret-would-not-be-used", "api_key_env": "TEST_TOKEN"}]}))
    service.model_config_path = config
    with pytest.raises(ValueError, match="secrets must use api_key_env"):
        service.start_api_job(str(cracked_image), [defect()], "unsafe")


def test_hybrid_refinement_preserves_raw_and_source_identity(service, cracked_image):
    first = service.segment(str(cracked_image), [defect()])
    second = service.refine(str(cracked_image), first["annotation_manifest"]["uri"])
    assert second["provenance"]["method"] == "hybrid_opencv_refinement"
    assert second["metrics"]["additional_api_requests"] == 0
    assert second["expert_validated"] is False
    row = second["instances"][0]
    original_mask = np.asarray(Image.open(first["instances"][0]["mask_path"]))
    assert np.array_equal(original_mask, np.asarray(Image.open(row["raw_mask"]["path"])))
    assert Path(row["candidate_mask"]["path"]).is_file()
    chosen = np.asarray(Image.open(row["mask_path"])) > 0
    assert np.array_equal(chosen, decode(row["segmentation_rle"]))
    assert row["refinement"]["selection"] in ("raw_fallback", "refined_candidate")
    Image.new("RGB", (128, 96), "red").save(cracked_image)
    with pytest.raises(ValueError, match="source image hash"):
        service.refine(str(cracked_image), first["annotation_manifest"]["uri"])


def test_mcp_tools_execute_real_service_without_large_rle_in_context(service, cracked_image):
    from fastmcp import Client
    from facade_mcp.mask_server import create_server

    async def exercise():
        async with Client(create_server(service)) as client:
            tools = await client.list_tools()
            assert {tool.name for tool in tools} == {"mask_validate_input", "mask_segment_opencv",
                    "mask_start_api_job", "mask_get_job", "mask_refine", "mask_get_artifact"}
            validation = await client.call_tool("mask_validate_input", {"image_path": str(cracked_image),
                                                                         "defects": [defect()]})
            assert validation.data["valid"]
            call = await client.call_tool("mask_segment_opencv", {"image_path": str(cracked_image),
                                                                   "defects": [defect()]})
            data = call.data
            assert data["instances"][0]["area_pixels"] > 0
            assert "segmentation_rle" not in data and "defects" not in data
            assert "segmentation_rle" not in data["instances"][0]
            stored = service.get_artifact(data["annotation_manifest"]["uri"])
            assert "segmentation_rle" in stored["instances"][0]

    asyncio.run(exercise())
