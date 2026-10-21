from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest

from facade_mcp.report_service import ReportService


class MemoryStore:
    """Small filesystem fixture using the public ArtifactStore protocol."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir()
        self.jobs = {}
        self.artifacts = {}

    def resolve_input(self, value):
        value = value.removeprefix("artifact:")
        path = Path(self.artifacts.get(value, value)).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Path is outside allowed input roots")
        if not path.exists():
            raise ValueError("File does not exist")
        return path

    def create_job(self, kind, inputs):
        job_id = f"job-{len(self.jobs) + 1}"
        job_dir = self.root / job_id
        job_dir.mkdir()
        self.jobs[job_id] = {"status": "running"}
        return job_id, job_dir

    def write_json(self, path, data):
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def artifact(self, path, role, job_id):
        path = Path(path)
        artifact_id = f"artifact-{len(self.artifacts) + 1}"
        self.artifacts[artifact_id] = str(path)
        return {"artifact_id": artifact_id, "path": str(path), "role": role, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}

    def complete_job(self, job_id, result):
        self.jobs[job_id] = {"status": "completed", "result": result}

    def fail_job(self, job_id, error):
        self.jobs[job_id] = {"status": "failed", "error": error}


@pytest.fixture
def report_fixture(tmp_path):
    store = MemoryStore(tmp_path / "workspace")
    image = store.root / "original.png"
    # Valid 1x1 PNG keeps the tests independent of OpenCV/Pillow.
    image.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jL7sAAAAASUVORK5CYII="))
    mask = store.root / "mask.png"
    mask.write_bytes(image.read_bytes())
    original = store.artifact(image, "original", "source")
    mask_artifact = store.artifact(mask, "mask", "source")
    segmentation = {"images": [{
        "schema_version": "facade-mask/v1", "image_id": "photo-a", "width": 20,
        "height": 10, "original_image": original,
        "instances": [{"instance_id": "d-1", "class_code": "crack", "class_name": "裂缝", "bbox_xyxy": [1, 1, 10, 8], "area_pixels": 7, "mask": mask_artifact, "method": "opencv", "severity": "待确认", "metrics": {"elapsed": 0.2}}],
    }]}
    project = {"project_id": "p-1", "name": "测试项目", "building_id": "b-1", "inspection_date": "2026-10-14", "responsible_party": "施工单位", "custom_metadata": {"floor": 3}}
    return ReportService(store), store, segmentation, project


def load_document(store, result):
    return json.loads(store.resolve_input(result["report_document"]["artifact_id"]).read_text(encoding="utf-8"))


@pytest.mark.parametrize("code,label,expected", [("M", "Meshcrack", "C"), ("L", "白华", "E"), ("L", "Leakage", "L")])
def test_report_normalizes_legacy_categories_without_rewriting_sources(report_fixture, code, label, expected):
    service, store, segmentation, project = report_fixture
    segmentation["images"][0]["instances"][0].update(class_code=code, class_name=label)
    original = copy.deepcopy(segmentation)
    result = service.build_draft(project, segmentation)
    document = load_document(store, result)
    assert document["defects"][0]["category_id"] == expected
    assert document["defects"][0]["class_code"] == expected
    assert document["source_data"]["segmentation_result"] == original
    assert segmentation == original


def test_unsupported_detection_is_visible_without_invented_mask(report_fixture):
    service,store,segmentation,project = report_fixture
    group = segmentation["images"][0]
    group["instances"] = []
    detection = {"images":[{"image_id":group["image_id"],"defects":[],"unmapped_defects":[
        {"instance_id":"u1","original_class_name":"Bulging","bbox_xyxy":[2,2,12,9],"reason":"unsupported_class"}
    ]}]}
    result = service.build_draft(project,segmentation,detection)
    document = load_document(store,result)
    assert result["defect_count"] == 1
    assert result["unsegmented_defect_count"] == 1
    row = document["defects"][0]
    assert row["category_name"] == "Bulging" and row["area_pixels"] is None
    assert row["segmentation_status"] == "not_supported"
    assert "mask" not in row
    assert any("unsupported_category" in field for field in result["validation"]["missing_fields"])
    rendered = service.render(result["report_document"]["path"])
    assert "Bulging" in Path(rendered["html"]["path"]).read_text(encoding="utf-8")


def test_draft_preserves_metadata_and_pixel_area_without_automatic_approval(report_fixture):
    service, store, segmentation, project = report_fixture
    input_copy = copy.deepcopy(segmentation)
    result = service.build_draft(project, segmentation)
    document = load_document(store, result)
    defect = document["defects"][0]
    assert document["project"] == project
    assert defect["metrics"] == {"elapsed": 0.2}
    assert defect["severity"] == "待确认"
    assert defect["area_pixels"] == 7  # Not bbox area (9 * 7).
    assert "estimated_area_m2" not in defect["quantity"]
    assert defect["expert_conclusion"]["status"] == "pending_confirmation"
    assert all(item["state"] == "model_estimate" and item["approved_redaction"] is False for item in defect["evidence"])
    assert result["validation"]["schema_valid"] is True
    assert result["validation"]["draft_ready"] is True
    assert result["validation"]["formal_ready"] is False
    assert segmentation == input_copy


def test_detection_join_uses_identifiers_not_list_position(report_fixture):
    service, store, segmentation, project = report_fixture
    image = segmentation["images"][0]
    first = image["instances"][0]
    second = copy.deepcopy(first)
    second["instance_id"] = "d-2"
    image["instances"].append(second)
    detection = {"images": [{"image_id": "photo-a", "detections": [
        {"instance_id": "d-2", "facade_id": "south", "confidence": 0.81},
        {"instance_id": "d-1", "facade_id": "north", "confidence": 0.92},
    ]}]}
    document = load_document(store, service.build_draft(project, segmentation, detection))
    assert [(d["instance_id"], d["facade_id"], d["confidence"]) for d in document["defects"]] == [("d-1", "north", 0.92), ("d-2", "south", 0.81)]


def test_html_is_standalone_escapes_user_text_and_has_explicit_evidence_roles(report_fixture):
    service, store, segmentation, project = report_fixture
    project["name"] = '<script>alert("x")</script>'
    result = service.build_draft(project, segmentation)
    rendered = service.render(result["report_document"]["artifact_id"])
    page = Path(rendered["html"]["path"]).read_text(encoding="utf-8")
    assert '<script>alert("x")</script>' not in page
    assert "&lt;script&gt;" in page
    assert "data:image/png;base64," in page
    assert 'data-role="original"' in page
    assert 'data-role="mask"' in page
    assert 'data-instance-id="d-1"' in page
    assert "localStorage" not in page
    assert "尚不具备" in page
    assert rendered["assets"]
    assert all(Path(asset["path"]).exists() for asset in rendered["assets"])


def test_missing_evidence_renders_draft_with_flags(report_fixture):
    service, store, segmentation, project = report_fixture
    segmentation["images"][0]["instances"][0]["mask"] = {"path": str(store.root / "missing.png")}
    result = service.build_draft({}, segmentation)
    assert "project.name" in result["validation"]["missing_fields"]
    rendered = service.render(result["report_document"]["path"])
    assert "证据暂不可用" in Path(rendered["html"]["path"]).read_text(encoding="utf-8")
    assert rendered["validation"]["formal_ready"] is False


def test_physical_area_requires_explicit_scale_and_source(report_fixture):
    service, store, segmentation, project = report_fixture
    project["scale"] = {"meters_per_pixel": 0.1, "source": "现场标尺标定"}
    document = load_document(store, service.build_draft(project, segmentation))
    assert document["defects"][0]["quantity"]["estimated_area_m2"] == pytest.approx(0.07)
    project["scale"].pop("source")
    document = load_document(store, service.build_draft(project, segmentation))
    assert "estimated_area_m2" not in document["defects"][0]["quantity"]


def test_formal_ready_requires_signed_review_files_and_redaction_approval(report_fixture):
    service, store, segmentation, project = report_fixture
    result = service.build_draft(project, segmentation)
    path = result["report_document"]["path"]
    document = load_document(store, result)
    signature = store.root / "signature.txt"
    signature.write_text("Explicit human review record", encoding="utf-8")
    review = {"reviewer_id": "reviewer-1", "signed_at": "2026-10-14T12:00:00+08:00", "signature_artifact": str(signature)}
    document["formal_review"] = copy.deepcopy(review)
    defect = document["defects"][0]
    defect["facade_id"] = "north"
    defect["expert_conclusion"] = {"status": "confirmed", **review}
    for group in document["images"] + document["defects"]:
        for item in group["evidence"]:
            item["approved_redaction"] = True
            item["redaction_review"] = copy.deepcopy(review)
    store.write_json(path, document)
    assert service.validate(path)["formal_ready"] is True
    signature.unlink()
    assert service.validate(path)["formal_ready"] is False


def test_duplicate_instance_ids_are_rejected(report_fixture):
    service, store, segmentation, project = report_fixture
    segmentation["images"][0]["instances"].append(copy.deepcopy(segmentation["images"][0]["instances"][0]))
    with pytest.raises(ValueError, match="Duplicate instance_id"):
        service.build_draft(project, segmentation)
    assert next(iter(store.jobs.values()))["status"] == "failed"


def test_real_artifact_store_uris_round_trip(tmp_path):
    from facade_mcp.storage import ArtifactStore

    store = ArtifactStore(tmp_path / "runtime", allowed_input_roots=[tmp_path])
    job_id, job_dir = store.create_job("source", {})
    image = job_dir / "original.png"
    image.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jL7sAAAAASUVORK5CYII="))
    source = store.artifact(image, "original", job_id)
    service = ReportService(store)
    result = service.build_draft({}, {"image_id": "a", "width": 1, "height": 1, "original_image": source, "instances": []})
    assert service.validate(result["report_document"]["uri"])["schema_valid"] is True
    rendered = service.render(result["report_document"]["uri"])
    assert "data:image/png;base64," in store.resolve_input(rendered["html"]["uri"]).read_text(encoding="utf-8")
    assert service.get_artifact(rendered["html"]["artifact_id"])["sha256"] == rendered["html"]["sha256"]
