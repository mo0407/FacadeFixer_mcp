"""Model MCP contracts, conditional workflow routing and secret-free config export."""
from __future__ import annotations

import asyncio
import json
import sys

import pytest

from facade_mcp.orchestration import build_workflow, server_config
from facade_mcp.storage import ArtifactStore


class RecordingGateway:
    def __init__(self, *, digest="photo-sha"):
        self.calls = []
        self.digest = digest

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "survey_validate_site":
            return {"site": arguments["site"]}
        if name == "survey_plan_route":
            return {"executable": False}
        if name == "media_organize_images":
            return {"images": [{"image_path": "archived-photo.png", "sha256": "photo-sha"}]}
        if name in ("defect_detect", "facade_models_detect"):
            return {"images": [{"image_path": "runner-copy.png", "source_sha256": self.digest,
                                 "defects": [{"instance_id": "D1", "class_code": "C", "bbox_xyxy": [1, 2, 8, 9]}],
                                 "unmapped_defects": [{"class_name": "unmapped-class"}]}]}
        if name in ("facade_models_segment", "mask_segment_opencv"):
            return {"schema_version": "facade-mask/v1", "instances": [], "expert_validated": False}
        if name == "report_build_draft":
            return {"report_document": {"uri": "artifact:report"}}
        if name == "report_validate":
            return {"formal_ready": False}
        if name == "report_render_html":
            return {"html": {"path": "draft.html"}}
        raise AssertionError(f"Unexpected tool {name}")


def request(detection=None, mask=None):
    return {"site": {"building_id": "B1"}, "image_paths": ["photo.png"],
            "detection": detection or {}, "mask": mask or {}}


@pytest.mark.parametrize("detector,segmentation,expected_detector,expected_mask", [
    ({"method": "source", "model_profile": "rtdetr", "threshold": .47},
     {"method": "source", "model_profile": "sam3"}, "facade_models_detect", "facade_models_segment"),
    ({"method": "source"}, {}, "facade_models_detect", "mask_segment_opencv"),
    ({"method": "annotations", "annotation_path": "annotations.json"},
     {"method": "source"}, "defect_detect", "facade_models_segment"),
    ({"method": "local_yolo"}, {}, "defect_detect", "mask_segment_opencv"),
])
def test_workflow_routes_to_selected_model_tools_and_keeps_evidence(detector, segmentation, expected_detector, expected_mask):
    gateway = RecordingGateway()
    result = asyncio.run(build_workflow(gateway).ainvoke({"request": request(detector, segmentation)}))
    assert [name for name, _ in gateway.calls] == ["survey_validate_site", "survey_plan_route",
            "media_organize_images", expected_detector, expected_mask, "report_build_draft",
            "report_validate", "report_render_html"]
    detection_args = gateway.calls[3][1]
    assert detection_args["image_paths"] == ["archived-photo.png"]
    if expected_detector == "facade_models_detect":
        assert detection_args["model_profile"] == detector.get("model_profile", "yolo12m")
        assert detection_args["threshold"] == detector.get("threshold", .25)
        assert "annotation_path" not in detection_args and "method" not in detection_args
    else:
        assert detection_args["method"] == detector["method"]
    mask_args = gateway.calls[4][1]
    assert mask_args["image_path"] == "archived-photo.png"
    assert mask_args["defects"][0]["instance_id"] == "D1"
    if expected_mask == "facade_models_segment":
        assert mask_args["model_profile"] == segmentation.get("model_profile", "mobile_sam")
    assert result["segmentation"]["images"][0]["unmapped_defects"] == [{"class_name": "unmapped-class"}]
    assert result["validation"]["formal_ready"] is False


def test_source_detector_must_match_archived_photo_hash():
    gateway = RecordingGateway(digest="different-photo")
    with pytest.raises(RuntimeError, match="differs from archived original"):
        asyncio.run(build_workflow(gateway).ainvoke({"request": request({"method": "source"}, {"method": "source"})}))
    assert gateway.calls[-1][0] == "facade_models_detect"


def test_local_model_routing_does_not_enable_paid_api():
    gateway = RecordingGateway()
    with pytest.raises(ValueError, match="allow_paid_api=true"):
        asyncio.run(build_workflow(gateway).ainvoke({"request": request({"method": "source"}, {"method": "api"})}))
    assert not any(name == "mask_start_api_job" for name, _ in gateway.calls)


def test_model_tools_publish_scoped_schemas_and_preserve_detection_boxes(tmp_path):
    from fastmcp import Client
    from facade_mcp.model_server import create_server

    store = ArtifactStore(tmp_path / "workspace", [tmp_path])
    job, directory = store.create_job("fixture", {})
    store.write_json(directory / "manifest.json", {"fixture": True})
    artifact = store.artifact(directory / "manifest.json", "fixture", job)

    class FixtureService:
        def __init__(self):
            self.store = store
            self.refinement_args = None

        def list_tools(self):
            return {"profiles": [{"id": "mobile_sam", "available": True}], "paid_api_requests": 0}

        def detect(self, image_paths, model_profile, threshold):
            return {"images": [{"defects": [{"instance_id": "D1", "bbox_xyxy": [1, 2, 8, 9]}]}],
                    "profile": model_profile, "threshold": threshold}

        def segment(self, image_path, defects, model_profile):
            return {"schema_version": "facade-mask/v1", "instances": [{"instance_id": "D1",
                    "area_pixels": 12, "segmentation_rle": {"counts": [0, 12]}}],
                    "segmentation_rle": {"counts": [0, 12]}, "annotation_manifest": artifact}

        def refine(self, *args, **kwargs):
            self.refinement_args = args, kwargs
            return {"instances": [{"instance_id": args[2], "segmentation_rle": {"counts": [1, 1]}}],
                    "expert_validated": False, "annotation_manifest": artifact}

        def get_job(self, job_id):
            return {"job_id": job_id, "result": {"images": [{"defects": [{"instance_id": "D1"}]}]}}

    service = FixtureService()

    async def exercise():
        async with Client(create_server(service)) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            assert set(tools) == {"facade_models_list", "facade_models_detect", "facade_models_segment",
                                  "facade_models_refine", "facade_models_get_job", "facade_models_get_artifact"}
            assert set(tools["facade_models_detect"].input_schema["properties"]) == {"image_paths", "model_profile", "threshold"}
            assert set(tools["facade_models_refine"].input_schema["required"]) == {
                "image_path", "annotation_manifest_path", "instance_id", "positive_points", "negative_points"}
            assert (await client.call_tool("facade_models_list", {})).data["profiles"][0]["available"]
            detection = (await client.call_tool("facade_models_detect", {"image_paths": ["photo.png"]})).data
            assert detection["profile"] == "yolo12m" and detection["images"][0]["defects"][0]["instance_id"] == "D1"
            mask = (await client.call_tool("facade_models_segment", {"image_path": "photo.png", "defects": []})).data
            assert "segmentation_rle" not in mask and "segmentation_rle" not in mask["instances"][0]
            assert mask["instances"][0]["area_pixels"] == 12 and mask["annotation_manifest"] == artifact
            await client.call_tool("facade_models_refine", {"image_path": "photo.png",
                "annotation_manifest_path": artifact["uri"], "instance_id": "D1",
                "positive_points": [[4, 5]], "negative_points": [[8, 9]]})
            assert service.refinement_args == (("photo.png", artifact["uri"], "D1", [[4., 5.]], [[8., 9.]]), {"model_profile": "mobile_sam"})
            state = (await client.call_tool("facade_models_get_job", {"job_id": job})).data
            assert state["result"]["images"][0]["defects"][0]["instance_id"] == "D1"
            assert (await client.call_tool("facade_models_get_artifact", {"artifact_id": artifact["uri"]})).data == artifact

    asyncio.run(exercise())


def test_four_server_config_and_cli_export_exclude_process_secrets(tmp_path, monkeypatch, capsys):
    from facade_mcp.cli import main

    monkeypatch.setenv("FACADE_CONFIG_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("FACADE_SOURCE_REPO", str(tmp_path / "source"))
    monkeypatch.setenv("FACADE_SOURCE_PYTHON", str(tmp_path / "source-python.exe"))
    monkeypatch.setenv("FACADE_SOURCE_DEVICE", "cpu")
    monkeypatch.setenv("FACADE_SOURCE_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("FACADE_LLM_API_KEY", "secret-sentinel-llm")
    monkeypatch.setenv("FACADE_DETECTOR_TOKEN", "secret-sentinel-detector")
    monkeypatch.setenv("UNRELATED_PASSWORD", "secret-sentinel-unrelated")
    workspace = tmp_path / "workspace"
    direct = server_config(workspace, [tmp_path])
    assert set(direct["mcpServers"]) == {"survey", "mask", "report", "model"}
    assert direct["mcpServers"]["model"]["args"] == ["-m", "facade_mcp.model_server"]
    destination = tmp_path / "mcp.json"
    monkeypatch.setattr(sys, "argv", ["facade-workflow", "--workspace", str(workspace), "mcp-config", "--output", str(destination)])
    main()
    serialized = destination.read_text(encoding="utf-8")
    assert "secret-sentinel" not in serialized and "secret-sentinel" not in capsys.readouterr().out
    for server in json.loads(serialized)["mcpServers"].values():
        assert server["env"]["FACADE_SOURCE_REPO"] == str(tmp_path / "source")
        assert server["env"]["FACADE_SOURCE_PYTHON"] == str(tmp_path / "source-python.exe")
        assert server["env"]["FACADE_SOURCE_DEVICE"] == "cpu"
        assert server["env"]["FACADE_SOURCE_TIMEOUT_SECONDS"] == "600"
        assert "FACADE_LLM_API_KEY" not in server["env"] and "FACADE_DETECTOR_TOKEN" not in server["env"]


def test_models_cli_uses_managed_catalog_without_running_inference(tmp_path, monkeypatch, capsys):
    from facade_mcp.cli import main

    calls = []

    class CatalogService:
        def __init__(self, store):
            calls.append(store.root)

        def list_tools(self):
            return {"schema_version": "facade-model-catalog/v1", "models": [{"model_profile": "yolo12m"}]}

    monkeypatch.setenv("FACADE_CONFIG_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setattr("facade_mcp.model_service.ModelService", CatalogService)
    monkeypatch.setattr(sys, "argv", ["facade-workflow", "--workspace", str(tmp_path / "workspace"), "models"])
    main()
    assert calls == [(tmp_path / "workspace").resolve()]
    assert json.loads(capsys.readouterr().out)["models"][0]["model_profile"] == "yolo12m"
