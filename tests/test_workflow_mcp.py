"""One full offline workflow across three actual stdio MCP processes."""
import asyncio
import json
import os
from pathlib import Path
import pytest

from facade_mcp.demo import create_demo
from facade_mcp.orchestration import run_workflow
from facade_mcp.storage import ArtifactStore


def find_mask_repo():
    configured = os.environ.get("FACADE_MASK_REPO")
    if configured:
        return Path(configured)
    project_root = Path(__file__).resolve().parents[1]
    vendor_repo = project_root / "vendor" / "FacadeFixer_mask_api"
    if vendor_repo.is_dir():
        return vendor_repo
    local = project_root.parent / ".task_tmp" / "mask-mcp-feasibility"
    return local if local.is_dir() else None


def test_full_workflow_uses_mcp_and_preserves_real_outputs(tmp_path):
    repo = find_mask_repo()
    if not repo:
        pytest.skip("Set FACADE_MASK_REPO for the real upstream algorithm integration")
    request = create_demo(tmp_path/"fixture")
    # Fractional supplied bbox exercises conversion into safe integer mask indices.
    annotations = Path(request["detection"]["annotation_path"])
    data = json.loads(annotations.read_text())
    data["images"][0]["defects"][0]["bbox_xyxy"] = [120.3,65.8,244.2,394.9]
    # Keep a detector category without a mask algorithm in the visible report.
    data["images"][0]["defects"].append({"id":3,"class_name":"Bulging","bbox_xyxy":[650,100,750,200]})
    annotations.write_text(json.dumps(data),encoding="utf-8")
    store = ArtifactStore(tmp_path/"workspace",[tmp_path])
    result = asyncio.run(run_workflow(request,store,repo))
    assert result["orchestration"] == "langgraph_over_mcp"
    outputs = result["outputs"]
    assert outputs["route"]["executable"] is False
    assert outputs["report"]["defect_count"] == 3
    assert outputs["report"]["unsegmented_defect_count"] == 1
    assert outputs["validation"]["formal_ready"] is False
    report = json.loads(Path(outputs["report"]["report_document"]["path"]).read_text(encoding="utf-8"))
    assert all(d["building_id"]=="demo-building" for d in report["defects"])
    assert report["defects"][0]["bbox_xyxy"]==[120,65,245,395]
    assert len(report["defects"])==3 and report["defects"][-1]["area_pixels"] is None
    html = Path(outputs["rendered"]["html"]["path"]).read_text(encoding="utf-8")
    assert "Bulging" in html and "data:image/png;base64," in html
    trace = [json.loads(line) for line in Path(result["trace"]["path"]).read_text(encoding="utf-8").splitlines()]
    assert [row["tool"] for row in trace]==["survey_validate_site","survey_plan_route","media_organize_images","defect_detect","mask_segment_opencv","report_build_draft","report_validate","report_render_html"]
    assert all(row["status"]=="success" for row in trace)
