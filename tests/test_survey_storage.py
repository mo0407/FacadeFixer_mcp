import json
from pathlib import Path
import pytest
from PIL import Image
from facade_mcp.storage import ArtifactStore
from facade_mcp.survey_service import SurveyService


def test_artifact_roots_hash_and_job_isolation(tmp_path):
    allowed = tmp_path/"allowed"
    allowed.mkdir()
    store = ArtifactStore(tmp_path/"workspace",[allowed])
    job_id,directory = store.create_job("x",{})
    path = directory/"result.json"
    store.write_json(path,{"x":1})
    artifact = store.artifact(path,"result",job_id)
    assert store.resolve_input(artifact["uri"])==path
    with pytest.raises(ValueError):
        store.resolve_input(str(tmp_path/"forbidden.png"))
    with pytest.raises(ValueError):
        store.job_dir("../escape")
    path.write_text("changed")
    with pytest.raises(ValueError,match="changed"):
        store.resolve_input(artifact["uri"])


def test_route_is_a_planning_draft_with_vertical_layers(tmp_path):
    service = SurveyService(ArtifactStore(tmp_path/"workspace",[tmp_path]))
    result = service.plan_route({"building_id":"B1","longitude":120,"latitude":30,"building_height_m":20},
                                {"capture_heights_m":[5,15],"stand_off_m":10})
    assert result["executable"] is False
    assert {p["relative_height_m"] for p in result["waypoints"]}=={5,15}
    assert result["geometry_method"]=="approximate_radius_envelope"
    assert len(result["required_checks"])>=4
    geo = json.loads(Path(result["route"]["path"]).read_text())
    assert geo["features"][0]["geometry"]["type"]=="LineString"
    with pytest.raises(ValueError):
        service.plan_route({"building_id":"B1","longitude":120,"latitude":30,"building_height_m":20},{"capture_heights_m":[21]})


def test_media_dedup_and_hash_snapshot(tmp_path):
    source = tmp_path/"source.jpg"
    Image.new("RGB",(50,40)).save(source)
    service = SurveyService(ArtifactStore(tmp_path/"workspace",[tmp_path]))
    result = service.organize_images([str(source),str(source)],"B1")
    assert result["duplicate_count"]==1
    row = result["images"][0]
    assert row["width"]==50 and row["height"]==40
    assert Path(row["image_path"])!=source
    assert row["image_id"]=="image_"+row["sha256"][:20]
