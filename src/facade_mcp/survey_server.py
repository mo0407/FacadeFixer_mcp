"""MCP surface for site, route draft, photo organization and existing detectors."""
from __future__ import annotations

import argparse
import os
from fastmcp import FastMCP
from .storage import ArtifactStore
from .survey_service import SurveyService
from .detection_service import DetectionService


def create_server(store: ArtifactStore | None = None) -> FastMCP:
    store = store or ArtifactStore.from_env()
    survey = SurveyService(store)
    detector = DetectionService(store, backend_url=os.environ.get("FACADE_DETECTOR_URL"), token=os.environ.get("FACADE_DETECTOR_TOKEN"))
    mcp = FastMCP("FacadeSurvey")

    @mcp.tool
    def survey_validate_site(site: dict) -> dict:
        """Validate building WGS84 longitude/latitude and height, without geocoding or aircraft control."""
        return survey.validate_site(site)

    @mcp.tool
    def survey_plan_route(site: dict, options: dict | None = None) -> dict:
        """Create a non-executable facade-photo route draft as GeoJSON. Needs building coordinates and height; stand_off_m and capture_heights_m are optional planning inputs."""
        result = survey.plan_route(site, options)
        return {key:value for key,value in result.items() if key != "waypoints"} | {"waypoint_count":len(result["waypoints"])}

    @mcp.tool
    def media_organize_images(image_paths: list[str], building_id: str, capture_metadata: dict | None = None) -> dict:
        """Hash, deduplicate and archive supplied local photos. Preserve stored pixel coordinates and EXIF orientation; return image manifest."""
        return survey.organize_images(image_paths, building_id, capture_metadata)

    @mcp.tool
    def defect_detect(image_paths: list[str], method: str = "annotations", annotation_path: str | None = None, threshold: float = 0.25) -> dict:
        """Get original-image defect bboxes from annotations, a configured FacadeFixer HTTP detector, or an existing local YOLO runtime. annotations needs annotation_path; http needs FACADE_DETECTOR_URL; local_yolo needs FACADE_DETECTOR_PYTHON and FACADE_DETECTOR_WEIGHTS. Does not fabricate detections or invoke paid justification."""
        return detector.detect(image_paths, method=method, annotation_path=annotation_path, threshold=threshold)

    @mcp.tool
    def survey_get_artifact(artifact_id: str) -> dict:
        """Return metadata/path/URI for a previously generated artifact."""
        return store.get_artifact(artifact_id.removeprefix("artifact:"))

    return mcp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["stdio","http"], default="stdio")
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=8761)
    args = parser.parse_args()
    mcp = create_server()
    if args.transport == "stdio":
        mcp.run(transport="stdio",show_banner=False)
    else:
        mcp.run(transport="http",host=args.host,port=args.port,show_banner=False)


if __name__ == "__main__":
    main()
