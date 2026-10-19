"""MCP tools for the local FacadeFixer model runner and its managed profiles."""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP

from .storage import ArtifactStore

if TYPE_CHECKING:
    from .model_service import ModelService


def _tool_result(value: Any) -> Any:
    """Leave original-resolution RLE in artifacts while retaining detector boxes."""
    if isinstance(value, dict):
        return {key: _tool_result(item) for key, item in value.items() if key != "segmentation_rle"}
    if isinstance(value, list):
        return [_tool_result(item) for item in value]
    return value


def create_server(service: ModelService | None = None, store: ArtifactStore | None = None) -> FastMCP:
    if service is None:
        from .model_service import ModelService
        service = ModelService(store or ArtifactStore.from_env())
    server = FastMCP("FacadeFixer Local Models")

    @server.tool()
    def facade_models_list() -> dict:
        """Inspect server-managed local detection/segmentation profiles and readiness.

        Missing weights or dependencies remain unavailable. Does not download
        weights, invoke models, call paid APIs, or evaluate accuracy.
        """
        return service.list_tools()

    @server.tool()
    def facade_models_detect(image_paths: list[str], model_profile: str = "yolo12m", threshold: float = 0.25) -> dict:
        """Run the configured FacadeFixer detector on supplied original photos.

        Choose a named profile returned by facade_models_list. Return real model
        boxes in original stored-pixel xyxy coordinates, image hashes, provenance,
        job state and artifacts. Confidence is not accuracy or human approval.
        """
        return service.detect(image_paths, model_profile=model_profile, threshold=threshold)

    @server.tool()
    def facade_models_segment(image_path: str, defects: list[dict], model_profile: str = "mobile_sam") -> dict:
        """Generate local model masks from original-photo defect boxes/classes.

        Choose a configured local profile; no paid API requests. Return
        facade-mask/v1 artifacts, original-resolution masks, actual pixel area
        and review state. Full RLE stays in the annotation manifest artifact.
        """
        return _tool_result(service.segment(image_path, defects, model_profile=model_profile))

    @server.tool()
    def facade_models_refine(image_path: str, annotation_manifest_path: str, instance_id: str,
                             positive_points: list[list[float]], negative_points: list[list[float]],
                             model_profile: str = "mobile_sam") -> dict:
        """Refine one prior mask with explicit foreground/background pixel points.

        Provide the exact original photo, its prior annotation manifest (path or
        artifact URI), and target instance id. Points use original stored pixels.
        Preserve prior evidence and write a new unreviewed result.
        """
        return _tool_result(service.refine(image_path, annotation_manifest_path, instance_id,
                                           positive_points, negative_points, model_profile=model_profile))

    @server.tool()
    def facade_models_get_job(job_id: str) -> dict:
        """Read a managed local-model job, including its result or failure state."""
        return _tool_result(service.get_job(job_id))

    @server.tool()
    def facade_models_get_artifact(artifact_id: str) -> dict:
        """Resolve an existing managed artifact id or artifact URI to file metadata."""
        return service.store.get_artifact(artifact_id.removeprefix("artifact:"))

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="FacadeFixer local detection and mask MCP server")
    parser.add_argument("--transport", choices=("stdio", "http", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8764)
    args = parser.parse_args()
    server = create_server()
    if args.transport == "stdio":
        server.run(transport="stdio", show_banner=False)
    else:
        server.run(transport="http", host=args.host, port=args.port, show_banner=False)


if __name__ == "__main__":
    main()
