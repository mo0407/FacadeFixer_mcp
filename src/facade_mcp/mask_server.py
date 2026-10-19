"""MCP transport for mask generation: stdio by default, local HTTP optionally."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

from fastmcp import FastMCP

from .mask_service import MaskService, mask_tool_result
from .storage import ArtifactStore


def create_server(service: MaskService | None = None, host: str = "127.0.0.1", port: int = 8762) -> FastMCP:
    if service is None:
        service = MaskService(ArtifactStore.from_env())
    server = FastMCP("FacadeFixer Mask")

    @server.tool()
    def mask_validate_input(image_path: str, defects: list[dict]) -> dict:
        """Check original image and frozen bbox/class inputs. Coordinates are max-exclusive xyxy.

        Classes: C crack, M network crack, P peeling, S spalling, T rust stain,
        R corrosion, D contamination, V vegetation, L leakage. Does not call APIs.
        """
        return service.validate_input(image_path, defects)

    @server.tool()
    def mask_segment_opencv(image_path: str, defects: list[dict]) -> dict:
        """Create new deterministic bbox/class masks with local OpenCV, without API costs.

        Return original-sized instance/union masks, overlay, actual pixel areas, RLE,
        provenance and a manifest. Results require visual or expert review.
        """
        return mask_tool_result(service.segment(image_path, defects, method="opencv"))

    @server.tool()
    def mask_start_api_job(image_path: str, defects: list[dict], model_profile: str) -> dict:
        """Explicitly submit paid image API requests through a named server-owned profile.

        One request per instance; returns a job id immediately. Requires configured
        credentials. Use only when the user/workflow has authorized API generation.
        """
        return service.start_api_job(image_path, defects, model_profile)

    @server.tool()
    def mask_get_job(job_id: str) -> dict:
        """Read a mask job's state and, when finished, its result artifact manifest."""
        return mask_tool_result(service.get_job(job_id))

    @server.tool()
    def mask_refine(image_path: str, annotation_manifest_path: str) -> dict:
        """Refine an explicit prior candidate mask with OpenCV, preserving raw/candidate masks.

        This is hybrid refinement, not the bbox/class automatic baseline. Requires
        the exact original image hash. No new API requests or correctness scores.
        """
        return mask_tool_result(service.refine(image_path, annotation_manifest_path))

    @server.tool()
    def mask_get_artifact(annotation_manifest_path: str) -> dict:
        """Read an existing facade-mask/v1 manifest from an allowed path or artifact id."""
        return mask_tool_result(service.get_artifact(annotation_manifest_path))

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="FacadeFixer Mask MCP server")
    parser.add_argument("--transport", choices=("stdio", "http", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8762)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    server = create_server(host=args.host, port=args.port)
    if args.transport == "stdio":
        server.run(transport="stdio", show_banner=False)
    else:
        server.run(transport="http", host=args.host, port=args.port, show_banner=False)


if __name__ == "__main__":
    main()
