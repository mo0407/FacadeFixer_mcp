"""MCP entry point for draft reports and standalone HTML export."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .report_service import ReportService
from .storage import ArtifactStore


def create_server(store: ArtifactStore | None = None) -> Any:
    from fastmcp import FastMCP

    service = ReportService(
        store or ArtifactStore.from_env(),
        Path(os.environ["FACADE_REPORT_REPO"]) if os.getenv("FACADE_REPORT_REPO") else None,
    )
    server = FastMCP("FacadeFixer Report")

    @server.tool
    def report_build_draft(project: dict[str, Any], segmentation_result: dict[str, Any], detection_result: dict[str, Any] | None = None) -> dict[str, Any]:
        """Import identified mask results into an unreviewed report draft.

        Evidence associations use image_id and instance_id, never list order.
        Pixel area comes from the segmentation mask, not bounding-box area.
        """
        return service.build_draft(project, segmentation_result, detection_result)

    @server.tool
    def report_validate(document_path: str) -> dict[str, Any]:
        """Check a draft JSON artifact; return missing fields and formal readiness.

        Formal readiness requires complete review records and signature files.
        This check does not authenticate identity or cryptographic signatures.
        """
        return service.validate(document_path)

    @server.tool
    def report_render_html(document_path: str) -> dict[str, Any]:
        """Render a standalone draft HTML artifact with embedded evidence images.

        Missing fields and unavailable evidence are shown in the draft.
        """
        return service.render(document_path)

    @server.tool
    def report_get_artifact(artifact_id: str) -> dict[str, Any]:
        """Resolve a permitted report artifact to its path, digest, and size."""
        return service.get_artifact(artifact_id)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="FacadeFixer report MCP server")
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8763)
    args = parser.parse_args()
    server = create_server()
    if args.transport == "stdio":
        server.run(transport="stdio", show_banner=False)
    else:
        server.run(transport="http", host=args.host, port=args.port, show_banner=False)


if __name__ == "__main__":
    main()
