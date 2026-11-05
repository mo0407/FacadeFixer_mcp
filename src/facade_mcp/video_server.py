"""Video preparation, overview and auditable video-review reports over MCP."""
import argparse
from fastmcp import FastMCP
from .storage import ArtifactStore
from .video_service import VideoService


def create_server(store: ArtifactStore | None = None):
    store = store or ArtifactStore.from_env()
    service = VideoService(store)
    mcp = FastMCP('FacadeVideo')

    @mcp.tool
    def video_inspect(video_path: str, modality: str = 'visible') -> dict:
        """Read video stream dimensions, fps and source hash. GPS/radiometry are uninspected, not inferred absent."""
        return service.inspect(video_path, modality)

    @mcp.tool
    def video_extract_frames(video_path: str, modality: str = 'visible', sample_count: int = 16, timestamps: list[float] | None = None) -> dict:
        """Extract original-sized PNGs with exact decoded frame indices, times and relative sharpness. Max 200 frames."""
        return service.extract(video_path, modality, sample_count, timestamps)

    @mcp.tool
    def video_pair_frames(visible_manifest: str, thermal_manifest: str, offset_seconds: float = 0, tolerance_seconds: float = .25) -> dict:
        """One-to-one nearest-time candidates. thermal_time = visible_time + offset. Requires landmark review; not pixel registration."""
        return service.pair(visible_manifest, thermal_manifest, offset_seconds, tolerance_seconds)

    @mcp.tool
    def video_track_transforms(video_path: str, frame_manifest: str, stride: int = 3, tracking_width: int = 640) -> dict:
        """Estimate translation-only mosaic transforms from dense video tracks. May reject weak tracks. Must visually review seams; no orthophoto claim."""
        return service.track(video_path, frame_manifest, stride, tracking_width)

    @mcp.tool
    def video_render_overview(frame_manifest: str, transforms: list[list[list[float]]], canvas_width: int, canvas_height: int, markers: list[dict] | None = None) -> dict:
        """Render original-pixel-to-canvas transforms and markers {frame_id, defect_id, bbox_xyxy}. Supports caller-reviewed homographies; max 24 MP. Preserve thermal palette."""
        return service.overview(frame_manifest, transforms, canvas_width, canvas_height, markers)

    @mcp.tool
    def video_curate_review(project: dict, defects: list[dict], actor: str, reason: str, parent_manifest: str | None = None, overviews: list[dict] | None = None) -> dict:
        """Snapshot explicitly grouped defects/evidence and masks; create location/crop/overlay images and audit trail. Each defect: defect_id,title,observation,interpretation,recommendation,review_status,evidence. Each evidence: evidence_id,modality,image_path,bbox_xyxy, optional seconds,frame_id,mask_path,mask_method,mask_semantics. Thermal masks require pseudocolor_candidate. No automatic diagnosis/deduplication."""
        return service.curate(project, defects, actor, reason, parent_manifest, overviews)

    @mcp.tool
    def video_report_html(review_manifest: str) -> dict:
        """Export self-contained HTML with embedded overviews, per-defect location images and masks. Always a draft."""
        return service.render_html(review_manifest)

    @mcp.tool
    def video_report_pdf(review_manifest: str, font_path: str) -> dict:
        """Export PDF draft using optional reportlab. Provide an allowed local CJK TrueType font; final visual pagination review still required."""
        return service.render_pdf(review_manifest, font_path)

    @mcp.tool
    def video_get_artifact(artifact_id: str) -> dict:
        """Resolve generated video evidence/artifact metadata."""
        return store.get_artifact(artifact_id.removeprefix('artifact:'))

    return mcp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transport',choices=['stdio','http'],default='stdio')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765)
    a=p.parse_args();server=create_server()
    if a.transport=='stdio':server.run(transport='stdio',show_banner=False)
    else:server.run(transport='http',host=a.host,port=a.port,show_banner=False)


if __name__=='__main__':main()
