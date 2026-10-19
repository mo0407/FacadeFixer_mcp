"""Coordinate planning drafts and photo manifests; never executes aircraft commands."""
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ExifTags
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .storage import ArtifactStore, sha256


class BuildingSite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    building_id: str = Field(min_length=1)
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-85, le=85)
    building_height_m: float = Field(gt=0, le=300)
    building_radius_m: float = Field(default=12, gt=0, le=500)
    footprint_lonlat: list[list[float]] | None = None
    name: str = ""

    @field_validator("footprint_lonlat")
    @classmethod
    def polygon(cls, values):
        if values is not None:
            if len(values) < 3:
                raise ValueError("footprint needs at least three vertices")
            for point in values:
                if len(point) != 2 or not (-180 <= point[0] <= 180 and -85 <= point[1] <= 85):
                    raise ValueError("footprint coordinates must be [longitude, latitude]")
        return values


class FlightPlanningOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stand_off_m: float = Field(default=10, ge=3, le=200)
    waypoint_spacing_m: float = Field(default=8, ge=1, le=100)
    capture_heights_m: list[float] | None = None
    vertical_spacing_m: float = Field(default=8, ge=1, le=100)
    speed_m_s: float = Field(default=2, gt=0, le=10)
    @field_validator("capture_heights_m")
    @classmethod
    def heights(cls, values):
        if values is not None and (not values or any(not math.isfinite(v) or v <= 0 or v > 300 for v in values)):
            raise ValueError("capture heights must be nonempty, positive, and <=300 m")
        return values


class SurveyService:
    def __init__(self, store: ArtifactStore):
        self.store = store

    def validate_site(self, site: dict) -> dict:
        building = BuildingSite.model_validate(site)
        return {"valid": True, "site": building.model_dump(), "coordinate_system": "EPSG:4326",
                "warnings": [] if building.footprint_lonlat else ["No footprint supplied: planning uses a building-radius envelope"]}

    def plan_route(self, site: dict, options: dict | None = None) -> dict:
        building = BuildingSite.model_validate(site)
        settings = FlightPlanningOptions.model_validate(options or {})
        job_id, directory = self.store.create_job("route", {"site": building.model_dump(), "options": settings.model_dump()})
        lat_scale = 111_320.0
        lon_scale = lat_scale * math.cos(math.radians(building.latitude))
        if building.footprint_lonlat:
            xy = [((p[0] - building.longitude) * lon_scale, (p[1] - building.latitude) * lat_scale) for p in building.footprint_lonlat]
            if any(math.hypot(x, y) > 2000 for x, y in xy):
                self.store.fail_job(job_id, "Footprint too far from building coordinate")
                raise ValueError("Footprint must lie within 2 km of building coordinate")
            x_min, x_max = min(p[0] for p in xy), max(p[0] for p in xy)
            y_min, y_max = min(p[1] for p in xy), max(p[1] for p in xy)
            if x_max - x_min < 1 or y_max - y_min < 1:
                self.store.fail_job(job_id, "Footprint envelope is degenerate")
                raise ValueError("Footprint envelope must span at least 1 m in each direction")
            method = "footprint_axis_aligned_envelope"
        else:
            x_min = y_min = -building.building_radius_m
            x_max = y_max = building.building_radius_m
            method = "approximate_radius_envelope"
        margin = settings.stand_off_m
        corners = [(x_min-margin, y_min-margin), (x_max+margin, y_min-margin), (x_max+margin,y_max+margin), (x_min-margin,y_max+margin)]
        heights = settings.capture_heights_m or list(range(3, math.ceil(building.building_height_m), math.ceil(settings.vertical_spacing_m)))
        if not settings.capture_heights_m:
            heights = sorted(set([float(h) for h in heights if h <= building.building_height_m] + [building.building_height_m]))
        if any(h > building.building_height_m for h in heights):
            self.store.fail_job(job_id, "Capture height exceeds building height")
            raise ValueError("Capture heights must not exceed supplied building height")
        waypoints = []
        for layer, altitude in enumerate(heights):
            ring = corners if layer % 2 == 0 else [corners[0],corners[3],corners[2],corners[1]]
            for side in range(4):
                start, end = ring[side], ring[(side+1)%4]
                count = max(1, math.ceil(math.dist(start,end)/settings.waypoint_spacing_m))
                for step in range(count):
                    fraction = step/count
                    x,y = start[0]+(end[0]-start[0])*fraction, start[1]+(end[1]-start[1])*fraction
                    waypoints.append({"waypoint_id": len(waypoints)+1, "longitude":building.longitude+x/lon_scale,
                                      "latitude":building.latitude+y/lat_scale,"relative_height_m":float(altitude),
                                      "heading_deg": math.degrees(math.atan2(-x,-y)) % 360, "capture":True,"layer":layer+1})
            # Close each layer so the next layer starts above the same vertex.
            x,y = ring[0]
            waypoints.append({"waypoint_id":len(waypoints)+1,"longitude":building.longitude+x/lon_scale,
                              "latitude":building.latitude+y/lat_scale,"relative_height_m":float(altitude),
                              "heading_deg":math.degrees(math.atan2(-x,-y))%360,"capture":False,"layer":layer+1})
        # Connect each layer to the next; distance is only an approximate planning quantity.
        distance = 0.0
        for a,b in zip(waypoints,waypoints[1:]):
            distance += math.sqrt(((b["longitude"]-a["longitude"])*lon_scale)**2+((b["latitude"]-a["latitude"])*lat_scale)**2+(b["relative_height_m"]-a["relative_height_m"])**2)
        checks = ["takeoff_reference_and_ground_elevation", "obstacles_and_clearance", "airspace_and_operation_permissions",
                  "wind_and_weather", "camera_overlap_and_visibility", "battery_return_and_emergency_path"]
        # Relative flight height is a property, not WGS84 ellipsoidal GeoJSON Z.
        features = [{"type":"Feature","geometry":{"type":"LineString","coordinates":[[p["longitude"],p["latitude"]] for p in waypoints]},
                     "properties":{"building_id":building.building_id,"status":"planning_draft","executable":False,
                                   "altitude_reference":"relative_to_launch_unverified","relative_heights_m":[p["relative_height_m"] for p in waypoints]}}]
        features += [{"type":"Feature","geometry":{"type":"Point","coordinates":[p["longitude"],p["latitude"]]},"properties":p} for p in waypoints]
        path = directory/"route.geojson"
        self.store.write_json(path, {"type":"FeatureCollection","features":features})
        result = {"job_id":job_id,"schema_version":"facade-route/v1","status":"planning_draft","executable":False,
                  "site":building.model_dump(),"options":settings.model_dump(),"geometry_method":method,
                  "waypoints":waypoints,"approximate_length_m":round(distance,2),"approximate_motion_seconds":round(distance/settings.speed_m_s,1),
                  "altitude_reference":"relative_to_launch_unverified","required_checks":checks,"route":self.store.artifact(path,"route_geojson",job_id)}
        self.store.write_json(directory/"route_plan.json",result)
        result["manifest"] = self.store.artifact(directory/"route_plan.json","route_manifest",job_id)
        self.store.complete_job(job_id,result)
        return result

    def organize_images(self, image_paths: list[str], building_id: str, capture_metadata: dict | None = None) -> dict:
        if not image_paths:
            raise ValueError("At least one image is required")
        job_id,directory = self.store.create_job("media", {"image_paths":image_paths,"building_id":building_id})
        rows, seen = [],set()
        try:
            for value in image_paths:
                original = self.store.resolve_input(value)
                digest = sha256(original)
                if digest in seen:
                    continue
                seen.add(digest)
                with Image.open(original) as image:
                    image.load()
                    if image.width*image.height > 100_000_000:
                        raise ValueError("Image exceeds 100 million pixel limit")
                    tags = {ExifTags.TAGS.get(k,str(k)):v for k,v in image.getexif().items()}
                    orientation = tags.get("Orientation",1)
                    # Preserve pixel layout: incoming detection boxes refer to stored pixels.
                    metadata = {"captured_at":tags.get("DateTimeOriginal") or tags.get("DateTime"),"exif_orientation":orientation}
                    width,height = image.size
                destination = directory/(digest[:16]+original.suffix.lower())
                shutil.copy2(original,destination)
                artifact = self.store.artifact(destination,"original_image",job_id)
                rows.append({"image_id":"image_"+digest[:20],"image_path":str(destination),"source_path":str(original),"sha256":digest,
                             "width":width,"height":height,"building_id":building_id,"metadata":metadata,
                             "capture_metadata":(capture_metadata or {}).get(original.name,{}),"artifact":artifact,
                             "warnings":[] if orientation==1 else ["EXIF orientation retained; bbox coordinates must use stored pixel layout"]})
            result = {"job_id":job_id,"schema_version":"facade-media/v1","status":"completed","images":rows,"duplicate_count":len(image_paths)-len(rows)}
            self.store.write_json(directory/"media_manifest.json",result)
            result["manifest"] = self.store.artifact(directory/"media_manifest.json","media_manifest",job_id)
            self.store.complete_job(job_id,result)
            return result
        except Exception as error:
            self.store.fail_job(job_id,str(error))
            raise
