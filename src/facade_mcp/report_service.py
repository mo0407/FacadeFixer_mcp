"""Draft report adapter and standalone HTML renderer.

The upstream report project models frozen expert snapshots. This adapter uses
its evidence concepts but keeps model output as an explicitly unreviewed draft.
It does not call an LLM or mutate the source mask/detection documents.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import html
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "facade-report/v1"
EVIDENCE_ROLES = {"original", "crop", "mask", "overlay", "location"}
EVIDENCE_STATES = {
    "model_estimate", "measured", "pending_confirmation", "not_collected", "inaccessible"
}
ROLE_LABELS = {
    "original": "原图", "crop": "缺陷局部图", "mask": "分割掩膜",
    "overlay": "分割叠加图", "location": "位置图",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(value: dict[str, Any], prefix: str) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else "未提供"), quote=True)


def _has_value(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _json_view(value: Any) -> str:
    return _escape(json.dumps(value, ensure_ascii=False, indent=2, default=str))


class ReportService:
    def __init__(self, store: Any, repo_path: Path | None = None):
        self.store = store
        self.repo_path = Path(repo_path).resolve() if repo_path else None

    def _resolve(self, reference: Any) -> Path:
        if isinstance(reference, dict):
            value = reference.get("uri")
            if not value and reference.get("artifact_id"):
                value = "artifact:" + str(reference["artifact_id"]).removeprefix("artifact:")
            value = value or reference.get("path")
        else:
            value = reference
        if not isinstance(value, str) or not value:
            raise ValueError("Missing artifact reference")
        if re.fullmatch(r"[a-f0-9]{32}", value):
            value = "artifact:" + value
        return self.store.resolve_input(value)

    def _read_document(self, document_path: str) -> dict[str, Any]:
        path = self._resolve(document_path)
        with path.open("r", encoding="utf-8") as source:
            document = json.load(source)
        if not isinstance(document, dict):
            raise ValueError("Report document must be a JSON object")
        return document

    @staticmethod
    def _reference(value: Any) -> Any:
        if isinstance(value, dict) and isinstance(value.get("artifact"), dict):
            return copy.deepcopy(value["artifact"])
        if isinstance(value, (str, dict)):
            return copy.deepcopy(value)
        return None

    def _evidence(self, source: dict[str, Any], image_id: str, instance_id: str | None) -> list[dict[str, Any]]:
        """Use explicit role keys; list position never determines association."""
        evidence: list[dict[str, Any]] = []
        candidates = {
            "original": ("original_image", "original_artifact", "original_path"),
            "crop": ("crop", "crop_artifact", "crop_path"),
            "mask": ("mask", "mask_artifact", "mask_path"),
            "overlay": ("overlay", "overlay_artifact", "overlay_path"),
            "location": ("location", "location_artifact", "location_path"),
        }
        for role, names in candidates.items():
            value = next((source[key] for key in names if _has_value(source.get(key))), None)
            reference = self._reference(value)
            if reference is None:
                continue
            evidence.append({
                "evidence_id": f"{image_id}:{instance_id or 'image'}:{role}",
                "image_id": image_id, "instance_id": instance_id, "role": role,
                "state": "model_estimate", "artifact": reference,
                "approved_redaction": False, "redaction_review": None,
            })
        # Additional evidence is supported only when it carries its own role.
        for item in source.get("evidence", []):
            if not isinstance(item, dict) or item.get("role") not in EVIDENCE_ROLES:
                continue
            reference = self._reference(item.get("artifact", item.get("path")))
            if reference is None or any(row["role"] == item["role"] for row in evidence):
                continue
            row = copy.deepcopy(item)
            row.update({
                "evidence_id": item.get("evidence_id") or f"{image_id}:{instance_id or 'image'}:{item['role']}",
                "image_id": image_id, "instance_id": instance_id,
                "state": "model_estimate", "artifact": reference,
                "approved_redaction": False, "redaction_review": None,
            })
            evidence.append(row)
        return evidence

    @staticmethod
    def _detection_lookup(result: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
        lookup: dict[tuple[str, str], dict[str, Any]] = {}
        if not result:
            return lookup
        groups = result.get("images", [result])
        for group in groups:
            if not isinstance(group, dict):
                continue
            image_id = str(group.get("image_id", group.get("image", {}).get("image_id", "")))
            for item in group.get("instances", group.get("detections", group.get("defects", []))):
                if not isinstance(item, dict):
                    continue
                instance_id = item.get("instance_id", item.get("defect_id"))
                if image_id and instance_id:
                    lookup[(image_id, str(instance_id))] = item
        return lookup

    def build_draft(self, project: dict[str, Any], segmentation_result: dict[str, Any], detection_result: dict[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(project, dict) or not isinstance(segmentation_result, dict):
            raise ValueError("project and segmentation_result must be objects")
        job_id, job_dir = self.store.create_job("report_draft", {"project": project, "segmentation_result": segmentation_result, "detection_result": detection_result})
        try:
            groups = segmentation_result.get("images", [segmentation_result])
            if not isinstance(groups, list):
                raise ValueError("segmentation_result.images must be a list")
            detections = self._detection_lookup(detection_result)
            images: list[dict[str, Any]] = []
            defects: list[dict[str, Any]] = []
            seen_images: set[str] = set()
            seen_defects: set[str] = set()
            for group in groups:
                if not isinstance(group, dict):
                    raise ValueError("Each segmentation image must be an object")
                image_metadata = copy.deepcopy(group.get("image", {}))
                image_id = str(group.get("image_id") or image_metadata.get("image_id") or _stable_id(group, "image"))
                if image_id in seen_images:
                    raise ValueError(f"Duplicate image_id: {image_id}")
                seen_images.add(image_id)
                image_metadata.update({
                    "image_id": image_id,
                    "width": group.get("width", image_metadata.get("width")),
                    "height": group.get("height", image_metadata.get("height")),
                    "sha256": group.get("source_sha256", image_metadata.get("sha256")),
                    "path": group.get("image_path", group.get("source_path", image_metadata.get("path"))),
                })
                evidence_source = copy.deepcopy(group)
                if not evidence_source.get("original_image") and image_metadata.get("path"):
                    evidence_source["original_image"] = image_metadata["path"]
                image_metadata["evidence"] = self._evidence(evidence_source, image_id, None)
                images.append(image_metadata)
                sources = list(group.get("instances", group.get("defects", [])))
                existing_ids = {str(item.get("instance_id",item.get("id"))) for item in sources if isinstance(item,dict)}
                unresolved = list(group.get("unmapped_defects",[]))
                for detected_image in (detection_result or {}).get("images",[]):
                    if detected_image.get("image_id") == image_id:
                        unresolved.extend(detected_image.get("unmapped_defects",[]))
                for item in unresolved:
                    unresolved_id = str(item.get("instance_id",item.get("id")))
                    if unresolved_id not in existing_ids:
                        sources.append({**item,"instance_id":unresolved_id,"class_code":None,
                                        "class_name":item.get("original_class_name"),"area_pixels":None,
                                        "segmentation_status":"not_supported","original_image":group.get("original_image"),
                                        "quality_flags":["unsupported_mask_category"]})
                        existing_ids.add(unresolved_id)
                for source in sources:
                    if not isinstance(source, dict):
                        raise ValueError("Each segmentation instance must be an object")
                    instance_id = str(source.get("instance_id") or source.get("defect_id") or _stable_id(source, "instance"))
                    defect_id = f"{image_id}:{instance_id}"
                    if defect_id in seen_defects:
                        raise ValueError(f"Duplicate instance_id within image: {defect_id}")
                    seen_defects.add(defect_id)
                    # Join only on explicit image/instance identifiers, never list order.
                    detection = copy.deepcopy(detections.get((image_id, instance_id), {}))
                    defect = detection
                    defect.update(copy.deepcopy(source))
                    prior_conclusion = defect.get("expert_conclusion")
                    defect.update({
                        "defect_id": defect_id, "instance_id": instance_id, "image_id": image_id,
                    "building_id": source.get("building_id", detection.get("building_id", project.get("building_id",project.get("project_id")))),
                        "facade_id": source.get("facade_id", detection.get("facade_id")),
                        "category_id": source.get("class_code", source.get("category_id", detection.get("class_code", detection.get("category_id")))),
                        "category_name": source.get("class_name", source.get("category_name", detection.get("class_name", detection.get("category_name")))),
                        "evidence_state": "model_estimate", "expert_validated": False,
                        "expert_conclusion": {"status": "pending_confirmation", "reviewer_id": None, "signed_at": None, "signature_artifact": None},
                        "evidence": self._evidence(defect, image_id, instance_id),
                    })
                    if prior_conclusion is not None:
                        defect["source_expert_conclusion"] = prior_conclusion
                    # Pixel measurements remain pixels. No bbox-derived or guessed m².
                    pixel_area = source.get("area_pixels")
                    defect["area_pixels"] = pixel_area
                    defect["quantity"] = {"area_pixels": pixel_area, "source": "segmentation_mask", "state": "model_estimate"}
                    scale = source.get("scale", group.get("scale", project.get("scale")))
                    if isinstance(scale, dict):
                        defect["scale"] = copy.deepcopy(scale)
                        meters = scale.get("meters_per_pixel")
                        if isinstance(meters, (int, float)) and not isinstance(meters, bool) and math.isfinite(meters) and meters > 0 and _has_value(scale.get("source")) and isinstance(pixel_area, (int, float)):
                            defect["quantity"]["estimated_area_m2"] = pixel_area * meters * meters
                            defect["quantity"]["calibration_source"] = scale["source"]
                    defects.append(defect)
            document = {
                "schema_version": SCHEMA_VERSION, "report_id": job_id,
                "report_version": "1", "created_at": _now(), "status": "draft",
                "project": copy.deepcopy(project), "images": images, "defects": defects,
                "source_data": {"segmentation_result": copy.deepcopy(segmentation_result), "detection_result": copy.deepcopy(detection_result)},
                "provenance": {"adapter": "facade_mcp.report_service", "upstream_repo": str(self.repo_path) if self.repo_path else "https://github.com/mo0407/FacadeFixer_report", "upstream_contract": "draft adapter; not a frozen upstream ReportDocument"},
                "review_note": "模型识别与像素分割结果，尚未经专家确认。正式报告资格须检查签署记录与图片脱敏审批。",
            }
            path = Path(job_dir) / "report_document.json"
            self.store.write_json(path, document)
            artifact = self.store.artifact(path, "report_document", job_id)
            validation = self.validate(str(path))
            result = {"job_id": job_id, "status": "completed", "report_document": artifact, "artifact": artifact, "defect_count": len(defects),
                      "unsegmented_defect_count":sum(d.get("segmentation_status")=="not_supported" for d in defects),"validation": validation}
            self.store.complete_job(job_id, result)
            return result
        except Exception as error:
            self.store.fail_job(job_id, str(error))
            raise

    def _signed_review(self, review: Any, path: str, missing: list[str]) -> bool:
        if not isinstance(review, dict):
            missing.append(path)
            return False
        complete = True
        for key in ("reviewer_id", "signed_at", "signature_artifact"):
            if not _has_value(review.get(key)):
                missing.append(f"{path}.{key}")
                complete = False
        if complete:
            try:
                signed = review["signed_at"]
                datetime.fromisoformat(str(signed).replace("Z", "+00:00"))
                reference = review["signature_artifact"]
                if not self._resolve(reference).is_file():
                    raise ValueError("Missing signature file")
            except (ValueError, TypeError, OSError):
                missing.append(f"{path}.valid_signature_record")
                complete = False
        return complete

    def validate(self, document_path: str) -> dict[str, Any]:
        errors: list[str] = []
        missing: list[str] = []
        try:
            document = self._read_document(document_path)
        except (OSError, ValueError, TypeError) as error:
            return {"schema_valid": False, "draft_ready": False, "formal_ready": False, "errors": [str(error)], "missing_fields": []}
        for key in ("schema_version", "report_id", "project", "images", "defects"):
            if key not in document:
                errors.append(f"Missing schema field: {key}")
        if document.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        project = document.get("project")
        if not isinstance(project, dict):
            errors.append("project must be an object")
            project = {}
        for key in ("project_id", "name", "building_id", "inspection_date"):
            if not _has_value(project.get(key)):
                missing.append(f"project.{key}")
        images = document.get("images", [])
        defects = document.get("defects", [])
        if not isinstance(images, list) or not isinstance(defects, list):
            errors.append("images and defects must be lists")
            images = images if isinstance(images, list) else []
            defects = defects if isinstance(defects, list) else []
        image_ids: set[str] = set()
        evidence_groups: list[tuple[str, list[Any]]] = []
        for index, image in enumerate(images):
            path = f"images[{index}]"
            if not isinstance(image, dict) or not image.get("image_id"):
                errors.append(f"{path} requires image_id")
                continue
            if image["image_id"] in image_ids:
                errors.append(f"Duplicate image_id: {image['image_id']}")
            image_ids.add(image["image_id"])
            for key in ("width", "height"):
                value = image.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    missing.append(f"{path}.{key}")
            evidence_groups.append((path, image.get("evidence", [])))
        defect_ids: set[str] = set()
        for index, defect in enumerate(defects):
            path = f"defects[{index}]"
            if not isinstance(defect, dict):
                errors.append(f"{path} must be an object")
                continue
            for key in ("defect_id", "image_id", "instance_id"):
                if not defect.get(key):
                    errors.append(f"{path} requires {key}")
            defect_id = defect.get("defect_id")
            if defect_id in defect_ids:
                errors.append(f"Duplicate defect_id: {defect_id}")
            defect_ids.add(defect_id)
            if defect.get("image_id") not in image_ids:
                errors.append(f"{path}.image_id has no matching image")
            for key in ("building_id", "facade_id", "category_id", "bbox_xyxy", "area_pixels"):
                if not _has_value(defect.get(key)):
                    missing.append(f"{path}.{key}")
            pixel_area = defect.get("area_pixels")
            if defect.get("segmentation_status") == "not_supported":
                missing.append(f"{path}.segmentation_for_unsupported_category")
            if pixel_area is not None and (not isinstance(pixel_area, (int, float)) or isinstance(pixel_area, bool) or not math.isfinite(pixel_area) or pixel_area < 0):
                errors.append(f"{path}.area_pixels must be non-negative and finite")
            bbox = defect.get("bbox_xyxy")
            if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4 or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in bbox) or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]):
                errors.append(f"{path}.bbox_xyxy must be [x1,y1,x2,y2] with positive extent")
            if defect.get("evidence_state") not in EVIDENCE_STATES:
                errors.append(f"{path}.evidence_state is unsupported")
            conclusion = defect.get("expert_conclusion", {})
            if not isinstance(conclusion, dict) or conclusion.get("status") not in {"confirmed", "signed", "frozen"}:
                missing.append(f"{path}.expert_conclusion.confirmed_status")
            self._signed_review(conclusion, f"{path}.expert_conclusion", missing)
            evidence_groups.append((path, defect.get("evidence", [])))
        for path, evidence in evidence_groups:
            if not isinstance(evidence, list):
                errors.append(f"{path}.evidence must be a list")
                continue
            if not evidence:
                missing.append(f"{path}.evidence")
            for index, item in enumerate(evidence):
                evidence_path = f"{path}.evidence[{index}]"
                if not isinstance(item, dict):
                    errors.append(f"{evidence_path} must be an object")
                    continue
                if item.get("role") not in EVIDENCE_ROLES:
                    errors.append(f"{evidence_path}.role must be explicit")
                if item.get("state") not in EVIDENCE_STATES:
                    errors.append(f"{evidence_path}.state is unsupported")
                reference = item.get("artifact")
                try:
                    if not self._resolve(reference).is_file():
                        raise ValueError("Evidence file not found")
                except (ValueError, TypeError, OSError):
                    missing.append(f"{evidence_path}.available_artifact")
                if item.get("approved_redaction") is not True:
                    missing.append(f"{evidence_path}.approved_redaction")
                self._signed_review(item.get("redaction_review"), f"{evidence_path}.redaction_review", missing)
        if not images:
            missing.append("images")
        formal_signed = self._signed_review(document.get("formal_review"), "formal_review", missing)
        missing = list(dict.fromkeys(missing))
        schema_valid = not errors
        return {
            "schema_valid": schema_valid, "draft_ready": schema_valid,
            "formal_ready": schema_valid and not missing and formal_signed,
            "errors": errors, "missing_fields": missing,
            "verification_scope": "Checks record completeness and signature file existence; does not authenticate reviewer identity or cryptographic signatures.",
        }

    def _render_evidence(self, evidence: list[dict[str, Any]], assets_dir: Path, job_id: str, assets: dict[str, dict[str, Any]]) -> str:
        cards: list[str] = []
        if not isinstance(evidence, list):
            return '<p class="missing">证据列表格式不正确</p>'
        image_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
        for item in evidence:
            if not isinstance(item, dict):
                cards.append('<p class="missing">证据条目格式不正确</p>')
                continue
            role = item.get("role", "unknown")
            reference = item.get("artifact")
            try:
                source = self._resolve(reference)
                content = source.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                suffix = source.suffix.lower()
                asset_key = digest + suffix
                target = assets_dir / asset_key
                if asset_key not in assets:
                    shutil.copyfile(source, target)
                    assets[asset_key] = self.store.artifact(target, role, job_id)
                image_type = image_types.get(suffix)
                if image_type:
                    uri = f"data:{image_type};base64,{base64.b64encode(content).decode('ascii')}"
                    preview = f'<img src="{uri}" alt="{_escape(ROLE_LABELS.get(role, role))}" loading="lazy">'
                else:
                    preview = f'<a href="assets/{_escape(asset_key)}">下载证据文件</a>'
                cards.append(f'<figure data-role="{_escape(role)}" data-image-id="{_escape(item.get("image_id"))}" data-instance-id="{_escape(item.get("instance_id"))}">{preview}<figcaption>{_escape(ROLE_LABELS.get(role, role))} · {_escape(item.get("state"))}<br><small>{_escape(item.get("evidence_id"))}</small></figcaption></figure>')
            except (OSError, ValueError, TypeError) as error:
                cards.append(f'<figure class="missing"><figcaption>{_escape(ROLE_LABELS.get(role, role))}：证据暂不可用<br>{_escape(error)}</figcaption></figure>')
        return '<div class="evidence-grid">' + "".join(cards) + "</div>"

    def render(self, document_path: str) -> dict[str, Any]:
        document = self._read_document(document_path)
        validation = self.validate(document_path)
        job_id, job_dir = self.store.create_job("report_render", {"document_path": document_path})
        try:
            assets_dir = Path(job_dir) / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)
            assets: dict[str, dict[str, Any]] = {}
            project = document.get("project", {})
            project = project if isinstance(project, dict) else {}
            content = [f'<header><p>FacadeFixer · 建筑立面检测</p><h1>{_escape(project.get("name", "建筑立面检测草稿"))}</h1><p>报告编号：{_escape(document.get("report_id"))} · 状态：草稿 / 待确认</p></header>']
            content.append('<aside class="notice"><strong>模型估算 / 待专家确认</strong><p>像素面积来自分割掩膜。未经标定不显示物理面积。HTML为检测草稿，未自动赋予专家签署或图片脱敏审批。</p></aside>')
            fields = [("项目名称",project.get("name")),("建筑编号",project.get("building_id",project.get("project_id"))),
                      ("检测期次",project.get("period")),("检测日期",project.get("inspection_date")),("责任单位",project.get("responsible_party"))]
            site = project.get("site",{})
            if isinstance(site,dict) and site:
                fields.extend([("经度 / 纬度",f"{site.get('longitude')} / {site.get('latitude')}"),("建筑高度",f"{site.get('building_height_m')} m")])
            content.append('<section><h2>项目资料</h2><table>'+''.join(f'<tr><th>{_escape(label)}</th><td>{_escape(value)}</td></tr>' for label,value in fields)+'</table>')
            if project.get("coordinate_note"):
                content.append('<p class="missing">'+_escape(project["coordinate_note"])+"</p>")
            route = project.get("route_plan",{})
            if isinstance(route,dict) and route:
                content.append(f'<p>航线规划草案：{_escape(route.get("waypoint_count"))} 个航点，估算长度 {_escape(route.get("approximate_length_m"))} m。尚未核验为可执行航线。</p>')
            content.append('<details><summary>项目完整元数据</summary><pre>'+_json_view(project)+'</pre></details></section>')
            missing_labels = set()
            for flag in validation["missing_fields"]:
                if "inspection_date" in flag: missing_labels.add("检测日期")
                elif "facade_id" in flag: missing_labels.add("缺陷所在立面")
                elif "unsupported_category" in flag: missing_labels.add("暂不支持类别的分割结果")
                elif "area_pixels" in flag: missing_labels.add("缺陷像素面积")
                elif "redaction" in flag: missing_labels.add("证据图片脱敏审核记录")
                elif "expert_conclusion" in flag: missing_labels.add("专家复核与签署记录")
                elif "formal_review" in flag: missing_labels.add("报告审核记录")
                elif "available_artifact" in flag: missing_labels.add("缺失的证据文件")
                else: missing_labels.add("其他项目或证据资料")
            content.append('<section><h2>完整性检查</h2><p>结构校验：' + ('通过' if validation['schema_valid'] else '未通过') + ' · 正式报告资格：' + ('记录齐备（身份和签名未认证）' if validation['formal_ready'] else '尚不具备') + '</p>')
            if missing_labels:
                content.append('<p>待补充：'+_escape('、'.join(sorted(missing_labels)))+'。</p>')
            content.append('<details><summary>检查明细</summary><ul>'+''.join(f'<li>{_escape(flag)}</li>' for flag in validation['errors']+validation['missing_fields'])+'</ul></details></section>')
            images = document.get("images", [])
            images = images if isinstance(images, list) else []
            for image in images:
                if not isinstance(image, dict):
                    continue
                content.append(f'<section><h2>影像 {_escape(image.get("image_id"))}</h2><p>{_escape(image.get("width"))} × {_escape(image.get("height"))} 像素</p>')
                content.append(self._render_evidence(image.get("evidence", []), assets_dir, job_id, assets))
                content.append('<details><summary>影像元数据</summary><pre>' + _json_view(image) + '</pre></details></section>')
            defects = document.get("defects", [])
            defects = defects if isinstance(defects, list) else []
            content.append(f'<section><h2>识别结果</h2><p>本批记录 {len(defects)} 条候选缺陷，待复核。</p></section>')
            if not defects:
                content.append('<p class="missing">当前识别没有返回候选框，需要复核原图；该结果不证明建筑不存在缺陷。</p>')
            for defect in defects:
                if not isinstance(defect, dict):
                    continue
                content.append(f'<section class="defect" data-defect-id="{_escape(defect.get("defect_id"))}"><h2>{_escape(defect.get("category_name", defect.get("category_id")))} · {_escape(defect.get("instance_id"))}</h2><p>影像：{_escape(defect.get("image_id"))} · 建筑：{_escape(defect.get("building_id"))} · 立面：{_escape(defect.get("facade_id"))}</p>')
                if defect.get("segmentation_status") == "not_supported":
                    content.append('<p class="missing">该类别暂未接入分割算法，保留检测框，等待人工处理。</p>')
                content.append(f'<p>缺陷框：{_escape(defect.get("bbox_xyxy"))} · mask面积：{_escape(defect.get("area_pixels"))} 像素</p>')
                quantity = defect.get("quantity", {})
                if isinstance(quantity, dict) and quantity.get("estimated_area_m2") is not None:
                    content.append(f'<p>标定后估算面积：{_escape(quantity["estimated_area_m2"])} m² · 标定来源：{_escape(quantity.get("calibration_source"))}</p>')
                content.append(self._render_evidence(defect.get("evidence", []), assets_dir, job_id, assets))
                content.append('<details><summary>完整缺陷数据与方法记录</summary><pre>' + _json_view(defect) + '</pre></details></section>')
            content.append('<details><summary>报告完整数据（含原始输入）</summary><pre>' + _json_view(document) + '</pre></details>')
            css = 'body{font:15px/1.7 system-ui,sans-serif;margin:0;background:#f1f4f7;color:#243144}main{max-width:1100px;margin:auto;padding:32px}header,section,aside,details{background:#fff;padding:22px;margin:18px 0;border-radius:10px}h1,h2{line-height:1.3}h2{font-size:20px}.notice{border-left:5px solid #d58b22}.evidence-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}figure{margin:0;padding:12px;border:1px solid #dbe2e8;border-radius:6px}img{display:block;width:100%;max-height:420px;object-fit:contain}figcaption{margin-top:8px;font-size:13px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.missing{border-color:#d58b22}small{color:#667085}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #e5eaf0}th{width:30%;font-weight:500;color:#667085}@media print{body{background:white}main{padding:0}section,figure{break-inside:avoid}details{display:none}}'
            page = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\'"><title>' + _escape(project.get("name", "建筑立面检测草稿")) + '</title><style>' + css + '</style></head><body><main>' + "".join(content) + '</main></body></html>'
            path = Path(job_dir) / "report.html"
            path.write_text(page, encoding="utf-8")
            artifact = self.store.artifact(path, "report_html", job_id)
            result = {"job_id": job_id, "status": "completed", "html": artifact, "artifact": artifact, "assets": list(assets.values()), "validation": validation, "format": "standalone_html", "draft": True}
            self.store.complete_job(job_id, result)
            return result
        except Exception as error:
            self.store.fail_job(job_id, str(error))
            raise

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        path = self._resolve(artifact_id)
        return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
