"""Job-scoped outputs and immutable artifact references shared by MCP servers."""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ArtifactStore:
    def __init__(self, root: Path, allowed_input_roots: list[Path] | None = None):
        self.root = Path(root).expanduser().resolve()
        self.allowed_input_roots = [Path(p).expanduser().resolve() for p in (allowed_input_roots or [Path.cwd()])]
        self.allowed_input_roots.append(self.root)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)
        (self.root / "artifacts").mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "ArtifactStore":
        from dotenv import load_dotenv
        load_dotenv(Path(os.environ.get("FACADE_CONFIG_FILE",str(Path.cwd()/".env"))),override=False)
        roots = os.environ.get("FACADE_ALLOWED_INPUT_ROOTS", "")
        allowed = [Path(p) for p in roots.split(";") if p.strip()] or [Path.cwd()]
        return cls(Path(os.environ.get("FACADE_WORKSPACE", "./workspace")), allowed)

    @staticmethod
    def write_json(path: Path, data: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)

    def resolve_input(self, value: str) -> Path:
        if value.startswith("artifact:"):
            artifact = self.get_artifact(value.removeprefix("artifact:"))
            path = Path(artifact["path"]).resolve()
            if sha256(path) != artifact["sha256"]:
                raise ValueError("Artifact content changed; create a new artifact instead")
            return path
        path = Path(value).expanduser().resolve()
        if not any(path.is_relative_to(root) for root in self.allowed_input_roots):
            raise ValueError("Input is outside FACADE_ALLOWED_INPUT_ROOTS")
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path

    def create_job(self, kind: str, inputs: dict) -> tuple[str, Path]:
        safe_kind = re.sub(r"[^a-zA-Z0-9_-]", "_", kind)[:40]
        job_id = safe_kind + "_" + uuid.uuid4().hex
        directory = self.root / "jobs" / job_id
        directory.mkdir()
        self.write_json(directory / "job.json", {"job_id": job_id, "kind": kind, "status": "running", "created_at": utc_now(), "inputs": inputs})
        return job_id, directory

    def job_dir(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", job_id):
            raise ValueError("Invalid job_id")
        directory = self.root / "jobs" / job_id
        if not (directory / "job.json").is_file():
            raise FileNotFoundError("Unknown job_id")
        return directory

    def job_status(self, job_id: str) -> dict:
        return json.loads((self.job_dir(job_id) / "job.json").read_text(encoding="utf-8"))

    def complete_job(self, job_id: str, result: dict) -> dict:
        record = self.job_status(job_id)
        record.update(status="completed", completed_at=utc_now(), result=result)
        self.write_json(self.job_dir(job_id) / "job.json", record)
        return record

    def fail_job(self, job_id: str, error: str | dict) -> dict:
        record = self.job_status(job_id)
        record.update(status="failed", completed_at=utc_now(), error=error)
        self.write_json(self.job_dir(job_id) / "job.json", record)
        return record

    def artifact(self, path: Path, role: str, job_id: str) -> dict:
        path = Path(path).resolve()
        # Output artifacts must belong to this job; callers copy originals first.
        if not path.is_relative_to(self.job_dir(job_id).resolve()):
            raise ValueError("Artifact must be inside its job directory")
        if not path.is_file():
            raise FileNotFoundError(str(path))
        digest = sha256(path)
        artifact_id = hashlib.sha256((job_id + "|" + str(path.relative_to(self.root)) + "|" + digest).encode()).hexdigest()[:32]
        result = {"artifact_id": artifact_id, "uri": "artifact:" + artifact_id, "job_id": job_id,
                  "role": role, "path": str(path), "sha256": digest, "size_bytes": path.stat().st_size,
                  "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream"}
        self.write_json(self.root / "artifacts" / (artifact_id + ".json"), result)
        return result

    def get_artifact(self, artifact_id: str) -> dict:
        if not re.fullmatch(r"[a-f0-9]{32}", artifact_id):
            raise ValueError("Invalid artifact_id")
        path = self.root / "artifacts" / (artifact_id + ".json")
        if not path.is_file():
            raise FileNotFoundError("Unknown artifact_id")
        return json.loads(path.read_text(encoding="utf-8"))
