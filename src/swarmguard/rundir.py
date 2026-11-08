"""Run directory creation and the allowlist of files it may contain (CLAUDE.md §4.7)."""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

ALLOWED_FILES = frozenset(
    {
        "config.yaml",
        "log.jsonl",
        "cache.sqlite",
        "safety_report.json",
        "metrics.json",
        "impostor_session.json",
        "checkpoint.json",
        "log.jsonl.gz",
        "STOP",  # kill-switch control file (§4.9); created by the user / `swarm stop`
    }
)

RUN_ID_RE = re.compile(r"^\d{8}-\d{6}_[0-9a-f]{8}$")


class RunDirError(Exception):
    pass


def new_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%d-%H%M%S')}_{secrets.token_hex(4)}"


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id):
        raise RunDirError(f"invalid run id {run_id!r}")
    return run_id


class RunDir:
    """A run directory under `root`. Only allowlisted file names can be addressed."""

    def __init__(self, root: str | Path, run_id: str):
        self.run_id = validate_run_id(run_id)
        self.root = Path(root).resolve()
        self.path = self.root / self.run_id

    @classmethod
    def create(cls, root: str | Path, run_id: str | None = None) -> "RunDir":
        rd = cls(root, run_id or new_run_id())
        rd.root.mkdir(parents=True, exist_ok=True)
        rd.path.mkdir(exist_ok=False)
        return rd

    @classmethod
    def open_existing(cls, root: str | Path, run_id: str) -> "RunDir":
        rd = cls(root, run_id)
        if not rd.path.is_dir():
            raise RunDirError(f"run directory not found: {rd.path}")
        return rd

    def file(self, name: str) -> Path:
        """Return the path of an allowlisted file inside this run dir."""
        if name not in ALLOWED_FILES:
            raise RunDirError(f"file name {name!r} is not allowlisted for run directories")
        p = (self.path / name).resolve()
        if p.parent != self.path.resolve():
            raise RunDirError(f"{name!r} escapes the run directory")
        return p

    def open(self, name: str, mode: str = "r", **kwargs) -> IO:
        return open(self.file(name), mode, **kwargs)

    def entries(self) -> list[Path]:
        return sorted(self.path.iterdir())

    def unexpected(self) -> list[str]:
        """Names present in the run dir that are not allowlisted regular files."""
        bad = []
        for p in self.entries():
            if p.name not in ALLOWED_FILES or not p.is_file() or p.is_symlink():
                bad.append(p.name)
        return bad

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.entries() if p.is_file() and not p.is_symlink())

    def file_count(self) -> int:
        return len(self.entries())
