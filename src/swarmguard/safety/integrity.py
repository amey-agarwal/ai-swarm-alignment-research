"""File integrity (CLAUDE.md §4.7).

MANIFEST.json lists every tracked repo file with its SHA-256. Pass 1 and postflight
compare the repo against it. Any difference is a failure; differences under
src/swarmguard/safety/ are additionally flagged as `safety_changed`.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from swarmguard.config import REPO_ROOT, RundirCfg
from swarmguard.rundir import RunDir

MANIFEST_NAME = "MANIFEST.json"
MANIFEST_VERSION = 1
SAFETY_PREFIX = "src/swarmguard/safety/"

# Directory names never descended into (mirrors .gitignore).
EXCLUDED_DIRS = frozenset(
    {".git", ".venv", "venv", "runs", "__pycache__", ".pytest_cache", ".mypy_cache",
     ".ruff_cache", ".cache", "build", "dist", "models", "weights", "hf_cache"}
)
# File-name patterns never listed (mirrors .gitignore, plus local-only Claude settings).
EXCLUDED_FILE_PATTERNS = (
    MANIFEST_NAME, ".agent_edits.log", ".DS_Store", "*.pyc", "*.pyo", "*.sqlite",
    "*.sqlite-journal", "*.safetensors", "*.bin", "*.gguf", "settings.local.json",
)
EXCLUDED_DIR_PATTERNS = ("*.egg-info",)


class IntegrityError(Exception):
    pass


def _excluded_dir(name: str) -> bool:
    return name in EXCLUDED_DIRS or any(fnmatch.fnmatch(name, p) for p in EXCLUDED_DIR_PATTERNS)


def _excluded_file(name: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in EXCLUDED_FILE_PATTERNS)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class TreeScan:
    files: dict[str, str] = field(default_factory=dict)  # rel posix path -> sha256
    symlinks: list[str] = field(default_factory=list)


def scan_tree(root: Path = REPO_ROOT) -> TreeScan:
    root = Path(root).resolve()
    out = TreeScan()

    def walk(d: Path) -> None:
        for p in sorted(d.iterdir()):
            rel = p.relative_to(root).as_posix()
            if p.is_symlink():
                out.symlinks.append(rel)
                continue
            if p.is_dir():
                if not _excluded_dir(p.name):
                    walk(p)
            elif p.is_file() and not _excluded_file(p.name):
                out.files[rel] = sha256_file(p)

    walk(root)
    return out


def generate_manifest(root: Path = REPO_ROOT) -> dict:
    scan = scan_tree(root)
    if scan.symlinks:
        raise IntegrityError(f"symlinks are not allowed in the repo tree: {scan.symlinks}")
    return {"version": MANIFEST_VERSION, "files": dict(sorted(scan.files.items()))}


def load_manifest(root: Path = REPO_ROOT) -> dict:
    path = Path(root) / MANIFEST_NAME
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or data.get("version") != MANIFEST_VERSION or not isinstance(
        data.get("files"), dict
    ):
        raise IntegrityError(f"{path} is not a valid manifest")
    return data


def write_manifest(manifest: dict, root: Path = REPO_ROOT) -> Path:
    path = Path(root) / MANIFEST_NAME
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


@dataclass
class ManifestDiff:
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    symlinks: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return not (self.missing or self.extra or self.modified or self.symlinks or self.error)

    @property
    def safety_changed(self) -> bool:
        return any(p.startswith(SAFETY_PREFIX) for p in self.missing + self.extra + self.modified)

    def summary(self) -> str:
        if self.error:
            return self.error
        if self.ok:
            return "repo matches manifest"
        parts = []
        for name in ("missing", "extra", "modified", "symlinks"):
            items = getattr(self, name)
            if items:
                shown = ", ".join(items[:5]) + (" ..." if len(items) > 5 else "")
                parts.append(f"{name}: {shown}")
        if self.safety_changed:
            parts.insert(0, "SAFETY FILES CHANGED")
        return "; ".join(parts)


def check_manifest(root: Path = REPO_ROOT, manifest: dict | None = None) -> ManifestDiff:
    try:
        manifest = manifest if manifest is not None else load_manifest(root)
    except (OSError, ValueError, IntegrityError) as exc:
        return ManifestDiff(error=f"manifest unavailable: {exc}")
    expected: dict[str, str] = manifest["files"]
    scan = scan_tree(root)
    actual = scan.files
    return ManifestDiff(
        missing=sorted(set(expected) - set(actual)),
        extra=sorted(set(actual) - set(expected)),
        modified=sorted(p for p in set(expected) & set(actual) if expected[p] != actual[p]),
        symlinks=scan.symlinks,
    )


def safety_files_changed(new: dict, old: dict) -> list[str]:
    """Paths under safety/ that differ between two manifests."""
    a = {k: v for k, v in old.get("files", {}).items() if k.startswith(SAFETY_PREFIX)}
    b = {k: v for k, v in new.get("files", {}).items() if k.startswith(SAFETY_PREFIX)}
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


@dataclass
class RundirCheck:
    ok: bool
    unexpected: list[str]
    file_count: int
    total_bytes: int
    reasons: list[str]


def check_rundir(rd: RunDir, cfg: RundirCfg) -> RundirCheck:
    reasons = []
    unexpected = rd.unexpected()
    count = rd.file_count()
    total = rd.total_bytes()
    if unexpected:
        reasons.append(f"non-allowlisted entries: {unexpected}")
    if count > cfg.max_files:
        reasons.append(f"file count {count} > max_files {cfg.max_files}")
    if total > cfg.max_bytes:
        reasons.append(f"total bytes {total} > max_bytes {cfg.max_bytes}")
    return RundirCheck(not reasons, unexpected, count, total, reasons)
