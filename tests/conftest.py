import shutil

import pytest

from swarmguard.config import REPO_ROOT
from swarmguard.safety import integrity


@pytest.fixture
def repo_copy(tmp_path):
    """A temp copy of src/swarmguard with a fresh MANIFEST.json (never touches the real repo)."""
    root = tmp_path / "repo"
    shutil.copytree(
        REPO_ROOT / "src" / "swarmguard", root / "src" / "swarmguard",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (root / "README.md").write_text("x\n")
    integrity.write_manifest(integrity.generate_manifest(root), root)
    return root


@pytest.fixture
def clean_env():
    """An environment with no secret-like variables."""
    return {"PATH": "/usr/bin", "HOME": "/nonexistent"}
