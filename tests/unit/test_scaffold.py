"""Scaffold tests for Phase 0 / Step 0.1.

They check packaging, import hygiene and repository rules (git-ignore policy for vendor
data, no secrets, no absolute paths). No network, no GPU, no real data.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import etfcorr

ROOT = Path(__file__).resolve().parents[2]
SUBPACKAGES = ["data", "correlation", "models", "evaluation", "viz"]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


# ------------------------------------------------------------------ packaging
def test_version_matches_pyproject() -> None:
    if etfcorr.__version__.endswith("+unknown"):
        pytest.skip("etfcorr is not installed (run `uv sync`); nothing to compare")
    assert etfcorr.__version__ == _pyproject()["project"]["version"]


def test_pyproject_contract() -> None:
    proj = _pyproject()["project"]
    assert proj["requires-python"] == ">=3.11,<3.14"
    assert set(proj["optional-dependencies"]) == {"dl", "xcheck", "app", "dev"}
    assert proj["scripts"]["etfcorr"] == "etfcorr.cli:app"
    deps = " ".join(proj["dependencies"])
    for lib in ["skfolio", "arch", "statsmodels", "lightgbm", "yfinance", "pydantic"]:
        assert lib in deps
    # torch must stay optional: CPU-only CI and the offline gate must not need it
    assert "torch" not in deps.replace("torch-geometric", "")


def test_python_version_file_is_allowed_by_pyproject() -> None:
    pin = (ROOT / ".python-version").read_text().strip()
    major, minor = (int(x) for x in pin.split(".")[:2])
    assert (3, 11) <= (major, minor) < (3, 14)


@pytest.mark.parametrize("sub", SUBPACKAGES)
def test_subpackages_import(sub: str) -> None:
    assert importlib.import_module(f"etfcorr.{sub}").__doc__


def test_import_has_no_side_effects(tmp_path: Path) -> None:
    """Import every module in a clean interpreter with sockets blocked and an empty cwd."""
    code = (
        "import socket\n"
        "def _blocked(*a, **k):\n"
        "    raise RuntimeError('network access at import time')\n"
        "socket.socket.connect = _blocked\n"
        "socket.create_connection = _blocked\n"
        "import etfcorr\n"
        + ("import etfcorr.cli\n" if importlib.util.find_spec("typer") else "")
        + "".join(f"import etfcorr.{s}\n" for s in SUBPACKAGES)
    )
    res = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert res.returncode == 0, res.stderr
    assert list(tmp_path.iterdir()) == []  # nothing written on import


# ------------------------------------------------------------------ repository rules
def test_repo_tree_matches_plan() -> None:
    expected = [
        "config", "data/raw", "data/interim", "data/processed", "data/external",
        "data/fixtures", "src/etfcorr/data", "src/etfcorr/correlation", "src/etfcorr/models",
        "src/etfcorr/evaluation", "src/etfcorr/viz", "scripts", "notebooks", "app", "artifacts",
        "reports/figures", "reports/tables", "tests/unit", "tests/integration", "tests/leakage",
        ".github/workflows",
    ]  # fmt: skip
    missing = [d for d in expected if not (ROOT / d).is_dir()]
    assert not missing, missing


def test_env_example_contains_no_values() -> None:
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            assert key.strip() and value.strip() == "", f"{key} must be empty in .env.example"


def test_no_absolute_user_paths_in_tracked_sources() -> None:
    pattern = re.compile(r"(/home/\w+|/Users/\w+|[A-Z]:\\\\Users)")
    offenders = []
    for folder in ["src", "scripts", "config"]:
        for f in (ROOT / folder).rglob("*"):
            is_text = f.is_file() and f.suffix in {".py", ".yaml", ".yml", ".toml", ".sh"}
            if is_text and pattern.search(f.read_text()):
                offenders.append(str(f.relative_to(ROOT)))
    assert not offenders, offenders


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_gitignore_policy(tmp_path: Path) -> None:
    """Vendor data and secrets are ignored; fixtures and download logs are not."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    shutil.copy(ROOT / ".gitignore", tmp_path / ".gitignore")

    def ignored(rel: str) -> bool:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
        rc = subprocess.run(["git", "check-ignore", "-q", rel], cwd=tmp_path, check=False)
        return rc.returncode == 0

    must_ignore = [
        ".env",
        "data/raw/prices_yf.parquet",
        "data/interim/prices_clean.parquet",
        "data/processed/returns_D.parquet",
        "data/external/fred.parquet",
        "artifacts/runs/abc/model.pt",
    ]
    must_track = [
        ".env.example",
        "data/raw/prices_yf.json",
        "data/fixtures/panel.parquet",
        "data/raw/.gitkeep",
    ]
    assert all(ignored(p) for p in must_ignore), [p for p in must_ignore if not ignored(p)]
    assert not any(ignored(p) for p in must_track), [p for p in must_track if ignored(p)]
