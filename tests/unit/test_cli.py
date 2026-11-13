"""CLI smoke tests (need `typer`, part of the core dependencies)."""

import shutil
from pathlib import Path

import pytest

import etfcorr

ROOT = Path(__file__).resolve().parents[2]

typer_testing = pytest.importorskip("typer.testing")


def test_cli_version() -> None:
    from etfcorr.cli import app

    result = typer_testing.CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == etfcorr.__version__


def test_cli_unknown_command_fails() -> None:
    from etfcorr.cli import app

    assert typer_testing.CliRunner().invoke(app, ["no-such-step"]).exit_code != 0


def test_cli_check_config_on_committed_files() -> None:
    from etfcorr.cli import app

    result = typer_testing.CliRunner().invoke(app, ["check-config", "--root", str(ROOT)])
    assert result.exit_code == 0, result.output
    assert "config ok" in result.output
    assert "H1" in result.output


def test_cli_check_config_fails_when_a_hypothesis_is_edited(tmp_path: Path) -> None:
    from etfcorr.cli import app

    shutil.copytree(ROOT / "config", tmp_path / "config")
    exp = tmp_path / "config" / "experiment.yaml"
    exp.write_text(exp.read_text().replace("more", "less", 1))
    result = typer_testing.CliRunner().invoke(app, ["check-config", "--root", str(tmp_path)])
    assert result.exit_code == 1
    assert "CONFIG ERROR" in result.output


def test_cli_lock_hypotheses_writes_once(tmp_path: Path) -> None:
    from etfcorr.cli import app

    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "config" / "hypotheses.lock").unlink()
    runner = typer_testing.CliRunner()
    first = runner.invoke(app, ["lock-hypotheses", "--root", str(tmp_path)])
    assert first.exit_code == 0, first.output
    assert (tmp_path / "config" / "hypotheses.lock").is_file()
    second = runner.invoke(app, ["lock-hypotheses", "--root", str(tmp_path)])
    assert second.exit_code == 1
    assert "already exists" in second.output
