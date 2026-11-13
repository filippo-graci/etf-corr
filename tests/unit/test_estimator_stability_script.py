"""Tests for the orchestration in ``scripts/estimator_stability_3_5.py`` (step 3.5), synthetic only.

The script is loaded by path (``scripts/`` is not a package). Parquet reading is replaced by a
stand-in (no parquet engine is needed offline), so what is tested is everything around the library
call: argument parsing, the sealed-window default, the core-ticker selection, the readable table and
the files written under ``reports/tables``. The real-data load is Gate 2.
"""

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

from etfcorr.correlation import stability as lib
from etfcorr.synthetic import make_factor_returns

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "estimator_stability_3_5.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("estimator_stability_3_5", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_script()


def _core_from_repo() -> list[str]:
    with open(REPO / "config" / "universe.yaml") as f:
        universe = yaml.safe_load(f)
    return [t["ticker"] for t in universe["tickers"] if t["panel"] == "core"]


def _blocks(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    n = len(frame) // k * k
    head = frame.iloc[:n]
    out = head.groupby(np.arange(n) // k).sum()
    out.index = head.index[k - 1 :: k]
    return out


@pytest.fixture(scope="module")
def tickers() -> list[str]:
    return _core_from_repo()


@pytest.fixture(scope="module")
def panels(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Synthetic returns for the real core tickers, 2014 to about 2026 (so the cut matters)."""
    sizes = (5, 5, 5, 5, 5)
    daily = make_factor_returns(3150, sizes, seed=3, start="2014-01-02").returns
    daily.columns = pd.Index(tickers)
    return {"D": daily, "W": _blocks(daily, 5), "M": _blocks(daily, 21), "Q": _blocks(daily, 63)}


@pytest.fixture
def fake_repo(
    tmp_path: Path, panels: dict[str, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> Path:
    (tmp_path / "config").mkdir()
    for name in ("universe.yaml", "experiment.yaml"):
        shutil.copy(REPO / "config" / name, tmp_path / "config" / name)
    monkeypatch.setattr(S, "find_repo_root", lambda: tmp_path)

    def fake_read(path: Path) -> pd.DataFrame:
        horizon = path.stem.split("_")[-1]
        extra = panels[horizon].copy()
        extra["NOT_CORE"] = 0.01 * np.arange(len(extra), dtype=float)  # must be ignored
        return extra

    monkeypatch.setattr(S, "read_returns", fake_read)
    return tmp_path


# --------------------------------------------------------------------------- configuration readers


def test_the_validation_end_and_default_draws_come_from_the_experiment_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "experiment.yaml").write_text(
        'splits:\n  val: ["2021-01-01", "2022-12-31"]\ninference:\n  n_bootstrap: 1234\n'
    )
    assert S.validation_end(tmp_path) == "2022-12-31"
    assert S.n_bootstrap_default(tmp_path) == 1234


def test_the_real_config_gives_the_documented_cut_and_draw_count() -> None:
    assert S.validation_end(REPO) == "2022-12-31"
    assert S.n_bootstrap_default(REPO) == 2000
    assert len(S.core_tickers(REPO)) == 25


def test_select_core_keeps_universe_order_and_rejects_a_missing_ticker(
    tickers: list[str], panels: dict[str, pd.DataFrame]
) -> None:
    shuffled = panels["D"][list(reversed(tickers))]
    shuffled = shuffled.assign(EXTRA=1.0)
    assert list(S.select_core(shuffled, tickers, "D").columns) == tickers
    with pytest.raises(SystemExit, match="missing core tickers"):
        S.select_core(shuffled.drop(columns=[tickers[0]]), tickers, "D")


def test_read_returns_flattens_a_two_level_column_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns = pd.MultiIndex.from_product([["AAA", "BBB"], ["log_return"]])
    frame = pd.DataFrame(np.arange(6.0).reshape(3, 2), columns=columns)
    monkeypatch.setattr(pd, "read_parquet", lambda path: frame)
    out = S.read_returns(Path("whatever.parquet"))
    assert list(out.columns) == ["AAA", "BBB"]
    plain = pd.DataFrame({"AAA": [1.0], "BBB": [2.0]})
    monkeypatch.setattr(pd, "read_parquet", lambda path: plain)
    assert list(S.read_returns(Path("whatever.parquet")).columns) == ["AAA", "BBB"]


# --------------------------------------------------------------------------- arguments


def test_default_arguments() -> None:
    args = S.parse_args([])
    assert args.quick is False and args.full_sample is False and args.end is None
    assert args.n_bootstrap is None and args.seed == 0
    assert tuple(args.estimators) == lib.ESTIMATORS and tuple(args.horizons) == S.HORIZONS


def test_end_and_full_sample_are_mutually_exclusive_and_choices_are_checked() -> None:
    with pytest.raises(SystemExit):
        S.parse_args(["--end", "2022-06-30", "--full-sample"])
    with pytest.raises(SystemExit):
        S.parse_args(["--estimators", "mystery"])
    with pytest.raises(SystemExit):
        S.parse_args(["--horizons", "X"])
    args = S.parse_args(
        ["--estimators", "pearson", "oas", "--horizons", "D", "Q", "--end", "2020-12-31"]
    )
    assert args.estimators == ["pearson", "oas"] and args.horizons == ["D", "Q"]
    assert args.end == "2020-12-31"


# --------------------------------------------------------------------------- end to end


def test_a_default_run_stops_at_the_validation_end_and_writes_both_files(
    fake_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = S.main(["--estimators", "pearson", "ledoit_wolf", "--n-bootstrap", "20"])
    assert code == 0
    out = capsys.readouterr().out
    assert "DESCRIPTIVE, NOT A TEST" in out and "sealed test window not read" in out
    csv_path = fake_repo / "reports" / "tables" / S.CSV_NAME
    md_path = fake_repo / "reports" / "tables" / S.MD_NAME
    table = pd.read_csv(csv_path)
    assert list(table.columns) == list(lib.TABLE_COLUMNS)
    assert len(table) == 4 * 2
    assert (pd.to_datetime(table["window_end"]) <= pd.Timestamp("2022-12-31")).all()
    assert (table["n_assets"] == 25).all()
    md = md_path.read_text()
    assert "sealed test window is not read" in md and "WARNING" not in md
    assert md.count("\n| ") >= 8 + 1  # header row plus one row per horizon and estimator
    assert "NOT_CORE" not in csv_path.read_text() + md


def test_full_sample_reads_the_sealed_window_and_says_so(
    fake_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = S.main(
        ["--estimators", "pearson", "--horizons", "D", "--n-bootstrap", "20", "--full-sample"]
    )
    assert code == 0
    assert "WARNING: reads the sealed test window" in capsys.readouterr().out
    table = pd.read_csv(fake_repo / "reports" / "tables" / S.CSV_NAME)
    assert pd.Timestamp(table.loc[0, "window_end"]) > pd.Timestamp("2022-12-31")
    assert (
        "WARNING: this table reads the sealed test window"
        in (fake_repo / "reports" / "tables" / S.MD_NAME).read_text()
    )


def test_an_explicit_end_later_than_the_validation_end_is_flagged_as_reading_the_sealed_window(
    fake_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    S.main(
        ["--estimators", "pearson", "--horizons", "D", "--n-bootstrap", "20", "--end", "2024-06-28"]
    )
    assert "WARNING: reads the sealed test window" in capsys.readouterr().out


def test_quick_run_prints_but_writes_nothing(
    fake_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = S.main(["--quick", "--estimators", "pearson", "--horizons", "D", "W"])
    assert code == 0
    out = capsys.readouterr().out
    assert "QUICK SMOKE RUN, NOT A RESULT" in out and "bootstrap draws 30" in out
    assert not (fake_repo / "reports").exists()


# --------------------------------------------------------------------------- markdown


def test_markdown_table_formats_singular_matrices_and_signs() -> None:
    panel = make_factor_returns(300, (3, 3, 3), seed=5).returns
    table = lib.estimator_stability_table({"D": panel}, ("pearson", "detoned"), n_bootstrap=20)
    text = S.markdown_table(table, ["# title"])
    rows = [line for line in text.splitlines() if line.startswith("| ")]
    assert len(rows) == 1 + 2  # header plus two data rows
    assert text.startswith("# title")
    assert "| detoned |" in text and "| inf |" in text
    assert all(row.count("|") == rows[0].count("|") for row in rows)


def test_markdown_header_lists_each_horizon_window_and_the_caveats() -> None:
    panel = make_factor_returns(400, (3, 3, 3), seed=6).returns
    table = lib.estimator_stability_table(
        {"D": panel, "W": _blocks(panel, 5)}, ("pearson",), n_bootstrap=20
    )
    lines = S.header_lines(table, n_bootstrap=20, seed=0, end="2022-12-31", reads_sealed=False)
    text = "\n".join(lines)
    assert "D: T=400" in text and "W: T=80" in text
    assert "not read (last date used: 2022-12-31)" in text
    assert "not higher accuracy" in text and "H5" in text
    flagged = "\n".join(S.header_lines(table, n_bootstrap=20, seed=0, end=None, reads_sealed=True))
    assert "WARNING" in flagged
