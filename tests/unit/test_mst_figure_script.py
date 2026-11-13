"""Tests for the orchestration in ``scripts/mst_figure_3_6.py`` (step 3.6), synthetic only.

The script is loaded by path (``scripts/`` is not a package). Parquet reading is replaced by a
stand-in (no parquet engine is needed offline), so what is tested is everything around the library
calls: argument parsing, the sealed-window default, the core-ticker selection, the tables, the
readable summary and the files written under ``reports/``. The real-data load is Gate 2.
"""

import importlib.util
import itertools
import shutil
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

from etfcorr.synthetic import make_factor_returns

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "mst_figure_3_6.py"
FAST = [
    "--n-bootstrap",
    "20",
    "--n-shuffles",
    "3",
    "--n-permutations",
    "20",
    "--n-null-draws",
    "20",
    "--rolling",
    "D:252:63",
    "W:104:13",
]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mst_figure_3_6", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_script()


def _core_from_repo() -> dict[str, str]:
    with open(REPO / "config" / "universe.yaml") as f:
        universe = yaml.safe_load(f)
    return {t["ticker"]: t["sector"] for t in universe["tickers"] if t["panel"] == "core"}


def _blocks(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    n = len(frame) // k * k
    head = frame.iloc[:n]
    out = head.groupby(np.arange(n) // k).sum()
    out.index = head.index[k - 1 :: k]
    return out


@pytest.fixture(scope="module")
def sectors() -> dict[str, str]:
    return _core_from_repo()


def _make_panels(sectors: dict[str, str], seed: int = 3) -> dict[str, pd.DataFrame]:
    """Synthetic returns for the real core tickers, 2014 to about 2026 (so the cut matters)."""
    sizes = tuple(len(list(g)) for _, g in itertools.groupby(sectors.values()))
    daily = make_factor_returns(
        3150, sizes, market_loading=0.3, sector_loading=0.8, seed=seed, start="2014-01-02"
    ).returns
    daily.columns = pd.Index(list(sectors))
    return {"D": daily, "W": _blocks(daily, 5), "M": _blocks(daily, 21), "Q": _blocks(daily, 63)}


@pytest.fixture(scope="module")
def panels(sectors: dict[str, str]) -> dict[str, pd.DataFrame]:
    return _make_panels(sectors)


def _install(
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


@pytest.fixture
def fake_repo(
    tmp_path: Path, panels: dict[str, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> Path:
    return _install(tmp_path, panels, monkeypatch)


def _tables(root: Path) -> Path:
    return root / "reports" / "tables"


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


def test_the_real_config_gives_the_documented_cut_draw_count_and_core_sectors(
    sectors: dict[str, str],
) -> None:
    assert S.validation_end(REPO) == "2022-12-31"
    assert S.n_bootstrap_default(REPO) == 2000
    assert S.core_sectors(REPO) == sectors
    assert len(sectors) == 25 and "control" not in sectors.values()


def test_select_core_keeps_universe_order_and_rejects_a_missing_ticker(
    sectors: dict[str, str], panels: dict[str, pd.DataFrame]
) -> None:
    tickers = list(sectors)
    shuffled = panels["D"][list(reversed(tickers))].assign(EXTRA=1.0)
    assert list(S.select_core(shuffled, tickers, "D").columns) == tickers
    with pytest.raises(SystemExit, match="missing core tickers"):
        S.select_core(shuffled.drop(columns=[tickers[0]]), tickers, "D")


def test_read_returns_flattens_a_two_level_column_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns = pd.MultiIndex.from_product([["AAA", "BBB"], ["log_return"]])
    frame = pd.DataFrame(np.arange(6.0).reshape(3, 2), columns=columns)
    monkeypatch.setattr(pd, "read_parquet", lambda path: frame)
    assert list(S.read_returns(Path("whatever.parquet")).columns) == ["AAA", "BBB"]
    plain = pd.DataFrame({"AAA": [1.0], "BBB": [2.0]})
    monkeypatch.setattr(pd, "read_parquet", lambda path: plain)
    assert list(S.read_returns(Path("whatever.parquet")).columns) == ["AAA", "BBB"]


# --------------------------------------------------------------------------- arguments


def test_default_arguments() -> None:
    args = S.parse_args([])
    assert args.quick is False and args.full_sample is False and args.end is None
    assert args.n_bootstrap is None and args.seed == 0
    assert tuple(args.horizons) == S.HORIZONS and tuple(args.rolling) == S.DEFAULT_ROLLING


def test_end_and_full_sample_are_mutually_exclusive_and_horizons_are_checked() -> None:
    with pytest.raises(SystemExit):
        S.parse_args(["--end", "2022-06-30", "--full-sample"])
    with pytest.raises(SystemExit):
        S.parse_args(["--horizons", "X"])
    args = S.parse_args(["--horizons", "D", "Q", "--end", "2020-12-31"])
    assert args.horizons == ["D", "Q"] and args.end == "2020-12-31"


def test_parse_rolling_reads_specs_and_drops_unselected_horizons() -> None:
    assert S.parse_rolling(["D:252:21", "W:104:4"], ["D", "W"]) == [("D", 252, 21), ("W", 104, 4)]
    assert S.parse_rolling(["D:252:21", "W:104:4"], ["W"]) == [("W", 104, 4)]
    assert S.parse_rolling([], ["D"]) == []


@pytest.mark.parametrize("spec", ["D", "D:252", "X:252:21", "D:a:21", "D:2:21", "D:252:0"])
def test_parse_rolling_rejects_malformed_specs(spec: str) -> None:
    with pytest.raises(SystemExit, match="bad --rolling spec"):
        S.parse_rolling([spec], ["D"])


# --------------------------------------------------------------------------- pieces


def test_edge_rows_carry_sector_labels_and_the_same_sector_flag(
    sectors: dict[str, str], panels: dict[str, pd.DataFrame]
) -> None:
    frame = panels["D"]
    support, _ = S.analyse_tree(
        frame, S.KINDS["raw"], sectors, n_bootstrap=20, n_permutations=20, seed=0
    )
    rows = S.edge_rows("D", "raw", support, sectors)
    assert len(rows) == len(sectors) - 1
    assert (rows["source_sector"] == rows["source"].map(sectors)).all()
    assert (rows["same_sector"] == (rows["source_sector"] == rows["target_sector"])).all()
    assert rows["support"].between(0, 1).all() and (rows["horizon"] == "D").all()


def test_analyse_tree_statistics_are_consistent_and_show_the_planted_sector_structure(
    sectors: dict[str, str], panels: dict[str, pd.DataFrame]
) -> None:
    support, stats = S.analyse_tree(
        panels["D"], S.KINDS["raw"], sectors, n_bootstrap=30, n_permutations=100, seed=0
    )
    assert stats["n_edges"] == len(sectors) - 1
    assert stats["mean_support"] == pytest.approx(support.edges["support"].mean())
    assert 0.0 <= stats["share_support_ge_half"] <= 1.0
    assert stats["purity"] > stats["purity_null_mean"] + 0.3  # sectors were planted
    assert stats["ari"] > 0.5 and stats["ari_p"] < 0.05
    assert stats["n_clusters"] == len(set(sectors.values()))
    assert stats["survival_lower"] <= stats["survival_mean"] <= stats["survival_upper"]


def test_cross_horizon_rows_compare_every_pair_with_the_noise_floor(
    sectors: dict[str, str], panels: dict[str, pd.DataFrame]
) -> None:
    trees = {}
    for kind, fn in S.KINDS.items():
        trees[kind] = {
            h: S.bootstrap_mst_edge_support(panels[h].iloc[-300:], corr_fn=fn, n_bootstrap=20).mst
            for h in ("D", "W", "M")
        }
    n_obs = {h: 300 for h in ("D", "W", "M")}
    table = S.cross_horizon_rows(trees, n_obs, n_null_draws=20, seed=0)
    assert len(table) == 2 * 3  # two matrix kinds, three horizon pairs
    assert set(table["matrix"]) == {"raw", "detoned"}
    np.testing.assert_allclose(table["noise_mean"], 2.0 / len(sectors))
    assert (table["noise_q95"] >= table["noise_mean"]).all()
    assert (table["above_noise_q95"] == (table["survival"] > table["noise_q95"])).all()
    assert table["survival"].between(0, 1).all()
    single = S.cross_horizon_rows({"raw": {"D": trees["raw"]["D"]}}, n_obs, n_null_draws=20, seed=0)
    assert single.empty and list(single.columns) == list(table.columns)


def test_the_noise_floor_of_a_pair_uses_the_smaller_number_of_observations(
    sectors: dict[str, str], panels: dict[str, pd.DataFrame]
) -> None:
    trees = {
        "raw": {
            "D": S.bootstrap_mst_edge_support(panels["D"].iloc[-200:], n_bootstrap=20).mst,
            "M": S.bootstrap_mst_edge_support(panels["M"].iloc[-40:], n_bootstrap=20).mst,
        }
    }
    table = S.cross_horizon_rows(trees, {"D": 200, "M": 40}, n_null_draws=40, seed=0)
    expected = S.noise_mst_survival_null(len(sectors), 40, 40, 0)
    assert table.loc[0, "noise_q95"] == expected.q95 and table.loc[0, "n_obs_a"] == 200


def test_the_share_of_well_supported_edges_counts_support_at_one_half(
    sectors: dict[str, str], panels: dict[str, pd.DataFrame]
) -> None:
    support, stats = S.analyse_tree(
        panels["M"], S.KINDS["raw"], sectors, n_bootstrap=20, n_permutations=20, seed=0
    )
    values = support.edges["support"].to_numpy()
    assert (values == 0.5).any(), "the fixture must contain a support of exactly one half"
    assert stats["share_support_ge_half"] == pytest.approx((values >= 0.5).mean())


def test_markdown_table_formats_numbers_booleans_and_strings() -> None:
    frame = pd.DataFrame({"a": ["x", "y"], "b": [1.23456, 2.0], "c": [True, False], "d": [3, 4]})
    text = S.markdown_table(frame, {"a": "str", "b": ".2f", "c": "bool", "d": "d"})
    assert text.splitlines() == [
        "| a | b | c | d |",
        "|---|---|---|---|",
        "| x | 1.23 | yes | 3 |",
        "| y | 2.00 | no | 4 |",
    ]


# --------------------------------------------------------------------------- end to end


def test_a_default_run_stops_at_the_validation_end_and_writes_every_output(
    fake_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert S.main(["--horizons", "D", "W", "M", *FAST]) == 0
    out = capsys.readouterr().out
    assert "DESCRIPTIVE AND EXPLORATORY, NOT A TEST" in out and "sealed test window not read" in out
    tables = _tables(fake_repo)
    figures = fake_repo / "reports" / "figures"
    assert (figures / "mst_grid_3_6.png").stat().st_size > 20_000
    assert b"<svg" in (figures / "mst_grid_3_6.svg").read_bytes()[:2000]

    edges = pd.read_csv(tables / S.EDGES_CSV)
    assert len(edges) == 3 * 2 * 24  # horizons x matrices x (N - 1) edges
    assert edges["support"].between(0, 1).all() and edges["correlation"].between(-1, 1).all()
    assert set(edges["matrix"]) == {"raw", "detoned"} and set(edges["horizon"]) == {"D", "W", "M"}

    summary = pd.read_csv(tables / S.SUMMARY_CSV)
    assert len(summary) == 3 * 2 and (summary["n_assets"] == 25).all()
    assert (pd.to_datetime(summary["window_end"]) <= pd.Timestamp("2022-12-31")).all()
    assert summary["purity_p"].between(0, 1).all() and summary["ari_p"].between(0, 1).all()
    daily_rows = summary[summary["horizon"] == "D"]
    assert (daily_rows["n_obs"] > 2000).all()  # about nine years of business days to 2022

    cross = pd.read_csv(tables / S.CROSS_CSV)
    assert len(cross) == 2 * 3
    rolling = pd.read_csv(tables / S.ROLLING_CSV)
    assert set(rolling["horizon"]) == {"D", "W"} and len(rolling) == 2 * 2
    assert rolling["lower_tail"].between(0, 1).all()

    md = (tables / S.SUMMARY_MD).read_text()
    assert "sealed test window is not read" in md and "WARNING" not in md
    assert "not a test of H1-H6" in md
    for heading in (
        "## Trees, edge support",
        "## Trees of different horizons",
        "## Rolling-window",
    ):
        assert heading in md
    assert "NOT_CORE" not in md + (tables / S.EDGES_CSV).read_text()


def test_outputs_are_reproducible_byte_for_byte(
    tmp_path: Path, panels: dict[str, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = []
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
        roots.append(_install(tmp_path / name, panels, monkeypatch))
        S.main(["--horizons", "D", "W", *FAST])
    first, second = roots
    files = [
        ("tables", S.EDGES_CSV),
        ("tables", S.SUMMARY_CSV),
        ("tables", S.CROSS_CSV),
        ("tables", S.ROLLING_CSV),
        ("tables", S.SUMMARY_MD),
        ("figures", "mst_grid_3_6.png"),
        ("figures", "mst_grid_3_6.svg"),
    ]
    for folder, name in files:
        a = (first / "reports" / folder / name).read_bytes()
        b = (second / "reports" / folder / name).read_bytes()
        assert a == b, name


def test_a_single_horizon_run_has_no_cross_horizon_section(
    fake_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert S.main(["--horizons", "D", *FAST]) == 0
    capsys.readouterr()
    assert pd.read_csv(_tables(fake_repo) / S.CROSS_CSV).empty
    md = (_tables(fake_repo) / S.SUMMARY_MD).read_text()
    assert "## Trees of different horizons" not in md and "## Rolling-window" in md
    only_d = pd.read_csv(_tables(fake_repo) / S.ROLLING_CSV)
    assert set(only_d["horizon"]) == {"D"}


def test_full_sample_reads_the_sealed_window_and_says_so(
    fake_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert S.main(["--horizons", "D", *FAST, "--full-sample"]) == 0
    assert "WARNING: reads the sealed test window" in capsys.readouterr().out
    summary = pd.read_csv(_tables(fake_repo) / S.SUMMARY_CSV)
    assert pd.Timestamp(summary.loc[0, "window_end"]) > pd.Timestamp("2022-12-31")
    md = (_tables(fake_repo) / S.SUMMARY_MD).read_text()
    assert "WARNING: this output reads the sealed test window" in md


def test_an_explicit_end_later_than_the_validation_end_is_flagged_as_reading_the_sealed_window(
    fake_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    S.main(["--horizons", "D", *FAST, "--end", "2024-06-28"])
    assert "WARNING: reads the sealed test window" in capsys.readouterr().out


def test_quick_run_prints_a_summary_but_writes_nothing(
    fake_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert S.main(["--quick", "--horizons", "D", "W", "--rolling", "D:252:63"]) == 0
    out = capsys.readouterr().out
    assert "QUICK SMOKE RUN, NOT A RESULT" in out and "bootstrap draws 30" in out
    assert "survival D-W" in out and "rolling D raw" in out and "purity" in out
    assert not (fake_repo / "reports").exists()


def test_a_missing_core_ticker_stops_the_run_before_any_output(
    fake_repo: Path, monkeypatch: pytest.MonkeyPatch, panels: dict[str, pd.DataFrame]
) -> None:
    def short_read(path: Path) -> pd.DataFrame:
        return panels["D"].drop(columns=[panels["D"].columns[0]])

    monkeypatch.setattr(S, "read_returns", short_read)
    with pytest.raises(SystemExit, match="missing core tickers"):
        S.main(["--horizons", "D", *FAST])
    assert not (fake_repo / "reports").exists()
