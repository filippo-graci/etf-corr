"""The committed config files must validate and match the committed pre-registration lock."""

import json
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path

import pytest

from etfcorr import config as C

ROOT = Path(__file__).resolve().parents[2]


def test_committed_configs_validate_and_lock_matches() -> None:
    project = C.load_project(ROOT)  # verifies hypotheses.lock too
    exp = project.experiment
    assert [h.id for h in exp.hypotheses] == ["H1", "H2", "H3", "H4", "H5", "H6"]
    # the follow-ups of docs/FOLLOW_UPS.md, declared 2026-10-12 after all six verdicts
    assert [h.id for h in exp.exploratory] == [f"E{k}" for k in range(1, 11)]


def test_committed_splits_match_project_definition() -> None:
    """train 2016-2020, validation 2021-2022, final test 2023 to 2026-09 (project instructions)."""
    s = C.load_experiment(ROOT / "config" / "experiment.yaml").splits
    assert (s.train.start.year, s.train.end.year) == (2016, 2020)
    assert (s.val.start.year, s.val.end.year) == (2021, 2022)
    assert (s.test.start.year, s.test.end.year, s.test.end.month) == (2023, 2026, 9)


def test_hypotheses_are_assigned_to_the_phases_that_test_them() -> None:
    exp = C.load_experiment(ROOT / "config" / "experiment.yaml")
    phases = {h.id: h.phase for h in exp.hypotheses}
    assert phases == {"H1": 3, "H2": 4, "H3": 7, "H4": 7, "H5": 8, "H6": 8}


def test_seed_universe_covers_the_seven_sectors_and_controls() -> None:
    uni = C.load_universe(ROOT / "config" / "universe.yaml")
    assert set(uni.sectors()) == {
        "ai_semis",
        "energy_trad",
        "clean_energy",
        "healthcare",
        "defense",
        "metals_mining",
        "fintech_crypto",
    }
    assert {"SPY", "^VIX"} <= set(uni.tickers_in("control"))
    assert next(t for t in uni.tickers if t.ticker == "USO").futures_based


def test_core_panel_follows_the_step_1_1b_rules() -> None:
    """Core = long history and liquid; the 1.1b panel plus IHF/REMX, promoted 2026-09-28
    (Question 3, docs/DECISION_LOG.md: lowest measured holdings overlap of the six eligible
    extended candidates). VDE, PBW, NLR and PICK stay extended."""
    uni = C.load_universe(ROOT / "config" / "universe.yaml")
    core = [t for t in uni.tickers if t.panel == "core"]
    assert (len(uni.tickers), len(core)) == (56, 25)
    assert len(uni.tickers_in("extended")) == 22
    assert len(uni.tickers_in("control")) == 9
    # the six ETFs moved out of core (history after the train start, or thin trading)
    moved = {"BOTZ", "AIQ", "BLOK", "ARKF", "FINX", "QCLN"}
    assert moved.isdisjoint({t.ticker for t in core})
    assert moved <= set(uni.tickers_in("extended"))
    # no core member starts later than the download start (+ 7 days: the audit's tolerance)
    cutoff = uni.start_date + timedelta(days=7)
    assert [t.ticker for t in core if t.inception is not None and t.inception > cutoff] == []
    # every extended ETF/series with a recorded inception is later than the download start
    assert all(
        t.inception is None or t.inception > uni.start_date
        for t in uni.tickers
        if t.panel != "control"
    )
    # sector coverage of core: six sectors, at least three members each, no fintech/crypto
    counts: dict[str, int] = {}
    for t in core:
        counts[t.sector] = counts.get(t.sector, 0) + 1
    assert counts == {
        "ai_semis": 3,
        "energy_trad": 4,
        "clean_energy": 4,
        "healthcare": 5,
        "defense": 3,
        "metals_mining": 6,
    }
    assert "fintech_crypto" not in counts
    # fintech/crypto exists only as extended sub-panels
    assert {t.panel for t in uni.tickers if t.sector == "fintech_crypto"} == {"extended"}


def test_futures_based_funds_are_never_core() -> None:
    """Roll yield makes futures-based funds a different object: never in the core panel."""
    uni = C.load_universe(ROOT / "config" / "universe.yaml")
    assert [t.ticker for t in uni.tickers if t.futures_based and t.panel == "core"] == []


def test_committed_lock_file_shape() -> None:
    lock = json.loads((ROOT / "config" / "hypotheses.lock").read_text())
    assert lock["algorithm"] == "sha256"
    assert len(lock["fingerprint"]) == 64


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_lock_file_is_not_gitignored(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    shutil.copy(ROOT / ".gitignore", tmp_path / ".gitignore")
    for rel in ("config/hypotheses.lock", "config/experiment.yaml", "config/universe.yaml"):
        (tmp_path / rel).parent.mkdir(exist_ok=True)
        (tmp_path / rel).write_text("x")
        rc = subprocess.run(["git", "check-ignore", "-q", rel], cwd=tmp_path, check=False)
        assert rc.returncode == 1, f"{rel} must be tracked"


def test_committed_forecasting_design_matches_the_h3_primary_target() -> None:
    """Step 5.1: H3's target is the 1-month-ahead average cross-sector correlation."""
    f = C.load_experiment(ROOT / "config" / "experiment.yaml").forecasting
    assert f is not None
    assert (f.primary_target, f.primary_horizon_days) == ("T1", 21)
    assert set(f.horizons_days) == {"T1", "T1_matrix", "T2", "T3"}
    assert {"persistence", "ewma", "har", "constant_correlation"} <= set(f.simple_baselines)


def test_committed_variants_and_gbm_budget_are_declared() -> None:
    """Step 6.0: two design candidates, two robustness runs, and the rung-3 budget."""
    exp = C.load_experiment(ROOT / "config" / "experiment.yaml")
    f = exp.forecasting
    assert f is not None and f.gbm is not None
    assert [v.id for v in f.variants] == ["V1", "V2", "R1", "R2"]
    assert f.variant("V2").ridge_alpha_grid[:6] == f.alpha_grid  # extended, not replaced
    assert f.variant("R2").assets == ("AMLP",)
    assert (f.gbm.n_trials, len(exp.seeds)) == (20, 5)


def test_committed_graph_kinds_and_final_plan_are_declared() -> None:
    """Step 7.0: seven graph kinds for H3's target, and the comparators of the final run."""
    f = C.load_experiment(ROOT / "config" / "experiment.yaml").forecasting
    assert f is not None and f.graph is not None and f.final is not None
    assert f.graph.targets == ("T1",) and len(f.graph.kinds) == 7 and "random" in f.graph.kinds
    assert (f.final.comparator_primary, f.final.comparator_strict) == ("ewma", "dcc")
    assert f.final.variant == "V1" and set(f.final.targets) == set(f.horizons_days)
    assert f.final.frozen == (f.final.primary_challenger is not None)


def test_committed_economics_design_is_declared() -> None:
    """Step 8.0: the H5 and H6 designs."""
    e = C.load_experiment(ROOT / "config" / "experiment.yaml").economics
    assert e is not None and dict(e.estimation) == {"D": 252, "W": 104, "M": 60, "Q": 32}
    assert {"rmt", "dcc"} < set(e.h5_estimators) and len(e.h5_estimators) == 3
    assert e.h5_variant is not None and e.h5_variant.candidate == "ledoit_wolf_corr"
    assert (e.hedge_instrument, e.stress_quantile, e.calm_quantile) == ("SPY", 0.9, 0.5)
    assert not {"UUP", "DBC", "^VIX"} & set(e.lasso_instruments)  # no futures-based fund
