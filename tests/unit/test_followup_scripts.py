"""Tests for the logic of the follow-up scripts (docs/FOLLOW_UPS.md), on synthetic data.

The scripts are loaded by path; nothing here reads the project's data.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from etfcorr.config import DateWindow
from etfcorr.evaluation import portfolio as P
from etfcorr.synthetic import make_factor_returns

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H1, H3, H5 = load("followup_h1"), load("followup_h3"), load("followup_h5")


def test_h1_statistics_are_one_for_identical_panels_and_report_after_minus_before() -> None:
    daily = make_factor_returns(2100, (4, 4, 4), sector_loading=0.6, seed=2).returns
    same = H1.statistics({"D": daily, "W": daily.copy()}, 3)
    np.testing.assert_allclose(same.xs("before", level=2), 1.0)
    np.testing.assert_allclose(same.xs("diff", level=2), 0.0, atol=1e-12)
    panels = H1.aggregate(daily)
    assert [len(p) for p in panels.values()] == [2100, 420, 100]
    np.testing.assert_allclose(panels["W"].iloc[0], daily.iloc[:5].sum())
    stats = H1.statistics(panels, 3)
    assert set(stats.index.get_level_values(0)) == {"D-W", "D-M", "W-M"}
    assert set(stats.index.get_level_values(1)) == {"ari", "eigvec_1", "subspace_2"}
    for key in (("D-M", "ari"), ("W-M", "subspace_2")):
        assert stats[(*key, "diff")] == pytest.approx(
            stats[(*key, "after")] - stats[(*key, "before")]
        )
    assert stats[("D-M", "ari", "after")] == 1.0  # strong planted sectors survive detoning
    boot = H1.bootstrap(panels, 3, 5, 0)
    assert boot.shape == (5, len(stats)) and boot.equals(H1.bootstrap(panels, 3, 5, 0))


def test_h3_simulation_separates_an_optimal_baseline_from_one_that_overreacts() -> None:
    optimal = H3.simulate_tests(600, 300, 63, 1.0, 12, 0)
    overreacts = H3.simulate_tests(600, 300, 63, 0.5, 12, 0)
    assert list(optimal.columns) == ["p_dm_hln", "p_clark_west", "mean_loss_diff"]
    assert optimal["p_clark_west"].between(0, 1).all() and len(optimal) == 12
    # Clark-West reacts to a baseline that only needs rescaling; the noise features cost both
    assert overreacts["p_clark_west"].median() < 0.5 * optimal["p_clark_west"].median()
    assert optimal["mean_loss_diff"].mean() > 0  # no information: the larger model loses
    x = H3.ar1(np.random.default_rng(0), 20000, 0.9)[:, 0]
    assert x.var() == pytest.approx(1.0, abs=0.1) and np.corrcoef(x[1:], x[:-1])[0, 1] > 0.85


def test_h5_tables_compare_each_cell_with_the_sample_cell_of_its_own_portfolio() -> None:
    daily = make_factor_returns(700, (3, 3), seed=1, start="2019-01-01").returns
    window = DateWindow.model_validate(["2021-01-01", "2021-06-30"])
    panels = {"D": daily, "W": daily.resample("W-FRI").sum()}
    runs, skipped = H5.run_cells(
        panels, {"D": 120, "W": 400}, ("sample", "nonlinear"), window, 0.94, None
    )
    assert skipped == ["W"] and len(runs) == 4  # 2 estimators x 2 portfolios, daily only
    cells, pooled = H5.tables(runs, 5, 50, 0.05)
    by = cells.set_index(["portfolio", "estimator"])
    assert by.loc[("minvar", "sample"), "gross"] == pytest.approx(1.0)
    assert by.loc[("minvar_ls", "sample"), "gross"] >= 1.0
    for portfolio in H5.PORTFOLIOS:
        name, base = (P.cell_name(portfolio, "D", e) for e in ("nonlinear", "sample"))
        vol = {k: runs[k].returns.std(ddof=1) * np.sqrt(252) for k in (name, base)}
        assert by.loc[(portfolio, "nonlinear"), "diff_vs_sample"] == pytest.approx(
            vol[name] - vol[base]
        )
        row = pooled[(pooled["portfolio"] == portfolio) & (pooled["pool"] == "nonlinear")].iloc[0]
        assert row["mean_diff"] == pytest.approx(vol[name] - vol[base]) and row["n_cells"] == 1
        assert row["ci_lower"] <= row["ci_upper"] and 0 < row["p_one_sided"] <= 1
    assert isinstance(cells, pd.DataFrame) and by.loc[("minvar", "sample"), "diff_vs_sample"] == 0
