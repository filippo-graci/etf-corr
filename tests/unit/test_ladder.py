"""Tests for ``etfcorr.evaluation.ladder`` (step 5.6) on synthetic panels with planted structure.

The real-data table is produced by ``scripts/baseline_table_5_6.py``; here the task builder, the
scoring of each loss and the summary are checked against known answers, and the whole ladder is
run on two small designs: a panel whose correlation switches between two persistent levels
(skill exists and must be found) and the sealed-window guard.
"""

import numpy as np
import pandas as pd
import pytest

from etfcorr import features as F
from etfcorr.config import Forecasting, Splits, Variant
from etfcorr.evaluation import ladder as L
from etfcorr.evaluation import metrics as M
from etfcorr.splits import SealedTestError, SplitError

SPLITS = Splits.model_validate(
    {
        "train": ["2016-01-01", "2018-12-31"],
        "val": ["2019-01-01", "2019-12-31"],
        "test": ["2020-01-01", "2020-12-31"],
        "walk_forward": {"initial_years": 3, "refit_every": "QE", "embargo_days": 2},
    }
)


def design(**changes: object) -> Forecasting:
    data: dict[str, object] = {
        "fixed_on": "2026-10-11",
        "primary_target": "T1",
        "primary_horizon_days": 21,
        "horizons_days": {"T1": [21], "T1_matrix": [21], "T2": [5], "T3": [5]},
        "feature_windows_days": [21, 63, 126],
        "simple_baselines": [
            "persistence",
            "historical_mean",
            "ewma",
            "constant_correlation",
            "har",
            "dcc",
        ],
        "linear_models": ["ridge", "pcr"],
        "losses": {"T1": "mse_fisher_z", "T1_matrix": "frobenius", "T2": "qlike", "T3": "rank_ic"},
        "inner_folds": 2,
        "alpha_grid": [0.01, 0.3],
        "l1_ratio": 0.5,
        "pcr_components": [1, 2],
    }
    data.update(changes)
    return Forecasting.model_validate(data)


def switching_panel(seed: int = 0) -> tuple[pd.DataFrame, dict[str, str]]:
    """Six assets in three sectors, business days 2016-2019. The market loading alternates
    between 0.2 and 0.8 in blocks of 130 days (cross-sector correlation 0.04 or 0.64), and the
    volatility of every asset doubles in the high blocks: both are persistent and forecastable."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2016-01-04", "2019-12-31", name="date")
    n = len(index)
    high = (np.arange(n) // 130) % 2 == 1
    loading = np.where(high, 0.8, 0.2)[:, None]
    market = rng.standard_normal((n, 1))
    noise = rng.standard_normal((n, 6))
    z = loading * market + np.sqrt(1 - loading**2) * noise
    vol = np.where(high, 0.02, 0.01)[:, None] * np.linspace(0.8, 1.2, 6)
    columns = [f"S{s}_{k}" for s in range(3) for k in range(2)]
    return pd.DataFrame(z * vol, index=index, columns=columns), {c: c[:2] for c in columns}


@pytest.fixture(scope="module")
def results() -> list[L.TaskResult]:
    returns, sectors = switching_panel()
    return L.run_ladder(returns, sectors, SPLITS, design(), n_bootstrap=50, seed=0)


# ---- task construction -----------------------------------------------------------------


def test_build_tasks_gives_each_target_its_ladder() -> None:
    returns, sectors = switching_panel()
    tasks = {t.name: t for t in L.build_tasks(returns, sectors, design(), embargo=2)}
    assert list(tasks) == ["T1_h21", "T1_matrix_h21", "T2_h5", "T3_h5"]
    names = {name: [m.name for m in t.models] for name, t in tasks.items()}
    assert names["T1_h21"] == [
        "persistence",
        "historical_mean",
        "ewma",
        "constant_correlation",
        "har",
        "ridge",
        "pcr",
    ]  # no DCC path was given, so the dcc baseline is absent
    assert names["T1_matrix_h21"] == [
        "persistence",
        "historical_mean",
        "ewma",
        "constant_correlation",
        "har",
    ]
    assert names["T2_h5"] == ["persistence", "historical_mean", "ewma", "har", "ridge", "pcr"]
    assert names["T3_h5"] == ["zero", "persistence", "momentum", "historical_mean", "ridge", "pcr"]
    assert {m.rung for m in tasks["T1_h21"].models} == {0, 1, 2}
    assert tasks["T3_h5"].higher_is_better and not tasks["T2_h5"].higher_is_better
    assert tasks["T3_h5"].benchmark == "zero" and tasks["T1_h21"].benchmark == "historical_mean"
    assert (
        tasks["T1_matrix_h21"].n_assets == 6 and len(tasks["T1_matrix_h21"].y) == len(returns) * 15
    )


def test_build_tasks_targets_filter_and_dcc_column() -> None:
    returns, sectors = switching_panel()
    base = np.linspace(0.1, 0.5, len(returns))
    avg = pd.DataFrame(
        {f"h{k}": base + 0.001 * k for k in range(1, 26)}, index=returns.index
    )  # the mean of h1..h21 is base + 0.011; columns h22..h25 must not be read
    (task,) = L.build_tasks(returns, sectors, design(), embargo=2, dcc_avg_corr=avg, targets=["T1"])
    assert "dcc" in [m.name for m in task.models]
    assert np.allclose(task.X["z_dcc_h21"], np.arctanh(base + 0.011))
    linear = next(m for m in task.models if m.name == "ridge").factory()
    assert "z_dcc_h21" not in linear.columns  # type: ignore[attr-defined]
    with pytest.raises(L.LadderError, match="lacks forecast columns"):
        L.dcc_feature(avg.iloc[:, :5], 21)


def test_build_tasks_needs_a_window_equal_to_the_horizon() -> None:
    returns, sectors = switching_panel()
    bad = design(
        horizons_days={"T1": [42], "T1_matrix": [21], "T2": [5], "T3": [5]}, primary_horizon_days=42
    )
    with pytest.raises(L.LadderError, match="persistence at horizon 42"):
        L.build_tasks(returns, sectors, bad, embargo=2)


# ---- scoring ---------------------------------------------------------------------------


def test_a_perfect_forecast_scores_zero_loss_and_unit_rank_ic() -> None:
    returns, sectors = switching_panel()
    for task in L.build_tasks(returns, sectors, design(), embargo=2):
        loss, name, _, share = L.per_date_loss(task, task.y)
        assert len(loss) > 900, task.name
        if task.higher_is_better:
            assert np.allclose(loss, 1.0) and name == "hit_rate"
        else:
            assert np.allclose(loss, 0.0, atol=1e-12), task.name
        if task.target == "T1_matrix":
            assert share == 0.0 and name == "psd_projected_share"


def test_matrix_scoring_projects_a_non_psd_forecast_and_reports_it() -> None:
    returns, sectors = switching_panel()
    task = next(
        t for t in L.build_tasks(returns, sectors, design(), embargo=2, targets=["T1_matrix"])
    )
    wide = task.y.unstack(F.UNIT)[list(task.y.index.get_level_values(F.UNIT).unique())]
    bad = wide.copy()
    bad.iloc[:, :] = np.arctanh(0.95)
    bad.iloc[:, 0] = np.arctanh(-0.95)  # one strongly negative pair among +0.95: not PSD
    forecast = F.stack_units({"y": bad})["y"].where(task.y.notna())
    loss, _, _, share = L.per_date_loss(task, forecast)
    assert share == 1.0 and (loss > 0).all()


def test_summarise_known_values() -> None:
    returns, sectors = switching_panel()
    task = next(t for t in L.build_tasks(returns, sectors, design(), embargo=2, targets=["T1"]))
    index = pd.bdate_range("2019-01-01", periods=100)
    names = [m.name for m in task.models]
    losses = pd.DataFrame({n: 1.0 for n in names}, index=index)
    losses["har"] = 0.5
    losses["ridge"] = 0.25
    secondary = {n: ("mae_correlation", 0.1) for n in names}
    best = L.best_simple(task, losses)
    table = L.summarise(task, losses, secondary, best, n_bootstrap=20).set_index("model")
    assert best == "har" and list(table.columns) == [c for c in L.TABLE_COLUMNS if c != "model"]
    assert table.loc["har", "best_simple"] and table["best_simple"].sum() == 1
    assert table.loc["ridge", "skill"] == pytest.approx(0.75)
    assert table.loc["har", "skill"] == pytest.approx(0.5)
    assert table.loc["historical_mean", "skill"] == pytest.approx(0.0)
    assert table.loc["ridge", "diff_vs_best_simple"] == pytest.approx(-0.25)
    assert table.loc["ridge", "diff_ci_upper"] == pytest.approx(-0.25)  # constant difference
    assert np.isnan(table.loc["har", "diff_ci_lower"])
    assert table.loc["har", "n_effective"] == pytest.approx(100 / 21)
    assert (table["rung"] <= 2).all() and table.loc["ridge", "rung"] == 2


def test_best_simple_ignores_rung_two_and_follows_the_loss_direction() -> None:
    returns, sectors = switching_panel()
    tasks = {t.target: t for t in L.build_tasks(returns, sectors, design(), embargo=2)}
    index = pd.bdate_range("2019-01-01", periods=10)
    t1 = pd.DataFrame({"persistence": 2.0, "har": 3.0, "ridge": 1.0}, index=index)
    assert L.best_simple(tasks["T1"], t1) == "persistence"  # ridge is better but is rung 2
    t3 = pd.DataFrame({"zero": 0.0, "momentum": 0.1, "ridge": 0.5}, index=index)
    assert L.best_simple(tasks["T3"], t3) == "momentum"  # higher rank IC is better
    with pytest.raises(L.LadderError):
        L.best_simple(tasks["T1"], t1[["ridge"]])


# ---- the whole ladder ------------------------------------------------------------------


def test_run_ladder_table_shape_and_dates(results: list[L.TaskResult]) -> None:
    table = L.ladder_table(results)
    assert list(table.columns) == list(L.TABLE_COLUMNS)
    assert len(table) == 7 + 5 + 6 + 6
    assert table.groupby("task")["best_simple"].sum().eq(1).all()
    assert np.isfinite(table["mean"]).all()
    for r in results:
        dates = r.losses.index
        assert dates.min() >= pd.Timestamp("2019-01-01") and dates.max() <= pd.Timestamp(
            "2019-12-31"
        )
        n = int(r.table["n_dates"].iloc[0])
        assert n == len(dates) and 200 < n <= 261
        # the last h origins of the sample have no complete target
        assert n == 261 - r.task.horizon
        assert r.predictions.loc[r.predictions.notna().any(axis=1)].index.get_level_values(
            0
        ).min() >= (pd.Timestamp("2019-01-01"))


def test_the_ladder_finds_planted_predictability(results: list[L.TaskResult]) -> None:
    """Correlation and volatility switch between two persistent levels, so the trailing
    statistics forecast both; the rank target has no planted signal."""
    by_name = {r.task.name: r.table.set_index("model") for r in results}
    for model in ("persistence", "ewma", "har", "ridge"):
        assert by_name["T1_h21"].loc[model, "skill"] > 0.3, model
    for model in ("ewma", "har", "ridge"):
        assert by_name["T2_h5"].loc[model, "skill"] > 0.25, model
    # Persistence of a 5-day variance is a different case: forecast and outcome are both
    # averages of 5 squared returns, so even with a constant true variance their ratio is a
    # ratio of two independent chi-square(5)/5 variables and the expected QLIKE is
    # E[v/f] - 1 = 5/3 - 1 = 0.67 (E[log(v/f)] = 0). It loses to smoother forecasts by
    # construction, here and on the real panel.
    assert by_name["T2_h5"].loc["persistence", "mean"] > 0.6
    assert by_name["T1_matrix_h21"].loc["har", "skill"] > 0.2
    assert by_name["T3_h5"].loc["zero", "mean"] == 0.0
    assert by_name["T3_h5"]["mean"].abs().max() < 0.15  # no rank signal was planted


def test_tuned_values_and_psd_shares_are_recorded(results: list[L.TaskResult]) -> None:
    by_name = {r.task.name: r for r in results}
    t1 = by_name["T1_h21"]
    assert set(t1.tuned) == {"ridge", "pcr"}
    assert sum(t1.tuned["ridge"].values()) == 4  # one tuned value per quarterly refit
    assert set(t1.tuned["ridge"]) <= {"0.01", "0.3"}
    assert sum(by_name["T2_h5"].tuned["ridge"].values()) == 4 * 6  # per refit and per asset
    matrix = by_name["T1_matrix_h21"]
    assert set(matrix.psd_projected_share) == {
        "persistence",
        "historical_mean",
        "ewma",
        "constant_correlation",
        "har",
    }
    for name in ("persistence", "ewma", "constant_correlation"):  # PSD by construction
        assert matrix.psd_projected_share[name] == 0.0


def test_table_lines_name_the_best_simple_baseline(results: list[L.TaskResult]) -> None:
    text = "\n".join(L.table_lines(results))
    assert text.count("(best simple baseline)") == 4
    assert "T3_h5: loss rank_ic (higher is better)" in text
    with pytest.raises(L.LadderError):
        L.ladder_table([])


def test_run_ladder_refuses_data_inside_the_sealed_window() -> None:
    returns, sectors = switching_panel()
    late = pd.bdate_range("2016-01-04", periods=len(returns) + 5, name="date")
    unsealed = pd.DataFrame(
        np.vstack([returns.to_numpy(), returns.to_numpy()[:5]]), index=late, columns=returns.columns
    )
    with pytest.raises(SealedTestError):
        L.run_ladder(unsealed, sectors, SPLITS, design(), n_bootstrap=10)
    market = pd.Series(0.0, index=late)
    with pytest.raises(SealedTestError):
        L.run_ladder(returns, sectors, SPLITS, design(), market=market, n_bootstrap=10)


def test_per_date_losses_agree_with_the_metric_functions(results: list[L.TaskResult]) -> None:
    t1 = next(r for r in results if r.task.name == "T1_h21")
    direct = M.squared_error(t1.task.y, t1.predictions["har"])
    assert np.allclose(t1.losses["har"], direct.loc[t1.losses.index])


# ---- declared variants (step 6.0) ------------------------------------------------------


def variant(**changes: object) -> Variant:
    data: dict[str, object] = {
        "id": "V1",
        "declared_on": "2026-10-11",
        "kind": "log_bias_correction",
        "targets": ["T2"],
        "note": "A declared deviation, run on validation only.",
    }
    data.update(changes)
    return Variant.model_validate(data)


@pytest.fixture(scope="module")
def base_t2(results: list[L.TaskResult]) -> L.TaskResult:
    return next(r for r in results if r.task.name == "T2_h5")


def test_bias_correction_variant_shifts_fitted_forecasts_only(base_t2: L.TaskResult) -> None:
    returns, sectors = switching_panel()
    (changed,) = L.run_ladder(returns, sectors, SPLITS, design(), variant=variant(), n_bootstrap=20)
    assert changed.task.name == "T2_h5"  # only the target the variant names is run
    for name in ("persistence", "ewma"):  # column forecasters are already variance levels
        pd.testing.assert_series_equal(changed.predictions[name], base_t2.predictions[name])
    for name in ("historical_mean", "har", "ridge", "pcr"):
        shift = (changed.predictions[name] - base_t2.predictions[name]).dropna()
        assert (shift > 0).all(), name  # half a residual variance is positive
    table = L.compare_with_base([base_t2], [changed], "V1", n_bootstrap=20)
    assert list(table.columns) == list(L.VARIANT_COLUMNS)
    assert set(table["model"]) == {
        "historical_mean",
        "har",
        "ridge",
        "pcr",
    }  # unchanged rows omitted
    assert (table["diff"] < 0).all()  # the planted variances are log-normal-like: QLIKE falls
    assert np.allclose(table["variant_mean"] - table["base_mean"], table["diff"])


def test_ridge_grid_variant_reaches_only_ridge() -> None:
    returns, sectors = switching_panel()
    wide = variant(id="V2", kind="ridge_grid", targets=["T1"], ridge_alpha_grid=[5.0, 50.0])
    (task,) = L.build_tasks(returns, sectors, design(), embargo=2, variant=wide, targets=["T1"])
    grids = {m.name: m.factory().grid for m in task.models if m.rung == 2}  # type: ignore[attr-defined]
    assert grids["ridge"] == (50.0, 5.0) and grids["pcr"] == (1, 2)
    (other,) = L.build_tasks(returns, sectors, design(), embargo=2, variant=wide, targets=["T3"])
    ridge = next(m for m in other.models if m.name == "ridge").factory()
    assert ridge.grid == (0.3, 0.01)  # type: ignore[attr-defined]  # T3 is not named by the variant


def test_training_window_variant_blanks_targets_and_must_precede_validation() -> None:
    returns, sectors = switching_panel()
    spring = variant(
        id="R1", kind="exclude_training_window", targets=["T1"], window=["2017-03-01", "2017-05-31"]
    )
    (task,) = L.build_tasks(returns, sectors, design(), embargo=2, variant=spring, targets=["T1"])
    (plain,) = L.build_tasks(returns, sectors, design(), embargo=2, targets=["T1"])
    gone = task.y.isna() & plain.y.notna()
    assert gone.sum() > 60 and gone[gone].index.min() < pd.Timestamp("2017-03-01")
    assert gone[gone].index.max() == pd.Timestamp("2017-05-31")
    late = variant(
        id="R1", kind="exclude_training_window", targets=["T1"], window=["2018-12-01", "2019-02-01"]
    )
    with pytest.raises(SplitError, match="before validation"):
        L.run_ladder(returns, sectors, SPLITS, design(), variant=late, n_bootstrap=10)


def test_compare_with_base_needs_the_task_under_the_design(base_t2: L.TaskResult) -> None:
    with pytest.raises(L.LadderError, match="was not run under the design"):
        L.compare_with_base([], [base_t2], "V1")
    same = L.compare_with_base([base_t2], [base_t2], "V1", n_bootstrap=10)
    assert same.empty and list(same.columns) == list(L.VARIANT_COLUMNS)


# ---- rung 3 in the ladder (step 6.3) and the test tables (step 6.4) --------------------

GBM_BLOCK = {
    "declared_on": "2026-10-11",
    "targets": ["T1", "T3"],
    "n_trials": 2,
    "n_jobs": 1,
    "search_space": {
        "num_leaves": [2, 4],
        "n_estimators": [20, 40],
        "min_data_in_leaf": [10, 20],
        "feature_fraction": [0.5, 0.9],
    },
}


@pytest.fixture(scope="module")
def gbm_results() -> list[L.TaskResult]:
    returns, sectors = switching_panel()
    return L.run_ladder(
        returns,
        sectors,
        SPLITS,
        design(gbm=GBM_BLOCK),
        targets=["T1", "T3"],
        gbm_seeds=(0, 1),
        n_bootstrap=30,
    )


def test_rung_three_is_added_only_with_seeds_and_only_to_its_targets() -> None:
    returns, sectors = switching_panel()
    with_gbm = design(gbm=GBM_BLOCK)
    plain = L.build_tasks(returns, sectors, with_gbm, embargo=2)
    assert all(m.rung <= 2 for t in plain for m in t.models)  # no seeds: the Phase 5 ladder
    tasks = {t.name: t for t in L.build_tasks(returns, sectors, with_gbm, embargo=2, gbm_seeds=[0])}
    has = {name: any(m.name == L.GBM_NAME for m in t.models) for name, t in tasks.items()}
    assert has == {"T1_h21": True, "T1_matrix_h21": False, "T2_h5": False, "T3_h5": True}
    spec = tasks["T1_h21"].models[-1]
    assert spec.rung == 3 and "z_cross_w21" in tasks["T1_h21"].gbm_columns
    with pytest.raises(L.LadderError, match="no gbm block"):
        L.gbm_factory(design(), 21, 2, ["a"], [0])


def test_rung_three_row_fold_parameters_and_the_comparator(
    gbm_results: list[L.TaskResult],
) -> None:
    t1 = gbm_results[0]
    table = t1.table.set_index("model")
    assert table.loc[L.GBM_NAME, "rung"] == 3 and np.isfinite(table.loc[L.GBM_NAME, "mean"])
    assert t1.best_simple != L.GBM_NAME  # the comparator is never a rung 2 or 3 model
    params = t1.fold_params[L.GBM_NAME]
    assert len(params) == 4 and all(p["n_trials"] == 2 and p["tuned"] for p in params)
    assert set(t1.fold_params) == {L.GBM_NAME}  # only rung 3 keeps its fitted parameters


def test_seed_spread_refits_single_seeds_with_the_tuned_parameters(
    gbm_results: list[L.TaskResult],
) -> None:
    returns, _ = switching_panel()
    calendar = pd.DatetimeIndex(returns.index)
    t1 = gbm_results[0]
    spread = L.seed_spread(t1, design(gbm=GBM_BLOCK), SPLITS, calendar, (0, 1))
    assert list(spread.index) == [0, 1] and spread.nunique() == 2
    ensemble = float(t1.losses[L.GBM_NAME].mean())
    assert ensemble <= spread.mean() + 1e-12  # squared error of a mean forecast (Jensen)
    # one seed alone reproduces the ladder's own forecast when the ladder used that one seed
    single = L.run_ladder(
        returns,
        {c: c[:2] for c in returns.columns},
        SPLITS,
        design(gbm=GBM_BLOCK),
        targets=["T1"],
        gbm_seeds=(1,),
        n_bootstrap=10,
    )[0]
    alone = L.seed_spread(single, design(gbm=GBM_BLOCK), SPLITS, calendar, (1,))
    assert alone[1] == pytest.approx(float(single.losses[L.GBM_NAME].mean()))
    no_gbm = L.run_ladder(
        returns,
        {c: c[:2] for c in returns.columns},
        SPLITS,
        design(),
        targets=["T1"],
        n_bootstrap=10,
    )[0]
    with pytest.raises(L.LadderError, match="without the rung-3 model"):
        L.seed_spread(no_gbm, design(gbm=GBM_BLOCK), SPLITS, calendar, (0,))


def test_dm_table_signs_and_roles(gbm_results: list[L.TaskResult]) -> None:
    from etfcorr.evaluation import report as R
    from etfcorr.evaluation.dm_mcs import diebold_mariano

    comparators = {
        "T1_h21": {"primary": "ewma", "stricter": "har"},
        "T3_h5": {"primary": "momentum"},
    }
    table = R.dm_table(gbm_results, L.GBM_NAME, comparators)
    assert list(table.columns) == list(R.DM_COLUMNS) and len(table) == 3
    assert table["role"].tolist() == ["primary", "stricter", "primary"]
    t1, t3 = gbm_results
    row = table.iloc[0]
    direct = diebold_mariano(t1.losses[L.GBM_NAME], t1.losses["ewma"], 21, alternative="less")
    assert row["dm_statistic"] == pytest.approx(direct.statistic)
    assert row["p_one_sided"] == pytest.approx(direct.p_value)
    assert row["mean_diff"] == pytest.approx(
        float((t1.losses[L.GBM_NAME] - t1.losses["ewma"]).mean())
    )
    # rank IC is a score: a model with the higher IC must get a negative mean_diff
    ic_gap = float((t3.losses[L.GBM_NAME] - t3.losses["momentum"]).mean())
    assert table.iloc[2]["mean_diff"] == pytest.approx(-ic_gap)
    assert R.dm_table(gbm_results, L.GBM_NAME, {}).empty
    with pytest.raises(L.LadderError, match="comparator"):
        R.dm_table(gbm_results, L.GBM_NAME, {"T1_h21": {"primary": "arima"}})
    with pytest.raises(L.LadderError, match="model"):
        R.dm_table(gbm_results, "arima", {"T1_h21": {"primary": "ewma"}})


def test_mcs_table_covers_every_model_of_every_task(gbm_results: list[L.TaskResult]) -> None:
    from etfcorr.evaluation import report as R

    table = R.mcs_table(gbm_results, size=0.10, reps=200, seed=0)
    assert set(table.columns) == {"task", "model", "p_value", "included", "mean_loss", "block_size"}
    assert len(table) == sum(r.losses.shape[1] for r in gbm_results)
    for name, block in table.groupby("task"):
        assert block["p_value"].max() == pytest.approx(1.0), name
        assert block["included"].any()
    t1 = table[table["task"] == "T1_h21"].set_index("model")
    assert t1.loc["har", "p_value"] > t1.loc["constant_correlation", "p_value"]  # planted ranking
    assert (t1["block_size"] == 42).all()
