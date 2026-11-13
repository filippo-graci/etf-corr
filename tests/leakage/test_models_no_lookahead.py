"""Leakage tests for the walk-forward driver and the model ladder (steps 5.4 and 5.5).

The whole pipeline ``returns -> features, target -> purged folds -> fitted model -> out-of-sample
forecast`` is treated as one feature function of the returns and run through the shared
detectors: the forecast dated ``t`` must not change when the sample is cut at any later date or
when the rows after it are replaced by noise. Cuts are placed at and just after fold starts
(standing lesson 2), where a model fitted with future labels would first show.

Leaky twins that must be caught: folds built without the purge, features standardised on the
whole sample before the walk-forward, and the target itself placed in the feature table.

The last block is the project's "synthetic noise gives about zero skill" rule: on a panel with no
structure the ladder shows no skill against the historical mean, while a model given the target
does (so the test can see skill when there is some). These are statements about fixed seeds of
one data-generating process, with thresholds set from the sampling error, not general results.
"""

import pandas as pd
import pytest

from etfcorr import features as F
from etfcorr import targets as T
from etfcorr.config import DateWindow
from etfcorr.evaluation import metrics as M
from etfcorr.leakage import LeakageError, assert_future_invariant, assert_no_lookahead
from etfcorr.models.base import Factory, LogBiasCorrected, walk_forward_predict
from etfcorr.models.baselines import HistoricalMean
from etfcorr.models.gbm import GBMForecaster
from etfcorr.models.har import HAR
from etfcorr.models.linear import PenalisedLinear
from etfcorr.splits import expanding_walk_forward
from etfcorr.synthetic import make_factor_returns

N = 560
H = 21
EMBARGO = 5
WINDOWS = (21, 63)
GRID = (0.01, 0.1, 1.0)
PANEL = make_factor_returns(N, (2, 2, 2), seed=17, start="2016-01-04")
ASSETS = list(PANEL.returns.columns)
SECTORS = dict(zip(ASSETS, PANEL.sectors, strict=True))
MASK = F.sector_pair_mask(ASSETS, SECTORS)
TEST = DateWindow.model_validate([str(PANEL.returns.index[300].date()), "2030-12-31"])
HAR_COLUMNS = ["z_cross_w21", "z_cross_w63"]
GBM_SPACE = {"num_leaves": (2, 4), "n_estimators": (20, 40), "min_data_in_leaf": (10, 20)}


def fold_starts() -> list[int]:
    index = pd.DatetimeIndex(PANEL.returns.index)
    folds = expanding_walk_forward(
        index, train_start=index[0], test=TEST, horizon=H, embargo=EMBARGO
    )
    return [int(f.test[0]) for f in folds]


STARTS = fold_starts()
# just after each fold start, well inside a fold, and the last row; all leave some future
CUTS = tuple(sorted({s + k for s in STARTS for k in (1, 6) if s + k < N} | {430, N - 1}))


def tables(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = F.build_features(d[ASSETS], SECTORS, windows=WINDOWS)
    y = pd.Series(F.fisher_z(T.forward_average_correlation(d[ASSETS], H, MASK)), index=d.index)
    return X, y.rename("y")


def pipeline(
    factory: Factory, *, purge_horizon: int = H, embargo: int = EMBARGO
) -> "callable[[pd.DataFrame], pd.Series]":  # type: ignore[valid-type]
    def run(d: pd.DataFrame) -> pd.Series:
        X, y = tables(d)
        index = pd.DatetimeIndex(d.index)
        folds = expanding_walk_forward(
            index, train_start=index[0], test=TEST, horizon=purge_horizon, embargo=embargo
        )
        return walk_forward_predict(factory, X, y, folds, index)

    return run


FACTORIES: dict[str, Factory] = {
    "historical_mean": HistoricalMean,
    "har": lambda: HAR(HAR_COLUMNS),
    "ridge": lambda: PenalisedLinear("ridge", GRID, horizon=H, embargo=EMBARGO),
    "lasso": lambda: PenalisedLinear("lasso", GRID, horizon=H, embargo=EMBARGO),
    "pcr": lambda: PenalisedLinear("pcr", (1, 2, 3), horizon=H, embargo=EMBARGO),
    "har_bias_corrected": lambda: LogBiasCorrected(lambda: HAR(HAR_COLUMNS)),
    "lightgbm": lambda: GBMForecaster(
        GBM_SPACE, n_trials=2, seeds=(0, 1), horizon=H, embargo=EMBARGO, inner_folds=2
    ),
}


def test_the_cut_points_cover_every_fold_start() -> None:
    assert len(STARTS) >= 4 and all(s + 1 in CUTS for s in STARTS)


# ---- the causal pipeline passes --------------------------------------------------------


@pytest.mark.parametrize("name", list(FACTORIES))
def test_out_of_sample_forecasts_have_no_lookahead(name: str) -> None:
    assert_no_lookahead(pipeline(FACTORIES[name]), PANEL.returns, CUTS)


@pytest.mark.parametrize("name", ["har", "ridge", "lightgbm"])
def test_out_of_sample_forecasts_ignore_a_noisy_future(name: str) -> None:
    assert_future_invariant(pipeline(FACTORIES[name]), PANEL.returns, CUTS, mode="noise")


def test_forecasts_exist_where_the_detectors_compare() -> None:
    """Guard against a vacuous pass: the compared range holds real forecasts."""
    pred = pipeline(FACTORIES["ridge"])(PANEL.returns)
    assert pred.iloc[STARTS[0] :].notna().all() and pred.iloc[: STARTS[0]].isna().all()


# ---- leaky twins are caught ------------------------------------------------------------


@pytest.mark.parametrize("name", ["historical_mean", "har", "ridge", "lightgbm"])
def test_a_walk_forward_without_the_purge_is_caught(name: str) -> None:
    """Folds built for a 1-day label: the last 20 training targets reach into the test fold,
    so they are known in the full sample and unknown when the sample is cut just after the
    fold starts."""
    leaky = pipeline(FACTORIES[name], purge_horizon=1, embargo=0)
    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, PANEL.returns, CUTS)
    with pytest.raises(LeakageError):
        assert_future_invariant(leaky, PANEL.returns, CUTS, mode="noise")


def test_full_sample_standardisation_before_the_walk_forward_is_caught() -> None:
    def leaky(d: pd.DataFrame) -> pd.Series:
        X, y = tables(d)
        X = (X - X.mean()) / X.std()  # the leak: statistics of the whole sample
        index = pd.DatetimeIndex(d.index)
        folds = expanding_walk_forward(
            index, train_start=index[0], test=TEST, horizon=H, embargo=EMBARGO
        )
        # a tree-like model that is NOT scale invariant in X: thresholded mean
        return walk_forward_predict(lambda: HAR(HAR_COLUMNS), X.clip(-1.0, 1.0), y, folds, index)

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, PANEL.returns, CUTS)
    with pytest.raises(LeakageError):
        assert_future_invariant(leaky, PANEL.returns, CUTS, mode="noise")


def test_the_target_inside_the_feature_table_is_caught() -> None:
    def leaky(d: pd.DataFrame) -> pd.Series:
        X, y = tables(d)
        index = pd.DatetimeIndex(d.index)
        folds = expanding_walk_forward(
            index, train_start=index[0], test=TEST, horizon=H, embargo=EMBARGO
        )
        return walk_forward_predict(lambda: HAR(["leak"]), X.assign(leak=y), y, folds, index)

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, PANEL.returns, CUTS)


# ---- synthetic noise gives about zero skill --------------------------------------------

NOISE_SEEDS = range(6)
NOISE_ROWS = 1500
NOISE_FIRST_TEST = 900


def noise_panel(seed: int) -> tuple[pd.DataFrame, dict[str, str], DateWindow]:
    panel = make_factor_returns(
        NOISE_ROWS, (3, 3, 3), market_loading=0.0, sector_loading=0.0, seed=seed
    )
    r = panel.returns
    window = DateWindow.model_validate([str(r.index[NOISE_FIRST_TEST].date()), "2030-12-31"])
    return r, dict(zip(r.columns, panel.sectors, strict=True)), window


def noise_r2(seed: int, factories: dict[str, Factory]) -> dict[str, float]:
    """Out-of-sample R2 against the historical mean on one noise panel, per model."""
    r, sectors, window = noise_panel(seed)
    mask = F.sector_pair_mask(list(r.columns), sectors)
    X = F.build_features(r, sectors, windows=(21, 63, 252))
    y = pd.Series(F.fisher_z(T.forward_average_correlation(r, H, mask)), index=r.index, name="y")
    X = X.assign(leak=y)  # read only by the leaky twin (the other models name their columns)
    index = pd.DatetimeIndex(r.index)
    folds = expanding_walk_forward(
        index, train_start=index[0], test=window, horizon=H, embargo=EMBARGO
    )
    benchmark = M.squared_error(y, walk_forward_predict(HistoricalMean, X, y, folds, index))
    assert len(benchmark) > 500
    out = {}
    for name, factory in factories.items():
        loss = M.squared_error(y, walk_forward_predict(factory, X, y, folds, index))
        out[name] = M.r2_oos(loss, benchmark)
    return out


def test_no_model_shows_skill_on_the_average_correlation_of_a_noise_panel() -> None:
    """True cross-sector correlation is zero at every date, so nothing is forecastable.

    One panel is not enough to say so: about 580 overlapping test dates are about 28
    independent ones, and over 30 seeds the R2 of each model had a mean between -0.03 and -0.16
    with a standard deviation of 0.05 to 0.18 and a largest single value of 0.10 (measured
    2026-10-11, this data-generating process only). The test therefore averages over six seeds:
    a mean R2 above 0.05 would be more than three standard errors above the measured null mean
    for every model.
    """
    columns = ["z_cross_w21", "z_cross_w63", "z_cross_w252"]
    factories: dict[str, Factory] = {
        "persistence_regression": lambda: HAR(columns[:1]),
        "har": lambda: HAR(columns),
        "ridge": lambda: PenalisedLinear(
            "ridge", GRID, horizon=H, embargo=EMBARGO, columns=columns
        ),
        "lightgbm": lambda: GBMForecaster(
            GBM_SPACE, n_trials=2, seeds=(0,), horizon=H, embargo=EMBARGO, columns=columns
        ),
        "leaky_twin": lambda: HAR(["leak"]),
    }
    results = pd.DataFrame([noise_r2(seed, factories) for seed in NOISE_SEEDS])
    means = results.mean()
    for name in ("persistence_regression", "har", "ridge", "lightgbm"):
        assert means[name] < 0.05, (name, results[name].round(3).tolist())
        assert results[name].max() < 0.35, (name, results[name].round(3).tolist())
    # the same harness reports skill on every panel when the target is handed to the model
    assert results["leaky_twin"].min() > 0.99


def test_no_model_shows_rank_skill_on_a_noise_panel() -> None:
    """Returns are independent over time, so no feature ranks the next 5-day returns. With 9
    assets a daily rank IC has a standard deviation near 0.35; over about 600 overlapping dates
    (about 120 independent ones) the mean IC has a standard error near 0.03."""
    r, _, window = noise_panel(20261011)
    X = F.build_asset_features(r, rank=True)
    y = F.stack_units({"y": T.forward_rank(r, 5)})["y"]
    index = pd.DatetimeIndex(r.index)
    folds = expanding_walk_forward(
        index, train_start=index[0], test=window, horizon=5, embargo=EMBARGO
    )
    for kind in ("ridge", "lasso"):
        pred = walk_forward_predict(
            lambda kind=kind: PenalisedLinear(kind, GRID, horizon=5, embargo=EMBARGO),  # type: ignore[misc]
            X,
            y,
            folds,
            index,
        )
        assert abs(M.rank_ic(y, pred).mean()) < 0.1, kind
    leak = walk_forward_predict(lambda: HAR(["leak"]), X.assign(leak=y), y, folds, index)
    assert M.rank_ic(y, leak).mean() > 0.99
