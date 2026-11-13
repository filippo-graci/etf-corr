"""Leakage tests for the graph models (step 7.1; plan section 9, item 1).

The graph must be built inside each training window. The pipeline ``returns -> features and
pair correlations -> purged folds -> graph + model -> forecast`` goes through the shared
detectors on a panel whose graph is pure sampling noise (one weak common factor, no sectors),
so that a graph built from any other sample is a different graph. The leaky twin builds its
graph once from the whole sample and must be caught.
"""

import numpy as np
import pandas as pd
import pytest

from etfcorr import features as F
from etfcorr import targets as T
from etfcorr.config import DateWindow
from etfcorr.leakage import LeakageError, assert_future_invariant, assert_no_lookahead
from etfcorr.models import graphs as G
from etfcorr.models.base import Factory, walk_forward_predict
from etfcorr.models.gbm import GBMForecaster
from etfcorr.models.har import HAR
from etfcorr.splits import expanding_walk_forward
from etfcorr.synthetic import make_factor_returns

N = 560
H = 21
EMBARGO = 5
WINDOWS = (21, 63)
PANEL = make_factor_returns(N, (8,), market_loading=0.3, sector_loading=0.0, seed=11)
ASSETS = list(PANEL.returns.columns)
SECTORS = {a: f"s{k % 2}" for k, a in enumerate(ASSETS)}  # arbitrary labels: the target's pairs
MASK = F.sector_pair_mask(ASSETS, SECTORS)
TEST = DateWindow.model_validate([str(PANEL.returns.index[300].date()), "2030-12-31"])
GRAPH_COLUMNS = G.graph_columns(WINDOWS)
GBM_SPACE = {"num_leaves": (2, 4), "n_estimators": (20, 40), "min_data_in_leaf": (10, 20)}


def folds_of(index: pd.DatetimeIndex) -> list:  # type: ignore[type-arg]
    return expanding_walk_forward(
        index, train_start=index[0], test=TEST, horizon=H, embargo=EMBARGO
    )


STARTS = [int(f.test[0]) for f in folds_of(pd.DatetimeIndex(PANEL.returns.index))]
CUTS = tuple(sorted({s + k for s in STARTS for k in (1, 6) if s + k < N} | {N - 1}))


def tables(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.concat([F.build_features(d, SECTORS, windows=WINDOWS), G.pair_table(d, WINDOWS)], axis=1)
    y = pd.Series(F.fisher_z(T.forward_average_correlation(d, H, MASK)), index=d.index)
    return X, y.rename("y")


def wrapped(kind: str, inner: Factory) -> Factory:
    return lambda: G.GraphAugmented(
        inner,
        kind,  # type: ignore[arg-type]
        ASSETS,
        SECTORS,
        windows=WINDOWS,
        similarity_window=63,
        knn_k=2,
        threshold_quantile=0.8,
    )


def linear() -> HAR:
    return HAR(GRAPH_COLUMNS)


def pipeline(factory: Factory) -> "callable[[pd.DataFrame], pd.Series]":  # type: ignore[valid-type]
    def run(d: pd.DataFrame) -> pd.Series:
        X, y = tables(d)
        index = pd.DatetimeIndex(d.index)
        return walk_forward_predict(factory, X, y, folds_of(index), index)

    return run


@pytest.mark.parametrize("kind", ["knn", "threshold", "mst", "pmfg", "cluster", "sector", "random"])
def test_graph_model_forecasts_have_no_lookahead(kind: str) -> None:
    run = pipeline(wrapped(kind, linear))
    assert_no_lookahead(run, PANEL.returns, CUTS)
    assert_future_invariant(run, PANEL.returns, CUTS, mode="noise")
    assert run(PANEL.returns).iloc[STARTS[0] :].notna().all()  # not a vacuous pass


def test_lightgbm_on_graph_features_has_no_lookahead() -> None:
    columns = [*F.build_features(PANEL.returns, SECTORS, windows=WINDOWS).columns, *GRAPH_COLUMNS]
    inner: Factory = lambda: GBMForecaster(  # noqa: E731
        GBM_SPACE,
        n_trials=2,
        seeds=(0,),
        horizon=H,
        embargo=EMBARGO,
        inner_folds=2,
        columns=columns,
    )
    assert_no_lookahead(pipeline(wrapped("mst", inner)), PANEL.returns, CUTS)


def test_the_graph_changes_between_refits() -> None:
    """Otherwise the twin below could not be told from the real thing."""
    X, y = tables(PANEL.returns)
    X = X.dropna()
    y = y.loc[X.index].fillna(0.0)
    early = wrapped("mst", linear)().fit(X.iloc[:200], y.iloc[:200])
    late = wrapped("mst", linear)().fit(X, y)
    assert (early.adjacency_ != late.adjacency_).any()  # type: ignore[attr-defined]


@pytest.mark.parametrize("kind", ["knn", "mst", "pmfg"])
def test_a_graph_built_from_the_whole_sample_is_caught(kind: str) -> None:
    def leaky(d: pd.DataFrame) -> pd.Series:
        X, y = tables(d)
        whole = wrapped(kind, linear)().fit(X.dropna(), y.loc[X.dropna().index].fillna(0.0))

        class FullSampleGraph(G.GraphAugmented):
            def fit(self, X: pd.DataFrame, y: pd.Series) -> "FullSampleGraph":
                self.adjacency_ = whole.adjacency_  # type: ignore[attr-defined]  # the leak
                self._model = self._factory().fit(self._augment(X), y)
                return self

        factory: Factory = lambda: FullSampleGraph(  # noqa: E731
            linear,
            kind,  # type: ignore[arg-type]
            ASSETS,
            SECTORS,
            windows=WINDOWS,
            similarity_window=63,
            knn_k=2,
            threshold_quantile=0.8,
        )
        index = pd.DatetimeIndex(d.index)
        return walk_forward_predict(factory, X, y, folds_of(index), index)

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, PANEL.returns, CUTS)
    with pytest.raises(LeakageError):
        assert_future_invariant(leaky, PANEL.returns, CUTS, mode="noise")


def test_the_ladder_rebuilds_the_graph_in_every_fold() -> None:
    """Through ``evaluation.ladder``: one fitted graph per fold, recorded with its edge count."""
    from etfcorr.config import Forecasting, Splits
    from etfcorr.evaluation import ladder as L

    splits = Splits.model_validate(
        {
            "train": ["2016-01-01", "2016-12-31"],
            "val": ["2017-01-01", "2017-12-31"],
            "test": ["2018-06-01", "2018-12-31"],
            "walk_forward": {"initial_years": 1, "refit_every": "QE", "embargo_days": 2},
        }
    )
    design = Forecasting.model_validate(
        {
            "fixed_on": "2026-10-11",
            "primary_target": "T1",
            "primary_horizon_days": 21,
            "horizons_days": {"T1": [21]},
            "feature_windows_days": [21, 63],
            "simple_baselines": ["persistence", "historical_mean", "ewma"],
            "losses": {"T1": "mse_fisher_z"},
            "inner_folds": 2,
            "alpha_grid": [0.1],
            "l1_ratio": 0.5,
            "pcr_components": [1],
            "gbm": {
                "declared_on": "2026-10-11",
                "targets": ["T1"],
                "n_trials": 2,
                "n_jobs": 1,
                "search_space": {k: list(v) for k, v in GBM_SPACE.items()},
            },
            "graph": {
                "declared_on": "2026-10-11",
                "targets": ["T1"],
                "kinds": ["mst", "random"],
                "windows_days": [21, 63],
                "similarity_window_days": 63,
                "knn_k": 2,
                "threshold_quantile": 0.8,
            },
        }
    )
    returns = PANEL.returns.loc[:"2017-12-31"]

    def forecasts(d: pd.DataFrame) -> pd.DataFrame:
        task = L.build_tasks(d, SECTORS, design, embargo=2, gbm_seeds=(0,))[0]
        return L.run_task(task, splits, pd.DatetimeIndex(d.index), n_bootstrap=5).predictions

    task = L.build_tasks(returns, SECTORS, design, embargo=2, gbm_seeds=(0,))[0]
    names = [m.name for m in task.models]
    assert names[-3:] == ["lightgbm", "lightgbm_graph_mst", "lightgbm_graph_random"]
    assert not any(c.startswith("pz") or c.startswith("g_z") for c in task.gbm_columns)
    result = L.run_task(task, splits, pd.DatetimeIndex(returns.index), n_bootstrap=5)
    folds = result.fold_params["lightgbm_graph_mst"]
    assert len(folds) == 4 and all(p["n_edges"] == len(ASSETS) - 1 for p in folds)
    assert np.isfinite(result.losses["lightgbm_graph_random"]).all()
    starts = [int(f.test[0]) for f in L.walk_forward_from_config(returns.index, splits, 21)]
    assert_no_lookahead(forecasts, returns, [s + 3 for s in starts[1:]])
