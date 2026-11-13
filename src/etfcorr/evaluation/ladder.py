"""Step 5.6: the baseline ladder (rungs 0 to 2) for targets T1 to T3 on the validation window.

This module turns the design fixed in ``config/experiment.yaml`` (``forecasting``) into tasks,
runs every model of the ladder through the purged walk-forward driver, and summarises the
out-of-sample losses. It is what ``scripts/baseline_table_5_6.py`` calls, and what Phase 6 will
call again to compare LightGBM with the same baselines on the same dates.

**Tasks** (one per target and horizon ``h``; origins are daily, so targets overlap):

========== ================================================= =========================== ==========
task       target (model space)                              feature table               loss
========== ================================================= =========================== ==========
T1         Fisher-z of the average cross-sector correlation  ``build_features``          squared
           over days ``t+1 .. t+h``                                                      error
T1_matrix  Fisher-z of every pairwise correlation            ``build_pair_features``     pair RMSE
T2         log of each asset's mean squared daily return     ``build_asset_features``    QLIKE
T3         cross-sectional rank of each asset's h-day return the same, ranked            rank IC
========== ================================================= =========================== ==========

**Ladder.** Rung 0: persistence (the trailing statistic over the last ``h`` rows), EWMA,
constant correlation (the longest-window average), historical mean. Rung 1: HAR on the trailing
statistic at three windows (per asset for T2, pooled over pairs for T1_matrix), and the causal
DCC forecast for T1 when a DCC path is supplied. Rung 2: ridge, lasso, elastic net and PCA
regression on the whole feature table (per asset for T2, pooled for T3). Not every baseline
exists for every target: T3 has no HAR, EWMA or constant-correlation analogue and gets a
zero forecast (the no-skill reference), persistence of the past ``h``-day return (its negative
is short-term reversal) and 12-1 momentum instead; T1_matrix stops at rung 1.

**"Best simple baseline"** in the table is the rung 0 or rung 1 model with the best mean
validation loss, over all of them (the historical mean and the DCC forecast included). The
pre-registered H3 names four baselines (persistence, EWMA, HAR-style, constant correlation):
by decision of 2026-10-11 the best of those four is H3's primary comparator and the best of
all rung 0-1 models is reported beside it as a stricter line. Both are chosen on validation
data and frozen before the test window is opened.

**Reading the table.** Every number is out-of-sample within the validation window (expanding
refits each quarter, purged by ``h`` plus the embargo), on the dates every model of the task
could forecast. ``skill`` is ``1 - loss / loss(historical mean)`` (for T3: the mean rank IC,
whose no-skill value is 0). ``diff_vs_best_simple`` is the mean per-date loss difference to the
best simple baseline with a stationary-bootstrap interval; negative is better (positive for
T3). These intervals are descriptive: the best baseline was picked on the same data, no
multiplicity correction is applied, and the formal tests (Diebold-Mariano with HLN, Clark-West
for nested pairs, Model Confidence Set) belong to Phase 6.

Limits: about 500 daily origins in validation are about ``500 / h`` independent observations
(24 at ``h = 21``, 8 at ``h = 63``); two years (2021-2022) are one market episode, not a sample
of regimes; the realised targets are noisy, so part of each loss is irreducible; a constant
forecast gets rank IC 0 by convention.
"""

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial

import numpy as np
import pandas as pd

from etfcorr import features as F
from etfcorr import targets as T
from etfcorr.config import (
    Forecasting,
    LinearModel,
    LossName,
    Project,
    Splits,
    TargetName,
    Variant,
)
from etfcorr.correlation.dcc import cross_sector_mask, rolling_dcc
from etfcorr.evaluation import metrics as M
from etfcorr.models import graphs as G
from etfcorr.models.base import (
    Factory,
    Forecaster,
    LogBiasCorrected,
    PerUnit,
    walk_forward_predict,
)
from etfcorr.models.baselines import ColumnForecaster, Constant, HistoricalMean
from etfcorr.models.gbm import GBMForecaster
from etfcorr.models.har import HAR
from etfcorr.models.linear import PenalisedLinear
from etfcorr.splits import (
    SplitError,
    assert_sealed,
    drop_training_window,
    final_walk_forward,
    walk_forward_from_config,
)

TABLE_COLUMNS: tuple[str, ...] = (
    "task",
    "target",
    "horizon",
    "model",
    "rung",
    "loss",
    "n_dates",
    "n_effective",
    "mean",
    "ci_lower",
    "ci_upper",
    "skill",
    "diff_vs_best_simple",
    "diff_ci_lower",
    "diff_ci_upper",
    "secondary_name",
    "secondary",
    "best_simple",
)
PCR_KIND: LinearModel = "pcr"
BENCHMARK = "historical_mean"  # reference of the skill column (T3 uses the zero forecast)


class LadderError(ValueError):
    """Invalid ladder request."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    rung: int
    factory: Factory


@dataclass(frozen=True)
class Task:
    """One target at one horizon, with its feature table and its ladder."""

    name: str
    target: TargetName
    horizon: int
    loss: LossName
    X: pd.DataFrame
    y: pd.Series
    models: tuple[ModelSpec, ...]
    benchmark: str
    n_assets: int = 0  # T1_matrix only
    gbm_columns: tuple[str, ...] = ()  # feature columns of the rung-3 model, if it is in the ladder
    bias_corrected: bool = False  # variant V1 applied to the fitted log-variance models

    @property
    def higher_is_better(self) -> bool:
        return self.loss == "rank_ic"


@dataclass(frozen=True)
class TaskResult:
    task: Task
    predictions: pd.DataFrame
    losses: pd.DataFrame  # per date (rows) and model (columns), on the common dates
    table: pd.DataFrame
    best_simple: str
    tuned: dict[str, dict[str, int]] = field(default_factory=dict)
    psd_projected_share: dict[str, float] = field(default_factory=dict)
    fold_params: dict[str, list[dict[str, object]]] = field(default_factory=dict)  # rung 3


# --------------------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------------------


def dcc_feature(avg_corr: pd.DataFrame, horizon: int) -> pd.Series:
    """Fisher-z of the DCC forecast of the average correlation over the next ``horizon`` days.

    ``avg_corr`` is ``rolling_dcc(...).avg_corr`` with columns ``h1 .. hK``, ``K >= horizon``:
    the value dated ``t`` in column ``hk`` forecasts day ``t + k`` with data up to ``t``. The
    mean over ``k = 1 .. horizon`` approximates the correlation realised over the window (the
    two differ: a realised correlation over a window is not the average of daily conditional
    correlations, and DCC correlates volatility-standardised returns).
    """
    columns = [f"h{k}" for k in range(1, horizon + 1)]
    missing = [c for c in columns if c not in avg_corr.columns]
    if missing:
        raise LadderError(f"the DCC path lacks forecast columns {missing[:3]}...")
    return pd.Series(F.fisher_z(avg_corr[columns].mean(axis=1)), index=avg_corr.index)


def _har_windows(design: Forecasting) -> tuple[int, ...]:
    w = design.feature_windows_days
    return tuple(dict.fromkeys((w[0], w[min(1, len(w) - 1)], w[-1])))


def _linear_specs(
    design: Forecasting,
    horizon: int,
    embargo: int,
    columns: Sequence[str],
    *,
    per_unit: bool,
    ridge_grid: Sequence[float] = (),
    bias_corrected: bool = False,
) -> list[ModelSpec]:
    specs = []
    for kind in design.linear_models:
        grid: Sequence[float] = design.pcr_components if kind == PCR_KIND else design.alpha_grid
        if kind == "ridge" and ridge_grid:
            grid = ridge_grid

        build: Factory = partial(
            PenalisedLinear,
            kind,
            grid,
            horizon=horizon,
            embargo=embargo,
            inner_folds=design.inner_folds,
            l1_ratio=design.l1_ratio,
            columns=tuple(columns),
        )
        if bias_corrected:
            build = partial(LogBiasCorrected, build)
        specs.append(ModelSpec(kind, 2, partial(PerUnit, build) if per_unit else build))
    return specs


GBM_NAME = "lightgbm"


def gbm_factory(
    design: Forecasting,
    horizon: int,
    embargo: int,
    columns: Sequence[str],
    seeds: Sequence[int],
    *,
    bias_corrected: bool = False,
    params_override: Mapping[str, float] | None = None,
) -> Factory:
    """Factory of the rung-3 model for one task (``forecasting.gbm`` of the config)."""
    if design.gbm is None:
        raise LadderError("the forecasting design has no gbm block")
    build: Factory = partial(
        GBMForecaster,
        design.gbm.search_space,
        n_trials=design.gbm.n_trials,
        seeds=tuple(seeds),
        horizon=horizon,
        embargo=embargo,
        inner_folds=design.inner_folds,
        columns=tuple(columns),
        n_jobs=design.gbm.n_jobs,
        params_override=params_override,
        name=GBM_NAME,
    )
    return partial(LogBiasCorrected, build) if bias_corrected else build


def _gbm_specs(
    design: Forecasting,
    target: TargetName,
    horizon: int,
    embargo: int,
    columns: Sequence[str],
    seeds: Sequence[int],
    *,
    bias_corrected: bool = False,
) -> list[ModelSpec]:
    if not seeds or design.gbm is None or target not in design.gbm.targets:
        return []
    factory = gbm_factory(design, horizon, embargo, columns, seeds, bias_corrected=bias_corrected)
    return [ModelSpec(GBM_NAME, 3, factory)]


def graph_name(kind: str) -> str:
    return f"{GBM_NAME}_graph_{kind}"


def _graph_specs(
    design: Forecasting,
    target: TargetName,
    horizon: int,
    embargo: int,
    columns: Sequence[str],
    seeds: Sequence[int],
    assets: Sequence[str],
    sectors: Mapping[str, str],
) -> list[ModelSpec]:
    """One rung-3 model per declared graph kind: LightGBM on ``columns`` plus graph aggregates."""
    graph = design.graph
    if not seeds or graph is None or design.gbm is None or target not in graph.targets:
        return []
    wide = [*columns, *G.graph_columns(graph.windows_days)]
    inner = gbm_factory(design, horizon, embargo, wide, seeds)
    return [
        ModelSpec(
            graph_name(kind),
            3,
            partial(
                G.GraphAugmented,
                inner,
                kind,
                tuple(assets),
                dict(sectors),
                windows=graph.windows_days,
                similarity_window=graph.similarity_window_days,
                knn_k=graph.knn_k,
                threshold_quantile=graph.threshold_quantile,
                seed=graph.seed,
                name=graph_name(kind),
            ),
        )
        for kind in graph.kinds
    ]


def _named(design: Forecasting, candidates: Mapping[str, ModelSpec]) -> list[ModelSpec]:
    return [candidates[name] for name in design.simple_baselines if name in candidates]


def _check_window(design: Forecasting, horizon: int, windows: Sequence[int], what: str) -> None:
    if horizon not in windows:
        raise LadderError(
            f"{what}: persistence at horizon {horizon} needs a trailing window of {horizon} "
            f"rows; available windows are {tuple(windows)}"
        )


def build_tasks(
    returns: pd.DataFrame,
    sectors: Mapping[str, str],
    design: Forecasting,
    *,
    embargo: int,
    market: pd.Series | None = None,
    vix: pd.Series | None = None,
    dcc_avg_corr: pd.DataFrame | None = None,
    lam: float = F.DEFAULT_LAMBDA,
    targets: Sequence[TargetName] | None = None,
    variant: Variant | None = None,
    gbm_seeds: Sequence[int] = (),
) -> list[Task]:
    """Every (target, horizon) task of ``design`` on ``returns`` (already sealed by the caller).

    ``variant`` applies one declared deviation (``config/experiment.yaml`` ``forecasting.variants``)
    to the targets it names: a lognormal bias correction of the fitted log-variance models, a
    different ridge grid, or training targets blanked over a date window. A ``dcc_drop_assets``
    variant changes nothing here: its caller supplies the other DCC path. ``gbm_seeds`` adds the
    rung-3 LightGBM model (pooled over assets for T2 and T3) to the targets of
    ``forecasting.gbm``; without seeds the ladder stops at rung 2, as in Phase 5. With seeds and
    a ``forecasting.graph`` block, the targets it names also get one LightGBM per graph kind
    (``models.graphs``), and their feature table gains the pairwise columns those models read.
    """
    wanted = tuple(design.horizons_days) if targets is None else tuple(targets)

    def applies(kind: str, target: TargetName) -> bool:
        return variant is not None and variant.kind == kind and target in variant.targets

    def ridge_grid(target: TargetName) -> Sequence[float]:
        return variant.ridge_alpha_grid if variant and applies("ridge_grid", target) else ()

    def training_target(task: Task) -> Task:
        if variant is None or not applies("exclude_training_window", task.target):
            return task
        assert variant.window is not None
        return replace(task, y=drop_training_window(task.y, variant.window, task.horizon))

    columns = [str(c) for c in returns.columns]
    windows = design.feature_windows_days
    common = F.build_features(returns, sectors, market=market, vix=vix, windows=windows, lam=lam)
    common_columns = list(common.columns)
    har_windows = _har_windows(design)
    tasks: list[Task] = []

    if "T1" in wanted:
        mask = F.sector_pair_mask(columns, sectors, "cross")
        graph = design.graph if gbm_seeds and design.gbm is not None else None
        with_graphs = graph is not None and "T1" in graph.targets
        pairs = G.pair_table(returns, graph.windows_days) if graph and with_graphs else None
        for h in design.horizons_days["T1"]:
            _check_window(design, h, windows, "T1")
            X = pd.concat([common, pairs], axis=1) if with_graphs else common.copy()
            y = pd.Series(
                F.fisher_z(T.forward_average_correlation(returns, h, mask)),
                index=returns.index,
                name=f"T1_h{h}",
            )
            har_columns = [f"z_cross_w{w}" for w in har_windows]
            candidates = {
                "persistence": ModelSpec(
                    "persistence", 0, partial(ColumnForecaster, f"z_cross_w{h}", "persistence")
                ),
                "ewma": ModelSpec("ewma", 0, partial(ColumnForecaster, "z_cross_ewma", "ewma")),
                "constant_correlation": ModelSpec(
                    "constant_correlation",
                    0,
                    partial(ColumnForecaster, f"z_cross_w{windows[-1]}", "constant_correlation"),
                ),
                "historical_mean": ModelSpec("historical_mean", 0, HistoricalMean),
                "har": ModelSpec("har", 1, partial(HAR, har_columns)),
            }
            if dcc_avg_corr is not None:
                X[f"z_dcc_h{h}"] = dcc_feature(dcc_avg_corr.reindex(X.index), h)
                candidates["dcc"] = ModelSpec(
                    "dcc", 1, partial(ColumnForecaster, f"z_dcc_h{h}", "dcc")
                )
            models = [
                *_named(design, candidates),
                *_linear_specs(
                    design, h, embargo, common_columns, per_unit=False, ridge_grid=ridge_grid("T1")
                ),
                *_gbm_specs(design, "T1", h, embargo, common_columns, gbm_seeds),
                *_graph_specs(
                    design, "T1", h, embargo, common_columns, gbm_seeds, columns, sectors
                ),
            ]
            tasks.append(
                Task(
                    f"T1_h{h}",
                    "T1",
                    h,
                    design.losses["T1"],
                    X,
                    y,
                    tuple(models),
                    BENCHMARK,
                    gbm_columns=tuple(common_columns),
                )
            )

    if "T1_matrix" in wanted:
        pair_x = F.build_pair_features(returns, windows=windows, lam=lam)
        for h in design.horizons_days["T1_matrix"]:
            _check_window(design, h, windows, "T1_matrix")
            wide = F.pairs_wide(T.forward_correlations(returns, h), returns.index, columns)
            y = F.stack_units({"y": wide})["y"].rename(f"T1_matrix_h{h}")
            har_columns = [f"z_w{w}" for w in har_windows]
            candidates = {
                "persistence": ModelSpec(
                    "persistence", 0, partial(ColumnForecaster, f"z_w{h}", "persistence")
                ),
                "ewma": ModelSpec("ewma", 0, partial(ColumnForecaster, "z_ewma", "ewma")),
                "constant_correlation": ModelSpec(
                    "constant_correlation",
                    0,
                    partial(ColumnForecaster, "z_const", "constant_correlation"),
                ),
                "historical_mean": ModelSpec("historical_mean", 0, HistoricalMean),
                "har": ModelSpec("har", 1, partial(HAR, har_columns)),
            }
            tasks.append(
                Task(
                    f"T1_matrix_h{h}",
                    "T1_matrix",
                    h,
                    design.losses["T1_matrix"],
                    pair_x,
                    y,
                    tuple(_named(design, candidates)),
                    BENCHMARK,
                    n_assets=len(columns),
                )
            )

    if "T2" in wanted:
        shared = [c for c in ("log_rv_xs_w21", "log_vix") if c in common.columns]
        asset_x = F.build_asset_features(returns, common=common[shared], lam=lam)
        asset_columns = list(asset_x.columns)
        har_columns = [f"log_rv_w{w}" for w in F.VOL_WINDOWS]
        corrected = applies("log_bias_correction", "T2")
        har_model: Factory = partial(HAR, har_columns)
        mean_model: Factory = HistoricalMean
        if corrected:  # per-asset residual variance for both fitted baselines
            har_model = partial(LogBiasCorrected, har_model)
            mean_model = partial(PerUnit, partial(LogBiasCorrected, HistoricalMean))
        for h in design.horizons_days["T2"]:
            _check_window(design, h, F.VOL_WINDOWS, "T2")
            y = F.stack_units({"y": F.log_variance(T.forward_variance(returns, h))})["y"]
            candidates = {
                "persistence": ModelSpec(
                    "persistence", 0, partial(ColumnForecaster, f"log_rv_w{h}", "persistence")
                ),
                "ewma": ModelSpec("ewma", 0, partial(ColumnForecaster, "log_ewma_var", "ewma")),
                "historical_mean": ModelSpec("historical_mean", 0, mean_model),
                "har": ModelSpec("har", 1, partial(PerUnit, har_model)),
            }
            models = [
                *_named(design, candidates),
                *_linear_specs(
                    design,
                    h,
                    embargo,
                    asset_columns,
                    per_unit=True,
                    ridge_grid=ridge_grid("T2"),
                    bias_corrected=corrected,
                ),
                *_gbm_specs(
                    design, "T2", h, embargo, asset_columns, gbm_seeds, bias_corrected=corrected
                ),
            ]
            tasks.append(
                Task(
                    f"T2_h{h}",
                    "T2",
                    h,
                    design.losses["T2"],
                    asset_x,
                    y.rename(f"T2_h{h}"),
                    tuple(models),
                    BENCHMARK,
                    gbm_columns=tuple(asset_columns),
                    bias_corrected=corrected,
                )
            )

    if "T3" in wanted:
        rank_x = F.build_asset_features(returns, lam=lam, rank=True)
        rank_columns = list(rank_x.columns)
        for h in design.horizons_days["T3"]:
            if f"ret_w{h}" not in rank_columns:
                raise LadderError(f"T3: no trailing-return column for horizon {h}")
            y = F.stack_units({"y": T.forward_rank(returns, h)})["y"].rename(f"T3_h{h}")
            models = [
                ModelSpec("zero", 0, partial(Constant, 0.0, "zero")),
                ModelSpec("persistence", 0, partial(ColumnForecaster, f"ret_w{h}", "persistence")),
                ModelSpec("momentum", 0, partial(ColumnForecaster, "mom_252_21", "momentum")),
                ModelSpec("historical_mean", 0, HistoricalMean),
                *_linear_specs(
                    design, h, embargo, rank_columns, per_unit=False, ridge_grid=ridge_grid("T3")
                ),
                *_gbm_specs(design, "T3", h, embargo, rank_columns, gbm_seeds),
            ]
            tasks.append(
                Task(
                    f"T3_h{h}",
                    "T3",
                    h,
                    design.losses["T3"],
                    rank_x,
                    y,
                    tuple(models),
                    "zero",
                    gbm_columns=tuple(rank_columns),
                )
            )
    return [training_target(task) for task in tasks]


# --------------------------------------------------------------------------------------
# Running and scoring
# --------------------------------------------------------------------------------------


def _chosen(model: Forecaster) -> list[object]:
    """The tuned value(s) of a fitted model: one, or one per unit; empty if nothing is tuned."""
    params = model.params
    if "chosen" in params:
        return [params["chosen"]]
    nested = [p["chosen"] for p in params.values() if isinstance(p, dict) and "chosen" in p]
    return list(nested)


def _matrix_losses(task: Task, forecast: pd.Series) -> tuple[pd.Series, float]:
    """Per-date pair RMSE in correlation units after a PSD projection, and how often it acted."""
    order = list(task.y.index.get_level_values(F.UNIT).unique())  # the order of pairs_wide
    if len(order) != task.n_assets * (task.n_assets - 1) // 2:
        raise LadderError("the matrix target does not hold every pair")
    true_wide = task.y.unstack(F.UNIT)[order]
    pred_wide = forecast.unstack(F.UNIT)[order]
    keep = np.asarray(pred_wide.notna().all(axis=1) & true_wide.notna().all(axis=1))
    dates = true_wide.index[keep]
    true = F.pairs_to_matrices(true_wide.loc[dates], task.n_assets)
    pred = F.pairs_to_matrices(pred_wide.loc[dates], task.n_assets)
    projected = 0
    for k in range(len(dates)):
        pred[k], changed = M.project_to_correlation(pred[k])
        projected += int(changed)
    share = projected / len(dates) if len(dates) else float("nan")
    return pd.Series(M.pair_rmse(true, pred), index=dates, name="pair_rmse"), share


def per_date_loss(task: Task, forecast: pd.Series) -> tuple[pd.Series, str, float, float]:
    """``(per-date loss, secondary metric name, its mean, PSD-projected share)`` for one model."""
    nan = float("nan")
    if task.loss == "mse_fisher_z":
        y, f = task.y, forecast
        secondary = M.absolute_error(
            pd.Series(F.inverse_fisher_z(y), index=y.index),
            pd.Series(F.inverse_fisher_z(f), index=f.index),
        )
        return M.squared_error(y, f), "mae_correlation", float(secondary.mean()), nan
    if task.loss == "qlike":
        loss = M.by_date(M.qlike(np.exp(task.y), np.exp(forecast)))
        mse_log = float(M.by_date(M.squared_error(task.y, forecast)).mean())
        return loss, "mse_log_variance", mse_log, nan
    if task.loss == "rank_ic":
        hit = float(M.hit_rate(task.y, forecast).mean())
        return M.rank_ic(task.y, forecast), "hit_rate", hit, nan
    if task.loss == "frobenius":
        loss, share = _matrix_losses(task, forecast)
        return loss, "psd_projected_share", share, share
    raise LadderError(f"no loss implemented for {task.loss!r}")


def run_task(
    task: Task,
    splits: Splits,
    calendar: pd.DatetimeIndex,
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    final: bool = False,
) -> TaskResult:
    """Walk-forward forecasts of every model of ``task`` on the validation window, scored.

    ``final=True`` evaluates on the test window instead (``splits.final_walk_forward``).
    """
    if final:
        folds = final_walk_forward(calendar, splits, task.horizon)
    else:
        folds = walk_forward_from_config(calendar, splits, task.horizon)
    predictions: dict[str, pd.Series] = {}
    losses: dict[str, pd.Series] = {}
    secondary: dict[str, tuple[str, float]] = {}
    tuned: dict[str, dict[str, int]] = {}
    psd: dict[str, float] = {}
    fold_params: dict[str, list[dict[str, object]]] = {}
    for spec in task.models:
        choices: Counter[str] = Counter()
        fitted: list[dict[str, object]] = []

        def record(
            _: int,
            model: Forecaster,
            choices: Counter[str] = choices,
            fitted: list[dict[str, object]] = fitted,
            keep: bool = spec.rung >= 3,
        ) -> None:
            choices.update(str(c) for c in _chosen(model))
            if keep:
                fitted.append(dict(model.params))

        pred = walk_forward_predict(spec.factory, task.X, task.y, folds, calendar, on_fit=record)
        predictions[spec.name] = pred
        loss, name, value, share = per_date_loss(task, pred)
        losses[spec.name] = loss
        secondary[spec.name] = (name, value)
        if choices:
            tuned[spec.name] = dict(sorted(choices.items()))
        if fitted:
            fold_params[spec.name] = fitted
        if not np.isnan(share):
            psd[spec.name] = share
    frame = pd.DataFrame(losses).dropna()
    if frame.empty:
        raise LadderError(f"{task.name}: no date on which every model has a forecast and a target")
    best = best_simple(task, frame)
    table = summarise(task, frame, secondary, best, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed)
    return TaskResult(task, pd.DataFrame(predictions), frame, table, best, tuned, psd, fold_params)


def best_simple(task: Task, losses: pd.DataFrame) -> str:
    """The rung 0 or rung 1 model with the best mean loss (first one on a tie)."""
    simple = [s.name for s in task.models if s.rung <= 1 and s.name in losses.columns]
    if not simple:
        raise LadderError(f"{task.name}: no simple baseline to compare with")
    means = losses[simple].mean()
    return str(means.idxmax() if task.higher_is_better else means.idxmin())


def summarise(
    task: Task,
    losses: pd.DataFrame,
    secondary: Mapping[str, tuple[str, float]],
    best: str,
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per model: mean loss with an interval, skill, and the gap to the best baseline."""
    n = len(losses)
    block = M.block_length_for(task.horizon, n)
    rung = {s.name: s.rung for s in task.models}

    def interval(values: pd.Series) -> M.MeanCI:
        return M.mean_ci(
            values, mean_block_length=block, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed
        )

    rows = []
    for name in losses.columns:
        level = interval(losses[name])
        gap = interval(losses[name] - losses[best])
        if task.higher_is_better:
            skill = level.mean - float(losses[task.benchmark].mean())
        else:
            skill = M.r2_oos(losses[name], losses[task.benchmark])
        rows.append(
            {
                "task": task.name,
                "target": task.target,
                "horizon": task.horizon,
                "model": name,
                "rung": rung[name],
                "loss": task.loss,
                "n_dates": n,
                "n_effective": n / task.horizon,
                "mean": level.mean,
                "ci_lower": level.lower,
                "ci_upper": level.upper,
                "skill": skill,
                "diff_vs_best_simple": gap.mean if name != best else 0.0,
                "diff_ci_lower": gap.lower if name != best else float("nan"),
                "diff_ci_upper": gap.upper if name != best else float("nan"),
                "secondary_name": secondary[name][0],
                "secondary": secondary[name][1],
                "best_simple": name == best,
            }
        )
    return pd.DataFrame(rows, columns=list(TABLE_COLUMNS))


def run_ladder(
    returns: pd.DataFrame,
    sectors: Mapping[str, str],
    splits: Splits,
    design: Forecasting,
    *,
    market: pd.Series | None = None,
    vix: pd.Series | None = None,
    dcc_avg_corr: pd.DataFrame | None = None,
    lam: float = F.DEFAULT_LAMBDA,
    targets: Sequence[TargetName] | None = None,
    variant: Variant | None = None,
    gbm_seeds: Sequence[int] = (),
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    final: bool = False,
) -> list[TaskResult]:
    """The whole baseline table. Refuses data that reaches the sealed test window.

    With a ``variant`` and no ``targets``, only the targets the variant names are run.
    ``final=True`` is the single test-window run of Phase 7 (``evaluation.final`` is its only
    caller): the seal is not applied and the folds are those of the test window.
    """
    if not final:
        assert_sealed(returns.index, splits)
    if variant is not None:
        targets = variant.targets if targets is None else targets
        if variant.window is not None and variant.window.end >= splits.val.start:
            raise SplitError("a training window to exclude must end before validation starts")
    for extra in (market, vix, dcc_avg_corr):
        if extra is not None and not final:
            assert_sealed(extra.index, splits)
    calendar = pd.DatetimeIndex(returns.index)
    tasks = build_tasks(
        returns,
        sectors,
        design,
        embargo=splits.walk_forward.embargo_days,
        market=market,
        vix=vix,
        dcc_avg_corr=dcc_avg_corr,
        lam=lam,
        targets=targets,
        variant=variant,
        gbm_seeds=gbm_seeds,
    )
    return [
        run_task(
            task, splits, calendar, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed, final=final
        )
        for task in tasks
    ]


def ladder_table(results: Sequence[TaskResult]) -> pd.DataFrame:
    """All task tables stacked (the baseline table)."""
    if not results:
        raise LadderError("no results")
    return pd.concat([r.table for r in results], ignore_index=True)


def table_lines(results: Sequence[TaskResult]) -> list[str]:
    """Plain-text report, one block per task."""
    out: list[str] = []
    for r in results:
        t = r.task
        better = "higher" if t.higher_is_better else "lower"
        n = int(r.table["n_dates"].iloc[0])
        out.append(
            f"{t.name}: loss {t.loss} ({better} is better), {n} validation dates "
            f"(about {n / t.horizon:.0f} independent), best simple baseline: {r.best_simple}"
        )
        for row in r.table.itertuples(index=False):
            gap = (
                "  (best simple baseline)"
                if row.best_simple
                else f"  vs best {row.diff_vs_best_simple:+.5f} "
                f"[{row.diff_ci_lower:+.5f}, {row.diff_ci_upper:+.5f}]"
            )
            out.append(
                f"  rung {row.rung} {row.model:<21} {row.mean:9.5f} "
                f"[{row.ci_lower:.5f}, {row.ci_upper:.5f}]  skill {row.skill:+.3f}{gap}"
            )
        out.append("")
    return out


def seed_spread(
    result: TaskResult,
    design: Forecasting,
    splits: Splits,
    calendar: pd.DatetimeIndex,
    seeds: Sequence[int],
) -> pd.Series:
    """Mean loss of the rung-3 model fitted with each single seed (index: seed).

    The hyperparameters are those the tuned run chose in each fold (``fold_params``), so only
    the fitting seed varies; the table's own row is the mean forecast over all seeds, which is
    usually better than the average single seed. The spread says how much of a difference
    between LightGBM and another model could be seed noise.
    """
    task = result.task
    params = result.fold_params.get(GBM_NAME)
    if not params:
        raise LadderError(f"{task.name}: the ladder was run without the rung-3 model")
    folds = walk_forward_from_config(calendar, splits, task.horizon)
    if len(params) != len(folds):
        raise LadderError(f"{task.name}: {len(params)} fitted folds for {len(folds)} folds")
    dates = result.losses.index
    out = {}
    for seed in seeds:
        queue = iter(params)

        def factory(queue: Iterator[dict[str, object]] = queue, seed: int = seed) -> Forecaster:
            best = next(queue)["best_params"]
            assert isinstance(best, dict)
            return gbm_factory(
                design,
                task.horizon,
                splits.walk_forward.embargo_days,
                task.gbm_columns,
                (seed,),
                bias_corrected=task.bias_corrected,
                params_override=best,
            )()

        pred = walk_forward_predict(factory, task.X, task.y, folds, calendar)
        loss = per_date_loss(task, pred)[0]
        out[seed] = float(loss.reindex(dates).mean())
    return pd.Series(out, name=f"{task.name}_{GBM_NAME}_by_seed")


VARIANT_COLUMNS: tuple[str, ...] = (
    "variant",
    "task",
    "model",
    "rung",
    "n_dates",
    "base_mean",
    "variant_mean",
    "diff",
    "diff_ci_lower",
    "diff_ci_upper",
)


def compare_with_base(
    base: Sequence[TaskResult],
    changed: Sequence[TaskResult],
    variant_id: str,
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    tol: float = 1e-12,
) -> pd.DataFrame:
    """Per task and model: loss under the design, under the variant, and their difference.

    ``diff`` is the mean over the common dates of (variant loss - design loss) with the same
    stationary-bootstrap interval as the table; rows whose losses are identical are omitted.
    For losses (everything but rank IC) a negative difference means the variant is better.
    """
    by_name = {r.task.name: r for r in base}
    rows = []
    for result in changed:
        name = result.task.name
        if name not in by_name:
            raise LadderError(f"variant {variant_id}: task {name} was not run under the design")
        reference = by_name[name].losses
        rung = {m.name: m.rung for m in result.task.models}
        for model in result.losses.columns:
            if model not in reference.columns:
                continue
            both = pd.concat(
                [reference[model], result.losses[model]], axis=1, join="inner"
            ).dropna()
            delta = both.iloc[:, 1] - both.iloc[:, 0]
            if len(delta) == 0 or float(delta.abs().max()) <= tol:
                continue
            block = M.block_length_for(result.task.horizon, len(delta))
            ci = M.mean_ci(
                delta, mean_block_length=block, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed
            )
            rows.append(
                {
                    "variant": variant_id,
                    "task": name,
                    "model": model,
                    "rung": rung[model],
                    "n_dates": ci.n,
                    "base_mean": float(both.iloc[:, 0].mean()),
                    "variant_mean": float(both.iloc[:, 1].mean()),
                    "diff": ci.mean,
                    "diff_ci_lower": ci.lower,
                    "diff_ci_upper": ci.upper,
                }
            )
    return pd.DataFrame(rows, columns=list(VARIANT_COLUMNS))


# --------------------------------------------------------------------------------------
# Loading the sealed development panel (shared by the Phase 5 and Phase 6 scripts)
# --------------------------------------------------------------------------------------

MARKET_TICKER = "SPY"
VIX_TICKER = "^VIX"


def load_sealed_panel(
    project: Project, *, final: bool = False
) -> tuple[pd.DataFrame, dict[str, str], pd.Series, pd.Series]:
    """Core returns, sectors, the SPY return and the VIX level, all cut at the end of validation.

    The cut is applied to each file as it is read, before any statistic
    is computed. Raises ``SystemExit`` with a plain message for a missing ticker or a gap.
    ``final=True`` cuts at the end of the test window instead: only ``evaluation.final`` may ask.
    """
    paths, splits = project.paths, project.experiment.splits
    end = pd.Timestamp(splits.test.end if final else splits.val.end)

    def cut(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[frame.index <= end]

    core = list(project.universe.tickers_in("core"))
    sectors = {t.ticker: str(t.sector) for t in project.universe.tickers if t.panel == "core"}
    returns_all = cut(pd.read_parquet(paths.data_processed / "returns_D.parquet"))
    missing = [t for t in [*core, MARKET_TICKER] if t not in returns_all.columns]
    if missing:
        raise SystemExit(f"missing from returns_D.parquet: {missing}")
    prices = cut(pd.read_parquet(paths.data_interim / "prices_clean.parquet"))
    vix = prices[VIX_TICKER]["close"].reindex(returns_all.index).rename(VIX_TICKER)
    returns = returns_all[core]
    if returns.isna().any().any() or vix.isna().any():
        raise SystemExit("the sealed core panel or the VIX level has missing values")
    if not final:
        assert_sealed(returns.index, splits)
    return returns, sectors, returns_all[MARKET_TICKER].rename(MARKET_TICKER), vix


def dcc_path(
    returns: pd.DataFrame,
    sectors: Mapping[str, str],
    max_horizon: int,
    *,
    drop: Sequence[str] = (),
) -> pd.DataFrame:
    """Causal DCC forecasts of the average cross-sector correlation, 1 to ``max_horizon`` days.

    ``drop`` removes assets from the DCC panel (robustness variant R2); the forecast is then
    the average over the cross-sector pairs of the remaining assets.
    """
    kept = returns[[c for c in returns.columns if c not in set(drop)]]
    mask = cross_sector_mask([str(c) for c in kept.columns], dict(sectors))
    path = rolling_dcc(
        kept, pair_mask=mask, horizons=tuple(range(1, max_horizon + 1)), keep_matrices=False
    )
    return path.avg_corr
