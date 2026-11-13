"""Step 5.5: rung 2, penalised linear models (ridge, lasso, elastic net, PCA regression).

One class, :class:`PenalisedLinear`, with four ``kind`` values. Everything that is estimated is
estimated inside ``fit``, i.e. inside one training window:

* features are standardised with the training mean and standard deviation
  (``sklearn.preprocessing.StandardScaler`` in a ``Pipeline``), and the target likewise
  (``TransformedTargetRegressor``), so the penalty grid has the same meaning for every target;
* the penalty (or the number of principal components) is chosen on **purged walk-forward folds
  inside the training window** (``splits.positional_walk_forward`` over the training dates, with
  the target's horizon and the embargo), never by leave-one-out or shuffled K-fold, because
  overlapping labels make neighbouring rows near-duplicates;
* the model is then refitted on the whole training window with the chosen value.

``columns`` restricts the model to a subset of the feature table (the table may carry columns
that only a baseline reads, such as the DCC forecast).

Grid semantics (``config/experiment.yaml`` ``forecasting.alpha_grid``, value ``a``): lasso and
elastic net use scikit-learn's ``alpha = a`` (their loss is already divided by the number of
rows); ridge uses ``alpha = a * n_rows`` so that the same ``a`` is a per-observation penalty in
both. Ties are resolved toward the stronger penalty (fewer components for ``pcr``).

``n_trials`` (grid size) and the chosen value are exposed in ``params`` so that the number of
configurations tried is on record (the project's multiple-testing rule). The models are
deterministic: no seed enters (coordinate descent in cyclic order, full SVD for the PCA).

Limits: with daily overlapping targets the effective training sample is roughly
``rows / horizon``, far below the row count, so the tuned penalty is noisy; the inner folds
select on mean squared error of the standardised target whatever the reporting loss of the
target is (QLIKE for volatility, rank IC for ranks). Method references (CHECKED
2026-10-11): Hoerl and Kennard (1970) ridge regression, Technometrics 12(1); Tibshirani (1996)
the lasso, JRSS B 58(1); Zou and Hastie (2005) the elastic net, JRSS B 67(2).
"""

from collections.abc import Sequence
from typing import Literal, get_args

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from etfcorr.models.base import ModelError, check_table, register, row_dates
from etfcorr.splits import SplitError, positional_walk_forward

Kind = Literal["ridge", "lasso", "elastic_net", "pcr"]
KINDS: tuple[Kind, ...] = get_args(Kind)
MAX_ITER = 20_000


@register("penalised_linear")
class PenalisedLinear:
    """Standardise, tune on purged inner folds, refit on the training window."""

    def __init__(
        self,
        kind: Kind,
        grid: Sequence[float],
        *,
        horizon: int,
        embargo: int = 0,
        inner_folds: int = 3,
        l1_ratio: float = 0.5,
        columns: Sequence[str] | None = None,
        name: str | None = None,
    ) -> None:
        if kind not in KINDS:
            raise ModelError(f"unknown kind {kind!r}; expected one of {KINDS}")
        if not grid or any(g <= 0 for g in grid):
            raise ModelError("grid must hold positive values")
        if kind == "pcr" and any(int(g) != g for g in grid):
            raise ModelError("the pcr grid is a list of component counts")
        self.kind: Kind = kind
        # strongest penalty first, so that a strict "<" keeps it on ties
        self.grid = tuple(sorted(grid, reverse=kind != "pcr"))
        self.horizon = horizon
        self.embargo = embargo
        self.inner_folds = inner_folds
        self.l1_ratio = l1_ratio
        self.columns = None if columns is None else tuple(columns)
        self.name = name or kind
        self.chosen_: float | None = None
        self.cv_scores_: dict[float, float] = {}
        self._model: TransformedTargetRegressor | None = None

    def _build(self, value: float, n_rows: int, n_features: int) -> TransformedTargetRegressor:
        steps: list[tuple[str, object]] = [("scale", StandardScaler())]
        if self.kind == "ridge":
            steps.append(("model", Ridge(alpha=value * n_rows)))
        elif self.kind == "lasso":
            steps.append(("model", Lasso(alpha=value, max_iter=MAX_ITER)))
        elif self.kind == "elastic_net":
            steps.append(
                ("model", ElasticNet(alpha=value, l1_ratio=self.l1_ratio, max_iter=MAX_ITER))
            )
        else:
            steps.append(("pca", PCA(n_components=min(int(value), n_features, n_rows))))
            steps.append(("model", LinearRegression()))
        return TransformedTargetRegressor(regressor=Pipeline(steps), transformer=StandardScaler())

    def _fit_one(self, value: float, x: np.ndarray, y: np.ndarray) -> TransformedTargetRegressor:
        return self._build(value, len(x), x.shape[1]).fit(x, y)

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        if self.columns is None:
            return X.to_numpy(dtype=float)
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ModelError(f"{self.name}: columns {missing} are not in the feature table")
        return X[list(self.columns)].to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PenalisedLinear":
        check_table(X, y)
        x = self._matrix(X)
        target = y.to_numpy(dtype=float)
        dates = row_dates(X.index)
        calendar = dates.unique().sort_values()
        position = calendar.get_indexer(dates)
        try:
            folds = positional_walk_forward(
                len(calendar), self.inner_folds, horizon=self.horizon, embargo=self.embargo
            )
        except SplitError as exc:
            raise ModelError(f"{self.name}: training window too short to tune ({exc})") from exc
        scale = float(target.std()) or 1.0
        self.cv_scores_ = {}
        best = float("inf")
        for value in self.grid:
            errors = []
            for fold in folds:
                tr, te = np.isin(position, fold.train), np.isin(position, fold.test)
                pred = self._fit_one(value, x[tr], target[tr]).predict(x[te])
                errors.append(float(np.mean(((pred - target[te]) / scale) ** 2)))
            score = float(np.mean(errors))
            self.cv_scores_[value] = score
            if score < best:
                best, self.chosen_ = score, value
        assert self.chosen_ is not None
        self._model = self._fit_one(self.chosen_, x, target)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise ModelError(f"{self.name}: predict called before fit")
        pred = self._model.predict(self._matrix(X))
        return pd.Series(np.asarray(pred, dtype=float), index=X.index)

    @property
    def params(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "chosen": self.chosen_,
            "n_trials": len(self.grid),
            "inner_folds": self.inner_folds,
            "cv_scores": dict(self.cv_scores_),
        }
