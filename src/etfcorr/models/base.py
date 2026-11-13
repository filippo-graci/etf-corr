"""Step 5.4: the forecaster interface, a registry, and the purged walk-forward driver.

Every model of the ladder (rungs 0 to 2 here, LightGBM and the graph models later) implements
the same small interface, :class:`Forecaster`: ``fit(X, y)`` on a training table and
``predict(X)`` on a test table, where ``X`` is a ``DataFrame`` indexed by date or by
``(date, unit)`` (``unit`` = an asset or a pair, see ``etfcorr.features.UNIT``) and ``y`` a
``Series`` on the same index. A model sees data **only** through these two calls, so the only
rows it can learn from are the ones the driver hands it.

:func:`walk_forward_predict` is that driver. For each fold of ``etfcorr.splits`` it builds a
fresh model, fits it on the fold's purged training origins and predicts the fold's test origins.
It is itself a pure function of the data up to each test date: the prediction dated ``t``
depends on features dated ``<= t`` and on targets whose label window ended before the fold
started, which ``tests/leakage/test_models_no_lookahead.py`` checks with the shared detectors
and with deliberately leaky twins.

Anything fitted (scalers, PCA, penalty choice, per-unit means) lives inside the model and is
therefore re-fitted inside every training window. Rows with a missing feature are not
predicted and not trained on; rows with a missing target are not trained on.

:class:`PerUnit` fits one copy of a model per unit (per-asset HAR for volatility).
No third-party code is adapted.
"""

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from etfcorr.features import UNIT
from etfcorr.splits import Fold


class ModelError(ValueError):
    """Invalid model request or input table."""


@runtime_checkable
class Forecaster(Protocol):
    """What the walk-forward driver needs from a model."""

    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster": ...

    def predict(self, X: pd.DataFrame) -> pd.Series: ...

    @property
    def params(self) -> dict[str, object]: ...


Factory = Callable[[], Forecaster]

REGISTRY: dict[str, Callable[..., Forecaster]] = {}


def register(name: str) -> Callable[[Callable[..., Forecaster]], Callable[..., Forecaster]]:
    """Class or function decorator: make a forecaster constructible by name."""

    def add(builder: Callable[..., Forecaster]) -> Callable[..., Forecaster]:
        if name in REGISTRY:
            raise ModelError(f"forecaster {name!r} is already registered")
        REGISTRY[name] = builder
        return builder

    return add


def make_forecaster(name: str, **kwargs: object) -> Forecaster:
    """Build a registered forecaster (the model modules register on import)."""
    if name not in REGISTRY:
        raise ModelError(f"unknown forecaster {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def row_dates(index: pd.Index) -> pd.DatetimeIndex:
    """The date of each row (first level of a ``(date, unit)`` index)."""
    level = index.get_level_values(0) if isinstance(index, pd.MultiIndex) else index
    return pd.DatetimeIndex(level)


def check_table(X: pd.DataFrame, y: pd.Series | None = None) -> None:
    if not isinstance(X, pd.DataFrame) or X.shape[1] == 0:
        raise ModelError("X must be a DataFrame with at least one column")
    if y is not None and not y.index.equals(X.index):
        raise ModelError("y must be on the same index as X")


def walk_forward_predict(
    factory: Factory,
    X: pd.DataFrame,
    y: pd.Series,
    folds: Sequence[Fold],
    calendar: pd.DatetimeIndex,
    *,
    on_fit: Callable[[int, Forecaster], None] | None = None,
) -> pd.Series:
    """Out-of-sample predictions on the test origins of every fold (``NaN`` elsewhere).

    ``calendar`` is the date index the folds' positions refer to. For each fold a new model from
    ``factory`` is fitted on the rows whose date is one of the fold's training origins (complete
    features and a known target) and predicts the rows of the fold's test origins that have
    complete features. A fold with no usable training row raises. ``on_fit(k, model)`` is
    called after each fit (to record the tuned parameters; it must not change the model).
    """
    check_table(X, y)
    dates = row_dates(X.index)
    complete = np.asarray(X.notna().all(axis=1))
    known = np.asarray(y.notna())
    out = pd.Series(np.nan, index=X.index, name=y.name, dtype=float)
    for k, fold in enumerate(folds):
        train = np.asarray(dates.isin(calendar[fold.train])) & complete & known
        test = np.asarray(dates.isin(calendar[fold.test])) & complete
        if not train.any():
            raise ModelError(f"fold {k}: no training row with complete features and a target")
        if not test.any():
            continue
        model = factory().fit(X.loc[train], y.loc[train])
        if on_fit is not None:
            on_fit(k, model)
        out.loc[test] = model.predict(X.loc[test]).to_numpy(dtype=float)
    return out


class PerUnit:
    """One independently fitted copy of a model per unit of a ``(date, unit)`` table."""

    def __init__(self, factory: Factory) -> None:
        self._factory = factory
        self._models: dict[object, Forecaster] = {}
        self.name = f"{factory().name}_per_unit"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PerUnit":
        check_table(X, y)
        self._models = {}
        for unit, block in X.groupby(level=UNIT, sort=False):
            self._models[unit] = self._factory().fit(block, y.loc[block.index])
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        out = pd.Series(np.nan, index=X.index, dtype=float)
        for unit, block in X.groupby(level=UNIT, sort=False):
            if unit in self._models:
                out.loc[block.index] = self._models[unit].predict(block).to_numpy(dtype=float)
        return out

    @property
    def params(self) -> dict[str, object]:
        return {str(unit): model.params for unit, model in self._models.items()}


class LogBiasCorrected:
    """A model of a log quantity whose forecasts are shifted by half its training residual variance.

    If the log of a positive quantity is forecast as ``f`` with residual variance ``s2``, then
    under log-normal errors the mean of the quantity is ``exp(f + s2 / 2)``, not ``exp(f)``
    (which is the median). This wrapper fits the wrapped model, measures ``s2`` on the same
    training rows (in-sample residuals, so slightly too small) and adds ``s2 / 2`` to every
    forecast, so that ``exp`` of the output is a mean-type forecast. Declared as variant V1 of
    the forecasting design (step 6.0); wrap it in :class:`PerUnit` for a per-asset ``s2``.
    """

    def __init__(self, factory: Factory) -> None:
        self._factory = factory
        self._model: Forecaster | None = None
        self.shift_ = 0.0
        self.name = f"{factory().name}_bias_corrected"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogBiasCorrected":
        check_table(X, y)
        self._model = self._factory().fit(X, y)
        residual = y.to_numpy(dtype=float) - self._model.predict(X).to_numpy(dtype=float)
        self.shift_ = 0.5 * float(np.var(residual))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise ModelError(f"{self.name}: predict called before fit")
        return self._model.predict(X) + self.shift_

    @property
    def params(self) -> dict[str, object]:
        inner = {} if self._model is None else dict(self._model.params)
        return {**inner, "log_bias_shift": self.shift_}
