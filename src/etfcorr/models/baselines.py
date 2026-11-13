"""Step 5.4: rung 0 of the ladder, the baselines that need no estimation or only a mean.

* :class:`ColumnForecaster`: the forecast is a column of the feature table, read at the origin.
  Persistence (the trailing statistic over the last ``h`` rows), EWMA, the constant-correlation
  level and the DCC forecast are all of this kind; ``fit`` does nothing, so there is nothing to
  leak. ``sign = -1`` gives a reversal signal.
* :class:`HistoricalMean`: the mean of the training targets (per unit when the table has a
  ``unit`` level, with the pooled mean for a unit never seen in training). It is the benchmark
  of the out-of-sample R squared (Campbell and Thompson (2008), *Predicting Excess Stock Returns
  Out of Sample: Can Anything Beat the Historical Average?*, Review of Financial Studies 21(4),
  1509-1531; CHECKED 2026-10-11). With quarterly refits the mean is up to one quarter stale,
  like every other fitted model of the ladder.
* :class:`Constant`: a fixed number (zero is the no-skill reference for the rank target).

For the scalar average-correlation target, "constant correlation" (Elton and Gruber (1973),
*Estimating the Dependence Structure of Share Prices*, Journal of Finance 28(5); CHECKED
2026-10-11) reduces to the long-window average correlation, i.e. a :class:`ColumnForecaster` on
the longest-window column; for the matrix target it is the ``z_const`` column of
``features.build_pair_features``.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from etfcorr.features import UNIT
from etfcorr.models.base import ModelError, check_table, register


@register("column")
@dataclass
class ColumnForecaster:
    """Forecast = ``sign`` times one feature column, as observed at the origin."""

    column: str
    name: str = ""
    sign: float = 1.0

    def __post_init__(self) -> None:
        self.name = self.name or self.column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ColumnForecaster":
        self._require(X)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        self._require(X)
        return (self.sign * X[self.column]).astype(float)

    def _require(self, X: pd.DataFrame) -> None:
        if self.column not in X.columns:
            raise ModelError(f"{self.name}: column {self.column!r} is not in the feature table")

    @property
    def params(self) -> dict[str, object]:
        return {"column": self.column, "sign": self.sign}


@register("historical_mean")
@dataclass
class HistoricalMean:
    """Forecast = mean of the training targets (per unit when the table is long)."""

    name: str = "historical_mean"
    _overall: float = field(default=float("nan"), init=False, repr=False)
    _by_unit: pd.Series | None = field(default=None, init=False, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistoricalMean":
        check_table(X, y)
        self._overall = float(y.mean())
        self._by_unit = y.groupby(level=UNIT).mean() if UNIT in y.index.names else None
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._by_unit is None:
            return pd.Series(self._overall, index=X.index, dtype=float)
        units = X.index.get_level_values(UNIT)
        values = self._by_unit.reindex(units).to_numpy(dtype=float)
        return pd.Series(np.where(np.isnan(values), self._overall, values), index=X.index)

    @property
    def params(self) -> dict[str, object]:
        return {"mean": self._overall}


@register("constant")
@dataclass
class Constant:
    """Forecast = a fixed value (no-skill reference)."""

    value: float = 0.0
    name: str = "constant"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Constant":
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=X.index, dtype=float)

    @property
    def params(self) -> dict[str, object]:
        return {"value": self.value}


@register("best_column")
@dataclass
class BestColumn:
    """Forecast = the one of ``columns`` with the lowest squared error on the training rows.

    Exploratory follow-up E4 (docs/FOLLOW_UPS.md item 4): an EWMA whose decay is chosen on the
    training window. The choice uses training targets only, so it is refitted like any model.
    """

    columns: tuple[str, ...]
    name: str = "best_column"
    column_: str = field(default="", init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BestColumn":
        missing = [c for c in self.columns if c not in X.columns]
        if missing or not self.columns:
            raise ModelError(f"{self.name}: columns {missing or '(none)'} are not in the table")
        errors = {c: float(((X[c] - y) ** 2).mean()) for c in self.columns}
        self.column_ = min(errors, key=lambda c: errors[c])
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.column_:
            raise ModelError(f"{self.name}: predict called before fit")
        return X[self.column_].astype(float)

    @property
    def params(self) -> dict[str, object]:
        return {"column": self.column_}
