"""Step 5.5: rung 1, the HAR regression (and its pooled "matrix-HAR" use).

HAR (Corsi (2009), *A Simple Approximate Long-Memory Model of Realized Volatility*, Journal of
Financial Econometrics 7(2), 174-196; CHECKED 2026-10-11) regresses the next-period realised
quantity on its own trailing averages over a short, a medium and a long window:

.. math:: y_{t \\to t+h} = c + \\sum_w \\beta_w \\, s^{(w)}_t + \\varepsilon_t

by ordinary least squares, with ``s^{(w)}_t`` the trailing statistic over ``w`` rows at the
origin. The same class serves three targets, only the columns differ:

* volatility (the original use): log variance on log trailing variances, one fit per asset
  through ``models.base.PerUnit``;
* average correlation in Fisher-z ("HAR-style" in the pre-registered H3 text): the trailing
  average correlation at several windows;
* the full matrix: one **pooled** regression over every (date, pair) row in Fisher-z, i.e. the
  same four coefficients for all pairs. This is the simplest matrix-HAR; it is not the
  matrix-log HAR of the literature, and its forecasts are not guaranteed positive semidefinite
  (``evaluation.metrics.project_to_correlation`` projects and counts).

Limits: OLS with overlapping targets has serially correlated errors, so coefficient standard
errors would need a HAC estimator; none is reported because only the forecasts are used.
Trailing windows of very different length are collinear, which inflates coefficient variance
(the penalised models of ``models.linear`` address that). ``numpy.linalg.lstsq`` gives the
minimum-norm solution if the design is rank deficient.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from etfcorr.models.base import ModelError, check_table, register


@register("har")
@dataclass
class HAR:
    """OLS of the target on an intercept and the given trailing-statistic columns."""

    columns: Sequence[str]
    name: str = "har"
    _coef: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.columns = tuple(self.columns)
        if not self.columns:
            raise ModelError("HAR needs at least one component column")

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ModelError(f"{self.name}: columns {missing} are not in the feature table")
        block = X[list(self.columns)].to_numpy(dtype=float)
        return np.column_stack([np.ones(len(block)), block])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HAR":
        check_table(X, y)
        design = self._design(X)
        if len(design) <= design.shape[1]:
            raise ModelError(f"{self.name}: {len(design)} rows cannot fit {design.shape[1]} terms")
        self._coef = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)[0]
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._coef is None:
            raise ModelError(f"{self.name}: predict called before fit")
        return pd.Series(self._design(X) @ self._coef, index=X.index, dtype=float)

    @property
    def params(self) -> dict[str, object]:
        if self._coef is None:
            return {}
        names = ("intercept", *self.columns)
        return {str(n): float(c) for n, c in zip(names, self._coef, strict=True)}
