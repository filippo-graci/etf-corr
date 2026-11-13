"""Step 5.2: forecasting targets T1 to T4 (project_plan.md section 6.5).

A target dated ``t`` (the forecast origin) is a statistic of the rows ``t + 1 .. t + h``, where
``h`` is the horizon in trading days. Each one is defined as *the trailing statistic of
``etfcorr.features`` over a window of ``h`` rows, read ``h`` rows later*:

.. math:: y_t = s_{t + h}, \\qquad s_u = \\text{stat}(r_{u - h + 1}, \\dots, r_u)

so the persistence forecast (``s_t``) and the target are the same statistic by construction, and
the last ``h`` origins have no target (``NaN``): they are never filled.

* **T1** :func:`forward_average_correlation`: average realised correlation over the masked pairs
  (cross-sector by default, the H2 statistic). H3's primary target is T1 at ``h = 21``.
  :func:`forward_correlations` is the full matrix (T1_matrix).
* **T2** :func:`forward_variance`: per-asset mean squared daily log return over the window.
* **T3** :func:`forward_rank`: cross-sectional rank of the ``h``-day log return, in
  ``[-0.5, 0.5]``; its sign says whether the asset ends above or below the cross-sectional
  median (a relative statement, not the sign of the return itself).
* **T4** :func:`forward_regime_label`: a regime label ``h`` rows ahead (pass an ``expanding``
  label from ``etfcorr.regimes``; a ``full_sample`` label would leak).

**Targets are forward-looking on purpose and must never enter a feature table.** With daily
origins and ``h > 1`` consecutive targets overlap in ``h - 1`` rows: errors are serially
correlated by construction, and train rows closer than ``h`` (plus the embargo) to a test row
share label days with it, which is what ``etfcorr.splits`` purges. :func:`label_end_dates` gives
the date at which each label is fully known.

Limits: realised correlation over 21 days is a noisy estimate of any underlying correlation, so
part of every forecast error is measurement noise that no model can remove; the zero-mean
variance convention matches ``features.trailing_variance``; no third-party code is adapted.
"""

import numpy as np
import pandas as pd

from etfcorr.features import (
    BoolArray,
    FloatArray,
    cross_sectional_rank,
    masked_average,
    trailing_correlations,
    trailing_return,
    trailing_variance,
)


class TargetError(ValueError):
    """Invalid input to a target function."""


def _check_horizon(horizon: int) -> None:
    if horizon < 1:
        raise TargetError(f"horizon must be >= 1, got {horizon}")


def forward_correlations(returns: pd.DataFrame, horizon: int) -> FloatArray:
    """T1_matrix: correlation matrix of rows ``t + 1 .. t + horizon`` at each origin ``t``."""
    _check_horizon(horizon)
    trailing = trailing_correlations(returns, horizon)
    out = np.full_like(trailing, np.nan)
    if len(trailing) > horizon:
        out[:-horizon] = trailing[horizon:]
    return out


def forward_average_correlation(returns: pd.DataFrame, horizon: int, mask: BoolArray) -> pd.Series:
    """T1: average of the masked pairs of :func:`forward_correlations` (correlation units)."""
    values = masked_average(forward_correlations(returns, horizon), mask)
    return pd.Series(values, index=returns.index, name=f"fwd_corr_h{horizon}")


def forward_variance(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """T2: mean squared daily return over rows ``t + 1 .. t + horizon``, per asset."""
    _check_horizon(horizon)
    if horizon == 1:
        return (returns**2).shift(-1)
    return trailing_variance(returns, horizon).shift(-horizon)


def forward_return(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Sum of log returns over rows ``t + 1 .. t + horizon``, per asset."""
    _check_horizon(horizon)
    return trailing_return(returns, horizon).shift(-horizon)


def forward_rank(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """T3: cross-sectional rank of :func:`forward_return`, mapped to ``[-0.5, 0.5]``."""
    return cross_sectional_rank(forward_return(returns, horizon))


def forward_regime_label(labels: pd.Series, horizon: int) -> pd.Series:
    """T4: the regime label ``horizon`` rows ahead (``NaN`` for the last ``horizon`` origins)."""
    _check_horizon(horizon)
    return labels.shift(-horizon).rename(f"fwd_{labels.name}_h{horizon}")


def label_end_dates(index: pd.Index, horizon: int) -> pd.Series:
    """Date of the last row each label uses (``NaT`` when the label is not complete)."""
    _check_horizon(horizon)
    ends = pd.Series(index, index=index).shift(-horizon)
    return ends.rename("label_end")
