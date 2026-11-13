"""Step 4.1: regime labels that need no fitting -- quantile (VIX tercile) and drawdown states.

Question served: Phase 4 (H2, ``config/experiment.yaml``) conditions correlations on market
regimes: "raw versus volatility-adjusted average cross-sector correlation by VIX tercile".
This module only *labels dates*; the conditional-correlation engine that consumes the labels is
the next step. Model-based regimes (the forward-filtered HMM of ``project_plan.md`` section 6.5)
are deliberately not here: they need fitting, and fitting is where look-ahead hides, so they get
their own step and their own leakage tests.

**Labels are integer codes stored as ``float``** (``0.0, 1.0, 2.0``; ``NaN`` where undefined, e.g.
warm-up or a missing input). Floats rather than a pandas Categorical so the shared detectors in
``etfcorr.leakage`` (which compare outputs numerically) apply to every function here unchanged.
Codes are ordered: a higher code is a higher value of the conditioning variable.

**Two modes for quantile cutpoints -- the leakage-relevant choice.**

* ``"expanding"`` (causal): the cutpoints at date ``t`` are the sample quantiles of the
  conditioning variable over observations dated ``<= t`` only, and no label is produced before
  ``min_obs`` valid observations exist. The label at ``t`` is a pure function of data up to ``t``
  (``f(data_up_to_t) -> value_at_t``) and is the only mode allowed for features, targets or any
  forecasting use (Phases 5-8).
* ``"full_sample"`` (**non-causal, descriptive only**): cutpoints are the quantiles of every valid
  observation passed in. The label at ``t`` therefore depends on later data. This is appropriate
  only for a descriptive conditional statistic about the sample itself (the confirmatory H2 test,
  decided with the user on 2026-10-02: full sample 2014 to 2026-09, no model fitted); anything
  reported from it must be labelled in-sample/descriptive. The tests verify that the shared
  detector *does* flag this mode, so the distinction cannot be lost silently.

**Alignment is not decided here.** A label at ``t`` uses the conditioning variable's value at
``t`` (the VIX close of day ``t``). Whether H2 should group day-``t`` returns by the day-``t``
label (contemporaneous) or by the label known at ``t-1`` (predetermined, see :func:`lag_regimes`)
is a design decision for the H2 engine: contemporaneous grouping selects on a variable that moves
with the same-day returns, which is exactly the selection problem that makes conditional
correlations misleading (Boyer, Gibson & Loretan (1999); Forbes & Rigobon (2002), both checked
for existence and venue on 2026-10-02, see ``REFERENCES_SEED.md``). Neither paper's *results* are
used by this module.

Ties and boundaries: a value equal to a cutpoint falls in the lower regime (label = number of
cutpoints strictly below the value), so a constant series is all-lowest and tercile sizes can be
unequal when the variable has many ties. Cutpoints use ``numpy.quantile`` with its default linear
interpolation.

**Drawdown states** use thresholds on the drawdown from the running maximum of a price series.
The defaults ``(-0.05, -0.20)`` (calm / drawdown / bear) are round-number market conventions
(**[Inference]**, no source checked) fixed before any result is seen; they are not tuned.
Drawdown from an adjusted-close series inherits the standing vendor caveat that adjusted prices
are revised retroactively when dividends are paid, so past drawdown values can differ slightly
between data vintages.

Not adapted from third-party code (no licence flag). Pure functions of their inputs: no I/O, no
randomness, no dependency on ``etfcorr.config`` or ``etfcorr.data``.
"""

from collections.abc import Sequence
from itertools import pairwise
from typing import Literal

import numpy as np
import pandas as pd

RegimeMode = Literal["expanding", "full_sample"]

#: Integer codes of the three VIX-tercile regimes (low, middle, high conditioning-variable tercile).
LOW: float = 0.0
MID: float = 1.0
HIGH: float = 2.0
TERCILE_NAMES: tuple[str, str, str] = ("low", "mid", "high")
TERCILE_PROBS: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)

DEFAULT_MIN_OBS = 252  # about one trading year of history before an expanding label exists
DEFAULT_DRAWDOWN_THRESHOLDS: tuple[float, float] = (-0.05, -0.20)
DRAWDOWN_NAMES: tuple[str, str, str] = ("calm", "drawdown", "bear")


class RegimeError(ValueError):
    """Invalid input to, or an invalid configuration of, a regime-labelling call."""


# --------------------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------------------


def _check_series(x: pd.Series, name: str) -> None:
    """A float-like series on a strictly increasing, duplicate-free index, with no +-inf."""
    if not isinstance(x, pd.Series):
        raise RegimeError(f"{name} must be a pandas Series, got {type(x)}")
    if not x.index.is_monotonic_increasing or not x.index.is_unique:
        raise RegimeError(f"{name} index must be strictly increasing and unique")
    values = x.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise RegimeError(f"{name} contains +-inf")


def _check_probs(probs: Sequence[float]) -> tuple[float, ...]:
    p = tuple(float(v) for v in probs)
    if not p:
        raise RegimeError("probs must contain at least one cutpoint probability")
    if any(not 0.0 < v < 1.0 for v in p):
        raise RegimeError(f"every probability must lie strictly in (0, 1), got {p}")
    if any(b <= a for a, b in pairwise(p)):
        raise RegimeError(f"probs must be strictly increasing, got {p}")
    return p


def _check_min_obs(min_obs: int) -> None:
    if min_obs < 2:
        raise RegimeError(f"min_obs must be >= 2, got {min_obs}")


# --------------------------------------------------------------------------------------
# Quantile regimes (VIX terciles are the H2 special case)
# --------------------------------------------------------------------------------------


def quantile_regimes(
    x: pd.Series,
    probs: Sequence[float],
    *,
    mode: RegimeMode = "expanding",
    min_obs: int = DEFAULT_MIN_OBS,
) -> pd.Series:
    """Label each date by which quantile bucket of ``x`` its value falls in.

    ``probs`` are the cutpoint probabilities, strictly increasing in (0, 1); ``k`` cutpoints give
    ``k + 1`` ordered regimes coded ``0.0 .. k`` (``(1/3, 2/3)`` gives terciles; a single
    ``(0.9,)`` gives a worst-decile indicator). The code at date ``t`` is the number of cutpoints
    strictly below ``x_t``; ``NaN`` where ``x_t`` is ``NaN`` or, in ``"expanding"`` mode, before
    ``min_obs`` valid observations (counting ``x_t``) have been seen.

    In ``"expanding"`` mode the cutpoints at ``t`` use only the valid values dated ``<= t``, so the
    output is causal (module docstring). In ``"full_sample"`` mode the cutpoints use all valid
    values and the output is **not** causal; it requires at least ``min_obs`` valid values.
    """
    _check_series(x, "x")
    p = _check_probs(probs)
    _check_min_obs(min_obs)
    if mode not in ("expanding", "full_sample"):
        raise RegimeError(f"unknown mode {mode!r}; expected 'expanding' or 'full_sample'")

    values = x.to_numpy(dtype=float)
    valid = ~np.isnan(values)
    n_valid = int(valid.sum())
    labels = np.full(len(values), np.nan)

    if mode == "full_sample":
        if n_valid < min_obs:
            raise RegimeError(f"need at least {min_obs} valid observations, got {n_valid}")
        cuts = np.quantile(values[valid], p)
        labels[valid] = (values[valid, None] > cuts[None, :]).sum(axis=1)
    else:
        seen = np.empty(len(values))
        k = 0
        for i in np.flatnonzero(valid):
            seen[k] = values[i]
            k += 1
            if k < min_obs:
                continue
            cuts = np.quantile(seen[:k], p)
            labels[i] = float((values[i] > cuts).sum())

    return pd.Series(labels, index=x.index, name=x.name, dtype=float)


def vix_terciles(
    vix: pd.Series,
    *,
    mode: RegimeMode = "expanding",
    min_obs: int = DEFAULT_MIN_OBS,
) -> pd.Series:
    """VIX tercile regime: ``0.0`` low, ``1.0`` middle, ``2.0`` high (:data:`TERCILE_NAMES`).

    ``vix`` is the VIX *level* (a strictly positive index level, not the log change that
    ``returns.py`` computes for the ``^VIX`` column). See :func:`quantile_regimes` for the modes;
    use ``"expanding"`` for anything that is not the descriptive H2 conditional statistic.
    """
    _check_series(vix, "vix")
    if (vix.dropna() <= 0).any():
        raise RegimeError("vix levels must be strictly positive (pass the level, not a return)")
    return quantile_regimes(vix, TERCILE_PROBS, mode=mode, min_obs=min_obs)


# --------------------------------------------------------------------------------------
# Drawdown states
# --------------------------------------------------------------------------------------


def drawdown(price: pd.Series) -> pd.Series:
    """Drawdown from the running maximum: ``price_t / max(price_{<=t}) - 1`` (``<= 0``).

    Causal by construction (the running maximum uses data up to ``t`` only). ``NaN`` prices give
    ``NaN`` and do not reset the running maximum. Prices must be strictly positive.
    """
    _check_series(price, "price")
    if (price.dropna() <= 0).any():
        raise RegimeError("prices must be strictly positive")
    return price / price.cummax() - 1.0


def drawdown_states(
    price: pd.Series,
    thresholds: Sequence[float] = DEFAULT_DRAWDOWN_THRESHOLDS,
) -> pd.Series:
    """Drawdown regime code: the number of ``thresholds`` the drawdown has reached or passed.

    With the default ``(-0.05, -0.20)``: ``0.0`` calm (drawdown above -5%), ``1.0`` drawdown
    (between -5% and -20%, -5% inclusive), ``2.0`` bear (-20% or worse, inclusive)
    (:data:`DRAWDOWN_NAMES`). ``thresholds`` must be strictly decreasing and negative. ``NaN``
    where the price is ``NaN``. Causal (see :func:`drawdown`).
    """
    th = tuple(float(v) for v in thresholds)
    if not th:
        raise RegimeError("thresholds must contain at least one value")
    if any(v >= 0.0 for v in th):
        raise RegimeError(f"drawdown thresholds must be negative, got {th}")
    if any(b >= a for a, b in pairwise(th)):
        raise RegimeError(f"drawdown thresholds must be strictly decreasing, got {th}")
    dd = drawdown(price)
    values = dd.to_numpy(dtype=float)
    codes = (values[:, None] <= np.asarray(th)[None, :]).sum(axis=1).astype(float)
    codes[np.isnan(values)] = np.nan
    return pd.Series(codes, index=price.index, name=price.name, dtype=float)


# --------------------------------------------------------------------------------------
# Alignment and diagnostics
# --------------------------------------------------------------------------------------


def lag_regimes(labels: pd.Series, periods: int = 1) -> pd.Series:
    """Label known ``periods`` rows earlier: the *predetermined* regime for each date.

    ``lag_regimes(labels, 1)`` at date ``t`` is ``labels`` at the previous row, i.e. the regime
    observable at the previous close. Rows, not calendar days: callers pass a series already on
    the trading calendar. Shifting never introduces look-ahead (``periods >= 1``); ``periods=0``
    is the contemporaneous label and is allowed so callers can treat both alignments uniformly.
    """
    if periods < 0:
        raise RegimeError(
            f"periods must be >= 0 (a negative shift would look ahead), got {periods}"
        )
    _check_series(labels, "labels")
    return labels.shift(periods)


def regime_summary(labels: pd.Series, n_regimes: int | None = None) -> pd.DataFrame:
    """Occupancy and persistence of each regime code (``NaN`` labels are excluded).

    One row per code ``0 .. n_regimes-1`` (default: highest observed code + 1) with ``n_obs``,
    ``share`` (of non-``NaN`` labels), ``n_runs`` (maximal blocks of consecutive equal labels; a
    ``NaN`` ends a run), ``mean_run_length`` and ``max_run_length``. Regimes cluster in time, so
    ``n_runs`` -- not ``n_obs`` -- is the honest count of independent-ish regime episodes behind
    any regime-conditional statistic (the reason H2 uses a block bootstrap).
    """
    _check_series(labels, "labels")
    values = labels.to_numpy(dtype=float)
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        raise RegimeError("labels has no non-NaN entries")
    if np.any(valid != np.floor(valid)) or np.any(valid < 0):
        raise RegimeError("labels must be non-negative integer codes")
    top = int(valid.max()) + 1
    n = top if n_regimes is None else n_regimes
    if n < top:
        raise RegimeError(f"n_regimes={n} is smaller than the highest code present ({top - 1})")

    runs: dict[int, list[int]] = {code: [] for code in range(n)}
    current: float | None = None
    length = 0
    for v in values:
        if np.isnan(v):
            if current is not None:
                runs[int(current)].append(length)
            current, length = None, 0
        elif current is not None and v == current:
            length += 1
        else:
            if current is not None:
                runs[int(current)].append(length)
            current, length = v, 1
    if current is not None:
        runs[int(current)].append(length)

    rows = []
    for code in range(n):
        lengths = runs[code]
        n_obs = int(sum(lengths))
        rows.append(
            {
                "n_obs": n_obs,
                "share": n_obs / valid.size,
                "n_runs": len(lengths),
                "mean_run_length": float(np.mean(lengths)) if lengths else float("nan"),
                "max_run_length": int(max(lengths)) if lengths else 0,
            }
        )
    out = pd.DataFrame(rows, index=pd.Index(range(n), name="regime"))
    return out
