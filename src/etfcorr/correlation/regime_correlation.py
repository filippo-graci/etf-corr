"""Step 4.2: the H2 engine -- regime-conditional average cross-sector correlation, raw and
Forbes-Rigobon adjusted, with stationary-bootstrap confidence intervals.

Question served: H2 (``config/experiment.yaml``, pre-registered): "Average cross-sector
correlation is higher in high-VIX regimes, but a large part of the increase is explained by
volatility (Forbes-Rigobon bias) rather than by a true change in dependence." Test text:
"Raw versus volatility-adjusted average cross-sector correlation by VIX tercile, with
stationary-bootstrap confidence intervals." This module computes exactly that table. It takes
regime labels from :mod:`etfcorr.regimes` (step 4.1) and decides nothing about the regimes
themselves.

**Method.** For every regime ``g`` the sample Pearson correlation matrix of the (log) returns on
the days assigned to ``g`` is computed (``numpy.corrcoef``; cross-checked against
``estimators.estimate`` in the tests). The reported statistic is the unweighted mean of the
off-diagonal entries over the pairs in scope (default: pairs whose two tickers are in different
sectors). The Forbes-Rigobon (2002) adjustment removes the part of the increase that a pure rise
in variance produces under their model:

.. math:: \\rho^* = \\rho \\big/ \\sqrt{1 + \\delta\\,(1 - \\rho^2)}, \\qquad
          \\delta = \\sigma^2_{g} / \\sigma^2_{\\mathrm{ref}} - 1,

where ``sigma2`` is the variance of the conditioning series in the regime ``g`` and in the
reference regime (default: the lowest VIX tercile, so the reference regime is left unchanged). The
square root is part of the formula; it was verified by derivation and by simulation before this
module was written (docs/DECISION_LOG.md, 2026-10-02) because a summary of the NBER working paper
rendered it without the root, and a known-answer test pins it.

**Definition of delta for an average over many pairs (decided with the user, 2026-10-02, before
any result):** ``delta_mode="pair"`` (primary). Each asset ``k`` gets its own
``delta_k = Var(r_k | g) / Var(r_k | ref) - 1``; for a pair ``(i, j)`` the adjusted correlation is
the mean of the two directional adjustments, ``0.5 * (rho*(rho, delta_i) + rho*(rho, delta_j))``
(the formula is not symmetric in the two assets, because it treats one of them as the source of
the volatility shock; averaging makes the pair value symmetric, **[Inference]**, a design choice
and not from the paper), and the statistic is the mean of the adjusted pair values.
``delta_mode="series"`` (robustness) uses one delta from a single conditioning series (for
example the VIX log change or an equal-weight market return) for every pair.

**Assumptions and failure modes -- state these wherever the adjusted numbers are reported.**
Forbes and Rigobon's adjustment assumes a bivariate model in which one variable is the
heteroskedastic source, with no omitted variables and no endogeneity. Returns of ETFs that load on
a common market factor violate that. A synthetic check shipped with this step
(``scripts/h2_calibration_4_2.py``, results in ``docs/checks/4.2.md``) measures what the adjustment
does when the *only* change is a rise in the common factor's variance: with the per-asset delta it
**under-corrects** (the adjusted correlation stays above the truth), i.e. the share of the increase
attributed to volatility is biased **downward**. Read a small explained share with that bias in
mind; read a large one as conservative. The same check reports the series-delta and the
volatility-standardised variants side by side. Further limits: regime-conditional correlations
select on the conditioning variable (Boyer, Gibson and Loretan (1999) analyse the resulting bias;
this module's answer is the predetermined alignment below, not a correction); sample correlations
inside a regime are noisy for fat-tailed returns; the adjusted matrices are an entrywise transform
and are **not** guaranteed positive semi-definite (only their off-diagonal mean is used); regime
cutpoints are fixed on the original sample and their sampling uncertainty is not propagated into
the bootstrap (**[Inference]**: small against the other uncertainty at about 3,000 days, not
checked).

**Alignment (decided with the user, 2026-10-02, before any result).** ``alignment="predetermined"``
(primary) groups the return of day ``t`` by the regime label known at the close of ``t-1``
(:func:`etfcorr.regimes.lag_regimes`), because grouping by the label of day ``t`` selects on a
variable that moves with the same-day returns. ``"contemporaneous"`` (robustness) groups by the
day-``t`` label. :func:`align_regime_labels` is the single place this happens, and requires the
label calendar and the return dates to be consecutive rows of the same trading calendar, so that
"one row earlier" is genuinely the previous return date.

**Inference.** :func:`bootstrap_regime_correlations` resamples *dates* with the stationary
bootstrap of Politis and Romano (1994) (reused from ``horizon_analysis``), and resamples the
return row, the aligned regime code (and the optional delta series) with the same indices, so the
label stays attached to its return and regime persistence is only broken at block boundaries.
Percentile intervals; the block length defaults to the project heuristic
``default_mean_block_length`` (not the Politis-White automatic rule), and the regime episodes
(about 100 per tercile on the real panel, step 4.1) are far fewer than the days, so the block
length should be varied as a sensitivity check. No multiplicity correction is applied here (the
cross-hypothesis Holm correction belongs to the scoreboard). The "explained share" is a ratio and
is unstable when the raw increase is close to zero: its interval is reported as ``NaN`` unless the
interval of the raw increase excludes zero.

**Everything computed here is a descriptive, in-sample conditional statistic** about the sample
it is given. It is not a forecast and it uses no fitted model, which is why the pre-registered H2
may use the full sample (decision of 2026-10-02); it must be labelled in-sample wherever it is
reported. Not adapted from third-party code (no licence flag).

Citations (existence and venue checked by web search on 2026-10-02; the paper's *formula* was
re-derived here rather than taken from a summary): Forbes, K. and Rigobon, R. (2002), *No
contagion, only interdependence: measuring stock market comovements*, Journal of Finance 57(5),
2223-2261. Boyer, B., Gibson, M. and Loretan, M. (1999), *Pitfalls in tests for changes in
correlations*, Federal Reserve Board IFDP 597 (1999; CHECKED 2026-11-02 in the
reference lists of Longin and Solnik and of Corsetti et al.).
Politis, D. and Romano, J. (1994), *The Stationary Bootstrap*, JASA 89(428), 1303-1313 (CHECKED,
step 3.2).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from etfcorr.correlation.horizon_analysis import (
    DEFAULT_ALPHA,
    DEFAULT_N_BOOTSTRAP,
    default_mean_block_length,
    stationary_bootstrap_indices,
)
from etfcorr.regimes import TERCILE_NAMES, lag_regimes, regime_summary

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

Alignment = Literal["predetermined", "contemporaneous"]
PairScope = Literal["cross", "within", "all"]
DeltaMode = Literal["pair", "series"]

DEFAULT_REFERENCE = 0  # the lowest regime code (the low VIX tercile)
DEFAULT_MIN_ROWS = 30  # fewest days a regime may have in a sample or in a bootstrap replicate
DEFAULT_VOL_LAMBDA = 0.94  # mirrors config/experiment.yaml's ewma_lambda (RiskMetrics convention)
DEFAULT_VOL_INIT = 30  # rows used to start the EWMA variance; those rows get no standardised value
_SHARE_EPS = 1e-12
_MIN_VAR_RATIO = 1e-12  # a regime variance below this fraction of the reference one counts as zero


class H2Error(ValueError):
    """Invalid input to, or an invalid configuration of, an H2-engine call."""


# --------------------------------------------------------------------------------------
# The Forbes-Rigobon adjustment
# --------------------------------------------------------------------------------------


def forbes_rigobon_adjust(rho: ArrayLike, delta: ArrayLike) -> FloatArray:
    """Forbes-Rigobon (2002) adjusted correlation ``rho / sqrt(1 + delta * (1 - rho**2))``.

    ``delta`` is the relative rise in the variance of the conditioning series
    (``sigma2_regime / sigma2_reference - 1``), so it must exceed -1; ``delta = 0`` returns
    ``rho``. ``rho`` and ``delta`` broadcast against each other. For ``delta > 0`` the result is
    smaller in absolute value than ``rho`` and keeps its sign; for ``-1 < delta < 0`` it is larger
    in absolute value. Raises :class:`H2Error` for non-finite input, ``|rho| > 1`` beyond
    rounding error, or ``delta <= -1``.
    """
    r = np.asarray(rho, dtype=float)
    d = np.asarray(delta, dtype=float)
    if not (np.isfinite(r).all() and np.isfinite(d).all()):
        raise H2Error("rho and delta must be finite")
    if np.any(np.abs(r) > 1.0 + 1e-9):
        raise H2Error("rho must lie in [-1, 1]")
    if np.any(d <= -1.0):
        raise H2Error("delta must be > -1 (it is a variance ratio minus one)")
    r = np.clip(r, -1.0, 1.0)
    return np.asarray(r / np.sqrt(1.0 + d * (1.0 - r**2)), dtype=float)


# --------------------------------------------------------------------------------------
# Input checks and helpers
# --------------------------------------------------------------------------------------


def _check_frame(frame: pd.DataFrame, name: str, min_columns: int = 1) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise H2Error(f"{name} must be a pandas DataFrame, got {type(frame)}")
    if frame.shape[1] < min_columns:
        raise H2Error(f"{name} needs at least {min_columns} columns, got {frame.shape[1]}")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise H2Error(f"{name} index must be strictly increasing and unique")
    if not frame.columns.is_unique:
        raise H2Error(f"{name} column names must be unique")


def _sector_vector(columns: pd.Index, sectors: Mapping[str, str] | pd.Series) -> list[str]:
    lookup = dict(sectors.items()) if isinstance(sectors, pd.Series) else dict(sectors)
    missing = [str(c) for c in columns if c not in lookup]
    if missing:
        raise H2Error(f"no sector given for tickers {missing}")
    return [str(lookup[c]) for c in columns]


def pair_mask(sector_of: Sequence[str], scope: PairScope = "cross") -> BoolArray:
    """Strictly-upper-triangular boolean mask of the ticker pairs in scope.

    ``"cross"``: pairs in different sectors (the H2 statistic); ``"within"``: pairs in the same
    sector (a contrast); ``"all"``: every pair. Raises :class:`H2Error` when the scope selects no
    pair (for example ``"cross"`` with a single sector).
    """
    if scope not in ("cross", "within", "all"):
        raise H2Error(f"unknown pair scope {scope!r}; expected 'cross', 'within' or 'all'")
    s = np.asarray(list(sector_of), dtype=object)
    n = len(s)
    same = s[:, None] == s[None, :]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    if scope == "cross":
        mask = upper & ~same
    elif scope == "within":
        mask = upper & same
    else:
        mask = upper
    if not mask.any():
        raise H2Error(f"pair scope {scope!r} selects no pair for sectors {sorted(set(s))}")
    return np.asarray(mask, dtype=bool)


def align_regime_labels(
    labels: pd.Series,
    index: pd.Index,
    alignment: Alignment = "predetermined",
) -> pd.Series:
    """The regime code to use for each date in ``index``, under the chosen alignment.

    ``"predetermined"``: the label of the *previous* row of ``labels`` (known at the previous
    close); ``"contemporaneous"``: the label of the same date. The previous row is taken in
    ``labels`` itself before restricting to ``index``, so the first return date still receives
    the label of the day before it. Raises :class:`H2Error` when a date of ``index`` is missing
    from ``labels`` or when the two calendars are not the same consecutive trading days (so that
    "one row earlier" would not be the previous return date).
    """
    if alignment not in ("predetermined", "contemporaneous"):
        raise H2Error(
            f"unknown alignment {alignment!r}; expected 'predetermined' or 'contemporaneous'"
        )
    shifted = lag_regimes(labels, 1 if alignment == "predetermined" else 0)
    pos = labels.index.get_indexer(index)
    if np.any(pos < 0):
        raise H2Error("some return dates are missing from the regime label index")
    if len(pos) > 1 and not np.all(np.diff(pos) == 1):
        raise H2Error(
            "the return dates are not consecutive rows of the label calendar; "
            "'one row earlier' would not be the previous return date"
        )
    return shifted.reindex(index)


# --------------------------------------------------------------------------------------
# Volatility standardisation (the robustness variant; feed its output to the same engine)
# --------------------------------------------------------------------------------------


def ewma_volatility(
    returns: pd.DataFrame,
    lam: float = DEFAULT_VOL_LAMBDA,
    init: int = DEFAULT_VOL_INIT,
) -> pd.DataFrame:
    """EWMA volatility ``sigma_t`` per column, using returns up to and including row ``t``.

    RiskMetrics-style recursion on raw (non-demeaned) returns: ``sigma2_t = lam * sigma2_{t-1} +
    (1 - lam) * r_t**2``, started at row ``init - 1`` with the mean square of the first ``init``
    returns. Rows before ``init - 1`` are ``NaN``. The input must contain no ``NaN`` or infinity
    (pass a complete-case panel).
    """
    _check_frame(returns, "returns")
    if not 0.0 < lam < 1.0:
        raise H2Error(f"lam must lie in (0, 1), got {lam}")
    if init < 2:
        raise H2Error(f"init must be >= 2, got {init}")
    x = returns.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise H2Error("returns must be finite (complete-case) for the EWMA recursion")
    if len(x) <= init:
        raise H2Error(f"need more than init={init} rows, got {len(x)}")
    var = np.full_like(x, np.nan)
    v = (x[:init] ** 2).mean(axis=0)
    var[init - 1] = v
    for t in range(init, len(x)):
        v = lam * v + (1.0 - lam) * x[t] ** 2
        var[t] = v
    if np.any(var[init - 1 :] <= 0.0):
        raise H2Error("an EWMA variance is zero (a column is constant over the start-up window)")
    return pd.DataFrame(np.sqrt(var), index=returns.index, columns=returns.columns)


def vol_standardize(
    returns: pd.DataFrame,
    lam: float = DEFAULT_VOL_LAMBDA,
    init: int = DEFAULT_VOL_INIT,
) -> pd.DataFrame:
    """Return divided by the *lagged* EWMA volatility: ``z_t = r_t / sigma_{t-1}``.

    ``sigma_{t-1}`` uses data through ``t-1`` only, so ``z_t`` is a function of data up to ``t``
    (causal). The first ``init`` rows are ``NaN``. The output is the robustness input for the
    engine: its regime-conditional correlations are not inflated by a rise in each asset's own
    volatility, but they still change if the common-factor share of that volatility changes, so it
    is a robustness check and not an exact deflator (see the module docstring and the check
    report).
    """
    sigma = ewma_volatility(returns, lam, init)
    return returns / sigma.shift(1)


# --------------------------------------------------------------------------------------
# Sample preparation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Sample:
    x: FloatArray  # (T, N) returns on the usable days
    codes: IntArray  # (T,) aligned regime code of each usable day
    s: FloatArray | None  # (T,) optional delta series on the usable days
    index: pd.Index
    columns: pd.Index
    mask: BoolArray  # (N, N) pairs in scope
    labels: pd.Series  # aligned labels on the usable days (float codes)


def _prepare(
    returns: pd.DataFrame,
    labels: pd.Series,
    sectors: Mapping[str, str] | pd.Series,
    *,
    alignment: Alignment,
    pairs: PairScope,
    delta_mode: DeltaMode,
    delta_series: pd.Series | None,
) -> _Sample:
    _check_frame(returns, "returns", min_columns=2)
    if not isinstance(labels, pd.Series):
        raise H2Error(f"labels must be a pandas Series, got {type(labels)}")
    if delta_mode not in ("pair", "series"):
        raise H2Error(f"unknown delta_mode {delta_mode!r}; expected 'pair' or 'series'")
    if delta_mode == "series" and delta_series is None:
        raise H2Error("delta_mode='series' needs a delta_series")
    if delta_mode == "pair" and delta_series is not None:
        raise H2Error("delta_series is only used with delta_mode='series'")
    values = returns.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise H2Error("returns contain +-inf")

    mask = pair_mask(_sector_vector(returns.columns, sectors), pairs)
    aligned = align_regime_labels(labels, returns.index, alignment)
    valid = returns.notna().all(axis=1).to_numpy() & aligned.notna().to_numpy()
    s_all: pd.Series | None = None
    if delta_series is not None:
        if not isinstance(delta_series, pd.Series):
            raise H2Error(f"delta_series must be a pandas Series, got {type(delta_series)}")
        s_all = delta_series.reindex(returns.index)
        if np.isinf(s_all.to_numpy(dtype=float)).any():
            raise H2Error("delta_series contains +-inf")
        valid = valid & s_all.notna().to_numpy()

    lab = aligned.to_numpy(dtype=float)[valid]
    if lab.size == 0:
        raise H2Error("no usable rows after aligning returns and regime labels")
    if np.any(lab != np.floor(lab)) or np.any(lab < 0):
        raise H2Error("regime labels must be non-negative integer codes")
    return _Sample(
        x=values[valid],
        codes=lab.astype(np.int64),
        s=None if s_all is None else s_all.to_numpy(dtype=float)[valid],
        index=returns.index[valid],
        columns=returns.columns,
        mask=mask,
        labels=aligned[valid],
    )


def _check_regimes(sample: _Sample, regimes: Sequence[int] | None, reference: int) -> list[int]:
    present = sorted({int(c) for c in sample.codes})
    chosen = present if regimes is None else [int(g) for g in regimes]
    if len(set(chosen)) != len(chosen) or not chosen:
        raise H2Error(f"regimes must be a non-empty list of distinct codes, got {chosen}")
    absent = [g for g in chosen if g not in present]
    if absent:
        raise H2Error(f"regimes {absent} do not occur in the usable sample (present: {present})")
    if reference not in chosen:
        raise H2Error(f"reference regime {reference} is not among the regimes {chosen}")
    return chosen


# --------------------------------------------------------------------------------------
# The core computation (shared by the point estimate and by every bootstrap replicate)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _RegimeBlock:
    raw: FloatArray
    adjusted: FloatArray
    delta: FloatArray  # (N,) per-asset delta ("pair") or the common delta repeated ("series")


def _regime_blocks(
    x: FloatArray,
    codes: IntArray,
    s: FloatArray | None,
    regimes: Sequence[int],
    reference: int,
    delta_mode: DeltaMode,
    min_rows: int,
) -> list[_RegimeBlock]:
    n_assets = x.shape[1]
    ref_rows = codes == reference
    if int(ref_rows.sum()) < min_rows:
        raise H2Error(
            f"reference regime {reference} has {int(ref_rows.sum())} days, need >= {min_rows}"
        )
    var_ref = x[ref_rows].var(axis=0, ddof=1)
    s_var_ref = float(s[ref_rows].var(ddof=1)) if s is not None else float("nan")
    if np.any(var_ref <= 0.0):
        raise H2Error("a ticker has zero variance in the reference regime")
    if delta_mode == "series" and not s_var_ref > 0.0:
        raise H2Error("the delta series has zero variance in the reference regime")

    blocks: list[_RegimeBlock] = []
    for g in regimes:
        rows = codes == g
        n_g = int(rows.sum())
        if n_g < min_rows:
            raise H2Error(f"regime {g} has {n_g} days, need >= {min_rows}")
        xg = x[rows]
        if np.any(xg.var(axis=0, ddof=1) <= _MIN_VAR_RATIO * var_ref):
            raise H2Error(f"a ticker has zero variance in regime {g}")
        raw = np.corrcoef(xg, rowvar=False)
        raw = np.clip((raw + raw.T) / 2.0, -1.0, 1.0)
        np.fill_diagonal(raw, 1.0)
        if delta_mode == "pair":
            d = xg.var(axis=0, ddof=1) / var_ref - 1.0
            adjusted = 0.5 * (
                forbes_rigobon_adjust(raw, d[:, None]) + forbes_rigobon_adjust(raw, d[None, :])
            )
        else:
            assert s is not None
            common = float(s[rows].var(ddof=1)) / s_var_ref - 1.0
            d = np.full(n_assets, common)
            adjusted = forbes_rigobon_adjust(raw, common)
        np.fill_diagonal(adjusted, 1.0)
        blocks.append(_RegimeBlock(raw=raw, adjusted=adjusted, delta=d))
    return blocks


def _averages(blocks: Sequence[_RegimeBlock], mask: BoolArray) -> tuple[FloatArray, FloatArray]:
    raw = np.array([b.raw[mask].mean() for b in blocks])
    adjusted = np.array([b.adjusted[mask].mean() for b in blocks])
    return raw, adjusted


# --------------------------------------------------------------------------------------
# Point estimates
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeCorrelationResult:
    """Point estimates of the regime-conditional correlation table (no confidence intervals).

    ``table`` is indexed by regime code with columns ``n_obs`` (days), ``n_runs`` (regime
    episodes, from :func:`etfcorr.regimes.regime_summary`), ``avg_raw`` and ``avg_adjusted`` (mean
    over the pairs in scope of the raw and the Forbes-Rigobon-adjusted correlation) and
    ``mean_delta`` (mean of the deltas used). ``raw_matrices`` and ``adjusted_matrices`` hold the
    per-regime matrices (labelled by ticker).
    """

    table: pd.DataFrame
    raw_matrices: dict[int, pd.DataFrame]
    adjusted_matrices: dict[int, pd.DataFrame]
    reference: int
    alignment: Alignment
    pairs: PairScope
    delta_mode: DeltaMode
    n_assets: int
    n_pairs: int
    n_rows: int


def _point(
    sample: _Sample,
    regimes: Sequence[int],
    reference: int,
    *,
    alignment: Alignment,
    pairs: PairScope,
    delta_mode: DeltaMode,
    min_rows: int,
) -> RegimeCorrelationResult:
    blocks = _regime_blocks(
        sample.x, sample.codes, sample.s, regimes, reference, delta_mode, min_rows
    )
    avg_raw, avg_adj = _averages(blocks, sample.mask)
    runs = regime_summary(sample.labels, n_regimes=int(sample.codes.max()) + 1)
    n_runs = dict(
        zip(runs.index.tolist(), runs["n_runs"].to_numpy(dtype=int).tolist(), strict=True)
    )
    table = pd.DataFrame(
        {
            "n_obs": [int((sample.codes == g).sum()) for g in regimes],
            "n_runs": [n_runs[g] for g in regimes],
            "avg_raw": avg_raw,
            "avg_adjusted": avg_adj,
            "mean_delta": [float(b.delta.mean()) for b in blocks],
        },
        index=pd.Index(list(regimes), name="regime"),
    )
    cols = sample.columns
    return RegimeCorrelationResult(
        table=table,
        raw_matrices={
            g: pd.DataFrame(b.raw, index=cols, columns=cols)
            for g, b in zip(regimes, blocks, strict=True)
        },
        adjusted_matrices={
            g: pd.DataFrame(b.adjusted, index=cols, columns=cols)
            for g, b in zip(regimes, blocks, strict=True)
        },
        reference=reference,
        alignment=alignment,
        pairs=pairs,
        delta_mode=delta_mode,
        n_assets=len(cols),
        n_pairs=int(sample.mask.sum()),
        n_rows=len(sample.index),
    )


def regime_correlations(
    returns: pd.DataFrame,
    labels: pd.Series,
    sectors: Mapping[str, str] | pd.Series,
    *,
    alignment: Alignment = "predetermined",
    pairs: PairScope = "cross",
    reference: int = DEFAULT_REFERENCE,
    regimes: Sequence[int] | None = None,
    delta_mode: DeltaMode = "pair",
    delta_series: pd.Series | None = None,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> RegimeCorrelationResult:
    """Regime-conditional raw and Forbes-Rigobon-adjusted average correlation (point estimates).

    ``returns``: a complete-case panel of (log) returns on a trading calendar. ``labels``: integer
    regime codes stored as floats (output of :mod:`etfcorr.regimes`) on the *same* trading
    calendar, covering at least the dates of ``returns``. ``sectors``: ticker to sector.
    ``regimes`` defaults to every code present in the usable sample; ``reference`` must be one of
    them. See the module docstring for ``alignment``, ``delta_mode`` and the assumptions.
    """
    sample = _prepare(
        returns,
        labels,
        sectors,
        alignment=alignment,
        pairs=pairs,
        delta_mode=delta_mode,
        delta_series=delta_series,
    )
    chosen = _check_regimes(sample, regimes, reference)
    return _point(
        sample,
        chosen,
        reference,
        alignment=alignment,
        pairs=pairs,
        delta_mode=delta_mode,
        min_rows=min_rows,
    )


# --------------------------------------------------------------------------------------
# Stationary-bootstrap confidence intervals
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class H2Stat:
    """One statistic of the H2 table with its percentile bootstrap interval.

    ``name`` is one of ``avg_raw``, ``avg_adjusted`` (every regime) or, for each non-reference
    regime, ``diff_raw`` and ``diff_adjusted`` (the regime minus the reference regime),
    ``vol_explained`` (``diff_raw - diff_adjusted``: the part of the raw increase the adjustment
    attributes to volatility) and ``vol_share`` (``vol_explained / diff_raw``; interval ``NaN``
    unless the interval of ``diff_raw`` excludes zero). The share is meaningful only when the raw
    increase is positive (the direction H2 predicts); with a negative raw difference it is just a
    ratio of two negative numbers and is not an "explained fraction".
    """

    name: str
    regime: int
    point: float
    lower: float
    upper: float
    replicates: FloatArray = field(repr=False, compare=False)

    @property
    def excludes_zero(self) -> bool:
        """True when the interval is defined and does not contain 0."""
        return bool(self.lower > 0.0 or self.upper < 0.0)


@dataclass(frozen=True)
class H2Result:
    """The H2 table: point estimates (:attr:`point`) plus one :class:`H2Stat` per statistic."""

    point: RegimeCorrelationResult
    stats: tuple[H2Stat, ...]
    alpha: float
    n_bootstrap: int
    mean_block_length: float
    seed: int

    def stat(self, name: str, regime: int) -> H2Stat:
        for s in self.stats:
            if s.name == name and s.regime == regime:
                return s
        raise KeyError(f"no statistic {name!r} for regime {regime}")

    def frame(self) -> pd.DataFrame:
        """Tidy table: one row per statistic with point, interval and ``excludes_zero``."""
        return pd.DataFrame(
            [
                {
                    "stat": s.name,
                    "regime": s.regime,
                    "point": s.point,
                    "lower": s.lower,
                    "upper": s.upper,
                    "excludes_zero": s.excludes_zero,
                }
                for s in self.stats
            ]
        )


def _interval(
    name: str, regime: int, point: float, reps: FloatArray, alpha: float, *, nan_safe: bool = False
) -> H2Stat:
    if nan_safe:
        lower = float(np.nanquantile(reps, alpha / 2.0))
        upper = float(np.nanquantile(reps, 1.0 - alpha / 2.0))
    else:
        lower = float(np.quantile(reps, alpha / 2.0))
        upper = float(np.quantile(reps, 1.0 - alpha / 2.0))
    return H2Stat(name, regime, point, lower, upper, reps)


def bootstrap_regime_correlations(
    returns: pd.DataFrame,
    labels: pd.Series,
    sectors: Mapping[str, str] | pd.Series,
    *,
    alignment: Alignment = "predetermined",
    pairs: PairScope = "cross",
    reference: int = DEFAULT_REFERENCE,
    regimes: Sequence[int] | None = None,
    delta_mode: DeltaMode = "pair",
    delta_series: pd.Series | None = None,
    min_rows: int = DEFAULT_MIN_ROWS,
    mean_block_length: float | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> H2Result:
    """The H2 table with stationary-bootstrap percentile intervals (module docstring: Inference).

    Every replicate resamples the usable days with one stationary-bootstrap index vector and
    applies it to the returns, the aligned regime codes and the optional delta series together;
    regime cutpoints and the alignment are fixed beforehand, on the original sample. A replicate in
    which a regime has fewer than ``min_rows`` days raises :class:`H2Error` (it is not silently
    skipped, which would bias the interval toward the stable replicates). Two calls with identical
    inputs and ``seed`` return identical intervals.
    """
    if n_bootstrap < 2:
        raise H2Error(f"n_bootstrap must be >= 2, got {n_bootstrap}")
    if not 0.0 < alpha < 1.0:
        raise H2Error(f"alpha must lie in (0, 1), got {alpha}")
    sample = _prepare(
        returns,
        labels,
        sectors,
        alignment=alignment,
        pairs=pairs,
        delta_mode=delta_mode,
        delta_series=delta_series,
    )
    chosen = _check_regimes(sample, regimes, reference)
    point = _point(
        sample,
        chosen,
        reference,
        alignment=alignment,
        pairs=pairs,
        delta_mode=delta_mode,
        min_rows=min_rows,
    )
    n_rows = len(sample.index)
    mbl = default_mean_block_length(n_rows) if mean_block_length is None else mean_block_length

    rng = np.random.default_rng(seed)
    raw_rep = np.empty((n_bootstrap, len(chosen)))
    adj_rep = np.empty((n_bootstrap, len(chosen)))
    for b in range(n_bootstrap):
        idx = stationary_bootstrap_indices(n_rows, mbl, rng)
        blocks = _regime_blocks(
            sample.x[idx],
            sample.codes[idx],
            None if sample.s is None else sample.s[idx],
            chosen,
            reference,
            delta_mode,
            min_rows,
        )
        raw_rep[b], adj_rep[b] = _averages(blocks, sample.mask)

    ref_pos = chosen.index(reference)
    p_raw = point.table["avg_raw"].to_numpy(dtype=float)
    p_adj = point.table["avg_adjusted"].to_numpy(dtype=float)
    stats: list[H2Stat] = []
    for k, g in enumerate(chosen):
        stats.append(_interval("avg_raw", g, float(p_raw[k]), raw_rep[:, k], alpha))
        stats.append(_interval("avg_adjusted", g, float(p_adj[k]), adj_rep[:, k], alpha))
    for k, g in enumerate(chosen):
        if g == reference:
            continue
        d_raw = raw_rep[:, k] - raw_rep[:, ref_pos]
        d_adj = adj_rep[:, k] - adj_rep[:, ref_pos]
        explained = d_raw - d_adj
        pt_raw = float(p_raw[k] - p_raw[ref_pos])
        pt_adj = float(p_adj[k] - p_adj[ref_pos])
        diff_raw = _interval("diff_raw", g, pt_raw, d_raw, alpha)
        stats.append(diff_raw)
        stats.append(_interval("diff_adjusted", g, pt_adj, d_adj, alpha))
        stats.append(_interval("vol_explained", g, pt_raw - pt_adj, explained, alpha))

        share_pt = (pt_raw - pt_adj) / pt_raw if abs(pt_raw) > _SHARE_EPS else float("nan")
        usable = np.abs(d_raw) > _SHARE_EPS
        share_rep = np.full_like(d_raw, np.nan)
        share_rep[usable] = explained[usable] / d_raw[usable]
        if diff_raw.excludes_zero and usable.any():
            share = _interval("vol_share", g, share_pt, share_rep, alpha, nan_safe=True)
        else:
            share = H2Stat("vol_share", g, share_pt, float("nan"), float("nan"), share_rep)
        stats.append(share)

    return H2Result(
        point=point,
        stats=tuple(stats),
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        mean_block_length=float(mbl),
        seed=seed,
    )


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def h2_summary_lines(result: H2Result, regime_names: Sequence[str] | None = None) -> list[str]:
    """Human-readable report lines (settings, per-regime table, contrasts, caveats).

    ``regime_names`` labels the regime codes in ascending order; by default the VIX tercile names
    when the codes are ``0, 1, 2``.
    """
    pt = result.point
    codes = [int(g) for g in pt.table.index]
    if regime_names is None:
        names = list(TERCILE_NAMES) if codes == [0, 1, 2] else [f"regime {g}" for g in codes]
    else:
        if len(regime_names) != len(codes):
            raise H2Error(f"need {len(codes)} regime names, got {len(regime_names)}")
        names = list(regime_names)
    label = dict(zip(codes, names, strict=True))
    level = round(100 * (1.0 - result.alpha))

    def fmt(s: H2Stat) -> str:
        if np.isnan(s.lower):
            return f"{s.point:+.4f} [CI not defined]"
        star = "*" if s.excludes_zero else ""
        return f"{s.point:+.4f} [{s.lower:+.4f}, {s.upper:+.4f}]{star}"

    lines = [
        f"H2 engine (in-sample, descriptive): alignment={pt.alignment}, pairs={pt.pairs}, "
        f"delta_mode={pt.delta_mode}, reference={label[pt.reference]}",
        f"  T={pt.n_rows} days, N={pt.n_assets} tickers, {pt.n_pairs} pairs in scope, "
        f"{result.n_bootstrap} stationary-bootstrap draws, mean block length "
        f"{result.mean_block_length:.1f}, {level}% percentile intervals, seed {result.seed}",
    ]
    n_obs = dict(
        zip(pt.table.index.tolist(), pt.table["n_obs"].to_numpy(dtype=int).tolist(), strict=True)
    )
    n_runs = dict(
        zip(pt.table.index.tolist(), pt.table["n_runs"].to_numpy(dtype=int).tolist(), strict=True)
    )
    mean_delta = dict(
        zip(
            pt.table.index.tolist(),
            pt.table["mean_delta"].to_numpy(dtype=float).tolist(),
            strict=True,
        )
    )
    for g in codes:
        lines.append(
            f"  {label[g]:>5}: n={n_obs[g]} days in {n_runs[g]} episodes, "
            f"mean delta {mean_delta[g]:+.3f}; "
            f"raw {fmt(result.stat('avg_raw', g))}; "
            f"adjusted {fmt(result.stat('avg_adjusted', g))}"
        )
    for g in codes:
        if g == pt.reference:
            continue
        lines.append(
            f"  {label[g]} minus {label[pt.reference]}: "
            f"raw {fmt(result.stat('diff_raw', g))}; "
            f"adjusted {fmt(result.stat('diff_adjusted', g))}; "
            f"explained by volatility {fmt(result.stat('vol_explained', g))}; "
            f"share {fmt(result.stat('vol_share', g))}"
        )
    lines.append("  * = interval excludes 0 (uncorrected for multiple comparisons)")
    if pt.delta_mode == "pair":
        direction = (
            "with a common factor the per-asset-delta adjustment under-corrects, so a small "
            "explained share is biased downward"
        )
    else:
        direction = (
            "with one delta from a market series the adjustment can over-attribute to "
            "volatility when dependence also rises"
        )
    lines.append(
        "  Forbes-Rigobon assumes no omitted variables or endogeneity; "
        + direction
        + " (see docs/checks/4.2.md)"
    )
    return lines
