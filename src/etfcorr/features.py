"""Step 5.3: leakage-safe features for the forecasting ladder (Phases 5 to 7).

Contract (project_plan.md section 9, the rule ``etfcorr.leakage`` tests): every function here is
a pure function ``f(data_up_to_t) -> value_at_t``. The value dated ``t`` uses rows dated ``<= t``
only, through *trailing* windows and one-pass recursions; nothing is centred, nothing is scaled
with full-sample statistics, and no scaler or PCA is fitted here (those are fitted inside each
training window, in ``etfcorr.models.linear``). A feature dated ``t`` is "lagged by at least one
step" relative to its target because every target in ``etfcorr.targets`` starts at ``t + 1``.

Three feature tables, all daily:

* :func:`build_features`: one row per date, the common state of the panel (trailing average
  cross-sector correlation in Fisher-z at several windows, its EWMA and rank-based versions,
  within-sector correlation, market-mode share, cross-sectional volatility and dispersion, and
  optional market and VIX columns).
* :func:`build_asset_features`: one row per (date, asset), for the volatility and rank targets.
* :func:`build_pair_features`: one row per (date, pair), for the full-matrix target.

Conventions: correlations enter models in Fisher-z space, ``z = atanh(r)`` with ``r`` clipped to
``+-0.999`` (:func:`fisher_z`); variances are *mean daily* squared log returns over the window
(zero mean, the RiskMetrics convention already used by ``estimators.ewma_corr``), in logs with a
floor of ``1e-12``. The Fisher-z of an average correlation is a variance-stabilising
re-parametrisation used for modelling, not Fisher's sampling distribution: an average of many
dependent pairwise correlations does not have the textbook ``1 / (T - 3)`` variance.

Assumptions and limits:

* A window with a missing value gives ``NaN`` for that date (no forward-fill, no pairwise
  deletion). The EWMA recursion needs complete rows and raises on a missing value.
* A 21-day window on 25 assets gives a rank-deficient, noisy correlation matrix. It is still a
  valid correlation matrix and its average over about 260 cross-sector pairs is usable, but
  single entries are very noisy.
* The rank-based (Spearman) column exists because step 4.5b found product-based statistics
  sensitive to the March 2020 cluster; it is a robustness input, not a claim that ranks are
  better.
* HMM-state and network features are deliberately absent: the Phase 4 and Phase 3 carry-ins
  (a simple-baseline comparison for the HMM tag, a real-data check of ``networks.py``) are open.

The trailing-window statistics are plain NumPy/pandas; nothing is adapted from third-party code.
"""

from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias, TypeVar, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import rankdata

from etfcorr.correlation.regime_correlation import PairScope, pair_mask

FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
CorrKind: TypeAlias = Literal["pearson", "spearman"]
_F = TypeVar("_F", pd.DataFrame, pd.Series)

DEFAULT_WINDOWS: tuple[int, ...] = (21, 63, 126, 252)
VOL_WINDOWS: tuple[int, ...] = (5, 21, 63)
STRUCTURE_WINDOW = 63  # window of the within-sector, rank-based and market-mode columns
MOMENTUM_WINDOW = 252  # "12-1" momentum: days t-252+1 .. t-21
MOMENTUM_SKIP = 21
DEFAULT_LAMBDA = 0.94  # mirrors config/experiment.yaml's ewma_lambda
Z_CLIP = 0.999
VAR_FLOOR = 1e-12
UNIT = "unit"  # second index level of the long tables (an asset or a pair)
PAIR_SEP = "|"


class FeatureError(ValueError):
    """Invalid input to a feature function."""


# --------------------------------------------------------------------------------------
# Small transforms
# --------------------------------------------------------------------------------------


def fisher_z(r: npt.ArrayLike) -> FloatArray:
    """``atanh`` of a correlation clipped to ``+-0.999`` (``NaN`` stays ``NaN``)."""
    return np.asarray(np.arctanh(np.clip(np.asarray(r, dtype=float), -Z_CLIP, Z_CLIP)))


def inverse_fisher_z(z: npt.ArrayLike) -> FloatArray:
    """``tanh``: back from Fisher-z to a correlation in ``(-1, 1)``."""
    return np.asarray(np.tanh(np.asarray(z, dtype=float)))


def log_variance(v: _F) -> _F:
    """Natural log of a variance floored at ``1e-12`` (an all-zero window stays finite)."""
    return cast(_F, np.log(v.clip(lower=VAR_FLOOR)))


def cross_sectional_rank(wide: pd.DataFrame) -> pd.DataFrame:
    """Per-date rank across columns, mapped to ``[-0.5, 0.5]`` (ties averaged, ``NaN`` kept).

    A date with fewer than two finite values is all ``NaN``.
    """
    n = wide.notna().sum(axis=1)
    rank = wide.rank(axis=1, method="average")
    out = rank.sub(1.0).div((n - 1.0).where(n > 1), axis=0) - 0.5
    return out


def _check_returns(returns: pd.DataFrame, min_columns: int = 1) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise FeatureError(f"returns must be a pandas DataFrame, got {type(returns)}")
    if returns.shape[1] < min_columns:
        raise FeatureError(f"need at least {min_columns} columns, got {returns.shape[1]}")
    if not returns.index.is_monotonic_increasing or not returns.index.is_unique:
        raise FeatureError("the index must be sorted and unique")


def _check_window(window: int) -> None:
    if window < 2:
        raise FeatureError(f"window must be >= 2, got {window}")


# --------------------------------------------------------------------------------------
# Trailing correlation
# --------------------------------------------------------------------------------------


def trailing_correlations(
    returns: pd.DataFrame, window: int, *, kind: CorrKind = "pearson"
) -> FloatArray:
    """Correlation matrix of the trailing ``window`` rows at each date: shape ``(T, N, N)``.

    Row ``t`` uses rows ``t - window + 1 .. t``. It is ``NaN`` before ``window`` rows exist or
    when the window holds a missing value; a column that is constant in the window gives ``NaN``
    in its row and column. ``kind="spearman"`` ranks each column inside the window first.
    """
    _check_returns(returns, 2)
    _check_window(window)
    if kind not in ("pearson", "spearman"):
        raise FeatureError(f"unknown correlation kind {kind!r}")
    x = returns.to_numpy(dtype=float)
    total, n = x.shape
    out = np.full((total, n, n), np.nan)
    finite = np.isfinite(x).all(axis=1)
    bad = np.concatenate([[0], np.cumsum(~finite)])
    for t in range(window - 1, total):
        if bad[t + 1] - bad[t + 1 - window]:
            continue
        w = x[t + 1 - window : t + 1]
        if kind == "spearman":
            w = rankdata(w, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t] = np.corrcoef(w, rowvar=False)
    return out


def ewma_correlations(returns: pd.DataFrame, lam: float = DEFAULT_LAMBDA) -> FloatArray:
    """EWMA correlation matrix as of each date, ``(T, N, N)``, in one causal pass.

    Same recursion as ``estimators.ewma_corr`` on the full history up to each date
    (``S_0 = r_0 r_0'``, ``S_t = lam S_{t-1} + (1 - lam) r_t r_t'``, zero mean), which a unit test
    checks row by row. Row 0 is ``NaN`` (``ewma_corr`` needs two rows).
    """
    _check_returns(returns, 2)
    if not 0.0 < lam < 1.0:
        raise FeatureError(f"lam must be in (0, 1), got {lam}")
    x = returns.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise FeatureError("the EWMA recursion needs complete rows (no missing value)")
    total, n = x.shape
    out = np.full((total, n, n), np.nan)
    if total == 0:
        return out
    s = np.outer(x[0], x[0])
    for t in range(1, total):
        s = lam * s + (1.0 - lam) * np.outer(x[t], x[t])
        d = np.sqrt(np.diag(s))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t] = s / np.outer(d, d)
    return out


def masked_average(corr: FloatArray, mask: BoolArray) -> FloatArray:
    """Mean of the entries of each ``(N, N)`` matrix in ``corr`` selected by ``mask``."""
    m = np.asarray(mask, dtype=bool)
    if corr.ndim != 3 or m.shape != corr.shape[1:]:
        raise FeatureError(f"mask shape {m.shape} does not fit matrices of shape {corr.shape}")
    if not m.any():
        raise FeatureError("the mask selects no pair")
    return np.asarray(corr[:, m].mean(axis=1), dtype=float)


def sector_pair_mask(
    columns: Sequence[str], sectors: Mapping[str, str], scope: PairScope = "cross"
) -> BoolArray:
    """Upper-triangular pair mask by sector (``"cross"``, ``"within"`` or ``"all"``).

    Thin wrapper over ``regime_correlation.pair_mask`` (the H2 definition), so the forecasting
    target is the same statistic as the one H2 describes.
    """
    missing = [c for c in columns if c not in sectors]
    if missing:
        raise FeatureError(f"no sector given for {missing}")
    return pair_mask([sectors[c] for c in columns], scope)


def trailing_average_correlation(
    returns: pd.DataFrame, window: int, mask: BoolArray, *, kind: CorrKind = "pearson"
) -> pd.Series:
    """Average of the masked pairs of the trailing-window correlation matrix, one value per date."""
    values = masked_average(trailing_correlations(returns, window, kind=kind), mask)
    return pd.Series(values, index=returns.index, name=f"corr_w{window}")


def market_mode_share(corr: FloatArray) -> FloatArray:
    """Largest eigenvalue divided by ``N`` for each matrix (``NaN`` where the matrix has one)."""
    out = np.full(corr.shape[0], np.nan)
    ok = np.isfinite(corr).all(axis=(1, 2))
    if ok.any():
        out[ok] = np.linalg.eigvalsh(corr[ok])[:, -1] / corr.shape[1]
    return out


# --------------------------------------------------------------------------------------
# Trailing volatility and returns
# --------------------------------------------------------------------------------------


def trailing_variance(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Mean squared daily return over the trailing ``window`` rows (``NaN`` until complete)."""
    _check_returns(returns)
    _check_window(window)
    return (returns**2).rolling(window, min_periods=window).mean()


def ewma_variance(returns: pd.DataFrame, lam: float = DEFAULT_LAMBDA) -> pd.DataFrame:
    """RiskMetrics variance ``v_t = lam v_{t-1} + (1 - lam) r_t^2`` with ``v_0 = r_0^2``."""
    _check_returns(returns)
    if not 0.0 < lam < 1.0:
        raise FeatureError(f"lam must be in (0, 1), got {lam}")
    return (returns**2).ewm(alpha=1.0 - lam, adjust=False).mean()


def trailing_return(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Sum of log returns over the trailing ``window`` rows (``NaN`` until complete)."""
    _check_returns(returns)
    if window < 1:
        raise FeatureError(f"window must be >= 1, got {window}")
    return returns.rolling(window, min_periods=window).sum()


# --------------------------------------------------------------------------------------
# Feature tables
# --------------------------------------------------------------------------------------


def build_features(
    returns: pd.DataFrame,
    sectors: Mapping[str, str],
    *,
    market: pd.Series | None = None,
    vix: pd.Series | None = None,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    lam: float = DEFAULT_LAMBDA,
) -> pd.DataFrame:
    """The common (one row per date) feature table. See the module docstring for the columns.

    ``market`` is a daily log-return series (SPY) and ``vix`` a positive level series, both on
    the index of ``returns``; each adds its columns only when given.
    """
    _check_returns(returns, 2)
    columns = [str(c) for c in returns.columns]
    cross = sector_pair_mask(columns, sectors, "cross")
    within = sector_pair_mask(columns, sectors, "within")
    out: dict[str, np.ndarray | pd.Series] = {}

    for w in windows:
        corr = trailing_correlations(returns, w)
        out[f"z_cross_w{w}"] = fisher_z(masked_average(corr, cross))
        if w == STRUCTURE_WINDOW:
            out[f"z_within_w{w}"] = fisher_z(masked_average(corr, within))
            out[f"mode_share_w{w}"] = market_mode_share(corr)
    out["z_cross_ewma"] = fisher_z(masked_average(ewma_correlations(returns, lam), cross))
    rank_corr = trailing_correlations(returns, STRUCTURE_WINDOW, kind="spearman")
    out[f"z_cross_spearman_w{STRUCTURE_WINDOW}"] = fisher_z(masked_average(rank_corr, cross))

    for w in VOL_WINDOWS:
        out[f"log_rv_xs_w{w}"] = log_variance(trailing_variance(returns, w).mean(axis=1))
    dispersion = returns.std(axis=1, ddof=0).rolling(21, min_periods=21).mean()
    out["log_dispersion_w21"] = np.log(dispersion.clip(lower=np.sqrt(VAR_FLOOR)))

    if market is not None:
        m = _aligned(market, returns.index, "market").to_frame("m")
        out["log_rv_mkt_w21"] = log_variance(trailing_variance(m, 21)["m"])
        out["ret_mkt_w21"] = trailing_return(m, 21)["m"]
    if vix is not None:
        v = _aligned(vix, returns.index, "vix")
        if (v.dropna() <= 0).any():
            raise FeatureError("vix must be a positive level series")
        log_v = np.log(v)
        out["log_vix"] = log_v
        out["dlog_vix_w5"] = log_v - log_v.shift(5)
        out["dlog_vix_w21"] = log_v - log_v.shift(21)
    return pd.DataFrame(out, index=returns.index)


def _aligned(series: pd.Series, index: pd.Index, name: str) -> pd.Series:
    if not series.index.equals(index):
        raise FeatureError(f"{name} must be on the same index as returns")
    return series.astype(float)


def stack_units(wide: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Long table indexed by ``(date, unit)`` from same-shaped wide ``date x unit`` frames."""
    if not wide:
        raise FeatureError("nothing to stack")
    first = next(iter(wide.values()))
    for name, frame in wide.items():
        if not (frame.index.equals(first.index) and frame.columns.equals(first.columns)):
            raise FeatureError(f"{name!r} does not share the index and columns of the others")
    index = pd.MultiIndex.from_product([first.index, first.columns], names=["date", UNIT])
    data = {name: frame.to_numpy(dtype=float).ravel() for name, frame in wide.items()}
    return pd.DataFrame(data, index=index)


def build_asset_features(
    returns: pd.DataFrame,
    *,
    common: pd.DataFrame | None = None,
    lam: float = DEFAULT_LAMBDA,
    rank: bool = False,
) -> pd.DataFrame:
    """Per-(date, asset) features: own volatility, own past return, momentum.

    Columns: ``log_rv_w{5,21,63}`` (log mean squared return), ``log_ewma_var``, ``ret_w5`` and
    ``ret_w21`` (trailing log return), ``mom_252_21`` (return over days ``t-251 .. t-21``).
    Columns of ``common`` (a per-date table such as a subset of :func:`build_features`) are
    repeated for every asset. ``rank=True`` replaces each own column by its cross-sectional rank
    per date (:func:`cross_sectional_rank`), the natural input for the rank target.
    """
    _check_returns(returns)
    wide: dict[str, pd.DataFrame] = {}
    for w in VOL_WINDOWS:
        wide[f"log_rv_w{w}"] = log_variance(trailing_variance(returns, w))
    wide["log_ewma_var"] = log_variance(ewma_variance(returns, lam))
    wide["ret_w5"] = trailing_return(returns, 5)
    wide["ret_w21"] = trailing_return(returns, 21)
    wide["mom_252_21"] = trailing_return(returns, MOMENTUM_WINDOW - MOMENTUM_SKIP).shift(
        MOMENTUM_SKIP
    )
    if rank:
        wide = {name: cross_sectional_rank(frame) for name, frame in wide.items()}
    if common is not None:
        if not common.index.equals(returns.index):
            raise FeatureError("common must be on the same index as returns")
        for name in common.columns:
            block = np.repeat(common[name].to_numpy(dtype=float)[:, None], returns.shape[1], 1)
            wide[str(name)] = pd.DataFrame(block, index=returns.index, columns=returns.columns)
    return stack_units(wide)


def pair_labels(columns: Sequence[str]) -> list[str]:
    """``"A|B"`` labels of the strictly-upper-triangular pairs, in ``np.triu_indices`` order."""
    i, j = np.triu_indices(len(columns), k=1)
    return [
        f"{columns[a]}{PAIR_SEP}{columns[b]}" for a, b in zip(i.tolist(), j.tolist(), strict=True)
    ]


def pairs_wide(corr: FloatArray, index: pd.Index, columns: Sequence[str]) -> pd.DataFrame:
    """Fisher-z of the upper-triangular entries of each matrix as a wide ``date x pair`` frame."""
    i, j = np.triu_indices(len(columns), k=1)
    return pd.DataFrame(fisher_z(corr[:, i, j]), index=index, columns=pair_labels(columns))


def pairs_to_matrices(wide_z: pd.DataFrame, n_assets: int) -> FloatArray:
    """Inverse of :func:`pairs_wide`: symmetric unit-diagonal matrices from Fisher-z pairs."""
    i, j = np.triu_indices(n_assets, k=1)
    if wide_z.shape[1] != len(i):
        raise FeatureError(f"expected {len(i)} pair columns, got {wide_z.shape[1]}")
    r = inverse_fisher_z(wide_z.to_numpy(dtype=float))
    out = np.empty((len(wide_z), n_assets, n_assets))
    out[:] = np.eye(n_assets)
    out[:, i, j] = r
    out[:, j, i] = r
    out[np.isnan(r).any(axis=1)] = np.nan
    return out


def build_pair_features(
    returns: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    lam: float = DEFAULT_LAMBDA,
) -> pd.DataFrame:
    """Per-(date, pair) features for the full-matrix target, all in Fisher-z.

    Columns: ``z_w{w}`` for each window (the pair's trailing correlation), ``z_ewma``, and
    ``z_const`` (the constant-correlation model of Elton and Gruber: every pair gets the average
    of all pairwise correlations over the longest window).
    """
    _check_returns(returns, 2)
    columns = [str(c) for c in returns.columns]
    wide: dict[str, pd.DataFrame] = {}
    longest: FloatArray | None = None
    for w in windows:
        longest = trailing_correlations(returns, w)
        wide[f"z_w{w}"] = pairs_wide(longest, returns.index, columns)
    if longest is None:
        raise FeatureError("windows is empty")
    wide["z_ewma"] = pairs_wide(ewma_correlations(returns, lam), returns.index, columns)
    upper = np.triu(np.ones((len(columns), len(columns)), dtype=bool), k=1)
    const = fisher_z(masked_average(longest, upper))
    first = next(iter(wide.values()))
    wide["z_const"] = pd.DataFrame(
        np.repeat(const[:, None], first.shape[1], 1), index=first.index, columns=first.columns
    )
    return stack_units(wide)
