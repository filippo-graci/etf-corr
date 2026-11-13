"""Step 4.2b (exploratory add-on to 4.2): a factor-model decomposition of the regime change in
average cross-sector correlation, without a Forbes-Rigobon delta.

Question served: the second clause of H2 ("a large part of the increase is explained by volatility
rather than by a true change in dependence"). The 4.2 engine answers it with the Forbes-Rigobon
adjustment, whose result depends on the delta that is plugged in (docs/checks/4.2.md: the explained
share spans 0.60 to 1.33 across three defensible deltas). This module answers the same question
structurally, so that no delta is needed. It is **exploratory**: it is not part of the
pre-registered H2 test, does not change H2, and its numbers never replace the 4.2 table. It is
descriptive and in-sample, like the 4.2 engine.

**Method.** In every regime ``g`` (days grouped as in 4.2: predetermined t-1 labels by default) fit
by ordinary least squares, with an intercept, ``r_t = a + B f_t + e_t`` for ``K`` factors ``f``.
Sample covariances use ``ddof=1``. With OLS residuals the *sample* identity

.. math:: \\widehat{\\mathrm{Cov}}(r) = B\\,\\Sigma_f\\,B^\\top + C_e

holds exactly (``Sigma_f`` the factor covariance, ``C_e`` the residual covariance), so the average
cross-sector correlation of the regime is reproduced to rounding error from the three components
``(Sigma_f, B, C_e)``. The *volatility-only counterfactual* for regime ``g`` keeps the reference
regime's loadings ``B`` and residual covariance ``C_e`` and changes only the factor covariance to
the one observed in ``g``: it answers "what would the average correlation be if only the volatility
of the common factors had changed". Its change relative to the reference regime, divided by the raw
change, is the delta-free analogue of the 4.2 ``vol_share`` (``var_only_share``). Because the three
components interact, a second, order-free attribution is reported as well: the Shapley value
(Shapley-Shorrocks decomposition) of the raw change over the three components (factor covariance,
loadings, residual covariance), computed from all eight combinations of "component taken from ``g``
or from the reference regime". The three Shapley terms sum to the raw change exactly. A fixed-order
decomposition would depend on the order of the components; the Shapley split does not
(**[Established]** as a procedure, Shorrocks 2013; its use here is **[Inference]**).

Descriptive checks of the "volatility only" reading are also reported per regime: the ratio of the
average systematic variance (``diag(B Sigma_f B')``), the ratio of the average residual variance
(``diag(C_e)``) and the ratio of the average loading on the first factor, each against the
reference regime. A pure volatility change would show a large first ratio and the other two near 1.

**What the result depends on -- state this wherever the numbers are reported.** The decomposition
is only as good as the factor set. Anything common to the assets that the chosen factors do not
span (a rates factor, a sector factor, a crypto factor) sits in the residual covariance, so when the
variance of such an omitted factor also rises in the high-volatility regime, its effect is booked
as a *change in residual covariance*, i.e. as a non-volatility change, and the volatility-only
share is biased downward; if instead the residual variances rise less than the chosen factor's, the
counterfactual overshoots and the share can exceed 1 (as the SPY-delta line of 4.2 did). The sign of
the bias is therefore not known in advance. Run several factor sets (the script does: SPY alone, an
equal-weight panel average, SPY plus latent principal components) and read the spread, not one
number. Further limits: OLS loadings are estimated with error inside each regime (about 1,000 days
each on the real panel) and the bootstrap propagates that, but the factor *choice* and, for
:func:`latent_factors`, the principal components (a full-sample, in-sample construction) are held
fixed in the bootstrap; regime cutpoints are fixed on the original sample as in 4.2; the pair
average mixes pairs of very different correlation; a linear factor model says nothing about tail
dependence. The statistic is a decomposition of an in-sample difference, not a causal statement.

**Alignment and inference** are exactly those of the 4.2 engine and reuse its functions
(:func:`etfcorr.correlation.regime_correlation.align_regime_labels`,
:func:`~etfcorr.correlation.regime_correlation.pair_mask`, the stationary bootstrap of
``horizon_analysis``): one index vector resamples the returns, the factors and the aligned regime
codes together, percentile intervals, a replicate in which a regime has too few days raises. As in
4.2 the interval of a ratio (``var_only_share``) is reported only when the interval of the raw
change excludes zero. No multiplicity correction here.

Not adapted from third-party code (no licence flag).

Citations (existence, authors, venue and year checked by web search/fetch on 2026-10-02; pages of
the first item checked 2026-11-02 in the published article: 24(8), 1177-1199): Corsetti, G.,
Pericoli, M. and Sbracia, M. (2005), *Some contagion, some interdependence: more pitfalls in tests
of financial contagion*, Journal of International Money and Finance 24 (abstract: a factor-model
critique of correlation-based contagion tests, whose "no contagion" conclusions depend on
restrictions on the variance of country-specific shocks, the same sensitivity found here). Bekaert,
G., Harvey, C. R. and Ng, A. (2005), *Market integration and contagion*, Journal of Business 78(1)
(defines contagion as correlation in excess of what fundamentals imply, within a factor model).
Shorrocks, A. F. (2013), *Decomposition procedures for distributional analysis: a unified framework
based on the Shapley value*, Journal of Economic Inequality 11(1), 99-126,
doi:10.1007/s10888-011-9214-z. Forbes, K. and Rigobon, R. (2002), Journal of Finance 57(5),
2223-2261 (see regime_correlation.py).
"""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from etfcorr.correlation.horizon_analysis import (
    DEFAULT_ALPHA,
    DEFAULT_N_BOOTSTRAP,
    default_mean_block_length,
    stationary_bootstrap_indices,
)
from etfcorr.correlation.regime_correlation import (
    DEFAULT_MIN_ROWS,
    DEFAULT_REFERENCE,
    Alignment,
    H2Error,
    H2Stat,
    PairScope,
    align_regime_labels,
    pair_mask,
)
from etfcorr.regimes import TERCILE_NAMES

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

COMPONENTS = ("factor_variance", "loadings", "residual_covariance")
_SHARE_EPS = 1e-12
_MIN_EIG_RATIO = 1e-12  # factor covariance is singular below this eigenvalue ratio
# Shapley weights |S|! (n - |S| - 1)! / n! for n = 3 components, S the other components present
_SHAPLEY_WEIGHT = {0: 1.0 / 3.0, 1: 1.0 / 6.0, 2: 1.0 / 3.0}
_FIELDS = (
    "avg_raw",
    "avg_var_only",
    "shap_factor_variance",
    "shap_loadings",
    "shap_residual_covariance",
    "sys_var_ratio",
    "idio_var_ratio",
    "loading_ratio",
)


# --------------------------------------------------------------------------------------
# The factor model of one regime
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorFit:
    """OLS fit of ``r = a + B f + e`` on one sample: ``loadings`` (N, K), ``factor_cov`` (K, K),
    ``resid_cov`` (N, N), all with ``ddof=1``."""

    loadings: FloatArray
    factor_cov: FloatArray
    resid_cov: FloatArray
    n_obs: int


def fit_factor_model(x: FloatArray, f: FloatArray) -> FactorFit:
    """Fit the multi-factor OLS model on the rows of ``x`` (n, N) and ``f`` (n, K).

    The factor covariance must be non-singular. By construction the sample covariance of ``x``
    equals ``B Sigma_f B' + C_e`` exactly (the residuals are orthogonal to the factors in-sample).
    """
    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)
    if x.ndim != 2 or f.ndim != 2 or x.shape[0] != f.shape[0]:
        raise H2Error(f"x and f must be 2-D with the same rows, got {x.shape} and {f.shape}")
    n, k = f.shape
    if n < k + 2:
        raise H2Error(f"need more than K + 1 = {k + 1} rows to fit the factor model, got {n}")
    xc = x - x.mean(axis=0)
    fc = f - f.mean(axis=0)
    sff = fc.T @ fc / (n - 1)
    sff = (sff + sff.T) / 2.0
    eig = np.linalg.eigvalsh(sff)
    if not eig.min() > _MIN_EIG_RATIO * max(float(eig.max()), np.finfo(float).tiny):
        raise H2Error("the factor covariance is singular in this sample")
    sxf = xc.T @ fc / (n - 1)
    loadings = np.linalg.solve(sff, sxf.T).T
    resid = xc - fc @ loadings.T
    c_e = resid.T @ resid / (n - 1)
    c_e = (c_e + c_e.T) / 2.0
    return FactorFit(loadings=loadings, factor_cov=sff, resid_cov=c_e, n_obs=int(n))


def implied_correlation(
    loadings: FloatArray, factor_cov: FloatArray, resid_cov: FloatArray
) -> FloatArray:
    """Correlation matrix implied by ``B Sigma_f B' + C_e`` (positive diagonal required)."""
    cov = loadings @ factor_cov @ loadings.T + resid_cov
    cov = (cov + cov.T) / 2.0
    d = np.diag(cov)
    if not np.all(d > 0.0):
        raise H2Error("the implied covariance has a non-positive variance")
    s = np.sqrt(d)
    return np.asarray(cov / np.outer(s, s), dtype=float)


def corner_averages(
    ref: FactorFit, other: FactorFit, mask: BoolArray
) -> dict[tuple[int, int, int], float]:
    """Mean implied pair correlation for each of the eight mixtures of the two regimes.

    The key ``(a, b, c)`` says whether the factor covariance, the loadings and the residual
    covariance (in this order, :data:`COMPONENTS`) are taken from ``other`` (1) or from ``ref`` (0).
    ``(0, 0, 0)`` is the reference regime itself, ``(1, 1, 1)`` the other regime itself.
    """
    out: dict[tuple[int, int, int], float] = {}
    for flags in itertools.product((0, 1), repeat=3):
        sig = other.factor_cov if flags[0] else ref.factor_cov
        bet = other.loadings if flags[1] else ref.loadings
        res = other.resid_cov if flags[2] else ref.resid_cov
        out[(flags[0], flags[1], flags[2])] = float(implied_correlation(bet, sig, res)[mask].mean())
    return out


def shapley_attribution(corner: Mapping[tuple[int, int, int], float]) -> tuple[float, float, float]:
    """Shapley value of each component for the change ``v(1,1,1) - v(0,0,0)``.

    The three values sum to that change exactly (efficiency) and do not depend on any ordering of
    the components.
    """
    values = []
    for c in range(3):
        others = [j for j in range(3) if j != c]
        total = 0.0
        for r in range(3):
            for subset in itertools.combinations(others, r):
                without = [0, 0, 0]
                for j in subset:
                    without[j] = 1
                with_c = list(without)
                with_c[c] = 1
                total += _SHAPLEY_WEIGHT[r] * (
                    corner[(with_c[0], with_c[1], with_c[2])]
                    - corner[(without[0], without[1], without[2])]
                )
        values.append(total)
    return values[0], values[1], values[2]


def latent_factors(
    returns: pd.DataFrame, base: pd.DataFrame | pd.Series, n_extra: int
) -> pd.DataFrame:
    """The first ``n_extra`` principal components of the returns left over after ``base``.

    Regress every column of ``returns`` on ``base`` (with an intercept) over the complete rows,
    standardise the residuals, and return the scores of their leading principal components
    (eigenvectors of the residual correlation matrix) as factor series named ``PC1``, ``PC2``...,
    aligned to ``returns.index`` (NaN where a row is incomplete). The sign of each component is
    fixed so that its mean loading is non-negative. **This is a full-sample, in-sample
    construction for a descriptive decomposition; it must not be used inside a training window of
    a forecasting model.**
    """
    if n_extra < 1:
        raise H2Error(f"n_extra must be >= 1, got {n_extra}")
    base_df = base.to_frame() if isinstance(base, pd.Series) else base
    base_df = base_df.reindex(returns.index)
    ok = returns.notna().all(axis=1).to_numpy() & base_df.notna().all(axis=1).to_numpy()
    if int(ok.sum()) < returns.shape[1] + base_df.shape[1] + 2:
        raise H2Error("too few complete rows to extract latent factors")
    x = returns.to_numpy(dtype=float)[ok]
    b = base_df.to_numpy(dtype=float)[ok]
    xc = x - x.mean(axis=0)
    bc = b - b.mean(axis=0)
    beta = np.linalg.lstsq(bc, xc, rcond=None)[0]
    resid = xc - bc @ beta
    sd = resid.std(axis=0, ddof=1)
    if np.any(sd <= 0.0):
        raise H2Error("a column has zero residual variance after removing the base factors")
    z = resid / sd
    corr = np.corrcoef(z, rowvar=False)
    eigval, eigvec = np.linalg.eigh(corr)
    order = np.argsort(eigval)[::-1][:n_extra]
    vecs = eigvec[:, order]
    signs = np.where(vecs.sum(axis=0) < 0.0, -1.0, 1.0)
    scores = z @ (vecs * signs)
    out = np.full((len(returns), n_extra), np.nan)
    out[ok] = scores
    return pd.DataFrame(out, index=returns.index, columns=[f"PC{j + 1}" for j in range(n_extra)])


# --------------------------------------------------------------------------------------
# Sample preparation and the shared core computation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Sample:
    x: FloatArray
    f: FloatArray
    codes: IntArray
    index: pd.Index
    columns: pd.Index
    factor_names: list[str]
    mask: BoolArray


def _sector_vector(columns: pd.Index, sectors: Mapping[str, str] | pd.Series) -> list[str]:
    lookup = dict(sectors.items()) if isinstance(sectors, pd.Series) else dict(sectors)
    missing = [str(c) for c in columns if c not in lookup]
    if missing:
        raise H2Error(f"no sector given for tickers {missing}")
    return [str(lookup[c]) for c in columns]


def _prepare(
    returns: pd.DataFrame,
    factors: pd.DataFrame | pd.Series,
    labels: pd.Series,
    sectors: Mapping[str, str] | pd.Series,
    alignment: Alignment,
    pairs: PairScope,
) -> _Sample:
    if not isinstance(returns, pd.DataFrame) or returns.shape[1] < 2:
        raise H2Error("returns must be a DataFrame with at least two columns")
    if not returns.index.is_monotonic_increasing or not returns.index.is_unique:
        raise H2Error("returns index must be strictly increasing and unique")
    if not returns.columns.is_unique:
        raise H2Error("returns column names must be unique")
    if not isinstance(labels, pd.Series):
        raise H2Error(f"labels must be a pandas Series, got {type(labels)}")
    if isinstance(factors, pd.Series):
        factors = factors.to_frame(name=str(factors.name) if factors.name is not None else "f1")
    if not isinstance(factors, pd.DataFrame) or factors.shape[1] < 1:
        raise H2Error("factors must be a Series or a DataFrame with at least one column")
    if not factors.columns.is_unique:
        raise H2Error("factor column names must be unique")
    values = returns.to_numpy(dtype=float)
    fvalues = factors.reindex(returns.index).to_numpy(dtype=float)
    if np.isinf(values).any() or np.isinf(fvalues).any():
        raise H2Error("returns or factors contain +-inf")
    mask = pair_mask(_sector_vector(returns.columns, sectors), pairs)
    aligned = align_regime_labels(labels, returns.index, alignment)
    valid = (
        returns.notna().all(axis=1).to_numpy()
        & ~np.isnan(fvalues).any(axis=1)
        & aligned.notna().to_numpy()
    )
    lab = aligned.to_numpy(dtype=float)[valid]
    if lab.size == 0:
        raise H2Error("no usable rows after aligning returns, factors and regime labels")
    if np.any(lab != np.floor(lab)) or np.any(lab < 0):
        raise H2Error("regime labels must be non-negative integer codes")
    return _Sample(
        x=values[valid],
        f=fvalues[valid],
        codes=lab.astype(np.int64),
        index=returns.index[valid],
        columns=returns.columns,
        factor_names=[str(c) for c in factors.columns],
        mask=mask,
    )


def _check_regimes(codes: IntArray, regimes: Sequence[int] | None, reference: int) -> list[int]:
    present = sorted({int(c) for c in codes})
    chosen = present if regimes is None else [int(g) for g in regimes]
    if len(set(chosen)) != len(chosen) or not chosen:
        raise H2Error(f"regimes must be a non-empty list of distinct codes, got {chosen}")
    absent = [g for g in chosen if g not in present]
    if absent:
        raise H2Error(f"regimes {absent} do not occur in the usable sample (present: {present})")
    if reference not in chosen:
        raise H2Error(f"reference regime {reference} is not among the regimes {chosen}")
    return chosen


def _decompose(
    x: FloatArray,
    f: FloatArray,
    codes: IntArray,
    regimes: Sequence[int],
    reference: int,
    mask: BoolArray,
    min_rows: int,
) -> FloatArray:
    """Array (n_regimes, len(_FIELDS)) of the per-regime quantities (shared by the point estimate
    and every bootstrap replicate)."""
    k = f.shape[1]
    need = max(min_rows, k + 2)

    def fit(g: int) -> FactorFit:
        rows = codes == g
        n_g = int(rows.sum())
        if n_g < need:
            raise H2Error(f"regime {g} has {n_g} days, need >= {need}")
        return fit_factor_model(x[rows], f[rows])

    ref = fit(reference)
    ref_sys = np.diag(ref.loadings @ ref.factor_cov @ ref.loadings.T).mean()
    ref_idio = np.diag(ref.resid_cov).mean()
    ref_load = ref.loadings[:, 0].mean()
    out = np.empty((len(regimes), len(_FIELDS)))
    for r, g in enumerate(regimes):
        cur = ref if g == reference else fit(g)
        corner = corner_averages(ref, cur, mask)
        shap = shapley_attribution(corner)
        sys_g = np.diag(cur.loadings @ cur.factor_cov @ cur.loadings.T).mean()
        idio_g = np.diag(cur.resid_cov).mean()
        out[r] = (
            corner[(1, 1, 1)],
            corner[(1, 0, 0)],
            shap[0],
            shap[1],
            shap[2],
            sys_g / ref_sys if ref_sys > 0.0 else np.nan,
            idio_g / ref_idio if ref_idio > 0.0 else np.nan,
            cur.loadings[:, 0].mean() / ref_load if ref_load != 0.0 else np.nan,
        )
    return out


# --------------------------------------------------------------------------------------
# Point estimates
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorDecompositionResult:
    """Per-regime table (index = regime code) and the settings it was computed under.

    Columns: ``n_obs``; ``avg_raw`` (the regime's own mean cross-sector correlation);
    ``avg_var_only`` (the volatility-only counterfactual: reference loadings and residual
    covariance, the regime's factor covariance); ``shap_factor_variance``, ``shap_loadings``,
    ``shap_residual_covariance`` (Shapley split of ``avg_raw - avg_raw[reference]``);
    ``sys_var_ratio``, ``idio_var_ratio``, ``loading_ratio`` (against the reference regime).
    """

    table: pd.DataFrame
    alignment: Alignment
    pairs: PairScope
    reference: int
    factor_names: tuple[str, ...]
    n_rows: int
    n_assets: int
    n_pairs: int


def _table(rows: FloatArray, regimes: Sequence[int], n_obs: Sequence[int]) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=list(_FIELDS), index=pd.Index(list(regimes), name="regime"))
    table.insert(0, "n_obs", list(n_obs))
    return table


def regime_factor_decomposition(
    returns: pd.DataFrame,
    labels: pd.Series,
    sectors: Mapping[str, str] | pd.Series,
    factors: pd.DataFrame | pd.Series,
    *,
    alignment: Alignment = "predetermined",
    pairs: PairScope = "cross",
    reference: int = DEFAULT_REFERENCE,
    regimes: Sequence[int] | None = None,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> FactorDecompositionResult:
    """Point estimates of the factor decomposition (module docstring: Method)."""
    sample = _prepare(returns, factors, labels, sectors, alignment, pairs)
    chosen = _check_regimes(sample.codes, regimes, reference)
    rows = _decompose(sample.x, sample.f, sample.codes, chosen, reference, sample.mask, min_rows)
    n_obs = [int((sample.codes == g).sum()) for g in chosen]
    return FactorDecompositionResult(
        table=_table(rows, chosen, n_obs),
        alignment=alignment,
        pairs=pairs,
        reference=reference,
        factor_names=tuple(sample.factor_names),
        n_rows=len(sample.index),
        n_assets=sample.x.shape[1],
        n_pairs=int(sample.mask.sum()),
    )


# --------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorBootstrapResult:
    """The decomposition with percentile bootstrap intervals.

    Statistic names: ``avg_raw`` (every regime); for each non-reference regime ``diff_raw``,
    ``var_only`` (volatility-only counterfactual minus the reference average), ``var_only_share``
    (``var_only / diff_raw``; interval ``NaN`` unless the interval of ``diff_raw`` excludes zero),
    ``shap_factor_variance``, ``shap_loadings``, ``shap_residual_covariance`` (sum to
    ``diff_raw``), ``sys_var_ratio``, ``idio_var_ratio`` and ``loading_ratio``.
    """

    point: FactorDecompositionResult
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


def _stat(
    name: str, regime: int, point: float, reps: FloatArray, alpha: float, *, nan_safe: bool = False
) -> H2Stat:
    if nan_safe:
        lower = float(np.nanquantile(reps, alpha / 2.0))
        upper = float(np.nanquantile(reps, 1.0 - alpha / 2.0))
    else:
        lower = float(np.quantile(reps, alpha / 2.0))
        upper = float(np.quantile(reps, 1.0 - alpha / 2.0))
    return H2Stat(name, regime, point, lower, upper, reps)


def bootstrap_regime_factor(
    returns: pd.DataFrame,
    labels: pd.Series,
    sectors: Mapping[str, str] | pd.Series,
    factors: pd.DataFrame | pd.Series,
    *,
    alignment: Alignment = "predetermined",
    pairs: PairScope = "cross",
    reference: int = DEFAULT_REFERENCE,
    regimes: Sequence[int] | None = None,
    min_rows: int = DEFAULT_MIN_ROWS,
    mean_block_length: float | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> FactorBootstrapResult:
    """The decomposition with stationary-bootstrap percentile intervals.

    One index vector resamples the usable days and is applied to the returns, the factors and the
    aligned regime codes together (labels stay attached to their returns; the factors stay attached
    to their dates). A replicate in which a regime has too few days, or a singular factor
    covariance, raises. Identical inputs and ``seed`` give identical intervals.
    """
    if n_bootstrap < 2:
        raise H2Error(f"n_bootstrap must be >= 2, got {n_bootstrap}")
    if not 0.0 < alpha < 1.0:
        raise H2Error(f"alpha must lie in (0, 1), got {alpha}")
    sample = _prepare(returns, factors, labels, sectors, alignment, pairs)
    chosen = _check_regimes(sample.codes, regimes, reference)
    point_rows = _decompose(
        sample.x, sample.f, sample.codes, chosen, reference, sample.mask, min_rows
    )
    n_obs = [int((sample.codes == g).sum()) for g in chosen]
    point = FactorDecompositionResult(
        table=_table(point_rows, chosen, n_obs),
        alignment=alignment,
        pairs=pairs,
        reference=reference,
        factor_names=tuple(sample.factor_names),
        n_rows=len(sample.index),
        n_assets=sample.x.shape[1],
        n_pairs=int(sample.mask.sum()),
    )
    n_rows = len(sample.index)
    mbl = default_mean_block_length(n_rows) if mean_block_length is None else mean_block_length

    rng = np.random.default_rng(seed)
    reps = np.empty((n_bootstrap, len(chosen), len(_FIELDS)))
    for b in range(n_bootstrap):
        idx = stationary_bootstrap_indices(n_rows, mbl, rng)
        reps[b] = _decompose(
            sample.x[idx],
            sample.f[idx],
            sample.codes[idx],
            chosen,
            reference,
            sample.mask,
            min_rows,
        )

    col = {name: j for j, name in enumerate(_FIELDS)}
    ref_pos = chosen.index(reference)
    stats: list[H2Stat] = []
    for k, g in enumerate(chosen):
        stats.append(
            _stat(
                "avg_raw",
                g,
                float(point_rows[k, col["avg_raw"]]),
                reps[:, k, col["avg_raw"]],
                alpha,
            )
        )
    for k, g in enumerate(chosen):
        if g == reference:
            continue
        raw_ref_pt = float(point_rows[ref_pos, col["avg_raw"]])
        d_raw_rep = reps[:, k, col["avg_raw"]] - reps[:, ref_pos, col["avg_raw"]]
        d_raw_pt = float(point_rows[k, col["avg_raw"]]) - raw_ref_pt
        diff_raw = _stat("diff_raw", g, d_raw_pt, d_raw_rep, alpha)
        stats.append(diff_raw)

        var_rep = reps[:, k, col["avg_var_only"]] - reps[:, ref_pos, col["avg_raw"]]
        var_pt = float(point_rows[k, col["avg_var_only"]]) - raw_ref_pt
        stats.append(_stat("var_only", g, var_pt, var_rep, alpha))

        share_pt = var_pt / d_raw_pt if abs(d_raw_pt) > _SHARE_EPS else float("nan")
        usable = np.abs(d_raw_rep) > _SHARE_EPS
        share_rep = np.full_like(d_raw_rep, np.nan)
        share_rep[usable] = var_rep[usable] / d_raw_rep[usable]
        if diff_raw.excludes_zero and usable.any():
            stats.append(_stat("var_only_share", g, share_pt, share_rep, alpha, nan_safe=True))
        else:
            stats.append(
                H2Stat("var_only_share", g, share_pt, float("nan"), float("nan"), share_rep)
            )

        for name in (
            "shap_factor_variance",
            "shap_loadings",
            "shap_residual_covariance",
            "sys_var_ratio",
            "idio_var_ratio",
            "loading_ratio",
        ):
            stats.append(
                _stat(name, g, float(point_rows[k, col[name]]), reps[:, k, col[name]], alpha)
            )
    return FactorBootstrapResult(
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


def factor_summary_lines(
    result: FactorBootstrapResult, regime_names: Sequence[str] | None = None
) -> list[str]:
    """Human-readable report lines (settings, per-regime values, contrasts, caveat)."""
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

    def fmt(s: H2Stat, digits: int = 4, centre: float = 0.0) -> str:
        if np.isnan(s.lower):
            return f"{s.point:+.{digits}f} [CI not defined]"
        star = "*" if (s.lower > centre or s.upper < centre) else ""
        return f"{s.point:+.{digits}f} [{s.lower:+.{digits}f}, {s.upper:+.{digits}f}]{star}"

    lines = [
        f"Factor decomposition (exploratory, in-sample, descriptive): factors="
        f"{'+'.join(pt.factor_names)}, alignment={pt.alignment}, pairs={pt.pairs}, "
        f"reference={label[pt.reference]}",
        f"  T={pt.n_rows} days, N={pt.n_assets} tickers, {pt.n_pairs} pairs in scope, "
        f"{result.n_bootstrap} stationary-bootstrap draws, mean block length "
        f"{result.mean_block_length:.1f}, {level}% percentile intervals, seed {result.seed}",
    ]
    n_obs = dict(zip(codes, pt.table["n_obs"].to_numpy(dtype=int).tolist(), strict=True))
    for g in codes:
        lines.append(f"  {label[g]:>5}: n={n_obs[g]} days; raw {fmt(result.stat('avg_raw', g))}")
    for g in codes:
        if g == pt.reference:
            continue
        lines.append(f"  {label[g]} minus {label[pt.reference]}:")
        lines.append(
            f"    raw change {fmt(result.stat('diff_raw', g))}; volatility-only counterfactual "
            f"{fmt(result.stat('var_only', g))}; share {fmt(result.stat('var_only_share', g))}"
        )
        lines.append(
            "    Shapley split of the raw change: "
            f"factor variance {fmt(result.stat('shap_factor_variance', g))}; "
            f"loadings {fmt(result.stat('shap_loadings', g))}; "
            f"residual covariance {fmt(result.stat('shap_residual_covariance', g))}"
        )
        lines.append(
            "    ratios vs reference: systematic variance "
            f"{fmt(result.stat('sys_var_ratio', g), 3, 1.0)}; residual variance "
            f"{fmt(result.stat('idio_var_ratio', g), 3, 1.0)}; first-factor loading "
            f"{fmt(result.stat('loading_ratio', g), 3, 1.0)}"
        )
    lines.append(
        "  * = interval excludes 0 (for the three ratios: excludes 1); uncorrected for multiplicity"
    )
    lines.append(
        "  Exploratory, not part of the pre-registered H2. Variance of any common factor the "
        "chosen factors do not span is booked as a residual-covariance change; the share's bias "
        "has no known sign (see docs/checks/4.2b.md)"
    )
    return lines
