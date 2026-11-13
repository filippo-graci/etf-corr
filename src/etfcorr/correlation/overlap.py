"""Step 2.5: mechanical vs economic co-movement -- regresses return correlation on holdings
overlap and reports the residual as the "economic" part.

Question served: this project's own standard (custom instructions section 2) states plainly
that "holdings overlap between ETFs creates mechanical correlation; separate it from economic
co-movement." ``data/holdings.py`` (step 2.4, LOCAL-PASSED) supplies one half of that
decomposition -- a point-in-time, symmetric weighted-Jaccard :func:`~etfcorr.data.holdings.
overlap_matrix` across tickers with a usable SEC N-PORT holdings snapshot -- and its own module
docstring names this file as the consumer: "so that ``correlation/overlap.py`` (step 2.5) can
regress return correlation on holdings overlap and report the residual as the 'economic' part."
``project_plan.md`` section 6.4 describes the same file identically: "Joins ``overlap_matrix``
with return correlation; regression of correlation on overlap; residual correlation as the
'economic' part." This module is exactly that join-and-regress step; nothing here is asserted as
a confirmatory result for H1-H6 (``config/experiment.yaml``) -- it is diagnostic/exploratory
infrastructure that supports interpreting H1 (market-mode removal) and any later network analysis,
labelled **[Exploratory]** throughout, per the project's own rule that anything outside the six
pre-registered hypotheses is exploratory.

**Why this is exploratory, not confirmatory.** Da, Z., Shive, S. (2018), *Exchange traded funds
and asset return correlations*, European Financial Management 24(1), 136-168 (CHECKED,
``REFERENCES_SEED.md``, DOI 10.1111/eufm.12137) and Ben-David, I., Franzoni, F., Moussawi, R.
(2018), *Do ETFs increase volatility?*, Journal of Finance 73(6), 2471-2535 (CHECKED,
``REFERENCES_SEED.md``, DOI 10.1111/jofi.12727) both show, in their own setting -- individual
*stocks*, a large cross-sectional panel, ETF *ownership*/*trading* as the mechanism -- that ETF
structure raises comovement beyond fundamentals. Neither paper regresses *ETF-pair* return
correlation on *portfolio-overlap* similarity the way this module does; that this project's
~30-40-ETF, N(N-1)/2-pair setting shows the same qualitative pattern is a plausible **[Inference]**
motivated by those findings, not a result either paper itself reports, and must be checked here
with this project's own data, not assumed by analogy.

**The dyadic-independence problem, stated before any inference is run.** A correlation matrix
and an overlap matrix each have N(N-1)/2 off-diagonal pairs, but those pairs are **not**
independent observations: ticker A appears in N-1 different pairs, so any pair sharing a ticker
shares that ticker's idiosyncratic contribution to both matrices -- a classic dyadic-dependence
structure. Ordinary least squares assumes independent errors; its reported standard errors on a
flattened N(N-1)/2-row "dataset" of pairs are therefore invalid here (understated, typically
severely so, because the *effective* number of independent units is N, not N(N-1)/2) -- this is
not a hypothetical concern, it is demonstrated concretely in ``tests/leakage/
test_overlap_purity.py`` (see below). The standard remedy for regressions on dyadic/network data
is a permutation test that reshuffles *node* labels jointly (holding the observed value at every
pair fixed, only relabelling which node owns which row/column) rather than treating pairs as
freely exchangeable: Mantel, N. (1967), *The Detection of Disease Clustering and a Generalized
Regression Approach*, Cancer Research 27(2 Part 1), 209-220 (CHECKED via search this session:
title/venue/volume/pages match, https://aacrjournals.org/cancerres/article/27/2_Part_1/209/476508)
introduced the permutation test for the association between two such matrices; Krackhardt, D.
(1988), *Predicting with networks: Nonparametric multiple regression analysis of dyadic data*,
Social Networks 10(4), 359-381 (CHECKED via search this session: title/author/journal/volume/
issue/pages confirmed via the publisher's own citation metadata) extended it to a regression
setting under the name Quadratic Assignment Procedure (QAP/MRQAP). :func:`qap_permutation_test`
implements this directly (node-label permutation, not independent-pair shuffling); it is this
module's *primary* inference tool. The naive OLS standard error/t-statistic
(:class:`OverlapRegressionResult`'s ``naive_se_slope``/``naive_t_stat``) is still reported, but
labelled invalid in every summary line, for reference and comparison against the QAP p-value only
-- never as the basis for a significance claim on its own.

**Library choice, stated on purpose (mirrors ``correlation/estimators.py``'s own precedent).**
``project_plan.md`` names no specific library for this file. This session checked for an existing
QAP/MRQAP implementation before writing one (per the project's own workflow step 3, "check
whether a maintained library already implements it"): none of this project's pinned dependencies
(``pyproject.toml``) provide it -- QAP/MRQAP implementations exist in R (the ``sna``/``statnet``
package family's ``netlm``), not in any Python package this project depends on -- so
:func:`qap_permutation_test` is this project's own code (not adapted from anywhere; nothing for
``THIRD_PARTY.md``). The point-estimate regression itself (a single-predictor OLS) uses
``sklearn.linear_model.LinearRegression`` (scikit-learn is already a core pinned dependency, used
directly rather than re-implemented, per the same workflow step); the association statistics use
``numpy.corrcoef`` (Pearson) and ``scipy.stats.spearmanr`` (Spearman, ``scipy`` also a core pinned
dependency) rather than ``statsmodels`` -- ``statsmodels`` is a pinned dependency
(``pyproject.toml``) but, like ``skfolio``/``arch``/``torch`` before it, is **not importable in
this planning sandbox** (confirmed this session: ``ModuleNotFoundError``, PyPI unreachable here,
the same recurring gap ``docs/STATUS.md`` has recorded at every earlier step). Because this
module's own inference is the *hand-derived* permutation test, not a ``statsmodels`` call, nothing
here needed to be isolated behind an injectable seam or marked NOT EXECUTED OFFLINE -- unlike
several of steps 1.1-1.3/2.4's I/O-touching functions, every line in this module is executable and
tested in this sandbox, matching ``estimators.py``'s/``horizon_analysis.py``'s own "pure
numpy/pandas library" precedent (extended here to ``scipy``/``scikit-learn``, both already
pinned).

**Scope and contract, mirroring ``estimators.py``'s/``horizon_analysis.py``'s own stated
precedent.** This module takes two already-computed, ticker-indexed matrices -- an
:func:`~etfcorr.data.holdings.overlap_matrix` output and a
:func:`~etfcorr.correlation.estimators.estimate`/:func:`~etfcorr.correlation.estimators.corr_report`
output -- and is a pure function of exactly those two inputs; it does no I/O, and does not import
``etfcorr.config``/``etfcorr.data.*`` beyond re-using ``holdings.validate_overlap_matrix`` and
``estimators.validate_correlation_matrix`` for each matrix's own structural checks (reused, not
reimplemented, matching this project's own repeated "what counts as valid lives in exactly one
place" pattern). Ensuring the two inputs are *themselves* point-in-time consistent -- that the
holdings snapshot's ``filing_date`` and the correlation matrix's estimation window are drawn from
a mutually sensible period -- is the caller's job (via ``holdings.latest_asof`` and the
correlation modules' own truncation discipline), not this module's; a single holdings snapshot
currently exists per ticker (one ``--asof`` fetch, step 2.4), so in practice this module currently
treats that one snapshot as representative of whatever correlation-estimation window it is joined
against -- a stated approximation, not yet checked for stability across time, disclosed rather
than silently assumed. No CLI command is added, following ``estimators.py``'s/
``horizon_analysis.py``'s own "pure library, not a per-step CLI precedent" pattern for this
package.

**What "residual correlation" means here.** :func:`fit_overlap_regression`'s
``residual_matrix`` is ``correlation - (intercept + slope * overlap)`` for every off-diagonal
pair, reshaped back into a labelled, symmetric matrix with the diagonal set to ``nan`` (self-pairs
are not part of the regression, the same exclusion :func:`~etfcorr.correlation.horizon_analysis.
average_offdiagonal_corr` already applies). Like ``holdings.py``'s own ``overlap_matrix``, this
matrix makes **no PSD or unit-diagonal claim** -- it is used descriptively (a diagnostic display
object, e.g. for a heatmap of "correlation left over after removing the overlap-explained part"),
never as a covariance/correlation input to anything downstream that would need those properties.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from etfcorr.config import Horizon
from etfcorr.correlation.estimators import CorrelationReport, validate_correlation_matrix
from etfcorr.data.holdings import validate_overlap_matrix

DEFAULT_N_PERMUTATIONS = 2000  # matches this project's own stationary-bootstrap default (step 0.3
# decision log: "stationary bootstrap with 2000 draws"); borrowed for consistency, not
# independently derived for this test -- [Inference].

MIN_TICKERS = 4  # -> 6 pairs, a practical floor for a permutation test to say anything at all;
# [Inference], not a proven minimum sample size for QAP inference specifically.

Statistic = Literal["pearson", "spearman"]
STATISTICS: tuple[Statistic, ...] = ("pearson", "spearman")


class OverlapAnalysisError(ValueError):
    """Invalid input to, or an invalid configuration of, an overlap-vs-correlation call."""


# --------------------------------------------------------------------------------------
# Alignment: two ticker-indexed matrices -> one common, sorted ticker set
# --------------------------------------------------------------------------------------


def _align(
    overlap: pd.DataFrame, corr: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Validate each matrix on its own terms (reusing each module's own validator), then restrict
    both to their common, sorted ticker set. Returns ``(overlap_aligned, corr_aligned,
    dropped_from_overlap, dropped_from_corr)`` -- the two drop lists are reported, not silently
    discarded, since step 2.4's real run covers only 42 of 56 tickers (holdings.py's own disclosed
    fetch failures) while a correlation matrix may cover a different, typically larger, set.
    """
    validate_overlap_matrix(overlap)
    validate_correlation_matrix(corr)
    overlap_tickers = set(overlap.index)
    corr_tickers = set(corr.index)
    common = sorted(overlap_tickers & corr_tickers)
    if len(common) < MIN_TICKERS:
        raise OverlapAnalysisError(
            f"only {len(common)} tickers are in both the overlap matrix and the correlation "
            f"matrix; need at least {MIN_TICKERS} (see MIN_TICKERS)"
        )
    dropped_from_overlap = tuple(sorted(overlap_tickers - corr_tickers))
    dropped_from_corr = tuple(sorted(corr_tickers - overlap_tickers))
    return (
        overlap.loc[common, common],
        corr.loc[common, common],
        dropped_from_overlap,
        dropped_from_corr,
    )


def _pair_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def pairs_frame(overlap: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
    """One row per unordered ticker pair in the common, aligned ticker set: ``ticker_i``,
    ``ticker_j``, ``overlap``, ``correlation``. Each pair appears exactly once (the strict upper
    triangle), not twice by symmetry. Ready for direct inspection or a scatter plot in a notebook,
    the same "ready for direct inspection" role ``estimators.CorrelationReport``/
    ``horizon_analysis.corr_by_horizon`` already play for their own outputs.
    """
    ov, cr, _, _ = _align(overlap, corr)
    tickers = list(ov.index)
    iu = _pair_indices(len(tickers))
    ov_values = ov.to_numpy(dtype=float)
    cr_values = cr.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "ticker_i": [tickers[i] for i in iu[0]],
            "ticker_j": [tickers[j] for j in iu[1]],
            "overlap": ov_values[iu],
            "correlation": cr_values[iu],
        }
    )


# --------------------------------------------------------------------------------------
# Association statistics (Pearson + Spearman, the project's own "robustness check" pairing)
# --------------------------------------------------------------------------------------


def _association(x: np.ndarray, y: np.ndarray, statistic: Statistic) -> float:
    if statistic == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    if statistic == "spearman":
        rho, _ = stats.spearmanr(x, y)
        return float(rho)
    raise OverlapAnalysisError(f"unknown statistic {statistic!r}; expected one of {STATISTICS}")


# --------------------------------------------------------------------------------------
# QAP / Mantel permutation test: the primary, dependence-robust inference tool
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QAPResult:
    """One permutation test's result. ``p_value`` is two-sided: the fraction of permutations whose
    |statistic| is at least as large as the observed |statistic|, with the standard "+1" smoothing
    (Davison & Hinkley's convention: a permutation test can never report an exact zero p-value)."""

    statistic: Statistic
    observed: float
    p_value: float
    n_permutations: int
    null_mean: float
    null_std: float


def qap_permutation_test(
    overlap: pd.DataFrame,
    corr: pd.DataFrame,
    *,
    statistic: Statistic = "pearson",
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 0,
) -> QAPResult:
    """Mantel (1967) / Krackhardt (1988) QAP permutation test for the association between the two
    matrices' off-diagonal pairs (see the module docstring for why this, not a flat/naive
    permutation of the pair vector, is the correct null here).

    Procedure: compute the observed ``statistic`` between the two matrices' aligned off-diagonal
    pair vectors; then, ``n_permutations`` times, apply one random permutation of the *node*
    (ticker) labels to the correlation matrix -- i.e. simultaneously reorder its rows and columns
    by the same permutation, which preserves every pairwise value and every ticker's row of
    dyadic dependence, only relabelling which ticker owns which row/column -- and recompute the
    same statistic between the (fixed) overlap pair vector and the newly relabelled correlation
    pair vector. This draws from the symmetric group on the N *nodes* (N! possible relabellings),
    not on the N(N-1)/2 *pairs* (whose own symmetric group is vastly larger) -- the effective
    sample size behind this null is N, not N(N-1)/2, demonstrated as a deterministic combinatorial
    fact (not an empirical, data-dependent one) in ``tests/leakage/test_overlap_purity.py``.

    Raises :class:`OverlapAnalysisError` if ``n_permutations < 2`` (``null_std`` is undefined from
    a single draw) or if the overlap pair vector has zero variance (association is undefined).
    """
    if n_permutations < 2:
        raise OverlapAnalysisError(f"n_permutations must be >= 2, got {n_permutations}")
    ov, cr, _, _ = _align(overlap, corr)
    n = len(ov)
    iu = _pair_indices(n)
    ov_values = ov.to_numpy(dtype=float)[iu]
    cr_matrix = cr.to_numpy(dtype=float)
    cr_values = cr_matrix[iu]
    if np.allclose(ov_values, ov_values[0]):
        raise OverlapAnalysisError(
            "overlap has zero variance across pairs; cannot test association"
        )
    observed = _association(ov_values, cr_values, statistic)

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for k in range(n_permutations):
        perm = rng.permutation(n)
        permuted_cr_values = cr_matrix[np.ix_(perm, perm)][iu]
        null[k] = _association(ov_values, permuted_cr_values, statistic)

    p_value = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_permutations + 1))
    return QAPResult(
        statistic=statistic,
        observed=observed,
        p_value=p_value,
        n_permutations=n_permutations,
        null_mean=float(null.mean()),
        null_std=float(null.std(ddof=1)),
    )


# --------------------------------------------------------------------------------------
# OLS regression of correlation on overlap, plus the residual ("economic") matrix
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlapRegressionResult:
    """Everything a skeptical reader will ask about one overlap-vs-correlation regression."""

    tickers: tuple[str, ...]
    n_tickers: int
    n_pairs: int
    dropped_from_overlap: tuple[str, ...]
    dropped_from_corr: tuple[str, ...]
    pairs: pd.DataFrame
    intercept: float
    slope: float
    r_squared: float
    naive_se_slope: float
    naive_t_stat: float
    pearson_r: float
    spearman_rho: float
    qap_pearson: QAPResult
    qap_spearman: QAPResult
    residual_matrix: pd.DataFrame


def _residual_matrix(
    ov: pd.DataFrame, cr: pd.DataFrame, intercept: float, slope: float
) -> pd.DataFrame:
    predicted = intercept + slope * ov.to_numpy(dtype=float)
    residual = cr.to_numpy(dtype=float) - predicted
    np.fill_diagonal(residual, np.nan)  # self-pairs are not part of the regression
    return pd.DataFrame(residual, index=ov.index, columns=ov.columns)


def fit_overlap_regression(
    overlap: pd.DataFrame,
    corr: pd.DataFrame,
    *,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 0,
) -> OverlapRegressionResult:
    """Regress correlation on overlap (OLS point estimate via ``sklearn.linear_model.
    LinearRegression``), run the QAP permutation test (Pearson and Spearman) for dependence-robust
    inference, and build the residual ("economic") correlation matrix.

    The naive OLS standard error/t-statistic (``naive_se_slope``/``naive_t_stat``) is computed by
    the textbook single-predictor formula (``se(beta) = sqrt(sigma^2 / sum((x - xbar)^2))``) purely
    for reference alongside the QAP p-value -- see the module docstring for why it is not a valid
    basis for a significance claim on its own (dyadic dependence among pairs). ``seed`` seeds the
    Pearson QAP test directly and ``seed + 1`` the Spearman one, so the two tests use independent
    (not identical) permutation draws while the whole call stays reproducible from one seed.
    """
    ov, cr, dropped_overlap, dropped_corr = _align(overlap, corr)
    tickers = tuple(ov.index)
    pairs = pairs_frame(ov, cr)
    x = pairs["overlap"].to_numpy(dtype=float)
    y = pairs["correlation"].to_numpy(dtype=float)
    if np.allclose(x, x[0]):
        raise OverlapAnalysisError("overlap has zero variance across pairs; cannot regress")

    model = LinearRegression().fit(x.reshape(-1, 1), y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    y_hat = model.predict(x.reshape(-1, 1))
    residuals = y - y_hat
    n_pairs = len(x)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    dof = n_pairs - 2
    sxx = float(np.sum((x - x.mean()) ** 2))
    if dof > 0 and sxx > 0:
        sigma2 = ss_res / dof
        naive_se_slope = float(np.sqrt(sigma2 / sxx))
        naive_t_stat = slope / naive_se_slope if naive_se_slope > 0 else float("nan")
    else:
        naive_se_slope = float("nan")
        naive_t_stat = float("nan")

    pearson_r = _association(x, y, "pearson")
    spearman_rho = _association(x, y, "spearman")
    qap_pearson = qap_permutation_test(
        ov, cr, statistic="pearson", n_permutations=n_permutations, seed=seed
    )
    qap_spearman = qap_permutation_test(
        ov, cr, statistic="spearman", n_permutations=n_permutations, seed=seed + 1
    )

    return OverlapRegressionResult(
        tickers=tickers,
        n_tickers=len(tickers),
        n_pairs=n_pairs,
        dropped_from_overlap=dropped_overlap,
        dropped_from_corr=dropped_corr,
        pairs=pairs,
        intercept=intercept,
        slope=slope,
        r_squared=r_squared,
        naive_se_slope=naive_se_slope,
        naive_t_stat=naive_t_stat,
        pearson_r=pearson_r,
        spearman_rho=spearman_rho,
        qap_pearson=qap_pearson,
        qap_spearman=qap_spearman,
        residual_matrix=_residual_matrix(ov, cr, intercept, slope),
    )


# --------------------------------------------------------------------------------------
# Per-horizon convenience (robustness across D/W/M/Q), mirroring corr_by_horizon's own style
# --------------------------------------------------------------------------------------


def overlap_regression_by_horizon(
    overlap: pd.DataFrame,
    corr_by_horizon: Mapping[Horizon, CorrelationReport],
    *,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 0,
) -> dict[Horizon, OverlapRegressionResult]:
    """One :func:`fit_overlap_regression` per horizon, against a single (typically point-in-time)
    overlap matrix and ``horizon_analysis.corr_by_horizon``'s own output shape. A robustness check
    (does the mechanical-overlap effect look similar at daily vs quarterly horizons?), not a new
    confirmatory hypothesis; every result here is **[Exploratory]**, matching this module's own
    scope note above. ``corr_by_horizon`` is typically ``horizon_analysis.corr_by_horizon(...)``'s
    return value; only its ``.corr`` field is used here.
    """
    if not corr_by_horizon:
        raise OverlapAnalysisError("corr_by_horizon is empty; nothing to compute")
    return {
        horizon: fit_overlap_regression(
            overlap, report.corr, n_permutations=n_permutations, seed=seed
        )
        for horizon, report in corr_by_horizon.items()
    }


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def summary_lines(result: OverlapRegressionResult) -> list[str]:
    """Human-readable lines: sample size, the OLS fit, and the primary QAP-based inference for
    both association statistics -- the naive t-stat is always labelled invalid, never presented
    bare, so nobody downstream can quote it without also seeing that caveat.
    """
    lines = [
        f"overlap vs correlation: N={result.n_tickers} tickers, {result.n_pairs} pairs",
        f"  OLS: correlation = {result.intercept:.4f} + {result.slope:.4f} * overlap "
        f"(R^2={result.r_squared:.4f})",
        f"  naive OLS t-stat={result.naive_t_stat:.3g} "
        "(INVALID under dyadic pair dependence -- see the QAP tests below, not this)",
        f"  pearson r={result.pearson_r:.4f}  QAP p={result.qap_pearson.p_value:.4f} "
        f"({result.qap_pearson.n_permutations} node-label permutations)",
        f"  spearman rho={result.spearman_rho:.4f}  QAP p={result.qap_spearman.p_value:.4f} "
        f"({result.qap_spearman.n_permutations} node-label permutations)",
    ]
    if result.dropped_from_overlap:
        lines.append(
            f"  in overlap matrix but not correlation matrix (dropped): "
            f"{', '.join(result.dropped_from_overlap)}"
        )
    if result.dropped_from_corr:
        lines.append(
            f"  in correlation matrix but not overlap matrix (dropped): "
            f"{', '.join(result.dropped_from_corr)}"
        )
    return lines
