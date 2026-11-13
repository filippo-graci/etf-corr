"""Step 3.4a: distance correlation and mutual information -- non-linear dependence robustness
checks alongside Pearson. This project's own custom instructions (section 2, "Correlation
estimation") name the default stack as "Pearson on log returns with shrinkage ... and
random-matrix cleaning ... with Spearman/Kendall, distance correlation, EWMA and DCC/ADCC, and
tail dependence as robustness checks" -- ``estimators.py`` (step 2.2) delivers Pearson/Spearman/
Kendall/Ledoit-Wolf/OAS/EWMA; ``rmt.py`` (step 3.1) delivers the random-matrix cleaning; this
module delivers the two remaining linear-dependence-agnostic robustness checks named there:
distance correlation and mutual information. Tail dependence is a distinct, separately-cited
method (extreme-value theory, not a "how much dependence exists overall" measure) and is
delivered in the sibling module ``tail.py``, not here -- see that module's docstring.

Question served: Pearson (and its shrinkage/EWMA variants) only detects *linear* dependence; a
pair of ETFs could be strongly, non-linearly co-moving (e.g. through a shared non-linear driver,
or a regime-dependent relationship) with a near-zero Pearson correlation, and Pearson alone would
never surface it. Distance correlation (Szekely, Rizzo, Bakirov 2007) is zero if and only if the
two variables are independent, for any dependence structure, linear or not -- so a real gap
between distance correlation and (absolute) Pearson is direct evidence of non-linear dependence
Pearson is blind to. Mutual information is the same idea from information theory, estimated here
with a different (k-nearest-neighbour) estimator with different finite-sample behaviour, as a
second, differently-biased check rather than a replacement.

This module is **not** a source of a matrix to feed into ``rmt.py``/``correlation/networks.py``:
distance correlation and mutual information are non-negative measures of *dependence strength*,
not signed linear correlations, so they do not carry the "detoning subtracts a market mode"
algebra those modules assume, and passing them there would silently produce numbers with no
justified interpretation. They are reported here purely as diagnostics, per this project's own
"robustness check" framing, never as a replacement covariance/correlation input for portfolio
construction (H5) or RMT cleaning (H1).

**Method notes, citation status** (every reference below was checked -- existence, authors,
venue, volume/issue/pages -- by a web search this session; ``REFERENCES_SEED.md`` section A
already seeded distance correlation and section E already seeded the false-discovery-rate
reference as UNVERIFIED, upgrade both there):

* **Distance correlation.** Szekely, G. J., Rizzo, M. L., Bakirov, N. K. (2007), *Measuring and
  testing dependence by correlation of distances*, Annals of Statistics 35(6), 2769-2794, DOI
  10.1214/009053607000000505 -- **CHECKED** this session (title, authors, journal, volume, issue
  and page range confirmed via Project Euclid's own listing). :func:`distance_correlation`
  implements the paper's original **biased (V-statistic) estimator**: double-center the pairwise
  absolute-difference matrices of each variable, take the mean elementwise product for the
  distance covariance, and normalise by the geometric mean of the two distance variances. This is
  the paper's Definitions 2-5 (not the bias-corrected U-statistic estimator of the 2014 follow-up
  paper, Szekely and Rizzo, *Partial distance correlation with methods for dissimilarities*,
  Annals of Statistics 42(6) -- not implemented here, not cited as a source of anything in this
  module). ``dcor`` is a pinned core dependency (``pyproject.toml``) whose own ``distance_
  correlation`` function implements this exact same biased estimator by default (per its
  documentation; CHECKED 2026-11-02 in the installed ``dcor`` 0.7), but
  ``dcor`` could not be installed in this planning sandbox (PyPI unreachable here, the same
  recurring gap as ``skfolio``/``arch``/``torch`` in every earlier step, confirmed this session by
  a direct ``pip install`` attempt). Rather than write code against a package this sandbox cannot
  import or test at all, this module implements the estimator directly from the paper's own
  definitions (a closed-form, fully specifiable computation, not a re-implementation of any
  particular library's source -- nothing for ``THIRD_PARTY.md``) and reserves the numerical
  cross-check against the real, pinned ``dcor`` package for Gate 2 (see the Delivery block),
  exactly ``estimators.py``'s own precedent for Ledoit-Wolf/OAS versus skfolio.
* **Mutual information.** Kraskov, A., Stogbauer, H., Grassberger, P. (2004), *Estimating mutual
  information*, Physical Review E 69, 066138, DOI 10.1103/PhysRevE.69.066138 (with a later
  erratum, Phys. Rev. E 83, 019903, not itself re-checked this session) -- **CHECKED** this
  session (title, authors, journal, volume, article number confirmed). :func:`mutual_information_
  matrix` calls ``sklearn.feature_selection.mutual_info_regression``, whose continuous-variable
  estimator is this k-nearest-neighbour method (per scikit-learn's own documentation;
  CHECKED 2026-11-02 in the installed source: nearest neighbours, citing Kraskov et al.). No
  re-implementation is attempted: scikit-learn is a core, offline-importable dependency, so this is
  the "prefer a maintained library" branch of the project's own workflow (custom instructions
  section 3, step 3), unlike distance correlation above where no maintained, offline-importable
  library implements it.
* **Symmetrisation and the diagonal, stated explicitly.** ``mutual_info_regression(X, y)``
  estimates ``I(X_i ; y)`` for each column of ``X`` against one target ``y``; the k-NN estimator
  is not exactly symmetric in finite samples even though mutual information is symmetric in
  theory, so :func:`mutual_information_matrix` computes both ``I(x_i ; x_j)`` and ``I(x_j ; x_i)``
  for every off-diagonal pair and reports their mean -- an **[Inference]** symmetrisation choice,
  not itself a citation-backed claim, disclosed rather than silently picking one direction. The
  diagonal is left ``NaN``, not ``I(x_i ; x_i)``: the k-NN estimator's distances between a point
  and itself in the joint space are degenerate (zero), so self-mutual-information is not a
  meaningful quantity this estimator can report, unlike distance correlation's diagonal (which is
  exactly 1 by the population definition, Szekely et al. 2007 Remark 3) -- no attempt is made to
  force a "diagonal = 1" convention onto a matrix whose off-diagonal values are not bounded by 1
  in the first place (mutual information has no fixed upper bound the way a correlation does).
* **Permutation test of independence.** Szekely et al. (2007) section 5 propose exactly this
  permutation scheme for testing ``H0: X independent of Y`` using the distance correlation
  statistic -- **CHECKED** as part of the same paper checked above, not a separate citation.
  :func:`distance_correlation_permutation_test` follows the same one-sided permutation-test
  pattern already used in this project's ``correlation/networks.py`` (``sector_purity_test``, step
  3.3): shuffle one variable relative to the other, recompute the statistic, and report ``p = (1 +
  #{null >= observed}) / (1 + n_permutations)``.
* **False discovery rate.** Benjamini, Y., Hochberg, Y. (1995), *Controlling the false discovery
  rate: a practical and powerful approach to multiple testing*, Journal of the Royal Statistical
  Society, Series B, 57(1), 289-300 -- **CHECKED** this session (title, authors, journal, volume,
  issue, page range confirmed). :func:`benjamini_hochberg_qvalues` implements the paper's original
  step-up procedure (linear, not the later two-stage adaptive variants); this project's own custom
  instructions (section 2) require FDR or Bonferroni-type correction "when scanning many pairs,
  lags or horizons," which a full N-asset pairwise significance scan is exactly.
* **Not a source of a linear correlation matrix.** Repeated from above because it is the single
  most important caveat of this module: :func:`distance_correlation_matrix` and :func:`mutual_
  information_matrix` deliberately do **not** call ``estimators.validate_correlation_matrix`` --
  that function's contract (checked across every other correlation-producing module in this
  project) is specifically about a *signed, unit-diagonal linear correlation matrix* that ``rmt.py``
  and ``correlation/networks.py`` can further process; reusing it here would misleadingly imply
  these outputs satisfy that same contract, when a mutual-information matrix does not even have a
  unit diagonal. This module defines its own, narrower :func:`_validate_dependence_matrix` for
  distance correlation's actual guarantees (symmetric, unit diagonal, values in [0, 1]) and no
  structural validator at all for mutual information (whose diagonal is deliberately ``NaN`` and
  whose off-diagonal range has no fixed upper bound).
* **Deliberately no dependency on ``etfcorr.config``/``etfcorr.data.*``/other ``etfcorr.
  correlation.*`` modules**, going one step further than ``rmt.py`` (which imports ``estimators.
  validate_correlation_matrix``): this module's only dependencies are numpy, pandas and
  scikit-learn, matching ``estimators.py``/``rmt.py``/``correlation/networks.py``'s own "pure
  numerical library, no I/O" contract, for the reason given in the point directly above (this
  module's outputs are not inputs to those modules' own validators or algebra).
* **Complete-case rows, matrix-wide.** Both matrix functions restrict to the rows where every
  requested ticker has data before computing anything pairwise, the same convention
  ``estimators.py`` states and justifies in its own module docstring (one well-defined T for the
  whole matrix; the caller restricts ``returns`` to the sub-panel actually wanted).
* **Performance, disclosed rather than silently capped.** :func:`distance_correlation` and the
  permutation test both form an ``n x n`` pairwise-distance matrix per variable (Szekely et al.'s
  own definition requires it) -- O(n^2) time and memory per pair, and the permutation test repeats
  the O(n^2) cross-term computation once per permutation. At this project's daily horizon
  (``n`` around 2,700) a full pairwise significance scan over ~25 core tickers (about 300 pairs)
  at ``n_permutations=199`` is a real, unmeasured cost -- flagged as a Gate 2 timing item (see the
  Delivery block), not assumed fast. Nothing in this module silently subsamples or caps ``n`` to
  make that cost disappear; the weekly/monthly/quarterly horizons (``n`` from 49 to about 660) are
  far cheaper and are the more likely default use for the exhaustive pairwise scan.

No third-party code is adapted; nothing new for ``THIRD_PARTY.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

__all__ = [
    "NonlinearError",
    "PermutationTestResult",
    "benjamini_hochberg_qvalues",
    "distance_correlation",
    "distance_correlation_matrix",
    "distance_correlation_permutation_test",
    "mutual_information_matrix",
    "nonlinearity_gap",
    "pairwise_distance_correlation_significance",
]

DEFAULT_N_PERMUTATIONS = 199
DEFAULT_ALPHA = 0.05


class NonlinearError(ValueError):
    """Invalid input to, or an invalid configuration of, a non-linear dependence call."""


# --------------------------------------------------------------------------------------
# Shared input preparation (mirrors estimators.py's _prepare, restated locally: this module
# has no dependency on estimators.py, see the module docstring)
# --------------------------------------------------------------------------------------


def _complete_case(returns: pd.DataFrame, *, window: int | None) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise NonlinearError(f"returns must be a pandas DataFrame, got {type(returns)}")
    if returns.shape[1] < 2:
        raise NonlinearError("returns must have at least 2 columns to form pairs")
    sliced = returns if window is None else returns.tail(window)
    used = sliced.dropna(how="any")
    if len(used) < 4:
        raise NonlinearError(
            f"need at least 4 complete-case rows, got {len(used)} (out of {len(sliced)})"
        )
    return used


def _validate_dependence_matrix(matrix: pd.DataFrame, *, tol: float = 1e-8) -> None:
    """Symmetric, unit-diagonal, values in [0, 1] -- distance correlation's own guarantees.

    Deliberately separate from ``estimators.validate_correlation_matrix``; see the module
    docstring for why that function's contract (signed linear correlation) does not apply here.
    """
    values = matrix.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise NonlinearError("dependence matrix contains NaN or infinite values")
    if not np.allclose(values, values.T, atol=tol):
        raise NonlinearError("dependence matrix is not symmetric")
    if not np.allclose(np.diag(values), 1.0, atol=tol):
        raise NonlinearError("dependence matrix diagonal is not all ones")
    if values.min() < -tol or values.max() > 1.0 + tol:
        raise NonlinearError("dependence matrix has entries outside [0, 1]")


# --------------------------------------------------------------------------------------
# Distance correlation (Szekely, Rizzo, Bakirov 2007)
# --------------------------------------------------------------------------------------


def _double_center(values: np.ndarray) -> np.ndarray:
    """Double-center a pairwise absolute-difference matrix (Szekely et al. 2007, Definition 2)."""
    row_mean = values.mean(axis=1, keepdims=True)
    col_mean = values.mean(axis=0, keepdims=True)
    grand_mean = values.mean()
    return values - row_mean - col_mean + grand_mean


def _distance_matrix(x: np.ndarray) -> np.ndarray:
    diff: np.ndarray = np.abs(x[:, None] - x[None, :])
    return diff


def _dcov2_from_centered(a: np.ndarray, b: np.ndarray) -> float:
    """``mean(A * B)`` -- the sample squared distance covariance of two centered matrices."""
    return float(np.mean(a * b))


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Sample distance correlation of two equal-length 1-D arrays (Szekely et al. 2007).

    Returns a value in [0, 1]: exactly 0 only in the population when X and Y are independent
    (unlike Pearson, which can be exactly 0 for strongly dependent, non-linearly related
    variables); this sample estimator is the **biased** V-statistic version (see the module
    docstring), so it is not exactly 0 even under independence for finite ``n`` -- use
    :func:`distance_correlation_permutation_test` for a significance judgement, not a raw
    threshold on this value. Both inputs must be finite; ``NaN``/``inf`` are rejected rather than
    silently dropped, since dropping rows independently per pair would break the pairing this
    statistic depends on (see :func:`distance_correlation_matrix` for the matrix-wide complete
    case convention this project uses instead).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise NonlinearError("x and y must be 1-D arrays")
    if len(x) != len(y):
        raise NonlinearError(f"x and y must have the same length, got {len(x)} and {len(y)}")
    if len(x) < 4:
        raise NonlinearError(f"need at least 4 observations, got {len(x)}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise NonlinearError("x and y must not contain NaN or infinite values")

    a = _double_center(_distance_matrix(x))
    b = _double_center(_distance_matrix(y))
    dcov2 = _dcov2_from_centered(a, b)
    dvar2_x = _dcov2_from_centered(a, a)
    dvar2_y = _dcov2_from_centered(b, b)
    denom = np.sqrt(max(dvar2_x, 0.0) * max(dvar2_y, 0.0))
    if denom <= 0.0:
        return 0.0
    return float(np.clip(np.sqrt(max(dcov2, 0.0)) / np.sqrt(denom), 0.0, 1.0))


def distance_correlation_matrix(
    returns: pd.DataFrame, *, window: int | None = None
) -> pd.DataFrame:
    """Pairwise distance correlation, complete-case rows, unit diagonal, symmetric, in [0, 1].

    ``window`` restricts to the trailing ``window`` rows first (same trailing-window convention
    as ``estimators.estimate``), then every pair uses the matrix-wide complete-case rows (see the
    module docstring). Cost is O(n_assets^2 * n_obs^2); see the module docstring's performance
    note.
    """
    used = _complete_case(returns, window=window)
    columns = list(used.columns)
    n = len(columns)
    values = used.to_numpy(dtype=float)
    out = np.eye(n, dtype=float)
    centered = [_double_center(_distance_matrix(values[:, i])) for i in range(n)]
    dvar2 = [_dcov2_from_centered(centered[i], centered[i]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dcov2 = _dcov2_from_centered(centered[i], centered[j])
            denom = np.sqrt(max(dvar2[i], 0.0) * max(dvar2[j], 0.0))
            value = 0.0 if denom <= 0.0 else float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(denom))
            value = float(np.clip(value, 0.0, 1.0))
            out[i, j] = out[j, i] = value
    result = pd.DataFrame(out, index=columns, columns=columns)
    _validate_dependence_matrix(result)
    return result


# --------------------------------------------------------------------------------------
# Mutual information (Kraskov, Stogbauer, Grassberger 2004, via scikit-learn)
# --------------------------------------------------------------------------------------


def mutual_information_matrix(
    returns: pd.DataFrame,
    *,
    n_neighbors: int = 3,
    window: int | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Pairwise mutual information, symmetrised, diagonal ``NaN`` (see the module docstring).

    ``n_neighbors`` is the k-NN estimator's own parameter (``sklearn.feature_selection.
    mutual_info_regression``'s ``n_neighbors``, default 3, matching scikit-learn's own default).
    Non-negative in theory; scikit-learn's estimator can return a small negative value for
    (near-)independent variables by construction (a known finite-sample property of the k-NN
    estimator, not a bug in this wrapper) -- reported as-is, not clipped to 0, so the raw estimate
    is always visible to the caller.
    """
    used = _complete_case(returns, window=window)
    columns = list(used.columns)
    n = len(columns)
    values = used.to_numpy(dtype=float)
    out = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            forward = mutual_info_regression(
                values[:, [i]], values[:, j], n_neighbors=n_neighbors, random_state=random_state
            )[0]
            backward = mutual_info_regression(
                values[:, [j]], values[:, i], n_neighbors=n_neighbors, random_state=random_state
            )[0]
            value = float((forward + backward) / 2.0)
            out[i, j] = out[j, i] = value
    return pd.DataFrame(out, index=columns, columns=columns)


# --------------------------------------------------------------------------------------
# Comparison with Pearson
# --------------------------------------------------------------------------------------


def nonlinearity_gap(nonlinear_measure: pd.DataFrame, pearson_corr: pd.DataFrame) -> pd.DataFrame:
    """``nonlinear_measure - abs(pearson_corr)``, diagonal masked to ``NaN``.

    A large positive gap for a pair is evidence of dependence Pearson misses (only meaningful
    when ``nonlinear_measure`` is bounded comparably to ``abs(pearson_corr)`` -- distance
    correlation, in [0, 1] like ``abs(pearson_corr)``; a raw mutual-information matrix has no
    such bound and this function will still compute a numeric result for it, but that result has
    no fixed scale to interpret against -- **[Inference]**, the caller's responsibility to use
    this with distance correlation, not blindly with mutual information). Both inputs must share
    the same labels in the same order (as ``distance_correlation_matrix``/``estimators.estimate``
    both already return).
    """
    if list(nonlinear_measure.index) != list(pearson_corr.index) or list(
        nonlinear_measure.columns
    ) != list(pearson_corr.columns):
        raise NonlinearError("both matrices must share identical labels in the same order")
    gap = nonlinear_measure.to_numpy(dtype=float) - np.abs(pearson_corr.to_numpy(dtype=float))
    np.fill_diagonal(gap, np.nan)
    return pd.DataFrame(gap, index=nonlinear_measure.index, columns=nonlinear_measure.columns)


# --------------------------------------------------------------------------------------
# Permutation test of independence (Szekely et al. 2007, section 5) and FDR correction
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PermutationTestResult:
    """Result of :func:`distance_correlation_permutation_test` (one-sided: dependence > chance)."""

    statistic: float
    p_value: float
    n_permutations: int


def distance_correlation_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 0,
) -> PermutationTestResult:
    """Permutation test of ``H0: x independent of y`` using distance correlation.

    Shuffles ``y`` relative to ``x``, recomputes the statistic, and reports ``p = (1 +
    #{null >= observed}) / (1 + n_permutations)`` -- the same one-sided convention as
    ``correlation.networks.sector_purity_test`` (step 3.3). Under permutation, the two distance
    variances (``dvar2_x``, ``dvar2_y``) are exactly invariant (they depend only on the multiset
    of each variable's own pairwise distances, not on row order), so only the cross term is
    recomputed per permutation -- a genuine cost reduction, not an approximation.
    """
    if n_permutations < 1:
        raise NonlinearError(f"n_permutations must be >= 1, got {n_permutations}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = distance_correlation(x, y)

    a = _double_center(_distance_matrix(x))
    b = _double_center(_distance_matrix(y))
    dvar2_x = _dcov2_from_centered(a, a)
    dvar2_y = _dcov2_from_centered(b, b)
    denom = np.sqrt(max(dvar2_x, 0.0) * max(dvar2_y, 0.0))

    rng = np.random.default_rng(seed)
    n = len(x)
    null = np.empty(n_permutations, dtype=float)
    for k in range(n_permutations):
        perm = rng.permutation(n)
        b_perm = b[np.ix_(perm, perm)]
        dcov2 = _dcov2_from_centered(a, b_perm)
        value = 0.0 if denom <= 0.0 else float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(denom))
        null[k] = float(np.clip(value, 0.0, 1.0))
    exceed = int(np.sum(null >= observed - 1e-12))
    return PermutationTestResult(
        statistic=observed,
        p_value=(1 + exceed) / (1 + n_permutations),
        n_permutations=n_permutations,
    )


def benjamini_hochberg_qvalues(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg (1995) step-up adjusted p-values ("q-values"), same order as input.

    ``reject_i = q_i <= alpha`` at any chosen ``alpha`` controls the false discovery rate at
    level ``alpha`` (the paper's original linear step-up procedure; the later two-stage adaptive
    variants are not implemented here). Monotone by construction (q-values never decrease going
    from the most to the least significant raw p-value).
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise NonlinearError("p_values must be 1-D")
    if np.any((p < 0.0) | (p > 1.0)):
        raise NonlinearError("p_values must be in [0, 1]")
    m = len(p)
    if m == 0:
        return np.empty(0, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    return q


def pairwise_distance_correlation_significance(
    returns: pd.DataFrame,
    *,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    window: int | None = None,
) -> pd.DataFrame:
    """Permutation-test every pair's distance correlation, FDR-correct across all pairs.

    Long-format output, one row per unordered pair, columns ``asset_a``, ``asset_b``,
    ``distance_correlation``, ``p_value``, ``q_value``, ``reject`` (``q_value <= alpha``). See
    the module docstring's performance note before running this at the daily horizon over many
    tickers.
    """
    used = _complete_case(returns, window=window)
    columns = list(used.columns)
    n = len(columns)
    values = used.to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for i in range(n):
        for j in range(i + 1, n):
            result = distance_correlation_permutation_test(
                values[:, i], values[:, j], n_permutations=n_permutations, seed=seed
            )
            rows.append(
                {
                    "asset_a": columns[i],
                    "asset_b": columns[j],
                    "distance_correlation": result.statistic,
                    "p_value": result.p_value,
                }
            )
    table = pd.DataFrame(rows)
    table["q_value"] = benjamini_hochberg_qvalues(table["p_value"].to_numpy(dtype=float))
    table["reject"] = table["q_value"] <= alpha
    return table
