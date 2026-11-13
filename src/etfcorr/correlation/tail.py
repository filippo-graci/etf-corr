"""Step 3.4b: empirical tail dependence with stationary-bootstrap confidence intervals -- the
"tail dependence as robustness checks" item of this project's own custom instructions (section 2,
"Correlation estimation"). Distance correlation and mutual information (the sibling module
``nonlinear.py``) ask "how much dependence exists overall, linear or not?"; this module asks a
different question specific to extreme-value theory: "does the dependence between two return
series get stronger specifically in the joint tail (both assets falling, or both rising,
together), beyond what a single overall correlation number would already imply?" These are
distinct enough in both citation base and method (extreme-value theory versus general dependence
measures) that ``project_plan.md`` section 4's own file layout keeps them as separate files
(``correlation/nonlinear.py`` and ``correlation/tail.py``); this module is the latter.

**Not the same question as H2's Forbes-Rigobon check.** This project's own custom instructions
(section 2) already flag that "correlation is heteroskedasticity-biased in high-volatility
regimes (Forbes and Rigobon)," and H2 (``config/experiment.yaml``, phase 4, not yet built) is the
volatility-adjusted regime-conditional comparison that follows from that. Tail dependence is a
different, complementary object: it is a **copula-based** property (from the joint distribution
of the two series' *probability-integral-transformed* ranks, not of their raw values or a
correlation coefficient computed on a volatility-conditioned subsample), estimating whether two
series co-move specifically in their joint extremes even after removing the marginal scale of
each series entirely (see the "pseudo-observations" step below) -- a genuinely different
diagnostic, not a duplicate of H2's own volatility-adjustment logic. Both are relevant to H6
(hedge effectiveness in worst-decile stress periods, phase 8): this module's output is a
candidate input to that later analysis, not itself an H6 test.

**Method notes and citation status** (checked by a web search this session; ``REFERENCES_SEED.md``
section B lists Longin and Solnik (2001) as the finance-specific application; read in full
2026-11-02):

* **The tail-dependence coefficient and its naive nonparametric estimator.** Frahm, G., Junker,
  M., Schmidt, R. (2005), *Estimating the tail-dependence coefficient: properties and pitfalls*,
  Insurance: Mathematics and Economics 37(1), 80-100, DOI 10.1016/j.insmatheco.2005.05.008 --
  **CHECKED** this session (title, authors, journal, volume, issue, page range confirmed via
  RePEc/IDEAS' own listing). The (lower) tail-dependence coefficient is defined as ``lambda_L =
  lim_{q -> 0+} P(F_Y(Y) <= q | F_X(X) <= q)`` for the population marginal CDFs ``F_X``, ``F_Y``
  (a **copula** property: invariant to any strictly increasing transform of either marginal, per
  the definition itself -- Sklar's theorem, not separately cited here). This paper's own title
  names exactly what :func:`empirical_tail_dependence` implements -- the simplest nonparametric
  ("naive"/plug-in) estimator: replace ``F_X``, ``F_Y`` by the empirical marginal ranks (``rank /
  (n + 1)``, avoiding a rank of exactly 1.0) and replace the limit by a fixed, small ``q`` --
  and, true to this paper's own title, explicitly discusses this estimator's "pitfalls": its
  variance is driven by the number of observations actually inside the joint tail region, roughly
  ``n * q``, which shrinks as ``q -> 0`` exactly when the estimator is supposed to become more
  accurate. :func:`empirical_tail_dependence` and :func:`tail_dependence_matrix` both report an
  ``effective_n = round(n * q)`` figure for exactly this reason (see below), rather than letting a
  q=0.05 estimate on this project's ~41-49-row quarterly panel be read at face value -- this
  project's own custom instructions already flag quarterly correlation estimation as ill-
  conditioned/exploratory for the same underlying reason (T/N too small), and the same caution
  applies here for a different statistic. Joe, H. (1997), *Multivariate Models and Dependence
  Concepts*, Chapman & Hall (the population definition's usual textbook source) -- CHECKED in
  library catalogues on 2026-11-02, cited only for the standard definition, not for any specific
  claim about this estimator.
* **Not implemented: parametric/semi-parametric tail-dependence estimators.** Frahm et al. (2005)
  also study smoother estimators (their own "logarithmic" and "smoothed" variants, and estimators
  derived from fitting an extreme-value copula) that reduce the naive estimator's variance at the
  cost of a parametric assumption on the tail's shape; none of those are implemented here -- this
  module delivers only the naive, assumption-light estimator the paper's title itself centres on,
  a deliberate simplification stated rather than silently adopted.
* **Stationary bootstrap for the confidence interval.** Politis, D. N., Romano, J. P. (1994),
  *The Stationary Bootstrap*, JASA 89(428), 1303-1313 -- already **CHECKED** in
  ``REFERENCES_SEED.md`` (step 3.2, 2026-09-28) via ``horizon_analysis.py``'s own citation of the
  same paper; not re-verified separately here since it is the same reference. :func:`_stationary_
  bootstrap_indices` and :func:`_default_mean_block_length` below are a **deliberate, disclosed
  duplication** of ``horizon_analysis.stationary_bootstrap_indices``/``default_mean_block_length``
  (step 3.2), not an import: importing them would pull in ``horizon_analysis``'s own dependency on
  ``etfcorr.config`` (needed there only for an unrelated ``Horizon`` type alias used elsewhere in
  that file), breaking this module's "pure numerical library, no ``etfcorr.config``/``etfcorr.
  data.*`` dependency" contract -- the same contract ``estimators.py``/``rmt.py``/``correlation.
  networks``/``nonlinear.py`` all already state and keep. Both copies are algorithmically
  identical (same geometric-block-length, circular-wrap construction) and independently unit-
  tested here; a numerical cross-check between the two copies with a real, shared seed is a Gate 2
  item (see the Delivery block), run in the real repository where both modules coexist alongside
  ``etfcorr.config``, rather than built as an extra offline dependency in this planning sandbox.
  **Unlike ``horizon_analysis.bootstrap_horizon_pair_similarity``'s two-horizon case (which
  resamples each horizon's returns independently, a stated conservative simplification), this
  module's bootstrap resamples ``x`` and ``y`` with the *same* index array every replicate**: they
  are the same pair's two return series over the same dates, so preserving their pairing (and each
  series' own serial dependence, via the block structure) is not a simplification but the only
  correct way to resample paired data -- there is no "independent versus joint" design choice to
  state here the way there was for two different, only loosely related horizons.
* **Percentile confidence interval.** Efron and Tibshirani, *An Introduction to the Bootstrap* --
  Chapman & Hall 1993, CHECKED in library catalogues on 2026-11-02; see
  ``horizon_analysis.py``'s own step-3.2 note on this same plain percentile-method simplification
  (more refined variants -- BCa, basic/pivotal -- are not implemented here either, for the same
  stated reason).
* **Complete-case rows and pseudo-observations computed per pair, not matrix-wide.** Unlike
  ``estimators.py``/``nonlinear.py``'s matrix-wide complete-case convention, tail dependence's
  marginal ranks must be computed from each *pair*'s own common sample (dropping rows where either
  of that pair's two series is missing) -- using the matrix-wide complete-case sample instead
  would silently shrink every pair's ``n`` (and its ranks) to whatever the *most incomplete*
  ticker in the whole panel allows, wasting most of the data for pairs whose own two series are
  actually both fully observed over a much longer common history. This is a stated, deliberate
  departure from ``estimators.py``'s own convention, made for exactly the reason that module's
  docstring itself flags as its known cost ("wasteful for a staggered-start extended sub-panel").
* **No dependency on ``etfcorr.config``/``etfcorr.data.*``/other ``etfcorr.correlation.*``
  modules** beyond the disclosed duplication above -- numpy and pandas only, the same "pure
  numerical library, no I/O" contract as ``estimators.py``/``rmt.py``/``correlation.networks``/
  ``nonlinear.py``.

No third-party code is adapted; nothing new for ``THIRD_PARTY.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "TailBootstrapCI",
    "TailError",
    "TailSide",
    "bootstrap_tail_dependence",
    "empirical_tail_dependence",
    "pairwise_tail_dependence_bootstrap",
    "tail_dependence_matrix",
]

DEFAULT_Q = 0.05
DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_ALPHA = 0.05

TailSide = Literal["lower", "upper"]


class TailError(ValueError):
    """Invalid input to, or an invalid configuration of, a tail-dependence call."""


# --------------------------------------------------------------------------------------
# Pseudo-observations and the naive nonparametric tail-dependence estimator
# --------------------------------------------------------------------------------------


def _pseudo_observations(x: np.ndarray) -> np.ndarray:
    """Empirical marginal ranks in (0, 1): ``rank / (n + 1)`` (never exactly 0 or 1)."""
    n = len(x)
    ranks = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    return ranks / (n + 1.0)


def empirical_tail_dependence(
    x: np.ndarray, y: np.ndarray, q: float = DEFAULT_Q, *, tail: TailSide = "lower"
) -> float:
    """Naive nonparametric tail-dependence coefficient (Frahm, Junker, Schmidt 2005).

    ``lambda_hat(q) = (1 / (n * q)) * #{i : U_i <= q, V_i <= q}`` for ``tail="lower"`` (with the
    mirrored ``>= 1 - q`` exceedance for ``tail="upper"``), where ``U``, ``V`` are the empirical
    marginal ranks of ``x``, ``y``. Values above 1 are possible for a small, noisy sample (more
    joint exceedances than ``n * q`` would suggest) and are not clipped -- see the module
    docstring's "pitfalls" note; read the paired ``effective_n = round(n * q)`` before trusting a
    single point estimate, especially at this project's monthly/quarterly horizons.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise TailError("x and y must be 1-D arrays")
    if len(x) != len(y):
        raise TailError(f"x and y must have the same length, got {len(x)} and {len(y)}")
    n = len(x)
    if n < 4:
        raise TailError(f"need at least 4 observations, got {n}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise TailError("x and y must not contain NaN or infinite values")
    if not 0.0 < q < 0.5:
        raise TailError(f"q must be in (0, 0.5), got {q}")
    if tail not in ("lower", "upper"):
        raise TailError(f"tail must be 'lower' or 'upper', got {tail!r}")

    u = _pseudo_observations(x)
    v = _pseudo_observations(y)
    if tail == "lower":
        joint = np.mean((u <= q) & (v <= q))
    else:
        joint = np.mean((u >= 1.0 - q) & (v >= 1.0 - q))
    return float(joint / q)


def effective_tail_n(n_obs: int, q: float) -> int:
    """``round(n_obs * q)`` -- the approximate number of observations inside the joint tail
    region an :func:`empirical_tail_dependence` estimate is actually based on (see the module
    docstring's "pitfalls" note); report this next to every estimate, never the estimate alone.
    """
    if n_obs < 1:
        raise TailError(f"n_obs must be >= 1, got {n_obs}")
    if not 0.0 < q < 0.5:
        raise TailError(f"q must be in (0, 0.5), got {q}")
    return round(n_obs * q)


def tail_dependence_matrix(
    returns: pd.DataFrame,
    q: float = DEFAULT_Q,
    *,
    tail: TailSide = "lower",
    window: int | None = None,
) -> pd.DataFrame:
    """Pairwise tail dependence, unit diagonal (trivially ``lambda(x, x) = 1``), symmetric.

    Each pair uses its own common (pairwise-complete) sample, not the matrix-wide complete case
    -- see the module docstring for why this deliberately departs from ``estimators.py``'s
    convention. ``window`` restricts to the trailing ``window`` rows of ``returns`` first, before
    the pairwise pairing.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TailError(f"returns must be a pandas DataFrame, got {type(returns)}")
    if returns.shape[1] < 2:
        raise TailError("returns must have at least 2 columns to form pairs")
    sliced = returns if window is None else returns.tail(window)
    columns = list(sliced.columns)
    n = len(columns)
    out = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            pair = sliced.iloc[:, [i, j]].dropna(how="any")
            if len(pair) < 4:
                raise TailError(
                    f"pair ({columns[i]!r}, {columns[j]!r}) has only {len(pair)} complete-case "
                    "rows, need at least 4"
                )
            value = empirical_tail_dependence(
                pair.iloc[:, 0].to_numpy(dtype=float),
                pair.iloc[:, 1].to_numpy(dtype=float),
                q,
                tail=tail,
            )
            out[i, j] = out[j, i] = value
    result = pd.DataFrame(out, index=columns, columns=columns)
    values = result.to_numpy(dtype=float)
    if not np.allclose(values, values.T, atol=1e-8):  # pragma: no cover
        raise TailError("internal error: resulting matrix is not symmetric")
    if not np.allclose(np.diag(values), 1.0, atol=1e-8):  # pragma: no cover
        raise TailError("internal error: resulting matrix diagonal is not all ones")
    return result


# --------------------------------------------------------------------------------------
# Stationary bootstrap (duplicated from horizon_analysis.py -- see the module docstring)
# --------------------------------------------------------------------------------------


def _stationary_bootstrap_indices(
    n: int, mean_block_length: float, rng: np.random.Generator
) -> np.ndarray:
    """Identical algorithm to ``horizon_analysis.stationary_bootstrap_indices`` (step 3.2); see
    the module docstring for why this is a disclosed duplication rather than an import."""
    if n < 1:
        raise TailError(f"n must be >= 1, got {n}")
    if mean_block_length <= 0:
        raise TailError(f"mean_block_length must be > 0, got {mean_block_length}")
    p = 1.0 / mean_block_length
    idx = np.empty(n, dtype=np.int64)
    filled = 0
    while filled < n:
        start = int(rng.integers(0, n))
        length = min(int(rng.geometric(p)), n - filled)
        end = start + length
        if end <= n:
            block = np.arange(start, end)
        else:
            block = np.concatenate([np.arange(start, n), np.arange(0, end - n)])
        idx[filled : filled + length] = block
        filled += length
    return idx


def _default_mean_block_length(n_obs: int) -> float:
    """Identical heuristic to ``horizon_analysis.default_mean_block_length`` (step 3.2): not a
    validated automatic block-length selection method, see that module's own docstring."""
    if n_obs < 2:
        raise TailError(f"n_obs must be >= 2, got {n_obs}")
    return float(max(2.0, round(n_obs ** (1.0 / 3.0))))


@dataclass(frozen=True)
class TailBootstrapCI:
    """A percentile stationary-bootstrap confidence interval for one tail-dependence estimate."""

    point_estimate: float
    lower: float
    upper: float
    alpha: float
    n_bootstrap: int
    mean_block_length: float
    effective_n: int
    replicates: tuple[float, ...]

    @property
    def excludes_zero(self) -> bool:
        """True when the interval does not contain 0 -- informal significance at ``alpha``."""
        return self.lower > 0.0 or self.upper < 0.0


def bootstrap_tail_dependence(
    x: np.ndarray,
    y: np.ndarray,
    q: float = DEFAULT_Q,
    *,
    tail: TailSide = "lower",
    mean_block_length: float | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> TailBootstrapCI:
    """Percentile stationary-bootstrap confidence interval on :func:`empirical_tail_dependence`.

    ``x`` and ``y`` are resampled with the **same** block-bootstrap index array at every
    replicate (see the module docstring for why this is not the same design choice as
    ``horizon_analysis``'s independent two-horizon resampling). ``mean_block_length`` defaults to
    :func:`_default_mean_block_length` of ``len(x)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    point = empirical_tail_dependence(x, y, q, tail=tail)
    n = len(x)
    block_length = _default_mean_block_length(n) if mean_block_length is None else mean_block_length
    if n_bootstrap < 1:
        raise TailError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if not 0.0 < alpha < 1.0:
        raise TailError(f"alpha must be in (0, 1), got {alpha}")

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_bootstrap, dtype=float)
    for k in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(n, block_length, rng)
        replicates[k] = empirical_tail_dependence(x[idx], y[idx], q, tail=tail)
    lower = float(np.quantile(replicates, alpha / 2.0))
    upper = float(np.quantile(replicates, 1.0 - alpha / 2.0))
    return TailBootstrapCI(
        point_estimate=point,
        lower=lower,
        upper=upper,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        mean_block_length=block_length,
        effective_n=effective_tail_n(n, q),
        replicates=tuple(float(v) for v in replicates),
    )


def pairwise_tail_dependence_bootstrap(
    returns: pd.DataFrame,
    q: float = DEFAULT_Q,
    *,
    tail: TailSide = "lower",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    window: int | None = None,
) -> pd.DataFrame:
    """:func:`bootstrap_tail_dependence` for every pair, long-format, with ``effective_n``.

    Columns: ``asset_a``, ``asset_b``, ``tail_dependence``, ``ci_lower``, ``ci_upper``,
    ``effective_n``, ``excludes_zero``. No multiple-comparison correction is applied here (unlike
    ``nonlinear.pairwise_distance_correlation_significance``'s Benjamini-Hochberg step): a
    confidence interval excluding zero is not itself a p-value, and this project's own workflow
    reserves the FDR/Holm correction for pre-registered hypothesis-testing tables (H1-H6), not
    every exploratory scan -- the caller applies a correction explicitly if this output is used
    for a formal multiple-testing claim.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TailError(f"returns must be a pandas DataFrame, got {type(returns)}")
    sliced = returns if window is None else returns.tail(window)
    columns = list(sliced.columns)
    n = len(columns)
    rows: list[dict[str, object]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pair = sliced.iloc[:, [i, j]].dropna(how="any")
            if len(pair) < 4:
                raise TailError(
                    f"pair ({columns[i]!r}, {columns[j]!r}) has only {len(pair)} complete-case "
                    "rows, need at least 4"
                )
            ci = bootstrap_tail_dependence(
                pair.iloc[:, 0].to_numpy(dtype=float),
                pair.iloc[:, 1].to_numpy(dtype=float),
                q,
                tail=tail,
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                seed=seed,
            )
            rows.append(
                {
                    "asset_a": columns[i],
                    "asset_b": columns[j],
                    "tail_dependence": ci.point_estimate,
                    "ci_lower": ci.lower,
                    "ci_upper": ci.upper,
                    "effective_n": ci.effective_n,
                    "excludes_zero": ci.excludes_zero,
                }
            )
    return pd.DataFrame(rows)
