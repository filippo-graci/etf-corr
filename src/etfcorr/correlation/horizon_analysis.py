"""Step 2.3 (basic scope): correlation matrices per horizon, and the rolling/EWMA "coupling
indicator" -- a date-indexed, leakage-safe scalar summary of average pairwise correlation.

Question served: ``correlation/estimators.py`` (step 2.2) answers "which correlation measure,
for exactly the rows I hand it?" for a *single* point-in-time matrix. Two things still need one
consistent place: (1) "what does that matrix look like at each of the four sampling horizons
(D/W/M/Q)?" -- the "heatmaps for D/W/M/Q" milestone of ``project_plan.md`` section 10, week 2 --
and (2) "how does average pairwise correlation move over time?" -- a single number per date,
computed only from data known as of that date, suitable for a time-series plot and, later, for
H1/H2 (``docs/STATUS.md``'s own step ordering: "the rolling/EWMA coupling indicator, calling step
2.2's ``estimate()``/``ewma_corr()`` repeatedly over a moving window"). This module answers both,
by calling ``estimators.py``'s single-shot functions repeatedly -- exactly the division of labour
``estimators.py``'s own module docstring and ``docs/STATUS.md``'s 2026-09-27 step-2.2 decision log
entry already name as this step's job ("A *rolling* (one-matrix-per-date) application of these
estimators ... is left to ``horizon_analysis.py``").

**Scope, stated explicitly (deviates from the full file description in `project_plan.md` section
6.4).** That table describes the *end state* of this file: "correlation matrices per horizon;
similarity across horizons (Frobenius distance, top-eigenvector overlap) before/after detoning
(H1); rolling average off-diagonal correlation; regime-conditional correlations with volatility
adjustment (H2); stationary block-bootstrap CIs via ``arch.bootstrap``." `docs/STATUS.md`'s own
proposed step order narrows step 2.3 to "basic: heatmaps per horizon, the rolling/EWMA 'coupling
indicator'" -- deliberately, because the rest of that table depends on modules that do not exist
yet: detoning and the market-mode eigenvector need ``correlation/rmt.py`` (``project_plan.md``
section 10, week 3); regime-conditional correlation needs ``regimes.py`` (week 4, H2's own
pre-registered phase in ``config/experiment.yaml`` is 4); stationary block-bootstrap CIs need
``arch.bootstrap`` (not installed in this planning sandbox, not yet imported by any module in this
repo). Writing "H1"/"H2" code against modules that are still placeholders would mean testing
against nothing real. This module therefore delivers only the two pieces that are fully
implementable *now*, against real, already-LOCAL-PASSED dependencies (``correlation/estimators.py``
step 2.2, ``etfcorr.leakage`` step 0.4): per-horizon correlation matrices, and a leakage-tested
rolling/EWMA average-off-diagonal-correlation series. Cross-horizon similarity (Frobenius
distance, eigenvector overlap, before/after detoning) and volatility-adjusted regime-conditional
correlation are left for the step that actually builds ``rmt.py``/``regimes.py`` and wires this
module's output into H1/H2's pre-registered tests; adding placeholder Frobenius/eigenvector code
against nothing to compare it to would be building the confirmatory-test machinery before its
result exists, which the project's own workflow (H1-H6 are touched once their dependencies are
real, not before) argues against.

**No CLI command, no file I/O -- a further stated deviation from `returns.py`'s own precedent.**
`docs/checks/2.2.md`'s design-decision note reads: "The natural CLI-facing use of this module --
computing and displaying heatmaps per horizon -- is ``horizon_analysis.py``'s job (step 2.3)."
*Displaying* needs ``viz/heatmaps.py`` (not built until Phase 8, per the file tree in
``project_plan.md`` section 4); *computing* is what :func:`corr_by_horizon` below does, but as a
pure function of an already-loaded ``{horizon: returns_frame}`` mapping, not by reading
``data/processed/returns_<horizon>.parquet`` itself. Wiring that read (and a
``data/processed/corr/*.parquet`` write, per the data-flow diagram in `project_plan.md` section 5)
into a CLI command is deferred to whichever later step actually needs the persisted matrices (the
Makefile's own ``corr: uv run etfcorr corr --all`` target is not part of week 2's build order) --
`notebooks/02_correlation_by_horizon.ipynb` (this step's other deliverable) calls this module's
functions directly on frames it loads itself with ``pandas.read_parquet``/
``etfcorr.returns.returns_by_horizon``, the same "notebooks call ``src/``, contain no logic" rule
every earlier notebook follows. Keeping this module free of ``etfcorr.config``/``etfcorr.data.*``/
CLI dependencies also means, like ``correlation/estimators.py`` before it, nothing here needs a
Parquet/typer injectable seam or a "NOT EXECUTED OFFLINE" marker: every line is executable and
tested in this planning sandbox.

**The rolling/EWMA coupling indicator is leakage-tested with `etfcorr.leakage`'s *real* shared
detectors, not a purity substitute.** ``estimators.py``'s own module docstring and
`tests/leakage/test_estimators_purity.py`'s docstring both explain why that module's point
estimators are the wrong output *shape* for ``assert_no_lookahead``/``assert_future_invariant``
(no per-date output for the detectors' truncation machinery to slice). :func:`rolling_coupling_
indicator` and :func:`ewma_coupling_indicator` below produce exactly the shape those detectors are
built for -- one scalar per date -- so this is the first module in the project able to use them
directly rather than a bespoke analog. See ``tests/leakage/test_horizon_analysis_no_lookahead.py``.

**Method notes and citation status.**

* **"Coupling indicator" = the sample mean of a correlation matrix's off-diagonal entries**
  (:func:`average_offdiagonal_corr`). This is a direct descriptive statistic (an arithmetic mean of
  already-computed correlations), not an estimator with its own citation to check --
  **[Inference]**, not an empirical claim requiring verification. The concept of tracking
  average pairwise correlation as a single number over time is common in the literature and
  practice (e.g. implied/realized average-correlation indices); no specific paper is cited here
  because none is needed for what is simply a mean.
* **Rolling window methods** (Pearson/Spearman/Kendall/Ledoit-Wolf/OAS): :func:`rolling_
  coupling_indicator` calls :func:`estimators.estimate` once per date, on ``returns`` truncated
  to that date and then further restricted to the trailing ``window`` rows by ``estimate``'s own
  ``window`` parameter -- the literal "calling step 2.2's ``estimate()`` ... repeatedly over a
  moving window" instruction, not a hand-rolled alternative. **Known limitation, disclosed rather
  than silently accepted**: this recomputes the full matrix from scratch at every date (no
  incremental update), so cost scales as ``O(T * window * N^2)`` -- fine for this step's synthetic
  fixtures and for exploratory use with a coarser ``step``, but a real run over the ~3,200-row,
  56-ticker daily core panel at ``step=1`` would be slow; ``step`` (default 1) exists precisely so
  a caller can trade date resolution for runtime (e.g. ``step=5`` for a twice-weekly reading of a
  daily panel). No performance tuning has been done, matching ``docs/checks/2.2.md``'s own stated
  limitation for ``estimators.py`` ("not yet a bottleneck at N=56") -- here it plausibly could be,
  for a full-resolution daily run, and that risk is stated rather than hidden.
* **EWMA**: :func:`ewma_coupling_indicator` calls :func:`estimators.ewma_corr` once per date on
  ``returns`` truncated to that date (optionally further capped by ``window``, EWMA's decay
  already gives it a soft effective memory without one). Same recompute-from-scratch cost
  structure and the same ``step`` escape hatch.
* **Delivered in step 3.2, below (``config/experiment.yaml``'s H1 test, in full)**:
  cross-horizon similarity of correlation matrices and of leading eigenvectors, before versus
  after market-mode removal, with stationary-bootstrap confidence intervals on the difference
  (:func:`horizon_pair_similarity`, :func:`bootstrap_horizon_pair_similarity`,
  :func:`h1_cross_horizon_test`). **Still not delivered**: regime-conditional volatility-adjusted
  correlation (H2, pre-registered at phase 4, needs ``regimes.py``, week 4).

No third-party code is adapted here; nothing new for ``THIRD_PARTY.md``.

---

**Step 3.2 addition: the H1 engine (cross-horizon similarity, before vs after detoning, with
stationary-bootstrap confidence intervals on the difference).**

Question served: step 3.1 (``correlation/rmt.py``) delivered the *tool* -- ``detone_corr``,
which removes the market mode from one correlation matrix -- and explicitly deferred "the actual
H1 comparison (correlation-matrix and leading-eigenvector similarity across horizons, before
versus after detoning, with stationary-bootstrap confidence intervals on the difference)" to this
step (``rmt.py``'s own module docstring, and ``docs/STATUS.md``'s "Next three tasks" entry
pointing here). ``config/experiment.yaml``'s H1 test text, verbatim: "Similarity of correlation
matrices and of leading eigenvectors across horizons, before versus after market-mode removal;
stationary-bootstrap confidence intervals on the difference." Every function below implements one
clause of that sentence: :func:`frobenius_distance`/:func:`leading_eigenvector_overlap` are the
two similarity metrics named ("correlation matrices" and "leading eigenvectors" respectively,
matching ``project_plan.md`` section 6.4's own "Frobenius distance, top-eigenvector overlap"
description); :func:`horizon_pair_similarity` computes both, before and after
``rmt.detone_corr``; :func:`stationary_bootstrap_indices`/:func:`bootstrap_horizon_pair_similarity`
build the confidence interval on the before/after difference; :func:`h1_cross_horizon_test` is the
one entry point that runs the whole test across every pair of horizons named in H1's statement
("across horizons (D/W/M)").

**Similarity metrics, method notes and citation status.**

* **Frobenius distance** (:func:`frobenius_distance`): the standard full-matrix Frobenius norm of
  the difference between two (unit-diagonal, same-tickers) correlation matrices -- a direct
  linear-algebra construction, **[Inference]**, not itself a citation-bearing claim. Lower means
  more similar. ``project_plan.md`` section 6.4 names this explicitly as part of H1's comparison.
* **Leading-eigenvector overlap** (:func:`leading_eigenvector_overlap`): the mean squared cosine
  of the principal angles between the top ``n_top`` eigenspaces of the two matrices -- for
  ``n_top=1`` (the default, matching ``rmt.detone_corr``'s own default ``n_market=1``) this is
  exactly the squared cosine between the two matrices' top eigenvectors, sign-independent by
  construction (squared, not absolute). A standard subspace-similarity measure, bounded in
  ``[0, 1]`` by the fact that singular values of a product of two orthonormal-column matrices lie
  in ``[0, 1]`` -- **[Inference]**, a direct linear-algebra fact. Higher means more similar.
  ``project_plan.md`` section 6.4's own "top-eigenvector overlap" is exactly this.
* **Literature precedent for the comparison itself** (not re-verified this session; already
  **CHECKED** in ``REFERENCES_SEED.md`` section B, per ``rmt.py``'s own module docstring, which
  named these as the literature precedent for this step without implementing it): Tumminello, M.,
  Di Matteo, T., Aste, T., Mantegna, R. N. (2007), *Correlation based networks of equity returns
  sampled at different time horizons*, Eur. Phys. J. B, arXiv:physics/0605251 -- structure varies
  with sampling horizon, the empirical motivation for H1's premise. Borghesi, C., Marsili, M.,
  Miccichè, S. (2007), *Emergence of time-horizon invariant correlation structure in financial
  returns by subtraction of the market mode*, Phys. Rev. E, arXiv:physics/0702106 -- the specific
  literature precedent for H1's own mechanism (market-mode removal makes structure more
  horizon-invariant), which this module tests on this project's own universe rather than assumes.

**Stationary bootstrap: method notes, citation status and a stated design simplification.**

* **Method**: Politis, D. N., Romano, J. P. (1994), *The Stationary Bootstrap*, Journal of the
  American Statistical Association 89(428), 1303-1313, DOI 10.1080/01621459.1994.10476870 --
  **CHECKED** this session (title, authors, year, journal, volume/issue/pages confirmed by a web
  search; upgrades the existing UNVERIFIED entry in ``REFERENCES_SEED.md`` section E, which
  already named it as "Bootstrap CIs for dependent data" and is exactly why
  ``config/experiment.yaml``'s ``inference.bootstrap`` field is named ``"stationary"``).
  :func:`stationary_bootstrap_indices` draws block lengths i.i.d. ``Geometric(p)`` with mean
  ``mean_block_length`` (``p = 1 / mean_block_length``) and uniform-random block start positions,
  wrapping circularly at the end of the sample so every resampled series has exactly the original
  length and every position has equal marginal probability of starting a block (the "stationary"
  property the circular block bootstrap, with a *fixed* block length, does not have). Confidence
  intervals are the plain **percentile** method (quantiles of the bootstrap replicate
  distribution) -- Efron and Tibshirani, *An Introduction to the Bootstrap* (Chapman &
  Hall 1993, CHECKED in library catalogues on 2026-11-02); more refined variants
  (bias-corrected-and-accelerated / BCa, the basic or "pivotal" method) are **not implemented
  here**, a stated simplification, not an oversight.
* **Block length is a stated heuristic, not a validated automatic selection**
  (:func:`default_mean_block_length`): Politis, D., White, H. (2004), *Automatic Block-Length
  Selection for the Dependent Bootstrap*, Econometric Reviews (with a 2009 correction by Patton,
  Politis and White) gives a real data-driven procedure -- 23(1), 53-70, correction 28(4),
  372-375, both CHECKED on Crossref on 2026-11-02, and **not implemented** here. The
  ``round(n ** (1/3))`` cube-root-growth default used instead is **[Inference]**, a commonly-cited
  *order* of growth for dependent-bootstrap block length, not a checked, specific citation; it
  exists so the bootstrap functions have a sensible, overridable, disclosed default rather than an
  unjustified fixed constant, and any caller who wants the fuller automatic selection can compute
  it separately and pass it in via ``mean_block_length_a``/``mean_block_length_b``.
* **Design simplification, stated explicitly**: :func:`bootstrap_horizon_pair_similarity` resamples
  the two horizons' return series **independently** at every replicate (see that function's own
  docstring for the full reasoning) rather than jointly from a shared underlying daily price path.
  This is the conservative direction for a difference-in-similarity confidence interval -- treating
  two horizons as independent is expected to *widen*, not narrow, the reported interval relative to
  a fully joint scheme, the safer error for a confirmatory test to make.
* **Known limitation, empirically measured this session, not swept under the rug**: on a
  *pure-noise* panel (population correlation = identity; see :func:`etfcorr.synthetic.
  make_noise_returns`), the Frobenius-distance metric's before/after-detoning difference is
  **not** null-centered -- across five independent noise-panel seeds (D vs a 5-day-subsampled
  "W," ``n_bootstrap=200`` each), ``frobenius_diff``'s point estimate was negative in every run
  (values roughly -0.7 to -1.2) and its 90% stationary-bootstrap CI excluded 0 on the negative
  side every time: detoning made two independently-sampled *noise* matrices systematically *less*
  Frobenius-similar to each other, not more. The mechanism: even under a null of no true
  cross-sectional correlation, a finite sample's top eigenvalue is inflated by sampling noise
  (Marchenko-Pastur), removing it leaves a residual with a smaller diagonal before
  renormalization, and :func:`~etfcorr.correlation.rmt.cov_to_corr`'s rescaling amplifies
  whatever (spurious, sample-specific) structure remains -- an artifact of the cleaning procedure
  itself, not evidence of genuine market-mode horizon-invariance. **The eigenvector-overlap
  metric did not show this bias** in the same runs (``eigvec_diff``'s CI contained 0 in every
  seed) -- so a real H1 verdict should weigh the eigenvector-overlap result more than the
  Frobenius result, or explicitly control for this null bias (e.g. by reporting the bootstrap
  distribution's location relative to a matched noise-panel null, not just whether it excludes
  raw 0) before treating a positive ``frobenius_diff`` alone as support for H1. This finding is
  carried into ``docs/STATUS.md``'s open questions, not treated as settled.
* **Not applied here, left to the scoreboard (Phase 8, not built yet)**:
  ``config/experiment.yaml``'s ``inference.multiplicity: holm`` correction across the six
  confirmatory hypotheses H1-H6. Every CI/interval this module reports is for one pair, one metric,
  uncorrected for the multiple pairs or multiple hypotheses being examined across the whole
  project -- see :func:`h1_cross_horizon_test`'s own docstring.

No third-party code is adapted here; nothing new for ``THIRD_PARTY.md``.
"""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from etfcorr.config import Horizon
from etfcorr.correlation.estimators import (
    DEFAULT_EWMA_LAMBDA,
    Backend,
    CorrelationReport,
    CorrMethod,
    corr_report,
    estimate,
    ewma_corr,
    validate_correlation_matrix,
)
from etfcorr.correlation.rmt import detone_corr


class HorizonAnalysisError(ValueError):
    """Invalid input to, or an invalid configuration of, a horizon/coupling-indicator call."""


# --------------------------------------------------------------------------------------
# The coupling indicator: one scalar per correlation matrix
# --------------------------------------------------------------------------------------


def average_offdiagonal_corr(corr: pd.DataFrame) -> float:
    """Mean of the strictly-upper-triangular entries of one correlation matrix.

    Each pair is counted once (not the full off-diagonal, which would double-count by symmetry).
    Does not itself validate ``corr`` (symmetry, unit diagonal, PSD) -- see
    :func:`etfcorr.correlation.estimators.validate_correlation_matrix`; every value this module
    actually feeds in comes from :func:`estimators.estimate`/:func:`estimators.ewma_corr`, both of
    which already validate their own output before returning it.
    """
    if not isinstance(corr, pd.DataFrame):
        raise HorizonAnalysisError(f"corr must be a pandas DataFrame, got {type(corr)}")
    if corr.shape[0] != corr.shape[1]:
        raise HorizonAnalysisError(f"corr must be square, got shape {corr.shape}")
    n = corr.shape[0]
    if n < 2:
        raise HorizonAnalysisError(f"need at least 2 assets to average an off-diagonal, got {n}")
    values = corr.to_numpy(dtype=float)
    iu = np.triu_indices(n, k=1)
    return float(values[iu].mean())


# --------------------------------------------------------------------------------------
# Correlation matrices per horizon ("heatmaps for D/W/M/Q")
# --------------------------------------------------------------------------------------


def corr_by_horizon(
    returns_by_horizon: Mapping[Horizon, pd.DataFrame],
    method: CorrMethod,
    *,
    window: int | None = None,
    backend: Backend = "sklearn",
    **kw: Any,
) -> dict[Horizon, CorrelationReport]:
    """One :func:`estimators.corr_report` per horizon; order of ``returns_by_horizon`` preserved.

    This is the data behind "heatmaps for D/W/M/Q" (``project_plan.md`` section 10's week-2
    milestone): each :class:`~etfcorr.correlation.estimators.CorrelationReport` bundles the
    labelled matrix with T/N/condition-number/PSD, ready for ``viz/heatmaps.py`` (Phase 8, not
    built yet -- plotting is out of this module's scope, see the module docstring) or direct
    inspection (e.g. ``report.corr`` as a ``DataFrame``) in a notebook. ``returns_by_horizon`` is
    typically ``etfcorr.returns.returns_by_horizon(prices, horizons)``'s output, or the equivalent
    read from ``data/processed/returns_<horizon>.parquet`` per horizon -- this function does no
    I/O of its own (see the module docstring's "no CLI" deviation note).
    """
    if not returns_by_horizon:
        raise HorizonAnalysisError("returns_by_horizon is empty; nothing to compute")
    return {
        horizon: corr_report(returns, method, window=window, backend=backend, **kw)
        for horizon, returns in returns_by_horizon.items()
    }


# --------------------------------------------------------------------------------------
# Rolling / EWMA coupling indicator: one scalar per date, leakage-safe by construction
# --------------------------------------------------------------------------------------


def rolling_coupling_indicator(
    returns: pd.DataFrame,
    method: CorrMethod,
    window: int,
    *,
    step: int = 1,
    backend: Backend = "sklearn",
    **kw: Any,
) -> pd.Series:
    """Average off-diagonal correlation, re-estimated on a trailing ``window`` at each date.

    For each row position ``i`` (starting once ``window`` rows are available, then every ``step``
    rows), calls ``estimate(returns.iloc[: i + 1], method, window=window, ...)`` -- i.e. the
    trailing ``window`` rows of exactly the data known up to and including date
    ``returns.index[i]`` -- and reports :func:`average_offdiagonal_corr` of the result at that
    date. Because every value at position ``i`` is computed from ``returns.iloc[: i + 1]`` alone,
    this function is no-lookahead *by construction*: calling it on ``returns.iloc[:t]`` reproduces
    exactly the same values, for every date ``<= t``, as calling it on the full ``returns`` (see
    ``tests/leakage/test_horizon_analysis_no_lookahead.py``, which checks this with
    ``etfcorr.leakage.assert_no_lookahead``/``assert_future_invariant`` directly rather than
    merely asserting it).

    Raises :class:`HorizonAnalysisError` for ``method="ewma"`` (EWMA has no fixed window -- use
    :func:`ewma_coupling_indicator`, or :func:`coupling_indicator`, which dispatches
    automatically), for ``window < 2``, for ``step < 1``, and when ``returns`` has fewer than
    ``window`` rows. ``**kw`` is forwarded to ``estimate`` (e.g. constructor keywords for
    Ledoit-Wolf/OAS).
    """
    if method == "ewma":
        raise HorizonAnalysisError(
            "method='ewma' has no fixed window; use ewma_coupling_indicator (or "
            "coupling_indicator, which dispatches automatically) instead"
        )
    if not isinstance(returns, pd.DataFrame):
        raise HorizonAnalysisError(f"returns must be a pandas DataFrame, got {type(returns)}")
    if window < 2:
        raise HorizonAnalysisError(f"window must be >= 2, got {window}")
    if step < 1:
        raise HorizonAnalysisError(f"step must be >= 1, got {step}")
    n = len(returns)
    if n < window:
        raise HorizonAnalysisError(f"returns has {n} rows, fewer than window={window}")

    dates: list[Any] = []
    values: list[float] = []
    for i in range(window - 1, n, step):
        corr = estimate(returns.iloc[: i + 1], method, window=window, backend=backend, **kw)
        dates.append(returns.index[i])
        values.append(average_offdiagonal_corr(corr))
    index = pd.Index(dates, name=returns.index.name)
    return pd.Series(values, index=index, name=f"{method}_w{window}_coupling", dtype=float)


def ewma_coupling_indicator(
    returns: pd.DataFrame,
    lam: float = DEFAULT_EWMA_LAMBDA,
    *,
    min_periods: int = 2,
    window: int | None = None,
    step: int = 1,
) -> pd.Series:
    """Average off-diagonal EWMA correlation, re-estimated on data up to each date.

    For each row position ``i`` (starting once ``min_periods`` rows are available, then every
    ``step`` rows), calls ``ewma_corr(returns.iloc[: i + 1], lam, min_periods=min_periods,
    window=window)`` and reports :func:`average_offdiagonal_corr` of the result at that date.
    Leakage-safe by construction, the same way and for the same reason as
    :func:`rolling_coupling_indicator` (see its docstring); ``window`` here is EWMA's own optional
    row-count cap (:func:`estimators.ewma_corr`'s parameter of the same name), not a required
    fixed window -- EWMA's exponential decay already gives it a soft effective memory without one,
    so ``window=None`` (the default) uses the full truncated history at every date.
    """
    if not isinstance(returns, pd.DataFrame):
        raise HorizonAnalysisError(f"returns must be a pandas DataFrame, got {type(returns)}")
    if step < 1:
        raise HorizonAnalysisError(f"step must be >= 1, got {step}")
    start = max(min_periods, 2) - 1
    n = len(returns)
    if n <= start:
        raise HorizonAnalysisError(
            f"returns has {n} rows, fewer than min_periods={min_periods} requires"
        )

    dates: list[Any] = []
    values: list[float] = []
    for i in range(start, n, step):
        corr = ewma_corr(returns.iloc[: i + 1], lam, min_periods=min_periods, window=window)
        dates.append(returns.index[i])
        values.append(average_offdiagonal_corr(corr))
    index = pd.Index(dates, name=returns.index.name)
    window_label = "full" if window is None else f"w{window}"
    return pd.Series(values, index=index, name=f"ewma{lam}_{window_label}_coupling", dtype=float)


def coupling_indicator(
    returns: pd.DataFrame,
    method: CorrMethod,
    *,
    window: int | None = None,
    lam: float | None = None,
    min_periods: int = 2,
    step: int = 1,
    backend: Backend = "sklearn",
    **kw: Any,
) -> pd.Series:
    """One entry point for the rolling/EWMA coupling indicator, dispatching on ``method``.

    Mirrors ``estimators.estimate``'s "one interface" style (``project_plan.md`` section 6.4),
    extended here to a date-indexed series: ``method="ewma"`` routes to
    :func:`ewma_coupling_indicator` (``window`` is then EWMA's optional row-count cap, ``lam``
    defaults to :data:`estimators.DEFAULT_EWMA_LAMBDA`); every other method routes to
    :func:`rolling_coupling_indicator` and requires ``window`` (raises
    :class:`HorizonAnalysisError` if omitted -- a fixed window is not optional for those methods).
    """
    if method == "ewma":
        return ewma_coupling_indicator(
            returns,
            lam if lam is not None else DEFAULT_EWMA_LAMBDA,
            min_periods=min_periods,
            window=window,
            step=step,
        )
    if window is None:
        raise HorizonAnalysisError(f"method={method!r} requires window (a fixed trailing window)")
    return rolling_coupling_indicator(returns, method, window, step=step, backend=backend, **kw)


# --------------------------------------------------------------------------------------
# H1, part 1: similarity of two correlation matrices (Frobenius distance, leading-eigenvector
# overlap) -- see the module docstring's "Step 3.2 addition" section for the full method notes.
# --------------------------------------------------------------------------------------

DEFAULT_H1_HORIZONS: tuple[Horizon, ...] = ("D", "W", "M")
DEFAULT_ALPHA = 0.05  # mirrors config/experiment.yaml's inference.alpha (not imported; see below)
DEFAULT_N_BOOTSTRAP = 2000  # mirrors config/experiment.yaml's inference.n_bootstrap


def _same_labels(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Both matrices must share identical row/column labels, in identical order.

    Cross-horizon similarity compares *the same* assets' correlation structure at two
    horizons; a silent positional comparison across a differently-ordered or differently-sized
    ticker set would compare unrelated entries. Order is required to match, not just the set,
    since every comparison below is entrywise on aligned positions.
    """
    if list(a.columns) != list(b.columns) or list(a.index) != list(b.index):
        raise HorizonAnalysisError(
            "the two correlation matrices must share identical row/column labels, in the same "
            "order, to be compared -- align returns_by_horizon to a common ticker set and order "
            "before calling"
        )


def frobenius_distance(corr_a: pd.DataFrame, corr_b: pd.DataFrame) -> float:
    """``||corr_a - corr_b||_F``, the standard full-matrix Frobenius norm of the difference.

    Every entry (both triangles, symmetric duplicates included) contributes; since both inputs
    are unit-diagonal correlation matrices the diagonal contributes exactly 0. Lower means more
    similar; 0 only when the two matrices are identical. See the module docstring's "Step 3.2
    addition" section for why this, rather than an upper-triangle-only variant, is used (a fixed,
    known ``sqrt(2)`` scale factor apart, immaterial since only relative before/after comparisons
    at a fixed horizon pair are used anywhere in this module).
    """
    validate_correlation_matrix(corr_a)
    validate_correlation_matrix(corr_b)
    _same_labels(corr_a, corr_b)
    diff = corr_a.to_numpy(dtype=float) - corr_b.to_numpy(dtype=float)
    return float(np.linalg.norm(diff, ord="fro"))


def _top_eigenvectors(corr: pd.DataFrame, n_top: int) -> np.ndarray:
    values, vectors = np.linalg.eigh(corr.to_numpy(dtype=float))
    order = np.argsort(values)[::-1]
    return np.asarray(vectors[:, order[:n_top]])


def leading_eigenvector_overlap(
    corr_a: pd.DataFrame, corr_b: pd.DataFrame, n_top: int = 1
) -> float:
    """Mean squared cosine of the principal angles between the top ``n_top`` eigenspaces.

    For ``n_top=1`` (matching ``rmt.detone_corr``'s own default ``n_market=1``) this is exactly
    the squared cosine between the two matrices' top eigenvectors -- squared, not absolute, so the
    result does not depend on either eigenvector's arbitrary sign. For ``n_top > 1``, computed as
    ``||V_a^T V_b||_F^2 / n_top`` for the two matrices' ``n_top`` leading eigenvectors ``V_a``,
    ``V_b`` (orthonormal columns) -- bounded in ``[0, 1]`` (1 = identical subspace, 0 = orthogonal
    subspaces; see the module docstring for why this bound holds). Higher means more similar.

    **Known limitation, disclosed rather than silently accepted**: this metric is only well
    defined when the ``n_top``-th and ``(n_top + 1)``-th eigenvalues are distinct. When the top
    eigenvalue(s) are (near-)degenerate (repeated, or nearly so -- an exact equicorrelation matrix
    is the sharpest example: its *non*-market eigenvalue has multiplicity ``n - 1``), the "top
    eigenvector(s)" are not uniquely determined -- any orthonormal basis of the degenerate
    eigenspace is an equally valid choice -- so ``numpy.linalg.eigh`` can return a basis that
    differs between two matrices with an (analytically) identical eigenspace purely from
    floating-point noise, and the reported overlap can then be small even though the underlying
    subspaces coincide. This is expected behaviour of the underlying eigen-decomposition, not a
    bug in this function; it is most likely to matter for a detoned matrix built from a stylised
    (e.g. exactly-equicorrelated) input, less likely for a real, heterogeneous asset panel, and not
    checked for here (no degeneracy warning is raised).
    """
    validate_correlation_matrix(corr_a)
    validate_correlation_matrix(corr_b)
    _same_labels(corr_a, corr_b)
    n = corr_a.shape[0]
    if not 1 <= n_top < n:
        raise HorizonAnalysisError(f"n_top must satisfy 1 <= n_top < {n} (n_assets), got {n_top}")
    va = _top_eigenvectors(corr_a, n_top)
    vb = _top_eigenvectors(corr_b, n_top)
    m = va.T @ vb
    return float(np.sum(m * m) / n_top)


@dataclass(frozen=True)
class HorizonPairSimilarity:
    """Cross-horizon similarity of one pair of correlation matrices, before vs after detoning.

    ``frobenius_diff = frobenius_before - frobenius_after``: positive means the two horizons'
    matrices are *more* similar (smaller Frobenius distance) after market-mode removal -- the
    direction H1 predicts. ``eigvec_diff = eigvec_after - eigvec_before``: positive means the two
    horizons' leading eigenspaces are *more* aligned after removal -- again the direction H1
    predicts. Both are defined so "positive supports H1" for either metric, deliberately, to keep
    sign conventions consistent for reporting.
    """

    horizon_a: Horizon
    horizon_b: Horizon
    method: CorrMethod
    n_market: int
    n_top: int
    frobenius_before: float
    frobenius_after: float
    frobenius_diff: float
    eigvec_before: float
    eigvec_after: float
    eigvec_diff: float


def horizon_pair_similarity(
    corr_a: pd.DataFrame,
    corr_b: pd.DataFrame,
    horizon_a: Horizon,
    horizon_b: Horizon,
    *,
    method: CorrMethod = "pearson",
    n_market: int = 1,
    n_top: int | None = None,
) -> HorizonPairSimilarity:
    """One horizon pair's Frobenius distance and leading-eigenvector overlap, before vs after
    detoning with :func:`etfcorr.correlation.rmt.detone_corr` (``n_market`` eigenmode(s) removed,
    matching that function's own default of 1 -- the market mode).

    ``method``/``n_market``/``n_top`` are carried in the result purely for reporting; this
    function does not itself estimate a correlation matrix from returns (see
    :func:`cross_horizon_similarity`/:func:`bootstrap_horizon_pair_similarity` for that).
    ``n_top`` defaults to ``n_market`` -- by default, the eigenvector-overlap metric compares
    exactly the modes ``detone_corr`` removed.
    """
    n_top_ = n_market if n_top is None else n_top
    detoned_a = detone_corr(corr_a, n_market=n_market)
    detoned_b = detone_corr(corr_b, n_market=n_market)
    fro_before = frobenius_distance(corr_a, corr_b)
    fro_after = frobenius_distance(detoned_a, detoned_b)
    eig_before = leading_eigenvector_overlap(corr_a, corr_b, n_top=n_top_)
    eig_after = leading_eigenvector_overlap(detoned_a, detoned_b, n_top=n_top_)
    return HorizonPairSimilarity(
        horizon_a=horizon_a,
        horizon_b=horizon_b,
        method=method,
        n_market=n_market,
        n_top=n_top_,
        frobenius_before=fro_before,
        frobenius_after=fro_after,
        frobenius_diff=fro_before - fro_after,
        eigvec_before=eig_before,
        eigvec_after=eig_after,
        eigvec_diff=eig_after - eig_before,
    )


def cross_horizon_similarity(
    returns_by_horizon: Mapping[Horizon, pd.DataFrame],
    method: CorrMethod = "pearson",
    *,
    horizons: Sequence[Horizon] = DEFAULT_H1_HORIZONS,
    window: int | None = None,
    n_market: int = 1,
    n_top: int | None = None,
    backend: Backend = "sklearn",
    **kw: Any,
) -> dict[tuple[Horizon, Horizon], HorizonPairSimilarity]:
    """Point-estimate :func:`horizon_pair_similarity` for every pair among ``horizons``.

    ``horizons`` defaults to ``("D", "W", "M")`` -- ``config/experiment.yaml``'s own H1 statement
    names only daily/weekly/monthly ("across horizons (D/W/M)"); quarterly is excluded by default
    (not forbidden -- pass it explicitly) since the project's own standard already flags the
    quarterly sample as ill-conditioned/exploratory-only at this universe size (T/N ~ 1-1.5), a
    different concern from what this function tests. Requires at least two horizons, and every
    named horizon present in ``returns_by_horizon``. No bootstrap CIs here -- point estimates only;
    see :func:`h1_cross_horizon_test` for the full H1 test with confidence intervals.
    """
    if len(horizons) < 2:
        raise HorizonAnalysisError(f"need at least 2 horizons to compare, got {horizons!r}")
    missing = [h for h in horizons if h not in returns_by_horizon]
    if missing:
        raise HorizonAnalysisError(f"horizons {missing} not present in returns_by_horizon")
    reports = corr_by_horizon(
        {h: returns_by_horizon[h] for h in horizons},
        method,
        window=window,
        backend=backend,
        **kw,
    )
    return {
        (a, b): horizon_pair_similarity(
            reports[a].corr, reports[b].corr, a, b, method=method, n_market=n_market, n_top=n_top
        )
        for a, b in itertools.combinations(horizons, 2)
    }


# --------------------------------------------------------------------------------------
# H1, part 2: the stationary bootstrap, and confidence intervals on the before/after difference
# -- see the module docstring's "Step 3.2 addition" section for the full method notes.
# --------------------------------------------------------------------------------------


def stationary_bootstrap_indices(
    n: int, mean_block_length: float, rng: np.random.Generator
) -> np.ndarray:
    """One resampled row-index array of length ``n``, via the stationary bootstrap of Politis and
    Romano (1994) -- see the module docstring for the full citation and method description.

    Draws block lengths i.i.d. ``Geometric(p)`` with mean ``mean_block_length``
    (``p = 1 / mean_block_length``) and a uniform-random start position for each block, filling
    ``n`` positions and wrapping circularly past the end of the sample so every resampled series
    is exactly ``n`` long and every position has equal marginal probability of starting a block.
    ``mean_block_length=1`` reduces to the ordinary i.i.d. bootstrap (every position redrawn
    independently, mean block length exactly 1); a very large ``mean_block_length`` approaches one
    contiguous circular block (a single random rotation of the original series).
    """
    if n < 1:
        raise HorizonAnalysisError(f"n must be >= 1, got {n}")
    if mean_block_length <= 0:
        raise HorizonAnalysisError(f"mean_block_length must be > 0, got {mean_block_length}")
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


def default_mean_block_length(n_obs: int) -> float:
    """A simple ``round(n_obs ** (1/3))`` heuristic mean block length, floored at 2.

    **Not a validated automatic block-length selection** -- see the module docstring for why
    (Politis and White (2004) gives one; not implemented here). Used only as an explicit,
    overridable, disclosed default so :func:`bootstrap_horizon_pair_similarity`/
    :func:`h1_cross_horizon_test` have a sensible starting point without silently picking an
    unjustified fixed constant.
    """
    if n_obs < 2:
        raise HorizonAnalysisError(f"n_obs must be >= 2, got {n_obs}")
    return float(max(2.0, round(n_obs ** (1.0 / 3.0))))


def _stationary_resample(
    frame: pd.DataFrame, mean_block_length: float, rng: np.random.Generator
) -> pd.DataFrame:
    """One stationary-bootstrap replicate of ``frame``'s rows.

    Returns a fresh default-integer-indexed frame (the resampled rows are not a real calendar, so
    carrying the original date index forward would be misleading); columns are preserved.
    """
    idx = stationary_bootstrap_indices(len(frame), mean_block_length, rng)
    return pd.DataFrame(frame.to_numpy(dtype=float)[idx], columns=frame.columns)


@dataclass(frozen=True)
class BootstrapCI:
    """A percentile stationary-bootstrap confidence interval for one before/after difference
    statistic (:attr:`HorizonPairSimilarity.frobenius_diff` or ``.eigvec_diff``)."""

    point_estimate: float
    lower: float
    upper: float
    alpha: float
    n_bootstrap: int
    mean_block_length_a: float
    mean_block_length_b: float
    replicates: tuple[float, ...]

    @property
    def excludes_zero(self) -> bool:
        """True when the interval does not contain 0 -- informal significance at ``alpha``."""
        return self.lower > 0.0 or self.upper < 0.0


def _percentile_ci(
    point_estimate: float,
    replicates: np.ndarray,
    *,
    alpha: float,
    n_bootstrap: int,
    mean_block_length_a: float,
    mean_block_length_b: float,
) -> BootstrapCI:
    lower = float(np.quantile(replicates, alpha / 2.0))
    upper = float(np.quantile(replicates, 1.0 - alpha / 2.0))
    return BootstrapCI(
        point_estimate=point_estimate,
        lower=lower,
        upper=upper,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        mean_block_length_a=mean_block_length_a,
        mean_block_length_b=mean_block_length_b,
        replicates=tuple(float(x) for x in replicates),
    )


def bootstrap_horizon_pair_similarity(
    returns_a: pd.DataFrame,
    returns_b: pd.DataFrame,
    horizon_a: Horizon,
    horizon_b: Horizon,
    method: CorrMethod = "pearson",
    *,
    window: int | None = None,
    n_market: int = 1,
    n_top: int | None = None,
    backend: Backend = "sklearn",
    mean_block_length_a: float | None = None,
    mean_block_length_b: float | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    **kw: Any,
) -> tuple[HorizonPairSimilarity, BootstrapCI, BootstrapCI]:
    """Point estimate plus stationary-bootstrap CIs on ``frobenius_diff``/``eigvec_diff`` for one
    horizon pair.

    **Design simplification, stated explicitly**: ``returns_a`` and ``returns_b`` are block-
    resampled *independently* at every bootstrap replicate (each with its own
    ``mean_block_length``, defaulting to :func:`default_mean_block_length` of its own,
    window-restricted row count), not jointly from a shared underlying daily price path. A fully
    joint bootstrap would resample the shared daily price history once per replicate and re-derive
    every horizon's returns from that one resampled path (``etfcorr.returns.horizon_returns``),
    preserving whatever dependence genuinely exists between, say, the daily and weekly matrices'
    sampling error; that is not implemented here (this module's own contract is "a pure function of
    already-loaded returns," see the module docstring's earlier "no CLI" note -- reaching into
    ``returns.py``'s resampling logic from inside a bootstrap loop would break that contract).
    Treating the two horizons as independently resampled is the **conservative** direction for a
    difference-in-similarity confidence interval: it adds resampling noise that a joint scheme
    would instead let cancel through shared randomness, so the resulting interval is expected to be
    wider, not narrower, than a fully joint bootstrap's would give -- the safer error for a
    confirmatory H1 test to make, not a free simplification.

    A bootstrap replicate that produces a degenerate resample (for example a numerically zero-
    variance column) raises the underlying estimator's own error (``CorrelationError`` from
    ``estimators.estimate``, or ``RMTError`` from ``rmt.detone_corr``) uncaught -- not silently
    skipped, since silently dropping unstable replicates would bias the reported interval toward
    only the stable ones.

    ``window`` (if given) restricts each horizon's returns to its trailing ``window`` rows *before*
    resampling, so every replicate has exactly ``window`` rows and the block-length default is
    computed from ``window``, not the full sample; the point estimate uses the same windowed
    sample. Every random draw is seeded deterministically from ``seed`` (``returns_a`` resampled
    before ``returns_b`` at every replicate), so two calls with identical inputs and ``seed``
    reproduce identical CIs.
    """
    ra = returns_a if window is None else returns_a.tail(window)
    rb = returns_b if window is None else returns_b.tail(window)
    mbl_a = (
        default_mean_block_length(len(ra)) if mean_block_length_a is None else mean_block_length_a
    )
    mbl_b = (
        default_mean_block_length(len(rb)) if mean_block_length_b is None else mean_block_length_b
    )

    point = horizon_pair_similarity(
        estimate(ra, method, backend=backend, **kw),
        estimate(rb, method, backend=backend, **kw),
        horizon_a,
        horizon_b,
        method=method,
        n_market=n_market,
        n_top=n_top,
    )

    rng = np.random.default_rng(seed)
    fro_diffs = np.empty(n_bootstrap)
    eig_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boot_a = _stationary_resample(ra, mbl_a, rng)
        boot_b = _stationary_resample(rb, mbl_b, rng)
        pair = horizon_pair_similarity(
            estimate(boot_a, method, backend=backend, **kw),
            estimate(boot_b, method, backend=backend, **kw),
            horizon_a,
            horizon_b,
            method=method,
            n_market=n_market,
            n_top=n_top,
        )
        fro_diffs[i] = pair.frobenius_diff
        eig_diffs[i] = pair.eigvec_diff

    fro_ci = _percentile_ci(
        point.frobenius_diff,
        fro_diffs,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        mean_block_length_a=mbl_a,
        mean_block_length_b=mbl_b,
    )
    eig_ci = _percentile_ci(
        point.eigvec_diff,
        eig_diffs,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        mean_block_length_a=mbl_a,
        mean_block_length_b=mbl_b,
    )
    return point, fro_ci, eig_ci


@dataclass(frozen=True)
class H1PairResult:
    """One horizon pair's point similarity plus both bootstrap CIs -- ``config/experiment.yaml``'s
    H1 test text ("similarity ... before versus after market-mode removal; stationary-bootstrap
    confidence intervals on the difference") applied to one pair."""

    similarity: HorizonPairSimilarity
    frobenius_ci: BootstrapCI
    eigvec_ci: BootstrapCI


def h1_cross_horizon_test(
    returns_by_horizon: Mapping[Horizon, pd.DataFrame],
    method: CorrMethod = "pearson",
    *,
    horizons: Sequence[Horizon] = DEFAULT_H1_HORIZONS,
    window: int | None = None,
    n_market: int = 1,
    n_top: int | None = None,
    backend: Backend = "sklearn",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    **kw: Any,
) -> dict[tuple[Horizon, Horizon], H1PairResult]:
    """The full H1 engine: :func:`horizon_pair_similarity` plus stationary-bootstrap CIs
    (:func:`bootstrap_horizon_pair_similarity`) for every pair among ``horizons`` --
    ``config/experiment.yaml``'s H1 test text, in full, applied to every horizon pair.

    Defaults mirror ``config/experiment.yaml``'s own ``inference`` block (``n_bootstrap=2000``,
    ``alpha=0.05``) without importing ``etfcorr.config`` -- the same "pure numerical library"
    contract ``estimators.py``'s ``DEFAULT_EWMA_LAMBDA`` already follows (see the module
    docstring); a caller reproducing the pre-registered H1 test exactly should pass
    ``config/experiment.yaml``'s own values explicitly. ``config/experiment.yaml``'s own
    ``inference.multiplicity: holm`` (a Holm-Bonferroni correction across the six confirmatory
    tests H1-H6) is **not** applied here: it is a cross-hypothesis correction that belongs where
    H1-H6's individual results are assembled into the scoreboard (``evaluation/scoreboard.py``,
    Phase 8, not built yet), not inside one hypothesis's own engine -- every CI this function
    returns is per-pair, per-metric, and uncorrected for the multiple pairs/metrics/hypotheses
    being examined across the whole project. The per-pair block length defaults to
    :func:`default_mean_block_length` of that horizon's own (window-restricted, if ``window`` is
    given) row count; ``seed`` is offset by each pair's position in ``itertools.combinations``
    order, so two calls with identical inputs and ``seed`` reproduce identical CIs for every pair.
    """
    if len(horizons) < 2:
        raise HorizonAnalysisError(f"need at least 2 horizons to compare, got {horizons!r}")
    missing = [h for h in horizons if h not in returns_by_horizon]
    if missing:
        raise HorizonAnalysisError(f"horizons {missing} not present in returns_by_horizon")
    result: dict[tuple[Horizon, Horizon], H1PairResult] = {}
    for offset, (a, b) in enumerate(itertools.combinations(horizons, 2)):
        point, fro_ci, eig_ci = bootstrap_horizon_pair_similarity(
            returns_by_horizon[a],
            returns_by_horizon[b],
            a,
            b,
            method,
            window=window,
            n_market=n_market,
            n_top=n_top,
            backend=backend,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=seed + offset,
            **kw,
        )
        result[(a, b)] = H1PairResult(similarity=point, frobenius_ci=fro_ci, eigvec_ci=eig_ci)
    return result


def h1_pair_summary_line(result: H1PairResult) -> str:
    """One human-readable line per horizon pair: both metrics, before/after, diff and CI.

    A trailing ``*`` marks a metric whose CI excludes 0 (see :attr:`BootstrapCI.excludes_zero`) --
    an informal per-pair significance flag, uncorrected for multiple comparisons (see
    :func:`h1_cross_horizon_test`'s own docstring on why the Holm correction is not applied here).
    """
    s = result.similarity
    fc, ec = result.frobenius_ci, result.eigvec_ci
    fro_flag = "*" if fc.excludes_zero else " "
    eig_flag = "*" if ec.excludes_zero else " "
    return (
        f"{s.horizon_a}-{s.horizon_b} ({s.method}, n_market={s.n_market}): "
        f"Frobenius {s.frobenius_before:.4f}->{s.frobenius_after:.4f} "
        f"diff={s.frobenius_diff:+.4f} CI=[{fc.lower:+.4f},{fc.upper:+.4f}]{fro_flag}  "
        f"eigvec {s.eigvec_before:.4f}->{s.eigvec_after:.4f} "
        f"diff={s.eigvec_diff:+.4f} CI=[{ec.lower:+.4f},{ec.upper:+.4f}]{eig_flag}"
    )
