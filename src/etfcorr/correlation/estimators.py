"""Step 2.2: the correlation estimator zoo -- Pearson/Spearman/Kendall, Ledoit-Wolf/OAS shrinkage,
EWMA -- behind one interface, with structural validation and estimator diagnostics.

Question served: every later Phase-2/3 module (``horizon_analysis.py``, ``rmt.py``, ``dcc.py``,
``networks.py``, ``overlap.py``) needs *a* correlation matrix from *some* returns frame, computed
one consistent way, always checked to actually be a valid correlation matrix, and always reported
with the numbers a skeptical statistician will ask for first: which estimator, over how many
observations (T), how many assets (N), how ill-conditioned, and whether it is even positive
semi-definite. This module is the single place that answers "which correlation measure?" (the
project's own standard, section 6) for the *static* (single-matrix) case; a *rolling* application
of these estimators across time -- the "rolling/EWMA coupling indicator" -- is
``horizon_analysis.py``'s job (step 2.3, per ``docs/STATUS.md``'s own step ordering), not this
module's. This module's contract ends at "a pure function of exactly the frame it is given";
Phase 3+ walk-forward code is responsible for passing it only ``returns.loc[:t]``.

**Deviation from project_plan.md section 6.4, stated on purpose.** The plan's own table reads:
"Ledoit-Wolf/OAS via skfolio (``LedoitWolf``, ``OAS``) with scikit-learn as cross-check." skfolio
is a pinned core dependency (``pyproject.toml``) and is confirmed installed on Filippo's machine
(``docs/STATUS.md`` decision log, 2026-09-21: "skfolio 1.3.0" resolved locally), but is NOT
installable in this planning sandbox (PyPI unreachable here, confirmed this session by a direct
``pip install`` attempt, same as every previous step's typer/pyarrow/pandas-stubs gaps). Rather
than write code against an API this sandbox cannot import or test at all, this module uses
``sklearn.covariance.LedoitWolf``/``OAS`` directly as the tested, offline-executable
implementation (scikit-learn is a core dependency present in *both* environments), and reserves a
``backend`` parameter naming ``"sklearn"`` as the only implemented value for now. A ``skfolio``
backend is a natural follow-up (skfolio's own ``LedoitWolf``/``OAS`` prior-covariance estimators
are, as far as this session can tell without being able to import or read their source,
convenience wrappers around the same scikit-learn estimators for portfolio-specific data handling
-- CHECKED 2026-11-02 in the installed skfolio: both subclass scikit-learn's estimators) and is
exactly the plan's own cross-check, run in reverse order of availability: at Gate 2, compare
``estimate(returns, "ledoit_wolf")`` against skfolio's own estimator on the same data and record
whether they agree, rather than asserting agreement now. Every call site can request a
non-``"sklearn"`` backend and gets a clear ``CorrelationError`` rather than a silent fallback.

**Method notes, citation status, and known limitations** (see ``REFERENCES_SEED.md`` for the
checked/unverified split; nothing here is asserted as a checked citation beyond what
``REFERENCES_SEED.md`` already marks CHECKED):

* **Pearson/Spearman/Kendall** (``pandas.DataFrame.corr``, per the plan's own "via pandas").
  Pearson and Spearman (Pearson correlation of within-column ranks) are Gram matrices of a
  standardized data matrix and are therefore positive semi-definite *by construction* whenever
  computed from a single common (complete-case) sample -- **[Established]**, a direct linear-
  algebra fact (``r = Z^T Z / (T-1)`` for column-standardized ``Z`` is always PSD), verified by
  running :func:`eigen_diagnostics` against many complete-case Pearson/Spearman matrices in this
  session's tests, not merely asserted. A **Kendall** tau matrix keeps
  that guarantee on complete rows (derived 2026-11-02, replacing an earlier [Unverified] note that
  doubted it): for each pair of days ``(s, t)`` let ``v_i = sign(x_is - x_it)``; tau-a is the
  average of ``v_i v_j`` over pairs, a Gram matrix, and tau-b (scipy's default) divides it by
  ``sqrt(sum v_i^2 * sum v_j^2)``, a cosine matrix; both are PSD. Every Kendall matrix and
  bootstrap draw in ``reports/tables/estimator_stability_3_5.csv`` is PSD.
  Accordingly :func:`eigen_diagnostics`'s ``is_psd`` is reported, never assumed, for every method
  including Kendall, and :func:`validate_correlation_matrix` deliberately does **not** enforce
  PSD as a hard invariant (only symmetry and a unit diagonal, which hold for every method here by
  construction) -- PSD is a *diagnostic*, not a postcondition this module can honestly guarantee
  for Kendall.
* **Ledoit-Wolf / OAS shrinkage.** Ledoit, O., Wolf, M. (2004), *Honey, I Shrunk the Sample
  Covariance Matrix*, Journal of Portfolio Management 30(4) -- **CHECKED**, ``REFERENCES_SEED.md``.
  Chen, Y., Wiesel, A., Eldar, Y., Hero, A. (2010), shrinkage/OAS -- IEEE Trans. Signal Process.
  58(10), 5016-5029, CHECKED on Crossref 2026-11-02, ``REFERENCES_SEED.md``. Implementation follows
  ``project_plan.md`` section 6.2's own draft ``lw_corr``/``cov_to_corr`` snippets (the project's
  own code, not third-party; nothing for ``THIRD_PARTY.md``): standardize each column (``z = (r -
  mean) / std``, ``ddof=1``) before fitting, so the shrinkage target (scaled identity, scaled by the
  trace of the empirical covariance) is effectively a shrinkage toward the *identity correlation
  matrix* rather than toward an arbitrary covariance scale -- exactly the interpretation
  ``project_plan.md``'s own synthetic test already exercised ("Ledoit-Wolf gave a well-conditioned
  matrix (min eigenvalue 0.20)" at the quarterly horizon). Requires a complete-case data matrix (see
  below); this is the primary tool for the T/N-small horizons (monthly, quarterly) the project's own
  standard (section 2) flags as noisy/singular.
* **EWMA.** Implemented as an explicit, own recursion (the plan's stated fallback: "via skfolio
  ``EWCovariance`` or own recursion"), not via ``pandas.DataFrame.ewm().cov()``, for full control
  over initialization and because a hand-verifiable closed-form recursion is far easier to give a
  known-answer test than ``pandas``' ``adjust=True`` weighting convention. Recursion (RiskMetrics-
  style, attributed to J.P. Morgan/Reuters, *RiskMetrics Technical
  Document*, fourth edition (1996): existence and attribution CHECKED 2026-11-02, its EWMA
  chapter not read): ``S_1 = x_1 x_1^T``; ``S_t = lam *
  S_{t-1} + (1 - lam) * x_t x_t^T``, on **raw, non-demeaned** returns (the standard convention for
  short-horizon covariance/correlation forecasting, since the mean is both close to zero and
  poorly estimated at these sample sizes) -- a stated deviation from Pearson/Spearman/Ledoit-
  Wolf/OAS above, which all demean. ``ewma_lambda`` (default 0.94) is ``config/experiment.yaml``'s
  own field, deliberately separate from ``windows_days``: EWMA has no fixed window, so this
  module's EWMA function has no ``window_days``-style parameter, only ``lam`` and an optional
  ``window`` row-count cap (see below).
* **Complete-case only, for every method, including Pearson/Spearman/Kendall.** ``pandas``'
  default pairwise-complete correlation (using, for each *pair* of columns, only the rows where
  *that pair* both have data) is not used here: with an unequal-history panel (this project's
  extended sub-panels) a pairwise-complete matrix has no single well-defined T, complicates the
  condition-number/PSD report this module must produce for every matrix, and is not guaranteed
  PSD even for Pearson. This module always restricts to the rows where *every* requested ticker
  has data (``DataFrame.dropna(how="any")``) before estimating anything, giving one well-defined T
  per call. This is appropriate for the core panel (no missing values in its own active window,
  per step 1.3's validation) and wasteful for a staggered-start extended sub-panel; the caller
  (not this module) is responsible for restricting ``returns`` to the sub-panel/period whose
  common sample is actually wanted (e.g. one sector's extended tickers only) rather than passing
  the full 56-ticker panel and hoping.
* **``window``**: an optional convenience equal to ``returns.tail(window)`` applied before
  estimation, reported back in :class:`CorrelationReport` so "which window" is always visible
  next to the matrix -- see ``tests/leakage/test_estimators_purity.py`` for why this is
  deliberately the *trailing* window and tested against a "leading window" twin to guard against
  a tail/head mix-up (the one directional bug that would actually matter if this parameter were
  ever used inside walk-forward code).

No third-party code is adapted here beyond the project's own ``project_plan.md`` draft snippets
(``cov_to_corr``, the Ledoit-Wolf standardize-then-shrink pattern), already the project's own
code; nothing new for ``THIRD_PARTY.md``.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.covariance import OAS, LedoitWolf

DEFAULT_EWMA_LAMBDA = 0.94

CorrMethod = Literal["pearson", "spearman", "kendall", "ledoit_wolf", "oas", "ewma"]
CORR_METHODS: tuple[CorrMethod, ...] = (
    "pearson",
    "spearman",
    "kendall",
    "ledoit_wolf",
    "oas",
    "ewma",
)
PandasCorrMethod = Literal["pearson", "spearman", "kendall"]

Backend = Literal["sklearn"]
IMPLEMENTED_BACKENDS: tuple[Backend, ...] = ("sklearn",)


class CorrelationError(ValueError):
    """Invalid input to, or an invalid result from, a correlation estimation call."""


# --------------------------------------------------------------------------------------
# Structural validation and eigen-diagnostics (used by every method, not just one)
# --------------------------------------------------------------------------------------


def validate_correlation_matrix(corr: pd.DataFrame, *, tol: float = 1e-8) -> None:
    """Hard structural invariants every returned matrix must satisfy: square, labelled the same
    way on both axes, finite, symmetric, unit diagonal.

    Deliberately does **not** check positive semi-definiteness -- see the module docstring for
    why that is a diagnostic (:func:`eigen_diagnostics`), not a postcondition this function can
    honestly enforce for every method (Kendall's tau is PSD on complete rows, see the module
    docstring).
    """
    if not isinstance(corr, pd.DataFrame):
        raise CorrelationError(f"a correlation matrix must be a pandas DataFrame, got {type(corr)}")
    if corr.shape[0] != corr.shape[1]:
        raise CorrelationError(f"a correlation matrix must be square, got shape {corr.shape}")
    if list(corr.index) != list(corr.columns):
        raise CorrelationError("index and columns must be identical, in the same order")
    values = corr.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise CorrelationError("correlation matrix contains NaN or infinite values")
    if not np.allclose(values, values.T, atol=tol):
        raise CorrelationError("correlation matrix is not symmetric")
    if not np.allclose(np.diag(values), 1.0, atol=tol):
        raise CorrelationError("correlation matrix diagonal is not all ones")


@dataclass(frozen=True)
class EigenDiagnostics:
    """Eigen-based diagnostics of one correlation matrix (a *report*, not an invariant)."""

    condition_number: float
    is_psd: bool
    min_eigenvalue: float
    max_eigenvalue: float


def eigen_diagnostics(corr: pd.DataFrame, *, psd_tol: float = -1e-8) -> EigenDiagnostics:
    """Condition number and PSD check from the matrix's own eigenvalues.

    ``eigvalsh`` assumes a symmetric input; call :func:`validate_correlation_matrix` first if that
    has not already been checked. ``psd_tol`` is a small negative tolerance (not zero) because a
    matrix that is PSD in theory can show a tiny negative eigenvalue from floating-point error
    alone; a genuinely non-PSD result (e.g. Kendall, or a shrinkage matrix computed on garbage
    input) will show a min eigenvalue far below this tolerance, not just past it.
    """
    values = np.linalg.eigvalsh(corr.to_numpy(dtype=float))
    condition_number = float(np.linalg.cond(corr.to_numpy(dtype=float)))
    return EigenDiagnostics(
        condition_number=condition_number,
        is_psd=bool(values.min() >= psd_tol),
        min_eigenvalue=float(values.min()),
        max_eigenvalue=float(values.max()),
    )


# --------------------------------------------------------------------------------------
# Shared input preparation
# --------------------------------------------------------------------------------------


def _complete_case(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows where every column has data -- the well-defined common sample (see module docstring
    for why pairwise-complete correlation is deliberately not offered here)."""
    return frame.dropna(how="any")


def _check_columns_have_variance(frame: pd.DataFrame) -> None:
    std = frame.std(ddof=1)
    bad = [str(c) for c in std.index[std.isna() | (std <= 0)]]
    if bad:
        raise CorrelationError(
            f"columns with zero or undefined variance cannot be correlated: {bad}"
        )


def _prepare(returns: pd.DataFrame, *, window: int | None, min_rows: int) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise CorrelationError(f"returns must be a pandas DataFrame, got {type(returns)}")
    if returns.shape[1] < 1:
        raise CorrelationError("returns has no columns (no assets to correlate)")
    if window is not None:
        if window < min_rows:
            raise CorrelationError(
                f"window {window} is smaller than the required minimum {min_rows}"
            )
        returns = returns.tail(window)
    used = _complete_case(returns)
    if len(used) < min_rows:
        raise CorrelationError(
            f"need at least {min_rows} complete-case rows, got {len(used)} "
            f"(out of {len(returns)} rows over {returns.shape[1]} tickers)"
        )
    _check_columns_have_variance(used)
    return used


# --------------------------------------------------------------------------------------
# Pearson / Spearman / Kendall
# --------------------------------------------------------------------------------------


def _pandas_corr(used: pd.DataFrame, method: PandasCorrMethod) -> pd.DataFrame:
    corr = used.corr(method=method)
    corr.index.name = corr.columns.name = None
    return corr


# --------------------------------------------------------------------------------------
# Ledoit-Wolf / OAS shrinkage
# --------------------------------------------------------------------------------------


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Covariance -> correlation, via ``project_plan.md`` section 6.2's own ``cov_to_corr``."""
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def _shrinkage_corr(
    used: pd.DataFrame, method: Literal["ledoit_wolf", "oas"], **kw: Any
) -> pd.DataFrame:
    z = (used - used.mean()) / used.std(ddof=1)
    estimator = LedoitWolf(**kw) if method == "ledoit_wolf" else OAS(**kw)
    fitted = estimator.fit(z.to_numpy(dtype=float))
    corr = _cov_to_corr(np.asarray(fitted.covariance_, dtype=float))
    return pd.DataFrame(corr, index=used.columns, columns=used.columns)


# --------------------------------------------------------------------------------------
# EWMA (own recursion; see module docstring)
# --------------------------------------------------------------------------------------


def _ewma_covariance(x: np.ndarray, lam: float) -> np.ndarray:
    """``x``: ``(T, N)`` raw (non-demeaned) values, ``T >= 1``. Pure, deterministic recursion."""
    s = np.outer(x[0], x[0])
    for row in x[1:]:
        s = lam * s + (1.0 - lam) * np.outer(row, row)
    return s


def ewma_corr(
    returns: pd.DataFrame,
    lam: float = DEFAULT_EWMA_LAMBDA,
    *,
    min_periods: int = 2,
    window: int | None = None,
) -> pd.DataFrame:
    """EWMA correlation as of the last row of ``returns`` (or of ``returns.tail(window)``).

    ``lam`` close to 1 means slow decay (long memory); ``config/experiment.yaml``'s
    ``ewma_lambda`` (0.94) is the project's own default, passed explicitly by callers rather than
    imported here (this module does not depend on ``etfcorr.config``, keeping it a pure numerical
    library with no I/O and nothing to isolate behind an injectable seam, unlike every earlier
    step's Parquet/typer-touching modules).
    """
    if not 0.0 < lam < 1.0:
        raise CorrelationError(f"lam must be in (0, 1), got {lam}")
    used = _prepare(returns, window=window, min_rows=max(2, min_periods))
    cov = _ewma_covariance(used.to_numpy(dtype=float), lam)
    corr = _cov_to_corr(cov)
    return pd.DataFrame(corr, index=used.columns, columns=used.columns)


# --------------------------------------------------------------------------------------
# The one interface: estimate(returns, method, **kw) -> DataFrame
# --------------------------------------------------------------------------------------


def estimate(
    returns: pd.DataFrame,
    method: CorrMethod,
    *,
    window: int | None = None,
    backend: Backend = "sklearn",
    **kw: Any,
) -> pd.DataFrame:
    """The estimator zoo behind one interface (``project_plan.md`` section 6.4).

    Always returns a labelled, symmetric, unit-diagonal correlation ``DataFrame`` (checked via
    :func:`validate_correlation_matrix` before it is returned -- a postcondition, not just a
    docstring claim). ``window`` restricts to the trailing ``window`` rows of ``returns`` first
    (see the module docstring); every method then uses the complete-case rows of whatever frame
    results. Extra ``**kw`` are passed to the underlying estimator: ``lam``/``min_periods`` for
    ``"ewma"``, constructor keywords for ``sklearn.covariance.LedoitWolf``/``OAS`` for
    ``"ledoit_wolf"``/``"oas"``, nothing for ``"pearson"``/``"spearman"``/``"kendall"`` (passing
    any raises :class:`CorrelationError`, so a typo in a keyword is not silently ignored).
    """
    if method not in CORR_METHODS:
        raise CorrelationError(f"unknown method {method!r}; expected one of {CORR_METHODS}")

    if method in ("pearson", "spearman", "kendall"):
        if kw:
            raise CorrelationError(f"{method!r} takes no extra keyword arguments, got {sorted(kw)}")
        used = _prepare(returns, window=window, min_rows=2)
        corr = _pandas_corr(used, method)
    elif method in ("ledoit_wolf", "oas"):
        if backend not in IMPLEMENTED_BACKENDS:
            raise CorrelationError(
                f"backend {backend!r} is not implemented offline; only {IMPLEMENTED_BACKENDS} is "
                "available in this session (see the module docstring's deviation note on skfolio)"
            )
        used = _prepare(returns, window=window, min_rows=2)
        corr = _shrinkage_corr(used, method, **kw)
    elif method == "ewma":
        corr = ewma_corr(returns, kw.pop("lam", DEFAULT_EWMA_LAMBDA), window=window, **kw)
    else:  # pragma: no cover - guarded by the membership check above
        raise CorrelationError(f"unhandled method {method!r}")

    validate_correlation_matrix(corr)
    return corr


# --------------------------------------------------------------------------------------
# Reporting: estimator, window, T, N, condition number, PSD -- for every reported matrix
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelationReport:
    """Everything a skeptical reader will ask about one reported correlation matrix."""

    corr: pd.DataFrame
    method: CorrMethod
    window: int | None
    n_obs: int
    n_assets: int
    condition_number: float
    is_psd: bool
    min_eigenvalue: float


def corr_report(
    returns: pd.DataFrame,
    method: CorrMethod,
    *,
    window: int | None = None,
    backend: Backend = "sklearn",
    **kw: Any,
) -> CorrelationReport:
    """:func:`estimate` plus the T/N/condition-number/PSD diagnostics, bundled for reporting."""
    corr = estimate(returns, method, window=window, backend=backend, **kw)
    diag = eigen_diagnostics(corr)
    sliced = returns if window is None else returns.tail(window)
    n_obs = len(_complete_case(sliced))
    return CorrelationReport(
        corr=corr,
        method=method,
        window=window,
        n_obs=n_obs,
        n_assets=corr.shape[0],
        condition_number=diag.condition_number,
        is_psd=diag.is_psd,
        min_eigenvalue=diag.min_eigenvalue,
    )


def summary_line(report: CorrelationReport) -> str:
    """One human-readable line: estimator, window, T, N, condition number, PSD."""
    window = "full sample" if report.window is None else f"last {report.window}"
    psd = "PSD" if report.is_psd else "NOT PSD"
    return (
        f"{report.method:<11} window={window:<12} T={report.n_obs:<5} N={report.n_assets:<4} "
        f"cond={report.condition_number:>10.3g}  {psd} (min eigenvalue {report.min_eigenvalue:.3g})"
    )
