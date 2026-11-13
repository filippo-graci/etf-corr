"""Step 3.1: Marchenko-Pastur denoising, market-mode detoning, and eigen-spectrum diagnostics for
a single correlation matrix -- the random-matrix-theory (RMT) "cleaning" building block that H1
(market-mode removal, ``config/experiment.yaml``'s own phase-3 hypothesis) and, later, H5
(denoised/RMT-cleaned covariances for minimum-variance portfolios, phase 8) both need.

Question served: ``correlation/estimators.py`` (step 2.2) answers "which correlation measure, for
exactly these rows?" and returns a *raw* sample-based matrix -- noisy whenever T is not much
larger than N (this project's own standard, section 2: "the raw sample correlation is noisy when
T/N is small... default stack: Pearson on log returns with shrinkage... and random-matrix
cleaning"). This module answers the next question: "how much of that matrix's eigen-structure is
signal versus sampling noise, and what does the matrix look like once the noise (denoising) or the
common market factor (detoning) is removed?" It is a pure function of a matrix (plus T, where the
Marchenko-Pastur edge needs it) -- like ``estimators.py`` before it, no I/O, no dependency on
``etfcorr.config``/``etfcorr.data.*``, and not itself an H1 test: the cross-horizon similarity
engine that actually assembles H1's before/after-detoning comparison with stationary-bootstrap
confidence intervals (``config/experiment.yaml``'s H1 test text) is left to the next step, which
extends ``horizon_analysis.py`` -- exactly the deferral that module's own docstring already
recorded ("cross-horizon similarity/detoning... needs correlation/rmt.py"). This module delivers
the tool that step needs, not the test itself.

**Two standing project caveats restated here because this module is where they first become
operationally relevant** (the project's own custom instructions, section 2, "Correlation
estimation"): (1) *"Detoned (market-removed) correlations are relative, not absolute; between-
sector values turn negative by construction."* :func:`detone_corr` subtracts a positive-
semi-definite rank-``n_market`` piece (the market mode) from every pairwise entry and then
renormalizes to unit diagonal; there is no reason the renormalized residual entries stay positive,
and for two assets in different sectors (whose only shared structure *was* mostly the market mode)
the residual is expected to go negative. This is not a data problem, and it is demonstrated
directly in this module's own tests on real fixture data, not merely asserted. (2) Do not confuse
this module's "detoning" (removing the *common market eigenmode* from a correlation matrix) with
``correlation/overlap.py``'s "residual correlation" (removing the *holdings-overlap-predicted*
part of correlation via linear regression, step 2.5) -- both produce a "residual" correlation-like
output and both are trying to separate "genuine" co-movement from something else, but the
*something else* and the *method* are unrelated (an RMT eigenmode here; a regression fit there).
They can be composed (detone, then check whether overlap still explains what remains) but neither
is a special case of the other, and this module does not depend on ``overlap.py`` or vice versa.

**Method notes and citation status** (see ``REFERENCES_SEED.md`` section A for the full seed
bibliography; every reference below was checked -- existence, authors, venue -- by a web search
this session, not carried over from memory):

* **Marchenko-Pastur edge.** Marchenko, V. A., Pastur, L. A. (1967), *Distribution of eigenvalues
  for some sets of random matrices*, Mat. Sb. 72(114):4 (English translation: Math. USSR-Sb.
  1(4), 457-483), DOI 10.1070/SM1967v001n04ABEH001994 -- **CHECKED** this session. Gives the
  asymptotic support ``[sigma2*(1-sqrt(q))^2, sigma2*(1+sqrt(q))^2]`` (``q = N/T``) of the
  eigenvalue distribution of a sample covariance/correlation matrix estimated from pure noise;
  :func:`marchenko_pastur_edge` implements only the *upper* edge, since this project's use (count
  eigenvalues that are "signal," per Laloux et al. and Bun et al. below) only needs it -- the
  lower edge is not computed or used anywhere in this module.
* **RMT applied to financial correlation matrices.** Laloux, L., Cizeau, P., Bouchaud, J.-P.,
  Potters, M. (1999), *Noise Dressing of Financial Correlation Matrices*, Phys. Rev. Lett. 83,
  1467-1470, DOI 10.1103/PhysRevLett.83.1467 -- **CHECKED** this session (already listed
  UNVERIFIED in ``REFERENCES_SEED.md``; upgrade its status there). Plerou, V., Gopikrishnan, P.,
  Rosenow, B., Amaral, L. A. N., Guhr, T., Stanley, H. E. (2002), *Random Matrix Approach to
  Cross Correlations in Financial Data*, Phys. Rev. E 65, 066126, DOI
  10.1103/PhysRevE.65.066126 -- **CHECKED** this session (upgrades the 2002 entry already seeded
  in ``REFERENCES_SEED.md`` section A, previously misattributed volume left blank). Both establish
  that the empirical spectrum of a large financial correlation matrix has a bulk consistent with
  MP noise plus a small number of large outlier eigenvalues (a dominant "market mode" and a few
  sector-like modes), the empirical basis for treating "eigenvalue above the MP edge" as "signal."
  Bun, J., Bouchaud, J.-P., Potters, M. (2017), *Cleaning large correlation matrices: tools from
  random matrix theory*, Physics Reports 666, 1-109, arXiv:1610.08104 -- already **CHECKED** in
  ``REFERENCES_SEED.md``; the review this project's own custom instructions name directly
  ("random-matrix cleaning (Marchenko-Pastur denoising, market-mode removal)").
* **Denoising (constant residual eigenvalue) and detoning.** Implementation follows
  ``project_plan.md`` section 6.2's own draft ``denoise_corr``/``detone_corr``/``cov_to_corr``
  snippets (the project's own prior code, already established; nothing new for
  ``THIRD_PARTY.md``), which in turn follow the published method in López de Prado, M. (2020),
  *Machine Learning for Asset Managers*, Cambridge Elements in Quantitative Finance, Chapter 2,
  "Denoising and Detoning" -- **CHECKED** this session (table of contents fetched directly from
  the publisher, Cambridge University Press; add to ``REFERENCES_SEED.md`` section H, currently
  listed only as an unchecked textbook entry). **Explicitly not read or copied this session**:
  ``mlfinlab`` (the companion code package for that book) is this project's own named
  [AVOID-COPY] example (custom instructions section 3, "e.g. the RSR repository, mlfinlab and its
  forks: never paste or paraphrase code from these; re-implement from the paper"); this module was
  written from the book's published method description and ``project_plan.md``'s own draft
  snippet, not from that repository, and no `mlfinlab` source was opened this session.
  *Denoising* (:func:`denoise_corr`): eigenvalues at or below the MP edge are replaced by their
  common mean (preserving the matrix's trace, hence its unit diagonal after renormalization) --
  the "constant residual eigenvalue" method, the simplest of the book's two denoising options
  (the other, "targeted shrinkage" of only the noise eigenvalues' eigenvectors, is not
  implemented here). *Detoning* (:func:`detone_corr`): the top ``n_market`` eigenmode(s) are
  subtracted outright and the residual is renormalized to unit diagonal -- the book applies this
  as a second, optional step on top of an already-denoised matrix, primarily to make correlation-
  based *clustering* (its own Chapter 4) less dominated by the one factor nearly every asset
  loads on; a minimum-variance *covariance* estimate (this project's H5) should use the denoised-
  but-not-detoned matrix, since detoning discards real, priceable market-factor risk, not noise --
  both functions are offered separately and independently composable here specifically so a
  caller (the H1 engine, step 3.2) can choose which is wanted rather than this module deciding for
  it.
* **Fixed ``sigma2 = 1.0`` by default, a stated simplification.** The book's fuller method fits
  ``sigma2`` (the noise variance) by matching a kernel density estimate of the empirical bulk
  spectrum to the theoretical MP density, appropriate for a general covariance matrix with
  unequal asset variances. Every matrix this module receives is already a *correlation* matrix
  (unit diagonal, mean eigenvalue exactly 1 by construction, checked by
  :func:`~etfcorr.correlation.estimators.validate_correlation_matrix` on every call), so
  ``sigma2 = 1.0`` -- the value implied under the null of "this whole matrix is noise" -- is used
  as the default rather than fitting a KDE against the bulk; ``sigma2`` remains an explicit,
  overridable parameter on every function here, and fitting it properly (a real follow-up, not
  done this session) is left as an open question in ``docs/STATUS.md``.
* **``eigen_spectrum``'s "market-mode share" and "effective number of bets."** Market-mode share
  is simply ``top_eigenvalue / n_assets`` (the fraction of total variance, i.e. of the trace,
  carried by the largest eigenvalue) -- **[Inference]**, a direct PCA variance-explained ratio,
  not a citation-bearing claim. "Effective number of bets" follows Meucci, A. (2009), *Managing
  Diversification*, Risk 22(6), 74-79 (SSRN 1358533) -- **CHECKED** this session, add to
  ``REFERENCES_SEED.md`` section A: the exponential of the Shannon entropy of the normalized
  eigenvalues (``p_i = lambda_i / sum(lambda)``), a diversification measure that equals ``N`` for
  an uncorrelated (identity) matrix and 1 for a matrix with a single dominant mode. **Stated
  narrowing of Meucci's own claim**: Meucci's paper defines this on the *risk contributions of a
  portfolio's principal-component exposures* (a portfolio-level diversification diagnostic); this
  module applies the identical entropy formula directly to a correlation matrix's own eigenvalues
  as a *structural* diagnostic of the matrix itself (how many effectively independent modes does
  this set of assets' correlations show), which is a narrower, simplified use than Meucci's own
  worked example -- flagged so this is not read as reproducing his portfolio-level result.
  **Only reliable for a PSD input**: with a genuinely negative eigenvalue (Kendall's tau is PSD on
  complete rows, see ``estimators.py``; every other method here is PSD by construction or argument),
  the normalized eigenvalues no longer form a valid probability distribution (they can sum to more
  than 1 once negative mass is excluded from the entropy sum) and both ``effective_bets`` and, less
  severely, ``market_mode_share`` should be read as order-of- magnitude only; :class:`EigenSpectrum`
  reports ``is_psd`` (same tolerance convention as
  :func:`~etfcorr.correlation.estimators.eigen_diagnostics`) precisely so this is visible, not
  silently assumed.
* **Not yet delivered here, left to step 3.2**: the actual H1 comparison (correlation-matrix and
  leading-eigenvector similarity across horizons, before versus after detoning, with stationary-
  bootstrap confidence intervals on the difference -- ``config/experiment.yaml``'s H1 test text).
  Borghesi, Marsili, Miccichè (2007) and Tumminello et al. (2007), both already **CHECKED** in
  ``REFERENCES_SEED.md`` section B, are the literature precedent for that comparison and are cited
  there, not repeated here since this module does not implement it.
* **skfolio cross-check, deferred, not implemented as a parameter.** ``project_plan.md`` section
  6.4 names a cross-check against "skfolio ``DenoiseCovariance``/``DetoneCovariance``." Unlike
  ``estimators.py``'s Ledoit-Wolf/OAS (where this project's own decision log already confirms
  skfolio 1.3.0 is installed locally and the *only* obstacle is this planning sandbox's own
  unreachable PyPI), this session could not even confirm those two class names exist in skfolio's
  installed API surface (skfolio cannot be imported here either, for the same reason) -- so, unlike
  ``estimators.py``'s ``backend`` parameter, no matching parameter is added here on the strength of
  an unverified guess at a third-party API (both class names were later confirmed in the
  installed skfolio, 2026-11-02; the cross-check itself is still not built). This is listed as a
  Gate 2 exploratory check (see the Delivery block) rather than built now.

No third-party code is adapted here beyond the project's own already-established
``project_plan.md`` draft snippets; nothing new for ``THIRD_PARTY.md``.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from etfcorr.correlation.estimators import (
    CorrelationReport,
    CorrMethod,
    validate_correlation_matrix,
)

DEFAULT_SIGMA2 = 1.0
PSD_TOL = -1e-8


class RMTError(ValueError):
    """Invalid input to, or an invalid configuration of, an RMT cleaning call."""


# --------------------------------------------------------------------------------------
# Shared primitive: renormalize any symmetric matrix with a positive diagonal to a
# labelled, unit-diagonal correlation matrix (project_plan.md section 6.2's own cov_to_corr)
# --------------------------------------------------------------------------------------


def cov_to_corr(matrix: pd.DataFrame, *, tol: float = 1e-8) -> pd.DataFrame:
    """Rescale rows/columns of a symmetric matrix so the diagonal becomes exactly 1.

    Used here on eigen-*reconstructions* (``vectors @ diag(cleaned_eigenvalues) @ vectors.T``),
    which are symmetric by construction but not unit-diagonal in general, unlike
    ``estimators.py``'s own private ``_cov_to_corr`` (which starts from an actual covariance
    matrix); kept as a separate, public function here rather than importing that private helper,
    since the two call sites have different preconditions worth documenting independently (this
    one must additionally guard against a non-positive diagonal, which a genuine covariance matrix
    never has but an eigen-reconstruction after removing modes can, in principle, produce for a
    heavily-loaded asset -- see the module docstring's detoning note). ``tol`` rejects a diagonal
    entry at or below it, not only strictly negative ones: a residual variance of a few times
    machine epsilon (confirmed by direct construction in this module's own tests -- a perfectly
    correlated pair, detoned by its one non-trivial mode, leaves a diagonal of ``~2e-16``, not
    exactly zero) is floating-point noise, not a real positive variance, and dividing by its square
    root would silently manufacture an unstable, meaningless correlation rather than raising.
    """
    if not isinstance(matrix, pd.DataFrame):
        raise RMTError(f"matrix must be a pandas DataFrame, got {type(matrix)}")
    if matrix.shape[0] != matrix.shape[1]:
        raise RMTError(f"matrix must be square, got shape {matrix.shape}")
    if list(matrix.index) != list(matrix.columns):
        raise RMTError("index and columns must be identical, in the same order")
    values = matrix.to_numpy(dtype=float)
    if not np.allclose(values, values.T, atol=1e-8):
        raise RMTError("matrix is not symmetric")
    diag = np.diag(values)
    bad = [
        str(c) for c, d in zip(matrix.columns, diag, strict=True) if not np.isfinite(d) or d <= tol
    ]
    if bad:
        raise RMTError(
            f"cannot renormalize to a correlation matrix: non-positive, non-finite or "
            f"numerically negligible (<= {tol:.1e}) residual variance for {bad} (this can happen "
            "when removing eigenmode(s) leaves an asset with no remaining variance -- see "
            "detone_corr's docstring)"
        )
    scale = np.sqrt(diag)
    corr = values / np.outer(scale, scale)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=matrix.index, columns=matrix.columns)


# --------------------------------------------------------------------------------------
# Marchenko-Pastur edge and the eigen-spectrum report
# --------------------------------------------------------------------------------------


def marchenko_pastur_edge(n_assets: int, n_obs: int, sigma2: float = DEFAULT_SIGMA2) -> float:
    """Upper edge ``sigma2 * (1 + sqrt(N/T))^2`` of the Marchenko-Pastur noise band.

    Eigenvalues of a correlation/covariance matrix estimated from ``n_obs`` independent draws of
    ``n_assets`` pure-noise variables concentrate, asymptotically, in
    ``[sigma2*(1-sqrt(q))^2, sigma2*(1+sqrt(q))^2]`` with ``q = n_assets / n_obs`` (Marchenko and
    Pastur, 1967, see the module docstring). Only the upper edge is returned (and used anywhere in
    this module): everything this project does with it is "count/replace eigenvalues *above* the
    noise band," per Laloux et al. (1999) and Bun et al. (2017); the lower edge is not needed.
    When ``q >= 1`` (``n_obs <= n_assets``, e.g. this project's own quarterly panel -- the
    project's custom instructions already flag quarterly as ill-conditioned/exploratory-only for
    this reason) the formula stays well-defined but the edge grows large, so aggressive noise
    classification (most or all eigenvalues below it) is the expected, correct behaviour for a
    small-T-relative-to-N sample, not a bug in this function.
    """
    if n_assets < 1:
        raise RMTError(f"n_assets must be >= 1, got {n_assets}")
    if n_obs < 1:
        raise RMTError(f"n_obs must be >= 1, got {n_obs}")
    if sigma2 <= 0:
        raise RMTError(f"sigma2 must be > 0, got {sigma2}")
    q = n_assets / n_obs
    return float(sigma2 * (1.0 + np.sqrt(q)) ** 2)


@dataclass(frozen=True)
class EigenSpectrum:
    """Everything a skeptical reader will ask about one correlation matrix's eigen-structure."""

    eigenvalues: tuple[float, ...]  # descending
    n_assets: int
    n_obs: int
    sigma2: float
    mp_edge: float
    n_signal: int
    is_psd: bool
    market_mode_share: float
    effective_bets: float


def eigen_spectrum(
    corr: pd.DataFrame, n_obs: int, *, sigma2: float = DEFAULT_SIGMA2
) -> EigenSpectrum:
    """Eigenvalue spectrum, MP edge, signal count, market-mode share, effective number of bets.

    ``n_obs`` must be the actual number of complete-case rows the caller estimated ``corr`` from
    (``estimators.CorrelationReport.n_obs``, not ``len(returns)`` if a window or missing data
    reduced the usable sample) -- see the module docstring's ``sigma2`` note for why no KDE fit is
    attempted here. See the module docstring for the PSD caveat on ``effective_bets``.
    """
    validate_correlation_matrix(corr)
    n = corr.shape[0]
    edge = marchenko_pastur_edge(n, n_obs, sigma2)
    values = np.linalg.eigvalsh(corr.to_numpy(dtype=float))[::-1]  # ascending -> descending
    is_psd = bool(values[-1] >= PSD_TOL)
    n_signal = int(np.sum(values > edge))
    total = float(values.sum())
    if total <= 0:
        raise RMTError(f"sum of eigenvalues must be positive to report a spectrum, got {total}")
    p = values / total
    mask = p > 1e-12
    entropy = float(-np.sum(p[mask] * np.log(p[mask])))
    return EigenSpectrum(
        eigenvalues=tuple(float(v) for v in values),
        n_assets=n,
        n_obs=n_obs,
        sigma2=sigma2,
        mp_edge=edge,
        n_signal=n_signal,
        is_psd=is_psd,
        market_mode_share=float(values[0] / n),
        effective_bets=float(np.exp(entropy)),
    )


def spectrum_summary_line(spectrum: EigenSpectrum) -> str:
    """One human-readable line: N, T, MP edge, signal count, market share, effective bets."""
    top = ", ".join(f"{v:.3g}" for v in spectrum.eigenvalues[:3])
    psd = "PSD" if spectrum.is_psd else "NOT PSD"
    return (
        f"N={spectrum.n_assets:<4} T={spectrum.n_obs:<6} sigma2={spectrum.sigma2:<5.3g} "
        f"MP_edge={spectrum.mp_edge:<8.3g} signal={spectrum.n_signal:<3} {psd}  "
        f"market_share={spectrum.market_mode_share:.3f} eff_bets={spectrum.effective_bets:.2f} "
        f"top_eigenvalues=[{top}, ...]"
    )


# --------------------------------------------------------------------------------------
# Denoising: constant residual eigenvalue (replace sub-edge eigenvalues by their mean)
# --------------------------------------------------------------------------------------


def denoise_corr(corr: pd.DataFrame, n_obs: int, *, sigma2: float = DEFAULT_SIGMA2) -> pd.DataFrame:
    """Replace eigenvalues at or below the Marchenko-Pastur edge by their common mean.

    Preserves the trace (hence, after :func:`cov_to_corr`'s renormalization, the unit diagonal)
    while collapsing the "noise" part of the spectrum to a single shared value -- the simplest of
    López de Prado (2020)'s two denoising options (see the module docstring). If every eigenvalue
    is already above the edge, the matrix is returned unchanged (up to floating-point
    reconstruction error) -- there is nothing classified as noise to average. ``n_obs`` and
    ``sigma2`` are passed through to :func:`marchenko_pastur_edge` exactly as in
    :func:`eigen_spectrum` (same value, so the two agree on what counts as signal).
    """
    validate_correlation_matrix(corr)
    edge = marchenko_pastur_edge(corr.shape[0], n_obs, sigma2)
    values, vectors = np.linalg.eigh(corr.to_numpy(dtype=float))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    is_noise = values <= edge
    cleaned = values.copy()
    if is_noise.any():
        cleaned[is_noise] = values[is_noise].mean()
    reconstructed = vectors @ np.diag(cleaned) @ vectors.T
    out = cov_to_corr(pd.DataFrame(reconstructed, index=corr.index, columns=corr.columns))
    validate_correlation_matrix(out)
    return out


@dataclass(frozen=True)
class DenoiseResult:
    """:func:`denoise_corr`'s output bundled with the diagnostics that justify it."""

    corr: pd.DataFrame
    method: CorrMethod
    n_obs: int
    n_assets: int
    sigma2: float
    mp_edge: float
    n_signal: int


def denoise_report(report: CorrelationReport, *, sigma2: float = DEFAULT_SIGMA2) -> DenoiseResult:
    """:func:`denoise_corr` applied to an ``estimators.CorrelationReport``.

    Convenience wrapper for the common case of denoising something ``estimators.corr_report``
    (or ``horizon_analysis.corr_by_horizon``) already produced: pulls ``corr``/``n_obs``/``method``
    from the report rather than requiring the caller to track them separately.
    """
    spectrum = eigen_spectrum(report.corr, report.n_obs, sigma2=sigma2)
    cleaned = denoise_corr(report.corr, report.n_obs, sigma2=sigma2)
    return DenoiseResult(
        corr=cleaned,
        method=report.method,
        n_obs=report.n_obs,
        n_assets=report.n_assets,
        sigma2=sigma2,
        mp_edge=spectrum.mp_edge,
        n_signal=spectrum.n_signal,
    )


# --------------------------------------------------------------------------------------
# Detoning: remove the top n_market eigenmode(s) outright (H1's "market mode")
# --------------------------------------------------------------------------------------


def detone_corr(corr: pd.DataFrame, n_market: int = 1) -> pd.DataFrame:
    """Subtract the top ``n_market`` eigenmode(s) and renormalize to unit diagonal.

    **Read the module docstring's standing caveat before interpreting the output**: between-sector
    (or, more generally, between-cluster) residual entries are expected to go negative -- this is
    what "detoned correlations are relative, not absolute" means in this project's own custom
    instructions, demonstrated directly in this module's tests, not merely asserted. Raises
    :class:`RMTError` (via :func:`cov_to_corr`) if removing the requested mode(s) leaves any
    asset with non-positive residual variance -- possible in principle for an asset whose variance
    is almost entirely explained by the removed mode(s), not expected for a realistic panel but
    not silently tolerated either.
    """
    validate_correlation_matrix(corr)
    n = corr.shape[0]
    if not 1 <= n_market < n:
        raise RMTError(f"n_market must satisfy 1 <= n_market < {n} (n_assets), got {n_market}")
    values, vectors = np.linalg.eigh(corr.to_numpy(dtype=float))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    top_vectors = vectors[:, :n_market]
    market = top_vectors @ np.diag(values[:n_market]) @ top_vectors.T
    residual = corr.to_numpy(dtype=float) - market
    out = cov_to_corr(pd.DataFrame(residual, index=corr.index, columns=corr.columns))
    validate_correlation_matrix(out)
    return out


@dataclass(frozen=True)
class DetoneResult:
    """:func:`detone_corr`'s output bundled with the eigenvalue(s) it removed."""

    corr: pd.DataFrame
    method: CorrMethod
    n_market: int
    market_eigenvalues: tuple[float, ...]


def detone_report(report: CorrelationReport, *, n_market: int = 1) -> DetoneResult:
    """:func:`detone_corr` applied to an ``estimators.CorrelationReport``."""
    values = np.linalg.eigvalsh(report.corr.to_numpy(dtype=float))[::-1]
    cleaned = detone_corr(report.corr, n_market=n_market)
    return DetoneResult(
        corr=cleaned,
        method=report.method,
        n_market=n_market,
        market_eigenvalues=tuple(float(v) for v in values[:n_market]),
    )
