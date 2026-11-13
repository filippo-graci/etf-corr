"""Step 4.5: lead-lag cross-correlations with false-discovery-rate control (exploratory).

Question served: ``EFT_project_v2.md`` section 5, analysis (6), "lead-lag cross-correlations with
FDR control". Does the lagged return of one ETF help to predict the current return of another,
beyond what the common market factor and each ETF's own autocorrelation already imply? Exploratory:
**not part of the pre-registered H1-H6**. It supplies (a) a descriptive map of daily lead-lag among
the core ETFs and (b) a screen for cross-asset lagged features in Phases 5-6, which is why the
real-data script fixes *discovery on the training window only* and keeps the final test window
unread.

**Statistic.** With standardised returns ``u`` (mean 0, sd 1 over the analysis window) the
directional lead-lag correlation at lag ``k`` is ``c_ij(k) = mean_t u_j,t * u_i,t-k`` ("i leads j").
Tested alone (``statistic="raw"``) it is contaminated: in a one-factor model ``r_i = b_i f + e_i``
the cross-autocovariance is ``Cov(r_j,t, r_i,t-k) = b_i b_j gamma_f(k)``, which is **symmetric in
(i, j)**, so any autocorrelation of the common factor produces "lead-lag" between every pair in both
directions with no asset leading another (the decomposition of cross- versus own-autocovariance is
the point of Lo and MacKinlay (1990); the one-line derivation here is mine and is checked by
simulation in the tests). The primary statistic is therefore the **asymmetry** ``D_ij(k) = c_ij(k) -
c_ji(k)`` for each unordered pair (``statistic="asymmetry"``, antisymmetric: ``D_ji = -D_ij``,
positive means the first-listed asset leads the second): the common-factor term cancels exactly.
What it does **not** remove, **[Inference]**: a pair-specific delayed reaction to a common factor
(stale prices, nonsynchronous trading, thin ETFs) shows up as the better-priced asset "leading" the
staler one, so a hit is a predictive relation, not a causal one.

**Inference.** Each test statistic is the mean of a product series ``z_t`` (the lag is built into
the product, so block resampling never breaks a lag across a block boundary). The standard error is
the standard deviation of the mean over ``n_bootstrap`` stationary-bootstrap replicates (Politis and
Romano (1994), the project's ``stationary_bootstrap_indices`` and block-length heuristic from
``horizon_analysis``), with **one set of index draws shared by all tests** (the cross-dependence
among the tests is preserved, and the whole family costs one matrix product). The p-value is
two-sided from ``z = estimate / se`` against the normal law. A percentile-bootstrap p-value is not
used: with ``m`` in the thousands the BH threshold for the top-ranked test is far below ``1 /
(n_bootstrap + 1)``. The normal tail is an approximation for fat-tailed, volatility-clustered data;
its calibration (size at 5%, 1%, 0.1% and false-discovery frequency under a GARCH-t null) is
measured in ``scripts/lead_lag_calibration_4_5.py`` and recorded in ``docs/checks/4.5.md``, not
assumed. A circular-shift permutation null is not used: it destroys the contemporaneous correlation
and tests independence, a stronger null than "no asymmetry".

**Multiplicity.** Benjamini-Hochberg (1995) q-values (primary, as in ``nonlinear.py``) and
Benjamini-Yekutieli (2001) q-values (valid under arbitrary dependence, printed as the conservative
check), both from ``scipy.stats.false_discovery_control``. The family is every (pair, lag) of one
scan.

**Replication and agreement.** :func:`replication_table` re-tests the discovered (pair, lag) cells
in a disjoint window with a one-sided test in the discovery direction (BH over the discovered set).
:func:`agreement_test` asks the question that does not need any single cell to survive: across all
tests, do the discovery z-statistics and the validation z-statistics rank-correlate? The null
reference comes from the two windows' centred bootstrap replicates (independent windows), which
carry the dependence among the tests.

Citation status (title, authors, venue, year checked by search/fetch on 2026-10-04;
``REFERENCES_SEED.md`` has the tags): Lo and MacKinlay (1990), Review of Financial Studies 3(2),
175-205 -- CHECKED (one source lists 3(1) and 206, the DOI is 10.1093/rfs/3.2.175); Boudoukh,
Richardson and Whitelaw (1994), Review of Financial Studies 7(3), 539-573 -- CHECKED (one listing);
Hou (2007), Review of Financial Studies 20(4), 1113-, industry lead-lag among stocks -- CHECKED for
existence, volume, issue and first page (pages 1113-1138 CHECKED on Crossref 2026-11-02); Benjamini
and Yekutieli (2001), Annals of Statistics 29(4), 1165-1188 -- CHECKED; Benjamini and Hochberg
(1995) and Politis and Romano (1994) were CHECKED in earlier steps. Chordia and Swaminathan (2000)
could not be verified and is not cited. No third-party code is adapted (no licence flag).

Limits (state wherever an output is used): in-sample, descriptive; **[Inference]** a planted or
real lead of A over B also appears as a (weaker) lead of every asset correlated with A over B,
because B inherits A's lagged information through the correlation (seen in the smoke run on
synthetic data), so a cluster of cells around one follower points to one source, not to many; one
window gives power only for large effects (see :attr:`LeadLagScan.minimum_detectable`); holdings
overlap, stale ETF prices, asynchronous closes (none among the core US-listed ETFs, but the
underlying baskets of LIT, COPX, REMX trade abroad) and adjusted-price revisions are standing
caveats; the standardisation uses window statistics, so a scan is a whole-window statistic and not
a feature function (a Phase 5 lagged-return feature is a plain ``shift`` and is leakage-tested
there); a rank transform (Spearman-type) is the outlier-robust variant; weekly returns are the same
code on weekly data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from etfcorr.correlation.horizon_analysis import (
    default_mean_block_length,
    stationary_bootstrap_indices,
)

__all__ = [
    "Agreement",
    "LeadLagError",
    "LeadLagScan",
    "agreement_test",
    "bootstrap_means",
    "lagged_products",
    "lead_lag_scan",
    "replication_table",
    "scan_summary_lines",
    "simulate_lead_lag",
    "slice_window",
    "standardise",
]

DEFAULT_LAGS: tuple[int, ...] = (1, 2, 3, 4, 5)
DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_ALPHA = 0.05
MIN_EFFECTIVE_OBS = 30

Statistic = Literal["asymmetry", "raw"]
Transform = Literal["none", "rank"]

TABLE_COLUMNS: tuple[str, ...] = (
    "asset_a",
    "asset_b",
    "lag",
    "estimate",
    "se",
    "z",
    "p_value",
    "q_bh",
    "q_by",
    "reject_bh",
    "reject_by",
    "leader",
    "follower",
)


class LeadLagError(ValueError):
    """Invalid input to, or an invalid configuration of, a lead-lag call."""


# --------------------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------------------


def _check_frame(returns: pd.DataFrame) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise LeadLagError(f"returns must be a pandas DataFrame, got {type(returns)}")
    if returns.shape[1] < 2:
        raise LeadLagError("returns must have at least 2 columns to form pairs")
    if not returns.index.is_monotonic_increasing or not returns.index.is_unique:
        raise LeadLagError("returns index must be strictly increasing and unique")
    if returns.columns.has_duplicates:
        raise LeadLagError("returns has duplicate column names")


def _check_lags(lags: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(k) for k in lags)
    if not out:
        raise LeadLagError("lags must not be empty")
    if any(k < 1 for k in out):
        raise LeadLagError(f"every lag must be >= 1, got {out}")
    if len(set(out)) != len(out):
        raise LeadLagError(f"lags must be distinct, got {out}")
    return out


def slice_window(returns: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Rows dated ``start <= date <= end`` (both inclusive). Raises on an empty selection.

    The single place where an analysis window is cut, so a discovery scan cannot see rows dated
    after ``end``: the result is a pure function of the rows inside the window.
    """
    _check_frame(returns)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    if lo > hi:
        raise LeadLagError(f"start {start} is after end {end}")
    mask = np.asarray((returns.index >= lo) & (returns.index <= hi))
    out = returns.loc[mask]
    if len(out) == 0:
        raise LeadLagError(f"no rows between {start} and {end}")
    return out


def standardise(returns: pd.DataFrame, transform: Transform = "none") -> np.ndarray:
    """Window-standardised returns as an ``(n, N)`` float array (mean 0, sd 1 per column).

    ``transform="rank"`` first replaces each column by its (average-tie) ranks, so that the product
    means below become Spearman-type cross-correlations, bounded in the influence of single days.
    NaN, infinite values and constant columns are rejected: dropping rows would break the lag
    adjacency.
    """
    _check_frame(returns)
    x = returns.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise LeadLagError("returns contain NaN or infinite values; lags need consecutive rows")
    if transform == "rank":
        x = np.asarray(stats.rankdata(x, axis=0), dtype=float)
    elif transform != "none":
        raise LeadLagError(f"transform must be 'none' or 'rank', got {transform!r}")
    sd = x.std(axis=0, ddof=1)
    if (sd <= 0.0).any():
        raise LeadLagError("a column is constant over the window")
    return np.asarray((x - x.mean(axis=0)) / sd, dtype=float)


# --------------------------------------------------------------------------------------
# Products and bootstrap
# --------------------------------------------------------------------------------------


def lagged_products(
    u: np.ndarray, lags: Sequence[int], statistic: Statistic = "asymmetry"
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Product series ``z`` (rows ``t = max(lags)..n-1``) and the ``(i, j, lag)`` label per column.

    ``raw``: ``z[t] = u[t, j] * u[t-k, i]`` for every ordered pair ``i != j`` ("i leads j").
    ``asymmetry``: ``z[t] = u[t, j] * u[t-k, i] - u[t, i] * u[t-k, j]`` for ``i < j``.
    The column mean is the estimate. All lags share the same rows, so one resampling index applies
    to every column.
    """
    ks = _check_lags(lags)
    if u.ndim != 2 or u.shape[1] < 2:
        raise LeadLagError("u must be a 2-D array with at least 2 columns")
    n, p = u.shape
    kmax = max(ks)
    if n - kmax < MIN_EFFECTIVE_OBS:
        raise LeadLagError(f"need at least {MIN_EFFECTIVE_OBS} usable rows, got {n - kmax}")
    cur = u[kmax:]
    iu, ju = np.triu_indices(p, k=1)
    off_i, off_j = np.nonzero(~np.eye(p, dtype=bool))
    blocks: list[np.ndarray] = []
    labels: list[tuple[int, int, int]] = []
    for k in ks:
        past = u[kmax - k : n - k]
        prod = past[:, :, None] * cur[:, None, :]  # prod[t, i, j] = u[t-k, i] * u[t, j]
        if statistic == "asymmetry":
            blocks.append(prod[:, iu, ju] - prod[:, ju, iu])
            labels += [(int(i), int(j), k) for i, j in zip(iu, ju, strict=True)]
        elif statistic == "raw":
            blocks.append(prod[:, off_i, off_j])
            labels += [(int(i), int(j), k) for i, j in zip(off_i, off_j, strict=True)]
        else:
            raise LeadLagError(f"statistic must be 'asymmetry' or 'raw', got {statistic!r}")
    return np.concatenate(blocks, axis=1), labels


def bootstrap_means(
    z: np.ndarray, mean_block_length: float, n_bootstrap: int, seed: int
) -> np.ndarray:
    """``(n_bootstrap, M)`` stationary-bootstrap means of every column of ``z``.

    The same resampled index vector is applied to all columns (``weights @ z`` with
    ``weights[b, t]`` the share of draw ``b`` that equals row ``t``).
    """
    if n_bootstrap < 2:
        raise LeadLagError(f"n_bootstrap must be >= 2, got {n_bootstrap}")
    if mean_block_length <= 0:
        raise LeadLagError(f"mean_block_length must be > 0, got {mean_block_length}")
    n = z.shape[0]
    rng = np.random.default_rng(seed)
    weights = np.zeros((n_bootstrap, n), dtype=float)
    for b in range(n_bootstrap):
        idx = stationary_bootstrap_indices(n, mean_block_length, rng)
        weights[b] = np.bincount(idx, minlength=n) / n
    return np.asarray(weights @ z, dtype=float)


# --------------------------------------------------------------------------------------
# The scan
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LeadLagScan:
    """Result of :func:`lead_lag_scan`; ``table`` and ``z`` / ``null_z`` columns share one order."""

    table: pd.DataFrame
    z: np.ndarray
    null_z: np.ndarray | None
    statistic: Statistic
    transform: Transform
    lags: tuple[int, ...]
    n_obs: int
    n_effective: int
    mean_block_length: float
    n_bootstrap: int
    alpha: float
    seed: int

    @property
    def n_tests(self) -> int:
        return len(self.table)

    @property
    def minimum_detectable(self) -> float:
        """``|estimate|`` the top-ranked test needs to pass BH at ``alpha`` (median se)."""
        threshold = float(stats.norm.isf(self.alpha / (2.0 * self.n_tests)))
        return threshold * float(np.median(self.table["se"].to_numpy(dtype=float)))

    def top(self, n: int = 10) -> pd.DataFrame:
        """The ``n`` smallest p-values."""
        return self.table.sort_values("p_value", kind="stable").head(n).reset_index(drop=True)


def lead_lag_scan(
    returns: pd.DataFrame,
    *,
    lags: Sequence[int] = DEFAULT_LAGS,
    statistic: Statistic = "asymmetry",
    transform: Transform = "none",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    mean_block_length: float | None = None,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    sectors: Mapping[str, str] | None = None,
    keep_null: bool = False,
) -> LeadLagScan:
    """Test every (pair, lag) for lead-lag; BH and BY q-values over the whole family.

    ``returns``: consecutive-row returns of one horizon (no NaN), already restricted to the analysis
    window (see :func:`slice_window`). ``keep_null=True`` stores the centred, standardised bootstrap
    replicates needed by :func:`agreement_test`.
    """
    ks = _check_lags(lags)
    if not 0.0 < alpha < 1.0:
        raise LeadLagError(f"alpha must be in (0, 1), got {alpha}")
    u = standardise(returns, transform)
    z_prod, labels = lagged_products(u, ks, statistic)
    n_eff = z_prod.shape[0]
    block = default_mean_block_length(n_eff) if mean_block_length is None else mean_block_length
    estimate = z_prod.mean(axis=0)
    means = bootstrap_means(z_prod, block, n_bootstrap, seed)
    se = means.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0.0, estimate / se, 0.0)
        null_z = np.where(se > 0.0, (means - estimate) / se, 0.0) if keep_null else None
    p = np.asarray(2.0 * stats.norm.sf(np.abs(z)), dtype=float)
    q_bh = np.asarray(stats.false_discovery_control(p, method="bh"), dtype=float)
    q_by = np.asarray(stats.false_discovery_control(p, method="by"), dtype=float)

    names = [str(c) for c in returns.columns]
    a = [names[i] for i, _, _ in labels]
    b = [names[j] for _, j, _ in labels]
    first_leads = estimate >= 0.0 if statistic == "asymmetry" else np.ones(len(a), dtype=bool)
    table = pd.DataFrame(
        {
            "asset_a": a,
            "asset_b": b,
            "lag": [k for _, _, k in labels],
            "estimate": estimate,
            "se": se,
            "z": z,
            "p_value": p,
            "q_bh": q_bh,
            "q_by": q_by,
            "reject_bh": q_bh <= alpha,
            "reject_by": q_by <= alpha,
            "leader": np.where(first_leads, a, b),
            "follower": np.where(first_leads, b, a),
        }
    )
    if sectors is not None:
        missing = sorted(set(names) - set(sectors))
        if missing:
            raise LeadLagError(f"sectors has no entry for {missing}")
        table["cross_sector"] = [sectors[x] != sectors[y] for x, y in zip(a, b, strict=True)]
    return LeadLagScan(
        table=table,
        z=z,
        null_z=null_z,
        statistic=statistic,
        transform=transform,
        lags=ks,
        n_obs=len(returns),
        n_effective=n_eff,
        mean_block_length=float(block),
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=seed,
    )


# --------------------------------------------------------------------------------------
# Replication and agreement across two disjoint windows
# --------------------------------------------------------------------------------------


def _same_family(a: LeadLagScan, b: LeadLagScan) -> None:
    keys = ["asset_a", "asset_b", "lag"]
    if (
        a.statistic != b.statistic
        or a.transform != b.transform
        or a.lags != b.lags
        or not a.table[keys].equals(b.table[keys])
    ):
        raise LeadLagError(
            "the two scans do not test the same family (statistic, transform, lags, pairs)"
        )


def replication_table(discovery: LeadLagScan, validation: LeadLagScan) -> pd.DataFrame:
    """Re-test the BH-discovered cells of ``discovery`` in ``validation`` (one-sided).

    One row per discovered cell. ``p_one_sided`` is the validation p-value in the discovery
    direction; ``q_rep`` is BH over the discovered set; ``replicated`` requires ``q_rep <= alpha``
    of the discovery scan. Empty frame (same columns) when nothing was discovered. The discovered
    set depends on the discovery scan only.
    """
    _same_family(discovery, validation)
    cols = [
        "asset_a",
        "asset_b",
        "lag",
        "estimate_disc",
        "z_disc",
        "q_disc",
        "estimate_val",
        "z_val",
        "p_one_sided",
        "q_rep",
        "replicated",
    ]
    hit = discovery.table["reject_bh"].to_numpy(dtype=bool)
    if not hit.any():
        return pd.DataFrame(columns=cols)
    d = discovery.table.loc[hit]
    v = validation.table.loc[hit]
    sign = np.sign(d["estimate"].to_numpy(dtype=float))
    signed_z = sign * v["z"].to_numpy(dtype=float)
    p1 = np.asarray(stats.norm.sf(signed_z), dtype=float)
    q = np.asarray(stats.false_discovery_control(p1, method="bh"), dtype=float)
    return pd.DataFrame(
        {
            "asset_a": d["asset_a"].to_numpy(),
            "asset_b": d["asset_b"].to_numpy(),
            "lag": d["lag"].to_numpy(),
            "estimate_disc": d["estimate"].to_numpy(),
            "z_disc": d["z"].to_numpy(),
            "q_disc": d["q_bh"].to_numpy(),
            "estimate_val": v["estimate"].to_numpy(),
            "z_val": v["z"].to_numpy(),
            "p_one_sided": p1,
            "q_rep": q,
            "replicated": (q <= discovery.alpha) & (signed_z > 0.0),
        }
    )


@dataclass(frozen=True)
class Agreement:
    """Spearman correlation of the z-statistics of two windows and its bootstrap-null p-value."""

    statistic: float
    p_value: float
    n_tests: int
    n_null: int


def _row_ranks(x: np.ndarray) -> np.ndarray:
    return np.asarray(stats.rankdata(x, axis=1), dtype=float)


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    num = (ac * bc).sum(axis=1)
    den = np.sqrt((ac**2).sum(axis=1) * (bc**2).sum(axis=1))
    return np.asarray(num / den, dtype=float)


def agreement_test(discovery: LeadLagScan, validation: LeadLagScan) -> Agreement:
    """Do the z-statistics of two disjoint windows rank-correlate across the whole family?

    Both scans need ``keep_null=True`` and different seeds. One-sided (positive agreement):
    ``p = (1 + #{null >= observed}) / (1 + n_null)``, the null being the Spearman correlation
    between the centred bootstrap replicates of the two windows (independent data, internal
    dependence of the tests preserved).
    """
    _same_family(discovery, validation)
    if discovery.null_z is None or validation.null_z is None:
        raise LeadLagError("both scans need keep_null=True")
    if discovery.seed == validation.seed:
        raise LeadLagError("use different seeds for the two scans")
    obs = float(stats.spearmanr(discovery.z, validation.z).statistic)
    n_null = min(discovery.null_z.shape[0], validation.null_z.shape[0])
    null = _row_corr(_row_ranks(discovery.null_z[:n_null]), _row_ranks(validation.null_z[:n_null]))
    exceed = int(np.sum(null >= obs - 1e-12))
    return Agreement(
        statistic=obs,
        p_value=(1 + exceed) / (1 + n_null),
        n_tests=discovery.n_tests,
        n_null=n_null,
    )


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def scan_summary_lines(scan: LeadLagScan) -> list[str]:
    """Plain-text description of a scan, with the caveats that must travel with it."""
    t = scan.table
    expected = scan.alpha * scan.n_tests
    lines = [
        f"lead-lag scan: statistic={scan.statistic}, transform={scan.transform}, lags={scan.lags}, "
        f"n_obs={scan.n_obs} (usable {scan.n_effective}), tests={scan.n_tests}",
        f"stationary bootstrap: {scan.n_bootstrap} draws, mean block length "
        f"{scan.mean_block_length:.1f}, seed {scan.seed}; p-values two-sided normal on "
        "estimate / bootstrap se",
        f"p < {scan.alpha}: {int((t['p_value'] < scan.alpha).sum())} (about {expected:.0f} "
        "expected by chance under the null, if the normal tail is well calibrated)",
        f"FDR level {scan.alpha}: BH rejects {int(t['reject_bh'].sum())}, "
        f"BY rejects {int(t['reject_by'].sum())}",
        f"minimum |estimate| the top-ranked test could pass BH with (median se): "
        f"{scan.minimum_detectable:.3f}",
        "exploratory, in-sample, descriptive; a hit is a predictive relation, not a causal one",
    ]
    return lines


# --------------------------------------------------------------------------------------
# Simulation (tests and calibration only)
# --------------------------------------------------------------------------------------


def simulate_lead_lag(
    n_days: int,
    n_assets: int = 8,
    *,
    market_loading: float = 0.6,
    factor_ar: float = 0.0,
    garch: tuple[float, float] | None = (0.05, 0.93),
    df: float | None = 5.0,
    planted: Sequence[tuple[int, int, int, float]] = (),
    seed: int = 0,
    burn: int = 200,
    start: str = "2016-01-04",
) -> pd.DataFrame:
    """Daily returns from a one-factor model with optional volatility clustering and lead-lag.

    ``r0_i = loading * f + sqrt(1 - loading^2) * x_i`` (unit variance before the 0.01 scale), the
    factor ``f`` is AR(1) with coefficient ``factor_ar`` (unit variance), and the shocks have
    GARCH(1,1) volatility (``garch=(alpha, beta)``, unit unconditional variance; ``None`` =
    constant) and Student-t (``df``, unit variance; ``None`` = Gaussian) innovations. ``planted``
    entries ``(leader, follower, lag, theta)`` add ``theta * r[t - lag, leader]`` to the follower's
    return, applied in time order. Columns are ``A00, A01, ...``.
    """
    if n_days < 2 or n_assets < 2:
        raise LeadLagError("need n_days >= 2 and n_assets >= 2")
    if not 0.0 <= market_loading < 1.0 or not -1.0 < factor_ar < 1.0:
        raise LeadLagError("market_loading must be in [0, 1) and factor_ar in (-1, 1)")
    if df is not None and df <= 2.0:
        raise LeadLagError("df must be > 2 for a finite variance")
    if garch is not None and (garch[0] < 0 or garch[1] < 0 or garch[0] + garch[1] >= 1.0):
        raise LeadLagError("garch must satisfy alpha, beta >= 0 and alpha + beta < 1")
    for lead, follow, lag, _ in planted:
        if not (0 <= lead < n_assets and 0 <= follow < n_assets and lead != follow and lag >= 1):
            raise LeadLagError(f"invalid planted entry {(lead, follow, lag)}")
    rng = np.random.default_rng(seed)
    total = n_days + burn
    k = n_assets + 1
    if df is None:
        eps = rng.standard_normal((total, k))
    else:
        eps = rng.standard_t(df, size=(total, k)) * np.sqrt((df - 2.0) / df)
    x = np.empty((total, k))
    if garch is None:
        x[:] = eps
    else:
        alpha, beta = garch
        omega = 1.0 - alpha - beta
        var = np.ones(k)
        x[0] = eps[0]
        for t in range(1, total):
            var = omega + alpha * x[t - 1] ** 2 + beta * var
            x[t] = np.sqrt(var) * eps[t]
    factor = np.empty(total)
    factor[0] = x[0, 0]
    scale = np.sqrt(1.0 - factor_ar**2)
    for t in range(1, total):
        factor[t] = factor_ar * factor[t - 1] + scale * x[t, 0]
    r = market_loading * factor[:, None] + np.sqrt(1.0 - market_loading**2) * x[:, 1:]
    if planted:
        for t in range(total):
            for lead, follow, lag, theta in planted:
                if t >= lag:
                    r[t, follow] += theta * r[t - lag, lead]
    index = pd.bdate_range(start, periods=n_days, name="date")
    columns = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(0.01 * r[burn:], index=index, columns=columns)
