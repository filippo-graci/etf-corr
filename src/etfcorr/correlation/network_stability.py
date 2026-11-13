"""Stability and null references for correlation networks (step 3.6).

Question served: the Phase 3 milestone "MST figure" and the open item "real-data check of
``networks.py``". ``networks.py`` (step 3.3) describes one tree; this module says how much of that
description to believe: how often each MST edge survives resampling, how similar two trees are
compared with chance, whether rolling-window trees persist more or less than i.i.d. row resampling
would give, and how well hierarchical clusters agree with the sector labels. Everything is
descriptive and exploratory; **none of it is part of H1-H6**.

Method
------
* **Edge support** (:func:`bootstrap_mst_edge_support`): the MST of the point-estimate correlation
  matrix, and for each of its edges the share of stationary-bootstrap replicates (Politis and
  Romano 1994, via :func:`etfcorr.correlation.horizon_analysis.stationary_bootstrap_indices`) whose
  own MST contains that edge. Bootstrap edge reliability is the idea of Tumminello et al. (2007);
  this implementation is written from the description, with the project's bootstrap. Also the
  mean and a percentile interval of the edge-survival ratio of each replicate tree against the
  point tree.
* **Survival matrix** (:func:`survival_matrix`): pairwise edge-survival ratios of several trees on
  the same nodes (for example the trees of different horizons).
* **Noise floor** (:func:`noise_mst_survival_null`): the survival ratio between the MSTs of two
  *independent* pure-noise panels with the same N and T, i.e. the similarity two trees show when
  there is no common structure at all. Its mean is ``2 / N`` exactly **[Inference, by symmetry]**:
  the MST of exchangeable noise is a label-exchangeable random tree, so each of the C(N, 2) edges
  is in it with probability (N - 1) / C(N, 2) = 2 / N, and for two independent trees the expected
  shared share is C(N, 2) (2 / N)^2 / (N - 1) = 2 / N (the simulation reproduces it). What the
  simulation adds is the *spread*: the 95th percentile of the noise survival, which is the bar a
  real survival ratio has to clear to count as more than chance.
* **Rolling null** (:func:`rolling_survival_shuffle_null`): the mean rolling-window survival
  (``networks.rolling_mst_survival``) of the real series against the same statistic on row-shuffled
  copies of the returns. Shuffling keeps every window's overlap (consecutive windows share
  ``window - step`` rows in both cases) and the marginal distribution of rows, and destroys serial
  and regime structure. A real mean above the shuffled distribution means the tree persists more
  than i.i.d. rows would give (local persistence of the correlation structure); below means less.
  Two-sided empirical position, not a significance claim: windows overlap and are not independent.
* **Cluster-sector agreement** (:func:`cluster_sector_agreement`): adjusted Rand index (Hubert and
  Arabie 1985) between a hierarchical clustering cut into as many clusters as there are sectors and
  the sector labels, with a permutation p-value that reshuffles the sector labels.

Assumptions and known failure modes
-----------------------------------
* The bootstrap resamples rows i.i.d.-in-blocks: it does not model volatility regimes. Support is
  the support under a stationary world.
* A tree built from a correlation matrix with T close to N (monthly and quarterly) is unstable:
  expect low support and low survival there, which is information, not a bug.
* MST edges are not independent, so edge supports are descriptive; no per-edge p-value is given.
* The cluster test conditions on the clustering method and the number of clusters, and on sector
  labels being the right benchmark (ETFs in one "sector" can be very different, and holdings
  overlap makes some agreement mechanical; see ``overlap.py``).
* No replicate is dropped: an estimator error on a replicate propagates, as in
  ``horizon_analysis`` and ``stability``.

Citations (status)
------------------
* Tumminello, M., Coronnello, C., Lillo, F., Micciche, S., Mantegna, R. N. (2007). Spanning trees
  and bootstrap reliability estimation in correlation-based networks. arXiv physics/0605116.
  CHECKED (title and author list seen on the arXiv listing in step 3.6). The journal reference
  (Int. J. Bifurcation and Chaos 17(7), 2319-2329) CHECKED on Crossref 2026-11-02.
* Hubert, L., Arabie, P. (1985). Comparing partitions. Journal of Classification. CHECKED (title,
  authors and journal on a Semantic Scholar record in step 3.6); volume and pages (2(1), 193-218)
  CHECKED on Crossref 2026-11-02.
* Politis, D. N., Romano, J. P. (1994): CHECKED (step 3.2). Mantegna (1999), Gower-Ross (1969),
  Kruskal (1956): CHECKED (``docs/checks/3.3.md``).

Third-party code: none adapted. ``sklearn.metrics.adjusted_rand_score`` is imported, not copied;
nothing new for ``THIRD_PARTY.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import adjusted_rand_score

from etfcorr.correlation.estimators import estimate
from etfcorr.correlation.horizon_analysis import (
    default_mean_block_length,
    stationary_bootstrap_indices,
)
from etfcorr.correlation.networks import (
    MSTResult,
    build_mst,
    edge_survival_ratio,
    hierarchical_clusters,
    rolling_mst_survival,
)
from etfcorr.correlation.rmt import detone_corr

__all__ = [
    "ClusterAgreement",
    "CorrFn",
    "EdgeSupport",
    "NetworkStabilityError",
    "NoiseSurvivalNull",
    "RollingSurvivalNull",
    "bootstrap_mst_edge_support",
    "cluster_sector_agreement",
    "detoned_pearson",
    "noise_mst_survival_null",
    "pearson",
    "rolling_survival_shuffle_null",
    "survival_matrix",
]

CorrFn = Callable[[pd.DataFrame], pd.DataFrame]

MIN_OBS = 8
MIN_BOOTSTRAP = 20


class NetworkStabilityError(ValueError):
    """Invalid input to a network-stability function."""


def pearson(sample: pd.DataFrame) -> pd.DataFrame:
    """Complete-case Pearson matrix (the default ``corr_fn``)."""
    return estimate(sample, "pearson")


def detoned_pearson(sample: pd.DataFrame, n_market: int = 1) -> pd.DataFrame:
    """Pearson matrix with the top ``n_market`` eigenmodes removed (a relative-scale matrix)."""
    return detone_corr(estimate(sample, "pearson"), n_market=n_market)


# --------------------------------------------------------------------------- edge support


@dataclass(frozen=True, eq=False)
class EdgeSupport:
    """Bootstrap reliability of the edges of the point-estimate MST.

    ``edges`` has the columns of ``MSTResult.edges`` plus ``support`` (share of replicate trees
    containing the edge). ``mean_survival`` and its percentile interval summarise the edge-survival
    ratio of each replicate tree against the point tree.
    """

    mst: MSTResult
    edges: pd.DataFrame
    mean_survival: float
    survival_lower: float
    survival_upper: float
    n_bootstrap: int
    mean_block_length: float
    n_obs: int

    @property
    def support(self) -> dict[tuple[str, str], float]:
        pairs = zip(self.edges["source"], self.edges["target"], strict=True)
        return {
            (str(s), str(t)): float(v)
            for (s, t), v in zip(pairs, self.edges["support"], strict=True)
        }


def bootstrap_mst_edge_support(
    returns: pd.DataFrame,
    *,
    corr_fn: CorrFn | None = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    mean_block_length: float | None = None,
    seed: int = 0,
) -> EdgeSupport:
    """Edge support of the MST of ``corr_fn(returns)`` under the stationary bootstrap.

    ``returns`` is reduced to its complete-case rows. The default ``mean_block_length`` is
    ``round(T ** (1/3))`` floored at 2 (an **[Inference]**, as in ``horizon_analysis``).
    """
    if not isinstance(returns, pd.DataFrame):
        raise NetworkStabilityError("returns must be a pandas DataFrame")
    if n_bootstrap < MIN_BOOTSTRAP:
        raise NetworkStabilityError(f"n_bootstrap must be >= {MIN_BOOTSTRAP}, got {n_bootstrap}")
    if not 0.0 < alpha < 0.5:
        raise NetworkStabilityError(f"alpha must be in (0, 0.5), got {alpha}")
    sample = returns.dropna(how="any")
    if len(sample) < MIN_OBS:
        raise NetworkStabilityError(
            f"need at least {MIN_OBS} complete-case rows, got {len(sample)}"
        )
    fn = pearson if corr_fn is None else corr_fn
    block = (
        default_mean_block_length(len(sample)) if mean_block_length is None else mean_block_length
    )

    values = sample.to_numpy(dtype=float)
    point = build_mst(fn(sample))
    counts = {edge: 0 for edge in point.edge_set}
    survival = np.empty(n_bootstrap, dtype=float)
    rng = np.random.default_rng(seed)
    for b in range(n_bootstrap):
        idx = stationary_bootstrap_indices(len(sample), block, rng)
        tree = build_mst(fn(pd.DataFrame(values[idx], columns=sample.columns)))
        shared = tree.edge_set & point.edge_set
        for edge in shared:
            counts[edge] += 1
        survival[b] = edge_survival_ratio(tree, point)

    edges = point.edges.copy()
    keys = zip(edges["source"], edges["target"], strict=True)
    edges["support"] = [counts[(str(s), str(t))] / n_bootstrap for s, t in keys]
    return EdgeSupport(
        mst=point,
        edges=edges,
        mean_survival=float(survival.mean()),
        survival_lower=float(np.quantile(survival, alpha / 2.0)),
        survival_upper=float(np.quantile(survival, 1.0 - alpha / 2.0)),
        n_bootstrap=n_bootstrap,
        mean_block_length=float(block),
        n_obs=len(sample),
    )


# --------------------------------------------------------------------------- survival matrix


def survival_matrix(msts: Mapping[str, MSTResult]) -> pd.DataFrame:
    """Pairwise edge-survival ratios of trees on the same nodes (symmetric, unit diagonal)."""
    if len(msts) < 2:
        raise NetworkStabilityError("need at least two trees")
    names = list(msts)
    out = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            value = edge_survival_ratio(msts[a], msts[b])
            out.loc[a, b] = value
            out.loc[b, a] = value
    return out


# --------------------------------------------------------------------------- noise floor


@dataclass(frozen=True, eq=False)
class NoiseSurvivalNull:
    """Survival ratio between MSTs of two independent noise panels (no common structure)."""

    values: NDArray[np.float64]
    n_nodes: int
    n_obs: int
    mean: float
    q95: float
    expected_mean: float  # 2 / N, the exact expectation under exchangeability


def noise_mst_survival_null(
    n_nodes: int, n_obs: int, n_draws: int = 1000, seed: int = 0
) -> NoiseSurvivalNull:
    """Simulated floor of the edge-survival ratio: two independent i.i.d. Gaussian panels."""
    if n_nodes < 3:
        raise NetworkStabilityError(f"n_nodes must be >= 3, got {n_nodes}")
    if n_obs < MIN_OBS:
        raise NetworkStabilityError(f"n_obs must be >= {MIN_OBS}, got {n_obs}")
    if n_draws < 1:
        raise NetworkStabilityError(f"n_draws must be >= 1, got {n_draws}")
    labels = [f"N{k:03d}" for k in range(n_nodes)]
    rng = np.random.default_rng(seed)
    values = np.empty(n_draws, dtype=float)
    for d in range(n_draws):
        a = pd.DataFrame(rng.standard_normal((n_obs, n_nodes)), columns=labels)
        b = pd.DataFrame(rng.standard_normal((n_obs, n_nodes)), columns=labels)
        values[d] = edge_survival_ratio(build_mst(a.corr()), build_mst(b.corr()))
    return NoiseSurvivalNull(
        values=values,
        n_nodes=n_nodes,
        n_obs=n_obs,
        mean=float(values.mean()),
        q95=float(np.quantile(values, 0.95)),
        expected_mean=2.0 / n_nodes,
    )


# --------------------------------------------------------------------------- rolling null


@dataclass(frozen=True, eq=False)
class RollingSurvivalNull:
    """Real mean rolling-window survival against its row-shuffled distribution."""

    real: pd.Series
    real_mean: float
    null_means: NDArray[np.float64]
    lower_tail: float  # (1 + #{null <= real}) / (1 + n)
    upper_tail: float  # (1 + #{null >= real}) / (1 + n)
    n_shuffles: int

    @property
    def p_two_sided(self) -> float:
        return float(min(1.0, 2.0 * min(self.lower_tail, self.upper_tail)))

    @property
    def null_mean(self) -> float:
        return float(self.null_means.mean())


def _mean_valid(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) < 3:
        raise NetworkStabilityError(
            f"only {len(valid)} defined survival values; use a smaller window or step"
        )
    return float(valid.mean())


def rolling_survival_shuffle_null(
    returns: pd.DataFrame,
    window: int,
    step: int,
    *,
    n_shuffles: int = 100,
    seed: int = 0,
    corr_fn: CorrFn | None = None,
    min_obs: int = 10,
) -> RollingSurvivalNull:
    """Real rolling survival versus the same statistic on ``n_shuffles`` row permutations.

    The real series is :func:`etfcorr.correlation.networks.rolling_mst_survival` (trailing windows,
    a pure function of rows up to each date); each shuffled copy keeps the date index and permutes
    the rows.
    """
    if n_shuffles < 1:
        raise NetworkStabilityError(f"n_shuffles must be >= 1, got {n_shuffles}")
    real = rolling_mst_survival(returns, window, step, corr_fn=corr_fn, min_obs=min_obs)
    real_mean = _mean_valid(real)
    values = returns.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    null = np.empty(n_shuffles, dtype=float)
    for s in range(n_shuffles):
        shuffled = pd.DataFrame(
            values[rng.permutation(len(values))], index=returns.index, columns=returns.columns
        )
        null[s] = _mean_valid(
            rolling_mst_survival(shuffled, window, step, corr_fn=corr_fn, min_obs=min_obs)
        )
    return RollingSurvivalNull(
        real=real,
        real_mean=real_mean,
        null_means=null,
        lower_tail=float((1 + np.sum(null <= real_mean + 1e-12)) / (1 + n_shuffles)),
        upper_tail=float((1 + np.sum(null >= real_mean - 1e-12)) / (1 + n_shuffles)),
        n_shuffles=n_shuffles,
    )


# --------------------------------------------------------------------------- cluster agreement


@dataclass(frozen=True)
class ClusterAgreement:
    """Adjusted Rand index of a hierarchical clustering against the sector labels."""

    ari: float
    null_mean: float
    p_value: float  # one-sided: agreement above chance
    n_clusters: int
    method: str
    n_permutations: int


def cluster_sector_agreement(
    corr: pd.DataFrame,
    sectors: Mapping[str, str],
    *,
    n_clusters: int | None = None,
    method: str = "average",
    n_permutations: int = 2000,
    seed: int = 0,
) -> ClusterAgreement:
    """ARI between ``hierarchical_clusters(corr, n_clusters, method)`` and the sector labels.

    ``n_clusters`` defaults to the number of distinct sectors among the nodes. The permutation null
    reshuffles the sector labels over nodes; the one-sided p-value is
    ``(1 + #{null >= observed}) / (1 + n_permutations)``.
    """
    if n_permutations < 1:
        raise NetworkStabilityError(f"n_permutations must be >= 1, got {n_permutations}")
    nodes = sorted(str(c) for c in corr.columns)
    missing = [n for n in nodes if n not in sectors]
    if missing:
        raise NetworkStabilityError(f"no sector for nodes: {missing}")
    labels = [sectors[n] for n in nodes]
    k = len(set(labels)) if n_clusters is None else n_clusters
    clusters = hierarchical_clusters(corr, k, method)
    found = clusters.loc[nodes].to_numpy()
    observed = float(adjusted_rand_score(labels, found))
    rng = np.random.default_rng(seed)
    label_array = np.asarray(labels)
    null = np.empty(n_permutations, dtype=float)
    for p in range(n_permutations):
        null[p] = float(adjusted_rand_score(rng.permutation(label_array), found))
    exceed = int(np.sum(null >= observed - 1e-12))
    return ClusterAgreement(
        ari=observed,
        null_mean=float(null.mean()),
        p_value=(1 + exceed) / (1 + n_permutations),
        n_clusters=k,
        method=method,
        n_permutations=n_permutations,
    )


def partition_similarity(
    corr_a: pd.DataFrame, corr_b: pd.DataFrame, n_clusters: int, method: str = "average"
) -> float:
    """ARI between the ``n_clusters`` hierarchical partitions of two matrices on the same nodes.

    Exploratory follow-up E2 (docs/FOLLOW_UPS.md item 2): unlike an eigenvector overlap it does
    not depend on the order of near-equal eigenvalues. 1 means the same partition, about 0 chance.
    """
    a = hierarchical_clusters(corr_a, n_clusters, method)
    b = hierarchical_clusters(corr_b, n_clusters, method)
    if list(a.index) != list(b.index):
        raise NetworkStabilityError("the two matrices must have the same nodes")
    return float(adjusted_rand_score(a.to_numpy(), b.to_numpy()))
