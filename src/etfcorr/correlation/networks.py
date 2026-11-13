"""Correlation networks: distance, minimum spanning tree, clustering, stability (step 3.3).

Question served: structure and visualisation for the Phase 3 milestone ("MST figure",
estimator-stability table). This module is NOT the H1 test: H1 is decided by
``horizon_analysis.h1_cross_horizon_test`` (step 3.2). Everything here is exploratory
description of one correlation matrix, or of the stability of that description.

Method
------
* Distance (Mantegna 1999): ``d_ij = sqrt(2 * (1 - rho_ij))``. It is a metric only when the
  input is a genuine (PSD) correlation matrix; shrunk or denoised matrices are fine, Kendall
  and pairwise-complete matrices may not be PSD. The functions below only need ``d`` to be
  symmetric, non-negative and finite, and do not require PSD.
* Minimum spanning tree (MST): Kruskal's algorithm with a deterministic tie-break on the
  (sorted) node labels, so the tree does not depend on the row order of the input.
* Single-linkage clustering equals the MST (Gower and Ross 1969): the sorted single-linkage
  merge heights equal the sorted MST edge weights. This identity is tested.
* Node statistics on the tree: degree, betweenness (closed form on a tree: removing node v
  leaves components of sizes s_k, so betweenness(v) = ((n - 1)^2 - sum s_k^2) / 2 pairs,
  reported normalised by (n - 1)(n - 2) / 2), and eigenvector centrality (Perron vector of the
  adjacency matrix, unit L2 norm). No networkx dependency; networkx is used in the tests
  only, as an independent cross-check.
* Edge-survival ratio between two trees on the same nodes: shared edges / (n - 1). Rolling
  version: MST of each trailing window versus the MST of the previous window.
* Sector purity: share of MST edges joining two ETFs of the same sector, with a permutation
  p-value that reshuffles sector labels over nodes (topology held fixed).

Assumptions and known failure modes
-----------------------------------
* MST edges are not independent observations and the tree is unstable when many distances are
  nearly equal (small T/N, flat correlation structure). The survival ratio should be read
  against a null, not against 1.
* Rolling windows with ``step < window`` overlap, so consecutive survival values share data;
  they are not independent draws and no p-value is attached to them.
* A detoned (market-mode-removed) matrix has negative cross-sector correlations by
  construction, so distances between sectors become large: MSTs of detoned matrices are
  relative structure, not absolute co-movement [Inference].
* Holdings overlap creates mechanical similarity between ETFs; use ``overlap.py`` residuals
  as input to separate it from economic co-movement.
* The permutation test conditions on the observed tree; it asks whether sector labels are
  associated with adjacency, not whether the tree itself is significant.

Citations (status)
------------------
* Mantegna, R. N. (1999). Hierarchical structure in financial markets. Eur. Phys. J. B
  11(1), 193-197, DOI 10.1007/s100510050929. CHECKED in this session (title, author, venue,
  abstract match a search result).
* Tumminello, M., Di Matteo, T., Aste, T., Mantegna, R. N. (2007). Correlation based networks
  of equity returns sampled at different time horizons. Eur. Phys. J. B 55(2), 209-217,
  arXiv physics/0605251. CHECKED (bibliographic record seen in this session).
* Gower, J. C., Ross, G. J. S. (1969). Minimum spanning trees and single linkage cluster
  analysis. Applied Statistics 18(1), 54-64. CHECKED on Crossref 2026-11-02.
* Kruskal, J. B. (1956). On the shortest spanning subtree of a graph. Proc. AMS 7(1), 48-50.
  CHECKED on Crossref 2026-11-02.
* Onnela, J.-P. et al. (2003). Dynamics of market correlations: taxonomy and portfolio
  analysis. Phys. Rev. E 68, 056110: the attribution of the "survival ratio" to this paper is
  CHECKED in the arXiv text (2026-11-02): it defines single- and multi-step survival ratios.

Third-party code: none adapted. The MST, centralities and permutation test are written from
the definitions above; scipy is used for hierarchical linkage only. No licence flags.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.cluster.hierarchy import cut_tree, leaves_list, linkage
from scipy.spatial.distance import squareform

__all__ = [
    "MSTResult",
    "NetworkError",
    "PurityTest",
    "build_mst",
    "correlation_distance",
    "dendrogram_order",
    "edge_survival_ratio",
    "hierarchical_clusters",
    "mst_summary_line",
    "rolling_mst_survival",
    "sector_edge_purity",
    "sector_purity_test",
]

_LINKAGE_METHODS = ("single", "average", "complete")


class NetworkError(ValueError):
    """Invalid input to a network function (bad matrix, labels, or parameters)."""


@dataclass(frozen=True, eq=False)
class MSTResult:
    """A minimum spanning tree of a correlation matrix.

    ``nodes`` are the labels in sorted order. ``edges`` has columns ``source``, ``target``
    (``source < target`` alphabetically), ``distance``, ``correlation``, in Kruskal order.
    ``node_stats`` is indexed by ``nodes`` with columns ``degree``, ``betweenness`` (normalised
    by (n - 1)(n - 2) / 2) and ``eigenvector`` (unit L2 norm).
    """

    nodes: tuple[str, ...]
    edges: pd.DataFrame
    node_stats: pd.DataFrame
    total_length: float

    @property
    def edge_set(self) -> frozenset[tuple[str, str]]:
        """Undirected edges as ``(min_label, max_label)`` pairs."""
        pairs = zip(self.edges["source"], self.edges["target"], strict=True)
        return frozenset((str(s), str(t)) for s, t in pairs)


@dataclass(frozen=True)
class PurityTest:
    """Result of :func:`sector_purity_test` (one-sided: purity above chance)."""

    observed: float
    null_mean: float
    p_value: float
    n_permutations: int


# --------------------------------------------------------------------------- validation


def _prepare(corr: pd.DataFrame, atol: float = 1e-8) -> tuple[list[str], NDArray[np.float64]]:
    """Validate a correlation matrix and return (sorted labels, clean sorted matrix)."""
    if not isinstance(corr, pd.DataFrame):
        raise NetworkError("corr must be a pandas DataFrame")
    n = corr.shape[0]
    if corr.shape[1] != n:
        raise NetworkError(f"corr must be square, got shape {corr.shape}")
    if n < 2:
        raise NetworkError("need at least 2 nodes")
    if not corr.index.equals(corr.columns):
        raise NetworkError("row and column labels must be identical and in the same order")
    labels = list(corr.index)
    if not all(isinstance(lab, str) for lab in labels):
        raise NetworkError("labels must be strings (tickers)")
    if len(set(labels)) != n:
        raise NetworkError("labels must be unique")
    values = corr.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise NetworkError("corr contains NaN or infinite values")
    if float(np.abs(values - values.T).max()) > atol:
        raise NetworkError("corr is not symmetric")
    if float(np.abs(np.diag(values) - 1.0).max()) > atol:
        raise NetworkError("corr diagonal must be 1")
    if float(np.abs(values).max()) > 1.0 + atol:
        raise NetworkError("corr has entries outside [-1, 1]")
    order = sorted(range(n), key=lambda i: labels[i])
    sorted_labels = [labels[i] for i in order]
    clean = values[np.ix_(order, order)]
    clean = (clean + clean.T) / 2.0
    np.fill_diagonal(clean, 1.0)
    return sorted_labels, np.clip(clean, -1.0, 1.0)


def _distance_from_corr(clean: NDArray[np.float64]) -> NDArray[np.float64]:
    dist: NDArray[np.float64] = np.sqrt(np.maximum(2.0 * (1.0 - clean), 0.0))
    np.fill_diagonal(dist, 0.0)
    return dist


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Mantegna distance ``sqrt(2 (1 - rho))`` with labels in sorted order.

    Equation 2 of Mantegna (1999), Eur. Phys. J. B 11, 193 (checked 2026-11-01). The arXiv
    preprint (cond-mat/9802256) had ``1 - rho**2``, which puts strongly anti-correlated assets
    close together; that matters for detoned matrices, so cite the published version.
    """
    labels, clean = _prepare(corr)
    return pd.DataFrame(_distance_from_corr(clean), index=labels, columns=labels)


# --------------------------------------------------------------------------- MST


def _kruskal(dist: NDArray[np.float64]) -> list[tuple[int, int, float]]:
    """Kruskal MST; ties broken by (i, j) in sorted-label index order."""
    n = dist.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    weights = dist[iu, ju]
    order = np.lexsort((ju, iu, weights))  # primary key: weight, then i, then j
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    chosen: list[tuple[int, int, float]] = []
    for k in order:
        i, j = int(iu[k]), int(ju[k])
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            chosen.append((i, j, float(weights[k])))
            if len(chosen) == n - 1:
                break
    return chosen


def _tree_stats(n: int, edges: list[tuple[int, int, float]]) -> pd.DataFrame:
    """Degree, normalised betweenness and eigenvector centrality of a tree (index order)."""
    adj = np.zeros((n, n), dtype=float)
    neighbours: list[list[int]] = [[] for _ in range(n)]
    for i, j, _ in edges:
        adj[i, j] = adj[j, i] = 1.0
        neighbours[i].append(j)
        neighbours[j].append(i)
    degree = adj.sum(axis=1)

    # Rooted at node 0: subtree sizes by iterative DFS.
    parent = [-1] * n
    visit_order: list[int] = []
    stack = [0]
    seen = [False] * n
    seen[0] = True
    while stack:
        v = stack.pop()
        visit_order.append(v)
        for w in neighbours[v]:
            if not seen[w]:
                seen[w] = True
                parent[w] = v
                stack.append(w)
    size = [1] * n
    for v in reversed(visit_order):
        if parent[v] >= 0:
            size[parent[v]] += size[v]
    pair_counts = np.zeros(n, dtype=float)
    for v in range(n):
        parts = [size[c] for c in neighbours[v] if parent[c] == v]
        if parent[v] >= 0:
            parts.append(n - size[v])
        pair_counts[v] = ((n - 1) ** 2 - sum(s * s for s in parts)) / 2.0
    denom = (n - 1) * (n - 2) / 2.0
    betweenness = pair_counts / denom if denom > 0 else np.zeros(n)

    # Perron vector: a tree is connected, so the top eigenvalue is simple.
    _, vectors = np.linalg.eigh(adj)
    top = np.abs(vectors[:, -1])
    eigenvector = top / float(np.linalg.norm(top))
    return pd.DataFrame(
        {"degree": degree.astype(int), "betweenness": betweenness, "eigenvector": eigenvector}
    )


def build_mst(corr: pd.DataFrame) -> MSTResult:
    """Minimum spanning tree of the Mantegna distance of ``corr``.

    Deterministic and invariant to the order of rows and columns of ``corr``. Does not modify
    its input. See the module docstring for assumptions and caveats.
    """
    labels, clean = _prepare(corr)
    n = len(labels)
    chosen = _kruskal(_distance_from_corr(clean))
    if len(chosen) != n - 1:  # pragma: no cover - a complete finite graph is always connected
        raise NetworkError("failed to build a spanning tree")
    edges = pd.DataFrame(
        {
            "source": [labels[i] for i, _, _ in chosen],
            "target": [labels[j] for _, j, _ in chosen],
            "distance": [w for _, _, w in chosen],
            "correlation": [float(clean[i, j]) for i, j, _ in chosen],
        }
    )
    stats = _tree_stats(n, chosen)
    stats.index = pd.Index(labels)
    return MSTResult(
        nodes=tuple(labels),
        edges=edges,
        node_stats=stats,
        total_length=float(edges["distance"].sum()),
    )


def mst_summary_line(mst: MSTResult) -> str:
    """One-line report: nodes, total length, hub (max degree), mean edge correlation."""
    hub = str(mst.node_stats["degree"].idxmax())
    hub_degree = int(mst.node_stats["degree"].max())
    return (
        f"MST n={len(mst.nodes)} edges={len(mst.edges)} length={mst.total_length:.4f} "
        f"hub={hub}(deg {hub_degree}) mean_edge_corr={float(mst.edges['correlation'].mean()):.4f}"
    )


# --------------------------------------------------------------------------- clustering


def hierarchical_clusters(corr: pd.DataFrame, n_clusters: int, method: str = "single") -> pd.Series:
    """Cut a hierarchical clustering of the Mantegna distance into exactly ``n_clusters``.

    ``method`` is ``single`` (equivalent to the MST), ``average`` or ``complete``. Cluster ids
    are 0-based and numbered by first appearance in sorted-label order, so they do not depend
    on the input row order. The returned Series is indexed by sorted label.
    """
    if method not in _LINKAGE_METHODS:
        raise NetworkError(f"method must be one of {_LINKAGE_METHODS}, got {method!r}")
    labels, clean = _prepare(corr)
    n = len(labels)
    if not 1 <= n_clusters <= n:
        raise NetworkError(f"n_clusters must be in [1, {n}], got {n_clusters}")
    tree = linkage(squareform(_distance_from_corr(clean), checks=False), method=method)
    raw = np.asarray(cut_tree(tree, n_clusters=n_clusters)).reshape(-1)
    relabel: dict[int, int] = {}
    ids = []
    for value in raw:
        key = int(value)
        if key not in relabel:
            relabel[key] = len(relabel)
        ids.append(relabel[key])
    return pd.Series(ids, index=pd.Index(labels), name="cluster", dtype=int)


def dendrogram_order(corr: pd.DataFrame, method: str = "average") -> list[str]:
    """Leaf order of a hierarchical clustering, for clustered heatmaps."""
    if method not in _LINKAGE_METHODS:
        raise NetworkError(f"method must be one of {_LINKAGE_METHODS}, got {method!r}")
    labels, clean = _prepare(corr)
    tree = linkage(squareform(_distance_from_corr(clean), checks=False), method=method)
    return [labels[int(i)] for i in leaves_list(tree)]


# --------------------------------------------------------------------------- stability


def edge_survival_ratio(a: MSTResult, b: MSTResult) -> float:
    """Shared edges divided by ``n - 1``; both trees must have the same node set."""
    if a.nodes != b.nodes:
        raise NetworkError("trees must have the same nodes")
    return len(a.edge_set & b.edge_set) / (len(a.nodes) - 1)


def _pearson_complete_case(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna().corr()


def rolling_mst_survival(
    returns: pd.DataFrame,
    window: int,
    step: int = 1,
    *,
    corr_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    min_obs: int = 10,
) -> pd.Series:
    """Edge-survival ratio of each trailing-window MST versus the previous window's MST.

    The value at date ``t`` uses only rows up to and including ``t`` (windows end at
    positions ``window - 1, window - 1 + step, ...``), so it is a pure function of
    ``returns`` up to ``t``. The first window has no predecessor and is NaN. Windows with
    fewer than ``min_obs`` complete-case rows give NaN for that date and the next one.
    ``corr_fn`` maps a window of returns to a labelled correlation matrix; the default is
    Pearson on complete-case rows. Overlapping windows are not independent.
    """
    if not isinstance(returns, pd.DataFrame):
        raise NetworkError("returns must be a pandas DataFrame")
    if not returns.index.is_monotonic_increasing or not returns.index.is_unique:
        raise NetworkError("returns index must be strictly increasing")
    if window < 3 or step < 1 or min_obs < 3:
        raise NetworkError("need window >= 3, step >= 1, min_obs >= 3")
    if window > len(returns):
        raise NetworkError(f"window {window} exceeds the {len(returns)} available rows")
    fn = corr_fn if corr_fn is not None else _pearson_complete_case

    positions = list(range(window - 1, len(returns), step))
    values: list[float] = []
    previous: MSTResult | None = None
    for pos in positions:
        chunk = returns.iloc[pos - window + 1 : pos + 1]
        if len(chunk.dropna()) < min_obs:
            current: MSTResult | None = None
        else:
            current = build_mst(fn(chunk))
        if previous is None or current is None:
            values.append(float("nan"))
        else:
            values.append(edge_survival_ratio(current, previous))
        previous = current
    return pd.Series(values, index=returns.index[positions], name="mst_survival", dtype=float)


# --------------------------------------------------------------------------- sector purity


def _sector_codes(nodes: tuple[str, ...], sectors: Mapping[str, str]) -> NDArray[np.int64]:
    missing = [node for node in nodes if node not in sectors]
    if missing:
        raise NetworkError(f"no sector for nodes: {missing}")
    names = sorted({sectors[node] for node in nodes})
    lookup = {name: k for k, name in enumerate(names)}
    return np.array([lookup[sectors[node]] for node in nodes], dtype=np.int64)


def _edge_index(mst: MSTResult) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    position = {node: k for k, node in enumerate(mst.nodes)}
    src = np.array([position[str(s)] for s in mst.edges["source"]], dtype=np.int64)
    dst = np.array([position[str(t)] for t in mst.edges["target"]], dtype=np.int64)
    return src, dst


def sector_edge_purity(mst: MSTResult, sectors: Mapping[str, str]) -> float:
    """Share of MST edges whose two endpoints belong to the same sector."""
    codes = _sector_codes(mst.nodes, sectors)
    src, dst = _edge_index(mst)
    return float(np.mean(codes[src] == codes[dst]))


def sector_purity_test(
    mst: MSTResult, sectors: Mapping[str, str], n_permutations: int = 2000, seed: int = 0
) -> PurityTest:
    """One-sided permutation test that sector purity exceeds chance.

    Sector labels are permuted over nodes (sector sizes preserved, tree topology fixed).
    ``p = (1 + #{null >= observed}) / (1 + n_permutations)``. Exploratory; not part of H1-H6.
    """
    if n_permutations < 1:
        raise NetworkError("n_permutations must be >= 1")
    codes = _sector_codes(mst.nodes, sectors)
    src, dst = _edge_index(mst)
    observed = float(np.mean(codes[src] == codes[dst]))
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for k in range(n_permutations):
        shuffled = rng.permutation(codes)
        null[k] = float(np.mean(shuffled[src] == shuffled[dst]))
    # small tolerance so exact ties count as "at least as pure"
    exceed = int(np.sum(null >= observed - 1e-12))
    return PurityTest(
        observed=observed,
        null_mean=float(null.mean()),
        p_value=(1 + exceed) / (1 + n_permutations),
        n_permutations=n_permutations,
    )
