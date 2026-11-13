"""Step 7.1: graphs built inside the training window, and graph aggregates for rung 3.

**Question served.** H3 allows "a machine-learning or graph model". In the MVP scope the graph
model is LightGBM (rung 3) given features that a graph selects: the average trailing correlation
over the graph's edges and over its non-edges. The GNN is deferred, so no PyG tensors are built.

**Leakage contract.** The graph is a fitted object, so it lives inside the model:
:class:`GraphAugmented` receives the same ``(X, y)`` as every other forecaster, builds the graph
in ``fit`` from the rows it is handed (the purged training origins of the fold) and reuses that
graph in ``predict``. It never sees returns, only the pairwise trailing correlations that
:func:`pair_table` adds to the feature table (each dated ``t`` and causal). A graph can therefore
not depend on a date after the last training origin, and it is rebuilt at every refit;
``tests/leakage/test_graphs_no_lookahead.py`` checks this with the shared detectors and a twin
whose graph is built from the whole sample.

**Graphs** (all undirected, unweighted, on the assets of the panel), from the training-window
similarity ``S = tanh(mean over training rows of the Fisher-z trailing correlation)``:

* ``knn``: each asset's ``k`` most correlated assets, symmetrised by union;
* ``threshold``: the pairs at or above a quantile of the off-diagonal entries of ``S``;
* ``mst``: the minimum spanning tree of the Mantegna distance (``correlation.networks``);
* ``pmfg``: the planar maximally filtered graph (Tumminello et al. 2005): pairs are added in
  decreasing order of ``S`` whenever the graph stays planar, up to ``3 (N - 2)`` edges;
  planarity through ``networkx.check_planarity``;
* ``cluster``: pairs inside the same average-linkage cluster, with as many clusters as sectors;
* ``sector``: pairs of the same sector (static taxonomy, uses no return data);
* ``random``: as many edges as the kNN graph, drawn uniformly with a fixed seed (the ablation:
  a graph with no information).

:func:`threshold_graph` works on any similarity matrix, so a holdings-overlap graph is that
function applied to ``data.holdings.overlap_matrix``. It is **not** in the ladder: the only
holdings snapshots on disk are dated 2026, after every training window, so using them would be
look-ahead.

**Features.** For each window ``w`` of ``windows``: ``g_z_edge_w{w}`` and ``g_z_nonedge_w{w}``,
the Fisher-z of the mean trailing-``w`` correlation over the pairs that are, and are not, edges.

Limits: ``S`` is an average of overlapping rolling correlations, not the correlation of the
whole training window (close, but not equal); an MST or PMFG on 25 assets changes edges between
refits (step 3.6 measured about 92% edge survival between consecutive windows), so the meaning
of a graph feature drifts across refits; the edge/non-edge split of the sector graph at the
structure window repeats two features the common table already has. A graph feature that helps
a forecast says nothing about a causal network.

Reference (CHECKED on Crossref 2026-11-02): Tumminello, Aste, Di Matteo and Mantegna
(2005), *A tool for filtering information in complex systems*, PNAS 102(30), 10421-10426.
"""

from collections.abc import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from etfcorr import features as F
from etfcorr.config import GraphKind
from etfcorr.correlation.networks import build_mst, hierarchical_clusters
from etfcorr.features import BoolArray, FloatArray
from etfcorr.models.base import Factory, Forecaster, ModelError, check_table

PAIR_PREFIX = "pz"


def pair_columns(assets: Sequence[str], window: int) -> list[str]:
    """Names of the pair columns of one window, in ``features.pair_labels`` order."""
    return [f"{PAIR_PREFIX}{window}{F.PAIR_SEP}{p}" for p in F.pair_labels(list(assets))]


def pair_table(returns: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    """Fisher-z trailing correlation of every pair at every window, one row per date."""
    assets = [str(c) for c in returns.columns]
    parts = []
    for w in windows:
        wide = F.pairs_wide(F.trailing_correlations(returns, w), returns.index, assets)
        wide.columns = pd.Index(pair_columns(assets, w))
        parts.append(wide)
    return pd.concat(parts, axis=1)


def graph_columns(windows: Sequence[int]) -> list[str]:
    """Names of the features :class:`GraphAugmented` adds."""
    return [f"g_z_{side}_w{w}" for w in windows for side in ("edge", "nonedge")]


# --------------------------------------------------------------------------------------
# Graphs from a similarity matrix (symmetric boolean adjacency, False on the diagonal)
# --------------------------------------------------------------------------------------


def _check(sim: FloatArray) -> int:
    n = sim.shape[0]
    if sim.ndim != 2 or sim.shape[1] != n or n < 3 or not np.isfinite(sim).all():
        raise ModelError("need a finite square similarity matrix of at least 3 assets")
    if not np.allclose(sim, sim.T):
        raise ModelError("the similarity matrix must be symmetric")
    return n


def _from_pairs(n: int, pairs: Sequence[tuple[int, int]]) -> BoolArray:
    adj = np.zeros((n, n), dtype=bool)
    for i, j in pairs:
        adj[i, j] = adj[j, i] = True
    return adj


def knn_graph(sim: FloatArray, k: int) -> BoolArray:
    """Each node's ``k`` most similar nodes (ties broken by position), symmetrised by union."""
    n = _check(sim)
    if not 1 <= k < n:
        raise ModelError(f"k must be in [1, {n - 1}], got {k}")
    s = sim.astype(float).copy()
    np.fill_diagonal(s, -np.inf)
    nearest = np.argsort(-s, axis=1, kind="stable")[:, :k]
    adj = np.zeros((n, n), dtype=bool)
    adj[np.arange(n)[:, None], nearest] = True
    return np.asarray(adj | adj.T)


def threshold_graph(sim: FloatArray, quantile: float) -> BoolArray:
    """The pairs whose similarity is at or above the ``quantile`` of all off-diagonal pairs."""
    n = _check(sim)
    cut = np.quantile(sim[np.triu_indices(n, k=1)], quantile)
    adj = sim >= cut
    np.fill_diagonal(adj, False)
    return np.asarray(adj)


def mst_graph(corr: FloatArray) -> BoolArray:
    """Minimum spanning tree of the Mantegna distance (``networks.build_mst``)."""
    n = _check(corr)
    labels = [f"{i:04d}" for i in range(n)]
    edges = build_mst(pd.DataFrame(corr, index=labels, columns=labels)).edge_set
    return _from_pairs(n, [(int(a), int(b)) for a, b in edges])


def pmfg_graph(sim: FloatArray) -> BoolArray:
    """Planar maximally filtered graph: add pairs by decreasing similarity while planar."""
    n = _check(sim)
    i, j = np.triu_indices(n, k=1)
    order = np.argsort(-sim[i, j], kind="stable")
    graph = nx.empty_graph(n)
    for a, b in zip(i[order].tolist(), j[order].tolist(), strict=True):
        graph.add_edge(a, b)
        if not nx.check_planarity(graph)[0]:
            graph.remove_edge(a, b)
        if graph.number_of_edges() == 3 * (n - 2):
            break
    return _from_pairs(n, list(graph.edges))


def cluster_graph(corr: FloatArray, n_clusters: int) -> BoolArray:
    """Pairs inside the same average-linkage cluster of the Mantegna distance."""
    n = _check(corr)
    labels = [f"{i:04d}" for i in range(n)]
    frame = pd.DataFrame(corr, index=labels, columns=labels)
    ids = hierarchical_clusters(frame, n_clusters, method="average").to_numpy()
    adj = ids[:, None] == ids[None, :]
    np.fill_diagonal(adj, False)
    return np.asarray(adj)


def sector_graph(assets: Sequence[str], sectors: Mapping[str, str]) -> BoolArray:
    """Pairs of the same sector."""
    labels = np.array([sectors[a] for a in assets])
    adj = labels[:, None] == labels[None, :]
    np.fill_diagonal(adj, False)
    return np.asarray(adj)


def random_graph(n: int, n_edges: int, seed: int) -> BoolArray:
    """``n_edges`` pairs drawn uniformly without replacement."""
    i, j = np.triu_indices(n, k=1)
    if not 0 < n_edges <= len(i):
        raise ModelError(f"n_edges must be in [1, {len(i)}], got {n_edges}")
    chosen = np.random.default_rng(seed).choice(len(i), size=n_edges, replace=False)
    return _from_pairs(n, list(zip(i[chosen].tolist(), j[chosen].tolist(), strict=True)))


def build_graph(
    kind: GraphKind,
    sim: FloatArray,
    assets: Sequence[str],
    sectors: Mapping[str, str],
    *,
    knn_k: int,
    threshold_quantile: float,
    seed: int = 0,
) -> BoolArray:
    """The graph of one ``kind`` on ``assets`` from the similarity matrix ``sim``."""
    if sim.shape[0] != len(assets):
        raise ModelError("the similarity matrix does not match the assets")
    if kind == "knn":
        return knn_graph(sim, knn_k)
    if kind == "threshold":
        return threshold_graph(sim, threshold_quantile)
    if kind == "mst":
        return mst_graph(sim)
    if kind == "pmfg":
        return pmfg_graph(sim)
    if kind == "cluster":
        return cluster_graph(sim, len({sectors[a] for a in assets}))
    if kind == "sector":
        return sector_graph(assets, sectors)
    if kind == "random":
        n_edges = int(np.triu(knn_graph(sim, knn_k), k=1).sum())
        return random_graph(len(assets), n_edges, seed)
    raise ModelError(f"unknown graph kind {kind!r}")


# --------------------------------------------------------------------------------------
# The forecaster wrapper
# --------------------------------------------------------------------------------------


class GraphAugmented:
    """A model fitted on its features plus graph aggregates; the graph is built in ``fit``.

    ``factory`` builds the wrapped model, which must read the columns of :func:`graph_columns`
    (for the same ``windows``) next to its own. ``X`` is a per-date table holding the columns of
    :func:`pair_table` for every window in ``windows``.
    """

    def __init__(
        self,
        factory: Factory,
        kind: GraphKind,
        assets: Sequence[str],
        sectors: Mapping[str, str],
        *,
        windows: Sequence[int],
        similarity_window: int,
        knn_k: int,
        threshold_quantile: float,
        seed: int = 0,
        name: str | None = None,
    ) -> None:
        if similarity_window not in windows:
            raise ModelError("similarity_window must be one of windows")
        self._factory = factory
        self.kind: GraphKind = kind
        self.assets = tuple(assets)
        self.sectors = dict(sectors)
        self.windows = tuple(windows)
        self.similarity_window = similarity_window
        self.knn_k = knn_k
        self.threshold_quantile = threshold_quantile
        self.seed = seed
        self.name = name or f"{factory().name}_graph_{kind}"
        self.adjacency_: BoolArray | None = None
        self._model: Forecaster | None = None

    def _correlations(self, X: pd.DataFrame, window: int) -> FloatArray:
        columns = pair_columns(self.assets, window)
        missing = [c for c in columns if c not in X.columns]
        if missing:
            raise ModelError(f"{self.name}: pair columns of window {window} are not in the table")
        return F.inverse_fisher_z(X[columns].to_numpy(dtype=float))

    def _augment(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.adjacency_ is not None
        n = len(self.assets)
        edge = self.adjacency_[np.triu_indices(n, k=1)]
        added = {}
        for w in self.windows:
            corr = self._correlations(X, w)
            added[f"g_z_edge_w{w}"] = F.fisher_z(corr[:, edge].mean(axis=1))
            added[f"g_z_nonedge_w{w}"] = F.fisher_z(corr[:, ~edge].mean(axis=1))
        return X.assign(**added)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GraphAugmented":
        check_table(X, y)
        if isinstance(X.index, pd.MultiIndex):
            raise ModelError(f"{self.name}: needs a per-date table")
        n = len(self.assets)
        z = np.arctanh(self._correlations(X, self.similarity_window)).mean(axis=0)
        sim = np.eye(n)
        i, j = np.triu_indices(n, k=1)
        sim[i, j] = sim[j, i] = np.tanh(z)
        adjacency = build_graph(
            self.kind,
            sim,
            self.assets,
            self.sectors,
            knn_k=self.knn_k,
            threshold_quantile=self.threshold_quantile,
            seed=self.seed,
        )
        n_edges = int(adjacency[i, j].sum())
        if n_edges in (0, len(i)):
            raise ModelError(f"{self.name}: the graph has no edge or is complete")
        self.adjacency_ = adjacency
        self._model = self._factory().fit(self._augment(X), y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise ModelError(f"{self.name}: predict called before fit")
        return self._model.predict(self._augment(X))

    @property
    def params(self) -> dict[str, object]:
        inner = {} if self._model is None else dict(self._model.params)
        n_edges = 0 if self.adjacency_ is None else int(self.adjacency_.sum()) // 2
        return {**inner, "graph": self.kind, "n_edges": n_edges}
