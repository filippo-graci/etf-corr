"""Unit tests for etfcorr.correlation.networks (step 3.3).

Synthetic data only. Known answers come from hand-built trees, graph theory (path/star
centralities, single linkage == MST) and independent implementations (scipy, networkx).
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.sparse.csgraph import minimum_spanning_tree as scipy_mst
from scipy.spatial.distance import squareform

from etfcorr.correlation import networks as nw


def corr_from_distance(dist: np.ndarray, labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(1.0 - dist**2 / 2.0, index=labels, columns=labels)


def chain_corr(labels: list[str], positions: list[float]) -> pd.DataFrame:
    x = np.asarray(positions, dtype=float) * 0.3
    return corr_from_distance(np.abs(x[:, None] - x[None, :]), labels)


def random_corr(n: int, seed: int, t: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = [f"T{k:02d}" for k in range(n)]
    data = pd.DataFrame(rng.standard_normal((t, n)), columns=labels)
    data = data + 0.5 * rng.standard_normal((t, 1))  # a common factor
    return data.corr()


def block_corr(sizes: list[int], within: float, across: float) -> pd.DataFrame:
    labels: list[str] = []
    block_of: list[int] = []
    for b, size in enumerate(sizes):
        for k in range(size):
            labels.append(f"S{b}_{k:02d}")
            block_of.append(b)
    n = len(labels)
    mat = np.full((n, n), across)
    for i in range(n):
        for j in range(n):
            if block_of[i] == block_of[j]:
                mat[i, j] = within
    np.fill_diagonal(mat, 1.0)
    return pd.DataFrame(mat, index=labels, columns=labels)


# ------------------------------------------------------------------ distance and validation


def test_distance_known_values() -> None:
    values = [[1.0, 1.0, 0.0], [1.0, 1.0, -1.0], [0.0, -1.0, 1.0]]
    corr = pd.DataFrame(values, index=list("ABC"), columns=list("ABC"))
    dist = nw.correlation_distance(corr)
    assert dist.loc["A", "B"] == pytest.approx(0.0)
    assert dist.loc["A", "C"] == pytest.approx(np.sqrt(2.0))
    assert dist.loc["B", "C"] == pytest.approx(2.0)
    assert np.allclose(np.diag(dist.to_numpy()), 0.0)
    assert np.allclose(dist.to_numpy(), dist.to_numpy().T)


def test_distance_is_exactly_symmetric_despite_float_level_asymmetry() -> None:
    corr = random_corr(6, seed=4)
    values = corr.to_numpy().copy()
    values[0, 1] += 1e-10  # below the validation tolerance
    dist = nw.correlation_distance(pd.DataFrame(values, index=corr.index, columns=corr.columns))
    assert (dist.to_numpy() == dist.to_numpy().T).all()


def test_distance_labels_are_sorted() -> None:
    corr = random_corr(6, seed=1)
    order = ["T03", "T00", "T05", "T01", "T04", "T02"]
    shuffled = corr.loc[order, order]
    assert list(nw.correlation_distance(shuffled).index) == sorted(shuffled.index)


@pytest.mark.parametrize(
    "case, message",
    [
        ("not_frame", "DataFrame"),
        ("not_square", "square"),
        ("one_node", "at least 2"),
        ("label_mismatch", "identical"),
        ("duplicate", "unique"),
        ("non_string", "strings"),
        ("nan", "NaN"),
        ("asymmetric", "symmetric"),
        ("diag", "diagonal"),
        ("out_of_range", "outside"),
    ],
)
def test_validation_errors(case: str, message: str) -> None:
    good = random_corr(4, seed=2)
    obj: object
    if case == "not_frame":
        obj = good.to_numpy()
    elif case == "not_square":
        obj = good.iloc[:, :3]
    elif case == "one_node":
        obj = good.iloc[:1, :1]
    elif case == "label_mismatch":
        obj = good.copy()
        obj.columns = pd.Index(list(reversed(list(good.columns))))
    elif case == "duplicate":
        obj = good.copy()
        obj.index = pd.Index(["T00", "T00", "T02", "T03"])
        obj.columns = obj.index
    elif case == "non_string":
        obj = good.copy()
        obj.index = pd.Index([0, 1, 2, 3])
        obj.columns = obj.index
    elif case == "nan":
        obj = good.copy()
        obj.iloc[0, 1] = np.nan
        obj.iloc[1, 0] = np.nan
    elif case == "asymmetric":
        obj = good.copy()
        obj.iloc[0, 1] += 0.1
    elif case == "diag":
        obj = good.copy()
        obj.iloc[0, 0] = 0.9
    else:
        obj = good.copy()
        obj.iloc[0, 1] = obj.iloc[1, 0] = 1.5
    with pytest.raises(nw.NetworkError, match=message):
        nw.build_mst(obj)  # type: ignore[arg-type]


def test_input_is_not_modified() -> None:
    corr = random_corr(7, seed=3)
    before = corr.copy()
    nw.build_mst(corr)
    nw.hierarchical_clusters(corr, 3)
    nw.dendrogram_order(corr)
    pd.testing.assert_frame_equal(corr, before)


# ------------------------------------------------------------------ MST known answers


def test_chain_mst_known_answer() -> None:
    mst = nw.build_mst(chain_corr(list("ABCDE"), [0, 1, 2, 3, 4]))
    assert mst.edge_set == {("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")}
    assert mst.total_length == pytest.approx(1.2)
    stats = mst.node_stats
    assert list(stats["degree"]) == [1, 2, 2, 2, 1]
    # path on 5 nodes: raw betweenness pair counts 0, 3, 4, 3, 0, divided by C(4, 2) = 6
    assert list(stats["betweenness"]) == pytest.approx([0.0, 0.5, 4 / 6, 0.5, 0.0])
    # Perron vector of the path P5: sin(k pi / 6) / sqrt(3)
    expected = np.sin(np.arange(1, 6) * np.pi / 6) / np.sqrt(3.0)
    assert list(stats["eigenvector"]) == pytest.approx(list(expected), abs=1e-10)


def test_star_from_constant_correlation_uses_first_label_as_hub() -> None:
    n = 6
    labels = list("FBEADC")
    corr = pd.DataFrame(np.full((n, n), 0.3), index=labels, columns=labels)
    for k in range(n):
        corr.iloc[k, k] = 1.0
    mst = nw.build_mst(corr)
    assert mst.edge_set == {("A", x) for x in "BCDEF"}
    stats = mst.node_stats
    assert stats.loc["A", "degree"] == n - 1
    assert stats.loc["A", "betweenness"] == pytest.approx(1.0)
    assert stats.loc["A", "eigenvector"] == pytest.approx(1 / np.sqrt(2))
    assert stats.loc["B", "eigenvector"] == pytest.approx(1 / np.sqrt(2 * (n - 1)))


def test_two_node_tree() -> None:
    mst = nw.build_mst(chain_corr(["X", "Y"], [0, 1]))
    assert mst.edge_set == {("X", "Y")}
    assert list(mst.node_stats["betweenness"]) == [0.0, 0.0]
    assert list(mst.node_stats["eigenvector"]) == pytest.approx([1 / np.sqrt(2)] * 2)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_mst_is_a_spanning_tree_of_minimum_weight(seed: int) -> None:
    corr = random_corr(15, seed=seed)
    mst = nw.build_mst(corr)
    assert len(mst.edges) == 14
    labels = list(mst.nodes)
    # connected: union-find over the returned edges
    parent = {lab: lab for lab in labels}

    def find(x: str) -> str:
        while parent[x] != x:
            x = parent[x]
        return x

    for s, t in mst.edge_set:
        parent[find(s)] = find(t)
    assert len({find(lab) for lab in labels}) == 1
    reference = scipy_mst(nw.correlation_distance(corr).to_numpy()).sum()
    assert mst.total_length == pytest.approx(float(reference))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_order_invariance(seed: int) -> None:
    corr = random_corr(12, seed=seed)
    rng = np.random.default_rng(100 + seed)
    perm = list(rng.permutation(corr.index))
    a = nw.build_mst(corr)
    b = nw.build_mst(corr.loc[perm, perm])
    assert a.edge_set == b.edge_set
    pd.testing.assert_frame_equal(a.node_stats, b.node_stats)
    assert a.total_length == b.total_length


def test_order_invariance_with_ties() -> None:
    corr = block_corr([3, 3], within=0.6, across=0.1)
    perm = list(corr.index[[4, 0, 5, 2, 1, 3]])
    assert nw.build_mst(corr).edge_set == nw.build_mst(corr.loc[perm, perm]).edge_set


def test_order_invariance_under_float_level_asymmetry() -> None:
    """Asymmetry below the validation tolerance must not make the tree depend on row order."""
    labels = list("ABCDEFGH")
    rng = np.random.default_rng(0)
    values = np.full((8, 8), 0.5) + 1e-10 * rng.standard_normal((8, 8))
    np.fill_diagonal(values, 1.0)
    corr = pd.DataFrame(values, index=labels, columns=labels)
    reference = nw.build_mst(corr).edge_set
    for seed in range(30):
        perm = list(np.random.default_rng(seed).permutation(labels))
        assert nw.build_mst(corr.loc[perm, perm]).edge_set == reference


def test_centralities_match_networkx() -> None:
    nx = pytest.importorskip("networkx")
    for seed in range(3):
        mst = nw.build_mst(random_corr(14, seed=seed))
        graph = nx.Graph()
        graph.add_nodes_from(mst.nodes)
        graph.add_edges_from(mst.edge_set)
        assert nx.is_tree(graph)
        stats = mst.node_stats
        deg = dict(graph.degree())
        bet = nx.betweenness_centrality(graph, normalized=True)
        eig = nx.eigenvector_centrality(graph, max_iter=100000, tol=1e-13)
        for node in mst.nodes:
            assert stats.loc[node, "degree"] == deg[node]
            assert stats.loc[node, "betweenness"] == pytest.approx(bet[node], abs=1e-10)
            assert stats.loc[node, "eigenvector"] == pytest.approx(eig[node], abs=1e-6)


def test_summary_line_mentions_hub() -> None:
    line = nw.mst_summary_line(nw.build_mst(chain_corr(list("ABCDE"), [0, 1, 2, 3, 4])))
    assert "n=5" in line and "edges=4" in line and "length=1.2000" in line


# ------------------------------------------------------------------ clustering


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_single_linkage_heights_equal_sorted_mst_weights(seed: int) -> None:
    corr = random_corr(13, seed=seed)
    mst = nw.build_mst(corr)
    condensed = squareform(nw.correlation_distance(corr).to_numpy(), checks=False)
    heights = linkage(condensed, "single")[:, 2]
    assert sorted(heights) == pytest.approx(sorted(mst.edges["distance"]))


def test_clusters_recover_blocks() -> None:
    corr = block_corr([4, 5, 3], within=0.7, across=0.05)
    for method in ("single", "average", "complete"):
        clusters = nw.hierarchical_clusters(corr, 3, method=method)
        by_block = clusters.groupby(lambda lab: str(lab).split("_")[0]).nunique()
        assert (by_block == 1).all()
        assert clusters.nunique() == 3


def test_cluster_extremes_and_numbering() -> None:
    corr = block_corr([3, 3], within=0.6, across=0.1)
    assert (nw.hierarchical_clusters(corr, 1) == 0).all()
    assert sorted(nw.hierarchical_clusters(corr, 6)) == list(range(6))
    two = nw.hierarchical_clusters(corr, 2)
    assert two.iloc[0] == 0  # numbered by first appearance in sorted-label order
    assert list(two.index) == sorted(two.index)


def test_cluster_order_invariance() -> None:
    corr = block_corr([4, 4], within=0.6, across=0.1)
    perm = list(corr.index[[7, 2, 5, 0, 3, 6, 1, 4]])
    pd.testing.assert_series_equal(
        nw.hierarchical_clusters(corr, 2, "average"),
        nw.hierarchical_clusters(corr.loc[perm, perm], 2, "average"),
    )


@pytest.mark.parametrize("n_clusters, method", [(0, "single"), (99, "single"), (2, "ward")])
def test_cluster_bad_arguments(n_clusters: int, method: str) -> None:
    with pytest.raises(nw.NetworkError):
        nw.hierarchical_clusters(random_corr(5, seed=0), n_clusters, method=method)


def test_dendrogram_order_is_a_permutation_with_contiguous_blocks() -> None:
    corr = block_corr([4, 4, 4], within=0.7, across=0.05)
    order = nw.dendrogram_order(corr)
    assert sorted(order) == sorted(corr.index)
    blocks = [lab.split("_")[0] for lab in order]
    changes = sum(1 for a, b in pairwise(blocks) if a != b)
    assert changes == 2


# ------------------------------------------------------------------ survival


def test_survival_ratio_known_answers() -> None:
    labels = list("ABCDE")
    a = nw.build_mst(chain_corr(labels, [0, 1, 2, 3, 4]))
    b = nw.build_mst(chain_corr(labels, [0, 1, 2, 4, 3]))  # chain order A,B,C,E,D
    assert nw.edge_survival_ratio(a, a) == 1.0
    assert nw.edge_survival_ratio(a, b) == pytest.approx(0.75)  # AB, BC, DE shared
    assert nw.edge_survival_ratio(a, b) == nw.edge_survival_ratio(b, a)


def test_survival_ratio_requires_same_nodes() -> None:
    a = nw.build_mst(chain_corr(list("ABC"), [0, 1, 2]))
    b = nw.build_mst(chain_corr(list("ABD"), [0, 1, 2]))
    with pytest.raises(nw.NetworkError, match="same nodes"):
        nw.edge_survival_ratio(a, b)


def _factor_returns(t: int, n: int, loading: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = [f"S{k // 4}_{k % 4}" for k in range(n)]
    sector = rng.standard_normal((t, n // 4))
    noise = rng.standard_normal((t, n))
    data = noise + loading * np.repeat(sector, 4, axis=1)
    index = pd.bdate_range("2016-01-01", periods=t)
    return pd.DataFrame(data, index=index, columns=labels)


def test_rolling_survival_shape_and_range() -> None:
    returns = _factor_returns(400, 12, 1.5, seed=0)
    out = nw.rolling_mst_survival(returns, window=100, step=20)
    assert out.name == "mst_survival"
    assert out.index[0] == returns.index[99]
    assert len(out) == len(range(99, 400, 20))
    assert np.isnan(out.iloc[0])
    assert out.iloc[1:].between(0.0, 1.0).all()


def test_rolling_survival_higher_with_stable_structure_than_pure_noise() -> None:
    structured = _factor_returns(1200, 12, 1.5, seed=1)
    noise = _factor_returns(1200, 12, 0.0, seed=1)
    s = nw.rolling_mst_survival(structured, window=250, step=125).mean()
    n = nw.rolling_mst_survival(noise, window=250, step=125).mean()
    assert s > n


def test_rolling_survival_passes_exact_windows_to_corr_fn() -> None:
    returns = _factor_returns(60, 8, 1.0, seed=2)
    seen: list[tuple[pd.Timestamp, int]] = []

    def spy(frame: pd.DataFrame) -> pd.DataFrame:
        seen.append((frame.index[-1], len(frame)))
        return frame.corr()

    out = nw.rolling_mst_survival(returns, window=30, step=10, corr_fn=spy)
    assert [length for _, length in seen] == [30] * len(out)
    assert [date for date, _ in seen] == list(out.index)


def test_rolling_survival_short_window_gives_nan() -> None:
    returns = _factor_returns(80, 8, 1.0, seed=3)
    returns.iloc[20:40, 0] = np.nan  # complete-case rows drop out around here
    out = nw.rolling_mst_survival(returns, window=30, step=10, min_obs=25)
    assert out.isna().any()
    assert out.iloc[-1] == out.iloc[-1]  # last window (rows 50..79) is complete: finite


@pytest.mark.parametrize(
    "window, step, min_obs",
    [(2, 1, 10), (30, 0, 10), (30, 1, 2), (10_000, 1, 10)],
)
def test_rolling_survival_bad_arguments(window: int, step: int, min_obs: int) -> None:
    with pytest.raises(nw.NetworkError):
        nw.rolling_mst_survival(_factor_returns(100, 8, 1.0, seed=0), window, step, min_obs=min_obs)


def test_rolling_survival_rejects_unsorted_index() -> None:
    returns = _factor_returns(100, 8, 1.0, seed=0).iloc[::-1]
    with pytest.raises(nw.NetworkError, match="increasing"):
        nw.rolling_mst_survival(returns, window=30)


# ------------------------------------------------------------------ sector purity


def test_sector_purity_known_answer() -> None:
    mst = nw.build_mst(chain_corr(list("ABCDE"), [0, 1, 2, 3, 4]))
    sectors = {"A": "x", "B": "x", "C": "y", "D": "y", "E": "y"}
    assert nw.sector_edge_purity(mst, sectors) == pytest.approx(0.75)  # BC crosses sectors


def test_sector_labels_must_cover_nodes() -> None:
    mst = nw.build_mst(chain_corr(list("ABC"), [0, 1, 2]))
    with pytest.raises(nw.NetworkError, match="no sector"):
        nw.sector_edge_purity(mst, {"A": "x", "B": "x"})


def test_purity_test_detects_real_structure() -> None:
    corr = block_corr([6, 6], within=0.7, across=0.05)
    rng = np.random.default_rng(0)
    noisy = corr.to_numpy() + 0.01 * rng.standard_normal(corr.shape)
    noisy = (noisy + noisy.T) / 2
    np.fill_diagonal(noisy, 1.0)
    mst = nw.build_mst(pd.DataFrame(noisy, index=corr.index, columns=corr.columns))
    sectors = {lab: lab.split("_")[0] for lab in corr.index}
    result = nw.sector_purity_test(mst, sectors, n_permutations=2000, seed=1)
    assert result.observed == pytest.approx(10 / 11)  # one unavoidable cross-block edge
    assert result.p_value < 0.01
    assert result.null_mean < result.observed


def test_purity_test_determinism_and_edge_cases() -> None:
    corr = random_corr(10, seed=5)
    mst = nw.build_mst(corr)
    sectors = {lab: "a" if k % 2 else "b" for k, lab in enumerate(mst.nodes)}
    first = nw.sector_purity_test(mst, sectors, n_permutations=200, seed=7)
    second = nw.sector_purity_test(mst, sectors, n_permutations=200, seed=7)
    assert first == second
    assert 0.0 < first.p_value <= 1.0
    single = nw.sector_purity_test(mst, {lab: "z" for lab in mst.nodes}, n_permutations=50)
    assert single.observed == 1.0 and single.p_value == 1.0
    with pytest.raises(nw.NetworkError):
        nw.sector_purity_test(mst, sectors, n_permutations=0)


def test_purity_test_is_calibrated_under_the_null() -> None:
    """Sector labels unrelated to returns: p < 0.05 should occur about 5% of the time."""
    hits = 0
    trials = 300
    for seed in range(trials):
        mst = nw.build_mst(random_corr(16, seed=seed, t=120))
        rng = np.random.default_rng(10_000 + seed)
        names = rng.permutation(np.repeat(["a", "b", "c", "d"], 4))
        sectors = dict(zip(mst.nodes, names.tolist(), strict=True))
        p = nw.sector_purity_test(mst, sectors, n_permutations=199, seed=seed).p_value
        hits += p < 0.05
    rate = hits / trials
    assert 0.01 <= rate <= 0.10
