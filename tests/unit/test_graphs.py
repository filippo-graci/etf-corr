"""Tests for ``etfcorr.models.graphs`` (step 7.1) on panels with a known sector structure."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from etfcorr import features as F
from etfcorr.models import graphs as G
from etfcorr.models.base import ModelError
from etfcorr.models.har import HAR
from etfcorr.synthetic import make_factor_returns

PANEL = make_factor_returns(2000, (3, 3, 3), seed=3)
ASSETS = list(PANEL.returns.columns)
SECTORS = dict(zip(ASSETS, PANEL.sectors, strict=True))
CORR = PANEL.returns.corr().to_numpy()  # within about 0.5, across about 0.25
N = len(ASSETS)
SECTOR_ADJ = G.sector_graph(ASSETS, SECTORS)
WINDOWS = (21, 63)


def edges(adj: np.ndarray) -> set[tuple[int, int]]:
    i, j = np.nonzero(np.triu(adj, k=1))
    return set(zip(i.tolist(), j.tolist(), strict=True))


def test_every_graph_is_symmetric_without_self_loops() -> None:
    for kind in ("knn", "threshold", "mst", "pmfg", "cluster", "sector", "random"):
        adj = G.build_graph(kind, CORR, ASSETS, SECTORS, knn_k=2, threshold_quantile=0.75)
        assert adj.dtype == bool and (adj == adj.T).all() and not adj.diagonal().any(), kind


def test_knn_threshold_and_cluster_recover_the_planted_sectors() -> None:
    assert (G.knn_graph(CORR, 2) == SECTOR_ADJ).all()  # the two sector mates
    assert (G.threshold_graph(CORR, 0.75) == SECTOR_ADJ).all()  # 9 of 36 pairs are within
    assert (G.cluster_graph(CORR, 3) == SECTOR_ADJ).all()
    assert len(edges(SECTOR_ADJ)) == 9


def test_mst_equals_networkx_and_pmfg_is_maximal_planar_and_contains_it() -> None:
    dist = np.sqrt(2.0 * (1.0 - CORR))
    reference = nx.minimum_spanning_tree(nx.from_numpy_array(dist))
    mst = G.mst_graph(CORR)
    assert edges(mst) == {(min(a, b), max(a, b)) for a, b in reference.edges}
    pmfg = G.pmfg_graph(CORR)
    assert len(edges(pmfg)) == 3 * (N - 2)
    assert nx.check_planarity(nx.from_numpy_array(pmfg.astype(int)))[0]
    assert edges(mst) <= edges(pmfg)


def test_random_graph_matches_the_knn_edge_count_and_its_seed() -> None:
    kwargs = {"knn_k": 2, "threshold_quantile": 0.75}
    a = G.build_graph("random", CORR, ASSETS, SECTORS, seed=0, **kwargs)
    b = G.build_graph("random", CORR, ASSETS, SECTORS, seed=0, **kwargs)
    c = G.build_graph("random", CORR, ASSETS, SECTORS, seed=1, **kwargs)
    assert len(edges(a)) == len(edges(G.knn_graph(CORR, 2))) == 9
    assert (a == b).all() and (a != c).any()


def test_bad_requests_are_refused() -> None:
    with pytest.raises(ModelError):
        G.knn_graph(CORR, N)
    with pytest.raises(ModelError):
        G.random_graph(N, 0, 0)
    with pytest.raises(ModelError):
        G.build_graph("knn", CORR[:4, :4], ASSETS, SECTORS, knn_k=2, threshold_quantile=0.5)
    with pytest.raises(ModelError, match="unknown graph kind"):
        G.build_graph("star", CORR, ASSETS, SECTORS, knn_k=2, threshold_quantile=0.5)  # type: ignore[arg-type]


def test_pair_table_holds_the_trailing_correlations() -> None:
    table = G.pair_table(PANEL.returns.iloc[:200], WINDOWS)
    assert table.shape == (200, 2 * N * (N - 1) // 2)
    corr = F.trailing_correlations(PANEL.returns.iloc[:200], 63)
    assert table[f"pz63|{ASSETS[0]}|{ASSETS[4]}"].iloc[-1] == pytest.approx(
        np.arctanh(corr[-1, 0, 4])
    )
    assert table["pz21|S0_0|S0_1"].iloc[:20].isna().all()


def augmented(kind: str = "sector") -> G.GraphAugmented:
    return G.GraphAugmented(
        lambda: HAR(G.graph_columns(WINDOWS)),
        kind,  # type: ignore[arg-type]
        ASSETS,
        SECTORS,
        windows=WINDOWS,
        similarity_window=63,
        knn_k=2,
        threshold_quantile=0.75,
    )


def test_graph_features_are_the_edge_and_non_edge_average_correlations() -> None:
    returns = PANEL.returns.iloc[:400]
    X = G.pair_table(returns, WINDOWS).dropna()
    y = pd.Series(np.linspace(0.0, 1.0, len(X)), index=X.index)
    model = augmented().fit(X, y)
    wide = model._augment(X)
    assert list(wide.columns[-4:]) == G.graph_columns(WINDOWS)
    within = F.sector_pair_mask(ASSETS, SECTORS, "within")
    cross = F.sector_pair_mask(ASSETS, SECTORS, "cross")
    for w in WINDOWS:
        corr = F.trailing_correlations(returns, w)[-len(X) :]
        np.testing.assert_allclose(
            wide[f"g_z_edge_w{w}"], F.fisher_z(F.masked_average(corr, within))
        )
        np.testing.assert_allclose(
            wide[f"g_z_nonedge_w{w}"], F.fisher_z(F.masked_average(corr, cross))
        )
    assert model.params["graph"] == "sector" and model.params["n_edges"] == 9
    assert model.predict(X).notna().all()


def test_the_graph_comes_from_the_rows_given_to_fit_and_is_reused_in_predict() -> None:
    """A noise panel has no stable graph: the first and second half give different trees, and
    predicting later rows does not change the fitted one."""
    noise = make_factor_returns(900, (9,), market_loading=0.3, sector_loading=0.0, seed=5)
    assets = list(noise.returns.columns)
    X = G.pair_table(noise.returns, WINDOWS).dropna()
    y = pd.Series(0.0, index=X.index)
    half = len(X) // 2

    def fit(rows: slice) -> G.GraphAugmented:
        model = G.GraphAugmented(
            lambda: HAR(G.graph_columns(WINDOWS)),
            "mst",
            assets,
            dict.fromkeys(assets, "s"),
            windows=WINDOWS,
            similarity_window=63,
            knn_k=2,
            threshold_quantile=0.75,
        )
        return model.fit(X.iloc[rows], y.iloc[rows])

    first, second = fit(slice(0, half)), fit(slice(half, None))
    assert first.adjacency_ is not None and second.adjacency_ is not None
    assert (first.adjacency_ != second.adjacency_).any()
    before = first.adjacency_.copy()
    first.predict(X.iloc[half:])
    assert (first.adjacency_ == before).all()


def test_wrapper_refusals() -> None:
    X = G.pair_table(PANEL.returns.iloc[:200], WINDOWS).dropna()
    y = pd.Series(0.0, index=X.index)
    with pytest.raises(ModelError, match="before fit"):
        augmented().predict(X)
    with pytest.raises(ModelError, match="pair columns"):
        augmented().fit(X[[c for c in X.columns if c.startswith("pz21")]], y)
    one_sector = G.GraphAugmented(
        lambda: HAR(G.graph_columns(WINDOWS)),
        "sector",
        ASSETS,
        dict.fromkeys(ASSETS, "s"),
        windows=WINDOWS,
        similarity_window=63,
        knn_k=2,
        threshold_quantile=0.75,
    )
    with pytest.raises(ModelError, match="complete"):
        one_sector.fit(X, y)
    with pytest.raises(ModelError, match="similarity_window"):
        G.GraphAugmented(
            lambda: HAR(["a"]),
            "knn",
            ASSETS,
            SECTORS,
            windows=(21,),
            similarity_window=63,
            knn_k=2,
            threshold_quantile=0.5,
        )


def test_threshold_graph_keeps_a_pair_exactly_at_the_cut() -> None:
    sim = np.eye(4)
    i, j = np.triu_indices(4, k=1)
    sim[i, j] = sim[j, i] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert np.quantile(sim[i, j], 0.8) == 0.5  # the cut is one of the six values
    assert edges(G.threshold_graph(sim, 0.8)) == {(1, 3), (2, 3)}


def test_the_graph_is_built_from_the_mean_of_the_training_rows_not_the_last_one() -> None:
    X = G.pair_table(PANEL.returns.iloc[:400], WINDOWS).dropna()
    y = pd.Series(0.0, index=X.index)
    changed = X.copy()
    columns = G.pair_columns(ASSETS, 63)
    changed.iloc[-1, [X.columns.get_loc(c) for c in columns]] = -X[columns].iloc[-1].to_numpy()
    base, moved = augmented("knn").fit(X, y), augmented("knn").fit(changed, y)
    assert base.adjacency_ is not None and moved.adjacency_ is not None
    assert (base.adjacency_ == moved.adjacency_).all()  # one row in 338 does not move the mean
    last = np.eye(N)
    i, j = np.triu_indices(N, k=1)
    last[i, j] = last[j, i] = F.inverse_fisher_z(changed[columns].iloc[-1].to_numpy())
    assert (G.knn_graph(last, 2) != base.adjacency_).any()  # the last row alone would
