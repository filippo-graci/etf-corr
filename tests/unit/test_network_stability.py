"""Unit tests for ``etfcorr.correlation.network_stability`` (step 3.6), synthetic data only.

Known answers come from theory. The noise floor of the edge-survival ratio is ``2 / N`` by label
exchangeability (the MST of exchangeable noise is a label-exchangeable random tree). A panel with
strong sector factors gives trees whose edges are mostly within-sector and mostly well supported; a
pure-noise panel does not. Cluster agreement is perfect (adjusted Rand index 1) when sector
structure dominates and about 0 when it does not. A panel whose structure flips every window gives
rolling persistence far below its row-shuffled version, which mixes the regimes.
"""

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import network_stability as NS
from etfcorr.correlation.estimators import estimate
from etfcorr.correlation.horizon_analysis import (
    default_mean_block_length,
    stationary_bootstrap_indices,
)
from etfcorr.correlation.networks import build_mst, edge_survival_ratio
from etfcorr.correlation.rmt import detone_corr
from etfcorr.synthetic import make_factor_returns, make_noise_returns

SEED = 20261004


@pytest.fixture(scope="module")
def strong() -> tuple[pd.DataFrame, dict[str, str]]:
    """Three sectors of five assets with dominant sector factors."""
    panel = make_factor_returns(600, (5, 5, 5), market_loading=0.3, sector_loading=0.85, seed=SEED)
    return panel.returns, dict(zip(panel.returns.columns, panel.sectors, strict=True))


@pytest.fixture(scope="module")
def noise() -> pd.DataFrame:
    return make_noise_returns(600, 15, seed=SEED).returns


# --------------------------------------------------------------------------- correlation callables


def test_pearson_and_detoned_are_symmetric_unit_diagonal_and_keep_labels(
    strong: tuple[pd.DataFrame, dict[str, str]],
) -> None:
    returns, _ = strong
    for fn in (NS.pearson, NS.detoned_pearson):
        corr = fn(returns)
        assert list(corr.columns) == list(returns.columns) == list(corr.index)
        values = corr.to_numpy()
        np.testing.assert_allclose(values, values.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(values), 1.0, atol=1e-12)


def test_pearson_matches_the_estimator_and_detoned_matches_detone_corr(
    strong: tuple[pd.DataFrame, dict[str, str]],
) -> None:
    returns, _ = strong
    pd.testing.assert_frame_equal(NS.pearson(returns), estimate(returns, "pearson"))
    for n_market in (1, 2):
        expected = detone_corr(estimate(returns, "pearson"), n_market=n_market)
        pd.testing.assert_frame_equal(NS.detoned_pearson(returns, n_market), expected)


def test_detoning_lowers_the_average_off_diagonal_correlation(
    strong: tuple[pd.DataFrame, dict[str, str]],
) -> None:
    returns, _ = strong
    off = ~np.eye(returns.shape[1], dtype=bool)
    assert (
        NS.detoned_pearson(returns).to_numpy()[off].mean()
        < NS.pearson(returns).to_numpy()[off].mean()
    )


# --------------------------------------------------------------------------- edge support


def test_edge_support_has_n_minus_one_edges_with_support_in_unit_interval(
    strong: tuple[pd.DataFrame, dict[str, str]],
) -> None:
    returns, _ = strong
    es = NS.bootstrap_mst_edge_support(returns, n_bootstrap=40, seed=1)
    n = returns.shape[1]
    assert len(es.edges) == n - 1 == len(es.mst.edges)
    assert es.edges["support"].between(0.0, 1.0).all()
    assert set(es.support) == set(es.mst.edge_set)
    assert es.n_bootstrap == 40 and es.n_obs == len(returns)
    assert 0.0 <= es.survival_lower <= es.mean_survival <= es.survival_upper <= 1.0
    assert es.mean_survival == pytest.approx(es.edges["support"].mean())  # shared / (N - 1)


def test_edge_support_columns_extend_the_mst_edge_table(
    strong: tuple[pd.DataFrame, dict[str, str]],
) -> None:
    returns, _ = strong
    es = NS.bootstrap_mst_edge_support(returns, n_bootstrap=20, seed=1)
    assert list(es.edges.columns) == [*es.mst.edges.columns, "support"]
    pd.testing.assert_frame_equal(es.edges.drop(columns="support"), es.mst.edges)


@pytest.mark.parametrize(
    ("corr_fn", "alpha"),
    [(NS.pearson, 0.05), (NS.pearson, 0.4), (NS.detoned_pearson, 0.4)],
)
def test_edge_support_matches_an_independent_recount(
    strong: tuple[pd.DataFrame, dict[str, str]], corr_fn: NS.CorrFn, alpha: float
) -> None:
    returns, _ = strong
    n_boot, seed = 40, 5
    es = NS.bootstrap_mst_edge_support(
        returns, corr_fn=corr_fn, n_bootstrap=n_boot, alpha=alpha, seed=seed
    )
    point = build_mst(corr_fn(returns))
    block = default_mean_block_length(len(returns))
    rng = np.random.default_rng(seed)
    values = returns.to_numpy()
    counts = {edge: 0 for edge in point.edge_set}
    survival = []
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(len(returns), block, rng)
        tree = build_mst(corr_fn(pd.DataFrame(values[idx], columns=returns.columns)))
        for edge in tree.edge_set & point.edge_set:
            counts[edge] += 1
        survival.append(len(tree.edge_set & point.edge_set) / (len(point.nodes) - 1))
    expected = {edge: c / n_boot for edge, c in counts.items()}
    assert es.support == pytest.approx(expected)
    assert es.mean_survival == pytest.approx(np.mean(survival))
    assert es.survival_lower == pytest.approx(np.quantile(survival, alpha / 2.0))
    assert es.survival_upper == pytest.approx(np.quantile(survival, 1.0 - alpha / 2.0))
    assert es.mean_block_length == block


def test_structured_panel_has_better_supported_and_more_sector_pure_edges_than_noise(
    strong: tuple[pd.DataFrame, dict[str, str]], noise: pd.DataFrame
) -> None:
    returns, sectors = strong
    structured = NS.bootstrap_mst_edge_support(returns, n_bootstrap=100, seed=1)
    flat = NS.bootstrap_mst_edge_support(noise, n_bootstrap=100, seed=1)
    assert structured.edges["support"].mean() > flat.edges["support"].mean() + 0.1
    same = [
        sectors[s] == sectors[t]
        for s, t in zip(structured.edges["source"], structured.edges["target"], strict=True)
    ]
    assert np.mean(same) > 0.7  # 12 of 14 edges are within a sector for these loadings
    # the within-sector edges are the ones the data actually support
    assert (
        structured.edges["support"][same].mean()
        > structured.edges["support"][[not s for s in same]].mean()
    )


def test_same_seed_reproduces_and_a_different_seed_changes_the_support(
    strong: tuple[pd.DataFrame, dict[str, str]],
) -> None:
    returns, _ = strong
    a = NS.bootstrap_mst_edge_support(returns, n_bootstrap=30, seed=3)
    b = NS.bootstrap_mst_edge_support(returns, n_bootstrap=30, seed=3)
    c = NS.bootstrap_mst_edge_support(returns, n_bootstrap=30, seed=4)
    pd.testing.assert_frame_equal(a.edges, b.edges)
    assert a.mean_survival == b.mean_survival
    assert not np.array_equal(a.edges["support"].to_numpy(), c.edges["support"].to_numpy())


def test_a_custom_corr_fn_builds_the_point_tree_and_replicates_from_that_matrix(
    strong: tuple[pd.DataFrame, dict[str, str]],
) -> None:
    returns, _ = strong
    es = NS.bootstrap_mst_edge_support(returns, corr_fn=NS.detoned_pearson, n_bootstrap=20, seed=1)
    assert es.mst.edge_set == build_mst(NS.detoned_pearson(returns)).edge_set
    assert es.mst.edge_set != build_mst(NS.pearson(returns)).edge_set


def test_block_length_default_and_override(strong: tuple[pd.DataFrame, dict[str, str]]) -> None:
    returns, _ = strong
    default = NS.bootstrap_mst_edge_support(returns, n_bootstrap=20, seed=1)
    assert default.mean_block_length == default_mean_block_length(len(returns))
    assert (
        NS.bootstrap_mst_edge_support(
            returns, n_bootstrap=20, seed=1, mean_block_length=12.0
        ).mean_block_length
        == 12.0
    )


def test_missing_rows_are_dropped_and_the_input_is_not_modified(
    strong: tuple[pd.DataFrame, dict[str, str]],
) -> None:
    returns, _ = strong
    holey = returns.copy()
    holey.iloc[10, 0] = np.nan
    holey.iloc[50, 3] = np.nan
    before = holey.copy()
    es = NS.bootstrap_mst_edge_support(holey, n_bootstrap=20, seed=1)
    assert es.n_obs == len(returns) - 2
    pd.testing.assert_frame_equal(holey, before)


def test_default_block_length_uses_the_complete_case_row_count() -> None:
    """round(T ** (1/3)) crosses 8 -> 9 between 612 and 616 rows, so the source of T shows."""
    returns = make_factor_returns(616, (3, 3, 3), seed=SEED).returns
    holey = returns.copy()
    holey.iloc[[3, 100, 200, 300], 0] = np.nan
    es = NS.bootstrap_mst_edge_support(holey, n_bootstrap=20, seed=1)
    assert es.n_obs == 612
    assert es.mean_block_length == default_mean_block_length(612)
    assert default_mean_block_length(612) != default_mean_block_length(616)


def test_edge_support_rejects_bad_input(strong: tuple[pd.DataFrame, dict[str, str]]) -> None:
    returns, _ = strong
    with pytest.raises(NS.NetworkStabilityError, match="DataFrame"):
        NS.bootstrap_mst_edge_support(returns.to_numpy())  # type: ignore[arg-type]
    with pytest.raises(NS.NetworkStabilityError, match="n_bootstrap"):
        NS.bootstrap_mst_edge_support(returns, n_bootstrap=NS.MIN_BOOTSTRAP - 1)
    with pytest.raises(NS.NetworkStabilityError, match="alpha"):
        NS.bootstrap_mst_edge_support(returns, n_bootstrap=20, alpha=0.7)
    with pytest.raises(NS.NetworkStabilityError, match="complete-case"):
        NS.bootstrap_mst_edge_support(returns.iloc[:5], n_bootstrap=20)
    with pytest.raises(NS.NetworkStabilityError, match="complete-case"):
        NS.bootstrap_mst_edge_support(returns.assign(S0_0=np.nan), n_bootstrap=20)


# --------------------------------------------------------------------------- survival matrix


def test_survival_matrix_is_symmetric_with_unit_diagonal_and_matches_pairwise_ratios(
    strong: tuple[pd.DataFrame, dict[str, str]],
) -> None:
    returns, _ = strong
    trees = {
        "all": build_mst(returns.corr()),
        "first": build_mst(returns.iloc[:300].corr()),
        "second": build_mst(returns.iloc[300:].corr()),
    }
    matrix = NS.survival_matrix(trees)
    assert list(matrix.index) == list(matrix.columns) == ["all", "first", "second"]
    np.testing.assert_allclose(matrix.to_numpy(), matrix.to_numpy().T)
    np.testing.assert_allclose(np.diag(matrix.to_numpy()), 1.0)
    assert matrix.loc["all", "first"] == pytest.approx(
        edge_survival_ratio(trees["all"], trees["first"])
    )
    assert matrix.to_numpy().min() >= 0.0


def test_survival_matrix_needs_two_trees(strong: tuple[pd.DataFrame, dict[str, str]]) -> None:
    returns, _ = strong
    with pytest.raises(NS.NetworkStabilityError, match="two trees"):
        NS.survival_matrix({"only": build_mst(returns.corr())})


# --------------------------------------------------------------------------- noise floor


@pytest.mark.parametrize("n_nodes", [3, 10])
def test_noise_floor_mean_matches_two_over_n(n_nodes: int) -> None:
    null = NS.noise_mst_survival_null(n_nodes, 60, n_draws=400, seed=0)
    assert null.expected_mean == pytest.approx(2.0 / n_nodes)
    assert abs(null.mean - 2.0 / n_nodes) < 0.04
    assert null.values.min() >= 0.0 and null.values.max() <= 1.0
    assert null.mean <= null.q95 <= 1.0
    assert null.n_nodes == n_nodes and null.n_obs == 60 and len(null.values) == 400


def test_noise_floor_is_reproducible_and_depends_on_the_seed() -> None:
    a = NS.noise_mst_survival_null(8, 40, n_draws=50, seed=1)
    b = NS.noise_mst_survival_null(8, 40, n_draws=50, seed=1)
    c = NS.noise_mst_survival_null(8, 40, n_draws=50, seed=2)
    np.testing.assert_array_equal(a.values, b.values)
    assert not np.array_equal(a.values, c.values)


def test_noise_floor_rejects_bad_arguments() -> None:
    with pytest.raises(NS.NetworkStabilityError, match="n_nodes"):
        NS.noise_mst_survival_null(2, 40)
    with pytest.raises(NS.NetworkStabilityError, match="n_obs"):
        NS.noise_mst_survival_null(8, NS.MIN_OBS - 1)
    with pytest.raises(NS.NetworkStabilityError, match="n_draws"):
        NS.noise_mst_survival_null(8, 40, n_draws=0)


def test_real_survival_between_halves_of_a_structured_panel_clears_the_noise_bar(
    strong: tuple[pd.DataFrame, dict[str, str]],
) -> None:
    returns, _ = strong
    a = build_mst(returns.iloc[:300].corr())
    b = build_mst(returns.iloc[300:].corr())
    null = NS.noise_mst_survival_null(returns.shape[1], 300, n_draws=300, seed=0)
    assert edge_survival_ratio(a, b) > null.q95


# --------------------------------------------------------------------------- rolling null


def _flipping_regimes(window: int = 120, blocks: int = 20) -> pd.DataFrame:
    """Sector structure that is reassigned to other columns in every second block."""
    base = make_factor_returns(
        window * blocks, (5, 5, 5), market_loading=0.3, sector_loading=0.85, seed=3
    ).returns
    perm = np.random.default_rng(0).permutation(base.shape[1])
    values = base.to_numpy()
    parts = []
    for k in range(blocks):
        chunk = values[k * window : (k + 1) * window]
        parts.append(chunk if k % 2 == 0 else chunk[:, perm])
    return pd.DataFrame(np.vstack(parts), index=base.index, columns=base.columns)


def test_flipping_structure_persists_far_less_than_its_row_shuffled_version() -> None:
    result = NS.rolling_survival_shuffle_null(_flipping_regimes(), 120, 120, n_shuffles=40, seed=0)
    assert result.real_mean < result.null_means.min() - 0.1
    assert result.lower_tail == pytest.approx(1.0 / 41.0)  # no shuffle is at or below the real
    assert result.upper_tail == pytest.approx(1.0)
    assert result.p_two_sided == pytest.approx(2.0 / 41.0)
    assert result.n_shuffles == 40 and len(result.null_means) == 40


def test_stationary_panel_sits_inside_its_shuffled_distribution() -> None:
    base = make_factor_returns(
        2400, (5, 5, 5), market_loading=0.3, sector_loading=0.85, seed=3
    ).returns
    result = NS.rolling_survival_shuffle_null(base, 120, 120, n_shuffles=50, seed=0)
    assert result.null_means.min() <= result.real_mean <= result.null_means.max()
    assert result.lower_tail > 0.1 and result.upper_tail > 0.1


def test_tied_values_count_on_both_sides_of_the_empirical_tails() -> None:
    """A chain A - B - C with a strong, well separated structure gives the same tree in every
    window of 1000 rows, so real and shuffled survival are all exactly 1 (a tie everywhere)."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal(4000)
    b = 0.8 * a + 0.6 * rng.standard_normal(4000)
    c = 0.8 * b + 0.6 * rng.standard_normal(4000)
    frame = pd.DataFrame({"A": a, "B": b, "C": c}, index=pd.bdate_range("2016-01-04", periods=4000))
    result = NS.rolling_survival_shuffle_null(frame, 1000, 1000, n_shuffles=10, seed=0)
    assert result.real_mean == 1.0 and (result.null_means == 1.0).all()
    assert result.lower_tail == 1.0 and result.upper_tail == 1.0
    assert result.p_two_sided == 1.0


def test_rolling_null_is_reproducible_and_does_not_modify_its_input() -> None:
    frame = _flipping_regimes(blocks=8)
    before = frame.copy()
    a = NS.rolling_survival_shuffle_null(frame, 120, 120, n_shuffles=10, seed=2)
    b = NS.rolling_survival_shuffle_null(frame, 120, 120, n_shuffles=10, seed=2)
    np.testing.assert_array_equal(a.null_means, b.null_means)
    pd.testing.assert_series_equal(a.real, b.real)
    pd.testing.assert_frame_equal(frame, before)


def test_rolling_null_accepts_a_corr_fn() -> None:
    frame = _flipping_regimes(blocks=8)
    raw = NS.rolling_survival_shuffle_null(frame, 120, 120, n_shuffles=5, seed=2)
    detoned = NS.rolling_survival_shuffle_null(
        frame, 120, 120, n_shuffles=5, seed=2, corr_fn=NS.detoned_pearson
    )
    assert not np.array_equal(raw.null_means, detoned.null_means)
    # (the first value is NaN by construction, so compare with equal_nan)
    assert not np.array_equal(raw.real.to_numpy(), detoned.real.to_numpy(), equal_nan=True)


def test_rolling_null_rejects_bad_arguments() -> None:
    frame = _flipping_regimes(blocks=8)
    with pytest.raises(NS.NetworkStabilityError, match="n_shuffles"):
        NS.rolling_survival_shuffle_null(frame, 120, 120, n_shuffles=0)
    with pytest.raises(NS.NetworkStabilityError, match="defined survival values"):
        NS.rolling_survival_shuffle_null(frame, 480, 480, n_shuffles=5)


def test_rolling_null_result_tail_arithmetic() -> None:
    real = pd.Series([0.5, 0.5], name="mst_survival")
    base = dict(real=real, real_mean=0.5, null_means=np.array([0.1, 0.2, 0.3]), n_shuffles=3)
    assert NS.RollingSurvivalNull(lower_tail=0.25, upper_tail=0.75, **base).p_two_sided == 0.5
    assert NS.RollingSurvivalNull(lower_tail=0.75, upper_tail=0.25, **base).p_two_sided == 0.5
    assert NS.RollingSurvivalNull(lower_tail=0.9, upper_tail=0.8, **base).p_two_sided == 1.0
    assert NS.RollingSurvivalNull(lower_tail=0.25, upper_tail=0.75, **base).null_mean == (
        pytest.approx(0.2)
    )


# --------------------------------------------------------------------------- cluster agreement


def test_dominant_sector_structure_gives_perfect_agreement_and_a_small_p_value(
    strong: tuple[pd.DataFrame, dict[str, str]],
) -> None:
    returns, sectors = strong
    result = NS.cluster_sector_agreement(returns.corr(), sectors, n_permutations=300, seed=0)
    assert result.ari == pytest.approx(1.0)
    assert result.p_value == pytest.approx(1.0 / 301.0)
    assert abs(result.null_mean) < 0.05
    assert result.n_clusters == 3 and result.method == "average" and result.n_permutations == 300


def test_noise_gives_agreement_near_zero_and_a_large_p_value(noise: pd.DataFrame) -> None:
    sectors = {c: f"s{i % 3}" for i, c in enumerate(noise.columns)}
    result = NS.cluster_sector_agreement(noise.corr(), sectors, n_permutations=300, seed=0)
    assert abs(result.ari) < 0.2
    assert result.p_value > 0.05
    assert abs(result.null_mean) < 0.05


def test_agreement_is_invariant_to_sector_names_and_reproducible(
    strong: tuple[pd.DataFrame, dict[str, str]],
) -> None:
    returns, sectors = strong
    renamed = {t: f"zz_{s[::-1]}" for t, s in sectors.items()}
    a = NS.cluster_sector_agreement(returns.corr(), sectors, n_permutations=50, seed=4)
    b = NS.cluster_sector_agreement(returns.corr(), renamed, n_permutations=50, seed=4)
    assert a.ari == b.ari
    again = NS.cluster_sector_agreement(returns.corr(), sectors, n_permutations=50, seed=4)
    assert (a.ari, a.null_mean, a.p_value) == (again.ari, again.null_mean, again.p_value)


def test_cluster_count_and_method_can_be_set(strong: tuple[pd.DataFrame, dict[str, str]]) -> None:
    returns, sectors = strong
    coarse = NS.cluster_sector_agreement(
        returns.corr(), sectors, n_clusters=2, method="single", n_permutations=20, seed=0
    )
    assert coarse.n_clusters == 2 and coarse.method == "single"
    assert coarse.ari < 1.0  # two clusters cannot reproduce three sectors


def test_cluster_agreement_rejects_bad_input(strong: tuple[pd.DataFrame, dict[str, str]]) -> None:
    returns, sectors = strong
    with pytest.raises(NS.NetworkStabilityError, match="n_permutations"):
        NS.cluster_sector_agreement(returns.corr(), sectors, n_permutations=0)
    incomplete = {k: v for k, v in sectors.items() if k != "S0_0"}
    with pytest.raises(NS.NetworkStabilityError, match="no sector for nodes"):
        NS.cluster_sector_agreement(returns.corr(), incomplete, n_permutations=10)


# ---- exploratory follow-up E2 (docs/FOLLOW_UPS.md item 2)


def test_partition_similarity_is_one_for_the_same_blocks_and_low_for_noise() -> None:
    a = make_factor_returns(800, (4, 4, 4), sector_loading=0.6, seed=1).returns
    b = make_factor_returns(800, (4, 4, 4), sector_loading=0.6, seed=2).returns
    assert NS.partition_similarity(NS.pearson(a), NS.pearson(a), 3) == 1.0
    assert NS.partition_similarity(NS.pearson(a), NS.pearson(b), 3) == 1.0  # same planted sectors
    assert NS.partition_similarity(NS.detoned_pearson(a), NS.detoned_pearson(b), 3) == 1.0
    noise_a = make_noise_returns(800, 12, seed=1).returns
    noise_b = make_noise_returns(800, 12, seed=2).returns
    assert NS.partition_similarity(NS.pearson(noise_a), NS.pearson(noise_b), 3) < 0.5
    with pytest.raises(NS.NetworkStabilityError):
        NS.partition_similarity(NS.pearson(a), NS.pearson(a.iloc[:, :6]), 3)
