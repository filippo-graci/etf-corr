"""Unit tests for ``etfcorr.correlation.stability`` (step 3.5), synthetic data only.

Known answers come from theory, not from fitting to a run: for independent noise the sampling
standard deviation of one sample correlation is about ``1 / sqrt(T)`` and the average of the
``N (N - 1) / 2`` (asymptotically uncorrelated) pairwise correlations has a standard deviation of
about ``1 / sqrt(T * N (N - 1) / 2)``.
"""

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import stability as S
from etfcorr.correlation.estimators import CorrelationError, estimate, ewma_corr
from etfcorr.correlation.horizon_analysis import default_mean_block_length
from etfcorr.correlation.rmt import denoise_corr, detone_corr
from etfcorr.synthetic import make_factor_returns, make_noise_returns

SEED = 20261004


def _blocks(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    """Sums of ``k`` consecutive daily rows, dated at the last day of each block."""
    n = len(frame) // k * k
    head = frame.iloc[:n]
    out = head.groupby(np.arange(n) // k).sum()
    out.index = head.index[k - 1 :: k]
    return out


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return make_factor_returns(1500, (3, 3, 3), seed=SEED).returns


@pytest.fixture(scope="module")
def horizons(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"D": daily, "W": _blocks(daily, 5), "M": _blocks(daily, 21)}


# --------------------------------------------------------------------------- restrict_window


def test_restrict_window_is_closed_on_both_sides_and_does_not_mutate(daily: pd.DataFrame) -> None:
    start, end = daily.index[100], daily.index[200]
    before = daily.copy()
    out = S.restrict_window(daily, str(start.date()), str(end.date()))
    assert out.index[0] == start and out.index[-1] == end and len(out) == 101
    pd.testing.assert_frame_equal(daily, before)
    out.iloc[0, 0] = 999.0  # a copy: editing the output must not touch the input
    pd.testing.assert_frame_equal(daily, before)


def test_restrict_window_without_bounds_returns_everything(daily: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(S.restrict_window(daily), daily)
    assert len(S.restrict_window(daily, end=str(daily.index[49].date()))) == 50
    assert len(S.restrict_window(daily, start=str(daily.index[-10].date()))) == 10


def test_restrict_window_rejects_bad_input(daily: pd.DataFrame) -> None:
    with pytest.raises(S.StabilityError, match="no rows"):
        S.restrict_window(daily, start="2090-01-01")
    with pytest.raises(S.StabilityError, match="DatetimeIndex"):
        S.restrict_window(daily.reset_index(drop=True))
    with pytest.raises(S.StabilityError, match="strictly increasing"):
        S.restrict_window(daily.iloc[::-1])
    dup = pd.concat([daily.iloc[:5], daily.iloc[:5]])
    with pytest.raises(S.StabilityError, match="strictly increasing"):
        S.restrict_window(dup)
    with pytest.raises(S.StabilityError, match="DataFrame"):
        S.restrict_window(daily["S0_0"])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- estimator_matrix


@pytest.mark.parametrize("name", S.ESTIMATORS)
def test_every_estimator_gives_a_labelled_symmetric_unit_diagonal_matrix(
    daily: pd.DataFrame, name: S.EstimatorName
) -> None:
    sample = daily.iloc[:300]
    m = S.estimator_matrix(sample, name)
    assert list(m.index) == list(sample.columns) == list(m.columns)
    values = m.to_numpy()
    np.testing.assert_allclose(values, values.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(values), 1.0, atol=1e-10)
    assert np.isfinite(values).all()


def test_estimator_matrix_matches_the_underlying_functions(daily: pd.DataFrame) -> None:
    sample = daily.iloc[:300]
    pd.testing.assert_frame_equal(
        S.estimator_matrix(sample, "pearson"), estimate(sample, "pearson")
    )
    pearson = estimate(sample, "pearson")
    pd.testing.assert_frame_equal(
        S.estimator_matrix(sample, "denoised"), denoise_corr(pearson, n_obs=300)
    )
    pd.testing.assert_frame_equal(
        S.estimator_matrix(sample, "detoned", n_market=2), detone_corr(pearson, n_market=2)
    )
    pd.testing.assert_frame_equal(
        S.estimator_matrix(sample, "ewma", ewma_lambda=0.9), ewma_corr(sample, 0.9)
    )


def test_estimator_matrix_rejects_an_unknown_name(daily: pd.DataFrame) -> None:
    with pytest.raises(S.StabilityError, match="unknown estimator"):
        S.estimator_matrix(daily.iloc[:50], "mystery")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- bootstrap_index_sets


def test_index_sets_have_the_right_shape_range_and_dtype() -> None:
    sets = S.bootstrap_index_sets(200, 30, 5.0, 0, "D")
    assert sets.shape == (30, 200) and sets.dtype == np.int64
    assert sets.min() >= 0 and sets.max() <= 199


def test_index_sets_are_deterministic_and_depend_on_seed_and_horizon() -> None:
    a = S.bootstrap_index_sets(100, 25, 4.0, 3, "W")
    np.testing.assert_array_equal(a, S.bootstrap_index_sets(100, 25, 4.0, 3, "W"))
    assert not np.array_equal(a, S.bootstrap_index_sets(100, 25, 4.0, 4, "W"))
    assert not np.array_equal(a, S.bootstrap_index_sets(100, 25, 4.0, 3, "M"))


def test_index_sets_have_the_requested_mean_block_length() -> None:
    """A step that continues a block moves to the next row; share of such steps = 1 - 1 / mbl."""
    n, mbl = 4000, 10.0
    sets = S.bootstrap_index_sets(n, 40, mbl, 1, "D")
    steps = (sets[:, 1:] - sets[:, :-1]) % n
    continuing = float(np.mean(steps == 1))
    # a new block also lands on "next row" by chance with probability 1 / n
    assert continuing == pytest.approx(1.0 - 1.0 / mbl, abs=0.01)


def test_index_sets_reject_bad_arguments() -> None:
    with pytest.raises(S.StabilityError, match="unknown horizon"):
        S.bootstrap_index_sets(50, 5, 3.0, 0, "X")
    with pytest.raises(S.StabilityError, match="n_bootstrap"):
        S.bootstrap_index_sets(50, 0, 3.0, 0, "D")


# --------------------------------------------------------------------------- the table


def test_table_has_one_row_per_horizon_and_estimator_with_the_documented_columns(
    horizons: dict[str, pd.DataFrame],
) -> None:
    estimators: tuple[S.EstimatorName, ...] = ("pearson", "ledoit_wolf", "detoned")
    table = S.estimator_stability_table(horizons, estimators, n_bootstrap=20)
    assert tuple(table.columns) == S.TABLE_COLUMNS
    assert len(table) == 3 * 3
    assert list(table["horizon"][::3]) == ["D", "W", "M"]
    assert list(table["estimator"][:3]) == list(estimators)
    d = table[table["horizon"] == "D"].iloc[0]
    assert d["n_obs"] == 1500 and d["n_assets"] == 9 and d["q"] == pytest.approx(9 / 1500)
    m = table[table["horizon"] == "M"].iloc[0]
    assert m["n_obs"] == 1500 // 21
    assert (table["n_bootstrap"] == 20).all()
    assert table["is_psd"].dtype == bool and table["n_obs"].dtype.kind == "i"


def test_table_values_are_finite_and_internally_consistent(
    horizons: dict[str, pd.DataFrame],
) -> None:
    table = S.estimator_stability_table(horizons, ("pearson", "spearman", "oas"), n_bootstrap=30)
    numeric = table.drop(columns=["horizon", "estimator", "window_start", "window_end"])
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()
    assert (table["ci_lower"] <= table["boot_mean"]).all()
    assert (table["boot_mean"] <= table["ci_upper"]).all()
    assert ((table["eigvec_overlap"] >= 0) & (table["eigvec_overlap"] <= 1 + 1e-12)).all()
    assert ((table["psd_share_boot"] >= 0) & (table["psd_share_boot"] <= 1)).all()
    assert (table["boot_sd"] > 0).all() and (table["rms_pair_sd"] > 0).all()
    np.testing.assert_allclose(table["boot_bias"], table["boot_mean"] - table["avg_corr"])


def test_pearson_point_estimate_columns_match_a_direct_computation(daily: pd.DataFrame) -> None:
    table = S.estimator_stability_table({"D": daily}, ("pearson",), n_bootstrap=20)
    row = table.iloc[0]
    corr = daily.corr().to_numpy()
    iu = np.triu_indices(corr.shape[0], k=1)
    assert row["avg_corr"] == pytest.approx(float(corr[iu].mean()))
    assert row["cond"] == pytest.approx(float(np.linalg.cond(corr)))
    assert row["min_eigenvalue"] == pytest.approx(float(np.linalg.eigvalsh(corr).min()))
    assert bool(row["is_psd"]) is True
    assert row["window_start"] == str(daily.index[0].date())
    assert row["window_end"] == str(daily.index[-1].date())


def test_pearson_spread_columns_match_an_independent_recomputation(daily: pd.DataFrame) -> None:
    """Recompute every bootstrap column with plain numpy from the same replicate index sets."""
    sample = daily.iloc[:400]
    n_boot, seed = 40, 7
    block = default_mean_block_length(len(sample))
    sets = S.bootstrap_index_sets(len(sample), n_boot, block, seed, "D")
    values = sample.to_numpy()
    iu = np.triu_indices(values.shape[1], k=1)
    point = np.corrcoef(values, rowvar=False)
    boots = [np.corrcoef(values[s], rowvar=False) for s in sets]
    pairs = np.array([m[iu] for m in boots])
    avg = pairs.mean(axis=1)
    pair_sd = pairs.std(axis=0, ddof=1)

    def lead(m: np.ndarray) -> np.ndarray:
        return np.asarray(np.linalg.eigh(m)[1][:, -1])

    overlap = np.mean([float(lead(point) @ lead(m)) ** 2 for m in boots])

    row = S.estimator_stability_table(
        {"D": sample}, ("pearson",), n_bootstrap=n_boot, seed=seed, alpha=0.1
    ).iloc[0]
    assert row["mean_block_length"] == block
    assert row["avg_corr"] == pytest.approx(float(point[iu].mean()), rel=1e-9)
    assert row["boot_mean"] == pytest.approx(float(avg.mean()), rel=1e-9)
    assert row["boot_sd"] == pytest.approx(float(avg.std(ddof=1)), rel=1e-9)
    assert row["ci_lower"] == pytest.approx(float(np.quantile(avg, 0.05)), rel=1e-9)
    assert row["ci_upper"] == pytest.approx(float(np.quantile(avg, 0.95)), rel=1e-9)
    rms = float(np.sqrt(np.mean(pair_sd**2)))
    assert row["rms_pair_sd"] == pytest.approx(rms, rel=1e-9)
    assert row["rel_spread"] == pytest.approx(rms / float(np.sqrt(np.mean(point[iu] ** 2))))
    assert row["eigvec_overlap"] == pytest.approx(float(overlap), rel=1e-9)
    assert row["psd_share_boot"] == 1.0


def test_the_table_is_deterministic_and_changes_with_the_seed(daily: pd.DataFrame) -> None:
    kw = {"estimators": ("pearson", "ewma"), "n_bootstrap": 20}
    a = S.estimator_stability_table({"D": daily}, seed=1, **kw)  # type: ignore[arg-type]
    b = S.estimator_stability_table({"D": daily}, seed=1, **kw)  # type: ignore[arg-type]
    c = S.estimator_stability_table({"D": daily}, seed=2, **kw)  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(a, b)
    assert not a["boot_sd"].equals(c["boot_sd"])


def test_a_row_does_not_depend_on_the_other_estimators_or_horizons(
    horizons: dict[str, pd.DataFrame],
) -> None:
    """Common random numbers: the replicates depend only on (seed, horizon, T)."""
    full = S.estimator_stability_table(
        horizons, ("pearson", "ledoit_wolf", "detoned"), n_bootstrap=20, seed=5
    )
    alone = S.estimator_stability_table({"W": horizons["W"]}, ("detoned",), n_bootstrap=20, seed=5)
    pick = full[(full["horizon"] == "W") & (full["estimator"] == "detoned")].reset_index(drop=True)
    pd.testing.assert_frame_equal(pick, alone)


def test_independent_noise_matches_the_theoretical_sampling_spread() -> None:
    t, n = 500, 9
    noise = make_noise_returns(t, n, seed=SEED).returns
    row = S.estimator_stability_table({"D": noise}, ("pearson",), n_bootstrap=200, seed=0).iloc[0]
    assert row["rms_pair_sd"] == pytest.approx(1.0 / np.sqrt(t), rel=0.2)
    expected_avg_sd = 1.0 / np.sqrt(t * n * (n - 1) / 2)
    assert 0.5 * expected_avg_sd < row["boot_sd"] < 2.0 * expected_avg_sd
    assert abs(row["avg_corr"]) < 4 * expected_avg_sd


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_shrinkage_has_lower_spread_than_the_sample_matrix_when_t_is_close_to_n(
    seed: int,
) -> None:
    """T = 30 rows, N = 20 assets: shrinkage toward the identity must cut the pairwise spread.

    A smoke property of this regime (q = 0.67), checked on three fixed seeds with a wide margin
    (the ratio is far below 0.95); it is not claimed as a general inequality.
    """
    panel = make_factor_returns(30, (5, 5, 5, 5), seed=seed).returns
    table = S.estimator_stability_table(
        {"Q": panel}, ("pearson", "ledoit_wolf"), n_bootstrap=60, seed=0
    )
    spread = table.set_index("estimator")["rms_pair_sd"]
    cond = table.set_index("estimator")["cond"]
    assert spread["ledoit_wolf"] < 0.95 * spread["pearson"]
    assert cond["ledoit_wolf"] < cond["pearson"]


def test_a_factor_panel_has_a_stable_leading_mode_and_a_clearly_positive_average(
    daily: pd.DataFrame,
) -> None:
    row = S.estimator_stability_table({"D": daily}, ("pearson",), n_bootstrap=40).iloc[0]
    assert row["avg_corr"] == pytest.approx(0.3333, abs=0.08)  # (3 * 0.5 + 6 * 0.25) / 9 pairs
    assert row["eigvec_overlap"] > 0.9


def test_t_not_larger_than_n_still_produces_a_row_and_flags_the_singular_matrix() -> None:
    panel = make_factor_returns(12, (4, 4, 4), seed=3).returns  # T = 12, N = 12
    table = S.estimator_stability_table(
        {"Q": panel}, ("pearson", "ledoit_wolf"), n_bootstrap=20, mean_block_length=2.0
    )
    pearson = table.set_index("estimator").loc["pearson"]
    assert pearson["q"] == 1.0
    assert pearson["cond"] > 1e6 or not np.isfinite(pearson["cond"])
    assert table.set_index("estimator").loc["ledoit_wolf", "cond"] < 1e3


def test_a_detoned_matrix_is_reported_as_singular_not_as_a_noise_condition_number(
    daily: pd.DataFrame,
) -> None:
    """Removing one mode leaves rank N - 1, so cond is inf (not a ~1e15 floating-point value)."""
    table = S.estimator_stability_table(
        {"D": daily}, ("pearson", "detoned"), n_bootstrap=20, seed=0
    ).set_index("estimator")
    assert table.loc["detoned", "cond"] == float("inf")
    assert table.loc["detoned", "cond_boot_median"] == float("inf")
    assert np.isfinite(table.loc["pearson", "cond"]) and table.loc["pearson", "cond"] < 1e3
    assert table.loc["detoned", "min_eigenvalue"] == pytest.approx(0.0, abs=1e-8)


def test_the_condition_helper_switches_to_inf_at_the_rank_tolerance() -> None:
    assert S._condition(0.5, 2.0, 4.0) == 4.0
    assert S._condition(3e-10, 2.0, 7e9) == 7e9  # above 1e-10 of the largest eigenvalue
    assert S._condition(2e-10, 2.0, 1e10) == float("inf")  # exactly at the tolerance: singular
    assert S._condition(1e-12, 2.0, 2e12) == float("inf")
    assert S._condition(0.0, 0.0, 1.0) == float("inf")


def test_incomplete_rows_are_dropped_once_so_every_estimator_shares_t(daily: pd.DataFrame) -> None:
    # 700 missing rows: round(T ** (1/3)) is 9 for T = 800 but 11 for T = 1500, so the reported
    # block length shows which row count the default was computed from.
    frame = daily.copy()
    frame.iloc[:700, 0] = np.nan
    table = S.estimator_stability_table({"D": frame}, ("pearson", "oas", "detoned"), n_bootstrap=20)
    assert (table["n_obs"] == len(frame) - 700).all()
    assert (table["window_start"] == str(frame.index[700].date())).all()
    assert (table["mean_block_length"] == default_mean_block_length(len(frame) - 700)).all()


def test_the_window_argument_selects_rows_and_is_reported(daily: pd.DataFrame) -> None:
    start, end = str(daily.index[200].date()), str(daily.index[699].date())
    table = S.estimator_stability_table(
        {"D": daily}, ("pearson",), start=start, end=end, n_bootstrap=20
    )
    row = table.iloc[0]
    assert row["n_obs"] == 500 and row["window_start"] == start and row["window_end"] == end


def test_a_degenerate_replicate_raises_instead_of_being_dropped() -> None:
    """A constant column cannot be correlated: the error propagates (no silent skipping)."""
    panel = make_factor_returns(100, (3, 3), seed=4).returns.copy()
    panel["S0_0"] = 0.0
    with pytest.raises(CorrelationError, match="zero or undefined variance"):
        S.estimator_stability_table({"D": panel}, ("pearson",), n_bootstrap=20)


def test_the_table_rejects_bad_arguments(daily: pd.DataFrame) -> None:
    ok = {"D": daily}
    with pytest.raises(S.StabilityError, match="empty"):
        S.estimator_stability_table({}, n_bootstrap=20)
    with pytest.raises(S.StabilityError, match="estimators is empty"):
        S.estimator_stability_table(ok, (), n_bootstrap=20)
    with pytest.raises(S.StabilityError, match="unknown estimators"):
        S.estimator_stability_table(ok, ("pearson", "mystery"), n_bootstrap=20)  # type: ignore[arg-type]
    with pytest.raises(S.StabilityError, match="n_bootstrap"):
        S.estimator_stability_table(ok, ("pearson",), n_bootstrap=5)
    with pytest.raises(S.StabilityError, match="alpha"):
        S.estimator_stability_table(ok, ("pearson",), n_bootstrap=20, alpha=0.7)
    with pytest.raises(S.StabilityError, match="unknown horizon"):
        S.estimator_stability_table({"X": daily}, ("pearson",), n_bootstrap=20)  # type: ignore[dict-item]
    with pytest.raises(S.StabilityError, match="complete-case rows"):
        S.estimator_stability_table({"Q": daily.iloc[:5]}, ("pearson",), n_bootstrap=20)
    with pytest.raises(S.StabilityError, match="at least 3 assets"):
        S.estimator_stability_table({"D": daily.iloc[:, :2]}, ("pearson",), n_bootstrap=20)


def test_ewma_lambda_and_n_market_are_passed_through_to_the_estimators(
    daily: pd.DataFrame,
) -> None:
    sample = daily.iloc[:300]
    iu = np.triu_indices(sample.shape[1], k=1)
    table = S.estimator_stability_table(
        {"D": sample},
        ("ewma", "detoned"),
        n_bootstrap=20,
        ewma_lambda=0.5,
        n_market=2,
    ).set_index("estimator")
    assert table.loc["ewma", "avg_corr"] == pytest.approx(
        float(ewma_corr(sample, 0.5).to_numpy()[iu].mean())
    )
    expected = detone_corr(estimate(sample, "pearson"), n_market=2).to_numpy()[iu].mean()
    assert table.loc["detoned", "avg_corr"] == pytest.approx(float(expected))
    default = S.estimator_stability_table({"D": sample}, ("ewma",), n_bootstrap=20)
    assert default.loc[0, "avg_corr"] != table.loc["ewma", "avg_corr"]


def test_the_psd_columns_react_to_a_matrix_that_is_not_psd(
    daily: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equicorrelation -0.9 over three assets has eigenvalue 1 + 2 * (-0.9) < 0."""

    def not_psd(sample: pd.DataFrame, name: str, **_: object) -> pd.DataFrame:
        k = sample.shape[1]
        values = np.full((k, k), -0.9)
        np.fill_diagonal(values, 1.0)
        return pd.DataFrame(values, index=sample.columns, columns=sample.columns)

    monkeypatch.setattr(S, "estimator_matrix", not_psd)
    row = S.estimator_stability_table(
        {"D": daily.iloc[:200, :3]}, ("pearson",), n_bootstrap=20
    ).iloc[0]
    assert bool(row["is_psd"]) is False and row["psd_share_boot"] == 0.0
    assert row["min_eigenvalue"] == pytest.approx(-0.8)
    assert row["avg_corr"] == pytest.approx(-0.9) and row["rms_pair_sd"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- summary lines


def test_summary_lines_have_one_line_per_row_and_carry_the_caveats(
    horizons: dict[str, pd.DataFrame],
) -> None:
    table = S.estimator_stability_table(
        {"D": horizons["D"]}, ("pearson", "detoned"), n_bootstrap=20
    )
    lines = S.stability_summary_lines(table)
    assert len(lines) == len(table) + 1
    assert "relative scale" in lines[1] and "relative scale" not in lines[0]
    assert "not higher accuracy" in lines[-1] and "H5" in lines[-1]
    with pytest.raises(S.StabilityError, match="missing columns"):
        S.stability_summary_lines(table.drop(columns=["boot_sd"]))
