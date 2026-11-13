"""Unit tests for ``etfcorr.features`` (step 5.3), on synthetic panels with known structure.

Look-ahead is tested separately in ``tests/leakage/test_features_no_lookahead.py``; here every
function is checked against a direct, independent computation.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from etfcorr import features as F
from etfcorr.correlation.estimators import ewma_corr
from etfcorr.synthetic import SyntheticPanel, make_factor_returns


@pytest.fixture(scope="module")
def panel() -> SyntheticPanel:
    return make_factor_returns(700, (3, 3, 3), seed=11)


@pytest.fixture(scope="module")
def sectors(panel: SyntheticPanel) -> dict[str, str]:
    return dict(zip(panel.returns.columns, panel.sectors, strict=True))


# ---- transforms ------------------------------------------------------------------------


def test_fisher_z_round_trips_and_clips() -> None:
    r = np.array([-0.9, -0.2, 0.0, 0.5, 0.95])
    assert np.allclose(F.inverse_fisher_z(F.fisher_z(r)), r)
    assert np.isfinite(F.fisher_z([1.0, -1.0])).all()
    assert F.fisher_z([1.0])[0] == pytest.approx(np.arctanh(F.Z_CLIP))
    assert np.isnan(F.fisher_z([np.nan])[0])


def test_log_variance_floors_zero() -> None:
    out = F.log_variance(pd.Series([0.0, 1e-4]))
    assert out.iloc[0] == pytest.approx(np.log(F.VAR_FLOOR))
    assert out.iloc[1] == pytest.approx(np.log(1e-4))


def test_cross_sectional_rank_range_ties_and_missing() -> None:
    wide = pd.DataFrame(
        [[3.0, 1.0, 2.0, 5.0], [1.0, 1.0, 4.0, np.nan], [np.nan, np.nan, 2.0, np.nan]],
        columns=list("abcd"),
    )
    out = F.cross_sectional_rank(wide)
    assert out.iloc[0].tolist() == pytest.approx([1 / 6, -0.5, -1 / 6, 0.5])
    assert out.iloc[1, :3].tolist() == pytest.approx([-0.25, -0.25, 0.5])  # tie averaged
    assert np.isnan(out.iloc[1, 3])
    assert out.iloc[2].isna().all()  # a single finite value has no rank


# ---- trailing correlation --------------------------------------------------------------


def test_trailing_correlations_match_numpy_and_warm_up(panel: SyntheticPanel) -> None:
    r = panel.returns.iloc[:120]
    out = F.trailing_correlations(r, 21)
    assert out.shape == (120, 9, 9)
    assert np.isnan(out[:20]).all() and np.isfinite(out[20:]).all()
    for t in (20, 57, 119):
        assert np.allclose(out[t], np.corrcoef(r.to_numpy()[t - 20 : t + 1], rowvar=False))
    assert np.allclose(out[20:], np.swapaxes(out[20:], 1, 2))
    assert np.allclose(np.diagonal(out[20:], axis1=1, axis2=2), 1.0)


def test_trailing_correlations_are_nan_exactly_while_the_window_holds_a_gap(
    panel: SyntheticPanel,
) -> None:
    r = panel.returns.iloc[:80].copy()
    r.iloc[40, 2] = np.nan
    out = F.trailing_correlations(r, 10)
    missing = np.isnan(out).all(axis=(1, 2))
    assert missing.tolist() == [t < 9 or 40 <= t <= 49 for t in range(80)]


def test_spearman_kind_matches_scipy(panel: SyntheticPanel) -> None:
    r = panel.returns.iloc[:60]
    out = F.trailing_correlations(r, 30, kind="spearman")
    expected = spearmanr(r.to_numpy()[30:60]).statistic
    assert np.allclose(out[59], expected)


def test_constant_column_gives_nan_in_its_row_and_column(panel: SyntheticPanel) -> None:
    r = panel.returns.iloc[:30].copy()
    r.iloc[:, 0] = 0.0
    out = F.trailing_correlations(r, 10)[29]
    assert np.isnan(out[0, 1:]).all() and np.isnan(out[1:, 0]).all()
    assert np.isfinite(out[1:, 1:]).all()


def test_trailing_correlations_reject_bad_input(panel: SyntheticPanel) -> None:
    with pytest.raises(F.FeatureError):
        F.trailing_correlations(panel.returns, 1)
    with pytest.raises(F.FeatureError):
        F.trailing_correlations(panel.returns.iloc[:, :1], 5)
    with pytest.raises(F.FeatureError):
        F.trailing_correlations(panel.returns, 5, kind="kendall")  # type: ignore[arg-type]
    with pytest.raises(F.FeatureError):
        F.trailing_correlations(panel.returns.iloc[::-1], 5)


def test_ewma_correlations_equal_the_estimator_at_every_checked_date(
    panel: SyntheticPanel,
) -> None:
    r = panel.returns.iloc[:150]
    out = F.ewma_correlations(r, 0.94)
    assert np.isnan(out[0]).all()
    for t in (1, 2, 40, 149):
        assert np.allclose(out[t], ewma_corr(r.iloc[: t + 1], 0.94).to_numpy(), atol=1e-12)


def test_ewma_correlations_reject_gaps_and_bad_lambda(panel: SyntheticPanel) -> None:
    r = panel.returns.iloc[:20].copy()
    with pytest.raises(F.FeatureError):
        F.ewma_correlations(r, 1.0)
    r.iloc[5, 0] = np.nan
    with pytest.raises(F.FeatureError):
        F.ewma_correlations(r)


def test_sector_masks_count_the_right_pairs(sectors: dict[str, str]) -> None:
    columns = list(sectors)
    assert int(F.sector_pair_mask(columns, sectors, "cross").sum()) == 27
    assert int(F.sector_pair_mask(columns, sectors, "within").sum()) == 9
    assert int(F.sector_pair_mask(columns, sectors, "all").sum()) == 36
    with pytest.raises(F.FeatureError):
        F.sector_pair_mask([*columns, "ZZZ"], sectors)


def test_trailing_average_correlation_recovers_the_planted_levels(
    panel: SyntheticPanel, sectors: dict[str, str]
) -> None:
    """Population values are 0.25 across and 0.5 within sectors; 700 days give about +-0.04."""
    columns = list(sectors)
    cross = F.trailing_average_correlation(
        panel.returns, 700, F.sector_pair_mask(columns, sectors, "cross")
    )
    within = F.trailing_average_correlation(
        panel.returns, 700, F.sector_pair_mask(columns, sectors, "within")
    )
    assert cross.iloc[-1] == pytest.approx(0.25, abs=0.06)
    assert within.iloc[-1] == pytest.approx(0.50, abs=0.06)
    assert cross.iloc[:-1].isna().all()


def test_masked_average_checks_the_mask() -> None:
    corr = np.tile(np.eye(3), (2, 1, 1))
    with pytest.raises(F.FeatureError):
        F.masked_average(corr, np.zeros((3, 3), dtype=bool))
    with pytest.raises(F.FeatureError):
        F.masked_average(corr, np.ones((2, 2), dtype=bool))


def test_market_mode_share_known_values() -> None:
    stack = np.stack([np.eye(4), np.ones((4, 4)), np.full((4, 4), np.nan)])
    out = F.market_mode_share(stack)
    assert out[0] == pytest.approx(0.25) and out[1] == pytest.approx(1.0) and np.isnan(out[2])


# ---- volatility and returns ------------------------------------------------------------


def test_trailing_variance_return_and_ewma_match_direct_formulas(panel: SyntheticPanel) -> None:
    r = panel.returns.iloc[:50, :2]
    x = r.to_numpy()
    var = F.trailing_variance(r, 5)
    assert var.iloc[:4].isna().all().all()
    assert np.allclose(var.iloc[30].to_numpy(), (x[26:31] ** 2).mean(axis=0))
    ret = F.trailing_return(r, 5)
    assert np.allclose(ret.iloc[30].to_numpy(), x[26:31].sum(axis=0))
    v = x[0] ** 2
    for row in x[1:]:
        v = 0.94 * v + 0.06 * row**2
    assert np.allclose(F.ewma_variance(r, 0.94).iloc[-1].to_numpy(), v)


# ---- feature tables --------------------------------------------------------------------


def test_build_features_columns_and_warm_up(panel: SyntheticPanel, sectors: dict[str, str]) -> None:
    out = F.build_features(panel.returns, sectors)
    expected = {
        "z_cross_w21",
        "z_cross_w63",
        "z_cross_w126",
        "z_cross_w252",
        "z_within_w63",
        "mode_share_w63",
        "z_cross_ewma",
        "z_cross_spearman_w63",
        "log_rv_xs_w5",
        "log_rv_xs_w21",
        "log_rv_xs_w63",
        "log_dispersion_w21",
    }
    assert set(out.columns) == expected
    assert out.index.equals(panel.returns.index)
    complete = out.notna().all(axis=1).to_numpy()
    assert not complete[:251].any() and complete[251:].all()
    assert np.isfinite(out.iloc[251:].to_numpy()).all()


def test_build_features_persistence_column_is_the_trailing_statistic(
    panel: SyntheticPanel, sectors: dict[str, str]
) -> None:
    out = F.build_features(panel.returns, sectors, windows=(21,))
    mask = F.sector_pair_mask(list(sectors), sectors)
    direct = F.trailing_average_correlation(panel.returns, 21, mask)
    assert np.allclose(out["z_cross_w21"].dropna(), np.arctanh(direct.dropna()))


def test_build_features_market_and_vix_columns(
    panel: SyntheticPanel, sectors: dict[str, str]
) -> None:
    r = panel.returns
    market = r.mean(axis=1)
    vix = pd.Series(np.exp(np.linspace(2.5, 3.5, len(r))), index=r.index)
    out = F.build_features(r, sectors, market=market, vix=vix, windows=(21, 63))
    assert {"log_rv_mkt_w21", "ret_mkt_w21", "log_vix", "dlog_vix_w5", "dlog_vix_w21"} <= set(out)
    assert out["ret_mkt_w21"].iloc[100] == pytest.approx(market.iloc[80:101].sum())
    assert out["dlog_vix_w5"].iloc[100] == pytest.approx(np.log(vix.iloc[100] / vix.iloc[95]))
    with pytest.raises(F.FeatureError):
        F.build_features(r, sectors, vix=-vix)
    with pytest.raises(F.FeatureError):
        F.build_features(r, sectors, market=market.iloc[1:])


def test_stack_units_orders_rows_by_date_then_unit() -> None:
    index = pd.bdate_range("2020-01-01", periods=2, name="date")
    a = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=index, columns=["x", "y"])
    out = F.stack_units({"a": a, "b": a * 10})
    assert out.index.names == ["date", F.UNIT]
    assert out["a"].tolist() == [1.0, 2.0, 3.0, 4.0] and out["b"].tolist() == [10, 20, 30, 40]
    assert out.loc[(index[1], "x"), "a"] == 3.0
    with pytest.raises(F.FeatureError):
        F.stack_units({"a": a, "b": a.iloc[:1]})
    with pytest.raises(F.FeatureError):
        F.stack_units({})


def test_build_asset_features_definitions(panel: SyntheticPanel) -> None:
    r = panel.returns.iloc[:400, :4]
    common = pd.DataFrame({"state": np.arange(400.0)}, index=r.index)
    out = F.build_asset_features(r, common=common)
    assert list(out.columns) == [
        "log_rv_w5",
        "log_rv_w21",
        "log_rv_w63",
        "log_ewma_var",
        "ret_w5",
        "ret_w21",
        "mom_252_21",
        "state",
    ]
    assert len(out) == 400 * 4
    name, t = r.columns[2], 350
    row = out.loc[(r.index[t], name)]
    x = r[name].to_numpy()
    assert row["mom_252_21"] == pytest.approx(x[t - 251 : t - 20].sum())  # days t-251 .. t-21
    assert row["log_rv_w21"] == pytest.approx(np.log((x[t - 20 : t + 1] ** 2).mean()))
    assert row["state"] == 350.0
    ranked = F.build_asset_features(r, rank=True).dropna()
    assert ranked.to_numpy().min() == -0.5 and ranked.to_numpy().max() == 0.5


def test_pair_tables_round_trip(panel: SyntheticPanel) -> None:
    r = panel.returns.iloc[:300, :4]
    columns = list(r.columns)
    assert F.pair_labels(columns)[:3] == [f"{columns[0]}|{columns[k]}" for k in (1, 2, 3)]
    corr = F.trailing_correlations(r, 21)
    wide = F.pairs_wide(corr, r.index, columns)
    assert wide.shape == (300, 6)
    back = F.pairs_to_matrices(wide, 4)
    assert np.allclose(back[20:], corr[20:]) and np.isnan(back[:20]).all()
    with pytest.raises(F.FeatureError):
        F.pairs_to_matrices(wide.iloc[:, :5], 4)


def test_build_pair_features_columns_and_constant_correlation(panel: SyntheticPanel) -> None:
    r = panel.returns.iloc[:300, :4]
    out = F.build_pair_features(r, windows=(21, 63))
    assert list(out.columns) == ["z_w21", "z_w63", "z_ewma", "z_const"]
    day = out.xs(r.index[200], level="date")
    corr63 = np.corrcoef(r.to_numpy()[138:201], rowvar=False)
    assert day["z_w63"].iloc[0] == pytest.approx(np.arctanh(corr63[0, 1]))
    assert day["z_const"].nunique() == 1
    assert day["z_const"].iloc[0] == pytest.approx(np.arctanh(corr63[np.triu_indices(4, 1)].mean()))
