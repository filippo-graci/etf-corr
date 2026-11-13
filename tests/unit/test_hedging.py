"""Tests for ``etfcorr.evaluation.hedging`` (step 8.2) on panels with planted betas."""

import numpy as np
import pandas as pd
import pytest

from etfcorr.config import DateWindow
from etfcorr.evaluation import hedging as H
from etfcorr.leakage import LeakageError, assert_future_invariant, assert_no_lookahead

RNG = np.random.default_rng(0)
N = 600
INDEX = pd.bdate_range("2020-01-01", periods=N, name="date")
MARKET = pd.Series(RNG.standard_normal(N) * 0.01, index=INDEX, name="SPY")
BETAS = np.array([0.5, 1.0, 1.5])
ASSETS = pd.DataFrame(
    np.outer(MARKET, BETAS) + RNG.standard_normal((N, 3)) * 0.005, index=INDEX, columns=list("abc")
)
WINDOW = DateWindow.model_validate([str(INDEX[300].date()), str(INDEX[-1].date())])


def test_rolling_beta_is_the_ols_slope_of_the_trailing_window() -> None:
    beta = H.rolling_beta(ASSETS, MARKET, 60)
    slope = np.polyfit(MARKET.iloc[240:300], ASSETS["b"].iloc[240:300], 1)[0]
    assert beta["b"].iloc[299] == pytest.approx(slope)
    assert beta.iloc[:59].isna().all().all()
    np.testing.assert_allclose(beta.iloc[-1], BETAS, atol=0.2)


def test_ewma_and_static_betas() -> None:
    np.testing.assert_allclose(H.ewma_beta(ASSETS, MARKET, 0.97).iloc[-1], BETAS, atol=0.25)
    static = H.static_beta(ASSETS, MARKET, INDEX[300])
    slope = np.polyfit(MARKET.iloc[:300], ASSETS["c"].iloc[:300], 1)[0]
    assert static["c"].nunique() == 1 and static["c"].iloc[0] == pytest.approx(slope)
    with pytest.raises(H.HedgingError):
        H.static_beta(ASSETS, MARKET, INDEX[10])


def test_the_hedge_applies_the_ratio_known_the_day_before() -> None:
    beta = H.rolling_beta(ASSETS, MARKET, 60)
    hedged = H.hedged_returns(ASSETS, MARKET, beta)
    t = 400
    expected = ASSETS["a"].iloc[t] - beta["a"].iloc[t - 1] * MARKET.iloc[t]
    assert hedged["a"].iloc[t] == pytest.approx(expected)


def frame(d: pd.DataFrame) -> pd.DataFrame:
    return H.hedged_returns(d[list("abc")], d["SPY"], H.rolling_beta(d[list("abc")], d["SPY"], 60))


def test_hedged_returns_have_no_lookahead_and_a_same_day_beta_is_caught() -> None:
    data = ASSETS.assign(SPY=MARKET)
    cuts = (150, 301, 450, N - 1)
    assert_no_lookahead(frame, data, cuts)
    assert_future_invariant(frame, data, cuts, mode="noise")

    def centred(d: pd.DataFrame) -> pd.DataFrame:  # the leak: a beta from a centred window
        beta = H.rolling_beta(d[list("abc")], d["SPY"], 60).shift(-30)
        return H.hedged_returns(d[list("abc")], d["SPY"], beta)

    with pytest.raises(LeakageError):
        assert_no_lookahead(centred, data, cuts)


def test_effectiveness_and_regime_masks() -> None:
    raw = ASSETS.to_numpy()
    everything = np.ones(N, dtype=bool)
    np.testing.assert_allclose(H.effectiveness(raw, raw, everything), 0.0)
    perfect = raw - np.outer(MARKET, BETAS)
    he = H.effectiveness(raw, perfect, everything)
    assert (np.diff(he) > 0).all() and he[2] > 0.85  # a larger beta leaves less noise share
    assert np.isnan(H.effectiveness(raw, raw, np.zeros(N, dtype=bool))).all()
    vix = pd.Series(np.arange(1.0, 101.0))
    stress, calm = H.regime_masks(vix, 0.9, 0.5)
    assert stress.sum() == 10 and calm.sum() == 50 and not (stress & calm).any()
    with pytest.raises(H.HedgingError):
        H.regime_masks(vix.iloc[:5], 0.9, 0.5)


def planted(noise_in_stress: float) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Stress = the days of a high-VIX stretch; idiosyncratic noise scaled inside it."""
    rng = np.random.default_rng(1)
    vix = np.where((np.arange(N) // 30) % 10 == 3, 40.0, 15.0) + rng.random(N)
    stress, calm = H.regime_masks(pd.Series(vix), 0.9, 0.5)
    market = rng.standard_normal(N) * 0.01
    noise = rng.standard_normal((N, 3)) * 0.005 * np.where(stress, noise_in_stress, 1.0)[:, None]
    raw = pd.DataFrame(np.outer(market, BETAS) + noise, index=INDEX, columns=list("abc"))
    hedged = raw - np.outer(market, BETAS)
    return raw, hedged, stress, calm


def test_h6_rule_detects_a_planted_drop_in_stress_and_not_its_opposite() -> None:
    raw, hedged, stress, calm = planted(4.0)
    summary, assets, verdict = H.h6_table(
        raw, {"rolling": hedged, "other": raw * 1.0}, stress, calm, block_days=5, n_bootstrap=200
    )
    assert verdict["h6_supported"] and verdict["statistic"] < -0.2  # type: ignore[operator]
    assert verdict["p_one_sided"] == pytest.approx(1 / 201)
    assert verdict["n_stress"] == int(stress.sum()) and list(summary["method"]) == [
        "rolling",
        "other",
    ]
    assert summary.set_index("method").loc["other", "he_all"] == pytest.approx(0.0)
    assert assets.shape == (3, 4) and (assets["rolling_stress"] < assets["rolling_calm"]).all()
    raw, hedged, stress, calm = planted(0.25)
    _, _, opposite = H.h6_table(raw, {"rolling": hedged}, stress, calm, n_bootstrap=200)
    assert not opposite["h6_supported"] and opposite["statistic"] > 0  # type: ignore[operator]
    with pytest.raises(H.HedgingError, match="primary"):
        H.h6_table(raw, {"ewma": hedged}, stress, calm, n_bootstrap=5)
    with pytest.raises(H.HedgingError, match="complete"):
        H.h6_table(raw, {"rolling": hedged.iloc[:-1]}, stress, calm, n_bootstrap=5)


def test_lasso_hedge_selects_the_instrument_that_matters() -> None:
    rng = np.random.default_rng(2)
    instruments = pd.DataFrame(
        rng.standard_normal((N, 3)) * 0.01, index=INDEX, columns=["SPY", "TLT", "GLD"]
    )
    assets = pd.DataFrame(
        {"x": 0.8 * instruments["TLT"] + rng.standard_normal(N) * 0.002}, index=INDEX
    )
    hedged, selected = H.lasso_hedged_returns(assets, instruments, 120, WINDOW)
    inside = hedged.loc[str(WINDOW.start) :]
    assert hedged.loc[: INDEX[299]].isna().all().all() and inside.notna().all().all()
    assert (selected["TLT"] == 1.0).all() and selected[["SPY", "GLD"]].mean().max() < 0.5
    assert inside["x"].var() < 0.2 * assets.loc[inside.index, "x"].var()
    with pytest.raises(H.HedgingError, match="fewer than"):
        H.lasso_hedged_returns(assets, instruments, 500, WINDOW)
    # the refit of a month must not read that month's first session
    refit = int(INDEX.searchsorted(selected.index[1]))
    shocked = assets.copy()
    shocked.iloc[refit] = 0.5
    again, _ = H.lasso_hedged_returns(shocked, instruments, 120, WINDOW)
    month = slice(refit + 1, int(INDEX.searchsorted(selected.index[2])))
    pd.testing.assert_frame_equal(again.iloc[month], hedged.iloc[month])


# ---- exploratory follow-up E1 (docs/FOLLOW_UPS.md item 1)

STATIC_HEDGED = H.hedged_returns(ASSETS, MARKET, H.static_beta(ASSETS, MARKET, INDEX[300]))
BIG = (MARKET.abs() > MARKET.abs().quantile(0.8)).to_numpy()
SMALL = (MARKET.abs() <= MARKET.abs().median()).to_numpy()


def test_selecting_large_market_days_inflates_effectiveness_and_the_adjustment_removes_it() -> None:
    """Constant betas: the true hedge is equally good every day, the measured one is not."""
    risk = H.regime_risk(
        ASSETS.iloc[1:], STATIC_HEDGED.iloc[1:], MARKET.iloc[1:], BIG[1:], SMALL[1:]
    )
    assert (risk["he_stress"] > risk["he_calm"] + 0.2).all()
    assert (risk["r2_stress"] > risk["r2_calm"] + 0.2).all()
    np.testing.assert_allclose(risk["r2_adj_stress"], risk["r2_calm"], atol=0.08)
    np.testing.assert_allclose(risk["he_stress"], risk["r2_stress"], atol=0.02)  # the beta is right


def test_regime_risk_columns_against_direct_computation() -> None:
    risk = H.regime_risk(
        ASSETS.iloc[1:], STATIC_HEDGED.iloc[1:], MARKET.iloc[1:], BIG[1:], SMALL[1:]
    )
    x = STATIC_HEDGED.iloc[1:].loc[BIG[1:], "b"].to_numpy()
    assert risk.loc["b", "vol_hedged_stress"] == pytest.approx(x.std(ddof=1) * np.sqrt(252))
    cut = np.quantile(x, 0.05)
    assert risk.loc["b", "var_hedged_stress"] == pytest.approx(-cut)
    assert risk.loc["b", "es_hedged_stress"] == pytest.approx(-x[x <= cut].mean())
    assert (risk["es_unhedged_calm"] >= risk["var_unhedged_calm"]).all()
    assert (risk["vol_hedged_stress"] < risk["vol_unhedged_stress"]).all()
    rho = np.corrcoef(ASSETS.iloc[1:].loc[SMALL[1:], "c"], MARKET.iloc[1:][SMALL[1:]])[0, 1]
    assert risk.loc["c", "r2_calm"] == pytest.approx(rho**2)
    with pytest.raises(H.HedgingError):
        H.regime_risk(ASSETS, STATIC_HEDGED.fillna(0.0), MARKET, np.zeros(N, dtype=bool), SMALL)


def test_regime_risk_table_agrees_with_the_h6_statistic_and_brackets_its_values() -> None:
    raw, hedged = ASSETS.iloc[1:], STATIC_HEDGED.iloc[1:]
    table = H.regime_risk_table(raw, hedged, MARKET.iloc[1:], BIG[1:], SMALL[1:], n_bootstrap=60)
    summary, _, _ = H.h6_table(raw, {"rolling": hedged}, BIG[1:], SMALL[1:], n_bootstrap=20)
    assert table.loc["he_diff", "value"] == pytest.approx(summary["diff"].iloc[0])
    assert (table["ci_lower"] <= table["ci_upper"]).all() and table["n_draws"].eq(60).all()
    inside = (table["ci_lower"] <= table["value"]) & (table["value"] <= table["ci_upper"])
    assert inside.mean() > 0.9
    assert table.loc["he_diff", "ci_lower"] > 0  # the planted inflation is detected
    assert abs(table.loc["r2_adj_diff", "value"]) < 0.08 < table.loc["r2_diff", "value"]
    again = H.regime_risk_table(raw, hedged, MARKET.iloc[1:], BIG[1:], SMALL[1:], n_bootstrap=60)
    pd.testing.assert_frame_equal(table, again)


def test_adjusted_r2_is_the_forbes_rigobon_formula_and_the_gaps_are_differences() -> None:
    from etfcorr.correlation.regime_correlation import forbes_rigobon_adjust

    raw, hedged, market = ASSETS.iloc[1:], STATIC_HEDGED.iloc[1:], MARKET.iloc[1:]
    risk = H.regime_risk(raw, hedged, market, BIG[1:], SMALL[1:])
    rho = np.corrcoef(raw.loc[BIG[1:], "a"], market[BIG[1:]])[0, 1]
    delta = market[BIG[1:]].var() / market[SMALL[1:]].var() - 1.0
    assert delta > 3 and risk.loc["a", "r2_adj_stress"] == pytest.approx(
        float(forbes_rigobon_adjust(rho, delta)) ** 2
    )
    table = H.regime_risk_table(raw, hedged, market, BIG[1:], SMALL[1:], n_bootstrap=20)["value"]
    for gap, high, low in (
        ("he_diff", "he_stress", "he_calm"),
        ("r2_diff", "r2_stress", "r2_calm"),
        ("r2_adj_diff", "r2_adj_stress", "r2_calm"),
        ("vol_hedged_diff", "vol_hedged_stress", "vol_hedged_calm"),
    ):
        assert table[gap] == pytest.approx(table[high] - table[low])


def test_draws_without_enough_regime_days_are_dropped_not_raised() -> None:
    rare = np.zeros(N - 1, dtype=bool)
    rare[[50, 200, 400]] = True
    table = H.regime_risk_table(
        ASSETS.iloc[1:], STATIC_HEDGED.iloc[1:], MARKET.iloc[1:], rare, SMALL[1:], n_bootstrap=80
    )
    assert 0 < table["n_draws"].iloc[0] < 80
