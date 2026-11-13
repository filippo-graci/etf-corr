"""Tests for ``etfcorr.evaluation.portfolio`` (step 8.1): weights, the walk-forward, the H5 rule."""

import numpy as np
import pandas as pd
import pytest

from etfcorr.config import DateWindow, Economics
from etfcorr.evaluation import portfolio as P
from etfcorr.synthetic import make_factor_returns

PANEL = make_factor_returns(700, (3, 3), seed=1, start="2019-01-01")
DAILY = PANEL.returns
WINDOW = DateWindow.model_validate(["2021-01-01", "2021-06-30"])
DESIGN = Economics.model_validate(
    {
        "declared_on": "2026-10-11",
        "estimation": {"D": 120, "W": 60},
        "estimators": ["sample", "ledoit_wolf", "rmt", "ewma"],
        "h5_estimators": ["ledoit_wolf", "rmt"],
        "hedge_instrument": "SPY",
        "hedge_window_days": 60,
        "stress_quantile": 0.9,
        "calm_quantile": 0.5,
        "block_days": 5,
    }
)
PANELS = {"D": DAILY, "W": DAILY.resample("W-FRI").sum()}


def test_minimum_variance_matches_the_closed_forms() -> None:
    var = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(
        P.min_variance_weights(np.diag(var)), (1 / var) / (1 / var).sum(), atol=1e-6
    )
    cov = np.array([[1.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 1.5]])
    inv = np.linalg.solve(cov, np.ones(3))
    np.testing.assert_allclose(P.min_variance_weights(cov), inv / inv.sum(), atol=1e-6)
    np.testing.assert_allclose(P.min_variance_weights(cov * 1e-6), inv / inv.sum(), atol=1e-6)


def test_minimum_variance_respects_the_long_only_constraint() -> None:
    cov = np.array([[1.0, 1.3, 0.0], [1.3, 2.0, 0.0], [0.0, 0.0, 1.0]])  # unconstrained: w2 < 0
    assert (np.linalg.solve(cov, np.ones(3)) < 0).any()
    w = P.min_variance_weights(cov)
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0) and w[1] == pytest.approx(0.0, abs=1e-6)
    rng = np.random.default_rng(0)
    for trial in rng.dirichlet(np.ones(3), size=200):
        assert w @ cov @ w <= trial @ cov @ trial + 1e-9


def test_hrp_is_inverse_variance_for_uncorrelated_assets_and_splits_two_blocks() -> None:
    var = np.array([1.0, 2.0, 4.0, 8.0])
    np.testing.assert_allclose(P.hrp_weights(np.diag(var)), (1 / var) / (1 / var).sum())
    block = np.array([[1.0, 0.9], [0.9, 1.0]])
    cov = np.block([[block, np.zeros((2, 2))], [np.zeros((2, 2)), 4.0 * block]])
    w = P.hrp_weights(cov)  # cluster variances 0.95 and 3.8: the first block gets 80%
    np.testing.assert_allclose(w, [0.4, 0.4, 0.1, 0.1])
    with pytest.raises(P.PortfolioError):
        P.weights(cov, "max_sharpe")


def test_hrp_agrees_with_skfolio_when_it_imports() -> None:
    skfolio = pytest.importorskip("skfolio", exc_type=ImportError)
    cluster = pytest.importorskip("skfolio.cluster", exc_type=ImportError)
    optimization = pytest.importorskip("skfolio.optimization", exc_type=ImportError)
    # skfolio orders the leaves optimally; López de Prado's original order gives other weights
    cov = np.cov(DAILY.to_numpy(), rowvar=False)
    model = optimization.HierarchicalRiskParity(
        risk_measure=skfolio.RiskMeasure.VARIANCE,
        hierarchical_clustering_estimator=cluster.HierarchicalClustering(
            linkage_method=cluster.LinkageMethod.SINGLE
        ),
    ).fit(DAILY)
    np.testing.assert_allclose(P.hrp_weights(cov, optimal_ordering=True), model.weights_, atol=1e-8)
    assert np.abs(P.hrp_weights(cov) - model.weights_).max() > 0.02


def test_covariance_estimators() -> None:
    window = DAILY.iloc[:120]
    x = window.to_numpy()
    sample = P.covariance(window, "sample")
    np.testing.assert_allclose(sample, np.cov(x, rowvar=False))
    lw, rmt = P.covariance(window, "ledoit_wolf"), P.covariance(window, "rmt")
    assert not np.allclose(lw, sample) and np.linalg.eigvalsh(lw).min() > 0
    np.testing.assert_allclose(np.diag(rmt), np.diag(sample))  # volatilities are kept
    assert not np.allclose(rmt, sample) and np.linalg.eigvalsh(rmt).min() > 0
    lwc = P.covariance(window, "ledoit_wolf_corr")
    np.testing.assert_allclose(np.diag(lwc), np.diag(sample))  # variances are not shrunk
    assert abs(lwc[0, 1]) < abs(sample[0, 1]) and not np.allclose(np.diag(lw), np.diag(sample))
    ewma = P.covariance(window, "ewma", lam=0.9)
    w = 0.1 * 0.9 ** np.arange(118, -1, -1)
    expected = 0.9**119 * np.outer(x[0], x[0]) + np.einsum("t,ti,tj->ij", w, x[1:], x[1:])
    np.testing.assert_allclose(ewma, expected)
    corr = np.full((6, 6), 0.3) + 0.7 * np.eye(6)
    dcc = P.covariance(window, "dcc", lam=0.9, dcc_corr=corr)
    np.testing.assert_allclose(np.diag(dcc), np.diag(ewma))
    np.testing.assert_allclose(dcc[0, 1] / np.sqrt(dcc[0, 0] * dcc[1, 1]), 0.3)
    with pytest.raises(P.PortfolioError):
        P.covariance(window, "dcc")
    with pytest.raises(P.PortfolioError):
        P.covariance(window.iloc[:2], "sample")


def test_rebalance_dates_are_the_first_sessions_of_each_month() -> None:
    dates = P.rebalance_dates(pd.DatetimeIndex(DAILY.index), WINDOW)
    assert list(dates.strftime("%Y-%m-%d")) == [
        "2021-01-01",
        "2021-02-01",
        "2021-03-01",
        "2021-04-01",
        "2021-05-03",
        "2021-06-01",
    ]


def test_hold_drifts_weights_and_measures_turnover() -> None:
    index = pd.bdate_range("2021-01-01", periods=4)
    daily = pd.DataFrame({"a": np.log([1.1, 1.0, 1.0, 1.0]), "b": 0.0}, index=index)
    targets = pd.DataFrame({"a": [0.5, 0.5], "b": [0.5, 0.5]}, index=index[[0, 2]])
    run = P.hold(daily, targets, index[-1])
    np.testing.assert_allclose(run.returns, [0.05, 0.0, 0.0, 0.0])
    drifted = 0.55 / 1.05  # weight of a after its 10% gain
    np.testing.assert_allclose(run.turnover, [1.0, 2 * (drifted - 0.5)])


def test_weights_use_only_rows_before_the_rebalance() -> None:
    """Changing the returns on and after a rebalance date leaves that rebalance's weights."""
    date = P.rebalance_dates(pd.DatetimeIndex(DAILY.index), WINDOW)[2]
    changed = DAILY.copy()
    changed.loc[date:] = changed.loc[date:].to_numpy()[::-1] * 3.0
    for source in ("D", "W"):
        panels = {"D": DAILY, "W": DAILY.resample("W-FRI").sum()}
        other = {"D": changed, "W": changed.resample("W-FRI").sum()}
        a = P.run_cell(DAILY, panels[source], 60, "ledoit_wolf", "minvar", WINDOW)
        b = P.run_cell(changed, other[source], 60, "ledoit_wolf", "minvar", WINDOW)
        np.testing.assert_allclose(a.weights.loc[:date], b.weights.loc[:date])
        assert not np.allclose(a.weights.iloc[-1], b.weights.iloc[-1])
    with pytest.raises(P.PortfolioError, match="fewer than"):
        P.run_cell(DAILY, DAILY, 5000, "sample", "minvar", WINDOW)


def test_run_all_has_every_cell_and_the_equal_weight_portfolio() -> None:
    runs = P.run_all(PANELS, DESIGN, WINDOW)
    assert len(runs) == 2 * 4 * 2 + 1 and "equal|-|-" in runs
    first = runs["minvar|D|sample"]
    assert first.returns.index[0] == pd.Timestamp("2021-01-01")
    assert first.returns.index[-1] <= pd.Timestamp("2021-06-30") and len(first.turnover) == 6
    assert np.allclose(first.weights.sum(axis=1), 1.0) and (first.weights >= 0).all().all()


def fake_runs(scale: float) -> dict[str, P.PortfolioRun]:
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2021-01-01", periods=300)
    base = pd.Series(rng.standard_normal(300) * 0.01, index=index)
    noise = pd.Series(rng.standard_normal(300) * 0.001, index=index)
    empty = pd.Series({index[0]: 1.0})
    weights = pd.DataFrame()
    runs = {}
    for freq in ("D", "W"):
        runs[f"minvar|{freq}|sample"] = P.PortfolioRun(base, empty, weights)
        for estimator in ("ledoit_wolf", "rmt"):
            runs[f"minvar|{freq}|{estimator}"] = P.PortfolioRun(
                scale * base + noise, empty, weights
            )
    return runs


def test_h5_rule_needs_a_lower_volatility_and_a_small_p_value() -> None:
    table, better = P.volatility_table(fake_runs(0.5), DESIGN, n_bootstrap=200)
    assert better["h5_supported"] and better["n_cells"] == 4
    assert better["p_one_sided"] == pytest.approx(1 / 201)
    row = table.set_index("cell").loc["minvar|D|rmt"]
    assert row["diff_vs_sample"] < 0 and row["diff_ci_upper"] < 0
    assert pd.isna(table.set_index("cell").loc["minvar|D|sample", "diff_vs_sample"])
    _, worse = P.volatility_table(fake_runs(1.5), DESIGN, n_bootstrap=200)
    assert not worse["h5_supported"] and worse["p_one_sided"] == pytest.approx(1.0)
    _, same = P.volatility_table(fake_runs(1.0), DESIGN, n_bootstrap=200)
    assert not same["h5_supported"] and abs(same["statistic"]) < 1e-2  # type: ignore[arg-type]


def test_backtest_table_charges_costs_on_turnover() -> None:
    runs = P.run_all(PANELS, DESIGN, WINDOW)
    table, pbo = P.backtest_table(runs, (0.0, 10.0), pbo_splits=4)
    assert len(table) == len(runs) * 2 and 0.0 <= pbo <= 1.0
    cell = table[table["cell"] == "minvar|D|sample"].set_index("cost_bps")
    run = runs["minvar|D|sample"]
    years = len(run.returns) / 252
    paid = run.turnover.sum() * 10 / 1e4 / years
    assert cell.loc[0.0, "ann_return"] - cell.loc[10.0, "ann_return"] == pytest.approx(paid)
    assert cell.loc[0.0, "turnover_per_year"] == pytest.approx(run.turnover.sum() / years)
    assert (table["sharpe_se"] > 0).all() and table["deflated_sharpe"].between(0, 1).all()


# ---- exploratory follow-up E3 (docs/FOLLOW_UPS.md item 3)


def test_long_short_minimum_variance_is_the_closed_form_and_may_go_short() -> None:
    cov = np.array([[1.0, 0.9, 0.1], [0.9, 2.0, 0.3], [0.1, 0.3, 1.5]])
    w = P.weights(cov, "minvar_ls")
    inv = np.linalg.inv(cov) @ np.ones(3)
    np.testing.assert_allclose(w, inv / inv.sum())
    assert w.sum() == pytest.approx(1.0) and w.min() < 0  # the highly correlated pair is shorted
    assert w @ cov @ w < P.min_variance_weights(cov) @ cov @ P.min_variance_weights(cov)
    assert "minvar_ls" not in P.PORTFOLIOS  # the declared design of run_all is unchanged


def test_nonlinear_shrinkage_keeps_eigenvectors_and_beats_the_sample_covariance() -> None:
    rng = np.random.default_rng(3)
    p, n = 30, 61
    truth = np.diag(np.linspace(0.5, 3.0, p))
    losses = {"sample": [], "nonlinear": []}
    for _ in range(20):
        x = rng.standard_normal((n, p)) * np.sqrt(np.diag(truth))
        sample, clean = np.cov(x, rowvar=False), P.nonlinear_shrinkage(x)
        np.testing.assert_allclose(clean @ sample, sample @ clean, atol=1e-8)  # same eigenvectors
        assert np.linalg.eigvalsh(clean).min() > np.linalg.eigvalsh(sample).min() > 0
        assert np.linalg.eigvalsh(clean).max() < np.linalg.eigvalsh(sample).max()
        losses["sample"].append(np.linalg.norm(sample - truth))
        losses["nonlinear"].append(np.linalg.norm(clean - truth))
    assert np.mean(losses["nonlinear"]) < 0.8 * np.mean(losses["sample"])
    with pytest.raises(P.PortfolioError):
        P.nonlinear_shrinkage(rng.standard_normal((30, 30)))


def test_nonlinear_shrinkage_changes_little_when_rows_are_many() -> None:
    x = np.random.default_rng(4).standard_normal((20000, 4)) * np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(
        P.nonlinear_shrinkage(x), np.cov(x, rowvar=False), rtol=0.05, atol=0.1
    )


def test_exploratory_covariance_estimators() -> None:
    window = DAILY.iloc[:120]
    sample = P.covariance(window, "sample")
    nl, nlc = P.covariance(window, "nonlinear"), P.covariance(window, "nonlinear_corr")
    np.testing.assert_allclose(nl, P.nonlinear_shrinkage(window.to_numpy()))
    np.testing.assert_allclose(np.diag(nlc), np.diag(sample))  # variances are not shrunk
    assert not np.allclose(nlc, sample) and not np.allclose(np.diag(nl), np.diag(sample))
    corr = np.full((6, 6), 0.3) + 0.7 * np.eye(6)
    var = np.linspace(1e-4, 4e-4, 6)
    full = P.covariance(window, "dcc_garch", dcc_corr=corr, dcc_var=var)
    np.testing.assert_allclose(np.diag(full), var)
    np.testing.assert_allclose(full[0, 5], 0.3 * np.sqrt(var[0] * var[5]))
    for bad in ({"dcc_corr": corr}, {"dcc_corr": corr, "dcc_var": -var}, {"dcc_var": var}):
        with pytest.raises(P.PortfolioError):
            P.covariance(window, "dcc_garch", **bad)


def test_the_dcc_garch_cell_reads_the_forecasts_made_the_day_before() -> None:
    index = pd.DatetimeIndex(DAILY.index)
    corr = np.tile(np.full((6, 6), 0.3) + 0.7 * np.eye(6), (len(index), 1, 1))
    var = np.tile(np.linspace(1e-4, 4e-4, 6), (len(index), 1))
    base = P.run_cell(
        DAILY, DAILY, 120, "dcc_garch", "minvar_ls", WINDOW, dcc_corr=corr, dcc_var=var
    )
    first = int(index.searchsorted(P.rebalance_dates(index, WINDOW)[0]))
    late = var.copy()
    late[first] *= np.linspace(1.0, 9.0, 6)  # dated on the rebalance day: must not be read
    same = P.run_cell(
        DAILY, DAILY, 120, "dcc_garch", "minvar_ls", WINDOW, dcc_corr=corr, dcc_var=late
    )
    pd.testing.assert_frame_equal(base.weights, same.weights)
    early = var.copy()
    early[first - 1] *= np.linspace(1.0, 9.0, 6)  # dated the day before: must be read
    moved = P.run_cell(
        DAILY, DAILY, 120, "dcc_garch", "minvar_ls", WINDOW, dcc_corr=corr, dcc_var=early
    )
    assert not np.allclose(base.weights.iloc[0], moved.weights.iloc[0])
    assert base.returns.notna().all() and base.weights.sum(axis=1).round(9).eq(1.0).all()


def test_nonlinear_shrinkage_recovers_a_flat_and_a_two_level_spectrum() -> None:
    """Known truths at p/n = 0.25, where sample eigenvalues spread over 0.28 to 2.2 times it."""
    rng = np.random.default_rng(0)
    p, n = 100, 401
    flat = np.linalg.eigvalsh(P.nonlinear_shrinkage(rng.standard_normal((n, p))))
    assert flat.min() > 0.9 and flat.max() < 1.3 and abs(flat.mean() - 1.0) < 0.05
    levels = np.r_[np.ones(50), np.full(50, 5.0)]
    two = np.linalg.eigvalsh(P.nonlinear_shrinkage(rng.standard_normal((n, p)) * np.sqrt(levels)))
    assert two[:50].min() > 0.9 and two[:50].max() < 1.7  # the sample ones reach 0.35 and 1.5
    assert two[50:].min() > 3.0 and two[50:].max() < 6.2  # the sample ones reach 2.2 and 8.9
    assert np.abs(two - levels).mean() < 0.35


def test_hrp_uses_single_linkage() -> None:
    cluster = pytest.importorskip("skfolio.cluster", exc_type=ImportError)
    optimization = pytest.importorskip("skfolio.optimization", exc_type=ImportError)
    skfolio = pytest.importorskip("skfolio", exc_type=ImportError)
    panel = make_factor_returns(700, (5, 4, 3), seed=1, start="2019-01-01").returns
    cov = np.cov(panel.to_numpy(), rowvar=False)
    weights = {}
    for method in ("SINGLE", "AVERAGE"):
        weights[method] = (
            optimization.HierarchicalRiskParity(
                risk_measure=skfolio.RiskMeasure.VARIANCE,
                hierarchical_clustering_estimator=cluster.HierarchicalClustering(
                    linkage_method=getattr(cluster.LinkageMethod, method)
                ),
            )
            .fit(panel)
            .weights_
        )
    ours = P.hrp_weights(cov, optimal_ordering=True)
    np.testing.assert_allclose(ours, weights["SINGLE"], atol=1e-8)
    assert np.abs(ours - weights["AVERAGE"]).max() > 1e-3


def test_the_dcc_cell_reads_the_correlation_forecast_made_the_day_before() -> None:
    index = pd.DatetimeIndex(DAILY.index)
    corr = np.tile(np.full((6, 6), 0.3) + 0.7 * np.eye(6), (len(index), 1, 1))
    first = int(index.searchsorted(P.rebalance_dates(index, WINDOW)[0]))
    other = np.full((6, 6), -0.15) + 1.15 * np.eye(6)
    base = P.run_cell(DAILY, DAILY, 120, "dcc", "minvar", WINDOW, dcc_corr=corr)
    late, early = corr.copy(), corr.copy()
    late[first], early[first - 1] = other, other
    same = P.run_cell(DAILY, DAILY, 120, "dcc", "minvar", WINDOW, dcc_corr=late)
    moved = P.run_cell(DAILY, DAILY, 120, "dcc", "minvar", WINDOW, dcc_corr=early)
    pd.testing.assert_frame_equal(base.weights, same.weights)
    assert not np.allclose(base.weights.iloc[0], moved.weights.iloc[0], atol=1e-4)


def test_the_variant_rule_needs_the_whole_interval_below_zero() -> None:
    """Two identical return series plus noise: a negative point with an interval across zero."""
    rng = np.random.default_rng(5)
    index = pd.bdate_range("2021-01-01", periods=300)
    base = pd.Series(rng.standard_normal(300) * 0.01, index=index)

    def run(scale: float, seed: int) -> P.PortfolioRun:
        noise = np.random.default_rng(seed).standard_normal(300) * 0.004
        return P.PortfolioRun(base * scale + noise, pd.Series(dtype=float), pd.DataFrame())

    variant_block = {
        "declared_on": "2026-10-11",
        "candidate": "rmt",
        "replaces": "ledoit_wolf",
        "note": "a candidate estimator against the one it would replace",
    }
    design = Economics.model_validate({**DESIGN.model_dump(), "h5_variant": variant_block})

    def variant(scale: float) -> dict[str, object]:
        runs = {
            "minvar|D|sample": run(1.0, 1),
            "minvar|D|ledoit_wolf": run(1.0, 2),
            "minvar|D|rmt": run(scale, 3),
        }
        out = P.volatility_table(runs, design, n_bootstrap=200)[1]["variant"]
        assert isinstance(out, dict)
        return out

    clear, close = variant(0.5), variant(0.995)
    assert clear["difference"] < 0 and clear["ci_upper"] < 0 and clear["rule_met"] is True
    assert close["ci_lower"] < 0 < close["ci_upper"] and close["rule_met"] is False
    assert close["difference"] < 0  # a negative point alone does not meet the rule


def test_nonlinear_shrinkage_equals_the_formula_written_as_loops() -> None:
    """Each kernel is centred on a sample eigenvalue and its width is proportional to *that*
    eigenvalue (the reference code's ``H = h * L'``), not to the point where it is evaluated."""
    x = np.random.default_rng(8).standard_normal((13, 4)) * np.array([1.0, 1.5, 2.0, 3.0])
    n, p = 12, 4
    lam, u = np.linalg.eigh(np.cov(x, rowvar=False))
    h, c, r5 = n ** (-1 / 3), p / n, np.sqrt(5.0)
    shrunk = np.empty(p)
    for i in range(p):
        density = hilbert = 0.0
        for j in range(p):
            width = h * lam[j]
            z = (lam[i] - lam[j]) / width
            density += 0.75 / r5 * max(1 - z**2 / 5, 0.0) / width / p
            log = np.log(abs((r5 - z) / (r5 + z)))
            hilbert += (-0.3 / np.pi * z + 0.75 / r5 / np.pi * (1 - z**2 / 5) * log) / width / p
        a, b = np.pi * c * lam[i] * density, 1 - c - np.pi * c * lam[i] * hilbert
        shrunk[i] = lam[i] / (a**2 + b**2)
    np.testing.assert_allclose(P.nonlinear_shrinkage(x), (u * shrunk) @ u.T)
