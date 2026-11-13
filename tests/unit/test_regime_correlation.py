"""Unit tests for ``etfcorr.correlation.regime_correlation`` (step 4.2, the H2 engine).

Known answers first (the Forbes-Rigobon formula against a simulation of their own model, an EWMA
recursion worked by hand, an independent loop-based re-computation of the whole table), then
structure (symmetry, unit diagonal, identities between pair scopes), then behaviour that pins the
two design decisions fixed before any result: predetermined (t-1) alignment and a per-pair delta.
"""

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import regime_correlation as R
from etfcorr.correlation.estimators import estimate
from etfcorr.correlation.horizon_analysis import default_mean_block_length

SIZES = (3, 3, 3)


def _names(sizes: tuple[int, ...] = SIZES) -> list[str]:
    return [f"S{s}_{k}" for s, size in enumerate(sizes) for k in range(size)]


def _sectors(sizes: tuple[int, ...] = SIZES) -> dict[str, str]:
    return {f"S{s}_{k}": f"sector{s}" for s, size in enumerate(sizes) for k in range(size)}


def _label_path(n: int, n_regimes: int, rng: np.random.Generator, kind: str) -> np.ndarray:
    if kind == "iid":
        return rng.integers(0, n_regimes, size=n).astype(float)
    out = np.empty(n)
    t, code = 0, int(rng.integers(0, n_regimes))
    while t < n:
        length = int(rng.geometric(1.0 / 40.0))
        out[t : t + length] = code
        t += length
        code = int((code + 1 + rng.integers(0, n_regimes - 1)) % n_regimes)
    return out


def _make(
    n: int,
    seed: int,
    loadings: dict[int, float],
    *,
    vol: dict[int, float] | None = None,
    kind: str = "persistent",
    driver: str = "lagged",
    sizes: tuple[int, ...] = SIZES,
) -> tuple[pd.DataFrame, pd.Series]:
    """Returns for ``n`` dates and labels for ``n + 1`` dates (one extra leading day).

    Every asset loads on one common factor with the regime's loading ``b`` (all pairwise
    population correlations are then ``b**2``). ``driver="lagged"`` lets the label of the
    previous day govern the day (what predetermined alignment recovers); ``"same"`` lets the
    day's own label govern it.
    """
    rng = np.random.default_rng(seed)
    k = len(loadings)
    labels = _label_path(n + 1, k, rng, kind)
    gov = labels[:-1] if driver == "lagged" else labels[1:]
    b = np.array([loadings[int(c)] for c in gov])
    scale = np.array([(vol or {}).get(int(c), 1.0) for c in gov])
    m = sum(sizes)
    f = rng.standard_normal((n, 1))
    e = rng.standard_normal((n, m))
    z = b[:, None] * f + np.sqrt(1.0 - b[:, None] ** 2) * e
    index = pd.bdate_range("2018-01-01", periods=n + 1, name="date")
    returns = pd.DataFrame(0.01 * scale[:, None] * z, index=index[1:], columns=_names(sizes))
    return returns, pd.Series(labels, index=index, name="regime")


# ---- forbes_rigobon_adjust ----------------------------------------------------------


def test_fr_with_zero_delta_is_the_identity() -> None:
    rho = np.array([-0.9, -0.2, 0.0, 0.35, 0.99])
    np.testing.assert_allclose(R.forbes_rigobon_adjust(rho, 0.0), rho)


def test_fr_known_value() -> None:
    # 0.8 / sqrt(1 + 2 * (1 - 0.64)) = 0.8 / sqrt(1.72)
    assert float(R.forbes_rigobon_adjust(0.8, 2.0)) == pytest.approx(0.609994, abs=1e-6)


def test_fr_shrinks_toward_zero_for_positive_delta_and_grows_for_negative() -> None:
    rho = np.array([-0.7, -0.1, 0.2, 0.8])
    up = R.forbes_rigobon_adjust(rho, 1.5)
    down = R.forbes_rigobon_adjust(rho, -0.5)
    assert np.all(np.abs(up) < np.abs(rho)) and np.all(np.sign(up) == np.sign(rho))
    assert np.all(np.abs(down) > np.abs(rho)) and np.all(np.sign(down) == np.sign(rho))


def test_fr_stays_in_bounds_over_a_grid() -> None:
    rho = np.linspace(-1.0, 1.0, 41)
    delta = np.array([-0.99, -0.5, 0.0, 1.0, 10.0, 1e4])
    out = R.forbes_rigobon_adjust(rho[:, None], delta[None, :])
    assert out.shape == (41, 6) and np.all(np.abs(out) <= 1.0 + 1e-12)


@pytest.mark.parametrize(
    ("rho", "delta"), [(0.5, -1.0), (0.5, -2.0), (1.2, 0.5), (np.nan, 0.5), (0.5, np.inf)]
)
def test_fr_rejects_invalid_input(rho: float, delta: float) -> None:
    with pytest.raises(R.H2Error):
        R.forbes_rigobon_adjust(rho, delta)


def test_fr_recovers_the_unconditional_correlation_in_their_own_model() -> None:
    """The decision-log check turned into a test: y = beta * x + e with the variance of x raised
    by delta. The high-variance correlation is inflated; the adjusted one matches the low-variance
    correlation. The version without the square root (what a summary of the NBER paper showed)
    must miss, otherwise this test could not tell the two apart."""
    rng = np.random.default_rng(7)
    n, beta, delta = 600_000, 1.0, 2.0
    x_low = rng.standard_normal(n)
    y_low = beta * x_low + rng.standard_normal(n)
    x_high = np.sqrt(1.0 + delta) * rng.standard_normal(n)
    y_high = beta * x_high + rng.standard_normal(n)
    rho_low = float(np.corrcoef(x_low, y_low)[0, 1])
    rho_high = float(np.corrcoef(x_high, y_high)[0, 1])
    d = float(x_high.var(ddof=1) / x_low.var(ddof=1) - 1.0)
    assert rho_high > rho_low + 0.1  # the heteroskedasticity bias is large
    assert float(R.forbes_rigobon_adjust(rho_high, d)) == pytest.approx(rho_low, abs=0.01)
    without_root = rho_high / (1.0 + d * (1.0 - rho_high**2))
    assert abs(without_root - rho_low) > 0.05


# ---- EWMA volatility and standardisation --------------------------------------------


def test_ewma_volatility_matches_a_hand_computed_recursion() -> None:
    r = pd.DataFrame({"a": [1.0, 3.0, 2.0, 4.0]}, index=pd.bdate_range("2020-01-01", periods=4))
    sigma = R.ewma_volatility(r, lam=0.5, init=2)
    # start: mean square of the first two returns = (1 + 9) / 2 = 5
    # t=2: 0.5 * 5 + 0.5 * 4 = 4.5 ; t=3: 0.5 * 4.5 + 0.5 * 16 = 10.25
    expected = np.sqrt([np.nan, 5.0, 4.5, 10.25])
    np.testing.assert_allclose(sigma["a"].to_numpy(), expected, equal_nan=True)


def test_vol_standardize_divides_by_the_lagged_volatility() -> None:
    rng = np.random.default_rng(0)
    r = pd.DataFrame(
        0.01 * rng.standard_normal((200, 3)),
        index=pd.bdate_range("2020-01-01", periods=200),
        columns=list("abc"),
    )
    z = R.vol_standardize(r, lam=0.94, init=30)
    sigma = R.ewma_volatility(r, lam=0.94, init=30)
    assert z.iloc[:30].isna().all().all() and z.iloc[30:].notna().all().all()
    np.testing.assert_allclose((z * sigma.shift(1)).iloc[30:], r.iloc[30:])
    # z_t does not move when only the *current* return's volatility contribution is changed:
    r2 = r.copy()
    r2.iloc[100, 0] *= 3.0
    z2 = R.vol_standardize(r2, lam=0.94, init=30)
    assert z2.iloc[100, 0] == pytest.approx(3.0 * z.iloc[100, 0])  # same sigma_{t-1}
    assert z2.iloc[101, 0] != pytest.approx(z.iloc[101, 0])  # the shock enters the next sigma


@pytest.mark.parametrize(
    ("lam", "init", "rows", "bad"),
    [(0.0, 30, 100, None), (1.0, 30, 100, None), (0.94, 1, 100, None), (0.94, 30, 20, None),
     (0.94, 30, 100, np.nan), (0.94, 30, 100, np.inf)],
)  # fmt: skip
def test_ewma_volatility_rejects_bad_arguments(
    lam: float, init: int, rows: int, bad: float | None
) -> None:
    x = np.random.default_rng(1).standard_normal((rows, 2))
    if bad is not None:
        x[5, 0] = bad
    frame = pd.DataFrame(x, index=pd.bdate_range("2020-01-01", periods=rows))
    with pytest.raises(R.H2Error):
        R.ewma_volatility(frame, lam, init)


def test_ewma_volatility_rejects_a_constant_start_window() -> None:
    x = np.random.default_rng(1).standard_normal((100, 2))
    x[:30, 1] = 0.0
    frame = pd.DataFrame(x, index=pd.bdate_range("2020-01-01", periods=100))
    with pytest.raises(R.H2Error, match="zero"):
        R.ewma_volatility(frame, 0.0 + 0.94, 30)


# ---- pair_mask ----------------------------------------------------------------------


def test_pair_mask_counts_and_structure() -> None:
    sector_of = [_sectors()[c] for c in _names()]
    cross = R.pair_mask(sector_of, "cross")
    within = R.pair_mask(sector_of, "within")
    both = R.pair_mask(sector_of, "all")
    assert (cross.sum(), within.sum(), both.sum()) == (27, 9, 36)
    assert not (cross & within).any() and np.array_equal(cross | within, both)
    assert not np.tril(both).any()  # strictly upper triangular


def test_pair_mask_rejects_empty_scopes_and_unknown_names() -> None:
    with pytest.raises(R.H2Error):
        R.pair_mask(["a", "a", "a"], "cross")
    with pytest.raises(R.H2Error):
        R.pair_mask(["a", "b", "c"], "within")
    with pytest.raises(R.H2Error):
        R.pair_mask(["a", "b"], "diagonal")  # type: ignore[arg-type]


# ---- align_regime_labels ------------------------------------------------------------


def _small_labels() -> tuple[pd.Series, pd.Index]:
    idx = pd.bdate_range("2020-01-01", periods=10)
    labels = pd.Series([0, 0, 1, 1, 2, 2, 0, 0, 1, 1], index=idx, dtype=float)
    return labels, idx[1:]


def test_predetermined_alignment_uses_the_previous_row_including_for_the_first_date() -> None:
    labels, dates = _small_labels()
    out = R.align_regime_labels(labels, dates, "predetermined")
    assert out.index.equals(dates)
    np.testing.assert_array_equal(out.to_numpy(), labels.to_numpy()[:-1])


def test_contemporaneous_alignment_uses_the_same_date() -> None:
    labels, dates = _small_labels()
    out = R.align_regime_labels(labels, dates, "contemporaneous")
    np.testing.assert_array_equal(out.to_numpy(), labels.to_numpy()[1:])


def test_alignment_rejects_missing_dates_and_a_different_calendar() -> None:
    labels, dates = _small_labels()
    with pytest.raises(R.H2Error, match="missing"):
        R.align_regime_labels(labels.iloc[3:], dates, "predetermined")
    # label calendar has a row that the return calendar skips: 'one row earlier' would be wrong
    with pytest.raises(R.H2Error, match="consecutive"):
        R.align_regime_labels(labels, dates.delete(4), "predetermined")
    with pytest.raises(R.H2Error, match="alignment"):
        R.align_regime_labels(labels, dates, "centred")  # type: ignore[arg-type]


# ---- regime_correlations: point estimates --------------------------------------------


def _expected_table(
    returns: pd.DataFrame,
    codes: np.ndarray,
    sectors: dict[str, str],
    reference: int,
    scope: str = "cross",
) -> dict[int, tuple[float, float]]:
    """Independent loop-based recomputation of (avg_raw, avg_adjusted) per regime."""
    cols = list(returns.columns)
    var_ref = returns[codes == reference].var(ddof=1)
    out: dict[int, tuple[float, float]] = {}
    for g in sorted({int(c) for c in codes}):
        sub = returns[codes == g]
        d = sub.var(ddof=1) / var_ref - 1.0
        raws, adjs = [], []
        for i, j in itertools.combinations(range(len(cols)), 2):
            same = sectors[cols[i]] == sectors[cols[j]]
            if (scope == "cross" and same) or (scope == "within" and not same):
                continue
            r = float(np.corrcoef(sub.iloc[:, i], sub.iloc[:, j])[0, 1])
            a_i = r / math.sqrt(1.0 + float(d.iloc[i]) * (1.0 - r * r))
            a_j = r / math.sqrt(1.0 + float(d.iloc[j]) * (1.0 - r * r))
            raws.append(r)
            adjs.append(0.5 * (a_i + a_j))
        out[g] = (float(np.mean(raws)), float(np.mean(adjs)))
    return out


@pytest.mark.parametrize("alignment", ["predetermined", "contemporaneous"])
@pytest.mark.parametrize("scope", ["cross", "within", "all"])
def test_point_estimates_match_an_independent_loop_recomputation(
    alignment: str, scope: str
) -> None:
    returns, labels = _make(900, 3, {0: 0.3, 1: 0.6, 2: 0.85}, vol={0: 1.0, 1: 1.4, 2: 2.2})
    res = R.regime_correlations(
        returns,
        labels,
        _sectors(),
        alignment=alignment,  # type: ignore[arg-type]
        pairs=scope,  # type: ignore[arg-type]
    )
    aligned = R.align_regime_labels(labels, returns.index, alignment)  # type: ignore[arg-type]
    keep = aligned.notna().to_numpy()
    expected = _expected_table(
        returns[keep], aligned[keep].to_numpy().astype(int), _sectors(), 0, scope
    )
    for g, (raw, adj) in expected.items():
        assert res.table.loc[g, "avg_raw"] == pytest.approx(raw, abs=1e-12)
        assert res.table.loc[g, "avg_adjusted"] == pytest.approx(adj, abs=1e-12)


def test_regime_matrices_agree_with_the_estimator_module_and_are_well_formed() -> None:
    returns, labels = _make(900, 4, {0: 0.3, 1: 0.8}, vol={0: 1.0, 1: 2.0})
    res = R.regime_correlations(returns, labels, _sectors())
    aligned = R.align_regime_labels(labels, returns.index, "predetermined")
    for g in (0, 1):
        sub = returns[(aligned == g).to_numpy()]
        reference = estimate(sub, "pearson")
        np.testing.assert_allclose(res.raw_matrices[g].to_numpy(), reference.to_numpy(), atol=1e-12)
        raw, adj = res.raw_matrices[g], res.adjusted_matrices[g]
        assert list(raw.columns) == _names() == list(adj.index)
        for m in (raw, adj):
            np.testing.assert_allclose(m.to_numpy(), m.to_numpy().T, atol=1e-12)
            np.testing.assert_allclose(np.diag(m.to_numpy()), 1.0)
            assert np.all(np.abs(m.to_numpy()) <= 1.0 + 1e-12)
        assert np.linalg.eigvalsh(raw.to_numpy()).min() > -1e-10  # raw is PSD (a Gram matrix)


def test_reference_regime_is_unchanged_by_the_adjustment() -> None:
    returns, labels = _make(900, 5, {0: 0.4, 1: 0.8}, vol={0: 1.0, 1: 2.0})
    res = R.regime_correlations(returns, labels, _sectors())
    assert res.table.loc[0, "avg_adjusted"] == res.table.loc[0, "avg_raw"]
    assert res.table.loc[0, "mean_delta"] == 0.0
    np.testing.assert_array_equal(
        res.adjusted_matrices[0].to_numpy(), res.raw_matrices[0].to_numpy()
    )


def test_zero_variance_ratio_gives_adjusted_equal_to_raw() -> None:
    returns, labels = _make(900, 6, {0: 0.4, 1: 0.8, 2: 0.6}, vol={0: 1.0, 1: 3.0, 2: 0.5})
    aligned = R.align_regime_labels(labels, returns.index, "predetermined")
    keep = aligned.notna().to_numpy()
    returns, codes = returns[keep], aligned[keep].to_numpy()
    # z-score every column inside every regime: all variance ratios are exactly one
    for g in (0.0, 1.0, 2.0):
        sel = codes == g
        sub = returns.loc[sel]
        returns.loc[sel] = (sub - sub.mean()) / sub.std(ddof=1)
    # the labels passed in are already aligned, so they are used as they are
    res = R.regime_correlations(
        returns, aligned[keep].rename("regime"), _sectors(), alignment="contemporaneous"
    )
    np.testing.assert_allclose(res.table["mean_delta"], 0.0, atol=1e-12)
    np.testing.assert_allclose(res.table["avg_adjusted"], res.table["avg_raw"], atol=1e-12)


def test_a_variance_rise_pulls_the_adjusted_correlation_below_the_raw_one() -> None:
    returns, labels = _make(1500, 8, {0: 0.6, 1: 0.6}, vol={0: 1.0, 1: 2.5})
    res = R.regime_correlations(returns, labels, _sectors())
    assert res.table.loc[1, "mean_delta"] > 3.0  # variance ratio is about 2.5**2 = 6.25
    assert res.table.loc[1, "avg_adjusted"] < res.table.loc[1, "avg_raw"] - 0.05


def test_pair_scopes_satisfy_the_exact_weighting_identity() -> None:
    returns, labels = _make(900, 9, {0: 0.3, 1: 0.7}, vol={0: 1.0, 1: 1.8})
    out = {
        s: R.regime_correlations(returns, labels, _sectors(), pairs=s)  # type: ignore[arg-type]
        for s in ("cross", "within", "all")
    }
    assert (out["cross"].n_pairs, out["within"].n_pairs, out["all"].n_pairs) == (27, 9, 36)
    for col in ("avg_raw", "avg_adjusted"):
        combined = (27 * out["cross"].table[col] + 9 * out["within"].table[col]) / 36
        np.testing.assert_allclose(out["all"].table[col], combined, atol=1e-12)


def test_table_bookkeeping_counts_days_and_episodes() -> None:
    returns, labels = _make(600, 10, {0: 0.3, 1: 0.7})
    res = R.regime_correlations(returns, labels, _sectors())
    assert (
        int(res.table["n_obs"].sum()) == res.n_rows == len(returns)
    )  # no NaN label for a lagged first row
    assert (res.table["n_runs"] >= 1).all() and (res.table["n_runs"] <= res.table["n_obs"]).all()
    assert res.n_assets == 9 and res.reference == 0 and res.alignment == "predetermined"


def test_a_nan_label_drops_its_day() -> None:
    returns, labels = _make(600, 11, {0: 0.3, 1: 0.7})
    labels.iloc[0] = np.nan  # predetermined: the label of the first return date is gone
    labels.iloc[100] = np.nan  # ... and the label of day 101 (predetermined) / day 100 (contemp.)
    pre = R.regime_correlations(returns, labels, _sectors())
    con = R.regime_correlations(returns, labels, _sectors(), alignment="contemporaneous")
    assert pre.n_rows == len(returns) - 2
    assert con.n_rows == len(returns) - 1  # label 0 is not needed; label 100 is


def test_predetermined_alignment_recovers_a_lagged_driver_and_contemporaneous_does_not() -> None:
    """Regimes flip every day, the previous day's regime sets the dependence. Grouping by the
    previous day's label recovers it; grouping by the same-day label mixes the two states."""
    returns, labels = _make(6000, 12, {0: 0.3, 1: 0.9}, kind="iid", driver="lagged")
    pre = R.regime_correlations(returns, labels, _sectors()).table
    con = R.regime_correlations(returns, labels, _sectors(), alignment="contemporaneous").table
    assert pre.loc[0, "avg_raw"] == pytest.approx(0.09, abs=0.03)
    assert pre.loc[1, "avg_raw"] == pytest.approx(0.81, abs=0.03)
    for g in (0, 1):  # a 50/50 mixture of the two states: 0.5 * (0.09 + 0.81)
        assert con.loc[g, "avg_raw"] == pytest.approx(0.45, abs=0.05)


def test_series_delta_mode_applies_one_common_delta() -> None:
    returns, labels = _make(900, 13, {0: 0.4, 1: 0.7}, vol={0: 1.0, 1: 2.0})
    aligned = R.align_regime_labels(labels, returns.index, "predetermined")
    rng = np.random.default_rng(0)
    mult = np.where(aligned.to_numpy() == 1.0, 2.0, 1.0)
    s = pd.Series(rng.standard_normal(len(returns)) * mult, index=returns.index)
    res = R.regime_correlations(returns, labels, _sectors(), delta_mode="series", delta_series=s)
    keep = aligned.notna().to_numpy()
    codes = aligned[keep].to_numpy()
    delta = float(s[keep][codes == 1].var(ddof=1) / s[keep][codes == 0].var(ddof=1) - 1.0)
    assert res.table.loc[1, "mean_delta"] == pytest.approx(delta)
    expected = R.forbes_rigobon_adjust(res.raw_matrices[1].to_numpy(), delta)
    mask = R.pair_mask([_sectors()[c] for c in _names()], "cross")
    assert res.table.loc[1, "avg_adjusted"] == pytest.approx(float(expected[mask].mean()))
    assert res.table.loc[0, "avg_adjusted"] == res.table.loc[0, "avg_raw"]


def test_input_is_not_mutated() -> None:
    returns, labels = _make(600, 14, {0: 0.3, 1: 0.7})
    r0, l0 = returns.copy(), labels.copy()
    R.regime_correlations(returns, labels, _sectors())
    pd.testing.assert_frame_equal(returns, r0)
    pd.testing.assert_series_equal(labels, l0)


# ---- regime_correlations: invalid input ---------------------------------------------


def test_invalid_inputs_raise_h2error() -> None:
    returns, labels = _make(600, 15, {0: 0.3, 1: 0.7})
    sectors = _sectors()
    with pytest.raises(R.H2Error, match="sector"):
        R.regime_correlations(returns, labels, {k: v for k, v in sectors.items() if k != "S0_0"})
    with pytest.raises(R.H2Error, match="DataFrame"):
        R.regime_correlations(returns["S0_0"], labels, sectors)  # type: ignore[arg-type]
    with pytest.raises(R.H2Error, match="at least 2 columns"):
        R.regime_correlations(returns[["S0_0"]], labels, sectors)
    with pytest.raises(R.H2Error, match="increasing"):
        R.regime_correlations(returns.iloc[::-1], labels, sectors)
    bad = returns.copy()
    bad.iloc[3, 0] = np.inf
    with pytest.raises(R.H2Error, match="inf"):
        R.regime_correlations(bad, labels, sectors)
    with pytest.raises(R.H2Error, match="reference"):
        R.regime_correlations(returns, labels, sectors, reference=5)
    with pytest.raises(R.H2Error, match="do not occur"):
        R.regime_correlations(returns, labels, sectors, regimes=[0, 1, 7])
    with pytest.raises(R.H2Error, match="integer"):
        R.regime_correlations(returns, labels + 0.5, sectors)
    with pytest.raises(R.H2Error, match="delta_series"):
        R.regime_correlations(returns, labels, sectors, delta_mode="series")
    with pytest.raises(R.H2Error, match="only used"):
        R.regime_correlations(
            returns, labels, sectors, delta_series=pd.Series(0.0, index=returns.index)
        )
    with pytest.raises(R.H2Error, match="delta_mode"):
        R.regime_correlations(returns, labels, sectors, delta_mode="mean")  # type: ignore[arg-type]


def test_too_few_days_in_a_regime_or_a_constant_column_raise() -> None:
    returns, labels = _make(600, 16, {0: 0.3, 1: 0.7})
    rare = labels.copy()
    rare[:] = 0.0
    rare.iloc[300:310] = 1.0  # only 10 days in regime 1
    with pytest.raises(R.H2Error, match="need >= 30"):
        R.regime_correlations(returns, rare, _sectors())
    flat = returns.copy()
    aligned = R.align_regime_labels(labels, returns.index, "predetermined")
    flat.loc[(aligned == 1.0).to_numpy(), "S1_1"] = 0.001
    with pytest.raises(R.H2Error, match="zero variance"):
        R.regime_correlations(flat, labels, _sectors())


# ---- bootstrap_regime_correlations --------------------------------------------------

B = 120  # replicates in the unit tests (the real run uses 2000)


def test_bootstrap_is_deterministic_for_a_seed_and_differs_across_seeds() -> None:
    returns, labels = _make(700, 20, {0: 0.3, 1: 0.7})
    kw = {"n_bootstrap": B}
    a = R.bootstrap_regime_correlations(returns, labels, _sectors(), seed=3, **kw)  # type: ignore[arg-type]
    b = R.bootstrap_regime_correlations(returns, labels, _sectors(), seed=3, **kw)  # type: ignore[arg-type]
    c = R.bootstrap_regime_correlations(returns, labels, _sectors(), seed=4, **kw)  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(a.frame(), b.frame())
    assert not a.frame()["lower"].equals(c.frame()["lower"])


def test_bootstrap_points_equal_the_point_estimate_function() -> None:
    returns, labels = _make(700, 21, {0: 0.3, 1: 0.5, 2: 0.8}, vol={0: 1.0, 1: 1.3, 2: 2.0})
    point = R.regime_correlations(returns, labels, _sectors())
    boot = R.bootstrap_regime_correlations(returns, labels, _sectors(), n_bootstrap=B)
    for g in (0, 1, 2):
        assert boot.stat("avg_raw", g).point == point.table.loc[g, "avg_raw"]
        assert boot.stat("avg_adjusted", g).point == point.table.loc[g, "avg_adjusted"]
    for g in (1, 2):
        raw = point.table.loc[g, "avg_raw"] - point.table.loc[0, "avg_raw"]
        adj = point.table.loc[g, "avg_adjusted"] - point.table.loc[0, "avg_adjusted"]
        assert boot.stat("diff_raw", g).point == pytest.approx(raw)
        assert boot.stat("diff_adjusted", g).point == pytest.approx(adj)
        assert boot.stat("vol_explained", g).point == pytest.approx(raw - adj)


def test_statistics_set_and_reference_exclusions() -> None:
    returns, labels = _make(700, 22, {0: 0.3, 1: 0.5, 2: 0.8})
    boot = R.bootstrap_regime_correlations(returns, labels, _sectors(), n_bootstrap=B)
    names = {(s.name, s.regime) for s in boot.stats}
    assert {("avg_raw", g) for g in (0, 1, 2)} <= names
    assert {
        (n, g) for n in ("diff_raw", "diff_adjusted", "vol_explained", "vol_share") for g in (1, 2)
    } <= names
    assert not any(n in ("diff_raw", "vol_share") and g == 0 for n, g in names)
    assert len(boot.stats) == 3 * 2 + 2 * 4
    with pytest.raises(KeyError):
        boot.stat("diff_raw", 0)
    frame = boot.frame()
    assert list(frame.columns) == ["stat", "regime", "point", "lower", "upper", "excludes_zero"]
    assert len(frame) == len(boot.stats)


def test_replicate_identities_and_interval_ordering() -> None:
    returns, labels = _make(700, 23, {0: 0.3, 1: 0.8}, vol={0: 1.0, 1: 2.0})
    boot = R.bootstrap_regime_correlations(returns, labels, _sectors(), n_bootstrap=B)
    d_raw, d_adj = boot.stat("diff_raw", 1), boot.stat("diff_adjusted", 1)
    np.testing.assert_allclose(
        boot.stat("vol_explained", 1).replicates, d_raw.replicates - d_adj.replicates, atol=1e-14
    )
    np.testing.assert_allclose(
        d_raw.replicates,
        boot.stat("avg_raw", 1).replicates - boot.stat("avg_raw", 0).replicates,
        atol=1e-14,
    )
    for s in boot.stats:
        assert len(s.replicates) == B
        if not math.isnan(s.lower):
            assert s.lower <= s.upper


def test_labels_travel_with_their_returns_in_every_replicate() -> None:
    """Regime 1 days are highly correlated (0.81), regime 0 days barely (0.09). If the bootstrap
    resampled returns and labels with different indices, every replicate would mix the two
    states. With one shared index every replicate keeps them apart."""
    returns, labels = _make(2500, 24, {0: 0.3, 1: 0.9}, kind="persistent")
    boot = R.bootstrap_regime_correlations(returns, labels, _sectors(), n_bootstrap=B)
    assert boot.stat("avg_raw", 1).replicates.min() > 0.65
    assert boot.stat("avg_raw", 0).replicates.max() < 0.25


def test_interval_covers_the_truth_and_a_null_difference_covers_zero() -> None:
    returns, labels = _make(3000, 25, {0: 0.6, 1: 0.6}, kind="persistent")
    boot = R.bootstrap_regime_correlations(returns, labels, _sectors(), n_bootstrap=250, seed=1)
    for g in (0, 1):  # true cross-sector correlation 0.6**2 = 0.36
        s = boot.stat("avg_raw", g)
        assert s.lower < 0.36 < s.upper
    diff = boot.stat("diff_raw", 1)
    assert diff.lower < 0.0 < diff.upper and not diff.excludes_zero
    assert math.isnan(boot.stat("vol_share", 1).lower)  # not defined when diff_raw spans zero


def test_a_real_difference_is_detected_and_its_share_interval_is_defined() -> None:
    returns, labels = _make(3000, 26, {0: 0.3, 1: 0.8}, vol={0: 1.0, 1: 1.8})
    boot = R.bootstrap_regime_correlations(returns, labels, _sectors(), n_bootstrap=250, seed=2)
    diff = boot.stat("diff_raw", 1)
    assert diff.excludes_zero and diff.lower > 0.2
    share = boot.stat("vol_share", 1)
    assert not math.isnan(share.lower) and share.lower <= share.point <= share.upper
    assert 0.0 < share.point < 1.0  # a rise in variance explains part, not all, of this increase


def test_block_length_default_and_override_are_recorded() -> None:
    returns, labels = _make(700, 27, {0: 0.3, 1: 0.7})
    default = R.bootstrap_regime_correlations(returns, labels, _sectors(), n_bootstrap=20)
    assert default.mean_block_length == default_mean_block_length(default.point.n_rows)
    custom = R.bootstrap_regime_correlations(
        returns, labels, _sectors(), n_bootstrap=20, mean_block_length=60.0
    )
    assert custom.mean_block_length == 60.0


def test_bootstrap_rejects_bad_settings_and_a_replicate_that_loses_a_regime() -> None:
    returns, labels = _make(700, 28, {0: 0.3, 1: 0.7})
    with pytest.raises(R.H2Error, match="n_bootstrap"):
        R.bootstrap_regime_correlations(returns, labels, _sectors(), n_bootstrap=1)
    with pytest.raises(R.H2Error, match="alpha"):
        R.bootstrap_regime_correlations(returns, labels, _sectors(), n_bootstrap=10, alpha=1.5)
    # regime 1 has 31 days: the point estimate is fine, a replicate will usually fall below 30
    thin = labels.copy()
    thin[:] = 0.0
    thin.iloc[100:131] = 1.0
    with pytest.raises(R.H2Error, match="need >= 30"):
        R.bootstrap_regime_correlations(returns, thin, _sectors(), n_bootstrap=200, seed=0)


def test_series_mode_bootstrap_runs_and_resamples_the_series_jointly() -> None:
    returns, labels = _make(900, 29, {0: 0.4, 1: 0.7}, vol={0: 1.0, 1: 2.0})
    s = returns.mean(axis=1)
    boot = R.bootstrap_regime_correlations(
        returns, labels, _sectors(), delta_mode="series", delta_series=s, n_bootstrap=B
    )
    assert boot.point.delta_mode == "series"
    assert boot.stat("avg_adjusted", 0).point == boot.stat("avg_raw", 0).point
    assert np.allclose(boot.stat("avg_adjusted", 0).replicates, boot.stat("avg_raw", 0).replicates)


# ---- reporting ----------------------------------------------------------------------


def test_summary_lines_name_the_regimes_and_state_the_caveats() -> None:
    returns, labels = _make(700, 30, {0: 0.3, 1: 0.5, 2: 0.8}, vol={0: 1.0, 1: 1.3, 2: 2.0})
    boot = R.bootstrap_regime_correlations(returns, labels, _sectors(), n_bootstrap=B)
    text = "\n".join(R.h2_summary_lines(boot))
    for needle in ("predetermined", "low", "mid", "high", "high minus low", "in-sample", "Forbes"):
        assert needle in text
    custom = "\n".join(R.h2_summary_lines(boot, ["calm", "busy", "storm"]))
    assert "storm minus calm" in custom
    with pytest.raises(R.H2Error, match="names"):
        R.h2_summary_lines(boot, ["a", "b"])


def test_the_caveat_line_names_the_direction_of_the_bias_of_the_delta_used() -> None:
    returns, labels = _make(700, 31, {0: 0.3, 1: 0.7}, vol={0: 1.0, 1: 2.0})
    pair = R.bootstrap_regime_correlations(returns, labels, _sectors(), n_bootstrap=20)
    series = R.bootstrap_regime_correlations(
        returns,
        labels,
        _sectors(),
        delta_mode="series",
        delta_series=returns.mean(axis=1),
        n_bootstrap=20,
    )
    assert "under-corrects" in "\n".join(R.h2_summary_lines(pair))
    assert "over-attribute" in "\n".join(R.h2_summary_lines(series))
