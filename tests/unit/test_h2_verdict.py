"""Unit tests for ``etfcorr.correlation.h2_verdict`` (step 4.6, the H2 verdict rule).

Three layers. (1) Pure logic on hand-built table lines: every branch and every boundary of the rule
fixed on 2026-10-02 (clause 1, the 0.5 threshold on the point estimate, the 'partly' branch that
reads only the market-series lines, the validation of the lines). (2) The p-value helper against
counts worked by hand. (3) End-to-end on planted-truth panels run through the real H2 engine: a null
world, a dependence-only world, a world where only the common factor's variance rises, and a world
where factor and idiosyncratic variances both rise (the configuration the real panel resembles),
each with the verdict the rule must give. The planted-truth worlds also document a measured
property of the rule: the per-asset delta cannot reach 0.5 when only the factor's variance rises
(it under-corrects), so such a world is read 'partly supported' through the market-series lines.
"""

import math

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import h2_verdict as V
from etfcorr.correlation import regime_correlation as R

SIZES = (3, 3, 3)
NAMES = [f"S{s}_{k}" for s, size in enumerate(SIZES) for k in range(size)]
SECTORS = {n: n.split("_")[0] for n in NAMES}


# ---- helpers for hand-built lines -----------------------------------------------------


def _stat(
    name: str,
    point: float,
    lower: float,
    upper: float,
    reps: np.ndarray | None = None,
    regime: int = 2,
) -> R.H2Stat:
    r = np.linspace(lower, upper, 101) if reps is None else reps
    return R.H2Stat(name, regime, point, lower, upper, r)


def _line(
    name: str,
    mode: str,
    *,
    raw: tuple[float, float, float] = (0.2, 0.1, 0.3),
    adj: tuple[float, float, float] = (0.05, -0.02, 0.12),
    share: tuple[float, float, float] = (0.8, 0.6, 1.0),
    pairs: str = "cross",
    alignment: str = "predetermined",
    n_rows: int = 3000,
    reps: np.ndarray | None = None,
) -> V.H2Line:
    return V.H2Line(
        name=name,
        delta_mode=mode,
        pairs=pairs,
        alignment=alignment,
        regime=2,
        reference=0,
        n_rows=n_rows,
        diff_raw=_stat("diff_raw", *raw, reps=reps),
        diff_adjusted=_stat("diff_adjusted", *adj),
        vol_share=_stat("vol_share", *share),
    )


def _primary(share: float = 0.88, **kw: object) -> V.H2Line:
    return _line("primary", "pair", share=(share, share - 0.1, share + 0.1), **kw)  # type: ignore[arg-type]


def _market(share: float, name: str = "spy") -> V.H2Line:
    return _line(name, "series", share=(share, share - 0.1, share + 0.1))


# ---- bootstrap_p_value ----------------------------------------------------------------


def test_p_value_is_never_zero_and_has_the_plus_one_floor() -> None:
    reps = np.linspace(0.1, 0.5, 99)  # all positive
    assert V.bootstrap_p_value(reps) == pytest.approx(2.0 / 100.0)


def test_p_value_known_counts() -> None:
    # B = 9: five replicates <= 0, four >= 0 (zero counted on both sides), so
    # p_le = (1+5)/10, p_ge = (1+5)/10 -> 2*0.6 capped at 1
    reps = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
    assert V.bootstrap_p_value(reps) == 1.0
    # B = 9: one replicate <= 0, eight > 0 -> p_le = 2/10, p_ge = 9/10 -> 0.4
    reps2 = np.array([-0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert V.bootstrap_p_value(reps2) == pytest.approx(0.4)


def test_p_value_is_symmetric_in_sign_and_respects_the_null_argument() -> None:
    reps = np.array([0.2, 0.3, 0.1, 0.5, 0.4, 0.25, 0.35])
    assert V.bootstrap_p_value(reps) == pytest.approx(V.bootstrap_p_value(-reps))
    assert V.bootstrap_p_value(reps, null=0.3) > V.bootstrap_p_value(reps, null=0.0)


@pytest.mark.parametrize(
    "bad",
    [np.array([]), np.array([[0.1, 0.2]]), np.array([0.1, np.nan]), np.array([0.1, np.inf])],
)
def test_p_value_rejects_bad_replicates(bad: np.ndarray) -> None:
    with pytest.raises(R.H2Error):
        V.bootstrap_p_value(bad)


# ---- clause 1 -------------------------------------------------------------------------


def test_clause1_needs_a_positive_point_and_an_interval_above_zero() -> None:
    assert V.clause1_holds(_primary(raw=(0.2, 0.05, 0.3)))
    assert not V.clause1_holds(_primary(raw=(0.2, 0.0, 0.3)))  # boundary: lower == 0
    assert not V.clause1_holds(_primary(raw=(0.2, -0.01, 0.3)))
    assert not V.clause1_holds(_primary(raw=(-0.2, -0.3, -0.05)))  # significant but reversed
    assert not V.clause1_holds(_primary(raw=(0.0, -0.1, 0.1)))


# ---- the rule: every branch and boundary ----------------------------------------------


def test_supported_when_the_primary_share_reaches_the_threshold() -> None:
    v = V.apply_h2_rule(_primary(0.88), [_market(1.33, "spy"), _market(0.60, "vix")])
    assert (v.verdict, v.clause1, v.clause2) == ("supported", True, "supported")
    assert v.primary_share == pytest.approx(0.88)
    assert v.market_shares == {"spy": 1.33, "vix": 0.60}


def test_threshold_boundary_is_inclusive_on_the_point_estimate() -> None:
    at = V.apply_h2_rule(_primary(0.5), [_market(0.1)])
    assert at.verdict == "supported"
    below = V.apply_h2_rule(_primary(0.4999), [_market(0.1)])
    assert below.verdict == "not supported" and below.clause2 == "not supported"


def test_partly_supported_reads_only_the_market_lines() -> None:
    v = V.apply_h2_rule(_primary(0.4), [_market(0.3, "spy"), _market(0.6, "vix")])
    assert (v.verdict, v.clause2) == ("partly supported", "partly supported")
    at = V.apply_h2_rule(_primary(0.4), [_market(0.5)])  # inclusive here too
    assert at.verdict == "partly supported"
    none = V.apply_h2_rule(_primary(0.4), [_market(0.3), _market(0.49, "vix")])
    assert none.verdict == "not supported"


def test_a_high_market_share_does_not_upgrade_a_supported_primary_or_a_failed_clause1() -> None:
    # primary below the threshold, market far above: 'partly', never 'supported'
    assert V.apply_h2_rule(_primary(0.3), [_market(1.4)]).verdict == "partly supported"
    # primary above: a low market line does not veto it
    assert V.apply_h2_rule(_primary(0.9), [_market(0.05)]).verdict == "supported"


def test_failed_clause1_gives_not_supported_whatever_the_shares() -> None:
    spans = _primary(0.95, raw=(0.2, -0.05, 0.4))
    v = V.apply_h2_rule(spans, [_market(1.2)])
    assert (v.verdict, v.clause1, v.clause2) == ("not supported", False, "not evaluated")
    assert any("includes 0" in n for n in v.notes)
    reversed_ = _primary(0.95, raw=(-0.2, -0.3, -0.1))
    v2 = V.apply_h2_rule(reversed_, [_market(1.2)])
    assert v2.verdict == "not supported" and v2.clause2 == "not evaluated"
    assert any("not positive" in n for n in v2.notes)


def test_a_custom_threshold_is_used_and_validated() -> None:
    assert V.apply_h2_rule(_primary(0.6), [_market(0.1)], threshold=0.7).verdict == "not supported"
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(R.H2Error, match="threshold"):
            V.apply_h2_rule(_primary(), [_market(0.5)], threshold=bad)


def test_the_verdict_does_not_depend_on_market_line_order_or_on_extra_weak_lines() -> None:
    a = V.apply_h2_rule(_primary(0.4), [_market(0.6, "vix"), _market(0.2, "spy")])
    b = V.apply_h2_rule(_primary(0.4), [_market(0.2, "spy"), _market(0.6, "vix")])
    assert a.verdict == b.verdict == "partly supported"
    c = V.apply_h2_rule(_primary(0.4), [_market(0.2, "spy"), _market(0.1, "extra")])
    assert c.verdict == "not supported"


def test_clause1_p_value_comes_from_the_primary_lines_raw_replicates() -> None:
    reps = np.linspace(0.05, 0.4, 199)
    v = V.apply_h2_rule(_primary(0.9, reps=reps), [_market(1.0)])
    assert v.clause1_p == pytest.approx(2.0 / 200.0)


# ---- reading notes (not rule inputs) ---------------------------------------------------


def test_notes_flag_an_interval_that_does_not_clear_the_threshold() -> None:
    wide = _line("primary", "pair", share=(0.55, 0.3, 0.9))
    notes = " ".join(V.apply_h2_rule(wide, [_market(0.9)]).notes)
    assert "does not clear" in notes
    tight = _line("primary", "pair", share=(0.88, 0.5, 1.07))  # lower bound exactly at 0.5
    assert "also >= 0.5" in " ".join(V.apply_h2_rule(tight, [_market(0.9)]).notes)
    undefined = _line("primary", "pair", share=(0.88, math.nan, math.nan))
    assert "not defined" in " ".join(V.apply_h2_rule(undefined, [_market(0.9)]).notes)


def test_notes_flag_a_share_above_one_and_name_the_line() -> None:
    v = V.apply_h2_rule(_primary(0.88), [_market(1.33, "spy"), _market(0.6, "vix")])
    text = " ".join(v.notes)
    assert "share above 1 on spy" in text and "vix," not in text
    assert "overshoots" in text
    assert not any(
        "share above 1" in n for n in V.apply_h2_rule(_primary(1.0), [_market(0.9)]).notes
    )


def test_residual_reading_requires_every_line_to_agree_and_exclude_zero() -> None:
    pos = [_line(f"l{i}", "pair" if i == 0 else "series", adj=(0.05, 0.01, 0.09)) for i in range(3)]
    assert V.residual_reading(pos) == "identified, positive"
    neg = [
        _line(f"l{i}", "pair" if i == 0 else "series", adj=(-0.05, -0.09, -0.01)) for i in range(3)
    ]
    assert V.residual_reading(neg) == "identified, negative"
    mixed = [pos[0], pos[1], neg[2]]
    assert V.residual_reading(mixed) == "not identified"
    spans = [pos[0], _line("x", "series", adj=(0.03, -0.01, 0.07))]
    assert V.residual_reading(spans) == "not identified"
    with pytest.raises(R.H2Error):
        V.residual_reading([])


def test_the_residual_note_is_printed_when_not_identified() -> None:
    lines = [
        _primary(0.88, adj=(0.027, -0.011, 0.064)),
        _line("spy", "series", adj=(-0.078, -0.106, -0.051), share=(1.33, 1.19, 1.59)),
    ]
    v = V.apply_h2_rule(lines[0], lines[1:])
    assert v.residual == "not identified"
    assert any("not identified" in n for n in v.notes)


# ---- validation of the lines ------------------------------------------------------------


def test_the_rule_rejects_lines_that_do_not_match_its_definition() -> None:
    ok_p, ok_m = _primary(), _market(0.5)
    with pytest.raises(R.H2Error, match="per-asset delta"):
        V.apply_h2_rule(_line("p", "series"), [ok_m])
    with pytest.raises(R.H2Error, match="market-series"):
        V.apply_h2_rule(ok_p, [_line("m", "pair")])
    with pytest.raises(R.H2Error, match="at least one market"):
        V.apply_h2_rule(ok_p, [])
    with pytest.raises(R.H2Error, match="cross-sector"):
        V.apply_h2_rule(_primary(pairs="within"), [ok_m])
    with pytest.raises(R.H2Error, match="predetermined"):
        V.apply_h2_rule(_primary(alignment="contemporaneous"), [ok_m])
    with pytest.raises(R.H2Error, match="same regimes"):
        V.apply_h2_rule(ok_p, [_line("m", "series", n_rows=2999)])
    with pytest.raises(R.H2Error, match="distinct"):
        V.apply_h2_rule(ok_p, [_market(0.5, "primary")])


# ---- sensitivity reading ----------------------------------------------------------------


def test_sensitivity_reading_needs_clause1_and_the_share_together() -> None:
    both = V.sensitivity_reading(_primary(0.7))
    assert both.same_reading and both.clause1 and both.share == pytest.approx(0.7)
    assert not V.sensitivity_reading(_primary(0.3)).same_reading
    assert not V.sensitivity_reading(_primary(0.9, raw=(0.2, -0.01, 0.3))).same_reading
    assert V.sensitivity_reading(_primary(0.5)).same_reading  # inclusive
    # the variants differ from the primary line in alignment or pair scope, which must be allowed
    contemporaneous = _primary(0.8, alignment="contemporaneous")
    within = _primary(0.8, pairs="within")
    assert V.sensitivity_reading(contemporaneous).same_reading
    assert V.sensitivity_reading(within).same_reading
    with pytest.raises(R.H2Error, match="threshold"):
        V.sensitivity_reading(_primary(), threshold=0.0)


# ---- report -----------------------------------------------------------------------------


def test_verdict_lines_state_the_rule_the_numbers_the_verdict_and_the_caveats() -> None:
    lines = [_primary(0.88), _market(1.33, "spy"), _market(0.6, "vix")]
    v = V.apply_h2_rule(lines[0], lines[1:])
    text = "\n".join(V.verdict_lines(v, lines))
    for needle in (
        "VERDICT: H2 SUPPORTED",
        "clause 1 (direction): MET",
        "clause 2 (a large part): SUPPORTED",
        ">= 0.5",
        "primary [pair delta]",
        "spy [series delta]",
        "Holm",
        "Forbes-Rigobon",
        "in-sample",
    ):
        assert needle in text
    failed = _primary(0.88, raw=(0.2, -0.1, 0.4))
    text2 = "\n".join(V.verdict_lines(V.apply_h2_rule(failed, [_market(1.0)]), [failed]))
    assert "NOT MET" in text2 and "NOT SUPPORTED" in text2 and "NOT EVALUATED" in text2


# ---- engine glue ------------------------------------------------------------------------


def _label_path(n: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(n)
    t, code = 0, int(rng.integers(0, 3))
    while t < n:
        length = int(rng.geometric(1.0 / 40.0))
        out[t : t + length] = code
        t += length
        code = int((code + 1 + rng.integers(0, 2)) % 3)
    return out


def _world(
    seed: int,
    beta: dict[int, float],
    fvol: dict[int, float],
    evol: dict[int, float],
    svol: dict[int, float],
    n: int = 3000,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Returns ``r = fvol * beta * f + evol * e`` governed by the label of the previous day.

    Cross-asset population correlation in regime g is ``(fvol*beta)**2 / ((fvol*beta)**2 +
    evol**2)``. Also returns the equal-weight series (an SPY stand-in) and an independent series
    whose scale follows ``svol`` (a stand-in for the VIX log change).
    """
    rng = np.random.default_rng(seed)
    labels = _label_path(n + 1, rng)
    gov = labels[:-1].astype(int)
    f = rng.standard_normal(n)
    e = rng.standard_normal((n, len(NAMES)))
    sys_part = np.array([fvol[g] * beta[g] for g in gov])
    idio = np.array([evol[g] for g in gov])
    x = 0.01 * sys_part[:, None] * f[:, None] + 0.01 * idio[:, None] * e
    index = pd.bdate_range("2018-01-01", periods=n + 1, name="date")
    returns = pd.DataFrame(x, index=index[1:], columns=NAMES)
    s2 = pd.Series(np.array([svol[g] for g in gov]) * rng.standard_normal(n), index=index[1:])
    return returns, pd.Series(labels, index=index, name="regime"), returns.mean(axis=1), s2


def _table_lines(
    world: tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series], n_boot: int = 100, seed: int = 0
) -> list[V.H2Line]:
    returns, labels, ew, s2 = world
    kw = {"n_bootstrap": n_boot, "seed": seed}
    primary = R.bootstrap_regime_correlations(returns, labels, SECTORS, **kw)  # type: ignore[arg-type]
    spy = R.bootstrap_regime_correlations(
        returns,
        labels,
        SECTORS,
        delta_mode="series",
        delta_series=ew,
        **kw,  # type: ignore[arg-type]
    )
    vix = R.bootstrap_regime_correlations(
        returns,
        labels,
        SECTORS,
        delta_mode="series",
        delta_series=s2,
        **kw,  # type: ignore[arg-type]
    )
    return [
        V.summarize_line("primary", primary),
        V.summarize_line("spy", spy),
        V.summarize_line("vix", vix),
    ]


FLAT = {0: 1.0, 1: 1.0, 2: 1.0}


def test_summarize_line_reads_the_engine_statistics() -> None:
    world = _world(1, dict.fromkeys(range(3), 0.7), FLAT, FLAT, FLAT)
    returns, labels, _, _ = world
    result = R.bootstrap_regime_correlations(returns, labels, SECTORS, n_bootstrap=30)
    line = V.summarize_line("primary", result)
    assert line.diff_raw is result.stat("diff_raw", 2)
    assert line.vol_share is result.stat("vol_share", 2)
    assert (line.delta_mode, line.pairs, line.alignment) == ("pair", "cross", "predetermined")
    assert line.n_rows == result.point.n_rows and line.reference == 0
    with pytest.raises(R.H2Error, match="reference"):
        V.summarize_line("x", result, regime=0)
    with pytest.raises(R.H2Error, match="not in the H2 table"):
        V.summarize_line("x", result, regime=7)


def test_planted_null_world_fails_clause1_and_is_not_supported() -> None:
    lines = _table_lines(_world(1, dict.fromkeys(range(3), 0.7), FLAT, FLAT, FLAT))
    v = V.apply_h2_rule(lines[0], lines[1:])
    assert (v.verdict, v.clause1, v.clause2) == ("not supported", False, "not evaluated")
    assert v.clause1_p > 0.2  # nothing changed; the p-value is far from the 5% region


def test_planted_dependence_only_world_has_clause1_but_no_volatility_share() -> None:
    """Total variance is constant, only the factor share of it rises: the correlation increase is
    a real change in dependence, so the rule must not call it volatility-driven."""
    b2 = {0: 0.3, 1: 0.5, 2: 0.8}
    beta = {g: math.sqrt(v) for g, v in b2.items()}
    evol = {g: math.sqrt(1.0 - v) for g, v in b2.items()}
    lines = _table_lines(_world(2, beta, FLAT, evol, FLAT))
    v = V.apply_h2_rule(lines[0], lines[1:])
    assert v.clause1 and v.verdict == "not supported" and v.clause2 == "not supported"
    assert abs(v.primary_share) < 0.1
    assert all(s < 0.5 for s in v.market_shares.values())
    assert v.residual == "identified, positive"


def test_planted_factor_variance_world_is_read_partly_supported() -> None:
    """Only the common factor's variance rises (x3.5). Truth: all of the increase is volatility.
    The per-asset delta under-corrects (share below 0.5, a measured property of the estimator on
    this design), the market-series lines do not, so the rule returns 'partly supported'."""
    beta = dict.fromkeys(range(3), 0.8)
    fvol = {0: 1.0, 1: math.sqrt(1.8), 2: math.sqrt(3.5)}
    lines = _table_lines(_world(3, beta, fvol, FLAT, {0: 1.0, 1: 1.3, 2: 2.0}))
    v = V.apply_h2_rule(lines[0], lines[1:])
    assert v.clause1 and v.verdict == "partly supported"
    assert 0.3 < v.primary_share < 0.5
    assert all(s >= 0.5 for s in v.market_shares.values())


def test_planted_world_where_factor_and_idiosyncratic_variance_rise_is_supported() -> None:
    """The configuration the real panel resembles (systematic variance x7, idiosyncratic x2): the
    per-asset delta then removes most of the raw increase and the rule reads 'supported'."""
    beta = dict.fromkeys(range(3), 0.8)
    fvol = {0: 1.0, 1: math.sqrt(3.0), 2: math.sqrt(7.0)}
    evol = {0: 1.0, 1: math.sqrt(1.5), 2: math.sqrt(2.0)}
    lines = _table_lines(_world(7, beta, fvol, evol, {0: 1.0, 1: 1.5, 2: 2.5}))
    v = V.apply_h2_rule(lines[0], lines[1:])
    assert (v.verdict, v.clause1, v.clause2) == ("supported", True, "supported")
    assert v.primary_share > 0.75
    assert v.clause1_p == pytest.approx(2.0 / 101.0)  # B = 100, every replicate positive


def test_the_table_and_the_verdict_are_deterministic_for_a_seed() -> None:
    world = _world(3, dict.fromkeys(range(3), 0.8), {0: 1.0, 1: 1.3, 2: 1.9}, FLAT, FLAT)
    a = _table_lines(world, n_boot=40, seed=5)
    b = _table_lines(world, n_boot=40, seed=5)
    va = V.apply_h2_rule(a[0], a[1:])
    vb = V.apply_h2_rule(b[0], b[1:])
    assert va == vb
    c = _table_lines(world, n_boot=40, seed=6)
    assert not np.array_equal(a[0].diff_raw.replicates, c[0].diff_raw.replicates)
    assert a[0].diff_raw.point == c[0].diff_raw.point  # the point estimate does not depend on it
