"""Point-in-time discipline for ``etfcorr.data.holdings`` (step 2.4).

``latest_asof`` plays the role ``etfcorr.leakage``'s ``assert_no_lookahead``/
``assert_future_invariant`` play for row-indexed feature functions, but it takes a *collection* of
point-in-time snapshots plus a cutoff date, not a ``DataFrame`` indexed by date -- the same reason
``correlation/estimators.py``'s ticker-indexed matrices needed a bespoke purity substitute
(``tests/leakage/test_estimators_purity.py``) instead of the shared detectors. This file follows
that precedent: it shows directly, against the real production function, that adding a snapshot
filed *after* the query date never changes what ``latest_asof`` returns for that date, and that a
deliberately leaky twin (one that picks by ``asof`` instead of ``filing_date``) is caught failing
the same property.
"""

from datetime import date

from etfcorr.data.holdings import Holding, HoldingsSnapshot, latest_asof

_HOLDING = Holding(constituent="111111111", name="X Corp", weight=0.1)


def _snap(asof: date, filing_date: date) -> HoldingsSnapshot:
    return HoldingsSnapshot(
        ticker="AAA", asof=asof, filing_date=filing_date, source="sec_nport", holdings=(_HOLDING,)
    )


def _history() -> list[HoldingsSnapshot]:
    """Four quarterly snapshots, each filed with the realistic ~45 day N-PORT reporting lag."""
    return [
        _snap(date(2023, 12, 31), date(2024, 2, 14)),
        _snap(date(2024, 3, 31), date(2024, 5, 15)),
        _snap(date(2024, 6, 30), date(2024, 8, 14)),
        _snap(date(2024, 9, 30), date(2024, 11, 14)),
    ]


def _leaky_latest_asof(snapshots: list[HoldingsSnapshot], as_of: date) -> HoldingsSnapshot | None:
    """A deliberately leaky twin: filters on the *reporting* date instead of the filing date --
    the exact bug this test suite exists to catch (using the "what the holdings describe" date
    as if it were "when the holdings became public")."""
    eligible = [s for s in snapshots if s.asof <= as_of]
    if not eligible:
        return None
    return max(eligible, key=lambda s: s.asof)


def test_future_filing_never_changes_a_past_query_real_function() -> None:
    history = _history()
    as_of = date(
        2024, 6, 1
    )  # strictly between the Q1 filing (2024-05-15) and Q2 filing (2024-08-14)
    before = latest_asof(history, as_of)
    assert before is not None
    assert before.asof == date(
        2024, 3, 31
    )  # the Q1 report, correctly the latest one *public* by June

    # Add a snapshot filed after `as_of` (the Q2 report, filed 2024-08-14) -- this must not change
    # the answer for a query made on 2024-06-01, when that filing did not yet exist publicly.
    with_future = [*history, _snap(date(2024, 6, 30), date(2024, 8, 14))]
    after = latest_asof(with_future, as_of)
    assert after == before


def test_future_filing_never_changes_a_past_query_many_cuts() -> None:
    """The truncation property checked at several cut points, mirroring
    ``etfcorr.leakage.assert_no_lookahead``'s "check several cuts, not just one" discipline."""
    history = _history()
    for as_of in (date(2024, 2, 20), date(2024, 6, 1), date(2024, 9, 1), date(2024, 11, 20)):
        eligible_now = latest_asof(history, as_of)
        # data "from the future" relative to as_of: every snapshot filed after as_of, shuffled in
        future_snap = _snap(date(2025, 12, 31), date(2026, 2, 14))
        assert latest_asof([*history, future_snap], as_of) == eligible_now


def test_leaky_twin_is_caught_by_the_same_property() -> None:
    """The leaky twin fails the identical check the real function passes, proving the test would
    have caught this bug had it been present in ``latest_asof`` itself (the project's O4 rule:
    every leakage test must also fail on a deliberately leaky version)."""
    history = _history()
    as_of = date(2024, 6, 1)
    # A report describing a period just before as_of (asof <= as_of, so the buggy asof-based rule
    # accepts it) but filed far in the future (filing_date way after as_of, so it was never
    # actually public on that date) -- exactly the leak this property is meant to catch.
    early_period_late_filing = _snap(date(2024, 5, 31), date(2026, 1, 1))
    baseline = _leaky_latest_asof(history, as_of)
    leaked = _leaky_latest_asof([*history, early_period_late_filing], as_of)
    assert leaked != baseline  # the leaky twin's answer changes: it saw a filing from the future

    # The real function is unaffected by the identical addition.
    real_baseline = latest_asof(history, as_of)
    real_after = latest_asof([*history, early_period_late_filing], as_of)
    assert real_after == real_baseline


def test_ties_on_filing_date_break_on_report_period() -> None:
    tie_a = _snap(date(2024, 3, 31), date(2024, 5, 15))
    tie_b = _snap(date(2024, 6, 30), date(2024, 5, 15))  # same filing_date, later report period
    result = latest_asof([tie_a, tie_b], date(2024, 12, 31))
    assert result == tie_b
