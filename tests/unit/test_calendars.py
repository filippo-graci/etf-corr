"""``etfcorr.data.calendars`` (step 1.2). No network; every function takes an injected calendar.

``nyse_sessions`` itself (the ``pandas_market_calendars`` adapter) is exercised against a small
stand-in module, exactly like ``yfinance`` in ``test_sources.py``: this proves the adapter reads
the documented API correctly, not that the real library behaves like the stand-in (NOT EXECUTED
OFFLINE; confirmed at Gate 2).
"""

import sys
import types
from datetime import date

import pandas as pd
import pytest

from etfcorr.data import calendars as C

# A small 2024 fixture calendar: ordinary business days minus New Year's Day (Mon 2024-01-01,
# observed) and MLK Day (Mon 2024-01-15) and Good Friday (Fri 2024-03-29) -- three NYSE holidays
# that fall on what would otherwise be a business day, so the fixture actually exercises holidays
# and not just weekends.
_HOLIDAYS = {date(2024, 1, 1), date(2024, 1, 15), date(2024, 3, 29)}


def fixture_sessions(start: date, end: date) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, end)
    return pd.DatetimeIndex([d for d in days if d.date() not in _HOLIDAYS])


# --------------------------------------------------------------------------------------
# is_24_7
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [
        ("BTC-USD", True),
        ("ETH-USD", True),
        ("SPY", False),
        ("^VIX", False),
        ("SMH", False),
        ("ARKF", False),
    ],
)
def test_is_24_7(ticker: str, expected: bool) -> None:
    assert C.is_24_7(ticker) is expected


# --------------------------------------------------------------------------------------
# last_session
# --------------------------------------------------------------------------------------


def test_last_session_on_a_session_day_is_itself() -> None:
    assert C.last_session(date(2024, 1, 2), fixture_sessions) == date(2024, 1, 2)


def test_last_session_on_a_weekend_steps_back_to_friday() -> None:
    assert C.last_session(date(2024, 1, 6), fixture_sessions) == date(2024, 1, 5)  # Sat -> Fri
    assert C.last_session(date(2024, 1, 7), fixture_sessions) == date(2024, 1, 5)  # Sun -> Fri


def test_last_session_on_a_holiday_steps_back_over_the_weekend_too() -> None:
    # MLK Day, Mon 2024-01-15: the fixture calendar has no session between Fri 01-12 and Tue 01-16
    assert C.last_session(date(2024, 1, 15), fixture_sessions) == date(2024, 1, 12)


def test_last_session_new_years_day_steps_back_to_the_prior_year() -> None:
    assert C.last_session(date(2024, 1, 1), fixture_sessions) == date(2023, 12, 29)  # Friday


def test_last_session_is_monotonic_non_decreasing_in_asof() -> None:
    days = [C.last_session(date(2024, 1, d), fixture_sessions) for d in range(1, 16)]
    assert days == sorted(days)


def test_last_session_raises_when_the_window_has_no_sessions() -> None:
    def empty_calendar(start: date, end: date) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([])

    with pytest.raises(C.CalendarError, match="no sessions"):
        C.last_session(date(2024, 1, 15), empty_calendar, lookback_days=1)


def test_last_session_lookback_days_must_be_positive() -> None:
    with pytest.raises(ValueError, match="lookback_days"):
        C.last_session(date(2024, 1, 15), fixture_sessions, lookback_days=0)


# --------------------------------------------------------------------------------------
# check_freshness / assert_fresh
# --------------------------------------------------------------------------------------


def test_freshness_ok_when_the_series_reaches_the_last_session() -> None:
    check = C.check_freshness("SMH", date(2024, 1, 5), date(2024, 1, 7), fixture_sessions)
    assert check == C.SessionCheck("SMH", "NYSE", date(2024, 1, 5), date(2024, 1, 5))
    assert not check.is_stale
    C.assert_fresh("SMH", date(2024, 1, 5), date(2024, 1, 7), fixture_sessions)  # does not raise


def test_freshness_stale_when_the_series_lags_the_last_session() -> None:
    check = C.check_freshness("SMH", date(2024, 1, 4), date(2024, 1, 7), fixture_sessions)
    assert check.is_stale
    with pytest.raises(C.StalenessError, match=r"SMH.*before the expected"):
        C.assert_fresh("SMH", date(2024, 1, 4), date(2024, 1, 7), fixture_sessions)


def test_freshness_a_series_ahead_of_asof_is_not_stale() -> None:
    """Ahead of asof cannot happen for a correctly-cut download, but the check must not misfire."""
    check = C.check_freshness("SMH", date(2024, 1, 10), date(2024, 1, 7), fixture_sessions)
    assert not check.is_stale


def test_freshness_24_7_ticker_is_expected_to_reach_asof_itself() -> None:
    """A crypto pair is expected to reach asof, not the NYSE last session (which may lag it)."""
    saturday = date(2024, 1, 6)
    check = C.check_freshness("BTC-USD", saturday, saturday, fixture_sessions)
    assert check == C.SessionCheck("BTC-USD", "24/7", saturday, saturday)
    assert not check.is_stale
    stale = C.check_freshness("BTC-USD", date(2024, 1, 5), saturday, fixture_sessions)
    assert stale.is_stale


def test_freshness_does_not_touch_the_calendar_for_24_7_tickers() -> None:
    """A 24/7 check must not even call sessions_fn (the crypto calendar has no such calendar)."""

    def boom(start: date, end: date) -> pd.DatetimeIndex:
        raise AssertionError("sessions_fn must not be called for a 24/7 ticker")

    check = C.check_freshness("ETH-USD", date(2024, 1, 6), date(2024, 1, 6), boom)
    assert not check.is_stale


# --------------------------------------------------------------------------------------
# nyse_sessions: adapter logic against a stand-in pandas_market_calendars module
# --------------------------------------------------------------------------------------


def install_stand_in(monkeypatch: pytest.MonkeyPatch, valid_days: pd.DatetimeIndex) -> list[str]:
    calls: list[str] = []

    class FakeCalendar:
        def valid_days(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
            calls.append(f"{start_date}..{end_date}")
            return valid_days

    def get_calendar(name: str) -> FakeCalendar:
        calls.append(f"get_calendar:{name}")
        return FakeCalendar()

    monkeypatch.setitem(
        sys.modules, "pandas_market_calendars", types.SimpleNamespace(get_calendar=get_calendar)
    )
    return calls


def test_nyse_sessions_reads_the_documented_api_and_normalises_tz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``valid_days`` returns a tz-aware (UTC) DatetimeIndex per the library's own docs."""
    raw = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="UTC")
    calls = install_stand_in(monkeypatch, raw)
    out = C.nyse_sessions(date(2024, 1, 1), date(2024, 1, 5))
    assert calls == ["get_calendar:NYSE", "2024-01-01..2024-01-05"]
    assert out.tz is None
    assert list(out.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]


def test_nyse_sessions_passes_through_the_calendar_name(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install_stand_in(monkeypatch, pd.DatetimeIndex([], tz="UTC"))
    C.nyse_sessions(date(2024, 1, 1), date(2024, 1, 2), calendar_name="XNYS")
    assert calls[0] == "get_calendar:XNYS"


def test_nyse_sessions_rejects_end_before_start() -> None:
    with pytest.raises(C.CalendarError, match="must not be before"):
        C.nyse_sessions(date(2024, 1, 5), date(2024, 1, 1))


def test_nyse_sessions_missing_library_is_an_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    with pytest.raises(ImportError):
        C.nyse_sessions(date(2024, 1, 1), date(2024, 1, 2))
