"""Step 1.2: exchange-session calendar helpers (last complete session, staleness, 24/7 series).

Question served: every later step in Phase 1 needs one answer to "what is the last date the
panel is allowed to reach right now" and "did series X actually reach it". Getting this wrong in
either direction is a standing risk already logged in ``docs/STATUS.md``: step 1.1's Gate 2 audit
was run mid-session and picked up an UNFINISHED trading day, because it used ``date.today()``
with no calendar check at all.

Design: a exchange session calendar is any callable ``(start, end) -> DatetimeIndex`` of the
dates the exchange was open, both inclusive (``SessionsFetcher``). Every pure function below
takes one as a parameter, defaulting to :func:`nyse_sessions` (the real, network/library-backed
calendar) so that production code needs no wiring, while tests inject a small synthetic
calendar and never touch the real library. This mirrors the ``Fetcher`` injection pattern of
``etfcorr.data.sources``/``universe_check``.

Scope: only the NYSE-listed equity/ETF calendar and a 24/7 "no calendar" mode for the crypto spot
pairs (Yahoo ``*-USD`` tickers). Gap-detection and calendar-based validation of a whole price
series (used to decide *why* a series is stale) is ``data/validate.py``'s job (step 1.3); this
module only answers "what date should a fresh series reach" and "is a given last date fresh".

Standing caveats (from ``docs/STATUS.md``, unchanged by this module):

* ``^VIX`` is treated as following the NYSE calendar (same as every other equity-index-style
  ticker). Checked on the vendor data on 2026-11-02: every NYSE session to 2026-09-25 has a VIX row,
  and the vendor also has VIX rows on two NYSE holidays (2026-05-25, 2026-09-07), dropped by the
  reindex; a 2026 Cboe holiday-schedule page found the *equities*
  schedule but nothing VIX-index-specific (checked 2026-09-27). Kept as the working assumption
  used since step 1.1's ``is_etf_like``.
* BTC-USD and ETH-USD bars are Yahoo daily bars closing at 00:00 UTC [Unverified, from step 1.1];
  treating their "session" as calendar day = ``asof`` (no NYSE alignment) is a simplification, not
  a claim that crypto and US-equity closes are synchronous (they are not; see the standing
  asynchronous-close caveat).

No third-party code is adapted here (the adapter below only calls the public
``pandas_market_calendars`` API); nothing to record in THIRD_PARTY.md.
"""

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

DEFAULT_CALENDAR_NAME = "NYSE"

# (start, end) -> the exchange's open session dates in [start, end], both inclusive.
SessionsFetcher = Callable[[date, date], pd.DatetimeIndex]


class CalendarError(ValueError):
    """A calendar or session computation could not be completed."""


class StalenessError(CalendarError):
    """A series' last observed date is earlier than the last complete session on/before asof."""


def _to_naive_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def nyse_sessions(
    start: date, end: date, *, calendar_name: str = DEFAULT_CALENDAR_NAME
) -> pd.DatetimeIndex:
    """Real NYSE (or another named exchange's) session index via ``pandas_market_calendars``.

    Lazy import, exactly like ``etfcorr.data.sources.fetch_yfinance``: this module imports
    without the library installed, and no network/library call happens at import time.

    STATUS: NOT EXECUTED OFFLINE. ``pandas_market_calendars`` is not installed in the planning
    sandbox and PyPI is unreachable there (same constraint as ``yfinance`` in step 1.1); the API
    used here (``get_calendar(name).valid_days(start_date=, end_date=)`` -> a tz-aware
    ``DatetimeIndex``) was checked against the library's own documentation
    (pandas-market-calendars.readthedocs.io/en/latest/usage.html, read 2026-09-27), not run.
    Every function below is fully tested offline against an injected synthetic calendar
    (``tests/unit/test_calendars.py``); confirm this one adapter function at Gate 2.
    """
    if end < start:
        raise CalendarError(f"end {end} must not be before start {start}")
    mcal = importlib.import_module("pandas_market_calendars")
    calendar = mcal.get_calendar(calendar_name)
    days = calendar.valid_days(start_date=start.isoformat(), end_date=end.isoformat())
    return _to_naive_days(days)


def is_24_7(ticker: str) -> bool:
    """True for a 24/7 spot series (Yahoo ``*-USD`` crypto pairs); False for everything else.

    Equities, ETFs and ``^VIX`` all use the exchange calendar (see the module-level caveat on
    ``^VIX``). No fuzzy matching: only the literal ``-USD`` suffix Yahoo uses for its crypto
    pairs (``BTC-USD``, ``ETH-USD``) is recognised.
    """
    return ticker.endswith("-USD")


def last_session(
    asof: date, sessions_fn: SessionsFetcher = nyse_sessions, *, lookback_days: int = 14
) -> date:
    """Latest exchange session on or before ``asof``.

    Pure given ``sessions_fn``; the default reaches the network/library. ``lookback_days`` only
    bounds how far back the search window is opened, not the answer: raising it cannot change
    the result, it only matters across a long holiday closure (never more than a few days for
    NYSE) or a thin/short injected test calendar.
    """
    if lookback_days < 1:
        raise ValueError("lookback_days must be at least 1")
    start = asof - timedelta(days=lookback_days)
    sessions = sessions_fn(start, asof)
    if len(sessions) == 0:
        raise CalendarError(
            f"no sessions found in [{start}, {asof}]; widen lookback_days or check the calendar"
        )
    return pd.Timestamp(sessions.max()).date()


@dataclass(frozen=True)
class SessionCheck:
    """Result of comparing a series' last observed date to what ``asof`` demands."""

    ticker: str
    calendar: str
    expected_last: date
    observed_last: date

    @property
    def is_stale(self) -> bool:
        return self.observed_last < self.expected_last


def check_freshness(
    ticker: str,
    observed_last: date,
    asof: date,
    sessions_fn: SessionsFetcher = nyse_sessions,
) -> SessionCheck:
    """Compare ``observed_last`` (a series' actual last row) to what ``asof`` demands for it.

    24/7 series (:func:`is_24_7`) are expected to reach ``asof`` itself; every other ticker is
    expected to reach :func:`last_session` (asof) on the given calendar.
    """
    if is_24_7(ticker):
        return SessionCheck(ticker, "24/7", asof, observed_last)
    expected = last_session(asof, sessions_fn)
    return SessionCheck(ticker, DEFAULT_CALENDAR_NAME, expected, observed_last)


def assert_fresh(
    ticker: str,
    observed_last: date,
    asof: date,
    sessions_fn: SessionsFetcher = nyse_sessions,
) -> None:
    """Raise :class:`StalenessError` unless ``observed_last`` reaches the expected last session.

    This is the "panel end date must equal the last session" gate named in
    EFT_project_v2.md section 4, item 11 and section 4's validation checklist, applied to one
    ticker; ``data/validate.py`` (step 1.3) applies it panel-wide as part of the quality gate.
    """
    check = check_freshness(ticker, observed_last, asof, sessions_fn)
    if check.is_stale:
        raise StalenessError(
            f"{ticker}: last observed date {check.observed_last} is before the expected last "
            f"session {check.expected_last} (calendar {check.calendar}) for asof {asof}"
        )
