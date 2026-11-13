"""Vendor adapters (step 1.1: yfinance only; Stooq and FRED follow in later Phase 1 steps).

Every adapter returns ``etfcorr.data.universe_check.FetchResult`` so that the rest of the data
layer never sees a vendor type. The vendor library is imported lazily inside the function: this
module imports without ``yfinance`` installed, and nothing here runs at import time.

Terms of use: yfinance wraps an unofficial Yahoo endpoint intended for personal use (see
EFT_project_v2.md section 4, item 1). Raw vendor data is never committed; only derived results
are. Requests are spaced by ``courtesy_pause`` seconds and retried with exponential backoff.

Prices: ``auto_adjust=False`` so that ``close`` is the split-adjusted, not dividend-adjusted,
price at which the fund traded; dollar volume (step 1.1's audit) is computed from it, never from
a total-return price. Step 1.2 adds ``adj_close``: yfinance's ``history(auto_adjust=False, ...)``
already returns an ``Adj Close`` column in the same response (split- and dividend-adjusted,
i.e. total-return), so capturing it costs no extra vendor call; it is read from the *same*
frame that ``close``/``volume`` come from, not fetched separately. ``adj_close`` is ``NaN`` when
the vendor omits the column (kept as a column of NaNs, not dropped, so callers see a stable
schema): step 1.1's audit (``etfcorr.data.universe_check``) reads only ``close``/``volume`` and
is unaffected by this addition. The adjusted series is retroactively revised by the vendor after
corporate actions, so ``download.py`` records a download timestamp and a hash, not just a value.

STATUS: the price path (``Ticker(...).history(start=, end=, auto_adjust=, actions=)`` returning a
frame with ``Close``, ``Adj Close`` and ``Volume`` and a tz-aware index) worked at Gate 2 of step
1.1 (56 of 56 tickers, no errors; ``Adj Close`` itself was not yet consumed at that point). The
metadata path did not: ``history_metadata`` is a lazy wrapper object in yfinance's current source
(github.com/ranaroussi/yfinance, ``scrapers/history.py``, read on 2026-09-21 on ``main``, not on
the installed 1.7.0), not a dict, and the first version silently returned ``None`` for every
ticker. It now reads the wrapper as dict-like and falls back to ``get_info()``
(``longName``/``shortName``, ``quoteType``); whether that works on the installed version is
UNVERIFIED until the next local run, and the audit summary reports how many names and instrument
types were obtained. The tests use a stand-in ``yfinance`` module, so they prove this module's
own logic, not the vendor's behaviour.
"""

import importlib
import time
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from etfcorr.data.universe_check import FetchResult


def _normalise_history(raw: Any) -> pd.DataFrame | None:
    """Vendor frame -> ``close``/``volume``/``adj_close`` on a tz-naive, day-normalised index.

    ``None`` if the vendor returned nothing. ``adj_close`` is ``NaN`` for every row when the
    vendor's frame has no ``Adj Close`` column (older yfinance versions, or a vendor that never
    provides it), so the schema is stable regardless of what the vendor sent.
    """
    if raw is None or len(raw) == 0:
        return None
    index = pd.DatetimeIndex(raw.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    if "Adj Close" in raw.columns:
        adj_close = pd.to_numeric(raw["Adj Close"], errors="coerce").to_numpy(dtype=float)
    else:
        adj_close = np.full(len(raw), np.nan)
    return pd.DataFrame(
        {
            "close": pd.to_numeric(raw["Close"], errors="coerce").to_numpy(dtype=float),
            "volume": pd.to_numeric(raw["Volume"], errors="coerce").to_numpy(dtype=float),
            "adj_close": adj_close,
        },
        index=index.normalize(),
    )


def _lookup(source: Any, key: str) -> Any:
    """Read ``key`` from a dict or a dict-like wrapper; ``None`` when absent or unreadable.

    Tries ``source.get(key)`` and then ``source[key]``, because yfinance's ``history_metadata``
    is a wrapper object rather than a plain dict (found at Gate 2 of step 1.1).
    """
    for read in (_via_get, _via_item):
        try:
            value = read(source, key)
        except Exception:  # advisory data: any vendor-side failure means "unknown"
            continue
        if value:
            return value
    return None


def _via_get(source: Any, key: str) -> Any:
    return source.get(key)


def _via_item(source: Any, key: str) -> Any:
    return source[key]


def _first(source: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _lookup(source, key)
        if value:
            return value
    return None


def _metadata(ticker_obj: Any) -> tuple[str | None, str | None]:
    """(vendor name, instrument type); ``None`` for what the vendor does not provide.

    First the history metadata (free: the price call already fetched it); only when a field is
    still missing, ``get_info()`` (one more request per ticker). Every vendor-side failure means
    "unknown": the audit reports how many names and types are missing instead of failing.
    """
    name: Any = None
    kind: Any = None
    try:
        meta = ticker_obj.history_metadata
    except Exception:
        meta = None
    if meta is not None:
        name = _first(meta, ("longName", "shortName"))
        kind = _lookup(meta, "instrumentType")
    if not name or not kind:
        try:
            info = ticker_obj.get_info()
        except Exception:
            info = None
        if info is not None:
            name = name or _first(info, ("longName", "shortName"))
            kind = kind or _lookup(info, "quoteType")
    return (str(name) if name else None, str(kind) if kind else None)


def fetch_yfinance(
    ticker: str,
    start: date,
    end: date,
    *,
    attempts: int = 3,
    pause: float = 2.0,
    courtesy_pause: float = 0.3,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchResult:
    """Daily close and volume for ``ticker`` on ``[start, end]`` (both inclusive).

    Raises the vendor's exception after ``attempts`` failed calls (the caller records it).
    An empty answer is returned as ``FetchResult(None, ...)``, not retried: it is the vendor's
    normal reply for an unknown or delisted symbol, but it can also be a rate limit, so a
    surprising number of ``no_data`` rows means "run again later".
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if end < start:
        raise ValueError("end must not be before start")
    yf = importlib.import_module("yfinance")
    sleep(courtesy_pause)
    obj = yf.Ticker(ticker)
    raw = None
    for attempt in range(1, attempts + 1):
        try:
            raw = obj.history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),  # the vendor's end is exclusive
                auto_adjust=False,
                actions=False,
            )
            break
        except Exception:
            if attempt == attempts:
                raise
            sleep(pause * 2 ** (attempt - 1))
    name, kind = _metadata(obj)
    return FetchResult(_normalise_history(raw), name, kind)
