"""yfinance adapter (step 1.1) against a stand-in ``yfinance`` module. No network.

The stand-in encodes the reading of the vendor API stated in ``etfcorr.data.sources``; these tests
therefore prove the adapter's own logic (normalisation, retry, argument mapping), not that the
real library behaves like the stand-in. The real call is the Gate 2 test.
"""

import sys
import types
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from etfcorr import config as C
from etfcorr.data import sources as S
from etfcorr.data.universe_check import CheckParams, FetchResult, check_history

START, END = date(2024, 1, 1), date(2024, 1, 10)


def vendor_frame(n: int = 3, tz: str | None = "America/New_York") -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=n, freq="B", tz=tz)
    return pd.DataFrame(
        {
            "Open": np.arange(n) + 9.0,
            "Close": np.arange(n) + 10.0,
            "Adj Close": np.arange(n) + 9.5,
            "Volume": (np.arange(n) + 1) * 100,
        },
        index=index,
    )


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.sleeps: list[float] = []


def install(
    monkeypatch: pytest.MonkeyPatch,
    frame: pd.DataFrame | None,
    *,
    meta: Any = None,
    meta_raises: bool = False,
    info: Any = None,
    info_raises: bool = False,
    failures: int = 0,
) -> Recorder:
    rec = Recorder()
    state = {"failures": failures}

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            rec.calls.append(("Ticker", symbol))

        def history(self, **kwargs: Any) -> pd.DataFrame | None:
            rec.calls.append(("history", kwargs))
            if state["failures"] > 0:
                state["failures"] -= 1
                raise ConnectionError("429 Too Many Requests")
            return frame

        @property
        def history_metadata(self) -> Any:
            if meta_raises:
                raise AttributeError("no metadata in this version")
            return meta

        def get_info(self) -> Any:
            rec.calls.append(("get_info", None))
            if info_raises:
                raise RuntimeError("401 Unauthorized")
            return info

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
    return rec


def fetch(rec: Recorder, **kw: Any) -> FetchResult:
    return S.fetch_yfinance("SMH", START, END, sleep=rec.sleeps.append, **kw)


def test_normalisation_gives_a_naive_day_index_and_three_float_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = install(monkeypatch, vendor_frame())
    result = fetch(rec)
    h = result.history
    assert h is not None
    assert list(h.columns) == ["close", "volume", "adj_close"]
    assert isinstance(h.index, pd.DatetimeIndex)
    assert h.index.tz is None
    assert list(h.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert (h.index == h.index.normalize()).all()
    assert list(h["close"]) == [10.0, 11.0, 12.0]  # the unadjusted Close, not Adj Close
    assert list(h["volume"]) == [100.0, 200.0, 300.0]
    assert list(h["adj_close"]) == [9.5, 10.5, 11.5]  # total-return series (step 1.2)
    assert h["close"].dtype == float
    assert h["volume"].dtype == float
    assert h["adj_close"].dtype == float


def test_a_utc_midnight_index_keeps_its_calendar_date(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = install(monkeypatch, vendor_frame(tz="UTC"))
    h = fetch(rec).history
    assert h is not None
    assert list(h.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-04"]


def test_timestamps_with_a_time_of_day_are_reduced_to_the_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = vendor_frame()
    raw.index = raw.index + pd.Timedelta(hours=16)  # e.g. a 16:00 close stamp
    h = fetch(install(monkeypatch, raw)).history
    assert h is not None
    assert list(h.index.strftime("%Y-%m-%d %H:%M")) == [
        "2024-01-02 00:00",
        "2024-01-03 00:00",
        "2024-01-04 00:00",
    ]


def test_a_naive_vendor_index_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = install(monkeypatch, vendor_frame(tz=None))
    h = fetch(rec).history
    assert h is not None and h.index.tz is None


def test_vendor_call_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = install(monkeypatch, vendor_frame())
    fetch(rec)
    assert rec.calls[0] == ("Ticker", "SMH")
    kind, kwargs = rec.calls[1]
    assert kind == "history"
    assert kwargs == {
        "start": "2024-01-01",
        "end": "2024-01-11",  # the vendor's end is exclusive, ours inclusive
        "auto_adjust": False,
        "actions": False,
    }
    assert rec.sleeps == [0.3]  # one courtesy pause before the request


def test_metadata_is_read_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    meta = {
        "longName": "VanEck Semiconductor ETF",
        "shortName": "VanEck Semi",
        "instrumentType": "ETF",
    }
    result = fetch(install(monkeypatch, vendor_frame(), meta=meta))
    assert (result.vendor_name, result.quote_type) == ("VanEck Semiconductor ETF", "ETF")


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        ({"shortName": "Short"}, ("Short", None)),
        ({"longName": "", "shortName": "Short", "instrumentType": "ETF"}, ("Short", "ETF")),
        ({}, (None, None)),
        (None, (None, None)),
        ("not a dict", (None, None)),
        ([("longName", "x")], (None, None)),
    ],
)
def test_metadata_fallbacks(
    monkeypatch: pytest.MonkeyPatch, meta: Any, expected: tuple[str | None, str | None]
) -> None:
    result = fetch(install(monkeypatch, vendor_frame(), meta=meta))
    assert (result.vendor_name, result.quote_type) == expected
    assert result.history is not None  # metadata problems never lose the prices


def test_metadata_exceptions_are_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    result = fetch(install(monkeypatch, vendor_frame(), meta_raises=True))
    assert (result.vendor_name, result.quote_type) == (None, None)
    assert result.history is not None


class GetOnlyMeta:
    """Dict-like wrapper with ``get`` but no item access."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def get(self, key: str) -> Any:
        return self._data.get(key)


class ItemOnlyMeta:
    """Dict-like wrapper with item access (KeyError when absent) but no ``get``."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]


@pytest.mark.parametrize("wrapper", [GetOnlyMeta, ItemOnlyMeta])
def test_metadata_wrapper_objects_are_read_like_dicts(
    monkeypatch: pytest.MonkeyPatch, wrapper: type
) -> None:
    """yfinance's ``history_metadata`` is a wrapper, not a dict (found at Gate 2)."""
    meta = wrapper({"longName": "VanEck Semiconductor ETF", "instrumentType": "ETF"})
    rec = install(monkeypatch, vendor_frame(), meta=meta)
    result = fetch(rec)
    assert (result.vendor_name, result.quote_type) == ("VanEck Semiconductor ETF", "ETF")
    assert ("get_info", None) not in rec.calls  # complete metadata: no extra request


def test_a_wrapper_with_missing_keys_falls_back_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = fetch(install(monkeypatch, vendor_frame(), meta=ItemOnlyMeta({})))
    assert (result.vendor_name, result.quote_type) == (None, None)


def test_complete_history_metadata_does_not_trigger_the_info_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = {"longName": "A", "instrumentType": "ETF"}
    rec = install(monkeypatch, vendor_frame(), meta=meta, info={"longName": "B", "quoteType": "X"})
    result = fetch(rec)
    assert (result.vendor_name, result.quote_type) == ("A", "ETF")
    assert ("get_info", None) not in rec.calls


def test_info_fills_what_the_history_metadata_lacks(monkeypatch: pytest.MonkeyPatch) -> None:
    info = {"longName": "From Info", "shortName": "Info", "quoteType": "ETF"}
    rec = install(monkeypatch, vendor_frame(), meta={}, info=info)
    result = fetch(rec)
    assert (result.vendor_name, result.quote_type) == ("From Info", "ETF")
    assert rec.calls.count(("get_info", None)) == 1
    # a field that the history metadata does provide is not overwritten by info
    partial = install(monkeypatch, vendor_frame(), meta={"longName": "From Meta"}, info=info)
    result = fetch(partial)
    assert (result.vendor_name, result.quote_type) == ("From Meta", "ETF")


def test_info_failures_are_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = install(monkeypatch, vendor_frame(), meta={}, info_raises=True)
    result = fetch(rec)
    assert (result.vendor_name, result.quote_type) == (None, None)
    assert result.history is not None
    assert ("get_info", None) in rec.calls  # it was tried


@pytest.mark.parametrize("empty", [None, pd.DataFrame()])
def test_an_empty_answer_is_returned_as_no_history_without_retry(
    monkeypatch: pytest.MonkeyPatch, empty: pd.DataFrame | None
) -> None:
    rec = install(monkeypatch, empty, meta={"longName": "Gone Fund"})
    result = fetch(rec)
    assert result.history is None
    assert result.vendor_name == "Gone Fund"
    assert [c for c in rec.calls if c[0] == "history"].__len__() == 1


def test_transient_failures_are_retried_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = install(monkeypatch, vendor_frame(), failures=2)
    result = fetch(rec)
    assert result.history is not None
    assert len([c for c in rec.calls if c[0] == "history"]) == 3
    assert rec.sleeps == [0.3, 2.0, 4.0]


def test_persistent_failure_raises_after_the_last_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = install(monkeypatch, vendor_frame(), failures=99)
    with pytest.raises(ConnectionError, match="429"):
        fetch(rec, attempts=3)
    assert len([c for c in rec.calls if c[0] == "history"]) == 3
    assert rec.sleeps == [0.3, 2.0, 4.0]  # no sleep after the final failure


def test_single_attempt_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = install(monkeypatch, vendor_frame(), failures=1)
    with pytest.raises(ConnectionError):
        fetch(rec, attempts=1)
    assert len([c for c in rec.calls if c[0] == "history"]) == 1


def test_invalid_arguments_fail_before_touching_the_vendor() -> None:
    with pytest.raises(ValueError, match="attempts"):
        S.fetch_yfinance("SMH", START, END, attempts=0)
    with pytest.raises(ValueError, match="end"):
        S.fetch_yfinance("SMH", END, START)


def test_a_missing_vendor_library_is_an_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "yfinance", None)  # makes `import yfinance` fail
    with pytest.raises(ImportError):
        S.fetch_yfinance("SMH", START, END, sleep=lambda s: None)


def test_normalise_history_edge_cases() -> None:
    assert S._normalise_history(None) is None
    assert S._normalise_history(pd.DataFrame()) is None
    raw = vendor_frame()
    raw.iloc[1, raw.columns.get_loc("Close")] = np.nan
    h = S._normalise_history(raw)
    assert h is not None and np.isnan(h["close"].iloc[1])  # NaNs are kept; the audit drops them


def test_adj_close_is_nan_when_the_vendor_frame_lacks_the_column() -> None:
    raw = vendor_frame().drop(columns=["Adj Close"])
    h = S._normalise_history(raw)
    assert h is not None
    assert list(h.columns) == ["close", "volume", "adj_close"]
    assert h["adj_close"].isna().all()
    assert list(h["close"]) == [10.0, 11.0, 12.0]  # close/volume unaffected


def test_adj_close_nans_from_the_vendor_are_kept() -> None:
    raw = vendor_frame()
    raw.iloc[0, raw.columns.get_loc("Adj Close")] = np.nan
    h = S._normalise_history(raw)
    assert h is not None and np.isnan(h["adj_close"].iloc[0])
    assert list(h["adj_close"].iloc[1:]) == [10.5, 11.5]


def test_adapter_output_passes_the_audit_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long tz-aware vendor frame must arrive at the audit as a valid, flag-free history."""
    index = pd.bdate_range("2014-01-02", "2026-09-18", tz="America/New_York")
    raw = pd.DataFrame({"Close": 100.0, "Volume": 200_000.0}, index=index)
    rec = install(monkeypatch, raw, meta={"longName": "Fund", "instrumentType": "ETF"})
    result = S.fetch_yfinance("SMH", date(2014, 1, 1), date(2026, 9, 18), sleep=rec.sleeps.append)
    entry = C.UniverseEntry(ticker="SMH", sector="ai_semis", panel="core", name="Fund")
    params = CheckParams(
        asof=date(2026, 9, 18),
        start=date(2014, 1, 1),
        train_start=date(2016, 1, 1),
        train_end=date(2020, 12, 31),
        min_adv_usd=10_000_000.0,
    )
    row = check_history(entry, result, params)
    assert row.flags == ()
    assert row.first_date == date(2014, 1, 2)
    assert row.adv_recent_usd == 20_000_000.0
    # ... and the audit really does reject a tz-aware frame (the bug this adapter prevents)
    assert check_history(entry, FetchResult(raw.rename(columns=str.lower)), params).flags == (
        "bad_history",
    )
