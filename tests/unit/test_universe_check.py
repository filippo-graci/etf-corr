"""Step 1.1: universe audit (pure checks, runner, summary, command). No network, no vendor data.

Histories are synthetic business-day frames with constant price and volume, so every statistic
has a closed form (dollar volume = price x volume). Each threshold is tested on both sides.
"""

import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from etfcorr import config as C
from etfcorr.data import universe_check as U
from etfcorr.data.universe_check import CheckParams, FetchResult

ROOT = Path(__file__).resolve().parents[2]
ASOF = date(2026, 9, 18)  # a Friday
PRICE, VOLUME = 100.0, 200_000.0  # 20M USD per day, above the 10M screen


def params(**kw: object) -> CheckParams:
    base: dict[str, object] = {
        "asof": ASOF,
        "start": date(2014, 1, 1),
        "train_start": date(2016, 1, 1),
        "train_end": date(2020, 12, 31),
        "min_adv_usd": 10_000_000.0,
    }
    base.update(kw)
    return CheckParams(**base)  # type: ignore[arg-type]


def entry(
    ticker: str = "XYZ", panel: str = "core", sector: str = "ai_semis", name: str = "Test ETF"
) -> C.UniverseEntry:
    return C.UniverseEntry(
        ticker=ticker,
        panel=panel,  # type: ignore[arg-type]
        sector=sector,  # type: ignore[arg-type]
        name=name,
    )


def history(
    start: str = "2014-01-02", end: str = "2026-09-18", price: float = PRICE, volume: float = VOLUME
) -> pd.DataFrame:
    index = pd.bdate_range(start, end)
    return pd.DataFrame({"close": price, "volume": volume}, index=index)


def fr(
    h: pd.DataFrame | None, name: str | None = "Vendor Name", qt: str | None = "ETF"
) -> FetchResult:
    return FetchResult(h, name, qt)


def run(h: pd.DataFrame | None, e: C.UniverseEntry | None = None, **kw: object) -> U.TickerCheck:
    return U.check_history(e or entry(), fr(h), params(**kw))


def set_volume(
    h: pd.DataFrame, where: slice | list[int] | np.ndarray, value: float
) -> pd.DataFrame:
    out = h.copy()
    out.iloc[where, out.columns.get_loc("volume")] = value
    return out


# --------------------------------------------------------------------------------------
# CheckParams
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"asof": date(2013, 12, 31)},
        {"asof": date(2014, 1, 1)},
        {"train_start": date(2021, 1, 1)},
        {"min_adv_usd": 0.0},
        {"min_adv_usd": -1.0},
        {"adv_window": 0},
        {"max_stale_days": -1},
        {"max_gap_days": 0},
        {"late_start_tolerance_days": -1},
        {"zero_volume_warn": -0.1},
        {"zero_volume_warn": 1.1},
    ],
)
def test_params_reject_invalid_values(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        params(**bad)


def test_params_from_project_reads_the_committed_configs() -> None:
    project = C.load_project(ROOT)
    p = CheckParams.from_project(project, ASOF)
    assert p.asof == ASOF
    assert p.start == project.universe.start_date
    assert (p.train_start, p.train_end) == (
        project.experiment.splits.train.start,
        project.experiment.splits.train.end,
    )
    assert p.min_adv_usd == float(project.universe.min_adv_usd)


# --------------------------------------------------------------------------------------
# check_history: known answers
# --------------------------------------------------------------------------------------


def test_clean_history_is_ok_and_matches_closed_forms() -> None:
    h = history()
    row = run(h)
    assert row.verdict == "ok"
    assert row.flags == ()
    assert row.first_date == date(2014, 1, 2)
    assert row.last_date == ASOF
    assert row.n_obs == len(h)
    assert row.adv_recent_usd == PRICE * VOLUME
    assert row.adv_train_usd == PRICE * VOLUME
    assert row.zero_volume_share == 0.0
    assert row.max_gap_days == 3  # Friday to Monday
    assert row.vendor_name == "Vendor Name"
    assert row.quote_type == "ETF"
    assert row.config_name == "Test ETF"


def test_single_observation_has_zero_gap() -> None:
    row = run(history(start="2026-09-18", end="2026-09-18"))
    assert row.n_obs == 1
    assert row.max_gap_days == 0


def test_adv_is_a_median_not_a_mean() -> None:
    h = set_volume(history(), [-10], 200_000_000.0)  # one 1000x spike inside the window
    assert run(h).adv_recent_usd == PRICE * VOLUME


def test_adv_uses_only_the_last_window_rows() -> None:
    h = set_volume(history(), slice(-3, None), 0.0)
    h.iloc[-3:, h.columns.get_loc("volume")] = [100_000.0, 300_000.0, 500_000.0]
    assert run(h, adv_window=3).adv_recent_usd == 30_000_000.0  # median of 10M, 30M, 50M
    assert run(h, adv_window=4).adv_recent_usd == 25_000_000.0  # median of 20M, 10M, 30M, 50M
    assert run(h, adv_window=1).adv_recent_usd == 50_000_000.0


def test_adv_window_longer_than_history_uses_all_rows() -> None:
    h = history(start="2026-09-01")
    assert run(h, adv_window=252).adv_recent_usd == PRICE * VOLUME


def test_adv_train_uses_only_the_training_window() -> None:
    h = history()
    in_train = (h.index >= "2016-01-01") & (h.index <= "2020-12-31")
    h.loc[in_train, "volume"] = 10_000.0  # 1M USD/day in the training window only
    row = run(h)
    assert row.adv_train_usd == PRICE * 10_000.0
    assert row.adv_recent_usd == PRICE * VOLUME
    assert row.flags == ("thin_in_train",)
    assert row.verdict == "ok"  # informational only


def test_adv_train_is_nan_without_training_rows() -> None:
    row = run(history(start="2022-01-03"))
    assert np.isnan(row.adv_train_usd)
    assert "thin_in_train" not in row.flags


def test_rows_after_asof_and_before_start_are_ignored() -> None:
    # A wildly different future and past must not change the result (no dependence on the future
    # of the audit date), while moving the audit date forward does (the test can detect a leak).
    base = history(start="2014-01-02", end="2026-09-18")
    future = history(start="2026-09-21", end="2027-09-17", price=1.0, volume=1.0)
    past = history(start="2013-01-02", end="2013-12-31", price=1.0, volume=1.0)
    extended = pd.concat([past, base, future])
    assert run(extended).to_row() == run(base).to_row()
    leaky = run(extended, asof=date(2027, 9, 17))
    assert leaky.to_row() != run(base).to_row()
    assert leaky.last_date == date(2027, 9, 17)


def test_history_only_in_the_future_is_no_data() -> None:
    row = run(history(start="2026-09-21", end="2026-12-31"))
    assert row.flags == ("no_data",)
    assert row.verdict == "fail"


# --------------------------------------------------------------------------------------
# check_history: each flag on both sides of its threshold
# --------------------------------------------------------------------------------------


def test_low_liquidity_boundary_and_severity() -> None:
    just_below = run(history(volume=99_999.0))  # 9,999,900 USD
    exactly = run(history(volume=100_000.0))  # 10,000,000 USD = the screen
    assert "low_liquidity" in just_below.flags
    assert just_below.verdict == "review"
    assert "low_liquidity" not in exactly.flags


def test_liquidity_gate_applies_to_core_and_extended_etfs_only() -> None:
    thin = history(volume=1_000.0)
    assert "low_liquidity" in run(thin, entry(panel="extended")).flags
    control = entry("SPY", panel="control", sector="control")
    assert "low_liquidity" not in run(thin, control).flags
    assert "low_liquidity" not in run(thin, entry("BTC-USD", "extended", "fintech_crypto")).flags
    assert "low_liquidity" not in run(thin, entry("^VIX", "control", "control")).flags


def test_zero_volume_for_an_index_is_informational_and_for_an_etf_a_review() -> None:
    zero = history(volume=0.0)
    vix = run(zero, entry("^VIX", "control", "control"))
    assert vix.flags == ("volume_not_available",)
    assert vix.verdict == "ok"
    etf = run(zero)
    assert etf.flags[0] == "volume_missing"
    assert etf.verdict == "review"
    assert "low_liquidity" not in etf.flags


def test_all_nan_volume_is_missing_volume() -> None:
    h = history()
    h["volume"] = np.nan
    row = run(h)
    assert "volume_missing" in row.flags
    assert np.isnan(row.adv_recent_usd)


def test_nan_volume_days_are_ignored_by_the_median() -> None:
    h = set_volume(history(), slice(-100, None), np.nan)
    assert run(h).adv_recent_usd == PRICE * VOLUME


def recent(n: int = 200) -> pd.DataFrame:
    """The last ``n`` business days up to the audit date (fresh, so never stale)."""
    index = pd.bdate_range(end=ASOF, periods=n)
    return pd.DataFrame({"close": PRICE, "volume": VOLUME}, index=index)


def test_zero_volume_days_share_and_threshold() -> None:
    n = 200
    h = recent(n)
    at = set_volume(h, [0, 100], 0.0)  # 2 of 200 = 1.0 %, exactly the default warning level
    below = set_volume(h, [0], 0.0)  # 0.5 %
    row = run(at, adv_window=n)
    assert "zero_volume_days" in row.flags
    assert "stale" not in row.flags
    assert row.zero_volume_share == 0.01
    assert "zero_volume_days" not in run(below, adv_window=n).flags


def test_zero_volume_days_are_kept_in_the_median() -> None:
    mostly_zero = set_volume(recent(200), slice(0, 120), 0.0)  # 60 % zero-volume sessions
    row = run(mostly_zero, adv_window=200)
    assert row.adv_recent_usd == 0.0
    assert "low_liquidity" in row.flags


@pytest.mark.parametrize(
    ("flags", "verdict"),
    [
        ((), "ok"),
        (("zero_volume_days",), "ok"),
        (("thin_in_train", "late_start_extended", "volume_not_available"), "ok"),
        (("late_start",), "review"),
        (("zero_volume_days", "low_liquidity"), "review"),
        (("stale",), "fail"),
        (("low_liquidity", "no_data"), "fail"),
    ],
)
def test_verdict_is_the_worst_severity(flags: tuple[str, ...], verdict: str) -> None:
    assert U.TickerCheck("T", "core", "ai_semis", flags).verdict == verdict


def test_late_start_boundary_and_panel_dependence() -> None:
    on_time = run(history(start="2016-01-08"))  # 7 days after the train start = tolerance
    late = run(history(start="2016-01-11"))  # 10 days after
    assert "late_start" not in on_time.flags
    assert late.flags == ("late_start",)
    assert late.verdict == "review"
    ext = run(history(start="2016-01-11"), entry(panel="extended"))
    assert ext.flags == ("late_start_extended",)
    assert ext.verdict == "ok"
    ctl = run(history(start="2016-01-11"), entry("SPY", "control", "control"))
    assert "late_start" in ctl.flags


def test_stale_boundary() -> None:
    fresh = run(history(end="2026-09-15"), max_stale_days=3)  # Tuesday: 3 days before asof
    stale = run(history(end="2026-09-14"), max_stale_days=3)  # Monday: 4 days
    assert "stale" not in fresh.flags
    assert "stale" in stale.flags
    assert stale.verdict == "fail"


def _with_gap(first_missing: str, last_missing: str) -> pd.DataFrame:
    h = history()
    return h.drop(h.loc[first_missing:last_missing].index)


def test_gap_boundary() -> None:
    seven = run(_with_gap("2020-06-02", "2020-06-05"))  # Mon 1 June -> Mon 8 June
    eight = run(_with_gap("2020-06-02", "2020-06-08"))  # Mon 1 June -> Tue 9 June
    assert seven.max_gap_days == 7
    assert "gap_over_limit" not in seven.flags
    assert eight.max_gap_days == 8
    assert eight.flags == ("gap_over_limit",)
    assert eight.verdict == "review"


def test_duplicate_or_unsorted_index_is_flagged_and_cleaned() -> None:
    h = history()
    dup = pd.concat([h, h.iloc[[100]]])
    unsorted = h.iloc[::-1]
    shuffled = h.sample(frac=1.0, random_state=0)
    for bad in (dup, unsorted, shuffled):
        row = run(bad)
        assert "index_not_unique_increasing" in row.flags
        assert row.verdict == "fail"
        assert row.n_obs == len(h)  # statistics use the cleaned frame
        assert row.first_date == date(2014, 1, 2)
        assert row.last_date == ASOF
        assert row.adv_recent_usd == PRICE * VOLUME


@pytest.mark.parametrize("bad_price", [0.0, -1.0])
def test_nonpositive_price_fails(bad_price: float) -> None:
    h = history()
    h.iloc[500, h.columns.get_loc("close")] = bad_price
    row = run(h)
    assert "nonpositive_price" in row.flags
    assert row.verdict == "fail"


def test_nan_close_rows_are_dropped() -> None:
    h = history()
    missing = np.arange(10, 100, 9)  # ten isolated sessions, so no calendar gap appears
    assert len(missing) == 10
    h.iloc[missing, h.columns.get_loc("close")] = np.nan
    row = run(h)
    assert row.n_obs == len(h) - 10
    assert row.flags == ()


def test_all_nan_close_is_no_data() -> None:
    h = history()
    h["close"] = np.nan
    assert run(h).flags == ("no_data",)


def test_non_numeric_cells_become_missing() -> None:
    h = history().astype(object)
    h.iloc[5, h.columns.get_loc("close")] = "n/a"
    row = run(h)
    assert row.n_obs == len(h) - 1


def test_no_data_variants() -> None:
    e = entry()
    p = params()
    for result in (
        None,
        FetchResult(None),
        fr(None),
        fr(pd.DataFrame({"close": [], "volume": []})),
    ):
        row = U.check_history(e, result, p)
        assert row.flags == ("no_data",)
        assert row.verdict == "fail"
        assert row.n_obs == 0
        assert row.first_date is None


def test_bad_history_variants() -> None:
    h = history()
    tz_aware = h.copy()
    tz_aware.index = tz_aware.index.tz_localize("America/New_York")
    range_index = h.reset_index(drop=True)
    missing_volume = h[["close"]]
    for bad in (tz_aware, range_index, missing_volume, [1, 2, 3], "text"):
        row = U.check_history(entry(), FetchResult(bad, "n", "ETF"), params())  # type: ignore[arg-type]
        assert row.flags == ("bad_history",), bad
        assert row.verdict == "fail"


def test_fetch_error_is_recorded() -> None:
    row = U.check_history(entry(), None, params(), error="RuntimeError: boom")
    assert row.flags == ("fetch_error",)
    assert row.error == "RuntimeError: boom"
    assert row.verdict == "fail"


def test_not_an_etf_uses_the_vendor_quote_type() -> None:
    e = entry()
    p = params()
    h = history()
    assert "not_an_etf" in U.check_history(e, fr(h, qt="EQUITY"), p).flags
    assert "not_an_etf" not in U.check_history(e, fr(h, qt="etf"), p).flags
    assert "not_an_etf" not in U.check_history(e, fr(h, qt=None), p).flags
    idx = entry("^VIX", "control", "control")
    assert "not_an_etf" not in U.check_history(idx, fr(h, qt="INDEX"), p).flags


def test_flags_combine_and_the_worst_severity_decides() -> None:
    h = history(start="2017-03-01", volume=1_000.0, end="2026-09-14")
    row = run(h, max_stale_days=3)
    assert {"late_start", "low_liquidity", "thin_in_train", "stale"} <= set(row.flags)
    assert row.verdict == "fail"


def test_every_flag_has_a_known_severity() -> None:
    assert set(U.FLAG_SEVERITY.values()) <= {"blocking", "review", "info"}


def test_row_has_the_documented_columns_in_order() -> None:
    row = run(history()).to_row()
    assert tuple(row) == U.COLUMNS


def test_is_etf_like() -> None:
    assert U.is_etf_like("SPY")
    assert not U.is_etf_like("^VIX")
    assert not U.is_etf_like("BTC-USD")


# --------------------------------------------------------------------------------------
# run_universe_check
# --------------------------------------------------------------------------------------

UNIVERSE_DICT = {
    "version": 1,
    "start_date": "2014-01-01",
    "min_adv_usd": 10_000_000,
    "tickers": [
        {"ticker": "AAA", "sector": "ai_semis", "panel": "core", "name": "Alpha"},
        {"ticker": "BBB", "sector": "ai_semis", "panel": "core", "name": "Bravo"},
        {"ticker": "CCC", "sector": "healthcare", "panel": "core", "name": "Charlie"},
        {"ticker": "DDD", "sector": "healthcare", "panel": "core", "name": "Delta"},
        {"ticker": "EEE", "sector": "defense", "panel": "extended", "name": "Echo"},
        {"ticker": "SPY", "sector": "control", "panel": "control", "name": "S&P"},
    ],
}


def small_universe() -> C.UniverseConfig:
    return C.UniverseConfig.model_validate(UNIVERSE_DICT)


def fake_fetcher(calls: list[tuple[str, date, date]] | None = None) -> U.Fetcher:
    table: dict[str, FetchResult] = {
        "AAA": fr(history(start="2014-01-02"), "Alpha ETF"),
        "BBB": fr(history(start="2016-01-04")),  # inside the tolerance of the train start
        "CCC": fr(history(start="2019-02-06")),
        "DDD": fr(history(start="2018-05-01")),
        "SPY": fr(history(start="2014-01-02")),
        # EEE deliberately absent -> no data
    }

    def fetch(ticker: str, start: date, end: date) -> FetchResult:
        if calls is not None:
            calls.append((ticker, start, end))
        return table.get(ticker, FetchResult(None))

    return fetch


def test_runner_returns_one_row_per_ticker_in_file_order() -> None:
    calls: list[tuple[str, date, date]] = []
    out = U.run_universe_check(small_universe(), fake_fetcher(calls), params())
    assert list(out["ticker"]) == ["AAA", "BBB", "CCC", "DDD", "EEE", "SPY"]
    assert tuple(out.columns) == U.COLUMNS
    assert list(out["verdict"]) == ["ok", "ok", "review", "review", "fail", "ok"]
    assert calls == [
        (t, date(2014, 1, 1), ASOF) for t in ["AAA", "BBB", "CCC", "DDD", "EEE", "SPY"]
    ]
    assert str(out["first_date"].dtype).startswith("datetime64")
    assert pd.isna(out.loc[out["ticker"] == "EEE", "first_date"]).all()
    assert out.loc[out["ticker"] == "AAA", "vendor_name"].item() == "Alpha ETF"


def test_runner_records_fetch_errors_and_continues() -> None:
    inner = fake_fetcher()

    def flaky(ticker: str, start: date, end: date) -> FetchResult:
        if ticker == "BBB":
            raise ConnectionError("rate limited " + "x" * 1000)
        return inner(ticker, start, end)

    out = U.run_universe_check(small_universe(), flaky, params()).set_index("ticker")
    assert out.loc["BBB", "flags"] == "fetch_error"
    assert out.loc["BBB", "error"].startswith("ConnectionError: rate limited")
    assert len(out.loc["BBB", "error"]) <= 300
    assert out.loc["AAA", "verdict"] == "ok"
    assert out.loc["SPY", "verdict"] == "ok"


def test_runner_lets_keyboard_interrupt_through() -> None:
    def interrupted(ticker: str, start: date, end: date) -> FetchResult:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        U.run_universe_check(small_universe(), interrupted, params())


def test_runner_reports_progress() -> None:
    seen: list[tuple[int, int, str]] = []
    U.run_universe_check(
        small_universe(), fake_fetcher(), params(), progress=lambda i, n, t: seen.append((i, n, t))
    )
    assert seen[0] == (1, 6, "AAA")
    assert seen[-1] == (6, 6, "SPY")
    assert len(seen) == 6


# --------------------------------------------------------------------------------------
# Panel summary
# --------------------------------------------------------------------------------------


def test_common_sample_start() -> None:
    s = pd.Series(pd.to_datetime(["2016-01-04", "2019-02-06", None]), index=["a", "b", "c"])
    assert U.common_sample_start(s) == pd.Timestamp("2019-02-06")
    assert U.common_sample_start(pd.Series(pd.to_datetime([None, None]))) is None
    assert U.common_sample_start(pd.Series([], dtype="datetime64[ns]")) is None


def test_start_if_dropped() -> None:
    s = pd.Series(pd.to_datetime(["2016-01-04", "2018-05-01", "2019-02-06"]), index=list("abc"))
    out = U.start_if_dropped(s)
    assert out["c"] == pd.Timestamp("2018-05-01")  # unique latest: dropping it moves the start
    assert out["b"] == pd.Timestamp("2019-02-06")  # not binding: start unchanged
    assert out["a"] == pd.Timestamp("2019-02-06")
    tie = pd.Series(pd.to_datetime(["2019-02-06", "2019-02-06"]), index=["x", "y"])
    assert U.start_if_dropped(tie) == {
        "x": pd.Timestamp("2019-02-06"),
        "y": pd.Timestamp("2019-02-06"),
    }
    assert U.start_if_dropped(pd.Series(pd.to_datetime(["2019-02-06"]), index=["x"])) == {"x": None}
    assert U.start_if_dropped(pd.Series([], dtype="datetime64[ns]")) == {}


def test_summary_lines_report_the_binding_tickers() -> None:
    out = U.run_universe_check(small_universe(), fake_fetcher(), params())
    text = "\n".join(U.summary_lines(out, params()))
    assert "core 4, extended 1, control 1, total 6" in text
    assert "verdicts: ok 3, review 2, fail 1" in text
    assert "core: 4 usable, common sample starts 2019-02-06 (2 of 4 cover the train start" in text
    assert "CCC       first 2019-02-06; common start if dropped: 2018-05-01" in text
    assert "DDD       first 2018-05-01; common start if dropped: 2019-02-06" in text
    assert "extended: no usable member" in text
    assert "EEE       extended no_data" in text
    assert "CCC       core     late_start" in text


def test_summary_reports_vendor_metadata_coverage_and_warns_when_incomplete() -> None:
    out = U.run_universe_check(small_universe(), fake_fetcher(), params())
    text = "\n".join(U.summary_lines(out, params()))
    assert "vendor metadata: name for 5 of 5 tickers with data, instrument type for 5 of 5" in text
    assert "WARNING" not in text  # the ticker without data does not count against coverage

    inner = fake_fetcher()

    def no_metadata(ticker: str, start: date, end: date) -> FetchResult:
        return FetchResult(inner(ticker, start, end).history)  # names and types dropped

    bare = U.run_universe_check(small_universe(), no_metadata, params())
    text = "\n".join(U.summary_lines(bare, params()))
    assert "name for 0 of 5 tickers with data, instrument type for 0 of 5" in text
    assert "WARNING: vendor metadata is incomplete" in text

    def only_names(ticker: str, start: date, end: date) -> FetchResult:
        return FetchResult(inner(ticker, start, end).history, "Some Name", None)

    half = U.run_universe_check(small_universe(), only_names, params())
    text = "\n".join(U.summary_lines(half, params()))
    assert "name for 5 of 5 tickers with data, instrument type for 0 of 5" in text
    assert "WARNING" in text


def test_summary_excludes_failed_tickers_from_the_common_start() -> None:
    inner = fake_fetcher()

    def stale_ccc(ticker: str, start: date, end: date) -> FetchResult:
        if ticker == "CCC":
            return fr(history(start="2019-02-06", end="2026-06-01"))  # stopped updating
        return inner(ticker, start, end)

    out = U.run_universe_check(small_universe(), stale_ccc, params())
    text = "\n".join(U.summary_lines(out, params()))
    assert "core: 3 usable, common sample starts 2018-05-01" in text
    assert "CCC       core     stale;late_start" in text


# --------------------------------------------------------------------------------------
# Command implementation (typer-free)
# --------------------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path: Path) -> C.Project:
    """Committed configs (lock included) with the small universe; nothing else is touched."""
    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "config" / "universe.yaml").write_text(yaml.safe_dump(UNIVERSE_DICT))
    return C.load_project(tmp_path)


def test_command_writes_the_csv_and_returns_the_report(tmp_project: C.Project) -> None:
    lines = U.command_check_universe(tmp_project, asof=ASOF, fetch=fake_fetcher())
    target = tmp_project.paths.root / "data" / "interim" / "universe_check.csv"
    assert target.is_file()
    assert lines[-1] == f"  written: {target}"
    back = pd.read_csv(target)
    assert tuple(back.columns) == U.COLUMNS
    assert list(back["ticker"]) == ["AAA", "BBB", "CCC", "DDD", "EEE", "SPY"]
    assert back.loc[back["ticker"] == "AAA", "first_date"].item() == "2014-01-02"
    assert any("common sample starts 2019-02-06" in line for line in lines)


def test_command_output_path_is_restricted(tmp_project: C.Project) -> None:
    ok = U.command_check_universe(
        tmp_project, asof=ASOF, fetch=fake_fetcher(), out="reports/tables/check.csv"
    )
    assert (tmp_project.paths.root / "reports" / "tables" / "check.csv").is_file()
    assert ok
    universe_yaml = tmp_project.paths.root / "config" / "universe.yaml"
    before = universe_yaml.read_text()
    for bad in ("config/universe.yaml", "../evil.csv", "/tmp/evil.csv", "src/x.csv", "data"):
        calls: list[tuple[str, date, date]] = []
        with pytest.raises(C.ConfigError):
            U.command_check_universe(tmp_project, asof=ASOF, fetch=fake_fetcher(calls), out=bad)
        assert calls == []  # refused before any download
    assert universe_yaml.read_text() == before  # a refused write leaves the configuration intact


def test_command_rejects_an_audit_date_before_the_download_start(tmp_project: C.Project) -> None:
    calls: list[tuple[str, date, date]] = []
    with pytest.raises(C.ConfigError, match="invalid audit parameters"):
        U.command_check_universe(tmp_project, asof=date(2013, 1, 1), fetch=fake_fetcher(calls))
    assert calls == []


def test_parse_iso_date() -> None:
    assert U.parse_iso_date("2026-09-18") == ASOF
    for bad in ("18/09/2026", "2026-13-01", "", "yesterday"):
        with pytest.raises(C.ConfigError):
            U.parse_iso_date(bad)


def test_committed_universe_is_consistent_with_the_check() -> None:
    """The committed universe: valid, unique, and every panel non-empty (panel rules: 1.1b file)."""
    uni = C.load_universe(ROOT / "config" / "universe.yaml")
    assert len(uni.tickers_in("core")) >= 20
    assert len(uni.tickers_in("extended")) > 0
    assert {"SPY", "QQQ", "TLT", "GLD", "^VIX"} <= set(uni.tickers_in("control"))
    futures = {t.ticker for t in uni.tickers if t.futures_based}
    assert {"USO", "UNG", "BITO"} <= futures  # the futures-based funds named in v2 section 4
