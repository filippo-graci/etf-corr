"""``etfcorr.data.holdings`` (step 2.4). No network: SEC calls are always an injected stand-in.

Same ``repo``/``project`` fixture pattern as ``test_download.py``: a small custom ``universe.yaml``
copied next to the real, committed ``experiment.yaml``/``hypotheses.lock``.
"""

import shutil
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml

from etfcorr import config as C
from etfcorr.data import holdings as H

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "config", tmp_path / "config")
    universe = {
        "version": 1,
        "start_date": "2014-01-01",
        "min_adv_usd": 10_000_000,
        "tickers": [
            {"ticker": "AAA", "sector": "ai_semis", "panel": "core", "name": "Alpha ETF"},
            {"ticker": "BBB", "sector": "ai_semis", "panel": "core", "name": "Beta ETF"},
            # a deliberate ticker-prefix collision (AAA vs AAAX): _cached_snapshots must not mix
            # AAA's files into AAAX's glob or vice versa.
            {"ticker": "AAAX", "sector": "ai_semis", "panel": "extended", "name": "Alpha-X ETF"},
            {"ticker": "GDX", "sector": "metals_mining", "panel": "core", "name": "Gold Miners"},
            {"ticker": "GDXJ", "sector": "metals_mining", "panel": "core", "name": "Junior Gold"},
            {"ticker": "SPY", "sector": "control", "panel": "control", "name": "S&P"},
            {"ticker": "^VIX", "sector": "control", "panel": "control", "name": "VIX"},
            {
                "ticker": "BTC-USD",
                "sector": "fintech_crypto",
                "panel": "extended",
                "name": "Bitcoin",
            },
        ],
    }
    (tmp_path / "config" / "universe.yaml").write_text(yaml.safe_dump(universe))
    return tmp_path


@pytest.fixture
def project(repo: Path) -> C.Project:
    return C.load_project(repo)


def hold(constituent: str, name: str, weight: float) -> H.Holding:
    return H.Holding(constituent=constituent, name=name, weight=weight)


def snapshot(
    ticker: str,
    asof: date,
    filing_date: date,
    holdings: tuple[H.Holding, ...],
    source: str = "sec_nport",
) -> H.HoldingsSnapshot:
    return H.HoldingsSnapshot(
        ticker=ticker, asof=asof, filing_date=filing_date, source=source, holdings=holdings
    )


# --------------------------------------------------------------------------------------
# Holding / HoldingsSnapshot
# --------------------------------------------------------------------------------------


def test_weights_sums_duplicate_constituents() -> None:
    snap = snapshot(
        "AAA",
        date(2024, 3, 31),
        date(2024, 5, 15),
        (hold("111111111", "X Corp", 0.05), hold("111111111", "X Corp (lot 2)", 0.02)),
    )
    assert snap.weights() == {"111111111": pytest.approx(0.07)}


# --------------------------------------------------------------------------------------
# weighted_jaccard
# --------------------------------------------------------------------------------------


def test_weighted_jaccard_identical_maps_is_one() -> None:
    a = {"X": 0.3, "Y": 0.2, "Z": 0.1}
    assert H.weighted_jaccard(a, dict(a)) == pytest.approx(1.0)


def test_weighted_jaccard_disjoint_is_zero() -> None:
    assert H.weighted_jaccard({"X": 0.5}, {"Y": 0.5}) == pytest.approx(0.0)


def test_weighted_jaccard_known_value() -> None:
    # union {X,Y,Z}; min sums: X min(0.4,0.1)=0.1, Y min(0.3,0.3)=0.3, Z min(0,0.2)=0 -> num=0.4
    # max sums: X max(0.4,0.1)=0.4, Y 0.3, Z 0.2 -> den=0.9 ; 0.4/0.9
    a = {"X": 0.4, "Y": 0.3}
    b = {"X": 0.1, "Y": 0.3, "Z": 0.2}
    assert H.weighted_jaccard(a, b) == pytest.approx(0.4 / 0.9)


def test_weighted_jaccard_is_symmetric() -> None:
    a = {"X": 0.4, "Y": 0.1}
    b = {"X": 0.1, "Y": 0.3, "Z": 0.2}
    assert H.weighted_jaccard(a, b) == pytest.approx(H.weighted_jaccard(b, a))


def test_weighted_jaccard_both_empty_is_nan() -> None:
    assert pd.isna(H.weighted_jaccard({}, {}))


def test_weighted_jaccard_both_zero_weight_is_nan() -> None:
    assert pd.isna(H.weighted_jaccard({"X": 0.0}, {"X": 0.0}))


def test_weighted_jaccard_rejects_negative_weight() -> None:
    with pytest.raises(H.HoldingsError):
        H.weighted_jaccard({"X": -0.1}, {"X": 0.1})


# --------------------------------------------------------------------------------------
# overlap_matrix / validate_overlap_matrix
# --------------------------------------------------------------------------------------


def _two_snapshots() -> dict[str, H.HoldingsSnapshot]:
    return {
        "AAA": snapshot(
            "AAA",
            date(2024, 3, 31),
            date(2024, 5, 15),
            (hold("111", "X", 0.4), hold("222", "Y", 0.3)),
        ),
        "BBB": snapshot(
            "BBB",
            date(2024, 3, 31),
            date(2024, 5, 20),
            (hold("111", "X", 0.1), hold("333", "Z", 0.2)),
        ),
    }


def test_overlap_matrix_needs_at_least_two() -> None:
    with pytest.raises(H.HoldingsError):
        H.overlap_matrix({"AAA": _two_snapshots()["AAA"]})


def test_overlap_matrix_symmetric_unit_diagonal_and_valid() -> None:
    matrix = H.overlap_matrix(_two_snapshots())
    assert list(matrix.index) == ["AAA", "BBB"]
    assert list(matrix.columns) == ["AAA", "BBB"]
    assert matrix.loc["AAA", "AAA"] == pytest.approx(1.0)
    assert matrix.loc["BBB", "BBB"] == pytest.approx(1.0)
    assert matrix.loc["AAA", "BBB"] == pytest.approx(matrix.loc["BBB", "AAA"])
    expected = H.weighted_jaccard({"111": 0.4, "222": 0.3}, {"111": 0.1, "333": 0.2})
    assert matrix.loc["AAA", "BBB"] == pytest.approx(expected)
    H.validate_overlap_matrix(matrix)  # must not raise


def test_overlap_matrix_deterministic_ticker_order() -> None:
    snaps = _two_snapshots()
    reordered = {"BBB": snaps["BBB"], "AAA": snaps["AAA"]}
    assert list(H.overlap_matrix(reordered).index) == ["AAA", "BBB"]


def test_validate_overlap_matrix_rejects_asymmetry() -> None:
    bad = pd.DataFrame([[1.0, 0.5], [0.9, 1.0]], index=["A", "B"], columns=["A", "B"])
    with pytest.raises(H.HoldingsError):
        H.validate_overlap_matrix(bad)


def test_validate_overlap_matrix_rejects_bad_diagonal() -> None:
    bad = pd.DataFrame([[0.9, 0.2], [0.2, 1.0]], index=["A", "B"], columns=["A", "B"])
    with pytest.raises(H.HoldingsError):
        H.validate_overlap_matrix(bad)


def test_validate_overlap_matrix_rejects_out_of_range() -> None:
    bad = pd.DataFrame([[1.0, 1.5], [1.5, 1.0]], index=["A", "B"], columns=["A", "B"])
    with pytest.raises(H.HoldingsError):
        H.validate_overlap_matrix(bad)


def test_validate_overlap_matrix_rejects_nonfinite() -> None:
    bad = pd.DataFrame(
        [[1.0, float("nan")], [float("nan"), 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    with pytest.raises(H.HoldingsError):
        H.validate_overlap_matrix(bad)


# --------------------------------------------------------------------------------------
# latest_asof (point-in-time selection)
# --------------------------------------------------------------------------------------


def test_latest_asof_none_when_nothing_eligible() -> None:
    snaps = [snapshot("AAA", date(2024, 3, 31), date(2024, 5, 15), (hold("1", "X", 0.1),))]
    assert H.latest_asof(snaps, date(2024, 1, 1)) is None


def test_latest_asof_picks_latest_eligible_filing() -> None:
    older = snapshot("AAA", date(2023, 12, 31), date(2024, 2, 10), (hold("1", "X", 0.1),))
    newer = snapshot("AAA", date(2024, 3, 31), date(2024, 5, 15), (hold("1", "X", 0.2),))
    snaps = [older, newer]
    assert H.latest_asof(snaps, date(2024, 6, 1)) is newer
    assert H.latest_asof(snaps, date(2024, 3, 1)) is older


def test_latest_asof_empty_input() -> None:
    assert H.latest_asof([], date(2024, 1, 1)) is None


# --------------------------------------------------------------------------------------
# CSV snapshot storage
# --------------------------------------------------------------------------------------


def test_snapshot_path_format(tmp_path: Path) -> None:
    p = H.snapshot_path(tmp_path, "aaa", date(2024, 3, 31))
    assert p == tmp_path / "AAA_2024-03-31.csv"


def test_write_read_snapshot_round_trip(tmp_path: Path) -> None:
    snap = snapshot(
        "AAA",
        date(2024, 3, 31),
        date(2024, 5, 15),
        (hold("111111111", "X Corp", 0.081), hold("NORMALIZED NAME", "Y Co", 0.02)),
    )
    path = H.snapshot_path(tmp_path, snap.ticker, snap.asof)
    H.write_snapshot_csv(snap, path)
    got = H.read_snapshot_csv(path)
    assert got.ticker == "AAA"
    assert got.asof == date(2024, 3, 31)
    assert got.filing_date == date(2024, 5, 15)
    assert got.source == "sec_nport"
    assert got.weights() == pytest.approx(snap.weights())


def test_write_snapshot_rejects_empty_holdings(tmp_path: Path) -> None:
    empty = snapshot("AAA", date(2024, 3, 31), date(2024, 5, 15), ())
    with pytest.raises(H.HoldingsError):
        H.write_snapshot_csv(empty, tmp_path / "AAA_2024-03-31.csv")


def test_read_snapshot_rejects_wrong_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(H.HoldingsError):
        H.read_snapshot_csv(path)


def test_read_snapshot_rejects_header_only_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(",".join(H.SNAPSHOT_COLUMNS) + "\n")
    with pytest.raises(H.HoldingsError):
        H.read_snapshot_csv(path)


def test_read_snapshot_rejects_malformed_weight(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    header = ",".join(H.SNAPSHOT_COLUMNS)
    path.write_text(f"{header}\nAAA,2024-03-31,2024-05-15,sec_nport,111,X,not-a-number\n")
    with pytest.raises(H.HoldingsError):
        H.read_snapshot_csv(path)


def test_cached_snapshots_ticker_prefix_no_collision(tmp_path: Path) -> None:
    aaa = snapshot("AAA", date(2024, 3, 31), date(2024, 5, 15), (hold("1", "X", 0.1),))
    aaax = snapshot("AAAX", date(2024, 3, 31), date(2024, 5, 15), (hold("2", "Y", 0.2),))
    H.write_snapshot_csv(aaa, H.snapshot_path(tmp_path, "AAA", aaa.asof))
    H.write_snapshot_csv(aaax, H.snapshot_path(tmp_path, "AAAX", aaax.asof))
    got_aaa = H._cached_snapshots(tmp_path, "AAA")
    got_aaax = H._cached_snapshots(tmp_path, "AAAX")
    assert [s.ticker for s in got_aaa] == ["AAA"]
    assert [s.ticker for s in got_aaax] == ["AAAX"]


def test_cached_snapshots_empty_dir_is_empty_list(tmp_path: Path) -> None:
    assert H._cached_snapshots(tmp_path / "does_not_exist", "AAA") == []


def test_cached_snapshots_propagates_malformed_file(tmp_path: Path) -> None:
    (tmp_path / "AAA_2024-03-31.csv").write_text("bad,columns\n1,2\n")
    with pytest.raises(H.HoldingsError):
        H._cached_snapshots(tmp_path, "AAA")


# --------------------------------------------------------------------------------------
# eligible_tickers
# --------------------------------------------------------------------------------------


def test_eligible_tickers_excludes_control_index_and_crypto(project: C.Project) -> None:
    assert H.eligible_tickers(project) == ("AAA", "BBB", "AAAX", "GDX", "GDXJ")


# --------------------------------------------------------------------------------------
# fetch_holdings_universe orchestration (injected fetcher; no network, no real Parquet)
# --------------------------------------------------------------------------------------


def _write_overlap_pickle(matrix: pd.DataFrame, path: Path) -> None:
    matrix.to_pickle(path)


def _fetcher_from(
    data: dict[str, H.HoldingsSnapshot | Exception | None],
) -> tuple[H.HoldingsFetcher, list[str]]:
    calls: list[str] = []

    def _fetch(ticker: str, asof: date) -> H.HoldingsSnapshot | None:
        calls.append(ticker)
        result = data.get(ticker)
        if isinstance(result, Exception):
            raise result
        return result

    return _fetch, calls


def test_fetch_holdings_universe_first_run_writes_snapshots_and_overlap(
    project: C.Project,
) -> None:
    asof = date(2024, 6, 1)
    data = {
        "AAA": snapshot("AAA", date(2024, 3, 31), date(2024, 5, 15), (hold("1", "X", 0.4),)),
        "BBB": snapshot("BBB", date(2024, 3, 31), date(2024, 5, 20), (hold("1", "X", 0.1),)),
        "AAAX": snapshot("AAAX", date(2024, 3, 31), date(2024, 5, 15), (hold("2", "Y", 0.3),)),
        "GDX": snapshot("GDX", date(2024, 3, 31), date(2024, 5, 15), (hold("3", "Z", 0.3),)),
        "GDXJ": snapshot("GDXJ", date(2024, 3, 31), date(2024, 5, 15), (hold("3", "Z", 0.1),)),
    }
    fetch, calls = _fetcher_from(dict(data))
    report = H.fetch_holdings_universe(
        project, fetch, asof=asof, write_overlap=_write_overlap_pickle
    )
    assert sorted(calls) == ["AAA", "AAAX", "BBB", "GDX", "GDXJ"]
    assert sorted(report.fetched) == ["AAA", "AAAX", "BBB", "GDX", "GDXJ"]
    assert report.failed == ()
    assert report.overlap_path is not None
    matrix = pd.read_pickle(report.overlap_path)
    H.validate_overlap_matrix(matrix)
    assert set(matrix.index) == {"AAA", "AAAX", "BBB", "GDX", "GDXJ"}
    for ticker, snap in data.items():
        path = H.snapshot_path(project.paths.data_external / "holdings", ticker, snap.asof)
        assert path.is_file()


def test_fetch_holdings_universe_second_run_uses_cache(project: C.Project) -> None:
    asof = date(2024, 6, 1)
    data: dict[str, H.HoldingsSnapshot | Exception | None] = {
        t: snapshot(t, date(2024, 3, 31), date(2024, 5, 15), (hold("1", "X", 0.1),))
        for t in H.eligible_tickers(project)
    }
    fetch, calls = _fetcher_from(dict(data))
    H.fetch_holdings_universe(project, fetch, asof=asof, write_overlap=_write_overlap_pickle)
    assert len(calls) == len(H.eligible_tickers(project))
    calls.clear()
    fetch2, calls2 = _fetcher_from(dict(data))
    report2 = H.fetch_holdings_universe(
        project, fetch2, asof=asof, write_overlap=_write_overlap_pickle
    )
    assert calls2 == []  # nothing re-fetched: every ticker already has a cached snapshot
    assert sorted(report2.cached) == sorted(H.eligible_tickers(project))


def test_fetch_holdings_universe_force_refetches(project: C.Project) -> None:
    asof = date(2024, 6, 1)
    data: dict[str, H.HoldingsSnapshot | Exception | None] = {
        t: snapshot(t, date(2024, 3, 31), date(2024, 5, 15), (hold("1", "X", 0.1),))
        for t in H.eligible_tickers(project)
    }
    fetch, _ = _fetcher_from(dict(data))
    H.fetch_holdings_universe(project, fetch, asof=asof, write_overlap=_write_overlap_pickle)
    fetch2, calls2 = _fetcher_from(dict(data))
    H.fetch_holdings_universe(
        project, fetch2, asof=asof, force=True, write_overlap=_write_overlap_pickle
    )
    assert sorted(calls2) == sorted(H.eligible_tickers(project))


def test_fetch_holdings_universe_failed_ticker_does_not_stop_others(project: C.Project) -> None:
    asof = date(2024, 6, 1)
    tickers = H.eligible_tickers(project)
    data: dict[str, H.HoldingsSnapshot | Exception | None] = {
        t: snapshot(t, date(2024, 3, 31), date(2024, 5, 15), (hold("1", "X", 0.1),))
        for t in tickers
    }
    data[tickers[0]] = RuntimeError("boom")
    data[tickers[1]] = None
    fetch, _ = _fetcher_from(data)
    report = H.fetch_holdings_universe(
        project, fetch, asof=asof, write_overlap=_write_overlap_pickle
    )
    assert tickers[0] in report.failed
    assert tickers[1] in report.failed
    assert "RuntimeError" in next(o.error for o in report.outcomes if o.ticker == tickers[0])
    assert all(t in report.fetched for t in tickers[2:])


def test_fetch_holdings_universe_failed_refresh_keeps_cached(project: C.Project) -> None:
    asof1 = date(2024, 6, 1)
    asof2 = date(2024, 9, 1)
    tickers = H.eligible_tickers(project)
    good: dict[str, H.HoldingsSnapshot | Exception | None] = {
        t: snapshot(t, date(2024, 3, 31), date(2024, 5, 15), (hold("1", "X", 0.1),))
        for t in tickers
    }
    fetch, _ = _fetcher_from(dict(good))
    H.fetch_holdings_universe(project, fetch, asof=asof1, write_overlap=_write_overlap_pickle)

    failing: dict[str, H.HoldingsSnapshot | Exception | None] = {t: None for t in tickers}
    fetch2, _ = _fetcher_from(failing)
    report2 = H.fetch_holdings_universe(
        project, fetch2, asof=asof2, force=True, write_overlap=_write_overlap_pickle
    )
    assert sorted(report2.failed) == sorted(tickers)
    # the overlap matrix is still built from the carried-forward cached snapshots
    assert report2.overlap_path is not None


def test_fetch_holdings_universe_no_overlap_written_with_fewer_than_two(
    project: C.Project,
) -> None:
    asof = date(2024, 6, 1)
    tickers = H.eligible_tickers(project)
    data: dict[str, H.HoldingsSnapshot | Exception | None] = {t: None for t in tickers}
    data[tickers[0]] = snapshot(
        tickers[0], date(2024, 3, 31), date(2024, 5, 15), (hold("1", "X", 0.1),)
    )
    fetch, _ = _fetcher_from(data)
    report = H.fetch_holdings_universe(
        project, fetch, asof=asof, write_overlap=_write_overlap_pickle
    )
    assert report.overlap_path is None


def test_summary_lines_mentions_failed_and_overlap(project: C.Project) -> None:
    asof = date(2024, 6, 1)
    tickers = H.eligible_tickers(project)
    data: dict[str, H.HoldingsSnapshot | Exception | None] = {
        t: snapshot(t, date(2024, 3, 31), date(2024, 5, 15), (hold("1", "X", 0.1),))
        for t in tickers
    }
    data[tickers[0]] = None
    fetch, _ = _fetcher_from(data)
    report = H.fetch_holdings_universe(
        project, fetch, asof=asof, write_overlap=_write_overlap_pickle
    )
    lines = H.summary_lines(report)
    joined = "\n".join(lines)
    assert "failed" in joined
    assert tickers[0] in joined
    assert "overlap matrix:" in joined


# --------------------------------------------------------------------------------------
# SEC series/class CSV parsing
# --------------------------------------------------------------------------------------

_SERIES_CSV_COLUMNS = (
    # Exact header of the real SEC Investment Company Series and Class dataset (confirmed
    # 2026-09-27 against a live download: the previous fixture used "CIK" -- self-consistent with
    # a bug in parse_series_class_csv's own column-name assumption, "CIK" instead of "CIK Number"
    # -- which made build_series_map silently return an empty map against the real file (every
    # row's CIK read as "", so every row was skipped); fixed in both places at once so the fixture
    # cannot mask a recurrence of the same class of bug).
    "Reporting File Number,CIK Number,Entity Name,Entity Org Type,"
    "Series ID,Series Name,Class ID,Class Name,Class Ticker,"
    "Address_1,Address_2,City,State,Zip Code"
)
_SERIES_CSV_ROWS = (
    "811-01234,320193,Example Trust,30,S000012345,Example Semiconductor ETF,"
    "C000098765,Example Semiconductor ETF,AAA,1 Main St,,New York,NY,10001",
    "811-01234,320193,Example Trust,30,S000012346,No Ticker Series,"
    "C000098766,No Ticker Class,,1 Main St,,New York,NY,10001",
    "811-05678,999999,Other Trust,30,S000055555,Beta Fund,"
    "C000011111,Beta Fund,bbb,2 Elm St,,Chicago,IL,60601",
)
_SERIES_CSV = _SERIES_CSV_COLUMNS + "\n" + "\n".join(_SERIES_CSV_ROWS) + "\n"


def test_parse_series_class_csv() -> None:
    out = H.parse_series_class_csv(_SERIES_CSV)
    assert set(out) == {"AAA", "BBB"}
    ref = out["AAA"]
    assert ref.cik == "0000320193"
    assert ref.series_id == "S000012345"
    assert ref.class_id == "C000098765"
    assert ref.series_name == "Example Semiconductor ETF"
    # lower-case ticker in the source CSV is upper-cased
    assert out["BBB"].cik == "0000999999"


def test_parse_series_class_csv_skips_rows_missing_ticker_cik_or_series() -> None:
    out = H.parse_series_class_csv(_SERIES_CSV)
    assert "NO TICKER SERIES" not in out
    assert len(out) == 2


# --------------------------------------------------------------------------------------
# submissions JSON parsing (_recent_nport_filings)
# --------------------------------------------------------------------------------------


def _submissions_payload() -> dict:
    return {
        "cik": "0000320193",
        "filings": {
            "recent": {
                "form": ["10-K", "NPORT-P", "NPORT-P", "NPORT-P"],
                "accessionNumber": [
                    "0001193125-24-000001",
                    "0001193125-24-000002",
                    "0001193125-24-000003",
                    "0001193125-23-000004",
                ],
                "filingDate": ["2024-08-01", "2024-05-15", "2024-08-20", "2023-11-10"],
                "reportDate": ["2024-06-30", "2024-03-31", "2024-06-30", "2023-09-30"],
                "primaryDocument": [
                    "10k.htm",
                    "primary_doc.xml",
                    "primary_doc.xml",
                    "primary_doc.xml",
                ],
            }
        },
    }


def test_recent_nport_filings_filters_to_nport_p() -> None:
    filings = H._recent_nport_filings(_submissions_payload())
    assert len(filings) == 3
    assert all(f.accession.startswith("0001193125") for f in filings)
    by_accession = {f.accession: f for f in filings}
    f = by_accession["0001193125-24-000002"]
    assert f.filing_date == date(2024, 5, 15)
    assert f.report_date == date(2024, 3, 31)
    assert f.primary_document == "primary_doc.xml"


def test_recent_nport_filings_rejects_inconsistent_array_lengths() -> None:
    payload = _submissions_payload()
    payload["filings"]["recent"]["form"].append("NPORT-P")  # now longer than the other arrays
    with pytest.raises(H.HoldingsError):
        H._recent_nport_filings(payload)


def test_recent_nport_filings_tolerates_missing_report_date_array() -> None:
    payload = _submissions_payload()
    del payload["filings"]["recent"]["reportDate"]
    filings = H._recent_nport_filings(payload)
    assert all(f.report_date is None for f in filings)


def test_recent_nport_filings_empty_payload() -> None:
    assert H._recent_nport_filings({}) == []


def test_filing_document_url() -> None:
    filing = H.FilingRef(
        accession="0001193125-24-000002",
        filing_date=date(2024, 5, 15),
        report_date=date(2024, 3, 31),
        primary_document="primary_doc.xml",
    )
    url = H._filing_document_url("0000320193", filing)
    assert url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000119312524000002/primary_doc.xml"
    )


def test_filing_document_url_strips_the_xslt_viewer_subfolder() -> None:
    """Gate 2 regression (2026-09-27): a real fetch showed ``primaryDocument`` can be
    ``"xslFormNPORT-P_X01/primary_doc.xml"`` -- SEC's XSLT-rendered HTML view, not the raw XML.
    The URL builder must use only the filename, dropping any viewer-folder prefix, since the raw
    document sits directly in the accession root (confirmed by a second real fetch to the
    stripped URL returning `text/xml`)."""
    filing = H.FilingRef(
        accession="0001410368-26-100095",
        filing_date=date(2026, 9, 25),
        report_date=date(2026, 6, 30),
        primary_document="xslFormNPORT-P_X01/primary_doc.xml",
    )
    url = H._filing_document_url("0001137360", filing)
    assert url == (
        "https://www.sec.gov/Archives/edgar/data/1137360/000141036826100095/primary_doc.xml"
    )


# --------------------------------------------------------------------------------------
# NPORT-P XML parsing
# --------------------------------------------------------------------------------------


def _nport_xml(*, with_namespace: bool = True, series_id: str = "S000012345") -> str:
    ns = ' xmlns="http://www.sec.gov/edgar/nport"' if with_namespace else ""
    return f"""<?xml version="1.0"?>
<edgarSubmission{ns}>
  <formData>
    <genInfo>
      <seriesId>{series_id}</seriesId>
      <repPdDate>2024-03-31</repPdDate>
    </genInfo>
    <invstOrSec>
      <name>Long Holding One</name>
      <cusip>111111111</cusip>
      <pctVal>8.100000</pctVal>
      <payoffProfile>Long</payoffProfile>
    </invstOrSec>
    <invstOrSec>
      <name>Long Holding No Cusip</name>
      <cusip>N/A</cusip>
      <pctVal>2.500000</pctVal>
      <payoffProfile>Long</payoffProfile>
    </invstOrSec>
    <invstOrSec>
      <name>Short Position</name>
      <cusip>222222222</cusip>
      <pctVal>1.000000</pctVal>
      <payoffProfile>Short</payoffProfile>
    </invstOrSec>
    <invstOrSec>
      <name>Zero Weight</name>
      <cusip>333333333</cusip>
      <pctVal>0.000000</pctVal>
      <payoffProfile>Long</payoffProfile>
    </invstOrSec>
  </formData>
</edgarSubmission>
"""


def test_parse_nport_xml_basic_fields() -> None:
    parsed = H.parse_nport_xml(_nport_xml())
    assert parsed.series_id == "S000012345"
    assert parsed.report_date == date(2024, 3, 31)
    # short and zero-weight positions excluded; no-cusip position falls back to normalised name
    assert len(parsed.holdings) == 2
    by_constituent = {h.constituent: h for h in parsed.holdings}
    assert by_constituent["111111111"].weight == pytest.approx(0.081)
    assert "LONG HOLDING NO CUSIP" in by_constituent


def test_parse_nport_xml_works_without_namespace() -> None:
    parsed = H.parse_nport_xml(_nport_xml(with_namespace=False))
    assert len(parsed.holdings) == 2
    assert parsed.series_id == "S000012345"


def test_parse_nport_xml_raises_on_malformed_xml() -> None:
    with pytest.raises(H.HoldingsError):
        H.parse_nport_xml("<not-well-formed")


def test_parse_nport_xml_raises_without_gen_info() -> None:
    with pytest.raises(H.HoldingsError):
        H.parse_nport_xml("<edgarSubmission><formData></formData></edgarSubmission>")


def test_matches_series_case_insensitive() -> None:
    ref = H.SeriesRef(cik="0000320193", series_id="s000012345", class_id="c1", series_name="x")
    assert H._matches_series("S000012345", ref)
    assert not H._matches_series("S000099999", ref)
    assert not H._matches_series(None, ref)


# --------------------------------------------------------------------------------------
# _user_agent
# --------------------------------------------------------------------------------------


def test_user_agent_explicit() -> None:
    assert H._user_agent("Explicit Agent x@example.com") == "Explicit Agent x@example.com"


def test_user_agent_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(H.SEC_USER_AGENT_ENV, "Env Agent x@example.com")
    assert H._user_agent(None) == "Env Agent x@example.com"


def test_user_agent_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(H.SEC_USER_AGENT_ENV, raising=False)
    with pytest.raises(H.HoldingsError):
        H._user_agent(None)


# --------------------------------------------------------------------------------------
# _request_with_retries
# --------------------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, *, text: str = "", json_data: object = None, ok: bool = True) -> None:
        self._text = text
        self._json = json_data
        self._ok = ok

    def raise_for_status(self) -> None:
        if not self._ok:
            raise RuntimeError("HTTP error")

    def json(self) -> object:
        return self._json

    @property
    def text(self) -> str:
        return self._text


def test_request_with_retries_succeeds_first_try() -> None:
    calls = []

    def get(url: str, *, headers: dict[str, str], timeout: float) -> _FakeResponse:
        calls.append(url)
        return _FakeResponse(text="ok")

    resp = H._request_with_retries(
        get, "http://x", {}, attempts=3, pause=0.0, sleep=lambda _s: None, timeout=1.0
    )
    assert resp.text == "ok"
    assert len(calls) == 1


def test_request_with_retries_retries_then_succeeds() -> None:
    attempts_made = {"n": 0}
    sleeps: list[float] = []

    def get(url: str, *, headers: dict[str, str], timeout: float) -> _FakeResponse:
        attempts_made["n"] += 1
        if attempts_made["n"] < 3:
            return _FakeResponse(ok=False)
        return _FakeResponse(text="ok")

    resp = H._request_with_retries(
        get, "http://x", {}, attempts=3, pause=0.01, sleep=sleeps.append, timeout=1.0
    )
    assert resp.text == "ok"
    assert attempts_made["n"] == 3
    assert len(sleeps) == 2


def test_request_with_retries_raises_after_exhausting_attempts() -> None:
    def get(url: str, *, headers: dict[str, str], timeout: float) -> _FakeResponse:
        return _FakeResponse(ok=False)

    with pytest.raises(RuntimeError):
        H._request_with_retries(
            get, "http://x", {}, attempts=2, pause=0.0, sleep=lambda _s: None, timeout=1.0
        )


# --------------------------------------------------------------------------------------
# fetch_sec_nport_holdings (network fully stubbed via `get`)
# --------------------------------------------------------------------------------------


def _fake_get_factory(
    *, submissions: dict, docs: dict[str, str]
) -> tuple[Callable[..., "_FakeResponse"], list[str]]:
    """A stand-in ``requests.get``: submissions JSON for the CIK URL, XML text for doc URLs."""
    urls: list[str] = []

    def get(url: str, *, headers: dict[str, str], timeout: float) -> _FakeResponse:
        urls.append(url)
        assert headers.get("User-Agent")
        if "data.sec.gov/submissions" in url:
            return _FakeResponse(json_data=submissions)
        for suffix, text in docs.items():
            if url.endswith(suffix):
                return _FakeResponse(text=text)
        raise AssertionError(f"unexpected URL {url}")

    return get, urls


def test_fetch_sec_nport_holdings_returns_none_for_unmapped_ticker() -> None:
    result = H.fetch_sec_nport_holdings(
        "ZZZ", date(2024, 6, 1), series_map={}, user_agent="Test x@example.com"
    )
    assert result is None


def test_fetch_sec_nport_holdings_happy_path() -> None:
    ref = H.SeriesRef(
        cik="0000320193", series_id="S000012345", class_id="C000098765", series_name="Example"
    )
    submissions = {
        "filings": {
            "recent": {
                "form": ["NPORT-P"],
                "accessionNumber": ["0001193125-24-000002"],
                "filingDate": ["2024-05-15"],
                "reportDate": ["2024-03-31"],
                "primaryDocument": ["primary_doc.xml"],
            }
        }
    }
    doc_url_suffix = "000119312524000002/primary_doc.xml"
    get, urls = _fake_get_factory(submissions=submissions, docs={doc_url_suffix: _nport_xml()})
    snap = H.fetch_sec_nport_holdings(
        "AAA",
        date(2024, 6, 1),
        series_map={"AAA": ref},
        get=get,
        user_agent="Test x@example.com",
        sleep=lambda _s: None,
    )
    assert snap is not None
    assert snap.ticker == "AAA"
    assert snap.asof == date(2024, 3, 31)
    assert snap.filing_date == date(2024, 5, 15)
    assert snap.source == "sec_nport"
    assert len(snap.holdings) == 2
    assert any("submissions" in u for u in urls)
    assert any(u.endswith(doc_url_suffix) for u in urls)


def test_fetch_sec_nport_holdings_filters_filings_after_asof() -> None:
    ref = H.SeriesRef(cik="0000320193", series_id="S000012345", class_id="c", series_name="x")
    submissions = {
        "filings": {
            "recent": {
                "form": ["NPORT-P"],
                "accessionNumber": ["0001193125-24-000002"],
                "filingDate": ["2024-05-15"],
                "reportDate": ["2024-03-31"],
                "primaryDocument": ["primary_doc.xml"],
            }
        }
    }
    get, _ = _fake_get_factory(submissions=submissions, docs={"primary_doc.xml": _nport_xml()})
    # asof is before the only filing's filing_date: nothing eligible
    snap = H.fetch_sec_nport_holdings(
        "AAA",
        date(2024, 1, 1),
        series_map={"AAA": ref},
        get=get,
        user_agent="Test x@example.com",
        sleep=lambda _s: None,
    )
    assert snap is None


def test_fetch_sec_nport_holdings_tries_next_candidate_on_series_mismatch() -> None:
    ref = H.SeriesRef(cik="0000320193", series_id="S000012345", class_id="c", series_name="x")
    submissions = {
        "filings": {
            "recent": {
                "form": ["NPORT-P", "NPORT-P"],
                "accessionNumber": ["0001193125-24-000003", "0001193125-24-000002"],
                "filingDate": ["2024-08-20", "2024-05-15"],
                "reportDate": ["2024-06-30", "2024-03-31"],
                "primaryDocument": ["primary_doc.xml", "primary_doc.xml"],
            }
        }
    }
    wrong_series_doc = _nport_xml(series_id="S000099999")
    right_series_doc = _nport_xml(series_id="S000012345")
    get, urls = _fake_get_factory(
        submissions=submissions,
        docs={
            "000119312524000003/primary_doc.xml": wrong_series_doc,
            "000119312524000002/primary_doc.xml": right_series_doc,
        },
    )
    snap = H.fetch_sec_nport_holdings(
        "AAA",
        date(2024, 9, 1),
        series_map={"AAA": ref},
        get=get,
        user_agent="Test x@example.com",
        sleep=lambda _s: None,
    )
    assert snap is not None
    assert snap.filing_date == date(2024, 5, 15)  # the second (older) candidate matched
    doc_urls = [u for u in urls if u.endswith("primary_doc.xml")]
    assert len(doc_urls) == 2  # both candidates were opened, newest first


def test_fetch_sec_nport_holdings_returns_none_when_no_candidate_matches() -> None:
    ref = H.SeriesRef(cik="0000320193", series_id="S000012345", class_id="c", series_name="x")
    submissions = {
        "filings": {
            "recent": {
                "form": ["NPORT-P"],
                "accessionNumber": ["0001193125-24-000002"],
                "filingDate": ["2024-05-15"],
                "reportDate": ["2024-03-31"],
                "primaryDocument": ["primary_doc.xml"],
            }
        }
    }
    wrong_series_doc = _nport_xml(series_id="S000099999")
    get, _ = _fake_get_factory(submissions=submissions, docs={"primary_doc.xml": wrong_series_doc})
    snap = H.fetch_sec_nport_holdings(
        "AAA",
        date(2024, 6, 1),
        series_map={"AAA": ref},
        get=get,
        user_agent="Test x@example.com",
        sleep=lambda _s: None,
    )
    assert snap is None


def _multi_series_submissions(*, n_other_series: int, target_series_id: str) -> dict:
    """``n_other_series`` wrong-series filings (newest first), then one target-series filing --
    mimics a large shared-CIK trust interleaving many series' filings ahead of the target's own
    (see :func:`H.fetch_sec_nport_holdings`'s docstring, Gate 2 round 3)."""
    n = n_other_series + 1
    forms = ["NPORT-P"] * n
    accns = [f"0001193125-24-{i:06d}" for i in range(n)]
    # strictly decreasing, one per day: index 0 filed latest, the last index filed earliest --
    # so the post-sort (newest filing_date first) order matches construction order exactly.
    base = date(2024, 12, 1)
    dates = [(base - timedelta(days=i)).isoformat() for i in range(n)]
    reports = ["2024-03-31"] * n
    docs = [f"doc{i}.xml" for i in range(n)]
    return {
        "filings": {
            "recent": {
                "form": forms,
                "accessionNumber": accns,
                "filingDate": dates,
                "reportDate": reports,
                "primaryDocument": docs,
            }
        }
    }


def test_fetch_sec_nport_holdings_default_max_candidates_covers_a_deep_shared_cik_trust() -> None:
    """Regression test for the ``max_candidates=8`` bug (Gate 2 round 3, 2026-09-27): SMH's real
    matching filing sat at index 58 of 569 candidates for its CIK. 20 wrong-series filings ahead of
    the target (well past the old default of 8) must still be found with the library default."""
    ref = H.SeriesRef(cik="0000320193", series_id="S000012345", class_id="c", series_name="x")
    submissions = _multi_series_submissions(n_other_series=20, target_series_id="S000012345")
    docs = {f"doc{i}.xml": _nport_xml(series_id="S000099999") for i in range(20)}
    docs["doc20.xml"] = _nport_xml(series_id="S000012345")
    get, urls = _fake_get_factory(submissions=submissions, docs=docs)
    snap = H.fetch_sec_nport_holdings(
        "AAA",
        date(2024, 12, 1),
        series_map={"AAA": ref},
        get=get,
        user_agent="Test x@example.com",
        sleep=lambda _s: None,
        # max_candidates intentionally omitted: exercises the library default (300), not an
        # explicit override, since the regression was the *default* being too small.
    )
    assert snap is not None
    assert snap.filing_date == date(2024, 11, 11)  # the deepest (oldest) candidate, index 20
    doc_urls = [u for u in urls if u.endswith(".xml") and "doc" in u]
    assert len(doc_urls) == 21  # all 21 candidates opened, newest first, until the match


def test_fetch_sec_nport_holdings_max_candidates_override_still_bounds_the_search() -> None:
    """An explicit ``max_candidates`` still caps the search -- the fix raises the *default*, it
    does not remove the safety ceiling."""
    ref = H.SeriesRef(cik="0000320193", series_id="S000012345", class_id="c", series_name="x")
    submissions = _multi_series_submissions(n_other_series=5, target_series_id="S000012345")
    docs = {f"doc{i}.xml": _nport_xml(series_id="S000099999") for i in range(5)}
    docs["doc5.xml"] = _nport_xml(series_id="S000012345")  # index 5: beyond a max_candidates=3 cap
    get, urls = _fake_get_factory(submissions=submissions, docs=docs)
    snap = H.fetch_sec_nport_holdings(
        "AAA",
        date(2024, 12, 1),
        series_map={"AAA": ref},
        max_candidates=3,
        get=get,
        user_agent="Test x@example.com",
        sleep=lambda _s: None,
    )
    assert snap is None
    doc_urls = [u for u in urls if u.endswith(".xml") and "doc" in u]
    assert len(doc_urls) == 3  # stopped at the cap, never reached the real match at index 5
