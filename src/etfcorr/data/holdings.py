"""Step 2.4: point-in-time ETF holdings snapshots and the weighted-Jaccard holdings-overlap matrix.

Question served: EFT_project_v2.md section 3 ("Overlap control") and section 4, item 7 both flag
that thematic ETFs sharing large-cap constituents (SMH/SOXX both hold NVDA, TSM, AVGO; ICLN/TAN
overlap in solar names) will show correlated returns for a *mechanical* reason -- shared holdings
-- independent of any genuine cross-sector co-movement the study wants to measure. This module
supplies the other half of that decomposition: a point-in-time holdings snapshot per ticker and a
symmetric weighted-Jaccard overlap matrix between tickers, so that ``correlation/overlap.py``
(step 2.5, per ``docs/STATUS.md``'s own step order: "overlap.py... needs both correlation and
holdings") can regress return correlation on holdings overlap and report the residual as the
"economic" part. This step delivers only the *data layer* (fetch, point-in-time storage, the
overlap matrix itself); the regression against return correlation is step 2.5's job.

**What is computed and stored**, per ``project_plan.md`` sections 4 and 6.3 ("Holdings snapshots
and overlap... saves ``data/external/holdings/<ticker>_<asof>.csv``; computes weighted-Jaccard
overlap -> ``overlap_matrix.parquet``") and section 4, item 11 ("every fetcher [needs] a primary +
fallback source"):

* :class:`HoldingsSnapshot` -- one ticker's portfolio composition as of a *reporting* date
  (``asof``, the period the holdings describe) and a *filing* date (``filing_date``, when that
  report became public). These are deliberately two different dates: EFT_project_v2.md section 3
  says holdings filings have "reporting lag -> use as-of filing date to stay point-in-time," and
  project_plan.md section 6.3 repeats it ("only use after the filing date"). :func:`latest_asof`
  is the point-in-time accessor every later module must use -- it is filtered on ``filing_date``,
  never on ``asof`` or the file's own timestamp, so a feature computed "as of" some historical date
  cannot see a filing the market did not yet have (the same discipline ``etfcorr.leakage`` enforces
  for row-indexed data; see "Point-in-time and leakage" below for why the shared detectors do not
  apply directly here).
* ``data/external/holdings/<ticker>_<asof>.csv`` -- one CSV per (ticker, reporting-period) pair,
  written by :func:`write_snapshot_csv`. Git-ignored like the rest of ``data/external`` (see
  ``.gitignore``); this is derived-but-still-external data (public regulatory filings, not a
  commercial vendor, but still not committed, matching every other Phase-1 raw/interim/external
  path).
* ``data/external/overlap_matrix.parquet`` -- the symmetric, unit-diagonal weighted-Jaccard
  overlap matrix (:func:`overlap_matrix`) across every ticker with a usable snapshot at the
  requested ``asof``.

**Source and fallback chain, stated on purpose.** EFT_project_v2.md's build-vs-reuse map
(section 9) flags ``data/holdings.py`` as **[REUSE-VERIFY]**, citing ``breadth-thrust-etf``'s own
finding that "the iShares US endpoint [was] blocked in 2026" -- exactly the endpoint-churn risk
section 4, item 11 warns about for *every* issuer's own holdings page (different HTML/CSV shape
per issuer -- iShares, SPDR, VanEck, Invesco, Global X, ARK, WisdomTree, ... -- across this
project's 56-ticker universe). This step ships one concrete, real network fetcher --
:func:`fetch_sec_nport_holdings`, against the SEC's own EDGAR system -- rather than a per-issuer
scraper for each of the ~15 issuers in ``config/universe.yaml``, for a reason stated rather than
silently assumed: SEC EDGAR is a single, uniform, documented, government-hosted API covering every
SEC-registered fund regardless of issuer, and does not depend on any one issuer's website staying
in the same shape. This is a deliberate scope narrowing versus project_plan.md's literal "issuer
holdings files *and/or* SEC N-PORT" (the "and/or" already allows either alone); the
``HoldingsFetcher`` protocol below is issuer-agnostic by construction, so an issuer-CSV fetcher
can be added later as a
second entry in a fallback chain (``fetch_holdings_universe`` already accepts any single injected
fetcher; a chain of several is a one-line composition a future step can add) without changing this
step's storage, point-in-time or overlap-matrix code at all. This mirrors
``docs/STATUS.md``'s own decision log: "the SEC N-PORT holdings pipeline... considered and NOT
adopted [for simplification]; stays as originally planned" -- N-PORT was always the fallback this
plan named, and it is the one implemented here.

**SEC EDGAR mechanics used by** :func:`fetch_sec_nport_holdings` **(researched this session via
web search/fetch against SEC.gov itself; see the function's own docstring for exactly what was
and was not verified -- none of it was *run*, since Gate 1 offline checks use no network):**

1. Ticker -> (CIK, Series ID) via the SEC's own published *Investment Company Series and Class*
   dataset, one CSV per calendar year, columns including ``CIK``, ``Series ID``, ``Series Name``,
   ``Class ID``, ``Class Name``, ``Class Ticker`` -- confirmed via
   https://www.sec.gov/open/datasets-investment_company (fetched 2026-09-27), current-year URL
   ``https://www.sec.gov/files/investment/data/other/investment-company-series-class-information/
   investment-company-series-class-<year>.csv``. This mapping is **not itself point-in-time**: it
   reflects *currently active* registrants as of whatever year's file is used, not the roster at a
   historical ``asof`` -- a disclosed limitation (a ticker merged or delisted before the dataset's
   year will not resolve even if it existed at the requested historical ``asof``; CIK/series/class
   identifiers themselves are permanent once assigned, so a still-active ticker resolves correctly
   regardless of ``asof``).
2. CIK -> recent filings via ``https://data.sec.gov/submissions/CIK<10-digit-CIK>.json``, whose
   ``filings.recent`` object holds parallel arrays including ``form``, ``accessionNumber``,
   ``filingDate``, ``reportDate`` and ``primaryDocument`` -- field names confirmed via two
   independent fetches this session (a direct fetch of a real submissions JSON, and a third-party
   guide's field list), not from a single unverified source. Filtered to ``form == "NPORT-P"``.
3. A trust CIK can cover *several* series (several tickers share one CIK), and the submissions
   list does not itself say which series a given ``NPORT-P`` accession belongs to -- so each
   candidate filing's own primary document must be opened and its ``genInfo/seriesId`` compared to
   the target series ID from step 1, newest filing first, until one matches or ``max_candidates``
   is exhausted. This is the one piece of this design this session could not independently confirm
   end-to-end (that a trust's per-series filings are not otherwise distinguishable from the
   submissions list alone); it is the load-bearing assumption to check first if Gate 2 finds wrong
   holdings for a multi-series issuer.
4. The primary document is an XML filing (``.../<accession-no-dashes>/<primaryDocument>``) whose
   repeating holdings element is ``invstOrSec``, with child fields ``name``, ``cusip``, ``pctVal``
   (percentage of the fund's net assets -- so ``weight = pctVal / 100``) and ``payoffProfile``
   (``Long``/``Short``) -- confirmed by fetching one real ``primary_doc.xml`` this session.
   :func:`parse_nport_xml` matches these by *local* tag name (stripping any XML namespace prefix)
   rather than a hardcoded namespace URI, since this session could not confirm the exact namespace
   string from the summarised fetches used for research; local-name matching is correct whether or
   not a namespace is present, so nothing here depends on that unconfirmed detail.

**Weighted-Jaccard / Ruzicka overlap.** :func:`weighted_jaccard` is the standard generalisation of
the Jaccard index to non-negative weights, ``sum(min(a_c, b_c)) / sum(max(a_c, b_c))`` over the
union of constituents -- also known as the Ruzicka similarity (name confirmed by web search this
session, e.g. the R package ``abdiv::ruzicka``; the original citation, commonly given as Ružička
(1958), was **not** independently verified this session -- flagged per the project's citation rule
rather than asserted, and to be checked before it goes in the README/paper). This is a direct
definition, not an empirical claim, so it needs no further evidentiary tagging beyond that
citation-status note. Constituents are matched by CUSIP when the filing provides one (the standard
security identifier, most robust to name formatting differences across issuers/filings); a holding
without a usable CUSIP falls back to a whitespace/case-normalised name, a materially weaker match
(two funds holding the same company under differently formatted names may fail to match) -- a
disclosed limitation, not silently smoothed over. Short positions (``payoffProfile == "Short"``)
and non-positive ``pctVal`` entries are dropped before computing weights: a short position is not
an "overlapping" long exposure in the sense this matrix is meant to capture, and this project's
thematic-ETF universe is overwhelmingly long-only (EFT_project_v2.md section 3), so the practical
effect of this filter is expected to be small.

**Point-in-time and leakage.** :func:`latest_asof` plays the role ``etfcorr.leakage``'s
:func:`assert_no_lookahead`/:func:`assert_future_invariant` play for row-indexed feature functions,
but its signature (a *collection* of point-in-time snapshots plus a cutoff date, not a
``DataFrame`` indexed by date) does not match ``FeatureFn``'s contract, the same reason
``correlation/estimators.py``'s ticker-indexed matrices needed a bespoke purity test suite
(``tests/leakage/test_estimators_purity.py``) instead of the shared detectors. ``tests/leakage/
test_holdings_no_lookahead.py`` follows that precedent: it builds a synthetic history of several
snapshots at different filing dates and shows directly that adding a snapshot filed *after* the
query date never changes :func:`latest_asof`'s answer for that date.

No third-party code is adapted here (the SEC EDGAR client below only calls the public documented
APIs, described, not copied, from SEC.gov's own pages); nothing to record in ``THIRD_PARTY.md``.
"""

import csv
import importlib
import io
import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from etfcorr.config import Project
from etfcorr.data.universe_check import is_etf_like

SEC_WWW_BASE = "https://www.sec.gov"
SEC_DATA_BASE = "https://data.sec.gov"
SEC_SERIES_CLASS_CSV_TEMPLATE = (
    "https://www.sec.gov/files/investment/data/other/investment-company-series-class-information/"
    "investment-company-series-class-{year}.csv"
)
SEC_USER_AGENT_ENV = "SEC_USER_AGENT"

SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "ticker",
    "asof",
    "filing_date",
    "source",
    "constituent",
    "name",
    "weight",
)


class HoldingsError(ValueError):
    """A holdings fetch, parse, storage or overlap-matrix computation could not be completed."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------------------
# Core data model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Holding:
    """One portfolio position. ``weight`` is a fraction of net assets (0-1), never a percentage."""

    constituent: str  # CUSIP when available, else a normalised security name (see the parser)
    name: str
    weight: float


@dataclass(frozen=True)
class HoldingsSnapshot:
    """One ticker's portfolio composition as of a reporting date, made public on a filing date.

    ``holdings`` must be non-empty (see :func:`write_snapshot_csv` for why the storage format
    cannot represent an empty snapshot); a fetcher that finds zero usable holdings should return
    ``None`` rather than an empty :class:`HoldingsSnapshot`.
    """

    ticker: str
    asof: date
    filing_date: date
    source: str
    holdings: tuple[Holding, ...]

    def weights(self) -> dict[str, float]:
        """``constituent -> weight``, summing duplicate constituents within this snapshot."""
        out: dict[str, float] = {}
        for h in self.holdings:
            out[h.constituent] = out.get(h.constituent, 0.0) + h.weight
        return out


# --------------------------------------------------------------------------------------
# Weighted-Jaccard overlap (pure; see module docstring for the citation-status note)
# --------------------------------------------------------------------------------------


def weighted_jaccard(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Generalised (weighted) Jaccard / Ruzicka similarity between two non-negative weight maps.

    ``sum(min(a[c], b[c])) / sum(max(a[c], b[c]))`` over the union of constituents; a constituent
    absent from one map counts as weight 0 there. Symmetric by construction. Returns ``nan`` when
    both maps have zero total weight (undefined ratio, not zero overlap). Always exactly 1.0 for
    two identical, non-empty maps, whatever they sum to (``min == max`` pointwise). Raises
    :class:`HoldingsError` on a negative weight -- this function is a general similarity measure,
    not tied to the NPORT parser's own filtering.
    """
    if any(v < 0 for v in a.values()) or any(v < 0 for v in b.values()):
        raise HoldingsError("weighted_jaccard: weights must be non-negative")
    keys = set(a) | set(b)
    if not keys:
        return float("nan")
    num = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    den = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return float("nan") if den == 0.0 else num / den


def overlap_matrix(snapshots: Mapping[str, HoldingsSnapshot]) -> pd.DataFrame:
    """Symmetric, unit-diagonal weighted-Jaccard overlap matrix across ``snapshots``' tickers.

    Tickers are sorted for a deterministic row/column order. Each pair is computed once
    (``weighted_jaccard`` is symmetric) and mirrored, not recomputed, for speed and to guarantee
    exact (not just numerically close) symmetry.
    """
    if len(snapshots) < 2:
        raise HoldingsError(f"overlap_matrix needs at least 2 tickers, got {len(snapshots)}")
    tickers = sorted(snapshots)
    weights = {t: snapshots[t].weights() for t in tickers}
    n = len(tickers)
    data = np.empty((n, n), dtype=float)
    for i, a in enumerate(tickers):
        for j in range(i, n):
            b = tickers[j]
            value = weighted_jaccard(weights[a], weights[b])
            data[i, j] = value
            data[j, i] = value
    return pd.DataFrame(data, index=tickers, columns=tickers)


def validate_overlap_matrix(matrix: pd.DataFrame, *, atol: float = 1e-9) -> None:
    """Hard structural invariants: square, symmetric, finite, unit diagonal, values in [0, 1].

    Unlike ``correlation/estimators.py``'s ``validate_correlation_matrix``, PSD is not discussed
    at all here: a similarity matrix built from ``min``/``max`` has no Gram-matrix argument for
    positive semi-definiteness and this module makes no claim either way; it is used descriptively
    (as a regressor in step 2.5), not as a covariance/correlation input that would need it.
    """
    if not matrix.index.equals(matrix.columns):
        raise HoldingsError("overlap matrix index and columns must match")
    values = matrix.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise HoldingsError("overlap matrix contains non-finite values (nan/inf)")
    if not np.allclose(values, values.T, atol=atol):
        raise HoldingsError("overlap matrix is not symmetric")
    diag = np.diag(values)
    if not np.allclose(diag, 1.0, atol=atol):
        raise HoldingsError("overlap matrix diagonal must be 1.0 (self-overlap)")
    if values.min() < -atol or values.max() > 1.0 + atol:
        raise HoldingsError("overlap values must lie in [0, 1]")


# --------------------------------------------------------------------------------------
# Point-in-time snapshot selection
# --------------------------------------------------------------------------------------


def latest_asof(snapshots: Iterable[HoldingsSnapshot], as_of: date) -> HoldingsSnapshot | None:
    """The most recent snapshot whose ``filing_date <= as_of`` (``None`` if none qualify).

    This is the point-in-time accessor: a caller reconstructing "what holdings were known as of
    date X" must go through this function, never pick "the newest file on disk" or filter on
    ``asof`` (the reporting period) directly -- either would leak a filing the market did not yet
    have. Ties on ``filing_date`` (should not occur in practice: SEC assigns one filing date per
    accession) break on the later reporting ``asof``.
    """
    eligible = [s for s in snapshots if s.filing_date <= as_of]
    if not eligible:
        return None
    return max(eligible, key=lambda s: (s.filing_date, s.asof))


# --------------------------------------------------------------------------------------
# CSV snapshot storage
# --------------------------------------------------------------------------------------


def snapshot_path(holdings_dir: Path, ticker: str, asof: date) -> Path:
    """``<holdings_dir>/<TICKER>_<asof>.csv``, the literal path project_plan.md 6.3 names."""
    return holdings_dir / f"{ticker.upper()}_{asof.isoformat()}.csv"


def write_snapshot_csv(snapshot: HoldingsSnapshot, path: Path) -> None:
    """Write one snapshot; metadata (``ticker``/``asof``/``filing_date``/``source``) is repeated
    on every row so a single CSV is self-describing without a separate manifest file.

    Refuses an empty snapshot (see :class:`HoldingsSnapshot`'s docstring): with zero holding rows
    there is nowhere in this row-per-holding format to store the repeated metadata, so an empty
    snapshot would silently lose its own ``asof``/``filing_date`` on the next read. A fetcher that
    finds nothing usable should not call this function at all.
    """
    if not snapshot.holdings:
        raise HoldingsError(
            f"{snapshot.ticker}: refusing to write an empty snapshot "
            "(zero usable holdings cannot round-trip through this storage format)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(SNAPSHOT_COLUMNS)
        for h in snapshot.holdings:
            writer.writerow(
                [
                    snapshot.ticker,
                    snapshot.asof.isoformat(),
                    snapshot.filing_date.isoformat(),
                    snapshot.source,
                    h.constituent,
                    h.name,
                    repr(h.weight),
                ]
            )


def read_snapshot_csv(path: Path) -> HoldingsSnapshot:
    """Inverse of :func:`write_snapshot_csv`. Raises :class:`HoldingsError` on a malformed file."""
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or tuple(reader.fieldnames) != SNAPSHOT_COLUMNS:
            raise HoldingsError(f"{path}: unexpected columns {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise HoldingsError(f"{path}: no holding rows (empty or header-only file)")
    try:
        ticker = rows[0]["ticker"]
        asof = date.fromisoformat(rows[0]["asof"])
        filing_date = date.fromisoformat(rows[0]["filing_date"])
        source = rows[0]["source"]
    except (KeyError, ValueError) as exc:
        raise HoldingsError(f"{path}: malformed metadata in the first row: {exc}") from exc
    holdings: list[Holding] = []
    for row in rows:
        try:
            holdings.append(
                Holding(
                    constituent=row["constituent"], name=row["name"], weight=float(row["weight"])
                )
            )
        except (KeyError, ValueError) as exc:
            raise HoldingsError(f"{path}: malformed holding row: {exc}") from exc
    return HoldingsSnapshot(
        ticker=ticker, asof=asof, filing_date=filing_date, source=source, holdings=tuple(holdings)
    )


def _cached_snapshots(holdings_dir: Path, ticker: str) -> list[HoldingsSnapshot]:
    """Every stored snapshot for ``ticker`` (all reporting periods on disk), unsorted.

    A malformed cached file raises rather than being silently skipped -- a corrupted cache entry
    is a structural problem worth surfacing, the same "fail loudly" choice ``data/validate.py``
    makes for hard errors versus the soft, advisory warnings elsewhere in the data layer.
    """
    if not holdings_dir.is_dir():
        return []
    return [read_snapshot_csv(p) for p in sorted(holdings_dir.glob(f"{ticker.upper()}_*.csv"))]


# --------------------------------------------------------------------------------------
# Universe eligibility
# --------------------------------------------------------------------------------------


def eligible_tickers(project: Project) -> tuple[str, ...]:
    """Core and extended, ETF-like tickers only.

    Controls, ``^VIX`` and the ``*-USD`` crypto spot series have no "holdings" concept and are
    excluded via ``etfcorr.data.universe_check.is_etf_like`` (reused, not reimplemented, so the
    "what counts as ETF-like" rule lives in exactly one place across the two modules that need it).
    """
    return tuple(
        e.ticker
        for e in project.universe.tickers
        if e.panel in ("core", "extended") and is_etf_like(e.ticker)
    )


# --------------------------------------------------------------------------------------
# Orchestration: fetch every eligible ticker, cache, build the overlap matrix
# --------------------------------------------------------------------------------------

HoldingsFetcher = Callable[[str, date], HoldingsSnapshot | None]
Progress = Callable[[int, int, str], None]
OverlapWriter = Callable[[pd.DataFrame, Path], None]


def _write_overlap_parquet(matrix: pd.DataFrame, path: Path) -> None:
    """Default overlap-matrix writer.

    STATUS: NOT EXECUTED OFFLINE, the same ``pyarrow``-availability gap noted in
    ``data/download.py``'s ``write_panel_parquet``. Every other function in this module is tested
    offline; ``fetch_holdings_universe`` accepts an injectable ``write_overlap`` so its own
    orchestration logic is fully tested without a real Parquet engine (``tests/unit/
    test_holdings.py`` injects a stand-in).
    """
    matrix.to_parquet(path)


@dataclass(frozen=True)
class TickerOutcome:
    ticker: str
    status: str  # "cached" | "fetched" | "failed"
    error: str = ""


@dataclass(frozen=True)
class HoldingsReport:
    asof: date
    fetched_at: str
    outcomes: tuple[TickerOutcome, ...]
    out_dir: Path
    overlap_path: Path | None

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(o.ticker for o in self.outcomes if o.status == "failed")

    @property
    def fetched(self) -> tuple[str, ...]:
        return tuple(o.ticker for o in self.outcomes if o.status == "fetched")

    @property
    def cached(self) -> tuple[str, ...]:
        return tuple(o.ticker for o in self.outcomes if o.status == "cached")


def fetch_holdings_universe(
    project: Project,
    fetch: HoldingsFetcher,
    *,
    asof: date,
    force: bool = False,
    progress: Progress | None = None,
    now: Callable[[], datetime] = _utc_now,
    write_overlap: OverlapWriter = _write_overlap_parquet,
) -> HoldingsReport:
    """Fetch/cache a point-in-time holdings snapshot for every eligible ticker, then the overlap
    matrix across whichever tickers ended up with a usable snapshot.

    Mirrors ``data/download.py``'s ``download_universe`` shape on purpose (per-ticker error
    isolation; a failed refresh keeps the last cached snapshot rather than losing it; ``force``
    re-fetches even when a fresh-enough cached snapshot exists): a ticker whose ``fetch`` raises or
    returns ``None`` is recorded as ``"failed"`` and does not stop the run for the others.
    ``fetch`` receives ``(ticker, asof)`` and must itself return only a snapshot filed on or before
    ``asof`` (see :func:`latest_asof`) or ``None``.
    """
    tickers = eligible_tickers(project)
    out_dir = project.paths.data_external / "holdings"
    out_dir.mkdir(parents=True, exist_ok=True)

    outcomes: list[TickerOutcome] = []
    snapshots: dict[str, HoldingsSnapshot] = {}
    for i, ticker in enumerate(tickers, start=1):
        if progress is not None:
            progress(i, len(tickers), ticker)
        cached = latest_asof(_cached_snapshots(out_dir, ticker), asof)
        if cached is not None and not force:
            snapshots[ticker] = cached
            outcomes.append(TickerOutcome(ticker, "cached"))
            continue
        try:
            fresh = fetch(ticker, asof)
        except Exception as exc:  # network boundary: record and continue (as download.py)
            outcomes.append(TickerOutcome(ticker, "failed", f"{type(exc).__name__}: {exc}"[:300]))
            if cached is not None:
                snapshots[ticker] = cached
            continue
        if fresh is None:
            outcomes.append(TickerOutcome(ticker, "failed", "no_data"))
            if cached is not None:
                snapshots[ticker] = cached
            continue
        write_snapshot_csv(fresh, snapshot_path(out_dir, ticker, fresh.asof))
        snapshots[ticker] = fresh
        outcomes.append(TickerOutcome(ticker, "fetched"))

    overlap_path: Path | None = None
    if len(snapshots) >= 2:
        matrix = overlap_matrix(snapshots)
        validate_overlap_matrix(matrix)
        overlap_path = project.paths.data_external / "overlap_matrix.parquet"
        write_overlap(matrix, overlap_path)

    return HoldingsReport(
        asof=asof,
        fetched_at=now().isoformat(),
        outcomes=tuple(outcomes),
        out_dir=out_dir,
        overlap_path=overlap_path,
    )


def summary_lines(report: HoldingsReport) -> list[str]:
    lines = [
        f"holdings as of {report.asof} ({len(report.outcomes)} tickers)",
        f"  fetched {len(report.fetched)}, cached {len(report.cached)}, "
        f"failed {len(report.failed)}",
        f"  snapshots: {report.out_dir}",
    ]
    if report.overlap_path is not None:
        lines.append(f"  overlap matrix: {report.overlap_path}")
    else:
        lines.append("  overlap matrix: not written (fewer than 2 tickers had a usable snapshot)")
    if report.failed:
        lines.append(f"  failed: {', '.join(report.failed)}")
    return lines


# --------------------------------------------------------------------------------------
# SEC EDGAR N-PORT: ticker -> CIK/series mapping
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesRef:
    """One ticker's SEC identifiers, from the Investment Company Series and Class dataset."""

    cik: str  # zero-padded to 10 digits
    series_id: str  # "S" + 9 digits
    class_id: str  # "C" + 9 digits
    series_name: str


def _user_agent(explicit: str | None) -> str:
    """SEC requires a descriptive User-Agent identifying the requester (name + contact), e.g.
    ``"Filippo Surname email@example.com"`` -- confirmed via
    https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm (fetched 2026-09-27), which
    also states a fair-access limit of 10 requests/second (informs this module's default
    ``courtesy_pause``). Reads ``config/.env``'s ``SEC_USER_AGENT`` (already reserved in
    ``.env.example``) via the environment when not given explicitly; this module does not load
    ``.env`` itself (no ``.env`` loader exists anywhere in this project yet), so Filippo must
    export it or use a tool that does.
    """
    ua = explicit if explicit is not None else os.environ.get(SEC_USER_AGENT_ENV, "").strip()
    if not ua:
        raise HoldingsError(
            f"{SEC_USER_AGENT_ENV} is not set; SEC requires a descriptive User-Agent for "
            "automated EDGAR access (name and contact, e.g. 'Filippo Surname email@example.com'); "
            "see .env.example and https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm"
        )
    return ua


def download_series_class_dataset(
    path: Path,
    *,
    year: int,
    get: Callable[..., Any] | None = None,
    user_agent: str | None = None,
    timeout: float = 60.0,
) -> Path:
    """Download and cache the SEC's Investment Company Series and Class CSV for ``year``.

    STATUS: NOT EXECUTED OFFLINE (network + ``requests``). URL template and column names verified
    via https://www.sec.gov/open/datasets-investment_company (fetched 2026-09-27); not run.
    """
    requests = importlib.import_module("requests")
    url = SEC_SERIES_CLASS_CSV_TEMPLATE.format(year=year)
    headers = {"User-Agent": _user_agent(user_agent)}
    fetch = get if get is not None else requests.get
    resp = fetch(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return path


def parse_series_class_csv(text: str) -> dict[str, SeriesRef]:
    """``ticker -> SeriesRef`` from the SEC series/class CSV text (last row wins on a duplicate
    ticker, e.g. a share class re-using a symbol after a merger -- an accepted simplification, not
    expected to matter for this project's 56-ticker universe of single-share-class ETFs). Rows
    missing a ticker, CIK or series ID are skipped (not every class has a public ticker).
    """
    reader = csv.DictReader(io.StringIO(text))
    out: dict[str, SeriesRef] = {}
    for row in reader:
        ticker = (row.get("Class Ticker") or "").strip().upper()
        cik = (row.get("CIK Number") or "").strip()
        series_id = (row.get("Series ID") or "").strip()
        class_id = (row.get("Class ID") or "").strip()
        name = (row.get("Series Name") or "").strip()
        if not ticker or not cik or not series_id:
            continue
        out[ticker] = SeriesRef(
            cik=cik.zfill(10), series_id=series_id, class_id=class_id, series_name=name
        )
    return out


def build_series_map(
    cache_path: Path,
    *,
    year: int,
    force: bool = False,
    get: Callable[..., Any] | None = None,
    user_agent: str | None = None,
) -> dict[str, SeriesRef]:
    """Download the series/class dataset if not already cached (or ``force``), then parse it."""
    if force or not cache_path.is_file():
        download_series_class_dataset(cache_path, year=year, get=get, user_agent=user_agent)
    return parse_series_class_csv(cache_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# SEC EDGAR N-PORT: submissions list and NPORT-P filing parsing
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FilingRef:
    accession: str
    filing_date: date
    report_date: date | None
    primary_document: str


def _recent_nport_filings(payload: Mapping[str, Any]) -> list[FilingRef]:
    """Parse ``form == "NPORT-P"`` entries out of a submissions JSON's ``filings.recent`` object.

    Pure; field names (``form``, ``accessionNumber``, ``filingDate``, ``reportDate``,
    ``primaryDocument``, all parallel arrays under ``filings.recent``) confirmed this session by
    fetching a real submissions JSON and cross-checking a third-party field-name guide (see module
    docstring); this function itself never touches the network.
    """
    filings = payload.get("filings", {})
    recent = filings.get("recent", {}) if isinstance(filings, dict) else {}
    forms = list(recent.get("form", []))
    accns = list(recent.get("accessionNumber", []))
    dates = list(recent.get("filingDate", []))
    docs = list(recent.get("primaryDocument", []))
    reports = list(recent.get("reportDate", []))
    n = len(forms)
    if len(accns) != n or len(dates) != n or len(docs) != n:
        raise HoldingsError("submissions JSON: filings.recent arrays have inconsistent lengths")
    if len(reports) < n:
        reports = reports + [None] * (n - len(reports))

    out: list[FilingRef] = []
    for i in range(n):
        if str(forms[i]).upper() != "NPORT-P":
            continue
        try:
            filing_date = date.fromisoformat(str(dates[i])[:10])
        except ValueError:
            continue
        report_date: date | None = None
        if reports[i]:
            try:
                report_date = date.fromisoformat(str(reports[i])[:10])
            except ValueError:
                report_date = None
        out.append(
            FilingRef(
                accession=str(accns[i]),
                filing_date=filing_date,
                report_date=report_date,
                primary_document=str(docs[i]),
            )
        )
    return out


def _filing_document_url(cik: str, filing: FilingRef) -> str:
    """``.../Archives/edgar/data/<unpadded CIK>/<accession, no dashes>/<primary document file>``.

    ``primaryDocument`` in the submissions JSON is **not always a bare filename**: for a form with
    an XSLT viewer template (NPORT-P among them), it is a path like
    ``"xslFormNPORT-P_X01/primary_doc.xml"`` -- that subfolder is SEC's human-readable, XSLT-
    rendered HTML view (confirmed this session, Gate 2, by a real fetch: that URL returns
    `text/html`, tens of megabytes, not XML). The machine-readable raw XML always sits one level up,
    directly in the accession root, under the *filename* component alone (confirmed by a second
    real fetch returning `text/xml`, the actual `edgarSubmission` document) -- so only the basename
    of ``primaryDocument`` is used here, discarding any leading viewer-folder segment. For a form
    with no XSLT viewer, ``primaryDocument`` is already a bare filename, so this is a no-op there.
    """
    accession_folder = filing.accession.replace("-", "")
    doc = filing.primary_document.rsplit("/", 1)[-1]
    return f"{SEC_WWW_BASE}/Archives/edgar/data/{int(cik)}/{accession_folder}/{doc}"


@dataclass(frozen=True)
class ParsedNport:
    series_id: str | None
    report_date: date | None
    holdings: tuple[Holding, ...]


def _local_tag(tag: str) -> str:
    """Strip an XML namespace prefix (``{uri}tag`` -> ``tag``); see module docstring for why this
    session matches by local name rather than a hardcoded, unconfirmed namespace URI."""
    return tag.rsplit("}", 1)[-1]


def _find_first(root: ET.Element, name: str) -> ET.Element | None:
    for elem in root.iter():
        if _local_tag(elem.tag) == name:
            return elem
    return None


def _child_text(elem: ET.Element | None, name: str) -> str | None:
    if elem is None:
        return None
    node = _find_first(elem, name)
    if node is None or node.text is None:
        return None
    text = node.text.strip()
    return text or None


def _normalise_name(name: str) -> str:
    return " ".join(name.split()).upper()


_BAD_CUSIPS: frozenset[str] = frozenset({"", "N/A", "NONE", "000000000"})


def parse_nport_xml(xml_text: str) -> ParsedNport:
    """Parse one NPORT-P ``primary_doc.xml``: the reporting series/date, and every usable holding.

    A holding is kept when it has a positive ``pctVal`` and is not flagged ``payoffProfile ==
    "Short"`` (see module docstring); its identifier is the CUSIP when present and not an obvious
    placeholder, else a normalised security name (a materially weaker match, disclosed above).
    Raises :class:`HoldingsError` on unparsable XML or a document missing ``genInfo`` entirely;
    returns a :class:`ParsedNport` with ``series_id``/``report_date`` possibly ``None`` and/or
    ``holdings`` possibly empty otherwise, leaving the "is this good enough" decision to the
    caller (:func:`fetch_sec_nport_holdings`), which tries the next candidate filing rather than
    failing the whole ticker on one bad document.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise HoldingsError(f"malformed NPORT-P XML: {exc}") from exc

    gen_info = _find_first(root, "genInfo")
    if gen_info is None:
        raise HoldingsError("NPORT-P XML has no genInfo element")
    series_id = _child_text(gen_info, "seriesId")
    rep_pd_text = _child_text(gen_info, "repPdDate")
    report_date = None
    if rep_pd_text:
        try:
            report_date = date.fromisoformat(rep_pd_text[:10])
        except ValueError:
            report_date = None

    holdings: list[Holding] = []
    for elem in root.iter():
        if _local_tag(elem.tag) != "invstOrSec":
            continue
        pct_text = _child_text(elem, "pctVal")
        if pct_text is None:
            continue
        try:
            pct = float(pct_text)
        except ValueError:
            continue
        if pct <= 0:
            continue
        payoff = (_child_text(elem, "payoffProfile") or "Long").strip().lower()
        if payoff == "short":
            continue
        name = _child_text(elem, "name") or ""
        cusip = (_child_text(elem, "cusip") or "").upper()
        constituent = cusip if cusip not in _BAD_CUSIPS else _normalise_name(name)
        if not constituent:
            continue
        holdings.append(Holding(constituent=constituent, name=name, weight=pct / 100.0))

    return ParsedNport(series_id=series_id, report_date=report_date, holdings=tuple(holdings))


def _matches_series(parsed_series_id: str | None, target: SeriesRef) -> bool:
    return (
        parsed_series_id is not None
        and parsed_series_id.strip().upper() == target.series_id.strip().upper()
    )


# --------------------------------------------------------------------------------------
# SEC EDGAR N-PORT: HTTP with retry (mirrors data/sources.py's fetch_yfinance pattern)
# --------------------------------------------------------------------------------------


def _request_with_retries(
    fetch: Callable[..., Any],
    url: str,
    headers: dict[str, str],
    *,
    attempts: int,
    pause: float,
    sleep: Callable[[float], None],
    timeout: float,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = fetch(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # broad: any vendor/network failure is retried, then raised
            last_exc = exc
            if attempt == attempts:
                raise
            sleep(pause * 2 ** (attempt - 1))
    raise HoldingsError(
        f"unreachable: {url}"
    ) from last_exc  # pragma: no cover - loop always exits above


# --------------------------------------------------------------------------------------
# SEC EDGAR N-PORT: the fetcher
# --------------------------------------------------------------------------------------


def fetch_sec_nport_holdings(
    ticker: str,
    asof: date,
    *,
    series_map: Mapping[str, SeriesRef],
    max_candidates: int = 300,
    attempts: int = 3,
    pause: float = 1.0,
    courtesy_pause: float = 0.15,
    timeout: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    get: Callable[..., Any] | None = None,
    user_agent: str | None = None,
) -> HoldingsSnapshot | None:
    """Real SEC EDGAR N-PORT fetcher for one ticker (a :data:`HoldingsFetcher`).

    Looks the ticker up in ``series_map`` (:func:`build_series_map`); ``None`` if it is not an
    SEC-registered series ticker in that dataset. Fetches the CIK's submissions JSON, keeps
    ``NPORT-P`` filings filed on or before ``asof``, and opens up to ``max_candidates`` of them
    (newest first) until one's ``genInfo/seriesId`` matches the target series. Returns ``None`` if
    no candidate matches or yields any usable holdings.

    ``max_candidates``: a CIK can be a multi-series trust that files ``NPORT-P`` separately for
    every series it manages, and the submissions JSON's ``filings.recent`` list interleaves all of
    a CIK's series (not just the target ticker's). Gate 2, round 3 (2026-09-27) confirmed this is
    not a rare edge case for this universe: for SMH (VanEck ETF Trust, CIK 0001137360), the
    matching filing for SMH's own series sat at index 58 (0-indexed) of 569 total eligible filings
    for that CIK -- the previous default of 8 made SMH itself fail. 300 is chosen with margin above
    that observed worst case; it is an **[Inference]**, not a verified bound, since this step could
    not independently confirm how many series the largest shared-CIK trusts in this universe carry
    (iShares Trust, which hosts SOXX and IGV, is a plausible candidate for needing more -- both
    were still failing after this fix's first real run, see ``docs/checks/2.4.md``). If a ticker
    from a still-larger trust keeps failing, raise ``max_candidates`` (exposed on
    :func:`command_holdings` and the ``--max-candidates`` CLI option) rather than assuming 300 is
    universally sufficient. Whether SEC's ``filings.recent`` array itself has an upper bound (older
    filings reportedly move to a separate paginated ``files`` index) is CHECKED (SEC EDGAR API
    documentation, 2026-11-02: the last year or 1,000 filings, older ones in ``files``) -- so
    ``max_candidates`` cannot be assumed to always be the binding constraint.

    STATUS: NOT EXECUTED OFFLINE -- real network calls to SEC EDGAR, no override available in this
    planning sandbox. Every function it composes (:func:`_recent_nport_filings`,
    :func:`parse_nport_xml`, :func:`_matches_series`, :func:`_filing_document_url`,
    :func:`_request_with_retries`) is unit-tested offline against small synthetic fixtures shaped
    like the real schema; this function's own control flow (loop over candidates, first match wins)
    is exercised offline too, via an injected ``get`` stub that serves canned JSON/XML text instead
    of a real HTTP response.
    """
    ref = series_map.get(ticker.upper())
    if ref is None:
        return None
    requests = importlib.import_module("requests")
    headers = {"User-Agent": _user_agent(user_agent)}
    fetch = get if get is not None else requests.get

    sleep(courtesy_pause)
    subs_resp = _request_with_retries(
        fetch,
        f"{SEC_DATA_BASE}/submissions/CIK{ref.cik}.json",
        headers,
        attempts=attempts,
        pause=pause,
        sleep=sleep,
        timeout=timeout,
    )
    filings = [f for f in _recent_nport_filings(subs_resp.json()) if f.filing_date <= asof]
    filings.sort(key=lambda f: f.filing_date, reverse=True)

    for filing in filings[:max_candidates]:
        sleep(courtesy_pause)
        doc_resp = _request_with_retries(
            fetch,
            _filing_document_url(ref.cik, filing),
            headers,
            attempts=attempts,
            pause=pause,
            sleep=sleep,
            timeout=timeout,
        )
        try:
            parsed = parse_nport_xml(doc_resp.text)
        except HoldingsError:
            continue
        if not _matches_series(parsed.series_id, ref) or not parsed.holdings:
            continue
        report_date = parsed.report_date if parsed.report_date is not None else filing.filing_date
        return HoldingsSnapshot(
            ticker=ticker.upper(),
            asof=report_date,
            filing_date=filing.filing_date,
            source="sec_nport",
            holdings=parsed.holdings,
        )
    return None


# --------------------------------------------------------------------------------------
# Command implementation (typer-free, per the project's CLI convention)
# --------------------------------------------------------------------------------------

SERIES_CLASS_CACHE_NAME = "sec_series_class.csv"


def command_holdings(
    project: Project,
    *,
    asof: date,
    force: bool = False,
    year: int | None = None,
    max_candidates: int | None = None,
    progress: Progress | None = None,
) -> list[str]:
    """Build/refresh the series map, fetch every eligible ticker's holdings, write the overlap
    matrix. ``year`` defaults to the current calendar year (the most complete "active registrants"
    list the series/class dataset can offer; see the module docstring's disclosed limitation about
    tickers that were merged or delisted before that year's snapshot). ``max_candidates`` overrides
    :func:`fetch_sec_nport_holdings`'s per-ticker candidate-filing search bound (default 300; see
    that function's docstring) -- raise it if a still-larger shared-CIK trust keeps failing.
    """
    holdings_dir = project.paths.data_external / "holdings"
    holdings_dir.mkdir(parents=True, exist_ok=True)
    series_cache = holdings_dir / SERIES_CLASS_CACHE_NAME
    resolved_year = year if year is not None else date.today().year
    series_map = build_series_map(series_cache, year=resolved_year, force=force)

    def fetch(ticker: str, cutoff: date) -> HoldingsSnapshot | None:
        kwargs: dict[str, Any] = {}
        if max_candidates is not None:
            kwargs["max_candidates"] = max_candidates
        return fetch_sec_nport_holdings(ticker, cutoff, series_map=series_map, **kwargs)

    report = fetch_holdings_universe(project, fetch, asof=asof, force=force, progress=progress)
    return summary_lines(report)
