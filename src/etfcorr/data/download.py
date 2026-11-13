"""Step 1.2: download orchestration, cache and provenance log for the raw price panel.

Question served: turn the audited universe (step 1.1/1.1b) into a cached, checksummed price
panel that later steps (``validate.py``/``build_panel.py``, step 1.3) can read without ever
touching the network again, and that never silently grows past the date it was asked to reach.

What is written, per ``project_plan.md`` sections 4 and 6.3:

* ``data/raw/prices_yf.parquet`` -- one wide panel, ``MultiIndex`` columns ``(ticker, field)``
  with ``field`` in ``close`` (split-adjusted, not dividend-adjusted; step 1.1's audit price),
  ``adj_close`` (total-return, step 1.2; see ``etfcorr.data.sources``) and ``volume``. Git-ignored
  (vendor data is never committed; see ``.gitignore`` and EFT_project_v2.md section 4, item 1).
* ``data/raw/prices_yf.json`` -- the download log: for every ticker, the ``asof`` it was fetched
  to, row count, a SHA-256 of that ticker's own history and the fetch timestamp, plus a whole-
  panel SHA-256. Committed (``.gitignore`` keeps ``.json`` logs under ``data/raw/``): this is the
  reproducibility record the plan asks for (section 11), since the vendor revises adjusted prices
  retroactively and the raw bytes themselves cannot be committed.

Design choices, stated on purpose:

* ``asof`` must already be the last COMPLETE session (``etfcorr.data.calendars.last_session``);
  this module does not consult the calendar itself and does not default to "today" anywhere, so a
  caller mistake is visible in the log's ``asof`` field, not hidden inside this function (the
  standing caveat from step 1.1's Gate 2: an audit run mid-session picked up an unfinished bar).
  Every fetched history is defensively clipped to ``<= asof`` before it is hashed or written, so a
  vendor or fetcher bug that returns a later row cannot leak into the cache.
* Per-ticker incremental caching: a ticker already fetched up to at least ``asof`` is skipped
  (``force=True`` overrides this); a ticker whose fetch fails this run keeps whatever the
  previous cache held for it rather than being wiped, and is reported as failed. This is the
  "skip if fresh, primary-source-if-not, never destroy existing good data on a partial failure"
  behaviour project_plan.md section 6.3 asks of ``download.py``.
* Tickers no longer present in ``config/universe.yaml`` are dropped from both the panel and the
  log on the next run (no stale leftovers silently kept forever).

Assumptions and known failure modes (standing caveats, unchanged from docs/STATUS.md):

* Survivorship and rate-limiting both show up as a fetch failure or ``no_data``; this module
  cannot tell them apart (same limitation as step 1.1's audit).
* Crypto (``*-USD``) "sessions" are not calendar-aligned here; see ``calendars.is_24_7`` and its
  module-level caveat on asynchronous closes.
* The adjusted (``adj_close``) series is retroactively revised by the vendor after corporate
  actions: the per-ticker hash changes on a re-download even when nothing about *this* run's
  request changed, and that is expected, not a bug.

No third-party code is adapted here; the download-log pattern (timestamp, requested/received,
sha256) follows the sketch in EFT_project_v2.md section 4 [INSPIRE: idea only, rewritten for a
wide multi-ticker panel and per-ticker incremental caching, which the sketch does not have].
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from etfcorr.config import ConfigError, Project
from etfcorr.data.universe_check import FetchResult

Fetcher = Callable[[str, date, date], FetchResult]
Progress = Callable[[int, int, str], None]
Clock = Callable[[], datetime]
PanelReader = Callable[[Path], pd.DataFrame]
PanelWriter = Callable[[pd.DataFrame, Path], None]

CACHE_NAME = "prices_yf.parquet"
LOG_NAME = "prices_yf.json"
FIELDS: tuple[str, ...] = ("close", "volume", "adj_close")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def read_panel_parquet(path: Path) -> pd.DataFrame:
    """Default panel reader.

    STATUS: NOT EXECUTED OFFLINE. The planning sandbox has neither ``pyarrow`` nor
    ``fastparquet`` installed (PyPI unreachable there), so ``pandas.read_parquet`` cannot run in
    Gate 1; every other function in this module is isolated from this one behind the injectable
    ``read_panel``/``write_panel`` parameters of :func:`download_universe` and is fully tested
    offline against a pickle-based stand-in (``tests/unit/test_download.py``). ``pyarrow`` is a
    core dependency (``pyproject.toml``), so the real Parquet round-trip is exercised at Gate 2.
    """
    return pd.read_parquet(path)


def write_panel_parquet(panel: pd.DataFrame, path: Path) -> None:
    """Default panel writer; see :func:`read_panel_parquet` for why this is NOT EXECUTED OFFLINE."""
    panel.to_parquet(path)


@dataclass(frozen=True)
class CacheEntry:
    """One ticker's row in the download log (what the *previous* run recorded, or the new one)."""

    ticker: str
    asof: date
    rows: int
    sha256: str
    fetched_at: str
    status: str  # "fetched" | "failed"
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "asof": self.asof.isoformat(),
            "rows": self.rows,
            "sha256": self.sha256,
            "fetched_at": self.fetched_at,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_json(cls, ticker: str, data: dict[str, Any]) -> "CacheEntry":
        return cls(
            ticker=ticker,
            asof=date.fromisoformat(data["asof"]),
            rows=int(data["rows"]),
            sha256=str(data["sha256"]),
            fetched_at=str(data["fetched_at"]),
            status=str(data["status"]),
            error=str(data.get("error", "")),
        )


@dataclass(frozen=True)
class TickerOutcome:
    """What happened to one ticker in *this* run: fresh cache hit, fetched, or failed."""

    ticker: str
    status: str  # "cached" | "fetched" | "failed"
    rows: int = 0
    error: str = ""


@dataclass(frozen=True)
class DownloadReport:
    asof: date
    start: date
    fetched_at: str
    outcomes: tuple[TickerOutcome, ...]
    sha256: str
    panel_path: Path
    log_path: Path

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(o.ticker for o in self.outcomes if o.status == "failed")

    @property
    def fetched(self) -> tuple[str, ...]:
        return tuple(o.ticker for o in self.outcomes if o.status == "fetched")

    @property
    def cached(self) -> tuple[str, ...]:
        return tuple(o.ticker for o in self.outcomes if o.status == "cached")


def _load_log(path: Path) -> dict[str, CacheEntry]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{path}: unreadable download log: {exc}") from exc
    tickers = raw.get("tickers", {})
    if not isinstance(tickers, dict):
        raise ConfigError(f"{path}: 'tickers' must be a JSON object")
    return {t: CacheEntry.from_json(t, e) for t, e in tickers.items()}


def _load_panel(path: Path, read_panel: PanelReader) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    panel = read_panel(path)
    if not isinstance(panel.columns, pd.MultiIndex):
        raise ConfigError(f"{path}: expected a MultiIndex-column panel, found {panel.columns!r}")
    return panel


def _tickers_in(panel: pd.DataFrame) -> frozenset[str]:
    if not isinstance(panel.columns, pd.MultiIndex):
        return frozenset()
    return frozenset(panel.columns.get_level_values("ticker"))


def _needs_fetch(ticker: str, asof: date, log: dict[str, CacheEntry], *, force: bool) -> bool:
    if force:
        return True
    prev = log.get(ticker)
    return prev is None or prev.status != "fetched" or prev.asof < asof


def _clip_to_asof(history: pd.DataFrame, asof: date) -> pd.DataFrame:
    """Defensive no-lookahead guard: never let a row dated after ``asof`` enter the cache."""
    return history.loc[history.index <= pd.Timestamp(asof)]


def _hash_frame(frame: pd.DataFrame) -> str:
    """Deterministic SHA-256 of a frame's content, independent of row/column order.

    Follows the ``sha256(px.to_csv().encode())`` pattern of EFT_project_v2.md section 4
    [INSPIRE], sorting both axes first so the hash does not depend on dict/iteration order.
    """
    ordered = frame if frame.empty else frame.sort_index(axis=0).sort_index(axis=1)
    return hashlib.sha256(ordered.to_csv().encode("utf-8")).hexdigest()


def _wide_panel(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble a ``(ticker, field)``-column panel from single-ticker ``close``/``volume``/
    ``adj_close`` frames, in the fixed column order ``FIELDS`` for every ticker."""
    if not histories:
        return pd.DataFrame()
    frames = {
        ticker: pd.DataFrame({f: hist[f] for f in FIELDS if f in hist.columns})
        for ticker, hist in histories.items()
    }
    panel = pd.concat(frames, axis=1).sort_index()
    panel.columns = panel.columns.set_names(["ticker", "field"])
    return panel


def download_universe(
    project: Project,
    fetch: Fetcher,
    *,
    asof: date,
    force: bool = False,
    progress: Progress | None = None,
    now: Clock = _utc_now,
    read_panel: PanelReader = read_panel_parquet,
    write_panel: PanelWriter = write_panel_parquet,
) -> DownloadReport:
    """Fetch every universe ticker up to ``asof`` (a COMPLETE session) and cache it to Parquet.

    ``asof`` is taken as given: this function does not call ``calendars.last_session`` itself
    (the caller resolves "today" to the last complete session, keeping that decision visible and
    testable at the call site rather than defaulted away here). Raises ``ConfigError`` if
    ``asof`` is before the universe's configured ``start_date``.

    ``read_panel``/``write_panel`` default to Parquet (see :func:`read_panel_parquet`); tests
    inject a stand-in so this function's own logic is fully tested offline without ``pyarrow``.
    """
    start = project.universe.start_date
    if asof < start:
        raise ConfigError(f"asof {asof} is before the universe start_date {start}")

    panel_path = project.paths.data_raw / CACHE_NAME
    log_path = project.paths.data_raw / LOG_NAME
    prev_log = _load_log(log_path)
    prev_panel = _load_panel(panel_path, read_panel)
    prev_tickers = _tickers_in(prev_panel)

    tickers = [t.ticker for t in project.universe.tickers]
    fetched_at = now().isoformat()

    histories: dict[str, pd.DataFrame] = {}
    outcomes: list[TickerOutcome] = []
    new_log: dict[str, CacheEntry] = {t: e for t, e in prev_log.items() if t in tickers}

    def _carry_forward(ticker: str) -> None:
        if ticker in prev_tickers:
            # prev_tickers is only non-empty when _load_panel already confirmed a MultiIndex
            # (ticker, field) column panel, so this is a DataFrame cross-section at runtime;
            # pandas-stubs' DataFrame.__getitem__ overloads don't special-case MultiIndex
            # columns and type a plain str key as returning Series[Any] regardless.
            histories[ticker] = cast(pd.DataFrame, prev_panel[ticker])

    for i, ticker in enumerate(tickers, start=1):
        if progress is not None:
            progress(i, len(tickers), ticker)
        if not _needs_fetch(ticker, asof, prev_log, force=force):
            cached = prev_log[ticker]
            outcomes.append(TickerOutcome(ticker, "cached", rows=cached.rows))
            _carry_forward(ticker)
            continue
        try:
            result: FetchResult | None = fetch(ticker, start, asof)
            error = ""
        except Exception as exc:  # network boundary: record and continue (as step 1.1)
            result, error = None, f"{type(exc).__name__}: {exc}"[:300]
        history = None if result is None else result.history
        if error or history is None or len(history) == 0:
            message = error or "no_data"
            outcomes.append(TickerOutcome(ticker, "failed", error=message))
            new_log[ticker] = CacheEntry(ticker, asof, 0, "", fetched_at, "failed", message)
            _carry_forward(ticker)  # a failed refresh keeps the last good data, if any
            continue
        clipped = _clip_to_asof(history, asof)
        histories[ticker] = clipped
        digest = _hash_frame(clipped)
        new_log[ticker] = CacheEntry(ticker, asof, len(clipped), digest, fetched_at, "fetched")
        outcomes.append(TickerOutcome(ticker, "fetched", rows=len(clipped)))

    panel = _wide_panel(histories)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    write_panel(panel, panel_path)
    whole_sha = _hash_frame(panel)
    log_path.write_text(
        json.dumps(
            {
                "asof": asof.isoformat(),
                "start": start.isoformat(),
                "fetched_at": fetched_at,
                "requested": tickers,
                "received": sorted(histories.keys()),
                "sha256": whole_sha,
                "tickers": {t: e.to_json() for t, e in new_log.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return DownloadReport(
        asof=asof,
        start=start,
        fetched_at=fetched_at,
        outcomes=tuple(outcomes),
        sha256=whole_sha,
        panel_path=panel_path,
        log_path=log_path,
    )


def summary_lines(report: DownloadReport) -> list[str]:
    """Human-readable report for the CLI: what was fetched, cached, or failed, and where."""
    lines = [
        f"download as of {report.asof} (start {report.start}, {len(report.outcomes)} tickers)",
        f"  fetched {len(report.fetched)}, cached {len(report.cached)}, "
        f"failed {len(report.failed)}",
        f"  panel:  {report.panel_path}",
        f"  log:    {report.log_path}",
        f"  sha256: {report.sha256[:12]}...",
    ]
    if report.failed:
        lines.append(f"  failed: {', '.join(report.failed)}")
    return lines
