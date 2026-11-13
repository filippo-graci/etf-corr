"""Step 1.1: audit the candidate ETF universe against a price vendor.

Question served: which of the candidate tickers in ``config/universe.yaml`` (EFT_project_v2.md
section 3) can enter the panels, and how much history do the panels really share? This is a data
audit, not a hypothesis test; it feeds every later phase through the universe it fixes.

What is computed per ticker, from daily ``close``/``volume`` between ``start`` and ``asof``:

* existence and freshness: first and last date, number of observations, staleness of the last
  date (a delisted, merged or renamed fund stops updating);
* index integrity (unique, increasing), non-positive prices, calendar gaps of more than a week;
* liquidity: the *median* daily dollar volume (close x volume) over the last ``adv_window``
  sessions, gated against ``min_adv_usd`` for ETF-like core/extended tickers, and over the
  training window (informational). Zero-volume days are kept in the median (a session with no
  trading is information about illiquidity, dropping it would bias the estimate upwards);
* the vendor's own name and instrument type, printed next to the configured name so that
  renames and mislabelled tickers are reviewed by a human (no fuzzy matching is attempted).

Across tickers: the *common sample start* of a panel (the latest first date among its members)
and, for the latest-starting members, how far the common start would move if that member were
dropped. The unbalanced-panel problem of EFT_project_v2.md section 4, item 4, is decided from
these numbers, not assumed.

Deviations from the plan, stated on purpose: (1) EFT_project_v2.md speaks of *average* daily
dollar volume; this module uses the *median*, which is robust to single-day spikes [Inference,
not a cited choice]. (2) AUM is not gated: vendor AUM figures are unreliable and are not fetched.

Assumptions and known failure modes (all are standing caveats of the project):

* Survivorship: a vendor that drops delisted funds returns nothing for them, so the audit sees
  only survivors; ``no_data`` may mean "delisted" or "rate-limited" and must be checked by hand.
* Dollar volume uses the vendor's split-adjusted close and volume, not NAV or creation-unit
  data; for indices (``^VIX``) and crypto pairs (``*-USD``) volume is not comparable and no
  liquidity gate is applied.
* ``first_date`` is the first row on or after ``start``: for a fund older than ``start`` it is
  the download start, not the inception date.
* Calendar gaps are judged without an exchange calendar (that arrives with ``calendars.py``):
  a gap of more than ``max_gap_days`` calendar days is flagged for review, not proven wrong.

Everything here is a function of the rows in ``[start, asof]``: rows after ``asof`` are ignored,
so the result cannot depend on the future of the audit date (tested). The network is touched only
through the injected ``Fetcher`` (see ``etfcorr.data.sources``); this module never imports a
vendor library. No third-party code is adapted and no citation is needed.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from etfcorr.config import ConfigError, Panel, Project, UniverseConfig, UniverseEntry

COLUMNS: tuple[str, ...] = (
    "ticker",
    "panel",
    "sector",
    "verdict",
    "flags",
    "first_date",
    "last_date",
    "n_obs",
    "adv_recent_usd",
    "adv_train_usd",
    "zero_volume_share",
    "max_gap_days",
    "quote_type",
    "vendor_name",
    "config_name",
    "error",
)

# Flag -> severity. "blocking" makes the verdict "fail", "review" makes it "review".
FLAG_SEVERITY: dict[str, str] = {
    "no_data": "blocking",
    "fetch_error": "blocking",
    "bad_history": "blocking",
    "stale": "blocking",
    "index_not_unique_increasing": "blocking",
    "nonpositive_price": "blocking",
    "low_liquidity": "review",
    "volume_missing": "review",
    "late_start": "review",
    "gap_over_limit": "review",
    "not_an_etf": "review",
    "late_start_extended": "info",
    "thin_in_train": "info",
    "zero_volume_days": "info",
    "volume_not_available": "info",
}

ETF_QUOTE_TYPES: frozenset[str] = frozenset({"ETF"})
_PANELS: tuple[Panel, ...] = ("core", "extended", "control")


class FetchResult(NamedTuple):
    """What a fetcher returns for one ticker.

    ``history`` has a tz-naive, day-normalised ``DatetimeIndex`` and the columns ``close`` and
    ``volume`` (``None`` or empty means the vendor returned nothing).
    """

    history: pd.DataFrame | None
    vendor_name: str | None = None
    quote_type: str | None = None


# (ticker, start, end inclusive) -> FetchResult. May raise; the runner records the error.
Fetcher = Callable[[str, date, date], FetchResult]
Progress = Callable[[int, int, str], None]


@dataclass(frozen=True)
class CheckParams:
    """Screening parameters; ``from_project`` reads them from the validated configs."""

    asof: date
    start: date
    train_start: date
    train_end: date
    min_adv_usd: float
    adv_window: int = 252
    max_stale_days: int = 5
    max_gap_days: int = 7
    zero_volume_warn: float = 0.01
    late_start_tolerance_days: int = 7

    def __post_init__(self) -> None:
        if not self.start < self.asof:
            raise ValueError(f"start {self.start} must be before asof {self.asof}")
        if not self.train_start < self.train_end:
            raise ValueError("train_start must be before train_end")
        if self.min_adv_usd <= 0:
            raise ValueError("min_adv_usd must be positive")
        if self.adv_window < 1:
            raise ValueError("adv_window must be at least 1")
        if self.max_stale_days < 0 or self.max_gap_days < 1 or self.late_start_tolerance_days < 0:
            raise ValueError("stale, gap and tolerance limits must be non-negative (gap >= 1)")
        if not 0.0 <= self.zero_volume_warn <= 1.0:
            raise ValueError("zero_volume_warn must lie in [0, 1]")

    @classmethod
    def from_project(cls, project: Project, asof: date) -> "CheckParams":
        train = project.experiment.splits.train
        return cls(
            asof=asof,
            start=project.universe.start_date,
            train_start=train.start,
            train_end=train.end,
            min_adv_usd=float(project.universe.min_adv_usd),
        )


@dataclass(frozen=True)
class TickerCheck:
    """One audited ticker; ``to_row`` follows ``COLUMNS``."""

    ticker: str
    panel: str
    sector: str
    flags: tuple[str, ...]
    first_date: date | None = None
    last_date: date | None = None
    n_obs: int = 0
    adv_recent_usd: float = float("nan")
    adv_train_usd: float = float("nan")
    zero_volume_share: float = float("nan")
    max_gap_days: int = 0
    quote_type: str | None = None
    vendor_name: str | None = None
    config_name: str = ""
    error: str = ""

    @property
    def verdict(self) -> str:
        severities = {FLAG_SEVERITY[f] for f in self.flags}
        if "blocking" in severities:
            return "fail"
        return "review" if "review" in severities else "ok"

    def to_row(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "panel": self.panel,
            "sector": self.sector,
            "verdict": self.verdict,
            "flags": ";".join(self.flags),
            "first_date": self.first_date,
            "last_date": self.last_date,
            "n_obs": self.n_obs,
            "adv_recent_usd": self.adv_recent_usd,
            "adv_train_usd": self.adv_train_usd,
            "zero_volume_share": self.zero_volume_share,
            "max_gap_days": self.max_gap_days,
            "quote_type": self.quote_type,
            "vendor_name": self.vendor_name,
            "config_name": self.config_name,
            "error": self.error,
        }


def is_etf_like(ticker: str) -> bool:
    """False for index (``^...``) and crypto-pair (``...-USD``) Yahoo series."""
    return not (ticker.startswith("^") or ticker.endswith("-USD"))


def _liquidity_gated(entry: UniverseEntry) -> bool:
    return entry.panel in ("core", "extended") and is_etf_like(entry.ticker)


def _make(
    entry: UniverseEntry, result: FetchResult | None, flags: list[str], **stats: Any
) -> TickerCheck:
    """Build the row for ``entry``; ``stats`` are the measured fields of ``TickerCheck``."""
    return TickerCheck(
        ticker=entry.ticker,
        panel=entry.panel,
        sector=entry.sector,
        flags=tuple(flags),
        quote_type=None if result is None else result.quote_type,
        vendor_name=None if result is None else result.vendor_name,
        config_name=entry.name,
        **stats,
    )


def _clean_frame(history: object, params: CheckParams) -> pd.DataFrame | None:
    """Rows in ``[start, asof]`` with a close price; ``None`` if the structure is unusable."""
    if not isinstance(history, pd.DataFrame):
        return None
    if not {"close", "volume"} <= set(history.columns):
        return None
    if not isinstance(history.index, pd.DatetimeIndex) or history.index.tz is not None:
        return None
    frame = pd.DataFrame(
        {
            "close": pd.to_numeric(history["close"], errors="coerce"),
            "volume": pd.to_numeric(history["volume"], errors="coerce"),
        },
        index=history.index,
    )
    inside = (frame.index >= pd.Timestamp(params.start)) & (
        frame.index <= pd.Timestamp(params.asof)
    )
    frame = frame.loc[inside]
    return frame.loc[frame["close"].notna()]


def check_history(
    entry: UniverseEntry,
    result: FetchResult | None,
    params: CheckParams,
    *,
    error: str = "",
) -> TickerCheck:
    """Audit one ticker. Pure: depends only on its arguments (rows after ``asof`` are ignored)."""
    if error:
        return _make(entry, None, ["fetch_error"], error=error)
    if result is None or result.history is None or len(result.history) == 0:
        return _make(entry, result, ["no_data"])

    frame = _clean_frame(result.history, params)
    if frame is None:
        return _make(entry, result, ["bad_history"])
    if len(frame) == 0:
        return _make(entry, result, ["no_data"])

    flags: list[str] = []
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        flags.append("index_not_unique_increasing")
        frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    if bool((frame["close"] <= 0).any()):
        flags.append("nonpositive_price")

    days = frame.index.to_numpy().astype("datetime64[D]")
    first = pd.Timestamp(days[0]).date()
    last = pd.Timestamp(days[-1]).date()
    max_gap = int(np.diff(days).astype(int).max()) if len(days) > 1 else 0
    if (params.asof - last).days > params.max_stale_days:
        flags.append("stale")
    if max_gap > params.max_gap_days:
        flags.append("gap_over_limit")

    if (first - params.train_start).days > params.late_start_tolerance_days:
        flags.append("late_start_extended" if entry.panel == "extended" else "late_start")

    dollar = (frame["close"] * frame["volume"]).dropna()
    volume = frame["volume"].dropna()
    adv_recent = float(dollar.iloc[-params.adv_window :].median()) if len(dollar) else float("nan")
    train_mask = (dollar.index >= pd.Timestamp(params.train_start)) & (
        dollar.index <= pd.Timestamp(params.train_end)
    )
    in_train = dollar.loc[train_mask]
    adv_train = float(in_train.median()) if len(in_train) else float("nan")
    zero_share = float((volume == 0).mean()) if len(volume) else float("nan")

    gated = _liquidity_gated(entry)
    if len(volume) == 0 or bool((volume == 0).all()):
        # no volume at all: unknown liquidity is a problem for a gated ETF, expected for an index
        flags.append("volume_missing" if gated else "volume_not_available")
    else:
        if zero_share >= params.zero_volume_warn and is_etf_like(entry.ticker):
            flags.append("zero_volume_days")
        if gated and adv_recent < params.min_adv_usd:
            flags.append("low_liquidity")
        if gated and adv_train < params.min_adv_usd:
            flags.append("thin_in_train")
    if gated and result.quote_type is not None and result.quote_type.upper() not in ETF_QUOTE_TYPES:
        flags.append("not_an_etf")

    return _make(
        entry,
        result,
        flags,
        first_date=first,
        last_date=last,
        n_obs=len(frame),
        adv_recent_usd=adv_recent,
        adv_train_usd=adv_train,
        zero_volume_share=zero_share,
        max_gap_days=max_gap,
    )


def run_universe_check(
    universe: UniverseConfig,
    fetch: Fetcher,
    params: CheckParams,
    *,
    progress: Progress | None = None,
) -> pd.DataFrame:
    """Fetch and audit every ticker; one row per ticker in file order, columns ``COLUMNS``.

    A fetcher that raises does not stop the run: the exception is recorded as ``fetch_error``
    (``KeyboardInterrupt`` and other non-``Exception`` signals still propagate).
    """
    entries = universe.tickers
    rows: list[dict[str, object]] = []
    for i, entry in enumerate(entries, start=1):
        if progress is not None:
            progress(i, len(entries), entry.ticker)
        try:
            result: FetchResult | None = fetch(entry.ticker, params.start, params.asof)
            error = ""
        except Exception as exc:  # network boundary: record and continue
            result, error = None, f"{type(exc).__name__}: {exc}"[:300]
        rows.append(check_history(entry, result, params, error=error).to_row())
    out = pd.DataFrame(rows, columns=list(COLUMNS))
    for col in ("first_date", "last_date"):
        out[col] = pd.to_datetime(out[col])
    return out


# --------------------------------------------------------------------------------------
# Panel-level summary: common sample start and what drives it
# --------------------------------------------------------------------------------------


def common_sample_start(first_dates: pd.Series) -> pd.Timestamp | None:
    """Latest first date among the members (``None`` if no member has data)."""
    valid = first_dates.dropna()
    return None if len(valid) == 0 else pd.Timestamp(valid.max())


def start_if_dropped(first_dates: pd.Series) -> dict[str, pd.Timestamp | None]:
    """For each member, the common start of the *other* members (``None`` if none remain)."""
    items = [(str(t), pd.Timestamp(v)) for t, v in first_dates.dropna().items()]
    out: dict[str, pd.Timestamp | None] = {}
    for ticker, _ in items:
        rest = [v for t, v in items if t != ticker]
        out[ticker] = max(rest) if rest else None
    return out


def _fmt_date(value: pd.Timestamp | None) -> str:
    if value is None or bool(pd.isna(value)):
        return "n/a"
    return str(pd.Timestamp(value).date())


def summary_lines(checks: pd.DataFrame, params: CheckParams, *, top: int = 6) -> list[str]:
    """Human-readable report: verdict counts, panel start dates, what needs a decision."""
    lines = [
        f"universe check as of {params.asof} (rows {params.start}..{params.asof}, "
        f"train start {params.train_start}, min median ADV {params.min_adv_usd:,.0f} USD)",
        "  "
        + ", ".join(f"{p} {int((checks['panel'] == p).sum())}" for p in _PANELS)
        + f", total {len(checks)}",
        "  verdicts: "
        + ", ".join(f"{v} {int((checks['verdict'] == v).sum())}" for v in ("ok", "review", "fail")),
    ]
    with_data = checks.loc[checks["n_obs"] > 0]
    n_names = int(with_data["vendor_name"].notna().sum())
    n_types = int(with_data["quote_type"].notna().sum())
    lines.append(
        f"  vendor metadata: name for {n_names} of {len(with_data)} tickers with data, "
        f"instrument type for {n_types} of {len(with_data)}"
    )
    if n_names < len(with_data) or n_types < len(with_data):
        lines.append(
            "  WARNING: vendor metadata is incomplete; the rename and instrument-type "
            "review is incomplete for the tickers without it"
        )
    usable = checks.loc[checks["verdict"] != "fail"]
    for panel in ("core", "extended"):
        members = usable.loc[usable["panel"] == panel]
        firsts = pd.Series(members["first_date"].to_numpy(), index=members["ticker"].to_numpy())
        start = common_sample_start(firsts)
        if start is None:
            lines.append(f"  {panel}: no usable member")
            continue
        limit = pd.Timestamp(params.train_start) + pd.Timedelta(
            days=params.late_start_tolerance_days
        )
        covering = int((members["first_date"] <= limit).sum())
        lines.append(
            f"  {panel}: {len(members)} usable, common sample starts {_fmt_date(start)} "
            f"({covering} of {len(members)} cover the train start {params.train_start})"
        )
        dropped = start_if_dropped(firsts)
        latest = firsts.dropna().sort_values(ascending=False).head(top)
        for ticker, first in latest.items():
            lines.append(
                f"    {ticker!s:<9} first {_fmt_date(first)}; "
                f"common start if dropped: {_fmt_date(dropped[str(ticker)])}"
            )
    for verdict in ("fail", "review"):
        bad = checks.loc[checks["verdict"] == verdict]
        if len(bad):
            lines.append(f"  {verdict}:")
            for _, row in bad.iterrows():
                lines.append(f"    {row['ticker']:<9} {row['panel']:<8} {row['flags']}")
    return lines


# --------------------------------------------------------------------------------------
# Command implementation (typer-free so that it is testable without the CLI library)
# --------------------------------------------------------------------------------------

DEFAULT_OUTPUT = ("data", "interim", "universe_check.csv")


def parse_iso_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ConfigError(f"not an ISO date (YYYY-MM-DD): {text!r}") from exc


def command_check_universe(
    project: Project,
    *,
    asof: date,
    fetch: Fetcher,
    out: str | None = None,
    progress: Progress | None = None,
) -> list[str]:
    """Run the audit for a loaded project, write the CSV, return the report lines.

    ``out`` is a path relative to the repository root and must lie under ``data/``,
    ``artifacts/`` or ``reports/`` (``Paths.output_path`` refuses anything else).
    """
    # Validate everything that can fail cheaply *before* the (slow, network-bound) download.
    target = project.paths.output_path(*(DEFAULT_OUTPUT if out is None else (out,)))
    try:
        params = CheckParams.from_project(project, asof)
    except ValueError as exc:
        raise ConfigError(f"invalid audit parameters: {exc}") from exc
    checks = run_universe_check(project.universe, fetch, params, progress=progress)
    target.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(target, index=False)
    return [*summary_lines(checks, params), f"  written: {target}"]
