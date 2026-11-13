"""Step 1.3a: data-quality gate on the cached price panel (``download.py``'s output).

Question served: is the panel `download_universe` wrote safe to turn into the analysis panel
(``build_panel.py``), or does it have a structural problem that must stop the pipeline rather
than be silently propagated. This is the checklist named in ``project_plan.md`` section 6.3 and
``EFT_project_v2.md`` section 4's "Validation checklist (fail loudly, run in CI on a small
fixture)": all tickers received; first valid date per ticker; % missing vs the exchange calendar;
duplicate/non-monotonic index; non-positive prices; returns beyond +-5 sigma flagged; zero-volume
days; ADV liquidity filter; **max abs diff vs a second source; panel end date equals the last
complete session** (staleness assertion).

Deviation from that checklist, stated on purpose (project instructions: say so explicitly rather
than silently drop it): **"max abs diff vs a second source" is NOT implemented.**
``etfcorr.data.sources`` only has a yfinance fetcher (steps 1.1/1.2); a second source (Stooq, per
``project_plan.md`` section 6.3's `sources.py` row) was never built, so there is nothing to
cross-check against yet. This is a real coverage gap, not an oversight, and it means a
single-vendor bad tick that is internally consistent (positive price, no duplicate, within 5
sigma, not stale) can pass this gate undetected. Recorded as an open item in ``docs/STATUS.md``.

Design: two severities, matching "fails the pipeline on errors, warns on soft issues"
(``project_plan.md`` 6.3):

* **Errors** (collected in ``ValidationReport.errors``; :func:`assert_valid` raises
  :class:`DataQualityError` if any exist) -- problems that make the panel unsafe to build on:
  a universe ticker missing from the panel entirely; a duplicate or non-monotonic panel index;
  a non-positive ``close`` price; a ticker whose last valid ``close`` is short of the last
  complete session for its calendar (:func:`etfcorr.data.calendars.check_freshness`, reused
  as-is rather than re-implemented, so the 24/7-crypto and ``^VIX`` exemptions stay in exactly
  one place).
* **Warnings** (collected in ``ValidationReport.warnings``, and in each ticker's
  :class:`TickerReport`; never raise) -- informational quality signals: % of expected sessions
  missing within the ticker's own active window, days with a |log return| beyond
  ``extreme_sigma`` times that ticker's own sample standard deviation, the zero-volume-day share,
  and the recent median-dollar-volume liquidity screen. The liquidity/zero-volume *gating rule*
  (core/extended, ETF-like tickers only) intentionally mirrors
  ``etfcorr.data.universe_check._liquidity_gated`` rather than importing it (that helper is
  private to its module); if the rule ever changes, both places need the same edit -- flagged in
  ``docs/STATUS.md`` as a follow-up, not fixed here, to avoid touching a LOCAL-PASSED file
  (``universe_check.py``, step 1.1) mid-project for a cosmetic refactor.

Assumptions and known failure modes (standing caveats):

* "Missing vs the exchange calendar" is computed only over each ticker's own active window
  (its first to last valid ``close``); pre-inception NaNs from the wide panel's outer join are
  not gaps and are correctly excluded by construction, not by a special case.
* The +-5 sigma extreme-return check uses each ticker's *own* full-sample standard deviation, not
  a shared or regime-conditional one; a ticker with a genuine multi-year vol regime shift will
  flag more or fewer extreme days than a rolling-vol version would. Acceptable for a soft warning,
  not for a quantitative claim.
* This module never reads the network or the vendor library; it only operates on the
  already-cached panel `download.py` wrote (or, in tests, a synthetic one).

No third-party code is adapted here; nothing to record in THIRD_PARTY.md.
"""

from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from etfcorr.config import Project
from etfcorr.data.calendars import SessionsFetcher, check_freshness, is_24_7, nyse_sessions
from etfcorr.data.universe_check import is_etf_like

ADV_WINDOW = 252  # trading days; mirrors CheckParams.adv_window, step 1.1
ZERO_VOLUME_WARN = 0.01  # mirrors CheckParams.zero_volume_warn, step 1.1
EXTREME_SIGMA = 5.0  # project_plan.md 6.3: "returns beyond +-5 sigma flagged"


class DataQualityError(ValueError):
    """The panel failed one or more hard checks; unsafe to pass to build_panel.py."""


@dataclass(frozen=True)
class TickerReport:
    """One ticker's measurements. ``None`` fields mean "not computable" (e.g. no valid rows)."""

    ticker: str
    first_valid: date | None
    last_valid: date | None
    n_valid: int
    expected_sessions: int | None
    missing_pct: float | None
    extreme_return_dates: tuple[str, ...]
    zero_volume_pct: float | None
    median_dollar_volume: float | None
    is_stale: bool
    errors: tuple[str, ...] = ()

    def to_json(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "first_valid": None if self.first_valid is None else self.first_valid.isoformat(),
            "last_valid": None if self.last_valid is None else self.last_valid.isoformat(),
            "n_valid": self.n_valid,
            "expected_sessions": self.expected_sessions,
            "missing_pct": self.missing_pct,
            "extreme_return_dates": list(self.extreme_return_dates),
            "zero_volume_pct": self.zero_volume_pct,
            "median_dollar_volume": self.median_dollar_volume,
            "is_stale": self.is_stale,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class ValidationReport:
    asof: date
    tickers: dict[str, TickerReport]
    missing_tickers: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def assert_valid(report: ValidationReport) -> None:
    """Raise :class:`DataQualityError` joining every error, if the report has any."""
    if not report.ok:
        raise DataQualityError("; ".join(report.errors))


def _liquidity_gated(ticker: str, panel_membership: str) -> bool:
    """Mirrors ``universe_check._liquidity_gated``: core/extended, ETF-like tickers only."""
    return panel_membership in ("core", "extended") and is_etf_like(ticker)


def _expected_sessions(ticker: str, first: date, last: date, sessions_fn: SessionsFetcher) -> int:
    if first > last:
        return 0
    if is_24_7(ticker):
        return int((last - first).days) + 1
    return len(sessions_fn(first, last))


def _extreme_return_dates(close: pd.Series, sigma_mult: float) -> tuple[str, ...]:
    # A non-positive price is already a hard error elsewhere (_check_one); guard here too so a
    # ticker failing that check does not also emit a log(0) = -inf RuntimeWarning on its way out.
    valid = close.dropna()
    valid = valid.loc[valid > 0]
    if len(valid) < 3:
        return ()
    log_ret = np.log(valid).diff().dropna()
    sigma = float(log_ret.std(ddof=1))
    if not sigma or np.isnan(sigma):
        return ()
    flagged = log_ret.loc[log_ret.abs() > sigma_mult * sigma]
    return tuple(pd.Timestamp(d).date().isoformat() for d in flagged.index)


def _check_one(
    ticker: str,
    panel_membership: str | None,
    close: pd.Series,
    volume: pd.Series,
    asof: date,
    *,
    sessions_fn: SessionsFetcher,
    extreme_sigma: float,
) -> TickerReport:
    errors: list[str] = []
    if bool((close.dropna() <= 0).any()):
        errors.append(f"{ticker}: non-positive close price in the panel")

    first_idx = close.first_valid_index()
    last_idx = close.last_valid_index()
    if first_idx is None or last_idx is None:
        errors.append(f"{ticker}: no valid close price in the panel")
        return TickerReport(
            ticker=ticker,
            first_valid=None,
            last_valid=None,
            n_valid=0,
            expected_sessions=None,
            missing_pct=None,
            extreme_return_dates=(),
            zero_volume_pct=None,
            median_dollar_volume=None,
            is_stale=True,
            errors=tuple(errors),
        )

    # pandas-stubs types `first_valid_index()`/`last_valid_index()` as a broad Hashable-like
    # union even for a DatetimeIndex-backed Series, where the runtime value is always a
    # pandas.Timestamp; cast once here rather than re-wrapping in `pd.Timestamp(...)` below
    # (which mypy still rejects, since it does not narrow the source type either way).
    first_ts, last_ts = cast(pd.Timestamp, first_idx), cast(pd.Timestamp, last_idx)
    first, last = first_ts.date(), last_ts.date()
    # Boolean masking, not `.loc[first_idx:last_idx]`: label-slicing raises on a non-monotonic
    # index, and a scrambled index is exactly one of the structural errors this module detects
    # (see the panel-wide check above) -- it must not also crash the rest of this function.
    # `.sort_index()` keeps the diff-based extreme-return check chronological even then.
    in_window = (close.index >= first_ts) & (close.index <= last_ts)
    window_close = close.loc[in_window].sort_index()
    window_volume = volume.loc[in_window].sort_index()
    n_valid = int(window_close.notna().sum())
    expected = _expected_sessions(ticker, first, last, sessions_fn)
    missing_pct = None if expected == 0 else max(0.0, 1.0 - n_valid / expected)

    freshness = check_freshness(ticker, last, asof, sessions_fn)
    if freshness.is_stale:
        errors.append(
            f"{ticker}: last valid close {last} is before the expected last session "
            f"{freshness.expected_last} (calendar {freshness.calendar})"
        )

    zero_volume_pct = None
    median_dollar_volume = None
    vol_valid = window_volume.dropna()
    if len(vol_valid) > 0:
        # Denominator is days volume was actually reported, not the whole window: a day with no
        # volume figure at all (NaN -- e.g. a non-session day pulled in by another ticker's wider
        # calendar in the shared wide panel, or a genuine vendor gap) is "no data", not "traded
        # zero shares"; `.fillna(0)` would have silently conflated the two.
        zero_volume_pct = float((vol_valid == 0).mean())
        dollar = (window_close * window_volume).dropna()
        if len(dollar) > 0:
            median_dollar_volume = float(dollar.tail(ADV_WINDOW).median())

    return TickerReport(
        ticker=ticker,
        first_valid=first,
        last_valid=last,
        n_valid=n_valid,
        expected_sessions=expected,
        missing_pct=missing_pct,
        extreme_return_dates=_extreme_return_dates(window_close, extreme_sigma),
        zero_volume_pct=zero_volume_pct,
        median_dollar_volume=median_dollar_volume,
        is_stale=freshness.is_stale,
        errors=tuple(errors),
    )


def validate_panel(
    panel: pd.DataFrame,
    project: Project,
    asof: date,
    *,
    sessions_fn: SessionsFetcher = nyse_sessions,
    extreme_sigma: float = EXTREME_SIGMA,
    min_adv_usd: float | None = None,
) -> ValidationReport:
    """Run the checklist against ``panel`` (a `download.py`-shaped ``(ticker, field)`` panel).

    ``min_adv_usd`` defaults to ``project.universe.min_adv_usd``; overridable for tests. Every
    ticker in ``project.universe.tickers`` is checked; one absent from ``panel`` is a hard error
    (not silently skipped) and is not otherwise checked.
    """
    threshold = project.universe.min_adv_usd if min_adv_usd is None else min_adv_usd
    panel_tickers = (
        frozenset(panel.columns.get_level_values("ticker"))
        if isinstance(panel.columns, pd.MultiIndex)
        else frozenset()
    )

    errors: list[str] = []
    warnings: list[str] = []
    if isinstance(panel.columns, pd.MultiIndex) and panel.index.has_duplicates:
        errors.append("panel index has duplicate dates")
    if isinstance(panel.columns, pd.MultiIndex) and not panel.index.is_monotonic_increasing:
        errors.append("panel index is not sorted ascending")

    reports: dict[str, TickerReport] = {}
    missing = tuple(t.ticker for t in project.universe.tickers if t.ticker not in panel_tickers)
    for ticker in missing:
        errors.append(f"{ticker}: configured in universe.yaml but absent from the panel")

    for entry in project.universe.tickers:
        if entry.ticker not in panel_tickers:
            continue
        frame = panel[entry.ticker]
        close = pd.to_numeric(frame["close"], errors="coerce")
        volume = (
            pd.to_numeric(frame["volume"], errors="coerce")
            if "volume" in frame.columns
            else pd.Series(np.nan, index=frame.index)
        )
        report = _check_one(
            entry.ticker,
            entry.panel,
            close,
            volume,
            asof,
            sessions_fn=sessions_fn,
            extreme_sigma=extreme_sigma,
        )
        reports[entry.ticker] = report
        errors.extend(report.errors)

        if report.missing_pct is not None and report.missing_pct > 0:
            warnings.append(
                f"{entry.ticker}: {report.missing_pct:.1%} of expected sessions missing"
            )
        if report.extreme_return_dates:
            warnings.append(
                f"{entry.ticker}: {len(report.extreme_return_dates)} day(s) beyond "
                f"{extreme_sigma}-sigma log returns"
            )
        if is_etf_like(entry.ticker) and (report.zero_volume_pct or 0.0) >= ZERO_VOLUME_WARN:
            warnings.append(f"{entry.ticker}: {report.zero_volume_pct:.1%} zero-volume days")
        if (
            _liquidity_gated(entry.ticker, entry.panel)
            and report.median_dollar_volume is not None
            and report.median_dollar_volume < threshold
        ):
            warnings.append(
                f"{entry.ticker}: median dollar volume {report.median_dollar_volume:,.0f} "
                f"< {threshold:,.0f}"
            )

    return ValidationReport(
        asof=asof,
        tickers=reports,
        missing_tickers=missing,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def summary_lines(report: ValidationReport) -> list[str]:
    lines = [
        f"validation as of {report.asof}: {len(report.tickers)} tickers checked, "
        f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)"
    ]
    if report.missing_tickers:
        lines.append(f"  missing from panel: {', '.join(report.missing_tickers)}")
    for line in report.warnings:
        lines.append(f"  WARNING: {line}")
    for line in report.errors:
        lines.append(f"  ERROR: {line}")
    return lines
