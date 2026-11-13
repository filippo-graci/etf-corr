"""Step 1.3b: turn the validated cache into the analysis panel (``project_plan.md`` section 6.3).

Question served: give every later phase one panel, aligned to a single calendar, that never
silently fills in data that was not observed. Precondition: the raw panel has already passed
:func:`etfcorr.data.validate.validate_panel` (:func:`command_build_panel` enforces this itself,
raising before writing anything if it has not).

What "align to the NYSE calendar" means here, precisely: the output panel's index is exactly the
NYSE session calendar over ``[project.universe.start_date, asof]`` (via
:func:`etfcorr.data.calendars.nyse_sessions`, one shared calendar, not a per-ticker one). Every
ticker, including the 24/7 crypto pairs and ``^VIX``, is reindexed onto that same index with a
plain ``reindex`` -- no ``ffill``/``bfill``, so a date the ticker did not report (before its
inception, or a genuine gap) is ``NaN`` in the output, per project_plan.md's "no forward-fill of
pre-inception history". This is also, in the loose and already-documented sense from step 1.1/1.2
[Unverified: Yahoo's crypto daily-bar close time], how "sample BTC-USD at the US close" is
implemented: a crypto pair's own daily bar is simply selected at each NYSE session date, not
resampled or interpolated.

Deviation from the ``project_plan.md`` 6.3 spec, stated on purpose (same standing gap as
``validate.py``): **"merge sources (prefer primary, patch gaps only if the second source
agrees)" is a no-op here.** ``sources.py`` has only one vendor (yfinance); there is no second
source to merge or patch from. The raw panel `download.py` wrote is passed through unmodified
except for the calendar reindex.

``coverage.csv`` columns (project_plan.md 6.3: "first date, % missing, panel membership"):
``ticker``, ``sector``, ``panel`` (core/extended/control), ``first_date``, ``last_date``,
``n_valid``, ``expected_sessions``, ``missing_pct`` -- all computed on the ALIGNED output panel
(not the raw one), so ``missing_pct`` describes exactly what a reader of ``prices_clean.parquet``
will see, not an intermediate.

No third-party code is adapted here; nothing to record in THIRD_PARTY.md.
"""

from datetime import date
from pathlib import Path
from typing import cast

import pandas as pd

from etfcorr.config import ConfigError, Project
from etfcorr.data.calendars import SessionsFetcher, nyse_sessions
from etfcorr.data.download import (
    CACHE_NAME,
    PanelReader,
    PanelWriter,
    read_panel_parquet,
    write_panel_parquet,
)
from etfcorr.data.validate import DataQualityError, summary_lines, validate_panel

CLEAN_NAME = "prices_clean.parquet"
COVERAGE_NAME = "coverage.csv"


def write_coverage_csv(coverage: pd.DataFrame, path: Path) -> None:
    """Plain ``pandas.to_csv``; unlike Parquet this needs no optional engine, so it *is* run
    offline (see ``tests/unit/test_build_panel.py``), not isolated behind an injectable seam."""
    coverage.to_csv(path, index=True)


def _coverage_row(entry_ticker: str, sector: str, panel_name: str, series: pd.Series) -> dict:
    first_idx = series.first_valid_index()
    last_idx = series.last_valid_index()
    if first_idx is None or last_idx is None:
        return {
            "ticker": entry_ticker,
            "sector": sector,
            "panel": panel_name,
            "first_date": None,
            "last_date": None,
            "n_valid": 0,
            "expected_sessions": 0,
            "missing_pct": None,
        }
    window = series.loc[first_idx:last_idx]
    expected = len(window)
    n_valid = int(window.notna().sum())
    # Same pandas-stubs gap as validate.py: first_valid_index()/last_valid_index() are typed as
    # a broad Hashable-like union even for a DatetimeIndex-backed Series, where the runtime value
    # is always a pandas.Timestamp.
    first_ts, last_ts = cast(pd.Timestamp, first_idx), cast(pd.Timestamp, last_idx)
    return {
        "ticker": entry_ticker,
        "sector": sector,
        "panel": panel_name,
        "first_date": first_ts.date().isoformat(),
        "last_date": last_ts.date().isoformat(),
        "n_valid": n_valid,
        "expected_sessions": expected,
        "missing_pct": max(0.0, 1.0 - n_valid / expected) if expected else None,
    }


def build_panel(
    panel: pd.DataFrame,
    project: Project,
    asof: date,
    *,
    sessions_fn: SessionsFetcher = nyse_sessions,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reindex every configured ticker onto the shared NYSE session calendar.

    Pure given ``panel`` and ``sessions_fn``. Tickers absent from ``panel`` (already reported as
    an error by ``validate_panel``) are silently skipped here -- this function does not
    re-validate, it assumes ``command_build_panel`` already refused to reach it otherwise.
    """
    start = project.universe.start_date
    sessions = sessions_fn(start, asof)
    if len(sessions) == 0:
        raise ConfigError(f"no NYSE sessions in [{start}, {asof}]; cannot align the panel")
    aligned_index = pd.DatetimeIndex(sessions).sort_values()

    panel_tickers = (
        frozenset(panel.columns.get_level_values("ticker"))
        if isinstance(panel.columns, pd.MultiIndex)
        else frozenset()
    )

    clean_frames: dict[str, pd.DataFrame] = {}
    coverage_rows: list[dict] = []
    for entry in project.universe.tickers:
        if entry.ticker not in panel_tickers:
            continue
        # panel_tickers is only non-empty when panel.columns is confirmed a MultiIndex above, so
        # this is a DataFrame cross-section at runtime; pandas-stubs' DataFrame.__getitem__
        # overloads don't special-case MultiIndex columns and type a plain str key as
        # Series[Any] regardless (same gap fixed in download.py's _carry_forward).
        raw = cast(pd.DataFrame, panel[entry.ticker])
        clean = raw.reindex(aligned_index)
        clean_frames[entry.ticker] = clean
        coverage_rows.append(_coverage_row(entry.ticker, entry.sector, entry.panel, clean["close"]))

    if not clean_frames:
        raise ConfigError("no tickers to build a panel from (all missing from the raw panel)")

    clean_panel = pd.concat(clean_frames, axis=1)
    clean_panel.columns = clean_panel.columns.set_names(["ticker", "field"])
    coverage = pd.DataFrame(coverage_rows).set_index("ticker")
    return clean_panel, coverage


def command_build_panel(
    project: Project,
    *,
    asof: date,
    sessions_fn: SessionsFetcher = nyse_sessions,
    read_panel: PanelReader = read_panel_parquet,
    write_clean: PanelWriter = write_panel_parquet,
    write_coverage: PanelWriter = write_coverage_csv,
) -> list[str]:
    """Validate the raw cache, then build and write the clean panel + coverage table.

    Raises ``ConfigError`` if the raw cache is missing, ``DataQualityError`` (message = the full
    validation report, both warnings and errors) if validation finds a hard error -- nothing is
    written in that case. Warnings alone do not block the write; they are included in the
    returned summary lines either way.
    """
    raw_path = project.paths.data_raw / CACHE_NAME
    if not raw_path.is_file():
        raise ConfigError(f"{raw_path} not found; run `etfcorr data download` first")
    panel = read_panel(raw_path)

    report = validate_panel(panel, project, asof, sessions_fn=sessions_fn)
    if not report.ok:
        raise DataQualityError("\n".join(summary_lines(report)))

    clean, coverage = build_panel(panel, project, asof, sessions_fn=sessions_fn)
    clean_path = project.paths.data_interim / CLEAN_NAME
    coverage_path = project.paths.data_processed / COVERAGE_NAME
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    write_clean(clean, clean_path)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    write_coverage(coverage, coverage_path)

    return [
        *summary_lines(report),
        f"  clean panel: {clean_path}",
        f"  coverage:    {coverage_path}",
    ]
