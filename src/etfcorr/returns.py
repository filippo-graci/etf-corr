"""Step 2.1: log returns at each configured horizon, from adjusted (total-return) prices.

Question served: every later phase (classical correlation in this phase; shrinkage/RMT in
Phase 3; regimes in Phase 4; features/targets from Phase 5 on) consumes returns, not prices.
This module is the single place that turns ``prices_clean.parquet`` (Phase 1) into
``data/processed/returns_<H>.parquet`` for each horizon in ``config/experiment.yaml``, so every
downstream module reads the same, once-computed series instead of re-deriving it.

**(A) sampling horizon vs (B) estimation window** (project's own standard, section 2 "Returns
and horizons"; project_plan.md section 6.2 draft): this module implements only (A) -- the
horizon *returns are sampled at* (D, W-Friday, M-end, Q-end, non-overlapping). The *estimation
window* (rolling 63/126/252 trading days, EWMA lambda 0.94) that ``config/experiment.yaml``
also lists is a separate concept, applied by the correlation estimator itself
(``correlation/estimators.py``, next step in this phase), not here.

Method: log returns from adjusted (total-return) close, ``r_t = log(p_t) - log(p_{t-1})``.
**[Established]** mathematical identity, not an empirical claim: log returns over non-overlapping
periods sum exactly to the log return over their union (``log(p_T/p_0) = sum_t log(p_t/p_{t-1})``),
which is why resampling to a lower frequency and then log-differencing (this module's approach)
agrees with log-differencing the daily series and summing within each period -- both give the
period's total log return. No citation search is needed for this: it follows directly from
``log(a) + log(b) = log(ab)``, not from a contested empirical result.

Horizon construction (project_plan.md section 6.2 draft, adapted to this repo's pandas-3
period-end aliases, matching ``config.WalkForward.refit_every``'s ``Literal["ME", "QE", "YE"]``):
``D`` = the daily aligned series as-is; ``W`` = Friday-to-Friday (``W-FRI``); ``M`` = calendar
month-end (``ME``); ``Q`` = calendar quarter-end (``QE``), non-overlapping. All four are read
from ``prices_clean.parquet`` (Phase 1), whose index is already the shared NYSE session calendar
over ``[universe.start_date, asof]`` with **no forward-fill** (``build_panel.py``): a period with
no valid price is silently ``NaN`` here too, never filled from an earlier period, per the
project's own "no forward-fill" rule and the pure-function contract of ``project_plan.md``
section 9 ("``f(data_up_to_t) -> value_at_t``", the same contract ``etfcorr.leakage`` tests).

**Incomplete final period, dropped** (project_plan.md section 6.2 draft; the project's own
standard, section 2: "drop the final incomplete period"): resampling labels a period by its
*end* date (e.g. the Friday of the current week) regardless of whether the input data actually
reaches that date. If the input's last row is dated before that label, the period is still
in progress (its return would be a partial-period observation, computed from fewer days than
every other period of the same horizon) and is dropped. This is the one leakage-relevant design
choice in this module: without it, a feature/model reading ``horizon_returns`` at some date `t`
that happens to fall inside an unfinished week/month would see a different (shorter, partial)
return for that week than a later run that had more data for the same week -- exactly what
``etfcorr.leakage.assert_no_lookahead``'s truncation test checks for, and what
``tests/leakage/test_returns_no_lookahead.py`` verifies including on a deliberately-leaky twin
(``drop_incomplete=False``) that the detector correctly flags.

Known limitations, disclosed rather than silently accepted:

* **Within-period gaps are "filled" from earlier in the same period, not treated as missing.**
  ``resample(...).last()`` takes each period's *last available* observation; if a ticker's last
  reported day of a week is Wednesday (a genuine gap on Thursday/Friday, not a holiday -- every
  ticker shares the same NYSE calendar per Phase 1), the week's return is computed through
  Wednesday's price under the Friday label. This is not look-ahead (only equal-or-earlier data is
  used) and is a data-quality question, not a returns-construction one: whether a ticker has such
  gaps is ``coverage.csv``'s / ``validate.py``'s job to surface (its ``missing_pct`` and gap
  checks), not this module's. Still, a horizon return computed this way is a weaker statement
  than "the full period's price change" and should not be treated as identical to one with no
  such gap.
* **Log of a non-positive price is treated as a hard error here, not a silent NaN/-inf.**
  ``validate.py`` (Phase 1) already makes a non-positive close a hard error before
  ``prices_clean.parquet`` is written, so this should never trigger in the normal pipeline; the
  check exists because this module's functions are public, pure, and reusable outside that
  pipeline (e.g. from a notebook or a future test), where that precondition is not guaranteed.
* **Vendor retroactive adjustment** (standing caveat, ``docs/STATUS.md``): ``adj_close`` is
  revised by the vendor after corporate actions, so re-running this module on a freshly
  downloaded panel can change historical returns slightly even for dates far in the past; the
  download log's per-ticker hash (``download.py``) is the record of which download a given
  ``returns_<H>.parquet`` was computed from.

No third-party code is adapted here; nothing to record in THIRD_PARTY.md.
"""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from etfcorr.config import ConfigError, Horizon, Project
from etfcorr.data.build_panel import CLEAN_NAME
from etfcorr.data.download import PanelReader, PanelWriter, read_panel_parquet, write_panel_parquet

RETURNS_NAME_TEMPLATE = "returns_{horizon}.parquet"

# Pandas period-end resampling rule per horizon; "D" needs no resampling (see horizon_returns).
# ME/QE match config.WalkForward.refit_every's Literal["ME", "QE", "YE"] (pandas-3 aliases; the
# pre-3.0 "M"/"Q" aliases this repo's own project_plan.md draft used no longer exist).
_RESAMPLE_RULE: dict[Horizon, str] = {"W": "W-FRI", "M": "ME", "Q": "QE"}

ADJUSTED_FIELD = "adj_close"


class ReturnsError(ValueError):
    """Invalid input to a returns computation (e.g. a non-positive price, an unknown horizon)."""


def select_field(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    """One field (``close``/``volume``/``adj_close``) of a ``(ticker, field)``-column panel.

    Returns a plain ticker-column frame. The project's standard input to every return/correlation
    computation is ``field="adj_close"`` (total-return, adjusted for splits and dividends, per
    the project's own standard, section 2: "Use log returns from adjusted (total-return)
    prices"); ``close`` (split-adjusted only) exists in the panel for the liquidity/audit checks
    of Phase 1 and as a diagnostic (e.g. isolating a dividend-adjustment artefact), never as the
    default input downstream of this module.
    """
    if not isinstance(panel.columns, pd.MultiIndex) or "field" not in panel.columns.names:
        raise ConfigError("expected a (ticker, field) MultiIndex-column panel")
    available = panel.columns.get_level_values("field").unique()
    if field not in available:
        raise ConfigError(f"panel has no {field!r} field (has {sorted(available)})")
    # Same pandas-stubs overload gap already fixed in download.py/build_panel.py: `.xs()` is
    # typed as `DataFrame | Series[Any]` because a single-level Index could collapse to a
    # Series, but `panel.columns` is confirmed a MultiIndex with a "field" level above, so
    # selecting one field still leaves the "ticker" level -- a DataFrame at runtime, always.
    return cast(pd.DataFrame, panel.xs(field, level="field", axis=1))


def adjusted_close(panel: pd.DataFrame) -> pd.DataFrame:
    """``select_field(panel, "adj_close")`` -- the project's standard price input."""
    return select_field(panel, ADJUSTED_FIELD)


def _assert_positive(prices: pd.DataFrame) -> None:
    values = np.asarray(prices, dtype=float)
    finite = ~np.isnan(values)
    if finite.any() and not bool(np.all(values[finite] > 0)):
        raise ReturnsError(
            "prices must be positive wherever not NaN (a non-positive price should already have "
            "been rejected by etfcorr.data.validate.validate_panel before this point)"
        )


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns ``log(p_t) - log(p_{t-1})``; the first row (no prior price) is dropped.

    Pure and NaN-preserving: a ``NaN`` price at ``t`` or ``t-1`` gives a ``NaN`` return at ``t``,
    never filled -- this is what carries Phase 1's "no forward-fill" rule through into returns
    (a pre-inception gap, or a genuine interior gap, stays absent rather than becoming a
    fabricated zero-return day).
    """
    _assert_positive(prices)
    # pandas-stubs does not special-case numpy ufuncs called on a DataFrame: `np.log(prices)` is
    # typed as returning an ndarray, though pandas' `__array_ufunc__` protocol makes the runtime
    # value a DataFrame (same column/index, log-transformed) -- confirmed, not assumed, by the
    # real `.diff()` call this line already made before pandas-stubs was installed offline.
    logged = cast(pd.DataFrame, np.log(prices))
    return logged.diff().iloc[1:]


def resample_prices(prices: pd.DataFrame, horizon: Horizon) -> pd.DataFrame:
    """Prices at ``horizon``'s period ends (``D`` = unchanged); see the module docstring."""
    if horizon == "D":
        return prices
    if horizon not in _RESAMPLE_RULE:
        raise ReturnsError(f"unknown horizon {horizon!r}")
    return prices.resample(_RESAMPLE_RULE[horizon]).last()


def _drop_incomplete_period(resampled: pd.DataFrame, last_observed: pd.Timestamp) -> pd.DataFrame:
    """Drop the last row if its period-end label is still in the future of the input data."""
    if len(resampled.index) and last_observed < resampled.index[-1]:
        return resampled.iloc[:-1]
    return resampled


def horizon_returns(
    prices: pd.DataFrame, horizon: Horizon, *, drop_incomplete: bool = True
) -> pd.DataFrame:
    """Log returns of ``prices`` at ``horizon``, dropping the final incomplete period by default.

    ``drop_incomplete=False`` exists only so ``tests/leakage/test_returns_no_lookahead.py`` can
    demonstrate that the guard is load-bearing (a deliberately leaky twin that
    ``assert_no_lookahead`` correctly flags); production code and :func:`command_returns` never
    pass it.
    """
    if horizon == "D":
        return log_returns(prices)
    resampled = resample_prices(prices, horizon)
    if drop_incomplete and len(prices.index):
        resampled = _drop_incomplete_period(resampled, prices.index[-1])
    return log_returns(resampled)


def returns_by_horizon(
    prices: pd.DataFrame, horizons: tuple[Horizon, ...]
) -> dict[Horizon, pd.DataFrame]:
    """:func:`horizon_returns` for every horizon in ``horizons`` (order preserved)."""
    return {h: horizon_returns(prices, h) for h in horizons}


def command_returns(
    project: Project,
    *,
    read_panel: PanelReader = read_panel_parquet,
    write_returns: PanelWriter = write_panel_parquet,
) -> list[str]:
    """Read ``prices_clean.parquet``, compute returns for every configured horizon, write each.

    Raises ``ConfigError`` if the clean panel (Phase 1, ``etfcorr data build``) does not exist
    yet. Writes ``data/processed/returns_<horizon>.parquet`` for every horizon in
    ``project.experiment.horizons``, in that order; returns are only ever written for horizons
    actually configured, so a smaller ``horizons`` list in ``experiment.yaml`` writes fewer files,
    never fewer rows within a file.
    """
    clean_path = project.paths.data_interim / CLEAN_NAME
    if not clean_path.is_file():
        raise ConfigError(f"{clean_path} not found; run `etfcorr data build` first")
    panel = read_panel(clean_path)
    prices = adjusted_close(panel)
    horizons = project.experiment.horizons
    by_horizon = returns_by_horizon(prices, horizons)

    lines = [f"returns from {clean_path} ({prices.shape[1]} tickers)"]
    for horizon in horizons:
        returns = by_horizon[horizon]
        out_path = project.paths.data_processed / RETURNS_NAME_TEMPLATE.format(horizon=horizon)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_returns(returns, out_path)
        lines.append(
            f"  {horizon}: {returns.shape[0]} rows x {returns.shape[1]} tickers -> {out_path}"
        )
    return lines


def returns_path(paths_data_processed: Path, horizon: Horizon) -> Path:
    """Where :func:`command_returns` writes (or a reader should look for) one horizon's returns."""
    return paths_data_processed / RETURNS_NAME_TEMPLATE.format(horizon=horizon)
