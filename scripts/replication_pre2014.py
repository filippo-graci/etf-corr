"""Pre-2014 replication of H2, H5 and H6 (E5-E7) and the crisis phases (E8). Run with:
``uv run python scripts/replication_pre2014.py --panel`` (reads only the first-row dates), then,
once, ``uv run python scripts/replication_pre2014.py --confirm-run`` (several minutes).

Design: the E5-E8 entries of ``config/experiment.yaml``, declared and tagged
(``replication-freeze-v1``) before any price before 2014-01-02 was downloaded. Data come from the
separate root ``replication/pre2014`` (universe from 2006-01-01, core funds and controls only;
``experiment.yaml`` and ``hypotheses.lock`` are symlinks to the main config), built with the
Phase 1 commands::

    uv run etfcorr data download --root replication/pre2014 --asof 2013-12-31
    uv run etfcorr data build --root replication/pre2014 --asof 2013-12-31
    uv run etfcorr returns --root replication/pre2014

``--panel`` applies the panel rule to ``coverage.csv`` (first valid date per ticker) and prints
it; it reads no price or return. ``--confirm-run`` is guarded like the final runs
(``evaluation.final.open_final``: clean working tree, no earlier marker; the marker
``reports/tables/replication_pre2014_marker.json`` is written before any data are read), so it
runs once.

What is run, with the declared settings reused rather than restated:

* E5: ``scripts/h2_table_4_6.run_study`` (the H2 lines and the step 4.6 rule) on 2007-07-02 to
  2013-12-31, VIX terciles of that window; no COVID line (no such window here).
* E6: ``portfolio.run_all`` and ``volatility_table`` with the ``economics`` block restricted to the
  D and W estimation windows. These cells and the DCC (504 daily rows before its first fit) are
  not all full on 2007-07-02, so E6 starts on the first month at which every cell has its full
  window (``e6_start``); expected July 2008. Not a declared detail: see the note in the summary.
* E7: ``hedging.h6_table`` (rolling, EWMA and static SPY betas; the verdict reads ``rolling``) and
  ``regime_risk_table`` (the E1 adjustment and leftover risk) on the E5 window; the lasso hedge is
  reported beside it from the first month with 126 complete instrument rows (HYG starts in
  April 2007).
* E8: per crisis phase, the average cross-sector correlation (raw and Forbes-Rigobon adjusted to
  the pre-crisis phase, per-asset delta) and the SPY hedge (effectiveness, adjusted squared
  correlation, hedged and unhedged volatility). The phases are fixed calendar periods, so the
  intervals resample days **within** each phase (stationary bootstrap, ``block_days``), not across
  them. Descriptive: no test.
* Beside everything: the share of zero-return days per fund (stale prices bias correlations down)
  and the 63-day rolling coupling with the phase boundaries.

Holm over the p-values of E5-E7 (m = 3). Nothing here changes an H1-H6 verdict.
"""

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from etfcorr.config import DateWindow, Project, config_hash, find_repo_root, load_project
from etfcorr.correlation.dcc import rolling_dcc
from etfcorr.correlation.horizon_analysis import (
    rolling_coupling_indicator,
    stationary_bootstrap_indices,
)
from etfcorr.correlation.regime_correlation import regime_correlations
from etfcorr.evaluation import hedging as H
from etfcorr.evaluation import portfolio as P
from etfcorr.evaluation.final import close_final, git_state, open_final
from etfcorr.evaluation.portfolio import rebalance_dates
from etfcorr.regimes import vix_terciles
from etfcorr.viz.style import LIGHT as T
from etfcorr.viz.style import save_figure

REPLICATION_ROOT = Path("replication") / "pre2014"
MARKER = "replication_pre2014_marker.json"
STEM = "replication_pre2014"

# The panel rule of E5-E7 (config/experiment.yaml, tag replication-freeze-v1).
DROP = ("SMH", "OIH")  # Merrill Lynch HOLDRS until 2011-12-20: a different product
FIRST_ROW_BY = pd.Timestamp("2006-07-10")  # panel start 2006-07-03 + the 7 days of the core rule
WINDOW = DateWindow(start="2007-07-02", end="2013-12-31")
WARMUP = (pd.Timestamp("2006-07-03"), pd.Timestamp("2007-06-29"))  # zero-return check only
E6_ESTIMATION = {"D": 252, "W": 104}  # economics.estimation restricted to the cells that fit
COUPLING_DAYS = 63
PHASES = {  # E8, fixed with the declaration; code 0 is the reference
    "pre_crisis": ("2007-01-02", "2007-07-31"),
    "onset": ("2007-08-01", "2008-09-12"),
    "peak": ("2008-09-15", "2009-03-09"),
    "recovery": ("2009-03-10", "2011-07-29"),
    "euro": ("2011-08-01", "2011-12-30"),
    "after": ("2012-01-03", "2013-12-31"),
}
HEDGE_STATS = ["he", "r2", "r2_adj", "vol_unhedged", "vol_hedged"]


def load_script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# Rules fixed before the data (tested in tests/unit/test_replication_script.py)
# --------------------------------------------------------------------------------------


def panel_rule(coverage: pd.DataFrame) -> list[str]:
    """Core funds, minus SMH and OIH, whose first valid row is on or before 2006-07-10."""
    core = coverage[coverage["panel"] == "core"]
    first = pd.to_datetime(core["first_date"])
    keep = (first <= FIRST_ROW_BY) & ~core["ticker"].isin(DROP)
    return core.loc[keep.to_numpy(), "ticker"].tolist()


def phase_codes(index: pd.DatetimeIndex) -> pd.Series:
    """Integer phase code per date (0 = pre-crisis, in the order of PHASES); NaN outside."""
    codes = pd.Series(np.nan, index=index)
    for code, (start, end) in enumerate(PHASES.values()):
        codes[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))] = code
    return codes


def first_month(
    index: pd.DatetimeIndex, window: DateWindow, ok: Callable[[pd.Timestamp], bool]
) -> pd.Timestamp:
    """The first rebalance date (first session of a month) in ``window`` for which ``ok`` holds."""
    for date in rebalance_dates(index, window):
        if ok(date):
            return date
    raise SystemExit(f"no month in {window.start}..{window.end} satisfies the start rule")


def e6_start(panels: dict[str, pd.DataFrame], dcc_corr: np.ndarray) -> pd.Timestamp:
    """First month at which every E6 cell has its full estimation window and the DCC a forecast."""
    index = pd.DatetimeIndex(panels["D"].index)

    def full(date: pd.Timestamp) -> bool:
        rows = all((panels[f].index < date).sum() >= n for f, n in E6_ESTIMATION.items())
        pos = int(index.searchsorted(date)) - 1
        return rows and pos >= 0 and bool(np.isfinite(dcc_corr[pos]).all())

    return first_month(index, WINDOW, full)


def within_phase_index(codes: np.ndarray, block: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary-bootstrap rows drawn inside each phase separately, phases kept in order."""
    parts = []
    for code in np.unique(codes):
        rows = np.flatnonzero(codes == code)
        parts.append(rows[stationary_bootstrap_indices(len(rows), block, rng)])
    return np.concatenate(parts)


def zero_return_share(daily: pd.DataFrame) -> pd.DataFrame:
    """Share of exactly-zero daily returns per fund, warm-up year and evaluation window."""
    inside = (daily.index >= pd.Timestamp(WINDOW.start)) & (daily.index <= pd.Timestamp(WINDOW.end))
    return pd.DataFrame(
        {
            "warmup": (
                daily.loc[(daily.index >= WARMUP[0]) & (daily.index <= WARMUP[1])] == 0
            ).mean(),
            "window": (daily.loc[inside] == 0).mean(),
        }
    )


def holm(p_values: dict[str, float], alpha: float) -> dict[str, float]:
    adjusted = multipletests(list(p_values.values()), alpha=alpha, method="holm")[1]
    return dict(zip(p_values, (float(p) for p in adjusted), strict=True))


# --------------------------------------------------------------------------------------
# E5-E8
# --------------------------------------------------------------------------------------


def e5(
    daily: pd.DataFrame,
    sectors: dict[str, str],
    spy: pd.Series,
    vix_change: pd.Series,
    vix: pd.Series,
    n_boot: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    h2 = load_script("h2_table_4_6")
    inside = (daily.index >= pd.Timestamp(WINDOW.start)) & (daily.index <= pd.Timestamp(WINDOW.end))
    labels = vix_terciles(vix.loc[inside], mode="full_sample")
    study = h2.run_study(
        daily.loc[inside],
        labels,
        sectors,
        spy.loc[inside],
        vix_change.loc[inside],
        n_boot=n_boot,
        covid_window=None,
    )
    payload = h2.verdict_payload(study, [])
    for key in ("step", "reproduces_4_2", "reproduction_differences"):
        payload.pop(key)
    payload["status"] = "E5, pre-2014 replication of H2; the step 4.6 rule unchanged"
    payload["supported"] = study.verdict.verdict == "supported"
    print(study.regime_table.to_string())
    print("\n".join(h2.report_lines(study, [])[:-1]))
    return h2.table_frame(study), payload


def e6(
    daily: pd.DataFrame, weekly: pd.DataFrame, project: Project, n_boot: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    exp = project.experiment
    design = exp.economics
    assert design is not None
    restricted = design.model_copy(update={"estimation": E6_ESTIMATION})
    panels = {"D": daily, "W": weekly}
    dcc_corr = rolling_dcc(daily, keep_matrices=True).corr_next
    assert dcc_corr is not None
    start = e6_start(panels, dcc_corr)
    window = DateWindow(start=start.date(), end=WINDOW.end)
    runs = P.run_all(panels, restricted, window, lam=exp.ewma_lambda, dcc_corr=dcc_corr)
    vol, verdict = P.volatility_table(
        runs, restricted, n_bootstrap=n_boot, alpha=exp.inference.alpha
    )
    verdict["window"] = [str(window.start), str(window.end)]
    shown = vol[["cell", "ann_vol", "diff_vs_sample", "diff_ci_lower", "diff_ci_upper"]]
    print(shown.round(4).to_string(index=False))
    print(
        f"E6: {verdict['n_cells']} cells from {window.start}, statistic "
        f"{verdict['statistic']:+.4f} [{verdict['ci_lower']:+.4f}, {verdict['ci_upper']:+.4f}], "
        f"one-sided p = {verdict['p_one_sided']:.4f}"
    )
    return vol, verdict


def e7(
    daily: pd.DataFrame,
    spy: pd.Series,
    instruments: pd.DataFrame,
    vix: pd.Series,
    project: Project,
    n_boot: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    exp = project.experiment
    design = exp.economics
    assert design is not None
    alpha = exp.inference.alpha
    inside = (daily.index >= pd.Timestamp(WINDOW.start)) & (daily.index <= pd.Timestamp(WINDOW.end))
    betas = {
        "rolling": H.rolling_beta(daily, spy, design.hedge_window_days),
        "ewma": H.ewma_beta(daily, spy, exp.ewma_lambda),
        "static": H.static_beta(daily, spy, pd.Timestamp(WINDOW.start)),
    }
    hedged = {k: H.hedged_returns(daily, spy, b).loc[inside] for k, b in betas.items()}
    if any(frame.isna().any().any() for frame in hedged.values()):
        raise SystemExit("a hedge ratio is missing inside the E7 window")
    stress, calm = H.regime_masks(vix.loc[inside], design.stress_quantile, design.calm_quantile)
    unhedged = daily.loc[inside]
    kw = {"block_days": design.block_days, "n_bootstrap": n_boot, "alpha": alpha}
    table, assets, verdict = H.h6_table(unhedged, hedged, stress, calm, **kw)
    risk = H.regime_risk_table(unhedged, hedged["rolling"], spy.loc[inside], stress, calm, **kw)

    # the lasso, reported beside: from the first month with complete instrument rows
    names = list(design.lasso_instruments)
    window_days = design.hedge_window_days
    x = instruments[names].reindex(daily.index)
    start = first_month(
        pd.DatetimeIndex(daily.index),
        WINDOW,
        lambda d: (
            bool(x.loc[x.index < d].iloc[-window_days:].notna().all().all())
            and (x.index < d).sum() >= window_days
        ),
    )
    lasso, _ = H.lasso_hedged_returns(
        daily, x, window_days, DateWindow(start=start.date(), end=WINDOW.end)
    )
    later = np.asarray(unhedged.index >= start)
    lasso_table, _, _ = H.h6_table(
        unhedged[later],
        {"rolling": hedged["rolling"][later], "lasso": lasso.loc[inside][later]},
        stress[later],
        calm[later],
        **kw,
    )
    lasso_table["from"] = str(start.date())
    verdict["window"] = [str(WINDOW.start), str(WINDOW.end)]
    print(table.round(4).to_string(index=False))
    print(risk.drop(columns="n_draws").round(4).to_string())
    print(
        f"E7: effectiveness {verdict['he_stress']:.4f} in stress ({verdict['n_stress']} days) "
        f"against {verdict['he_calm']:.4f} in calm ({verdict['n_calm']} days), difference "
        f"{verdict['statistic']:+.4f} [{verdict['ci_lower']:+.4f}, {verdict['ci_upper']:+.4f}], "
        f"one-sided p = {verdict['p_one_sided']:.4f}; lasso beside from {start.date()}"
    )
    frames = {"hedge": table, "hedge_assets": assets, "risk": risk, "lasso": lasso_table}
    return frames, verdict


def phase_correlation(
    returns: pd.DataFrame, codes: np.ndarray, sectors: dict[str, str]
) -> pd.DataFrame:
    """Raw and adjusted (to the pre-crisis phase) average cross-sector correlation per phase."""
    frame = returns.reset_index(drop=True)  # contemporaneous labels: row order is irrelevant
    table = regime_correlations(
        frame,
        pd.Series(codes, dtype=float),
        sectors,
        alignment="contemporaneous",
        reference=0,
    ).table[["avg_raw", "avg_adjusted"]]
    table["diff_raw"] = table["avg_raw"] - table.loc[0, "avg_raw"]
    table["diff_adjusted"] = table["avg_adjusted"] - table.loc[0, "avg_adjusted"]
    return table


def phase_hedge(
    unhedged: pd.DataFrame, hedged: pd.DataFrame, spy: pd.Series, codes: np.ndarray
) -> pd.DataFrame:
    """Means over funds of the SPY hedge statistics per phase (adjusted to the pre-crisis phase)."""
    reference = codes == 0
    rows = {}
    for code in np.unique(codes):
        risk = H.regime_risk(unhedged, hedged, spy, codes == code, reference).mean()
        rows[int(code)] = {s: risk[f"{s}_stress"] for s in HEDGE_STATS}
    return pd.DataFrame(rows).T


def e8(
    daily: pd.DataFrame,
    sectors: dict[str, str],
    spy: pd.Series,
    project: Project,
    n_boot: int,
) -> pd.DataFrame:
    design = project.experiment.economics
    assert design is not None
    codes_all = phase_codes(pd.DatetimeIndex(daily.index))
    rows = codes_all.notna().to_numpy()
    codes = codes_all[rows].to_numpy(dtype=int)
    returns = daily.loc[rows]
    beta = H.rolling_beta(daily, spy, design.hedge_window_days)
    hedged = H.hedged_returns(daily, spy, beta).loc[rows]
    if hedged.isna().any().any():
        raise SystemExit("a hedge ratio is missing inside the E8 phases")

    def stats(idx: np.ndarray | slice) -> pd.DataFrame:
        c = codes[idx]
        corr = phase_correlation(returns.iloc[idx], c, sectors)
        hedge = phase_hedge(
            returns.iloc[idx], hedged.iloc[idx], spy.loc[returns.index].iloc[idx], c
        )
        return corr.join(hedge)

    point = stats(slice(None))
    rng = np.random.default_rng(0)
    draws = [stats(within_phase_index(codes, design.block_days, rng)) for _ in range(n_boot)]
    stacked = np.stack([d.to_numpy() for d in draws])
    alpha = project.experiment.inference.alpha
    lower = pd.DataFrame(
        np.quantile(stacked, alpha / 2, axis=0), index=point.index, columns=point.columns
    )
    upper = pd.DataFrame(
        np.quantile(stacked, 1 - alpha / 2, axis=0), index=point.index, columns=point.columns
    )
    out = pd.concat(
        [
            frame.rename_axis(index="phase", columns="statistic").stack().rename(bound)
            for bound, frame in (("value", point), ("ci_lower", lower), ("ci_upper", upper))
        ],
        axis=1,
    ).reset_index()
    names = list(PHASES)
    out["phase"] = [names[int(c)] for c in out["phase"]]
    out["n_days"] = [int((codes == names.index(p)).sum()) for p in out["phase"]]
    print(point.rename(index=dict(enumerate(names))).round(3).to_string())
    return out


def coupling_figure(daily: pd.DataFrame, project: Project) -> None:
    coupling = rolling_coupling_indicator(daily, "pearson", COUPLING_DAYS).dropna()
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=T.surface)
    ax.plot(coupling.index, coupling.to_numpy(), color=T.slots[0], linewidth=1.2)
    for name, (start, _) in PHASES.items():
        ax.axvline(pd.Timestamp(start), color=T.muted, linewidth=0.8, linestyle="--")
        ax.annotate(
            name,
            (pd.Timestamp(start), 0.97),
            xycoords=("data", "axes fraction"),
            fontsize=7,
            color=T.ink_secondary,
            ha="left",
            va="top",
            xytext=(3, 0),
            textcoords="offset points",
        )
    ax.set_ylabel(f"average pairwise correlation, trailing {COUPLING_DAYS} days", fontsize=8)
    ax.set_title(
        f"Coupling of the {daily.shape[1]} replication-panel ETFs, 2006-2013 (E8, descriptive)",
        loc="left",
        fontsize=10,
        color=T.ink,
    )
    ax.set_facecolor(T.surface)
    ax.tick_params(colors=T.ink_secondary, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=T.gridline, linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, f"{STEM}_coupling", project.paths)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------


def load_panel(
    replication: Project, tickers: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Daily and weekly panel returns, SPY, the VIX log change, the lasso instruments, VIX level."""
    paths = replication.paths
    end = pd.Timestamp(WINDOW.end)
    returns = pd.read_parquet(paths.data_processed / "returns_D.parquet")
    returns = returns.loc[returns.index <= end]
    daily = returns[tickers].dropna()
    weekly = pd.read_parquet(paths.data_processed / "returns_W.parquet")
    weekly = weekly.loc[weekly.index <= end, tickers].dropna()
    prices = pd.read_parquet(paths.data_interim / "prices_clean.parquet")
    vix = prices["^VIX"]["close"].reindex(daily.index)
    spy = returns["SPY"].reindex(daily.index)
    if vix.isna().any() or spy.isna().any():
        raise SystemExit("SPY or the VIX level is missing on the panel calendar")
    design = replication.experiment.economics
    assert design is not None
    instruments = returns[list(design.lasso_instruments)].reindex(daily.index)
    return daily, weekly, spy, returns["^VIX"].reindex(daily.index), instruments, vix


def show_panel(replication: Project) -> list[str]:
    coverage = pd.read_csv(replication.paths.data_processed / "coverage.csv")
    tickers = panel_rule(coverage)
    core = coverage[coverage["panel"] == "core"][["ticker", "sector", "first_date"]]
    core["in_panel"] = core["ticker"].isin(tickers)
    print(core.to_string(index=False))
    sectors = sorted(set(core.loc[core["in_panel"], "sector"]))
    print(f"\npanel: {len(tickers)} funds, {len(sectors)} sectors ({', '.join(sectors)})")
    return tickers


def run(project: Project, replication: Project) -> None:
    exp = project.experiment
    n_boot, alpha = exp.inference.n_bootstrap, exp.inference.alpha
    marker = project.paths.tables / MARKER
    open_final(project, marker, True, "", True)
    commit, _ = git_state(project)

    tickers = show_panel(replication)
    sectors = {t.ticker: str(t.sector) for t in replication.universe.tickers if t.ticker in tickers}
    daily, weekly, spy, vix_change, instruments, vix = load_panel(replication, tickers)
    print(f"\npanel from {daily.index[0].date()} to {daily.index[-1].date()}, {len(daily)} days\n")

    zeros = zero_return_share(daily)
    print("share of zero-return days (beside, not an input):")
    print(zeros.round(3).to_string())
    print("\n== E5 (H2 replicated) ==")
    e5_table, e5_verdict = e5(daily, sectors, spy, vix_change, vix, n_boot)
    print("\n== E6 (H5 replicated) ==")
    e6_vol, e6_verdict = e6(daily, weekly, replication, n_boot)
    print("\n== E7 (H6 replicated) ==")
    e7_frames, e7_verdict = e7(daily, spy, instruments, vix, replication, n_boot)
    print("\n== E8 (crisis phases, descriptive) ==")
    e8_table = e8(daily, sectors, spy, replication, n_boot)
    coupling_figure(daily, project)

    p = {
        "E5": float(e5_verdict["clause1_p_bootstrap"]),
        "E6": float(e6_verdict["p_one_sided"]),
        "E7": float(e7_verdict["p_one_sided"]),
    }
    own = {
        "E5": bool(e5_verdict["supported"]),
        "E6": bool(e6_verdict["h5_supported"]),
        "E7": bool(e7_verdict["h6_supported"]),
    }
    adjusted = holm(p, alpha)
    print("\n== Holm over E5-E7 (m = 3) ==")
    for key in p:
        holds = own[key] and adjusted[key] < alpha
        print(
            f"{key}: own rule {own[key]}, p {p[key]:.4f}, Holm {adjusted[key]:.4f}, holds {holds}"
        )

    out = project.paths.tables
    e5_table.to_csv(out / f"{STEM}_e5_h2.csv", index=False)
    e6_vol.to_csv(out / f"{STEM}_e6_vol.csv", index=False)
    for key, frame in e7_frames.items():
        frame.to_csv(out / f"{STEM}_e7_{key}.csv", index=key in ("hedge_assets", "risk"))
    e8_table.to_csv(out / f"{STEM}_e8_phases.csv", index=False)
    zeros.to_csv(out / f"{STEM}_zero_returns.csv")
    summary = {
        "status": "pre-2014 replication, run once; E5-E7 confirmatory as a family of their own, "
        "E8 descriptive",
        "config_hash": config_hash(exp)[:12],
        "commit": commit,
        "panel": tickers,
        "panel_first_day": str(daily.index[0].date()),
        "window": [str(WINDOW.start), str(WINDOW.end)],
        "e6_start_note": "E6 starts on the first month at which every D and W cell and the DCC "
        "have their full estimation window (e6_start)",
        "E5": e5_verdict,
        "E6": e6_verdict,
        "E7": e7_verdict,
        "holm": {k: {"p": p[k], "p_holm": adjusted[k], "own_rule": own[k]} for k in p},
        "bootstrap": {"n": n_boot, "seed": 0},
    }
    (out / f"{STEM}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    close_final(marker)
    print(f"\nwrote {STEM}_*.csv/json under {out} and the coupling figure")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--panel", action="store_true", help="apply the panel rule; reads no prices")
    mode.add_argument("--confirm-run", action="store_true", help="the single run (writes a marker)")
    args = parser.parse_args()
    root = find_repo_root()
    project = load_project(root)
    replication = load_project(root / REPLICATION_ROOT)
    if args.panel:
        show_panel(replication)
    else:
        run(project, replication)
    return 0


if __name__ == "__main__":
    sys.exit(main())
