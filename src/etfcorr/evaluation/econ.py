"""Step 8.3: ``etfcorr econ``, the portfolio (H5) and hedging (H6) evaluation.

Same pattern as ``etfcorr eval`` (``evaluation.final``): without ``--final`` the design of
``config/experiment.yaml`` (``economics``) is run on the **validation** window (a rehearsal);
with ``--final --confirm-unseal`` it is run **once** on the test window, behind the same guards
(``economics.frozen_on`` set, clean working tree, no marker of an earlier run; the marker
``reports/tables/econ_8_final_marker.json`` is written before any data are read).

The test window was read for the forecasting ladder in Phase 7. The covariance estimators, the
portfolios and the hedge ratios here use no fitted forecasting model and nothing tuned on that
window; the DCC correlation path is the causal expanding-window one. What is new in the final
run is every portfolio and hedging statistic of 2023-2026.

Both verdict rules are written next to the design in the config and applied by
``portfolio.volatility_table`` and ``hedging.h6_table``. Limits are in those modules; the ones
to repeat with every number: long-only monthly portfolios of 25 thematic ETFs on vendor closes,
flat costs, one window, a stress regime made of a few clusters of days, and no claim of
tradeable alpha.
"""

import json

import numpy as np
import pandas as pd

from etfcorr.config import Project, config_hash
from etfcorr.correlation.dcc import rolling_dcc
from etfcorr.evaluation import hedging as H
from etfcorr.evaluation import portfolio as P
from etfcorr.evaluation.final import EvalError, close_final, git_state, open_final
from etfcorr.evaluation.ladder import load_sealed_panel

MARKER = "econ_8_final_marker.json"


def load_panels(
    project: Project, *, final: bool
) -> tuple[dict[str, pd.DataFrame], pd.Series, pd.DataFrame, pd.Series]:
    """Core panels per frequency, the hedge instrument, the lasso instruments and the VIX level.

    Everything is cut at the end of validation, or at the end of the test window with ``final``.
    """
    exp, paths = project.experiment, project.paths
    design = exp.economics
    assert design is not None
    daily, _, _, vix = load_sealed_panel(project, final=final)
    end = daily.index[-1]
    core = list(daily.columns)
    panels = {"D": daily}
    for freq in design.estimation:
        if freq != "D":
            frame = pd.read_parquet(paths.data_processed / f"returns_{freq}.parquet")
            panels[freq] = frame.loc[frame.index <= end, core].dropna()
    everything = pd.read_parquet(paths.data_processed / "returns_D.parquet")
    wanted = list(dict.fromkeys([design.hedge_instrument, *design.lasso_instruments]))
    missing = [t for t in wanted if t not in everything.columns]
    if missing:
        raise EvalError(f"missing from returns_D.parquet: {missing}")
    instruments = everything.loc[daily.index, wanted]
    if instruments.isna().any().any():
        raise EvalError("a hedge instrument has missing returns on the core calendar")
    return panels, instruments[design.hedge_instrument], instruments, vix


def evaluate(
    panels: dict[str, pd.DataFrame],
    hedge: pd.Series,
    instruments: pd.DataFrame,
    vix: pd.Series,
    project: Project,
    *,
    final: bool,
    dcc_corr: np.ndarray | None,
    n_bootstrap: int | None = None,
) -> dict[str, object]:
    """Every table and both verdicts for one window."""
    exp = project.experiment
    design = exp.economics
    assert design is not None
    window = exp.splits.test if final else exp.splits.val
    reps = exp.inference.n_bootstrap if n_bootstrap is None else n_bootstrap
    alpha = exp.inference.alpha
    daily = panels["D"]

    runs = P.run_all(
        panels, design, window, lam=exp.ewma_lambda, dcc_corr=dcc_corr, skip_short=not final
    )
    ran = {name.split("|")[1] for name in runs}
    vol, h5 = P.volatility_table(runs, design, n_bootstrap=reps, alpha=alpha)
    backtest, pbo = P.backtest_table(runs, exp.costs_bps)

    inside = (daily.index >= pd.Timestamp(window.start)) & (daily.index <= pd.Timestamp(window.end))
    betas = {
        "rolling": H.rolling_beta(daily, hedge, design.hedge_window_days),
        "ewma": H.ewma_beta(daily, hedge, exp.ewma_lambda),
        "static": H.static_beta(daily, hedge, pd.Timestamp(window.start)),
    }
    hedged = {k: H.hedged_returns(daily, hedge, b).loc[inside] for k, b in betas.items()}
    names = list(design.lasso_instruments)
    selected = pd.DataFrame()
    if names:
        lasso, selected = H.lasso_hedged_returns(
            daily, instruments[names], design.hedge_window_days, window
        )
        hedged["lasso"] = lasso.loc[inside]
    first = max(frame.dropna().index[0] for frame in hedged.values())
    keep = daily.index[inside] >= first  # the lasso starts at the first rebalance of the window
    stress, calm = H.regime_masks(
        vix.loc[inside][keep], design.stress_quantile, design.calm_quantile
    )
    hedge_table, hedge_assets, h6 = H.h6_table(
        daily.loc[inside][keep],
        {k: v[keep] for k, v in hedged.items()},
        stress,
        calm,
        block_days=design.block_days,
        n_bootstrap=reps,
        alpha=alpha,
    )
    return {
        "vol": vol,
        "backtest": backtest,
        "pbo": pbo,
        "h5": h5,
        "hedge": hedge_table,
        "hedge_assets": hedge_assets,
        "lasso_selected": selected,
        "h6": h6,
        "window": [str(window.start), str(window.end)],
        "frequencies_skipped": [f for f in design.estimation if f not in ran],
    }


def report_lines(run: dict[str, object], final: bool) -> list[str]:
    vol, hedge = run["vol"], run["hedge"]
    h5, h6 = run["h5"], run["h6"]
    assert isinstance(vol, pd.DataFrame) and isinstance(hedge, pd.DataFrame)
    assert isinstance(h5, dict) and isinstance(h6, dict)
    tag = "TEST WINDOW: THE VERDICT" if final else "validation rehearsal, NOT a verdict"
    shown = vol[["cell", "ann_vol", "diff_vs_sample", "diff_ci_lower", "diff_ci_upper"]]
    lines = [
        "Annualised realised volatility; diff_vs_sample < 0 means lower than the sample covariance",
        shown.round(4).to_string(index=False),
        "",
        f"H5 reading ({tag}): mean volatility difference over {h5['n_cells']} minimum-variance "
        f"cells {h5['statistic']:+.4f} [{h5['ci_lower']:+.4f}, {h5['ci_upper']:+.4f}], one-sided "
        f"p = {h5['p_one_sided']:.4f} ({h5['n_days']} days)",
    ]
    gap = h5.get("monthly_quarterly_minus_daily")
    if isinstance(gap, dict):
        lines.append(
            f"  monthly and quarterly cells {gap['monthly_quarterly']:+.4f}, daily cells "
            f"{gap['daily']:+.4f}, difference {gap['difference']:+.4f} "
            f"[{gap['ci_lower']:+.4f}, {gap['ci_upper']:+.4f}] (descriptive)"
        )
    variant = h5.get("variant")
    if isinstance(variant, dict):
        lines.append(
            f"  declared variant {variant['candidate']} against {variant['replaces']}: "
            f"{variant['difference']:+.4f} [{variant['ci_lower']:+.4f}, "
            f"{variant['ci_upper']:+.4f}], adoption rule met: {variant['rule_met']}"
        )
    lines += [
        f"  probability of backtest overfitting across cells (Sharpe): {run['pbo']:.2f}",
        "",
        "Hedge effectiveness (mean over ETFs) by regime; diff = stress - calm",
        hedge.round(4).to_string(index=False),
        "",
        f"H6 reading ({tag}): effectiveness {h6['he_stress']:.4f} in stress ({h6['n_stress']} "
        f"days) against {h6['he_calm']:.4f} in calm ({h6['n_calm']} days), difference "
        f"{h6['statistic']:+.4f} [{h6['ci_lower']:+.4f}, {h6['ci_upper']:+.4f}], one-sided "
        f"p = {h6['p_one_sided']:.4f}",
    ]
    if final:
        for name, key in (("H5", "h5_supported"), ("H6", "h6_supported")):
            verdict = h5 if name == "H5" else h6
            word = "SUPPORTED" if verdict[key] else "NOT SUPPORTED"
            lines.append(
                f"  {name} {word} at alpha = {verdict['alpha']} by the rule fixed before the run"
            )
    return lines


def command_econ(
    project: Project, *, final: bool = False, confirm_unseal: bool = False
) -> list[str]:
    """Run ``etfcorr econ`` and write its tables; returns the lines to print."""
    exp, paths = project.experiment, project.paths
    design = exp.economics
    if design is None:
        raise EvalError("config/experiment.yaml has no economics block (step 8.0)")
    mode = "final" if final else "validation"
    commit, dirty = git_state(project)
    marker = paths.tables / MARKER
    if final:
        open_final(
            project,
            marker,
            design.frozen_on is not None,
            "economics is not frozen (frozen_on)",
            confirm_unseal,
        )
    panels, hedge, instruments, vix = load_panels(project, final=final)
    daily = panels["D"]
    dcc_corr = None
    if "dcc" in design.estimators:
        cache = paths.artifacts / "econ_8" / mode / "dcc_corr.npy"
        if not final and cache.is_file() and len(np.load(cache)) == len(daily):
            dcc_corr = np.load(cache)  # the validation calendar is fixed; the path takes minutes
        else:
            dcc_corr = rolling_dcc(daily, keep_matrices=True).corr_next
            cache.parent.mkdir(parents=True, exist_ok=True)
            if dcc_corr is not None:
                np.save(cache, dcc_corr)
    run = evaluate(panels, hedge, instruments, vix, project, final=final, dcc_corr=dcc_corr)

    head = "TEST WINDOW, SINGLE FINAL RUN." if final else "VALIDATION WINDOW ONLY."
    lines = [
        head,
        f"core panel N={daily.shape[1]}, last date read {daily.index[-1].date()}, config "
        f"{config_hash(exp)[:12]}, commit {commit[:7]}{' (dirty)' if dirty else ''}",
        f"frequencies without a full estimation window, skipped: {run['frequencies_skipped']}",
        "",
        *report_lines(run, final),
    ]
    stem = f"econ_8_{mode}"
    paths.tables.mkdir(parents=True, exist_ok=True)
    for key in ("vol", "backtest", "hedge"):
        frame = run[key]
        assert isinstance(frame, pd.DataFrame)
        frame.to_csv(paths.tables / f"{stem}_{key}.csv", index=False)
    for key in ("hedge_assets", "lasso_selected"):
        frame = run[key]
        assert isinstance(frame, pd.DataFrame)
        frame.to_csv(paths.tables / f"{stem}_{key}.csv")
    summary = {
        "step": "8.5" if final else "8.3",
        "mode": mode,
        "config_hash": config_hash(exp)[:12],
        "commit": commit,
        "dirty": dirty,
        "window": run["window"],
        "last_date_read": str(daily.index[-1].date()),
        "design": design.model_dump(mode="json"),
        "frequencies_skipped": run["frequencies_skipped"],
        "h5": run["h5"],
        "h6": run["h6"],
        "pbo": run["pbo"],
        "bootstrap": {"n": exp.inference.n_bootstrap, "seed": 0},
    }
    (paths.tables / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if final:
        close_final(marker)
    lines.append(f"\nwrote {stem}_*.csv/json under {paths.tables}")
    return lines
