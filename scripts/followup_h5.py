"""Exploratory follow-up E3 to H5 (docs/FOLLOW_UPS.md item 3). Run with:
``uv run python scripts/followup_h5.py`` (several minutes the first time: one DCC path).

H5 compared cleaned covariances with the sample covariance in long-only minimum-variance
portfolios of 25 ETFs. The long-only constraint already acts like shrinkage and N/T was low, so
the cleaned estimators had little room. This script gives them four settings where they could
matter, with the walk-forward of step 8.1 unchanged (monthly rebalance, data dated before it):

* ``minvar_ls``: fully invested minimum variance with no sign constraint (closed form);
* a larger panel: the core and extended ETFs with returns since September 2019 (41 funds; the two
  spot crypto series and the four funds launched later are left out, because the common period of
  all 47 starts in 2024), at the daily and weekly frequency;
* ``nonlinear`` and ``nonlinear_corr``: the Ledoit-Wolf (2020) analytical nonlinear shrinkage of
  the covariance, and of the correlation with the sample volatilities kept;
* ``dcc_garch``: the DCC correlation forecast with the GARCH variance forecasts of the same fit,
  in place of the EWMA variances of the declared ``dcc`` cell (core panel, daily).

``diff_vs_sample`` is the annualised realised volatility of a cell minus that of the sample
covariance in the same panel, portfolio and frequency; negative is better. ``pooled`` rows are
means of that difference over frequencies, with a joint stationary bootstrap over days, read like
the H5 statistic. The long-only core rows with the declared estimators repeat step 8.5.

**Status: exploratory, after the H5 verdict, which it does not change.** The test window
(2023-01 to 2026-09) has been read: every number on it is in-sample; the validation rows
(2021-2022) are a second, short sample. Dozens of cells are compared and nothing is corrected
for that. Long-short weights ignore borrow costs and leverage limits (``gross`` is the mean sum
of absolute weights). Output: ``reports/tables/followup_h5_{cells,pooled}.csv``.
"""

import numpy as np
import pandas as pd

from etfcorr.config import DateWindow, load_project
from etfcorr.correlation.dcc import rolling_dcc
from etfcorr.evaluation import portfolio as P
from etfcorr.evaluation.ladder import load_sealed_panel

PORTFOLIOS = ("minvar", "minvar_ls")
BASE = ("sample", "ledoit_wolf_corr", "rmt", "nonlinear", "nonlinear_corr")
LARGE_SINCE = "2019-09-12"
POOLS = {
    "declared (ledoit_wolf_corr, rmt, dcc)": ("ledoit_wolf_corr", "rmt", "dcc"),
    "nonlinear": ("nonlinear",),
    "nonlinear_corr": ("nonlinear_corr",),
    "dcc_garch": ("dcc_garch",),
}


def run_cells(
    panels: dict[str, pd.DataFrame],
    estimation: dict[str, int],
    estimators: tuple[str, ...],
    window: DateWindow,
    lam: float,
    dcc: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[dict[str, P.PortfolioRun], list[str]]:
    """Every (portfolio, frequency, estimator) cell that has a full estimation window."""
    daily = panels["D"]
    start = P.rebalance_dates(pd.DatetimeIndex(daily.index), window)[0]
    runs, skipped = {}, []
    for freq, n_rows in estimation.items():
        if (panels[freq].index < start).sum() < n_rows:
            skipped.append(freq)
            continue
        names = [*estimators, *(("dcc", "dcc_garch") if dcc is not None and freq == "D" else ())]
        for estimator in names:
            for portfolio in PORTFOLIOS:
                runs[P.cell_name(portfolio, freq, estimator)] = P.run_cell(
                    daily,
                    panels[freq],
                    n_rows,
                    estimator,  # type: ignore[arg-type]
                    portfolio,
                    window,
                    lam=lam,
                    dcc_corr=None if dcc is None else dcc[0],
                    dcc_var=None if dcc is None else dcc[1],
                )
    return runs, skipped


def tables(
    runs: dict[str, P.PortfolioRun], block: int, n_bootstrap: int, alpha: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cell volatility against the sample cell, and the pooled differences."""
    returns = pd.DataFrame({name: run.returns for name, run in runs.items()}).dropna()
    vols = returns.std(ddof=1) * np.sqrt(P.TRADING_DAYS)
    boot = pd.DataFrame(P._bootstrap_vols(returns, n_bootstrap, block, 0), columns=returns.columns)
    rows = []
    for name in returns.columns:
        portfolio, freq, estimator = name.split("|")
        base = P.cell_name(portfolio, freq, "sample")
        diff = boot[name] - boot[base]
        rows.append(
            {
                "portfolio": portfolio,
                "frequency": freq,
                "estimator": estimator,
                "n_days": len(returns),
                "ann_vol": float(vols[name]),
                "gross": float(runs[name].weights.abs().sum(axis=1).mean()),
                "diff_vs_sample": float(vols[name] - vols[base]),
                "diff_ci_lower": float(diff.quantile(alpha / 2)),
                "diff_ci_upper": float(diff.quantile(1 - alpha / 2)),
            }
        )
    cells = pd.DataFrame(rows)
    pooled = []
    for portfolio in PORTFOLIOS:
        for pool, members in POOLS.items():
            names = [
                n
                for n in returns.columns
                if n.split("|")[0] == portfolio and n.split("|")[2] in members
            ]
            if not names:
                continue
            bases = [P.cell_name(portfolio, n.split("|")[1], "sample") for n in names]
            draws = (boot[names].to_numpy() - boot[bases].to_numpy()).mean(axis=1)
            pooled.append(
                {
                    "portfolio": portfolio,
                    "pool": pool,
                    "n_cells": len(names),
                    "mean_diff": float((vols[names].to_numpy() - vols[bases].to_numpy()).mean()),
                    "ci_lower": float(np.quantile(draws, alpha / 2)),
                    "ci_upper": float(np.quantile(draws, 1 - alpha / 2)),
                    "p_one_sided": float((1 + (draws >= 0).sum()) / (n_bootstrap + 1)),
                }
            )
    return cells, pd.DataFrame(pooled)


def main() -> int:
    project = load_project()
    exp, paths = project.experiment, project.paths
    design = exp.economics
    assert design is not None
    daily, _, _, _ = load_sealed_panel(project, final=True)
    end = daily.index[-1]
    core = list(daily.columns)

    cache = paths.artifacts / "followups"
    cache.mkdir(parents=True, exist_ok=True)
    files = cache / "dcc_corr.npy", cache / "dcc_var.npy"
    if all(f.is_file() for f in files) and len(np.load(files[0])) == len(daily):
        dcc = np.load(files[0]), np.load(files[1])
    else:
        path = rolling_dcc(daily, keep_matrices=True)
        assert path.corr_next is not None and path.var_next is not None
        dcc = path.corr_next, path.var_next
        np.save(files[0], dcc[0])
        np.save(files[1], dcc[1])
    recorded = paths.artifacts / "econ_8" / "final" / "dcc_corr.npy"
    if recorded.is_file():  # the correlation path must be the one of step 8.5
        np.testing.assert_allclose(dcc[0], np.load(recorded), equal_nan=True)
        print("DCC correlation path equals the one recorded in step 8.5")

    frames = {f: pd.read_parquet(paths.data_processed / f"returns_{f}.parquet") for f in "DWMQ"}
    frames = {f: x.loc[x.index <= end] for f, x in frames.items()}
    wanted = [
        t
        for t in (*project.universe.tickers_in("core"), *project.universe.tickers_in("extended"))
        if not t.endswith("-USD")
    ]
    large = [t for t in wanted if frames["D"][t].first_valid_index() <= pd.Timestamp(LARGE_SINCE)]
    setups = {
        f"core N={len(core)}": (
            {f: frames[f][core].dropna() for f in design.estimation},
            dict(design.estimation),
            dcc,
        ),
        f"core+extended N={len(large)}": (
            {f: frames[f].loc[frames[f].index >= LARGE_SINCE, large].dropna() for f in "DW"},
            {f: design.estimation[f] for f in "DW"},  # type: ignore[index]
            None,
        ),
    }
    windows = {"validation 2021-2022": exp.splits.val, "test 2023-2026 (seen)": exp.splits.test}
    all_cells, all_pooled = [], []
    for window_name, window in windows.items():
        for panel_name, (panels, estimation, forecasts) in setups.items():
            runs, skipped = run_cells(panels, estimation, BASE, window, exp.ewma_lambda, forecasts)
            cells, pooled = tables(
                runs, design.block_days, exp.inference.n_bootstrap, exp.inference.alpha
            )
            ratio = {f: round(panels["D"].shape[1] / n, 2) for f, n in estimation.items()}
            print(
                f"\n{window_name} | {panel_name} | N/T {ratio} | skipped (short history) {skipped}"
            )
            print(cells.drop(columns="n_days").round(4).to_string(index=False))
            print(pooled.round(4).to_string(index=False))
            all_cells.append(cells.assign(window=window_name, panel=panel_name))
            all_pooled.append(pooled.assign(window=window_name, panel=panel_name))
    pd.concat(all_cells).to_csv(paths.tables / "followup_h5_cells.csv", index=False)
    pd.concat(all_pooled).to_csv(paths.tables / "followup_h5_pooled.csv", index=False)
    print(f"\nwrote followup_h5_cells.csv and followup_h5_pooled.csv under {paths.tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
