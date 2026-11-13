"""Result figures for the README and the write-up (Phase 8, step 8.7). Run with:
``uv run python scripts/figures_8_7.py`` (seconds; reads recorded tables, and the returns up to
the end of validation for the heatmaps: no test-window data are read here).

* ``h3_models_8_7``: test-window loss of every model for H3's target, with 95% intervals.
* ``h5_h6_8_7``: left, minimum-variance volatility of each cleaned covariance minus the sample
  covariance; right, hedge effectiveness of each ETF in calm and in stress.
* ``heatmaps_8_7``: core correlation matrices per horizon (D, W, M, Q), 2014-2022, by sector.

Every panel is descriptive; the verdicts and their caveats are in ``docs/checks/7.5.md`` and
``docs/checks/8.5.md``. Colours are the fixed sector slots of ``viz.style``; where colour carries
meaning it is doubled by position or marker shape.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from etfcorr.config import Project, load_project
from etfcorr.viz.style import LIGHT, SECTOR_ORDER, save_figure, sector_style

T = LIGHT
RUNG_NAMES = {0: "naive", 1: "econometric", 2: "penalised linear", 3: "LightGBM (with graphs)"}
RUNG_MARKERS = {0: "o", 1: "s", 2: "^", 3: "D"}


def tidy(ax: Axes, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=T.ink)
    ax.set_facecolor(T.surface)
    ax.tick_params(colors=T.ink_secondary, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T.baseline)
    ax.grid(axis="x", color=T.gridline, linewidth=0.6)
    ax.set_axisbelow(True)


def h3_figure(project: Project) -> None:
    table = pd.read_csv(project.paths.tables / "eval_7_final_table.csv")
    table = table[table["task"] == "T1_h21"].sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor=T.surface)
    y = np.arange(len(table))
    for rung, block in table.groupby("rung"):
        pos = y[(table["rung"] == rung).to_numpy()]
        err = [block["mean"] - block["ci_lower"], block["ci_upper"] - block["mean"]]
        ax.errorbar(
            block["mean"],
            pos,
            xerr=err,
            fmt=RUNG_MARKERS[int(rung)],
            color=T.slots[int(rung)],
            ecolor=T.muted,
            elinewidth=1,
            markersize=6,
            label=f"rung {rung}: {RUNG_NAMES[int(rung)]}",
        )
    ax.set_yticks(y, table["model"])
    ax.set_xlabel(
        "squared error of the 21-day average cross-sector correlation (Fisher-z)", fontsize=8
    )
    tidy(ax, "Test window 2023-2026: no model is separable from the simple baselines (H3)")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    fig.tight_layout()
    save_figure(fig, "h3_models_8_7", project.paths)
    plt.close(fig)


def h5_h6_figure(project: Project, sectors: dict[str, str]) -> None:
    vol = pd.read_csv(project.paths.tables / "econ_8_final_vol.csv")
    cells = vol[(vol["portfolio"] == "minvar") & vol["diff_vs_sample"].notna()].iloc[::-1]
    assets = pd.read_csv(project.paths.tables / "econ_8_final_hedge_assets.csv", index_col=0)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 6), facecolor=T.surface)

    y = np.arange(len(cells))
    err = [
        100 * (cells["diff_vs_sample"] - cells["diff_ci_lower"]),
        100 * (cells["diff_ci_upper"] - cells["diff_vs_sample"]),
    ]
    left.errorbar(
        100 * cells["diff_vs_sample"], y, xerr=err, fmt="o", color=T.slots[0], ecolor=T.muted
    )
    left.axvline(0.0, color=T.ink_secondary, linewidth=0.8)
    left.set_yticks(y, cells["frequency"] + " " + cells["estimator"])
    left.set_xlabel("annualised volatility minus the sample covariance's, points", fontsize=8)
    tidy(left, "H5: cleaned covariances do not lower minimum-variance volatility")

    order = sorted(assets.index, key=lambda t: (SECTOR_ORDER.index(sectors[t]), t))
    for k, ticker in enumerate(order[::-1]):
        colour, marker = sector_style(sectors[ticker], T)
        calm, stress = assets.loc[ticker, "rolling_calm"], assets.loc[ticker, "rolling_stress"]
        right.plot([calm, stress], [k, k], color=T.baseline, linewidth=1.5, zorder=1)
        right.scatter(
            calm, k, s=28, facecolors=T.surface, edgecolors=colour, marker=marker, zorder=2
        )
        right.scatter(stress, k, s=28, color=colour, marker=marker, zorder=2)
    right.set_yticks(np.arange(len(order)), order[::-1])
    right.set_xlabel(
        "share of variance removed by the SPY hedge (hollow: calm; filled: stress)", fontsize=8
    )
    tidy(right, "H6: hedges remove more variance in stress, not less")
    fig.tight_layout()
    save_figure(fig, "h5_h6_8_7", project.paths)
    plt.close(fig)


def heatmap_figure(project: Project, sectors: dict[str, str]) -> None:
    end = pd.Timestamp(project.experiment.splits.val.end)
    order = sorted(sectors, key=lambda t: (SECTOR_ORDER.index(sectors[t]), t))
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), facecolor=T.surface)
    for ax, freq in zip(axes, project.experiment.horizons, strict=True):
        frame = pd.read_parquet(project.paths.data_processed / f"returns_{freq}.parquet")
        used = frame.loc[frame.index <= end, order].dropna()
        image = ax.imshow(used.corr().to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(np.arange(len(order)), order, rotation=90, fontsize=5)
        ax.set_yticks(np.arange(len(order)), order, fontsize=5)
        ax.set_title(f"{freq}: T = {len(used)}, N = {len(order)}", fontsize=9, color=T.ink)
    fig.colorbar(image, ax=axes, shrink=0.8, label="Pearson correlation")
    fig.suptitle(
        "Core ETF correlations by horizon, 2014-2022, ordered by sector (in-sample)",
        fontsize=10,
        x=0.02,
        ha="left",
    )
    save_figure(fig, "heatmaps_8_7", project.paths)
    plt.close(fig)


def main() -> int:
    project = load_project()
    sectors = {t.ticker: str(t.sector) for t in project.universe.tickers if t.panel == "core"}
    h3_figure(project)
    h5_h6_figure(project, sectors)
    heatmap_figure(project, sectors)
    print(f"wrote three figures under {project.paths.figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
