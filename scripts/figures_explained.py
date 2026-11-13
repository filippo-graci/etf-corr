"""Figures for the explainers in ``docs/explained/`` (2026-10-19). Run with:
``uv run python scripts/figures_explained.py`` (seconds).

Reads only recorded tables, plus the core daily returns up to the end of validation for the
eigenvalue spectrum. No statistic is recomputed except that spectrum, so every figure shows numbers
already reported. Descriptive; the verdicts and their caveats are in the explainers.

* ``spectrum_explained``: eigenvalues of the daily core correlation matrix against the noise band.
* ``h1_noise_explained``: follow-up E2, the real after-minus-before differences against panels whose
  structure is identical at every horizon.
* ``h2_adjustment_explained``: the H2 table, raw and volatility-adjusted high-minus-low differences.
* ``h3_null_explained``: test-window losses of every model against 200 noise-fed models.
* ``h5_nt_explained``: follow-up E3, cleaned minus sample volatility by N/T, long-only and
  long-short.
* ``h6_gap_explained``: follow-up E1, the stress-calm gap before and after the volatility
  adjustment.
* ``h6_sectors_explained``: follow-up E1, hedged volatility in calm and in stress by sector.

Colours: the project's validated slots (``viz.style``), assigned in fixed order; text in ink tokens.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from etfcorr.config import Project, load_project
from etfcorr.correlation.rmt import marchenko_pastur_edge
from etfcorr.viz.style import LIGHT, SECTOR_ORDER, save_figure, sector_style

T = LIGHT
BLUE, ORANGE, GREEN = T.slots[0], T.slots[1], T.slots[2]
RING = {"edgecolors": T.surface, "linewidths": 1.5, "zorder": 3}


def tidy(ax: Axes, title: str, grid: str = "x") -> None:
    ax.set_title(title, loc="left", fontsize=9.5, color=T.ink)
    ax.set_facecolor(T.surface)
    ax.tick_params(colors=T.ink_secondary, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T.baseline)
    ax.grid(axis=grid, color=T.gridline, linewidth=0.6)
    ax.set_axisbelow(True)


def zero(ax: Axes, vertical: bool = True) -> None:
    (ax.axvline if vertical else ax.axhline)(0.0, color=T.ink_secondary, linewidth=0.8)


def finish(
    fig: plt.Figure,
    title: str,
    name: str,
    project: Project,
    note: str = "",
    legend: list[Line2D] | None = None,
) -> None:
    """Title, an optional figure legend in its own band at the bottom, the footnote, then save."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=T.ink)
    if note:
        fig.text(0.01, 0.005, note, fontsize=7.5, color=T.ink_secondary, ha="left", va="bottom")
    bottom = 0.04 if note else 0.0
    if legend:
        fig.legend(
            handles=legend,
            loc="lower center",
            ncol=len(legend),
            fontsize=7.5,
            frameon=False,
            bbox_to_anchor=(0.5, bottom),
        )
        bottom += 0.07
    fig.tight_layout(rect=(0, bottom, 1, 0.95))
    save_figure(fig, name, project.paths)
    plt.close(fig)


def spectrum(project: Project) -> None:
    core = list(project.universe.tickers_in("core"))
    frame = pd.read_parquet(project.paths.data_processed / "returns_D.parquet")[core]
    daily = frame.loc[frame.index <= pd.Timestamp(project.experiment.splits.val.end)].dropna()
    values = np.sort(np.linalg.eigvalsh(np.corrcoef(daily.to_numpy(), rowvar=False)))[::-1]
    edge = marchenko_pastur_edge(len(core), len(daily))
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=T.surface)
    rank = np.arange(1, len(values) + 1)
    above = values > edge
    ax.bar(rank[above], values[above], width=0.7, color=BLUE, label="above the noise band")
    ax.bar(rank[~above], values[~above], width=0.7, color=T.baseline, label="inside the noise band")
    ax.axhline(edge, color=ORANGE, linewidth=1.5, label=f"noise-band edge {edge:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue rank", fontsize=8)
    ax.set_ylabel("eigenvalue (log scale)", fontsize=8)
    share = values[0] / values.sum()
    ax.annotate(
        f"market mode: {values[0]:.1f}, {share:.0%} of the variance",
        (1, values[0]),
        xytext=(3.5, values[0]),
        fontsize=8,
        color=T.ink_secondary,
        va="center",
    )
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    tidy(ax, f"Daily core matrix, 25 ETFs, {len(daily)} days 2014-2022", grid="y")
    finish(
        fig,
        f"{above.sum()} eigenvalues stand above what pure noise gives; the first is the market",
        "spectrum_explained",
        project,
        "Noise band: Marchenko-Pastur upper edge (1 + sqrt(N/T))^2. In-sample, descriptive.",
    )


def h1_noise(project: Project) -> None:
    sim = pd.read_csv(project.paths.tables / "followup_h1_simulated.csv")
    sim = sim[(sim["world"] == "real_corr") & (sim["when"] == "diff")]
    panels = (
        ("eigvec_1", "Leading-eigenvector overlap\n(the pre-registered statistic)"),
        ("ari", "Cluster agreement (ARI)"),
        ("subspace_5", "5-dimensional subspace overlap"),
    )
    pairs = ["D-W", "D-M", "W-M"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), facecolor=T.surface, sharey=True)
    for ax, (stat, title) in zip(axes, panels, strict=True):
        rows = sim[sim["statistic"] == stat].set_index("pair").loc[pairs]
        y = np.arange(len(pairs))
        ax.hlines(y, rows["sim_q05"], rows["sim_q95"], color=T.baseline, linewidth=7)
        ax.scatter(rows["sim_mean"], y, marker="|", s=160, color=T.ink_secondary, zorder=2)
        ax.scatter(rows["real"], y, s=60, color=BLUE, **RING)
        zero(ax)
        ax.set_yticks(y, pairs)
        ax.invert_yaxis()
        ax.set_xlabel("after minus before removing the market mode", fontsize=8)
        tidy(ax, title)
    handles = [
        Line2D([], [], color=T.baseline, linewidth=7, label="simulated, 5%-95% (truth identical)"),
        Line2D(
            [],
            [],
            color=T.ink_secondary,
            marker="|",
            linestyle="",
            markersize=12,
            label="simulated mean",
        ),
        Line2D([], [], color=BLUE, marker="o", linestyle="", markersize=7, label="real data"),
    ]
    finish(
        fig,
        "H1: the pre-registered statistic falls even when the structure is equal at every horizon",
        "h1_noise_explained",
        project,
        "Follow-up E2, core 25, 2014-2022; 300 Gaussian panels with the real daily correlation, "
        "aggregated to W and M. Exploratory.",
        legend=handles,
    )


def h2_adjustment(project: Project) -> None:
    table = pd.read_csv(project.paths.tables / "h2_table_4_6.csv")
    names = {
        "primary": "primary (per-asset delta)",
        "VIX log-change delta": "delta from VIX changes",
        "SPY return delta": "delta from SPY returns",
        "day-t labels (contemporaneous)": "same-day labels",
        "mean block length 60": "bootstrap block 60",
        "mean block length 120": "bootstrap block 120",
        "without 2020-02-24 to 2020-06-30": "without spring 2020",
        "within-sector pairs (contrast)": "within-sector pairs",
    }
    table = table[table["line"].isin(names)]
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=T.surface)
    for col, colour, label, shift in (
        ("diff_raw", ORANGE, "raw correlation", -0.15),
        ("diff_adjusted", BLUE, "after the Forbes-Rigobon adjustment", 0.15),
    ):
        err = [table[col] - table[f"{col}_lower"], table[f"{col}_upper"] - table[col]]
        ax.errorbar(table[col], y + shift, xerr=err, fmt="none", ecolor=colour, elinewidth=1.5)
        ax.scatter(table[col], y + shift, s=50, color=colour, label=label, **RING)
    zero(ax)
    ax.set_yticks(y, [names[n] for n in table["line"]])
    ax.invert_yaxis()
    ax.set_xlabel("average correlation, high-VIX tercile minus low-VIX tercile", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    tidy(ax, "Core 25, 257 cross-sector pairs, 2014-2026, 95% intervals")
    finish(
        fig,
        "H2: correlation rises by 0.24 in volatile markets; adjusted for volatility, about zero",
        "h2_adjustment_explained",
        project,
        "Step 4.6. Primary line: volatility share 0.88 [0.76, 1.07]. In-sample, descriptive.",
    )


def h3_null(project: Project) -> None:
    table = pd.read_csv(project.paths.tables / "null_benchmark_8_4.csv").sort_values("loss")
    y = np.arange(len(table))
    q05, q50, q95 = (float(table[c].iloc[0]) for c in ("null_q05", "null_median", "null_q95"))
    fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=T.surface)
    ax.axvspan(q05, q95, color=T.gridline, alpha=0.7, label="noise-fed models, 5%-95%")
    ax.axvline(q50, color=T.muted, linewidth=1, label="noise-fed models, median")
    ax.scatter(table["loss"], y, s=50, color=BLUE, label="real models", **RING)
    for k, (loss, rank) in enumerate(zip(table["loss"], table["percentile_rank"], strict=True)):
        ax.annotate(
            f"{rank:.0%}",
            (loss, k),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=7,
            color=T.ink_secondary,
            va="center",
        )
    ax.set_yticks(y, table["model"])
    ax.invert_yaxis()
    ax.set_xlabel(
        "test-window loss, 21-day average cross-sector correlation (Fisher-z MSE)", fontsize=8
    )
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    tidy(ax, "Percent: share of 200 noise-fed ridge models at or below the model's loss")
    finish(
        fig,
        "H3: the best models sit in the good tail of noise; EWMA is worse than 95%",
        "h3_null_explained",
        project,
        "Step 8.4, test window 2023-2026, read once in step 7.5. Descriptive, after the verdict.",
    )


def h5_nt(project: Project) -> None:
    cells = pd.read_csv(project.paths.tables / "followup_h5_cells.csv")
    cells = cells[cells["panel"] == "core N=25"]
    ratio = {"D": 25 / 252, "W": 25 / 104, "M": 25 / 60, "Q": 25 / 32}
    series = (
        ("ledoit_wolf_corr", BLUE, "Ledoit-Wolf on correlation"),
        ("rmt", ORANGE, "RMT clipping"),
        ("nonlinear", GREEN, "nonlinear shrinkage"),
    )
    windows = ["validation 2021-2022", "test 2023-2026 (seen)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), facecolor=T.surface)
    for row, window in enumerate(windows):
        for col, (portfolio, label) in enumerate(
            (("minvar", "long-only"), ("minvar_ls", "long-short"))
        ):
            ax = axes[row, col]
            for estimator, colour, name in series:
                rows = cells[
                    (cells["window"] == window)
                    & (cells["portfolio"] == portfolio)
                    & (cells["estimator"] == estimator)
                ]
                rows = rows.sort_values("frequency", key=lambda f: f.map(ratio))
                x = rows["frequency"].map(ratio)
                d = 100 * rows["diff_vs_sample"]
                err = [d - 100 * rows["diff_ci_lower"], 100 * rows["diff_ci_upper"] - d]
                ax.errorbar(
                    x, d, yerr=err, color=colour, linewidth=2, elinewidth=1, capsize=0, label=name
                )
                ax.scatter(x, d, s=40, color=colour, **RING)
            zero(ax, vertical=False)
            ax.set_xticks(list(ratio.values()), [f"{f}\n{v:.2f}" for f, v in ratio.items()])
            ax.set_ylabel("volatility points vs sample", fontsize=8)
            tidy(ax, f"{label}, {window}", grid="y")
    axes[1, 0].set_xlabel("estimation frequency and N/T", fontsize=8)
    axes[1, 1].set_xlabel("estimation frequency and N/T", fontsize=8)
    handles = [Line2D([], [], color=c, marker="o", linewidth=2, label=n) for _, c, n in series]
    finish(
        fig,
        "H5: cleaning does nothing long-only; long-short, nonlinear shrinkage helps as N/T rises",
        "h5_nt_explained",
        project,
        "Follow-up E3, core 25, minimum variance, 95% intervals. Below zero = less volatile. "
        "No quarterly window before 2022. Exploratory; test window seen.",
        legend=handles,
    )


def h6_gap(project: Project) -> None:
    table = pd.read_csv(project.paths.tables / "followup_h6_summary.csv")
    labels = {
        "vix_same_day (H6 design)": "same-day VIX (H6)",
        "vix_previous_day": "previous-day VIX",
        "spy_realised_vol_21d_previous_day": "previous 21-day SPY vol",
    }
    rows = []
    for (window, label), block in table.groupby(["window", "label"], sort=False):
        block = block.set_index("statistic")
        rows.append(
            (
                f"{window.split(' (')[0]}, {labels[label]}",
                block.loc["he_diff"],
                block.loc["r2_adj_diff"],
            )
        )
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.5, 4), facecolor=T.surface)
    for idx, colour, name, shift in (
        (1, ORANGE, "share of variance removed", -0.15),
        (2, BLUE, "after the volatility adjustment", 0.15),
    ):
        v = np.array([r[idx]["value"] for r in rows])
        lo = np.array([r[idx]["ci_lower"] for r in rows])
        hi = np.array([r[idx]["ci_upper"] for r in rows])
        ax.errorbar(v, y + shift, xerr=[v - lo, hi - v], fmt="none", ecolor=colour, elinewidth=1.5)
        ax.scatter(v, y + shift, s=50, color=colour, label=name, **RING)
    zero(ax)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("stress minus calm, mean over 25 ETFs (SPY hedge, 126-day beta)", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    tidy(ax, "Positive = the hedge looks better in stress; 95% intervals")
    finish(
        fig,
        "H6: the hedge only looks better in stress; adjusted for volatility the gap disappears",
        "h6_gap_explained",
        project,
        "Follow-up E1. Exploratory; 2023-2026 seen, 2016-2022 a second sample.",
    )


def h6_sectors(project: Project) -> None:
    table = pd.read_csv(project.paths.tables / "followup_h6_sectors.csv")
    table = table[table["label"] == "vix_same_day (H6 design)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=T.surface, sharey=True)
    order = [s for s in SECTOR_ORDER if s in set(table["sector"])]
    for ax, window in zip(
        axes, ["2016-2022 (train and validation)", "test 2023-2026 (seen, in-sample)"], strict=True
    ):
        rows = table[table["window"] == window].set_index("sector").loc[order]
        for k, sector in enumerate(order):
            colour, marker = sector_style(sector, T)
            calm, stress = 100 * rows.loc[sector, ["vol_hedged_calm", "vol_hedged_stress"]]
            ax.plot([calm, stress], [k, k], color=T.baseline, linewidth=2, zorder=1)
            ax.scatter(
                calm,
                k,
                s=55,
                marker=marker,
                facecolors=T.surface,
                edgecolors=colour,
                linewidths=1.5,
                zorder=2,
            )
            ax.scatter(stress, k, s=80, marker=marker, color=colour, **RING)
        ax.set_yticks(np.arange(len(order)), order)
        ax.invert_yaxis()
        ax.set_xlabel("annualised volatility of the hedged ETF, %", fontsize=8)
        tidy(ax, window.split(" (")[0])
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor=T.surface,
            markeredgecolor=T.ink_secondary,
            label="calm days (hollow)",
        ),
        Line2D(
            [], [], marker="o", linestyle="", color=T.ink_secondary, label="stress days (filled)"
        ),
    ]
    finish(
        fig,
        "H6: after hedging with SPY, every sector is at least as risky in stress; commodities most",
        "h6_sectors_explained",
        project,
        "Follow-up E1, stress = top VIX decile of the window, calm = at or below its median. "
        "Sector colour and marker as in every project figure.",
        legend=handles,
    )


def main() -> int:
    project = load_project()
    for make in (spectrum, h1_noise, h2_adjustment, h3_null, h5_nt, h6_gap, h6_sectors):
        make(project)
    print(f"wrote seven figures under {project.paths.figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
