"""The two planned figures that were never built (docs/FOLLOW_UPS.md item 6). Run with:
``uv run python scripts/figures_followup.py`` (under a minute).

* ``coupling_followup``: the 63-day rolling average correlation among the core ETFs with the
  dates of well-known market events marked. The events are context chosen after the fact, not a
  test: no claim is made that an event caused a move.
* ``mst_regimes_followup``: the minimum spanning tree of the core ETFs inside each full-sample
  VIX tercile (the regime labels of H2).

Both are descriptive and in-sample: 2014-2022 only, as in Phases 3 and 4; no test-window data.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from etfcorr.config import load_project
from etfcorr.correlation.horizon_analysis import rolling_coupling_indicator
from etfcorr.correlation.network_stability import pearson
from etfcorr.correlation.networks import build_mst, sector_edge_purity
from etfcorr.regimes import vix_terciles
from etfcorr.viz.networks_plot import MSTPanel, plot_mst_grid
from etfcorr.viz.style import LIGHT as T
from etfcorr.viz.style import save_figure

WINDOW = 63
# Dates checked 2026-11-02 against news records and the project's SPY and VIX data: 2016-06-24 is
# the day the Brexit result was declared, 2018-12-24 SPY's lowest close of Q4 2018, 2020-03-16 its
# largest one-day fall of 2014-2026.
EVENTS = {
    "2015-08-24": "China devaluation sell-off",
    "2016-06-24": "Brexit vote",
    "2018-02-05": "volatility spike",
    "2018-12-24": "Q4 2018 low",
    "2020-03-16": "COVID crash",
    "2022-02-24": "Ukraine invasion",
}


def main() -> int:
    project = load_project()
    paths = project.paths
    end = pd.Timestamp(project.experiment.splits.val.end)
    core = list(project.universe.tickers_in("core"))
    sectors = {t.ticker: str(t.sector) for t in project.universe.tickers if t.panel == "core"}
    frame = pd.read_parquet(paths.data_processed / "returns_D.parquet")
    daily = frame.loc[frame.index <= end, core].dropna()

    coupling = rolling_coupling_indicator(daily, "pearson", WINDOW).dropna()
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=T.surface)
    ax.plot(coupling.index, coupling.to_numpy(), color=T.slots[0], linewidth=1.2)
    for k, (date, label) in enumerate(EVENTS.items()):
        ax.axvline(pd.Timestamp(date), color=T.muted, linewidth=0.8, linestyle="--")
        ax.annotate(
            label,
            (pd.Timestamp(date), 0.97 - 0.07 * (k % 2)),
            xycoords=("data", "axes fraction"),
            fontsize=7,
            color=T.ink_secondary,
            ha="left",
            va="top",
            xytext=(3, 0),
            textcoords="offset points",
        )
    ax.set_ylabel(f"average pairwise correlation, trailing {WINDOW} days", fontsize=8)
    ax.set_title(
        f"Coupling of the {len(core)} core ETFs, 2014-2022 (in-sample; events are context only)",
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
    save_figure(fig, "coupling_followup", paths)
    plt.close(fig)
    print(
        f"coupling: min {coupling.min():.2f} on {coupling.idxmin().date()}, "
        f"max {coupling.max():.2f} on {coupling.idxmax().date()}, median {coupling.median():.2f}"
    )

    prices = pd.read_parquet(paths.data_interim / "prices_clean.parquet")
    vix = prices["^VIX"]["close"].reindex(daily.index).rename("^VIX")
    labels = vix_terciles(vix, mode="full_sample")
    panels = []
    for code, name in enumerate(("low VIX", "middle VIX", "high VIX")):
        rows = daily.loc[labels == code]
        mst = build_mst(pearson(rows))
        purity = sector_edge_purity(mst, sectors)
        panels.append(
            MSTPanel(name, mst, subtitle=f"{len(rows)} days, {purity:.0%} same-sector edges")
        )
        print(
            f"{name}: {len(rows)} days, same-sector edges {purity:.2f}, "
            f"mean edge correlation {mst.edges['correlation'].mean():.2f}"
        )
    shared = len(panels[0].mst.edge_set & panels[2].mst.edge_set)
    print(f"edges shared by the low and high trees: {shared} of {len(core) - 1}")
    fig = plot_mst_grid(
        [panels],
        sectors,
        title="Minimum spanning tree of the core ETFs by VIX tercile, 2014-2022 (in-sample)",
        note="Daily Pearson correlations inside each full-sample VIX tercile; descriptive.",
    )
    save_figure(fig, "mst_regimes_followup", paths)
    plt.close(fig)
    print(f"wrote two figures under {paths.figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
