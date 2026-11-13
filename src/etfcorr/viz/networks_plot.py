"""The MST figure (step 3.6): a grid of minimum spanning trees coloured by sector.

Question served: the Phase 3 milestone "MST figure". Each panel draws one
:class:`etfcorr.correlation.networks.MSTResult`: nodes are ETFs, marker colour *and shape* give the
sector (shape is the secondary channel, see ``viz/style.py``), node size grows with the node's
degree, and, when a bootstrap support is supplied, edge width grows with the share of resamples
that contain the edge (``network_stability.bootstrap_mst_edge_support``). Every node carries its
ticker. A sector legend sits under the grid.

This module only draws; it computes nothing about the data. Layout is a Kamada-Kawai embedding of
the tree (networkx), deterministic for a given tree; positions have no meaning beyond adjacency,
and panels are laid out independently, so compare trees by their edges, not by where nodes sit.

Assumptions and caveats: an MST keeps only N - 1 of the N(N - 1)/2 correlations, so a missing edge
is not a zero correlation; trees from a matrix with T close to N (monthly, quarterly) are unstable
(the support shows it); a detoned matrix has negative cross-sector entries by construction, so its
tree is relative structure. See ``correlation/networks.py`` and
``correlation/network_stability.py``.

Third-party code: networkx's ``kamada_kawai_layout`` is imported, not copied (BSD-licensed library
already a project dependency); nothing for ``THIRD_PARTY.md``. Figure rules: ``viz/style.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from etfcorr.correlation.networks import MSTResult
from etfcorr.viz.style import LIGHT, SECTOR_ORDER, Theme, sector_style

__all__ = ["MSTPanel", "PlotError", "mst_layout", "plot_mst_grid"]

NODE_BASE_AREA = 38.0  # points squared
NODE_AREA_PER_DEGREE = 26.0
EDGE_BASE_WIDTH = 0.8
EDGE_SUPPORT_WIDTH = 2.4
EDGE_PLAIN_WIDTH = 1.4
LABEL_FONT = 6.5


class PlotError(ValueError):
    """Invalid input to a plotting function."""


@dataclass(frozen=True, eq=False)
class MSTPanel:
    """One panel: a tree, an optional edge support, and its title lines."""

    title: str
    mst: MSTResult
    support: Mapping[tuple[str, str], float] | None = None
    subtitle: str = ""


def mst_layout(mst: MSTResult) -> dict[str, tuple[float, float]]:
    """Deterministic 2-D positions of the nodes, scaled into ``[-1, 1]``.

    Kamada-Kawai on the tree with the Mantegna distance as edge length. Two-node trees are placed
    by hand (the embedding needs at least three nodes).
    """
    nodes = list(mst.nodes)
    if len(nodes) == 1:
        return {nodes[0]: (0.0, 0.0)}
    if len(nodes) == 2:
        return {nodes[0]: (-1.0, 0.0), nodes[1]: (1.0, 0.0)}
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    edge_distance = mst.edges["distance"].to_numpy(dtype=float)
    for u, v, d in zip(mst.edges["source"], mst.edges["target"], edge_distance, strict=True):
        graph.add_edge(str(u), str(v), distance=float(d))
    raw = nx.kamada_kawai_layout(graph, weight="distance")
    xy = np.array([raw[n] for n in nodes], dtype=float)
    xy = xy - xy.mean(axis=0)
    scale = float(np.abs(xy).max())
    if scale > 0:
        xy = xy / scale
    return {n: (float(x), float(y)) for n, (x, y) in zip(nodes, xy, strict=True)}


def _draw_panel(ax: Axes, panel: MSTPanel, sectors: Mapping[str, str], theme: Theme) -> list[str]:
    """Draw one tree on ``ax``; returns the sectors present (for the legend)."""
    mst = panel.mst
    pos = mst_layout(mst)
    ax.set_facecolor(theme.surface)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.28, 1.28)

    segments = []
    widths = []
    for row in mst.edges.itertuples(index=False):
        a, b = str(row.source), str(row.target)
        segments.append([pos[a], pos[b]])
        if panel.support is None:
            widths.append(EDGE_PLAIN_WIDTH)
        else:
            support = float(panel.support.get((a, b), panel.support.get((b, a), 0.0)))
            widths.append(EDGE_BASE_WIDTH + EDGE_SUPPORT_WIDTH * support)
    ax.add_collection(
        LineCollection(segments, linewidths=widths, colors=theme.muted, capstyle="round", zorder=1)
    )

    degree = mst.node_stats["degree"]
    present: list[str] = []
    for sector in sorted({sectors.get(n, "other") for n in mst.nodes}, key=_sector_key):
        members = [n for n in mst.nodes if sectors.get(n, "other") == sector]
        colour, marker = sector_style(sector, theme)
        xs = [pos[n][0] for n in members]
        ys = [pos[n][1] for n in members]
        sizes = [NODE_BASE_AREA + NODE_AREA_PER_DEGREE * float(degree[n]) for n in members]
        ax.scatter(
            xs,
            ys,
            s=sizes,
            c=colour,
            marker=marker,
            edgecolors=theme.surface,
            linewidths=1.5,
            zorder=3,
        )
        present.append(sector)

    ring = [patheffects.withStroke(linewidth=2.2, foreground=theme.surface)]
    for n in mst.nodes:
        ax.annotate(
            n,
            pos[n],
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=LABEL_FONT,
            color=theme.ink,
            path_effects=ring,
            zorder=4,
        )

    ax.text(
        0.0,
        1.075 if panel.subtitle else 1.02,
        panel.title,
        transform=ax.transAxes,
        fontsize=10,
        color=theme.ink,
        ha="left",
        va="bottom",
    )
    if panel.subtitle:
        ax.text(
            0.0,
            1.02,
            panel.subtitle,
            transform=ax.transAxes,
            fontsize=7.5,
            color=theme.ink_secondary,
            ha="left",
            va="bottom",
        )
    return present


def _sector_key(sector: str) -> tuple[int, str]:
    return (SECTOR_ORDER.index(sector), sector) if sector in SECTOR_ORDER else (99, sector)


def plot_mst_grid(
    rows: Sequence[Sequence[MSTPanel]],
    sectors: Mapping[str, str],
    *,
    row_labels: Sequence[str] | None = None,
    title: str | None = None,
    note: str | None = None,
    theme: Theme = LIGHT,
    panel_size: float = 4.4,
) -> Figure:
    """A rectangular grid of MST panels, with a sector legend and an encoding note underneath.

    ``rows`` is a list of rows of panels (all rows the same length). ``sectors`` maps every node to
    its sector; a node missing from the map is an error (a silent gray node would hide a typo).
    """
    if not rows or not rows[0]:
        raise PlotError("rows must contain at least one panel")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise PlotError("all rows must have the same number of panels")
    if row_labels is not None and len(row_labels) != len(rows):
        raise PlotError("row_labels must have one entry per row")
    for row in rows:
        for panel in row:
            missing = [n for n in panel.mst.nodes if n not in sectors]
            if missing:
                raise PlotError(f"no sector for nodes: {missing}")

    n_rows = len(rows)
    legend_room = 0.9
    top_room = 0.95 if title else 0.5  # inches: room for the panel titles above the axes
    fig = Figure(
        figsize=(width * panel_size, n_rows * panel_size * 1.02 + legend_room + top_room),
        facecolor=theme.surface,
    )
    axes = fig.subplots(n_rows, width, squeeze=False)
    fig.subplots_adjust(
        left=0.04,
        right=0.99,
        bottom=legend_room / fig.get_figheight(),
        top=1.0 - top_room / fig.get_figheight(),
        wspace=0.04,
        hspace=0.16,
    )

    present: set[str] = set()
    for r, panels in enumerate(rows):
        for c, panel in enumerate(panels):
            present.update(_draw_panel(axes[r][c], panel, sectors, theme))
        if row_labels is not None:
            axes[r][0].text(
                -0.02,
                0.5,
                row_labels[r],
                transform=axes[r][0].transAxes,
                rotation=90,
                fontsize=9,
                color=theme.ink_secondary,
                ha="right",
                va="center",
            )

    handles = []
    labels = []
    for sector in sorted(present, key=_sector_key):
        colour, marker = sector_style(sector, theme)
        handles.append(
            Line2D(
                [],
                [],
                linestyle="",
                marker=marker,
                markersize=8,
                markerfacecolor=colour,
                markeredgecolor=theme.surface,
                markeredgewidth=1.5,
            )
        )
        labels.append(sector)
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 7),
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.045 / n_rows),
        handletextpad=0.3,
        columnspacing=1.4,
    )
    for text in legend.get_texts():
        text.set_color(theme.ink_secondary)
    footer = note or (
        "Node size grows with degree. Edge width grows with bootstrap support when shown. "
        "Layouts are independent per panel: compare edges, not positions."
    )
    fig.text(0.5, 0.012, footer, ha="center", va="bottom", fontsize=7.5, color=theme.ink_secondary)
    if title:
        fig.text(
            0.04, 1.0 - 0.18 / fig.get_figheight(), title, fontsize=12, color=theme.ink, va="top"
        )
    return fig
