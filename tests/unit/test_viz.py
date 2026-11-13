"""Tests for ``etfcorr.viz.style`` and ``etfcorr.viz.networks_plot`` (step 3.6), synthetic only.

What is checked is the structure of the figure object (counts of nodes, labels, legend entries,
marker shapes, edge widths, text colours), not pixels, plus the file side effects of
``save_figure`` (location, determinism, no timestamp metadata) and the absence of import-time side
effects (no pyplot, no backend).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from etfcorr.config import ConfigError, Paths
from etfcorr.correlation import network_stability as NS
from etfcorr.correlation.networks import MSTResult, build_mst
from etfcorr.synthetic import make_factor_returns
from etfcorr.viz import networks_plot as P
from etfcorr.viz import style as ST

SECTORS_3 = ("ai_semis", "energy_trad", "healthcare")


def _panel_inputs(
    sizes: tuple[int, ...] = (4, 4, 4), seed: int = 1
) -> tuple[pd.DataFrame, dict[str, str]]:
    returns = make_factor_returns(
        400, sizes, market_loading=0.3, sector_loading=0.85, seed=seed
    ).returns
    names = (*SECTORS_3, "defense", "metals_mining")
    labels = [names[s] for s, n in enumerate(sizes) for _ in range(n)]
    returns.columns = pd.Index([f"T{i:02d}" for i in range(returns.shape[1])])
    return returns, dict(zip(returns.columns, labels, strict=True))


@pytest.fixture(scope="module")
def inputs() -> tuple[pd.DataFrame, dict[str, str]]:
    return _panel_inputs()


@pytest.fixture(scope="module")
def mst(inputs: tuple[pd.DataFrame, dict[str, str]]) -> MSTResult:
    return build_mst(inputs[0].corr())


# --------------------------------------------------------------------------- style


def test_every_sector_has_a_slot_and_a_distinct_marker() -> None:
    assert len(ST.SECTOR_ORDER) == len(ST.SECTOR_MARKERS) == 7
    assert len(set(ST.SECTOR_MARKERS)) == 7
    for theme in (ST.LIGHT, ST.DARK):
        assert len(theme.slots) >= len(ST.SECTOR_ORDER)
        assert len(set(theme.slots[:7])) == 7
        assert all(c.startswith("#") and len(c) == 7 for c in theme.slots)


def test_sector_style_is_fixed_per_sector_and_unknowns_are_muted_hexagons() -> None:
    for k, sector in enumerate(ST.SECTOR_ORDER):
        assert ST.sector_style(sector, ST.LIGHT) == (ST.LIGHT.slots[k], ST.SECTOR_MARKERS[k])
        assert ST.sector_style(sector, ST.DARK) == (ST.DARK.slots[k], ST.SECTOR_MARKERS[k])
    assert ST.sector_style("control", ST.LIGHT) == (ST.LIGHT.muted, ST.OTHER_MARKER)
    assert ST.OTHER_MARKER not in ST.SECTOR_MARKERS
    assert ST.sector_style("ai_semis") == ST.sector_style("ai_semis", ST.LIGHT)


def test_themes_are_immutable_and_differ_in_surface_and_ink() -> None:
    with pytest.raises(AttributeError):
        ST.LIGHT.surface = "#000000"  # type: ignore[misc]
    assert ST.LIGHT.surface != ST.DARK.surface and ST.LIGHT.ink != ST.DARK.ink
    assert ST.LIGHT.name == "light" and ST.DARK.name == "dark"


def _tiny_figure() -> Figure:
    fig = Figure(figsize=(2, 2))
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


def test_save_figure_writes_png_and_svg_under_reports_figures(tmp_path: Path) -> None:
    paths = Paths.from_root(tmp_path)
    written = ST.save_figure(_tiny_figure(), "demo", paths)
    assert [p.name for p in written] == ["demo.png", "demo.svg"]
    for p in written:
        assert p.parent == (tmp_path / "reports" / "figures").resolve() and p.stat().st_size > 100
    assert written[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"<svg" in written[1].read_bytes()[:2000]
    assert [
        p.name for p in ST.save_figure(_tiny_figure(), "only_png", paths, formats=("png",))
    ] == ["only_png.png"]


def test_save_figure_output_is_byte_deterministic_and_carries_no_timestamp(
    tmp_path: Path,
) -> None:
    paths = Paths.from_root(tmp_path)
    first = [p.read_bytes() for p in ST.save_figure(_tiny_figure(), "a", paths)]
    second = [p.read_bytes() for p in ST.save_figure(_tiny_figure(), "b", paths)]
    assert first[0] == second[0] and first[1] == second[1]
    assert b"<dc:date>" not in first[1] and b"Software" not in first[0]
    assert b"Matplotlib" not in first[0][:2000]


@pytest.mark.parametrize("name", ["", "../escape", "sub/dir", "/abs", ".hidden", "a/../b"])
def test_save_figure_rejects_names_that_are_not_bare_stems(tmp_path: Path, name: str) -> None:
    with pytest.raises(ST.StyleError, match="bare file stem"):
        ST.save_figure(_tiny_figure(), name, Paths.from_root(tmp_path))
    assert not (tmp_path / "reports").exists()


def test_save_figure_rejects_unknown_formats(tmp_path: Path) -> None:
    paths = Paths.from_root(tmp_path)
    for formats in ((), ("pdf",), ("png", "jpg")):
        with pytest.raises(ST.StyleError, match="formats"):
            ST.save_figure(_tiny_figure(), "x", paths, formats=formats)
    assert not (tmp_path / "reports").exists()


def test_the_output_guard_of_paths_still_blocks_other_locations(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        Paths.from_root(tmp_path).output_path("src", "x.png")


def test_importing_the_viz_modules_does_not_load_pyplot_or_pick_a_backend() -> None:
    code = (
        "import sys\n"
        "import etfcorr.viz.style, etfcorr.viz.networks_plot\n"
        "assert 'matplotlib.pyplot' not in sys.modules, 'pyplot imported'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- layout


def test_layout_is_deterministic_complete_and_inside_the_unit_square(mst: MSTResult) -> None:
    a = P.mst_layout(mst)
    b = P.mst_layout(mst)
    assert a == b and set(a) == set(mst.nodes)
    xy = np.array(list(a.values()))
    assert np.abs(xy).max() == pytest.approx(1.0)
    assert len({tuple(np.round(p, 6)) for p in a.values()}) == len(mst.nodes)  # no stacked nodes


def test_layout_rescales_whatever_the_embedding_returns(
    mst: MSTResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scaling is a safeguard: networkx already returns a centred unit layout, so a stub
    embedding that is off-centre and five times too large is what exercises it."""
    nodes = list(mst.nodes)
    stub = {n: np.array([5.0 * (k % 7) + 20.0, 3.0 * (k % 5) - 40.0]) for k, n in enumerate(nodes)}
    monkeypatch.setattr(P.nx, "kamada_kawai_layout", lambda graph, weight=None: stub)
    xy = np.array(list(P.mst_layout(mst).values()))
    assert np.abs(xy).max() == pytest.approx(1.0)
    np.testing.assert_allclose(xy.mean(axis=0), 0.0, atol=1e-12)


def test_layout_of_one_and_two_node_trees() -> None:
    two = build_mst(pd.DataFrame([[1, 0.5], [0.5, 1]], index=list("ab"), columns=list("ab")))
    assert P.mst_layout(two) == {"a": (-1.0, 0.0), "b": (1.0, 0.0)}
    one = MSTResult(
        nodes=("a",),
        edges=two.edges.iloc[:0],
        node_stats=two.node_stats.iloc[:1],
        total_length=0.0,
    )
    assert P.mst_layout(one) == {"a": (0.0, 0.0)}


# --------------------------------------------------------------------------- the grid


def _grid(
    mst: MSTResult, sectors: dict[str, str], *, support: bool, theme: ST.Theme = ST.LIGHT
) -> Figure:
    values = {edge: 0.25 for edge in mst.edge_set} if support else None
    first = next(iter(mst.edge_set))
    if values is not None:
        values[first] = 1.0
    panel = P.MSTPanel("Daily", mst, values, subtitle="T = 400")
    return P.plot_mst_grid(
        [[panel, panel], [panel, panel]],
        sectors,
        row_labels=["raw", "detoned"],
        title="A title",
        theme=theme,
    )


def test_grid_is_a_plain_figure_with_one_axes_per_panel(
    mst: MSTResult, inputs: tuple[pd.DataFrame, dict[str, str]]
) -> None:
    fig = _grid(mst, inputs[1], support=True)
    assert type(fig) is Figure
    assert len(fig.axes) == 4
    assert to_hex(fig.get_facecolor()) == ST.LIGHT.surface
    dark = _grid(mst, inputs[1], support=False, theme=ST.DARK)
    assert to_hex(dark.get_facecolor()) == ST.DARK.surface


def test_every_node_carries_its_ticker_in_every_panel(
    mst: MSTResult, inputs: tuple[pd.DataFrame, dict[str, str]]
) -> None:
    fig = _grid(mst, inputs[1], support=True)
    for ax in fig.axes:
        labels = sorted(t.get_text() for t in ax.texts if t.get_text() in mst.nodes)
        assert labels == sorted(mst.nodes)


def test_each_sector_is_one_scatter_with_its_own_colour_and_marker(
    mst: MSTResult, inputs: tuple[pd.DataFrame, dict[str, str]]
) -> None:
    _, sectors = inputs
    fig = _grid(mst, sectors, support=True)
    for ax in fig.axes:
        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(scatters) == len(set(sectors.values())) == 3
        colours = {to_hex(c.get_facecolor()[0]) for c in scatters}
        expected = {ST.sector_style(s, ST.LIGHT)[0] for s in set(sectors.values())}
        assert colours == expected
        total = sum(len(c.get_offsets()) for c in scatters)
        assert total == len(mst.nodes)
        shapes = {tuple(np.round(c.get_paths()[0].vertices.ravel()[:6], 3)) for c in scatters}
        assert len(shapes) == 3  # three different marker outlines


def test_node_area_grows_with_degree(
    mst: MSTResult, inputs: tuple[pd.DataFrame, dict[str, str]]
) -> None:
    _, sectors = inputs
    ax = _grid(mst, sectors, support=False).axes[0]
    pos = P.mst_layout(mst)
    by_xy = {tuple(np.round(p, 9)): n for n, p in pos.items()}
    degree = mst.node_stats["degree"]
    for coll in (c for c in ax.collections if isinstance(c, PathCollection)):
        for xy, size in zip(coll.get_offsets(), coll.get_sizes(), strict=True):
            node = by_xy[tuple(np.round(xy, 9))]
            assert size == pytest.approx(P.NODE_BASE_AREA + P.NODE_AREA_PER_DEGREE * degree[node])


def test_edge_width_follows_support_and_is_constant_without_it(
    mst: MSTResult, inputs: tuple[pd.DataFrame, dict[str, str]]
) -> None:
    _, sectors = inputs
    with_support = _grid(mst, sectors, support=True).axes[0]
    (lines,) = [c for c in with_support.collections if isinstance(c, LineCollection)]
    widths = sorted(round(float(w), 6) for w in lines.get_linewidths())
    lo = P.EDGE_BASE_WIDTH + P.EDGE_SUPPORT_WIDTH * 0.25
    hi = P.EDGE_BASE_WIDTH + P.EDGE_SUPPORT_WIDTH * 1.0
    assert widths == sorted([round(hi, 6)] + [round(lo, 6)] * (len(mst.edges) - 1))
    plain = _grid(mst, sectors, support=False).axes[0]
    (lines,) = [c for c in plain.collections if isinstance(c, LineCollection)]
    assert {round(float(w), 6) for w in lines.get_linewidths()} == {P.EDGE_PLAIN_WIDTH}
    # a tree drawn without support sits strictly inside the range of supported widths
    assert P.EDGE_BASE_WIDTH < P.EDGE_PLAIN_WIDTH < P.EDGE_BASE_WIDTH + P.EDGE_SUPPORT_WIDTH


def test_an_edge_missing_from_the_support_map_is_drawn_at_the_base_width(
    mst: MSTResult, inputs: tuple[pd.DataFrame, dict[str, str]]
) -> None:
    panel = P.MSTPanel("x", mst, support={})
    fig = P.plot_mst_grid([[panel]], inputs[1])
    (lines,) = [c for c in fig.axes[0].collections if isinstance(c, LineCollection)]
    assert {round(float(w), 6) for w in lines.get_linewidths()} == {P.EDGE_BASE_WIDTH}


def test_support_may_be_keyed_in_either_node_order(
    mst: MSTResult, inputs: tuple[pd.DataFrame, dict[str, str]]
) -> None:
    flipped = {(t, s): 1.0 for s, t in mst.edge_set}
    fig = P.plot_mst_grid([[P.MSTPanel("x", mst, support=flipped)]], inputs[1])
    (lines,) = [c for c in fig.axes[0].collections if isinstance(c, LineCollection)]
    expected = P.EDGE_BASE_WIDTH + P.EDGE_SUPPORT_WIDTH
    assert {round(float(w), 6) for w in lines.get_linewidths()} == {round(expected, 6)}


def test_legend_lists_the_sectors_present_in_fixed_order_with_matching_markers(
    mst: MSTResult, inputs: tuple[pd.DataFrame, dict[str, str]]
) -> None:
    _, sectors = inputs
    fig = _grid(mst, sectors, support=False)
    (legend,) = fig.legends
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == [s for s in ST.SECTOR_ORDER if s in set(sectors.values())]
    markers = [h.get_marker() for h in legend.legend_handles]
    assert markers == [ST.sector_style(s)[1] for s in labels]
    colours = [to_hex(h.get_markerfacecolor()) for h in legend.legend_handles]
    assert colours == [ST.sector_style(s)[0] for s in labels]


def test_legend_order_is_the_fixed_sector_order_not_alphabetical() -> None:
    returns, sectors = _panel_inputs((3, 3, 3, 3))
    assert sorted(set(sectors.values())) != [
        s for s in ST.SECTOR_ORDER if s in set(sectors.values())
    ]
    fig = P.plot_mst_grid([[P.MSTPanel("x", build_mst(returns.corr()))]], sectors)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert labels == [s for s in ST.SECTOR_ORDER if s in set(sectors.values())]


def test_a_control_ticker_gets_the_muted_hexagon_and_a_legend_entry(
    inputs: tuple[pd.DataFrame, dict[str, str]],
) -> None:
    returns, sectors = inputs
    with_control = {**sectors, "T00": "control"}
    tree = build_mst(returns.corr())
    fig = P.plot_mst_grid([[P.MSTPanel("x", tree)]], with_control)
    (legend,) = fig.legends
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels[-1] == "control"
    assert legend.legend_handles[-1].get_marker() == ST.OTHER_MARKER


def test_all_text_is_in_ink_tokens_never_a_series_colour(
    mst: MSTResult, inputs: tuple[pd.DataFrame, dict[str, str]]
) -> None:
    for theme in (ST.LIGHT, ST.DARK):
        fig = _grid(mst, inputs[1], support=True, theme=theme)
        allowed = {theme.ink, theme.ink_secondary}
        texts = [t for ax in fig.axes for t in ax.texts] + list(fig.texts)
        texts += [t for legend in fig.legends for t in legend.get_texts()]
        assert texts and all(to_hex(t.get_color()) in allowed for t in texts)
        assert not allowed & set(theme.slots)


def test_titles_row_labels_and_note_are_present(
    mst: MSTResult, inputs: tuple[pd.DataFrame, dict[str, str]]
) -> None:
    fig = _grid(mst, inputs[1], support=True)
    everything = [t.get_text() for ax in fig.axes for t in ax.texts] + [
        t.get_text() for t in fig.texts
    ]
    assert everything.count("Daily") == 4 and everything.count("T = 400") == 4
    assert "raw" in everything and "detoned" in everything and "A title" in everything
    assert any("compare edges, not positions" in t for t in everything)
    custom = P.plot_mst_grid([[P.MSTPanel("x", mst)]], inputs[1], note="my note")
    assert "my note" in [t.get_text() for t in custom.texts]


def test_grid_figure_renders_and_saves(
    mst: MSTResult, inputs: tuple[pd.DataFrame, dict[str, str]], tmp_path: Path
) -> None:
    fig = _grid(mst, inputs[1], support=True)
    written = ST.save_figure(fig, "mst_demo", Paths.from_root(tmp_path))
    assert written[0].stat().st_size > 10_000 and written[1].stat().st_size > 10_000


def test_grid_rejects_bad_input(
    mst: MSTResult, inputs: tuple[pd.DataFrame, dict[str, str]]
) -> None:
    _, sectors = inputs
    panel = P.MSTPanel("x", mst)
    with pytest.raises(P.PlotError, match="at least one panel"):
        P.plot_mst_grid([], sectors)
    with pytest.raises(P.PlotError, match="at least one panel"):
        P.plot_mst_grid([[]], sectors)
    with pytest.raises(P.PlotError, match="same number"):
        P.plot_mst_grid([[panel, panel], [panel]], sectors)
    with pytest.raises(P.PlotError, match="row_labels"):
        P.plot_mst_grid([[panel]], sectors, row_labels=["a", "b"])
    with pytest.raises(P.PlotError, match="no sector for nodes"):
        P.plot_mst_grid([[panel]], {k: v for k, v in sectors.items() if k != "T00"})


def test_panel_inputs_can_come_from_the_bootstrap_support_object(
    inputs: tuple[pd.DataFrame, dict[str, str]],
) -> None:
    returns, sectors = inputs
    es = NS.bootstrap_mst_edge_support(returns, n_bootstrap=20, seed=1)
    fig = P.plot_mst_grid([[P.MSTPanel("x", es.mst, es.support)]], sectors)
    (lines,) = [c for c in fig.axes[0].collections if isinstance(c, LineCollection)]
    widths = sorted(float(w) for w in lines.get_linewidths())
    expected = sorted(P.EDGE_BASE_WIDTH + P.EDGE_SUPPORT_WIDTH * v for v in es.support.values())
    np.testing.assert_allclose(widths, expected)
