"""Shared figure style: palette, themes, sector encoding and the save helper (step 3.6).

Palette and rules follow the project's data-visualisation method: eight categorical hues in a fixed
order (used here for the sectors, in a fixed sector-to-slot assignment so a sector keeps its colour
in every figure), recessive chrome, text in ink tokens (never in a series colour), thin marks and a
surface-coloured ring around nodes.

Validation record (``validate_palette.js`` of the data-visualisation skill, run in step 3.6,
2026-10-04): the seven slots used here pass the lightness band, chroma floor, adjacent-pair CVD
separation (worst 9.1 light / 8.4 dark, protan) and the normal-vision floor (worst 19.6 light /
19.3 dark) on the adjacent pairlist, in both modes. On the **all-pairs** list (any two nodes of a
network can touch) six slots **fail** the floors (worst CVD 3.2, worst normal-vision 12.9 in light
mode): this is the documented series cap of that chart form. Sector identity is therefore carried
by a second channel, a distinct marker shape per sector, plus a legend and a ticker label on every
node; colour is redundant, not the only code. Three light-mode slots (aqua, yellow, magenta) are
below 3:1 against the surface, the contrast relief is the always-visible ticker labels and the CSV
edge table that accompanies each figure.

Nothing here touches the filesystem except :func:`save_figure`, which writes only under
``reports/`` through :meth:`etfcorr.config.Paths.output_path`. No pyplot, no backend selection:
figures are built with :class:`matplotlib.figure.Figure` directly, so importing this module has no
side effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from matplotlib import rc_context
from matplotlib.figure import Figure

from etfcorr.config import Paths

__all__ = [
    "DARK",
    "LIGHT",
    "SECTOR_MARKERS",
    "SECTOR_ORDER",
    "StyleError",
    "Theme",
    "save_figure",
    "sector_style",
]


class StyleError(ValueError):
    """Invalid theme, sector or output request."""


@dataclass(frozen=True)
class Theme:
    """Chart chrome and the categorical slots of one mode (hex strings)."""

    name: str
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    gridline: str
    baseline: str
    slots: tuple[str, ...]


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    gridline="#e1e0d9",
    baseline="#c3c2b7",
    slots=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    muted="#898781",
    gridline="#2c2c2a",
    baseline="#383835",
    slots=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"),
)

# Fixed sector -> slot assignment (colour follows the entity, never its rank in a figure).
SECTOR_ORDER: tuple[str, ...] = (
    "ai_semis",
    "energy_trad",
    "clean_energy",
    "defense",
    "healthcare",
    "metals_mining",
    "fintech_crypto",
)
# One distinct marker shape per sector: the secondary encoding that the all-pairs colour check
# requires (see the module docstring).
SECTOR_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X")
OTHER_MARKER = "h"
SVG_HASH_SALT = "etfcorr-figures"


def sector_style(sector: str, theme: Theme = LIGHT) -> tuple[str, str]:
    """``(colour, marker)`` of a sector; unknown sectors (controls) are muted gray hexagons."""
    if sector in SECTOR_ORDER:
        k = SECTOR_ORDER.index(sector)
        return theme.slots[k], SECTOR_MARKERS[k]
    return theme.muted, OTHER_MARKER


def save_figure(
    fig: Figure,
    name: str,
    paths: Paths,
    *,
    formats: tuple[str, ...] = ("png", "svg"),
    dpi: int = 150,
) -> list[Path]:
    """Write ``fig`` to ``reports/figures/<name>.<format>`` and return the written paths.

    ``name`` is a bare file stem (no directories). Output is limited to ``reports/`` by
    :meth:`Paths.output_path`. Metadata that would embed a timestamp or the software version is
    dropped, and a fixed SVG hash salt replaces matplotlib's random element ids, so re-running a
    script reproduces the same bytes (same library versions).
    """
    if not name or Path(name).name != name or name.startswith("."):
        raise StyleError(f"name must be a bare file stem, got {name!r}")
    unsupported = [f for f in formats if f not in ("png", "svg")]
    if unsupported or not formats:
        raise StyleError(f"formats must be a non-empty subset of ('png', 'svg'), got {formats}")
    written: list[Path] = []
    for fmt in formats:
        target = paths.output_path("reports", "figures", f"{name}.{fmt}")
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"Date": None} if fmt == "svg" else {"Software": None}
        # a fixed hash salt: SVG element ids are otherwise random and change on every run
        with rc_context({"svg.hashsalt": SVG_HASH_SALT}):
            fig.savefig(
                target, format=fmt, dpi=dpi, facecolor=fig.get_facecolor(), metadata=metadata
            )
        written.append(target)
    return written
