"""The MST figure and the real-data check of ``networks.py`` (Phase 3, step 3.6), core panel.

Run with ``uv run python scripts/mst_figure_3_6.py`` (a few minutes at the defaults; ``--quick`` is
a tiny smoke run that prints only and writes nothing: not a result).

What it does: for each horizon (D, W, M, Q) it takes the **core** tickers of
``config/universe.yaml`` (complete-case rows), keeps the rows up to the **validation end of the
sealed split** (2022-12-31, read from ``config/experiment.yaml``) and, for the raw Pearson matrix
and the detoned one (market mode removed, a relative scale):

* builds the MST (Mantegna distance) and its stationary-bootstrap edge support
  (``network_stability.bootstrap_mst_edge_support``);
* tests sector edge purity and the hierarchical-cluster agreement with the sector labels (adjusted
  Rand index), each against a label-permutation null;
* compares the trees of different horizons (edge-survival ratio) with the pure-noise floor
  (``2/N`` mean and a simulated 95th percentile);
* measures rolling-window tree persistence on D and W against a row-shuffled null.

Outputs (not under ``--quick``): the figure ``reports/figures/mst_grid_3_6.png`` and ``.svg``
(rows: raw and detoned; columns: horizons; node colour **and marker shape** = sector, edge width =
bootstrap support) and the tables ``reports/tables/mst_edges_3_6.csv`` (one row per tree edge),
``mst_summary_3_6.csv``, ``mst_cross_horizon_3_6.csv``, ``mst_rolling_3_6.csv`` and
``mst_summary_3_6.md`` (a readable version with the settings and caveats).

**Status: descriptive and exploratory, not a test of H1-H6.** An MST keeps N - 1 of the N(N - 1)/2
correlations; support and survival describe stability under a stationary world; quarterly rows are
exploratory (T close to N). Sector labels are a convenient benchmark, not truth, and holdings
overlap makes some within-sector agreement mechanical. The purity and cluster p-values (two per
tree) are not corrected for multiplicity because none is a confirmatory test. No claim of
tradeable value.
``--full-sample`` removes the 2022 cut, reads the sealed test window and says so: do not use such
output to make choices that are later judged on 2023 onward.
"""

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pandas as pd
import yaml

from etfcorr.config import Horizon, Paths, find_repo_root
from etfcorr.correlation.network_stability import (
    CorrFn,
    EdgeSupport,
    bootstrap_mst_edge_support,
    cluster_sector_agreement,
    detoned_pearson,
    noise_mst_survival_null,
    pearson,
    rolling_survival_shuffle_null,
    survival_matrix,
)
from etfcorr.correlation.networks import MSTResult, sector_purity_test
from etfcorr.correlation.stability import restrict_window
from etfcorr.viz.networks_plot import MSTPanel, plot_mst_grid
from etfcorr.viz.style import save_figure

HORIZONS: tuple[Horizon, ...] = ("D", "W", "M", "Q")
HORIZON_NAMES: dict[str, str] = {
    "D": "Daily",
    "W": "Weekly (Friday close)",
    "M": "Monthly",
    "Q": "Quarterly (exploratory)",
}
KINDS: dict[str, CorrFn] = {"raw": pearson, "detoned": detoned_pearson}
KIND_LABELS: dict[str, str] = {"raw": "raw Pearson", "detoned": "detoned (market mode removed)"}
DEFAULT_ROLLING: tuple[str, ...] = ("D:252:21", "W:104:4")
SEED = 0
FIGURE_NAME = "mst_grid_3_6"
EDGES_CSV = "mst_edges_3_6.csv"
SUMMARY_CSV = "mst_summary_3_6.csv"
CROSS_CSV = "mst_cross_horizon_3_6.csv"
ROLLING_CSV = "mst_rolling_3_6.csv"
SUMMARY_MD = "mst_summary_3_6.md"

QUICK = {"n_bootstrap": 30, "n_shuffles": 5, "n_permutations": 100, "n_null_draws": 30}
DEFAULTS = {"n_shuffles": 200, "n_permutations": 2000, "n_null_draws": 1000}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--quick", action="store_true", help="tiny run, print only, write nothing")
    parser.add_argument("--n-bootstrap", type=int, default=None, help="default: config value")
    parser.add_argument("--n-shuffles", type=int, default=None)
    parser.add_argument("--n-permutations", type=int, default=None)
    parser.add_argument("--n-null-draws", type=int, default=None)
    parser.add_argument("--horizons", nargs="+", default=list(HORIZONS), choices=HORIZONS)
    parser.add_argument(
        "--rolling",
        nargs="*",
        default=list(DEFAULT_ROLLING),
        help="rolling-survival settings as HORIZON:WINDOW:STEP (default: D:252:21 W:104:4)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    window = parser.add_mutually_exclusive_group()
    window.add_argument("--end", default=None, help="last date used (default: validation end)")
    window.add_argument(
        "--full-sample", action="store_true", help="no end date: READS THE SEALED TEST WINDOW"
    )
    return parser.parse_args(argv)


def parse_rolling(specs: Sequence[str], horizons: Sequence[str]) -> list[tuple[str, int, int]]:
    """``HORIZON:WINDOW:STEP`` strings as tuples; settings for unselected horizons are dropped."""
    out: list[tuple[str, int, int]] = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3 or parts[0] not in HORIZONS:
            raise SystemExit(f"bad --rolling spec {spec!r}; expected HORIZON:WINDOW:STEP")
        try:
            window, step = int(parts[1]), int(parts[2])
        except ValueError:
            raise SystemExit(f"bad --rolling spec {spec!r}; window and step are integers") from None
        if window < 3 or step < 1:
            raise SystemExit(f"bad --rolling spec {spec!r}; need window >= 3 and step >= 1")
        if parts[0] in horizons:
            out.append((parts[0], window, step))
    return out


def read_returns(path: Path) -> pd.DataFrame:
    """One ``returns_<h>.parquet``, flattened to a plain ticker-column frame."""
    frame = pd.read_parquet(path)
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.xs(frame.columns.levels[-1][0], axis=1, level=-1)
    return frame


def core_sectors(root: Path) -> dict[str, str]:
    """``{ticker: sector}`` for the core panel, in universe order."""
    with open(root / "config" / "universe.yaml") as f:
        universe = yaml.safe_load(f)
    return {t["ticker"]: str(t["sector"]) for t in universe["tickers"] if t["panel"] == "core"}


def _experiment(root: Path) -> dict[str, object]:
    with open(root / "config" / "experiment.yaml") as f:
        loaded = yaml.safe_load(f)
    return cast(dict[str, object], loaded)


def validation_end(root: Path) -> str:
    """End of the validation split: the last date this script may read by default."""
    splits = cast(Mapping[str, Sequence[object]], _experiment(root)["splits"])
    return str(splits["val"][1])


def n_bootstrap_default(root: Path) -> int:
    inference = cast(Mapping[str, object], _experiment(root)["inference"])
    return int(cast(int, inference["n_bootstrap"]))


def select_core(frame: pd.DataFrame, tickers: Sequence[str], horizon: str) -> pd.DataFrame:
    """The core columns in universe order; a missing core ticker is an error, not a warning."""
    missing = [t for t in tickers if t not in frame.columns]
    if missing:
        raise SystemExit(f"returns_{horizon}.parquet is missing core tickers: {missing}")
    return frame[list(tickers)]


# --------------------------------------------------------------------------- computation


def analyse_tree(
    frame: pd.DataFrame,
    fn: CorrFn,
    sectors: Mapping[str, str],
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> tuple[EdgeSupport, dict[str, float]]:
    """Edge support of the MST of ``fn(frame)``, plus purity and cluster-agreement numbers."""
    support = bootstrap_mst_edge_support(frame, corr_fn=fn, n_bootstrap=n_bootstrap, seed=seed)
    purity = sector_purity_test(support.mst, sectors, n_permutations=n_permutations, seed=seed)
    clusters = cluster_sector_agreement(
        fn(frame), sectors, n_permutations=n_permutations, seed=seed
    )
    degree = support.mst.node_stats["degree"]
    values = support.edges["support"].to_numpy(dtype=float)
    stats = {
        "n_edges": float(len(support.edges)),
        "total_length": float(support.mst.total_length),
        "max_degree": float(degree.max()),
        "purity": purity.observed,
        "purity_null_mean": purity.null_mean,
        "purity_p": purity.p_value,
        "ari": clusters.ari,
        "ari_null_mean": clusters.null_mean,
        "ari_p": clusters.p_value,
        "n_clusters": float(clusters.n_clusters),
        "mean_support": float(values.mean()),
        "share_support_ge_half": float((values >= 0.5).mean()),
        "survival_mean": support.mean_survival,
        "survival_lower": support.survival_lower,
        "survival_upper": support.survival_upper,
    }
    return support, stats


def edge_rows(
    horizon: str, kind: str, support: EdgeSupport, sectors: Mapping[str, str]
) -> pd.DataFrame:
    """One row per tree edge with its correlation, distance, support and sector labels."""
    edges = support.edges
    out = pd.DataFrame(
        {
            "horizon": horizon,
            "matrix": kind,
            "source": edges["source"].astype(str).to_numpy(),
            "target": edges["target"].astype(str).to_numpy(),
            "correlation": edges["correlation"].to_numpy(dtype=float),
            "distance": edges["distance"].to_numpy(dtype=float),
            "support": edges["support"].to_numpy(dtype=float),
        }
    )
    out["source_sector"] = out["source"].map(sectors)
    out["target_sector"] = out["target"].map(sectors)
    out["same_sector"] = out["source_sector"] == out["target_sector"]
    return out


def cross_horizon_rows(
    trees: Mapping[str, Mapping[str, MSTResult]],
    n_obs: Mapping[str, int],
    *,
    n_null_draws: int,
    seed: int,
) -> pd.DataFrame:
    """Edge survival between the trees of every pair of horizons, against the noise floor.

    The floor is two independent pure-noise panels with the same N and the smaller T of the pair:
    its mean is 2/N, and its 95th percentile is the bar a real survival ratio has to clear.
    """
    rows: list[dict[str, object]] = []
    cache: dict[int, tuple[float, float]] = {}
    for kind, by_horizon in trees.items():
        if len(by_horizon) < 2:
            continue
        matrix = survival_matrix(by_horizon)
        names = list(matrix.columns)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                t_min = min(n_obs[a], n_obs[b])
                n_nodes = len(by_horizon[a].nodes)
                if t_min not in cache:
                    null = noise_mst_survival_null(n_nodes, t_min, n_null_draws, seed)
                    cache[t_min] = (null.expected_mean, null.q95)
                expected, q95 = cache[t_min]
                value = float(matrix.loc[a, b])
                rows.append(
                    {
                        "matrix": kind,
                        "horizon_a": a,
                        "horizon_b": b,
                        "n_obs_a": n_obs[a],
                        "n_obs_b": n_obs[b],
                        "survival": value,
                        "noise_mean": expected,
                        "noise_q95": q95,
                        "above_noise_q95": value > q95,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "matrix",
            "horizon_a",
            "horizon_b",
            "n_obs_a",
            "n_obs_b",
            "survival",
            "noise_mean",
            "noise_q95",
            "above_noise_q95",
        ],
    )


def rolling_rows(
    frames: Mapping[str, pd.DataFrame],
    settings: Sequence[tuple[str, int, int]],
    *,
    n_shuffles: int,
    seed: int,
) -> pd.DataFrame:
    """Mean rolling-window survival of the real series against row-shuffled copies."""
    rows: list[dict[str, object]] = []
    for horizon, window, step in settings:
        for kind, fn in KINDS.items():
            result = rolling_survival_shuffle_null(
                frames[horizon], window, step, n_shuffles=n_shuffles, seed=seed, corr_fn=fn
            )
            rows.append(
                {
                    "horizon": horizon,
                    "matrix": kind,
                    "window": window,
                    "step": step,
                    "n_windows": int(result.real.notna().sum()),
                    "real_mean": result.real_mean,
                    "null_mean": result.null_mean,
                    "null_min": float(result.null_means.min()),
                    "null_max": float(result.null_means.max()),
                    "lower_tail": result.lower_tail,
                    "upper_tail": result.upper_tail,
                    "p_two_sided": result.p_two_sided,
                    "n_shuffles": result.n_shuffles,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- readable output


def markdown_table(table: pd.DataFrame, formats: Mapping[str, str]) -> str:
    """A GitHub-style table of ``formats``' columns; each spec is a format spec or ``bool``."""
    columns = list(formats)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for record in table[columns].itertuples(index=False):
        cells: list[str] = []
        for value, spec in zip(record, formats.values(), strict=True):
            if spec == "bool":
                cells.append("yes" if bool(value) else "no")
            elif spec == "str":
                cells.append(str(value))
            else:
                cells.append(format(value, spec))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def header_lines(
    summary: pd.DataFrame,
    *,
    n_bootstrap: int,
    n_permutations: int,
    n_shuffles: int,
    n_null_draws: int,
    seed: int,
    end: str | None,
    reads_sealed: bool,
) -> list[str]:
    windows = (
        summary.groupby("horizon", sort=False)
        .agg(
            n_obs=("n_obs", "first"), start=("window_start", "first"), stop=("window_end", "first")
        )
        .reset_index()
    )
    spans = "; ".join(
        f"{w.horizon}: T={int(w.n_obs)} ({w.start} to {w.stop})" for w in windows.itertuples()
    )
    blocks = "; ".join(
        f"{h}: {b:g}"
        for h, b in summary.groupby("horizon", sort=False)["mean_block_length"].first().items()
    )
    sealed = (
        "WARNING: this output reads the sealed test window (2023 onward)."
        if reads_sealed
        else f"The sealed test window is not read (last date used: {end})."
    )
    n_assets = int(summary["n_assets"].iloc[0])
    return [
        "# MST figure and network check (step 3.6)",
        "",
        f"Core panel, N = {n_assets}. {spans}.",
        f"Stationary bootstrap, {n_bootstrap} draws, mean block lengths {blocks}; permutation "
        f"nulls {n_permutations}; rolling row-shuffles {n_shuffles}; noise-floor draws "
        f"{n_null_draws}; seed {seed}.",
        sealed,
        "",
        "Descriptive and exploratory, not a test of H1-H6. An MST keeps N - 1 of the N(N - 1)/2 "
        "correlations: a missing edge is not a zero correlation. Support is the support under a "
        "stationary world. Detoned matrices are relative (cross-sector entries negative by "
        "construction). Quarterly rows are exploratory (T close to N). P-values are not corrected "
        "for multiplicity because none is a confirmatory test; sector labels are a benchmark, not "
        "truth, and holdings overlap makes some within-sector agreement mechanical.",
    ]


SUMMARY_FORMATS: dict[str, str] = {
    "horizon": "str",
    "matrix": "str",
    "n_obs": "d",
    "purity": ".3f",
    "purity_null_mean": ".3f",
    "purity_p": ".4f",
    "ari": "+.3f",
    "ari_null_mean": "+.3f",
    "ari_p": ".4f",
    "mean_support": ".3f",
    "share_support_ge_half": ".2f",
    "survival_mean": ".3f",
    "survival_lower": ".3f",
    "survival_upper": ".3f",
    "max_degree": ".0f",
}
CROSS_FORMATS: dict[str, str] = {
    "matrix": "str",
    "horizon_a": "str",
    "horizon_b": "str",
    "survival": ".3f",
    "noise_mean": ".3f",
    "noise_q95": ".3f",
    "above_noise_q95": "bool",
}
ROLLING_FORMATS: dict[str, str] = {
    "horizon": "str",
    "matrix": "str",
    "window": "d",
    "step": "d",
    "n_windows": "d",
    "real_mean": ".3f",
    "null_mean": ".3f",
    "null_min": ".3f",
    "null_max": ".3f",
    "lower_tail": ".4f",
    "upper_tail": ".4f",
}


def render_markdown(
    header: Sequence[str], summary: pd.DataFrame, cross: pd.DataFrame, rolling: pd.DataFrame
) -> str:
    parts = [
        "\n".join(header),
        "",
        "## Trees, edge support, sector purity and cluster agreement",
        "",
        "purity = share of edges joining two ETFs of the same sector (null: sector labels "
        "permuted over nodes); ari = adjusted Rand index of an average-linkage clustering against "
        "the sector labels; support = bootstrap share of replicates containing each edge; "
        "survival = edge-survival ratio of a replicate tree against the point tree.",
        "",
        markdown_table(summary, SUMMARY_FORMATS),
    ]
    if not cross.empty:
        parts += [
            "## Trees of different horizons against the noise floor",
            "",
            "survival = shared edges / (N - 1); noise = two independent pure-noise panels (mean "
            "2/N exactly by exchangeability, 95th percentile simulated at the smaller T of the "
            "pair, raw Pearson; for detoned trees the mean is the same and the percentile "
            "approximate).",
            "",
            markdown_table(cross, CROSS_FORMATS),
        ]
    if not rolling.empty:
        parts += [
            "## Rolling-window persistence against row-shuffled returns",
            "",
            "real_mean = mean edge-survival ratio of consecutive trailing-window trees; the null "
            "shuffles rows (overlap kept, serial and regime structure destroyed). lower_tail = "
            "share of shuffles at or below the real value (add-one); windows overlap, so these are "
            "empirical positions, not significance claims.",
            "",
            markdown_table(rolling, ROLLING_FORMATS),
        ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- main


def summary_row(
    horizon: str,
    kind: str,
    frame: pd.DataFrame,
    support: EdgeSupport,
    stats: Mapping[str, float],
) -> dict[str, object]:
    return {
        "horizon": horizon,
        "matrix": kind,
        "n_obs": len(frame),
        "n_assets": frame.shape[1],
        "q": frame.shape[1] / len(frame),
        **stats,
        "n_bootstrap": support.n_bootstrap,
        "mean_block_length": support.mean_block_length,
        "window_start": str(frame.index[0].date()),
        "window_end": str(frame.index[-1].date()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = find_repo_root()
    paths = Paths.from_root(root)
    sectors = core_sectors(root)
    tickers = list(sectors)
    sealed_from = validation_end(root)
    end: str | None = None if args.full_sample else (args.end or sealed_from)
    reads_sealed = end is None or pd.Timestamp(end) > pd.Timestamp(sealed_from)

    def pick(name: str, quick_value: int, default: int) -> int:
        given = getattr(args, name)
        return quick_value if args.quick else (default if given is None else int(given))

    n_boot = pick("n_bootstrap", QUICK["n_bootstrap"], n_bootstrap_default(root))
    n_shuf = pick("n_shuffles", QUICK["n_shuffles"], DEFAULTS["n_shuffles"])
    n_perm = pick("n_permutations", QUICK["n_permutations"], DEFAULTS["n_permutations"])
    n_null = pick("n_null_draws", QUICK["n_null_draws"], DEFAULTS["n_null_draws"])
    rolling_settings = parse_rolling(args.rolling or [], args.horizons)

    frames: dict[str, pd.DataFrame] = {}
    for horizon in args.horizons:
        raw = read_returns(root / "data" / "processed" / f"returns_{horizon}.parquet")
        windowed = restrict_window(select_core(raw, tickers, horizon), end=end)
        frames[horizon] = windowed.dropna(how="any")

    smoke = "QUICK SMOKE RUN, NOT A RESULT. " if args.quick else ""
    print(
        f"DESCRIPTIVE AND EXPLORATORY, NOT A TEST. {smoke}core N={len(tickers)}, "
        f"bootstrap draws {n_boot}, permutations {n_perm}"
    )
    print(
        "WARNING: reads the sealed test window (2023 onward)."
        if reads_sealed
        else f"sealed test window not read (last date used {end})"
    )

    start = time.perf_counter()
    summary_list: list[dict[str, object]] = []
    edge_frames: list[pd.DataFrame] = []
    trees: dict[str, dict[str, MSTResult]] = {kind: {} for kind in KINDS}
    supports: dict[tuple[str, str], EdgeSupport] = {}
    stats_by: dict[tuple[str, str], dict[str, float]] = {}
    for horizon, frame in frames.items():
        for kind, fn in KINDS.items():
            support, stats = analyse_tree(
                frame, fn, sectors, n_bootstrap=n_boot, n_permutations=n_perm, seed=args.seed
            )
            supports[(horizon, kind)] = support
            stats_by[(horizon, kind)] = stats
            trees[kind][horizon] = support.mst
            summary_list.append(summary_row(horizon, kind, frame, support, stats))
            edge_frames.append(edge_rows(horizon, kind, support, sectors))
    summary = pd.DataFrame(summary_list)
    edges = pd.concat(edge_frames, ignore_index=True)
    cross = cross_horizon_rows(
        trees, {h: len(f) for h, f in frames.items()}, n_null_draws=n_null, seed=args.seed
    )
    rolling = (
        rolling_rows(frames, rolling_settings, n_shuffles=n_shuf, seed=args.seed)
        if rolling_settings
        else pd.DataFrame()
    )
    print(f"computed in {time.perf_counter() - start:.1f}s")

    panels: list[list[MSTPanel]] = []
    for kind in KINDS:
        row: list[MSTPanel] = []
        for horizon in frames:
            stats = stats_by[(horizon, kind)]
            row.append(
                MSTPanel(
                    title=HORIZON_NAMES[horizon],
                    mst=supports[(horizon, kind)].mst,
                    support=supports[(horizon, kind)].support,
                    subtitle=(
                        f"T = {len(frames[horizon])}, edge purity {stats['purity']:.2f}, "
                        f"mean support {stats['mean_support']:.2f}"
                    ),
                )
            )
        panels.append(row)
    last = max(f.index[-1] for f in frames.values())
    figure = plot_mst_grid(
        panels,
        sectors,
        row_labels=[KIND_LABELS[k] for k in KINDS],
        title="Minimum spanning trees of the core ETF panel (Mantegna distance)",
        note=(
            f"Exploratory, last date {last.date()}"
            + (", sealed test window read" if reads_sealed else ", sealed test window not read")
            + ". Node size grows with degree; edge width grows with bootstrap support "
            f"({n_boot} draws). Layouts are independent per panel: compare edges, not positions."
        ),
    )

    print()
    for _, r in summary.iterrows():
        print(
            f"{r['horizon']} {r['matrix']:<8} T={int(r['n_obs']):>4} purity {r['purity']:.3f} "
            f"(null {r['purity_null_mean']:.3f}, p {r['purity_p']:.4f}); ARI {r['ari']:+.3f} "
            f"(p {r['ari_p']:.4f}); mean support {r['mean_support']:.3f}"
        )
    if not cross.empty:
        print()
        for _, r in cross.iterrows():
            print(
                f"survival {r['horizon_a']}-{r['horizon_b']} {r['matrix']:<8} {r['survival']:.3f} "
                f"(noise mean {r['noise_mean']:.3f}, q95 {r['noise_q95']:.3f})"
            )
    if not rolling.empty:
        print()
        for _, r in rolling.iterrows():
            print(
                f"rolling {r['horizon']} {r['matrix']:<8} real {r['real_mean']:.3f} vs shuffled "
                f"{r['null_mean']:.3f} (lower tail {r['lower_tail']:.4f})"
            )

    if not args.quick:
        header = header_lines(
            summary,
            n_bootstrap=n_boot,
            n_permutations=n_perm,
            n_shuffles=n_shuf,
            n_null_draws=n_null,
            seed=args.seed,
            end=end,
            reads_sealed=reads_sealed,
        )
        tables = paths.output_path("reports", "tables", SUMMARY_MD).parent
        tables.mkdir(parents=True, exist_ok=True)
        edges.to_csv(paths.output_path("reports", "tables", EDGES_CSV), index=False)
        summary.to_csv(paths.output_path("reports", "tables", SUMMARY_CSV), index=False)
        cross.to_csv(paths.output_path("reports", "tables", CROSS_CSV), index=False)
        rolling.to_csv(paths.output_path("reports", "tables", ROLLING_CSV), index=False)
        paths.output_path("reports", "tables", SUMMARY_MD).write_text(
            render_markdown(header, summary, cross, rolling)
        )
        written = save_figure(figure, FIGURE_NAME, paths)
        print(f"\nwrote tables under {tables} and figures {[str(p) for p in written]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
