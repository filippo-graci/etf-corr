"""The estimator-stability table (Phase 3, step 3.5), on the real core panel.

Run with ``uv run python scripts/estimator_stability_3_5.py`` (about 15 minutes at the default 2000
draws, most of it Kendall's tau; ``--n-bootstrap 500`` is about 4 minutes; ``--quick`` is a 30-draw
smoke run that prints only and writes nothing: not a result; ``--estimators pearson ledoit_wolf``
restricts the list).

What it does: for each horizon (D, W, M, Q) it takes the **core** tickers of
``config/universe.yaml`` (complete-case rows), keeps the rows up to the **validation end of the
sealed split** (2022-12-31, read from ``config/experiment.yaml``) and runs
``etfcorr.correlation.stability.estimator_stability_table``: the condition number, the average
correlation with its stationary-bootstrap spread, the RMS pairwise spread, the leading-eigenvector
stability and the PSD share, for Pearson, Spearman, Kendall, Ledoit-Wolf, OAS, EWMA, denoised and
detoned matrices. The default number of draws is ``inference.n_bootstrap`` of the experiment
config.

Why the 2022 cut: the table is a candidate input to the choice of covariance estimator in Phase 8
(H5) and to feature design in Phases 5-7, and 2023 onward is the sealed test window (touched once,
in Phase 7). ``--full-sample`` removes the cut and the output says so; do not use such a table to
choose an estimator that is later judged on 2023 onward.

**Status: descriptive, not a test.** The bootstrap spread measures variance only; shrinkage and
denoising trade variance for bias, so a lower spread is not a better estimator (H5 is the accuracy
comparison, Phase 8). No claim of tradeable value. Quarterly rows are exploratory (T close to N).

Outputs (not under ``--quick``): ``reports/tables/estimator_stability_3_5.csv`` (one row per
horizon and estimator, all columns of ``TABLE_COLUMNS``) and
``reports/tables/estimator_stability_3_5.md`` (a readable version with the settings and caveats).
"""

import argparse
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pandas as pd
import yaml

from etfcorr.config import Horizon, Paths, find_repo_root
from etfcorr.correlation.stability import (
    ESTIMATORS,
    EstimatorName,
    estimator_stability_table,
    stability_summary_lines,
)

HORIZONS: tuple[Horizon, ...] = ("D", "W", "M", "Q")
QUICK_DRAWS = 30
SEED = 0
CSV_NAME = "estimator_stability_3_5.csv"
MD_NAME = "estimator_stability_3_5.md"

# Columns of the readable table, with their headers.
MD_COLUMNS: tuple[tuple[str, str], ...] = (
    ("horizon", "hor"),
    ("estimator", "estimator"),
    ("n_obs", "T"),
    ("q", "N/T"),
    ("cond", "cond"),
    ("is_psd", "PSD"),
    ("avg_corr", "avg corr"),
    ("ci", "95% interval"),
    ("boot_bias", "boot bias"),
    ("boot_sd", "boot sd"),
    ("rms_pair_sd", "rms pair sd"),
    ("rel_spread", "rel spread"),
    ("eigvec_overlap", "eigvec overlap"),
    ("psd_share_boot", "PSD share"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--quick", action="store_true", help="30 draws, print only, write nothing")
    parser.add_argument("--n-bootstrap", type=int, default=None, help="default: config value")
    parser.add_argument("--estimators", nargs="+", default=list(ESTIMATORS), choices=ESTIMATORS)
    parser.add_argument("--horizons", nargs="+", default=list(HORIZONS), choices=HORIZONS)
    parser.add_argument("--seed", type=int, default=SEED)
    window = parser.add_mutually_exclusive_group()
    window.add_argument("--end", default=None, help="last date used (default: validation end)")
    window.add_argument(
        "--full-sample", action="store_true", help="no end date: READS THE SEALED TEST WINDOW"
    )
    return parser.parse_args(argv)


def read_returns(path: Path) -> pd.DataFrame:
    """One ``returns_<h>.parquet``, flattened to a plain ticker-column frame."""
    frame = pd.read_parquet(path)
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.xs(frame.columns.levels[-1][0], axis=1, level=-1)
    return frame


def core_tickers(root: Path) -> list[str]:
    with open(root / "config" / "universe.yaml") as f:
        universe = yaml.safe_load(f)
    return [t["ticker"] for t in universe["tickers"] if t["panel"] == "core"]


def validation_end(root: Path) -> str:
    """End of the validation split: the last date this script may read by default."""
    with open(root / "config" / "experiment.yaml") as f:
        experiment = yaml.safe_load(f)
    return str(experiment["splits"]["val"][1])


def n_bootstrap_default(root: Path) -> int:
    with open(root / "config" / "experiment.yaml") as f:
        experiment = yaml.safe_load(f)
    return int(experiment["inference"]["n_bootstrap"])


def select_core(frame: pd.DataFrame, tickers: Sequence[str], horizon: str) -> pd.DataFrame:
    """The core columns in universe order; a missing core ticker is an error, not a warning."""
    missing = [t for t in tickers if t not in frame.columns]
    if missing:
        raise SystemExit(f"returns_{horizon}.parquet is missing core tickers: {missing}")
    return frame[list(tickers)]


def _fmt(value: float, spec: str) -> str:
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return format(value, spec)


def markdown_table(table: pd.DataFrame, header: Sequence[str] = ()) -> str:
    """A GitHub-style table of the main columns, with ``header`` lines above it."""
    rows: list[list[str]] = []
    for r in table.itertuples(index=False):
        rows.append(
            [
                str(r.horizon),
                str(r.estimator),
                str(int(r.n_obs)),
                f"{r.q:.3f}",
                _fmt(float(r.cond), ".3g"),
                "yes" if bool(r.is_psd) else "NO",
                f"{r.avg_corr:+.3f}",
                f"[{r.ci_lower:+.3f}, {r.ci_upper:+.3f}]",
                f"{r.boot_bias:+.4f}",
                f"{r.boot_sd:.4f}",
                f"{r.rms_pair_sd:.4f}",
                f"{r.rel_spread:.3f}",
                f"{r.eigvec_overlap:.3f}",
                f"{r.psd_share_boot:.2f}",
            ]
        )
    heads = [h for _, h in MD_COLUMNS]
    lines = [*header, ""] if header else []
    lines.append("| " + " | ".join(heads) + " |")
    lines.append("|" + "|".join("---" for _ in heads) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def header_lines(
    table: pd.DataFrame, *, n_bootstrap: int, seed: int, end: str | None, reads_sealed: bool
) -> list[str]:
    windows = (
        table.groupby("horizon", sort=False)
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
        for h, b in table.groupby("horizon", sort=False)["mean_block_length"].first().items()
    )
    n_assets = int(table["n_assets"].iloc[0])
    sealed = (
        "WARNING: this table reads the sealed test window (2023 onward)."
        if reads_sealed
        else f"The sealed test window is not read (last date used: {end})."
    )
    return [
        "# Estimator-stability table (step 3.5)",
        "",
        f"Core panel, N = {n_assets}. {spans}.",
        f"Stationary bootstrap, {n_bootstrap} draws, seed {seed}, mean block lengths {blocks}; "
        "the same replicate rows are used for every estimator within a horizon.",
        sealed,
        "",
        "Descriptive, not a test. Spread is variance only: lower spread is not higher accuracy "
        "(shrinkage and denoising add bias; the accuracy comparison is H5, Phase 8).",
        "Detoned: singular by construction (cond = inf); its average correlation is on a relative "
        "scale, compare rel spread only. EWMA: order-dependent, its spread mixes sampling error "
        "with time variation. Quarterly rows are exploratory (T close to N).",
    ]


def run(
    returns_by_horizon: dict[Horizon, pd.DataFrame],
    estimators: Sequence[EstimatorName],
    *,
    end: str | None,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    return estimator_stability_table(
        returns_by_horizon, estimators, end=end, n_bootstrap=n_bootstrap, seed=seed
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = find_repo_root()
    paths = Paths.from_root(root)
    tickers = core_tickers(root)
    sealed_from = validation_end(root)
    end: str | None = None if args.full_sample else (args.end or sealed_from)
    reads_sealed = end is None or pd.Timestamp(end) > pd.Timestamp(sealed_from)
    n_boot = QUICK_DRAWS if args.quick else (args.n_bootstrap or n_bootstrap_default(root))

    frames: dict[Horizon, pd.DataFrame] = {}
    for horizon in args.horizons:
        raw = read_returns(root / "data" / "processed" / f"returns_{horizon}.parquet")
        frames[cast(Horizon, horizon)] = select_core(raw, tickers, horizon)

    smoke = "QUICK SMOKE RUN, NOT A RESULT. " if args.quick else ""
    print(f"DESCRIPTIVE, NOT A TEST. {smoke}core N={len(tickers)}, bootstrap draws {n_boot}")
    print(
        "WARNING: reads the sealed test window (2023 onward)."
        if reads_sealed
        else f"sealed test window not read (last date used {end})"
    )
    start = time.perf_counter()
    table = run(frames, args.estimators, end=end, n_bootstrap=n_boot, seed=args.seed)
    print(f"computed in {time.perf_counter() - start:.1f}s")
    print()
    for line in stability_summary_lines(table):
        print(line)

    if not args.quick:
        header = header_lines(
            table, n_bootstrap=n_boot, seed=args.seed, end=end, reads_sealed=reads_sealed
        )
        out_csv = paths.output_path("reports", "tables", CSV_NAME)
        out_md = paths.output_path("reports", "tables", MD_NAME)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_csv, index=False)
        out_md.write_text(markdown_table(table, header))
        print(f"\nwrote {out_csv} and {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
