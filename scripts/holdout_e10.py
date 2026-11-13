"""Forward hold-out E10 of E3: long-short minimum variance with nonlinear shrinkage. Run with:
``uv run python scripts/holdout_e10.py --confirm-run``, once, after 2027-10-07.

Design: E10 in ``config/experiment.yaml``, declared and tagged on 2026-11-01, when no price dated in
the hold-out window (2026-10-01 to 2027-09-30) was on disk. The statistic is E3's, computed with
``scripts/followup_h5.py`` (``run_cells`` and ``tables``) unchanged: the ``minvar_ls`` portfolio
of the 25 core funds, ``nonlinear`` against ``sample`` at the D, W, M and Q frequencies, the mean
of the four volatility differences, a joint stationary bootstrap of days. Supported if its one-sided
p is below ``inference.alpha``.

Data come from the separate root ``holdout/post2026`` (its three config files are symlinks to the
main ones), built with the Phase 1 commands once the window has closed::

    uv run etfcorr data download --root holdout/post2026 --asof 2027-09-30
    uv run etfcorr data build --root holdout/post2026 --asof 2027-09-30
    uv run etfcorr returns --root holdout/post2026

The script refuses to run before 2027-10-07 or when ``coverage.csv`` says a core fund stops before
2027-09-30 (both checked before the marker, reading no price). Then it writes the marker
``reports/tables/e10_holdout_marker.json`` (``evaluation.final.open_final``: clean working tree, no
earlier run) and reads the returns. Outputs: ``reports/tables/e10_holdout_{cells,pooled}.csv`` and
``e10_holdout_summary.json``. Long-short weights ignore borrow costs and leverage limits.
"""

import argparse
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from types import ModuleType

import pandas as pd

from etfcorr.config import DateWindow, Economics, config_hash, find_repo_root, load_project
from etfcorr.evaluation.final import close_final, git_state, open_final

HOLDOUT_ROOT = Path("holdout") / "post2026"
WINDOW = DateWindow(start="2026-10-01", end="2027-09-30")
NOT_BEFORE = date(2027, 10, 7)
MARKER = "e10_holdout_marker.json"
STEM = "e10_holdout"
ESTIMATORS = ("sample", "ledoit_wolf_corr", "rmt", "nonlinear", "nonlinear_corr")  # E3's BASE
PRIMARY = ("minvar_ls", "nonlinear")
FREQUENCIES = 4


def load_script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_ready(today: date, coverage: pd.DataFrame | None) -> None:
    """Refuse before 2027-10-07, or when a core fund's last valid row is before the window end."""
    if today < NOT_BEFORE:
        raise SystemExit(f"E10 may not run before {NOT_BEFORE} (today is {today})")
    if coverage is None:
        raise SystemExit(f"no coverage.csv under {HOLDOUT_ROOT}: run the data commands first")
    core = coverage[coverage["panel"] == "core"]
    short = core.loc[pd.to_datetime(core["last_date"]) < pd.Timestamp(WINDOW.end), "ticker"]
    if len(short):
        raise SystemExit(f"data stop before {WINDOW.end} for {short.tolist()}: download again")


def evaluate(
    panels: dict[str, pd.DataFrame],
    design: Economics,
    *,
    lam: float,
    n_bootstrap: int,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """E3's cells and pooled table on the hold-out window, and the E10 reading."""
    h5 = load_script("followup_h5")
    estimation = dict(design.estimation)
    runs, skipped = h5.run_cells(panels, estimation, ESTIMATORS, WINDOW, lam, None)
    if skipped:
        raise SystemExit(f"frequencies without a full estimation window: {skipped}")
    cells, pooled = h5.tables(runs, design.block_days, n_bootstrap, alpha)
    row = pooled[(pooled["portfolio"] == PRIMARY[0]) & (pooled["pool"] == PRIMARY[1])]
    if len(row) != 1 or int(row["n_cells"].iloc[0]) != FREQUENCIES:
        raise SystemExit("the E10 pool must hold exactly the four nonlinear long-short cells")
    reading = row.iloc[0].to_dict()
    reading["supported"] = bool(reading["p_one_sided"] < alpha)
    reading["alpha"] = alpha
    return cells, pooled, reading


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--confirm-run", action="store_true", required=True)
    parser.parse_args()
    root = find_repo_root()
    project = load_project(root)
    holdout = load_project(root / HOLDOUT_ROOT)
    coverage = holdout.paths.data_processed / "coverage.csv"
    check_ready(date.today(), pd.read_csv(coverage) if coverage.is_file() else None)
    marker = project.paths.tables / MARKER
    open_final(project, marker, True, "", True)
    commit, _ = git_state(project)

    exp = project.experiment
    design = exp.economics
    assert design is not None
    core = list(project.universe.tickers_in("core"))
    end = pd.Timestamp(WINDOW.end)
    panels = {}
    for freq in design.estimation:
        frame = pd.read_parquet(holdout.paths.data_processed / f"returns_{freq}.parquet")
        panels[freq] = frame.loc[frame.index <= end, core].dropna()
    cells, pooled, reading = evaluate(
        panels,
        design,
        lam=exp.ewma_lambda,
        n_bootstrap=exp.inference.n_bootstrap,
        alpha=exp.inference.alpha,
    )
    print(cells.drop(columns="n_days").round(4).to_string(index=False))
    print(pooled.round(4).to_string(index=False))
    word = "SUPPORTED" if reading["supported"] else "NOT SUPPORTED"
    print(
        f"\nE10 {word}: nonlinear minus sample, long-short, mean over {FREQUENCIES} frequencies "
        f"{reading['mean_diff']:+.4f} [{reading['ci_lower']:+.4f}, {reading['ci_upper']:+.4f}], "
        f"one-sided p = {reading['p_one_sided']:.4f} ({cells['n_days'].iloc[0]} days)"
    )

    out = project.paths.tables
    cells.to_csv(out / f"{STEM}_cells.csv", index=False)
    pooled.to_csv(out / f"{STEM}_pooled.csv", index=False)
    summary = {
        "status": "E10, forward hold-out of E3, run once",
        "config_hash": config_hash(exp)[:12],
        "commit": commit,
        "window": [str(WINDOW.start), str(WINDOW.end)],
        "last_date_read": str(panels["D"].index[-1].date()),
        "E10": reading,
        "bootstrap": {"n": exp.inference.n_bootstrap, "block_days": design.block_days, "seed": 0},
    }
    (out / f"{STEM}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    close_final(marker)
    print(f"wrote {STEM}_*.csv/json under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
