"""The baseline table for targets T1 to T3 (Phase 5, step 5.6). Run with:
``uv run python scripts/baseline_table_5_6.py`` (about 5 minutes, most of it the causal DCC fit;
``--no-dcc`` skips that baseline; ``--quick`` is a smoke run with 100 bootstrap draws and no DCC
that prints only and writes nothing: not a result).

What it does: loads the daily returns of the core panel, **drops every row dated after the
validation window before anything is computed** (``splits.seal_test``; the script prints the last
date it read and ``run_ladder`` refuses unsealed data), builds the features and targets, and runs
the ladder fixed in ``config/experiment.yaml`` (``forecasting``, fixed 2026-10-11) through the
purged, embargoed, expanding walk-forward of ``etfcorr.splits`` over the validation window
(2021-2022, quarterly refits). See ``etfcorr.evaluation.ladder`` for the tasks, the losses and
how to read the table.

**Status: out-of-sample within validation, descriptive.** The test window (2023 onwards) is not
read. The intervals are stationary-bootstrap percentile intervals on per-date losses; they are
not tests, the "best simple baseline" is picked on the same data, and nothing is corrected for
multiplicity. Nothing here is a trading result.

Outputs (not under ``--quick``): ``reports/tables/baseline_table_5_6.csv`` and ``.md`` (one row
per task and model), ``reports/tables/baseline_summary_5_6.json`` (settings, the best simple
baseline per task, tuned penalties, counts of configurations), and under
``artifacts/baselines_5_6/`` the validation forecasts of every model per task and the DCC
forecast path, for the Diebold-Mariano and Model-Confidence-Set comparisons of Phase 6.
"""

import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

from etfcorr.config import Project, config_hash, load_project
from etfcorr.evaluation.ladder import (
    TaskResult,
    dcc_path,
    ladder_table,
    run_ladder,
    table_lines,
)
from etfcorr.evaluation.ladder import load_sealed_panel as load_sealed

N_BOOT_FULL = 2000
SEED = 0
H3_NAMED = ("persistence", "ewma", "har", "constant_correlation")  # the baselines H3 names


def _num(x: float) -> float | None:
    return None if not math.isfinite(x) else float(x)


def summary_payload(
    project: Project, results: list[TaskResult], *, n_boot: int, seed: int, with_dcc: bool
) -> dict[str, object]:
    exp = project.experiment
    design = exp.forecasting
    assert design is not None
    tasks = {}
    for r in results:
        best = r.table[r.table["best_simple"]].iloc[0]
        named = r.table[r.table["model"].isin(H3_NAMED)]
        ascending = not r.task.higher_is_better
        tasks[r.task.name] = {
            "target": r.task.target,
            "horizon_days": r.task.horizon,
            "loss": r.task.loss,
            "higher_is_better": r.task.higher_is_better,
            "n_dates": int(best["n_dates"]),
            "n_effective": float(best["n_effective"]),
            "first_date": str(r.losses.index.min().date()),
            "last_date": str(r.losses.index.max().date()),
            "best_simple_baseline": r.best_simple,
            "best_simple_mean": _num(float(best["mean"])),
            "best_of_the_four_named_in_h3": (
                None
                if named.empty
                else str(named.sort_values("mean", ascending=ascending)["model"].iloc[0])
            ),
            "models": [str(m) for m in r.table["model"]],
            "tuned_values_count": r.tuned,
            "psd_projected_share": {k: _num(v) for k, v in r.psd_projected_share.items()},
        }
    return {
        "step": "5.6",
        "status": "validation window only (walk-forward, purged, embargoed); test window unread",
        "config_hash": config_hash(exp)[:12],
        "design_fixed_on": str(design.fixed_on),
        "primary": {"target": design.primary_target, "horizon_days": design.primary_horizon_days},
        "splits": {
            "train_start": str(exp.splits.train.start),
            "validation": [str(exp.splits.val.start), str(exp.splits.val.end)],
            "refit_every": exp.splits.walk_forward.refit_every,
            "embargo_days": exp.splits.walk_forward.embargo_days,
        },
        "bootstrap": {"n": n_boot, "seed": seed, "mean_block_length": "2 x horizon"},
        "dcc_baseline_included": with_dcc,
        "configurations": {
            "tasks": len(results),
            "task_model_cells": int(sum(len(r.table) for r in results)),
            "hyperparameter_values_per_refit": design.n_linear_trials,
            "seeds": "none (every model here is deterministic)",
        },
        "tasks": tasks,
    }


def markdown_table(table: pd.DataFrame) -> str:
    lines = [
        "| task | model | rung | loss | mean [95% interval] | skill | vs best simple [95%] |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in table.itertuples(index=False):
        gap = (
            "**best simple**"
            if row.best_simple
            else f"{row.diff_vs_best_simple:+.5f} [{row.diff_ci_lower:+.5f}, "
            f"{row.diff_ci_upper:+.5f}]"
        )
        lines.append(
            f"| {row.task} | {row.model} | {row.rung} | {row.loss} | {row.mean:.5f} "
            f"[{row.ci_lower:.5f}, {row.ci_upper:.5f}] | {row.skill:+.3f} | {gap} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    root: Path, results: list[TaskResult], payload: dict[str, object], dcc: pd.DataFrame | None
) -> list[Path]:
    tables = root / "reports" / "tables"
    artifacts = root / "artifacts" / "baselines_5_6"
    tables.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    table = ladder_table(results)
    written = [
        tables / "baseline_table_5_6.csv",
        tables / "baseline_table_5_6.md",
        tables / "baseline_summary_5_6.json",
    ]
    table.to_csv(written[0], index=False)
    written[1].write_text(markdown_table(table), encoding="utf-8")
    with open(written[2], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    for r in results:
        frame = r.predictions.assign(target=r.task.y).dropna(
            how="all", subset=r.predictions.columns
        )
        path = artifacts / f"predictions_{r.task.name}.parquet"
        frame.to_parquet(path)
        written.append(path)
    if dcc is not None:
        dcc.to_parquet(artifacts / "dcc_avg_corr.parquet")
        written.append(artifacts / "dcc_avg_corr.parquet")
    return written


def main() -> int:
    quick = "--quick" in sys.argv
    with_dcc = not quick and "--no-dcc" not in sys.argv
    n_boot = 100 if quick else N_BOOT_FULL
    project = load_project()
    design = project.experiment.forecasting
    if design is None:
        raise SystemExit("config/experiment.yaml has no forecasting block (step 5.1)")
    returns, sectors, market, vix = load_sealed(project)
    smoke = "QUICK SMOKE RUN, NOT A RESULT. " if quick else ""
    print(f"VALIDATION WINDOW ONLY, TEST WINDOW UNREAD. {smoke}")
    print(
        f"core panel N={returns.shape[1]}, {len(returns)} daily returns "
        f"{returns.index[0].date()} to {returns.index[-1].date()} (last date read); "
        f"design fixed {design.fixed_on}, config {config_hash(project.experiment)[:12]}"
    )
    dcc = None
    if with_dcc:
        start = time.perf_counter()
        dcc = dcc_path(returns, sectors, max(design.horizons_days["T1"]))
        print(f"causal DCC path computed in {time.perf_counter() - start:.0f}s")
    start = time.perf_counter()
    results = run_ladder(
        returns,
        sectors,
        project.experiment.splits,
        design,
        market=market,
        vix=vix,
        dcc_avg_corr=dcc,
        lam=project.experiment.ewma_lambda,
        n_bootstrap=n_boot,
        alpha=project.experiment.inference.alpha,
        seed=SEED,
    )
    print(f"ladder computed in {time.perf_counter() - start:.0f}s\n")
    for line in table_lines(results):
        print(line)
    if not quick:
        payload = summary_payload(project, results, n_boot=n_boot, seed=SEED, with_dcc=with_dcc)
        for path in write_outputs(project.paths.root, results, payload, dcc):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
