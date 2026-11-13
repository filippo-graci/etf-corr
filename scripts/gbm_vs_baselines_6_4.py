"""LightGBM against the baseline ladder on the validation window (Phase 6, step 6.4). Run with:
``uv run python scripts/gbm_vs_baselines_6_4.py`` (roughly half an hour: the tuned LightGBM
refits dominate; ``--quick`` runs T1 only with 100 bootstrap draws and no DCC, prints and writes
nothing: not a result).

What it does, on the core panel cut at the end of validation (the test window is not read):

1. reads which declared design variants were adopted (``reports/tables/variants_6_0.json``,
   step 6.0) and applies them; with no such file the design of tag ``design-v1`` is used;
2. runs the ladder of ``etfcorr.evaluation.ladder`` for T1, T2 and T3 with rung 3 added:
   LightGBM tuned by Optuna inside each training window (``forecasting.gbm``: 20 trials per
   refit on purged inner folds), refitted for the five seeds of the config and averaged;
3. tests LightGBM against the frozen comparators with the HLN-corrected Diebold-Mariano test
   (for T1: EWMA, the best of the four baselines H3 names, as primary, and DCC as the stricter
   line; for the other tasks the best simple baseline of the table), and builds the Model
   Confidence Set of every task;
4. refits LightGBM with each single seed (same hyperparameters) to show the seed spread, and
   reports mean absolute TreeSHAP contributions of the first refit for the H3 target.

**Status: validation only, "H3 preliminary".** H3 is read once, on the test window, in Phase 7.
The comparators were chosen on this same window, several tasks are tested without a
multiplicity correction, and the DM test is oversized at these sample sizes
(``reports/tables/dm_calibration_6_2.csv``). Nothing here is a trading result.

Outputs (not under ``--quick``): ``reports/tables/gbm_table_6_4.{csv,md}``,
``gbm_tests_6_4.csv``, ``gbm_mcs_6_4.csv``, ``gbm_summary_6_4.json``; validation forecasts under
``artifacts/gbm_6_4/``.
"""

import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

from etfcorr.config import Forecasting, Project, Variant, config_hash, load_project
from etfcorr.evaluation.ladder import (
    GBM_NAME,
    TaskResult,
    dcc_path,
    gbm_factory,
    ladder_table,
    load_sealed_panel,
    run_ladder,
    seed_spread,
    table_lines,
)
from etfcorr.evaluation.report import dm_table, mcs_table
from etfcorr.splits import walk_forward_from_config

N_BOOT_FULL = 2000
SEED = 0
TARGETS = ("T1", "T2", "T3")
H3_NAMED = ("persistence", "ewma", "har", "constant_correlation")
DCC_CACHE = Path("artifacts") / "baselines_5_6" / "dcc_avg_corr.parquet"


def adopted_variant(project: Project) -> Variant | None:
    """The single adopted design variant recorded by step 6.0, if any."""
    design = project.experiment.forecasting
    path = project.paths.tables / "variants_6_0.json"
    if design is None or not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        adopted = [d["variant"] for d in json.load(f)["adoption"] if d["adopted"]]
    if len(adopted) > 1:
        raise SystemExit(f"more than one adopted variant ({adopted}): combine them explicitly")
    return design.variant(adopted[0]) if adopted else None


def comparators(results: list[TaskResult]) -> dict[str, dict[str, str]]:
    """Frozen comparators per task: best of the four H3 names (T1), and the best simple model."""
    out: dict[str, dict[str, str]] = {}
    for r in results:
        roles: dict[str, str] = {}
        if r.task.target == "T1":
            named = r.table[r.table["model"].isin(H3_NAMED)].sort_values("mean")
            roles["primary_named_in_h3"] = str(named["model"].iloc[0])
        roles["best_simple"] = r.best_simple
        out[r.task.name] = roles
    return out


def shap_first_refit(
    project: Project, design: Forecasting, result: TaskResult, calendar: pd.DatetimeIndex
) -> dict[str, float]:
    """Mean absolute TreeSHAP per feature for the first refit, on that fold's test origins."""
    task, splits = result.task, project.experiment.splits
    fold = walk_forward_from_config(calendar, splits, task.horizon)[0]
    best = result.fold_params[GBM_NAME][0]["best_params"]
    assert isinstance(best, dict)
    model = gbm_factory(
        design,
        task.horizon,
        splits.walk_forward.embargo_days,
        task.gbm_columns,
        project.experiment.seeds,
        params_override=best,
    )()
    X, y = task.X, task.y
    train = X.index.isin(calendar[fold.train]) & X.notna().all(axis=1) & y.notna()
    test = X.index.isin(calendar[fold.test]) & X.notna().all(axis=1)
    model.fit(X.loc[train], y.loc[train])
    values = model.mean_abs_contributions(X.loc[test])  # type: ignore[attr-defined]
    return {str(k): float(v) for k, v in values.items()}


def _num(x: float) -> float | None:
    return None if not math.isfinite(x) else float(x)


def markdown_table(table: pd.DataFrame) -> str:
    lines = [
        "| task | model | rung | mean [95% interval] | skill | vs best simple [95%] |",
        "|---|---|---|---|---|---|",
    ]
    for row in table.itertuples(index=False):
        gap = (
            "**best simple**"
            if row.best_simple
            else f"{row.diff_vs_best_simple:+.5f} [{row.diff_ci_lower:+.5f}, "
            f"{row.diff_ci_upper:+.5f}]"
        )
        lines.append(
            f"| {row.task} | {row.model} | {row.rung} | {row.mean:.5f} "
            f"[{row.ci_lower:.5f}, {row.ci_upper:.5f}] | {row.skill:+.3f} | {gap} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    quick = "--quick" in sys.argv
    n_boot = 100 if quick else N_BOOT_FULL
    project = load_project()
    exp = project.experiment
    design = exp.forecasting
    if design is None or design.gbm is None:
        raise SystemExit("config/experiment.yaml has no forecasting.gbm block (step 6.0)")
    returns, sectors, market, vix = load_sealed_panel(project)
    calendar = pd.DatetimeIndex(returns.index)
    variant = adopted_variant(project)
    smoke = "QUICK SMOKE RUN, NOT A RESULT. " if quick else ""
    print(f"VALIDATION WINDOW ONLY, TEST WINDOW UNREAD. {smoke}")
    print(
        f"core panel N={returns.shape[1]}, last date read {returns.index[-1].date()}, config "
        f"{config_hash(exp)[:12]}; adopted variant: {variant.id if variant else 'none'}; "
        f"LightGBM {design.gbm.n_trials} trials per refit, seeds {list(exp.seeds)}"
    )
    dcc = None
    if not quick:
        cache = project.paths.root / DCC_CACHE
        if cache.is_file() and "--recompute-dcc" not in sys.argv:
            dcc = pd.read_parquet(cache)
            if not dcc.index.equals(returns.index):
                raise SystemExit(f"{cache} is on another calendar; rerun with --recompute-dcc")
            print(f"DCC path reused from {cache} (written by scripts/baseline_table_5_6.py)")
        else:
            dcc = dcc_path(returns, sectors, max(design.horizons_days["T1"]))
    start = time.perf_counter()
    results = run_ladder(
        returns,
        sectors,
        exp.splits,
        design,
        market=market,
        vix=vix,
        dcc_avg_corr=dcc,
        lam=exp.ewma_lambda,
        targets=("T1",) if quick else TARGETS,
        variant=variant,
        gbm_seeds=exp.seeds[:2] if quick else exp.seeds,
        n_bootstrap=n_boot,
        alpha=exp.inference.alpha,
        seed=SEED,
    )
    print(f"ladder with rung 3 computed in {time.perf_counter() - start:.0f}s\n")
    for line in table_lines(results):
        print(line)

    roles = comparators(results)
    tests = dm_table(results, GBM_NAME, roles)
    print("Diebold-Mariano (HLN), LightGBM against each comparator; mean_diff < 0 favours LightGBM")
    print(tests.round(4).to_string(index=False))
    mcs = mcs_table(results, size=design.mcs_size, reps=n_boot, seed=SEED)
    print(f"\nModel Confidence Set at level {design.mcs_size} (block 2h)")
    for name, block in mcs.groupby("task", sort=False):
        kept = block[block["included"]]["model"].tolist()
        print(f"  {name}: {len(kept)} of {len(block)} kept: {kept}")
    seeds = exp.seeds[:2] if quick else exp.seeds
    spread = {r.task.name: seed_spread(r, design, exp.splits, calendar, seeds) for r in results}
    print("\nLightGBM mean loss by single seed (same hyperparameters), and the ensemble row")
    for r in results:
        s = spread[r.task.name]
        ensemble = float(r.losses[GBM_NAME].mean())
        print(
            f"  {r.task.name}: seeds {s.min():.5f} to {s.max():.5f} (mean {s.mean():.5f}); "
            f"ensemble {ensemble:.5f}"
        )
    primary = next(r for r in results if r.task.name == f"T1_h{design.primary_horizon_days}")
    shap = shap_first_refit(project, design, primary, calendar)
    print(f"\nmean |TreeSHAP| of the first refit, {primary.task.name}, top 6:")
    for name, value in list(shap.items())[:6]:
        print(f"  {name:<24} {value:.4f}")

    if not quick:
        tables = project.paths.tables
        artifacts = project.paths.artifacts / "gbm_6_4"
        tables.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)
        table = ladder_table(results)
        table.to_csv(tables / "gbm_table_6_4.csv", index=False)
        (tables / "gbm_table_6_4.md").write_text(markdown_table(table), encoding="utf-8")
        tests.to_csv(tables / "gbm_tests_6_4.csv", index=False)
        mcs.to_csv(tables / "gbm_mcs_6_4.csv", index=False)
        n_refits = sum(len(r.fold_params[GBM_NAME]) for r in results)
        payload = {
            "step": "6.4",
            "status": "validation window only; H3 preliminary, not a verdict",
            "config_hash": config_hash(exp)[:12],
            "adopted_variant": variant.id if variant else None,
            "comparators": roles,
            "gbm": {
                "n_trials_per_refit": design.gbm.n_trials,
                "refits": n_refits,
                "trials_total": design.gbm.n_trials * n_refits,
                "seeds": list(exp.seeds),
                "best_params_by_fold": {
                    r.task.name: [p["best_params"] for p in r.fold_params[GBM_NAME]]
                    for r in results
                },
                "loss_by_seed": {
                    k: {str(s): _num(v) for s, v in x.items()} for k, x in spread.items()
                },
                "ensemble_loss": {
                    r.task.name: _num(float(r.losses[GBM_NAME].mean())) for r in results
                },
            },
            "shap_first_refit_primary": shap,
            "mcs_size": design.mcs_size,
            "bootstrap": {"n": n_boot, "seed": SEED},
        }
        with open(tables / "gbm_summary_6_4.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        for r in results:
            frame = r.predictions.assign(target=r.task.y)
            frame = frame.dropna(how="all", subset=list(r.predictions.columns))
            frame.to_parquet(artifacts / f"predictions_{r.task.name}.parquet")
        print(f"\nwrote tables under {tables} and forecasts under {artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
