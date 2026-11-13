"""Declared design variants and robustness runs against the design (Phase 6, step 6.0). Run with:
``uv run python scripts/variants_6_0.py`` (about 10 minutes: two causal DCC fits and several
ladder runs; ``--quick`` is a smoke run with 100 bootstrap draws and no DCC that prints only and
writes nothing: not a result).

The variants are the ones declared in ``config/experiment.yaml`` (``forecasting.variants``,
tag ``design-v2-declared``) **before any of them was run**:

* V1: lognormal bias correction of the fitted log-variance forecasts (T2);
* V2: the ridge penalty grid extended upward (T1, T2, T3);
* R1: spring 2020 excluded from training (T1), robustness only;
* R2: the DCC baseline fitted without AMLP (T1), robustness only.

For each one the script re-runs the ladder on the validation window with that single change and
reports, per task and model, the loss under the design, under the variant, and the mean per-date
difference with a stationary-bootstrap interval. It then applies the adoption rules written in
the config for V1 and V2: a variant is adopted for Phase 6 only if its named loss falls with a
95% interval that excludes zero.

**Status: validation window only; the test window is not read.** The variants were chosen after
the Phase 5 table was seen, so they are extra configurations (counted in the output), and an
adopted variant is a design choice made on validation data, not a finding.

Outputs (not under ``--quick``): ``reports/tables/variants_6_0.csv`` and ``.md``, and
``reports/tables/variants_6_0.json`` (adoption decisions, settings, configuration counts).
"""

import json
import sys
import time

import pandas as pd

from etfcorr.config import Forecasting, Project, Variant, config_hash, load_project
from etfcorr.evaluation.ladder import (
    TaskResult,
    compare_with_base,
    dcc_path,
    load_sealed_panel,
    run_ladder,
)

N_BOOT_FULL = 2000
SEED = 0
BASE_TARGETS = ("T1", "T2", "T3")
# adoption rules of the config, as (variant, model, tasks that must all improve)
RULES: dict[str, tuple[str, tuple[str, ...]]] = {
    "V1": ("har", ("T2_h5", "T2_h21")),
    "V2": ("ridge", ("T1_h21",)),
}


def adoption(table: pd.DataFrame, variant_id: str) -> dict[str, object]:
    """Apply one adoption rule: every named row must have a difference interval below zero."""
    model, tasks = RULES[variant_id]
    rows = table[(table["variant"] == variant_id) & (table["model"] == model)].set_index("task")
    checks = {}
    for task in tasks:
        if task not in rows.index:
            checks[task] = {"diff": 0.0, "upper": 0.0, "improves": False, "note": "no change"}
            continue
        row = rows.loc[task]
        upper = float(row["diff_ci_upper"])
        checks[task] = {"diff": float(row["diff"]), "upper": upper, "improves": upper < 0.0}
    return {
        "variant": variant_id,
        "model": model,
        "checks": checks,
        "adopted": all(bool(c["improves"]) for c in checks.values()),
    }


def run_variants(
    project: Project,
    returns: pd.DataFrame,
    sectors: dict[str, str],
    market: pd.Series,
    vix: pd.Series,
    *,
    n_boot: int,
    with_dcc: bool,
    variants: tuple[Variant, ...] | None = None,
) -> tuple[pd.DataFrame, list[TaskResult]]:
    """The design once, then each declared variant; returns the comparison table and the base."""
    exp = project.experiment
    design: Forecasting | None = exp.forecasting
    if design is None:
        raise SystemExit("config/experiment.yaml has no forecasting block")
    todo = design.variants if variants is None else variants
    horizon = max(design.horizons_days["T1"])
    kw = {"market": market, "vix": vix, "lam": exp.ewma_lambda, "n_bootstrap": n_boot}
    kw["alpha"], kw["seed"] = exp.inference.alpha, SEED
    dcc = dcc_path(returns, sectors, horizon) if with_dcc else None
    base = run_ladder(
        returns, sectors, exp.splits, design, dcc_avg_corr=dcc, targets=BASE_TARGETS, **kw
    )
    tables = []
    for variant in todo:
        path = dcc
        if variant.kind == "dcc_drop_assets":
            if not with_dcc:
                continue
            path = dcc_path(returns, sectors, horizon, drop=variant.assets)
        changed = run_ladder(
            returns, sectors, exp.splits, design, dcc_avg_corr=path, variant=variant, **kw
        )
        tables.append(
            compare_with_base(
                base, changed, variant.id, n_bootstrap=n_boot, alpha=exp.inference.alpha, seed=SEED
            )
        )
    return pd.concat(tables, ignore_index=True), base


def markdown_table(table: pd.DataFrame) -> str:
    lines = [
        "| variant | task | model | design | variant | difference [95% interval] |",
        "|---|---|---|---|---|---|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.task} | {row.model} | {row.base_mean:.5f} | "
            f"{row.variant_mean:.5f} | {row.diff:+.5f} [{row.diff_ci_lower:+.5f}, "
            f"{row.diff_ci_upper:+.5f}] |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    quick = "--quick" in sys.argv
    n_boot = 100 if quick else N_BOOT_FULL
    project = load_project()
    returns, sectors, market, vix = load_sealed_panel(project)
    smoke = "QUICK SMOKE RUN, NOT A RESULT. " if quick else ""
    print(f"VALIDATION WINDOW ONLY, TEST WINDOW UNREAD. {smoke}")
    print(
        f"core panel N={returns.shape[1]}, last date read {returns.index[-1].date()}, "
        f"config {config_hash(project.experiment)[:12]}"
    )
    start = time.perf_counter()
    table, _ = run_variants(
        project, returns, sectors, market, vix, n_boot=n_boot, with_dcc=not quick
    )
    print(f"computed in {time.perf_counter() - start:.0f}s\n")
    print(markdown_table(table))
    decisions = [adoption(table, v) for v in RULES]
    for d in decisions:
        print(f"{d['variant']} ({d['model']}): {'ADOPTED' if d['adopted'] else 'not adopted'}")
        for task, check in d["checks"].items():  # type: ignore[attr-defined]
            print(f"  {task}: diff {check['diff']:+.5f}, interval upper {check['upper']:+.5f}")
    if not quick:
        out = project.paths.tables
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "variants_6_0.csv", index=False)
        (out / "variants_6_0.md").write_text(markdown_table(table), encoding="utf-8")
        design = project.experiment.forecasting
        assert design is not None
        payload = {
            "step": "6.0",
            "status": "validation window only; variants declared before they were run",
            "config_hash": config_hash(project.experiment)[:12],
            "variants_run": [v.id for v in design.variants],
            "extra_configurations": len(design.variants),
            "bootstrap": {"n": n_boot, "seed": SEED},
            "adoption": decisions,
        }
        with open(out / "variants_6_0.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"\nwrote {out / 'variants_6_0.csv'}, .md and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
