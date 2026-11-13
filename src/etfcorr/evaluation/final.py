"""Step 7.3: ``etfcorr eval``, the frozen ladder on validation, or once on the test window.

One code path, two windows. Without ``--final`` it runs the ladder of ``forecasting.final``
(rungs 0 to 3, with one LightGBM per declared graph kind) on the **validation** window: this is
the rehearsal, and the run that names the primary challenger for the freeze. With ``--final`` it
reads the test window, **once**:

* it refuses unless ``forecasting.final`` is frozen (``primary_challenger`` and ``frozen_on``
  set), the working tree is clean, ``--confirm-unseal`` is given, and no marker of an earlier
  final run exists;
* the marker (``reports/tables/eval_7_final_marker.json``) is written **before** the data are
  read, so a run that crashes still counts as having opened the window. Removing the marker is a
  decision for a visible commit, not for this code.

**How H3 is read** (fixed here before the test window is opened; the pre-registered test is
"Diebold-Mariano with the HLN correction, Clark-West if the models are nested, on out-of-sample
loss against the best simple baseline, plus a Model Confidence Set"):

* target: ``forecasting.primary_target`` at ``primary_horizon_days`` (T1 at 21 days);
* comparator: ``final.comparator_primary`` (EWMA: the best of the four baselines H3 names, chosen
  on validation); ``final.comparator_strict`` (DCC) is reported beside it;
* challengers: ``final.primary_challenger``, the rung 2-3 model with the lowest validation loss
  (a penalised regression), and ``final.second_challenger``, the best tree/graph model on
  validation (decision of 2026-10-11: test both, so that H3 does not hinge on whether a
  penalised regression counts as machine learning);
* test: one-sided DM-HLN for each challenger, multiplied by the number of challengers
  (Bonferroni). **H3 is supported if, for at least one challenger, that adjusted p-value is
  below ``inference.alpha`` and the mean loss difference favours it.** The smallest adjusted
  p-value is the one Phase 8 carries into Holm. Clark-West is **not** the verdict test:
  step 6.2 already recorded that penalised and tree
  models are not nested in the simple baselines in the regression sense (the baselines are
  unfitted columns, the challengers are penalised), and Clark-West's size was never calibrated
  here. Its p-value is reported beside the verdict for the penalised linear models against the
  named baselines, as a sensitivity line (on validation it is far more liberal than DM-HLN).
* beside the verdict, not part of it: the same test for every rung 2-3 model with a Bonferroni
  bound over that family, the stricter comparator, and the Model Confidence Set.

Limits to state with the verdict: the DM test over-rejects at this horizon (measured size 7 to
10% at nominal 5%, ``reports/tables/dm_calibration_6_2.csv``); the challenger and the comparators
were chosen on about 23 independent months; the test window is one stretch of market history;
the secondary tasks get a table and a Model Confidence Set only, and their "best simple baseline"
column is picked on the evaluated window itself (descriptive). Nothing here is a trading result.
"""

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from etfcorr.config import ExperimentConfig, Final, Project, TargetName, config_hash
from etfcorr.evaluation.dm_mcs import clark_west
from etfcorr.evaluation.ladder import (
    GBM_NAME,
    TaskResult,
    dcc_path,
    ladder_table,
    load_sealed_panel,
    run_ladder,
    table_lines,
)
from etfcorr.evaluation.report import dm_table, mcs_table

H3_NAMED: tuple[str, ...] = ("persistence", "ewma", "har", "constant_correlation")
NESTING_LINEAR: tuple[str, ...] = ("ridge", "lasso", "elastic_net")
MARKER = "eval_7_final_marker.json"
VALIDATION_DCC = ("baselines_5_6", "dcc_avg_corr.parquet")


class EvalError(RuntimeError):
    """The evaluation was refused or cannot run."""


@dataclass(frozen=True)
class Evaluation:
    results: list[TaskResult]
    table: pd.DataFrame
    tests: pd.DataFrame
    mcs: pd.DataFrame
    verdict: dict[str, object]


def challenger_tests(
    results: Sequence[TaskResult], roles: Mapping[str, Mapping[str, str]]
) -> pd.DataFrame:
    """Every rung 2-3 model against each comparator of its task: DM-HLN, Clark-West beside it.

    ``roles`` maps a task name to ``{role: comparator}``; comparators absent from the run are
    skipped. ``p_one_sided`` (DM-HLN) is the p-value the verdict uses; ``p_clark_west`` is a
    sensitivity line, filled for a penalised linear model against a named baseline only.
    ``mean_diff < 0`` favours the model.
    """
    parts = []
    for r in results:
        wanted = {k: v for k, v in roles.get(r.task.name, {}).items() if v in r.losses.columns}
        if not wanted:
            continue
        forecasts = r.predictions.loc[r.losses.index]
        for spec in r.task.models:
            if spec.rung < 2:
                continue
            block = dm_table([r], spec.name, {r.task.name: wanted})
            block.insert(2, "rung", spec.rung)
            nested = (
                (r.task.loss == "mse_fisher_z")
                & (spec.name in NESTING_LINEAR)
                & block["comparator"].isin(H3_NAMED)
            )
            block["p_clark_west"] = [
                clark_west(r.task.y, forecasts[c], forecasts[spec.name], r.task.horizon).p_value
                if is_nested
                else float("nan")
                for c, is_nested in zip(block["comparator"], nested, strict=True)
            ]
            parts.append(block)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def best_challenger(result: TaskResult) -> str:
    """The rung 2-3 model with the best mean loss (what the freeze names, on validation)."""
    names = [s.name for s in result.task.models if s.rung >= 2]
    means = result.losses[names].mean()
    return str(means.idxmax() if result.task.higher_is_better else means.idxmin())


def h3_verdict(
    result: TaskResult, tests: pd.DataFrame, mcs: pd.DataFrame, final: Final, alpha: float
) -> dict[str, object]:
    """The H3 reading of one evaluated window (see the module docstring for the rule)."""
    name = result.task.name
    challenger = final.primary_challenger or best_challenger(result)
    challengers = [challenger, *([final.second_challenger] if final.second_challenger else [])]
    mine = tests[tests["task"] == name]

    def line(comparator: str, challenger: str = challenger) -> dict[str, object] | None:
        row = mine[(mine["model"] == challenger) & (mine["comparator"] == comparator)]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "comparator": comparator,
            "loss_challenger": float(result.losses[challenger].mean()),
            "loss_comparator": float(result.losses[comparator].mean()),
            "mean_diff": float(r["mean_diff"]),
            "test": "dm_hln",
            "p_one_sided": float(r["p_one_sided"]),
            "p_clark_west_sensitivity": float(r["p_clark_west"]),
            "p_adjusted": float(min(1.0, r["p_one_sided"] * len(challengers))),
            "better_and_significant": bool(
                r["mean_diff"] < 0 and r["p_one_sided"] * len(challengers) < alpha
            ),
        }

    primary = line(final.comparator_primary)
    if primary is None:
        raise EvalError(f"{name}: no test of {challenger!r} against {final.comparator_primary!r}")
    second = None
    if final.second_challenger:
        second = line(final.comparator_primary, final.second_challenger)
        if second is None:
            raise EvalError(
                f"{name}: the second challenger {final.second_challenger!r} was not run"
            )
        second = {"challenger": final.second_challenger, **second}
        second["strict"] = line(final.comparator_strict, final.second_challenger)
    lines = [primary, *([second] if second else [])]
    family = mine[mine["comparator"] == final.comparator_primary]
    best = family.sort_values("p_one_sided").iloc[0]
    kept = mcs[(mcs["task"] == name) & mcs["included"]]["model"].tolist()
    return {
        "task": name,
        "n_dates": len(result.losses),
        "n_effective": len(result.losses) / result.task.horizon,
        "challenger": challenger,
        "challenger_frozen": final.primary_challenger is not None,
        "alpha": alpha,
        "primary": primary,
        "strict": line(final.comparator_strict),
        "second": second,
        "n_challengers": len(challengers),
        "p_for_holm": min(float(x["p_adjusted"]) for x in lines),  # type: ignore[arg-type]
        "h3_supported": any(x["better_and_significant"] for x in lines),
        "family": {
            "n_models": len(family),
            "smallest_p_model": str(best["model"]),
            "smallest_p": float(best["p_one_sided"]),
            "bonferroni_p": float(min(1.0, best["p_one_sided"] * len(family))),
        },
        "mcs": {
            "n_models": int((mcs["task"] == name).sum()),
            "kept": kept,
            "challenger_kept": challenger in kept,
            "comparator_kept": final.comparator_primary in kept,
        },
    }


def evaluate(
    returns: pd.DataFrame,
    sectors: Mapping[str, str],
    exp: ExperimentConfig,
    *,
    market: pd.Series | None = None,
    vix: pd.Series | None = None,
    dcc_avg_corr: pd.DataFrame | None = None,
    final: bool = False,
    targets: Sequence[TargetName] | None = None,
    seeds: Sequence[int] | None = None,
    n_bootstrap: int | None = None,
) -> Evaluation:
    """Run the frozen ladder on the validation window, or on the test window with ``final``."""
    design = exp.forecasting
    if design is None or design.gbm is None or design.final is None:
        raise EvalError("config/experiment.yaml needs forecasting.gbm and forecasting.final")
    plan = design.final
    reps = exp.inference.n_bootstrap if n_bootstrap is None else n_bootstrap
    results = run_ladder(
        returns,
        sectors,
        exp.splits,
        design,
        market=market,
        vix=vix,
        dcc_avg_corr=dcc_avg_corr,
        lam=exp.ewma_lambda,
        targets=plan.targets if targets is None else targets,
        variant=design.variant(plan.variant) if plan.variant else None,
        gbm_seeds=exp.seeds if seeds is None else seeds,
        n_bootstrap=reps,
        alpha=exp.inference.alpha,
        final=final,
    )
    roles = {
        r.task.name: {"primary": plan.comparator_primary, "strict": plan.comparator_strict}
        for r in results
        if r.task.target == design.primary_target
    }
    tests = challenger_tests(results, roles)
    mcs = mcs_table(results, size=design.mcs_size, reps=reps)
    primary_name = f"{design.primary_target}_h{design.primary_horizon_days}"
    primary = next((r for r in results if r.task.name == primary_name), None)
    if primary is None:
        raise EvalError(f"the primary task {primary_name} was not run")
    verdict = h3_verdict(primary, tests, mcs, plan, exp.inference.alpha)
    return Evaluation(results, ladder_table(results), tests, mcs, verdict)


def trial_count(results: Sequence[TaskResult], n_trials: int) -> dict[str, int]:
    """Rung-3 refits and Optuna trials of one run (for the multiplicity record)."""
    refits = sum(len(folds) for r in results for folds in r.fold_params.values())
    return {"rung3_refits": refits, "optuna_trials": refits * n_trials}


def verdict_lines(verdict: Mapping[str, object], final: bool) -> list[str]:
    primary = verdict["primary"]
    family, mcs = verdict["family"], verdict["mcs"]
    assert isinstance(primary, dict) and isinstance(family, dict) and isinstance(mcs, dict)
    window = "TEST WINDOW: THE H3 VERDICT" if final else "validation rehearsal, NOT a verdict"
    out = [
        f"H3 reading ({window}), task {verdict['task']}, {verdict['n_dates']} dates "
        f"(about {verdict['n_effective']:.0f} independent)",
        f"  challenger {verdict['challenger']}"
        + (
            ""
            if verdict["challenger_frozen"]
            else " (best of rungs 2-3 on this window, not frozen)"
        ),
    ]
    for label in ("primary", "strict"):
        row = verdict[label]
        if not isinstance(row, dict):
            continue
        out.append(
            f"  {label:<7} vs {row['comparator']}: loss {row['loss_challenger']:.5f} against "
            f"{row['loss_comparator']:.5f}, diff {row['mean_diff']:+.5f}, DM-HLN "
            f"p = {row['p_one_sided']:.4f} one-sided (Clark-West sensitivity: "
            f"{row['p_clark_west_sensitivity']:.4f})"
        )
    second = verdict.get("second")
    if isinstance(second, dict):
        out.append(
            f"  second challenger {second['challenger']} vs {second['comparator']}: loss "
            f"{second['loss_challenger']:.5f}, diff {second['mean_diff']:+.5f}, DM-HLN "
            f"p = {second['p_one_sided']:.4f} one-sided"
        )
    out.append(
        f"  over {verdict['n_challengers']} challenger(s), Bonferroni: p = "
        f"{verdict['p_for_holm']:.4f} (the H3 p-value)"
    )
    out.append(
        f"  all {family['n_models']} rung 2-3 models vs the primary comparator: smallest p "
        f"{family['smallest_p']:.4f} ({family['smallest_p_model']}), Bonferroni "
        f"{family['bonferroni_p']:.4f}"
    )
    kept = mcs["kept"]
    assert isinstance(kept, list)
    out.append(
        f"  Model Confidence Set keeps {len(kept)} of {mcs['n_models']}; challenger "
        f"{'in' if mcs['challenger_kept'] else 'out'}, comparator "
        f"{'in' if mcs['comparator_kept'] else 'out'}"
    )
    if final:
        word = "SUPPORTED" if verdict["h3_supported"] else "NOT SUPPORTED"
        out.append(f"  H3 {word} at alpha = {verdict['alpha']} by the rule fixed before the run")
    return out


def _git(project: Project, *args: str) -> str:
    done = subprocess.run(
        ["git", *args], cwd=project.paths.root, capture_output=True, text=True, check=False
    )
    return done.stdout.strip()


def git_state(project: Project) -> tuple[str, bool]:
    """``(HEAD commit, whether the working tree is dirty)``."""
    return _git(project, "rev-parse", "HEAD"), bool(_git(project, "status", "--porcelain"))


def open_final(
    project: Project, marker: Path, frozen: bool, not_frozen: str, confirm_unseal: bool
) -> None:
    """The guards of a single test-window run; writes ``marker`` before any data are read."""
    commit, dirty = git_state(project)
    if not frozen:
        raise EvalError(not_frozen)
    if marker.exists():
        raise EvalError(f"{marker} exists: the test window has already been opened")
    if dirty:
        raise EvalError("the working tree is not clean: commit the frozen state first")
    if not confirm_unseal:
        raise EvalError("this reads the sealed test window, once: pass --confirm-unseal")
    marker.parent.mkdir(parents=True, exist_ok=True)
    opened = {
        "opened_on": date.today().isoformat(),
        "commit": commit,
        "config_hash": config_hash(project.experiment)[:12],
        "status": "started",
    }
    marker.write_text(json.dumps(opened, indent=2) + "\n", encoding="utf-8")


def close_final(marker: Path) -> None:
    done = {**json.loads(marker.read_text(encoding="utf-8")), "status": "completed"}
    marker.write_text(json.dumps(done, indent=2) + "\n", encoding="utf-8")


def command_eval(
    project: Project,
    *,
    final: bool = False,
    confirm_unseal: bool = False,
    recompute_dcc: bool = False,
) -> list[str]:
    """Run ``etfcorr eval`` and write its tables; returns the lines to print."""
    exp = project.experiment
    design = exp.forecasting
    if design is None or design.gbm is None or design.final is None:
        raise EvalError("config/experiment.yaml needs forecasting.gbm and forecasting.final")
    paths = project.paths
    mode = "final" if final else "validation"
    commit, dirty = git_state(project)
    marker = paths.tables / MARKER
    if final:
        open_final(
            project,
            marker,
            design.final.frozen,
            "forecasting.final is not frozen (primary_challenger, frozen_on)",
            confirm_unseal,
        )

    returns, sectors, market, vix = load_sealed_panel(project, final=final)
    cache = paths.artifacts.joinpath(*VALIDATION_DCC)
    out_dir = paths.artifacts / "eval_7" / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    if not final and cache.is_file() and not recompute_dcc:
        dcc = pd.read_parquet(cache)
        if not dcc.index.equals(returns.index):
            raise EvalError(f"{cache} is on another calendar; rerun with --recompute-dcc")
    else:
        dcc = dcc_path(returns, sectors, max(design.horizons_days[design.primary_target]))
        dcc.to_parquet(out_dir / "dcc_avg_corr.parquet")

    run = evaluate(returns, sectors, exp, market=market, vix=vix, dcc_avg_corr=dcc, final=final)

    head = (
        "TEST WINDOW, SINGLE FINAL RUN." if final else "VALIDATION WINDOW ONLY, TEST WINDOW UNREAD."
    )
    lines = [
        head,
        f"core panel N={returns.shape[1]}, last date read {returns.index[-1].date()}, config "
        f"{config_hash(exp)[:12]}, commit {commit[:7]}{' (dirty)' if dirty else ''}; variant "
        f"{design.final.variant}; LightGBM {design.gbm.n_trials} trials per refit, seeds "
        f"{list(exp.seeds)}",
        "",
        *(
            line.replace("validation dates", "test dates") if final else line
            for line in table_lines(run.results)
        ),
        "Rung 2-3 models against the frozen comparators (mean_diff < 0 favours the model)",
        run.tests.drop(columns=["kernel", "n_effective"]).round(4).to_string(index=False),
        "",
        f"Model Confidence Set at level {design.mcs_size}",
    ]
    for name, block in run.mcs.groupby("task", sort=False):
        kept = block[block["included"]]["model"].tolist()
        lines.append(f"  {name}: {len(kept)} of {len(block)} kept: {kept}")
    lines += ["", *verdict_lines(run.verdict, final)]

    primary = next(r for r in run.results if r.task.name == run.verdict["task"])
    summary = {
        "step": "7.3" if not final else "7.5",
        "mode": mode,
        "status": "single final run on the test window" if final else "validation rehearsal",
        "config_hash": config_hash(exp)[:12],
        "commit": commit,
        "dirty": dirty,
        "window": [str(primary.losses.index[0].date()), str(primary.losses.index[-1].date())],
        "last_date_read": str(returns.index[-1].date()),
        "plan": design.final.model_dump(mode="json"),
        "best_rung_2_3_on_this_window": best_challenger(primary),
        "models": {r.task.name: [s.name for s in r.task.models] for r in run.results},
        "gbm": {
            "n_trials_per_refit": design.gbm.n_trials,
            "seeds": list(exp.seeds),
            **trial_count(run.results, design.gbm.n_trials),
            "graph_edges_by_fold": {
                name: [p.get("n_edges") for p in folds]
                for name, folds in primary.fold_params.items()
                if name != GBM_NAME
            },
        },
        "h3": run.verdict,
        "bootstrap": {"n": exp.inference.n_bootstrap, "seed": 0},
    }
    stem = f"eval_7_{mode}"
    paths.tables.mkdir(parents=True, exist_ok=True)
    run.table.to_csv(paths.tables / f"{stem}_table.csv", index=False)
    run.tests.to_csv(paths.tables / f"{stem}_tests.csv", index=False)
    run.mcs.to_csv(paths.tables / f"{stem}_mcs.csv", index=False)
    (paths.tables / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for r in run.results:
        frame = r.predictions.assign(target=r.task.y)
        frame = frame.dropna(how="all", subset=list(r.predictions.columns))
        frame.to_parquet(out_dir / f"predictions_{r.task.name}.parquet")
    if final:
        close_final(marker)
    lines.append(f"\nwrote {stem}_*.csv/json under {paths.tables} and forecasts under {out_dir}")
    return lines
