"""Random-signal null for H3's target on the test window (Phase 8, step 8.4). Run with:
``uv run python scripts/null_benchmark_8_4.py`` (a few minutes).

For the 21-day average cross-sector correlation it fits, 200 times, a tuned ridge on random
persistent features through the same purged test-window folds as the Phase 7 ladder
(``evaluation.null_benchmark``), and ranks the loss of every real model of
``reports/tables/eval_7_final_table.csv`` in that null distribution.

**Status: descriptive, after the H3 verdict.** The forecasting test window was opened once in
Phase 7; this script reads the same returns again to rebuild the target and adds no model. It
does not change the verdict. Output: ``reports/tables/null_benchmark_8_4.csv``.
"""

import pandas as pd

from etfcorr.config import load_project
from etfcorr.evaluation.ladder import build_tasks, load_sealed_panel
from etfcorr.evaluation.null_benchmark import percentile_ranks, random_signal_losses
from etfcorr.splits import final_walk_forward

N_DRAWS = 200
PHI = 0.97
SEED = 0


def main() -> int:
    project = load_project()
    exp = project.experiment
    design = exp.forecasting
    assert design is not None
    task_name = f"{design.primary_target}_h{design.primary_horizon_days}"
    returns, sectors, market, vix = load_sealed_panel(project, final=True)
    embargo = exp.splits.walk_forward.embargo_days
    task = next(
        t
        for t in build_tasks(
            returns, sectors, design, embargo=embargo, market=market, vix=vix, targets=("T1",)
        )
        if t.name == task_name
    )
    calendar = pd.DatetimeIndex(returns.index)
    folds = final_walk_forward(calendar, exp.splits, task.horizon)
    table = pd.read_csv(project.paths.tables / "eval_7_final_table.csv")
    table = table[table["task"] == task_name]
    scored = pd.read_parquet(
        project.paths.artifacts / "eval_7" / "final" / f"predictions_{task_name}.parquet"
    ).dropna()
    dates = scored.index
    if len(dates) != int(table["n_dates"].iloc[0]):
        raise SystemExit("the saved forecasts do not cover the dates of the Phase 7 table")
    null = random_signal_losses(
        task.y,
        folds,
        dates,
        n_features=len(task.gbm_columns),
        n_draws=N_DRAWS,
        horizon=task.horizon,
        embargo=embargo,
        alpha_grid=design.alpha_grid,
        inner_folds=design.inner_folds,
        phi=PHI,
        seed=SEED,
    )
    ranks = percentile_ranks(null, dict(zip(table["model"], table["mean"], strict=True)))
    print(
        f"{task_name}, {len(dates)} test dates, {N_DRAWS} random-signal ridge models with "
        f"{len(task.gbm_columns)} AR({PHI}) features"
    )
    q = null.quantile([0.05, 0.5, 0.95])
    print(f"null loss: median {q[0.5]:.5f}, 5% {q[0.05]:.5f}, 95% {q[0.95]:.5f}")
    print(ranks.round(5).to_string(index=False))
    ranks.assign(
        null_median=q[0.5], null_q05=q[0.05], null_q95=q[0.95], n_draws=N_DRAWS, phi=PHI
    ).to_csv(project.paths.tables / "null_benchmark_8_4.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
