"""Step 6.4: test tables for a ladder run (Diebold-Mariano rows and Model Confidence Sets).

Thin glue between ``evaluation.ladder`` (per-date losses of every model of a task) and
``evaluation.dm_mcs`` (the tests). Nothing is estimated here.

* :func:`dm_table`: for each task, the HLN-corrected DM test of one model against each named
  comparator, one-sided ("the model has the lower expected loss") and two-sided. For the rank
  target, where higher is better, the series are negated first, so "mean_diff < 0" always means
  the model is better.
* :func:`mcs_table`: the Model Confidence Set of every task at one level, with the block length
  ``2 h`` used for the bootstrap intervals of the ladder.

Reading: these are tests on the **validation** window. The comparators were chosen on the same
window, several tasks and models are tested, and nothing is corrected for multiplicity; the
size of the DM test at this project's sample sizes is measured in
``scripts/dm_calibration_6_2.py`` and is above nominal. A p-value here is a description of the
validation sample, not a verdict on H3, which is read once on the test window in Phase 7.
"""

from collections.abc import Mapping, Sequence

import pandas as pd

from etfcorr.evaluation.dm_mcs import diebold_mariano, model_confidence_set
from etfcorr.evaluation.ladder import LadderError, TaskResult
from etfcorr.evaluation.metrics import block_length_for

DM_COLUMNS: tuple[str, ...] = (
    "task",
    "model",
    "comparator",
    "role",
    "n_dates",
    "n_effective",
    "mean_diff",
    "dm_statistic",
    "p_one_sided",
    "p_two_sided",
    "kernel",
)


def dm_table(
    results: Sequence[TaskResult], model: str, comparators: Mapping[str, Mapping[str, str]]
) -> pd.DataFrame:
    """DM rows of ``model`` against the comparators of each task.

    ``comparators`` maps a task name to ``{role: comparator model}`` (for example
    ``{"primary": "ewma", "stricter": "dcc"}``); tasks without an entry are skipped.
    """
    rows = []
    for result in results:
        task = result.task
        wanted = comparators.get(task.name, {})
        if not wanted:
            continue
        if model not in result.losses.columns:
            raise LadderError(f"{task.name}: model {model!r} is not in the ladder run")
        sign = -1.0 if task.higher_is_better else 1.0
        for role, other in wanted.items():
            if other not in result.losses.columns:
                raise LadderError(f"{task.name}: comparator {other!r} is not in the ladder run")
            a, b = sign * result.losses[model], sign * result.losses[other]
            less = diebold_mariano(a, b, task.horizon, alternative="less")
            two = diebold_mariano(a, b, task.horizon)
            rows.append(
                {
                    "task": task.name,
                    "model": model,
                    "comparator": other,
                    "role": role,
                    "n_dates": less.n,
                    "n_effective": less.n / task.horizon,
                    "mean_diff": less.mean_diff,
                    "dm_statistic": less.statistic,
                    "p_one_sided": less.p_value,
                    "p_two_sided": two.p_value,
                    "kernel": less.kernel,
                }
            )
    return pd.DataFrame(rows, columns=list(DM_COLUMNS))


def mcs_table(
    results: Sequence[TaskResult], *, size: float = 0.10, reps: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Model Confidence Set of each task: one row per task and model."""
    parts = []
    for result in results:
        task = result.task
        block = int(block_length_for(task.horizon, len(result.losses)))
        out = model_confidence_set(
            result.losses,
            size=size,
            block_size=block,
            reps=reps,
            seed=seed,
            higher_is_better=task.higher_is_better,
        ).reset_index()
        out.insert(0, "task", task.name)
        out["block_size"] = block
        parts.append(out)
    if not parts:
        raise LadderError("no results")
    return pd.concat(parts, ignore_index=True)
