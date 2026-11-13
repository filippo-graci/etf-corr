"""Tests for ``scripts/variants_6_0.py`` (step 6.0): the adoption rules and the report table.

The script is loaded by path. The rules are the ones written in ``config/experiment.yaml`` for V1
and V2; they are checked here on hand-made comparison tables, including the boundary case.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from etfcorr.evaluation.ladder import VARIANT_COLUMNS

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "variants_6_0.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("variants_6_0", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_script()


def row(variant: str, task: str, model: str, diff: float, upper: float) -> dict[str, object]:
    return {
        "variant": variant,
        "task": task,
        "model": model,
        "rung": 1,
        "n_dates": 480,
        "base_mean": 0.3,
        "variant_mean": 0.3 + diff,
        "diff": diff,
        "diff_ci_lower": diff - 0.05,
        "diff_ci_upper": upper,
    }


def table(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(VARIANT_COLUMNS))


def test_v1_needs_har_to_improve_at_both_horizons() -> None:
    both = table(row("V1", "T2_h5", "har", -0.09, -0.05), row("V1", "T2_h21", "har", -0.04, -0.01))
    assert S.adoption(both, "V1")["adopted"] is True
    one = table(row("V1", "T2_h5", "har", -0.09, -0.05), row("V1", "T2_h21", "har", -0.04, 0.01))
    decision = S.adoption(one, "V1")
    assert decision["adopted"] is False and decision["checks"]["T2_h21"]["improves"] is False
    other_model = table(
        row("V1", "T2_h5", "ridge", -0.09, -0.05), row("V1", "T2_h21", "ridge", -0.04, -0.01)
    )
    assert S.adoption(other_model, "V1")["adopted"] is False  # the rule names HAR


def test_v2_reads_ridge_on_the_primary_task_only() -> None:
    elsewhere = table(row("V2", "T3_h21", "ridge", -0.01, -0.001))
    decision = S.adoption(elsewhere, "V2")
    assert decision["adopted"] is False and decision["checks"]["T1_h21"]["note"] == "no change"
    assert S.adoption(table(row("V2", "T1_h21", "ridge", -0.002, -0.0001)), "V2")["adopted"]


@pytest.mark.parametrize("upper", [0.0, 1e-9])
def test_an_interval_touching_zero_is_not_an_improvement(upper: float) -> None:
    assert S.adoption(table(row("V2", "T1_h21", "ridge", -0.002, upper)), "V2")["adopted"] is False


def test_rules_name_declared_variants_and_markdown_has_one_row_each() -> None:
    assert set(S.RULES) == {"V1", "V2"}
    text = S.markdown_table(table(row("V1", "T2_h5", "har", -0.09, -0.05)))
    assert len(text.strip().splitlines()) == 3 and "-0.09000 [-0.14000, -0.05000]" in text
