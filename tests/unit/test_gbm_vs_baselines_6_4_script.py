"""Tests for the orchestration in ``scripts/gbm_vs_baselines_6_4.py`` (step 6.4), synthetic only.

The script is loaded by path. Checked here: how the adopted design variant is read from the
step 6.0 record, how the frozen comparators are derived from a ladder run, the SHAP helper and
the Markdown table. The ladder, the tests and LightGBM themselves have their own test files.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from etfcorr.config import Paths, Project, load_project
from etfcorr.evaluation import ladder as L

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load("gbm_vs_baselines_6_4")
REAL = load_project(ROOT)
_ladder_tests = importlib.util.spec_from_file_location(
    "ladder_fixtures_6_4", ROOT / "tests" / "unit" / "test_ladder.py"
)
assert _ladder_tests is not None and _ladder_tests.loader is not None
F = importlib.util.module_from_spec(_ladder_tests)
_ladder_tests.loader.exec_module(F)


def project_at(root: Path) -> Project:
    return Project(paths=Paths.from_root(root), experiment=REAL.experiment, universe=REAL.universe)


def write_record(root: Path, adopted: dict[str, bool]) -> None:
    tables = root / "reports" / "tables"
    tables.mkdir(parents=True)
    payload = {"adoption": [{"variant": k, "adopted": v} for k, v in adopted.items()]}
    (tables / "variants_6_0.json").write_text(json.dumps(payload), encoding="utf-8")


def test_no_record_means_the_design_as_tagged(tmp_path: Path) -> None:
    assert S.adopted_variant(project_at(tmp_path)) is None


def test_the_adopted_variant_is_read_from_the_step_6_0_record(tmp_path: Path) -> None:
    write_record(tmp_path, {"V1": True, "V2": False})
    variant = S.adopted_variant(project_at(tmp_path))
    assert variant is not None and variant.id == "V1" and variant.kind == "log_bias_correction"


def test_nothing_adopted_and_two_adopted(tmp_path: Path) -> None:
    write_record(tmp_path / "a", {"V1": False, "V2": False})
    assert S.adopted_variant(project_at(tmp_path / "a")) is None
    write_record(tmp_path / "b", {"V1": True, "V2": True})
    with pytest.raises(SystemExit, match="more than one"):
        S.adopted_variant(project_at(tmp_path / "b"))


@pytest.fixture(scope="module")
def results() -> list[L.TaskResult]:
    returns, sectors = F.switching_panel()
    return L.run_ladder(
        returns,
        sectors,
        F.SPLITS,
        F.design(gbm=F.GBM_BLOCK),
        targets=["T1", "T3"],
        gbm_seeds=(0,),
        n_bootstrap=20,
    )


def test_comparators_name_the_best_h3_baseline_for_t1_and_the_best_simple_one(
    results: list[L.TaskResult],
) -> None:
    roles = S.comparators(results)
    assert set(roles) == {"T1_h21", "T3_h5"}
    t1 = results[0]
    named = t1.table[t1.table["model"].isin(S.H3_NAMED)].sort_values("mean")["model"].iloc[0]
    assert roles["T1_h21"] == {"primary_named_in_h3": named, "best_simple": t1.best_simple}
    assert roles["T3_h5"] == {"best_simple": results[1].best_simple}  # H3 names no T3 baseline
    assert L.GBM_NAME not in roles["T1_h21"].values()


def test_shap_helper_covers_the_model_features(results: list[L.TaskResult]) -> None:
    returns, _ = F.switching_panel()
    project = Project(
        paths=REAL.paths,
        experiment=REAL.experiment.model_copy(update={"splits": F.SPLITS, "seeds": (0,)}),
        universe=REAL.universe,
    )
    shap = S.shap_first_refit(
        project, F.design(gbm=F.GBM_BLOCK), results[0], pd.DatetimeIndex(returns.index)
    )
    assert set(shap) == set(results[0].task.gbm_columns)
    values = list(shap.values())
    assert values == sorted(values, reverse=True) and values[0] > 0


def test_markdown_table_has_a_row_per_model_including_rung_three(
    results: list[L.TaskResult],
) -> None:
    table = L.ladder_table(results)
    lines = S.markdown_table(table).strip().splitlines()
    assert len(lines) == 2 + len(table)
    assert sum("| lightgbm | 3 |" in line for line in lines) == 2
