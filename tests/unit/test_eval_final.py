"""Tests for ``etfcorr.evaluation.final`` (step 7.3) and the test-window folds, on synthetic data.

Nothing here reads the project's data. The refusals of the final run are tested with a project
rooted in a temporary folder, never with ``--confirm-unseal`` against the repository.
"""

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from etfcorr import config as C
from etfcorr.evaluation import final as E
from etfcorr.splits import SealedTestError, SplitError, final_walk_forward, walk_forward_from_config
from etfcorr.synthetic import make_factor_returns

ROOT = Path(__file__).resolve().parents[2]
SPLITS = C.Splits.model_validate(
    {
        "train": ["2016-01-01", "2018-12-31"],
        "val": ["2019-01-01", "2019-12-31"],
        "test": ["2020-01-01", "2020-09-30"],
        "walk_forward": {"initial_years": 3, "refit_every": "QE", "embargo_days": 2},
    }
)
PANEL = make_factor_returns(1300, (2, 2, 2), seed=4, start="2016-01-04")  # ends late in 2020
RETURNS = PANEL.returns
SECTORS = dict(zip(RETURNS.columns, PANEL.sectors, strict=True))


def experiment(**final: object) -> C.ExperimentConfig:
    plan = {
        "declared_on": "2026-10-11",
        "targets": ["T1"],
        "comparator_primary": "ewma",
        "comparator_strict": "dcc",
        **final,
    }
    design = C.Forecasting.model_validate(
        {
            "fixed_on": "2026-10-11",
            "primary_target": "T1",
            "primary_horizon_days": 21,
            "horizons_days": {"T1": [21]},
            "feature_windows_days": [21, 63],
            "simple_baselines": ["persistence", "historical_mean", "ewma", "har"],
            "linear_models": ["ridge", "pcr"],
            "losses": {"T1": "mse_fisher_z"},
            "inner_folds": 2,
            "alpha_grid": [0.01, 0.3],
            "l1_ratio": 0.5,
            "pcr_components": [1, 2],
            "gbm": {
                "declared_on": "2026-10-11",
                "targets": ["T1"],
                "n_trials": 2,
                "n_jobs": 1,
                "search_space": {"num_leaves": [2, 4], "n_estimators": [20, 40]},
            },
            "graph": {
                "declared_on": "2026-10-11",
                "targets": ["T1"],
                "kinds": ["knn", "random"],
                "windows_days": [21, 63],
                "similarity_window_days": 63,
                "knn_k": 1,
                "threshold_quantile": 0.8,
            },
            "final": plan,
        }
    )
    base = C.load_experiment(ROOT / "config" / "experiment.yaml")
    return base.model_copy(update={"splits": SPLITS, "forecasting": design, "seeds": (0, 1)})


def run(final: bool, **plan: object) -> E.Evaluation:
    end = SPLITS.test.end if final else SPLITS.val.end
    data = RETURNS.loc[: str(end)]
    return E.evaluate(data, SECTORS, experiment(**plan), final=final, n_bootstrap=40)


@pytest.fixture(scope="module")
def validation() -> E.Evaluation:
    return run(False)


@pytest.fixture(scope="module")
def test_window() -> E.Evaluation:
    return run(True, primary_challenger="ridge", frozen_on="2026-10-11")


# ---- folds -----------------------------------------------------------------------------


def test_final_folds_cover_the_test_window_and_train_through_validation() -> None:
    index = pd.DatetimeIndex(RETURNS.loc[: str(SPLITS.test.end)].index)
    folds = final_walk_forward(index, SPLITS, 21)
    assert len(folds) == 3  # 2020 Q1 to Q3
    assert index[folds[0].test[0]] >= pd.Timestamp("2020-01-01")
    assert index[folds[-1].test[-1]] <= pd.Timestamp("2020-09-30")
    assert index[folds[0].train[-1]] > pd.Timestamp("2019-06-30")  # validation is training data
    assert all(f.test[0] - f.train[-1] >= 21 + 2 for f in folds)
    with pytest.raises(SplitError, match="after the test window"):
        final_walk_forward(pd.DatetimeIndex(RETURNS.index), SPLITS, 21)
    with pytest.raises(SealedTestError):  # the validation folds still refuse the same index
        walk_forward_from_config(index, SPLITS, 21)


# ---- the two windows -------------------------------------------------------------------


def test_validation_and_final_runs_score_different_windows(
    validation: E.Evaluation, test_window: E.Evaluation
) -> None:
    v, t = validation.results[0].losses.index, test_window.results[0].losses.index
    assert v.min() >= pd.Timestamp("2019-01-01") and v.max() <= pd.Timestamp("2019-12-31")
    assert t.min() >= pd.Timestamp("2020-01-01") and t.max() <= pd.Timestamp("2020-09-30")
    models = set(test_window.table["model"])
    assert {"lightgbm", "lightgbm_graph_knn", "lightgbm_graph_random", "ridge", "ewma"} <= models


def test_the_validation_window_refuses_unsealed_data() -> None:
    with pytest.raises(SealedTestError):
        E.evaluate(RETURNS.loc[:"2020-09-30"], SECTORS, experiment(), final=False, n_bootstrap=5)


def test_tests_cover_every_rung_2_3_model_and_clark_west_is_only_a_sensitivity_line(
    test_window: E.Evaluation,
) -> None:
    tests = test_window.tests
    assert set(tests["model"]) == {
        "ridge",
        "pcr",
        "lightgbm",
        "lightgbm_graph_knn",
        "lightgbm_graph_random",
    }
    assert set(tests["comparator"]) == {"ewma"}  # no DCC path was given: the strict line is absent
    cw = tests.set_index("model")["p_clark_west"]
    assert pd.notna(cw["ridge"]) and cw.drop("ridge").isna().all()  # pcr and trees: no line
    assert test_window.verdict["primary"]["p_one_sided"] == pytest.approx(  # type: ignore[index]
        tests.set_index("model").loc["ridge", "p_one_sided"]
    )


def test_verdict_follows_the_frozen_rule(test_window: E.Evaluation) -> None:
    v = test_window.verdict
    primary = v["primary"]
    assert isinstance(primary, dict)
    assert v["challenger"] == "ridge" and v["challenger_frozen"] and v["strict"] is None
    result = test_window.results[0]
    diff = float((result.losses["ridge"] - result.losses["ewma"]).mean())
    assert primary["mean_diff"] == pytest.approx(diff)
    assert v["h3_supported"] == (diff < 0 and primary["p_one_sided"] < 0.05)
    assert v["second"] is None and v["p_for_holm"] == primary["p_one_sided"]
    family = v["family"]
    assert isinstance(family, dict)
    assert family["n_models"] == 5
    assert family["bonferroni_p"] == pytest.approx(min(1.0, 5 * family["smallest_p"]))
    assert any("H3" in line and "SUPPORTED" in line for line in E.verdict_lines(v, True))


def test_an_unfrozen_plan_reports_the_best_challenger_of_the_window(
    validation: E.Evaluation,
) -> None:
    v = validation.verdict
    result = validation.results[0]
    names = ["ridge", "pcr", "lightgbm", "lightgbm_graph_knn", "lightgbm_graph_random"]
    assert v["challenger"] == result.losses[names].mean().idxmin() == E.best_challenger(result)
    assert not v["challenger_frozen"]
    lines = E.verdict_lines(v, False)
    assert "NOT a verdict" in lines[0] and not any("SUPPORTED" in line for line in lines)


def test_a_significant_better_challenger_is_supported_and_a_worse_one_is_not(
    test_window: E.Evaluation,
) -> None:
    """The rule itself, on hand-made tests: both conditions are needed."""
    result = test_window.results[0]
    plan = experiment(primary_challenger="lightgbm", frozen_on="2026-10-11").forecasting.final  # type: ignore[union-attr]
    for diff, p, expected in ((-0.1, 0.01, True), (-0.1, 0.2, False), (0.1, 0.01, False)):
        tests = test_window.tests.copy()
        tests["mean_diff"], tests["p_one_sided"] = diff, p
        tests["p_clark_west"] = 0.0001  # must not rescue a failed DM test
        verdict = E.h3_verdict(result, tests, test_window.mcs, plan, 0.05)
        assert verdict["h3_supported"] is expected


def test_two_challengers_are_read_with_a_bonferroni_factor(test_window: E.Evaluation) -> None:
    result = test_window.results[0]
    plan = experiment(
        primary_challenger="ridge", second_challenger="lightgbm_graph_knn", frozen_on="2026-10-11"
    ).forecasting.final  # type: ignore[union-attr]
    tests = test_window.tests.copy()
    tests["mean_diff"] = -0.1
    tests["p_one_sided"] = [0.04 if m == "lightgbm_graph_knn" else 0.5 for m in tests["model"]]
    v = E.h3_verdict(result, tests, test_window.mcs, plan, 0.05)
    assert v["second"]["challenger"] == "lightgbm_graph_knn"  # type: ignore[index]
    assert v["p_for_holm"] == pytest.approx(0.08) and v["h3_supported"] is False  # 0.04 x 2
    tests["p_one_sided"] = [0.02 if m == "lightgbm_graph_knn" else 0.5 for m in tests["model"]]
    v = E.h3_verdict(result, tests, test_window.mcs, plan, 0.05)
    assert v["p_for_holm"] == pytest.approx(0.04) and v["h3_supported"] is True
    tests["p_one_sided"] = 0.7
    capped = E.h3_verdict(result, tests, test_window.mcs, plan, 0.05)
    assert capped["primary"]["p_adjusted"] == 1.0 == capped["p_for_holm"]  # type: ignore[index]
    assert any("second challenger" in line for line in E.verdict_lines(v, True))
    with pytest.raises(E.EvalError, match="second challenger"):
        E.h3_verdict(
            result,
            tests,
            test_window.mcs,
            plan.model_copy(update={"second_challenger": "no_such_model"}),
            0.05,
        )


def test_trial_count_counts_every_rung_three_refit(test_window: E.Evaluation) -> None:
    assert E.trial_count(test_window.results, 2) == {"rung3_refits": 9, "optuna_trials": 18}


# ---- the guards of the final run -------------------------------------------------------


def project_in(tmp_path: Path, **plan: object) -> C.Project:
    real = C.load_project(ROOT)
    return C.Project(C.Paths.from_root(tmp_path), experiment(**plan), real.universe)


def test_final_run_is_refused_until_frozen_confirmed_clean_and_unopened(tmp_path: Path) -> None:
    with pytest.raises(E.EvalError, match="not frozen"):
        E.command_eval(project_in(tmp_path), final=True, confirm_unseal=True)
    frozen = project_in(tmp_path, primary_challenger="ridge", frozen_on="2026-10-11")
    with pytest.raises(E.EvalError, match="--confirm-unseal"):
        E.command_eval(frozen, final=True)
    assert not (tmp_path / "reports").exists()  # nothing was written by a refusal

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "untracked.txt").write_text("x")
    with pytest.raises(E.EvalError, match="not clean"):
        E.command_eval(frozen, final=True, confirm_unseal=True)


def test_the_marker_is_written_before_the_data_are_read_and_blocks_a_second_run(
    tmp_path: Path,
) -> None:
    frozen = project_in(tmp_path, primary_challenger="ridge", frozen_on="2026-10-11")
    with pytest.raises(FileNotFoundError):  # no data in the temporary project
        E.command_eval(frozen, final=True, confirm_unseal=True)
    marker = json.loads((tmp_path / "reports" / "tables" / E.MARKER).read_text())
    assert marker["status"] == "started"
    with pytest.raises(E.EvalError, match="already been opened"):
        E.command_eval(frozen, final=True, confirm_unseal=True)


def test_cli_refuses_the_final_run_without_confirmation() -> None:
    """Safe against the repository in every state: without --confirm-unseal nothing is read."""
    from typer.testing import CliRunner

    from etfcorr.cli import app

    result = CliRunner().invoke(app, ["eval", "--final", "--root", str(ROOT)])
    assert result.exit_code == 1 and "EVAL REFUSED" in result.output
