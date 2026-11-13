"""Tests for the orchestration in ``scripts/baseline_table_5_6.py`` (step 5.6).

The script is loaded by path (``scripts/`` is not a package). Synthetic data only: the loader is
run against a temporary repository whose files reach into the test window, to check that the
sealed rows are dropped before anything is computed; the payload, the Markdown table and the
written files are checked on a small ladder run.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from etfcorr.config import Paths, Project, load_project
from etfcorr.evaluation import ladder as L

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "baseline_table_5_6.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("baseline_table_5_6", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_script()
REAL = load_project(ROOT)


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Project:
    """The committed configs on a temporary data folder whose files run to mid-2023."""
    root = tmp_path_factory.mktemp("repo")
    paths = Paths.from_root(root)
    paths.data_processed.mkdir(parents=True)
    paths.data_interim.mkdir(parents=True)
    index = pd.bdate_range("2014-01-03", "2023-06-30", name="date")
    rng = np.random.default_rng(0)
    core = list(REAL.universe.tickers_in("core"))
    tickers = [*core, "SPY", "^VIX"]
    market = rng.standard_normal((len(index), 1))
    values = 0.01 * (0.6 * market + 0.8 * rng.standard_normal((len(index), len(tickers))))
    pd.DataFrame(values, index=index, columns=tickers).to_parquet(
        paths.data_processed / "returns_D.parquet"
    )
    vix = np.exp(np.log(18.0) + 0.2 * np.sin(np.arange(len(index)) / 50.0))
    prices = pd.DataFrame(
        {("^VIX", "close"): vix, ("^VIX", "adj_close"): vix}, index=index
    ).rename_axis(columns=["ticker", "field"])
    prices.to_parquet(paths.data_interim / "prices_clean.parquet")
    return Project(paths=paths, experiment=REAL.experiment, universe=REAL.universe)


def test_loader_drops_every_row_after_the_validation_window(project: Project) -> None:
    returns, sectors, market, vix = S.load_sealed(project)
    last = pd.Timestamp(project.experiment.splits.val.end)
    for frame in (returns, market, vix):
        assert frame.index.max() <= last and frame.index.max() == pd.Timestamp("2022-12-30")
        assert frame.index.equals(returns.index)
    assert list(returns.columns) == list(project.universe.tickers_in("core"))
    assert set(sectors) == set(returns.columns) and "control" not in set(sectors.values())
    assert market.name == "SPY" and (vix > 0).all()


def test_loader_reports_a_missing_ticker(project: Project, tmp_path: Path) -> None:
    paths = Paths.from_root(tmp_path)
    paths.data_processed.mkdir(parents=True)
    full = pd.read_parquet(project.paths.data_processed / "returns_D.parquet")
    full.drop(columns=["SPY"]).to_parquet(paths.data_processed / "returns_D.parquet")
    broken = Project(paths=paths, experiment=project.experiment, universe=project.universe)
    with pytest.raises(SystemExit, match="SPY"):
        S.load_sealed(broken)


@pytest.fixture(scope="module")
def results(project: Project) -> list[L.TaskResult]:
    """T1 only, on the first six core tickers of the sealed synthetic panel."""
    returns, sectors, market, vix = S.load_sealed(project)
    design = project.experiment.forecasting
    assert design is not None
    return L.run_ladder(
        returns,
        sectors,
        project.experiment.splits,
        design,
        market=market,
        vix=vix,
        targets=["T1"],
        n_bootstrap=30,
    )


def test_summary_payload_records_the_design_and_the_best_baselines(
    project: Project, results: list[L.TaskResult]
) -> None:
    payload = S.summary_payload(project, results, n_boot=30, seed=0, with_dcc=False)
    json.dumps(payload)  # serialisable: no NaN objects, no numpy scalars
    assert payload["step"] == "5.6" and payload["dcc_baseline_included"] is False
    assert payload["primary"] == {"target": "T1", "horizon_days": 21}
    assert payload["splits"]["validation"] == ["2021-01-01", "2022-12-31"]
    assert set(payload["tasks"]) == {"T1_h21", "T1_h63"}
    task = payload["tasks"]["T1_h21"]
    assert task["best_simple_baseline"] in task["models"]
    assert task["best_of_the_four_named_in_h3"] in S.H3_NAMED
    assert task["last_date"] <= "2022-12-30" and task["first_date"] >= "2021-01-01"
    assert set(task["tuned_values_count"]) == {"ridge", "lasso", "elastic_net", "pcr"}
    cells = payload["configurations"]
    assert cells["task_model_cells"] == sum(len(r.table) for r in results)
    design = project.experiment.forecasting
    assert design is not None
    assert cells["hyperparameter_values_per_refit"] == design.n_linear_trials


def test_markdown_table_has_one_row_per_model(results: list[L.TaskResult]) -> None:
    table = L.ladder_table(results)
    text = S.markdown_table(table)
    lines = text.strip().splitlines()
    assert len(lines) == 2 + len(table)
    assert text.count("**best simple**") == len(results)
    assert all(line.count("|") == 8 for line in lines)


def test_write_outputs_stays_under_reports_and_artifacts(
    project: Project, results: list[L.TaskResult], tmp_path: Path
) -> None:
    payload = S.summary_payload(project, results, n_boot=30, seed=0, with_dcc=False)
    written = S.write_outputs(tmp_path, results, payload, None)
    assert all(p.is_file() for p in written)
    allowed = (tmp_path / "reports" / "tables", tmp_path / "artifacts" / "baselines_5_6")
    assert all(p.parent in allowed for p in written)
    table = pd.read_csv(tmp_path / "reports" / "tables" / "baseline_table_5_6.csv")
    assert list(table.columns) == list(L.TABLE_COLUMNS)
    saved = pd.read_parquet(tmp_path / "artifacts" / "baselines_5_6" / "predictions_T1_h21.parquet")
    assert "target" in saved.columns and "persistence" in saved.columns
    assert saved.index.max() <= pd.Timestamp("2022-12-30")
    assert saved.index.min() >= pd.Timestamp("2021-01-01")  # validation forecasts only
    with open(tmp_path / "reports" / "tables" / "baseline_summary_5_6.json") as f:
        assert json.load(f)["tasks"]["T1_h21"]["loss"] == "mse_fisher_z"


def test_quick_mode_writes_nothing(
    project: Project, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(S, "load_project", lambda: project)
    monkeypatch.setattr(S, "run_ladder", lambda *a, **k: [])
    monkeypatch.setattr(S, "table_lines", lambda results: ["(stub)"])
    monkeypatch.setattr(sys, "argv", ["baseline_table_5_6.py", "--quick"])
    assert S.main() == 0
    out = capsys.readouterr().out
    assert "QUICK SMOKE RUN, NOT A RESULT" in out and "TEST WINDOW UNREAD" in out
    assert "2022-12-30 (last date read)" in out and "wrote" not in out
    assert not (project.paths.root / "reports").exists()
    assert not (project.paths.root / "artifacts").exists()
