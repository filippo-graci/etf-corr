"""Tests for ``etfcorr.evaluation.econ`` (step 8.3): the two windows and the guards, synthetic."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etfcorr import config as C
from etfcorr.evaluation import econ as E
from etfcorr.evaluation.final import EvalError
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
PANEL = make_factor_returns(1300, (3, 3), seed=6, start="2016-01-04")
RNG = np.random.default_rng(3)
SPY = pd.Series(
    PANEL.returns.mean(axis=1) + RNG.standard_normal(1300) * 0.002, PANEL.returns.index, name="SPY"
)
OTHER = pd.Series(RNG.standard_normal(1300) * 0.01, PANEL.returns.index, name="TLT")
VIX = pd.Series(15.0 + 10.0 * RNG.random(1300), PANEL.returns.index)


def project(tmp_path: Path, **changes: object) -> C.Project:
    design = C.Economics.model_validate(
        {
            "declared_on": "2026-10-11",
            "estimation": {"D": 120, "W": 60},
            "estimators": ["sample", "ledoit_wolf", "rmt", "ewma"],
            "h5_estimators": ["ledoit_wolf", "rmt"],
            "hedge_instrument": "SPY",
            "hedge_window_days": 60,
            "lasso_instruments": ["SPY", "TLT"],
            "stress_quantile": 0.9,
            "calm_quantile": 0.5,
            "block_days": 5,
            **changes,
        }
    )
    real = C.load_project(ROOT)
    exp = real.experiment.model_copy(update={"splits": SPLITS, "economics": design})
    return C.Project(C.Paths.from_root(tmp_path), exp, real.universe)


def run(tmp_path: Path, final: bool) -> dict[str, object]:
    end = str(SPLITS.test.end if final else SPLITS.val.end)
    daily = PANEL.returns.loc[:end]
    panels = {"D": daily, "W": daily.resample("W-FRI").sum().iloc[:-1]}
    instruments = pd.concat([SPY, OTHER], axis=1).loc[:end]
    return E.evaluate(
        panels,
        SPY.loc[:end],
        instruments,
        VIX.loc[:end],
        project(tmp_path),
        final=final,
        dcc_corr=None,
        n_bootstrap=40,
    )


def test_the_two_windows_are_evaluated_separately(tmp_path: Path) -> None:
    validation, test = run(tmp_path, False), run(tmp_path, True)
    assert validation["window"] == ["2019-01-01", "2019-12-31"]
    assert test["window"] == ["2020-01-01", "2020-09-30"]
    for out in (validation, test):
        vol, hedge, h5, h6 = out["vol"], out["hedge"], out["h5"], out["h6"]
        assert isinstance(vol, pd.DataFrame) and isinstance(hedge, pd.DataFrame)
        assert isinstance(h5, dict) and isinstance(h6, dict)
        assert len(vol) == 17 and h5["n_cells"] == 4  # 2 frequencies x 2 H5 estimators
        assert list(hedge["method"]) == ["rolling", "ewma", "static", "lasso"]
        assert (hedge["he_all"] > 0.3).all()  # the hedge is the panel's own market factor
        assert h6["n_stress"] + h6["n_calm"] < hedge["n_days"].iloc[0]
        assert 0 < h5["p_one_sided"] <= 1 and 0 < h6["p_one_sided"] <= 1
    lines = E.report_lines(test, True)
    assert sum("SUPPORTED" in line for line in lines) == 2
    assert not any("SUPPORTED" in line for line in E.report_lines(validation, False))


def test_final_run_is_refused_until_frozen_and_confirmed(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="not frozen"):
        E.command_econ(project(tmp_path), final=True, confirm_unseal=True)
    frozen = project(tmp_path, frozen_on="2026-10-11")
    with pytest.raises(EvalError, match="--confirm-unseal"):
        E.command_econ(frozen, final=True)
    assert not (tmp_path / "reports").exists()
    with pytest.raises(FileNotFoundError):  # no data in the temporary project
        E.command_econ(frozen, final=True, confirm_unseal=True)
    marker = json.loads((tmp_path / "reports" / "tables" / E.MARKER).read_text())
    assert marker["status"] == "started"
    with pytest.raises(EvalError, match="already been opened"):
        E.command_econ(frozen, final=True, confirm_unseal=True)


def test_cli_refuses_the_final_run_without_confirmation() -> None:
    """Safe against the repository in every state: without --confirm-unseal nothing is read."""
    from typer.testing import CliRunner

    from etfcorr.cli import app

    result = CliRunner().invoke(app, ["econ", "--final", "--root", str(ROOT)])
    assert result.exit_code == 1 and "EVAL REFUSED" in result.output


def test_the_design_values_reach_the_hedge_and_the_regimes(tmp_path: Path) -> None:
    """The hedge window, the stress quantile and the short-history rule come from the design."""
    end = str(SPLITS.val.end)
    daily = PANEL.returns.loc[:end]
    panels = {"D": daily, "W": daily.resample("W-FRI").sum().iloc[:-1]}
    instruments = pd.concat([SPY, OTHER], axis=1).loc[:end]

    def hedge(**changes: object) -> tuple[pd.DataFrame, dict[str, object]]:
        out = E.evaluate(
            panels,
            SPY.loc[:end],
            instruments,
            VIX.loc[:end],
            project(tmp_path, **changes),
            final=False,
            dcc_corr=None,
            n_bootstrap=10,
        )
        table, h6 = out["hedge"], out["h6"]
        assert isinstance(table, pd.DataFrame) and isinstance(h6, dict)
        return table.set_index("method"), h6

    base, h6 = hedge()
    n_days = int(base["n_days"].iloc[0])
    assert h6["n_stress"] == int(np.floor(n_days * 0.1)) or h6["n_stress"] == int(
        np.ceil(n_days * 0.1)
    )
    _, h6_wider = hedge(stress_quantile=0.75)
    assert h6_wider["n_stress"] > 2 * h6["n_stress"]  # type: ignore[operator]
    short, _ = hedge(hedge_window_days=20)
    assert short.loc["rolling", "he_all"] != pytest.approx(base.loc["rolling", "he_all"], abs=1e-6)
    assert short.loc["static", "he_all"] == pytest.approx(base.loc["static", "he_all"])
    long_window = project(tmp_path, estimation={"D": 120, "W": 400})
    kw = {"final": False, "dcc_corr": None, "n_bootstrap": 10}
    out = E.evaluate(panels, SPY.loc[:end], instruments, VIX.loc[:end], long_window, **kw)  # type: ignore[arg-type]
    assert out["frequencies_skipped"] == ["W"]  # a rehearsal skips a window it cannot fill
    full = PANEL.returns.loc[: str(SPLITS.test.end)]
    panels = {"D": full, "W": full.resample("W-FRI").sum().iloc[:-1]}
    instruments = pd.concat([SPY, OTHER], axis=1).loc[full.index]
    with pytest.raises(ValueError, match="fewer than 400"):  # the final run must have it
        E.evaluate(
            panels,
            SPY.loc[full.index],
            instruments,
            VIX.loc[full.index],
            long_window,
            final=True,
            dcc_corr=None,
            n_bootstrap=10,
        )
