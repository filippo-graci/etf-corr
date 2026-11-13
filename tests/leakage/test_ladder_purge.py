"""Leakage tests for the baseline ladder as it is actually run (step 5.6).

``tests/leakage/test_models_no_lookahead.py`` checks the driver with folds the test builds
itself. Here the folds are the ones ``evaluation.ladder.run_task`` builds from the config, for
every task of the design, so that a ladder that purged with the wrong horizon (or not at all)
is caught: the validation forecasts of every model must not change when the sample is cut just
after a refit date, where the last training labels of a badly purged fold would still be
unknown.
"""

import importlib.util
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from etfcorr.config import Splits
from etfcorr.evaluation import ladder as L
from etfcorr.leakage import LeakageError, assert_no_lookahead
from etfcorr.splits import Fold, walk_forward_from_config

# the small design and the planted panel of the ladder's unit tests (tests/ is not a package)
_PATH = Path(__file__).resolve().parents[1] / "unit" / "test_ladder.py"
_spec = importlib.util.spec_from_file_location("ladder_fixtures", _PATH)
assert _spec is not None and _spec.loader is not None
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
SPLITS: Splits = _fixtures.SPLITS
design = _fixtures.design

RETURNS, SECTORS = _fixtures.switching_panel(seed=2)
CALENDAR = pd.DatetimeIndex(RETURNS.index)
STARTS = [int(f.test[0]) for f in walk_forward_from_config(CALENDAR, SPLITS, 21)]
# a few rows after the second, third and fourth quarterly refit, and the last row
CUTS = (*[s + 4 for s in STARTS[1:]], len(RETURNS) - 1)
TASKS = ("T1_h21", "T1_matrix_h21", "T2_h5", "T3_h5")


def forecasts(task_name: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def run(d: pd.DataFrame) -> pd.DataFrame:
        tasks = L.build_tasks(d, SECTORS, design(), embargo=SPLITS.walk_forward.embargo_days)
        task = next(t for t in tasks if t.name == task_name)
        result = L.run_task(task, SPLITS, pd.DatetimeIndex(d.index), n_bootstrap=5)
        return result.predictions

    return run


def test_the_cuts_sit_just_after_refit_dates() -> None:
    assert len(STARTS) == 4 and len(CUTS) == 4
    assert all(RETURNS.index[c] > pd.Timestamp("2019-03-31") for c in CUTS)


@pytest.mark.parametrize("task_name", TASKS)
def test_validation_forecasts_of_every_model_have_no_lookahead(task_name: str) -> None:
    assert_no_lookahead(forecasts(task_name), RETURNS, CUTS)


@pytest.mark.parametrize("task_name", ["T1_h21", "T2_h5"])
def test_a_ladder_purged_with_the_wrong_horizon_is_caught(
    task_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leaky twin: the ladder asks for folds of a 1-day label whatever the target's horizon."""
    no_embargo = SPLITS.model_copy(
        update={"walk_forward": SPLITS.walk_forward.model_copy(update={"embargo_days": 0})}
    )

    def understated(calendar: pd.DatetimeIndex, splits: Splits, horizon: int) -> list[Fold]:
        return walk_forward_from_config(calendar, no_embargo, 1)

    monkeypatch.setattr(L, "walk_forward_from_config", understated)
    with pytest.raises(LeakageError):
        assert_no_lookahead(forecasts(task_name), RETURNS, CUTS)


def test_run_task_asks_for_folds_at_the_task_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[int] = []
    real = L.walk_forward_from_config

    def spy(calendar: pd.DatetimeIndex, splits: Splits, horizon: int) -> list[Fold]:
        asked.append(horizon)
        return real(calendar, splits, horizon)

    monkeypatch.setattr(L, "walk_forward_from_config", spy)
    wide = design(horizons_days={"T1": [21, 63], "T1_matrix": [21], "T2": [5, 21], "T3": [5]})
    L.run_ladder(RETURNS, SECTORS, SPLITS, wide, n_bootstrap=5)
    assert asked == [21, 63, 21, 5, 21, 5]
