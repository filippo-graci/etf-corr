"""Purge integrity for ``etfcorr.splits`` (step 5.2; project_plan.md section 9, "purge integrity").

A target dated ``p`` uses the rows ``p + 1 .. p + horizon``. The tests here check, by brute force
on the label rows themselves, that no training label shares a row with a test label, that the
purge is exactly tight when the embargo is zero (one more training origin would overlap), that
the fold construction agrees with the independent ``purgedcv.purge``, and that a splitter
without a purge is caught. The sealed-window guard is tested as a procedure: the folds built
from the committed config never reach a test-window date.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import purgedcv
import pytest

from etfcorr import splits as S
from etfcorr.config import load_experiment
from etfcorr.targets import label_end_dates

ROOT = Path(__file__).resolve().parents[2]
SPLITS = load_experiment(ROOT / "config" / "experiment.yaml").splits
CALENDAR = pd.bdate_range("2014-01-03", "2022-12-30", name="date")


def label_rows(origins: np.ndarray, horizon: int) -> set[int]:
    return {int(p) + k for p in origins for k in range(1, horizon + 1)}


@pytest.mark.parametrize("horizon", [1, 5, 21, 63])
@pytest.mark.parametrize("embargo", [0, 5])
def test_no_training_label_shares_a_row_with_a_test_label(horizon: int, embargo: int) -> None:
    folds = S.expanding_walk_forward(
        CALENDAR,
        train_start=pd.Timestamp("2016-01-01"),
        test=SPLITS.val,
        horizon=horizon,
        embargo=embargo,
    )
    for fold in folds:
        assert not label_rows(fold.train, horizon) & label_rows(fold.test, horizon)
        assert fold.test.min() - fold.train.max() == horizon + embargo
        assert fold.train.max() < fold.test.min()


@pytest.mark.parametrize("horizon", [5, 21])
def test_the_purge_is_tight_one_more_origin_would_overlap(horizon: int) -> None:
    fold = S.walk_forward_from_config(CALENDAR, SPLITS.model_copy(), horizon)[0]
    extra = np.arange(fold.train.max() + 1, fold.test.min() - horizon + 2)  # embargo, then one
    assert not label_rows(extra[:-1], horizon) & label_rows(fold.test, horizon)
    assert label_rows(extra[-1:], horizon) & label_rows(fold.test, horizon)


@pytest.mark.parametrize("horizon", [1, 5, 21, 63])
def test_folds_agree_with_purgedcv_purge(horizon: int) -> None:
    """Independent check: the vetted library keeps exactly our training origins (embargo 0)."""
    # label end dates; the last `horizon` origins get dates past the calendar so every row has one
    padded = pd.bdate_range(CALENDAR[0], periods=len(CALENDAR) + horizon)
    prediction = pd.Series(CALENDAR)
    evaluation = pd.Series(padded[horizon:])
    known = label_end_dates(CALENDAR, horizon).dropna()
    assert (evaluation.iloc[: len(known)].to_numpy() == known.to_numpy()).all()

    folds = S.expanding_walk_forward(
        CALENDAR, train_start=CALENDAR[0], test=SPLITS.val, horizon=horizon, embargo=0
    )
    for fold in folds:
        candidates = np.arange(0, fold.test.min())
        kept = purgedcv.purge(candidates, fold.test, prediction, evaluation)
        assert np.array_equal(kept, fold.train)


def test_inner_folds_are_purged_too() -> None:
    for horizon, embargo in [(1, 0), (21, 5), (63, 5)]:
        for fold in S.positional_walk_forward(1200, 3, horizon=horizon, embargo=embargo):
            assert not label_rows(fold.train, horizon) & label_rows(fold.test, horizon)


def test_a_splitter_without_a_purge_is_caught() -> None:
    """Leaky twin: train on everything before the fold, as a plain TimeSeriesSplit would."""
    good = S.walk_forward_from_config(CALENDAR, SPLITS, 21)
    leaky = [S.Fold(train=np.arange(0, f.test.min(), dtype=np.int64), test=f.test) for f in good]
    assert all(label_rows(f.train, 21) & label_rows(f.test, 21) for f in leaky)
    with pytest.raises(S.SplitError):
        S.assert_purged(leaky, horizon=21, embargo=0)
    S.assert_purged(good, horizon=21, embargo=SPLITS.walk_forward.embargo_days)


def test_a_horizon_understated_to_the_splitter_is_caught() -> None:
    """Leaky twin: folds built for horizon 1 while the labels span 21 rows."""
    understated = S.walk_forward_from_config(CALENDAR, SPLITS, 1)
    with pytest.raises(S.SplitError):
        S.assert_purged(understated, horizon=21, embargo=SPLITS.walk_forward.embargo_days)


def test_config_folds_never_reach_the_sealed_test_window() -> None:
    test_start = pd.Timestamp(SPLITS.test.start)
    for horizon in (5, 21, 63):
        for fold in S.walk_forward_from_config(CALENDAR, SPLITS, horizon):
            assert CALENDAR[fold.test].max() < test_start
            # the last label row of a kept target must also be before the test window
            ends = label_end_dates(CALENDAR, horizon).iloc[fold.train]
            assert ends.max() < CALENDAR[fold.test.min()]


def test_unsealed_data_cannot_be_split() -> None:
    full = pd.bdate_range("2014-01-03", "2026-09-18", name="date")
    with pytest.raises(S.SealedTestError):
        S.walk_forward_from_config(full, SPLITS, 21)
    sealed = S.seal_test(pd.Series(0.0, index=full), SPLITS)
    assert len(S.walk_forward_from_config(pd.DatetimeIndex(sealed.index), SPLITS, 21)) == 8
