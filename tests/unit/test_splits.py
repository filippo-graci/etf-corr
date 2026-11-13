"""Unit tests for ``etfcorr.splits`` (step 5.2): sealing, calendar folds, inner folds.

Label-overlap (purge integrity) and the ``purgedcv`` cross-check are in
``tests/leakage/test_splits_purge.py``.
"""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from etfcorr import splits as S
from etfcorr.config import DateWindow, Splits

CALENDAR = pd.bdate_range("2014-01-03", "2022-12-30", name="date")


def make_splits(**changes: object) -> Splits:
    data: dict[str, object] = {
        "train": ["2016-01-01", "2020-12-31"],
        "val": ["2021-01-01", "2022-12-31"],
        "test": ["2023-01-01", "2026-09-18"],
        "walk_forward": {"initial_years": 5, "refit_every": "QE", "embargo_days": 5},
    }
    data.update(changes)
    return Splits.model_validate(data)


SPLITS = make_splits()

# ---- sealing ---------------------------------------------------------------------------


def test_seal_test_keeps_rows_up_to_the_end_of_validation() -> None:
    index = pd.bdate_range("2022-12-20", "2023-01-10", name="date")
    frame = pd.DataFrame({"x": np.arange(len(index))}, index=index)
    sealed = S.seal_test(frame, SPLITS)
    assert sealed.index.max() == pd.Timestamp("2022-12-30")
    assert len(sealed) == int((index <= "2022-12-31").sum())
    assert S.seal_test(frame["x"], SPLITS).index.equals(sealed.index)
    S.assert_sealed(sealed.index, SPLITS)


def test_seal_test_handles_a_date_unit_index() -> None:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2022-12-30", "2023-01-03"]), ["a", "b"]], names=["date", "unit"]
    )
    frame = pd.DataFrame({"x": range(4)}, index=index)
    sealed = S.seal_test(frame, SPLITS)
    assert sealed["x"].tolist() == [0, 1]
    with pytest.raises(S.SealedTestError):
        S.assert_sealed(frame.index, SPLITS)


def test_assert_sealed_flags_the_first_test_day() -> None:
    with pytest.raises(S.SealedTestError, match="2023-01-02"):
        S.assert_sealed(pd.DatetimeIndex(["2022-12-30", "2023-01-02"]), SPLITS)
    S.assert_sealed(pd.DatetimeIndex(["2022-12-31"]), SPLITS)  # the gap day is not test


def test_window_positions_are_closed_on_both_ends() -> None:
    index = pd.DatetimeIndex(["2020-12-30", "2020-12-31", "2021-01-04", "2022-12-30"])
    window = DateWindow.model_validate(["2020-12-31", "2021-01-04"])
    assert S.window_positions(index, window).tolist() == [1, 2]


# ---- calendar walk-forward -------------------------------------------------------------


def folds_for(horizon: int = 21, embargo: int = 5, refit: str = "QE") -> list[S.Fold]:
    return S.expanding_walk_forward(
        CALENDAR,
        train_start=pd.Timestamp("2016-01-01"),
        test=SPLITS.val,
        refit_every=refit,
        horizon=horizon,
        embargo=embargo,
    )


def test_quarterly_folds_tile_the_test_window_once() -> None:
    folds = folds_for()
    assert len(folds) == 8
    tested = np.concatenate([f.test for f in folds])
    assert tested.tolist() == S.window_positions(CALENDAR, SPLITS.val).tolist()
    for fold in folds:
        quarters = CALENDAR[fold.test].to_period("Q")
        assert quarters.nunique() == 1


def test_training_windows_expand_from_the_train_start_and_stop_at_the_purge() -> None:
    folds = folds_for(horizon=21, embargo=5)
    first_train = int(np.searchsorted(CALENDAR, pd.Timestamp("2016-01-01")))
    for fold in folds:
        assert fold.train[0] == first_train
        assert fold.train[-1] == fold.test[0] - 26
        assert np.array_equal(fold.train, np.arange(fold.train[0], fold.train[-1] + 1))
    assert all(len(a.train) < len(b.train) for a, b in pairwise(folds))


@pytest.mark.parametrize(("refit", "n"), [("ME", 24), ("QE", 8), ("YE", 2)])
def test_refit_frequency_sets_the_number_of_folds(refit: str, n: int) -> None:
    assert len(folds_for(refit=refit)) == n


def test_expanding_walk_forward_rejects_bad_requests() -> None:
    kw = {"train_start": pd.Timestamp("2016-01-01"), "test": SPLITS.val, "horizon": 21}
    with pytest.raises(S.SplitError, match="sorted"):
        S.expanding_walk_forward(CALENDAR[::-1], **kw)  # type: ignore[arg-type]
    with pytest.raises(S.SplitError, match="no origin"):
        S.expanding_walk_forward(CALENDAR[:500], **kw)  # type: ignore[arg-type]
    with pytest.raises(S.SplitError, match="refit_every"):
        S.expanding_walk_forward(CALENDAR, refit_every="W", **kw)  # type: ignore[arg-type]
    with pytest.raises(S.SplitError, match="horizon"):
        S.expanding_walk_forward(CALENDAR, **{**kw, "horizon": 0})  # type: ignore[arg-type]
    with pytest.raises(S.SplitError, match="embargo"):
        S.expanding_walk_forward(CALENDAR, embargo=-1, **kw)  # type: ignore[arg-type]
    late = {**kw, "train_start": pd.Timestamp("2020-12-15")}
    with pytest.raises(S.SplitError, match="no training origin"):
        S.expanding_walk_forward(CALENDAR, **late)  # type: ignore[arg-type]


def test_walk_forward_from_config_starts_at_the_validation_window() -> None:
    folds = S.walk_forward_from_config(CALENDAR, SPLITS, 21)
    assert len(folds) == 8
    assert CALENDAR[folds[0].test[0]] == pd.Timestamp("2021-01-01")
    assert CALENDAR[folds[-1].test[-1]] == pd.Timestamp("2022-12-30")
    assert CALENDAR[folds[0].train[0]] == pd.Timestamp("2016-01-01")
    assert folds[0].train[-1] == folds[0].test[0] - 21 - 5  # embargo_days from the config


def test_walk_forward_from_config_refuses_unsealed_data_and_a_short_initial_window() -> None:
    unsealed = pd.bdate_range("2014-01-03", "2023-01-05")
    with pytest.raises(S.SealedTestError):
        S.walk_forward_from_config(unsealed, SPLITS, 21)
    short = make_splits(walk_forward={"initial_years": 6, "refit_every": "QE", "embargo_days": 5})
    with pytest.raises(S.SplitError, match="initial"):
        S.walk_forward_from_config(CALENDAR, short, 21)


# ---- positional (inner) folds ----------------------------------------------------------


def test_positional_folds_tile_the_second_half_and_are_purged() -> None:
    folds = S.positional_walk_forward(103, 3, horizon=5, embargo=2)
    assert [f.test[0] for f in folds] == [51, 68, 85]
    assert np.concatenate([f.test for f in folds]).tolist() == list(range(51, 103))
    assert [int(f.train[-1]) for f in folds] == [44, 61, 78]
    assert all(f.train[0] == 0 for f in folds)


def test_positional_folds_reject_samples_that_are_too_short() -> None:
    with pytest.raises(S.SplitError):
        S.positional_walk_forward(4, 3, horizon=1)
    with pytest.raises(S.SplitError):
        S.positional_walk_forward(40, 3, horizon=21)
    with pytest.raises(S.SplitError):
        S.positional_walk_forward(100, 0, horizon=1)
    with pytest.raises(S.SplitError):
        S.positional_walk_forward(100, 3, horizon=1, first_test_fraction=1.0)


# ---- the purge checker -----------------------------------------------------------------


def test_assert_purged_accepts_the_boundary_and_rejects_one_row_less() -> None:
    ok = S.Fold(train=np.arange(0, 75, dtype=np.int64), test=np.arange(100, 120, dtype=np.int64))
    S.assert_purged([ok], horizon=21, embargo=5)  # gap 100 - 74 = 26
    bad = S.Fold(train=np.arange(0, 76, dtype=np.int64), test=ok.test)
    with pytest.raises(S.SplitError, match="need >= 26"):
        S.assert_purged([bad], horizon=21, embargo=5)
    empty = S.Fold(train=np.array([], dtype=np.int64), test=ok.test)
    with pytest.raises(S.SplitError, match="empty"):
        S.assert_purged([empty], horizon=1, embargo=0)


# ---- excluding an episode from training (variant R1, step 6.0) -------------------------


def test_drop_training_window_blanks_origins_whose_label_touches_the_window() -> None:
    index = pd.bdate_range("2020-01-01", periods=120, name="date")
    y = pd.Series(np.arange(120.0), index=index)
    window = DateWindow.model_validate([str(index[50].date()), str(index[60].date())])
    out = S.drop_training_window(y, window, horizon=5)
    blank = out.isna().to_numpy()
    # origin 44 has label rows 45..49 (kept); origin 45 reaches row 50 (blanked); 60 is the last
    assert np.flatnonzero(blank).tolist() == list(range(45, 61))
    assert y.notna().all()  # the input is not modified
    assert out[~blank].equals(y[~blank])


def test_drop_training_window_on_a_long_table_and_outside_the_sample() -> None:
    dates = pd.bdate_range("2020-01-01", periods=30, name="date")
    index = pd.MultiIndex.from_product([dates, ["a", "b"]], names=["date", "unit"])
    y = pd.Series(1.0, index=index)
    window = DateWindow.model_validate([str(dates[10].date()), str(dates[12].date())])
    out = S.drop_training_window(y, window, horizon=2)
    blanked = out[out.isna()].index.get_level_values("date").unique()
    assert blanked.tolist() == dates[8:13].tolist()
    far = DateWindow.model_validate(["2030-01-01", "2030-02-01"])
    assert S.drop_training_window(y, far, horizon=2).equals(y)
