"""Step 5.2: chronological splits, the sealed test window, and purged walk-forward folds.

Three jobs (project_plan.md sections 6.5 and 9; ``config/experiment.yaml`` ``splits``):

1. **Sealing the test window.** :func:`seal_test` drops every row dated after the validation
   window ends, and :func:`assert_sealed` raises :class:`SealedTestError` when a frame still
   holds a row dated in or after the test window. Every Phase 5 and Phase 6 entry point calls
   both on the raw returns *before* it computes a feature or a target, so the sealed rows are
   never read into a statistic. The single final run of Phase 7 is the only caller allowed to
   skip the seal.
2. **Expanding walk-forward folds with purge and embargo** (:func:`expanding_walk_forward`,
   :func:`walk_forward_from_config`). Test folds are the calendar periods (``QE`` in the config)
   of the evaluation window; each fold trains on every earlier origin from the train start that
   is far enough from the fold.
3. **Inner folds for tuning** (:func:`positional_walk_forward`): the same purge, by position,
   inside one training window.

**Purge rule.** A target dated ``p`` uses rows ``p + 1 .. p + horizon`` (``etfcorr.targets``).
For a test fold starting at position ``s``, the training origins are those with

.. math:: p \\le s - \\text{horizon} - \\text{embargo}

so the last training label ends at row ``s - embargo`` at the latest and the first test label
starts at row ``s + 1``: no label row is shared, and every train index is at least
``horizon + embargo`` positions before every test index (:func:`assert_purged` checks exactly
this). In walk-forward all training rows precede the test fold, so the embargo acts as an extra
gap on the same side as the purge. The fold construction is a few lines of index arithmetic; a
test cross-checks it against ``purgedcv.purge`` (MIT, already a dependency) rather than
re-deriving that library's logic here. skfolio's ``WalkForward`` / ``CombinatorialPurgedCV`` are
named in the plan but skfolio does not import in the current environment (``cvxpy`` missing), so
they are not used; combinatorial purged CV is not needed for a baseline table.

**Initial window.** ``walk_forward.initial_years`` is counted from ``splits.train.start``. With
the committed config (train 2016-2020, ``initial_years: 5``) the first test fold is 2021Q1, the
start of the validation window: the history from 2014 only warms up the features. This is a
reading of the config fixed in step 5.1, not a derived fact.

Limits: purging removes *label* overlap only. It does not remove dependence that outlasts the
embargo (volatility clustering lasts longer than five days), and it cannot detect a feature that
looks ahead: that is the job of the ``etfcorr.leakage`` detectors.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from etfcorr.config import DateWindow, Splits

IntArray: TypeAlias = npt.NDArray[np.int64]
_F = TypeVar("_F", pd.DataFrame, pd.Series)

_PERIOD_FREQ = {"ME": "M", "QE": "Q", "YE": "Y"}


class SplitError(ValueError):
    """Invalid split request, or a fold that violates the purge rule."""


class SealedTestError(SplitError):
    """Data dated inside the sealed test window reached code that must not read it."""


@dataclass(frozen=True)
class Fold:
    """Positions (into the date index the folds were built on) of one train/test split."""

    train: IntArray
    test: IntArray


def _dates(index: pd.Index) -> pd.DatetimeIndex:
    level = index.get_level_values(0) if isinstance(index, pd.MultiIndex) else index
    return pd.DatetimeIndex(level)


# --------------------------------------------------------------------------------------
# The sealed test window
# --------------------------------------------------------------------------------------


def seal_test(data: _F, splits: Splits) -> _F:
    """Rows of ``data`` dated at or before the end of the validation window."""
    return data.loc[np.asarray(_dates(data.index) <= pd.Timestamp(splits.val.end))]


def assert_sealed(index: pd.Index, splits: Splits) -> None:
    """Raise :class:`SealedTestError` if ``index`` holds a date in or after the test window."""
    dates = _dates(index)
    late = dates[dates >= pd.Timestamp(splits.test.start)]
    if len(late):
        raise SealedTestError(
            f"{len(late)} rows dated from {late.min().date()} are inside the sealed test "
            f"window (starts {splits.test.start}); call seal_test first"
        )


def window_positions(index: pd.DatetimeIndex, window: DateWindow) -> IntArray:
    """Positions of the rows of ``index`` dated inside the closed ``window``."""
    inside = (index >= pd.Timestamp(window.start)) & (index <= pd.Timestamp(window.end))
    return np.flatnonzero(np.asarray(inside)).astype(np.int64)


# --------------------------------------------------------------------------------------
# Purged walk-forward folds
# --------------------------------------------------------------------------------------


def _check_gap(horizon: int, embargo: int) -> None:
    if horizon < 1:
        raise SplitError(f"horizon must be >= 1, got {horizon}")
    if embargo < 0:
        raise SplitError(f"embargo must be >= 0, got {embargo}")


def assert_purged(folds: Sequence[Fold], horizon: int, embargo: int) -> None:
    """Every train position is at least ``horizon + embargo`` before every test position."""
    _check_gap(horizon, embargo)
    for k, fold in enumerate(folds):
        if len(fold.train) == 0 or len(fold.test) == 0:
            raise SplitError(f"fold {k} has an empty train or test set")
        gap = int(fold.test.min()) - int(fold.train.max())
        if gap < horizon + embargo:
            raise SplitError(
                f"fold {k}: last train position {int(fold.train.max())} is {gap} rows before "
                f"the first test position {int(fold.test.min())}; need >= {horizon + embargo}"
            )


def expanding_walk_forward(
    index: pd.DatetimeIndex,
    *,
    train_start: pd.Timestamp,
    test: DateWindow,
    refit_every: str = "QE",
    horizon: int,
    embargo: int = 0,
) -> list[Fold]:
    """Expanding-window folds: one test fold per calendar period of ``test``, purged trains.

    ``index`` is the sorted, unique calendar of forecast origins. Fold ``k`` tests the origins of
    the ``k``-th period (``ME``, ``QE`` or ``YE``) inside ``test`` and trains on the origins from
    ``train_start`` up to ``horizon + embargo`` positions before that period's first origin.
    """
    _check_gap(horizon, embargo)
    if refit_every not in _PERIOD_FREQ:
        raise SplitError(f"refit_every must be one of {sorted(_PERIOD_FREQ)}, got {refit_every!r}")
    if not (index.is_monotonic_increasing and index.is_unique):
        raise SplitError("the origin index must be sorted and unique")
    test_pos = window_positions(index, test)
    if len(test_pos) == 0:
        raise SplitError(f"no origin dated inside the test window {test.start}..{test.end}")
    lo = int(np.searchsorted(index.to_numpy(), np.datetime64(pd.Timestamp(train_start))))
    periods, _ = pd.factorize(index[test_pos].to_period(_PERIOD_FREQ[refit_every]))
    folds: list[Fold] = []
    for period in np.unique(periods):
        fold_test = test_pos[periods == period]
        last_train = int(fold_test[0]) - horizon - embargo
        if last_train < lo:
            raise SplitError(
                f"no training origin is left before {index[int(fold_test[0])].date()} after "
                f"purging {horizon} + {embargo} rows (train starts {index[lo].date()})"
            )
        folds.append(Fold(train=np.arange(lo, last_train + 1, dtype=np.int64), test=fold_test))
    assert_purged(folds, horizon, embargo)
    return folds


def walk_forward_from_config(index: pd.DatetimeIndex, splits: Splits, horizon: int) -> list[Fold]:
    """The validation-window folds of ``config/experiment.yaml`` for one target horizon.

    Train from ``splits.train.start``, one test fold per ``walk_forward.refit_every`` period of
    the validation window, embargo ``walk_forward.embargo_days`` rows. Raises if ``index``
    reaches the test window, or if the first fold starts before ``initial_years`` have passed.
    """
    assert_sealed(index, splits)
    wf = splits.walk_forward
    train_start = pd.Timestamp(splits.train.start)
    earliest = train_start + pd.DateOffset(years=wf.initial_years)
    if pd.Timestamp(splits.val.start) < earliest:
        raise SplitError(
            f"validation starts {splits.val.start}, before the {wf.initial_years} initial "
            f"years counted from {splits.train.start} have passed"
        )
    return expanding_walk_forward(
        index,
        train_start=train_start,
        test=splits.val,
        refit_every=wf.refit_every,
        horizon=horizon,
        embargo=wf.embargo_days,
    )


def final_walk_forward(index: pd.DatetimeIndex, splits: Splits, horizon: int) -> list[Fold]:
    """The **test-window** folds: the same expanding scheme with the test window as target.

    Train from ``splits.train.start`` (so the validation window becomes training data), one fold
    per ``refit_every`` period of the test window, same purge and embargo. Only the single final
    run of Phase 7 calls this. Raises if ``index`` holds a date after the test window ends.
    """
    if len(index) and index.max() > pd.Timestamp(splits.test.end):
        raise SplitError(f"the index reaches {index.max().date()}, after the test window ends")
    wf = splits.walk_forward
    return expanding_walk_forward(
        index,
        train_start=pd.Timestamp(splits.train.start),
        test=splits.test,
        refit_every=wf.refit_every,
        horizon=horizon,
        embargo=wf.embargo_days,
    )


def positional_walk_forward(
    n: int, n_folds: int, *, horizon: int, embargo: int = 0, first_test_fraction: float = 0.5
) -> list[Fold]:
    """``n_folds`` purged expanding folds over positions ``0 .. n - 1`` (inner tuning folds).

    The test folds are equal consecutive blocks tiling the positions from
    ``floor(n * first_test_fraction)`` to the end (the last block takes the remainder).
    """
    _check_gap(horizon, embargo)
    if n_folds < 1 or not 0.0 < first_test_fraction < 1.0:
        raise SplitError("need n_folds >= 1 and 0 < first_test_fraction < 1")
    first = int(n * first_test_fraction)
    size = (n - first) // n_folds
    if size < 1 or first - horizon - embargo < 1:
        raise SplitError(
            f"{n} positions are too few for {n_folds} folds with horizon {horizon} and "
            f"embargo {embargo}"
        )
    folds: list[Fold] = []
    for k in range(n_folds):
        start = first + k * size
        stop = n if k == n_folds - 1 else start + size
        folds.append(
            Fold(
                train=np.arange(0, start - horizon - embargo + 1, dtype=np.int64),
                test=np.arange(start, stop, dtype=np.int64),
            )
        )
    assert_purged(folds, horizon, embargo)
    return folds


def drop_training_window(y: _F, window: DateWindow, horizon: int) -> _F:
    """``y`` with every target whose origin or label rows touch ``window`` set to ``NaN``.

    A target dated at position ``p`` uses rows ``p + 1 .. p + horizon``, so the origins from
    ``horizon`` rows before the first date of the window to its last date are blanked. Rows
    without a target are not trained on by ``models.base.walk_forward_predict`` (and not
    scored), which is how a robustness run excludes an episode from training. The dates are
    those of ``y`` (first index level for a long table).
    """
    _check_gap(horizon, 0)
    dates = _dates(y.index)
    calendar = dates.unique().sort_values()
    inside = window_positions(calendar, window)
    if len(inside) == 0:
        return y.copy()
    first = max(int(inside[0]) - horizon, 0)
    blanked = calendar[first : int(inside[-1]) + 1]
    out = y.copy()
    out[np.asarray(dates.isin(blanked))] = np.nan
    return out
