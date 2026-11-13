"""Leakage and purity tests for ``etfcorr.correlation.stability`` (step 3.5).

The table is not a date-indexed series, so the shared row-aligned detectors in ``etfcorr.leakage``
do not apply directly. The risk that matters here is the **sealed window**: the table must not read
any row dated after ``end``. Three checks express that (truncation, noise after the cut, shuffle
after the cut), and each is shown to FAIL on a deliberately leaky twin that ignores ``end``, so the
detector is itself tested. A last check confirms the guard is not vacuous: moving ``end`` later does
change the table.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import stability as S
from etfcorr.synthetic import make_factor_returns

ESTIMATORS: tuple[S.EstimatorName, ...] = ("pearson", "ledoit_wolf", "ewma", "detoned")
N_BOOT = 20

Table = Callable[[dict[str, pd.DataFrame], str], pd.DataFrame]


def _frames() -> dict[str, pd.DataFrame]:
    daily = make_factor_returns(900, (3, 3, 3), seed=11).returns
    weekly = daily.groupby(np.arange(len(daily)) // 5).sum()
    weekly.index = daily.index[4::5][: len(weekly)]
    return {"D": daily, "W": weekly}


def good(frames: dict[str, pd.DataFrame], end: str) -> pd.DataFrame:
    return S.estimator_stability_table(frames, ESTIMATORS, end=end, n_bootstrap=N_BOOT, seed=2)


def leaky_ignores_end(frames: dict[str, pd.DataFrame], end: str) -> pd.DataFrame:
    """Deliberately leaky twin: the window is applied to the labels but not to the data."""
    table = S.estimator_stability_table(frames, ESTIMATORS, n_bootstrap=N_BOOT, seed=2)
    table["window_end"] = end
    return table


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)
    except AssertionError:
        return False
    return True


def _end_date(frames: dict[str, pd.DataFrame], position: int) -> str:
    return str(frames["D"].index[position].date())


def check_truncation(fn: Table) -> None:
    frames = _frames()
    for position in (400, 600, 750):
        end = _end_date(frames, position)
        full = fn(frames, end)
        cut = {h: f.loc[f.index <= pd.Timestamp(end)] for h, f in frames.items()}
        # the cut data given the same end: nothing after `end` exists, so nothing can differ
        if not _same(full, fn(cut, end)):
            raise AssertionError(f"lookahead detected by truncation at {end}")


def _replace_after(frames: dict[str, pd.DataFrame], end: str, mode: str) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(99)
    out: dict[str, pd.DataFrame] = {}
    for horizon, frame in frames.items():
        changed = frame.copy()
        later = (changed.index > pd.Timestamp(end)).nonzero()[0]
        if later.size:
            block = changed.iloc[later].to_numpy().copy()
            if mode == "noise":
                changed.iloc[later] = rng.standard_normal(block.shape) * 5.0
            else:
                changed.iloc[later] = block[rng.permutation(len(block))]
        out[horizon] = changed
    return out


def check_future_noise(fn: Table) -> None:
    frames = _frames()
    for position in (400, 600):
        end = _end_date(frames, position)
        if not _same(fn(frames, end), fn(_replace_after(frames, end, "noise"), end)):
            raise AssertionError(f"lookahead detected by future noise after {end}")


def check_future_shuffle(fn: Table) -> None:
    frames = _frames()
    for position in (400, 600):
        end = _end_date(frames, position)
        if not _same(fn(frames, end), fn(_replace_after(frames, end, "shuffle"), end)):
            raise AssertionError(f"lookahead detected by future shuffle after {end}")


CHECKS = [check_truncation, check_future_noise, check_future_shuffle]


@pytest.mark.parametrize("check", CHECKS)
def test_the_table_never_reads_rows_after_end(check: Callable[[Table], None]) -> None:
    check(good)


@pytest.mark.parametrize("check", CHECKS)
def test_the_detectors_catch_the_leaky_twin(check: Callable[[Table], None]) -> None:
    with pytest.raises(AssertionError, match="lookahead detected"):
        check(leaky_ignores_end)


def test_moving_end_later_changes_the_table_so_the_guard_is_not_vacuous() -> None:
    frames = _frames()
    early = good(frames, _end_date(frames, 400))
    late = good(frames, _end_date(frames, 750))
    assert not _same(early, late)
    assert int(early.loc[0, "n_obs"]) < int(late.loc[0, "n_obs"])


def test_the_input_frames_are_not_modified() -> None:
    frames = _frames()
    snapshot = {h: f.copy() for h, f in frames.items()}
    good(frames, _end_date(frames, 600))
    for horizon, frame in frames.items():
        pd.testing.assert_frame_equal(frame, snapshot[horizon])


def test_the_column_order_of_the_input_changes_labels_not_the_spread() -> None:
    """Permuting asset order permutes entries but not the sorted pairwise spread summaries."""
    frames = _frames()
    end = _end_date(frames, 600)
    reordered = {h: f[list(f.columns[::-1])] for h, f in frames.items()}
    a = good(frames, end).set_index(["horizon", "estimator"])
    b = good(reordered, end).set_index(["horizon", "estimator"])
    for column in ("n_obs", "cond", "avg_corr", "boot_mean", "min_eigenvalue"):
        np.testing.assert_allclose(
            a[column].to_numpy(dtype=float), b[column].to_numpy(dtype=float), rtol=1e-8, atol=1e-10
        )
