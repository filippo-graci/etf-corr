"""Leakage and purity tests for etfcorr.correlation.networks (step 3.3).

``rolling_mst_survival`` takes a date-indexed frame, so it could use the shared detectors in
``etfcorr.leakage``; those are not imported here because their signatures were not visible
when this file was written (see docs/checks/3.3.md). The three checks below are the same
three ideas (truncation, noise after the cut, shuffle after the cut), implemented locally,
and each is shown to FAIL on a deliberately leaky twin so the detector is itself tested.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import networks as nw

WINDOW = 60
STEP = 10


def make_returns(t: int = 400, n: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = [f"E{k}" for k in range(n)]
    factor = rng.standard_normal((t, 1))
    data = rng.standard_normal((t, n)) + 0.8 * factor
    return pd.DataFrame(data, index=pd.bdate_range("2018-01-01", periods=t), columns=labels)


def good(returns: pd.DataFrame) -> pd.Series:
    return nw.rolling_mst_survival(returns, WINDOW, STEP)


def leaky_centred(returns: pd.DataFrame) -> pd.Series:
    """Deliberately leaky twin: each window is centred on t, so it uses the future half."""
    half = WINDOW // 2
    positions = list(range(WINDOW - 1, len(returns), STEP))
    previous: nw.MSTResult | None = None
    values: list[float] = []
    for pos in positions:
        chunk = returns.iloc[pos - half : min(pos + half, len(returns))]
        current = nw.build_mst(chunk.corr())
        if previous is None:
            values.append(float("nan"))
        else:
            values.append(nw.edge_survival_ratio(current, previous))
        previous = current
    return pd.Series(values, index=returns.index[positions], dtype=float)


def _same(a: pd.Series, b: pd.Series) -> bool:
    return a.index.equals(b.index) and bool(np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True))


def check_truncation(fn: Callable[[pd.DataFrame], pd.Series], data: pd.DataFrame) -> None:
    full = fn(data)
    for cut in (120, 200, 300, 399):
        head = fn(data.iloc[:cut])
        if not _same(head, full.loc[head.index]):
            raise AssertionError(f"lookahead detected by truncation at row {cut}")


def check_future_noise(fn: Callable[[pd.DataFrame], pd.Series], data: pd.DataFrame) -> None:
    full = fn(data)
    rng = np.random.default_rng(99)
    for cut in (150, 250):
        changed = data.copy()
        changed.iloc[cut:] = rng.standard_normal(changed.iloc[cut:].shape) * 5.0
        out = fn(changed)
        before = full.index[full.index <= data.index[cut - 1]]
        if not _same(out.loc[before], full.loc[before]):
            raise AssertionError(f"lookahead detected by future noise after row {cut}")


def check_future_shuffle(fn: Callable[[pd.DataFrame], pd.Series], data: pd.DataFrame) -> None:
    full = fn(data)
    rng = np.random.default_rng(7)
    for cut in (150, 250):
        changed = data.copy()
        tail = changed.iloc[cut:].to_numpy().copy()
        changed.iloc[cut:] = tail[rng.permutation(len(tail))]
        out = fn(changed)
        before = full.index[full.index <= data.index[cut - 1]]
        if not _same(out.loc[before], full.loc[before]):
            raise AssertionError(f"lookahead detected by future shuffle after row {cut}")


@pytest.mark.parametrize("check", [check_truncation, check_future_noise, check_future_shuffle])
def test_rolling_mst_survival_has_no_lookahead(check: Callable[..., None]) -> None:
    check(good, make_returns())


@pytest.mark.parametrize("check", [check_truncation, check_future_noise, check_future_shuffle])
def test_detectors_catch_the_leaky_twin(check: Callable[..., None]) -> None:
    with pytest.raises(AssertionError, match="lookahead detected"):
        check(leaky_centred, make_returns())


def test_survival_of_early_dates_ignores_later_history_in_custom_corr_fn() -> None:
    """A user-supplied corr_fn only ever receives rows up to the window end."""
    data = make_returns()
    last_seen: list[pd.Timestamp] = []

    def spy(frame: pd.DataFrame) -> pd.DataFrame:
        last_seen.append(frame.index[-1])
        return frame.corr()

    out = nw.rolling_mst_survival(data, WINDOW, STEP, corr_fn=spy)
    assert last_seen == list(out.index)


def test_seed_determinism_of_permutation_test() -> None:
    mst = nw.build_mst(make_returns().corr())
    sectors = {lab: "a" if k < 4 else "b" for k, lab in enumerate(mst.nodes)}
    a = nw.sector_purity_test(mst, sectors, n_permutations=100, seed=3)
    b = nw.sector_purity_test(mst, sectors, n_permutations=100, seed=3)
    assert a == b
