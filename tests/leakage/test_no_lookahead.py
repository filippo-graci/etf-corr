"""No-lookahead and shuffle-the-future tests, each proven able to fail on a leaky twin.

Every feature function used later in the project (features.py, regimes.py, graph features,
forward-filtered HMM, ...) gets the same three assertions. This module (a) fixes the pattern on
small known functions and (b) shows that each detector fails on deliberately leaky versions,
including the subtle ones (full-sample PCA, a scaler fitted on a length-dependent "train" share).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from etfcorr.leakage import (
    FeatureFn,
    LeakageError,
    assert_future_invariant,
    assert_no_lookahead,
)
from etfcorr.synthetic import SyntheticPanel

# --------------------------------------------------------------------------------------
# clean features: value at t uses data up to t only
# --------------------------------------------------------------------------------------


def rolling_corr_with_first(df: pd.DataFrame) -> pd.DataFrame:
    """The pattern of project_plan.md section 9: 63-day correlation with the first column."""
    return df.rolling(63).corr(df.iloc[:, 0])


def rolling_mean_std_ratio(df: pd.DataFrame) -> pd.DataFrame:
    roll = df.rolling(40)
    return roll.mean() / roll.std()


def ewma_vol(df: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    return df.pow(2).ewm(alpha=1 - lam, adjust=False).mean().pow(0.5)


def expanding_zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.expanding(30).mean()) / df.expanding(30).std()


def cumulative_return(df: pd.DataFrame) -> pd.DataFrame:
    return df.cumsum()


def _remove_top_pc(train: np.ndarray, x: np.ndarray) -> np.ndarray:
    _, vecs = np.linalg.eigh(np.cov(train, rowvar=False))
    v = vecs[:, -1:]
    return x - (x @ v) @ v.T


def past_only_pc_removal(df: pd.DataFrame, step: int = 50, min_fit: int = 100) -> pd.DataFrame:
    """Remove the leading principal component, refitted every ``step`` rows on earlier rows."""
    x = df.to_numpy()
    out = np.full_like(x, np.nan)
    for start in range(min_fit, len(df), step):
        end = min(start + step, len(df))
        out[start:end] = _remove_top_pc(x[:start], x[start:end])
    return pd.DataFrame(out, index=df.index, columns=df.columns)


CLEAN: dict[str, FeatureFn] = {
    "rolling_corr_with_first": rolling_corr_with_first,
    "rolling_mean_std_ratio": rolling_mean_std_ratio,
    "ewma_vol": ewma_vol,
    "expanding_zscore": expanding_zscore,
    "cumulative_return": cumulative_return,
    "past_only_pc_removal": past_only_pc_removal,
}

# --------------------------------------------------------------------------------------
# leaky twins
# --------------------------------------------------------------------------------------


def leaky_centered_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.rolling(21, center=True).mean()


def leaky_full_sample_zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std()


def leaky_negative_shift(df: pd.DataFrame) -> pd.DataFrame:
    return df.shift(-1)


def leaky_forward_label(df: pd.DataFrame) -> pd.DataFrame:
    """A 5-day forward return used as if it were a feature."""
    return df.rolling(5).sum().shift(-5)


def leaky_future_max_scaling(df: pd.DataFrame) -> pd.DataFrame:
    return df / df.abs().max()


def leaky_full_sample_pc_removal(df: pd.DataFrame) -> pd.DataFrame:
    """Market-mode removal with the eigenvector estimated on the whole sample."""
    x = df.to_numpy()
    return pd.DataFrame(_remove_top_pc(x, x), index=df.index, columns=df.columns)


def leaky_train_share_scaler(df: pd.DataFrame) -> pd.DataFrame:
    """Scaler fitted on 'the first 60 percent': the share is relative to the data length."""
    cut = int(0.6 * len(df))
    return (df - df.iloc[:cut].mean()) / df.iloc[:cut].std()


def leaky_tiny_peek(df: pd.DataFrame) -> pd.DataFrame:
    """A clean rolling mean plus a one-millionth weight on tomorrow's return."""
    return df.rolling(20).mean() + 1e-6 * df.shift(-1).fillna(0.0)


def leaky_corr_shifted(df: pd.DataFrame) -> pd.DataFrame:
    return rolling_corr_with_first(df).shift(-1)


LEAKY: dict[str, FeatureFn] = {
    "centered_mean": leaky_centered_mean,
    "full_sample_zscore": leaky_full_sample_zscore,
    "negative_shift": leaky_negative_shift,
    "forward_label": leaky_forward_label,
    "future_max_scaling": leaky_future_max_scaling,
    "full_sample_pc_removal": leaky_full_sample_pc_removal,
    "train_share_scaler": leaky_train_share_scaler,
    "tiny_peek": leaky_tiny_peek,
    "corr_shifted": leaky_corr_shifted,
}

# Statistics that do not change when future rows are only re-ordered: the shuffle mode cannot
# see these leaks (the noise mode and truncation can). Documented limitation of the shuffle test.
SHUFFLE_BLIND = {"full_sample_zscore", "future_max_scaling", "full_sample_pc_removal"}


@pytest.fixture(params=["factor_panel", "noise_panel"])
def returns(request: pytest.FixtureRequest) -> pd.DataFrame:
    frame: pd.DataFrame = request.getfixturevalue(request.param).returns
    return frame


# --------------------------------------------------------------------------------------
# clean features pass every detector
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CLEAN))
def test_clean_feature_passes_truncation(name: str, returns: pd.DataFrame) -> None:
    assert_no_lookahead(CLEAN[name], returns)


@pytest.mark.parametrize("name", list(CLEAN))
@pytest.mark.parametrize("mode", ["noise", "shuffle"])
def test_clean_feature_passes_future_replacement(
    name: str, mode: str, returns: pd.DataFrame
) -> None:
    assert_future_invariant(CLEAN[name], returns, mode=mode)  # type: ignore[arg-type]


def test_plan_pattern_last_row_equals_full_sample_row(factor_panel: SyntheticPanel) -> None:
    """project_plan.md section 9, test 1, written out by hand (independent of the helpers)."""
    panel = factor_panel.returns
    t = 1000
    a = rolling_corr_with_first(panel.iloc[:t]).iloc[-1]
    b = rolling_corr_with_first(panel).iloc[t - 1]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


# --------------------------------------------------------------------------------------
# leaky twins are caught (the detectors can fail)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(LEAKY))
def test_truncation_catches_every_leaky_feature(name: str, returns: pd.DataFrame) -> None:
    with pytest.raises(LeakageError):
        assert_no_lookahead(LEAKY[name], returns)


@pytest.mark.parametrize("name", list(LEAKY))
def test_noise_future_catches_every_leaky_feature(name: str, returns: pd.DataFrame) -> None:
    with pytest.raises(LeakageError):
        assert_future_invariant(LEAKY[name], returns, mode="noise")


@pytest.mark.parametrize("name", [n for n in LEAKY if n not in SHUFFLE_BLIND])
def test_shuffle_future_catches_order_dependent_leaks(name: str, returns: pd.DataFrame) -> None:
    with pytest.raises(LeakageError):
        assert_future_invariant(LEAKY[name], returns, mode="shuffle")


@pytest.mark.parametrize("name", sorted(SHUFFLE_BLIND))
def test_shuffle_future_is_blind_to_permutation_invariant_leaks(
    name: str, returns: pd.DataFrame
) -> None:
    """Documents a limit: re-ordering the future keeps its mean, std, max and covariance, so
    this detector passes leaky full-sample statistics. That is why truncation and noise are the
    primary detectors and the shuffle mode is only a complement."""
    assert_future_invariant(LEAKY[name], returns, mode="shuffle")


@pytest.mark.parametrize("k", [1, 5, 20])
def test_peeks_of_any_horizon_are_caught(k: int, returns: pd.DataFrame) -> None:
    def peek(df: pd.DataFrame) -> pd.DataFrame:
        return df.rolling(20).mean() + df.shift(-k).fillna(0.0)

    with pytest.raises(LeakageError):
        assert_no_lookahead(peek, returns)


# --------------------------------------------------------------------------------------
# late-onset leak: detection depends on the cuts, so the default cuts span the sample
# --------------------------------------------------------------------------------------


def _late_onset_leak(onset: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def fn(df: pd.DataFrame) -> pd.DataFrame:
        out = df.rolling(10).mean()
        out.iloc[onset:] = df.shift(-1).iloc[onset:]  # leaks only from row `onset` on
        return out

    return fn


def test_late_onset_leak_is_caught_by_default_cuts(returns: pd.DataFrame) -> None:
    with pytest.raises(LeakageError):
        assert_no_lookahead(_late_onset_leak(1000), returns)


def test_late_onset_leak_is_missed_if_all_cuts_precede_it(returns: pd.DataFrame) -> None:
    """Limit of the method: a leak that only exists after every cut point stays invisible."""
    assert_no_lookahead(_late_onset_leak(1000), returns, cuts=[300, 600, 900])
