"""Leakage tests for ``etfcorr.regime_hmm`` (step 4.3), using the shared ``etfcorr.leakage``
detectors.

The causal pipeline (observations -> expanding-window fit -> forward filter) is checked by
truncation and by replacing the future. Each way the same model can be made to leak must be
*caught* by the same detectors, so a green run proves they can see the leak they are meant to see
(project_plan.md O4 rule): smoothed posteriors, parameters fitted once on the full sample, and a
refit schedule that depends on the length of the data.

Mode choice: observations are signed, so the ``noise`` mode is valid on them; the returns frame is
signed as well.
"""

import numpy as np
import pandas as pd
import pytest

from etfcorr.leakage import LeakageError, assert_future_invariant, assert_no_lookahead
from etfcorr.regime_hmm import (
    HmmParams,
    fit_gaussian_hmm,
    forward_filtered_hmm,
    forward_probabilities,
    hmm_observations,
    smoothed_probabilities,
)

N = 380
MIN_OBS = 100
REFIT = 90
FAST = {"n_init": 2, "max_iter": 40}

A = np.array([[0.97, 0.03], [0.04, 0.96]])
MU = np.array([[0.0, 0.0], [2.5, 1.5]])
COV = np.stack([0.5 * np.eye(2), np.array([[1.0, 0.4], [0.4, 0.8]])])


def _obs(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    states = np.empty(N, dtype=int)
    states[0] = 0
    for t in range(1, N):
        states[t] = rng.choice(2, p=A[states[t - 1]])
    x = np.stack([rng.multivariate_normal(MU[s], COV[s]) for s in states])
    index = pd.bdate_range("2018-01-01", periods=N, name="date")
    return pd.DataFrame(x, index=index, columns=["log_vol", "z_cross_corr"])


def _returns(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scale = np.where(np.arange(N) < 200, 0.5, 2.0)[:, None]
    data = (scale * rng.standard_normal((N, 1)) + rng.standard_normal((N, 6))) * 0.01
    index = pd.bdate_range("2018-01-01", periods=N, name="date")
    return pd.DataFrame(data, index=index, columns=[f"T{i}" for i in range(6)])


SECTORS = {f"T{i}": ("a" if i < 3 else "b") for i in range(6)}


def _causal(d: pd.DataFrame) -> pd.DataFrame:
    return forward_filtered_hmm(d, 2, min_obs=MIN_OBS, refit_every=REFIT, **FAST).probs


# ---- the causal pipeline passes ------------------------------------------------------


def test_forward_filtered_probabilities_have_no_lookahead() -> None:
    assert_no_lookahead(_causal, _obs())


def test_no_lookahead_exactly_at_the_refit_boundaries() -> None:
    """The default cut points rarely land on a refit date; a fit that peeks one row ahead is only
    visible when the sample is cut exactly there (and one row either side)."""
    boundaries = [MIN_OBS - 1, MIN_OBS, MIN_OBS + 1, MIN_OBS + REFIT - 1, MIN_OBS + REFIT]
    boundaries += [MIN_OBS + REFIT + 1, MIN_OBS + 2 * REFIT]
    assert_no_lookahead(_causal, _obs(), cuts=boundaries)
    assert_future_invariant(_causal, _obs(), cuts=boundaries, mode="noise")


def test_forward_filtered_probabilities_ignore_a_noisy_future() -> None:
    assert_future_invariant(_causal, _obs(), mode="noise")


def test_forward_filtered_probabilities_ignore_a_shuffled_future() -> None:
    assert_future_invariant(_causal, _obs(), mode="shuffle")


def test_hard_labels_have_no_lookahead() -> None:
    assert_no_lookahead(
        lambda d: forward_filtered_hmm(
            d, 2, min_obs=MIN_OBS, refit_every=REFIT, **FAST
        ).labels.to_frame(),
        _obs(),
    )


def test_observations_from_returns_have_no_lookahead() -> None:
    assert_no_lookahead(lambda d: hmm_observations(d, SECTORS, window=20), _returns())
    assert_future_invariant(lambda d: hmm_observations(d, SECTORS, window=20), _returns())


def test_returns_to_regimes_pipeline_has_no_lookahead() -> None:
    def pipeline(d: pd.DataFrame) -> pd.DataFrame:
        obs = hmm_observations(d, SECTORS, window=20)
        return forward_filtered_hmm(obs, 2, min_obs=MIN_OBS, refit_every=REFIT, **FAST).probs

    assert_no_lookahead(pipeline, _returns())
    assert_future_invariant(pipeline, _returns(), mode="noise")


def test_cold_start_refits_have_no_lookahead() -> None:
    assert_no_lookahead(
        lambda d: (
            forward_filtered_hmm(
                d, 2, min_obs=MIN_OBS, refit_every=REFIT, warm_start=False, **FAST
            ).probs
        ),
        _obs(),
    )


# ---- leaky twins are caught ----------------------------------------------------------


def _causal_params(x: np.ndarray) -> HmmParams:
    return fit_gaussian_hmm(x[:MIN_OBS], 2, seed=0, n_init=2).params


def test_smoothed_posteriors_are_flagged() -> None:
    """Same parameters, fitted on the first MIN_OBS rows only (causal); the leak is smoothing."""

    def leaky(d: pd.DataFrame) -> pd.DataFrame:
        x = d.to_numpy(dtype=float)
        probs = smoothed_probabilities(x, _causal_params(x))
        return pd.DataFrame(probs, index=d.index, columns=["p_0", "p_1"])

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, _obs())
    with pytest.raises(LeakageError):
        assert_future_invariant(leaky, _obs(), mode="noise")


def test_the_same_twin_with_filtering_instead_of_smoothing_passes() -> None:
    """Control for the twin above: swap smoothing for filtering and the detector is silent."""

    def control(d: pd.DataFrame) -> pd.DataFrame:
        x = d.to_numpy(dtype=float)
        probs, _ = forward_probabilities(x, _causal_params(x))
        return pd.DataFrame(probs, index=d.index, columns=["p_0", "p_1"])

    assert_no_lookahead(control, _obs())


def test_parameters_fitted_once_on_the_full_sample_are_flagged() -> None:
    """Forward-filtered, but the parameters saw the whole sample: the subtle HMM leak."""

    def leaky(d: pd.DataFrame) -> pd.DataFrame:
        x = d.to_numpy(dtype=float)
        params = fit_gaussian_hmm(x, 2, seed=0, n_init=2).params
        probs, _ = forward_probabilities(x, params)
        return pd.DataFrame(probs, index=d.index, columns=["p_0", "p_1"])

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, _obs())
    with pytest.raises(LeakageError):
        assert_future_invariant(leaky, _obs(), mode="noise")


def test_a_refit_schedule_that_depends_on_the_sample_length_is_flagged() -> None:
    def leaky(d: pd.DataFrame) -> pd.DataFrame:
        every = max(len(d) // 4, 30)
        return forward_filtered_hmm(d, 2, min_obs=MIN_OBS, refit_every=every, **FAST).probs

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, _obs())


def test_a_centred_window_in_the_observations_is_flagged() -> None:
    def leaky(d: pd.DataFrame) -> pd.DataFrame:
        vol = d.abs().rolling(20, center=True, min_periods=20).mean()
        return vol

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, _returns())
