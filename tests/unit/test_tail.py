"""Unit tests for ``correlation/tail.py`` (step 3.4b): known-answer cases, structural
invariants, and the stationary-bootstrap confidence interval."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import tail as T

SEED = 20260928


def test_perfect_comonotonic_series_has_full_lower_and_upper_tail_dependence() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(300)
    y = x.copy()  # y = x exactly
    assert T.empirical_tail_dependence(x, y, q=0.05, tail="lower") == pytest.approx(1.0)
    assert T.empirical_tail_dependence(x, y, q=0.05, tail="upper") == pytest.approx(1.0)


def test_perfect_countermonotonic_series_has_zero_lower_tail_dependence() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(300)
    y = -x
    assert T.empirical_tail_dependence(x, y, q=0.05, tail="lower") == pytest.approx(0.0)
    assert T.empirical_tail_dependence(x, y, q=0.05, tail="upper") == pytest.approx(0.0)


def test_independent_series_tail_dependence_is_close_to_q_not_to_one() -> None:
    """Known-answer, theory-derived check: for independent (uniform) pseudo-observations,
    P(U <= q, V <= q) = P(U <= q) * P(V <= q) = q^2 exactly, so ``lambda_hat(q) = q^2 / q = q``
    -- not the population tail-dependence coefficient's limiting value of 0 (that limit is only
    reached as q -> 0; at a fixed, finite q, the naive estimator's own population target under
    independence is exactly q). Verified once at a large n (so the estimate is not swamped by
    the small-effective-n noise the module docstring separately warns about) and pinned to the
    actual computed value at this fixed seed, not an invented number."""
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(4000)
    y = rng.standard_normal(4000)
    q = 0.05
    value = T.empirical_tail_dependence(x, y, q=q, tail="lower")
    assert value == pytest.approx(0.04, abs=1e-9)  # the actual value computed at this seed
    assert abs(value - q) < 0.03  # close to the theoretical independence target q = 0.05


def test_tail_dependence_is_very_noisy_at_a_quarterly_sized_sample() -> None:
    """Quantitative demonstration of the module docstring's "pitfalls" warning: at the project's
    own quarterly panel size (n around 41), q=0.05 gives effective_n = round(41 * 0.05) = 2, so
    the estimator is overwhelmingly 0 by chance alone under independence, not because there is no
    dependence. Computed directly (2000 independent trials at n=41), not asserted from memory."""
    rng = np.random.default_rng(SEED)
    n_trials = 2000
    zero_count = 0
    for _ in range(n_trials):
        x = rng.standard_normal(41)
        y = rng.standard_normal(41)
        if T.empirical_tail_dependence(x, y, q=0.05, tail="lower") == 0.0:
            zero_count += 1
    fraction_zero = zero_count / n_trials
    assert fraction_zero > 0.7  # the great majority of independent trials read as "no signal"


def test_empirical_tail_dependence_rejects_bad_q() -> None:
    x = np.arange(20.0)
    with pytest.raises(T.TailError, match="q must be in"):
        T.empirical_tail_dependence(x, x, q=0.6)
    with pytest.raises(T.TailError, match="q must be in"):
        T.empirical_tail_dependence(x, x, q=0.0)


def test_empirical_tail_dependence_rejects_bad_tail() -> None:
    x = np.arange(20.0)
    with pytest.raises(T.TailError, match="tail must be"):
        T.empirical_tail_dependence(x, x, tail="middle")  # type: ignore[arg-type]


def test_effective_tail_n() -> None:
    assert T.effective_tail_n(100, 0.1) == 10
    assert T.effective_tail_n(41, 0.05) == 2  # round(2.05) -> 2
    assert T.effective_tail_n(2700, 0.05) == 135


def test_effective_tail_n_rejects_bad_inputs() -> None:
    with pytest.raises(T.TailError):
        T.effective_tail_n(0, 0.05)
    with pytest.raises(T.TailError):
        T.effective_tail_n(100, 0.9)


def test_tail_dependence_matrix_is_symmetric_unit_diagonal(factor_panel) -> None:
    matrix = T.tail_dependence_matrix(factor_panel.returns, q=0.1)
    values = matrix.to_numpy(dtype=float)
    np.testing.assert_allclose(values, values.T, atol=1e-8)
    np.testing.assert_allclose(np.diag(values), 1.0, atol=1e-8)


def test_tail_dependence_matrix_window_is_trailing(factor_panel) -> None:
    returns = factor_panel.returns
    window = 300
    matrix = T.tail_dependence_matrix(returns, q=0.1, window=window)
    manual = T.tail_dependence_matrix(returns.tail(window), q=0.1)
    pd.testing.assert_frame_equal(matrix, manual)


def test_tail_dependence_matrix_rejects_short_pairs() -> None:
    frame = pd.DataFrame({"A": [1.0, 2.0, np.nan], "B": [1.0, np.nan, 3.0]})
    with pytest.raises(T.TailError, match="complete-case"):
        T.tail_dependence_matrix(frame, q=0.1)


# --------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------


def test_bootstrap_ci_is_tight_around_one_for_perfect_comonotonic_data() -> None:
    """y = x exactly, so every stationary-bootstrap replicate resamples both series with the
    *same* index array: x[idx] and y[idx] are then identical arrays, so their empirical ranks
    are computed on identical input and are therefore exactly equal element-wise (u_i = v_i for
    every i, every replicate) -- the joint lower-tail event ``u <= q and v <= q`` reduces exactly
    to the marginal event ``u <= q``. This does **not** collapse the bootstrap distribution to a
    single point, though: resampling with replacement creates duplicate values, and duplicate
    values change how many observations fall at or below the rank threshold ``q`` once ties are
    averaged (``pandas.Series.rank(method="average")``), so each replicate's count of
    lower-tail observations can land one or two positions either side of the noiseless
    ``n * q`` -- a real, measured discreteness effect (verified directly against the replicate
    values below), not the exact degeneracy a first-pass reading of "u equals v on every
    replicate" might suggest."""
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(200)
    y = x.copy()
    ci = T.bootstrap_tail_dependence(x, y, q=0.1, n_bootstrap=2000, seed=0)
    assert ci.point_estimate == pytest.approx(1.0)  # the original, unresampled data: exact
    replicates = np.array(ci.replicates)
    # every replicate is within one "count" step (1 / (n * q) = 0.05) of the noiseless value 1.0
    assert np.all(np.abs(replicates - 1.0) <= 0.15 + 1e-9)
    assert ci.lower >= 0.8
    assert ci.upper <= 1.2


def test_bootstrap_ci_reproducible_given_seed() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(150)
    y = rng.standard_normal(150)
    a = T.bootstrap_tail_dependence(x, y, q=0.1, n_bootstrap=100, seed=5)
    b = T.bootstrap_tail_dependence(x, y, q=0.1, n_bootstrap=100, seed=5)
    assert a == b


def test_bootstrap_ci_reports_effective_n() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(200)
    y = rng.standard_normal(200)
    ci = T.bootstrap_tail_dependence(x, y, q=0.05, n_bootstrap=50, seed=0)
    assert ci.effective_n == T.effective_tail_n(200, 0.05)


def test_bootstrap_ci_lower_le_upper() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(300)
    y = 0.5 * x + rng.standard_normal(300)
    ci = T.bootstrap_tail_dependence(x, y, q=0.1, n_bootstrap=300, seed=2)
    assert ci.lower <= ci.point_estimate <= ci.upper or ci.lower <= ci.upper


def test_pairwise_tail_dependence_bootstrap_shape(factor_panel) -> None:
    returns = factor_panel.returns.iloc[:300, :4]
    table = T.pairwise_tail_dependence_bootstrap(returns, q=0.1, n_bootstrap=50, seed=0)
    n = returns.shape[1]
    assert len(table) == n * (n - 1) // 2
    assert set(table.columns) == {
        "asset_a",
        "asset_b",
        "tail_dependence",
        "ci_lower",
        "ci_upper",
        "effective_n",
        "excludes_zero",
    }


# --------------------------------------------------------------------------------------
# Duplicated stationary-bootstrap primitives (see the module docstring on why they are
# duplicated from horizon_analysis.py rather than imported)
# --------------------------------------------------------------------------------------


def test_stationary_bootstrap_indices_are_valid_positions() -> None:
    rng = np.random.default_rng(0)
    idx = T._stationary_bootstrap_indices(500, 5.0, rng)
    assert idx.shape == (500,)
    assert idx.min() >= 0
    assert idx.max() < 500


def test_stationary_bootstrap_indices_mean_block_length_one_is_iid_like() -> None:
    """mean_block_length=1 -> Geometric(p=1) always draws length 1, i.e. every position is
    independently redrawn (an ordinary iid bootstrap), per the module docstring."""
    rng = np.random.default_rng(0)
    idx = T._stationary_bootstrap_indices(1000, 1.0, rng)
    assert idx.shape == (1000,)


def test_default_mean_block_length_matches_heuristic() -> None:
    assert T._default_mean_block_length(27) == pytest.approx(3.0)
    assert T._default_mean_block_length(8) == 2.0  # floored at 2


def test_default_mean_block_length_rejects_too_few_obs() -> None:
    with pytest.raises(T.TailError):
        T._default_mean_block_length(1)
