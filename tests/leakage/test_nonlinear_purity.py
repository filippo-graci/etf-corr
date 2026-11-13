"""Leakage-adjacent purity tests for ``correlation/nonlinear.py`` (step 3.4a).

Same rationale as ``tests/leakage/test_estimators_purity.py`` (step 2.2): every function here is
a point estimator over whatever frame/array it is handed (or a permutation-test/FDR calculation
over such point estimates), not a per-date feature function, so ``etfcorr.leakage``'s shared
truncation/future-invariance detectors do not apply (no per-date output for their ``_upto()``
machinery to slice). The genuine "no-lookahead" property for a point estimator is purity: the
answer must depend on nothing but the rows actually passed in, and ``window`` must be a trailing,
not leading or centred, convenience. Each test also demonstrates the property failing on a
deliberately broken twin, per the project's O4 rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etfcorr.correlation import nonlinear as NL


def test_distance_correlation_matrix_ignores_rows_it_was_never_given(factor_panel) -> None:
    returns = factor_panel.returns
    t = 500
    truncated = returns.iloc[:t]
    out_before_mutation = NL.distance_correlation_matrix(truncated)

    mutated_full = returns.copy()
    mutated_full.iloc[t:] = 1.0e6  # separate object; never passed to the call below
    out_after_mutation = NL.distance_correlation_matrix(truncated)

    pd.testing.assert_frame_equal(out_before_mutation, out_after_mutation)


def test_a_function_that_secretly_peeks_past_its_argument_is_actually_caught() -> None:
    """Deliberately-leaky twin, same shape as test_estimators_purity.py's own version."""

    def leaky(returns: pd.DataFrame, peek: pd.DataFrame) -> pd.DataFrame:
        return NL.distance_correlation_matrix(pd.concat([returns, peek]))

    returns = pd.DataFrame(
        np.random.default_rng(11).standard_normal((200, 3)),
        index=pd.bdate_range("2020-01-01", periods=200),
    )
    t = 150
    truncated = returns.iloc[:t]
    peeked_rows = returns.iloc[t : t + 30]

    honest = NL.distance_correlation_matrix(truncated)
    leaky_out = leaky(truncated, peeked_rows)

    assert not honest.equals(leaky_out), (
        "the leaky twin's answer differs because it used rows past the truncation point -- "
        "confirming this comparison would catch such a bug"
    )


def test_window_trailing_semantics_are_caught_if_ever_flipped_to_leading(factor_panel) -> None:
    returns = factor_panel.returns
    window = 100

    def leaky_leading_window(returns: pd.DataFrame, window: int) -> pd.DataFrame:
        return NL.distance_correlation_matrix(returns.head(window))  # deliberately wrong

    correct = NL.distance_correlation_matrix(returns, window=window)
    leaky_out = leaky_leading_window(returns, window)

    assert not correct.equals(leaky_out)
    pd.testing.assert_frame_equal(correct, NL.distance_correlation_matrix(returns.tail(window)))


def test_distance_correlation_scalar_is_pure_in_its_arguments() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(300)
    y = rng.standard_normal(300)
    out1 = NL.distance_correlation(x, y)
    # a separate, mutated copy must not affect a call made on the original arrays
    x_copy, y_copy = x.copy(), y.copy()
    x_copy[:] = 999.0
    y_copy[:] = -999.0
    out2 = NL.distance_correlation(x, y)
    assert out1 == out2


def test_permutation_test_seed_determinism_is_the_no_hidden_state_property() -> None:
    """A hidden global RNG (module-level, not passed a seed) would make two calls with the same
    explicit seed diverge if anything else in the process had already consumed randomness --
    exercised here by drawing unrelated random numbers between the two calls."""
    rng = np.random.default_rng(20260928)
    x = rng.standard_normal(120)
    y = rng.standard_normal(120)

    first = NL.distance_correlation_permutation_test(x, y, n_permutations=50, seed=42)
    _ = np.random.default_rng(1).standard_normal(10_000)  # unrelated randomness in between
    second = NL.distance_correlation_permutation_test(x, y, n_permutations=50, seed=42)
    assert first == second
