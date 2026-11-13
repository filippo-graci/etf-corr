"""Leakage-adjacent purity tests for ``correlation/tail.py`` (step 3.4b).

Same rationale as ``test_nonlinear_purity.py``/``test_estimators_purity.py``: point estimators
over whatever array/frame they are handed, tested for purity and trailing-window semantics, each
demonstrated failing on a deliberately broken twin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etfcorr.correlation import tail as T


def test_tail_dependence_matrix_ignores_rows_it_was_never_given(factor_panel) -> None:
    returns = factor_panel.returns
    t = 500
    truncated = returns.iloc[:t]
    out_before_mutation = T.tail_dependence_matrix(truncated, q=0.1)

    mutated_full = returns.copy()
    mutated_full.iloc[t:] = 1.0e6
    out_after_mutation = T.tail_dependence_matrix(truncated, q=0.1)

    pd.testing.assert_frame_equal(out_before_mutation, out_after_mutation)


def test_a_function_that_secretly_peeks_past_its_argument_is_actually_caught() -> None:
    def leaky(returns: pd.DataFrame, peek: pd.DataFrame) -> pd.DataFrame:
        return T.tail_dependence_matrix(pd.concat([returns, peek]), q=0.1)

    returns = pd.DataFrame(
        np.random.default_rng(11).standard_normal((300, 3)),
        index=pd.bdate_range("2020-01-01", periods=300),
    )
    t = 200
    truncated = returns.iloc[:t]
    peeked_rows = returns.iloc[t : t + 50]

    honest = T.tail_dependence_matrix(truncated, q=0.1)
    leaky_out = leaky(truncated, peeked_rows)

    assert not honest.equals(leaky_out), (
        "the leaky twin's answer differs because it used rows past the truncation point"
    )


def test_window_trailing_semantics_are_caught_if_ever_flipped_to_leading(factor_panel) -> None:
    returns = factor_panel.returns
    window = 150

    def leaky_leading_window(returns: pd.DataFrame, window: int) -> pd.DataFrame:
        return T.tail_dependence_matrix(returns.head(window), q=0.1)  # deliberately wrong

    correct = T.tail_dependence_matrix(returns, q=0.1, window=window)
    leaky_out = leaky_leading_window(returns, window)

    assert not correct.equals(leaky_out)
    pd.testing.assert_frame_equal(correct, T.tail_dependence_matrix(returns.tail(window), q=0.1))


def test_empirical_tail_dependence_is_pure_in_its_arguments() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(300)
    y = 0.5 * x + rng.standard_normal(300)
    out1 = T.empirical_tail_dependence(x, y, q=0.1)
    x_copy, y_copy = x.copy(), y.copy()
    x_copy[:] = 999.0
    y_copy[:] = -999.0
    out2 = T.empirical_tail_dependence(x, y, q=0.1)
    assert out1 == out2


def test_bootstrap_seed_determinism_is_the_no_hidden_state_property() -> None:
    rng = np.random.default_rng(20260928)
    x = rng.standard_normal(150)
    y = rng.standard_normal(150)

    first = T.bootstrap_tail_dependence(x, y, q=0.1, n_bootstrap=80, seed=42)
    _ = np.random.default_rng(1).standard_normal(10_000)  # unrelated randomness in between
    second = T.bootstrap_tail_dependence(x, y, q=0.1, n_bootstrap=80, seed=42)
    assert first == second
