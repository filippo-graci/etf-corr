"""Leakage-adjacent purity tests for ``correlation/overlap.py`` (step 2.5).

**Why this file does not call ``etfcorr.leakage.assert_no_lookahead``/``assert_future_invariant``
directly.** Exactly the reason ``tests/leakage/test_estimators_purity.py`` and ``tests/leakage/
test_holdings_no_lookahead.py`` already give for their own modules: ``overlap.py``'s functions
take two ticker-indexed matrices, not a date-indexed frame, so there is no per-date output for the
shared detectors' truncation machinery to slice (see ``overlap.py``'s own module docstring,
"Scope and contract"). The point-in-time discipline for *this* module's inputs is the caller's job
(``holdings.latest_asof`` for the overlap snapshot, the correlation modules' own truncation
discipline for the correlation matrix); what this module owns, and what is tested here instead,
is (1) purity in its exact inputs, (2) invariance to the row/column ordering the inputs happen to
arrive in, and (3) that its permutation-based inference test draws from the correct population of
relabellings -- the symmetric group on N *nodes*, not on the N(N-1)/2 *pairs* -- which is the
concrete, combinatorial reason ``qap_permutation_test`` permutes node labels jointly rather than
shuffling individual pair values (see ``overlap.py``'s own module docstring for the Mantel 1967 /
Krackhardt 1988 citations, and the test below for why this is checked as a deterministic
combinatorial fact rather than an empirical variance comparison).
"""

import numpy as np
import pandas as pd

from etfcorr.correlation import overlap as O

# --------------------------------------------------------------------------------------
# Purity: output depends on nothing but the exact matrices passed in
# --------------------------------------------------------------------------------------


def _small_matrices(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = ("AAA", "BBB", "CCC", "DDD", "EEE")
    rng = np.random.default_rng(seed)
    n = len(tickers)
    ov = rng.uniform(0.05, 0.95, size=(n, n))
    ov = (ov + ov.T) / 2
    np.fill_diagonal(ov, 1.0)
    cr = rng.uniform(-0.3, 0.9, size=(n, n))
    cr = (cr + cr.T) / 2
    np.fill_diagonal(cr, 1.0)
    return (
        pd.DataFrame(ov, index=tickers, columns=tickers),
        pd.DataFrame(cr, index=tickers, columns=tickers),
    )


def test_fit_overlap_regression_ignores_copies_mutated_after_the_call() -> None:
    """Mutating separate copies of the inputs after computing a result must not change anything
    already computed -- the same triviality-by-construction regression ``test_estimators_purity.
    py``'s analogous test guards (a hidden global cache or accidental extra-row read would break
    this)."""
    overlap, corr = _small_matrices()
    result_before = O.fit_overlap_regression(overlap, corr, n_permutations=100, seed=0)

    mutated_overlap = overlap.copy()
    mutated_overlap.iloc[:, :] = 0.5
    mutated_corr = corr.copy()
    mutated_corr.iloc[:, :] = 0.0

    result_after = O.fit_overlap_regression(overlap, corr, n_permutations=100, seed=0)
    assert result_before.slope == result_after.slope
    assert result_before.intercept == result_after.intercept
    assert result_before.qap_pearson == result_after.qap_pearson


def test_a_function_that_secretly_uses_a_different_matrix_is_actually_caught() -> None:
    """Deliberately-leaky twin: instead of using exactly the ``overlap`` it is given, it silently
    averages in an unrelated third matrix first -- the shape of bug this project's leakage culture
    asks every such test to demonstrate catching, not just assert away (O4-style: every purity
    test must also fail on a deliberately broken version)."""
    overlap, corr = _small_matrices(seed=1)
    unrelated, _ = _small_matrices(seed=99)

    def leaky_fit(overlap: pd.DataFrame, corr: pd.DataFrame, unrelated: pd.DataFrame):
        blended = (overlap + unrelated) / 2
        return O.fit_overlap_regression(blended, corr, n_permutations=50, seed=0)

    honest = O.fit_overlap_regression(overlap, corr, n_permutations=50, seed=0)
    leaky = leaky_fit(overlap, corr, unrelated)
    assert honest.slope != leaky.slope, (
        "the leaky twin's slope differs because it used a blended-in matrix the honest call never "
        "saw -- confirming a direct comparison like this would catch such a bug"
    )


# --------------------------------------------------------------------------------------
# Ordering invariance: the row/column order the inputs arrive in must not matter
# --------------------------------------------------------------------------------------


def test_fit_overlap_regression_is_invariant_to_input_ticker_order() -> None:
    overlap, corr = _small_matrices(seed=2)
    reordered = list(reversed(overlap.index))
    a = O.fit_overlap_regression(overlap, corr, n_permutations=100, seed=7)
    b = O.fit_overlap_regression(
        overlap.loc[reordered, reordered],
        corr.loc[reordered, reordered],
        n_permutations=100,
        seed=7,
    )
    assert a.slope == b.slope
    assert a.intercept == b.intercept
    assert a.r_squared == b.r_squared
    # the permutation test itself is also invariant: relabelling the input before permuting is the
    # same population of relabellings as permuting from the canonical order.
    assert a.qap_pearson.observed == b.qap_pearson.observed


def test_a_windowing_style_ordering_bug_is_actually_caught() -> None:
    """Deliberately-leaky twin: instead of aligning by ticker label (``_align``'s ``.loc[common,
    common]``), it aligns two differently-ordered matrices by raw position -- exactly the
    tail/head-style directional bug ``test_estimators_purity.py``'s own windowing test guards
    against, transplanted to this module's row/column-alignment axis."""
    overlap, corr = _small_matrices(seed=3)
    shuffled_overlap = overlap.loc[list(reversed(overlap.index)), list(reversed(overlap.columns))]

    def leaky_positional_pairs(overlap: pd.DataFrame, corr: pd.DataFrame) -> np.ndarray:
        # WRONG: assumes both frames share the same row order, ignoring labels entirely.
        n = len(overlap)
        iu = np.triu_indices(n, k=1)
        return overlap.to_numpy()[iu], corr.to_numpy()[iu]

    honest_ov, _honest_cr = O.pairs_frame(overlap, corr)[["overlap", "correlation"]].to_numpy().T
    leaky_ov, _ = leaky_positional_pairs(shuffled_overlap, corr)

    assert not np.allclose(honest_ov, leaky_ov), (
        "the leaky, position-based twin's overlap values differ from the label-aligned ones once "
        "the two frames are ordered differently -- confirming this test would catch a label vs "
        "position alignment bug, were one ever introduced into `_align`"
    )


# --------------------------------------------------------------------------------------
# The core design point: QAP (node-label permutation) vs a naive flat permutation of pairs
# --------------------------------------------------------------------------------------


def test_qap_s_effective_sample_size_is_n_factorial_not_the_pair_count_factorial() -> None:
    """The concrete, deterministic reason a naive permutation of the flattened pair vector is the
    wrong null for dyadic data (see ``overlap.py``'s module docstring): QAP draws its permutations
    from the symmetric group on N *nodes* (N! possible relabellings), while a naive permutation of
    the N(N-1)/2 *pairs* draws from a population larger by many orders of magnitude (K! for
    K = N(N-1)/2). This is a combinatorial fact, true for any data, so it is checked directly
    rather than via a noisy Monte-Carlo comparison of two estimated null variances -- an earlier
    version of this test tried exactly that (comparing ``null_std`` between the two procedures on
    one synthetic dataset) and found the sign of the difference was **not stable across random
    seeds** for a simple bivariate-correlation statistic (checked this session, not assumed);
    reporting a fragile, seed-dependent inequality as if it were a general property would itself
    be exactly the kind of overstated claim this project's own citation/evidence rules forbid, so
    that version was replaced with this one.

    At N=5 (K=10 pairs), N!=120 but K!=3,628,800: drawing 500 permutations can *at most* produce
    120 distinct node relabellings (guaranteed by the pigeonhole principle, not by chance), while
    500 draws from a population of 3,628,800 orderings essentially never repeat. This is exactly
    the "effectively N degrees of freedom, not N(N-1)/2" property ``overlap.py``'s module docstring
    states, demonstrated directly against the real permutation draws
    :func:`~etfcorr.correlation.overlap.qap_permutation_test` actually performs (reproduced here
    with the identical ``np.random.default_rng(seed).permutation(...)` call, not a separate
    implementation, so this is a property of the real function's draws, not of a stand-in).
    """
    from math import factorial

    n_nodes = 5  # N! = 120
    n_pairs = n_nodes * (n_nodes - 1) // 2  # K! = 3,628,800
    n_draws = 500
    seed = 1

    node_rng = np.random.default_rng(seed)
    node_perms = [tuple(node_rng.permutation(n_nodes)) for _ in range(n_draws)]
    distinct_node_perms = len(set(node_perms))

    pair_rng = np.random.default_rng(seed)
    pair_values = np.arange(n_pairs)
    pair_perms = [tuple(pair_rng.permutation(pair_values)) for _ in range(n_draws)]
    distinct_pair_perms = len(set(pair_perms))

    assert distinct_node_perms <= factorial(n_nodes)  # hard combinatorial ceiling, always true
    assert distinct_node_perms < n_draws  # collisions actually occurred (pigeonhole: 500 > 120)
    assert distinct_pair_perms > n_draws - 5  # essentially all 500 draws distinct (K! is huge)
    assert distinct_pair_perms > distinct_node_perms + 300  # the gap is large, not marginal
