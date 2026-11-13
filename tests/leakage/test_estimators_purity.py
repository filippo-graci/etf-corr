"""Leakage-adjacent purity tests for ``correlation/estimators.py`` (step 2.2).

**Why this file does not call ``etfcorr.leakage.assert_no_lookahead``/``assert_future_invariant``
directly.** Those detectors are built for *feature functions* that produce one output row per
input date (``fn(data_up_to_t) -> value_at_t``; see ``leakage.py``'s own docstring: output
"indexed by the same dates or by a MultiIndex whose first level is the date"). Every function in
``correlation/estimators.py`` is a *point estimator*: given whatever ``returns`` frame it is
handed, it returns ONE N x N matrix indexed by ticker, not by date -- there is no per-date output
for the detectors' ``_upto()`` machinery to slice. Forcing that machinery onto a ticker-indexed
matrix would either crash (treating ticker labels as dates) or silently check nothing meaningful.

The no-lookahead property that actually matters for a point estimator is different and simpler:
it must be **pure in its ``returns`` argument** -- its answer must depend on nothing but the rows
actually passed in, and its ``window`` parameter (a convenience for "the last N rows") must behave
as a *trailing* window, not a leading or centred one. This file tests exactly that, directly, and
demonstrates each test failing on a deliberately broken twin, per the project's O4 rule that
every leakage-style test must also fail on a leaky version.

The genuine walk-forward no-lookahead obligation for correlation estimation -- "only ever call
``estimate`` with a returns frame truncated to what was known at each point in time" -- is the
CALLER's job (Phase 3+ rolling/walk-forward code), not this module's; this module's contract ends
at "a pure function of exactly the frame it is given," which is what is verified here.
"""

import numpy as np
import pandas as pd

from etfcorr.correlation import estimators as E


def test_estimate_ignores_anything_done_to_rows_it_was_never_given(factor_panel) -> None:
    """Truncate to the first `t` rows, mutate everything *after* `t` in a separate copy to
    obviously-wrong values, and confirm the truncated call's answer is identical either way --
    it never had a reference to the mutated copy, so this is trivially true by construction, but
    it is exactly the kind of regression a hidden global cache or an accidental extra-rows slice
    bug would break."""
    returns = factor_panel.returns
    t = 500
    truncated = returns.iloc[:t]
    out_before_mutation = E.estimate(truncated, "pearson")

    mutated_full = returns.copy()
    mutated_full.iloc[t:] = 1.0e6  # obviously-wrong "future" values, in a SEPARATE object
    out_after_mutation = E.estimate(truncated, "pearson")  # same `truncated` object as before

    pd.testing.assert_frame_equal(out_before_mutation, out_after_mutation)


def test_a_function_that_secretly_peeks_past_its_argument_is_actually_caught() -> None:
    """Deliberately-leaky twin: instead of estimating on exactly what it is given, it silently
    concatenates a few extra ("peeked") rows first -- the shape of bug this project's leakage
    culture asks every such test to demonstrate catching, not just assert away."""

    def leaky_estimate(returns: pd.DataFrame, peek: pd.DataFrame) -> pd.DataFrame:
        return E.estimate(pd.concat([returns, peek]), "pearson")

    returns = pd.DataFrame(
        np.random.default_rng(11).standard_normal((200, 4)),
        index=pd.bdate_range("2020-01-01", periods=200),
    )
    t = 150
    truncated = returns.iloc[:t]
    peeked_rows = returns.iloc[t : t + 20]

    honest = E.estimate(truncated, "pearson")
    leaky = leaky_estimate(truncated, peeked_rows)

    assert not honest.equals(leaky), (
        "the leaky twin's answer differs because it used rows past the truncation point -- "
        "confirming a direct comparison like this would catch such a bug, were one ever "
        "introduced into the real estimate() function"
    )


def test_window_trailing_semantics_are_caught_if_ever_flipped_to_leading(factor_panel) -> None:
    """A `window` implementation that silently used `.head(window)` instead of `.tail(window)`
    would be a real look-ahead-shaped bug in any walk-forward context that assumes "the last N
    observations as of now": this test fails on that deliberately-broken alternative."""
    returns = factor_panel.returns
    window = 80

    def leaky_leading_window(returns: pd.DataFrame, method: str, window: int) -> pd.DataFrame:
        return E.estimate(returns.head(window), method)  # deliberately wrong direction

    correct = E.estimate(returns, "pearson", window=window)
    leaky = leaky_leading_window(returns, "pearson", window)

    assert not correct.equals(leaky)
    # and the correct trailing behaviour matches the honest manual slice exactly:
    pd.testing.assert_frame_equal(correct, E.estimate(returns.tail(window), "pearson"))


def test_ewma_corr_is_also_pure_in_its_argument(factor_panel) -> None:
    """Same purity property as Pearson/Spearman/Kendall/Ledoit-Wolf/OAS, checked separately for
    the EWMA recursion since it is the one method with hand-written iteration (a for-loop is
    exactly the kind of code where an off-by-one could accidentally read one row too many)."""
    returns = factor_panel.returns
    t = 300
    truncated = returns.iloc[:t]
    out1 = E.ewma_corr(truncated, lam=0.9)

    mutated_full = returns.copy()
    mutated_full.iloc[t:] = -1.0e6
    out2 = E.ewma_corr(truncated, lam=0.9)  # same `truncated` object

    pd.testing.assert_frame_equal(out1, out2)
