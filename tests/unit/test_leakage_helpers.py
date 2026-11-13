"""Unit tests for etfcorr.leakage: cuts, index handling, NaNs, messages, replace_future."""

import numpy as np
import pandas as pd
import pytest

from etfcorr.leakage import (
    LeakageError,
    assert_future_invariant,
    assert_no_lookahead,
    default_cuts,
    replace_future,
)
from etfcorr.synthetic import make_factor_returns


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return make_factor_returns(400, (2, 2), seed=5).returns


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.rolling(20).mean()


# --- cuts ---------------------------------------------------------------------------------


def test_default_cuts_are_sorted_distinct_and_in_range() -> None:
    cuts = default_cuts(1500)
    assert list(cuts) == sorted(set(cuts))
    assert all(1 <= c < 1500 for c in cuts)
    assert len(cuts) == 5
    assert default_cuts(2) == (1,)  # tiny samples collapse to the only valid cut
    assert all(1 <= c < 10 for c in default_cuts(10))


def test_default_cuts_needs_two_rows() -> None:
    with pytest.raises(ValueError, match="at least 2 rows"):
        default_cuts(1)


@pytest.mark.parametrize("bad", [0, 400, 401, -1])
def test_invalid_cuts_are_rejected(data: pd.DataFrame, bad: int) -> None:
    with pytest.raises(ValueError, match="cut"):
        assert_no_lookahead(clean, data, cuts=[bad])
    with pytest.raises(ValueError, match="cut"):
        assert_future_invariant(clean, data, cuts=[bad])


def test_empty_cuts_are_rejected(data: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="no cut points"):
        assert_no_lookahead(clean, data, cuts=[])


# --- passing behaviour ------------------------------------------------------------------------


def test_clean_function_passes_all_detectors(data: pd.DataFrame) -> None:
    assert_no_lookahead(clean, data)
    assert_future_invariant(clean, data, mode="noise")
    assert_future_invariant(clean, data, mode="shuffle")


def test_series_output_and_nan_warmup_are_supported(data: pd.DataFrame) -> None:
    def series_fn(df: pd.DataFrame) -> pd.Series:
        return df.iloc[:, 0].rolling(30).std()

    assert_no_lookahead(series_fn, data)
    assert_future_invariant(series_fn, data)


def test_multiindex_output_is_compared_by_date(data: pd.DataFrame) -> None:
    def pairwise(df: pd.DataFrame) -> pd.DataFrame:
        return df.rolling(30).corr()  # index: (date, column)

    assert isinstance(pairwise(data).index, pd.MultiIndex)
    assert_no_lookahead(pairwise, data, cuts=[150, 300])
    assert_future_invariant(pairwise, data, cuts=[150, 300])


def test_multiindex_leak_is_detected(data: pd.DataFrame) -> None:
    def leaky_pairwise(df: pd.DataFrame) -> pd.DataFrame:
        return df.rolling(30, center=True).corr()

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky_pairwise, data, cuts=[150, 300])


# --- failure modes and messages -------------------------------------------------------------------


def test_value_leak_message_names_cut_row_and_size(data: pd.DataFrame) -> None:
    def leaky(df: pd.DataFrame) -> pd.DataFrame:
        return (df - df.mean()) / df.std()  # full-sample scaling: values differ, no NaN change

    with pytest.raises(LeakageError, match=r"truncation at row \d+ .*max abs diff"):
        assert_no_lookahead(leaky, data)


def test_nan_pattern_change_is_a_leak(data: pd.DataFrame) -> None:
    def leaky(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.iloc[: len(df) // 3] = np.nan  # blank length depends on how much data exists
        return out

    with pytest.raises(LeakageError, match="NaN pattern"):
        assert_no_lookahead(leaky, data, cuts=[150, 300])


def test_index_change_is_a_leak(data: pd.DataFrame) -> None:
    def leaky(df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[len(df) // 4 :]  # drops a share of rows that depends on the length

    with pytest.raises(LeakageError, match="index differs"):
        assert_no_lookahead(leaky, data, cuts=[150, 300])


def test_column_change_is_a_leak(data: pd.DataFrame) -> None:
    def leaky(df: pd.DataFrame) -> pd.DataFrame:
        keep = 2 if len(df) < 200 else 3
        return df.iloc[:, :keep]

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, data, cuts=[150, 300])


def test_relabelled_columns_are_a_leak(data: pd.DataFrame) -> None:
    """Same shape and values, different column labels: only the column check can see it."""

    def leaky(df: pd.DataFrame) -> pd.DataFrame:
        out = df.rolling(20).mean()
        out.columns = [f"{c}_{'a' if len(df) < 200 else 'b'}" for c in out.columns]
        return out

    with pytest.raises(LeakageError, match="columns differ"):
        assert_no_lookahead(leaky, data, cuts=[150, 300])


def test_type_change_is_a_leak(data: pd.DataFrame) -> None:
    def leaky(df: pd.DataFrame) -> pd.DataFrame | pd.Series:
        return df.iloc[:, 0] if len(df) < 200 else df

    with pytest.raises(LeakageError, match="type"):
        assert_no_lookahead(leaky, data, cuts=[150, 300])


def test_tolerance_is_respected(data: pd.DataFrame) -> None:
    """A 1e-6 relative peek is caught by default and accepted with a loose tolerance."""

    def tiny_leak(df: pd.DataFrame) -> pd.DataFrame:
        return df.rolling(20).mean() + 1e-6 * df.shift(-1).fillna(0.0)

    with pytest.raises(LeakageError):
        assert_no_lookahead(tiny_leak, data, cuts=[150, 300])
    assert_no_lookahead(tiny_leak, data, cuts=[150, 300], atol=1e-3)


def test_output_dates_after_the_cut_are_not_compared(data: pd.DataFrame) -> None:
    """Only dates up to the cut count: what the full run reports later is irrelevant."""
    assert_no_lookahead(lambda df: df.cumsum(), data, cuts=[100])


# --- replace_future ------------------------------------------------------------


def test_replace_future_keeps_the_past_and_the_input_untouched(data: pd.DataFrame) -> None:
    original = data.copy()
    rng = np.random.default_rng(0)
    for mode in ("noise", "shuffle"):
        out = replace_future(data, 250, rng, mode)  # type: ignore[arg-type]
        pd.testing.assert_frame_equal(out.iloc[:250], data.iloc[:250])
        assert out.index.equals(data.index)
        assert not out.iloc[250:].equals(data.iloc[250:])
    pd.testing.assert_frame_equal(data, original)


def test_shuffle_preserves_the_multiset_of_future_rows(data: pd.DataFrame) -> None:
    out = replace_future(data, 250, np.random.default_rng(1), "shuffle")
    a = np.sort(out.iloc[250:].to_numpy(), axis=0)
    b = np.sort(data.iloc[250:].to_numpy(), axis=0)
    np.testing.assert_array_equal(a, b)


def test_noise_future_uses_only_past_moments(data: pd.DataFrame) -> None:
    """Changing the future must not change the noise's parameters (checked via equal draws)."""
    changed = data.copy()
    changed.iloc[250:] = 999.0
    a = replace_future(data, 250, np.random.default_rng(2), "noise")
    b = replace_future(changed, 250, np.random.default_rng(2), "noise")
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("t", [0, 1, 400, 500])
def test_replace_future_rejects_bad_cut(data: pd.DataFrame, t: int) -> None:
    with pytest.raises(ValueError, match="t must satisfy"):
        replace_future(data, t, np.random.default_rng(0))


def test_future_detector_is_seed_deterministic(data: pd.DataFrame) -> None:
    def leaky(df: pd.DataFrame) -> pd.DataFrame:
        return df.rolling(10, center=True).mean()

    messages = []
    for _ in range(2):
        with pytest.raises(LeakageError) as info:
            assert_future_invariant(leaky, data, seed=7)
        messages.append(str(info.value))
    assert messages[0] == messages[1]
