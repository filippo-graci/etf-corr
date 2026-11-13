"""Reusable look-ahead detectors for row-aligned feature functions.

Contract: a *feature function* maps a time-indexed frame ``data`` to a frame or series whose
value at date ``t`` may depend only on ``data`` up to and including ``t``. This is the project's
"pure function ``f(data_up_to_t) -> value_at_t``" rule (project_plan.md section 9, tests 1-2).
Output may be indexed by the same dates or by a ``MultiIndex`` whose first level is the date
(for example ``DataFrame.rolling(w).corr()``).

Two detectors, applied at several cut points ``t`` (number of rows kept):

* :func:`assert_no_lookahead` (truncation): ``fn(data.iloc[:t])`` must equal ``fn(data)``
  restricted to the same dates. Catches centred windows, full-sample scaling or PCA, negative
  shifts and any use of later rows.
* :func:`assert_future_invariant` (shuffle/noise the future): outputs up to date ``t`` must not
  change when every row after ``t`` is replaced by noise or randomly permuted.

Limits (what these detectors cannot see): they check the function's own outputs on the given
data only, so a leak that stays invisible on that data (for example a threshold never reached)
passes; they say nothing about *label* overlap between train and test rows (that is the job of
the purge/embargo tests); and both detectors assume ``fn`` is deterministic. Every detector is
itself tested against deliberately leaky functions (tests/leakage/test_no_lookahead.py).
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd

Frame: TypeAlias = pd.DataFrame | pd.Series
FeatureFn: TypeAlias = Callable[[pd.DataFrame], Frame]

DEFAULT_CUT_FRACTIONS: tuple[float, ...] = (0.40, 0.55, 0.70, 0.85, 0.97)


class LeakageError(AssertionError):
    """A feature function used information from after the date it reports for."""


def default_cuts(
    n_rows: int, fractions: Sequence[float] = DEFAULT_CUT_FRACTIONS
) -> tuple[int, ...]:
    """Distinct cut points (rows kept) at the given fractions of the sample, all in [1, n-1]."""
    if n_rows < 2:
        raise ValueError(f"need at least 2 rows to cut, got {n_rows}")
    cuts = sorted({min(max(int(n_rows * f), 1), n_rows - 1) for f in fractions})
    return tuple(cuts)


def _check_cuts(n_rows: int, cuts: Sequence[int]) -> tuple[int, ...]:
    if not cuts:
        raise ValueError("no cut points given")
    for t in cuts:
        if not 1 <= t < n_rows:
            raise ValueError(f"cut {t} must satisfy 1 <= cut < {n_rows} (some future must remain)")
    return tuple(cuts)


def _dates(index: pd.Index) -> pd.Index:
    return index.get_level_values(0) if isinstance(index, pd.MultiIndex) else index


def _day(label: Any) -> str:
    return str(label)[:10]


def _upto(frame: Frame, last: Any) -> Frame:
    """Rows of ``frame`` dated at or before ``last`` (first index level for a MultiIndex)."""
    mask = np.asarray(_dates(frame.index) <= last)
    return frame.loc[mask]


def _compare(actual: Frame, expected: Frame, label: str, rtol: float, atol: float) -> None:
    if type(actual) is not type(expected):
        raise LeakageError(f"{label}: output type changed ({type(actual)} vs {type(expected)})")
    if not actual.index.equals(expected.index):
        raise LeakageError(
            f"{label}: output index differs ({len(actual)} vs {len(expected)} rows); "
            "rows exist in one run and not in the other"
        )
    if (
        isinstance(actual, pd.DataFrame)
        and isinstance(expected, pd.DataFrame)
        and not actual.columns.equals(expected.columns)
    ):
        raise LeakageError(f"{label}: output columns differ")
    a = np.asarray(actual, dtype=float)
    b = np.asarray(expected, dtype=float)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        raise LeakageError(f"{label}: NaN pattern differs (warm-up or fill depends on the future)")
    both = ~nan_a
    if not np.allclose(a[both], b[both], rtol=rtol, atol=atol):
        diff = np.where(both, np.abs(a - b), 0.0)
        pos = tuple(int(i) for i in np.unravel_index(int(np.argmax(diff)), diff.shape))
        raise LeakageError(
            f"{label}: values differ, max abs diff {diff[pos]:.3e} at output row "
            f"{actual.index[pos[0]]!r}" + (f", column {pos[1]}" if a.ndim > 1 else "")
        )


def assert_no_lookahead(
    fn: FeatureFn,
    data: pd.DataFrame,
    cuts: Sequence[int] | None = None,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """Truncation test: ``fn`` on the first ``t`` rows equals ``fn`` on all rows, up to ``t``."""
    points = _check_cuts(len(data), default_cuts(len(data)) if cuts is None else cuts)
    full = fn(data)
    for t in points:
        last = data.index[t - 1]
        truncated = fn(data.iloc[:t])
        _compare(truncated, _upto(full, last), f"truncation at row {t} ({_day(last)})", rtol, atol)


def replace_future(
    data: pd.DataFrame,
    t: int,
    rng: np.random.Generator,
    mode: Literal["noise", "shuffle"] = "noise",
) -> pd.DataFrame:
    """Copy of ``data`` whose rows from position ``t`` on are noise or randomly permuted.

    ``noise`` draws Gaussian values with each column's mean and standard deviation over the
    first ``t`` rows (only the past is used); ``shuffle`` permutes the future rows. Rows before
    ``t`` and the index are untouched.
    """
    n = len(data)
    if not 2 <= t < n:
        raise ValueError(f"t must satisfy 2 <= t < {n}, got {t}")
    out = data.copy()
    future = np.asarray(data.iloc[t:], dtype=float)
    if mode == "noise":
        past = np.asarray(data.iloc[:t], dtype=float)
        future = rng.normal(past.mean(axis=0), past.std(axis=0), size=future.shape)
    elif mode == "shuffle":
        future = future[rng.permutation(len(future))]
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown mode {mode!r}")
    out.iloc[t:] = future
    return out


def assert_future_invariant(
    fn: FeatureFn,
    data: pd.DataFrame,
    cuts: Sequence[int] | None = None,
    *,
    mode: Literal["noise", "shuffle"] = "noise",
    seed: int = 0,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """Outputs dated up to ``t`` must not change when the data after ``t`` is replaced."""
    points = _check_cuts(len(data), default_cuts(len(data)) if cuts is None else cuts)
    rng = np.random.default_rng(seed)
    base = fn(data)
    for t in points:
        last = data.index[t - 1]
        altered = fn(replace_future(data, t, rng, mode))
        _compare(
            _upto(altered, last),
            _upto(base, last),
            f"{mode} the future after row {t} ({_day(last)})",
            rtol,
            atol,
        )
