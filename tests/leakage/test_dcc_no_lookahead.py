"""Leakage tests for ``etfcorr.correlation.dcc`` (step 4.4), using the shared detectors.

The causal pipeline (GARCH fits and DCC fit on an expanding window, filter re-run with the frozen
parameters) is checked by truncation and by replacing the future. Each way the same model can be
made to leak must be *caught* by the same detectors, so a green run proves they can see the leak
they are meant to see (project_plan.md O4 rule): parameters fitted once on the full sample, a refit
schedule that depends on the length of the data, a correlation target ``Qbar`` taken from the whole
sample, a variance backcast taken from the whole sample, and a forecast dated one day too early.
Controls swap each leak for its causal counterpart and must pass.

Mode choice: the returns are signed, so the ``noise`` mode is valid; ``shuffle`` is run as well.
The default cut points rarely land on a refit date, so cuts exactly at and around the refit dates
are tested separately (the lesson of step 4.3).
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from etfcorr.correlation import dcc
from etfcorr.leakage import LeakageError, assert_future_invariant, assert_no_lookahead

RB = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
N = 360
MIN_OBS = 120
REFIT = 80
KW = {"dist": "normal", "min_obs": MIN_OBS, "refit_every": REFIT}


def _returns(seed: int = 0) -> pd.DataFrame:
    return dcc.simulate_dcc(N, RB, a=0.08, b=0.85, seed=seed).returns


def _causal(d: pd.DataFrame) -> pd.DataFrame:
    return dcc.rolling_dcc(d, **KW).avg_corr  # type: ignore[arg-type]


def _frame(values: np.ndarray, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({"h1": values}, index=index)


def _avg_next(corr_next: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Average correlation of one-step forecasts: row t of ``corr_next`` is R_{t+1|t}."""
    return _frame(np.asarray(dcc.average_correlation(corr_next)), index)


BOUNDARIES = [
    MIN_OBS - 1,
    MIN_OBS,
    MIN_OBS + 1,
    MIN_OBS + REFIT - 1,
    MIN_OBS + REFIT,
    MIN_OBS + REFIT + 1,
    MIN_OBS + 2 * REFIT,
]


# ---- the causal pipeline passes ------------------------------------------------------


def test_rolling_dcc_has_no_lookahead() -> None:
    assert_no_lookahead(_causal, _returns())


def test_no_lookahead_exactly_at_the_refit_boundaries() -> None:
    assert_no_lookahead(_causal, _returns(), cuts=BOUNDARIES)
    assert_future_invariant(_causal, _returns(), cuts=BOUNDARIES, mode="noise")


def test_rolling_dcc_ignores_a_noisy_future() -> None:
    assert_future_invariant(_causal, _returns(), mode="noise")


def test_rolling_dcc_ignores_a_shuffled_future() -> None:
    assert_future_invariant(_causal, _returns(), mode="shuffle")


def test_multi_step_forecasts_have_no_lookahead() -> None:
    fn = lambda d: dcc.rolling_dcc(d, horizons=(1, 21), **KW).avg_corr  # type: ignore[arg-type]  # noqa: E731
    assert_no_lookahead(fn, _returns(), cuts=BOUNDARIES)
    assert_future_invariant(fn, _returns(), mode="noise")


def test_asymmetric_dcc_has_no_lookahead() -> None:
    fn = lambda d: dcc.rolling_dcc(d, asymmetric=True, **KW).avg_corr  # type: ignore[arg-type]  # noqa: E731
    assert_no_lookahead(fn, _returns())
    assert_future_invariant(fn, _returns(), mode="noise")


def test_cold_start_refits_have_no_lookahead() -> None:
    fn = lambda d: dcc.rolling_dcc(d, warm_start=False, **KW).avg_corr  # type: ignore[arg-type]  # noqa: E731
    assert_no_lookahead(fn, _returns(), cuts=BOUNDARIES)


def test_student_t_marginals_have_no_lookahead() -> None:
    fn = lambda d: dcc.rolling_dcc(d, dist="t", min_obs=MIN_OBS, refit_every=REFIT).avg_corr  # noqa: E731
    assert_no_lookahead(fn, _returns(), cuts=[MIN_OBS, MIN_OBS + REFIT + 1])


def test_full_matrices_have_no_lookahead() -> None:
    def fn(d: pd.DataFrame) -> pd.DataFrame:
        path = dcc.rolling_dcc(d, **KW)  # type: ignore[arg-type]
        assert path.corr_next is not None
        return pd.DataFrame(path.corr_next.reshape(len(d), -1), index=d.index)

    assert_no_lookahead(fn, _returns(), cuts=BOUNDARIES)
    assert_future_invariant(fn, _returns(), cuts=BOUNDARIES, mode="noise")


def test_leading_missing_rows_do_not_change_later_outputs() -> None:
    d = _returns()
    padded = d.copy()
    padded.iloc[:5, 0] = np.nan
    np.testing.assert_allclose(
        _causal(padded).iloc[5:].to_numpy(), _causal(d.iloc[5:]).to_numpy(), equal_nan=True
    )
    assert _causal(padded).iloc[: 5 + MIN_OBS - 1].isna().all().all()


# ---- the model fitted on a fixed early window, then filtered, passes (control) -------


def _fixed_model_forecast(
    d: pd.DataFrame, *, qbar_from_all: bool = False, h0_from_all: bool = False
):
    x = d.to_numpy(dtype=float)
    model = dcc.fit_dcc_model(x[:MIN_OBS], list(d.columns), dist="normal")
    if qbar_from_all or h0_from_all:
        garch = model.garch
        dcc_fit = model.dcc
        if h0_from_all:
            garch = tuple(
                dataclasses.replace(g, h0=float(np.mean((x[:, i] - g.mu) ** 2)))
                for i, g in enumerate(garch)
            )
        if qbar_from_all:
            flt = dcc.filter_dcc(dataclasses.replace(model, garch=garch), x)
            qbar = (flt.z[:, :, None] * flt.z[:, None, :]).mean(axis=0)
            dcc_fit = dataclasses.replace(dcc_fit, qbar=qbar)
        model = dataclasses.replace(model, garch=garch, dcc=dcc_fit)
    flt = dcc.filter_dcc(model, x)
    return _avg_next(flt.corr[1:], d.index)


def test_a_model_fitted_on_the_first_rows_only_and_filtered_forward_passes() -> None:
    cuts = [MIN_OBS, MIN_OBS + 1, 200, 300]
    assert_no_lookahead(_fixed_model_forecast, _returns(), cuts=cuts)
    assert_future_invariant(_fixed_model_forecast, _returns(), cuts=cuts, mode="noise")


# ---- leaky twins are caught ----------------------------------------------------------


def test_parameters_fitted_once_on_the_full_sample_are_flagged() -> None:
    """Filtered forward, but every parameter saw the whole sample."""

    def leaky(d: pd.DataFrame) -> pd.DataFrame:
        x = d.to_numpy(dtype=float)
        model = dcc.fit_dcc_model(x, list(d.columns), dist="normal")
        return _avg_next(dcc.filter_dcc(model, x).corr[1:], d.index)

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, _returns())
    with pytest.raises(LeakageError):
        assert_future_invariant(leaky, _returns(), mode="noise")


def test_a_refit_schedule_that_depends_on_the_sample_length_is_flagged() -> None:
    def leaky(d: pd.DataFrame) -> pd.DataFrame:
        every = max(len(d) // 4, 60)
        return dcc.rolling_dcc(d, dist="normal", min_obs=MIN_OBS, refit_every=every).avg_corr

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, _returns())


def test_a_correlation_target_taken_from_the_whole_sample_is_flagged() -> None:
    def leaky(d: pd.DataFrame) -> pd.DataFrame:
        return _fixed_model_forecast(d, qbar_from_all=True)

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, _returns(), cuts=[MIN_OBS, 200, 300])
    with pytest.raises(LeakageError):
        assert_future_invariant(leaky, _returns(), cuts=[MIN_OBS, 200, 300], mode="noise")


def test_a_variance_backcast_taken_from_the_whole_sample_is_flagged() -> None:
    """The subtlest one: a one-time initial value whose effect decays geometrically."""

    def leaky(d: pd.DataFrame) -> pd.DataFrame:
        return _fixed_model_forecast(d, h0_from_all=True)

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, _returns(), cuts=[MIN_OBS, 200, 300])


def test_a_forecast_dated_one_day_too_early_is_flagged() -> None:
    """Value at row t is R_{t+2|t+1}: it needs the return of day t+1."""

    def leaky(d: pd.DataFrame) -> pd.DataFrame:
        x = d.to_numpy(dtype=float)
        model = dcc.fit_dcc_model(x[:MIN_OBS], list(d.columns), dist="normal")
        corr = dcc.filter_dcc(model, x).corr  # (T + 1, N, N)
        values = np.full(len(d), np.nan)
        values[:-1] = np.asarray(dcc.average_correlation(corr[2:]))
        return _frame(values, d.index)

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, _returns(), cuts=[MIN_OBS, 200, 300])


def test_a_centred_smoothing_of_the_output_is_flagged() -> None:
    def leaky(d: pd.DataFrame) -> pd.DataFrame:
        return _causal(d).rolling(9, center=True, min_periods=1).mean()

    with pytest.raises(LeakageError):
        assert_no_lookahead(leaky, _returns(), cuts=[MIN_OBS + 5, 200, 300])
