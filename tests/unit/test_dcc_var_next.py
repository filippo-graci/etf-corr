"""``rolling_dcc`` exports the one-step GARCH variance forecasts (follow-up E3)."""

import numpy as np

from etfcorr.correlation import dcc
from etfcorr.leakage import assert_future_invariant

RB = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
KW = {"dist": "normal", "min_obs": 120, "refit_every": 80}
RETURNS = dcc.simulate_dcc(360, RB, a=0.08, b=0.85, seed=0).returns


def test_var_next_is_the_garch_recursion_one_day_ahead() -> None:
    path = dcc.rolling_dcc(RETURNS, **KW)  # type: ignore[arg-type]
    assert path.var_next is not None and path.last_model is not None
    assert np.isnan(path.var_next[:119]).all() and np.isfinite(path.var_next[119:]).all()
    # inside the last segment the forecast for day t+1 is the filtered variance of day t+1
    h = dcc.filter_dcc(path.last_model, RETURNS.to_numpy()).h
    last_fit = 119 + 80 * 3
    np.testing.assert_allclose(path.var_next[last_fit:-1], h[last_fit + 1 :])
    fit = path.last_model.garch[1]
    e = RETURNS.to_numpy()[-1, 1] - fit.mu
    assert path.var_next[-1, 1] == fit.omega + fit.alpha * e**2 + fit.beta * h[-1, 1]
    assert dcc.rolling_dcc(RETURNS, keep_matrices=False, **KW).var_next is None  # type: ignore[arg-type]
    assert np.isnan(dcc.rolling_dcc(RETURNS.iloc[:50], **KW).var_next).all()  # type: ignore[arg-type]


def test_var_next_does_not_read_the_future() -> None:
    def causal(d):  # type: ignore[no-untyped-def]
        out = dcc.rolling_dcc(d, **KW).var_next  # type: ignore[arg-type]
        return d * 0.0 + out

    assert_future_invariant(causal, RETURNS, cuts=[119, 120, 199, 200, 300])
