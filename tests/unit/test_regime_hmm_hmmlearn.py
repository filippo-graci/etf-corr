"""Cross-check of ``etfcorr.regime_hmm``'s inference against hmmlearn (step 4.3, Gate 2).

hmmlearn is a project dependency but cannot be installed in the offline gate sandbox, so this file
is skipped there (``importorskip``) and is **NOT EXECUTED OFFLINE**; it runs on Filippo's machine.
Only deterministic inference is compared (log-likelihood, smoothed posteriors, and the fact that
hmmlearn's posterior at the last row equals the filtered probability), never fitted parameters:
two EM implementations legitimately land on different local optima.

If hmmlearn changes how fixed parameters are assigned, these tests fail at construction, not in
the comparison; paste the traceback and the file will be adapted with a minimal diff.
"""

from typing import Any

import numpy as np
import pytest

from etfcorr.regime_hmm import HmmParams, forward_probabilities, smoothed_probabilities

hmm = pytest.importorskip("hmmlearn.hmm")

PARAMS = HmmParams(
    startprob=np.array([0.6, 0.4]),
    transmat=np.array([[0.95, 0.05], [0.08, 0.92]]),
    means=np.array([[0.0, 0.0], [2.0, 1.0]]),
    covars=np.stack([0.5 * np.eye(2), np.array([[1.0, 0.3], [0.3, 0.8]])]),
)


def _model() -> Any:
    model = hmm.GaussianHMM(n_components=2, covariance_type="full")
    model.startprob_ = PARAMS.startprob
    model.transmat_ = PARAMS.transmat
    model.means_ = PARAMS.means
    model.covars_ = PARAMS.covars
    return model


def _data(n: int = 300) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, 2)) * 1.2 + np.where(rng.random((n, 1)) < 0.4, 2.0, 0.0)


def test_log_likelihood_matches_hmmlearn_score() -> None:
    x = _data()
    _, ll = forward_probabilities(x, PARAMS)
    assert ll == pytest.approx(float(_model().score(x)), rel=1e-9)


def test_smoothed_posteriors_match_hmmlearn_predict_proba() -> None:
    """hmmlearn's posterior is the smoothed one (uses the whole sequence); ours must agree."""
    x = _data()
    np.testing.assert_allclose(
        smoothed_probabilities(x, PARAMS), _model().predict_proba(x), rtol=1e-8, atol=1e-10
    )


def test_hmmlearn_posterior_at_the_last_row_is_our_filtered_probability() -> None:
    """The look-ahead pitfall in one line: truncating the data changes hmmlearn's posterior for
    earlier rows, but its last row equals the forward-filtered probability at that date."""
    x = _data()
    filtered, _ = forward_probabilities(x, PARAMS)
    model = _model()
    for t in (50, 120, 299):
        np.testing.assert_allclose(
            model.predict_proba(x[:t])[-1], filtered[t - 1], rtol=1e-8, atol=1e-10
        )
    full = model.predict_proba(x)
    assert float(np.abs(full[:299] - filtered[:299]).max()) > 1e-3
