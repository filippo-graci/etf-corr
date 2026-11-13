"""Cross-check of the univariate GARCH(1,1) stage of ``etfcorr.correlation.dcc`` against ``arch``.

**Needs the ``arch`` package** (``uv pip install arch``); skipped when it is not importable. It was
not importable in the offline sandbox that wrote the module, so this file is **NOT EXECUTED
OFFLINE** and the first run is Gate 2 (local). The ``arch`` calls below follow its documented
public API as remembered (``arch_model(..., mean="Zero", vol="GARCH", dist=..., rescale=False)``,
``fit(disp="off")``, ``params``, ``loglikelihood``, ``conditional_volatility``): confirmed by the
step 4.4 local run; an AttributeError/KeyError here is an interface problem of this
file, not evidence about the estimator.

What is compared. The series is demeaned first, so the mean is the sample mean in both fits and
``arch`` gets ``mean="Zero"``. (1) *Likelihood surface:* at ``arch``'s own estimates, our
``garch11_loglik`` (with our backcast) must reproduce ``arch``'s reported log-likelihood and its
conditional variances; this is a direct test of the likelihood, the recursion and the backcast
convention (decay 0.94 over the first 75 rows), checked in the installed ``arch`` source.
(2) *Optimum:* our fit must not end below ``arch``'s, and the parameters must be close.
"""

import numpy as np
import pytest

from etfcorr.correlation import dcc

arch = pytest.importorskip("arch")

RB3 = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])


def _series(nu: float | None, seed: int) -> np.ndarray:
    sim = dcc.simulate_dcc(2500, RB3, a=0.0, b=0.0, nu=nu, seed=seed)
    x = 100.0 * sim.returns.iloc[:, 0].to_numpy(dtype=float)
    return x - x.mean()


def _arch_fit(x: np.ndarray, dist: str):
    model = arch.arch_model(x, mean="Zero", vol="GARCH", p=1, q=1, dist=dist, rescale=False)
    return model.fit(disp="off")


@pytest.mark.parametrize(("dist", "nu"), [("normal", None), ("t", 6.0)])
def test_our_likelihood_reproduces_archs_at_archs_estimates(dist: str, nu: float | None) -> None:
    x = _series(nu, seed=1)
    res = _arch_fit(x, dist)
    p = res.params
    h0 = dcc.garch_backcast(x)
    ours = dcc.garch11_loglik(
        x,
        float(p["omega"]),
        float(p["alpha[1]"]),
        float(p["beta[1]"]),
        h0,
        dist="t" if dist == "t" else "normal",
        nu=float(p["nu"]) if dist == "t" else None,
    )
    assert ours == pytest.approx(float(res.loglikelihood), rel=1e-6)
    h = dcc.garch11_variance(x, float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"]), h0)
    np.testing.assert_allclose(np.sqrt(h), np.asarray(res.conditional_volatility), rtol=1e-6)


@pytest.mark.parametrize(("dist", "nu"), [("normal", None), ("t", 6.0)])
def test_our_fit_is_at_least_as_good_as_archs_and_close(dist: str, nu: float | None) -> None:
    x = _series(nu, seed=2)
    res = _arch_fit(x, dist)
    ours = dcc.fit_garch11(x, dist="t" if dist == "t" else "normal", demean=False)
    assert ours.converged
    assert ours.loglik >= float(res.loglikelihood) - 1e-3
    p = res.params
    assert ours.alpha == pytest.approx(float(p["alpha[1]"]), abs=0.01)
    assert ours.beta == pytest.approx(float(p["beta[1]"]), abs=0.03)
    assert ours.omega == pytest.approx(float(p["omega"]), rel=0.3, abs=0.05)
    if dist == "t":
        assert ours.nu == pytest.approx(float(p["nu"]), rel=0.1)
