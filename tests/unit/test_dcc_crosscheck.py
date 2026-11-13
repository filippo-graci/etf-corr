"""Independent minimal DCC cross-check for ``etfcorr.correlation.dcc`` (step 4.4).

``MinimalDcc`` below is written for this test only and shares no code with the module: a plain
Python loop over days, ``np.linalg.inv``/``slogdet`` per day, a different parametrisation
(logistic maps keep ``a, b > 0`` and ``a + b < 1`` without constraints) and a different optimiser
(Nelder-Mead instead of SLSQP). Both are run on the same standardised shocks; the optimum of the
stage-2 likelihood, and the filtered correlations at that optimum, must agree. Stationarity note:
the symmetric model only (the asymmetric one is checked against explicit loops in ``test_dcc.py``).
"""

import numpy as np
import pytest
from scipy.optimize import minimize

from etfcorr.correlation import dcc

RB3 = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])


class MinimalDcc:
    """Engle (2002) correlation-targeting DCC(1,1) stage 2, loops only (Q_0 = Qbar)."""

    def __init__(self, z: np.ndarray) -> None:
        self.z = z
        self.t, self.n = z.shape
        self.qbar = z.T @ z / self.t

    def paths(self, a: float, b: float) -> list[np.ndarray]:
        q = self.qbar.copy()
        out = []
        for t in range(self.t):
            d = np.diag(1.0 / np.sqrt(np.diag(q)))
            out.append(d @ q @ d)
            q = (1 - a - b) * self.qbar + a * np.outer(self.z[t], self.z[t]) + b * q
        return out

    def loglik(self, a: float, b: float) -> float:
        total = 0.0
        for t, r in enumerate(self.paths(a, b)):
            _, logdet = np.linalg.slogdet(r)
            zt = self.z[t]
            total += -0.5 * (logdet + zt @ np.linalg.inv(r) @ zt - zt @ zt)
        return float(total)

    @staticmethod
    def _ab(u: np.ndarray) -> tuple[float, float]:
        s = 1.0 / (1.0 + np.exp(-u[0]))  # a + b in (0, 1)
        w = 1.0 / (1.0 + np.exp(-u[1]))  # a / (a + b) in (0, 1)
        return float(s * w), float(s * (1.0 - w))

    def fit(self) -> tuple[float, float, float]:
        res = minimize(
            lambda u: -self.loglik(*self._ab(u)),
            x0=np.array([3.5, -2.5]),  # a + b about 0.97, a share about 0.08
            method="Nelder-Mead",
            options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 2000},
        )
        a, b = self._ab(res.x)
        return a, b, -float(res.fun)


@pytest.fixture(scope="module")
def shocks() -> np.ndarray:
    sim = dcc.simulate_dcc(1200, RB3, a=0.05, b=0.90, seed=21)
    # unit-variance standardisation by the true conditional variances, as stage 2 sees it
    return np.asarray(sim.z, dtype=float)


def test_minimal_dcc_and_the_module_agree_on_the_likelihood_surface(shocks: np.ndarray) -> None:
    ref = MinimalDcc(shocks)
    qbar = ref.qbar
    for a, b in ((0.0, 0.0), (0.03, 0.95), (0.08, 0.80), (0.2, 0.5)):
        assert dcc.dcc_loglik(shocks, qbar, None, a, b) == pytest.approx(
            ref.loglik(a, b), rel=1e-9, abs=1e-8
        )


def test_minimal_dcc_and_the_module_find_the_same_optimum(shocks: np.ndarray) -> None:
    ref = MinimalDcc(shocks)
    a_ref, b_ref, ll_ref = ref.fit()
    fit = dcc.fit_dcc(shocks)
    assert fit.loglik == pytest.approx(ll_ref, abs=1e-3)
    assert fit.a == pytest.approx(a_ref, abs=5e-3)
    assert fit.b == pytest.approx(b_ref, abs=2e-2)
    # the module never ends below the independent optimiser
    assert fit.loglik >= ll_ref - 1e-4


def test_filtered_correlations_at_the_optimum_agree(shocks: np.ndarray) -> None:
    fit = dcc.fit_dcc(shocks)
    ref = np.array(MinimalDcc(shocks).paths(fit.a, fit.b))
    mine = dcc.dcc_correlations(dcc.dcc_q_path(shocks, fit.qbar, None, fit.a, fit.b))
    np.testing.assert_allclose(mine[: len(shocks)], ref, atol=1e-10)
