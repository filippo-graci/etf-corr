"""Shared fixtures: synthetic panels with known population truth (no data files, no network)."""

import pytest

from etfcorr.synthetic import SyntheticPanel, make_factor_returns, make_noise_returns

SEED = 20260921


@pytest.fixture(scope="session")
def factor_panel() -> SyntheticPanel:
    """1500 business days, 3 sectors of 3 assets, corr 0.5 within sector and 0.25 across."""
    return make_factor_returns(1500, (3, 3, 3), seed=SEED)


@pytest.fixture(scope="session")
def noise_panel() -> SyntheticPanel:
    """1500 business days of independent noise (true correlation = identity)."""
    return make_noise_returns(1500, 9, seed=SEED)
