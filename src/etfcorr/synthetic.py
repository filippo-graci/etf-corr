"""Synthetic return panels with a *known* correlation matrix (for tests and CI fixtures).

Model (one market factor, one factor per sector, idiosyncratic noise; all standard normal and
independent over time and across factors):

.. math:: z_{it} = b_m f_{m,t} + b_s f_{s(i),t} + \\sqrt{1 - b_m^2 - b_s^2}\\, e_{it},
          \\qquad r_{it} = \\sigma_i z_{it}

so every :math:`z_i` has unit variance and the population correlation is
:math:`b_m^2 + b_s^2` within a sector and :math:`b_m^2` across sectors. Setting both loadings to
zero gives a pure-noise panel (all off-diagonal correlations are exactly zero), which is the null
for the "synthetic noise gives about zero skill" tests.

Assumptions and limits: returns are i.i.d. Gaussian, so there is no volatility clustering, fat
tail or regime; the panel is only meant to have known structure and no look-ahead hazards, not
to look like ETFs. Values are log-return-like numbers, not prices. The generator has no
external dependency and is not adapted from third-party code.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticPanel:
    """Simulated daily log returns together with the population truth they were drawn from."""

    returns: pd.DataFrame
    true_corr: np.ndarray
    sectors: tuple[str, ...]  # sector label of each column, same order as the columns


def true_correlation(
    sector_sizes: tuple[int, ...], market_loading: float, sector_loading: float
) -> np.ndarray:
    """Population correlation matrix implied by the factor model in the module docstring."""
    labels = np.repeat(np.arange(len(sector_sizes)), sector_sizes)
    same = labels[:, None] == labels[None, :]
    corr = np.where(same, market_loading**2 + sector_loading**2, market_loading**2)
    np.fill_diagonal(corr, 1.0)
    return np.asarray(corr, dtype=float)


def make_factor_returns(
    n_days: int = 1500,
    sector_sizes: tuple[int, ...] = (3, 3, 3),
    *,
    market_loading: float = 0.5,
    sector_loading: float = 0.5,
    vol_range: tuple[float, float] = (0.008, 0.020),
    start: str = "2016-01-04",
    seed: int = 0,
) -> SyntheticPanel:
    """Draw ``n_days`` business days of returns for ``sum(sector_sizes)`` assets.

    ``vol_range`` sets the per-asset daily volatilities (evenly spaced); they scale the columns
    and do not change correlations. Tickers are named ``S<sector>_<k>``.
    """
    if n_days < 2:
        raise ValueError(f"n_days must be >= 2, got {n_days}")
    if not sector_sizes or any(s < 1 for s in sector_sizes):
        raise ValueError(f"sector_sizes must be positive integers, got {sector_sizes}")
    if market_loading < 0 or sector_loading < 0:
        raise ValueError("loadings must be non-negative")
    if market_loading**2 + sector_loading**2 >= 1.0:
        raise ValueError("market_loading**2 + sector_loading**2 must be < 1 (unit variance)")
    lo, hi = vol_range
    if not 0 < lo <= hi:
        raise ValueError(f"vol_range must satisfy 0 < low <= high, got {vol_range}")

    rng = np.random.default_rng(seed)
    n_assets = sum(sector_sizes)
    labels = np.repeat(np.arange(len(sector_sizes)), sector_sizes)
    idio = np.sqrt(1.0 - market_loading**2 - sector_loading**2)

    market = rng.standard_normal((n_days, 1))
    sector = rng.standard_normal((n_days, len(sector_sizes)))
    noise = rng.standard_normal((n_days, n_assets))
    z = market_loading * market + sector_loading * sector[:, labels] + idio * noise
    vols = np.linspace(lo, hi, n_assets)

    columns = [f"S{s}_{k}" for s, size in enumerate(sector_sizes) for k in range(size)]
    index = pd.bdate_range(start, periods=n_days, name="date")
    returns = pd.DataFrame(z * vols, index=index, columns=columns)
    return SyntheticPanel(
        returns=returns,
        true_corr=true_correlation(sector_sizes, market_loading, sector_loading),
        sectors=tuple(f"sector{s}" for s in labels),
    )


def make_noise_returns(
    n_days: int = 1500,
    n_assets: int = 9,
    *,
    seed: int = 0,
    start: str = "2016-01-04",
) -> SyntheticPanel:
    """Independent noise panel: the true correlation is the identity (null for skill tests)."""
    return make_factor_returns(
        n_days,
        (n_assets,),
        market_loading=0.0,
        sector_loading=0.0,
        start=start,
        seed=seed,
    )
