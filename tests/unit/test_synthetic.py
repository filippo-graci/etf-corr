"""Tests for the synthetic panel generator: structure, known answer, determinism, failures."""

import numpy as np
import pandas as pd
import pytest

from etfcorr.synthetic import make_factor_returns, make_noise_returns, true_correlation


def test_true_correlation_known_values_and_structure() -> None:
    c = true_correlation((2, 2), market_loading=0.5, sector_loading=0.5)
    expected = np.array(
        [
            [1.0, 0.5, 0.25, 0.25],
            [0.5, 1.0, 0.25, 0.25],
            [0.25, 0.25, 1.0, 0.5],
            [0.25, 0.25, 0.5, 1.0],
        ]
    )
    np.testing.assert_allclose(c, expected)
    np.testing.assert_allclose(c, c.T)
    assert np.linalg.eigvalsh(c).min() > 0  # positive definite


def test_panel_shape_index_and_labels() -> None:
    p = make_factor_returns(300, (2, 3), seed=1)
    assert p.returns.shape == (300, 5)
    assert isinstance(p.returns.index, pd.DatetimeIndex)
    assert p.returns.index.is_monotonic_increasing
    assert p.returns.index.is_unique
    assert set(p.returns.index.dayofweek) <= {0, 1, 2, 3, 4}  # business days only
    assert p.returns.index.name == "date"
    assert list(p.returns.columns) == ["S0_0", "S0_1", "S1_0", "S1_1", "S1_2"]
    assert p.sectors == ("sector0", "sector0", "sector1", "sector1", "sector1")
    assert p.true_corr.shape == (5, 5)
    assert not p.returns.isna().any().any()


def test_determinism_and_seed_dependence() -> None:
    a = make_factor_returns(200, seed=3).returns
    b = make_factor_returns(200, seed=3).returns
    c = make_factor_returns(200, seed=4).returns
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sample_correlation_converges_to_truth(seed: int) -> None:
    """Known-answer: with T = 40000 the standard error is about 1/sqrt(T) = 0.005 per entry."""
    p = make_factor_returns(40_000, (3, 3, 3), seed=seed)
    sample = np.corrcoef(p.returns.to_numpy(), rowvar=False)
    assert np.abs(sample - p.true_corr).max() < 0.03  # about 6 standard errors


@pytest.mark.parametrize(("b_m", "b_s"), [(0.6, 0.3), (0.3, 0.7)])
def test_unequal_loadings_match_the_closed_form(b_m: float, b_s: float) -> None:
    """Within-sector correlation b_m^2 + b_s^2, across-sector b_m^2 (distinct loadings)."""
    p = make_factor_returns(40_000, (3, 3), market_loading=b_m, sector_loading=b_s, seed=0)
    assert p.true_corr[0, 1] == pytest.approx(b_m**2 + b_s**2)
    assert p.true_corr[0, 3] == pytest.approx(b_m**2)
    sample = np.corrcoef(p.returns.to_numpy(), rowvar=False)
    assert np.abs(sample - p.true_corr).max() < 0.03


def test_volatilities_follow_vol_range_and_do_not_change_correlation() -> None:
    p = make_factor_returns(40_000, (3, 3), vol_range=(0.01, 0.03), seed=0)
    stds = p.returns.std().to_numpy()
    np.testing.assert_allclose(stds, np.linspace(0.01, 0.03, 6), rtol=0.03)


def test_noise_panel_has_identity_truth() -> None:
    p = make_noise_returns(40_000, 6, seed=0)
    np.testing.assert_array_equal(p.true_corr, np.eye(6))
    off = np.corrcoef(p.returns.to_numpy(), rowvar=False) - np.eye(6)
    assert np.abs(off).max() < 0.03


def test_market_mode_is_the_first_eigenvector() -> None:
    """Sanity: with a strong market loading the leading eigenvector has same-sign entries."""
    p = make_factor_returns(20_000, (4, 4), market_loading=0.7, sector_loading=0.2, seed=0)
    vals, vecs = np.linalg.eigh(np.corrcoef(p.returns.to_numpy(), rowvar=False))
    lead = vecs[:, np.argmax(vals)]
    assert (lead > 0).all() or (lead < 0).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_days": 1},
        {"sector_sizes": ()},
        {"sector_sizes": (3, 0)},
        {"market_loading": -0.1},
        {"sector_loading": -0.1},
        {"market_loading": 0.8, "sector_loading": 0.7},
        {"market_loading": 1.0, "sector_loading": 0.0},
        {"vol_range": (0.0, 0.01)},
        {"vol_range": (0.02, 0.01)},
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_factor_returns(**kwargs)  # type: ignore[arg-type]
