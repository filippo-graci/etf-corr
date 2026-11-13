"""Unit tests for ``etfcorr.evaluation.dm_mcs`` (step 6.2): known answers and planted truth."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from etfcorr.evaluation import dm_mcs as D

INDEX = pd.bdate_range("2021-01-01", periods=400, name="date")


def series(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=INDEX[: len(values)])


def ma_noise(rng: np.random.Generator, n: int, order: int) -> np.ndarray:
    """Moving average of ``order + 1`` white-noise terms: autocorrelated up to lag ``order``."""
    e = rng.standard_normal(n + order)
    return np.convolve(e, np.ones(order + 1), mode="valid") / np.sqrt(order + 1)


# ---- long-run variance -----------------------------------------------------------------


def test_autocovariances_match_the_direct_formula() -> None:
    x = np.array([1.0, 3.0, 2.0, 6.0, 3.0])
    c = x - x.mean()
    got = D.autocovariances(x, 2)
    assert got == pytest.approx([(c @ c) / 5, (c[1:] @ c[:-1]) / 5, (c[2:] @ c[:-2]) / 5])
    assert len(D.autocovariances(x, 99)) == 5  # capped at n - 1 lags


def test_long_run_variance_kernels_and_the_negative_fallback() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    gamma = D.autocovariances(x, 3)
    value, kernel = D.long_run_variance(x, 3)
    assert kernel == "rectangular" and value == pytest.approx(gamma[0] + 2 * gamma[1:].sum())
    bartlett, kernel = D.long_run_variance(x, 3, "bartlett")
    weights = np.array([0.75, 0.5, 0.25])
    assert kernel == "bartlett" and bartlett == pytest.approx(gamma[0] + 2 * weights @ gamma[1:])
    alternating = np.tile([1.0, -1.0], 100)  # gamma_1 = -gamma_0: rectangular estimate < 0
    assert alternating.var() + 2 * D.autocovariances(alternating, 1)[1] < 0
    value, kernel = D.long_run_variance(alternating, 1)
    assert kernel == "bartlett" and value > 0
    with pytest.raises(D.ComparisonError):
        D.long_run_variance(x, -1)
    with pytest.raises(D.ComparisonError):
        D.long_run_variance(x, 1, "parzen")  # type: ignore[arg-type]


# ---- Diebold-Mariano -------------------------------------------------------------------


def test_dm_at_horizon_one_is_exactly_the_paired_t_test() -> None:
    rng = np.random.default_rng(1)
    a = series(rng.standard_normal(200) ** 2)
    b = series(rng.standard_normal(200) ** 2 * 1.3)
    result = D.diebold_mariano(a, b, 1)
    reference = stats.ttest_1samp((a - b).to_numpy(), 0.0)
    assert result.statistic == pytest.approx(reference.statistic)
    assert result.p_value == pytest.approx(reference.pvalue)
    assert result.n == 200 and result.kernel == "rectangular"
    assert result.mean_diff == pytest.approx(float((a - b).mean()))


def test_dm_at_a_longer_horizon_matches_the_formula() -> None:
    rng = np.random.default_rng(2)
    d = ma_noise(rng, 300, 4) + 0.2
    a, b = series(d), series(np.zeros(300))
    h, n = 5, 300
    gamma = D.autocovariances(d, h - 1)
    omega = gamma[0] + 2 * gamma[1:].sum()
    expected = d.mean() / np.sqrt(omega / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    result = D.diebold_mariano(a, b, h)
    assert result.statistic == pytest.approx(expected)
    assert result.p_value == pytest.approx(2 * stats.t(n - 1).sf(abs(expected)))
    assert result.long_run_variance == pytest.approx(omega)


def test_dm_alternatives_are_consistent() -> None:
    rng = np.random.default_rng(3)
    a = series(rng.standard_normal(300) ** 2)
    b = series(rng.standard_normal(300) ** 2 + 0.5)  # a is clearly better
    less = D.diebold_mariano(a, b, 1, alternative="less")
    greater = D.diebold_mariano(a, b, 1, alternative="greater")
    two = D.diebold_mariano(a, b, 1)
    assert less.statistic < 0 and less.p_value < 0.001 and greater.p_value > 0.999
    assert two.p_value == pytest.approx(2 * less.p_value)
    swapped = D.diebold_mariano(b, a, 1, alternative="greater")
    assert swapped.statistic == pytest.approx(-less.statistic)
    assert swapped.p_value == pytest.approx(less.p_value)


def test_dm_uses_common_dates_and_handles_degenerate_input() -> None:
    rng = np.random.default_rng(4)
    a = series(rng.standard_normal(100) ** 2)
    b = series(rng.standard_normal(100) ** 2)
    b.iloc[:10] = np.nan
    assert D.diebold_mariano(a, b, 1).n == 90
    same = D.diebold_mariano(a, a, 3)
    assert np.isnan(same.statistic) and np.isnan(same.p_value) and same.mean_diff == 0.0
    with pytest.raises(D.ComparisonError, match="too few"):
        D.diebold_mariano(a.iloc[:5], b.iloc[:5], 21)
    with pytest.raises(D.ComparisonError, match="horizon"):
        D.diebold_mariano(a, b, 0)
    with pytest.raises(D.ComparisonError, match="alternative"):
        D.diebold_mariano(a, b, 1, alternative="both")  # type: ignore[arg-type]


def test_dm_rejects_about_as_often_as_its_level_on_white_noise_differentials() -> None:
    """Size at h = 1, n = 200 over 400 simulated nulls (binomial standard error about 0.011)."""
    rng = np.random.default_rng(5)
    zero = series(np.zeros(200))
    rejections = sum(
        D.diebold_mariano(series(rng.standard_normal(200)), zero, 1).p_value < 0.05
        for _ in range(400)
    )
    assert 0.02 <= rejections / 400 <= 0.09


# ---- Clark-West ------------------------------------------------------------------------


def test_clark_west_matches_its_formula() -> None:
    rng = np.random.default_rng(6)
    y = series(rng.standard_normal(250))
    small = series(np.zeros(250))
    large = series(0.3 * y.to_numpy() + 0.5 * rng.standard_normal(250))
    adjusted = (y - small) ** 2 - ((y - large) ** 2 - (small - large) ** 2)
    expected = adjusted.mean() / np.sqrt(adjusted.var(ddof=0) / 250)
    result = D.clark_west(y, small, large, 1)
    assert result.statistic == pytest.approx(expected)
    assert result.p_value == pytest.approx(stats.norm.sf(expected))
    assert result.mean_adjusted_diff == pytest.approx(float(adjusted.mean()))


def test_clark_west_finds_a_true_larger_model_that_plain_mse_understates() -> None:
    """y = 0.25 x + noise. The larger model's forecast is 0.25 x plus estimation noise, so its
    raw squared error is barely below the small model's, while the adjusted difference is not."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(400)
    y = series(0.25 * x + rng.standard_normal(400))
    small = series(np.zeros(400))
    large = series(0.25 * x + 0.2 * rng.standard_normal(400))
    cw = D.clark_west(y, small, large, 1)
    dm = D.diebold_mariano((y - large) ** 2, (y - small) ** 2, 1, alternative="less")
    assert cw.p_value < 0.01 and cw.statistic > 0
    assert cw.p_value < dm.p_value  # the adjustment adds power for nested models


def test_clark_west_does_not_reject_when_the_extra_term_is_noise() -> None:
    rng = np.random.default_rng(8)
    rejections = 0
    for _ in range(200):
        y = series(rng.standard_normal(200))
        large = series(0.3 * rng.standard_normal(200))  # unrelated to y
        rejections += D.clark_west(y, series(np.zeros(200)), large, 1).p_value < 0.05
    assert rejections / 200 <= 0.10
    with pytest.raises(D.ComparisonError):
        D.clark_west(y.iloc[:3], y.iloc[:3], y.iloc[:3], 5)


# ---- Model Confidence Set --------------------------------------------------------------


def losses_frame(seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((400, 1)) ** 2
    return pd.DataFrame(
        {
            "good_a": base[:, 0] + 0.05 * rng.standard_normal(400),
            "good_b": base[:, 0] + 0.05 * rng.standard_normal(400),
            "bad": base[:, 0] + 1.0 + 0.05 * rng.standard_normal(400),
        },
        index=INDEX,
    )


def test_mcs_drops_the_clearly_worse_model_and_keeps_the_equal_ones() -> None:
    out = D.model_confidence_set(losses_frame(), size=0.10, block_size=5, reps=500, seed=0)
    assert list(out.columns) == ["p_value", "included", "mean_loss"]
    assert not out.loc["bad", "included"] and out.loc["bad", "p_value"] < 0.01
    assert out.loc["good_a", "included"] and out.loc["good_b", "included"]
    assert out["p_value"].max() == pytest.approx(1.0)  # the best model has p-value 1
    again = D.model_confidence_set(losses_frame(), size=0.10, block_size=5, reps=500, seed=0)
    pd.testing.assert_frame_equal(out, again)


def test_mcs_direction_flag_and_bad_input() -> None:
    scores = -losses_frame()  # as scores, "bad" is the lowest
    out = D.model_confidence_set(
        scores, size=0.10, block_size=5, reps=300, seed=0, higher_is_better=True
    )
    assert not out.loc["bad", "included"] and out.loc["good_a", "included"]
    with pytest.raises(D.ComparisonError):
        D.model_confidence_set(losses_frame()[["bad"]], block_size=5)
    with pytest.raises(D.ComparisonError):
        D.model_confidence_set(losses_frame().iloc[:5], block_size=10)


@pytest.mark.parametrize("h", [1, 5, 21])
def test_dm_with_the_bartlett_kernel_agrees_with_statsmodels(h: int) -> None:
    """Independent cross-check: statsmodels' HLN-adjusted DM test uses a Newey-West variance,
    which is this module's Bartlett path with the same number of lags."""
    from statsmodels.tsa.stattools import diebold_mariano_test

    rng = np.random.default_rng(10)
    n = 300
    y = rng.standard_normal(n)
    fa = 0.3 * y + 0.5 * rng.standard_normal(n)
    fb = 0.6 * rng.standard_normal(n)
    reference = diebold_mariano_test(y, fa, fb, lags=h - 1, harvey_adj=True, horizon=h)
    mine = D.diebold_mariano(series((y - fa) ** 2), series((y - fb) ** 2), h, kernel="bartlett")
    assert mine.statistic == pytest.approx(float(reference.statistic))
    assert mine.p_value == pytest.approx(float(reference.pvalue))
    assert mine.kernel == "bartlett"
