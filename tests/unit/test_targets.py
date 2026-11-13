"""Unit tests for ``etfcorr.targets`` (step 5.2): each target against a direct computation."""

import numpy as np
import pandas as pd
import pytest

from etfcorr import features as F
from etfcorr import targets as T
from etfcorr.synthetic import make_factor_returns

PANEL = make_factor_returns(300, (2, 2, 2), seed=5)
R = PANEL.returns
SECTORS = dict(zip(R.columns, PANEL.sectors, strict=True))
MASK = F.sector_pair_mask(list(R.columns), SECTORS)


@pytest.mark.parametrize("h", [5, 21, 63])
def test_forward_correlation_is_the_trailing_statistic_read_h_rows_later(h: int) -> None:
    target = T.forward_average_correlation(R, h, MASK)
    trailing = F.trailing_average_correlation(R, h, MASK)
    assert target.index.equals(R.index)
    assert target.iloc[-h:].isna().all() and target.iloc[:-h].notna().all()
    assert np.allclose(target.iloc[:-h].to_numpy(), trailing.iloc[h:].to_numpy())


def test_forward_correlations_use_rows_t_plus_1_to_t_plus_h() -> None:
    out = T.forward_correlations(R, 21)
    x = R.to_numpy()
    for t in (0, 100, 278):
        assert np.allclose(out[t], np.corrcoef(x[t + 1 : t + 22], rowvar=False))
    assert np.isnan(out[279:]).all()


def test_forward_correlations_on_a_sample_shorter_than_the_horizon_are_all_nan() -> None:
    assert np.isnan(T.forward_correlations(R.iloc[:10], 21)).all()


@pytest.mark.parametrize("h", [1, 5, 21])
def test_forward_variance_and_return_match_direct_sums(h: int) -> None:
    x = R.to_numpy()
    var, ret = T.forward_variance(R, h), T.forward_return(R, h)
    t = 150
    assert np.allclose(var.iloc[t].to_numpy(), (x[t + 1 : t + 1 + h] ** 2).mean(axis=0))
    assert np.allclose(ret.iloc[t].to_numpy(), x[t + 1 : t + 1 + h].sum(axis=0))
    assert var.iloc[-h:].isna().all().all() and ret.iloc[-h:].isna().all().all()
    assert var.iloc[:-h].notna().all().all()


def test_forward_rank_is_the_cross_sectional_rank_of_the_forward_return() -> None:
    rank = T.forward_rank(R, 5)
    assert rank.iloc[:-5].min().min() == -0.5 and rank.iloc[:-5].max().max() == 0.5
    t = 40
    order = T.forward_return(R, 5).iloc[t].rank()
    assert np.allclose(rank.iloc[t].to_numpy(), ((order - 1) / 5 - 0.5).to_numpy())
    assert np.allclose(rank.iloc[:-5].sum(axis=1), 0.0)  # centred


def test_forward_regime_label_and_label_end_dates() -> None:
    labels = pd.Series(np.arange(300.0) % 3, index=R.index, name="regime")
    out = T.forward_regime_label(labels, 21)
    assert out.iloc[10] == labels.iloc[31] and out.iloc[-21:].isna().all()
    assert out.name == "fwd_regime_h21"
    ends = T.label_end_dates(R.index, 21)
    assert ends.iloc[10] == R.index[31] and ends.iloc[-21:].isna().all()


@pytest.mark.parametrize(
    "call",
    [
        lambda: T.forward_correlations(R, 0),
        lambda: T.forward_variance(R, 0),
        lambda: T.forward_return(R, -1),
        lambda: T.forward_regime_label(pd.Series([1.0], name="x"), 0),
        lambda: T.label_end_dates(R.index, 0),
    ],
)
def test_non_positive_horizons_are_rejected(call: object) -> None:
    with pytest.raises(T.TargetError):
        call()  # type: ignore[operator]
