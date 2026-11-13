"""Leakage tests for ``etfcorr.features`` and ``etfcorr.targets`` (steps 5.2 and 5.3).

Every feature table is checked with both shared detectors (truncation, and replacing the future
by noise or by a permutation), at explicit cut points placed on and around the warm-up
boundaries of the trailing windows as well as inside the sample (standing lesson 2 in
``status/cross_cutting.md``: default fractions rarely land on a boundary). Deliberately leaky
twins must be *caught*, so a green run shows the detectors can see these leaks.

Targets are forward-looking by definition. Two things are checked for them instead: the
detectors do flag a target when it is treated as a feature (so a target that slips into a
feature table cannot pass unnoticed), and each target depends on the rows ``t + 1 .. t + h``
only (the label span that ``etfcorr.splits`` purges).

Mode choice: ``noise`` for return-like data; the VIX level must stay positive, so the table that
includes it uses ``shuffle`` (standing lesson 3).
"""

import numpy as np
import pandas as pd
import pytest

from etfcorr import features as F
from etfcorr import targets as T
from etfcorr.leakage import LeakageError, assert_future_invariant, assert_no_lookahead
from etfcorr.synthetic import make_factor_returns

N = 420
WINDOWS = (21, 63, 126)
CUTS = (21, 22, 63, 64, 126, 127, 200, 333, N - 1)
PANEL = make_factor_returns(N, (2, 2, 2), seed=3)
ASSETS = list(PANEL.returns.columns)
SECTORS = dict(zip(ASSETS, PANEL.sectors, strict=True))
MASK = F.sector_pair_mask(ASSETS, SECTORS)


def _data() -> pd.DataFrame:
    """Asset returns plus a market return and a positive VIX-like level, one frame."""
    rng = np.random.default_rng(8)
    data = PANEL.returns.copy()
    data["mkt"] = data[ASSETS].mean(axis=1)
    log_v = np.log(18.0) + np.cumsum(0.03 * rng.standard_normal(N)) * 0.3
    data["vix"] = np.exp(log_v)
    return data


def common(d: pd.DataFrame) -> pd.DataFrame:
    return F.build_features(d[ASSETS], SECTORS, market=d["mkt"], windows=WINDOWS)


def common_with_vix(d: pd.DataFrame) -> pd.DataFrame:
    return F.build_features(d[ASSETS], SECTORS, market=d["mkt"], vix=d["vix"], windows=WINDOWS)


def per_asset(d: pd.DataFrame) -> pd.DataFrame:
    return F.build_asset_features(d[ASSETS], common=common(d)[["log_rv_xs_w21"]])


def per_asset_ranked(d: pd.DataFrame) -> pd.DataFrame:
    return F.build_asset_features(d[ASSETS], rank=True)


def per_pair(d: pd.DataFrame) -> pd.DataFrame:
    return F.build_pair_features(d[ASSETS], windows=WINDOWS)


# ---- causal tables pass ----------------------------------------------------------------


@pytest.mark.parametrize("fn", [common, common_with_vix, per_asset, per_asset_ranked, per_pair])
def test_feature_tables_have_no_lookahead(fn: object) -> None:
    assert_no_lookahead(fn, _data(), CUTS)  # type: ignore[arg-type]


@pytest.mark.parametrize("fn", [common, per_asset, per_asset_ranked, per_pair])
def test_feature_tables_ignore_a_noisy_future(fn: object) -> None:
    assert_future_invariant(fn, _data(), CUTS[1:], mode="noise")  # type: ignore[arg-type]


def test_vix_features_ignore_a_shuffled_future() -> None:
    assert_future_invariant(common_with_vix, _data(), CUTS[1:], mode="shuffle")


def test_feature_tables_are_not_all_nan_on_the_checked_range() -> None:
    """Guard against a vacuous pass: every column holds real numbers on the compared rows
    (the slowest one, 12-1 momentum, needs 252 of the 420 rows to warm up)."""
    for fn in (common, per_asset, per_pair):
        assert (fn(_data()).notna().mean() > 0.3).all()


# ---- leaky twins are caught ------------------------------------------------------------


def centred_window(d: pd.DataFrame) -> pd.DataFrame:
    """Leak: the window is centred on t, so half of it is in the future."""
    corr = F.trailing_correlations(d[ASSETS], 21)
    centred = np.full_like(corr, np.nan)
    centred[:-10] = corr[10:]
    return pd.DataFrame({"z": F.fisher_z(F.masked_average(centred, MASK))}, index=d.index)


def full_sample_scaled(d: pd.DataFrame) -> pd.DataFrame:
    """Leak: a causal table standardised with the mean and deviation of the whole sample."""
    table = common(d)
    return (table - table.mean()) / table.std()


def backward_ewma(d: pd.DataFrame) -> pd.DataFrame:
    """Leak: the EWMA recursion run from the end of the sample backwards."""
    reverse = d[ASSETS].iloc[::-1].reset_index(drop=True)
    corr = F.ewma_correlations(reverse)[::-1]
    return pd.DataFrame({"z": F.fisher_z(F.masked_average(corr, MASK))}, index=d.index)


def momentum_without_the_skip_shift(d: pd.DataFrame) -> pd.DataFrame:
    """Leak: a negative shift (the skip applied in the wrong direction)."""
    return F.trailing_return(d[ASSETS], 60).shift(-F.MOMENTUM_SKIP)


LEAKY = [centred_window, full_sample_scaled, backward_ewma, momentum_without_the_skip_shift]


@pytest.mark.parametrize("fn", LEAKY)
def test_truncation_detector_catches_each_leaky_twin(fn: object) -> None:
    with pytest.raises(LeakageError):
        assert_no_lookahead(fn, _data(), CUTS)  # type: ignore[arg-type]


@pytest.mark.parametrize("fn", LEAKY)
def test_noisy_future_detector_catches_each_leaky_twin(fn: object) -> None:
    with pytest.raises(LeakageError):
        assert_future_invariant(fn, _data(), CUTS[1:], mode="noise")  # type: ignore[arg-type]


# ---- targets: flagged as features, and confined to their label span --------------------


def t1(d: pd.DataFrame) -> pd.Series:
    return T.forward_average_correlation(d[ASSETS], 21, MASK)


def t2(d: pd.DataFrame) -> pd.DataFrame:
    return T.forward_variance(d[ASSETS], 5)


def t3(d: pd.DataFrame) -> pd.DataFrame:
    return T.forward_rank(d[ASSETS], 5)


@pytest.mark.parametrize("fn", [t1, t2, t3])
def test_a_target_used_as_a_feature_is_flagged_by_both_detectors(fn: object) -> None:
    with pytest.raises(LeakageError):
        assert_no_lookahead(fn, _data(), CUTS)  # type: ignore[arg-type]
    with pytest.raises(LeakageError):
        assert_future_invariant(fn, _data(), CUTS[1:], mode="noise")  # type: ignore[arg-type]


@pytest.mark.parametrize(("fn", "h"), [(t1, 21), (t2, 5), (t3, 5)])
def test_a_target_depends_only_on_its_label_rows(fn: object, h: int) -> None:
    data = _data()
    t = 200
    base = fn(data).iloc[t]  # type: ignore[operator]
    rng = np.random.default_rng(1)

    outside = data.copy()
    rows = np.r_[0 : t + 1, t + h + 1 : N]  # everything except rows t+1 .. t+h
    outside.iloc[rows] = rng.standard_normal((len(rows), data.shape[1])) * 0.01
    assert np.allclose(np.asarray(fn(outside).iloc[t]), np.asarray(base))  # type: ignore[operator]

    for row in (t + 1, t + h):  # both ends of the label span matter
        inside = data.copy()
        inside.iloc[row] = rng.standard_normal(data.shape[1]) * 0.05
        changed = np.asarray(fn(inside).iloc[t])  # type: ignore[operator]
        assert not np.allclose(changed, np.asarray(base))


def test_the_persistence_feature_and_the_target_share_no_row() -> None:
    """The feature dated t ends at row t; the target dated t starts at row t + 1."""
    data = _data()
    t = 250
    changed = data.copy()
    changed.iloc[t + 1 :] = 0.123  # rewrite the whole future
    for name in ("z_cross_w21", "z_cross_ewma", "log_rv_xs_w5"):
        assert common(changed)[name].iloc[t] == common(data)[name].iloc[t]
    past_changed = data.copy()
    past_changed.iloc[: t + 1] = 0.0
    assert t1(past_changed).iloc[t] == pytest.approx(t1(data).iloc[t])
