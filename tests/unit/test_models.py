"""Unit tests for ``etfcorr.models`` (steps 5.4 and 5.5): interface, baselines, HAR, linear.

Small planted-truth tables with exact answers. The walk-forward driver's look-ahead tests are in
``tests/leakage/test_models_no_lookahead.py``.
"""

import numpy as np
import pandas as pd
import pytest

from etfcorr.features import UNIT
from etfcorr.models import base as B
from etfcorr.models.baselines import ColumnForecaster, Constant, HistoricalMean
from etfcorr.models.har import HAR
from etfcorr.models.linear import KINDS, PenalisedLinear
from etfcorr.splits import Fold

GRID = (0.003, 0.03, 0.3)


def table(n: int = 400, seed: int = 0, noise: float = 0.1) -> tuple[pd.DataFrame, pd.Series]:
    """y = 1 + 2 a - b + noise, with an irrelevant column c."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=n, name="date")
    X = pd.DataFrame(rng.standard_normal((n, 3)), index=index, columns=list("abc"))
    y = 1.0 + 2.0 * X["a"] - X["b"] + noise * rng.standard_normal(n)
    return X, y.rename("y")


def long_table(
    n: int = 120, units: tuple[str, ...] = ("u", "v", "w")
) -> tuple[pd.DataFrame, pd.Series]:
    """Per-unit level (10, 20, 30) plus slope 1 on x."""
    rng = np.random.default_rng(4)
    index = pd.MultiIndex.from_product(
        [pd.bdate_range("2018-01-01", periods=n, name="date"), list(units)], names=["date", UNIT]
    )
    X = pd.DataFrame({"x": rng.standard_normal(len(index))}, index=index)
    level = pd.Series(index.get_level_values(UNIT), index=index).map(
        {u: 10.0 * (k + 1) for k, u in enumerate(units)}
    )
    return X, (level + X["x"]).rename("y")


# ---- interface and registry ------------------------------------------------------------


def test_every_model_satisfies_the_forecaster_protocol() -> None:
    models = [
        ColumnForecaster("a"),
        HistoricalMean(),
        Constant(),
        HAR(["a"]),
        PenalisedLinear("ridge", GRID, horizon=1),
        B.PerUnit(lambda: HAR(["x"])),
    ]
    for model in models:
        assert isinstance(model, B.Forecaster) and isinstance(model.name, str) and model.name


def test_registry_builds_by_name_and_rejects_unknown_and_duplicate_names() -> None:
    assert {"column", "historical_mean", "constant", "har", "penalised_linear"} <= set(B.REGISTRY)
    model = B.make_forecaster("har", columns=["a", "b"])
    assert isinstance(model, HAR) and model.columns == ("a", "b")
    with pytest.raises(B.ModelError, match="unknown"):
        B.make_forecaster("arima")
    with pytest.raises(B.ModelError, match="already"):
        B.register("har")(HAR)


# ---- rung 0 ----------------------------------------------------------------------------


def test_column_forecaster_reads_the_column_with_its_sign() -> None:
    X, y = table()
    assert ColumnForecaster("a").fit(X, y).predict(X).equals(X["a"])
    assert ColumnForecaster("a", sign=-1.0).predict(X).equals(-X["a"])
    assert ColumnForecaster("a", name="persistence").name == "persistence"
    with pytest.raises(B.ModelError):
        ColumnForecaster("zzz").fit(X, y)


def test_historical_mean_is_the_training_mean() -> None:
    X, y = table()
    model = HistoricalMean().fit(X.iloc[:100], y.iloc[:100])
    pred = model.predict(X.iloc[100:])
    assert np.allclose(pred, y.iloc[:100].mean()) and pred.index.equals(X.index[100:])
    assert model.params["mean"] == pytest.approx(y.iloc[:100].mean())


def test_historical_mean_is_per_unit_with_a_pooled_fallback() -> None:
    X, y = long_table()
    seen = X.index.get_level_values(UNIT) != "w"
    model = HistoricalMean().fit(X[seen], y[seen])
    pred = model.predict(X)
    by_unit = pred.groupby(level=UNIT).first()
    assert by_unit["u"] == pytest.approx(y[seen].xs("u", level=UNIT).mean())
    assert by_unit["v"] == pytest.approx(y[seen].xs("v", level=UNIT).mean())
    assert by_unit["w"] == pytest.approx(y[seen].mean())  # never seen in training


def test_constant_predicts_its_value() -> None:
    X, y = table()
    assert (Constant(0.0).fit(X, y).predict(X) == 0.0).all()


# ---- HAR -------------------------------------------------------------------------------


def test_har_recovers_planted_coefficients_exactly_without_noise() -> None:
    X, y = table(noise=0.0)
    model = HAR(["a", "b", "c"]).fit(X.iloc[:200], y.iloc[:200])
    params = model.params
    assert params["intercept"] == pytest.approx(1.0)
    assert (params["a"], params["b"]) == (pytest.approx(2.0), pytest.approx(-1.0))
    assert params["c"] == pytest.approx(0.0, abs=1e-10)
    assert np.allclose(model.predict(X.iloc[200:]), y.iloc[200:])


def test_har_uses_only_its_columns_and_rejects_bad_calls() -> None:
    X, y = table()
    assert set(HAR(["a"]).fit(X, y).params) == {"intercept", "a"}
    with pytest.raises(B.ModelError, match="before fit"):
        HAR(["a"]).predict(X)
    with pytest.raises(B.ModelError, match="cannot fit"):
        HAR(["a", "b", "c"]).fit(X.iloc[:4], y.iloc[:4])
    with pytest.raises(B.ModelError, match="not in"):
        HAR(["zzz"]).fit(X, y)
    with pytest.raises(B.ModelError):
        HAR([])
    with pytest.raises(B.ModelError, match="same index"):
        HAR(["a"]).fit(X, y.iloc[1:])


def test_per_unit_fits_one_model_per_unit() -> None:
    X, y = long_table()
    model = B.PerUnit(lambda: HAR(["x"])).fit(X, y)
    assert model.name == "har_per_unit"
    intercepts = {u: p["intercept"] for u, p in model.params.items()}  # type: ignore[index]
    assert intercepts == {
        "u": pytest.approx(10.0),
        "v": pytest.approx(20.0),
        "w": pytest.approx(30.0),
    }
    assert np.allclose(model.predict(X), y)
    pooled = HAR(["x"]).fit(X, y)
    assert pooled.params["intercept"] == pytest.approx(20.0, abs=0.5)  # pooling loses the levels


# ---- penalised linear ------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_each_kind_learns_a_strong_linear_signal(kind: str) -> None:
    X, y = table(n=500)
    grid = (1, 2, 3) if kind == "pcr" else GRID
    model = PenalisedLinear(kind, grid, horizon=1).fit(X.iloc[:400], y.iloc[:400])  # type: ignore[arg-type]
    pred = model.predict(X.iloc[400:])
    assert np.corrcoef(pred, y.iloc[400:])[0, 1] > 0.95
    params = model.params
    assert params["chosen"] in grid and params["n_trials"] == 3
    assert set(params["cv_scores"]) == set(grid)  # type: ignore[call-overload]


def test_penalised_linear_is_deterministic() -> None:
    X, y = table()
    a = PenalisedLinear("lasso", GRID, horizon=5, embargo=2).fit(X, y).predict(X)
    b = PenalisedLinear("lasso", GRID, horizon=5, embargo=2).fit(X, y).predict(X)
    assert a.equals(b)


def test_standardisation_is_fitted_on_the_training_rows() -> None:
    """Rescaling a feature leaves the forecasts unchanged, and so does changing the test rows'
    own distribution: only training statistics enter the scaler."""
    X, y = table()
    train, test = slice(0, 300), slice(300, 400)
    base = PenalisedLinear("ridge", GRID, horizon=1).fit(X.iloc[train], y.iloc[train])
    scaled = X.assign(a=X["a"] * 1000.0 + 7.0)
    other = PenalisedLinear("ridge", GRID, horizon=1).fit(scaled.iloc[train], y.iloc[train])
    assert np.allclose(base.predict(X.iloc[test]), other.predict(scaled.iloc[test]))
    one_row = base.predict(X.iloc[[350]]).iloc[0]
    assert one_row == pytest.approx(base.predict(X.iloc[test]).loc[X.index[350]])


def test_a_heavy_ridge_penalty_shrinks_to_the_training_mean() -> None:
    X, y = table()
    model = PenalisedLinear("ridge", (1e9,), horizon=1).fit(X.iloc[:300], y.iloc[:300])
    assert np.allclose(model.predict(X.iloc[300:]), y.iloc[:300].mean(), atol=1e-4)


def test_ties_go_to_the_stronger_penalty() -> None:
    """With a constant target every penalty scores the same, so the largest one is kept."""
    X, _ = table()
    y = pd.Series(1.0, index=X.index)
    assert PenalisedLinear("lasso", GRID, horizon=1).fit(X, y).params["chosen"] == max(GRID)
    assert PenalisedLinear("pcr", (3, 1, 2), horizon=1).fit(X, y).params["chosen"] == 1


def test_columns_restrict_the_model_and_pcr_caps_its_components() -> None:
    X, y = table()
    model = PenalisedLinear("ridge", GRID, horizon=1, columns=["c"]).fit(X, y)
    assert abs(np.corrcoef(model.predict(X), y)[0, 1]) < 0.3  # c carries no signal
    with pytest.raises(B.ModelError, match="not in"):
        PenalisedLinear("ridge", GRID, horizon=1, columns=["zzz"]).fit(X, y)
    capped = PenalisedLinear("pcr", (10,), horizon=1).fit(X, y)  # only 3 features
    assert np.corrcoef(capped.predict(X), y)[0, 1] > 0.95


def test_penalised_linear_tunes_on_dates_of_a_long_table() -> None:
    X, y = long_table(n=200)
    model = PenalisedLinear("ridge", GRID, horizon=5, embargo=1).fit(X, y)
    assert model.params["chosen"] in GRID and np.isfinite(model.predict(X)).all()


def test_penalised_linear_rejects_bad_requests() -> None:
    X, y = table()
    with pytest.raises(B.ModelError, match="unknown kind"):
        PenalisedLinear("ols", GRID, horizon=1)  # type: ignore[arg-type]
    with pytest.raises(B.ModelError, match="positive"):
        PenalisedLinear("ridge", (0.0,), horizon=1)
    with pytest.raises(B.ModelError, match="component counts"):
        PenalisedLinear("pcr", (1.5,), horizon=1)
    with pytest.raises(B.ModelError, match="too short"):
        PenalisedLinear("ridge", GRID, horizon=63).fit(X.iloc[:60], y.iloc[:60])
    with pytest.raises(B.ModelError, match="before fit"):
        PenalisedLinear("ridge", GRID, horizon=1).predict(X)


# ---- the walk-forward driver -----------------------------------------------------------


def folds_for(n: int = 400) -> list[Fold]:
    return [
        Fold(train=np.arange(0, 190, dtype=np.int64), test=np.arange(200, 300, dtype=np.int64)),
        Fold(train=np.arange(0, 290, dtype=np.int64), test=np.arange(300, n, dtype=np.int64)),
    ]


def test_driver_predicts_test_rows_only_with_a_fresh_model_per_fold() -> None:
    X, y = table()
    calendar = pd.DatetimeIndex(X.index)
    seen: list[tuple[int, object]] = []
    pred = B.walk_forward_predict(
        HistoricalMean, X, y, folds_for(), calendar, on_fit=lambda k, m: seen.append((k, m))
    )
    assert pred.iloc[:200].isna().all() and pred.iloc[200:].notna().all()
    assert np.allclose(pred.iloc[200:300], y.iloc[:190].mean())  # purged rows 190..199 unused
    assert np.allclose(pred.iloc[300:], y.iloc[:290].mean())
    assert [k for k, _ in seen] == [0, 1] and seen[0][1] is not seen[1][1]


def test_driver_skips_rows_with_missing_features_or_targets() -> None:
    X, y = table()
    X, y = X.copy(), y.copy()
    X.iloc[250, 0] = np.nan  # a test row without features: not predicted
    X.iloc[10, 1] = np.nan  # a training row without features: not trained on
    y.iloc[20] = np.nan  # a training row without a target: not trained on
    pred = B.walk_forward_predict(HistoricalMean, X, y, folds_for(), pd.DatetimeIndex(X.index))
    assert np.isnan(pred.iloc[250]) and pred.iloc[251] == pytest.approx(
        y.iloc[:190].drop(y.index[[10, 20]]).mean()
    )


def test_driver_handles_long_tables_by_date() -> None:
    X, y = long_table(n=120)
    calendar = pd.DatetimeIndex(X.index.get_level_values("date").unique())
    folds = [Fold(train=np.arange(0, 80, dtype=np.int64), test=np.arange(90, 120, dtype=np.int64))]
    pred = B.walk_forward_predict(lambda: B.PerUnit(lambda: HAR(["x"])), X, y, folds, calendar)
    tested = pred.notna()
    assert tested.sum() == 30 * 3 and np.allclose(pred[tested], y[tested])
    assert set(pred[tested].index.get_level_values("date")) == set(calendar[90:120])


def test_driver_rejects_a_fold_without_training_rows_and_misaligned_tables() -> None:
    X, y = table()
    calendar = pd.DatetimeIndex(X.index)
    with pytest.raises(B.ModelError, match="no training row"):
        B.walk_forward_predict(HistoricalMean, X, y * np.nan, folds_for(), calendar)
    with pytest.raises(B.ModelError, match="same index"):
        B.walk_forward_predict(HistoricalMean, X, y.iloc[1:], folds_for(), calendar)
    with pytest.raises(B.ModelError, match="at least one column"):
        B.walk_forward_predict(HistoricalMean, X.iloc[:, :0], y, folds_for(), calendar)


def test_driver_drops_training_rows_without_a_target_for_a_regression_model() -> None:
    """A missing target inside the training window must not reach the least-squares fit (the
    historical mean would hide it, because a mean skips missing values on its own)."""
    X, y = table(noise=0.0)
    y = y.copy()
    y.iloc[[20, 21, 150]] = np.nan
    pred = B.walk_forward_predict(
        lambda: HAR(["a", "b", "c"]), X, y, folds_for(), pd.DatetimeIndex(X.index)
    )
    clean = table(noise=0.0)[1]
    assert np.allclose(pred.iloc[200:], clean.iloc[200:])


def test_ridge_penalty_is_per_observation() -> None:
    """With one standardised feature and target, ridge with alpha = a * n has slope
    1 / (1 + a) times the least-squares slope: a = 1 halves it, whatever the sample size."""
    for n in (200, 400):
        X, y = table(n=n, noise=0.0)
        X, y = X[["a"]], (3.0 * X["a"]).rename("y")
        model = PenalisedLinear("ridge", (1.0,), horizon=1).fit(X, y)
        pred = model.predict(X)
        assert np.allclose(pred - y.mean(), 0.5 * (y - y.mean()))


def test_inner_folds_use_the_target_horizon_and_the_embargo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tuning folds must be purged with the label span of the target, not with 1."""
    from etfcorr.models import linear

    calls: list[tuple[int, int, int, int]] = []
    real = linear.positional_walk_forward

    def spy(n: int, n_folds: int, *, horizon: int, embargo: int = 0) -> list[Fold]:
        calls.append((n, n_folds, horizon, embargo))
        return real(n, n_folds, horizon=horizon, embargo=embargo)

    monkeypatch.setattr(linear, "positional_walk_forward", spy)
    X, y = table()
    PenalisedLinear("ridge", GRID, horizon=21, embargo=5, inner_folds=2).fit(X, y)
    assert calls == [(400, 2, 21, 5)]


# ---- lognormal bias correction (variant V1, step 6.0) ----------------------------------


def test_log_bias_correction_adds_half_the_training_residual_variance() -> None:
    """log(v) = a + noise with noise variance 0.64: exp(mean log) underestimates the mean of v
    by exp(0.32); the corrected forecast does not."""
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2018-01-01", periods=4000, name="date")
    X = pd.DataFrame({"x": np.zeros(4000)}, index=index)
    log_v = pd.Series(-9.0 + 0.8 * rng.standard_normal(4000), index=index, name="y")
    plain = HistoricalMean().fit(X.iloc[:3000], log_v.iloc[:3000])
    fixed = B.LogBiasCorrected(HistoricalMean).fit(X.iloc[:3000], log_v.iloc[:3000])
    assert fixed.shift_ == pytest.approx(0.32, abs=0.03)
    assert fixed.params["log_bias_shift"] == fixed.shift_
    shift = fixed.predict(X.iloc[3000:]) - plain.predict(X.iloc[3000:])
    assert np.allclose(shift, fixed.shift_)
    realised = float(np.exp(log_v.iloc[3000:]).mean())
    assert float(np.exp(fixed.predict(X.iloc[3000:])).iloc[0]) == pytest.approx(realised, rel=0.08)
    assert float(np.exp(plain.predict(X.iloc[3000:])).iloc[0]) < 0.8 * realised
    assert fixed.name == "historical_mean_bias_corrected"
    with pytest.raises(B.ModelError, match="before fit"):
        B.LogBiasCorrected(HistoricalMean).predict(X)


def test_log_bias_correction_is_per_unit_inside_per_unit() -> None:
    X, y = long_table()
    rng = np.random.default_rng(1)
    scale = pd.Series(X.index.get_level_values(UNIT), index=X.index).map(
        {"u": 0.1, "v": 1.0, "w": 2.0}
    )
    noisy = (y + scale * rng.standard_normal(len(y))).rename("y")
    model = B.PerUnit(lambda: B.LogBiasCorrected(lambda: HAR(["x"]))).fit(X, noisy)
    shifts = {u: p["log_bias_shift"] for u, p in model.params.items()}  # type: ignore[index]
    assert shifts["u"] < 0.01 < shifts["v"] < shifts["w"]
    assert shifts["w"] == pytest.approx(2.0, rel=0.3)  # 0.5 * 2.0**2


# ---- exploratory follow-up E4 (docs/FOLLOW_UPS.md item 4)


def test_best_column_is_chosen_on_the_rows_it_is_fitted_on() -> None:
    from etfcorr.models.baselines import BestColumn

    index = pd.bdate_range("2020-01-01", periods=40, name="date")
    y = pd.Series(np.arange(40.0), index=index)
    X = pd.DataFrame({"a": y + 1.0, "b": y.where(index >= index[20], y + 9.0)}, index=index)
    early = BestColumn(("a", "b")).fit(X.iloc[:20], y.iloc[:20])
    assert early.column_ == "a" and early.params == {"column": "a"}  # b is bad on these rows
    assert early.predict(X.iloc[20:]).equals(X["a"].iloc[20:])
    assert BestColumn(("a", "b")).fit(X.iloc[20:], y.iloc[20:]).column_ == "b"
    assert isinstance(early, B.Forecaster)
    with pytest.raises(B.ModelError):
        BestColumn(("a", "b")).predict(X)
    with pytest.raises(B.ModelError):
        BestColumn(("a", "zzz")).fit(X, y)
