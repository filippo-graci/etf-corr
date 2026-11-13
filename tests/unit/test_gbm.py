"""Unit tests for ``etfcorr.models.gbm`` (step 6.3), on small planted tables with tiny budgets."""

import numpy as np
import pandas as pd
import pytest

from etfcorr.models import base as B
from etfcorr.models import gbm as G
from etfcorr.models.linear import PenalisedLinear
from etfcorr.splits import Fold

SPACE = {
    "num_leaves": (2, 8),
    "min_data_in_leaf": (5, 20),
    "learning_rate": (0.05, 0.2),
    "n_estimators": (30, 60),
}


def table(n: int = 500, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """y = 2 * sign(a) + |b| + noise: no linear signal in either feature; c is irrelevant."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=n, name="date")
    X = pd.DataFrame(rng.standard_normal((n, 3)), index=index, columns=list("abc"))
    y = 2.0 * np.sign(X["a"]) + X["b"].abs() + 0.2 * rng.standard_normal(n)
    return X, y.rename("y")


def make(**changes: object) -> G.GBMForecaster:
    kw: dict[str, object] = {"n_trials": 4, "seeds": (0, 1), "horizon": 1}
    kw.update(changes)
    return G.GBMForecaster(SPACE, **kw)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def fitted() -> tuple[G.GBMForecaster, pd.DataFrame, pd.Series]:
    X, y = table()
    return make().fit(X.iloc[:400], y.iloc[:400]), X, y


def test_gbm_is_a_registered_forecaster() -> None:
    model = B.make_forecaster("gbm", search_space=SPACE, n_trials=2, seeds=[0], horizon=1)
    assert isinstance(model, B.Forecaster) and model.name == "lightgbm"


def test_gbm_learns_a_nonlinear_signal_that_ridge_cannot(
    fitted: tuple[G.GBMForecaster, pd.DataFrame, pd.Series],
) -> None:
    model, X, y = fitted
    test = slice(400, 500)
    gbm_error = float(((model.predict(X.iloc[test]) - y.iloc[test]) ** 2).mean())
    ridge = PenalisedLinear("ridge", (0.01, 0.1), horizon=1).fit(X.iloc[:400], y.iloc[:400])
    ridge_error = float(((ridge.predict(X.iloc[test]) - y.iloc[test]) ** 2).mean())
    assert gbm_error < 0.3 * ridge_error


def test_every_trial_is_logged_and_the_best_one_is_inside_the_search_space(
    fitted: tuple[G.GBMForecaster, pd.DataFrame, pd.Series],
) -> None:
    model = fitted[0]
    assert len(model.trials_) == 4 and model.params["n_trials"] == 4
    assert model.cv_score_ == pytest.approx(min(t["score"] for t in model.trials_))
    for key, (low, high) in SPACE.items():
        assert low <= model.best_params_[key] <= high
    assert model.params["tuned"] is True and model.params["seeds"] == [0, 1]


def test_forecast_is_the_mean_over_seeds() -> None:
    """With column sampling the seeds give different forecasts, so the mean is none of them."""
    X, y = table(n=300)
    space = {**SPACE, "feature_fraction": (0.4, 0.6)}
    model = G.GBMForecaster(space, n_trials=2, seeds=(0, 1, 2), horizon=1).fit(X, y)
    by_seed = model.predict_by_seed(X)
    assert list(by_seed.columns) == ["seed_0", "seed_1", "seed_2"]
    assert np.allclose(model.predict(X), by_seed.mean(axis=1))
    assert not np.allclose(model.predict(X), by_seed["seed_0"])


def test_gbm_is_reproducible_and_the_tuning_seed_matters() -> None:
    X, y = table(n=300)
    a = make().fit(X, y)
    b = make().fit(X, y)
    assert a.trials_ == b.trials_ and a.predict(X).equals(b.predict(X))
    c = make(tune_seed=7).fit(X, y)
    assert c.trials_ != a.trials_


def test_seeds_differ_only_when_the_model_has_randomness() -> None:
    X, y = table(n=300)
    plain = make().fit(X, y).predict_by_seed(X)
    assert np.allclose(plain["seed_0"], plain["seed_1"])  # no column sampling: seeds are idle
    space = {**SPACE, "feature_fraction": (0.4, 0.6)}
    sampled = G.GBMForecaster(space, n_trials=2, seeds=(0, 1), horizon=1).fit(X, y)
    by_seed = sampled.predict_by_seed(X)
    assert not np.allclose(by_seed["seed_0"], by_seed["seed_1"])


def test_params_override_skips_the_tuning() -> None:
    X, y = table(n=300)
    tuned = make(seeds=(3,)).fit(X, y)
    fixed = make(seeds=(3,), params_override=tuned.best_params_).fit(X, y)
    assert fixed.trials_ == [] and fixed.params["tuned"] is False
    assert np.allclose(fixed.predict(X), tuned.predict(X))


def test_inner_folds_use_the_target_horizon_and_the_embargo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, int, int]] = []
    real = G.positional_walk_forward

    def spy(n: int, n_folds: int, *, horizon: int, embargo: int = 0) -> list[Fold]:
        calls.append((n, n_folds, horizon, embargo))
        return real(n, n_folds, horizon=horizon, embargo=embargo)

    monkeypatch.setattr(G, "positional_walk_forward", spy)
    X, y = table(n=300)
    make(n_trials=1, horizon=21, embargo=5, inner_folds=2).fit(X, y)
    assert calls == [(300, 2, 21, 5)]


def test_columns_restrict_the_model_and_contributions_rank_the_signal(
    fitted: tuple[G.GBMForecaster, pd.DataFrame, pd.Series],
) -> None:
    model, X, y = fitted
    shap = model.mean_abs_contributions(X.iloc[:400])
    assert list(shap.index[:2]) == ["a", "b"] and shap["c"] < 0.2 * shap["b"]
    only_c = make(columns=["c"]).fit(X.iloc[:400], y.iloc[:400])
    assert list(only_c.mean_abs_contributions(X).index) == ["c"]
    assert abs(np.corrcoef(only_c.predict(X.iloc[400:]), y.iloc[400:])[0, 1]) < 0.3


def test_gbm_rejects_bad_requests() -> None:
    X, y = table(n=100)
    with pytest.raises(B.ModelError, match="unknown search-space"):
        G.GBMForecaster({"max_bin": (2, 3)}, n_trials=1, seeds=(0,), horizon=1)
    with pytest.raises(B.ModelError, match="low <= high"):
        G.GBMForecaster({"num_leaves": (8, 2)}, n_trials=1, seeds=(0,), horizon=1)
    with pytest.raises(B.ModelError, match="at least one seed"):
        G.GBMForecaster(SPACE, n_trials=1, seeds=(), horizon=1)
    with pytest.raises(B.ModelError, match="before fit"):
        make().predict(X)
    with pytest.raises(B.ModelError, match="too short"):
        make(horizon=63).fit(X.iloc[:60], y.iloc[:60])
    with pytest.raises(B.ModelError, match="not in"):
        make(columns=["zzz"]).fit(X, y)
