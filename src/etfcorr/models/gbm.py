"""Step 6.3: rung 3, LightGBM with a fixed, logged tuning budget.

:class:`GBMForecaster` follows the same contract as the linear rung (``models.linear``):

* hyperparameters are chosen **inside the training window** on purged walk-forward folds
  (``splits.positional_walk_forward`` over the training dates, with the target's horizon and the
  embargo), by Optuna's TPE sampler with a fixed seed and a fixed number of trials
  (``config/experiment.yaml`` ``forecasting.gbm``); every trial is kept in ``trials_`` so the
  number of configurations tried is on record;
* the model is then refitted on the whole training window once per seed in ``seeds`` and the
  forecast is the **mean over seeds**; :meth:`predict_by_seed` gives the single-seed forecasts
  so that the spread across seeds can be reported next to the result;
* ``params_override`` skips the tuning and fits the given hyperparameters (used to re-run a
  single seed with the hyperparameters the tuned run chose in the same fold).

Trees do not need scaled inputs, so nothing is standardised; the tuning objective is the mean
squared error on the inner test folds, whatever the reporting loss of the target is.

Determinism: ``deterministic=True`` and ``force_row_wise=True`` with a fixed ``seed`` and
``n_jobs`` make LightGBM reproducible on one machine; results can differ in the last digits
across machines or LightGBM versions. The TPE sampler is seeded, so the trial sequence is fixed.

Limits: with daily overlapping targets a training window of about 1250 rows holds about 60
independent months at ``h = 21``; a tree ensemble can memorise that easily, which is why the
search space is deliberately small (few leaves, a floor on leaf size) and the budget is 20
trials. The inner folds select on a noisy score, so the chosen hyperparameters vary from refit
to refit. Feature attributions (:func:`mean_abs_contributions`, LightGBM's own TreeSHAP) describe
the fitted model, not the data-generating process. References (checked by search on 2026-10-11):
Ke et al. (2017), *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS 30;
Akiba et al. (2019), *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from etfcorr.models.base import ModelError, check_table, register, row_dates
from etfcorr.splits import SplitError, positional_walk_forward

INTEGER_PARAMS: tuple[str, ...] = ("num_leaves", "min_data_in_leaf", "n_estimators")
LOG_PARAMS: tuple[str, ...] = ("learning_rate", "lambda_l2")
KNOWN_PARAMS: tuple[str, ...] = (*INTEGER_PARAMS, *LOG_PARAMS, "feature_fraction")

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _regressor(params: Mapping[str, float], seed: int, n_jobs: int) -> lgb.LGBMRegressor:
    settings: dict[str, Any] = {
        k: (int(v) if k in INTEGER_PARAMS else float(v)) for k, v in params.items()
    }
    settings.update(
        objective="regression",
        seed=seed,
        n_jobs=n_jobs,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )
    return lgb.LGBMRegressor(**settings)


@register("gbm")
class GBMForecaster:
    """LightGBM regression tuned on purged inner folds, refitted per seed, averaged."""

    def __init__(
        self,
        search_space: Mapping[str, tuple[float, float]],
        *,
        n_trials: int,
        seeds: Sequence[int],
        horizon: int,
        embargo: int = 0,
        inner_folds: int = 3,
        columns: Sequence[str] | None = None,
        n_jobs: int = 1,
        tune_seed: int = 0,
        params_override: Mapping[str, float] | None = None,
        name: str = "lightgbm",
    ) -> None:
        unknown = [k for k in search_space if k not in KNOWN_PARAMS]
        if unknown:
            raise ModelError(f"unknown search-space entries {unknown}; known: {KNOWN_PARAMS}")
        if any(not 0 < low <= high for low, high in search_space.values()):
            raise ModelError("every search-space range needs 0 < low <= high")
        if not seeds or n_trials < 1:
            raise ModelError("need at least one seed and one trial")
        self.search_space = {k: (float(a), float(b)) for k, (a, b) in search_space.items()}
        self.n_trials = n_trials
        self.seeds = tuple(int(s) for s in seeds)
        self.horizon = horizon
        self.embargo = embargo
        self.inner_folds = inner_folds
        self.columns = None if columns is None else tuple(columns)
        self.n_jobs = n_jobs
        self.tune_seed = tune_seed
        self.params_override = None if params_override is None else dict(params_override)
        self.name = name
        self.best_params_: dict[str, float] = {}
        self.cv_score_ = float("nan")
        self.trials_: list[dict[str, float]] = []
        self._models: list[lgb.LGBMRegressor] = []
        self._features: list[str] = []

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            return X
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ModelError(f"{self.name}: columns {missing} are not in the feature table")
        return X[list(self.columns)]

    def _suggest(self, trial: optuna.Trial) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, (low, high) in self.search_space.items():
            if key in INTEGER_PARAMS:
                out[key] = float(trial.suggest_int(key, int(low), int(high)))
            else:
                out[key] = trial.suggest_float(key, low, high, log=key in LOG_PARAMS)
        return out

    def _tune(self, x: np.ndarray, y: np.ndarray, position: np.ndarray, n_dates: int) -> None:
        try:
            folds = positional_walk_forward(
                n_dates, self.inner_folds, horizon=self.horizon, embargo=self.embargo
            )
        except SplitError as exc:
            raise ModelError(f"{self.name}: training window too short to tune ({exc})") from exc
        masks = [(np.isin(position, f.train), np.isin(position, f.test)) for f in folds]

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest(trial)
            errors = []
            for train, test in masks:
                model = _regressor(params, self.tune_seed, self.n_jobs).fit(x[train], y[train])
                errors.append(float(np.mean((model.predict(x[test]) - y[test]) ** 2)))
            return float(np.mean(errors))

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.tune_seed)
        )
        study.optimize(objective, n_trials=self.n_trials)
        self.trials_ = [
            {**t.params, "score": float(t.value)} for t in study.trials if t.value is not None
        ]
        self.best_params_ = {k: float(v) for k, v in study.best_params.items()}
        self.cv_score_ = float(study.best_value)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GBMForecaster":
        check_table(X, y)
        frame = self._frame(X)
        self._features = [str(c) for c in frame.columns]
        x = frame.to_numpy(dtype=float)
        target = y.to_numpy(dtype=float)
        if self.params_override is not None:
            self.best_params_ = {k: float(v) for k, v in self.params_override.items()}
            self.trials_, self.cv_score_ = [], float("nan")
        else:
            dates = row_dates(X.index)
            calendar = dates.unique().sort_values()
            self._tune(x, target, calendar.get_indexer(dates), len(calendar))
        self._models = [
            _regressor(self.best_params_, seed, self.n_jobs).fit(x, target) for seed in self.seeds
        ]
        return self

    def predict_by_seed(self, X: pd.DataFrame) -> pd.DataFrame:
        """One column of forecasts per seed (columns named ``seed_<s>``)."""
        if not self._models:
            raise ModelError(f"{self.name}: predict called before fit")
        x = self._frame(X).to_numpy(dtype=float)
        data = {f"seed_{s}": m.predict(x) for s, m in zip(self.seeds, self._models, strict=True)}
        return pd.DataFrame(data, index=X.index, dtype=float)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.predict_by_seed(X).mean(axis=1)

    def mean_abs_contributions(self, X: pd.DataFrame) -> pd.Series:
        """Mean absolute TreeSHAP contribution of each feature over ``X``, averaged over seeds."""
        if not self._models:
            raise ModelError(f"{self.name}: called before fit")
        x = self._frame(X).to_numpy(dtype=float)
        parts = [
            np.abs(np.asarray(m.predict(x, pred_contrib=True))[:, :-1]).mean(axis=0)
            for m in self._models
        ]
        values = np.mean(parts, axis=0)
        return pd.Series(values, index=self._features, name="mean_abs_shap").sort_values(
            ascending=False
        )

    @property
    def params(self) -> dict[str, object]:
        return {
            "best_params": dict(self.best_params_),
            "cv_score": self.cv_score_,
            "n_trials": len(self.trials_),
            "seeds": list(self.seeds),
            "tuned": self.params_override is None,
        }
