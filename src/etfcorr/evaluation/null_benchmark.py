"""Step 8.4: a null distribution from random signals, for the percentile rank of a forecast loss.

**Question served.** "The model's loss is 0.033": is that better than what the same pipeline
produces when it is fed noise? For each of ``n_draws`` draws a table of random features is
generated (independent Gaussian AR(1) series with coefficient ``phi``, as persistent as trailing
window statistics, and independent of the returns), a ridge regression is tuned and fitted on it
through the **same** purged walk-forward folds as the real models, and its mean loss on the same
dates is recorded. The percentile rank of a model is the add-one share of null draws whose loss
is at or below the model's: small means the model beats almost every noise-fed pipeline.

What it is and is not: the null model can still learn the unconditional mean of the target, so
this null is "the historical mean plus the noise of fitting useless features", not "zero skill
against a naive forecast". It complements the Diebold-Mariano test against a named baseline; it
does not replace it. The rank depends on the number of features, their persistence and the
model class of the null (ridge), all fixed by the caller and reported with the result.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from etfcorr.models.base import walk_forward_predict
from etfcorr.models.linear import PenalisedLinear
from etfcorr.splits import Fold


def random_features(
    index: pd.Index, n_features: int, phi: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Independent stationary Gaussian AR(1) columns with unit variance."""
    if not 0.0 <= phi < 1.0 or n_features < 1:
        raise ValueError("need 0 <= phi < 1 and at least one feature")
    shocks = rng.standard_normal((len(index), n_features)) * np.sqrt(1.0 - phi**2)
    x = np.empty_like(shocks)
    x[0] = rng.standard_normal(n_features)
    for t in range(1, len(index)):
        x[t] = phi * x[t - 1] + shocks[t]
    return pd.DataFrame(x, index=index, columns=[f"noise_{k}" for k in range(n_features)])


def random_signal_losses(
    y: pd.Series,
    folds: Sequence[Fold],
    dates: pd.Index,
    *,
    n_features: int,
    n_draws: int,
    horizon: int,
    embargo: int,
    alpha_grid: Sequence[float],
    inner_folds: int = 3,
    phi: float = 0.97,
    seed: int = 0,
) -> pd.Series:
    """Mean squared error on ``dates`` of a ridge fed random features, one value per draw."""
    calendar = pd.DatetimeIndex(y.index)
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws)
    for k in range(n_draws):
        X = random_features(y.index, n_features, phi, rng)

        def factory() -> PenalisedLinear:
            return PenalisedLinear(
                "ridge", alpha_grid, horizon=horizon, embargo=embargo, inner_folds=inner_folds
            )

        pred = walk_forward_predict(factory, X, y, folds, calendar)
        out[k] = float(((y - pred) ** 2).reindex(dates).mean())
    return pd.Series(out, name="null_loss")


def percentile_ranks(null: pd.Series, losses: Mapping[str, float]) -> pd.DataFrame:
    """Add-one share of null draws with a loss at or below each model's (lower is better)."""
    values = null.to_numpy(dtype=float)
    rows = [
        {
            "model": name,
            "loss": loss,
            "percentile_rank": float((1 + (values <= loss).sum()) / (len(values) + 1)),
        }
        for name, loss in losses.items()
    ]
    return pd.DataFrame(rows)
