"""Synthetic calibration of the step 4.3 HMM regimes (offline, no real data). Run with:
``uv run python scripts/hmm_calibration_4_3.py`` (about a minute).

Two kinds of panel with N=25 assets and 3201 days, the size of the core panel:

* **noise**: ``make_factor_returns`` (a fixed, time-invariant correlation structure, no regimes).
  Any "regime" the HMM reports here is an artefact. Printed: how persistent and how well
  separated the fitted states look anyway.
* **planted**: a common factor whose loading switches between 0.5 and 2.0 with probability 0.01
  per day (two persistent regimes, known truth). Printed: label accuracy of the causal filter,
  which is limited by the trailing 63-day window the observations are built from, and the
  separation diagnostic.

These are properties of the estimator on two data-generating processes, not evidence about ETFs.
Nothing is written.
"""

import numpy as np
import pandas as pd

from etfcorr.regime_hmm import forward_filtered_hmm, hmm_observations, hmm_summary_lines
from etfcorr.synthetic import make_factor_returns

SIZES = (4, 4, 4, 4, 4, 5)
SECTOR_OF = np.repeat(range(len(SIZES)), SIZES)


def _sectors(columns: pd.Index) -> dict[str, str]:
    return {str(c): f"s{i}" for c, i in zip(columns, SECTOR_OF, strict=True)}


print("NOISE panels (no regimes exist): the states below are artefacts")
for seed in (1, 2, 3, 4, 5):
    panel = make_factor_returns(3201, SIZES, seed=seed)
    obs = hmm_observations(panel.returns, _sectors(panel.returns.columns))
    out = forward_filtered_hmm(obs, 2)
    fits = out.fits
    print(
        f"  seed {seed}: separation min/median/max "
        f"{fits['min_separation'].min():.2f}/{fits['min_separation'].median():.2f}/"
        f"{fits['min_separation'].max():.2f}; p_stay medians "
        f"{fits['p_stay_0'].median():.3f}/{fits['p_stay_1'].median():.3f}"
    )

print("\nPLANTED two-regime panel (switch probability 0.01 per day, loading 0.5 vs 2.0)")
rng = np.random.default_rng(5)
n, m = 3201, int(sum(SIZES))
state = np.zeros(n, dtype=int)
for t in range(1, n):
    state[t] = state[t - 1] if rng.random() > 0.01 else 1 - state[t - 1]
loading = np.where(state == 1, 2.0, 0.5)[:, None]
data = (loading * rng.standard_normal((n, 1)) + rng.standard_normal((n, m))) * 0.01
columns = [f"T{i}" for i in range(m)]
returns = pd.DataFrame(data, index=pd.bdate_range("2014-01-02", periods=n), columns=columns)
obs = hmm_observations(returns, _sectors(returns.columns))
out = forward_filtered_hmm(obs, 2)
labelled = out.labels.notna().to_numpy()
accuracy = float((out.labels.to_numpy()[labelled] == state[labelled]).mean())
print(f"  causal label accuracy over {int(labelled.sum())} labelled days: {accuracy:.3f}")
for line in hmm_summary_lines(out):
    print("  " + line)
