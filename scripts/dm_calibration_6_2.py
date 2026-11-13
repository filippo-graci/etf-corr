"""Size of the Diebold-Mariano test (HLN-corrected) at this project's sample sizes (step 6.2).
Run with ``uv run python scripts/dm_calibration_6_2.py`` (under a minute; synthetic data only).

The loss differential of two equally good ``h``-day forecasts made at daily origins is
autocorrelated by construction. Two null worlds are simulated, both with mean zero:

* ``ma``: a moving average of ``h`` white-noise terms (errors of an optimal ``h``-step forecast:
  autocorrelated up to lag ``h - 1`` and not beyond, the case DM's rectangular kernel assumes);
* ``ar_ma``: the same moving average applied to an AR(1) series with coefficient 0.9 (dependence
  beyond lag ``h - 1``, closer to losses driven by persistent volatility).

For each (n, h) used in the validation tables the script reports how often the two-sided test
rejects at 5% and 10%, per kernel, and how often the rectangular long-run variance was negative
(the test then falls back to Bartlett). **A property of the test on two data-generating
processes, not evidence about ETFs.** Output: ``reports/tables/dm_calibration_6_2.csv``.
"""

import numpy as np
import pandas as pd

from etfcorr.config import find_repo_root
from etfcorr.evaluation.dm_mcs import diebold_mariano, long_run_variance

CASES = ((498, 5), (482, 21), (440, 63))
N_SIM = 2000
SEED = 0


def null_differential(rng: np.random.Generator, n: int, h: int, world: str) -> np.ndarray:
    e = rng.standard_normal(n + h - 1 + 200)
    if world == "ar_ma":
        for t in range(1, len(e)):
            e[t] += 0.9 * e[t - 1]
    e = e[200:]
    return np.convolve(e, np.ones(h), mode="valid")[:n]


def calibrate(n_sim: int = N_SIM, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for n, h in CASES:
        index = pd.RangeIndex(n)
        zero = pd.Series(np.zeros(n), index=index)
        for world in ("ma", "ar_ma"):
            p = {"rectangular": [], "bartlett": []}
            negative = 0
            for _ in range(n_sim):
                d = null_differential(rng, n, h, world)
                negative += long_run_variance(d, h - 1)[1] == "bartlett"
                for kernel in p:
                    result = diebold_mariano(pd.Series(d, index=index), zero, h, kernel=kernel)  # type: ignore[arg-type]
                    p[kernel].append(result.p_value)
            for kernel, values in p.items():
                v = np.asarray(values)
                rows.append(
                    {
                        "n": n,
                        "horizon": h,
                        "world": world,
                        "kernel": kernel,
                        "reject_5pct": float((v < 0.05).mean()),
                        "reject_10pct": float((v < 0.10).mean()),
                        "rectangular_negative_share": negative / n_sim,
                        "n_sim": n_sim,
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    table = calibrate()
    print("SYNTHETIC NULL WORLDS ONLY (size of the test, not a result about ETFs)")
    print(table.round(3).to_string(index=False))
    out = find_repo_root() / "reports" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "dm_calibration_6_2.csv", index=False)
    print(f"wrote {out / 'dm_calibration_6_2.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
