"""Exploratory follow-up E2 to H1 (docs/FOLLOW_UPS.md item 2). Run with:
``uv run python scripts/followup_h1.py`` (a few minutes).

H1 was read on the overlap of the single leading eigenvector, which came out *lower* after the
market mode was removed. After removal the remaining eigenvalues are close together, and
eigenvectors of close eigenvalues are unstable under sampling noise. This script compares the
structure across horizons (D, W, M) before and after removal with statistics that do not depend
on the order of near-equal eigenvalues:

* ``ari``: adjusted Rand index between the average-linkage partitions into as many clusters as
  there are sectors;
* ``subspace_k``: overlap of the spaces spanned by the first ``k`` eigenvectors (mean squared
  cosine of the principal angles), for ``k = 2`` and ``k = sectors - 1``;
* ``eigvec_1``: the statistic of the H1 verdict, for reference.

Intervals: stationary bootstrap, each horizon resampled on its own with the block length of the
H1 engine (the design of step 3.2). **Noise calibration:** the same statistics on simulated
panels of the real size whose true correlation is the same at every horizon, so that any
before/after difference they show is sampling noise: ``factor`` is one market factor plus sector
factors with the real average within- and cross-sector correlations, ``real_corr`` is Gaussian
with the real daily correlation matrix (it keeps the real eigenvalue gaps).

**Status: exploratory, after the H1 verdict, which it does not change.** Core panel, 2014-2022
only (as Phase 3); no test-window data are read. Output:
``reports/tables/followup_h1_{real,simulated}.csv``.
"""

from itertools import combinations

import numpy as np
import pandas as pd

from etfcorr.config import load_project
from etfcorr.correlation import network_stability as NS
from etfcorr.correlation.horizon_analysis import (
    default_mean_block_length,
    leading_eigenvector_overlap,
    stationary_bootstrap_indices,
)
from etfcorr.synthetic import make_factor_returns

HORIZONS = ("D", "W", "M")
AGGREGATE = {"D": 1, "W": 5, "M": 21}  # trading days per row in the simulated panels
N_BOOTSTRAP = 2000
N_SIM = 300
SEED = 0


def statistics(panels: dict[str, pd.DataFrame], n_clusters: int) -> pd.Series:
    """Every statistic for every horizon pair, before and after detoning, and the difference."""
    raw = {h: NS.pearson(p) for h, p in panels.items()}
    detoned = {h: NS.detoned_pearson(p) for h, p in panels.items()}
    out: dict[tuple[str, str, str], float] = {}
    for a, b in combinations(panels, 2):
        for when, corr in (("before", raw), ("after", detoned)):
            out[f"{a}-{b}", "ari", when] = NS.partition_similarity(corr[a], corr[b], n_clusters)
            for k in (1, 2, n_clusters - 1):
                name = "eigvec_1" if k == 1 else f"subspace_{k}"
                out[f"{a}-{b}", name, when] = leading_eigenvector_overlap(corr[a], corr[b], k)
    series = pd.Series(out)
    diff = series.xs("after", level=2) - series.xs("before", level=2)
    return pd.concat([series, pd.concat({"diff": diff}).reorder_levels([1, 2, 0])]).sort_index()


def bootstrap(panels: dict[str, pd.DataFrame], n_clusters: int, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        sample = {}
        for h, p in panels.items():
            idx = stationary_bootstrap_indices(len(p), default_mean_block_length(len(p)), rng)
            sample[h] = pd.DataFrame(p.to_numpy()[idx], columns=p.columns)
        draws.append(statistics(sample, n_clusters))
    return pd.DataFrame(draws)


def aggregate(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Non-overlapping sums of a simulated daily panel: the same truth at every horizon."""
    out = {}
    for h, step in AGGREGATE.items():
        rows = len(daily) // step * step
        block = daily.to_numpy()[:rows].reshape(-1, step, daily.shape[1]).sum(axis=1)
        out[h] = pd.DataFrame(block, columns=daily.columns)
    return out


def simulate(
    world: str, corr: pd.DataFrame, sectors: dict[str, str], n_days: int, n: int, seed: int
) -> pd.DataFrame:
    labels = pd.Series(sectors).loc[list(corr.columns)]
    sizes = tuple(labels.value_counts().tolist())
    same = labels.to_numpy()[:, None] == labels.to_numpy()[None, :]
    off = ~np.eye(len(labels), dtype=bool)
    cross = float(corr.to_numpy()[~same].mean())
    within = float(corr.to_numpy()[same & off].mean())
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(corr.to_numpy())
    draws = []
    for s in range(n):
        if world == "factor":
            daily = make_factor_returns(
                n_days,
                sizes,
                market_loading=np.sqrt(cross),
                sector_loading=np.sqrt(within - cross),
                seed=seed + s,
            ).returns
        else:
            daily = pd.DataFrame(rng.standard_normal((n_days, len(chol))) @ chol.T)
            daily.columns = corr.columns
        draws.append(statistics(aggregate(daily), len(sizes)))
    return pd.DataFrame(draws)


def main() -> int:
    project = load_project()
    core = list(project.universe.tickers_in("core"))
    sectors = {t.ticker: str(t.sector) for t in project.universe.tickers if t.panel == "core"}
    end = pd.Timestamp(project.experiment.splits.val.end)
    panels = {}
    for h in HORIZONS:
        frame = pd.read_parquet(project.paths.data_processed / f"returns_{h}.parquet")[core]
        panels[h] = frame.loc[frame.index <= end].dropna()
    k = len(set(sectors.values()))
    print(
        f"core N={len(core)}, {k} sectors, rows "
        + ", ".join(f"{h} {len(p)}" for h, p in panels.items())
        + f", last date {max(p.index[-1] for p in panels.values()).date()} (2014-2022 only)"
    )

    point = statistics(panels, k)
    boot = bootstrap(panels, k, N_BOOTSTRAP, SEED)
    real = pd.DataFrame(
        {"value": point, "ci_lower": boot.quantile(0.025), "ci_upper": boot.quantile(0.975)}
    ).rename_axis(["pair", "statistic", "when"])
    real["boot_share_above_zero"] = (boot > 0).mean()
    print("\nREAL DATA (95% stationary-bootstrap intervals; diff = after - before)")
    print(real.round(3).to_string())

    frames = []
    for world in ("factor", "real_corr"):
        sim = simulate(world, NS.pearson(panels["D"]), sectors, len(panels["D"]), N_SIM, SEED)
        frames.append(
            pd.DataFrame(
                {
                    "world": world,
                    "sim_mean": sim.mean(),
                    "sim_q05": sim.quantile(0.05),
                    "sim_q95": sim.quantile(0.95),
                    "real": point,
                    "share_of_sims_below_real": (sim.lt(point, axis=1)).mean(),
                }
            ).rename_axis(["pair", "statistic", "when"])
        )
    simulated = pd.concat(frames)
    print(
        f"\nSIMULATED PANELS WITH A HORIZON-INVARIANT TRUTH ({N_SIM} per world; in the truth every "
        "similarity is 1 before and after)"
    )
    print(simulated.round(3).to_string())
    out = project.paths.tables
    real.to_csv(out / "followup_h1_real.csv")
    simulated.to_csv(out / "followup_h1_simulated.csv")
    print(f"\nwrote followup_h1_real.csv and followup_h1_simulated.csv under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
