"""Exploratory follow-up E4 to H3 (docs/FOLLOW_UPS.md item 4). Run with:
``uv run python scripts/followup_h3.py`` (a few minutes; the second part is synthetic).

**Part 1, a fairer naive baseline.** H3's primary comparator was the EWMA of the average
cross-sector correlation with a decay of 0.94 (a memory of about a month, a convention for
one-day horizons). Here the same forecast is built with slower decays, and with the decay chosen
on each training window (``models.baselines.BestColumn``, refitted with the folds of the ladder),
and scored on the dates and targets of the recorded Phase 7 forecasts. Tests are Diebold-Mariano
with the HLN correction, one-sided, as in the verdict.

**Part 2, the size of the Clark-West test.** The H3 verdict used DM-HLN (p = 0.19); Clark-West
gave 0.016 and was reported as uncalibrated. Here both tests are run on simulated forecasts with
the sample sizes of the final run (expanding training window, quarterly refits, 21-day
overlapping target), where the larger model is a tuned ridge on the baseline forecast plus pure
noise features, so it has **no information the baseline lacks**:

* ``baseline_optimal``: the target is the baseline forecast plus noise;
* ``baseline_overreacts``: the target moves half as much as the baseline forecast, so a fitted
  model wins by rescaling the baseline alone.

**Status: exploratory, after the H3 verdict, which it does not change.** The test window has
been read: the 2023-2026 rows are in-sample, and a baseline picked from this table would be
picked on seen data. Part 2 is a property of the tests in two simulated worlds, not evidence
about ETFs. Output: ``reports/tables/followup_h3_{baselines,clark_west}.csv``.
"""

from functools import partial

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from etfcorr import features as F
from etfcorr import targets as T
from etfcorr.config import Project, load_project
from etfcorr.evaluation.dm_mcs import clark_west, diebold_mariano
from etfcorr.evaluation.ladder import load_sealed_panel
from etfcorr.models.base import walk_forward_predict
from etfcorr.models.baselines import BestColumn
from etfcorr.splits import final_walk_forward, walk_forward_from_config

DECAYS = (0.94, 0.97, 0.98, 0.99, 0.995)
TASK, HORIZON = "T1_h21", 21
RECORDED = ("historical_mean", "ewma", "har", "dcc", "elastic_net", "lightgbm_graph_knn")
N_SIM = 500
N_NOISE = 20
SEED = 0


def baselines(project: Project, *, final: bool) -> pd.DataFrame:
    """Mean loss of the slow EWMAs beside the recorded models, on the recorded dates."""
    exp = project.experiment
    mode = "final" if final else "validation"
    returns, sectors, _, _ = load_sealed_panel(project, final=final)
    recorded = pd.read_parquet(
        project.paths.artifacts / "eval_7" / mode / f"predictions_{TASK}.parquet"
    ).dropna()
    cross = F.sector_pair_mask([str(c) for c in returns.columns], sectors, "cross")
    X = pd.DataFrame(
        {
            f"ewma_{lam}": F.fisher_z(F.masked_average(F.ewma_correlations(returns, lam), cross))
            for lam in DECAYS
        },
        index=returns.index,
    )
    np.testing.assert_allclose(X.loc[recorded.index, "ewma_0.94"], recorded["ewma"])
    y = pd.Series(
        F.fisher_z(T.forward_average_correlation(returns, HORIZON, cross)), index=returns.index
    )
    np.testing.assert_allclose(y.loc[recorded.index], recorded["target"])
    calendar = pd.DatetimeIndex(returns.index)
    make = final_walk_forward if final else walk_forward_from_config
    chosen: list[str] = []
    tuned = walk_forward_predict(
        partial(BestColumn, tuple(X.columns), "ewma_tuned"),
        X,
        y,
        make(calendar, exp.splits, HORIZON),
        calendar,
        on_fit=lambda _, model: chosen.append(str(model.params["column"])),
    )
    forecasts = pd.concat(
        [recorded[list(RECORDED)], X.drop(columns="ewma_0.94"), tuned.rename("ewma_tuned")],
        axis=1,
        join="inner",
    ).dropna()
    losses = forecasts.sub(recorded["target"].loc[forecasts.index], axis=0) ** 2
    rows = []
    for name in losses.columns:
        row: dict[str, object] = {
            "window": "test 2023-2026 (seen)" if final else "validation 2021-2022",
            "model": name,
            "mse_fisher_z": float(losses[name].mean()),
            "n_dates": len(losses),
        }
        for other in ("ewma", "ewma_0.99"):  # is `name` better than this baseline?
            if name != other:
                test = diebold_mariano(losses[name], losses[other], HORIZON, alternative="less")
                row[f"diff_vs_{other}"] = test.mean_diff
                row[f"p_better_than_{other}"] = test.p_value
        rows.append(row)
    table = pd.DataFrame(rows)
    table["decays_chosen"] = ""
    table.loc[table["model"] == "ewma_tuned", "decays_chosen"] = " ".join(
        f"{c.split('_')[1]}x{chosen.count(c)}" for c in dict.fromkeys(chosen)
    )
    return table


def ar1(rng: np.random.Generator, n: int, phi: float, k: int = 1) -> np.ndarray:
    e = rng.standard_normal((n + 300, k))
    for t in range(1, len(e)):
        e[t] += phi * e[t - 1]
    return e[300:] * np.sqrt(1.0 - phi**2)


def simulate_tests(
    n_train: int, n_test: int, refit: int, slope: float, n_sim: int, seed: int
) -> pd.DataFrame:
    """p-values of both tests when the larger model only adds noise features to the baseline."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    out = []
    for _ in range(n_sim):
        base = ar1(rng, n, 0.98)[:, 0]
        shocks = rng.standard_normal(n + HORIZON - 1)
        noise = np.convolve(shocks, np.ones(HORIZON) / np.sqrt(HORIZON), mode="valid")
        y = slope * base + noise  # the error overlaps like a 21-day forward window
        X = np.column_stack([base, ar1(rng, n, 0.97, N_NOISE)])
        large = np.full(n, np.nan)
        for start in range(n_train, n, refit):
            fit = slice(0, start - HORIZON)  # purge: targets known at the refit date only
            model = RidgeCV(alphas=np.logspace(-2, 3, 6)).fit(X[fit], y[fit])
            large[start : start + refit] = model.predict(X[start : start + refit])
        index = pd.RangeIndex(n_test)
        target, small, big = (pd.Series(v[n_train:], index=index) for v in (y, base, large))
        dm = diebold_mariano(
            (target - big) ** 2, (target - small) ** 2, HORIZON, alternative="less"
        )
        out.append((dm.p_value, clark_west(target, small, big, HORIZON).p_value, dm.mean_diff))
    return pd.DataFrame(out, columns=["p_dm_hln", "p_clark_west", "mean_loss_diff"])


def calibration(n_train: int, n_test: int, n_sim: int = N_SIM, seed: int = SEED) -> pd.DataFrame:
    rows = []
    for world, slope in (("baseline_optimal", 1.0), ("baseline_overreacts", 0.5)):
        sim = simulate_tests(n_train, n_test, 63, slope, n_sim, seed)
        for test in ("p_dm_hln", "p_clark_west"):
            p = sim[test].dropna()
            rows.append(
                {
                    "world": world,
                    "test": test[2:],
                    "reject_5pct": float((p < 0.05).mean()),
                    "reject_10pct": float((p < 0.10).mean()),
                    "share_p_below_0.016": float((p < 0.016).mean()),
                    "larger_model_better_share": float((sim["mean_loss_diff"] < 0).mean()),
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_sim": len(p),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    project = load_project()
    table = pd.concat([baselines(project, final=False), baselines(project, final=True)])
    print("PART 1: Fisher-z mean squared error, 21-day average cross-sector correlation")
    print("(p_better_than_x: one-sided DM-HLN p-value that the row's model beats x)")
    print(table.round(4).to_string(index=False))
    n_test = int(table["n_dates"].iloc[-1])
    calendar = load_sealed_panel(project, final=True)[0].index
    splits = project.experiment.splits
    n_train = int(
        (
            (calendar >= pd.Timestamp(splits.train.start))
            & (calendar <= pd.Timestamp(splits.val.end))
        ).sum()
    )
    sizes = calibration(n_train, n_test)
    print(f"\nPART 2: SYNTHETIC ({N_SIM} draws per world, nominal one-sided tests)")
    print(sizes.round(3).to_string(index=False))
    table.to_csv(project.paths.tables / "followup_h3_baselines.csv", index=False)
    sizes.to_csv(project.paths.tables / "followup_h3_clark_west.csv", index=False)
    print(f"\nwrote followup_h3_*.csv under {project.paths.tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
