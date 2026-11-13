"""Synthetic calibration of the H2 engine (step 4.2, not gating, no real data).

Run: ``uv run python scripts/h2_calibration_4_2.py`` (about 3 minutes); ``--quick`` for a smoke
run, ``--seeds=N`` to change the number of simulated panels (default 30).

Three questions, each answered on simulated panels whose truth is known analytically, with the
same N=25 / six-sector shape as the core panel and about 3,200 days:

1. When the *only* change in the high-volatility regime is a rise in the common-factor variance
   (loadings and idiosyncratic variance fixed), how much of the raw correlation increase does each
   adjustment remove? The truth is 100%: no dependence parameter changed.
2. When the dependence also changes (loading up as well as variance), does the adjustment keep
   the true change? The truth share is computed analytically.
3. How often does the 95% stationary-bootstrap interval for the raw high-minus-low difference
   exclude zero when the true difference is zero, for several mean block lengths, when returns
   have volatility clustering? A well-calibrated interval rejects about 5% of the time.

Model: ``z_it = sqrt(m) * (bm * f_t + bs * f_s(i),t) + sqrt(1 - bm**2 - bs**2) * e_it`` with
``m`` the regime's factor-variance multiplier, so every cross-sector pair has population
correlation ``rho = bm**2 * m / (m * (bm**2 + bs**2) + 1 - bm**2 - bs**2)``. Regimes are a
persistent Markov chain (mean spell 25 days); the label known at ``t-1`` is the true state that
governs day ``t`` (a perfect, predetermined label: the best case for the engine).
"""

import sys

import numpy as np
import pandas as pd

from etfcorr.correlation.regime_correlation import (
    bootstrap_regime_correlations,
    regime_correlations,
    vol_standardize,
)

SIZES = (3, 4, 4, 5, 3, 6)
N_DAYS = 3200
MEAN_SPELL = 25.0
MULT = {0: 1.0, 1: 1.7, 2: 3.5}  # factor-variance multiplier per regime
BM2, BS2 = 0.30, 0.15  # market and sector factor shares of variance in the reference regime
QUICK = "--quick" in sys.argv
_SEEDS_ARG = [a for a in sys.argv if a.startswith("--seeds=")]
N_SEEDS = int(_SEEDS_ARG[0].split("=")[1]) if _SEEDS_ARG else (6 if QUICK else 30)
N_BOOT = 100 if QUICK else 300


def sectors() -> dict[str, str]:
    return {f"S{s}_{k}": f"sector{s}" for s, size in enumerate(SIZES) for k in range(size)}


def rho(m: float, bm2: float = BM2, bs2: float = BS2) -> float:
    """Population cross-sector correlation for factor-variance multiplier ``m``."""
    return bm2 * m / (m * (bm2 + bs2) + 1.0 - bm2 - bs2)


def markov_path(n: int, rng: np.random.Generator, n_states: int = 3) -> np.ndarray:
    path = np.empty(n, dtype=int)
    state = int(rng.integers(0, n_states))
    leave = rng.random(n) < 1.0 / MEAN_SPELL
    jump = rng.integers(1, n_states, size=n)
    for t in range(n):
        if leave[t]:
            state = (state + int(jump[t])) % n_states
        path[t] = state
    return path


def simulate(
    seed: int,
    *,
    bm2_by_state: dict[int, float] | None = None,
    state_for_vol: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray]:
    """Returns (n days), labels (n + 1 days), the true factor, and the governing state per day."""
    rng = np.random.default_rng(seed)
    path = markov_path(N_DAYS + 1, rng)
    gov = path[:-1] if state_for_vol is None else state_for_vol
    m = np.array([MULT[int(s)] for s in gov])
    bm2 = np.array([(bm2_by_state or {}).get(int(s), BM2) for s in gov])
    labels_arr = path.astype(float)
    n_assets = sum(SIZES)
    sector_of = np.repeat(np.arange(len(SIZES)), SIZES)
    f = rng.standard_normal(N_DAYS)
    fs = rng.standard_normal((N_DAYS, len(SIZES)))
    e = rng.standard_normal((N_DAYS, n_assets))
    common = np.sqrt(m * bm2)[:, None] * f[:, None] + np.sqrt(m * BS2)[:, None] * fs[:, sector_of]
    z = common + np.sqrt(1.0 - bm2 - BS2)[:, None] * e
    index = pd.bdate_range("2014-01-02", periods=N_DAYS + 1)
    cols = list(sectors())
    returns = pd.DataFrame(0.01 * z, index=index[1:], columns=cols)
    factor = pd.Series(np.sqrt(m * bm2) * f, index=index[1:])
    return returns, pd.Series(labels_arr, index=index), factor, gov


def one_seed(seed: int, bm2_by_state: dict[int, float] | None) -> dict[str, float]:
    returns, labels, factor, _ = simulate(seed, bm2_by_state=bm2_by_state)
    pair = regime_correlations(returns, labels, sectors()).table
    market = regime_correlations(
        returns,
        labels,
        sectors(),
        delta_mode="series",
        delta_series=returns.mean(axis=1),
    ).table
    oracle = regime_correlations(
        returns, labels, sectors(), delta_mode="series", delta_series=factor
    ).table
    std = regime_correlations(vol_standardize(returns), labels, sectors()).table
    out: dict[str, float] = {}
    for g in (1, 2):
        d_raw = pair.loc[g, "avg_raw"] - pair.loc[0, "avg_raw"]
        out[f"raw{g}"] = pair.loc[g, "avg_raw"]
        out[f"pair{g}"] = pair.loc[g, "avg_adjusted"]
        out[f"market{g}"] = market.loc[g, "avg_adjusted"]
        out[f"oracle{g}"] = oracle.loc[g, "avg_adjusted"]
        out[f"std{g}"] = std.loc[g, "avg_raw"]
        out[f"share_pair{g}"] = (
            d_raw - (pair.loc[g, "avg_adjusted"] - pair.loc[0, "avg_adjusted"])
        ) / d_raw
        out[f"share_market{g}"] = (
            d_raw - (market.loc[g, "avg_adjusted"] - market.loc[0, "avg_adjusted"])
        ) / d_raw
        out[f"share_oracle{g}"] = (
            d_raw - (oracle.loc[g, "avg_adjusted"] - oracle.loc[0, "avg_adjusted"])
        ) / d_raw
        out[f"share_std{g}"] = (d_raw - (std.loc[g, "avg_raw"] - std.loc[0, "avg_raw"])) / d_raw
    out["raw0"] = pair.loc[0, "avg_raw"]
    out["std0"] = std.loc[0, "avg_raw"]
    return out


def section(title: str, bm2_by_state: dict[int, float] | None) -> None:
    rows = pd.DataFrame([one_seed(s, bm2_by_state) for s in range(N_SEEDS)])
    mean = rows.mean()
    print(f"\n=== {title} ({N_SEEDS} seeds, N={sum(SIZES)}, T={N_DAYS}) ===")
    bm2 = bm2_by_state or {}
    truth_low = rho(MULT[0], bm2.get(0, BM2))
    for g in (1, 2):
        truth_raw = rho(MULT[g], bm2.get(g, BM2))
        truth_vol_only = rho(MULT[g], BM2)  # variance changes, dependence parameters as in state 0
        true_share = (truth_vol_only - truth_low) / (truth_raw - truth_low)
        print(
            f"state {g}: population correlation {truth_raw:.4f} (reference state {truth_low:.4f}); "
            f"volatility-only counterfactual {truth_vol_only:.4f}; true explained share "
            f"{true_share:.3f}"
        )
        print(
            f"  mean estimate: raw {mean[f'raw{g}']:.4f}"
            f" | FR per-asset delta {mean[f'pair{g}']:.4f}"
            f" | FR delta from equal-weight market {mean[f'market{g}']:.4f}"
            f" | FR delta from true factor (oracle) {mean[f'oracle{g}']:.4f}"
            f" | lagged-EWMA standardised {mean[f'std{g}']:.4f}"
        )
        print(
            f"  explained share: per-asset {mean[f'share_pair{g}']:.3f} | market "
            f"{mean[f'share_market{g}']:.3f} | oracle {mean[f'share_oracle{g}']:.3f} | "
            f"standardised {mean[f'share_std{g}']:.3f}   (seed sd of per-asset share "
            f"{rows[f'share_pair{g}'].std():.3f})"
        )


def null_calibration() -> None:
    """Type-I error of the bootstrap interval for the raw high-minus-low difference.

    The returns have volatility clustering (their own persistent factor-variance state) that is
    independent of the labels, so the true difference is zero; the labels are persistent too.
    """
    print(f"\n=== Null calibration of the diff_raw interval ({N_SEEDS} seeds, {N_BOOT} draws) ===")
    results: dict[float, int] = {}
    blocks = (None, 60.0, 120.0)
    for mbl in blocks:
        rejections = 0
        for seed in range(N_SEEDS):
            returns, _, _, _ = simulate(1000 + seed)
            rng = np.random.default_rng(5000 + seed)
            labels = pd.Series(
                markov_path(N_DAYS + 1, rng).astype(float),
                index=pd.bdate_range("2014-01-02", periods=N_DAYS + 1),
            )
            boot = bootstrap_regime_correlations(
                returns,
                labels,
                sectors(),
                regimes=[0, 2],
                n_bootstrap=N_BOOT,
                mean_block_length=mbl,
                seed=seed,
            )
            rejections += int(boot.stat("diff_raw", 2).excludes_zero)
        key = -1.0 if mbl is None else mbl
        results[key] = rejections
        label = "default heuristic" if mbl is None else f"{mbl:.0f}"
        print(
            f"  mean block length {label:>17}: rejects in {rejections}/{N_SEEDS} "
            f"= {rejections / N_SEEDS:.2f} (nominal 0.05)"
        )


if __name__ == "__main__":
    section("1. Volatility-only regime change (truth: 100% explained)", None)
    section("2. Dependence also rises in state 2 (market share 0.30 -> 0.45)", {2: 0.45})
    null_calibration()
