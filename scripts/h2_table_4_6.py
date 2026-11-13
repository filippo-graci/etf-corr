"""The H2 table and verdict (Phase 4, step 4.6). Run with:
``uv run python scripts/h2_table_4_6.py`` (about 1 to 2 minutes; ``--quick`` is a smoke run with
100 draws that prints only and writes nothing: not a result).

It re-runs the three lines of the H2 table that the confirmed reading rule uses, with the settings
of ``scripts/exploratory_4_2.py`` (same inputs, seed 0, 2000 stationary-bootstrap draws, default
mean block length, 95% percentile intervals), and applies the rule in
``etfcorr.correlation.h2_verdict``:

* primary line: t-1 (predetermined) VIX-tercile labels, low tercile as the reference, cross-sector
  pairs, per-asset delta (two directions averaged);
* market-series lines (the "partly supported" branch): delta from the SPY return and delta from
  the VIX log change;
* robustness lines, printed whatever they show and **not** inputs to the verdict: day-t labels,
  mean block lengths 60 and 120, the primary line without a fixed COVID window (2020-02-24 to
  2020-06-30, returns set to missing), and the within-sector contrast.

**Status: descriptive and in-sample** (the VIX terciles use the full sample; no model is fitted).
The rule (clauses, threshold 0.5, which line decides) was fixed and confirmed on 2026-10-02, before
this script existed. **The 4.2 numbers were already known when it was written**, so the script is a
deterministic reproduction plus the formal reading, not a new test: it checks its own table against
the values recorded for 4.2 and says so when they differ. Nothing here is a trading result.

Outputs (not under ``--quick``): ``reports/tables/h2_table_4_6.csv`` (one row per line) and
``reports/tables/h2_verdict_4_6.json`` (verdict, rule inputs, settings, sensitivities).
"""

import json
import math
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from etfcorr.config import find_repo_root
from etfcorr.correlation.h2_verdict import (
    H2Line,
    H2Verdict,
    SensitivityReading,
    apply_h2_rule,
    sensitivity_reading,
    summarize_line,
    verdict_lines,
)
from etfcorr.correlation.regime_correlation import (
    Alignment,
    DeltaMode,
    H2Result,
    PairScope,
    bootstrap_regime_correlations,
)
from etfcorr.regimes import vix_terciles

N_BOOT_FULL = 2000
SEED = 0
COVID_WINDOW = ("2020-02-24", "2020-06-30")  # fixed here, before the run; returns set to NaN
TOLERANCE = 1.5e-3  # the recorded 4.2 values are rounded to three decimals

# High-minus-low values recorded for step 4.2 (docs/DECISION_LOG.md, 2026-10-02): (point, lower,
# upper); None where the record has no interval. Used only to check the reproduction.
RECORDED_4_2: dict[str, dict[str, tuple[float, float | None, float | None]]] = {
    "primary": {
        "diff_raw": (0.235, 0.159, 0.305),
        "diff_adjusted": (0.027, -0.011, 0.064),
        "vol_share": (0.884, 0.761, 1.065),
    },
    "SPY return delta": {
        "diff_adjusted": (-0.078, -0.106, -0.051),
        "vol_share": (1.333, None, None),
    },
    "VIX log-change delta": {
        "diff_adjusted": (0.094, 0.028, 0.152),
        "vol_share": (0.602, 0.435, 0.829),
    },
}

PRIMARY = "primary"
SPY_LINE = "SPY return delta"
VIX_LINE = "VIX log-change delta"


@dataclass(frozen=True)
class Study:
    """Everything the report, the CSV and the JSON need."""

    primary: H2Line
    market: list[H2Line]
    verdict: H2Verdict
    variants: list[tuple[H2Line, SensitivityReading]]
    results: dict[str, H2Result]
    n_boot: int
    seed: int
    n_assets: int
    n_pairs: int
    mean_block_length: float
    regime_table: pd.DataFrame


def _run_line(
    name: str,
    returns: pd.DataFrame,
    labels: pd.Series,
    sectors: Mapping[str, str],
    *,
    n_boot: int,
    seed: int,
    delta_mode: DeltaMode = "pair",
    delta_series: pd.Series | None = None,
    alignment: Alignment = "predetermined",
    pairs: PairScope = "cross",
    mean_block_length: float | None = None,
) -> tuple[H2Line, H2Result]:
    result = bootstrap_regime_correlations(
        returns,
        labels,
        sectors,
        alignment=alignment,
        pairs=pairs,
        delta_mode=delta_mode,
        delta_series=delta_series,
        mean_block_length=mean_block_length,
        n_bootstrap=n_boot,
        seed=seed,
    )
    return summarize_line(name, result), result


def mask_window(returns: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Return ``returns`` with every row dated in ``[start, end]`` set to NaN.

    The calendar is kept (the engine requires consecutive trading days); the engine drops
    incomplete rows. The regime label of the first kept day after the window is still the one
    known the day before it, so predetermined alignment is preserved.
    """
    out = returns.copy()
    out.loc[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))] = np.nan
    return out


def run_study(
    returns: pd.DataFrame,
    labels: pd.Series,
    sectors: Mapping[str, str],
    spy: pd.Series,
    vix_change: pd.Series,
    *,
    n_boot: int = N_BOOT_FULL,
    seed: int = SEED,
    covid_window: tuple[str, str] | None = COVID_WINDOW,
) -> Study:
    """Compute the three rule lines, the verdict and the robustness lines."""
    kw = {"n_boot": n_boot, "seed": seed}
    primary, primary_res = _run_line(PRIMARY, returns, labels, sectors, **kw)
    spy_line, spy_res = _run_line(
        SPY_LINE, returns, labels, sectors, delta_mode="series", delta_series=spy, **kw
    )
    vix_line, vix_res = _run_line(
        VIX_LINE, returns, labels, sectors, delta_mode="series", delta_series=vix_change, **kw
    )
    market = [spy_line, vix_line]
    verdict = apply_h2_rule(primary, market)

    variants: list[tuple[H2Line, SensitivityReading]] = []
    results: dict[str, H2Result] = {
        PRIMARY: primary_res,
        SPY_LINE: spy_res,
        VIX_LINE: vix_res,
    }

    def add(name: str, data: pd.DataFrame = returns, **extra: object) -> None:
        line, res = _run_line(name, data, labels, sectors, **kw, **extra)  # type: ignore[arg-type]
        variants.append((line, sensitivity_reading(line)))
        results[name] = res

    add("day-t labels (contemporaneous)", alignment="contemporaneous")
    add("mean block length 60", mean_block_length=60.0)
    add("mean block length 120", mean_block_length=120.0)
    if covid_window is not None:
        add(
            f"without {covid_window[0]} to {covid_window[1]}",
            mask_window(returns, *covid_window),
        )
    add("within-sector pairs (contrast)", pairs="within")

    table = primary_res.point.table
    return Study(
        primary=primary,
        market=market,
        verdict=verdict,
        variants=variants,
        results=results,
        n_boot=n_boot,
        seed=seed,
        n_assets=primary_res.point.n_assets,
        n_pairs=primary_res.point.n_pairs,
        mean_block_length=primary_res.mean_block_length,
        regime_table=table,
    )


def check_reproduction(study: Study, *, with_intervals: bool) -> list[str]:
    """Differences (above the rounding tolerance) from the values recorded for step 4.2."""
    lines = {line.name: line for line in [study.primary, *study.market]}
    problems: list[str] = []
    for name, stats in RECORDED_4_2.items():
        line = lines[name]
        for stat_name, (point, lower, upper) in stats.items():
            stat = getattr(line, stat_name)
            pairs = [("point", stat.point, point)]
            if with_intervals:
                if lower is not None:
                    pairs.append(("lower", stat.lower, lower))
                if upper is not None:
                    pairs.append(("upper", stat.upper, upper))
            for part, got, want in pairs:
                if not math.isfinite(got) or abs(got - want) > TOLERANCE:
                    problems.append(
                        f"{name} {stat_name} {part}: got {got:+.4f}, recorded {want:+.3f}"
                    )
    return problems


def _num(x: float) -> float | None:
    return None if not math.isfinite(x) else float(x)


def table_frame(study: Study) -> pd.DataFrame:
    """One row per line: the three statistics with their intervals."""
    rows = []
    roles = [(study.primary, "primary"), *[(m, "market") for m in study.market]]
    roles += [(line, "robustness") for line, _ in study.variants]
    for line, role in roles:
        row: dict[str, object] = {
            "line": line.name,
            "role": role,
            "delta_mode": line.delta_mode,
            "pairs": line.pairs,
            "alignment": line.alignment,
            "n_rows": line.n_rows,
        }
        for stat_name in ("diff_raw", "diff_adjusted", "vol_share"):
            stat = getattr(line, stat_name)
            row[stat_name] = stat.point
            row[f"{stat_name}_lower"] = stat.lower
            row[f"{stat_name}_upper"] = stat.upper
        rows.append(row)
    return pd.DataFrame(rows)


def verdict_payload(study: Study, reproduction: list[str]) -> dict[str, object]:
    v = study.verdict
    return {
        "step": "4.6",
        "status": "descriptive, in-sample; rule confirmed 2026-10-02",
        "verdict": v.verdict,
        "clause1_direction": v.clause1,
        "clause1_p_bootstrap": v.clause1_p,
        "clause2": v.clause2,
        "threshold": v.threshold,
        "primary": {
            "name": v.primary_name,
            "share": _num(v.primary_share),
            "share_lower": _num(v.primary_share_lower),
        },
        "market_shares": {k: _num(s) for k, s in v.market_shares.items()},
        "residual_reading": v.residual,
        "notes": list(v.notes),
        "settings": {
            "n_bootstrap": study.n_boot,
            "seed": study.seed,
            "mean_block_length": study.mean_block_length,
            "n_assets": study.n_assets,
            "n_cross_sector_pairs": study.n_pairs,
            "n_rows": study.primary.n_rows,
            "regime": study.primary.regime,
            "reference": study.primary.reference,
        },
        "robustness": [
            {
                "line": line.name,
                "clause1": r.clause1,
                "share": _num(r.share),
                "share_lower": _num(r.share_lower),
                "same_reading_as_primary_rule": r.same_reading,
            }
            for line, r in study.variants
        ],
        "reproduces_4_2": not reproduction,
        "reproduction_differences": reproduction,
    }


def report_lines(study: Study, reproduction: list[str]) -> list[str]:
    out = verdict_lines(study.verdict, [study.primary, *study.market])
    out.append("")
    out.append("robustness lines (not inputs to the verdict; 'same' = clause 1 and share >= 0.5):")
    for line, r in study.variants:
        share_ci = "CI n/a" if math.isnan(r.share_lower) else f"lower {r.share_lower:+.3f}"
        out.append(
            f"  {line.name}: raw {line.diff_raw.point:+.4f}, adjusted "
            f"{line.diff_adjusted.point:+.4f}, share {r.share:+.3f} ({share_ci}), clause 1 "
            f"{'met' if r.clause1 else 'not met'}, same reading: "
            f"{'yes' if r.same_reading else 'no'}"
        )
    out.append("")
    if reproduction:
        out.append("REPRODUCTION OF 4.2: MISMATCH (this is NOT the recorded 4.2 table)")
        out.extend(f"  {p}" for p in reproduction)
    else:
        out.append("reproduction of 4.2: every recorded high-minus-low number matches (3 decimals)")
    return out


def _load(root: Path) -> tuple[pd.DataFrame, pd.Series, dict[str, str], pd.Series, pd.Series]:
    with open(root / "config" / "universe.yaml") as f:
        universe = yaml.safe_load(f)
    core = [t for t in universe["tickers"] if t["panel"] == "core"]
    core_tickers = [t["ticker"] for t in core]
    sectors = {t["ticker"]: t["sector"] for t in core}

    returns_all = pd.read_parquet(root / "data" / "processed" / "returns_D.parquet")
    if isinstance(returns_all.columns, pd.MultiIndex):
        returns_all = returns_all.xs(returns_all.columns.levels[-1][0], axis=1, level=-1)
    missing = [t for t in [*core_tickers, "SPY", "^VIX"] if t not in returns_all.columns]
    if missing:
        raise SystemExit(f"missing from returns_D.parquet: {missing}")

    prices = pd.read_parquet(root / "data" / "interim" / "prices_clean.parquet")
    if not isinstance(prices.columns, pd.MultiIndex):
        raise SystemExit("expected (ticker, field) MultiIndex columns in prices_clean.parquet")
    labels = vix_terciles(prices["^VIX"]["close"].rename("^VIX"), mode="full_sample")
    return (
        returns_all[core_tickers],
        labels,
        sectors,
        returns_all["SPY"].rename("SPY"),
        returns_all["^VIX"].rename("^VIX"),
    )


def main() -> int:
    quick = "--quick" in sys.argv
    n_boot = 100 if quick else N_BOOT_FULL
    root = find_repo_root()
    returns, labels, sectors, spy, vix_change = _load(root)
    smoke = "QUICK SMOKE RUN, NOT A RESULT. " if quick else ""
    print(f"IN-SAMPLE, DESCRIPTIVE (no model fitted). {smoke}")
    print(
        f"core panel N={returns.shape[1]}, {len(returns)} daily returns "
        f"{returns.index[0].date()} to {returns.index[-1].date()}; "
        f"VIX terciles from {int(labels.notna().sum())} levels"
    )
    start = time.perf_counter()
    study = run_study(returns, labels, sectors, spy, vix_change, n_boot=n_boot)
    print(f"computed in {time.perf_counter() - start:.1f}s; {study.n_pairs} cross-sector pairs")
    print(study.regime_table.to_string())
    reproduction = check_reproduction(study, with_intervals=not quick)
    print()
    for text in report_lines(study, reproduction):
        print(text)

    if not quick:
        out_dir = root / "reports" / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        table_frame(study).to_csv(out_dir / "h2_table_4_6.csv", index=False)
        with open(out_dir / "h2_verdict_4_6.json", "w") as f:
            json.dump(verdict_payload(study, reproduction), f, indent=2)
            f.write("\n")
        print(f"\nwrote {out_dir / 'h2_table_4_6.csv'} and {out_dir / 'h2_verdict_4_6.json'}")
    return 2 if reproduction else 0


if __name__ == "__main__":
    raise SystemExit(main())
