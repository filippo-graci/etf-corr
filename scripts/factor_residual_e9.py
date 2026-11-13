"""E9: does co-movement beyond common factors rise in high-VIX periods? Run with:
``uv run python scripts/factor_residual_e9.py --fetch-factors`` (downloads two files, computes
nothing), then, once, ``uv run python scripts/factor_residual_e9.py --confirm-run`` (a few minutes).

Design: E9 in ``config/experiment.yaml``, declared and tagged on 2026-11-01 before any statistic
here was computed. H2 found the high-VIX rise of cross-sector correlation to be mostly volatility,
but its Forbes-Rigobon adjustment over-corrected (E5, E7) and left the non-volatility part
unidentified. This script uses a factor model instead.

Within each VIX tercile (t-1 labels, the H2 convention), each core fund's daily return is regressed
on nine factors with an intercept. In-sample OLS makes the factor fit and the residuals
uncorrelated, so for every pair ``cov(r_i, r_j) = b_i' S_F b_j + cov(e_i, e_j)`` exactly; divided by
the two volatilities this splits the average cross-sector correlation into a **factor part** and a
**residual part**. The E9 statistic is the high-minus-low difference of the average cross-sector
**residual correlation** ``corr(e_i, e_j)``: co-movement the factors do not carry, on its own scale.

Factors: Fama-French five and momentum (Kenneth French data library, daily, percent turned into
decimals) and the daily log returns of USO, GLD and TLT. ``--fetch-factors`` downloads the two
library files over HTTPS, parses them and writes ``data/external/ff_factors_daily.parquet`` with a
JSON log (URLs, time, sha256 of each zip). The library revises its history and keeps no vintages:
harmless for an in-sample description of regimes, not for a forecast.

Bootstrap: stationary, days resampled with returns, factors and labels together, mean block length
``round(n^(1/3))`` as in H2, regressions refitted in every draw; one-sided
``p = (1 + #{draws <= 0}) / (B + 1)``. Reported beside the verdict, not part of it: the factor and
residual parts of the raw rise, full-window betas, the four-factor model (Mkt-RF, SMB, HML, Mom),
day-t labels, and the same statistics on 2023-01-03 to 2026-09-18 (to the last factor date).

**Status: in-sample and descriptive, like H2.** One test (a family of one); it changes no verdict.
Outputs: ``reports/tables/e9_factor_{table.csv,summary.json}``, marker ``e9_factor_marker.json``.
"""

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from etfcorr.config import Project, config_hash, find_repo_root, load_project
from etfcorr.correlation.horizon_analysis import (
    default_mean_block_length,
    stationary_bootstrap_indices,
)
from etfcorr.correlation.regime_correlation import pair_mask
from etfcorr.evaluation.final import close_final, git_state, open_final
from etfcorr.evaluation.ladder import load_sealed_panel
from etfcorr.regimes import lag_regimes, vix_terciles

FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "F-F_Momentum_Factor_daily_CSV.zip")
FACTOR_FILE = "ff_factors_daily.parquet"
FF = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
ETF_FACTORS = ["USO", "GLD", "TLT"]
FOUR = ["Mkt-RF", "SMB", "HML", "Mom"]
LOW, HIGH = 0, 2
MARKER = "e9_factor_marker.json"
STEM = "e9_factor"
STATS = ["raw", "factor_part", "residual_part", "residual_corr"]


# --------------------------------------------------------------------------------------
# Factors
# --------------------------------------------------------------------------------------


def parse_french_csv(text: str) -> pd.DataFrame:
    """Daily rows (``YYYYMMDD,v1,...``) of a French library CSV, in decimals, by date.

    The header is the last non-data line before the first data row; parsing stops at the first
    non-data line after it (the files end with a copyright line or an annual block).
    """
    header: list[str] | None = None
    rows: list[list[str]] = []
    last: list[str] = []
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if fields and len(fields[0]) == 8 and fields[0].isdigit():
            if header is None:
                header = last
            rows.append(fields)
        elif rows:
            break
        elif line.strip():
            last = fields
    if header is None or not rows:
        raise SystemExit("no daily rows found in a French library file")
    names = [h for h in header[1:] if h]
    frame = pd.DataFrame([r[1 : len(names) + 1] for r in rows], columns=names, dtype=float)
    frame.index = pd.to_datetime([r[0] for r in rows], format="%Y%m%d")
    return frame / 100.0


def fetch_factors(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, log = [], {"fetched_at": datetime.now(UTC).isoformat(), "files": {}}
    for name in FILES:
        with urllib.request.urlopen(FRENCH + name, timeout=60) as response:
            blob = response.read()
        log["files"][name] = {"url": FRENCH + name, "sha256": hashlib.sha256(blob).hexdigest()}
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            inner = archive.namelist()[0]
            frames.append(parse_french_csv(archive.read(inner).decode("latin-1")))
    factors = pd.concat(frames, axis=1, join="inner").rename(columns=lambda c: c.strip())
    missing = [c for c in FF if c not in factors.columns]
    if missing:
        raise SystemExit(f"factor columns missing after parsing: {missing}")
    path = out_dir / FACTOR_FILE
    factors[[*FF, "RF"]].to_parquet(path)
    log["rows"] = len(factors)
    log["first"], log["last"] = str(factors.index[0].date()), str(factors.index[-1].date())
    (out_dir / "ff_factors_daily.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"wrote {path}: {log['rows']} days, {log['first']} to {log['last']}")
    return path


# --------------------------------------------------------------------------------------
# The decomposition
# --------------------------------------------------------------------------------------


def regime_split(
    y: np.ndarray, f: np.ndarray, mask: np.ndarray, beta: np.ndarray | None = None
) -> dict[str, float]:
    """Average over ``mask`` pairs of the raw correlation and its factor and residual parts.

    ``y`` (days x funds), ``f`` (days x factors). ``beta=None`` fits OLS with an intercept on these
    rows, so ``raw = factor_part + residual_part`` exactly; a given ``beta`` (full-window betas)
    breaks that identity and only ``residual_corr`` is comparable.
    """
    x = np.column_stack([np.ones(len(f)), f])
    coef = np.linalg.lstsq(x, y, rcond=None)[0] if beta is None else beta
    fit = f @ coef[1:]
    resid = y - x @ coef
    sd = y.std(axis=0, ddof=1)
    scale = np.outer(sd, sd)
    raw = np.cov(y, rowvar=False) / scale
    factor_part = np.cov(fit, rowvar=False) / scale
    residual_part = np.cov(resid, rowvar=False) / scale
    residual_corr = np.corrcoef(resid, rowvar=False)
    return {
        "raw": float(raw[mask].mean()),
        "factor_part": float(factor_part[mask].mean()),
        "residual_part": float(residual_part[mask].mean()),
        "residual_corr": float(residual_corr[mask].mean()),
    }


def high_minus_low(
    y: np.ndarray, f: np.ndarray, codes: np.ndarray, mask: np.ndarray, full_beta: bool = False
) -> pd.Series:
    beta = None
    if full_beta:
        x = np.column_stack([np.ones(len(f)), f])
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
    parts = {g: regime_split(y[codes == g], f[codes == g], mask, beta) for g in (LOW, HIGH)}
    out = {f"{s}_{name}": parts[g][s] for g, name in ((LOW, "low"), (HIGH, "high")) for s in STATS}
    out.update({f"{s}_diff": parts[HIGH][s] - parts[LOW][s] for s in STATS})
    out["factor_share"] = out["factor_part_diff"] / out["raw_diff"]
    return pd.Series(out)


def line(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    codes: pd.Series,
    sectors: dict[str, str],
    *,
    n_bootstrap: int,
    alpha: float,
    full_beta: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """Point, interval and the one-sided p of ``residual_corr_diff > 0`` for one specification."""
    # every usable day, middle tercile included, is resampled (as in H2); only low and high enter
    rows = returns.notna().all(axis=1) & factors.notna().all(axis=1) & codes.notna()
    y = returns.loc[rows].to_numpy(dtype=float)
    f = factors.loc[rows].to_numpy(dtype=float)
    c = codes.loc[rows].to_numpy(dtype=int)
    mask = pair_mask([sectors[t] for t in returns.columns], "cross")
    point = high_minus_low(y, f, c, mask, full_beta)
    rng = np.random.default_rng(seed)
    block = default_mean_block_length(len(y))
    draws = []
    for _ in range(n_bootstrap):
        idx = stationary_bootstrap_indices(len(y), block, rng)
        draws.append(high_minus_low(y[idx], f[idx], c[idx], mask, full_beta))
    boot = pd.DataFrame(draws)
    s = boot["residual_corr_diff"].to_numpy()
    table = pd.DataFrame(
        {
            "value": point,
            "ci_lower": boot.quantile(alpha / 2),
            "ci_upper": boot.quantile(1 - alpha / 2),
        }
    ).rename_axis("statistic")
    table.attrs.update(
        p_one_sided=float((1 + (s <= 0).sum()) / (n_bootstrap + 1)),
        n_days=len(y),
        n_low=int((c == LOW).sum()),
        n_high=int((c == HIGH).sum()),
        block=block,
        first=str(returns.index[rows][0].date()),
        last=str(returns.index[rows][-1].date()),
    )
    return table


def tercile_codes(vix: pd.Series, lag: int) -> pd.Series:
    return lag_regimes(vix_terciles(vix, mode="full_sample"), lag)


def specifications(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    vix: pd.Series,
    sectors: dict[str, str],
    *,
    n_bootstrap: int,
    alpha: float,
    primary: bool,
) -> dict[str, pd.DataFrame]:
    """The primary line, then the lines reported beside it (only the first on a second window)."""
    t1 = tercile_codes(vix, 1)
    kw = {"n_bootstrap": n_bootstrap, "alpha": alpha}
    out = {"nine factors, regime betas, t-1 labels": line(returns, factors, t1, sectors, **kw)}
    if primary:
        out["nine factors, full-window betas"] = line(
            returns, factors, t1, sectors, full_beta=True, **kw
        )
        out["four factors (Mkt-RF, SMB, HML, Mom)"] = line(
            returns, factors[FOUR], t1, sectors, **kw
        )
        out["day-t labels"] = line(returns, factors, tercile_codes(vix, 0), sectors, **kw)
    return out


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------


def load_window(
    project: Project, factors: pd.DataFrame, *, final: bool, start: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, str]]:
    returns, sectors, _, vix = load_sealed_panel(project, final=final)
    if start is not None:
        keep = returns.index >= pd.Timestamp(start)
        returns, vix = returns.loc[keep], vix.loc[keep]
    etfs = pd.read_parquet(project.paths.data_processed / "returns_D.parquet")
    frame = factors[FF].reindex(returns.index).join(etfs[ETF_FACTORS].reindex(returns.index))
    last = factors.index[-1]
    inside = returns.index <= last
    return returns.loc[inside], frame.loc[inside], vix.loc[inside], sectors


def show(name: str, table: pd.DataFrame) -> None:
    a = table.attrs
    print(
        f"\n{name}: {a['n_days']} days {a['first']} to {a['last']} (low {a['n_low']}, high "
        f"{a['n_high']}), block {a['block']:.0f}, one-sided p(residual_corr_diff > 0) = "
        f"{a['p_one_sided']:.4f}"
    )
    print(table.round(4).to_string())


def run(project: Project) -> None:
    exp = project.experiment
    n_boot, alpha = exp.inference.n_bootstrap, exp.inference.alpha
    path = project.paths.data_external / FACTOR_FILE
    if not path.is_file():
        raise SystemExit(f"{path} not found: run with --fetch-factors first")
    marker = project.paths.tables / MARKER
    open_final(project, marker, True, "", True)
    commit, _ = git_state(project)
    factors = pd.read_parquet(path)

    kw = {"n_bootstrap": n_boot, "alpha": alpha}
    windows: dict[str, dict[str, pd.DataFrame]] = {}
    returns, frame, vix, sectors = load_window(project, factors, final=False)
    windows["2014-2022 (the verdict)"] = specifications(
        returns, frame, vix, sectors, primary=True, **kw
    )
    test_start = str(exp.splits.test.start)
    returns, frame, vix, sectors = load_window(project, factors, final=True, start=test_start)
    windows["2023-2026 (beside)"] = specifications(
        returns, frame, vix, sectors, primary=False, **kw
    )
    rows: list[pd.DataFrame] = []
    for window, specs in windows.items():
        for spec, table in specs.items():
            show(f"{window} | {spec}", table)
            rows.append(table.reset_index().assign(window=window, specification=spec))
    primary = next(iter(windows["2014-2022 (the verdict)"].values()))
    p = primary.attrs["p_one_sided"]
    supported = bool(p < alpha)
    value = primary.loc["residual_corr_diff"]
    print(
        f"\nE9 {'SUPPORTED' if supported else 'NOT SUPPORTED'}: residual correlation high minus "
        f"low {value['value']:+.4f} [{value['ci_lower']:+.4f}, {value['ci_upper']:+.4f}], "
        f"one-sided p = {p:.4f}"
    )
    out = project.paths.tables
    pd.concat(rows).to_csv(out / f"{STEM}_table.csv", index=False)
    log = json.loads((project.paths.data_external / "ff_factors_daily.json").read_text())
    summary = {
        "status": "E9, in-sample and descriptive, run once",
        "config_hash": config_hash(exp)[:12],
        "commit": commit,
        "factors": [*FF, *ETF_FACTORS],
        "factor_files": log,
        "E9": {
            "supported": supported,
            "p_one_sided": p,
            "alpha": alpha,
            **{k: float(v) for k, v in value.items()},
            **primary.attrs,
        },
        "bootstrap": {"n": n_boot, "seed": 0},
    }
    (out / f"{STEM}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    close_final(marker)
    print(f"wrote {STEM}_table.csv and {STEM}_summary.json under {out}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fetch-factors", action="store_true", help="download; computes nothing")
    mode.add_argument("--confirm-run", action="store_true", help="the single run (writes a marker)")
    args = parser.parse_args(argv)
    project = load_project(find_repo_root())
    if args.fetch_factors:
        fetch_factors(project.paths.data_external)
    else:
        run(project)
    return 0


if __name__ == "__main__":
    sys.exit(main())
