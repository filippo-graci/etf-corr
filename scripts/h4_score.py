"""H4, step 3 of 3: score the temporal graph network against the recorded models (this project's
environment). Run with ``uv run python scripts/h4_score.py`` (validation) or, after the single
test-window export and training, ``uv run python scripts/h4_score.py --final``.

Reads ``artifacts/h4/<mode>/predictions.parquet`` (``scripts/h4_tgnn.py``) and the Phase 7
forecasts of the same window and target (``artifacts/eval_7/<mode>/predictions_T1_h21.parquet``,
made by the ladder on the same purged folds), keeps the dates where every model has a forecast,
and reports the H3 loss (squared error of the Fisher-z target) with stationary-bootstrap intervals,
one-sided DM-HLN tests of every network against the comparator ``forecasting.h4.comparator`` and
against EWMA and DCC, and the Model Confidence Set over all models.

**H4 reading** (rule declared in ``config/experiment.yaml`` before any network was fitted): H4,
"the temporal graph network does not significantly beat LightGBM with graph features", is
supported unless the primary network (graph ``forecasting.h4.primary_graph``) has the lower mean
loss with a one-sided DM-HLN p-value below ``inference.alpha``. On validation this is a rehearsal,
not a verdict. With ``--final`` the marker written by the export is closed.

Outputs: ``reports/tables/h4_<mode>_{table,tests,mcs}.csv`` and ``h4_<mode>_summary.json``.
"""

import argparse
import json

import numpy as np
import pandas as pd

from etfcorr.config import config_hash, load_project
from etfcorr.evaluation import metrics as M
from etfcorr.evaluation.dm_mcs import diebold_mariano, model_confidence_set
from etfcorr.evaluation.final import close_final, git_state

CONTEXT = ("ewma", "dcc", "historical_mean", "elastic_net", "lightgbm")


def score(
    preds: pd.DataFrame,
    recorded: pd.DataFrame,
    comparator: str,
    primary: str,
    horizon: int,
    alpha: float,
    n_bootstrap: int,
    mcs_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Loss table, tests, MCS and the H4 reading on the dates every model covers."""
    models = [*preds.columns, comparator, *[c for c in CONTEXT if c in recorded.columns]]
    frame = pd.concat([preds, recorded[[*models[len(preds.columns) :], "target"]]], axis=1)
    frame = frame.dropna()
    losses = frame[models].sub(frame["target"], axis=0) ** 2
    block = M.block_length_for(horizon, len(losses))
    rows = []
    for name in models:
        ci = M.mean_ci(losses[name], mean_block_length=block, n_bootstrap=n_bootstrap)
        rows.append(
            {
                "model": name,
                "mean": ci.mean,
                "ci_lower": ci.lower,
                "ci_upper": ci.upper,
                "n_dates": len(losses),
            }
        )
    table = pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)
    tests = []
    for name in preds.columns:
        for against in (comparator, "ewma", "dcc"):
            if against in losses.columns:
                dm = diebold_mariano(losses[name], losses[against], horizon, alternative="less")
                tests.append(
                    {
                        "model": name,
                        "comparator": against,
                        "mean_diff": dm.mean_diff,
                        "dm_statistic": dm.statistic,
                        "p_one_sided": dm.p_value,
                    }
                )
    tests = pd.DataFrame(tests)
    mcs = model_confidence_set(losses, size=mcs_size, block_size=round(block)).reset_index()
    line = tests[(tests["model"] == primary) & (tests["comparator"] == comparator)].iloc[0]
    beats = bool(line["mean_diff"] < 0 and line["p_one_sided"] < alpha)
    reading: dict[str, object] = {
        "primary": primary,
        "comparator": comparator,
        "loss_primary": float(losses[primary].mean()),
        "loss_comparator": float(losses[comparator].mean()),
        "mean_diff": float(line["mean_diff"]),
        "p_one_sided": float(line["p_one_sided"]),
        "alpha": alpha,
        "network_significantly_better": beats,
        "h4_supported": not beats,
        "n_dates": len(losses),
        "n_effective": len(losses) / horizon,
        "mcs_kept": mcs.loc[mcs["included"], "model"].tolist(),
    }
    return table, tests, mcs, reading


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    project = load_project()
    exp, paths = project.experiment, project.paths
    assert exp.forecasting is not None and exp.forecasting.h4 is not None
    h4 = exp.forecasting.h4
    mode = "final" if args.final else "validation"
    folder = paths.artifacts / "h4" / mode
    preds = pd.read_parquet(folder / "predictions.parquet")
    task = f"{h4.target}_h{h4.horizon_days}"
    recorded = pd.read_parquet(paths.artifacts / "eval_7" / mode / f"predictions_{task}.parquet")
    with np.load(folder / "inputs.npz") as f:
        y = pd.Series(f["y"], index=pd.DatetimeIndex(f["dates"]))
    both = recorded["target"].dropna().index
    np.testing.assert_allclose(y.reindex(both), recorded.loc[both, "target"])  # same target
    table, tests, mcs, reading = score(
        preds,
        recorded,
        h4.comparator,
        f"tgnn_{h4.primary_graph}",
        h4.horizon_days,
        exp.inference.alpha,
        exp.inference.n_bootstrap,
        exp.forecasting.mcs_size,
    )
    tag = "TEST WINDOW: THE H4 VERDICT" if args.final else "validation rehearsal, NOT a verdict"
    print(f"{task}, {reading['n_dates']} dates (about {reading['n_effective']:.0f} independent)")
    print(table.round(5).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    print(f"MCS at {exp.forecasting.mcs_size:.0%} keeps {len(reading['mcs_kept'])} of {len(mcs)}")
    print(
        f"H4 ({tag}): {reading['primary']} {reading['loss_primary']:.5f} against "
        f"{reading['comparator']} {reading['loss_comparator']:.5f}, difference "
        f"{reading['mean_diff']:+.5f}, one-sided DM-HLN p = {reading['p_one_sided']:.3f} -> "
        f"network significantly better: {reading['network_significantly_better']}; "
        f"H4 {'SUPPORTED' if reading['h4_supported'] else 'NOT SUPPORTED'}"
        + ("" if args.final else " (rehearsal)")
    )
    stem = f"h4_{mode}"
    table.to_csv(paths.tables / f"{stem}_table.csv", index=False)
    tests.to_csv(paths.tables / f"{stem}_tests.csv", index=False)
    mcs.to_csv(paths.tables / f"{stem}_mcs.csv", index=False)
    commit, dirty = git_state(project)
    summary = {
        "mode": mode,
        "config_hash": config_hash(exp)[:12],
        "commit": commit,
        "dirty": dirty,
        "h4": reading,
        "design": h4.model_dump(mode="json"),
    }
    (paths.tables / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if args.final:
        close_final(paths.tables / "h4_final_marker.json")
    print(f"wrote {stem}_*.csv/json under {paths.tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
