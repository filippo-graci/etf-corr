"""H4, step 1 of 3: the inputs of the temporal graph network (run in this project's environment).
Run with ``uv run python scripts/h4_export.py`` (validation, under a minute) or, once only and after
``forecasting.h4.frozen_on`` is set and committed,
``uv run python scripts/h4_export.py --final --confirm-unseal`` (test window).

Writes ``artifacts/h4/<mode>/inputs.npz`` and ``meta.json`` for ``scripts/h4_tgnn.py``, which needs
torch and runs in another environment. Everything a network may learn from is fixed here:

* ``y``: H3's target (Fisher-z of the average cross-sector correlation over the next 21 days),
  ``NaN`` where it is not known by the end of the data;
* ``nodes`` (dates x funds x 8): per-fund trailing features, causal (each dated ``t`` uses returns
  up to ``t``; ``tests/unit/test_h4_scripts.py`` checks this with the truncation detector);
* ``common`` (dates x 17): the per-date features LightGBM uses (``features.build_features``);
* the purged walk-forward folds of the ladder (``splits``), and for each fold one adjacency per
  graph kind, built from the fold's **training origins only**, as ``models.graphs`` does.

With ``--final`` the guards of the other final runs apply (``evaluation.final.open_final``):
frozen design, clean working tree, no earlier run, explicit confirmation; the marker
``reports/tables/h4_final_marker.json`` is written before any data are read and is closed by
``scripts/h4_score.py --final``.
"""

import argparse
import json

import numpy as np
import pandas as pd

from etfcorr import features as F
from etfcorr import targets as T
from etfcorr.config import Project, load_project
from etfcorr.evaluation.final import open_final
from etfcorr.evaluation.ladder import build_tasks, load_sealed_panel
from etfcorr.models import graphs as G
from etfcorr.splits import Fold, final_walk_forward, walk_forward_from_config

MARKER = "h4_final_marker.json"
NODE_FEATURES = (
    "log_rv_w5",
    "log_rv_w21",
    "log_rv_w63",
    "log_ewma_var",
    "ret_w5",
    "ret_w21",
    "mean_z_w21",
    "mean_z_w63",
)
GRAPH_WINDOW = 63  # the similarity window of `forecasting.graph`


def node_features(returns: pd.DataFrame) -> np.ndarray:
    """``(dates, funds, 8)`` causal per-fund features, in the order of ``NODE_FEATURES``."""
    stacked = F.build_asset_features(returns)
    own = [
        stacked[c].unstack(level=-1)[list(returns.columns)].to_numpy(dtype=float)
        for c in NODE_FEATURES[:6]
    ]
    n = returns.shape[1]
    off = ~np.eye(n, dtype=bool)
    mean_z = []
    for w in (21, 63):
        z = F.fisher_z(np.clip(F.trailing_correlations(returns, w), -0.999999, 0.999999))
        mean_z.append(np.where(off, z, 0.0).sum(axis=2) / (n - 1))
    return np.stack([*own, *mean_z], axis=2)


def fold_graphs(
    returns: pd.DataFrame,
    sectors: dict[str, str],
    folds: list[Fold],
    kinds: tuple[str, ...],
    project: Project,
) -> np.ndarray:
    """``(folds, kinds, funds, funds)`` adjacency, each from the fold's training origins only."""
    design = project.experiment.forecasting
    assert design is not None and design.graph is not None
    assets = [str(c) for c in returns.columns]
    pairs = G.pair_table(returns, (GRAPH_WINDOW,))
    columns = G.pair_columns(assets, GRAPH_WINDOW)
    n = len(assets)
    i, j = np.triu_indices(n, k=1)
    out = np.zeros((len(folds), len(kinds), n, n), dtype=np.float32)
    for k, fold in enumerate(folds):
        rows = pairs.iloc[fold.train][columns].dropna()
        sim = np.eye(n)
        sim[i, j] = sim[j, i] = np.tanh(rows.to_numpy(dtype=float).mean(axis=0))  # Fisher-z mean
        for g, kind in enumerate(kinds):
            if kind == "none":
                continue  # no message passing: the network sees each fund alone
            out[k, g] = G.build_graph(
                kind,  # type: ignore[arg-type]
                sim,
                assets,
                sectors,
                knn_k=design.graph.knn_k,
                threshold_quantile=design.graph.threshold_quantile,
                seed=design.graph.seed,
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--final", action="store_true", help="the single test-window run")
    parser.add_argument("--confirm-unseal", action="store_true")
    args = parser.parse_args()
    project = load_project()
    exp, paths = project.experiment, project.paths
    design = exp.forecasting
    assert design is not None and design.h4 is not None
    h4 = design.h4
    mode = "final" if args.final else "validation"
    if args.final:
        open_final(
            project,
            paths.tables / MARKER,
            h4.frozen_on is not None,
            "forecasting.h4 is not frozen (frozen_on)",
            args.confirm_unseal,
        )
    returns, sectors, market, vix = load_sealed_panel(project, final=args.final)
    embargo = exp.splits.walk_forward.embargo_days
    task = next(
        t
        for t in build_tasks(
            returns, sectors, design, embargo=embargo, market=market, vix=vix, targets=("T1",)
        )
        if t.name == f"{h4.target}_h{h4.horizon_days}"
    )
    calendar = pd.DatetimeIndex(returns.index)
    make = final_walk_forward if args.final else walk_forward_from_config
    folds = make(calendar, exp.splits, h4.horizon_days)
    kinds = (h4.primary_graph, *h4.ablations)
    mask = F.sector_pair_mask([str(c) for c in returns.columns], sectors, "cross")
    y = F.fisher_z(T.forward_average_correlation(returns, h4.horizon_days, mask))
    np.testing.assert_allclose(y, task.y.to_numpy(), equal_nan=True)  # the ladder's target
    common = task.X[list(task.gbm_columns)].to_numpy(dtype=float)
    arrays: dict[str, np.ndarray] = {
        "dates": calendar.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        "y": np.asarray(y, dtype=float),
        "nodes": node_features(returns),
        "common": common,
        "adjacency": fold_graphs(returns, sectors, folds, kinds, project),
    }
    for k, fold in enumerate(folds):
        arrays[f"train_{k}"], arrays[f"test_{k}"] = fold.train, fold.test
    out = paths.artifacts / "h4" / mode
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "inputs.npz", **arrays)
    meta = {
        "mode": mode,
        "last_date_read": str(calendar[-1].date()),
        "assets": [str(c) for c in returns.columns],
        "node_features": list(NODE_FEATURES),
        "common_features": list(task.gbm_columns),
        "graphs": list(kinds),
        "n_folds": len(folds),
        "horizon_days": h4.horizon_days,
        "embargo_days": embargo,
        "seeds": list(exp.seeds),
        "design": h4.model_dump(mode="json"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    edges = arrays["adjacency"].sum(axis=(2, 3)) / 2
    print(
        f"{mode}: {len(calendar)} dates to {calendar[-1].date()}, {len(folds)} folds, "
        f"nodes {arrays['nodes'].shape}, common {common.shape}, "
        f"edges per graph (first fold) {dict(zip(kinds, edges[0].tolist(), strict=True))}"
    )
    print(f"wrote {out / 'inputs.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
