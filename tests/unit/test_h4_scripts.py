"""Tests for the H4 pipeline scripts that run here (``scripts/h4_export.py``, ``h4_score.py``).

The network itself (``scripts/h4_tgnn.py``) needs torch, which this environment does not have; its
own ``--selftest`` (a planted signal is learned; forecasts ignore targets after the training
window) is run in the torch environment and recorded in ``status/phase_7.md``.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from etfcorr import config as C
from etfcorr import features as F
from etfcorr.leakage import assert_no_lookahead
from etfcorr.models import graphs as G
from etfcorr.splits import Fold
from etfcorr.synthetic import make_factor_returns

ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPORT, SCORE = load("h4_export"), load("h4_score")
PANEL = make_factor_returns(500, (3, 3, 3), seed=4)
RETURNS = PANEL.returns
SECTORS = dict(zip(RETURNS.columns, PANEL.sectors, strict=True))


def test_node_features_are_causal_and_in_the_declared_order() -> None:
    nodes = EXPORT.node_features(RETURNS)
    assert nodes.shape == (500, 9, len(EXPORT.NODE_FEATURES))
    var5 = (RETURNS.iloc[-5:] ** 2).mean()
    np.testing.assert_allclose(nodes[-1, :, 0], np.log(var5.to_numpy()))
    corr = RETURNS.iloc[-21:].corr().to_numpy()
    expected = np.arctanh(corr[0, 1:]).mean()
    assert nodes[-1, 0, 6] == pytest.approx(expected)

    def as_frame(d: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(EXPORT.node_features(d).reshape(len(d), -1), index=d.index)

    assert_no_lookahead(as_frame, RETURNS, cuts=[100, 250, 400])


def test_fold_graphs_use_the_training_rows_only_and_match_the_graph_model() -> None:
    project = C.load_project(ROOT)
    folds = [Fold(train=np.arange(70, 300), test=np.arange(330, 400))]
    kinds = ("knn", "sector", "random", "none")
    adj = EXPORT.fold_graphs(RETURNS, SECTORS, folds, kinds, project)
    assert adj.shape == (1, 4, 9, 9) and not adj[0, 3].any()
    assert (adj[0, 1] == G.sector_graph(list(RETURNS.columns), SECTORS)).all()
    assert adj[0, 0].sum() == adj[0, 2].sum()  # the random graph has the kNN edge count
    later = RETURNS.copy()
    later.iloc[300:] = later.iloc[300:].to_numpy()[::-1] * 3.0  # change everything after training
    np.testing.assert_array_equal(EXPORT.fold_graphs(later, SECTORS, folds, kinds, project), adj)
    # GraphAugmented.fit: S = tanh(mean of arctanh(inverse_fisher_z(pair columns))) on its rows
    pairs = G.pair_table(RETURNS, (63,))[G.pair_columns(list(RETURNS.columns), 63)]
    rows = pairs.iloc[folds[0].train].dropna().to_numpy()
    z = np.arctanh(F.inverse_fisher_z(rows)).mean(axis=0)
    sim = np.eye(9)
    i, j = np.triu_indices(9, k=1)
    sim[i, j] = sim[j, i] = np.tanh(z)
    np.testing.assert_array_equal(adj[0, 0], G.knn_graph(sim, 3))


def test_the_h4_reading_follows_the_declared_rule() -> None:
    index = pd.bdate_range("2021-01-01", periods=300)
    rng = np.random.default_rng(0)
    target = pd.Series(rng.standard_normal(300), index=index)
    recorded = pd.DataFrame(
        {
            "target": target,
            "lightgbm_graph_knn": target + 0.5,
            "ewma": target + 1.0,
            "dcc": target + 0.8,
        }
    )

    def read(gap: float) -> dict[str, object]:
        noise = pd.Series(rng.standard_normal(300) * 0.05, index=index)
        preds = pd.DataFrame({"tgnn_knn": target + gap + noise})
        out = SCORE.score(preds, recorded, "lightgbm_graph_knn", "tgnn_knn", 5, 0.05, 50, 0.1)
        return out[3]

    better, worse = read(0.0), read(1.0)
    assert better["network_significantly_better"] is True and better["h4_supported"] is False
    assert worse["network_significantly_better"] is False and worse["h4_supported"] is True
    assert better["n_dates"] == 300


def test_the_h4_design_rejects_a_primary_graph_among_its_ablations() -> None:
    block = {
        "declared_on": "2026-11-01",
        "target": "T1",
        "horizon_days": 21,
        "comparator": "lightgbm_graph_knn",
        "primary_graph": "knn",
        "ablations": ["knn"],
        "lookback_steps": 8,
        "stride_days": 5,
        "hidden": 16,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_epochs": 300,
        "patience": 30,
        "early_stop_share": 0.2,
    }
    with pytest.raises(ValueError, match="ablation"):
        C.H4Design.model_validate(block)
    assert C.H4Design.model_validate({**block, "ablations": ["sector"]}).frozen_on is None
