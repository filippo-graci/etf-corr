"""H4, step 2 of 3: train the temporal graph network and forecast (needs torch).

This script imports only numpy, pandas and torch, never ``etfcorr``, so it runs in any Python with
those three (the results were produced with torch 2.11 and CUDA):

    /path/to/torch-env/bin/python -I scripts/h4_tgnn.py --mode validation
    /path/to/torch-env/bin/python -I scripts/h4_tgnn.py --selftest

It reads ``artifacts/h4/<mode>/inputs.npz`` and ``meta.json`` (``scripts/h4_export.py``) and writes
``predictions.parquet`` (one column per graph, ``tgnn_<graph>``) and ``training_log.json``.

**Network** (declared in ``config/experiment.yaml``, ``forecasting.h4``): for a forecast origin
``t`` the input is the fund features at ``lookback_steps`` dates ``stride_days`` apart, ending at
``t``. At each step, ``m = A_hat relu(x W)`` (one graph convolution; ``A_hat`` is the symmetric
normalised adjacency with self-loops, ``D^-1/2 (A + I) D^-1/2``) updates a GRU cell per fund. The
mean of the final fund states, after dropout, is joined with the 17 common features and mapped to
the target by a two-layer MLP. Graph ``none`` has no edges, so ``A_hat = I``: a plain GRU per fund.

**Training, per fold** (the purged walk-forward folds of the export): origins with complete inputs
and a known target; the last ``early_stop_share`` of them is an early-stopping set, separated from
the rest by ``horizon + embargo`` origins so that no early-stopping target overlaps a training
target; inputs and target standardised with the remaining training origins only; full-batch Adam;
the weights of the best early-stopping epoch predict the fold's test origins. Each seed of
``seeds`` trains its own network and the forecasts are averaged (as for LightGBM).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


class TGNN(nn.Module):
    def __init__(self, n_node: int, n_common: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.inp = nn.Linear(n_node, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_common, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, 1)
        )
        self.hidden = hidden

    def forward(self, x: torch.Tensor, common: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        batch, steps, n, _ = x.shape
        h = x.new_zeros(batch * n, self.hidden)
        for s in range(steps):
            m = torch.einsum("ij,bjh->bih", a_hat, torch.relu(self.inp(x[:, s])))
            h = self.gru(m.reshape(batch * n, self.hidden), h)
        pooled = self.drop(h.view(batch, n, self.hidden).mean(dim=1))
        return self.head(torch.cat([pooled, common], dim=1)).squeeze(-1)


def normalised_adjacency(adj: np.ndarray) -> np.ndarray:
    a = adj + np.eye(len(adj))
    d = 1.0 / np.sqrt(a.sum(axis=1))
    return a * d[:, None] * d[None, :]


def sequence_index(origins: np.ndarray, steps: int, stride: int) -> np.ndarray:
    """``(origins, steps)`` row positions, oldest first, ending at each origin."""
    return origins[:, None] - stride * np.arange(steps - 1, -1, -1)[None, :]


def usable(origins: np.ndarray, nodes: np.ndarray, common: np.ndarray, steps: int, stride: int):
    idx = sequence_index(origins, steps, stride)
    ok = (idx[:, 0] >= 0) & np.isfinite(common[origins]).all(axis=1)
    ok[ok] = np.isfinite(nodes[idx[ok]]).all(axis=(1, 2, 3))
    return origins[ok]


def train_one(
    data: dict[str, np.ndarray],
    a_hat: np.ndarray,
    fit: np.ndarray,
    stop: np.ndarray,
    test: np.ndarray,
    design: dict,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    """Forecasts for ``test`` from one network trained on ``fit``, early-stopped on ``stop``."""
    steps, stride = design["lookback_steps"], design["stride_days"]
    nodes, common, y = data["nodes"], data["common"], data["y"]
    mu_n = nodes[fit].reshape(-1, nodes.shape[2]).mean(axis=0)
    sd_n = nodes[fit].reshape(-1, nodes.shape[2]).std(axis=0) + 1e-8
    mu_c, sd_c = common[fit].mean(axis=0), common[fit].std(axis=0) + 1e-8
    mu_y, sd_y = y[fit].mean(), y[fit].std() + 1e-8

    def tensors(origins: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = (nodes[sequence_index(origins, steps, stride)] - mu_n) / sd_n
        c = (common[origins] - mu_c) / sd_c
        return (
            torch.tensor(x, dtype=torch.float32, device=device),
            torch.tensor(c, dtype=torch.float32, device=device),
        )

    torch.manual_seed(seed)
    model = TGNN(nodes.shape[2], common.shape[1], design["hidden"], design["dropout"]).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=design["learning_rate"], weight_decay=design["weight_decay"]
    )
    a = torch.tensor(a_hat, dtype=torch.float32, device=device)
    (xf, cf), (xs, cs) = tensors(fit), tensors(stop)
    yf = torch.tensor((y[fit] - mu_y) / sd_y, dtype=torch.float32, device=device)
    ys = torch.tensor((y[stop] - mu_y) / sd_y, dtype=torch.float32, device=device)
    best, best_epoch, best_state, waited = math.inf, 0, None, 0
    for epoch in range(design["max_epochs"]):
        model.train()
        opt.zero_grad()
        loss = torch.mean((model(xf, cf, a) - yf) ** 2)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val = float(torch.mean((model(xs, cs, a) - ys) ** 2))
        if val < best - 1e-6:
            best, best_epoch, waited = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= design["patience"]:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        xt, ct = tensors(test)
        pred = model(xt, ct, a).cpu().numpy() * sd_y + mu_y
    return pred, {"best_epoch": best_epoch, "stop_mse_std_units": best, "epochs_run": epoch + 1}


def run(data: dict[str, np.ndarray], meta: dict, device: torch.device) -> tuple[pd.DataFrame, list]:
    design = meta["design"]
    steps, stride = design["lookback_steps"], design["stride_days"]
    gap = meta["horizon_days"] + meta["embargo_days"]
    dates = pd.DatetimeIndex(data["dates"])
    out = pd.DataFrame(np.nan, index=dates, columns=[f"tgnn_{g}" for g in meta["graphs"]])
    log = []
    for k in range(meta["n_folds"]):
        train = usable(data[f"train_{k}"], data["nodes"], data["common"], steps, stride)
        train = train[np.isfinite(data["y"][train])]
        test = usable(data[f"test_{k}"], data["nodes"], data["common"], steps, stride)
        n_stop = math.ceil(design["early_stop_share"] * len(train))
        stop, fit = train[-n_stop:], train[: len(train) - n_stop - gap]
        for g, graph in enumerate(meta["graphs"]):
            a_hat = normalised_adjacency(data["adjacency"][k, g])
            preds = []
            for seed in meta["seeds"]:
                pred, info = train_one(data, a_hat, fit, stop, test, design, seed, device)
                preds.append(pred)
                log.append({"fold": k, "graph": graph, "seed": seed, **info})
            out.iloc[test, g] = np.mean(preds, axis=0)
        print(
            f"fold {k}: train {len(fit)} + stop {len(stop)} origins, test {len(test)} "
            f"({dates[test[0]].date()} to {dates[test[-1]].date()})"
        )
    return out, log


def selftest(device: torch.device) -> None:
    """Planted signal and a causality check on a synthetic panel."""
    rng = np.random.default_rng(0)
    t, n, f = 1200, 6, 3
    nodes = rng.standard_normal((t, n, f))
    for i in range(1, t):  # stationary AR(1) features, so the test range is like the training one
        nodes[i] = 0.9 * nodes[i - 1] + np.sqrt(1 - 0.81) * nodes[i]
    signal = nodes[:, :, 0].mean(axis=1)
    y = signal + 0.05 * rng.standard_normal(t)
    common = rng.standard_normal((t, 2))
    adj = np.zeros((1, 1, n, n))
    data = {
        "nodes": nodes,
        "common": common,
        "y": y,
        "adjacency": adj,
        "dates": pd.bdate_range("2010-01-01", periods=t).strftime("%Y-%m-%d").to_numpy(),
        "train_0": np.arange(0, 900),
        "test_0": np.arange(926, 1200),
    }
    design = {
        "lookback_steps": 4,
        "stride_days": 5,
        "hidden": 8,
        "dropout": 0.0,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "max_epochs": 300,
        "patience": 30,
        "early_stop_share": 0.2,
    }
    meta = {
        "design": design,
        "horizon_days": 21,
        "embargo_days": 5,
        "n_folds": 1,
        "graphs": ["none"],
        "seeds": [0],
    }
    pred, _ = run(data, meta, device)
    p = pred["tgnn_none"].to_numpy()[926:]
    r2 = 1 - np.mean((y[926:] - p) ** 2) / np.var(y[926:])
    assert r2 > 0.5, f"the planted signal was not learned (R2 {r2:.2f})"
    changed = dict(data, y=np.where(np.arange(t) >= 900, y + 5.0, y))  # targets after training
    again, _ = run(changed, meta, device)
    np.testing.assert_allclose(again.to_numpy()[926:], pred.to_numpy()[926:], rtol=1e-5)
    print(f"selftest passed: out-of-sample R2 {r2:.2f}; forecasts ignore targets after training")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("validation", "final"), default="validation")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.selftest:
        selftest(device)
        return 0
    folder = ROOT / "artifacts" / "h4" / args.mode
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    with np.load(folder / "inputs.npz") as f:
        data = {k: f[k] for k in f.files}
    start = time.perf_counter()
    preds, log = run(data, meta, device)
    preds.to_parquet(folder / "predictions.parquet")
    (folder / "training_log.json").write_text(
        json.dumps({"device": str(device), "torch": torch.__version__, "runs": log}, indent=1)
        + "\n",
        encoding="utf-8",
    )
    epochs = pd.DataFrame(log).groupby("graph")["best_epoch"].median().to_dict()
    print(
        f"{args.mode}: {len(log)} networks on {device} in {time.perf_counter() - start:.0f}s; "
        f"median best epoch {epochs}"
    )
    print(f"wrote {folder / 'predictions.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
