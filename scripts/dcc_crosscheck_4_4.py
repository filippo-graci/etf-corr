"""Local cross-check of the GARCH stage of step 4.4 against ``arch`` on the real core panel.

Run with ``uv run python scripts/dcc_crosscheck_4_4.py`` (a few seconds). Needs ``arch``
(``uv pip install arch``), which the offline sandbox that wrote ``dcc.py`` did not have, so this is
**Gate 2 only**. Prints only; nothing is written. **Technical check of an implementation, not a
result about the ETFs.**

For each core ETF the demeaned daily log returns (in percent) are fitted twice with a
zero-mean GARCH(1,1) and Student-t innovations: by ``arch`` (``mean="Zero"`` on the
demeaned series) and by ``dcc.fit_garch11(..., demean=False)``. Printed: the largest absolute
differences in alpha and beta, the smallest and largest log-likelihood difference (ours minus
arch; the optimiser that ends higher is the better maximiser, and ours should never be materially
lower), and, for the likelihood-surface check, the difference between ``arch``'s reported
log-likelihood and ours *evaluated at arch's parameters* (this should be at numerical-noise level
if the backcast convention is the same; a systematic offset means it is not).

The ``arch`` calls follow its documented public interface; the step 4.4 local run confirmed them
(the GARCH stage matched ``arch`` on all 25 core ETFs).
"""

import numpy as np
import pandas as pd
import yaml
from arch import arch_model

from etfcorr.config import find_repo_root
from etfcorr.correlation import dcc

root = find_repo_root()
with open(root / "config" / "universe.yaml") as f:
    universe = yaml.safe_load(f)
core_tickers = [t["ticker"] for t in universe["tickers"] if t["panel"] == "core"]

returns_all = pd.read_parquet(root / "data" / "processed" / "returns_D.parquet")
if isinstance(returns_all.columns, pd.MultiIndex):
    returns_all = returns_all.xs(returns_all.columns.levels[-1][0], axis=1, level=-1)
returns = returns_all[core_tickers].dropna()
print(f"core panel N={len(core_tickers)}, {len(returns)} complete daily returns")

rows = []
for ticker in core_tickers:
    x = 100.0 * returns[ticker].to_numpy(dtype=float)
    x = x - x.mean()
    res = arch_model(x, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off")
    p = res.params
    ours = dcc.fit_garch11(x, dist="t", demean=False)
    at_arch = dcc.garch11_loglik(
        x,
        float(p["omega"]),
        float(p["alpha[1]"]),
        float(p["beta[1]"]),
        dcc.garch_backcast(x),
        dist="t",
        nu=float(p["nu"]),
    )
    rows.append(
        {
            "ticker": ticker,
            "d_alpha": ours.alpha - float(p["alpha[1]"]),
            "d_beta": ours.beta - float(p["beta[1]"]),
            "d_nu": ours.nu - float(p["nu"]),
            "ll_ours_minus_arch": ours.loglik - float(res.loglikelihood),
            "ll_surface_gap": at_arch - float(res.loglikelihood),
        }
    )
table = pd.DataFrame(rows).set_index("ticker")
print(table.round(5).to_string())
print(
    f"\nmax |d_alpha| {table['d_alpha'].abs().max():.5f}; max |d_beta| "
    f"{table['d_beta'].abs().max():.5f}; max |d_nu| {table['d_nu'].abs().max():.4f}"
)
print(
    f"loglik ours - arch: min {table['ll_ours_minus_arch'].min():.5f}, "
    f"max {table['ll_ours_minus_arch'].max():.5f}"
)
print(
    f"loglik surface gap at arch's parameters: max abs {np.abs(table['ll_surface_gap']).max():.2e}"
)
