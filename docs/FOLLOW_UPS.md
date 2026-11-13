# Possible follow-up implementations

Written 2026-10-11, after the Phase 8 verdicts and a literature check. **Status on 2026-10-12: items 1-4 (except
intraday data and a longer history) and item 6 are built and run; item 5 is not. Results:
`docs/FOLLOW_UPS_RESULTS.md`.** The text below is the original proposal, unchanged.

Two rules apply to every item:

- **The six verdicts are fixed.** Anything below is exploratory: add it to the `exploratory` block of
  `config/experiment.yaml` (ids E1, E2, ...) before running it, and report it as exploratory.
- **The test window (2023-01 to 2026-09) has been seen.** A new method judged on it is no longer
  out-of-sample. Either say so next to the number, or develop on 2014-2022 only and hold out data after
  2026-09-18 for a later check.

Ordered by payoff for effort.

## 1. H6: remove the conditioning bias and report what a hedger needs

**Problem.** Stress was defined by the same day's VIX. High-VIX days are days with large market moves, and
conditioning on large market moves inflates the market's share of every ETF's variance. This is the
Forbes-Rigobon bias applied to hedging. The direction of the H6 result (hedges remove a larger share of
variance in stress) is plausible; its size (54% against 19%) is overstated by construction.

**Implement.**

- Stress label from information known beforehand: the previous day's VIX, or the previous 21-day realised
  volatility of SPY, in place of the same-day VIX. One argument in `hedging.regime_masks` and its caller in
  `evaluation/econ.py`.
- A volatility-adjusted effectiveness: apply the Forbes-Rigobon correction to the in-regime correlation
  before squaring it (`correlation/regime_correlation.py` already has the adjustment used for H2).
- Leftover risk, not only the share removed: annualised volatility of the hedged return by regime, and
  the 5% worst daily loss and expected shortfall of hedged against unhedged positions.
- The same table by sector (energy and miners were barely hedged by SPY in calm markets).

**Data needed:** none new; `artifacts` and `reports/tables/econ_8_final_hedge_assets.csv` hold most of it.
**Effort:** small (half a day). **Expected:** a smaller stress-calm gap, possibly none; higher leftover
volatility in stress despite the larger share removed.

## 2. H1: compare structure the way the source paper does

**Problem.** H1 came out reversed on eigenvector overlap, while Borghesi, Marsili and Miccichè (2007) find
that removing the market mode makes the structure more stable across horizons. After the dominant mode is
removed the remaining eigenvalues are close together, which makes eigenvectors unstable under sampling
noise. The Phase 3 table already shows cluster-to-sector agreement rising after removal at every horizon
(D 0.40 to 0.62, W 0.38 to 0.78, M 0.59 to 0.78).

**Implement.**

- Cross-horizon similarity of the **cluster partition** (adjusted Rand index between the average-linkage
  clusters at D, W and M), before and after removal, with the stationary bootstrap of step 3.2.
- Cross-horizon similarity of the **eigen-subspace** spanned by the first k residual eigenvectors
  (principal angles), which does not depend on the order of near-equal eigenvalues.
- A noise calibration: the same statistics on a simulated one-factor-plus-sectors panel at the real T and
  N, to see how much of the eigenvector result is sampling noise.

**Code:** `correlation/networks.py` (`hierarchical_clusters`), `correlation/horizon_analysis.py`,
`correlation/network_stability.py` (`cluster_sector_agreement`). **Effort:** small to medium.
**Data:** 2014-2022 only, as in Phase 3.

## 3. H5: give the cleaned estimators a setting where they can matter

**Problem.** Long-only minimum variance already acts like shrinkage (Jagannathan and Ma 2003), and 25
assets on 252 daily rows is an easy case for the sample covariance (N/T = 0.10).

**Implement, one at a time:**

- **Long-short minimum variance** (weights sum to one, no sign constraint; closed form). Estimation error
  is not hidden by the constraint. One new entry in `portfolio.PORTFOLIOS`.
- **A larger universe:** the 47 ETFs of core plus extended, over their common period, which raises N/T.
- **Nonlinear shrinkage** (Ledoit and Wolf 2017, 2020, analytical form) as a sixth estimator in
  `portfolio.covariance`. Check first whether an installed library provides it (skfolio does, but it does
  not import here until `cvxpy` is installed) before writing it.
- **A full DCC covariance:** export the GARCH variance forecasts from `correlation/dcc.py` instead of
  pairing the DCC correlation with EWMA variances.

**Effort:** small for long-short, medium for the rest. **Expected:** shrinkage helps in long-short and at
higher N/T; that would be consistent with the literature, not a rescue of H5 as pre-registered.

## 4. H3: a cleaner target and a fairer baseline

**Problem.** The target is a correlation measured from 21 daily returns, so much of every model's loss is
measurement noise. EWMA with a decay of 0.94 has a memory of about a month, tuned for one-day horizons;
on the test window it was worse than the historical mean.

**Implement.**

- **A slower EWMA** (decay 0.97 to 0.99) and an EWMA whose decay is chosen on the training window, as
  extra rung-0 baselines in `evaluation/ladder.py`.
- **Intraday realised covariances** as the target and as HAR inputs. Needs a new data source (minute bars)
  with a point-in-time record; a blocker until one is found and its terms checked.
- **A longer history:** many of the 25 ETFs exist before 2014; extending the download start adds
  independent months, the scarcest resource in this study.
- **Calibrate the Clark-West test** at this project's sample sizes (the Diebold-Mariano calibration of step
  6.2 is the template), so the 0.016 beside the H3 verdict can be read against its real size.

**Effort:** small for the EWMA and the calibration, large for intraday data.

## 5. H4 and the deferred rungs

Sequence models (LSTM, GRU, TCN) and a temporal graph neural network were never built, so H4 is "not
attempted". With about 60 independent months of training data they are unlikely to beat DCC, and any
result on 2023-2026 would be in-sample in the sense above. Worth doing only with a longer history (item 4)
and a fresh hold-out window.

## 6. Housekeeping that affects trust in the results

- Install `cvxpy` so skfolio imports; the HRP cross-check in `tests/unit/test_portfolio.py` then runs
  instead of being skipped.
- Verify the futures-based flags of UUP, DBC and BITO from the prospectuses (`config/universe.yaml`).
- Mutation testing for the Phase 7 and Phase 8 modules, as was done through Phase 6.
- The two planned figures not built: the rolling coupling indicator with event annotations, and the MST by
  regime.

## References

Checked by search on 2026-10-11 (title, authors and venue seen in a search listing; full texts not read):

- Jagannathan and Ma (2003), *Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints
  Helps*, Journal of Finance 58(4), 1651-1684.
- Borghesi, Marsili and Miccichè (2007), *Emergence of time-horizon invariant correlation structure in
  financial returns by subtraction of the market mode*, arXiv physics/0702106.
- Ledoit and Wolf (2017), *Nonlinear Shrinkage of the Covariance Matrix for Portfolio Selection: Markowitz
  Meets Goldilocks*, Review of Financial Studies.
- Clark and West (2007), *Approximately normal tests for equal predictive accuracy in nested models*,
  Journal of Econometrics 138(1), 291-311 (checked on 2026-10-11 in step 6.2).

From memory, unverified: Forbes and Rigobon (2002), *No Contagion, Only Interdependence*, Journal of
Finance 57(5); Ledoit and Wolf (2020), analytical nonlinear shrinkage, Annals of Statistics.
