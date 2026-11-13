# 7. The scoreboard, and why the results can be trusted

Step 8.6 (`status/phase_8.md`), `reports/tables/scoreboard_8_6.md`, and the checks that run through every phase
(`status/cross_cutting.md`, `tests/`, `scripts/mutation_followup.py`).

## 7.1 The six verdicts together

`scripts/scoreboard_8_6.py` reads each recorded verdict and applies the Holm (1979) correction across the six
confirmatory tests, as pre-registered. An untested hypothesis enters with p = 1.

| | Verdict | p (one-sided) | Holm p | Holds after Holm | Window | What the follow-up adds |
|---|---|---|---|---|---|---|
| H1 | not supported | 0.998 (derived) | 1.000 | no | full sample, in-sample | the test could not tell the directions apart (E2) |
| **H2** | **supported** | **0.0010** | **0.0060** | **yes** | full sample, in-sample | |
| H3 | not supported | 0.378 | 1.000 | no | test, read once | Clark-West's 0.016 reflects a badly scaled baseline (E4) |
| H4 | supported (a non-superiority claim) | 0.597 | 1.000 | yes | test, read once (2026-11-01) | the network did not beat LightGBM-graph; the graph inside it did nothing |
| H5 | not supported | 0.997 | 1.000 | no | test, read once | long-short, nonlinear shrinkage helps (E3) |
| H6 | not supported | 1.000 | 1.000 | no | test, read once | the gap is the volatility effect (E1) |

**Holm's procedure.** Sort the p-values; compare the smallest with alpha/6, the next with alpha/5, and so on,
stopping at the first that fails. It controls the chance of any false rejection among the six without assuming
the tests are independent. H2's 0.0010 is the floor of a 2000-draw bootstrap; times six it is 0.006, still
below 0.05.

**H4 is the odd one out.** Its claim is that a model does *not* win, so it is supported when its test does
not reject. It enters the family with that test's p-value (0.597), and it holds after Holm because a
correction can only make a rejection harder, never turn a non-rejection into one. "Two of six held" should
therefore be read as one effect found (H2) and one model that failed to beat another (H4).

**Reading the reversals.** H1, H5 and H6 came out with the opposite sign, which a first reading takes as three
surprising findings. The follow-ups (files 02, 05, 06) show that each has a plain explanation: a statistic that
moves the wrong way under noise (H1), a design where the constraint already does the shrinking (H5), and the H2
volatility effect seen through a hedge (H6). The verdicts stand as recorded.

**Later declared tests (2026-11-01), each family kept apart from H1-H6.** E5-E7 repeat H2, H5 and H6 on
2007-2013 with Holm over the three: E5 holds (p 0.0010, Holm 0.0030); E6 and E7 do not, with the same sign as H5
and H6 (`docs/REPLICATION_PRE2014_RESULTS.md`). E9, a test of its own, asks whether co-movement beyond nine common
factors rises in high-VIX periods: not supported (p 0.158; `docs/E9_FACTOR_RESIDUAL_RESULTS.md`). E8 is
descriptive. E10 is declared and runs after September 2027.

## 7.2 One picture

A single market factor carries about half of the common variation (file 02), and its share grows when
volatility rises (H2). That one fact explains the hedging result: a market hedge removes a larger share of a
larger variance in stress, while more risk is still left (H6). What remains after the market factor is small
and noisy: its eigenvectors cannot be compared reliably across horizons (H1), flexible forecasters, a graph neural network included, do not beat
standard ones (H3), and cleaning the covariance does nothing for a long-only portfolio of 25 assets (H5).
Cleaning matters once the portfolio may go short.

## 7.3 Why a null result here can be believed

| Safeguard | Where | What it prevents |
|---|---|---|
| Pre-registration with a fingerprint | `config/hypotheses.lock`, `etfcorr check-config`, tag `prereg-v1` | Rewriting a hypothesis after seeing data |
| Declared design changes, committed before running | `config/experiment.yaml` (`variants`, `h5_variant`, `exploratory`), tags `design-v1` to `design-v4b-declared` | Silent tuning; every configuration tried is counted |
| A sealed test window, opened once | `splits.assert_sealed`, `evaluation/final.py` guards (frozen design, clean working tree, marker written before data are read) | Peeking, and re-running until a result appears |
| Purged and embargoed walk-forward folds | `splits.py`, checked against `purgedcv.purge` | Overlapping targets leaking from training into test |
| Leakage detectors with leaky twins | `tests/leakage/` (22 files), `etfcorr.leakage` | Look-ahead in features, targets, regimes, graphs, weights and hedge ratios; each detector is shown to catch a deliberately leaky copy |
| Calibration of every test on simulated data | `h2_calibration_4_2.py`, `dcc_calibration_4_4.py`, `hmm_calibration_4_3.py`, `lead_lag_calibration_4_5.py`, `dm_calibration_6_2.py`, `followup_h1.py`, `followup_h3.py` | Trusting a test whose size or power at these sample sizes is unknown |
| Cross-checks against independent libraries | `arch` (GARCH stage, MCS), `dcor`, `hmmlearn`, `statsmodels` and scipy (DM test), skfolio (HRP) | Errors in code written in this repository |
| A null benchmark of noise-fed models | `null_benchmark_8_4.py` | Mistaking an ordinary loss for skill |
| Deflated Sharpe ratio and PBO | `evaluation/portfolio.py` through `purgedcv` (Bailey and López de Prado 2014; Bailey et al. 2017) | Picking the best of many backtests |
| Mutation testing | per phase in the check reports; `scripts/mutation_followup.py` for Phases 7-8 and the follow-ups | Tests that pass whatever the code does |

**Mutation testing**, in practice: each mutation replaces one line of source with a plausible mistake (a `<`
for `<=` that reads one extra day, a missing `-1` that uses today's forecast, a dropped Bonferroni factor) and
the tests must fail. Phases 1-6 ran mutations at every step (for example 53 of 54 in Phase 5, the last one
equivalent, and 40 of 40 in Phase 6). On 2026-10-12, 70
mutations of the Phase 7-8 modules and the new code caught 52 at first. The gaps were all in the original Phase 8
tests: nothing checked that the DCC forecast and the monthly lasso refit use only data from before the
rebalance day, that the design's hedge window and stress quantile reach the computation, or that a tiny p-value
alone cannot mark a hypothesis as holding. The code was correct each time; the tests could not have noticed
otherwise. After targeted new tests: **70 of 70**. Full suite: 1804 tests.

## 7.4 What to hold against all of this

- **One test window** (2023 to September 2026), mostly a rising, low-volatility market. "Not supported" means
  "not detected here", with wide intervals.
- **Small effective samples:** about 43 independent months for the forecasting target, about 100 high-VIX
  episodes for H2, 93 stress days in a few clusters for H6.
- **Designs left open by the pre-registration were chosen by the project** (long-only monthly portfolios, a
  single-instrument hedge, stress as a VIX decile, DM-HLN as the H3 test); each choice is dated in the config,
  and the H3 test choice was made after validation p-values were seen.
- **Follow-ups were run after every verdict** and partly on the seen window. They explain results; they confirm
  nothing.
- **Data:** one vendor, survivorship in the choice of funds, no NAV premium or asynchronous-close modelling.

## 7.5 What would turn an explanation into a result

The one follow-up worth confirming is nonlinear shrinkage in long-short minimum-variance portfolios (file 05):
it held in both windows and matches the literature. That design was frozen on 2026-11-01 as E10, to be evaluated
once on 2026-10-01 to 2027-09-30. It will either be the project's second confirmed effect or show that the result
belonged to the seen windows. The other way to new evidence, an earlier window, has been used: the
2007-2013 replication (E5-E7) repeats the H2, H5 and H6 verdicts.
