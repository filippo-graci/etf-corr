# E9: correlation beyond common factors, by VIX regime

Written 2026-11-01. Covers E9 of `config/experiment.yaml`.

**Read this first.**

- **Declared before it was run.** Design committed and tagged `e9-freeze-v1` (commit `6bb6179`); the
  factor files were downloaded afterwards; run once by Filippo (marker
  `reports/tables/e9_factor_marker.json`, completed). A family of one; it changes no H verdict.
- **In-sample and descriptive, like H2.** No model is fitted for prediction; the regimes use VIX terciles
  of the window being described.
- **A labelling error found on the way.** The declaration called 2014-2022 "the H2 window". H2 (and H1)
  used all the data, 2014-01-03 to 2026-09-25. The E9 dates were binding and are unchanged; its two
  windows (2014-2022 for the verdict, 2023-2026 beside) together cover H2's. The README, paper, scoreboard
  and explainers said "2014-2022" for H1 and H2 and were corrected the same day.
- **It overlaps step 4.2b**, an exploratory factor counterfactual (SPY plus latent components, fixed in
  the bootstrap, so its intervals were too narrow) that found no rise in dependence beyond volatility.
  E9 asks the same question as a declared test, with observable factors, betas fitted inside each regime,
  an exact decomposition and regressions refitted in every bootstrap draw.

## In one paragraph

H2 said cross-sector correlation rises in high-VIX periods mostly because of volatility, but its
Forbes-Rigobon adjustment over-corrected and could not say whether anything else changed. With nine
common factors the answer is clear: they carry 99% of the rise (+0.252 of +0.256), and the correlation
between what they leave behind is about 0.01 in both regimes. The residual correlation is +0.009 higher
in high-VIX periods, with an interval of [-0.009, +0.028]: not detected, and at most a few hundredths if
it exists. E9 is therefore not supported, which is a positive finding about H2's open clause: across
sectors, the stress rise in correlation is the common factors, not extra co-movement.

## Method

Within each VIX tercile (t-1 labels, low and high compared), each of the 25 core funds is regressed on
nine factors with an intercept: the Fama-French five (Mkt-RF, SMB, HML, RMW, CMA) and momentum, daily from
the Kenneth French data library (2026 version, sha256 in `e9_factor_summary.json`), and the daily log
returns of USO, GLD and TLT. In-sample OLS makes the fit and the residuals uncorrelated, so for every pair

  correlation = factor part (b_i' S_F b_j / s_i s_j) + residual part (cov(e_i, e_j) / s_i s_j)

holds exactly, and so does its average over the 257 cross-sector pairs. The test statistic is the
high-minus-low difference of the average residual **correlation** corr(e_i, e_j), co-movement the
factors do not carry on its own scale. Stationary bootstrap over days (2000 draws, mean block 13 =
round(n^(1/3)) as in H2), regressions refitted in every draw; one-sided p = (1 + draws <= 0) / 2001.

## Result (2014-01-06 to 2022-12-30, 2264 days; 756 low-VIX, 754 high-VIX)

| | Low VIX | High VIX | High - low |
|---|---|---|---|
| Raw cross-sector correlation | 0.278 | 0.533 | +0.256 [+0.166, +0.333] |
| Factor part | 0.273 | 0.525 | +0.252 [+0.162, +0.329] |
| Residual part | 0.005 | 0.008 | +0.003 [-0.004, +0.010] |
| **Residual correlation (the test)** | 0.006 | 0.016 | **+0.009 [-0.009, +0.028]** |
| Share of the rise carried by the factors | | | 0.988 [0.958, 1.016] |

One-sided p = 0.158: **not supported**.

## Beside the verdict

| Line | Residual correlation, high - low | p | Factor share |
|---|---|---|---|
| nine factors, full-window betas | +0.003 [-0.015, +0.022] | 0.369 | 1.01 |
| four factors (Mkt-RF, SMB, HML, Mom) | +0.015 [-0.007, +0.037] | 0.096 | 1.00 |
| day-t labels | +0.009 [-0.009, +0.026] | 0.170 | 0.99 |
| 2023-01-04 to 2026-09-18 (930 days) | +0.017 [-0.016, +0.048] | 0.175 | 0.97 [0.66, 1.21] |

- **Without oil, gold and Treasuries** the residuals stay more correlated in both regimes (0.027 and
  0.042): energy and mining funds share commodity exposure the equity factors do not span. The
  conclusion does not change.
- **2023-2026** gives the same reading on a window where the raw rise is smaller (+0.134).

## What it does and does not say

- **"Carried by the factors" is not "volatility".** The factor part changes when factor variances rise
  and when betas change; E9 does not split the two. Step 4.2b's Shapley split (SPY only, 2014-2026) gave
  factor variance +0.378, more than the whole rise of +0.235, offset by residual covariance (-0.117),
  with loadings slightly down (-0.026).
- **Cross-sector pairs only.** Co-movement within a sector was not tested.
- **The factor list was a choice** (fixed before the run). More factors leave less residual by
  construction; the four-factor line shows how much the three ETF factors matter.
- **In-sample regimes and a 2026 vintage of the French data**, which revises history. Neither matters for
  this description; both would for a forecast.

Outputs: `reports/tables/e9_factor_{table.csv,summary.json}`.
