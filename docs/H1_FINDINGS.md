# H1 — Cross-horizon correlation structure and market-mode removal

Phase 3, step 3.2. Last updated 2026-09-28. Status of the underlying code: **LOCAL-PASSED**
(`src/etfcorr/correlation/horizon_analysis.py`, `docs/checks/3.2.md`). Status of the result below:
a formal, multiplicity-corrected confirmatory test at one specification (`n_market=1`), not yet a
final scientific verdict — open robustness checks are listed at the end.

This file collects everything about H1 in one place: the hypothesis, the method, the real-data
result (raw and multiplicity-corrected), the literature behind each design choice, and open
questions. It duplicates some material already in `docs/DECISION_LOG.md` and `docs/STATUS.md` by
design, so it can be read standalone.

## 1. The hypothesis

`config/experiment.yaml`, verbatim:

> **H1** (statement): After removing the first eigenmode (market mode), the correlation structure
> is more similar across horizons (D/W/M) than before removal.
>
> **H1** (test): Similarity of correlation matrices and of leading eigenvectors across horizons,
> before versus after market-mode removal; stationary-bootstrap confidence intervals on the
> difference.

In plain terms: every ETF in this panel is exposed to a common, dominant "market" effect — on a
bad day for equities broadly, almost everything in the panel falls together, largely irrespective
of sector. This single effect can make the correlation structure look similar at every sampling
frequency (daily, weekly, monthly) simply because the market factor itself is present at every
frequency, potentially masking whatever *sector-specific* structure sits underneath it and
whether *that* structure is stable across horizons. H1 asks: once you subtract the market effect
out, does the remaining (sector-level) structure look more alike across D/W/M than it did before?
A "yes" would suggest the market factor is largely what makes the horizons look different from
each other, and the true, market-adjusted economic structure is closer to horizon-invariant. This
is confirmatory (pre-registered), not exploratory.

## 2. Method

### 2.1 Market-mode removal ("detoning")

Given a correlation matrix, decompose it into eigenvalues/eigenvectors. The single largest
eigenvalue's eigenvector is, empirically, close to a uniform, all-positive vector — "everything
moves together" — and is identified with the market factor. Detoning subtracts that eigenmode's
outer-product contribution from the matrix and renormalizes back to unit diagonal (implemented
in `src/etfcorr/correlation/rmt.py::detone_corr`, step 3.1, LOCAL-PASSED).

- **Marchenko & Pastur (1967)**, "Distribution of eigenvalues for some sets of random matrices,"
  *Mathematics of the USSR-Sbornik* 1(4). **[CHECKED, step 3.1]** — gives the random-matrix noise
  edge used to separate "signal" eigenvalues from pure-noise ones (used in step 3.1's
  `eigen_spectrum`, referenced below, not directly in this step's code).
- **Laloux, Cizeau, Bouchaud & Potters (1999/2000)**, "Random Matrix Theory and Financial
  Correlations," *Int. J. Theor. Appl. Finance* 3(3). **[CHECKED, this session]** — identifies the
  top eigenvalue's eigenvector as "the market itself... roughly equal components on all N
  stocks." Their own cleaning method is eigenvalue shrinkage toward the identity within the noise
  band, not subtraction/detoning, and the paper does not discuss what happens to off-diagonal
  signs after removing the market mode.
- **Bun, Bouchaud & Potters (2017)**, "Cleaning Large Correlation Matrices: Tools from Random
  Matrix Theory," *Physics Reports* 666, 1–109, arXiv:1610.08104. **[CHECKED, this session,
  partial]** — reviews eigenvalue "spikes" as signal versus Marchenko-Pastur bulk noise. The
  excerpt retrieved this session does not explicitly address sign reversal from market-mode
  subtraction; it may be discussed elsewhere in the 109-page review (not confirmed either way).
- **López de Prado (2020)**, *Machine Learning for Asset Managers*, Cambridge Elements in
  Quantitative Finance, the "Denoising and Detoning" material (commonly Ch. 2). **[CHECKED,
  partial, this session]** — gives the detoning formula (subtract the market eigenmode's
  contribution, rescale to unit diagonal) and frames it as letting sector-specific structure "be
  heard" once the market "tone" is removed. **The retrieved text does not explicitly state that
  this produces negative off-diagonal correlations from originally-positive ones** — that
  specific claim should not be attributed to this source as a direct quote or named result.

**Sign reversal after detoning — status corrected from an earlier draft of this project's own
notes.** An earlier pass attributed "detoned correlations can go negative" directly to the RMT/
detoning literature above. On checking this session, none of the four sources above states this
explicitly. The claim is nonetheless **[Inference]**, grounded in ordinary multivariate
statistics: subtracting a common factor from a correlation matrix is the same operation as
computing a *partial* correlation (the correlation between two variables after conditioning out a
third), and it is a standard, textbook fact that marginal and partial correlation can differ in
sign (**Baba, Shibata & Sibuya (2004)**, "Partial correlation and conditional correlation as
measures of conditional independence," *Australian & New Zealand Journal of Statistics* 46(4),
657–664, DOI 10.1111/j.1467-842X.2004.00360.x — **[CHECKED existence/venue only, this session]**;
a general treatment of the concept, not written about detoning specifically). Separately, this
project has its own **direct empirical confirmation**, not just a citation: step 3.1's real-data
exploratory check found that detoning the daily core-23 correlation matrix (`n_market=1`) sent
84.5% of cross-sector pairs negative and only 5.9% of within-sector pairs negative — the
mechanism demonstrated quantitatively on this project's own data, independent of what any paper
says.

### 2.2 Similarity metrics

Two ways of comparing a pair of correlation matrices (e.g., the daily-horizon one and the
weekly-horizon one), computed before and after detoning:

- **Frobenius distance** — treat each matrix as a grid of numbers and take the Euclidean distance
  between the two grids. Lower = more similar. **[Inference]**, a generic linear-algebra
  distance, not a citation-bearing method.
- **Leading eigenvector overlap** — compare only the single most dominant remaining pattern
  (after detoning: no longer the market mode, since that was removed; the next-most-dominant
  mode) between the two horizons, via squared cosine similarity of the eigenvectors (subspace
  overlap for the top `n_top` eigenvectors in general). Bounded in [0, 1]; 1 = identical
  direction. **[Inference]**, standard linear-algebra subspace-overlap definition.

Both diffs are defined so that a **positive** value supports H1 (the matrices/eigenvectors become
*more* alike after detoning).

### 2.3 Uncertainty: stationary bootstrap

A single point estimate of "did detoning help" could be noise. Confidence intervals are built via
the stationary bootstrap: resample blocks of rows (block lengths drawn from a geometric
distribution, with circular wrap-around), which — unlike an ordinary i.i.d. bootstrap — respects
the fact that daily/weekly/monthly returns are serially dependent (in particular, dependent in
volatility), so naively resampling individual rows would understate the true uncertainty.

- **Politis & Romano (1994)**, "The Stationary Bootstrap," *Journal of the American Statistical
  Association* 89(428), 1303–1313, DOI 10.1080/01621459.1994.10476870. **[CHECKED via WebSearch,
  this session]** — the source method for the block-resampling scheme itself.
- **Politis & White (2004)**, automatic block-length selection. **[UNVERIFIED, NOT
  IMPLEMENTED]** — this project uses a simple heuristic default instead
  (`max(2.0, round(n_obs ** (1/3)))`), disclosed as **[Inference]**, not the validated method.
- **Efron & Tibshirani**, *An Introduction to the Bootstrap* — the percentile method used to turn
  a distribution of bootstrap replicates into a confidence interval. **[UNVERIFIED, cited from
  memory, edition unchecked]**.
- Confidence intervals are computed independently per horizon (each horizon's returns resampled
  on its own, not jointly), a stated conservative simplification: real D/W/M returns share the
  same underlying daily data, and a joint scheme would likely narrow the intervals rather than
  widen them. Not yet implemented.

### 2.4 What the additional ("signal") eigenvalues might represent

Step 3.1's own real-data check found more than one eigenvalue above the Marchenko-Pastur noise
edge: 4 signal eigenvalues at the daily horizon, 2 at quarterly (core-23 panel) — i.e., detoning
with `n_market=1` removes only the single largest of several non-noise patterns.

- **Plerou, Gopikrishnan, Rosenow, Amaral, Guhr & Stanley (2002)**, "Random Matrix Approach to
  Cross-Correlations in Financial Data," *Physical Review E* 65, 066126. **[CHECKED, this
  session]** — beyond the market mode, the next several eigenvectors "contain stocks belonging to
  similar or related industries as significant contributors"; they identify each eigenvector's
  economic meaning by inspecting which assets have the largest-magnitude loadings on it (in their
  data: electronics/semiconductors, gold mining — twice, banking, oil/gas, autos, pharma). This is
  the strongest available source for "what do the 2nd/3rd/4th eigenvalues mean."
- Laloux et al. (1999/2000) **[CHECKED]** only characterizes the top (market) eigenvalue; it
  should not be cited for the sector-interpretation claim specifically.

**Confirmed on this project's own data (2026-09-28, `scripts/eigenvector_loadings_3_2.py`,
sanity-tested on a synthetic sector-clustered panel before running on real data).** Real core-23
daily and quarterly correlation matrices, top-8-by-|loading| tickers per eigenvector:

*Daily* (3 signal eigenvalues beyond market, all above the Marchenko-Pastur edge 1.1767,
matching step 3.1's `n_signal=4`):

| Rank | Eigenvalue (% of trace) | Dominant cluster | Top loadings |
|---|---|---|---|
| 2 | 2.8207 (12.3%) | **metals_mining** | GDX +0.499, GDXJ +0.489, SIL +0.480, XME +0.193, COPX +0.174 (vs. healthcare XLV/IHI/IBB, −0.16 to −0.19) |
| 3 | 2.1954 (9.5%) | **energy_trad** | XOP +0.419, XLE +0.416, OIH +0.405, AMLP +0.350 (vs. healthcare IBB/XBI/IHI and ai_semis IGV, −0.18 to −0.24) |
| 4 | 1.2537 (5.5%, barely above the 1.1767 edge) | mixed style axis | healthcare (XLV +0.40, IBB +0.29, IHI +0.28) vs. clean_energy+ai_semis (TAN/SOXX/SMH/ICLN/LIT, −0.28 to −0.33) |

*Quarterly* (only 1 signal eigenvalue beyond market, matching step 3.1's `n_signal=2`; ranks 3-4
fall below the — much wider, T=49-driven — Marchenko-Pastur edge 2.8396 and should not be read as
real structure):

| Rank | Eigenvalue (% of trace) | Above MP edge? | Dominant cluster | Top loadings |
|---|---|---|---|---|
| 2 | 3.0289 (13.2%) | Yes | **energy_trad** | XLE +0.334, XOP +0.288, OIH +0.266 (vs. ai_semis IGV/SOXX/SMH and healthcare IBB/XBI, −0.25 to −0.30) |
| 3 | 2.2902 (10.0%) | No (noise-level at this T) | metals_mining vs. energy_trad | GDX/SIL/GDXJ +0.42 to +0.44 vs. AMLP/XLE/OIH/XOP ≈ −0.22 |
| 4 | 1.3077 (5.7%) | No (noise-level) | ai_semis vs. mixed | SMH/IGV/SOXX +0.29 to +0.34 vs. TAN/IBB/XLV/XBI −0.28 to −0.32 |

**Interpretation, connecting directly to the H1 result (§4-5): [Inference], grounded in this
real-data check, not just speculation.** H1's `n_market=1` test compares the *leading residual
eigenvector* (original rank 2) across horizons after detoning. At the daily horizon that
eigenvector is a **metals_mining** cluster; at the quarterly horizon it is an **energy_trad**
cluster. These are genuinely different economic stories, not the same sector structure merely
reshuffled in magnitude — a different sector actually dominates the residual at each horizon
(energy is present at both, but only overtakes metals at quarterly, where metals itself drops to a
noise-level rank). This gives a concrete, data-grounded mechanism for the low, statistically
significant post-detoning eigenvector overlap found in §4: it is not that sector structure
disappears after removing the market mode, but that *which* sector dominates the leading residual
mode changes with sampling frequency — exactly the kind of horizon-dependence H1 asked whether
detoning would remove, and on this evidence it does not.

## 3. Real-data result (uncorrected)

Run: `scripts/exploratory_3_2.py`, core-23 panel (`config/universe.yaml`, `panel: core`),
Pearson correlation, `n_market=1`, `n_bootstrap=2000`, seed=0, 2026-09-28. Wall-clock 35.28s.

| Pair | Frobenius before → after (diff) | Frobenius 95% CI | Eigvec overlap before → after (diff) | Eigvec 95% CI |
|---|---|---|---|---|
| D–W | 0.7564 → 0.8564 (−0.1000) | [−0.6500, +1.2304] | 0.9982 → 0.9440 (−0.0542) | [−0.9743, −0.0082]* |
| D–M | 1.3575 → 1.6001 (−0.2426) | [−0.9591, +1.2379] | 0.9950 → 0.9535 (−0.0415) | [−0.9815, −0.0191]* |
| W–M | 1.0964 → 1.5482 (−0.4518) | [−1.1017, +1.2261] | 0.9979 → 0.9639 (−0.0340) | [−0.9857, −0.0285]* |

(* = CI excludes zero, uncorrected alpha = 0.05.)

Both metrics move in the direction **opposite** H1 at every horizon pair (negative diff = matrices
less similar after detoning, not more). The Frobenius CIs are all wide and include zero —
inconclusive on their own, and consistent with a separately disclosed problem with this metric
(§5). The eigenvector-overlap CIs exclude zero at all three pairs, uncorrected.

## 4. Real-data result (multiplicity-corrected)

Three simultaneous horizon-pair comparisons were tested; without correction, a "significant" CI
is more likely to appear by chance. Run: `scripts/h1_table_3_2.py`, same run as §3 (identical
seed and inputs), computing a two-sided percentile bootstrap p-value directly from each test's
stored replicate distribution (p = 2 · min(P(replicate ≤ 0), P(replicate ≥ 0)), **[Inference /
standard practice]**, not itself a citation-specific formula), then applying Holm-Bonferroni
correction (**Holm (1979)**, "A Simple Sequentially Rejective Multiple Test Procedure,"
*Scandinavian Journal of Statistics* 6(2), 65–70 — **[UNVERIFIED this session, standard citation
from memory]**) at alpha = 0.05.

| Test | Raw p | Holm-significant (within-metric, m=3) | Holm-significant (all 6 tests, m=6) |
|---|---|---|---|
| Frobenius D–W | 0.975 | No | No |
| Frobenius D–M | 0.531 | No | No |
| Frobenius W–M | 0.513 | No | No |
| Eigvec D–W | 0.004 | **Yes** | **Yes** |
| Eigvec D–M | 0.001 | **Yes** | **Yes** |
| Eigvec W–M | 0.001 | **Yes** | **Yes** |

The eigenvector-overlap finding survives even the more conservative correction (treating all 6
tests, both metrics, as one family), not just the more lenient within-metric one. The Frobenius
metric shows essentially no signal once tested properly (p's near 0.5–1.0).

## 4A. Robustness check: n_market = n_signal (exploratory)

Motivated by §5 caveat 2 and §6: `n_market=1` removes only the single largest eigenmode, but step
3.1's real-data spectrum found more non-noise ("signal") eigenvalues than that at every horizon.
This checks whether detoning by each horizon's own detected signal count — rather than always
just 1 — changes the H1 verdict.

**API constraint and design choice** (stated in the script's own docstring,
`scripts/h1_nmarket_signal_3_2.py`): `bootstrap_horizon_pair_similarity` takes one `n_market`
shared by both horizons in a pair — there is no per-horizon-asymmetric detoning in the current
API. For each pair, `n_market_pair = max(n_signal[horizon_a], n_signal[horizon_b])` was used — the
more conservative (larger) of the two horizons' own signal counts — which also sets `n_top`
(eigenvector overlap compares exactly the modes being removed, per `horizon_pair_similarity`'s own
default).

Run: `scripts/h1_nmarket_signal_3_2.py`, core-23 panel, Pearson, `n_bootstrap=2000`, 2026-09-28.
Wall-clock 35.51s.

**Step 1 — real n_signal per horizon** (core-23 panel; adds W and M to the D=4/Q=2 figures already
quoted in §2.4/§6 — quarterly is outside H1's D/W/M scope and is unaffected):

| Horizon | T | MP edge | n_signal |
|---|---|---|---|
| D | 3201 | 1.18 | 4 |
| W | 664 | 1.41 | 3 |
| M | 151 | 1.93 | 3 |

**Step 2 — n_market_pair used:**

| Pair | n_signal(a) | n_signal(b) | n_market_pair |
|---|---|---|---|
| D–W | 4 | 3 | 4 |
| D–M | 4 | 3 | 4 |
| W–M | 3 | 3 | 3 |

**Step 3 — result (uncorrected):**

| Pair | Frobenius before → after (diff) | Frobenius 95% CI | Eigvec overlap before → after (diff) | Eigvec 95% CI |
|---|---|---|---|---|
| D–W | 0.7564 → 2.9694 (−2.2131) | [−4.9567, +0.0902] | 0.9520 → 0.9211 (−0.0308) | [−0.1129, −0.0185]* |
| D–M | 1.3575 → 2.6496 (−1.2920) | [−3.5381, −0.2264]* | 0.9746 → 0.9138 (−0.0609) | [−0.2140, −0.0555]* |
| W–M | 1.0964 → 1.9147 (−0.8183) | [−3.4818, +0.2913] | 0.9844 → 0.9535 (−0.0309) | [−0.3570, −0.0460]* |

(* = CI excludes zero, uncorrected alpha = 0.05.)

**Holm-Bonferroni correction (alpha = 0.05), same two families as §4:**

| Test | Raw p | Holm-significant (within-metric, m=3) | Holm-significant (all 6, m=6) |
|---|---|---|---|
| Frobenius D–W | 0.061 | No | No |
| Frobenius D–M | 0.031 | No | No |
| Frobenius W–M | 0.090 | No | No |
| Eigvec D–W | 0.001 | **Yes** | **Yes** |
| Eigvec D–M | 0.000 | **Yes** | **Yes** |
| Eigvec W–M | 0.000 | **Yes** | **Yes** |

Frobenius D–M's raw p = 0.031 looks nominally below 0.05 but fails Holm's own first, most
stringent hurdle (0.05/3 = 0.0167), so none of the three Frobenius tests survive correction — the
same qualitative outcome as §4.

**Interpretation.** The H1 verdict is **unchanged**: at `n_market=n_signal` as at `n_market=1`,
the eigenvector-overlap metric finds correlation structure *less* similar across horizons after
detoning, not more — opposite H1's predicted direction — and this remains significant at all
three pairs under both the within-metric and the more conservative all-6-tests Holm correction.
This robustness check does not rescue H1; if anything it reinforces §5's verdict by showing it is
not an artifact of the specific `n_market=1` choice.

Two things did change from §3, and both are flagged rather than smoothed over:

- `frobenius_before` is unchanged from §3 (0.7564/1.3575/1.0964, as expected — "before" does not
  depend on `n_market`), but `frobenius_after` grew substantially once more eigenmodes are
  subtracted (e.g. D–W's after-distance rose from 0.8564 at `n_market=1` to 2.9694 here), making
  the Frobenius diffs much larger in magnitude than at `n_market=1`. This is consistent with, but
  not new confirmation of, the Frobenius null-bias problem already disclosed in §5 caveat 1: that
  diagnostic was only validated on synthetic null panels at `n_market=1`. Whether the same
  renormalization artifact scales with `n_market`, or something else is happening, has not been
  separately tested — **new open item, added to §6.**
- `eigvec_before` (now a 3–4-dimensional subspace overlap, not just the single leading
  eigenvector, since `n_top` tracks `n_market`) dropped from the near-ceiling 0.995–0.998 range in
  §3 to 0.952–0.984 here — still high, but a real move off the ceiling §5 caveat 6 flagged. The
  `eigvec_diff` magnitudes changed non-monotonically versus §3 (D–W shrank from −0.0542 to
  −0.0308; D–M grew from −0.0415 to −0.0609; W–M was roughly unchanged, −0.0340 to −0.0309) — no
  clean pattern of "more detoning changes the result more," which argues against treating either
  specification as more or less conservative than the other on this axis.

Given the known, and now possibly larger, Frobenius artifact, the eigenvector-overlap metric
remains the more trustworthy one here, exactly as in §5 — and it continues to reject H1's
predicted direction at this specification too.

## 4B. Null-bias calibration check (methodology cross-check)

**Full detail, method and additional findings: `docs/METHOD_ROBUSTNESS.md`.** Summarized here
because it bears directly on how much to trust §3-4A's verdict, not just on methodology in the
abstract.

Every CI reported in §3-4A is the real data's own bootstrap CI — it says whether *that* interval
excludes 0, but says nothing about how the point estimate compares to what pure noise would
produce at this project's actual dimensions. This gap is closed by benchmarking against a
matched-dimension null: independent noise panels (population correlation = identity) at N=23 and
the exact real row counts (D=3201, W=664, M=151), at both `n_market` specifications actually used.

| Pair | Spec | Null mean eigvec diff | Real eigvec diff | Gap |
|---|---|---|---|---|
| D–W | n_market=1 | +0.0064 | −0.0542 | 0.0606 |
| D–M | n_market=1 | +0.0235 | −0.0415 | 0.0650 |
| W–M | n_market=1 | −0.0276 | −0.0340 | 0.0064 |
| D–W | n_market=n_signal | −0.0109 | −0.0308 | 0.0199 |
| D–M | n_market=n_signal | +0.0214 | −0.0609 | 0.0823 |
| W–M | n_market=n_signal | −0.0123 | −0.0309 | 0.0186 |

The real eigenvector-overlap finding is more negative than the matched null-noise mean at every
one of the six pair/specification combinations — at the real project's own N and T, this null's
own CI never excludes 0 (well-calibrated, no inflated false-positive rate). This is genuine,
previously-missing corroboration of §3-4A's verdict, not just a restatement of it. **One
qualification, not smoothed over**: the W–M gap is thin (0.006–0.019, versus 0.06–0.08 for the
pairs involving D), so W–M's significance rests more on its own bootstrap CI than on standing
clearly apart from this null floor — worth keeping in mind before leaning on W–M alone in any
future argument. The same study also re-confirms, at real dimensions, that the Frobenius null
bias grows substantially worse at larger `n_market` (from roughly −0.4 to −0.8 at `n_market=1` to
−1.2 to −2.3 at `n_market=n_signal`), closing the open item §4A raised.

## 5. Verdict and caveats

**Verdict, stated plainly per this project's "report results plainly when they disappoint"
rule**: at the pre-registered specification (`n_market=1`, Pearson, core-23 panel), **H1 is not
supported**. The eigenvector-overlap metric shows a statistically robust effect in the *opposite*
direction from what H1 predicts — cross-horizon similarity of the leading residual eigenvector
*decreases*, not increases, after market-mode removal — and this holds up under multiple-
comparison correction. This is a formal confirmatory result for this one specification, not a
final scientific verdict on the underlying question (see open items below).

**Caveats, disclosed rather than used to rescue the result:**

1. **The Frobenius metric has a known null bias, confirmed to worsen with `n_market`.** Testing on
   pure-noise synthetic panels (no true market mode to remove) found `frobenius_diff`'s bootstrap
   CI systematically biased *negative* in all 5 tested seeds — the opposite of the "no effect" a
   null should show — diagnosed as a renormalization artifact of comparing matrices after removing
   a noise-inflated top eigenvalue, not a real effect. `docs/METHOD_ROBUSTNESS.md` §1-2 (2026-09-28)
   confirms this bias grows substantially larger as `n_market` increases, both on the legacy N=8
   fixture and at the real project's own N=23/T dimensions — so the larger Frobenius diffs seen at
   `n_market=n_signal` in §4A are consistent with a worse artifact, not evidence of a stronger real
   effect. This is why the Frobenius result here (also negative, but not significant) is read as
   uninformative rather than as independent confirmation of "no support for H1." The
   eigenvector-overlap metric, by contrast, was validated as well-calibrated under the same
   null-noise test (its CI mostly contained zero, including at real project dimensions — §4B), and
   its point estimate on real data clears the matched null-noise floor at every pair/specification
   tested, which is part of why it is treated as the more trustworthy metric here.
2. **Only one specification is confirmatory.** `n_market=1` removes exactly the single largest
   eigenmode, as H1 literally states ("the first eigenmode") — that remains the pre-registered
   result in §3-4. Step 3.1's own real-data check found 4 signal eigenvalues at the daily horizon
   and 2 at quarterly — i.e., more non-noise structure exists than a single mode captures. The
   `n_market = n_signal` **exploratory** robustness check is now run (§4A): it leaves the verdict
   unchanged (eigenvector overlap still significantly opposes H1's direction at every pair), but
   it is reported and read as exploratory, not substituted for the confirmatory §3-4 result.
3. **No correction across H1–H6.** `config/experiment.yaml`'s `inference.multiplicity: holm` is a
   *cross-hypothesis* correction (across all six pre-registered hypotheses), deferred to the
   Phase 8 scoreboard (not built yet). The correction in §4 is *within* H1 only (across its own 3
   horizon-pair tests); it does not substitute for that eventual, larger correction.
4. **Independent, not joint, bootstrap resampling.** A conservative simplification (real D/W/M
   returns share the same underlying daily data; independent resampling likely widens rather than
   narrows the reported intervals), so this is not a reason the eigenvector-overlap significance
   might be spurious — if anything, a joint scheme might sharpen it further, though this has not
   been tested.
5. **Single whole-sample point estimate, no regime conditioning.** This result does not
   distinguish calm from high-volatility periods; that is H2's job (not yet built).
6. **[Inference], now grounded in the real-data eigenvector-loading check in §2.4, not just
   speculation**: the *before*-detoning eigenvector overlap is already very high at every pair
   (0.995–0.998) — a ceiling effect, consistent with the raw leading eigenvector (the market
   factor) already being highly stable across sampling horizons before any cleaning. Detoning
   removes exactly that stable component and compares whichever mode is *next* most dominant in
   the residual instead — §2.4 shows this is a **metals_mining** cluster at the daily horizon but
   an **energy_trad** cluster at the quarterly horizon: a different sector genuinely dominates the
   residual at each horizon, which is a concrete, data-grounded mechanism for the observed
   direction, not merely a plausible story.

## 6. Open follow-ups (not yet run)

- ~~Inspect real eigenvector loadings (2nd–4th eigenvalues)~~ **Done, 2026-09-28** —
  `scripts/eigenvector_loadings_3_2.py`; results and interpretation in §2.4 above.
- ~~`n_market = n_signal` sensitivity check (exploratory)~~ **Done, 2026-09-28** —
  `scripts/h1_nmarket_signal_3_2.py`; results and interpretation in §4A above. Verdict unchanged
  (eigenvector overlap still significantly opposes H1 at every pair under both Holm families).
- ~~Whether the Frobenius null-bias diagnostic scales with `n_market`~~ **Done, 2026-09-28** —
  `docs/METHOD_ROBUSTNESS.md` §1-2: confirmed, grows roughly linearly worse with `n_market` both
  at the legacy N=8 fixture and at real project dimensions.
- ~~Null-bias calibration of the eigenvector-overlap finding at real project dimensions~~ **Done,
  2026-09-28** — §4B above / `docs/METHOD_ROBUSTNESS.md` §2: real finding clears the matched-null
  floor at all six pair/specification combinations, though thinly at W-M.
- ~~Whether joint (shared-calendar) bootstrap resampling would narrow the CI, as the module
  docstring predicts~~ **Partially done, 2026-09-28** (synthetic test, not implemented in the
  library) — `docs/METHOD_ROBUSTNESS.md` §3: confirmed for Frobenius, but the eigenvector-overlap
  CI width was statistically indistinguishable between joint and independent resampling on the
  tested synthetic panel, so building the real joint-resampling feature is not obviously
  worthwhile purely for this metric. The library itself still only does independent resampling;
  actually implementing joint resampling (reaching into `etfcorr.returns`) remains not done.
- Formal H1–H6 cross-hypothesis Holm correction, once H2–H6 have results (Phase 8 scoreboard).
- Politis & White (2004) automatic block-length selection, replacing the current heuristic
  default — coverage sensitivity to block length also not yet tested even on synthetic data (see
  `docs/METHOD_ROBUSTNESS.md` §4).
- A joint-dependence null-noise floor (matching §4B's design but with both horizons drawn from one
  shared daily path, not independent panels) — `docs/METHOD_ROBUSTNESS.md` §2's own caveat.
- Signal/noise eigenvalue classification (`eigen_spectrum`, fixed `sigma2=1.0`) under fat-tailed
  or volatility-clustered returns, rather than only the i.i.d. Gaussian synthetic panels used so
  far — `docs/METHOD_ROBUSTNESS.md` §4.

## 7. Provenance

- Code: `src/etfcorr/correlation/horizon_analysis.py` (LOCAL-PASSED, `docs/checks/3.2.md`),
  functions `h1_cross_horizon_test`, `horizon_pair_similarity`, `bootstrap_horizon_pair_similarity`,
  `frobenius_distance`, `leading_eigenvector_overlap`.
- Scripts: `scripts/exploratory_3_2.py` (uncorrected table, §3), `scripts/h1_table_3_2.py`
  (Holm-corrected table, §4), `scripts/eigenvector_loadings_3_2.py` (real eigenvector loadings,
  §2.4), `scripts/h1_nmarket_signal_3_2.py` (n_market=n_signal robustness check, §4A). All
  restrict to the core-23 panel (`config/universe.yaml`, `panel: core`); all exploratory, not part
  of Gate 1/Gate 2. Run by Filippo on his own machine, all four scripts, 2026-09-28; output
  reproduced verbatim in §3–4 and §4A above (no fabricated numbers, per this project's evidence
  rules).
- §4B's null-calibration check: `scripts/frobenius_bias_nmarket_3_2.py`,
  `scripts/null_calibration_real_dims_3_2.py`, `scripts/bootstrap_joint_vs_independent_3_2.py`
  (full detail and additional findings in `docs/METHOD_ROBUSTNESS.md`). Synthetic-data-only,
  executed directly in this session (not by Filippo), reproducible from a clean checkout given the
  fixed seeds — no real data or repository-specific environment involved.
- Data: `data/processed/returns_{D,W,M}.parquet` (step 2.1, LOCAL-PASSED) for §3-4A;
  `etfcorr.synthetic` (no real data) for §4B.
