# Method robustness — synthetic studies on the H1 bootstrap/detoning engine

Phase 3, step 3.2 follow-up. 2026-09-28. Status: exploratory, not part of Gate 1/Gate 2 — these
studies validate the *methodology* behind `src/etfcorr/correlation/horizon_analysis.py`'s bootstrap
engine, which is reused as-is by every future confirmatory hypothesis that calls
`bootstrap_horizon_pair_similarity` (H1 today; potentially others later), not just the H1 result
itself. All three studies use only synthetic data with a known ground truth
(`etfcorr.synthetic`), so no real (vendor) data or Filippo's local environment was needed — every
number below was produced by code run directly in this session (see each study's own script and
this doc's provenance section), reproducible from a clean checkout given the fixed seeds.

**Why this exists.** `docs/H1_FINDINGS.md` documents the H1 result itself. This file documents
three targeted checks on the *engine* that result depends on: whether the already-disclosed
Frobenius null bias gets worse at higher `n_market` (motivated by `docs/H1_FINDINGS.md` §4A);
whether the real H1 finding is distinguishable from a null-noise floor measured at the project's
actual dimensions, not just a legacy N=8 test fixture; and whether the module's own claim that
independent per-horizon resampling is "conservative" (only widens, never narrows, CIs relative to
a joint scheme) holds for both metrics equally.

## 1. Does the Frobenius null bias scale with `n_market`, and does eigenvector overlap have one too?

**Script:** `scripts/frobenius_bias_nmarket_3_2.py`. **Method:** the existing regression tests'
own null-noise fixture (`etfcorr.synthetic.make_noise_returns(500, 8, seed)`, "W" a 5-day
subsample of "D", population correlation = identity, so any nonzero `frobenius_diff`/`eigvec_diff`
is by construction a false positive), extended across `n_market ∈ {1,...,5}`, 15 seeds,
`n_bootstrap=150`, `alpha=0.10`.

| n_market | mean Frobenius diff | Frobenius excl.% | mean eigvec diff | eigvec excl.% |
|---|---|---|---|---|
| 1 | −1.0582 | 100% | −0.1358 | 0% |
| 2 | −1.4146 | 100% | −0.2055 | 0% |
| 3 | −1.9688 | 100% | −0.1593 | 0% |
| 4 | −2.7637 | 100% | −0.0171 | 0% |
| 5 | −3.6674 | 100% | −0.0964 | 0% |

(excl.% = fraction of seeds whose 90% CI excludes 0; a well-calibrated CI should exclude 0 in
about 10% of seeds by chance.)

**Findings.** (1) The Frobenius null bias, already known at `n_market=1` (module docstring,
"Known limitation, empirically measured"), grows monotonically worse as `n_market` increases —
roughly linear, from −1.06 at `n_market=1` to −3.67 at `n_market=5` — confirming the concern
raised in `docs/H1_FINDINGS.md` §4A as more than speculation. **[Supported by this synthetic
study]**. (2) A previously uncharacterized nuance: the eigenvector-overlap metric's *point
estimate* is also consistently negative under this null (−0.02 to −0.21 across `n_market`
values), even though its CI essentially never excludes 0 (0% here, vs. the existing test's
looser "≥4/5 seeds" bar) — the existing test suite only ever checked CI-exclusion frequency, never
the point estimate's magnitude, so this small negative bias was not previously visible.
**[New finding, this study]**. It does not scale monotonically with `n_market` in this fixture
(peaks at `n_market=2`, not at the largest value), unlike the Frobenius bias.

**Caveat.** N=8, T=500/100 is the legacy unit-test fixture, not the project's real dimensions —
see study 2 for the dimension that matters for interpreting the actual H1 result.

## 2. Null-bias calibration at the real project's dimensions (N=23, real D/W/M row counts)

**Script:** `scripts/null_calibration_real_dims_3_2.py`. **Method:** independent noise panels
(population correlation = identity, no shared draw between the two panels — closer to how the
real D/W/M return panels are actually produced, independently estimated from resampled prices,
not literally subsampled from each other) at N=23 and the *exact* row counts used in the real H1
run (D=3201, W=664, M=151), at both `n_market` specifications `docs/H1_FINDINGS.md` actually
reports: `n_market=1` (§3-4) and the §4A `n_market=n_signal` values (D-W=4, D-M=4, W-M=3). 12
seeds, `n_bootstrap=250`, `alpha=0.10`.

**`n_market=1`:**

| Pair | T_a, T_b | null mean Frobenius diff | Frobenius excl.% | null mean eigvec diff | eigvec excl.% | real eigvec diff (§3) |
|---|---|---|---|---|---|---|
| D–W | 3201, 664 | −0.7808 | 100% | +0.0064 | 0% | −0.0542 |
| D–M | 3201, 151 | −0.4625 | 100% | +0.0235 | 0% | −0.0415 |
| W–M | 664, 151 | −0.4115 | 100% | −0.0276 | 0% | −0.0340 |

**`n_market=n_signal`:**

| Pair | T_a, T_b | n_market | null mean Frobenius diff | Frobenius excl.% | null mean eigvec diff | eigvec excl.% | real eigvec diff (§4A) |
|---|---|---|---|---|---|---|---|
| D–W | 3201, 664 | 4 | −2.3293 | 100% | −0.0109 | 0% | −0.0308 |
| D–M | 3201, 151 | 4 | −1.7254 | 100% | +0.0214 | 0% | −0.0609 |
| W–M | 664, 151 | 3 | −1.1670 | 100% | −0.0123 | 0% | −0.0309 |

**Findings.** (1) At the project's real N and T, the eigenvector-overlap null bias is
substantially *smaller* than the legacy N=8/T=500 fixture suggested — roughly ±0.01 to ±0.03 here,
vs. −0.02 to −0.21 there — and its 90% CI never excludes 0 across any of the six pair/`n_market`
combinations tested (0/12 seeds each), i.e. well-calibrated, no inflated false-positive rate, at
this N/T scale. **[This study]**. (2) The real H1 point estimates are more negative than the
matched null mean at every one of the six combinations, by a margin ranging from small (W–M:
real −0.0340 vs. null −0.0276 at `n_market=1`, a gap of only 0.0064) to substantial (D–M: real
−0.0609 vs. null +0.0214 at `n_market=n_signal`, a gap of 0.083). **This is the first time the H1
eigenvector-overlap result has been benchmarked against a null-noise floor matched to the
project's actual dimensions, rather than judged only by its own bootstrap CI** — and it survives
the comparison at every pair, though the W–M margin is thin enough to flag rather than treat as
comfortably clear. (3) The Frobenius bias is confirmed severe at real dimensions too, and (as in
study 1) worse at the larger `n_market` — from roughly −0.4 to −0.8 at `n_market=1` to roughly
−1.2 to −2.3 at `n_market=n_signal` — reinforcing that this metric should not be used to assess
H1 at either specification.

**Caveat, stated plainly.** This null uses fully *independent* noise panels for each horizon,
which does not reproduce whatever additional dependence exists between real D/W/M panels sharing
the same underlying calendar and economic history (they are independently *estimated*, per step
2.1, but not statistically independent processes). Study 3 addresses a related but distinct
question (does resampling scheme, not the null distribution itself, change CI width) using a
genuinely shared daily-to-weekly generator; a full joint-dependence null-noise floor (matching
study 2's design but with a shared, not independent, daily path underlying both horizons) has not
been built and is a natural next step, not yet run.

## 3. Independent vs. jointly-resampled bootstrap: does joint resampling narrow the CI?

**Script:** `scripts/bootstrap_joint_vs_independent_3_2.py`. **Method:** a synthetic daily panel
(3 sectors of 5 assets, market_loading=0.6, sector_loading=0.5 — the same sanity-tested setup
already used in `scripts/eigenvector_loadings_3_2.py`) with a "weekly" series derived by summing
every 5 consecutive daily log-returns (a simplified, disclosed stand-in for the real pipeline's
Friday-close resampling, not a claim that it reproduces `etfcorr.returns` exactly). Two schemes
compared at `n_market=1`, `n_bootstrap=300`, 8 seeds: **independent** (the library's own
`bootstrap_horizon_pair_similarity`, unmodified — daily and weekly frames block-resampled
separately) vs. **joint** (implemented locally in the script only, never touching
`horizon_analysis.py`: one stationary-bootstrap index drawn over the daily calendar per replicate,
the weekly series re-derived from that *same* resampled daily path, then run through the
library's own `estimate`/`horizon_pair_similarity` functions).

| Metric | mean CI-width ratio (joint / independent) | range across 8 seeds |
|---|---|---|
| Frobenius | 0.802 | 0.634 – 0.969 (narrower in all 8 seeds) |
| Eigenvector overlap | 1.019 | 0.936 – 1.119 (no consistent direction) |

**Findings.** The module docstring's prediction — that joint resampling should narrow the CI
relative to independent resampling, since shared randomness cancels some of the resampling noise
— **holds clearly for the Frobenius metric** (narrower in all 8 tested seeds, ~20% narrower on
average) but **does not hold for the eigenvector-overlap metric**, whose CI width is statistically
indistinguishable between the two schemes on this synthetic panel (ratio scattered symmetrically
around 1.0, both above and below). **[This study, one synthetic panel design, 8 seeds — not
exhaustive]**.

**Practical implication for future evolution.** Since the eigenvector-overlap metric is the one
`docs/H1_FINDINGS.md` actually relies on (Frobenius is separately disqualified by its own null
bias — see section 1-2 above and `docs/H1_FINDINGS.md` §5 caveat 1), and joint resampling does not
measurably change that metric's CI width here, building the real joint-bootstrap feature (which
would mean reaching into `etfcorr.returns`'s resampling logic from inside the bootstrap loop, a
real engineering investment the module docstring explicitly avoided) is **not obviously
worthwhile purely for tightening the H1 eigenvector-overlap result** — though it may still matter
for other future confirmatory hypotheses (H2-H6) that reuse the same engine with different
metrics, which this study does not speak to.

## 4. What was not tested (identified, not yet run)

Stated plainly, not silently skipped, per this project's evidence rules:

- **Fat-tailed / volatility-clustered returns.** All three studies above use i.i.d. Gaussian
  synthetic returns (`etfcorr.synthetic`'s only generator). Real daily returns are fat-tailed and
  volatility-clustered; whether `eigen_spectrum`'s Marchenko-Pastur signal/noise classification
  (fixed `sigma2=1.0`) misclassifies signal vs. noise eigenvalues under those violations — the
  substance of `docs/STATUS.md`'s open "should `sigma2` be KDE-fit" question — was not tested this
  session. Would need a new synthetic generator (e.g. a Student-t or GARCH-like copula on top of
  the existing factor model), not yet built.
- **Stationary-bootstrap block-length sensitivity under serial dependence.** The current
  `default_mean_block_length` heuristic (`round(n**(1/3))`) has not been checked for CI coverage
  accuracy against genuinely autocorrelated/vol-clustered synthetic data (all studies here are
  i.i.d. over time, so block length has no serial structure to capture). Politis & White (2004)'s
  automatic selection remains unimplemented and uncompared.
- **A joint-dependence null-noise floor** (study 2's design, but with both horizons drawn from one
  shared daily path rather than independent panels) — flagged as the natural next step in section
  2's caveat above.
- **H2-H6.** None of these studies say anything about the other five confirmatory hypotheses,
  which reuse parts of this same bootstrap/RMT machinery but are not yet built (Phase 4+).

## 5. Provenance

- Scripts: `scripts/frobenius_bias_nmarket_3_2.py`, `scripts/null_calibration_real_dims_3_2.py`,
  `scripts/bootstrap_joint_vs_independent_3_2.py`. All offline-verified (`ruff check`,
  `ruff format --check`, `py_compile`) and **executed directly in this session** (not by Filippo
  locally) since they use only `etfcorr.synthetic` and already-LOCAL-PASSED library code, no real
  data or repository-specific environment — reproducible on any machine with the pinned
  dependencies given the fixed seeds. All exploratory, not part of Gate 1/Gate 2.
- Depends on: `src/etfcorr/correlation/horizon_analysis.py`, `src/etfcorr/correlation/rmt.py`,
  `src/etfcorr/correlation/estimators.py`, `src/etfcorr/synthetic.py` (all LOCAL-PASSED,
  unmodified).
- Referenced from: `docs/H1_FINDINGS.md` §4B (headline null-calibration result), §6 (open
  follow-ups), `docs/STATUS.md`, `docs/DECISION_LOG.md`.
