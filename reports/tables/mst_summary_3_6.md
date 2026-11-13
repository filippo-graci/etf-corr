# MST figure and network check (step 3.6)

Core panel, N = 25. D: T=2265 (2014-01-03 to 2022-12-30); W: T=469 (2014-01-10 to 2022-12-30); M: T=107 (2014-02-28 to 2022-12-31); Q: T=35 (2014-06-30 to 2022-12-31).
Stationary bootstrap, 2000 draws, mean block lengths D: 13; W: 8; M: 5; Q: 3; permutation nulls 2000; rolling row-shuffles 200; noise-floor draws 1000; seed 0.
The sealed test window is not read (last date used: 2022-12-31).

Descriptive and exploratory, not a test of H1-H6. An MST keeps N - 1 of the N(N - 1)/2 correlations: a missing edge is not a zero correlation. Support is the support under a stationary world. Detoned matrices are relative (cross-sector entries negative by construction). Quarterly rows are exploratory (T close to N). P-values are not corrected for multiplicity because none is a confirmatory test; sector labels are a benchmark, not truth, and holdings overlap makes some within-sector agreement mechanical.

## Trees, edge support, sector purity and cluster agreement

purity = share of edges joining two ETFs of the same sector (null: sector labels permuted over nodes); ari = adjusted Rand index of an average-linkage clustering against the sector labels; support = bootstrap share of replicates containing each edge; survival = edge-survival ratio of a replicate tree against the point tree.

| horizon | matrix | n_obs | purity | purity_null_mean | purity_p | ari | ari_null_mean | ari_p | mean_support | share_support_ge_half | survival_mean | survival_lower | survival_upper | max_degree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D | raw | 2265 | 0.750 | 0.142 | 0.0005 | +0.402 | +0.001 | 0.0005 | 0.885 | 0.88 | 0.885 | 0.792 | 0.958 | 5 |
| D | detoned | 2265 | 0.750 | 0.143 | 0.0005 | +0.620 | +0.001 | 0.0005 | 0.884 | 1.00 | 0.884 | 0.792 | 0.958 | 4 |
| W | raw | 469 | 0.750 | 0.143 | 0.0005 | +0.376 | +0.001 | 0.0005 | 0.885 | 0.96 | 0.885 | 0.792 | 1.000 | 4 |
| W | detoned | 469 | 0.750 | 0.142 | 0.0005 | +0.783 | +0.002 | 0.0005 | 0.883 | 0.96 | 0.883 | 0.792 | 0.958 | 3 |
| M | raw | 107 | 0.750 | 0.144 | 0.0005 | +0.591 | +0.002 | 0.0005 | 0.848 | 0.92 | 0.848 | 0.750 | 0.958 | 4 |
| M | detoned | 107 | 0.750 | 0.143 | 0.0005 | +0.783 | +0.002 | 0.0005 | 0.830 | 0.88 | 0.830 | 0.708 | 0.958 | 3 |
| Q | raw | 35 | 0.750 | 0.143 | 0.0005 | +0.371 | +0.001 | 0.0005 | 0.748 | 0.75 | 0.748 | 0.625 | 0.875 | 3 |
| Q | detoned | 35 | 0.708 | 0.143 | 0.0005 | +0.572 | +0.001 | 0.0005 | 0.743 | 0.75 | 0.743 | 0.625 | 0.875 | 4 |

## Trees of different horizons against the noise floor

survival = shared edges / (N - 1); noise = two independent pure-noise panels (mean 2/N exactly by exchangeability, 95th percentile simulated at the smaller T of the pair, raw Pearson; for detoned trees the mean is the same and the percentile approximate).

| matrix | horizon_a | horizon_b | survival | noise_mean | noise_q95 | above_noise_q95 |
|---|---|---|---|---|---|---|
| raw | D | W | 0.833 | 0.080 | 0.167 | yes |
| raw | D | M | 0.750 | 0.080 | 0.167 | yes |
| raw | D | Q | 0.708 | 0.080 | 0.167 | yes |
| raw | W | M | 0.875 | 0.080 | 0.167 | yes |
| raw | W | Q | 0.833 | 0.080 | 0.167 | yes |
| raw | M | Q | 0.875 | 0.080 | 0.167 | yes |
| detoned | D | W | 0.792 | 0.080 | 0.167 | yes |
| detoned | D | M | 0.750 | 0.080 | 0.167 | yes |
| detoned | D | Q | 0.625 | 0.080 | 0.167 | yes |
| detoned | W | M | 0.750 | 0.080 | 0.167 | yes |
| detoned | W | Q | 0.667 | 0.080 | 0.167 | yes |
| detoned | M | Q | 0.750 | 0.080 | 0.167 | yes |

## Rolling-window persistence against row-shuffled returns

real_mean = mean edge-survival ratio of consecutive trailing-window trees; the null shuffles rows (overlap kept, serial and regime structure destroyed). lower_tail = share of shuffles at or below the real value (add-one); windows overlap, so these are empirical positions, not significance claims.

| horizon | matrix | window | step | n_windows | real_mean | null_mean | null_min | null_max | lower_tail | upper_tail |
|---|---|---|---|---|---|---|---|---|---|---|
| D | raw | 252 | 21 | 95 | 0.924 | 0.920 | 0.900 | 0.935 | 0.7463 | 0.2786 |
| D | detoned | 252 | 21 | 95 | 0.916 | 0.920 | 0.900 | 0.937 | 0.2935 | 0.7413 |
| W | raw | 104 | 4 | 91 | 0.939 | 0.940 | 0.920 | 0.956 | 0.4328 | 0.5920 |
| W | detoned | 104 | 4 | 91 | 0.930 | 0.936 | 0.918 | 0.955 | 0.2139 | 0.8109 |
