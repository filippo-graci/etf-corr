# 8. Our results against the papers

Written 2026-10-19 after reading the full text of 19 of the cited papers (section 8.10, on the follow-ups
E5-E9, added 2026-11-01 after reading 7 more) (`references/README.md` lists which
copies were obtained and from where). Each claim about a paper below was checked in its text, and is given with
the page of the PDF that was read (`p.` = PDF page; for working papers this is not the page of the published
article). Papers that could not be obtained are marked *(abstract only)*. Statements about a paper are
paraphrases; where its setting differs from ours, the difference is stated, because most disagreements turn out
to be differences of design rather than of evidence.

## Summary

| Topic | What the literature leads one to expect | What we found | Verdict of the comparison |
|---|---|---|---|
| Spectrum of the correlation matrix | most eigenvalues are noise; one market mode; a few sector modes | 3-4 of 25 eigenvalues above the noise edge; market mode 53% of variance | **agrees** |
| Sector structure in trees | trees and clusters recover industry groups | 75% of tree edges inside a sector (14% by chance) | **agrees** |
| H1, structure across horizons after removing the market | more similar (Borghesi et al.) | pre-registered eigenvector statistic fell; cluster agreement rose, not significantly | **agrees in direction on the statistic Borghesi et al. used; our test statistic was the wrong tool** |
| H2, correlation in volatile markets | the rise is largely a volatility bias (Forbes and Rigobon) | +0.235 raw, +0.027 adjusted, 88% explained | **agrees** |
| Time-varying correlation models | DCC competitive with other multivariate GARCH; simple exponential smoothing hard to beat | DCC the best simple baseline for next month's correlation; EWMA (0.94) poor | **partly**: different horizon |
| H3, forecasting tests | Clark-West slightly undersized for nested OLS models; unadjusted DM badly undersized | Clark-West 7% at a nominal 5%; DM-HLN 0% | **agrees for DM; Clark-West slightly oversized in our setting (outside their design)** |
| Model Confidence Set | uninformative data give a large set | all 18 models kept | **agrees** (uninformative data) |
| H5, long-only minimum variance | the constraint acts as shrinkage; sample covariance as good as cleaned ones (best, by a small margin, in Ledoit and Wolf's long-only check) | no estimator beats the sample covariance | **agrees** |
| H5, long-short, nonlinear shrinkage | beats the sample covariance, more as N/T rises | the same pattern; our gains 40-70% larger at matched N/T | **agrees in pattern and order of magnitude** |
| Eigenvalue clipping in portfolios | noise in small eigenvalues spoils minimum-risk portfolios (Laloux, stated, not tested); filtering improves risk *prediction* (Plerou, a different filter) | clipping raised *realised* long-short volatility | **no contradiction**: different filter and different measure |
| H6, hedging in stress | correlations measured on volatile days are biased up | the stress gap disappears after the adjustment | **agrees** (an application of Forbes and Rigobon to hedging) |
| Lead-lag | large stocks lead small ones | no lead-lag among thematic ETFs after FDR control | **differs**, plausibly by design |

## 8.1 The spectrum and the market mode (file 02)

**Laloux et al. (1999)**, arXiv version. 406 S&P 500 stocks, daily, 1991-1996 (T = 1309). The largest eigenvalue
is about 25 times the noise edge and is the market; after fitting the noise level, about 94% of the spectrum is
consistent with a random matrix and about 6% lies above it (p. 2).
**Ours:** 25 ETFs, 2014-2022 (T = 2265): 22 of 25 eigenvalues inside the band (88%), 3 above it; the market
eigenvalue is 13.3, about 11 times the edge of 1.22. **Agrees.** The proportion of signal is higher here because
25 sector baskets carry more common structure per asset than 406 single stocks, and the market multiple is
lower because N is smaller. Note that Laloux et al. fit the noise variance (0.74 rather than 1); `rmt.py` uses
1, a difference already recorded as an open item in `status/phase_3.md`.

**Plerou et al. (2002)**, arXiv version. Most eigenvalues fall within the random-matrix bounds (p. 1); the
largest is an influence common to all stocks; the other deviating eigenvectors group stocks by industry, one of
them gold mining (p. 8).
**Ours:** the second daily eigenvector loads on the gold and silver miners (GDX, GDXJ, SIL). **Agrees**, down to
the sector.

**Bollerslev (1986)**, journal version from the author's page. Defines GARCH(p, q) and studies GARCH(1,1) in
detail (p. 2). The DCC first stage here is a Student-t GARCH(1,1), cross-checked against `arch`.

## 8.2 Trees and sectors (file 02)

**Mantegna (1999)**, arXiv version. Minimum spanning trees of the Dow Jones and S&P 500 stocks, July 1989 to
October 1995 (p. 2); the groups in the tree are, with few exceptions, homogeneous by industry and often by
sub-industry (p. 5).
**Ours:** 75% of edges join funds of the same sector against 14% under random labels, at every horizon.
**Agrees.** Some of our agreement is mechanical: funds of a sector hold overlapping stocks (file 01).
**The distance.** The published EPJ B article defines `d = sqrt(2 (1 - rho))` (equation 2, p. 193 of the
journal, p. 1 of the PDF), as `networks.py` does; the arXiv preprint had `1 - rho^2` (p. 3), which was changed
before publication. The difference matters on detoned matrices, which have negative entries: `1 - rho^2` puts
strongly anti-correlated funds close together, the published distance puts them far apart.

## 8.3 H1: is the structure the same across horizons once the market is removed? (file 02)

**Borghesi, Marsili and Miccichè (2007)**, arXiv version. This is the paper H1 was built on, and its full text
changes how H1 should be read:

- **Horizons.** They compare returns from 5 minutes to one day, on 100 NYSE stocks in 2002 and on London, Paris
  and Milan (p. 1-2). H1 compares one day, one week and one month on 25 ETFs over 2014-2022: a different range
  of horizons, where microstructure effects (the Epps effect they discuss, p. 1) no longer apply.
- **How the market is removed.** They subtract the average market return (set B), or regress each return on
  the average market return (set D) or on a single "optimal" factor fitted to the data (set E), and use the
  residuals (p. 3). They prefer fitting the factor to zeroing the top eigenvalue of the matrix, but note that
  the set-E residuals coincide with subtracting the leading singular-value component of the returns, and
  conclude that removing the market is most meaningful when it sets the market eigenvalue to zero (p. 3). Our
  `detone_corr` zeroes the top eigenmode of the correlation matrix and rescales to unit diagonal: in effect
  their set E, computed on standardised rather than raw returns. The removal method is not where H1 departs
  from the paper; the horizons and the statistic are.
- **How similarity is measured.** They compare **cluster structures** with an information-based overlap and
  find more than 90% overlap across intraday horizons once the market is removed, against much less when it is
  not; the market-removed clusters carry about 80% of the information in the sector labels, constant across
  horizons (p. 5-6). Cluster membership is stable, but "some reshuffling takes place in the order of clusters"
  across horizons (p. 5). They also compare MST degree distributions (p. 7), and find that with 100 stocks the
  daily MSTs of the market-removed sets cannot be told apart from those of random walks; only with 500 stocks
  does structure show (p. 8). They do not compare leading eigenvectors.

**Ours:** the pre-registered statistic, leading-eigenvector overlap, fell after removal. The follow-up's
cluster statistic, the adjusted Rand index of the partitions, rose from 0.36 to 0.77 between daily and monthly,
not significantly. **Agrees in direction on the statistic closest to theirs; the pre-registered statistic was
not the one the paper used.**

**Plerou et al. (2002)** explain why our eigenvector statistic misbehaved. They measured how stable each
deviating eigenvector is over time: the top two or three are stable for long periods, and stability falls as an
eigenvalue approaches the noise edge (p. 2, p. 9). After detoning, the leading eigenvector is the one with
eigenvalue about 3, close to the next ones (2.2, 1.2) and not far above the edge: exactly the unstable kind. Our
simulation (file 02, follow-up E2) showed the same thing directly: with an identical structure at every
horizon, the statistic still drops. Borghesi et al.'s own observation fits the same reading: stable cluster
membership with a shifting order of clusters (p. 5) is what our data show, cluster agreement rising while the
leading residual mode changes sector between horizons (miners daily, energy quarterly). Their p. 8 result is
also a caution for us: if 100 stocks were too few for market-removed trees to differ from noise, statements
about our 25-fund detoned trees are weak evidence.

## 8.4 H2: correlation in volatile markets (file 03)

**Forbes and Rigobon (1999)**, NBER version of the 2002 article. The correlation between two markets measured
when one of them is more volatile is biased upwards, even if the underlying relationship is unchanged, and the
bias can be corrected with the relative increase in that market's variance (p. 15, equations 7-8; the OCR text
of the equations is unreadable, the derivation is in their appendix). After the correction, they find no
contagion in the 1997 East Asian crisis, the 1994 Mexican peso collapse or the 1987 crash, only continued
interdependence (p. 2). The correction assumes no endogeneity and no omitted common shocks (footnote on p. 13).
**Ours:** high-minus-low VIX correlation +0.235 raw, +0.027 adjusted [-0.011, +0.064]; 88% of the rise
explained. **Agrees.** Differences: they condition on the variance of the market where the crisis starts, over
a crisis window; we condition on VIX terciles over twelve years and use each fund's own variance ratio, averaged
over pairs. Their derivation splits the sample into a low- and a high-variance group (p. 14), as we do (bottom
against top tercile), but their empirical tests compare the turmoil period with the full period, turmoil
included (p. 21), which gives a smaller variance ratio and a smaller correction. Their two assumptions carry
over unchanged.

**Longin and Solnik (2001)** *(published article, read 2026-11-02; first written from the abstract)*. Monthly
returns of five markets, 1959-1996 (p. 9). Conditioning a correlation on large moves raises it even when the true
correlation is constant (p. 2); against a null that accounts for this, the correlation of large losses is higher
than normality predicts and that of large gains is not (p. 22). **Not compared directly:** H2 splits regimes by
the VIX level, not by the size or sign of returns; their conditioning warning is the reason H2 needs a volatility
adjustment at all. See 8.10 for E5 and E8.

## 8.5 Time-varying correlation (files 03, 04)

**Engle (2002)**, NYU working-paper version. DCC is estimated in two steps from univariate GARCH fits (p. 29);
in simulations it is often the most accurate of the compared estimators (p. 18, 30). Two details matter here:
the paper estimates only bivariate systems (p. 29), and in its empirical comparison the simple exponential
smoother with decay 0.94 has the fewest rejections at the 1% level (p. 28-29). That ranking counts rejections
in in-sample misspecification tests (remaining ARCH in standardised residuals, value-at-risk hits), not forecast
losses, and Engle adds that the differences are probably "not large enough to be convincing" (p. 29).
**Ours:** DCC with 25 assets is the best simple baseline for next month's average correlation (loss 0.0320);
EWMA with decay 0.94 is poor (0.0411, worse than 95% of noise-fed models). **Partly agrees.** Engle judges
models by how well they fit daily correlation, where an exponential smoother with a one-month memory is
appropriate; our target is the out-of-sample average over the next 21 days, where it over-reacts.

## 8.6 H3: comparing forecasts (file 04)

**Diebold and Mariano (1994)**, NBER version (page scans). Their test allows non-quadratic and asymmetric losses
and forecast errors that are non-Gaussian, non-zero mean and serially correlated (abstract, p. 2). Harvey,
Leybourne and Newbold (1997) *(not obtained)* add the small-sample correction used here.

**Clark and West (2006)**, NBER version. When a larger model nests a smaller one, the larger model's
estimation noise inflates its error, and they adjust for it (p. 2). In 48 simulation designs with one-step
least-squares forecasts, their adjusted test at the 5% normal critical value has a median size of about 0.04,
a little below nominal; rolling estimation is better sized than recursive. The unadjusted comparison is
badly undersized, below 0.01 at a nominal 0.10 (p. 4).
**Ours** (follow-up E4, 500 simulations at our sizes): Clark-West 7.0% at a nominal 5%; DM-HLN 0.0%. **Agrees on
DM** (badly undersized for nested models). **Clark-West slightly oversized here**, which is outside their
main design in three ways: 21-day overlapping targets, a recursive (expanding) window, and a ridge rather than
a least-squares fit. (Multistep forecasts are covered only in an unpublished appendix, where a bootstrap does
distinctly better than their test in small samples, p. 5.) In the world where the baseline only needs rescaling, Clark-West rejects 74% of the time,
consistent with what the test is built to detect: that the larger model's population forecast differs from
the baseline's, not that it adds information.

**Hansen, Lunde and Nason (2010)**, CREATES version. The Model Confidence Set reflects the informativeness of
the data: uninformative data give a set with many models, informative data a set with few (p. 2).
**Ours:** the set kept all 18 models on the test window. **Agrees**, and it is the most direct statement that
43 independent months cannot rank these models.

**Campbell and Thompson (2005)**, NBER version. Following Goyal and Welch, the historical average is a hard
benchmark out of sample; with sign restrictions some predictors beat it slightly (p. 2).
**Ours:** the historical mean of the target (0.0346) beat EWMA (0.0411) and was beaten only by DCC, HAR and
PCA regression, and not significantly. **Agrees** with the general lesson.

**Patton (2011)**, Oxford archive version. Some loss functions give rankings of volatility forecasts that change
when the volatility proxy is noisy (p. 1); QLIKE belongs to the robust class, and is the only robust loss that
depends only on the standardised forecast error (Proposition 3, p. 16).
**Ours:** the per-asset variance target is scored with QLIKE. **Consistent use.**

**Lo and MacKinlay (1989)**, NBER version. Weekly returns of size-sorted portfolios show positive
cross-autocorrelations, with large stocks leading small ones (p. 4-5).
**Ours** (step 4.5): no lead-lag survived false-discovery control among the 25 ETFs, daily or weekly; the
strongest uncorrected cells had REMX, a small and less liquid fund, as the follower. **Differs, plausibly by
design:** ETF prices are tied to their baskets by creation and redemption, and our sample is 25 funds, not
portfolios of thousands of stocks; the direction of the strongest cells is the one Lo and MacKinlay would
predict.

## 8.7 H5: cleaned covariances in minimum-variance portfolios (file 05)

**Jagannathan and Ma (2002)**, NBER version. A no-short-sales constraint is equivalent to shrinking the large
covariance estimates and then solving without constraints (p. 2, Proposition 1 on p. 9). With the constraint,
minimum-variance portfolios built from the sample covariance do about as well as those from factor models,
shrinkage or daily data; when short sales are allowed, the sample covariance from daily returns does best
(p. 7). They also expect sampling error to matter less, and the constraint to hurt, when the assets are
themselves large portfolios (p. 3-4); ETFs are such assets.
**Ours:** long-only, no cleaned estimator beat the sample covariance (H5 +0.14 points, p = 0.997). Long-short
with daily estimation (N/T = 0.10), nonlinear shrinkage gained only 0.1 points. **Agrees on both counts.** Their
p. 3-4 argument is a second reason, besides the constraint, why cleaning had little to fix for 25 diversified
funds.

**Ledoit and Wolf (2017)**, the authors' copy of the RFS article. Their backtest is the global minimum-variance
portfolio **without short-sales constraints**, estimated from the last 250 daily returns and updated monthly
(p. 18-19), our long-short design almost exactly. Their Table 1 (p. 22), annualised volatility, 1973-2011:

| Their N (N/T) | Sample | Nonlinear | Gain | Our row (N/T) | Our gain, test window |
|---|---|---|---|---|---|
| 30 (0.12) | 14.21% | 14.08% | 0.13 | daily (0.10) | 0.11 |
| 50 (0.20) | 13.15% | 12.68% | 0.47 | weekly (0.24) | 0.65 |
| 100 (0.40) | 13.11% | 11.52% | 1.59 | monthly (0.42) | 2.66 |
| 250 (1.00) | sample unusable | 10.34% | | quarterly (0.78) | 16.9 |

**Agrees in pattern and order of magnitude**: the gain grows with N/T, and is a fraction of a point at N/T near
0.1. At matched N/T our gains are 40-70% larger (0.65 against 0.47, 2.66 against 1.59), so "the same
magnitude" would overstate it. The designs also differ: they raise N/T by adding assets with T = 250 daily
rows, we by using fewer, lower-frequency returns for the same 25 funds; the mechanism (more estimation error
per parameter) is the same, but a weekly or monthly covariance applied to a month of daily returns adds a
mismatch theirs does not have.

The same paper also runs the backtest **with a no-short-sales constraint** (Table 8, p. 27): the sample
covariance is then uniformly the best of the rotation-equivariant estimators, by small margins, and forbidding
short sales helps the sample covariance but hurts linear and nonlinear shrinkage, which the authors tie to
Jagannathan and Ma. This is the closest published counterpart of the long-only H5 result, on US stocks, and it
points the same way.

**Ledoit and Wolf (2020)**, the authors' copy. Our implementation was checked against the paper: the kernel is
Epanechnikov with a bandwidth proportional to each sample eigenvalue, `h_j = lambda_j * n^(-1/3)` (p. 12-13,
equations 4.8-4.9); with demeaned data, n is replaced by n - 1 (p. 3, Remark 2.1); the last term of the Hilbert
transform is set to zero at the kernel's edge (p. 12). All four match `portfolio.nonlinear_shrinkage`, and a test
pins the formula written as explicit loops. They advise against the method for fewer than 12 variables (p. 13);
we use 25 and 41.

**Ledoit and Wolf (2004)**, authors' copy. Linear shrinkage is the optimal convex combination of the sample
covariance and a scaled identity (p. 1), which pulls every variance towards the average. Our design rehearsal
found this hurt minimum variance, and replaced it by the same shrinkage applied to the correlation only (file
05). Not a disagreement: the paper targets covariance accuracy, not this portfolio.

**Laloux et al. (1999)** and **Plerou et al. (2002)** on portfolios. Laloux et al. conclude that Markowitz
optimisation on a historical correlation matrix is unreliable because its smallest eigenvalues, which define the
lowest-risk portfolios, are dominated by noise (p. 3); the short paper states this as a consequence of its
spectrum fit, and neither proposes a cleaning method nor tests a portfolio. Plerou et al. filter the matrix by
keeping only the deviating eigenvalues, setting the others to zero and the diagonal back to one, and find that
realised risk then comes much closer to predicted risk (p. 9).
**Ours:** replacing the noise eigenvalues by their average (`rmt`) raised the realised volatility of long-short
portfolios by 2.6-5.4 points. **No contradiction**, for two reasons. Plerou et al. measure how well risk is
*predicted*, not how low realised risk is, and a matrix can predict its portfolio's risk better while choosing a
riskier portfolio; and their filter (zeroing) is not the one used here (averaging). Laloux et al. give no
portfolio evidence to compare with. Ledoit and Wolf (2017) compare realised risk directly but do not include
clipping among their estimators (p. 20-21), so "nonlinear shrinkage rather than clipping" is our result, not
theirs.

**DeMiguel, Garlappi and Uppal (2009)** *(abstract only)*. No optimised model consistently beats 1/N on Sharpe
ratio, certainty equivalent or turnover. **Not directly comparable:** their criteria involve expected returns;
for pure risk, our minimum-variance portfolios were far less volatile than equal weight (11.4% against 18.9%).

**Bailey and López de Prado (2014)** and **Bailey et al. (2017)**. The deflated Sharpe ratio corrects for the
number of trials and for non-normal returns (p. 2); the probability of backtest overfitting is the chance that
the strategy chosen as best in sample falls below the median out of sample (p. 6).
**Ours:** PBO across portfolio cells 0.80, so picking the best cell in sample would usually have picked a
below-median one; Sharpe ratios 1.0-1.3 with a standard error of 0.52. **Consistent with their warning**, and
the reason no Sharpe ratio is reported as a result.

## 8.8 H6: hedging in stress (file 06)

No paper in the reference list studies hedge effectiveness by volatility regime, so this is an application, not
a replication. **Ederington (1979)** *(not obtained)* is cited only for the standard definition of
effectiveness as the share of variance removed. The mechanism is **Forbes and Rigobon's** (p. 15): choosing days
by high VIX chooses days with high market variance, which biases each fund's correlation with SPY upwards, and
for a one-instrument hedge the share of variance removed is about the squared correlation. **Ours:** with their
correction the stress-calm gap goes from +0.357 to -0.04 [-0.08, +0.04]. **Agrees**, as an extension of their
argument from contagion to hedging.

## 8.9 What changed in the explainers after reading the texts

- **File 02:** the description of Borghesi et al. now states their intraday horizons, their regression-based
  removal of the market, and their cluster-based measure; and cites Plerou et al. on eigenvector stability near
  the noise edge.
- **File 04:** the Clark-West size comparison now cites their own simulation sizes.
- **File 05:** the Jagannathan-Ma daily-data finding and the Ledoit-Wolf (2017) Table 1 comparison are added.
- **Nothing contradicted a claim already made.** The closest was file 02, which had said Borghesi et al. found
  horizon invariance "in their title" without saying they used intraday horizons and cluster overlap. That
  omission made H1 look like a closer test of their claim than it was.

**Second check (2026-11-01),** every claim above re-read against `references/md`. Corrected: Borghesi et al.'s
set E is in effect our detoning, so the removal method is not a difference (8.3); Laloux et al. (1999) give
no portfolio test and no cleaning method, and Plerou et al.'s filter zeroes rather than averages (8.7, file 05);
our long-short gains match Ledoit and Wolf (2017) in pattern and order of magnitude, not closely in size (8.7);
Engle's exponential-smoother ranking is a count of misspecification rejections (8.5). Checked: the Mantegna
distance in the arXiv copy is `1 - rho^2`, but the published EPJ B version, obtained afterwards, uses
`sqrt(2 (1 - rho))`, the formula in `networks.py` (8.2). Added: Ledoit and Wolf's long-only check (Table 8) and Jagannathan and Ma
on assets that are themselves portfolios (8.7); Borghesi et al. on cluster reordering and on MSTs with 100
stocks (8.3); the Forbes-Rigobon reference period (8.4); Clark and West's multistep appendix (8.6).

## 8.10 The follow-ups E5-E9 against the papers (added 2026-11-01)

Seven more texts were read for this section, all obtained on 2026-11-01 from free copies (`references/README.md`):
Bekaert, Harvey and Ng (2005); Bekaert, Ehrmann, Fratzscher and Mehl (NBER version of the 2014 article);
Preis, Kenett, Stanley, Helbing and Ben-Jacob (2012); Büyükşahin and Robe (2011 working paper of the 2014
article); Bruno, Büyükşahin and Robe (2013); Ledoit and Wolf (2022); Ben-David, Franzoni, Kim and Moussawi (2021
working paper of the 2023 article). Corsetti, Pericoli and Sbracia (2005) and Longin and Solnik (2001), first
marked *(abstract only)*, were then obtained by Filippo through institutional access (2026-11-02, published
versions) and their paragraphs below rewritten from the texts.

**How these papers can be compared with ours.** Most of them study other assets (national indices, country-sector
portfolios, Dow stocks, commodity indices), other frequencies and, for the older ones, other years. Their
numbers are therefore not compared with ours. What is compared is the method (E9 is a known test), the
mechanism (why an adjustment over-corrects, why correlation rises) and the direction and timing of effects.
Three of them cover the years of our replication window (2007-2013): Bekaert et al. (2014) study the
2007-09 crisis itself, and Preis et al. and Büyükşahin and Robe run to 2010.

**Bekaert, Harvey and Ng (2005) and E9.** They define contagion as correlation beyond what a factor model
implies, measured as the correlation of the model's residuals, and note that a factor model by itself raises
correlation when a factor's volatility rises, by an amount set by the loadings (p. 2). Weekly national markets,
world and regional factors, betas varying with instruments. No evidence of contagion in the Mexican crisis
(p. 18); an economically meaningful rise in residual correlation in Asia in 1997 (p. 21). **Ours:** E9 is their
test, on US sector funds, with VIX regimes, nine observed factors and betas fitted inside each regime. The
residual correlation is about 0.01 in both regimes (+0.009 [-0.009, +0.028] higher in stress). **Agrees** with
their Mexican case: the stress rise is interdependence through the factors.

**Bekaert, Ehrmann, Fratzscher and Mehl (2014) and E9, E5.** The 2007-09 crisis on 415 country-sector portfolios
in 55 countries, weekly 1995 to March 2009, three factors (US, global financial, domestic), crisis from
7 August 2007 (p. 7); contagion is co-movement beyond the factor model with pre-crisis exposures (p. 3). Once
exposures may shift in the crisis, the residual correlation within countries disappears, the average residual
correlation is negligible and the residual-contagion coefficient is small and insignificant (p. 21).
Correlations with the US and global factors rose by about 0.2 in the crisis, which the fixed-exposure model
already reproduces because factor volatility rose relative to the portfolios' (p. 24); fixed exposures explain
75% of crisis return variation, exposure shifts 25% and residual contagion a negligible part (p. 25). The shifts
that matter are mostly domestic: domestic betas rose by about half (p. 21). **Ours:** with betas fitted inside each
regime the factors carry 99% of the rise; with one set of betas for the whole window they still carry all of it
(101%; residual correlation +0.003). **Agrees on the residual: nothing left beyond the factors. Differs on beta
shifts**, which mattered for them and not for us. The settings differ in the way that would produce this: their
"domestic contagion" is co-movement rising within a country, while our panel is one country, and our regimes
are VIX terciles over 2014-2022 rather than a crisis dummy.

**Corsetti, Pericoli and Sbracia (2005) and the overshoot in E5 and E7.** In a single-factor model of returns
(p. 5), the Forbes-Rigobon benchmark is correct only if the whole rise in the crisis market's variance comes from
the common factor (p. 10, footnote 13). If the market-specific noise also becomes more volatile in the crisis, the
benchmark expects too much correlation from interdependence alone, so the test leans towards "no contagion"
(p. 10); the bias grows with the share of the variance rise that is market-specific (p. 12). On the Hong Kong
crisis of October 1997, with plausible values for that noise, they find contagion for at least five of 17
countries, where the unadjusted test finds it for one (p. 1). **Ours:** the per-asset adjustment of H2 treats
each fund's variance rise as common. Step 4.2b measured idiosyncratic variance roughly doubling in stress, and the
adjustment over-corrects: the adjusted rise turns negative (volatility share 1.51 in E5; stress-minus-calm gap
-0.37 after adjustment in E7). **Agrees:** this is the bias they derive, in the direction they sign. Their remedy
is to model the factor and the specific noise separately, which is what E9 does; E9 finds no co-movement beyond
the factors, so in our data the corrected answer is still "interdependence".

**Preis et al. (2012) and E5, E8.** Daily prices of the 30 Dow stocks, 1939-2010 (p. 3). The average correlation
over windows of 10 to 60 days rises linearly with the size of the index's normalised return over the same window,
for large gains as well as large losses, more steeply for losses (slopes of about 0.064 for gains and 0.085 for
losses, p. 5); with shuffled data the relation disappears (p. 5). Their correlations are raw, with no volatility
adjustment. **Ours:** raw cross-sector correlation rises in high-VIX periods (E5: +0.197) and the whole rise is
carried by the common factors (E9). **Agrees** on the raw relation. Their rise after large gains should not be
read as evidence about the true correlation: Longin and Solnik (p. 2) show that conditioning on large absolute
returns raises the measured correlation even when the true one is constant, and Preis et al.'s control
(shuffled data, which removes correlation altogether) does not test against that.

**Büyükşahin and Robe (2011) and E8.** Dynamic conditional correlations between commodity indices (GSCI, DJ-UBS)
and the S&P 500, weekly and daily, 1991 to March 2010; DCC is used because rolling correlations carry the
Forbes-Rigobon bias (p. 10). Before Lehman the correlation swung between -0.38 and 0.4 with no upward trend; in the
18 months after Lehman it rose to levels unseen in the previous two decades (p. 10). Commodities outperformed
equities in 2003-2008 (p. 9). Bruno, Büyükşahin and Robe (2013) report the same post-Lehman rise for
agricultural commodities, with high correlations with equities until late 2011 or early 2012 (p. 10).
**Ours:** the coupling of the 13 equity funds fell to 0.26 on 2008-08-22 and was back to 0.81 by 2008-12-16; the
2011 euro phase was the most coupled (0.80). **Consistent:** our energy and mining funds hold equities whose
returns follow commodities, so a commodity boom that ran against the rest of the market until mid-2008 is a
plausible cause of the onset decoupling, and their post-Lehman break matches our recoupling. Not shown: we did
not compute correlations by sector pair, so the commodity link remains a reading.

**Longin and Solnik (2001) and E5, E8.** Monthly returns of five national markets, January 1959 to December
1996 (p. 9). Their starting point is the conditioning bias: splitting normal returns with a constant correlation
of 0.50 by the size of the move gives 0.21 for small moves and 0.62 for large ones (p. 2), so a test must
compare the conditional correlation with what the null predicts. With extreme value theory they do so for the
tails: correlation of large positive returns is what normality predicts, correlation of large negative returns
is higher (p. 22). Their "bear market" is this tail of large losses, not a period of falling prices (p. 2, p. 22).
**Ours:** E8's onset (August 2007 to Lehman) was a falling market without extreme losses, and coupling fell
(0.43); in the peak phase, the months of extreme losses, it rose to 0.75. **Consistent** once their term is read as
they define it. Their warning also applies to us: H2 and E5 condition on the VIX, which selects days of large
moves, and that is the bias the Forbes-Rigobon adjustment and E9 address.

**Ledoit and Wolf (2022) and E6, E10.** Their review reports that, when DeMiguel et al.'s (2009) norm-constrained
portfolios (a short-sale ban is one such constraint) are compared with shrinkage on small universes of
portfolios, such as ten industry or 25 Fama-French portfolios, the results are inconclusive, neither approach
dominating (p. 12, footnote 5). In stock-level backtests (De Nard, Ledoit and Wolf 2021, daily data) dynamic
estimators beat static ones, and the best estimator combines a factor model with DCC and nonlinear shrinkage of
the residuals (p. 27). **Ours:** E6, 13 diversified funds and long-only, finds that cleaning adds nothing (+0.17
points). **Agrees** with their small-universe footnote. For the long-short case, E3 found static nonlinear
shrinkage helping and a DCC-GARCH covariance not helping; their best estimators are dynamic with nonlinear
shrinkage (DCC-NL), which this project has not tried. E10 tests the static estimator only.

**Ben-David et al. (2021) and the universe.** Specialised ETFs (sector and thematic) lose about 30% risk-adjusted
over their first five years, which the authors trace to the overvaluation of the stocks they hold at launch
(p. 2). A large batch of specialised ETFs was launched in 2006 and another in 2011, and closures are more frequent
among specialised ETFs than among broad ones (p. 17). **For us:** seven of the 13 replication funds first trade in
2006 (XBI, IHI, IHF, ITA, GDX, XOP, XME), the first wave they describe, and the panel was chosen in 2026 among
survivors, so funds of that wave that closed are missing. Their return result is not something this project
tests.

**Not found.** No paper was found on the 2011 euro crisis as a period of record cross-sector coupling, no study
applying the volatility adjustment to US sectors in 2020, and the one recent study of sector-ETF betas in the
tails (*Studies in Economics and Finance*, 2026) sits behind a bot challenge that was not worked around.

| Follow-up | What the literature leads one to expect | What we found | Verdict of the comparison |
|---|---|---|---|
| E9, correlation beyond factors | residual correlation small once factors and their volatility are modelled (Bekaert et al. 2005, 2014) | about 0.01 in both regimes; factors carry 99% of the rise | **agrees**; beta shifts, which mattered in the 2007-09 global crisis, do not matter in our one-country panel |
| E5, E7, the overshoot | the Forbes-Rigobon benchmark over-predicts interdependence when specific noise also rises in a crisis (Corsetti et al.) | volatility share 1.51; adjusted E7 gap -0.37; idiosyncratic variance about doubled | **agrees**, in the direction they derive |
| E5, E8, correlation and stress | raw correlation rises with the size of market moves (Preis et al.), partly a conditioning effect (Longin and Solnik) | raw rise +0.197 in high VIX, all carried by the factors (E9) | **agrees** on the raw relation |
| E8, 2007-2009 | commodity-equity correlation without trend until August 2008, at record levels after Lehman (Büyükşahin and Robe) | decoupling to 0.26 three weeks before Lehman, back to 0.81 by December | **consistent**; the commodity link is not tested |
| E8, bear markets | tail correlation is high for large losses, not for large gains (Longin and Solnik) | coupling fell in the falling but non-extreme onset, rose to 0.75 in the extreme-loss peak | **consistent** with their definition of a bear market |
| E6, long-only cleaning | inconclusive on small universes of portfolios (Ledoit and Wolf 2022) | nothing gained | **agrees** |
