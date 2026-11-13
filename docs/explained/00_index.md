# etf-corr explained

Written 2026-10-19; updated 2026-11-02 (H4, the 2007-2013 replication E5-E8, E9, E10). These files explain the whole study for a reader who has not followed it: what was
asked, what the literature says, which code does what, what came out, and how to read it. They summarise the
project's own records; the numbers come from the files under `reports/tables/` and from the per-step reports
of the private research repository (not included here), and each file says which.

| File | Topic | Hypothesis |
|---|---|---|
| [01_data_and_design.md](01_data_and_design.md) | The ETFs, the data, the splits, and the rules every test follows | all |
| [02_structure_H1.md](02_structure_H1.md) | Correlation structure across daily, weekly and monthly horizons; random-matrix cleaning; networks | H1 |
| [03_regimes_H2.md](03_regimes_H2.md) | Correlation in calm and volatile markets; the volatility bias; HMM, DCC and lead-lag side studies | H2 |
| [04_forecasting_H3_H4.md](04_forecasting_H3_H4.md) | Forecasting next month's correlation with baselines, linear models, LightGBM and graphs | H3, H4 |
| [05_portfolios_H5.md](05_portfolios_H5.md) | Cleaned covariance matrices in minimum-variance portfolios | H5 |
| [06_hedging_H6.md](06_hedging_H6.md) | Hedging each ETF with SPY in stress and in calm | H6 |
| [07_scoreboard_and_validation.md](07_scoreboard_and_validation.md) | The six verdicts together, the multiplicity correction, and why the results can be trusted | all |
| [08_literature_comparison.md](08_literature_comparison.md) | Every result set against the papers' own findings, checked in their full text with page numbers | all |

Results written up outside these files: `docs/FOLLOW_UPS_RESULTS.md` (E1-E4), `docs/REPLICATION_PRE2014_RESULTS.md`
(E5-E8, covered in files 03, 05, 06) and `docs/E9_FACTOR_RESIDUAL_RESULTS.md` (E9, file 03).

## The study in five sentences

Twenty-five US-listed thematic ETFs in six sectors, daily prices from 2014 to September 2026. Six hypotheses
were written and fingerprinted before any result existed, then each was tested once. One held as an effect:
cross-sector correlation rises in high-volatility periods, and most of the rise is a volatility effect (H2).
One held as a "no improvement" claim: a graph neural network did not beat LightGBM (H4, run 2026-11-01). The
other four did not, and three came out with the opposite sign; follow-ups run after the verdicts show that none of the three
reversals is a finding in its own right. For forecasting, portfolio construction and hedging, a simple
standard method was never beaten on the test window. Declared tests run afterwards repeat the H2, H5 and H6
verdicts on 2007-2013 (E5-E7) and show that nine common factors carry the whole stress rise in correlation (E9).

## How to read the numbers

- **Windows.** Train 2016-2020, validation 2021-2022, test 2023-01-03 to 2026-09-18. "Test" numbers labelled
  *verdict* were computed once, by a command that refuses to run twice. Numbers labelled *follow-up* were
  computed afterwards on data already seen: they explain, they do not confirm.
- **Intervals** are 95% stationary-bootstrap intervals (Politis and Romano 1994) unless said otherwise.
- **p-values** are one-sided in the direction of the hypothesis unless said otherwise.
- **Tags** used in the project records and kept here: *Established* (a verdict by a rule fixed in advance),
  *Supported* (a result with an interval that excludes zero, not a verdict), *Inference* (a reading, not
  tested), *Unverified* (a fact or reference not checked).

## Map from question to code

| You want | Look at |
|---|---|
| The design and the pre-registration | `config/experiment.yaml`, `config/hypotheses.lock` |
| The library code | `src/etfcorr/` (`data/`, `correlation/`, `features.py`, `models/`, `evaluation/`) |
| The script that produced a table | `scripts/<name>_<phase>_<step>.py`, named in each file below |
| The tables and figures | `reports/tables/`, `reports/figures/` |

## References used across these files

Every reference below was checked for authors, title, venue and year by web search, either in this session
(2026-10-19) or in the session that first cited it (marked with the date). Full texts were not read unless
stated; where a file attributes a specific claim to a paper, it is a claim stated in the paper's abstract or
listing, or a standard description of a well-known method.

- Bailey, D. H. and López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management* 40(5), 94-107.
- Bailey, D. H., Borwein, J., López de Prado, M. and Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance* 20(4).
- Benjamini, Y. and Hochberg, Y. (1995). Controlling the False Discovery Rate. *JRSS B* 57(1) (checked 2026-09-28).
- Benjamini, Y. and Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics* 29(4) (checked 2026-10-04).
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics* 31(3), 307-327.
- Borghesi, C., Marsili, M. and Miccichè, S. (2007). Emergence of time-horizon invariant correlation structure in financial returns by subtraction of the market mode. arXiv physics/0702106 (checked 2026-10-11).
- Campbell, J. Y. and Thompson, S. B. (2008). Predicting Excess Stock Returns Out of Sample. *Review of Financial Studies* 21(4), 1509-1531 (checked 2026-10-11).
- Clark, T. E. and West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics* 138(1), 291-311 (checked 2026-10-11).
- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics* 7(2), 174-196.
- DeMiguel, V., Garlappi, L. and Uppal, R. (2009). Optimal versus naive diversification: how inefficient is the 1/N portfolio strategy? *Review of Financial Studies* 22(5), 1915-1953.
- Diebold, F. X. and Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics* 13(3), 253-263.
- Ederington, L. H. (1979). The Hedging Performance of the New Futures Markets. *Journal of Finance* 34(1), 157-170.
- Elton, E. J. and Gruber, M. J. (1973). Estimating the Dependence Structure of Share Prices. *Journal of Finance* 28(5) (checked 2026-10-11).
- Engle, R. (2002). Dynamic Conditional Correlation: A Simple Class of Multivariate GARCH Models. *Journal of Business & Economic Statistics* 20(3), 339-350.
- Forbes, K. J. and Rigobon, R. (2002). No Contagion, Only Interdependence: Measuring Stock Market Comovements. *Journal of Finance* 57(5), 2223-2261 (checked 2026-10-02).
- Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. *Econometrica* 57(2), 357-384 (checked 2026-10-02).
- Hansen, P. R., Lunde, A. and Nason, J. M. (2011). The Model Confidence Set. *Econometrica* 79(2), 453-497.
- Harvey, D., Leybourne, S. and Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting* 13(2), 281-291.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6(2), 65-70.
- Hubert, L. and Arabie, P. (1985). Comparing partitions. *Journal of Classification* 2, 193-218.
- Jagannathan, R. and Ma, T. (2003). Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps. *Journal of Finance* 58(4), 1651-1684 (checked 2026-10-11).
- J.P. Morgan / Reuters (1996). *RiskMetrics Technical Document*, 4th edition.
- Laloux, L., Cizeau, P., Bouchaud, J.-P. and Potters, M. (1999). Noise Dressing of Financial Correlation Matrices. *Physical Review Letters* 83(7).
- Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis* 88(2), 365-411.
- Ledoit, O. and Wolf, M. (2017). Nonlinear Shrinkage of the Covariance Matrix for Portfolio Selection: Markowitz Meets Goldilocks. *Review of Financial Studies* (checked 2026-10-11).
- Ledoit, O. and Wolf, M. (2020). Analytical nonlinear shrinkage of large-dimensional covariance matrices. *Annals of Statistics* 48(5).
- Lo, A. W. and MacKinlay, A. C. (1990). When Are Contrarian Profits Due to Stock Market Overreaction? *Review of Financial Studies* 3(2) (checked 2026-10-04).
- Longin, F. and Solnik, B. (2001). Extreme Correlation of International Equity Markets. *Journal of Finance* 56(2), 649-676.
- López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of Sample. *Journal of Portfolio Management* 42(4), 59-69.
- Mantegna, R. N. (1999). Hierarchical structure in financial markets. *European Physical Journal B* 11, 193-197.
- Marchenko, V. A. and Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. *Mathematics of the USSR-Sbornik* 1(4) (checked 2026-09-28).
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics* 160(1), 246-256.
- Plerou, V., Gopikrishnan, P., Rosenow, B., Amaral, L. A. N., Guhr, T. and Stanley, H. E. (2002). Random matrix approach to cross correlations in financial data. *Physical Review E* 65, 066126.
- Politis, D. N. and Romano, J. P. (1994). The Stationary Bootstrap. *JASA* 89(428), 1303-1313 (checked 2026-09-28).

**Checked against the full text** on 2026-10-19 (copies in `references/`, git-ignored; page numbers in
file 08): Bailey and López de Prado (2014), Bailey et al. (2017), Bollerslev (1986), Borghesi et al. (2007),
Campbell and Thompson (as NBER WP 11468), Clark and West (as NBER TWP 326), Diebold and Mariano (as NBER TWP 169,
page scans), Engle (2002, working-paper version), Forbes and Rigobon (as NBER WP 7267), Hansen, Lunde and Nason
(as CREATES RP 2010-76), Jagannathan and Ma (as NBER WP 8922), Laloux et al. (1999), Ledoit and Wolf (2004,
2017, 2020), Lo and MacKinlay (as NBER WP 2977), Mantegna (1999), Patton (2011), Plerou et al. (2002). The others
were checked for bibliographic details only; `references/README.md` says why each could not be obtained.

Checked for bibliographic details only (2026-11-02): Boyer, Gibson and Loretan (1999), *Pitfalls in tests for
changes in correlations*, Federal Reserve International Finance Discussion Paper 597 (as listed in the reference
lists of Longin and Solnik 2001 and Corsetti et al. 2005); Onnela, Chakraborti, Kaski, Kertész and Kanto (2003),
*Physical Review E* 68, 056110 (arXiv cond-mat/0302546; one scanned copy shows only the first three authors).
