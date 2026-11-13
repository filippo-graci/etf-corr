# Reference papers

Research copies of the papers cited in `docs/explained/`, downloaded on 2026-10-19 so that the explainers can
be checked against the actual text. **The PDFs (`pdf/`) and their text versions (`md/`) are git-ignored**: they
are copyrighted, and only this file and `sources.tsv` are committed. If the repository is made public, nothing
here is redistributed.

- `sources.tsv`: one row per paper, with the version downloaded and its URL. Only legitimate free copies were
  used: arXiv, NBER and CREATES working papers, the authors' own university pages, institutional repositories
  and publishers' open-access pages.
- `pdf/<key>.pdf`: the downloaded file. Reproduce with the URLs in `sources.tsv`.
- `md/<key>.md`: the text, one `## Page N` section per PDF page, from `uv run python scripts/papers_to_md.py`
  (poppler's `pdftotext`). Equations and tables are often garbled in the text: check the PDF before quoting
  one. Page numbers are those of the PDF file; for a working paper they differ from the published article.

## Obtained (28)

| Key | Version | Note |
|---|---|---|
| bailey_borwein_lopezdeprado_zhu_2017 | eScholarship copy of the J. Comput. Finance article | |
| bailey_lopezdeprado_2014 | authors' copy of the JPM article | |
| bekaert_et_al_2011 | NBER WP 17121, revised 2012 | published JF 2014; added 2026-11-01 |
| bekaert_harvey_ng_2005 | authors' copy (Duke) of the J. Business article | added 2026-11-01 |
| ben_david_et_al_2021 | NBER WP 28369, revision 2 | published RFS 2023; added 2026-11-01 |
| bollerslev_1986 | author-hosted copy (Duke) of the J. Econometrics article | added on a second attempt |
| borghesi_marsili_micciche_2007 | arXiv preprint | |
| bruno_buyuksahin_robe_2013 | Bank of Canada WP 2013-39, "The Financialization of Food?" | added 2026-11-01; first downloaded in error for Büyükşahin and Robe |
| buyuksahin_robe_2011 | working paper, February 2011, University of Oklahoma conference copy | published JIMF 2014; added 2026-11-01 |
| campbell_thompson_2005 | NBER WP 11468 | published RFS 2008 with a different title |
| clark_west_2006 | NBER technical WP 326 | published J. Econometrics 2007 |
| corsetti_pericoli_sbracia_2005 | the published JIMF article | obtained by Filippo through institutional access, 2026-11-02 |
| diebold_mariano_1994 | NBER technical WP 169 | **page scans with no text layer**: no Markdown, read the PDF (no OCR tool installed) |
| engle_2002 | NYU Stern WP, revised Jan 2002 | published JBES 2002 |
| forbes_rigobon_1999 | NBER WP 7267 | published JF 2002; the text layer is an OCR of a scan, equations unreadable |
| hansen_lunde_nason_2010 | CREATES RP 2010-76 | published Econometrica 2011 |
| jagannathan_ma_2002 | NBER WP 8922 | published JF 2003 |
| laloux_et_al_1999 | arXiv preprint | published PRL 1999 |
| ledoit_wolf_2004 | authors' copy of the JMVA article | |
| ledoit_wolf_2017 | authors' copy of the RFS article | |
| ledoit_wolf_2020 | authors' copy of the Annals of Statistics article | |
| ledoit_wolf_2022 | authors' copy (UZH) of the J. Financial Econometrics article | added 2026-11-01 |
| lo_mackinlay_1989 | NBER WP 2977 | published RFS 1990 |
| longin_solnik_2001 | the published JF article | obtained by Filippo through institutional access, 2026-11-02 |
| mantegna_1999 | arXiv preprint | published EPJ B 1999; its distance formula differs from the article's |
| mantegna_1999_epjb | the published EPJ B article | obtained through institutional access, 2026-11-01 |
| patton_2011 | Oxford Research Archive copy | |
| plerou_et_al_2002 | arXiv preprint | published PRE 2002 |
| preis_et_al_2012 | the published Scientific Reports article (Warwick repository, CC BY-NC-ND) | added 2026-11-01 |

## Not obtained

| Paper | Why |
|---|---|
| Büyükşahin, Haigh and Robe (2010) | no free copy found (doi 10.3905/JAI.2010.12.3.076); its question is covered by Büyükşahin and Robe (2011) |
| Sector-ETF betas in the tails (*Studies in Economics and Finance*, 2026) | the Emerald page is behind a bot challenge |
| Benjamini and Yekutieli (2001) | the open-access link returned a web page, not the PDF |
| DeMiguel, Garlappi and Uppal (2009) | the LBS repository has no full text |
| RiskMetrics Technical Document (1996) | MSCI requires a form with personal details; not filled in |
| Corsi (2009); López de Prado (2016); Harvey, Leybourne and Newbold (1997); Ederington (1979); Holm (1979); Hubert and Arabie (1985); Hamilton (1989); Marchenko and Pastur (1967); Elton and Gruber (1973); Benjamini and Hochberg (1995); Politis and Romano (1994) | no free copy from the authors or an institutional repository was found; copies on course pages and file-sharing sites were not used |

A second attempt on 2026-10-19 searched author pages (ESSEC, LBS, Tel Aviv, UCSD, Purdue) and found only Bollerslev
(1986). The CEPR server's refusal is a bot block, which was not worked around; the HEC, LBS and Purdue pages
list the papers without a file. For these, the explainers rely on the abstract or a standard description of the method, and say so.

On 2026-11-01 the published EPJ B version of Mantegna (1999), doi 10.1007/s100510050929, was sought to check
its distance formula against the arXiv copy (`1 - rho^2`, p. 3). Springer and epjb.epj.org answered with a bot
challenge (not worked around), no author or institutional copy was found, and a university-library proxy link
was not used. Filippo then obtained it through institutional access (`mantegna_1999_epjb`): the published
article uses `sqrt(2 (1 - rho))` (equation 2, p. 193), the formula in `networks.py`.

On 2026-11-02 Filippo obtained the published versions of Longin and Solnik (2001) and Corsetti, Pericoli and
Sbracia (2005) through institutional access, after the free copies were refused (CEPR, HTTP 403; HAL, bot
challenge; the journal version subscription-only).
