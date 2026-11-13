# Third-party code

One row per adapted snippet. Only [REUSE] / [REUSE-VERIFY] sources; nothing from AGPL,
proprietary or unlicensed repositories.
Libraries used as ordinary dependencies are listed in `pyproject.toml` / `uv.lock`, not here.

| File in this repo | Source URL | Commit | License | Notes |
|---|---|---|---|---|
| `src/etfcorr/evaluation/portfolio.py`, `nonlinear_shrinkage` (follow-up E3, 2026-10-12) | Ledoit and Wolf (2020), *Annals of Statistics* 48(5): equations 4.8-4.9 and the 20-line Matlab code in Appendix E of the article's Supplementary Material (p. 13 and p. 21 of the authors' copy, `references/sources.tsv`) | n/a (journal supplement) | published with the article; no separate software licence stated | Translated to NumPy, not copied: the kernel width proportional to each sample eigenvalue follows that code (`tests/unit/test_portfolio.py` checks it against the formula written as loops). The supplement itself was not re-read on 2026-11-02. Kept as is by Filippo's decision (2026-11-02): a translation of code published with the article. |

Checked 2026-11-02 (Phase 9): every module under `src/` and `scripts/` states either that nothing is adapted or which
library it imports; the one adapted algorithm is the row above. `correlation/rmt.py` follows the book's published method
and the project's own draft snippets, and states that no `mlfinlab` source was opened.
