"""Check real measured holdings overlap for specific pairs, to replace supposition with data.

Context: while discussing which of the six core-eligible extended ETFs (VDE, PBW, NLR, IHF,
REMX, PICK) add genuine sector depth vs. duplicate an existing core member's exposure, several
"redundant" and "adds depth" claims were made from general knowledge of what these funds hold,
not from this project's own measured data. This script checks them against the real weighted-
Jaccard overlap matrix from step 2.4/2.5 (data/external/overlap_matrix.parquet, computed from
real SEC N-PORT holdings, 42/45 tickers fetched).

Run with: uv run python scripts/check_overlap_pairs.py
"""

import pandas as pd

from etfcorr.config import find_repo_root

root = find_repo_root()
overlap = pd.read_parquet(root / "data" / "external" / "overlap_matrix.parquet")

print(f"Overlap matrix shape: {overlap.shape}")
print(f"Columns: {list(overlap.columns)}")
print()


def show_pair(a: str, b: str, note: str = "") -> None:
    if a not in overlap.columns or b not in overlap.columns:
        missing = [t for t in (a, b) if t not in overlap.columns]
        print(f"  {a}-{b}: MISSING from overlap matrix ({missing}) -- {note}")
        return
    v = overlap.loc[a, b]
    print(f"  {a}-{b}: overlap={v:.4f}  {note}")


print("=== ai_semis: was flagged 'overlaps_with' in v2 (design-time note, not measured) ===")
show_pair("SMH", "SOXX", "flagged in universe.yaml/v2 as overlapping")
show_pair("SMH", "IGV")
show_pair("SOXX", "IGV")
print()

print("=== Claimed 'mostly redundant' vs. existing core member (my supposition, not checked) ===")
show_pair("VDE", "XLE", "both broad energy")
show_pair("PBW", "ICLN", "both broad clean energy")
show_pair("NLR", "URA", "both uranium/nuclear-themed")
show_pair("PICK", "XME", "both broad metals & mining")
print()

print("=== Claimed 'genuinely adds depth' (my supposition, not checked) ===")
for core_healthcare in ["XLV", "IBB", "XBI", "IHI"]:
    show_pair("IHF", core_healthcare, "IHF = healthcare providers/insurers, distinct sub-theme")
print()
for core_metals in ["GDX", "GDXJ", "XME", "COPX", "SIL"]:
    show_pair("REMX", core_metals, "REMX = rare earth/strategic metals, distinct sub-theme")
print()

print("=== For context: average overlap for each of these six vs. ALL current core members ===")
core_by_sector = {
    "VDE": ["XLE", "XOP", "OIH", "AMLP"],
    "PBW": ["ICLN", "TAN", "LIT", "URA"],
    "NLR": ["ICLN", "TAN", "LIT", "URA"],
    "IHF": ["XLV", "IBB", "XBI", "IHI"],
    "REMX": ["GDX", "GDXJ", "XME", "COPX", "SIL"],
    "PICK": ["GDX", "GDXJ", "XME", "COPX", "SIL"],
}
for candidate, core_members in core_by_sector.items():
    if candidate not in overlap.columns:
        print(f"  {candidate}: missing")
        continue
    vals = [overlap.loc[candidate, c] for c in core_members if c in overlap.columns]
    if vals:
        print(
            f"  {candidate} vs its sector's core members {core_members}: "
            f"mean={sum(vals) / len(vals):.4f}, max={max(vals):.4f}"
        )
