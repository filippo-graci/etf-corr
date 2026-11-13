"""Step 8.6: the hypothesis scoreboard with the Holm correction across H1-H6.

One row per pre-registered hypothesis: its verdict by its own pre-fixed rule, the p-value of
the hypothesised direction, and the Holm-adjusted p-value over the whole family
(``statsmodels.stats.multitest.multipletests``, not re-implemented). A hypothesis that was not
tested enters the family with ``p = 1``, which keeps the family size at six and cannot help any
other hypothesis: failures and gaps are shown, not dropped.

``supported_after_holm`` needs both the hypothesis's own rule and an adjusted p-value below
``alpha``; a rule can be stricter than its p-value (H2 has two clauses), never looser.

A **non-superiority claim** (``null_claim``, H4: "the network does not significantly beat ...")
is supported when its test does *not* reject. It enters the family with the p-value of that test
(so the family has six real tests), and it holds after Holm exactly when its own rule holds: the
correction only makes rejections harder, so it cannot turn a non-rejection into a rejection. Such
a "supported" is a failure to detect a difference, not evidence of equality (2026-11-01).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from statsmodels.stats.multitest import multipletests


@dataclass(frozen=True)
class Entry:
    """One hypothesis: ``p_value`` is ``None`` when it was not tested."""

    id: str
    statement: str
    verdict: str
    supported: bool
    p_value: float | None
    window: str
    source: str
    caveat: str = ""
    null_claim: bool = False  # supported when its test does not reject


def scoreboard(entries: Sequence[Entry], alpha: float = 0.05) -> pd.DataFrame:
    """The table, in the order given, with Holm-adjusted p-values over all entries."""
    if len({e.id for e in entries}) != len(entries):
        raise ValueError("hypothesis ids must be unique")
    for e in entries:
        if e.p_value is not None and not 0.0 <= e.p_value <= 1.0:
            raise ValueError(f"{e.id}: p-value {e.p_value} is outside [0, 1]")
        if e.supported and e.p_value is None:
            raise ValueError(f"{e.id}: a hypothesis that was not tested cannot be supported")
    raw = [1.0 if e.p_value is None else e.p_value for e in entries]
    adjusted = multipletests(raw, alpha=alpha, method="holm")[1]
    rows = [
        {
            "hypothesis": e.id,
            "statement": e.statement,
            "verdict": e.verdict,
            "p_value": e.p_value,
            "p_holm": float(p) if e.p_value is not None else None,
            "supported_after_holm": bool(
                e.supported if e.null_claim else e.supported and p < alpha
            ),
            "window": e.window,
            "source": e.source,
            "caveat": e.caveat,
        }
        for e, p in zip(entries, adjusted, strict=True)
    ]
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame) -> str:
    def fmt(value: float | None) -> str:
        return "not tested" if value is None or value != value else f"{value:.4f}"

    lines = [
        "| | Hypothesis | Verdict | p | Holm p (m = 6) | Holds after Holm | Window | Caveat |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in table.to_dict("records"):
        lines.append(
            f"| {row['hypothesis']} | {row['statement']} | **{row['verdict']}** | "
            f"{fmt(row['p_value'])} | {fmt(row['p_holm'])} | "
            f"{'yes' if row['supported_after_holm'] else 'no'} | {row['window']} | "
            f"{row['caveat']} |"
        )
    return "\n".join(lines) + "\n"
