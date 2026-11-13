"""The hypothesis scoreboard with the Holm correction across H1-H6 (Phase 8, step 8.6). Run with:
``uv run python scripts/scoreboard_8_6.py`` (instant; reads recorded results only).

Sources: H1 from the record of step 3.2 (``docs/H1_FINDINGS.md``; the three eigenvector tests are
significant in the direction **opposite** to the hypothesis, so the p-value of the hypothesised
direction is taken as ``1 - p / 2`` of the least extreme two-sided value, 0.004: a derived
number, any value above 0.05 gives the same table); H2 from ``h2_verdict_4_6.json``; H3 from
``eval_7_final_summary.json``; H5 and H6 from ``econ_8_final_summary.json``; H4 from
``h4_final_summary.json`` (attempted 2026-11-01; until then it entered with p = 1).
Output: ``reports/tables/scoreboard_8_6.{csv,md}``.
The caveats of H1, H3, H5 and H6 were extended on 2026-10-19 with the exploratory follow-ups
(``docs/FOLLOW_UPS_RESULTS.md``); verdicts and p-values are unchanged.
"""

import json

from etfcorr.config import load_project
from etfcorr.evaluation.scoreboard import Entry, scoreboard, to_markdown

H1_TWO_SIDED_LEAST_EXTREME = 0.004  # docs/H1_FINDINGS.md, eigenvector overlap, pair D-W


def main() -> int:
    project = load_project()
    tables = project.paths.tables

    def read(name: str) -> dict:  # type: ignore[type-arg]
        with open(tables / name, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    h2, h3 = read("h2_verdict_4_6.json"), read("eval_7_final_summary.json")["h3"]
    econ = read("econ_8_final_summary.json")
    h4 = read("h4_final_summary.json")["h4"]
    h5, h6 = econ["h5"], econ["h6"]
    statement = {h.id: " ".join(h.statement.split()) for h in project.experiment.hypotheses}

    def word(supported: bool) -> str:
        return "supported" if supported else "not supported"

    entries = [
        Entry(
            "H1",
            statement["H1"],
            "not supported",
            False,
            1.0 - H1_TWO_SIDED_LEAST_EXTREME / 2.0,
            "2014-01 to 2026-09 (all data), in-sample",
            "docs/H1_FINDINGS.md",
            "significant in the opposite direction (two-sided p 0.004, 0.001, 0.001), but the "
            "statistic falls the same way on simulated panels with a horizon-invariant truth "
            "(follow-up E2): the test could not tell the directions apart; p derived for the "
            "hypothesised direction",
        ),
        Entry(
            "H2",
            statement["H2"],
            h2["verdict"],
            h2["verdict"] == "supported",
            float(h2["clause1_p_bootstrap"]),
            "2014-01 to 2026-09 (all data), in-sample",
            "docs/explained/03_regimes_H2.md",
            "descriptive; the p-value is at the floor of 2000 bootstrap draws; the volatility "
            "clause is a pre-fixed threshold, not a test",
        ),
        Entry(
            "H3",
            statement["H3"],
            word(h3["h3_supported"]),
            bool(h3["h3_supported"]),
            float(h3["p_for_holm"]),
            "test 2023-2026, read once",
            "docs/explained/04_forecasting_H3_H4.md",
            "DM-HLN over two challengers; Clark-West gives 0.016 for elastic net against EWMA, "
            "but rejects 74% of the time when the baseline is merely badly scaled (follow-up E4)",
        ),
        Entry(
            "H4",
            statement["H4"],
            word(h4["h4_supported"]),
            bool(h4["h4_supported"]),
            float(h4["p_one_sided"]),
            "test 2023-2026, read once (2026-11-01)",
            "docs/explained/04_forecasting_H3_H4.md",
            "a non-superiority claim: supported because the network did not beat LightGBM-graph "
            "(it was worse, +0.0008); a failure to detect a difference, not proof of equality; "
            "designed after the H3 test window had been read",
            null_claim=True,
        ),
        Entry(
            "H5",
            statement["H5"],
            word(h5["h5_supported"]),
            bool(h5["h5_supported"]),
            float(h5["p_one_sided"]),
            "test 2023-2026, read once",
            "docs/explained/05_portfolios_H5.md",
            "the difference has the opposite sign; long-only monthly portfolios of 25 ETFs, where "
            "the constraint already shrinks; long-short, nonlinear shrinkage lowers volatility "
            "(follow-up E3, on seen data)",
        ),
        Entry(
            "H6",
            statement["H6"],
            word(h6["h6_supported"]),
            bool(h6["h6_supported"]),
            float(h6["p_one_sided"]),
            "test 2023-2026, read once",
            "docs/explained/06_hedging_H6.md",
            "the difference has the opposite sign, but it is a volatility effect: about zero "
            "after a Forbes-Rigobon adjustment, and hedged volatility is higher in stress "
            "(follow-up E1); stress is 93 days in a few clusters",
        ),
    ]
    table = scoreboard(entries, project.experiment.inference.alpha)
    table.to_csv(tables / "scoreboard_8_6.csv", index=False)
    (tables / "scoreboard_8_6.md").write_text(to_markdown(table), encoding="utf-8")
    shown = table[["hypothesis", "verdict", "p_value", "p_holm", "supported_after_holm"]]
    print(shown.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
