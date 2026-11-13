"""Exploratory follow-up E1 to H6 (docs/FOLLOW_UPS.md item 1). Run with:
``uv run python scripts/followup_h6.py`` (about a minute).

H6 labelled stress by the VIX close of the same day, which selects days with large market moves
and so inflates the share of variance a market hedge removes (the Forbes-Rigobon bias). This
script recomputes the SPY hedge of step 8.5 (126-day rolling beta, known the day before) with

* three stress labels: the same-day VIX (the H6 design, to reproduce +0.357), the VIX of the day
  before, and the previous 21-day realised volatility of SPY (both known before the day starts);
* the squared in-regime correlation with SPY, raw and Forbes-Rigobon adjusted to the calm variance;
* what is left after hedging: annualised volatility, 5% worst daily loss and expected shortfall;
* the same per sector.

**Status: exploratory, after the H6 verdict, which it does not change.** The test window
(2023-01 to 2026-09) has been read: every number on it is in-sample. The 2016-2022 window is shown
beside it as a second sample, not as a hold-out (it contains the data every design was built on).
The quantile thresholds are those of the evaluated window, so the labels are still descriptions,
not tradeable signals; only the information inside each label is lagged. With about 90 stress
days the 5% tail rests on four or five observations per ETF.
Output: ``reports/tables/followup_h6_{summary,sectors}.csv``.
"""

import pandas as pd

from etfcorr.config import DateWindow, load_project
from etfcorr.evaluation import hedging as H
from etfcorr.evaluation.ladder import load_sealed_panel

SECTOR_COLUMNS = ["he_calm", "he_stress", "r2_calm", "r2_adj_stress"] + [
    f"{kind}_{regime}" for kind in ("vol_unhedged", "vol_hedged") for regime in ("calm", "stress")
]
SHOWN = ["he_stress", "he_calm", "he_diff", "r2_adj_diff", "vol_hedged_stress"]
SHOWN += ["vol_hedged_calm", "es_hedged_stress", "es_unhedged_stress"]


def main() -> int:
    project = load_project()
    exp = project.experiment
    design = exp.economics
    assert design is not None
    daily, sectors, spy, vix = load_sealed_panel(project, final=True)
    hedged = H.hedged_returns(daily, spy, H.rolling_beta(daily, spy, design.hedge_window_days))
    labels = {
        "vix_same_day (H6 design)": vix,
        "vix_previous_day": vix.shift(1),
        "spy_realised_vol_21d_previous_day": spy.rolling(21).std().shift(1),
    }
    windows = {
        "test 2023-2026 (seen, in-sample)": exp.splits.test,
        "2016-2022 (train and validation)": DateWindow(
            start=exp.splits.train.start, end=exp.splits.val.end
        ),
    }
    summary, by_sector = [], []
    for window_name, window in windows.items():
        inside = (daily.index >= pd.Timestamp(window.start)) & (
            daily.index <= pd.Timestamp(window.end)
        )
        for label_name, signal in labels.items():
            stress, calm = H.regime_masks(
                signal.loc[inside], design.stress_quantile, design.calm_quantile
            )
            args = (daily.loc[inside], hedged.loc[inside], spy.loc[inside], stress, calm)
            table = H.regime_risk_table(
                *args,
                block_days=design.block_days,
                n_bootstrap=exp.inference.n_bootstrap,
                alpha=exp.inference.alpha,
            )
            summary.append(
                table.reset_index().assign(
                    window=window_name,
                    label=label_name,
                    n_stress=int(stress.sum()),
                    n_calm=int(calm.sum()),
                )
            )
            per_sector = H.regime_risk(*args).groupby(pd.Series(sectors)).mean()[SECTOR_COLUMNS]
            by_sector.append(
                per_sector.rename_axis("sector")
                .reset_index()
                .assign(window=window_name, label=label_name)
            )
            show = table.loc[SHOWN]
            print(f"\n{window_name} | {label_name} | {stress.sum()} stress, {calm.sum()} calm days")
            print(show.drop(columns="n_draws").round(4).to_string())
            print(per_sector.round(3).to_string())
    out = project.paths.tables
    pd.concat(summary).to_csv(out / "followup_h6_summary.csv", index=False)
    pd.concat(by_sector).to_csv(out / "followup_h6_sectors.csv", index=False)
    print(f"\nwrote followup_h6_summary.csv and followup_h6_sectors.csv under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
