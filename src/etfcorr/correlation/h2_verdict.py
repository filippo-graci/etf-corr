"""Step 4.6: the H2 verdict rule -- the pre-fixed reading of the H2 table, as code.

Question served: H2 (``config/experiment.yaml``, pre-registered, locked): "Average cross-sector
correlation is higher in high-VIX regimes, but a large part of the increase is explained by
volatility (Forbes-Rigobon bias) rather than by a true change in dependence." Test text: "Raw versus
volatility-adjusted average cross-sector correlation by VIX tercile, with stationary-bootstrap
confidence intervals." The table itself is produced by :mod:`etfcorr.correlation.regime_correlation`
(step 4.2); this module decides nothing about estimation. It only applies a reading rule to that
table, so that the verdict is a pure function of the table and can be tested, instead of a sentence
written after the numbers were seen.

**The rule (confirmed by the user on 2026-10-02, after the first real-data table existed but with
the wording unchanged from the version proposed before it; docs/DECISION_LOG.md).** It is evaluated
on the high VIX tercile against the low tercile (the reference), cross-sector pairs, predetermined
(t-1) labels:

* *Clause 1 (direction)*: the raw high-minus-low difference of the average cross-sector correlation
  is positive and its bootstrap interval excludes zero (lower bound above 0).
* *Clause 2 ("a large part")*: the ``vol_share`` point estimate, ``(raw difference - adjusted
  difference) / raw difference``, is at least ``THRESHOLD = 0.5`` on the **primary line** (per-asset
  delta); "partly" when it is below 0.5 on the primary line but at least 0.5 on a **market-series
  line** (delta taken from the SPY return or from the VIX log change); "not supported" when it is
  below 0.5 on every line.
* *Verdict*: clause 1 failed gives "not supported" (clause 2 is then not evaluated); clause 1
  holding gives "supported", "partly supported" or "not supported" according to clause 2.

The threshold applies to the **point estimate**, as fixed on 2026-10-02. The interval of the share
is reported next to it (and flagged when its lower bound is also at least 0.5) but does not enter
the rule. Two reading notes are computed from the table and printed beside the verdict; they are
**presentation added after the 4.2 table was seen** and do not enter the rule: whether any line has
a share above 1 (the adjusted difference then has the opposite sign to the raw one, so the
adjustment overshoots; it does not mean that volatility explains more than 100%), and whether the
non-volatility part of the change (the adjusted high-minus-low difference) is identified, which it
is only if every line's adjusted-difference interval excludes zero on the same side.

**What the verdict does and does not say.** "Supported" means: the pre-registered rule, applied to
an in-sample, descriptive table, is met. The second clause is a statement about the
Forbes-Rigobon-adjusted correlation, which assumes a bivariate model with one heteroskedastic
source, no omitted variables and no endogeneity (Forbes and Rigobon, 2002); ETF returns with a
common factor violate it, and the per-asset-delta variant under-corrects in a simulation
(docs/checks/4.2.md, one data-generating process). The residual, non-volatility part of the change
is *not* identified by this design when the sign of the adjusted difference depends on which delta
is used. Nothing here is causal and nothing is a forecast. Regime cutpoints are full-sample terciles
of the VIX (descriptive, decided 2026-10-02); the number of high-VIX episodes (about 100), not the
number of days, bounds the information.

**p-value for the scoreboard.** :func:`bootstrap_p_value` turns the replicates of the raw difference
into a two-sided percentile-inversion p-value with a ``+1`` correction in numerator and denominator,
so it is never zero with a finite number of draws. The ``+1`` follows the argument of Phipson and
Smyth (2010) for randomly drawn permutations; applying it to bootstrap replicates is this module's
extension (**[Inference]**), and the inversion itself is generic practice, not a cited estimator.
Only clause 1 has a p-value; clause 2 is a decision rule on a ratio. The cross-hypothesis Holm
correction over H1-H6 (``inference.multiplicity``) is the scoreboard's job in Phase 8 and is not
applied here.

Citations (existence and venue checked by web search on 2026-10-04 unless noted): Forbes, K. J. and
Rigobon, R. (2002), *No Contagion, Only Interdependence: Measuring Stock Market Comovements*,
Journal of Finance 57(5), 2223-2261, doi 10.1111/0022-1082.00494 (CHECKED; the formula was
re-derived in step 4.2, the paper's empirical results are not used). Phipson, B. and Smyth, G. K.
(2010), *Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations
Are Randomly Drawn*, Statistical Applications in Genetics and Molecular Biology 9(1), Article 39
(CHECKED for existence, authors, venue; the volume and article number come from the RePEc listing,
not from the paper). No third-party code adapted (no licence flag). Pure functions of their inputs:
no I/O, no randomness.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from etfcorr.correlation.regime_correlation import H2Error, H2Result, H2Stat

THRESHOLD = 0.5  # "a large part": vol_share point estimate >= 0.5 (fixed 2026-10-02)
HIGH_REGIME = 2  # the high VIX tercile code (etfcorr.regimes.HIGH)

Clause2 = Literal["supported", "partly supported", "not supported", "not evaluated"]
Verdict = Literal["supported", "partly supported", "not supported"]
ResidualReading = Literal["identified, positive", "identified, negative", "not identified"]


@dataclass(frozen=True)
class H2Line:
    """The three statistics of one H2 table line that the rule reads (one regime vs. reference).

    ``name`` labels the line in reports; ``delta_mode`` is ``"pair"`` (per-asset delta, the primary
    line) or ``"series"`` (one delta from a market series); ``pairs`` and ``alignment`` are those of
    the engine call (the rule is defined for cross-sector pairs and predetermined labels).
    """

    name: str
    delta_mode: str
    pairs: str
    alignment: str
    regime: int
    reference: int
    n_rows: int
    diff_raw: H2Stat
    diff_adjusted: H2Stat
    vol_share: H2Stat

    @property
    def share_exceeds_one(self) -> bool:
        """True when the share is above 1 (the adjusted difference has the opposite sign)."""
        return bool(self.vol_share.point > 1.0)


def summarize_line(name: str, result: H2Result, regime: int = HIGH_REGIME) -> H2Line:
    """Pull ``diff_raw``, ``diff_adjusted`` and ``vol_share`` of ``regime`` out of an engine result.

    Raises :class:`~etfcorr.correlation.regime_correlation.H2Error` when ``regime`` is the
    reference regime or is not in the table (``KeyError`` from the engine is converted).
    """
    pt = result.point
    if regime == pt.reference:
        raise H2Error(f"regime {regime} is the reference regime; it has no contrast")
    try:
        return H2Line(
            name=name,
            delta_mode=pt.delta_mode,
            pairs=pt.pairs,
            alignment=pt.alignment,
            regime=regime,
            reference=pt.reference,
            n_rows=pt.n_rows,
            diff_raw=result.stat("diff_raw", regime),
            diff_adjusted=result.stat("diff_adjusted", regime),
            vol_share=result.stat("vol_share", regime),
        )
    except KeyError as exc:
        raise H2Error(f"regime {regime} is not in the H2 table") from exc


def bootstrap_p_value(replicates: ArrayLike, null: float = 0.0) -> float:
    """Two-sided percentile-inversion bootstrap p-value of ``H0: statistic = null``, never zero.

    ``p = 2 * min((1 + #{rep <= null}) / (B + 1), (1 + #{rep >= null}) / (B + 1))``, capped at 1.
    With ``B`` draws the smallest value is ``2 / (B + 1)``. Raises
    :class:`~etfcorr.correlation.regime_correlation.H2Error` for no replicates or a non-finite one.
    """
    arr = np.asarray(replicates, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise H2Error("replicates must be a non-empty one-dimensional array")
    if not np.isfinite(arr).all():
        raise H2Error("replicates must be finite")
    b = arr.size
    p_le = (1.0 + float(np.sum(arr <= null))) / (b + 1.0)
    p_ge = (1.0 + float(np.sum(arr >= null))) / (b + 1.0)
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def clause1_holds(line: H2Line) -> bool:
    """Clause 1: the raw high-minus-low difference is positive and its interval excludes zero."""
    return bool(line.diff_raw.point > 0.0 and line.diff_raw.lower > 0.0)


@dataclass(frozen=True)
class H2Verdict:
    """The outcome of the rule, with the numbers it used.

    ``clause1_p`` is the bootstrap p-value of the primary line's raw difference (the scoreboard
    input). ``primary_share`` / ``primary_share_lower`` are the share's point estimate and interval
    lower bound (NaN when the share interval is not defined). ``market_shares`` maps each
    market-series line to its share point estimate. ``notes`` are the reading notes described in the
    module docstring (not rule inputs).
    """

    verdict: Verdict
    clause1: bool
    clause1_p: float
    clause2: Clause2
    threshold: float
    primary_name: str
    primary_share: float
    primary_share_lower: float
    market_shares: Mapping[str, float]
    residual: ResidualReading
    notes: tuple[str, ...]


def residual_reading(lines: Sequence[H2Line]) -> ResidualReading:
    """Is the non-volatility part of the change (the adjusted high-minus-low difference) identified?

    "identified, positive" only if every line's adjusted-difference interval lies above zero,
    "identified, negative" only if every one lies below zero; anything else (an interval spanning
    zero on any line, or intervals on opposite sides) is "not identified". Presentation, not a rule
    input.
    """
    if not lines:
        raise H2Error("need at least one line")
    if all(line.diff_adjusted.lower > 0.0 for line in lines):
        return "identified, positive"
    if all(line.diff_adjusted.upper < 0.0 for line in lines):
        return "identified, negative"
    return "not identified"


def _check_lines(primary: H2Line, market: Sequence[H2Line]) -> None:
    if primary.delta_mode != "pair":
        raise H2Error(f"the primary line must use the per-asset delta, got {primary.delta_mode!r}")
    if not market:
        raise H2Error("need at least one market-series line (the 'partly supported' branch)")
    everything = [primary, *market]
    for line in market:
        if line.delta_mode != "series":
            raise H2Error(f"line {line.name!r} must use a market-series delta")
    for line in everything:
        if line.pairs != "cross":
            raise H2Error(f"line {line.name!r}: the rule is defined for cross-sector pairs")
        if line.alignment != "predetermined":
            raise H2Error(f"line {line.name!r}: the rule is defined for predetermined labels")
    if len({(line.regime, line.reference, line.n_rows) for line in everything}) != 1:
        raise H2Error("all lines must contrast the same regimes on the same sample")
    names = [line.name for line in everything]
    if len(set(names)) != len(names):
        raise H2Error(f"line names must be distinct, got {names}")


def apply_h2_rule(
    primary: H2Line,
    market: Sequence[H2Line],
    *,
    threshold: float = THRESHOLD,
) -> H2Verdict:
    """Apply the pre-fixed H2 reading rule (module docstring) to the table lines.

    ``primary`` is the per-asset-delta line, ``market`` the market-series lines (SPY return, VIX log
    change). Raises :class:`~etfcorr.correlation.regime_correlation.H2Error` when the lines do not
    match the rule's definition (wrong delta mode, pair scope, alignment or sample).
    """
    if not 0.0 < threshold <= 1.0:
        raise H2Error(f"threshold must lie in (0, 1], got {threshold}")
    _check_lines(primary, market)
    all_lines = [primary, *market]

    c1 = clause1_holds(primary)
    p_val = bootstrap_p_value(primary.diff_raw.replicates)
    market_shares = {line.name: float(line.vol_share.point) for line in market}
    share = float(primary.vol_share.point)

    notes: list[str] = []
    if c1 and np.isnan(share):
        raise H2Error("the primary share is undefined although clause 1 holds")
    if not c1:
        clause2: Clause2 = "not evaluated"
        verdict: Verdict = "not supported"
        if primary.diff_raw.point <= 0.0:
            notes.append("clause 1 failed: the raw high-minus-low difference is not positive")
        else:
            notes.append(
                "clause 1 failed: the raw difference is positive but its interval includes 0"
            )
    else:
        if share >= threshold:
            clause2, verdict = "supported", "supported"
        elif any(s >= threshold for s in market_shares.values()):
            clause2, verdict = "partly supported", "partly supported"
        else:
            clause2, verdict = "not supported", "not supported"
        lower = float(primary.vol_share.lower)
        if clause2 == "supported":
            if np.isnan(lower):
                notes.append("the share interval is not defined")
            elif lower >= threshold:
                notes.append(
                    f"the lower bound of the primary share interval is also >= {threshold}"
                )
            else:
                notes.append(
                    f"the lower bound of the primary share interval ({lower:.3f}) is below "
                    f"{threshold}: the point rule is met, the interval does not clear it"
                )
    over = [line.name for line in all_lines if line.share_exceeds_one]
    if over:
        notes.append(
            "share above 1 on " + ", ".join(over) + ": the adjustment overshoots (the adjusted "
            "difference has the opposite sign to the raw one); this is not 'more than 100% "
            "explained by volatility'"
        )
    residual = residual_reading(all_lines)
    if residual == "not identified":
        notes.append(
            "the non-volatility part of the change is not identified: the adjusted high-minus-low "
            "difference changes sign or spans zero across the lines"
        )
    return H2Verdict(
        verdict=verdict,
        clause1=c1,
        clause1_p=p_val,
        clause2=clause2,
        threshold=float(threshold),
        primary_name=primary.name,
        primary_share=share,
        primary_share_lower=float(primary.vol_share.lower),
        market_shares=market_shares,
        residual=residual,
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class SensitivityReading:
    """Whether one robustness variant of the primary line gives the same reading as the rule.

    ``same_reading`` is True when, on that variant alone, clause 1 holds and the primary-style share
    is at least the threshold. It is a stability check, not a second verdict.
    """

    name: str
    clause1: bool
    share: float
    share_lower: float
    same_reading: bool


def sensitivity_reading(line: H2Line, *, threshold: float = THRESHOLD) -> SensitivityReading:
    """Read one robustness variant of the primary line (day-t labels, other block lengths, ...).

    Unlike :func:`apply_h2_rule` this does not require predetermined alignment or cross-sector
    pairs, because the variants differ from the primary line exactly in those respects.
    """
    if not 0.0 < threshold <= 1.0:
        raise H2Error(f"threshold must lie in (0, 1], got {threshold}")
    c1 = clause1_holds(line)
    share = float(line.vol_share.point)
    return SensitivityReading(
        name=line.name,
        clause1=c1,
        share=share,
        share_lower=float(line.vol_share.lower),
        same_reading=bool(c1 and share >= threshold),
    )


def _fmt_stat(s: H2Stat) -> str:
    if np.isnan(s.lower):
        return f"{s.point:+.4f} [CI not defined]"
    return f"{s.point:+.4f} [{s.lower:+.4f}, {s.upper:+.4f}]"


def verdict_lines(verdict: H2Verdict, lines: Sequence[H2Line]) -> list[str]:
    """Human-readable report of the rule and its inputs (settings, per-line numbers, verdict)."""
    out = [
        "H2 verdict (pre-fixed rule, in-sample and descriptive; high tercile minus low tercile)",
        f"  rule: clause 1 = raw difference > 0 with an interval excluding 0; clause 2 = "
        f"vol_share point estimate >= {verdict.threshold} on the primary line "
        "('partly' if only a market-series line reaches it)",
    ]
    for line in lines:
        out.append(
            f"  {line.name} [{line.delta_mode} delta]: raw {_fmt_stat(line.diff_raw)}; adjusted "
            f"{_fmt_stat(line.diff_adjusted)}; share {_fmt_stat(line.vol_share)}"
        )
    out.append(
        f"  clause 1 (direction): {'MET' if verdict.clause1 else 'NOT MET'} "
        f"(bootstrap p = {verdict.clause1_p:.4f}, two-sided, +1 corrected)"
    )
    out.append(f"  clause 2 (a large part): {verdict.clause2.upper()}")
    out.append(f"  VERDICT: H2 {verdict.verdict.upper()}")
    for note in verdict.notes:
        out.append(f"  note: {note}")
    out.append(
        "  caveats: Forbes-Rigobon assumes no omitted variables or endogeneity; in-sample "
        "descriptive statistic; about 100 high-VIX episodes; Holm across H1-H6 is Phase 8"
    )
    return out
