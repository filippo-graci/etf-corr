"""Tests for ``evaluation.null_benchmark`` (step 8.4) and ``evaluation.scoreboard`` (step 8.6)."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from etfcorr.config import DateWindow
from etfcorr.evaluation import null_benchmark as NB
from etfcorr.evaluation import scoreboard as S
from etfcorr.splits import expanding_walk_forward

INDEX = pd.bdate_range("2018-01-01", periods=700, name="date")
TEST = DateWindow.model_validate([str(INDEX[400].date()), "2030-12-31"])
FOLDS = expanding_walk_forward(INDEX, train_start=INDEX[0], test=TEST, horizon=5, embargo=2)


def test_random_features_are_persistent_unit_variance_and_seeded() -> None:
    x = NB.random_features(pd.RangeIndex(20000), 3, 0.9, np.random.default_rng(0))
    np.testing.assert_allclose(x.var(), 1.0, atol=0.08)
    assert x["noise_0"].autocorr(1) == pytest.approx(0.9, abs=0.02)
    assert abs(x.corr().to_numpy()[0, 1]) < 0.05
    again = NB.random_features(pd.RangeIndex(20000), 3, 0.9, np.random.default_rng(0))
    pd.testing.assert_frame_equal(x, again)
    with pytest.raises(ValueError):
        NB.random_features(pd.RangeIndex(5), 3, 1.0, np.random.default_rng(0))


def test_a_skilful_forecast_ranks_first_and_noise_ranks_in_the_middle() -> None:
    rng = np.random.default_rng(1)
    signal = pd.Series(rng.standard_normal(700), index=INDEX)
    y = signal + 0.5 * pd.Series(rng.standard_normal(700), index=INDEX)
    dates = INDEX[400:]
    null = NB.random_signal_losses(
        y, FOLDS, dates, n_features=4, n_draws=40, horizon=5, embargo=2, alpha_grid=(0.1, 1.0)
    )
    assert len(null) == 40 and null.nunique() == 40
    assert null.mean() == pytest.approx(
        float(y.loc[dates].var()), rel=0.15
    )  # no skill: the variance
    skilful = float(((y - signal) ** 2).loc[dates].mean())
    ranks = NB.percentile_ranks(null, {"skilful": skilful, "typical": float(null.median())})
    by_model = ranks.set_index("model")["percentile_rank"]
    assert by_model["skilful"] == pytest.approx(1 / 41) and 0.4 < by_model["typical"] < 0.6


def entries() -> list[S.Entry]:
    return [
        S.Entry("H1", "a", "not supported", False, 0.998, "w", "s"),
        S.Entry("H2", "b", "supported", True, 0.001, "w", "s"),
        S.Entry("H3", "c", "not supported", False, 0.378, "w", "s"),
        S.Entry("H4", "d", "deferred, not attempted", False, None, "-", "s"),
        S.Entry("H5", "e", "supported", True, 0.02, "w", "s"),
        S.Entry("H6", "f", "not supported", False, 0.9, "w", "s"),
    ]


def test_holm_is_statsmodels_over_six_with_untested_entered_as_one() -> None:
    table = S.scoreboard(entries()).set_index("hypothesis")
    expected = multipletests([0.998, 0.001, 0.378, 1.0, 0.02, 0.9], method="holm")[1]
    assert table.loc["H2", "p_holm"] == pytest.approx(0.006) == pytest.approx(expected[1])
    assert table.loc["H5", "p_holm"] == pytest.approx(0.1)  # 0.02 x 5: lost to the correction
    assert table["supported_after_holm"].to_dict() == {
        "H1": False,
        "H2": True,
        "H3": False,
        "H4": False,
        "H5": False,
        "H6": False,
    }
    assert pd.isna(table.loc["H4", "p_holm"]) and pd.isna(table.loc["H4", "p_value"])
    text = S.to_markdown(S.scoreboard(entries()))
    assert text.count("\n") == 8 and "not tested" in text and "| H5 |" in text


def test_scoreboard_refuses_inconsistent_entries() -> None:
    with pytest.raises(ValueError, match="unique"):
        S.scoreboard([entries()[0], entries()[0]])
    with pytest.raises(ValueError, match="not tested"):
        S.scoreboard([S.Entry("H4", "d", "supported", True, None, "-", "s")])
    with pytest.raises(ValueError, match="outside"):
        S.scoreboard([S.Entry("H1", "a", "x", False, 1.5, "w", "s")])


def test_null_losses_are_scored_on_the_given_dates_only() -> None:
    rng = np.random.default_rng(2)
    y = pd.Series(rng.standard_normal(700), index=INDEX)
    y.iloc[650:] *= 10.0  # a loud tail: a loss over all dates would be far larger
    kw = {"n_features": 2, "n_draws": 3, "horizon": 5, "embargo": 2, "alpha_grid": (1.0,)}
    quiet = NB.random_signal_losses(y, FOLDS, INDEX[400:650], **kw)  # type: ignore[arg-type]
    everything = NB.random_signal_losses(y, FOLDS, INDEX[400:], **kw)  # type: ignore[arg-type]
    assert (quiet < 2.0).all() and (everything > 5.0).all()


def test_a_small_p_value_does_not_make_an_unsupported_hypothesis_hold() -> None:
    """H5 and H6 had tiny p-values in the wrong direction: the verdict flag decides, not p alone."""
    rows = [S.Entry("H1", "a", "not supported", False, 0.0001, "w", "s"), entries()[1]]
    table = S.scoreboard(rows).set_index("hypothesis")
    assert table["supported_after_holm"].to_dict() == {"H1": False, "H2": True}


def test_a_supported_non_superiority_claim_holds_after_holm_and_still_counts_in_the_family() -> (
    None
):
    """H4's claim is that a model does NOT beat another: a non-rejection, which Holm cannot undo."""
    rows = [
        entries()[1],
        S.Entry("H4", "d", "supported", True, 0.6, "w", "s", null_claim=True),
        S.Entry("H9", "e", "not supported", False, 0.6, "w", "s", null_claim=True),
    ]
    table = S.scoreboard(rows).set_index("hypothesis")
    assert table["supported_after_holm"].to_dict() == {"H2": True, "H4": True, "H9": False}
    assert table.loc["H2", "p_holm"] == pytest.approx(0.003)  # three real tests in the family
    assert table.loc["H4", "p_holm"] == pytest.approx(1.0)
