"""Leakage and purity tests for step 3.6 (``network_stability`` and ``scripts/mst_figure_3_6.py``).

Two risks matter. (1) The **sealed window**: by default the script must not read any row dated after
the validation end, so every output (tables, summary, figure) must be unchanged when later rows are
altered. (2) **Lookahead inside the library**: the rolling persistence at date t must be a function
of rows up to t. Each detector is shown to FAIL on a deliberately leaky twin (``restrict_window``
replaced by the identity, or a figure computed on the full sample), so the detector itself is
tested. A last check confirms the guard is not vacuous: moving the end later does change the output.
"""

import importlib.util
import itertools
import shutil
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

from etfcorr.correlation import network_stability as NS
from etfcorr.synthetic import make_factor_returns

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "mst_figure_3_6.py"
CUT = pd.Timestamp("2022-12-31")
ARGS = [
    "--horizons",
    "D",
    "W",
    "--n-bootstrap",
    "20",
    "--n-shuffles",
    "3",
    "--n-permutations",
    "20",
    "--n-null-draws",
    "20",
    "--rolling",
    "D:252:63",
    "W:104:13",
]
OUTPUTS = [
    ("tables", "mst_edges_3_6.csv"),
    ("tables", "mst_summary_3_6.csv"),
    ("tables", "mst_cross_horizon_3_6.csv"),
    ("tables", "mst_rolling_3_6.csv"),
    ("tables", "mst_summary_3_6.md"),
    ("figures", "mst_grid_3_6.svg"),
]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mst_figure_3_6_leak", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sectors() -> dict[str, str]:
    with open(REPO / "config" / "universe.yaml") as f:
        universe = yaml.safe_load(f)
    return {t["ticker"]: t["sector"] for t in universe["tickers"] if t["panel"] == "core"}


def _blocks(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    n = len(frame) // k * k
    head = frame.iloc[:n]
    out = head.groupby(np.arange(n) // k).sum()
    out.index = head.index[k - 1 :: k]
    return out


def _panels() -> dict[str, pd.DataFrame]:
    sectors = _sectors()
    sizes = tuple(len(list(g)) for _, g in itertools.groupby(sectors.values()))
    daily = make_factor_returns(
        3150, sizes, market_loading=0.3, sector_loading=0.8, seed=9, start="2014-01-02"
    ).returns
    daily.columns = pd.Index(list(sectors))
    return {"D": daily, "W": _blocks(daily, 5)}


def _alter_after_cut(panels: dict[str, pd.DataFrame], how: str) -> dict[str, pd.DataFrame]:
    """Copies of the panels with the rows dated after the cut replaced (noise or shuffled)."""
    out = {}
    rng = np.random.default_rng(1)
    for key, frame in panels.items():
        copy = frame.copy()
        later = copy.index > CUT
        assert later.sum() > 20
        values = copy.loc[later].to_numpy()
        if how == "noise":
            copy.loc[later] = rng.normal(0.0, 0.05, size=values.shape)
        else:
            copy.loc[later] = values[rng.permutation(len(values))][:, ::-1]
        out[key] = copy
    return out


def _run(
    root: Path,
    panels: dict[str, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str] | None = None,
    leaky: bool = False,
) -> dict[str, bytes]:
    root.mkdir(parents=True)
    (root / "config").mkdir()
    for name in ("universe.yaml", "experiment.yaml"):
        shutil.copy(REPO / "config" / name, root / "config" / name)
    module = _load_script()
    monkeypatch.setattr(module, "find_repo_root", lambda: root)
    monkeypatch.setattr(module, "read_returns", lambda path: panels[path.stem.split("_")[-1]])
    if leaky:  # the twin: the sealed-window cut is silently not applied
        monkeypatch.setattr(module, "restrict_window", lambda frame, start=None, end=None: frame)
    assert module.main([*ARGS, *(extra or [])]) == 0
    return {name: (root / "reports" / folder / name).read_bytes() for folder, name in OUTPUTS}


def _differs(a: dict[str, bytes], b: dict[str, bytes]) -> bool:
    return any(a[name] != b[name] for name in a)


@pytest.fixture(scope="module")
def panels() -> dict[str, pd.DataFrame]:
    return _panels()


@pytest.fixture
def baseline(
    tmp_path: Path, panels: dict[str, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> dict[str, bytes]:
    return _run(tmp_path / "base", panels, monkeypatch)


# --------------------------------------------------------------------------- the sealed window


@pytest.mark.parametrize("how", ["noise", "shuffle"])
def test_every_output_is_unchanged_when_the_rows_after_the_cut_are_altered(
    tmp_path: Path,
    panels: dict[str, pd.DataFrame],
    baseline: dict[str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    how: str,
) -> None:
    altered = _run(tmp_path / how, _alter_after_cut(panels, how), monkeypatch)
    assert not _differs(baseline, altered)


@pytest.mark.parametrize("how", ["noise", "shuffle"])
def test_the_detector_fails_on_a_twin_that_ignores_the_cut(
    tmp_path: Path,
    panels: dict[str, pd.DataFrame],
    baseline: dict[str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    how: str,
) -> None:
    clean = _run(tmp_path / "leaky_clean", panels, monkeypatch, leaky=True)
    dirty = _run(tmp_path / f"leaky_{how}", _alter_after_cut(panels, how), monkeypatch, leaky=True)
    assert _differs(clean, dirty), "altered future rows must show up in a leaky run"


def test_moving_the_end_later_changes_the_output(
    tmp_path: Path,
    panels: dict[str, pd.DataFrame],
    baseline: dict[str, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    later = _run(tmp_path / "later", panels, monkeypatch, extra=["--end", "2024-12-31"])
    assert _differs(baseline, later)
    assert b"WARNING" in later["mst_summary_3_6.md"]


def test_no_frame_passed_to_the_library_reaches_past_the_cut(
    tmp_path: Path, panels: dict[str, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    def spy_run(leaky: bool) -> list[pd.Timestamp]:
        module = _load_script()
        seen: list[pd.Timestamp] = []

        def watch(name: str) -> None:
            original: Callable[..., object] = getattr(module, name)

            def wrapper(frame: pd.DataFrame, *args: object, **kwargs: object) -> object:
                seen.append(pd.Timestamp(frame.index.max()))
                return original(frame, *args, **kwargs)

            monkeypatch.setattr(module, name, wrapper)

        watch("bootstrap_mst_edge_support")
        watch("rolling_survival_shuffle_null")
        root = tmp_path / ("spy_leaky" if leaky else "spy")
        (root / "config").mkdir(parents=True)
        for name in ("universe.yaml", "experiment.yaml"):
            shutil.copy(REPO / "config" / name, root / "config" / name)
        monkeypatch.setattr(module, "find_repo_root", lambda: root)
        monkeypatch.setattr(module, "read_returns", lambda path: panels[path.stem.split("_")[-1]])
        if leaky:
            monkeypatch.setattr(
                module, "restrict_window", lambda frame, start=None, end=None: frame
            )
        module.main([*ARGS, "--quick"])
        return seen

    honest = spy_run(leaky=False)
    assert honest and max(honest) <= CUT
    assert max(spy_run(leaky=True)) > CUT  # non-vacuity: the spy sees the future in the twin


# --------------------------------------------------------------------------- library lookahead


def _frame() -> pd.DataFrame:
    return make_factor_returns(
        1200, (5, 5, 5), market_loading=0.3, sector_loading=0.8, seed=2
    ).returns


def test_the_real_rolling_series_at_t_does_not_depend_on_later_rows() -> None:
    frame = _frame()
    full = NS.rolling_survival_shuffle_null(frame, 120, 60, n_shuffles=2, seed=0).real
    cut = 700
    truncated = NS.rolling_survival_shuffle_null(
        frame.iloc[:cut], 120, 60, n_shuffles=2, seed=0
    ).real
    pd.testing.assert_series_equal(full.loc[truncated.index], truncated)
    changed = frame.copy()
    changed.iloc[cut:] = np.random.default_rng(0).normal(0, 0.02, size=changed.iloc[cut:].shape)
    again = NS.rolling_survival_shuffle_null(changed, 120, 60, n_shuffles=2, seed=0).real
    pd.testing.assert_series_equal(again.loc[truncated.index], truncated)


def test_a_look_ahead_twin_of_the_rolling_series_is_caught_by_the_same_check() -> None:
    """Twin: windows are centred (they include 60 future rows). The prefix property must break."""
    frame = _frame()

    def centred(returns: pd.DataFrame, window: int, step: int) -> pd.Series:
        from etfcorr.correlation.networks import build_mst, edge_survival_ratio

        values, index, previous = [], [], None
        for pos in range(window - 1, len(returns), step):
            chunk = returns.iloc[max(0, pos - window // 2) : pos + window // 2 + 1]
            current = build_mst(chunk.corr())
            values.append(np.nan if previous is None else edge_survival_ratio(current, previous))
            index.append(returns.index[pos])
            previous = current
        return pd.Series(values, index=index)

    full = centred(frame, 120, 60)
    truncated = centred(frame.iloc[:700], 120, 60)
    common = truncated.index.intersection(full.index)
    a, b = full.loc[common].to_numpy(), truncated.loc[common].to_numpy()
    assert not np.allclose(a, b, equal_nan=True), "the detector must flag centred windows"


def test_the_library_functions_do_not_modify_their_inputs() -> None:
    frame = _frame()
    sectors = {c: f"s{i // 5}" for i, c in enumerate(frame.columns)}
    before = frame.copy()
    es = NS.bootstrap_mst_edge_support(frame, n_bootstrap=20, seed=0)
    NS.rolling_survival_shuffle_null(frame, 120, 120, n_shuffles=2, seed=0)
    NS.cluster_sector_agreement(frame.corr(), sectors, n_permutations=10, seed=0)
    NS.survival_matrix({"a": es.mst, "b": es.mst})
    pd.testing.assert_frame_equal(frame, before)
