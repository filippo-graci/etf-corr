"""Unit tests for etfcorr.config (step 0.3): loader, models, lock, paths, seeds, threads."""

import copy
import hashlib
import json
import os
import random
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from etfcorr import config as C
from etfcorr.config import ConfigError

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

_STATEMENTS = {
    "H1": "After removing the market mode the structure is more similar across horizons.",
    "H2": "Cross-sector correlation is higher in high-VIX regimes, largely through volatility.",
    "H3": "A machine-learning model beats the best simple baseline for monthly correlation.",
    "H4": "A graph neural network does not beat LightGBM with graph-derived features.",
    "H5": "Shrunk covariance estimates give lower out-of-sample minimum-variance volatility.",
    "H6": "Hedge effectiveness is lower in the worst-decile stress periods than in calm ones.",
}


def exp_dict() -> dict[str, Any]:
    return {
        "version": 1,
        "seeds": [0, 1, 2],
        "device": "cuda",
        "threads": 8,
        "horizons": ["D", "W", "M", "Q"],
        "windows_days": [63, 126, 252],
        "ewma_lambda": 0.94,
        "splits": {
            "train": ["2016-01-01", "2020-12-31"],
            "val": ["2021-01-01", "2022-12-31"],
            "test": ["2023-01-01", "2026-09-18"],
            "walk_forward": {"initial_years": 5, "refit_every": "QE", "embargo_days": 5},
        },
        "costs_bps": [0, 5, 10],
        "models": {},
        "inference": {
            "alpha": 0.05,
            "multiplicity": "holm",
            "bootstrap": "stationary",
            "n_bootstrap": 2000,
        },
        "hypotheses": [
            {
                "id": h,
                "phase": 3,
                "statement": text,
                "test": "A pre-specified test with confidence intervals.",
            }
            for h, text in _STATEMENTS.items()
        ],
        "exploratory": [],
    }


def uni_dict() -> dict[str, Any]:
    return {
        "version": 1,
        "start_date": "2014-01-01",
        "min_adv_usd": 10_000_000,
        "tickers": [
            {"ticker": "SMH", "sector": "ai_semis", "panel": "core", "name": "Semis"},
            {
                "ticker": "SOXX",
                "sector": "ai_semis",
                "panel": "core",
                "name": "Semis 2",
                "overlaps_with": ["SMH"],
            },
            {"ticker": "XLE", "sector": "energy_trad", "panel": "core", "name": "Energy"},
            {"ticker": "SPY", "sector": "control", "panel": "control", "name": "Market"},
        ],
    }


def dump(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def make_exp(**changes: Any) -> C.ExperimentConfig:
    data = exp_dict()
    data.update(changes)
    return C.ExperimentConfig.model_validate(data)


def make_uni(**changes: Any) -> C.UniverseConfig:
    data = uni_dict()
    data.update(changes)
    return C.UniverseConfig.model_validate(data)


def make_repo(tmp_path: Path, exp: dict[str, Any] | None = None, lock: bool = True) -> Path:
    (tmp_path / "config").mkdir()
    dump(tmp_path / "config" / "experiment.yaml", exp if exp is not None else exp_dict())
    dump(tmp_path / "config" / "universe.yaml", uni_dict())
    if lock:
        paths = C.Paths.from_root(tmp_path)
        cfg = C.load_experiment(paths.experiment_yaml)
        C.write_hypotheses_lock(cfg, paths.hypotheses_lock, locked_on=date(2026, 9, 21))
    return tmp_path


# --------------------------------------------------------------------------------------
# strict YAML loader
# --------------------------------------------------------------------------------------


def test_duplicate_top_level_key_is_rejected(tmp_path: Path) -> None:
    f = tmp_path / "x.yaml"
    f.write_text("a: 1\nb: 2\na: 3\n")
    with pytest.raises(ConfigError, match="duplicate key 'a'"):
        C.load_yaml(f)


def test_duplicate_nested_key_is_rejected(tmp_path: Path) -> None:
    f = tmp_path / "x.yaml"
    f.write_text("outer:\n  alpha: 0.05\n  alpha: 0.10\n")
    with pytest.raises(ConfigError, match="duplicate key 'alpha'"):
        C.load_yaml(f)


def test_plain_safe_load_would_accept_the_duplicate() -> None:
    """Control: proves the strict loader is needed (yaml.safe_load keeps the last value)."""
    assert yaml.safe_load("alpha: 0.05\nalpha: 0.10\n") == {"alpha": 0.10}


def test_load_yaml_failure_modes(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        C.load_yaml(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [1, 2\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        C.load_yaml(bad)
    seq = tmp_path / "seq.yaml"
    seq.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError, match="mapping"):
        C.load_yaml(seq)
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(ConfigError, match="mapping"):
        C.load_yaml(empty)


# --------------------------------------------------------------------------------------
# experiment model
# --------------------------------------------------------------------------------------


def test_valid_experiment_loads_from_file(tmp_path: Path) -> None:
    cfg = C.load_experiment(dump(tmp_path / "e.yaml", exp_dict()))
    assert cfg.splits.train.start == date(2016, 1, 1)
    assert cfg.horizons == ("D", "W", "M", "Q")
    assert [h.id for h in cfg.hypotheses] == list(C.CONFIRMATORY_IDS)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    data = exp_dict()
    data["seedz"] = [1]
    with pytest.raises(ConfigError, match="seedz"):
        C.load_experiment(dump(tmp_path / "e.yaml", data))


def test_config_is_frozen() -> None:
    cfg = make_exp()
    with pytest.raises(Exception, match="frozen"):
        cfg.threads = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("splits", "val"), ["2020-06-01", "2022-12-31"]),  # overlaps train
        (("splits", "val"), ["2020-12-31", "2022-12-31"]),  # shares its first day with train
        (("splits", "test"), ["2022-12-31", "2026-09-18"]),  # shares its first day with val
        (("splits", "test"), ["2022-06-01", "2026-09-18"]),  # overlaps val
        (("splits", "train"), ["2020-12-31", "2016-01-01"]),  # reversed
        (("splits", "train"), ["2016-01-01", "2016-01-01"]),  # zero length
        (("splits", "train"), ["2016-01-01", "2018-01-01", "2020-12-31"]),  # three items
        (("splits", "test"), ["not-a-date", "2026-09-18"]),
        (("seeds",), [1, 1]),
        (("seeds",), [-1]),
        (("seeds",), [2**32]),
        (("seeds",), []),
        (("horizons",), ["D", "X"]),
        (("horizons",), ["D", "D"]),
        (("windows_days",), [126, 63]),
        (("windows_days",), [5, 63]),
        (("ewma_lambda",), 1.0),
        (("ewma_lambda",), 0.0),
        (("costs_bps",), [5, 10]),  # zero-cost case missing
        (("costs_bps",), [0, -5]),
        (("threads",), 0),
        (("device",), "tpu"),
        (("inference", "alpha"), 0.9),
        (("inference", "multiplicity"), "none"),
        (("inference", "n_bootstrap"), 10),
        (("splits", "walk_forward", "refit_every"), "W"),
        (("splits", "walk_forward", "embargo_days"), -1),
    ],
)
def test_invalid_experiment_values_are_rejected(path: tuple[str, ...], value: Any) -> None:
    data = exp_dict()
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        C.ExperimentConfig.model_validate(data)


def test_hypothesis_set_must_be_exactly_h1_to_h6() -> None:
    base = exp_dict()["hypotheses"]
    for bad in (base[:5], base[::-1], [*base, {**base[0], "id": "H7"}], base[:2] + base[3:]):
        data = exp_dict()
        data["hypotheses"] = bad
        with pytest.raises(ValueError, match="H1"):
            C.ExperimentConfig.model_validate(data)


def test_exploratory_ids_must_be_e_prefixed_and_unique() -> None:
    hyp = {**exp_dict()["hypotheses"][0], "id": "E1"}
    assert make_exp(exploratory=[hyp]).exploratory[0].id == "E1"
    for bad in ([{**hyp, "id": "H7"}], [hyp, hyp]):
        with pytest.raises(ValueError):
            make_exp(exploratory=bad)


def test_hypothesis_text_whitespace_is_normalised() -> None:
    h = C.Hypothesis(id="H1", phase=3, statement="a  b\n c " * 8, test="x\n y " * 10)
    assert "  " not in h.statement
    assert "\n" not in h.statement
    assert h.statement == h.statement.strip()


# --------------------------------------------------------------------------------------
# universe model
# --------------------------------------------------------------------------------------


def test_valid_universe_and_helpers(tmp_path: Path) -> None:
    uni = C.load_universe(dump(tmp_path / "u.yaml", uni_dict()))
    assert uni.tickers_in("core") == ("SMH", "SOXX", "XLE")
    assert uni.tickers_in("control") == ("SPY",)
    assert uni.tickers_in("extended") == ()
    assert uni.sectors() == ("ai_semis", "energy_trad")  # control excluded, order preserved


def test_universe_rejects_inconsistent_entries() -> None:
    def mutate(fn: Any) -> dict[str, Any]:
        data = copy.deepcopy(uni_dict())
        fn(data)
        return data

    bad_cases = {
        "duplicate": mutate(lambda d: d["tickers"].append(d["tickers"][0])),
        "control sector, core panel": mutate(lambda d: d["tickers"][3].update(panel="core")),
        "core sector, control panel": mutate(lambda d: d["tickers"][0].update(panel="control")),
        "unknown overlap": mutate(lambda d: d["tickers"][1].update(overlaps_with=["QQQ"])),
        "self overlap": mutate(lambda d: d["tickers"][1].update(overlaps_with=["SOXX"])),
        "lowercase ticker": mutate(lambda d: d["tickers"][0].update(ticker="smh")),
        "empty list": mutate(lambda d: d.update(tickers=[])),
        "unknown sector": mutate(lambda d: d["tickers"][0].update(sector="crypto")),
        "bad adv": mutate(lambda d: d.update(min_adv_usd=0)),
        "extra key": mutate(lambda d: d["tickers"][0].update(color="red")),
    }
    for label, data in bad_cases.items():
        with pytest.raises(ValueError):
            C.UniverseConfig.model_validate(data)
        assert label  # keeps the failing case visible in tracebacks


def test_cross_check_warmup() -> None:
    exp = make_exp()
    assert C.cross_check(exp, make_uni(start_date="2014-01-01")) == []
    # 252 trading days need about 353 calendar days of history before the train start
    assert C.cross_check(exp, make_uni(start_date="2015-06-01")) != []
    assert C.cross_check(exp, make_uni(start_date="2015-01-01")) == []  # 365 days: enough


# --------------------------------------------------------------------------------------
# fingerprints and the pre-registration lock
# --------------------------------------------------------------------------------------


def test_fingerprint_known_answer() -> None:
    """Independent re-computation of the documented recipe (canonical JSON, SHA-256)."""
    cfg = make_exp()
    payload = {
        "hypotheses": [
            {
                "id": h,
                "phase": 3,
                "statement": t,
                "test": "A pre-specified test with confidence intervals.",
            }
            for h, t in _STATEMENTS.items()
        ],
        "inference": {
            "alpha": 0.05,
            "multiplicity": "holm",
            "bootstrap": "stationary",
            "n_bootstrap": 2000,
        },
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert C.preregistration_fingerprint(cfg) == hashlib.sha256(text.encode()).hexdigest()


def test_lock_roundtrip_and_file_content(tmp_path: Path) -> None:
    cfg = make_exp()
    lock_path = tmp_path / "h.lock"
    lock = C.write_hypotheses_lock(cfg, lock_path, locked_on=date(2026, 9, 21))
    assert json.loads(lock_path.read_text()) == lock
    assert lock["locked_on"] == "2026-09-21"
    assert lock["hypothesis_ids"] == list(C.CONFIRMATORY_IDS)
    C.verify_hypotheses_lock(cfg, lock_path)


def _tampered(field_path: tuple[Any, ...], value: Any) -> C.ExperimentConfig:
    data = exp_dict()
    node: Any = data
    for key in field_path[:-1]:
        node = node[key]
    node[field_path[-1]] = value
    return C.ExperimentConfig.model_validate(data)


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("hypotheses", 0, "statement"), _STATEMENTS["H1"].replace("more", "less")),
        (("hypotheses", 2, "statement"), _STATEMENTS["H3"] + " (amended)"),
        (("hypotheses", 5, "test"), "A different pre-specified test, chosen after results."),
        (("hypotheses", 4, "phase"), 9),
        (("inference", "alpha"), 0.10),
        (("inference", "multiplicity"), "bh"),
        (("inference", "n_bootstrap"), 500),
        (("inference", "bootstrap"), "circular"),
    ],
)
def test_lock_detects_any_change_to_preregistered_content(
    tmp_path: Path, field_path: tuple[Any, ...], value: Any
) -> None:
    lock_path = tmp_path / "h.lock"
    C.write_hypotheses_lock(make_exp(), lock_path, locked_on=date(2026, 9, 21))
    with pytest.raises(ConfigError, match="changed after locking"):
        C.verify_hypotheses_lock(_tampered(field_path, value), lock_path)


def test_naive_id_only_check_would_miss_tampering(tmp_path: Path) -> None:
    """Control for the test above: a lock that only compared ids passes a rewritten hypothesis."""
    original = make_exp()
    edited = _tampered(("hypotheses", 0, "statement"), "H1 was rewritten after seeing results.")
    ids = lambda c: [h.id for h in c.hypotheses]  # noqa: E731
    assert ids(original) == ids(edited)  # naive check: passes
    lock_path = tmp_path / "h.lock"
    C.write_hypotheses_lock(original, lock_path, locked_on=date(2026, 9, 21))
    with pytest.raises(ConfigError):  # real check: fails
        C.verify_hypotheses_lock(edited, lock_path)


def test_lock_ignores_rewrapping_and_non_preregistered_fields(tmp_path: Path) -> None:
    lock_path = tmp_path / "h.lock"
    base = make_exp()
    C.write_hypotheses_lock(base, lock_path, locked_on=date(2026, 9, 21))

    rewrapped = exp_dict()
    rewrapped["hypotheses"][0]["statement"] = _STATEMENTS["H1"].replace(" ", "\n  ")
    hyp_e1 = {**exp_dict()["hypotheses"][0], "id": "E1"}
    variants = [
        C.ExperimentConfig.model_validate(rewrapped),
        make_exp(seeds=[7, 8]),
        make_exp(threads=4),
        make_exp(models={"lgbm": {"num_leaves": [15, 31]}}),
        make_exp(exploratory=[hyp_e1]),
    ]
    for cfg in variants:
        C.verify_hypotheses_lock(cfg, lock_path)  # must not raise
    # ...but the whole-config hash does move for every real change
    assert C.config_hash(variants[1]) != C.config_hash(base)
    assert C.config_hash(variants[3]) != C.config_hash(base)
    assert C.config_hash(variants[0]) == C.config_hash(base)  # re-wrapping is a no-op


def test_lock_failure_modes(tmp_path: Path) -> None:
    cfg = make_exp()
    lock_path = tmp_path / "h.lock"
    with pytest.raises(ConfigError, match="missing"):
        C.verify_hypotheses_lock(cfg, lock_path)
    lock_path.write_text("{not json")
    with pytest.raises(ConfigError, match="unreadable"):
        C.verify_hypotheses_lock(cfg, lock_path)
    lock_path.write_text("[1, 2]")
    with pytest.raises(ConfigError, match="JSON object"):
        C.verify_hypotheses_lock(cfg, lock_path)
    good = C.write_hypotheses_lock(cfg, lock_path, locked_on=date(2026, 9, 21), overwrite=True)
    for key, bad in (("algorithm", "md5"), ("covers", ["hypotheses"]), ("hypothesis_ids", ["H1"])):
        lock_path.write_text(json.dumps({**good, key: bad}))
        with pytest.raises(ConfigError):
            C.verify_hypotheses_lock(cfg, lock_path)


def test_lock_is_never_silently_overwritten(tmp_path: Path) -> None:
    cfg = make_exp()
    lock_path = tmp_path / "h.lock"
    C.write_hypotheses_lock(cfg, lock_path, locked_on=date(2026, 9, 21))
    before = lock_path.read_text()
    with pytest.raises(ConfigError, match="already exists"):
        C.write_hypotheses_lock(cfg, lock_path, locked_on=date(2027, 1, 1))
    assert lock_path.read_text() == before
    C.write_hypotheses_lock(cfg, lock_path, locked_on=date(2027, 1, 1), overwrite=True)
    assert "2027-01-01" in lock_path.read_text()


def test_run_id() -> None:
    cfg = make_exp()
    rid = C.run_id(cfg, "baseline")
    assert rid == f"baseline-{C.config_hash(cfg)[:12]}"
    assert C.run_id(make_exp(seeds=[9]), "baseline") != rid
    for bad in ("", "a b", "a/b", "a\\b"):
        with pytest.raises(ValueError, match="invalid run name"):
            C.run_id(cfg, bad)


# --------------------------------------------------------------------------------------
# paths and project loading
# --------------------------------------------------------------------------------------


def test_paths_layout(tmp_path: Path) -> None:
    p = C.Paths.from_root(tmp_path)
    assert p.experiment_yaml == p.root / "config" / "experiment.yaml"
    assert p.hypotheses_lock.name == "hypotheses.lock"
    assert p.data_raw == p.root / "data" / "raw"
    assert p.figures == p.root / "reports" / "figures"


def test_output_path_allows_only_writable_roots(tmp_path: Path) -> None:
    p = C.Paths.from_root(tmp_path)
    assert p.output_path("data", "raw", "x.parquet") == p.root / "data" / "raw" / "x.parquet"
    assert p.output_path("artifacts/runs/r1") == p.root / "artifacts" / "runs" / "r1"
    assert p.output_path(p.figures / "f.png") == p.figures / "f.png"
    for bad in (
        "config/experiment.yaml",
        "src/etfcorr/x.py",
        "data",  # the folder itself
        "data/../config/x",
        "../outside.txt",
        "/etc/passwd",
        "",
    ):
        with pytest.raises(ConfigError, match="refusing"):
            p.output_path(bad)


def test_output_path_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    (root / "data" / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ConfigError, match="refusing"):
        C.Paths.from_root(root).output_path("data", "link", "x.txt")


def test_find_repo_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    make_repo(repo)
    nested = repo / "a" / "b"
    nested.mkdir(parents=True)
    assert C.find_repo_root(nested) == repo.resolve()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with pytest.raises(ConfigError, match="no config"):
        C.find_repo_root(elsewhere)


def test_load_project_end_to_end(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    project = C.load_project(repo)
    assert project.experiment.threads == 8
    assert len(project.universe.tickers) == 4
    lines = C.describe_project(project)
    assert lines[0].startswith("config ok")
    assert any("H1" in line and "locked" in line for line in lines)


def test_load_project_detects_edited_hypothesis_on_disk(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    exp_file = repo / "config" / "experiment.yaml"
    exp_file.write_text(exp_file.read_text().replace("more similar", "less similar"))
    with pytest.raises(ConfigError, match="changed after locking"):
        C.load_project(repo)
    C.load_project(repo, verify_lock=False)  # explicit opt-out still loads


def test_load_project_requires_lock_and_consistent_pair(tmp_path: Path) -> None:
    no_lock = tmp_path / "no_lock"
    no_lock.mkdir()
    make_repo(no_lock, lock=False)
    with pytest.raises(ConfigError, match="missing"):
        C.load_project(no_lock)
    C.load_project(no_lock, verify_lock=False)

    short_warmup = tmp_path / "short_warmup"
    short_warmup.mkdir()
    make_repo(short_warmup)
    uni = uni_dict()
    uni["start_date"] = "2016-01-01"
    dump(short_warmup / "config" / "universe.yaml", uni)
    with pytest.raises(ConfigError, match="longest window"):
        C.load_project(short_warmup)


# --------------------------------------------------------------------------------------
# device, seeds, threads
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "usable", "expected"),
    [
        ("cpu", True, ("cpu", False)),
        ("cpu", False, ("cpu", False)),
        ("cuda", True, ("cuda", False)),
        ("cuda", False, ("cpu", True)),
        ("auto", True, ("cuda", False)),
        ("auto", False, ("cpu", False)),
    ],
)
def test_resolve_device(requested: Any, usable: bool, expected: tuple[str, bool]) -> None:
    assert tuple(C.resolve_device(requested, usable)) == expected


def _draws(seed: int) -> tuple[float, float, np.ndarray]:
    gen = C.set_seed(seed, use_torch=False)
    return random.random(), float(np.random.rand()), gen.standard_normal(4)  # noqa: NPY002


def test_set_seed_is_deterministic_and_seed_dependent() -> None:
    a, b, c = _draws(7), _draws(7), _draws(8)
    assert a[:2] == b[:2]
    np.testing.assert_array_equal(a[2], b[2])
    assert a[0] != c[0]
    assert a[1] != c[1]
    assert not np.array_equal(a[2], c[2])


@pytest.mark.parametrize("bad", [-1, 2**32])
def test_set_seed_rejects_out_of_range(bad: int) -> None:
    with pytest.raises(ValueError, match="seed must be"):
        C.set_seed(bad, use_torch=False)


def _fake_torch(cuda: bool) -> Any:
    calls: list[tuple[str, Any]] = []
    return SimpleNamespace(
        calls=calls,
        manual_seed=lambda s: calls.append(("manual_seed", s)),
        use_deterministic_algorithms=lambda flag, warn_only=False: calls.append(
            ("deterministic", (flag, warn_only))
        ),
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            manual_seed_all=lambda s: calls.append(("cuda_seed", s)),
        ),
        backends=SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True)),
    )


def test_set_seed_configures_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    fake = _fake_torch(cuda=True)
    C.set_seed(3, torch_module=fake)
    assert fake.calls == [
        ("manual_seed", 3),
        ("cuda_seed", 3),
        ("deterministic", (True, True)),
    ]
    assert fake.backends.cudnn.deterministic is True
    assert fake.backends.cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_set_seed_torch_cpu_only_and_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    cpu = _fake_torch(cuda=False)
    C.set_seed(3, torch_module=cpu)
    assert ("cuda_seed", 3) not in cpu.calls
    off = _fake_torch(cuda=True)
    C.set_seed(3, torch_module=off, use_torch=False)
    assert off.calls == []


def test_set_seed_without_torch_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(C, "_try_import_torch", lambda: None)
    assert isinstance(C.set_seed(1), np.random.Generator)


def test_thread_env_and_apply() -> None:
    env = C.thread_env(6)
    assert set(env) == set(C.THREAD_ENV_VARS)
    assert set(env.values()) == {"6"}
    for bad in (0, -2):
        with pytest.raises(ValueError, match="thread count"):
            C.thread_env(bad)

    target = {"OMP_NUM_THREADS": "2"}
    applied = C.apply_thread_env(6, target)
    assert target["OMP_NUM_THREADS"] == "2"  # existing value kept
    assert "OMP_NUM_THREADS" not in applied
    assert target["MKL_NUM_THREADS"] == "6"
    C.apply_thread_env(4, target, override=True)
    assert target["OMP_NUM_THREADS"] == "4"


def test_apply_thread_env_defaults_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in C.THREAD_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    C.apply_thread_env(3)
    assert os.environ["OPENBLAS_NUM_THREADS"] == "3"
    for key in C.THREAD_ENV_VARS:  # monkeypatch cannot undo keys it did not see set
        monkeypatch.setenv(key, os.environ[key])


# ---- forecasting block (step 5.1) ----------------------------------------------------


def forecasting_dict() -> dict[str, Any]:
    return {
        "fixed_on": "2026-10-11",
        "primary_target": "T1",
        "primary_horizon_days": 21,
        "horizons_days": {"T1": [21, 63], "T2": [5, 21]},
        "feature_windows_days": [21, 63, 252],
        "simple_baselines": ["persistence", "historical_mean", "har"],
        "linear_models": ["ridge", "lasso", "pcr"],
        "losses": {"T1": "mse_fisher_z", "T2": "qlike"},
        "inner_folds": 3,
        "alpha_grid": [0.01, 0.1, 1.0],
        "l1_ratio": 0.5,
        "pcr_components": [1, 2],
    }


def test_forecasting_block_is_optional_and_validated() -> None:
    assert make_exp().forecasting is None
    f = make_exp(forecasting=forecasting_dict()).forecasting
    assert f is not None and f.primary_horizon_days == 21
    assert f.n_linear_trials == 3 + 3 + 2  # ridge and lasso use the alpha grid, pcr its own


@pytest.mark.parametrize(
    "change",
    [
        {"primary_horizon_days": 5},  # not one of T1's horizons
        {"primary_target": "T3"},  # target without horizons
        {"horizons_days": {"T1": [63, 21], "T2": [5]}},  # not ascending
        {"losses": {"T1": "mse_fisher_z"}},  # targets differ from horizons_days
        {"feature_windows_days": [63, 21]},
        {"alpha_grid": [0.0, 1.0]},
        {"pcr_components": [0]},
        {"simple_baselines": ["arima"]},
        {"unknown_key": 1},
    ],
)
def test_forecasting_block_rejects_inconsistent_designs(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        make_exp(forecasting={**forecasting_dict(), **change})


def test_forecasting_block_moves_the_config_hash_but_not_the_lock() -> None:
    base, with_block = make_exp(), make_exp(forecasting=forecasting_dict())
    assert C.config_hash(base) != C.config_hash(with_block)
    assert C.preregistration_fingerprint(base) == C.preregistration_fingerprint(with_block)


# ---- declared variants and the gbm block (step 6.0) ------------------------------------


def variant_dict(**changes: Any) -> dict[str, Any]:
    data = {
        "id": "V1",
        "declared_on": "2026-10-11",
        "kind": "log_bias_correction",
        "targets": ["T2"],
        "note": "A declared deviation, run on validation only.",
    }
    data.update(changes)
    return data


def test_variants_are_validated_and_found_by_id() -> None:
    grid = variant_dict(id="V2", kind="ridge_grid", ridge_alpha_grid=[1.0, 10.0])
    f = make_exp(forecasting={**forecasting_dict(), "variants": [variant_dict(), grid]}).forecasting
    assert f is not None and f.variant("V2").ridge_alpha_grid == (1.0, 10.0)
    assert f.mcs_size == 0.10 and f.gbm is None
    with pytest.raises(KeyError):
        f.variant("V9")


@pytest.mark.parametrize(
    "variants",
    [
        [variant_dict(), variant_dict()],  # duplicate id
        [variant_dict(id="X1")],
        [variant_dict(kind="ridge_grid")],  # no grid
        [variant_dict(kind="ridge_grid", ridge_alpha_grid=[0.0])],
        [variant_dict(kind="exclude_training_window")],  # no window
        [variant_dict(kind="dcc_drop_assets")],  # no assets
        [variant_dict(note="short")],
    ],
)
def test_malformed_variants_are_rejected(variants: list[dict[str, Any]]) -> None:
    with pytest.raises(ValidationError):
        make_exp(forecasting={**forecasting_dict(), "variants": variants})


def test_gbm_block_checks_its_search_space() -> None:
    gbm = {
        "declared_on": "2026-10-11",
        "targets": ["T1"],
        "n_trials": 5,
        "n_jobs": 2,
        "search_space": {"num_leaves": [2, 15]},
    }
    f = make_exp(forecasting={**forecasting_dict(), "gbm": gbm}).forecasting
    assert f is not None and f.gbm is not None and f.gbm.search_space["num_leaves"] == (2.0, 15.0)
    with pytest.raises(ValidationError):
        bad = {**gbm, "search_space": {"num_leaves": [15, 2]}}
        make_exp(forecasting={**forecasting_dict(), "gbm": bad})
