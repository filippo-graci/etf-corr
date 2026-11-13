"""Configuration layer: validated YAML configs, pre-registration lock, paths, seeds, threads.

Responsibilities (project_plan.md section 6.1):

* ``config/experiment.yaml`` and ``config/universe.yaml`` are parsed by a *strict* YAML loader
  (duplicate keys raise, because ``yaml.safe_load`` would silently keep the last one) and
  validated by frozen pydantic models that forbid unknown keys.
* Pre-registration: the confirmatory hypotheses H1-H6 and the ``inference`` block are hashed
  (SHA-256 over a canonical JSON with whitespace-normalised text) into ``config/hypotheses.lock``.
  ``verify_hypotheses_lock`` fails when they differ. The lock is a tripwire, not a proof of
  honesty: the git history is the record, so commit the lock together with the YAML before any
  result exists and tag that commit. Exploratory hypotheses (E1, E2, ...) sit outside the lock.
* ``config_hash`` fingerprints the whole experiment config; it is meant to be embedded in run ids.
* ``Paths`` centralises directories and only allows output under ``data/``, ``artifacts/`` and
  ``reports/``.
* ``set_seed`` seeds ``random``, NumPy and (if installed) PyTorch; ``thread_env`` /
  ``apply_thread_env`` set BLAS/OpenMP thread counts.

Assumptions and limits: environment thread variables only take effect if they are set before
NumPy/BLAS are first imported; ``set_seed`` cannot make every GPU kernel deterministic
(``torch.use_deterministic_algorithms(..., warn_only=True)`` warns instead of failing), so GPU
results are reproducible only up to floating-point non-determinism. No third-party code is
adapted here, so there is nothing to record in THIRD_PARTY.md. Method reference for the
stationary bootstrap named in the ``inference`` block: Politis and Romano (1994), "The Stationary
Bootstrap", JASA 89(428), 1303-1313 (CHECKED, see ``correlation/horizon_analysis.py``).
"""

import hashlib
import importlib
import importlib.util
import json
import os
import random
from collections.abc import MutableMapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, TypeVar

import numpy as np
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

Horizon = Literal["D", "W", "M", "Q"]
Sector = Literal[
    "ai_semis",
    "energy_trad",
    "clean_energy",
    "healthcare",
    "defense",
    "metals_mining",
    "fintech_crypto",
    "control",
]
Panel = Literal["core", "extended", "control"]
Seed = Annotated[int, Field(ge=0, lt=2**32)]

CONFIRMATORY_IDS: tuple[str, ...] = ("H1", "H2", "H3", "H4", "H5", "H6")
LOCK_ALGORITHM = "sha256"
LOCK_COVERS: tuple[str, ...] = ("hypotheses", "inference")
WRITABLE_TOPLEVEL: tuple[str, ...] = ("data", "artifacts", "reports")
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
)


class ConfigError(ValueError):
    """Invalid, inconsistent or tampered configuration."""


# --------------------------------------------------------------------------------------
# YAML loading
# --------------------------------------------------------------------------------------


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of keeping the last one."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ConfigError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML file whose top level must be a mapping; duplicate keys are errors."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    try:
        data = yaml.load(text, Loader=_StrictLoader)  # _StrictLoader is a SafeLoader subclass
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DateWindow(_Frozen):
    """Closed date interval, written ``[start, end]`` in YAML."""

    start: date
    end: date

    @model_validator(mode="before")
    @classmethod
    def _from_pair(cls, data: Any) -> Any:
        if isinstance(data, list | tuple):
            if len(data) != 2:
                raise ValueError("a date window is [start, end]")
            return {"start": data[0], "end": data[1]}
        return data

    @model_validator(mode="after")
    def _ordered(self) -> "DateWindow":
        if self.start >= self.end:
            raise ValueError(f"start {self.start} must be before end {self.end}")
        return self


class WalkForward(_Frozen):
    initial_years: int = Field(ge=1)
    refit_every: Literal["ME", "QE", "YE"]
    embargo_days: int = Field(ge=0)


class Splits(_Frozen):
    train: DateWindow
    val: DateWindow
    test: DateWindow
    walk_forward: WalkForward

    @model_validator(mode="after")
    def _chronological(self) -> "Splits":
        if not self.train.end < self.val.start:
            raise ValueError("train must end before validation starts")
        if not self.val.end < self.test.start:
            raise ValueError("validation must end before test starts")
        return self


class Inference(_Frozen):
    """Inference settings; part of the pre-registration fingerprint."""

    alpha: float = Field(gt=0.0, lt=0.5)
    multiplicity: Literal["holm", "bh", "bonferroni"]
    bootstrap: Literal["stationary", "circular", "moving_block"]
    n_bootstrap: int = Field(ge=100)


class Hypothesis(_Frozen):
    id: str = Field(pattern=r"^[HE][0-9]+$")
    phase: int = Field(ge=1, le=9)
    statement: str = Field(min_length=30)
    test: str = Field(min_length=20)

    @field_validator("statement", "test")
    @classmethod
    def _normalise_whitespace(cls, value: str) -> str:
        # Re-wrapping the YAML must not break the lock; changing a word must.
        return " ".join(value.split())


TargetName = Literal["T1", "T1_matrix", "T2", "T3"]
SimpleBaseline = Literal[
    "persistence", "historical_mean", "ewma", "constant_correlation", "har", "dcc"
]
LinearModel = Literal["ridge", "lasso", "elastic_net", "pcr"]
LossName = Literal["mse_fisher_z", "frobenius", "qlike", "rank_ic"]


VariantKind = Literal[
    "log_bias_correction", "ridge_grid", "exclude_training_window", "dcc_drop_assets"
]


class Variant(_Frozen):
    """One declared deviation from the forecasting design, run on validation data only.

    Ids ``V<n>`` are candidate design changes (adopted only by the rule in ``note``), ids
    ``R<n>`` are robustness runs (never adopted). Every variant counts as a configuration tried.
    """

    id: str = Field(pattern=r"^[VR][0-9]+$")
    declared_on: date
    kind: VariantKind
    targets: tuple[TargetName, ...] = Field(min_length=1)
    note: str = Field(min_length=20)
    ridge_alpha_grid: tuple[float, ...] = ()
    window: DateWindow | None = None
    assets: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _has_its_setting(self) -> "Variant":
        needs = {
            "ridge_grid": bool(self.ridge_alpha_grid) and all(a > 0 for a in self.ridge_alpha_grid),
            "exclude_training_window": self.window is not None,
            "dcc_drop_assets": bool(self.assets),
            "log_bias_correction": True,
        }
        if not needs[self.kind]:
            raise ValueError(f"variant {self.id}: kind {self.kind!r} lacks its setting")
        return self


class Gbm(_Frozen):
    """Rung 3 (Phase 6): LightGBM with a fixed, logged Optuna budget per refit."""

    declared_on: date
    targets: tuple[TargetName, ...] = Field(min_length=1)
    n_trials: int = Field(ge=1)
    n_jobs: int = Field(ge=1)
    search_space: dict[str, tuple[float, float]]

    @model_validator(mode="after")
    def _ranges(self) -> "Gbm":
        for name, (low, high) in self.search_space.items():
            if not 0 < low <= high:
                raise ValueError(f"search_space[{name}]: need 0 < low <= high")
        return self


GraphKind = Literal["knn", "threshold", "mst", "pmfg", "cluster", "sector", "random"]


class Graph(_Frozen):
    """Phase 7: graph-derived features for rung 3, one LightGBM model per graph kind.

    The graph is rebuilt inside every training window (``models.graphs``). Each kind is a
    configuration tried.
    """

    declared_on: date
    targets: tuple[TargetName, ...] = Field(min_length=1)
    kinds: tuple[GraphKind, ...] = Field(min_length=1)
    windows_days: tuple[int, ...] = Field(min_length=1)
    similarity_window_days: int = Field(ge=2)
    knn_k: int = Field(ge=1)
    threshold_quantile: float = Field(gt=0.0, lt=1.0)
    seed: int = 0

    @model_validator(mode="after")
    def _similarity_window_is_aggregated(self) -> "Graph":
        if self.similarity_window_days not in self.windows_days:
            raise ValueError("similarity_window_days must be one of windows_days")
        if len(set(self.kinds)) != len(self.kinds):
            raise ValueError("graph kinds must be unique")
        return self


class Final(_Frozen):
    """Phase 7: what the single test-window run evaluates and how H3 is read from it.

    ``primary_challenger`` and ``frozen_on`` are filled at the freeze, from validation results
    only; ``etfcorr eval --final`` refuses to run while either is missing.
    """

    declared_on: date
    variant: str | None = None
    targets: tuple[TargetName, ...] = Field(min_length=1)
    comparator_primary: SimpleBaseline
    comparator_strict: SimpleBaseline
    primary_challenger: str | None = None
    second_challenger: str | None = None
    frozen_on: date | None = None

    @property
    def frozen(self) -> bool:
        return self.primary_challenger is not None and self.frozen_on is not None


H4Graph = Literal["knn", "sector", "random", "none"]


class H4Design(_Frozen):
    """H4: a temporal graph network against LightGBM with graph features (declared 2026-11-01).

    Declared after the Phase 7 test window had been read for H3, and before any temporal graph
    network was fitted on any window. The network is trained outside this package (it needs
    torch, which this environment does not have): ``scripts/h4_export.py`` writes the inputs,
    ``scripts/h4_tgnn.py`` trains and predicts, ``scripts/h4_score.py`` reads the verdict.
    ``frozen_on`` is set after the validation rehearsal; the test-window export refuses to run
    without it.
    """

    declared_on: date
    target: TargetName
    horizon_days: int = Field(ge=1)
    comparator: str = Field(min_length=1)
    primary_graph: H4Graph
    ablations: tuple[H4Graph, ...] = ()
    lookback_steps: int = Field(ge=1)
    stride_days: int = Field(ge=1)
    hidden: int = Field(ge=1)
    dropout: float = Field(ge=0.0, lt=1.0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    max_epochs: int = Field(ge=1)
    patience: int = Field(ge=1)
    early_stop_share: float = Field(gt=0.0, lt=0.5)
    frozen_on: date | None = None

    @model_validator(mode="after")
    def _graphs(self) -> "H4Design":
        if self.primary_graph in self.ablations or len(set(self.ablations)) != len(self.ablations):
            raise ValueError("the ablation graphs must be unique and differ from the primary graph")
        return self


class Forecasting(_Frozen):
    """Phase 5 forecasting design: targets, baseline ladder, losses, tuning grids.

    Fixed (``fixed_on``) before any forecasting model was fitted. It is outside the
    pre-registration lock (the lock covers H1-H6 and ``inference``), but it is part of
    ``config_hash``, so a later change is visible in every run id.
    """

    fixed_on: date
    primary_target: TargetName
    primary_horizon_days: int = Field(ge=1)
    horizons_days: dict[TargetName, tuple[int, ...]]
    feature_windows_days: tuple[int, ...] = Field(min_length=1)
    simple_baselines: tuple[SimpleBaseline, ...] = Field(min_length=1)
    linear_models: tuple[LinearModel, ...] = ()
    losses: dict[TargetName, LossName]
    inner_folds: int = Field(ge=2)
    alpha_grid: tuple[float, ...] = Field(min_length=1)
    l1_ratio: float = Field(gt=0.0, lt=1.0)
    pcr_components: tuple[int, ...] = Field(min_length=1)
    variants: tuple[Variant, ...] = ()
    gbm: Gbm | None = None
    graph: Graph | None = None
    final: Final | None = None
    h4: H4Design | None = None
    mcs_size: float = Field(default=0.10, gt=0.0, lt=0.5)

    @model_validator(mode="after")
    def _consistent(self) -> "Forecasting":
        ids = [v.id for v in self.variants]
        if len(set(ids)) != len(ids):
            raise ValueError("variant ids must be unique")
        if set(self.horizons_days) != set(self.losses):
            raise ValueError("horizons_days and losses must name the same targets")
        for target, hs in self.horizons_days.items():
            if not hs or any(h < 1 for h in hs) or list(hs) != sorted(set(hs)):
                raise ValueError(f"{target}: horizons must be positive and strictly ascending")
        if self.primary_horizon_days not in self.horizons_days.get(self.primary_target, ()):
            raise ValueError("the primary horizon must be one of the primary target's horizons")
        w = self.feature_windows_days
        if any(x < 2 for x in w) or list(w) != sorted(set(w)):
            raise ValueError("feature windows must be >= 2 and strictly ascending")
        if any(a <= 0 for a in self.alpha_grid) or any(k < 1 for k in self.pcr_components):
            raise ValueError("alpha_grid must be positive and pcr_components >= 1")
        if self.graph is not None and not set(self.graph.windows_days) <= set(w):
            raise ValueError("graph windows must be feature windows")
        if self.final is not None:
            if self.final.variant is not None and self.final.variant not in ids:
                raise ValueError(f"final.variant {self.final.variant!r} is not a declared variant")
            if not set(self.final.targets) <= set(self.horizons_days):
                raise ValueError("final.targets must be targets of the design")
        return self

    def variant(self, variant_id: str) -> Variant:
        """The declared variant with this id."""
        for v in self.variants:
            if v.id == variant_id:
                return v
        raise KeyError(f"no variant {variant_id!r} is declared in the forecasting block")

    @property
    def n_linear_trials(self) -> int:
        """Hyperparameter values tried per refit, summed over the penalised-linear models."""
        per_model = {"pcr": len(self.pcr_components)}
        return sum(per_model.get(m, len(self.alpha_grid)) for m in self.linear_models)


# the last three are exploratory (docs/FOLLOW_UPS.md item 3) and enter no declared design
CovEstimator = Literal[
    "sample",
    "ledoit_wolf",
    "ledoit_wolf_corr",
    "rmt",
    "ewma",
    "dcc",
    "nonlinear",
    "nonlinear_corr",
    "dcc_garch",
]


class H5Variant(_Frozen):
    """A declared candidate replacement of one H5 estimator, decided on validation only."""

    declared_on: date
    candidate: CovEstimator
    replaces: CovEstimator
    note: str = Field(min_length=20)
    adopted: bool | None = None  # filled at the freeze by the rule in ``note``


class Economics(_Frozen):
    """Phase 8: the portfolio (H5) and hedging (H6) designs and how their verdicts are read.

    Declared before any portfolio or hedge was computed. ``frozen_on`` is set at the freeze,
    after the validation rehearsal; ``etfcorr econ --final`` refuses to run without it.
    """

    declared_on: date
    estimation: dict[Horizon, int]  # return frequency -> rows in the estimation window
    estimators: tuple[CovEstimator, ...] = Field(min_length=2)
    h5_estimators: tuple[CovEstimator, ...] = Field(min_length=1)
    hedge_instrument: str
    hedge_window_days: int = Field(ge=20)
    lasso_instruments: tuple[str, ...] = ()
    stress_quantile: float = Field(gt=0.5, lt=1.0)
    calm_quantile: float = Field(gt=0.0, le=0.5)
    block_days: int = Field(ge=1)
    h5_variant: H5Variant | None = None
    frozen_on: date | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "Economics":
        if "sample" not in self.estimators or not set(self.h5_estimators) < set(self.estimators):
            raise ValueError("estimators must hold 'sample' and every h5 estimator")
        if "D" not in self.estimation or any(w < 2 for w in self.estimation.values()):
            raise ValueError("estimation needs the daily frequency and windows >= 2")
        return self


class ExperimentConfig(_Frozen):
    version: Literal[1]
    seeds: tuple[Seed, ...] = Field(min_length=1)
    device: Literal["cuda", "cpu", "auto"]
    threads: int = Field(ge=1, le=256)
    horizons: tuple[Horizon, ...] = Field(min_length=1)
    windows_days: tuple[int, ...] = Field(min_length=1)
    ewma_lambda: float = Field(gt=0.0, lt=1.0)
    splits: Splits
    costs_bps: tuple[float, ...] = Field(min_length=1)
    models: dict[str, Any]
    forecasting: Forecasting | None = None  # Phase 5; None until the design is fixed
    economics: Economics | None = None  # Phase 8
    inference: Inference
    hypotheses: tuple[Hypothesis, ...]
    exploratory: tuple[Hypothesis, ...] = ()

    @field_validator("seeds", "horizons")
    @classmethod
    def _unique(cls, value: tuple[Any, ...]) -> tuple[Any, ...]:
        if len(set(value)) != len(value):
            raise ValueError("entries must be unique")
        return value

    @field_validator("windows_days")
    @classmethod
    def _windows(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(w < 10 for w in value):
            raise ValueError("each window must be at least 10 trading days")
        if list(value) != sorted(set(value)):
            raise ValueError("windows must be strictly ascending")
        return value

    @field_validator("costs_bps")
    @classmethod
    def _costs(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(c < 0 for c in value):
            raise ValueError("costs must be non-negative")
        if 0.0 not in value:
            raise ValueError("the zero-cost case must be included (costs accompany, not replace)")
        return value

    @field_validator("hypotheses")
    @classmethod
    def _confirmatory(cls, value: tuple[Hypothesis, ...]) -> tuple[Hypothesis, ...]:
        ids = tuple(h.id for h in value)
        if ids != CONFIRMATORY_IDS:
            raise ValueError(
                f"confirmatory hypotheses must be exactly {CONFIRMATORY_IDS}, got {ids}"
            )
        return value

    @field_validator("exploratory")
    @classmethod
    def _exploratory(cls, value: tuple[Hypothesis, ...]) -> tuple[Hypothesis, ...]:
        ids = [h.id for h in value]
        if any(not i.startswith("E") for i in ids):
            raise ValueError("exploratory hypothesis ids must look like E1, E2, ...")
        if len(set(ids)) != len(ids):
            raise ValueError("exploratory hypothesis ids must be unique")
        return value


class UniverseEntry(_Frozen):
    ticker: str = Field(pattern=r"^[A-Z0-9^.=-]+$")
    sector: Sector
    panel: Panel
    name: str = Field(min_length=1)
    inception: date | None = None
    futures_based: bool = False
    overlaps_with: tuple[str, ...] = ()
    notes: str = ""


class UniverseConfig(_Frozen):
    version: Literal[1]
    start_date: date
    min_adv_usd: int = Field(gt=0)
    tickers: tuple[UniverseEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> "UniverseConfig":
        names = [t.ticker for t in self.tickers]
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate tickers: {dup}")
        known = set(names)
        for entry in self.tickers:
            if (entry.sector == "control") != (entry.panel == "control"):
                raise ValueError(
                    f"{entry.ticker}: sector 'control' and panel 'control' go together"
                )
            for other in entry.overlaps_with:
                if other == entry.ticker:
                    raise ValueError(f"{entry.ticker}: cannot overlap with itself")
                if other not in known:
                    raise ValueError(f"{entry.ticker}: overlaps_with unknown ticker {other!r}")
        return self

    def tickers_in(self, panel: Panel) -> tuple[str, ...]:
        """Tickers of one panel, in file order."""
        return tuple(t.ticker for t in self.tickers if t.panel == panel)

    def sectors(self) -> tuple[Sector, ...]:
        """Distinct non-control sectors, in order of first appearance."""
        out: list[Sector] = []
        for t in self.tickers:
            if t.sector != "control" and t.sector not in out:
                out.append(t.sector)
        return tuple(out)


# --------------------------------------------------------------------------------------
# Loading, cross-checks
# --------------------------------------------------------------------------------------


_M = TypeVar("_M", bound=BaseModel)


def _validate(model: type[_M], data: dict[str, Any], path: Path) -> _M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def load_experiment(path: Path) -> ExperimentConfig:
    return _validate(ExperimentConfig, load_yaml(path), path)


def load_universe(path: Path) -> UniverseConfig:
    return _validate(UniverseConfig, load_yaml(path), path)


def cross_check(exp: ExperimentConfig, uni: UniverseConfig) -> list[str]:
    """Problems that only appear when the two configs are read together."""
    problems: list[str] = []
    # Estimation windows are in trading days; 7/5 converts to calendar days (holidays ignored).
    warmup_days = -(-max(exp.windows_days) * 7 // 5)
    available = (exp.splits.train.start - uni.start_date).days
    if available < warmup_days:
        problems.append(
            f"universe start_date {uni.start_date} leaves {available} calendar days before "
            f"train start {exp.splits.train.start}; the longest window needs about {warmup_days}"
        )
    return problems


# --------------------------------------------------------------------------------------
# Fingerprints and the pre-registration lock
# --------------------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def preregistration_payload(cfg: ExperimentConfig) -> dict[str, Any]:
    """The part of the config covered by the lock: confirmatory hypotheses and inference."""
    return {
        "hypotheses": [h.model_dump(mode="json") for h in cfg.hypotheses],
        "inference": cfg.inference.model_dump(mode="json"),
    }


def preregistration_fingerprint(cfg: ExperimentConfig) -> str:
    return _sha256(_canonical_json(preregistration_payload(cfg)))


def config_hash(cfg: ExperimentConfig) -> str:
    """Hash of the whole experiment config (all fields), for embedding in run ids."""
    return _sha256(_canonical_json(cfg.model_dump(mode="json")))


def run_id(cfg: ExperimentConfig, name: str) -> str:
    """``<name>-<first 12 hex of config_hash>``."""
    if not name or any(c in name for c in "/\\ "):
        raise ValueError(f"invalid run name {name!r}")
    return f"{name}-{config_hash(cfg)[:12]}"


def write_hypotheses_lock(
    cfg: ExperimentConfig, path: Path, *, locked_on: date, overwrite: bool = False
) -> dict[str, Any]:
    """Write the lock file. Refuses to replace an existing lock unless ``overwrite`` is set."""
    if path.exists() and not overwrite:
        raise ConfigError(
            f"{path} already exists; the pre-registration is not overwritten. "
            "Add an exploratory hypothesis instead, or delete the lock in a visible commit "
            "if no result exists yet."
        )
    lock = {
        "algorithm": LOCK_ALGORITHM,
        "covers": list(LOCK_COVERS),
        "hypothesis_ids": [h.id for h in cfg.hypotheses],
        "fingerprint": preregistration_fingerprint(cfg),
        "locked_on": locked_on.isoformat(),
    }
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def verify_hypotheses_lock(cfg: ExperimentConfig, path: Path) -> None:
    """Raise ``ConfigError`` if the pre-registered part of ``cfg`` differs from the lock."""
    if not path.is_file():
        raise ConfigError(f"{path} is missing; create it once with `etfcorr lock-hypotheses`")
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{path}: unreadable lock file: {exc}") from exc
    if not isinstance(lock, dict):
        raise ConfigError(f"{path}: lock file must be a JSON object")
    if lock.get("algorithm") != LOCK_ALGORITHM or lock.get("covers") != list(LOCK_COVERS):
        raise ConfigError(f"{path}: lock uses an unknown algorithm or coverage")
    ids = [h.id for h in cfg.hypotheses]
    if lock.get("hypothesis_ids") != ids:
        raise ConfigError(
            f"{path}: hypothesis ids {ids} differ from locked {lock.get('hypothesis_ids')}"
        )
    found = preregistration_fingerprint(cfg)
    expected = lock.get("fingerprint")
    if found != expected:
        raise ConfigError(
            "pre-registered hypotheses/inference changed after locking "
            f"(locked {str(expected)[:12]}..., now {found[:12]}...). "
            "Do not edit H1-H6 after results exist: add an exploratory hypothesis (E1, ...)."
        )


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Paths:
    """All project directories, derived from the repository root."""

    root: Path

    @classmethod
    def from_root(cls, root: Path) -> "Paths":
        return cls(root=root.resolve())

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def experiment_yaml(self) -> Path:
        return self.config / "experiment.yaml"

    @property
    def universe_yaml(self) -> Path:
        return self.config / "universe.yaml"

    @property
    def hypotheses_lock(self) -> Path:
        return self.config / "hypotheses.lock"

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def data_external(self) -> Path:
        return self.root / "data" / "external"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def tables(self) -> Path:
        return self.reports / "tables"

    def output_path(self, *parts: str | Path) -> Path:
        """Resolve an output location; refuse anything outside data/, artifacts/, reports/."""
        target = self.root.joinpath(*parts).resolve()
        for top in WRITABLE_TOPLEVEL:
            if target.is_relative_to((self.root / top).resolve()) and target != self.root / top:
                return target
        raise ConfigError(f"refusing to write outside {WRITABLE_TOPLEVEL}: {target}")


def find_repo_root(start: Path | None = None) -> Path:
    """Nearest ancestor of ``start`` (default: cwd) that contains ``config/experiment.yaml``."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "config" / "experiment.yaml").is_file():
            return candidate
    raise ConfigError(f"no config/experiment.yaml found at or above {here}")


@dataclass(frozen=True)
class Project:
    paths: Paths
    experiment: ExperimentConfig
    universe: UniverseConfig


def load_project(root: Path | None = None, *, verify_lock: bool = True) -> Project:
    """Load and cross-check both configs; verify the pre-registration lock by default."""
    paths = Paths.from_root(root if root is not None else find_repo_root())
    exp = load_experiment(paths.experiment_yaml)
    uni = load_universe(paths.universe_yaml)
    problems = cross_check(exp, uni)
    if problems:
        raise ConfigError("; ".join(problems))
    if verify_lock:
        verify_hypotheses_lock(exp, paths.hypotheses_lock)
    return Project(paths=paths, experiment=exp, universe=uni)


def describe_project(project: Project) -> list[str]:
    """Human-readable summary lines for ``etfcorr check-config``."""
    exp, uni = project.experiment, project.universe
    tr, va, te = exp.splits.train, exp.splits.val, exp.splits.test
    return [
        f"config ok  (root: {project.paths.root})",
        f"  horizons {list(exp.horizons)}, windows {list(exp.windows_days)}, "
        f"seeds {list(exp.seeds)}",
        f"  splits train {tr.start}..{tr.end}, val {va.start}..{va.end}, test {te.start}..{te.end}",
        f"  hypotheses {[h.id for h in exp.hypotheses]} locked, {len(exp.exploratory)} exploratory",
        f"  universe {len(uni.tickers)} tickers (core {len(uni.tickers_in('core'))}, "
        f"extended {len(uni.tickers_in('extended'))}, control {len(uni.tickers_in('control'))}), "
        f"{len(uni.sectors())} sectors",
        f"  preregistration {preregistration_fingerprint(exp)[:12]}  "
        f"config {config_hash(exp)[:12]}",
    ]


# --------------------------------------------------------------------------------------
# Reproducibility helpers
# --------------------------------------------------------------------------------------


class DeviceChoice(NamedTuple):
    device: Literal["cuda", "cpu"]
    fell_back: bool


def resolve_device(requested: Literal["cuda", "cpu", "auto"], cuda_usable: bool) -> DeviceChoice:
    """Pure device rule: cuda/auto use the GPU only when the GPU chain check passed."""
    if requested == "cpu":
        return DeviceChoice("cpu", False)
    if cuda_usable:
        return DeviceChoice("cuda", False)
    return DeviceChoice("cpu", requested == "cuda")


def _try_import_torch() -> Any | None:
    if importlib.util.find_spec("torch") is None:
        return None
    return importlib.import_module("torch")


def set_seed(
    seed: int, *, torch_module: Any | None = None, use_torch: bool = True
) -> np.random.Generator:
    """Seed ``random``, NumPy's legacy global state and (if available) PyTorch.

    Returns a fresh ``np.random.Generator``; new code should draw from it rather than from the
    global state. ``torch_module`` exists for testing.
    """
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must be in [0, 2**32), got {seed}")
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - third-party libraries still read the global state
    if use_torch:
        torch = torch_module if torch_module is not None else _try_import_torch()
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
    return np.random.default_rng(seed)


def thread_env(n: int) -> dict[str, str]:
    """Environment variables that cap BLAS/OpenMP/Polars threads at ``n``."""
    if n < 1:
        raise ValueError(f"thread count must be >= 1, got {n}")
    return dict.fromkeys(THREAD_ENV_VARS, str(n))


def apply_thread_env(
    n: int, environ: MutableMapping[str, str] | None = None, *, override: bool = False
) -> dict[str, str]:
    """Set the thread variables in ``environ`` (default ``os.environ``); return what was set.

    Existing values are kept unless ``override``. Effective only if called before NumPy/BLAS
    are first imported (entry-point scripts call it first thing).
    """
    target = os.environ if environ is None else environ
    applied: dict[str, str] = {}
    for key, value in thread_env(n).items():
        if override or key not in target:
            target[key] = value
            applied[key] = value
    return applied
