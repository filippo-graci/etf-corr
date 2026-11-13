"""Environment gate (Phase 0, step 0.2): versions, imports, GPU / CUDA architecture, threads.

Purpose: answer, before any experiment runs, (a) is the Python environment the one the project
expects, (b) can the RTX 5060 (Blackwell, compute capability 12.0, ``sm_120``) really be used by
PyTorch, or must the deep-learning rungs fall back to CPU (``device: cpu`` in the config)?

The script only *reports*: it never installs anything, never changes environment variables
persistently and never writes files. Threads, seeds and determinism are configured centrally in
``config.py`` (step 0.3).

Design: every decision is a pure function of injected inputs (``nvidia-smi`` text, a module-like
``torch`` object, an environment mapping), so the logic is tested offline; only the last mile
(``nvidia-smi`` subprocess, the real ``torch`` import, a real GPU matmul) needs the real machine.

CUDA rule used in :func:`sm_support` [Inference, from the CUDA compatibility model, not verified in
this session]: a PyTorch build runs natively on a GPU of capability ``X.Y`` if its architecture list
contains ``sm_XY``; otherwise it can still run through JIT compilation of embedded PTX if the list
contains ``compute_N`` with ``N <= XY``; SASS for another architecture (for example ``sm_100``) is
not enough. The matmul self-test is the final arbiter.

No third-party code is adapted here (see THIRD_PARTY.md).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

Status = Literal["OK", "WARN", "FAIL", "SKIP"]
Group = Literal["core", "threads", "gpu"]

# Must match ``requires-python`` in pyproject.toml (checked by a unit test).
PYTHON_MIN = (3, 11)
PYTHON_MAX_EXCLUSIVE = (3, 14)
RECOMMENDED_THREADS = 8  # project_plan.md section 2.4

# (distribution name, import name)
CORE_PACKAGES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("statsmodels", "statsmodels"),
    ("pyarrow", "pyarrow"),
    ("pydantic", "pydantic"),
    ("pyyaml", "yaml"),
    ("typer", "typer"),
    ("networkx", "networkx"),
    ("yfinance", "yfinance"),
    ("fredapi", "fredapi"),
    ("pandas-market-calendars", "pandas_market_calendars"),
    ("skfolio", "skfolio"),
    ("arch", "arch"),
    ("pymgarch", "pymgarch"),
    ("hmmlearn", "hmmlearn"),
    ("dcor", "dcor"),
    ("purgedcv", "purgedcv"),
    ("lightgbm", "lightgbm"),
    ("shap", "shap"),
    ("optuna", "optuna"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("plotly", "plotly"),
    ("etfcorr", "etfcorr"),
)
# Optional extras: missing means SKIP, installed-but-broken means WARN.
OPTIONAL_PACKAGES: tuple[tuple[str, str], ...] = (
    ("torch-geometric", "torch_geometric"),
    ("torch-geometric-temporal", "torch_geometric_temporal"),
    ("riskfolio-lib", "riskfolio"),
    ("pyportfolioopt", "pypfopt"),
    ("streamlit", "streamlit"),
)

_LEVEL_STATUS: dict[str, Status] = {"native": "OK", "ptx": "WARN", "none": "FAIL"}
_INCOMPATIBLE_WARNING = re.compile(r"not compatible|no kernel image|sm_\d+.*not", re.IGNORECASE)


@dataclass(frozen=True)
class Check:
    """One line of the report."""

    group: Group
    name: str
    status: Status
    detail: str


@dataclass(frozen=True)
class GpuInfo:
    """One row of ``nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap``."""

    name: str
    driver: str
    memory_mib: int | None
    capability: tuple[int, int] | None


# ---------------------------------------------------------------------------- python
def check_python(version_info: Sequence[int] | None = None) -> Check:
    """Python must lie in ``[PYTHON_MIN, PYTHON_MAX_EXCLUSIVE)``."""
    vi = tuple(version_info) if version_info is not None else tuple(sys.version_info[:3])
    text = ".".join(str(x) for x in vi)
    ok = PYTHON_MIN <= vi[:2] < PYTHON_MAX_EXCLUSIVE
    lo = ".".join(map(str, PYTHON_MIN))
    hi = ".".join(map(str, PYTHON_MAX_EXCLUSIVE))
    rng = f"{lo} <= python < {hi}"
    return Check("core", "python", "OK" if ok else "FAIL", f"{text} (supported: {rng})")


# ---------------------------------------------------------------------------- packages
def check_packages(
    packages: Sequence[tuple[str, str]],
    *,
    optional: bool = False,
    do_import: bool = True,
    importer: Callable[[str], Any] = importlib.import_module,
    version_of: Callable[[str], str] = md.version,
) -> list[Check]:
    """Report version and importability of each ``(distribution, import name)`` pair."""
    out: list[Check] = []
    for dist, module in packages:
        try:
            ver = version_of(dist)
        except md.PackageNotFoundError:
            status: Status = "SKIP" if optional else "FAIL"
            hint = " (optional extra not installed)" if optional else " (not installed)"
            out.append(Check("core", dist, status, f"missing{hint}"))
            continue
        if not do_import:
            out.append(Check("core", dist, "OK", f"{ver} (import not tested)"))
            continue
        t0 = time.perf_counter()
        try:
            importer(module)
        except Exception as exc:  # any import-time failure is a finding, not a crash
            status = "WARN" if optional else "FAIL"
            out.append(Check("core", dist, status, f"{ver}, import failed: {exc!r}"))
            continue
        out.append(Check("core", dist, "OK", f"{ver} (import {time.perf_counter() - t0:.2f} s)"))
    return out


# ---------------------------------------------------------------------------- threads
def check_threads(
    environ: Mapping[str, str], cpu_count: int | None, torch_threads: int | None = None
) -> list[Check]:
    """Report CPU count and thread-related variables; never changes anything."""
    cpus = cpu_count or 1
    out = [Check("threads", "cpu_count", "OK", str(cpus))]
    wanted = min(RECOMMENDED_THREADS, cpus)
    raw = environ.get("OMP_NUM_THREADS")
    if raw is None:
        out.append(
            Check(
                "threads",
                "OMP_NUM_THREADS",
                "WARN",
                f"unset; recommended `export OMP_NUM_THREADS={wanted}` for long runs",
            )
        )
    else:
        try:
            n = int(raw)
        except ValueError:
            out.append(Check("threads", "OMP_NUM_THREADS", "WARN", f"not an integer: {raw!r}"))
        else:
            status: Status = "OK" if 1 <= n <= cpus else "WARN"
            out.append(Check("threads", "OMP_NUM_THREADS", status, f"{n} (cpu_count {cpus})"))
    if torch_threads is not None:
        out.append(Check("threads", "torch_threads", "OK", str(torch_threads)))
    return out


# ---------------------------------------------------------------------------- nvidia-smi
def parse_nvidia_smi_csv(text: str) -> list[GpuInfo]:
    """Parse the CSV printed by ``nvidia-smi --query-gpu=name,driver_version,memory.total,
    compute_cap --format=csv,noheader,nounits``.

    Rows with fewer than four fields are ignored; ``[N/A]`` and other non-numeric values
    become None.
    """
    gpus: list[GpuInfo] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[0]:
            continue
        name, driver, mem, cap = parts[0], parts[1], parts[2], parts[3]
        memory = int(mem) if mem.isdigit() else None
        m = re.fullmatch(r"(\d+)\.(\d+)", cap)
        capability = (int(m.group(1)), int(m.group(2))) if m else None
        gpus.append(GpuInfo(name, driver, memory, capability))
    return gpus


def query_nvidia_smi(
    run: Callable[..., Any] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> tuple[bool, str]:
    """Return ``(found, stdout_or_error)``; ``found`` is False when ``nvidia-smi`` is absent."""
    if which("nvidia-smi") is None:
        return False, "nvidia-smi not found on PATH (NVIDIA driver not installed?)"
    try:
        res = run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return True, f"nvidia-smi failed to run: {exc!r}"
    if res.returncode != 0:
        return True, f"nvidia-smi exit code {res.returncode}: {res.stderr.strip()[:200]}"
    return True, str(res.stdout)


def sm_support(
    arch_list: Sequence[str], capability: tuple[int, int]
) -> Literal["native", "ptx", "none"]:
    """How a PyTorch build with ``arch_list`` can serve a GPU of ``capability`` (see top)."""
    code = capability[0] * 10 + capability[1]
    sms = {int(m.group(2)) for a in arch_list if (m := re.fullmatch(r"(sm)_(\d+)a?", a))}
    ptx = {int(m.group(2)) for a in arch_list if (m := re.fullmatch(r"(compute)_(\d+)a?", a))}
    if code in sms:
        return "native"
    if any(n <= code for n in ptx):
        return "ptx"
    return "none"


# ---------------------------------------------------------------------------- torch
def gpu_matmul_selftest(
    torch: Any, n: int = 256, atol: float = 1e-3, rtol: float = 1e-3
) -> tuple[bool, str]:
    """Multiply two random matrices on the GPU and compare with the CPU result."""
    try:
        torch.manual_seed(0)
        a = torch.randn(n, n)
        b = torch.randn(n, n)
        ref = torch.matmul(a, b)
        out = torch.matmul(a.to("cuda"), b.to("cuda"))
        torch.cuda.synchronize()
        ok = bool(torch.allclose(out.cpu(), ref, atol=atol, rtol=rtol))
    except Exception as exc:  # e.g. "no kernel image is available for execution on the device"
        return False, f"matmul raised {exc!r}"
    return ok, f"{n}x{n} float32 GPU result {'matches' if ok else 'DIFFERS FROM'} CPU (atol={atol})"


def check_torch(torch: Any, smi_capability: tuple[int, int] | None) -> list[Check]:
    """CUDA availability, architecture list, matmul self-test and stray compatibility warnings."""
    out: list[Check] = []
    cuda_build = getattr(getattr(torch, "version", None), "cuda", None)
    out.append(Check("gpu", "torch", "OK", f"{torch.__version__} (built for CUDA {cuda_build})"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            available = bool(torch.cuda.is_available())
        except Exception as exc:
            available = False
            out.append(Check("gpu", "cuda_available", "FAIL", f"is_available raised {exc!r}"))
        else:
            out.append(
                Check(
                    "gpu",
                    "cuda_available",
                    "OK" if available else "FAIL",
                    "True" if available else "False (driver/CUDA runtime not usable by this build)",
                )
            )
        if available:
            capability = smi_capability
            try:
                cc = torch.cuda.get_device_capability(0)
                capability = (int(cc[0]), int(cc[1]))
                name = torch.cuda.get_device_name(0)
                out.append(Check("gpu", "device", "OK", f"{name}, capability {cc[0]}.{cc[1]}"))
            except Exception as exc:
                out.append(
                    Check("gpu", "device", "WARN", f"could not read device properties: {exc!r}")
                )
            arch_list = list(torch.cuda.get_arch_list())
            if capability is None:
                out.append(Check("gpu", "arch_list", "WARN", f"{arch_list}; capability unknown"))
            else:
                level = sm_support(arch_list, capability)
                status = _LEVEL_STATUS[level]
                out.append(Check("gpu", "arch_list", status, f"{level}: build has {arch_list}"))
            ok, detail = gpu_matmul_selftest(torch)
            out.append(Check("gpu", "matmul", "OK" if ok else "FAIL", detail))
    bad = [str(w.message)[:160] for w in caught if _INCOMPATIBLE_WARNING.search(str(w.message))]
    if bad:
        out.append(Check("gpu", "warnings", "FAIL", "; ".join(bad)))
    return out


def check_gpu_chain(
    smi_found: bool,
    smi_text: str,
    torch_loader: Callable[[], Any],
) -> list[Check]:
    """nvidia-smi -> torch import -> CUDA -> architecture -> matmul; each failure is a gpu check."""
    out: list[Check] = []
    capability: tuple[int, int] | None = None
    if not smi_found:
        out.append(Check("gpu", "nvidia-smi", "FAIL", smi_text))
    else:
        gpus = parse_nvidia_smi_csv(smi_text)
        if not gpus:
            out.append(Check("gpu", "nvidia-smi", "FAIL", f"no GPU reported: {smi_text[:200]!r}"))
        else:
            g = gpus[0]
            cap = f"{g.capability[0]}.{g.capability[1]}" if g.capability else "unknown"
            mem = f"{g.memory_mib} MiB" if g.memory_mib is not None else "unknown memory"
            out.append(
                Check(
                    "gpu",
                    "nvidia-smi",
                    "OK",
                    f"{g.name}, driver {g.driver}, {mem}, capability {cap}",
                )
            )
            capability = g.capability
            if capability is not None and capability != (12, 0):
                out.append(
                    Check(
                        "gpu",
                        "capability",
                        "WARN",
                        f"expected 12.0 (sm_120) for the RTX 5060, got {cap}",
                    )
                )
    try:
        torch = torch_loader()
    except Exception as exc:
        out.append(
            Check(
                "gpu",
                "torch",
                "FAIL",
                f"cannot import torch ({exc!r}); install the deep-learning extra "
                "(`uv sync --all-packages --extra dev --extra dl` from the workspace root)",
            )
        )
        return out
    out.extend(check_torch(torch, capability))
    return out


# ---------------------------------------------------------------------------- decision
def gpu_usable(checks: Sequence[Check]) -> bool:
    """CUDA is usable only if the gpu group has checks, none failed, and the matmul passed."""
    gpu = [c for c in checks if c.group == "gpu"]
    matmul = [c for c in gpu if c.name == "matmul"]
    return (
        bool(gpu)
        and all(c.status != "FAIL" for c in gpu)
        and len(matmul) == 1
        and matmul[0].status == "OK"
    )


def exit_code(checks: Sequence[Check], cpu_ok: bool) -> int:
    """0 = all good (or CPU fallback accepted), 1 = core environment broken, 2 = GPU unusable."""
    if any(c.group == "core" and c.status == "FAIL" for c in checks):
        return 1
    if not gpu_usable(checks) and not cpu_ok:
        return 2
    return 0


def render(checks: Sequence[Check], device: str, code: int) -> str:
    """Plain-text report."""
    lines = []
    for group in ("core", "threads", "gpu"):
        rows = [c for c in checks if c.group == group]
        if rows:
            lines.append(f"== {group}")
            lines.extend(f"[{c.status:<4}] {c.name}: {c.detail}" for c in rows)
    lines.append(f"DECISION: device={device}  exit_code={code}")
    return "\n".join(lines)


def run_all(
    *,
    do_import: bool = True,
    environ: Mapping[str, str] | None = None,
    cpu_count: int | None = None,
    smi: Callable[[], tuple[bool, str]] = query_nvidia_smi,
    torch_loader: Callable[[], Any] = lambda: importlib.import_module("torch"),
) -> list[Check]:
    """Run every check with injectable collaborators (real ones by default)."""
    env = os.environ if environ is None else environ
    cpus = os.cpu_count() if cpu_count is None else cpu_count
    checks = [check_python()]
    checks += check_packages(CORE_PACKAGES, do_import=do_import)
    checks += check_packages(OPTIONAL_PACKAGES, optional=True, do_import=do_import)
    found, text = smi()
    checks += check_gpu_chain(found, text, torch_loader)
    torch_threads: int | None = None
    try:
        torch_threads = int(importlib.import_module("torch").get_num_threads())
    except Exception:
        torch_threads = None
    checks += check_threads(env, cpus, torch_threads)
    return checks


def main(
    argv: Sequence[str] | None = None,
    *,
    checks_fn: Callable[..., list[Check]] = run_all,
) -> int:
    """Entry point; returns the process exit code (0 ok, 1 core broken, 2 GPU unusable)."""
    p = argparse.ArgumentParser(description="etf-corr environment gate (Phase 0)")
    p.add_argument(
        "--cpu-ok", action="store_true", help="accept a CPU fallback (exit 0 without GPU)"
    )
    p.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p.add_argument("--no-import", action="store_true", help="check versions only, do not import")
    args = p.parse_args(argv)
    checks = checks_fn(do_import=not args.no_import)
    device = "cuda" if gpu_usable(checks) else "cpu"
    code = exit_code(checks, args.cpu_ok)
    if args.json:
        print(
            json.dumps(
                {"device": device, "exit_code": code, "checks": [asdict(c) for c in checks]},
                indent=2,
            )
        )
    else:
        print(render(checks, device, code))
    return code
