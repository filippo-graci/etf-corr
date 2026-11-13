"""Tests for etfcorr.envcheck with injected fakes: no GPU, no torch, no network needed."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from etfcorr import envcheck as ec

ROOT = Path(__file__).resolve().parents[2]

SMI_OK = "NVIDIA GeForce RTX 5060 Laptop GPU, 595.45.04, 8151, 12.0\n"


# ------------------------------------------------------------------ fake torch (numpy backed)
class FakeTensor:
    def __init__(self, a: np.ndarray) -> None:
        self.a = a

    def to(self, device: str) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self


def make_torch(
    *,
    available: bool = True,
    arch: tuple[str, ...] = ("sm_86", "sm_120", "compute_120"),
    capability: tuple[int, int] = (12, 0),
    matmul_error: float = 0.0,
    warn: str | None = None,
    raises: str | None = None,
) -> Any:
    """A module-like object with the small torch surface that envcheck uses."""
    rng = np.random.default_rng(0)

    def randn(n: int, m: int) -> FakeTensor:
        return FakeTensor(rng.standard_normal((n, m), dtype=np.float32))

    calls = {"n": 0}

    def matmul(x: FakeTensor, y: FakeTensor) -> FakeTensor:
        calls["n"] += 1
        if raises and calls["n"] == 2:  # second call is the "GPU" one
            raise RuntimeError(raises)
        out = x.a @ y.a
        if matmul_error and calls["n"] == 2:
            out = out + matmul_error
        return FakeTensor(out)

    def is_available() -> bool:
        if warn:
            warnings.warn(warn, UserWarning, stacklevel=2)
        return available

    cuda = SimpleNamespace(
        is_available=is_available,
        get_arch_list=lambda: list(arch),
        get_device_capability=lambda i=0: capability,
        get_device_name=lambda i=0: "Fake RTX 5060",
        synchronize=lambda: None,
    )
    return SimpleNamespace(
        __version__="9.9.9",
        version=SimpleNamespace(cuda="12.8"),
        cuda=cuda,
        manual_seed=lambda s: None,
        randn=randn,
        matmul=matmul,
        allclose=lambda x, y, atol, rtol: bool(np.allclose(x.a, y.a, atol=atol, rtol=rtol)),
        get_num_threads=lambda: 8,
    )


def gpu_checks(**kw: Any) -> list[ec.Check]:
    return ec.check_gpu_chain(True, SMI_OK, lambda: make_torch(**kw))


def by_name(checks: list[ec.Check], name: str) -> ec.Check:
    return next(c for c in checks if c.name == name)


# ------------------------------------------------------------------ python
def test_python_range() -> None:
    assert ec.check_python((3, 12, 8)).status == "OK"
    assert ec.check_python((3, 11, 0)).status == "OK"
    assert ec.check_python((3, 10, 14)).status == "FAIL"
    assert ec.check_python((3, 14, 4)).status == "FAIL"


def test_python_range_matches_pyproject() -> None:
    spec = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
    lo, hi = spec.replace(" ", "").split(",")
    assert lo == f">={ec.PYTHON_MIN[0]}.{ec.PYTHON_MIN[1]}"
    assert hi == f"<{ec.PYTHON_MAX_EXCLUSIVE[0]}.{ec.PYTHON_MAX_EXCLUSIVE[1]}"


def test_core_packages_are_declared_in_pyproject() -> None:
    deps = " ".join(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"])
    for dist, _ in ec.CORE_PACKAGES:
        assert dist == "etfcorr" or dist in deps, dist


# ------------------------------------------------------------------ packages
def test_packages_ok_missing_and_broken() -> None:
    versions = {"a": "1.0", "c": "3.0", "d": "4.0"}

    def version_of(name: str) -> str:
        if name in versions:
            return versions[name]
        raise ec.md.PackageNotFoundError(name)

    def importer(name: str) -> None:
        if name == "c_mod":
            raise ImportError("boom")

    pk = [("a", "a_mod"), ("b", "b_mod"), ("c", "c_mod")]
    core = ec.check_packages(pk, importer=importer, version_of=version_of)
    assert [c.status for c in core] == ["OK", "FAIL", "FAIL"]
    opt = ec.check_packages(pk, optional=True, importer=importer, version_of=version_of)
    assert [c.status for c in opt] == ["OK", "SKIP", "WARN"]
    noimp = ec.check_packages(pk[:1], do_import=False, importer=importer, version_of=version_of)
    assert noimp[0].status == "OK" and "not tested" in noimp[0].detail


# ------------------------------------------------------------------ threads
def test_threads_variants() -> None:
    assert by_name(ec.check_threads({}, 16), "OMP_NUM_THREADS").status == "WARN"
    assert "8" in by_name(ec.check_threads({}, 16), "OMP_NUM_THREADS").detail
    assert "4" in by_name(ec.check_threads({}, 4), "OMP_NUM_THREADS").detail
    assert by_name(ec.check_threads({"OMP_NUM_THREADS": "8"}, 16), "OMP_NUM_THREADS").status == "OK"
    assert (
        by_name(ec.check_threads({"OMP_NUM_THREADS": "64"}, 16), "OMP_NUM_THREADS").status == "WARN"
    )
    assert (
        by_name(ec.check_threads({"OMP_NUM_THREADS": "0"}, 16), "OMP_NUM_THREADS").status == "WARN"
    )
    assert (
        by_name(ec.check_threads({"OMP_NUM_THREADS": "x"}, 16), "OMP_NUM_THREADS").status == "WARN"
    )
    assert by_name(ec.check_threads({}, None), "cpu_count").detail == "1"
    assert by_name(ec.check_threads({}, 16, torch_threads=8), "torch_threads").detail == "8"


# ------------------------------------------------------------------ nvidia-smi
def test_parse_nvidia_smi() -> None:
    (g,) = ec.parse_nvidia_smi_csv(SMI_OK)
    assert g == ec.GpuInfo("NVIDIA GeForce RTX 5060 Laptop GPU", "595.45.04", 8151, (12, 0))
    two = ec.parse_nvidia_smi_csv(SMI_OK + "Other GPU, 595.45.04, [N/A], [N/A]\n")
    assert len(two) == 2 and two[1].memory_mib is None and two[1].capability is None


@pytest.mark.parametrize("text", ["", "\n\n", "garbage line", "a, b, c", ", , , "])
def test_parse_nvidia_smi_garbage(text: str) -> None:
    assert ec.parse_nvidia_smi_csv(text) == []


def test_query_nvidia_smi_paths() -> None:
    assert ec.query_nvidia_smi(which=lambda _: None)[0] is False

    def ok(*a: Any, **k: Any) -> Any:
        return SimpleNamespace(returncode=0, stdout=SMI_OK, stderr="")

    def bad(*a: Any, **k: Any) -> Any:
        return SimpleNamespace(returncode=9, stdout="", stderr="driver mismatch")

    def crash(*a: Any, **k: Any) -> Any:
        raise subprocess.TimeoutExpired("nvidia-smi", 20)

    assert ec.query_nvidia_smi(run=ok, which=lambda _: "/x") == (True, SMI_OK)
    assert "driver mismatch" in ec.query_nvidia_smi(run=bad, which=lambda _: "/x")[1]
    assert "failed to run" in ec.query_nvidia_smi(run=crash, which=lambda _: "/x")[1]


# ------------------------------------------------------------------ architecture rule
@pytest.mark.parametrize(
    ("arch", "cc", "expected"),
    [
        (["sm_75", "sm_86", "sm_120", "compute_120"], (12, 0), "native"),
        (["sm_120a"], (12, 0), "native"),
        (["sm_90", "compute_90"], (12, 0), "ptx"),
        (["sm_90", "sm_100"], (12, 0), "none"),  # SASS of another architecture is not enough
        (["compute_130"], (12, 0), "none"),  # PTX newer than the device cannot be used
        ([], (12, 0), "none"),
        (["sm_86", "compute_86"], (8, 6), "native"),
    ],
)
def test_sm_support(arch: list[str], cc: tuple[int, int], expected: str) -> None:
    assert ec.sm_support(arch, cc) == expected


# ------------------------------------------------------------------ GPU chain
def test_gpu_happy_path() -> None:
    checks = gpu_checks()
    assert all(c.status == "OK" for c in checks), checks
    assert ec.gpu_usable(checks)
    assert ec.exit_code(checks, cpu_ok=False) == 0


def test_no_driver_is_gpu_failure() -> None:
    checks = ec.check_gpu_chain(False, "nvidia-smi not found", lambda: make_torch())
    assert by_name(checks, "nvidia-smi").status == "FAIL"
    assert not ec.gpu_usable(checks) or by_name(checks, "matmul").status == "OK"
    # torch alone must not rescue a missing driver: the chain has a FAIL entry
    assert not ec.gpu_usable(checks)


def test_torch_missing_points_to_dl_extra() -> None:
    def loader() -> Any:
        raise ModuleNotFoundError("No module named 'torch'")

    checks = ec.check_gpu_chain(True, SMI_OK, loader)
    c = by_name(checks, "torch")
    assert c.status == "FAIL" and "--extra dl" in c.detail
    assert not ec.gpu_usable(checks)


def test_cuda_unavailable() -> None:
    checks = gpu_checks(available=False)
    assert by_name(checks, "cuda_available").status == "FAIL"
    assert not ec.gpu_usable(checks)


def test_arch_missing_sm120_fails() -> None:
    checks = gpu_checks(arch=("sm_86", "sm_90", "sm_100"))
    assert by_name(checks, "arch_list").status == "FAIL"
    assert not ec.gpu_usable(checks)


def test_ptx_only_is_warning_but_usable_if_matmul_passes() -> None:
    checks = gpu_checks(arch=("sm_90", "compute_90"))
    assert by_name(checks, "arch_list").status == "WARN"
    assert ec.gpu_usable(checks)


def test_matmul_mismatch_and_exception_fail() -> None:
    assert not ec.gpu_usable(gpu_checks(matmul_error=1.0))
    failing = gpu_checks(raises="no kernel image is available for execution on the device")
    assert by_name(failing, "matmul").status == "FAIL"
    assert "no kernel image" in by_name(failing, "matmul").detail


def test_incompatibility_warning_fails_even_if_matmul_passes() -> None:
    msg = (
        "NVIDIA GeForce RTX 5060 with CUDA capability sm_120 is not compatible "
        "with the current PyTorch installation."
    )
    checks = gpu_checks(warn=msg)
    assert by_name(checks, "matmul").status == "OK"
    assert by_name(checks, "warnings").status == "FAIL"
    assert not ec.gpu_usable(checks)


def test_unrelated_warning_is_ignored() -> None:
    checks = gpu_checks(warn="something about deprecation")
    assert not any(c.name == "warnings" for c in checks)
    assert ec.gpu_usable(checks)


def test_unexpected_capability_warns() -> None:
    checks = ec.check_gpu_chain(
        True, "Some GPU, 550.0, 8000, 8.6", lambda: make_torch(capability=(8, 6))
    )
    assert by_name(checks, "capability").status == "WARN"


def test_empty_gpu_group_is_not_usable() -> None:
    assert not ec.gpu_usable([])


def test_gpu_group_without_matmul_is_not_usable() -> None:
    # e.g. a future refactor that forgets to run the self-test must not report "cuda"
    assert not ec.gpu_usable([ec.Check("gpu", "torch", "OK", "9.9.9")])


# ------------------------------------------------------------------ exit codes and rendering
def _core_ok() -> list[ec.Check]:
    return [ec.Check("core", "python", "OK", "3.12")]


def test_exit_codes() -> None:
    good = _core_ok() + gpu_checks()
    no_gpu = _core_ok() + gpu_checks(available=False)
    broken = [ec.Check("core", "numpy", "FAIL", "missing"), *gpu_checks()]
    assert ec.exit_code(good, False) == 0
    assert ec.exit_code(no_gpu, False) == 2
    assert ec.exit_code(no_gpu, True) == 0  # CPU fallback accepted
    assert ec.exit_code(broken, True) == 1  # a broken core beats --cpu-ok
    assert ec.exit_code(broken, False) == 1


def test_threads_never_affect_exit_code() -> None:
    checks = _core_ok() + gpu_checks() + [ec.Check("threads", "OMP_NUM_THREADS", "FAIL", "x")]
    assert ec.exit_code(checks, False) == 0


def test_main_text_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    def fake_checks(**kw: Any) -> list[ec.Check]:
        return _core_ok() + gpu_checks(available=False)

    assert ec.main([], checks_fn=fake_checks) == 2
    out = capsys.readouterr().out
    assert "DECISION: device=cpu  exit_code=2" in out and "[FAIL]" in out
    assert ec.main(["--cpu-ok", "--json"], checks_fn=fake_checks) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["device"] == "cpu" and payload["exit_code"] == 0
    assert {"group", "name", "status", "detail"} <= set(payload["checks"][0])


def test_run_all_with_injected_collaborators() -> None:
    checks = ec.run_all(
        do_import=False,
        environ={"OMP_NUM_THREADS": "8"},
        cpu_count=16,
        smi=lambda: (False, "nvidia-smi not found"),
        torch_loader=lambda: make_torch(),
    )
    groups = {c.group for c in checks}
    assert groups == {"core", "threads", "gpu"}
    assert by_name(checks, "python").status == "OK"


def test_module_import_has_no_side_effects(tmp_path: Path) -> None:
    code = (
        "import socket\n"
        "def _b(*a, **k):\n    raise RuntimeError('network at import')\n"
        "socket.socket.connect = _b\n"
        "import etfcorr.envcheck\n"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert list(tmp_path.iterdir()) == []
