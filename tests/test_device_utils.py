"""Tests for the device selection helpers."""

from __future__ import annotations

import sys
import types

import pytest

from src.utils import device as device_utils


def _install_torch_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda_available: bool,
    mps_available: bool = False,
    total_memory_gb: int = 8,
) -> None:
    """Register a lightweight torch stub tailored to each scenario."""
    module = types.ModuleType("torch")

    def cuda_is_available() -> bool:
        return cuda_available

    module.cuda = types.SimpleNamespace(
        is_available=cuda_is_available,
        get_device_name=lambda idx: "Stub GPU",
        get_device_properties=lambda idx: types.SimpleNamespace(
            total_memory=total_memory_gb * 1024**3,
        ),
    )
    module.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps_available),
    )
    module.version = types.SimpleNamespace(cuda="12.1")
    module.__version__ = "2.4.0"
    monkeypatch.setitem(sys.modules, "torch", module)


def test_resolve_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_torch_stub(monkeypatch, cuda_available=True, mps_available=True)
    assert device_utils.resolve_device("auto") == "cuda"


def test_resolve_device_auto_falls_back_to_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_torch_stub(monkeypatch, cuda_available=False, mps_available=True)
    assert device_utils.resolve_device("auto") == "mps"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_torch_stub(monkeypatch, cuda_available=False, mps_available=False)
    assert device_utils.resolve_device("auto") == "cpu"


def test_resolve_device_cuda_requires_available_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_torch_stub(monkeypatch, cuda_available=False)
    with pytest.raises(ValueError):
        device_utils.resolve_device("cuda")


def test_resolve_device_mps_requires_supported_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_torch_stub(monkeypatch, cuda_available=False, mps_available=False)
    with pytest.raises(ValueError):
        device_utils.resolve_device("mps")


def test_get_optimal_batch_size_scales_with_gpu_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_torch_stub(monkeypatch, cuda_available=True, total_memory_gb=24)
    assert device_utils.get_optimal_batch_size("cuda", default=200) == 128


def test_get_optimal_batch_size_cpu_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_torch_stub(monkeypatch, cuda_available=False)
    assert device_utils.get_optimal_batch_size("cpu", default=64) == 32


