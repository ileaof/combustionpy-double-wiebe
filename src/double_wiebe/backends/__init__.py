# -*- coding: utf-8 -*-
"""
backends — Camada de execução do plano HPC.

Uso:
    from double_wiebe.backends import select_backend, get_available_backends
    backend = select_backend("auto")          # mini-benchmark
    backend = select_backend("cpu-parallel")  # explícito

Nomes: "serial" | "cpu" | "cpu-parallel" | "cuda" | "auto".
cuda: RK4 em lote na GPU (CuPy, extra [cuda]) — veja docs/hpc.md.
OpenCL: apenas detecção de hardware (detection.py).
"""
from __future__ import annotations

from typing import Optional

from .base import BackendNotAvailableError, ComputeBackend, BackendError
from .serial_backend import (CPUBackend, MultiprocessingBackend,
                             SerialBackend)
from .cuda_backend import CUDABackend
from . import detection

__all__ = ["ComputeBackend", "BackendError", "BackendNotAvailableError",
           "SerialBackend", "CPUBackend", "MultiprocessingBackend", "CUDABackend",
           "get_available_backends", "get_all_backends", "select_backend",
           "detect_hardware", "hardware_report"]

detect_hardware = detection.detect_hardware
hardware_report = detection.hardware_report

_BACKEND_CLASSES = {
    "serial": SerialBackend,
    "cpu": CPUBackend,
    "cpu-parallel": MultiprocessingBackend,
    "cuda": CUDABackend,
}


def get_all_backends() -> dict:
    """Nome -> classe (todos os backends conhecidos)."""
    return dict(_BACKEND_CLASSES)


def get_available_backends() -> dict:
    """Somente os backends utilizáveis nesta máquina: nome -> instância."""
    out = {}
    for nome, cls in _BACKEND_CLASSES.items():
        try:
            b = cls()
            if b.is_available():
                out[nome] = b
        except Exception:                       # pragma: no cover
            pass
    return out


def select_backend(nome: str = "serial", workers: Optional[int] = None,
                   integrator: str = "auto",
                   fallback: bool = True) -> ComputeBackend:
    """Seleciona o backend pedido; com fallback=True, degrada com aviso:
    cpu-parallel -> serial (1 núcleo); cuda sem GPU/CuPy -> serial. "auto" escolhe por mini-benchmark
    (ver benchmark.py: escolhe o mais rápido em uma população pequena).

    ``integrator`` só afeta o backend `cpu`: "auto" usa numba quando
    disponível (senão NumPy); "rk4_numpy"/"rk4_numba" forçam o integrador;
    "scipy" avalia com solve_ivp — equivalente ao backend serial."""
    nome = (nome or "serial").lower()
    if nome == "auto":
        from .benchmark import choose_backend
        return choose_backend(workers=workers)
    if nome == "cpu" and integrator == "scipy":
        return SerialBackend()
    cls = _BACKEND_CLASSES.get(nome)
    if cls is None:
        raise BackendError(
            f"Backend desconhecido: '{nome}'. Válidos: "
            f"{sorted(_BACKEND_CLASSES) + ['auto']}.")
    if nome == "cpu":
        b = CPUBackend(integrator=integrator)
    elif nome == "cpu-parallel":
        b = cls(workers=workers)
    else:
        b = cls()
    if not b.is_available():
        if fallback:
            import warnings
            warnings.warn(f"Backend '{nome}' indisponível; usando 'serial'. "
                          f"({cls.__doc__ or ''})".strip())
            return SerialBackend()
        raise BackendNotAvailableError(
            f"Backend '{nome}' indisponível nesta máquina.")
    return b