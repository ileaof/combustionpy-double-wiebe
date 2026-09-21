# -*- coding: utf-8 -*-
"""
cuda_backend.py — Backend `cuda`: RK4 em lote na GPU (CuPy, extra [cuda]).

Um thread CUDA por candidato (integrators/rk4_cuda.py). Para minimizar a
transferência host<->device, ``theta``/``P_exp`` ficam em cache na GPU entre
gerações e o RMSE é reduzido na própria GPU: por geração sobem só os
candidatos (S x 10) e descem só S valores. A regularização (barata) é
calculada no host com as mesmas fórmulas dos demais backends.

Mesma semântica do backend `cpu` (modo acelerado): o MELHOR candidato final
é re-integrado com solve_ivp e a diferença de objetivo é relatada (ver
calibration.run_calibration).
"""
from __future__ import annotations

import numpy as np

from ..thermodynamics import PENALTY
from .base import ComputeBackend, regularize_batch
from .serial_backend import SerialBackend


class CUDABackend(ComputeBackend):
    """Backend `cuda`: RK4 em lote na GPU (requer CuPy e GPU NVIDIA)."""
    name = "cuda"
    description = ("RK4 em lote na GPU (CUDA/CuPy, 1 thread por candidato) "
                   "— modo acelerado; float32 recomendado em GPUs de consumo")

    def __init__(self, device: int = 0):
        self.device = device
        self._cache_key = None
        self._theta_d = None
        self._P_exp_d = None

    def is_available(self) -> bool:
        from ..integrators.rk4_cuda import cuda_available
        return cuda_available()

    def capabilities(self) -> dict:
        from .detection import detect_cuda
        info = detect_cuda()
        return {"integrador": "RK4 passo fixo (rk4_cuda)",
                "precision": "float64|float32", "paralelo": True,
                "vetorizado": True, "gpus": info.get("gpus", [])}

    def simulate(self, theta, P_exp, engine, wiebe, sim):
        return SerialBackend().simulate(theta, P_exp, engine, wiebe, sim)

    def synchronize(self) -> None:
        import cupy as cp
        cp.cuda.Device(self.device).synchronize()

    def close(self) -> None:
        self._cache_key = self._theta_d = self._P_exp_d = None

    def _dados_na_gpu(self, theta, P_exp):
        """theta/P_exp na GPU, reaproveitados enquanto não mudarem."""
        import cupy as cp
        chave = (theta.tobytes(), P_exp.tobytes())
        if chave != self._cache_key:
            self._theta_d = cp.asarray(theta)
            self._P_exp_d = cp.asarray(P_exp)
            self._cache_key = chave
        return self._theta_d, self._P_exp_d

    def evaluate_population(self, X_full, theta, P_exp, engine, wiebe, sim,
                            calib, precision="float64", substeps=4,
                            batch_size=0):
        import cupy as cp
        from ..integrators.rk4_cuda import batch_integrate_cuda_device

        X_full = np.atleast_2d(np.asarray(X_full, dtype=float))
        theta = np.ascontiguousarray(theta, dtype=np.float64)
        P_exp = np.ascontiguousarray(P_exp, dtype=np.float64)
        S = X_full.shape[0]
        step = batch_size if batch_size and batch_size > 0 else max(S, 1)
        out = np.empty(S)
        with cp.cuda.Device(self.device):
            theta_d, P_exp_d = self._dados_na_gpu(theta, P_exp)
            for ini in range(0, S, step):
                bloco = X_full[ini:ini + step]
                P_sim = batch_integrate_cuda_device(
                    theta_d, float(P_exp[0]), bloco, engine, substeps,
                    precision)
                # RMSE na GPU; linha com NaN -> PENALTY (= batch_objective)
                ok = cp.isfinite(P_sim).all(axis=1)
                d = P_sim - P_exp_d[None, :]
                rmse = cp.where(ok, cp.sqrt(cp.mean(d * d, axis=1)),
                                float(PENALTY))
                out[ini:ini + step] = cp.asnumpy(rmse)
        return out + regularize_batch(X_full, calib)
