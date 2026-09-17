# -*- coding: utf-8 -*-
"""
base.py — Contrato dos backends de computação (API do plano HPC).

Todo backend implementa a interface :class:`ComputeBackend`:

    name            identificador ("serial", "cpu", "cpu-parallel", ...)
    description     texto curto para CLI/GUI
    is_available()  True se pode ser usado nesta máquina
    capabilities()  dict com detalhes (integrador, precision, workers...)
    simulate(...)   1 simulação -> (P_sim, Tg, Q_wall)  [referência]
    evaluate_population(candidates, theta, P_exp, engine, wiebe, sim,
                        calib, precision="float64", substeps=4, batch_size=0)
                    -> valores da função objetivo (N,)   [lote]
    synchronize()   barra de sincronização (no-op em CPU; explícito em GPU)

A referência numérica é SEMPRE a implementação serial (solve_ivp/DOP853,
float64) — regra fundamental do projeto: toda otimização é validada contra
ela e o melhor candidato final é re-integrado com solve_ivp.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..models import (CalibrationConfig, EngineConfig, PARAM_ORDER,
                      SimulationConfig)


class BackendError(Exception):
    """Backend indisponível ou pedido inválido (ex.: CUDA sem GPU)."""


class BackendNotAvailableError(BackendError):
    """O backend solicitado não pode executar nesta máquina."""


class ComputeBackend(ABC):
    """Interface única usada por CLI/GUI/calibração."""

    name: str = "base"
    description: str = ""

    # ---------------------------------------------------------------- status
    @abstractmethod
    def is_available(self) -> bool:
        """True se o backend pode executar nesta máquina."""

    @abstractmethod
    def capabilities(self) -> dict:
        """Metadados para `double-wiebe devices` e a GUI."""

    # -------------------------------------------------------------------- A0
    @abstractmethod
    def simulate(self, theta, P_exp, engine: EngineConfig,
                 wiebe, sim: SimulationConfig):
        """1 simulação de referência -> (P_sim, Tg, Q_wall)."""

    # -------------------------------------------------------------- populac.
    @abstractmethod
    def evaluate_population(self, candidates, theta, P_exp,
                            engine: EngineConfig, wiebe, sim: SimulationConfig,
                            calib: CalibrationConfig,
                            precision: str = "float64", substeps: int = 4,
                            batch_size: int = 0) -> np.ndarray:
        """Avalia a função objetivo (RMSE + regularização) para uma
        população de candidatos (S, len(selected))."""

    # ------------------------------------------------------------ sincronia
    def synchronize(self) -> None:
        """Ponto de sincronização explícito (no-op em CPU)."""

    def close(self) -> None:
        """Libera recursos (pools)."""


# ---------------------------------------------------------------------------
# Objetivo em lote (compartilhado pelos backends acelerados)
# ---------------------------------------------------------------------------
def batch_objective(P_sim, X_full, P_exp, calib, regularize):
    """RMSE + regularização por candidato a partir de P_sim (S, n).

    Linha com NaN (candidato falho) recebe a PENALTY — mesma semântica da
    referência serial (ODEFailure -> 1e10).
    """
    from ..thermodynamics import PENALTY

    S = P_sim.shape[0]
    rmse = np.full(S, float(PENALTY))
    ok = np.isfinite(P_sim).all(axis=1)
    if ok.any():
        d = P_sim[ok] - P_exp[None, :]
        rmse[ok] = np.sqrt(np.mean(d * d, axis=1))
    if regularize is not None:
        rmse = rmse + regularize(X_full)
    return rmse


def regularize_batch(X_full, calib) -> np.ndarray:
    """Regularização vetorizada (mesmas fórmulas de calibration._regularization)."""
    import math

    theta01, delta1 = X_full[:, 1], X_full[:, 2]
    theta02 = X_full[:, 5]
    delta2 = X_full[:, 6]
    delta_min = math.radians(calib.delta_min_deg)
    pen = calib.w_order * np.maximum(0.0, theta01 - theta02) ** 2
    pen = pen + calib.w_overlap * np.maximum(
        0.0, (theta01 + delta1) - theta02) ** 2
    pen = pen + calib.w_min_duration * (
        np.maximum(0.0, delta_min - delta1) ** 2
        + np.maximum(0.0, delta_min - delta2) ** 2)
    if calib.w_ridge > 0.0:
        from ..calibration import PARAM_SPECS
        for j, nome in enumerate(PARAM_ORDER):
            lo, hi = PARAM_SPECS[nome]["lower"], PARAM_SPECS[nome]["upper"]
            span = hi - lo
            margem = np.minimum(X_full[:, j] - lo, hi - X_full[:, j]) / span
            pen = pen + calib.w_ridge * (
                np.maximum(0.0, 0.02 - margem) / 0.02) ** 2
    return pen