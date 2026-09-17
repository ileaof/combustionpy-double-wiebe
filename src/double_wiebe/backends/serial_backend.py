# -*- coding: utf-8 -*-
"""
serial_backend.py — Backend de REFERÊNCIA (regra 1 da especificação).

Implementação NumPy/SciPy original, serial, float64: solve_ivp adaptativo
(DOP853 por padrão) avaliado exatamente nos ângulos experimentais, uma
simulação por candidato. Toda otimização é validada numericamente contra
este backend e o melhor candidato final é re-integrado aqui.
"""
from __future__ import annotations

import numpy as np

from ..models import (CalibrationConfig, EngineConfig, PARAM_ORDER,
                      SimulationConfig, WiebeParameters)
from ..thermodynamics import PENALTY
from .base import ComputeBackend, batch_objective, regularize_batch


def _row_to_configs(row, engine: EngineConfig, wiebe: WiebeParameters):
    """Linha (10,) na ordem PARAM_ORDER -> (EngineConfig, WiebeParameters)."""
    eng = EngineConfig(**vars(engine))
    wieb = WiebeParameters(**vars(wiebe))
    eng.Rc = float(row[0])
    wieb.theta01, wieb.delta1, wieb.m1, wieb.a1 = (float(v) for v in row[1:5])
    wieb.theta02, wieb.delta2, wieb.m2, wieb.a2 = (float(v) for v in row[5:9])
    wieb.alpha = float(row[9])
    return eng, wieb


def objective_from_row(row, theta, P_exp, engine, wiebe, sim, calib) -> float:
    """Objetivo (RMSE + regularização) de UM candidato (linha PARAM_ORDER).
    Sempre via solve_ivp — exatamente o caminho serial original."""
    from ..simulation import evaluate_rmse

    eng, wieb = _row_to_configs(row, engine, wiebe)
    valor = evaluate_rmse(theta, P_exp, eng, wieb, sim)
    if valor >= PENALTY:
        return float(PENALTY)
    return float(valor) + _regularization_full(row, calib)


def _regularization_full(x_full, calib) -> float:
    """Regularização escalar do vetor completo (10,)."""
    from ..calibration import _regularization
    return _regularization(np.asarray(x_full, dtype=float), calib)


def _reg_batch(X_full, calib) -> np.ndarray:
    return regularize_batch(np.asarray(X_full, dtype=float), calib)


class SerialBackend(ComputeBackend):
    name = "serial"
    description = ("referência NumPy/SciPy (solve_ivp adaptativo, float64) — "
                   "caminho original, sem alterações")

    def is_available(self) -> bool:
        return True                      # só precisa de numpy+scipy

    def capabilities(self) -> dict:
        return {"integrador": "solve_ivp (adaptativo)", "precision": "float64",
                "paralelo": False, "vetorizado": False}

    def simulate(self, theta, P_exp, engine, wiebe, sim):
        from ..thermodynamics import integrate_ode
        return integrate_ode(np.asarray(theta, dtype=float), engine, wiebe,
                             float(P_exp[0]), method=sim.method,
                             rtol=sim.rtol, atol=sim.atol)

    def evaluate_population(self, X_full, theta, P_exp, engine, wiebe, sim,
                            calib, precision="float64", substeps=4,
                            batch_size=0):
        X_full = np.atleast_2d(np.asarray(X_full, dtype=float))
        S = X_full.shape[0]
        out = np.empty(S)
        for i in range(S):
            eng, wieb = _row_to_configs(X_full[i], engine, wiebe)
            valor = evaluate_rmse_np(theta, P_exp, eng, wieb, sim)
            if valor >= PENALTY:
                out[i] = PENALTY
            else:
                out[i] = valor + float(_reg_batch(X_full[i:i + 1], calib)[0])
        return out


def evaluate_rmse_np(theta, P_exp, eng, wieb, sim) -> float:
    from ..simulation import evaluate_rmse
    return float(evaluate_rmse(np.asarray(theta, dtype=float),
                               np.asarray(P_exp, dtype=float),
                               eng, wieb, sim))


class CPUBackend(ComputeBackend):
    """Backend `cpu`: avalia a população em LOTE com RK4 de passo fixo
    (NumPy, ou Numba compilada se o extra [cpu] estiver instalado).

    Equações idênticas à referência; muda apenas o integrador (modo
    acelerado). O MELHOR candidato final é re-integrado com solve_ivp e a
    diferença de objetivo é relatada (ver calibration.run_calibration)."""
    name = "cpu"
    description = ("RK4 em lote vetorizado (NumPy/Numba) — avalia a "
                   "população inteira por geração; modo acelerado")

    def __init__(self, integrator: str = "auto"):
        # "auto": numba se disponível, senão numpy
        self.integrator = integrator

    def is_available(self) -> bool:
        return True

    def capabilities(self) -> dict:
        from .detection import detect_numba
        nb = detect_numba()
        return {"integrador": f"RK4 passo fixo ({self._fn_name()})",
                "precision": "float64|float32", "paralelo": False,
                "vetorizado": True, "numba": nb["disponivel"]}

    def _fn(self):
        if self.integrator == "rk4_numpy":
            from ..integrators.rk4_numpy import batch_integrate_np
            return batch_integrate_np
        from ..integrators.rk4_numba import batch_integrate_numba
        return batch_integrate_numba

    def _fn_name(self) -> str:
        return ("rk4_numpy" if self.integrator == "rk4_numpy"
                else "rk4_numba")

    def simulate(self, theta, P_exp, engine, wiebe, sim):
        return SerialBackend().simulate(theta, P_exp, engine, wiebe, sim)

    def synchronize(self) -> None:
        pass

    def evaluate_population(self, X_full, theta, P_exp, engine, wiebe, sim,
                            calib, precision="float64", substeps=4,
                            batch_size=0):
        X_full = np.atleast_2d(np.asarray(X_full, dtype=float))
        theta = np.asarray(theta, dtype=float)
        P1 = float(P_exp[0])
        fn = self._fn()
        S = X_full.shape[0]
        step = batch_size if batch_size and batch_size > 0 else max(S, 1)
        out = np.empty(S)
        for ini in range(0, S, step):
            bloco = X_full[ini:ini + step]
            P_sim = fn(theta, P1, bloco, engine, sim, substeps, precision)
            out[ini:ini + step] = batch_objective(
                P_sim, bloco, np.asarray(P_exp, dtype=float), calib,
                lambda X: _reg_batch(X, calib))
        return out


class MultiprocessingBackend(ComputeBackend):
    """Backend `cpu-parallel`: avalia a população em processos separados
    (ProcessPoolExecutor, contexto spawn no Windows).

    Cada candidato é avaliado com a MESMA implementação serial (solve_ivp)
    — os números são idênticos aos do backend serial, apenas em paralelo.
    Paraleliza ENTRE candidatos (população/PSO/DE/sensibilidade), nunca
    dentro de uma partícula."""
    name = "cpu-parallel"
    description = ("população em processos (ProcessPoolExecutor/spawn) com a "
                   "implementação serial — resultados idênticos ao serial")

    def __init__(self, workers: int | None = None):
        self.workers = workers if workers and workers > 0 \
            else max(1, (os_cpu() or 1) - 1)
        self._pool = None

    def is_available(self) -> bool:
        return (os_cpu() or 1) > 1

    def capabilities(self) -> dict:
        return {"integrador": "solve_ivp (adaptativo)", "precision": "float64",
                "paralelo": True, "workers": self.workers}

    def _get_pool(self):
        if self._pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            ctx = mp.get_context("spawn")     # Windows-safe
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=ctx)
        return self._pool

    def simulate(self, theta, P_exp, engine, wiebe, sim):
        return SerialBackend().simulate(theta, P_exp, engine, wiebe, sim)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def evaluate_population(self, X_full, theta, P_exp, engine, wiebe, sim,
                            calib, precision="float64", substeps=4,
                            batch_size=0):
        import numpy as np
        X_full = np.atleast_2d(np.asarray(X_full, dtype=float))
        pool = self._get_pool()
        tarefas = [(X_full[i], np.asarray(theta, dtype=float),
                    np.asarray(P_exp, dtype=float), engine, wiebe, sim,
                    calib) for i in range(X_full.shape[0])]
        vals = list(pool.map(_mp_worker, tarefas, chunksize=max(
            1, len(tarefas) // (self.workers * 4))))
        return np.asarray(vals, dtype=float)


def os_cpu() -> int | None:
    import os
    return os.cpu_count()


def _mp_worker(tarefa):
    """Top-level (picklable): avalia UM candidato com o caminho serial."""
    (row, theta, P_exp, engine, wiebe, sim, calib) = tarefa
    return objective_from_row(row, theta, P_exp, engine, wiebe, sim, calib)