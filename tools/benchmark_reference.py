# -*- coding: utf-8 -*-
"""
benchmark_reference.py — Benchmark de referência ANTES das otimizações
(Etapa 1 do plano HPC). Mede a implementação CPU serial original e salva
o baseline em outputs/baseline_serial.json, para comparação posterior.

Reproduzível:  python tools/benchmark_reference.py
"""
from __future__ import annotations

import io
import json
import platform
import time
from pathlib import Path

import numpy as np

from double_wiebe.data_processing import read_table, prepare_series
from double_wiebe.models import (CalibrationConfig, EngineConfig,
                                 SimulationConfig, WiebeParameters)
from double_wiebe.simulation import evaluate_rmse, run_simulation
from double_wiebe.calibration import run_calibration

DATA = Path(__file__).resolve().parents[2] / "P_exp-Carga-3_45%.txt"
OUT = Path(__file__).resolve().parents[1] / "outputs" / "baseline_serial.json"
REP = 20


def main() -> None:
    theta, P, _ = prepare_series(
        read_table(DATA, sep="auto", has_header=False),
        angle_unit="radianos", pressure_unit="bar",
        theta_min=-2.0, theta_max=2.0)
    engine, wiebe, sim = EngineConfig(), WiebeParameters(), SimulationConfig()

    # 1 simulação completa
    run_simulation(theta, P, engine, wiebe, sim)          # warm-up
    ts = []
    for _ in range(REP):
        t0 = time.perf_counter()
        run_simulation(theta, P, engine, wiebe, sim)
        ts.append(time.perf_counter() - t0)
    t_sim = float(np.mean(ts))
    sd_sim = float(np.std(ts))

    # 1 avaliação da função objetivo
    evaluate_rmse(theta, P, engine, wiebe, sim)           # warm-up
    ts = []
    for _ in range(REP):
        t0 = time.perf_counter()
        evaluate_rmse(theta, P, engine, wiebe, sim)
        ts.append(time.perf_counter() - t0)
    t_obj = float(np.mean(ts))
    sd_obj = float(np.std(ts))

    # calibração DE curta (reprodutível, seed fixa)
    calib = CalibrationConfig(
        method="differential-evolution",
        selected=["Rc", "theta01", "delta1", "m1", "theta02", "delta2", "m2",
                  "alpha"],
        seed=42, maxiter=20, popsize=15, tol=1e-10, polish=False)
    t0 = time.perf_counter()
    r = run_calibration(theta, P, engine, wiebe, sim, calib)
    t_cal = time.perf_counter() - t0

    baseline = {
        "data": DATA.name,
        "n_obs": int(theta.size),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "n_cores": int(__import__("os").cpu_count()),
        "t_simulacao_s": t_sim,
        "sd_simulacao_s": sd_sim,
        "t_objetivo_s": t_obj,
        "sd_objetivo_s": sd_obj,
        "t_calibracao_DE_maxiter20_popsize15_s": t_cal,
        "rmse_calibracao_kPa": float(r["rmse"]),
        "params_calibracao": {k: float(v) for k, v in r["params"].items()},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(json.dumps(baseline, indent=2))
    print(f"\nbaseline salvo em: {OUT}")


if __name__ == "__main__":
    main()