# -*- coding: utf-8 -*-
"""
check_equivalence.py — Equivalência por candidato (regra 4): população FIXA,
compara SerialBackend.evaluate_population vs cpu (RK4) vs cpu-parallel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from double_wiebe.data_processing import read_table, prepare_series
from double_wiebe.models import (CalibrationConfig, DEFAULT_SELECTED,
                                 EngineConfig, PARAM_ORDER,
                                 SimulationConfig, WiebeParameters)
from double_wiebe.backends import (CPUBackend, MultiprocessingBackend,
                                   SerialBackend)

DATA = Path(__file__).resolve().parents[2] / "P_exp-Carga-3_45%.txt"


def main() -> None:
    theta, P, _ = prepare_series(
        read_table(DATA, sep="auto", has_header=False),
        angle_unit="radianos", pressure_unit="bar",
        theta_min=-2.0, theta_max=2.0)
    engine, wiebe, sim = EngineConfig(), WiebeParameters(), SimulationConfig()
    calib = CalibrationConfig()

    rng = np.random.default_rng(7)
    selected = list(DEFAULT_SELECTED)
    full0 = np.array([engine.Rc, wiebe.theta01, wiebe.delta1, wiebe.m1,
                      wiebe.a1, wiebe.theta02, wiebe.delta2, wiebe.m2,
                      wiebe.a2, wiebe.alpha])
    X_sel = np.column_stack([
        rng.uniform(0.2, 0.8) * (1.0 - 0.0) + 0.0 for _ in selected])
    # popula dentro dos limites reais
    from double_wiebe.calibration import PARAM_SPECS
    X_sel = np.column_stack([
        rng.uniform(PARAM_SPECS[n]["lower"] + 0.1 * (PARAM_SPECS[n]["upper"]
                                                    - PARAM_SPECS[n]["lower"]),
                    PARAM_SPECS[n]["upper"] - 0.1 * (PARAM_SPECS[n]["upper"]
                                                     - PARAM_SPECS[n]["lower"]),
                    40) for n in selected])
    idx = [PARAM_ORDER.index(n) for n in selected]
    X_full = np.tile(full0, (40, 1))
    X_full[:, idx] = X_sel

    # referência: objetivo serial candidato a candidato
    sb = SerialBackend()
    f_serial = sb.evaluate_population(X_full, theta, P, engine, wiebe, sim,
                                      calib, selected, full0)

    for backend in (CPUBackend(), MultiprocessingBackend()):
        f = backend.evaluate_population(X_full, theta, P, engine, wiebe,
                                        sim, calib)
        d = np.abs(f - f_serial)
        rel = d / np.maximum(np.abs(f_serial), 1e-12)
        print(f"{backend.name:>14}: max|df|={d.max():.6e} kPa  "
              f"max rel={rel.max():.3e}  "
              f"penalty iguais={bool(((f >= 1e9) == (f_serial >= 1e9)).all())}")

    # cpu vs cpu-parallel (mesma matemática? não: mp = solve_ivp serial)
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()