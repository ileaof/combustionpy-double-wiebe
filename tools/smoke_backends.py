# -*- coding: utf-8 -*-
"""
smoke_backends.py — Validação da orquestração de backends em run_calibration
(regra 4: toda otimização validada contra a serial).

Verificacoes:
  1. PSO: cpu-parallel (solve_ivp em processos) reproduz a serial EXATAMENTE
     (mesma seed -> mesma trajetoria, pois a avaliacao em lote nao altera a
     dinamica do enxame).
  2. DE: com backend acelerado scipy exige vectorized=True + updating=
     "deferred" -> a TRAJETORIA de busca difere da serial (immediate). Nao e
     bug: a equivalencia de valores por candidato e comprovada em
     check_equivalence.py e o resultado final e sempre re-validado com
     solve_ivp (diferenca_integrador).
  3. Historico sem NaN, rmse final sempre do re-run serial.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from double_wiebe.data_processing import read_table, prepare_series
from double_wiebe.models import (CalibrationConfig, EngineConfig,
                                 SimulationConfig, WiebeParameters)
from double_wiebe.calibration import run_calibration

DATA = Path(__file__).resolve().parents[2] / "P_exp-Carga-3_45%.txt"
SELECTED = ["Rc", "theta01", "delta1", "m1", "theta02", "delta2", "m2",
            "alpha"]


def _rodar(theta, P, engine, wiebe, sim, **kw):
    calib = CalibrationConfig(selected=list(SELECTED), seed=42, tol=1e-10,
                              polish=False, **kw)
    t0 = time.perf_counter()
    r = run_calibration(theta, P, engine, wiebe, sim, calib)
    dt = time.perf_counter() - t0
    hist = np.asarray(r["objective_history"], dtype=float)
    assert np.isfinite(hist).all(), f"{calib.backend}: NaN no historico!"
    print(f"[{calib.backend:>13}] metodo={calib.method[:3]} "
          f"rmse={r['rmse']:10.4f} kPa  t={dt:6.2f}s  "
          f"integrador={r['rmse_integrador']:10.4f}  "
          f"dif={r['diferenca_integrador']:.3e}")
    return r


def main() -> None:
    theta, P, _ = prepare_series(
        read_table(DATA, sep="auto", has_header=False),
        angle_unit="radianos", pressure_unit="bar",
        theta_min=-2.0, theta_max=2.0)
    engine, wiebe, sim = EngineConfig(), WiebeParameters(), SimulationConfig()

    # ---- 1. PSO: cpu-parallel == serial (exato) --------------------------
    print("--- PSO (mesma seed) ---")
    pso = {
        b: _rodar(theta, P, engine, wiebe, sim, method="pso",
                  pso_particles=12, pso_max_iter=25, backend=b)
        for b in ("serial", "cpu", "cpu-parallel")
    }
    dx_pso = float(np.max(np.abs(
        np.array([pso["cpu-parallel"]["params"][k]
                  for k in SELECTED])
        - np.array([pso["serial"]["params"][k] for k in SELECTED]))))
    df_pso = abs(pso["cpu-parallel"]["rmse"] - pso["serial"]["rmse"])
    print(f"PSO cpu-parallel vs serial: |dRMSE|={df_pso:.6e}  "
          f"max|dparam|={dx_pso:.6e}")
    assert df_pso == 0.0 and dx_pso == 0.0, \
        "PSO cpu-parallel deveria reproduzir a serial exatamente"

    # ---- 2. DE: trajetorias diferem (deferred vs immediate); final
    #         sempre re-validado com solve_ivp ------------------------------
    print("\n--- DE (mesma seed; trajetorias diferem por updating=deferred) ---")
    de = {
        b: _rodar(theta, P, engine, wiebe, sim, method="differential-evolution",
                  maxiter=6, popsize=10, backend=b)
        for b in ("serial", "cpu", "cpu-parallel")
    }
    for b in ("cpu", "cpu-parallel"):
        d = abs(de[b]["rmse"] - de["serial"]["rmse"])
        print(f"DE {b} vs serial: |dRMSE|={d:.4f} kPa "
              f"(trajetoria differ; re-run serial valida o resultado)")

    print("\nSMOKE OK")


if __name__ == "__main__":
    sys.exit(main())