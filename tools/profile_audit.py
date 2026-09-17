# -*- coding: utf-8 -*-
"""
profile_audit.py — Etapa 1 do plano HPC: auditoria de desempenho.

Mede o tempo de cada etapa do núcleo Double Wiebe (dados, geometria,
Wiebe, EDOs, função objetivo, calibração, gráficos, exportação), conta
chamadas do RHS e da função objetivo e roda cProfile na avaliação unitária.

Uso:  python tools/profile_audit.py
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time
from pathlib import Path

import numpy as np

from double_wiebe.data_processing import read_table, prepare_series
from double_wiebe.geometry import cylinder_volume, dV_dtheta, heat_area
from double_wiebe.models import (CalibrationConfig, EngineConfig,
                                 SimulationConfig, WiebeParameters)
from double_wiebe.simulation import run_simulation
from double_wiebe.thermodynamics import integrate_ode
from double_wiebe.wiebe import double_burned_fraction

DATA = Path(__file__).resolve().parents[2] / "P_exp-Carga-3_45%.txt"
REP = 20


def t_medio(fn, rep=REP):
    """Tempo médio (s) e desvio padrão de fn() com warm-up."""
    fn()                                   # warm-up
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return float(ts.mean()), float(ts.std())


def main() -> None:
    linhas = []                            # (etapa, tempo_medio, desvio)

    # ------------------------------------------------------------------ dados
    raw = DATA.read_bytes()
    t, sd = t_medio(lambda: prepare_series(
        read_table(io.BytesIO(raw), sep="auto", has_header=False),
        angle_unit="radianos", pressure_unit="bar",
        theta_min=-2.0, theta_max=2.0), rep=10)
    linhas.append(("leitura dos dados (read_table+prepare_series)", t, sd))
    theta, P, _ = prepare_series(
        read_table(io.BytesIO(raw), sep="auto", has_header=False),
        angle_unit="radianos", pressure_unit="bar",
        theta_min=-2.0, theta_max=2.0)
    n = theta.size

    engine, wiebe, sim = EngineConfig(), WiebeParameters(), SimulationConfig()

    # -------------------------------------------------------------- geometria
    def _geo():
        V = cylinder_volume(theta, engine.Rc, engine)
        dV = dV_dtheta(theta, engine.Rc, engine)
        A = heat_area(theta, engine.Rc, engine)
        return V, dV, A

    t, sd = t_medio(_geo)
    linhas.append((f"geometria V+dV+A ({n} pts)", t, sd))

    # ----------------------------------------------------------------- wiebe
    t, sd = t_medio(lambda: double_burned_fraction(theta, wiebe))
    linhas.append((f"Double Wiebe x1,x2,xb,dxb ({n} pts)", t, sd))

    # -------------------------------------------------------------- EDO (RHS)
    t, sd = t_medio(lambda: integrate_ode(
        theta, engine, wiebe, float(P[0]),
        method=sim.method, rtol=sim.rtol, atol=sim.atol))
    linhas.append((f"integrate_ode {sim.method} rtol={sim.rtol:g} "
                   f"({n} pontos)", t, sd))

    # contagem de chamadas do RHS via cProfile
    prof = cProfile.Profile()
    prof.enable()
    integrate_ode(theta, engine, wiebe, float(P[0]),
                  method=sim.method, rtol=sim.rtol, atol=sim.atol)
    prof.disable()
    import pstats as ps
    st = ps.Stats(prof)
    rhs_calls = sum(nc for func, (cc, nc, tt, ct, callers)
                    in st.stats.items()
                    if func[2] == "rhs")
    print(f"[contagem] chamadas do RHS em 1 integração ({sim.method}): "
          f"{rhs_calls}")

    # ------------------------------------------------------- função objetivo
    from double_wiebe.simulation import evaluate_rmse
    t, sd = t_medio(lambda: evaluate_rmse(theta, P, engine, wiebe, sim))
    linhas.append(("evaluate_rmse (1 avaliação da função objetivo)", t, sd))

    # cProfile da função objetivo: onde o tempo vai dentro dela?
    prof = cProfile.Profile()
    prof.enable()
    for _ in range(5):
        evaluate_rmse(theta, P, engine, wiebe, sim)
    prof.disable()
    st = ps.Stats(prof)
    st.sort_stats("cumulative")
    saida = io.StringIO()
    st.stream = saida
    st.print_stats(12)
    print("\n=== cProfile: evaluate_rmse (10 chamadas) ===")
    print("\n".join(saida.getvalue().splitlines()[4:20]))

    # ------------------------------------------------------------ calibração
    calib = CalibrationConfig(
        method="differential-evolution",
        selected=["Rc", "theta01", "delta1", "m1", "theta02", "delta2", "m2",
                  "alpha"],
        seed=42, maxiter=5, popsize=8, tol=1e-10, polish=False)
    from double_wiebe.calibration import run_calibration
    t0 = time.perf_counter()
    r = run_calibration(theta, P, engine, wiebe, sim, calib)
    t_cal = time.perf_counter() - t0
    linhas.append((f"calibração DE (maxiter=5, popsize=8 -> {r['iteracoes']} "
                   f"iterações)", t_cal, 0.0))
    # n chamadas ~= maxiter*popsize*nvar (scipy DE)
    print(f"[contagem] DE maxiter=5 popsize=8 8params: iterações="
          f"{r['iteracoes']}, RMSE={r['rmse']:.1f}")

    # ------------------------------------------------------------ pós-processo
    from double_wiebe.metrics import summary_metrics
    P_sim = r2_ = None
    P_sim, _, _ = integrate_ode(theta, engine, wiebe, float(P[0]))
    t, sd = t_medio(lambda: summary_metrics(P, P_sim))
    linhas.append(("métricas (summary_metrics)", t, sd))

    # ----------------------------------------------------- gráficos + export
    from double_wiebe.plotting import (fig_burned, fig_heat_release,
                                       fig_pressure, fig_pv, fig_temperature)
    from double_wiebe.reporting import (convergence_csv_bytes,
                                        metrics_json_bytes,
                                        parameters_yaml_bytes, report_html_bytes,
                                        results_csv_bytes)

    res = run_simulation(theta, P, engine, wiebe, sim)
    cal = run_calibration(theta, P, engine, wiebe, sim, calib)

    def _graficos():
        for f in (fig_pressure(res, False, "kPa"), fig_burned(res, False),
                  fig_temperature(res, False), fig_pv(res, "kPa"),
                  fig_heat_release(res, False)):
            f.to_json()

    t, sd = t_medio(_graficos, rep=5)
    linhas.append(("5 gráficos plotly (to_json)", t, sd))

    def _export():
        results_csv_bytes(res)
        parameters_yaml_bytes(res)
        metrics_json_bytes(res, cal)
        report_html_bytes(res, cal)

    t, sd = t_medio(_export, rep=5)
    linhas.append(("exportação (csv+yaml+json+report.html)", t, sd))

    # ------------------------------------------------------------------ total
    t_total = sum(x[1] for x in linhas)
    print("\n================ TABELA DE AUDITORIA ================")
    print(f"{'etapa':<58}{'média (s)':>10}{'desvio':>9}{'% do total':>11}")
    print("-" * 88)
    for nome, tm, s in linhas:
        print(f"{nome:<58}{tm:>10.4f}s{s:>9.4f}{100*tm/t_total:>10.1f}%")
    print("-" * 88)
    print(f"{'TOTAL (1 sim + 1 calibração curta + plots + export)':<58}"
          f"{t_total:>10.4f}s")
    print(f"\nObservação: 1 avaliação da função objetivo = "
          f"{linhas[4][1]*1000:.1f} ms -> calibração DE default "
          f"(popsize 20, 8 params, maxiter 200) ~= "
          f"{20*8*200*linhas[4][1]/60:.1f} min só de avaliações.")


if __name__ == "__main__":
    main()