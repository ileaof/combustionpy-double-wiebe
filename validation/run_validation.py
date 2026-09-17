# -*- coding: utf-8 -*-
"""
run_validation.py
=================
Harness de validação de paridade Mathematica (MOD0d, notebook
``Modelo_Double_Wiebe_v2.nb``) x pacote Python ``double_wiebe``.

Executa as etapas 2-13 da especificação:
  2. dados experimentais (503 -> 459, filtro -2 <= theta <= 2, bar -> kPa x100)
  3. constantes do motor
  4. geometria em 8 angulos + diferencas finitas
  5. Double Wiebe (zeros antes do inicio, derivadas por DF, combinacao beta,
     reducao ao single em beta = 0/1)
  6. calor liberado Q = mcomb * PCI * x
  7. correlacao de Hohenberg
  8. EDOs (solve_ivp DOP853) -> casos-ancora do Mathematica
  9. funcao de erro do notebook sqrt(SSres/(q-2)) x RMSE do Python sqrt(SSE/q)
 10. caso obrigatorio erro ~= 100.538 kPa (tolerancia 0.5 kPa)
 11. vetor ativo do PSO erro ~= 55.381 kPa (tolerancia 0.5 kPa)
 12. CSV ponto a ponto (459 linhas)
 13. graficos (experimental '+', Mathematica tracejada preta, Python verde)

Nao modifica nenhum arquivo do pacote: apenas le o codigo e grava os
artefatos de validacao (CSV/JSON/PNG) nesta pasta.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]          # .../double_wiebe
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

import mathematica_reference as mm                  # noqa: E402
from double_wiebe import data_processing as dp      # noqa: E402
from double_wiebe.geometry import (                 # noqa: E402
    cylinder_volume, dV_dtheta, heat_area, piston_disp)
from double_wiebe.models import EngineConfig, SimulationConfig, WiebeParameters  # noqa: E402
from double_wiebe.simulation import run_simulation  # noqa: E402
from double_wiebe.thermodynamics import hohenberg_h  # noqa: E402
from double_wiebe.wiebe import double_burned_fraction  # noqa: E402

DATA_PATH = Path(r"C:\Users\ileao\OneDrive\Documentos\Larissa"
                 r"\P_exp-Carga-3_45%.txt")
OUT = Path(__file__).resolve().parent

# Vetores de referencia (ordem do notebook: Rc, m1, theta01, d1, m2, theta02, d2, beta)
CASE_A = [15.34, 0.6, math.radians(-8.0), math.radians(15.0),
          0.141, math.radians(2.83), math.radians(62.06), 0.063]
CASE_A_ANCHOR = 100.538          # erro do Mathematica (caso obrigatorio)
CASE_B = [15.3496, 0.6, -0.193038, 0.436332,
          0.9, 0.0317214, 0.523599, 0.075285]
CASE_B_ANCHOR = 55.381           # erro do Mathematica (PSO, iteracao 190)
CASE_C = [15.340724668321869, 0.5999990714343, -0.13962634015954636,
          0.2617993877991494, 0.14162900274166848, 0.04937752133906362,
          1.082970478069876, 0.06333317846901508]
CASE_C_ANCHOR = 34.25858860927897   # vetor comentado no notebook (auxiliar)

TOL_ERRO = 0.5        # kPa  (erros globais)
TOL_P = 1.0           # kPa  (ponto a ponto ref x python)
TOL_TG = 1.0          # K
TOL_REL = 1e-3        # erro relativo ponto a ponto
RTOL_ALG = 1e-10
ATOL_ALG = 1e-12

CHECKS = []          # [(etapa, item, valor_ref, valor_py, tol, ok, obs)]


def check(etapa, item, ref, py, tol, obs=""):
    ok = bool(np.all(np.isfinite([ref, py]))) and abs(ref - py) <= tol
    CHECKS.append((etapa, item, ref, py, tol, ok, obs))
    return ok


def rel_err(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


# =============================================================================
# Etapa 2 — dados
# =============================================================================
def etapa_dados():
    print("== Etapa 2: dados experimentais ==")
    theta_all, p_bar_all, theta_ref, P_ref = mm.load_and_filter(str(DATA_PATH))
    df = dp.read_table(str(DATA_PATH))
    theta_py, P_py, resumo = dp.prepare_series(
        df, angle_unit="radianos", pressure_unit="bar",
        theta_min=-2.0, theta_max=2.0)
    r = {
        "n_total": int(theta_all.size),
        "n_filtrado": int(theta_ref.size),
        "theta_min": float(theta_ref[0]),
        "theta_max": float(theta_ref[-1]),
        "passo_medio": float(np.mean(np.diff(theta_ref))),
        "P1_kPa": float(P_ref[0]),
    }
    check(2, "n_total", 503, r["n_total"], 0)
    check(2, "n_filtrado", 459, r["n_filtrado"], 0)
    check(2, "theta_min [rad]", -1.998401994, r["theta_min"], 1e-9)
    check(2, "theta_max [rad]", 1.998401994, r["theta_max"], 1e-9)
    check(2, "passo_medio [rad]", 0.008726646, r["passo_medio"], 1e-9)
    check(2, "P1 [kPa]", 138.2, r["P1_kPa"], 1e-6)
    ok_series = (theta_ref.size == theta_py.size
                 and np.allclose(theta_ref, theta_py, rtol=0, atol=0)
                 and np.allclose(P_ref, P_py, rtol=0, atol=0))
    CHECKS.append((2, "serie identica (ref x prepare_series)",
                   0.0, 0.0 if ok_series else 1.0, 0.0, ok_series,
                   "elemento a elemento, bit a bit"))
    print(f"   n={r['n_filtrado']}/{r['n_total']}  "
          f"theta=[{r['theta_min']:.9f}, {r['theta_max']:.9f}]  "
          f"passo={r['passo_medio']:.9f}  P1={r['P1_kPa']:.1f} kPa  "
          f"serie identica: {ok_series}")
    return theta_ref, P_ref, r


# =============================================================================
# Etapa 3 — constantes
# =============================================================================
def etapa_constantes():
    print("== Etapa 3: constantes do motor ==")
    eng = EngineConfig()
    pares = [
        ("bore d [m]", mm.D, eng.bore), ("stroke s [m]", mm.S, eng.stroke),
        ("rod l [m]", mm.L, eng.rod_length),
        ("kappa", mm.KP, eng.kappa), ("rpm/60 [rev/s]", mm.OMEGA, eng.omega_rev_s),
        ("A_p [m2]", mm.ACIL, eng.A_p), ("Vp [m/s]", mm.VP, eng.Vp),
        ("r [m]", mm.R_CRANK, eng.r_crank), ("R [-]", mm.R_ROD, eng.R),
        ("Vd [m3]", mm.VD, eng.Vd), ("mcomb [kg]", mm.MCOMB, eng.m_fuel),
        ("PCI [kJ/kg]", mm.PCI, eng.LHV), ("T1 [K]", mm.T1, eng.T1),
        ("Tw [K]", mm.TW, eng.Tw),
    ]
    ok = True
    for nome, ref, py in pares:
        ok &= check(3, nome, ref, py, RTOL_ALG * max(abs(ref), 1.0))
    print(f"   14 grandezas comparadas: {'OK' if ok else 'FALHOU'}")
    return ok


# =============================================================================
# Etapa 4 — geometria em 8 angulos + diferencas finitas
# =============================================================================
def etapa_geometria(engine, theta):
    print("== Etapa 4: geometria (8 angulos + DF) ==")
    Rc = 15.34
    ang8 = np.linspace(theta[0], theta[-1], 8)
    V_ref, V_py = mm.V(ang8, Rc), cylinder_volume(ang8, Rc, engine)
    dV_ref, dV_py = mm.dVdtheta(ang8, Rc), dV_dtheta(ang8, Rc, engine)
    y_ref, y_py = mm.y_piston(ang8), piston_disp(ang8, Rc, engine)
    As_ref, As_py = mm.As(ang8, Rc), heat_area(ang8, Rc, engine)
    ok = True
    ok &= check(4, "max|dV| V ref x py", 0.0,
                float(np.max(np.abs(V_ref - V_py))), ATOL_ALG)
    ok &= check(4, "max|dV| dV ref x py", 0.0,
                float(np.max(np.abs(dV_ref - dV_py))), ATOL_ALG)
    ok &= check(4, "max|dV| y ref x py", 0.0,
                float(np.max(np.abs(y_ref - y_py))), ATOL_ALG)
    ok &= check(4, "max|dV| As ref x py", 0.0,
                float(np.max(np.abs(As_ref - As_py))), ATOL_ALG)
    h = 1e-6
    dV_fd = (mm.V(ang8 + h, Rc) - mm.V(ang8 - h, Rc)) / (2 * h)
    err_fd = float(np.max(np.abs(dV_fd - dV_ref) / np.abs(dV_fd)))
    ok &= check(4, "DF central dV/dtheta (erro relativo)", 0.0, err_fd, 1e-6)
    # V no TDC = volume de folga = Vd/(Rc-1) (checagem fisica)
    ok &= check(4, "V(0)/Vd vs 1/(Rc-1)", 1.0 / (Rc - 1.0),
                float(mm.V(np.array([0.0]), Rc)[0] / mm.VD), 1e-12)
    print(f"   V, dV, y, As identicos (atol 1e-12); DF dV/dtheta "
          f"err rel = {err_fd:.2e}")
    return ok


# =============================================================================
# Etapa 5 — Double Wiebe
# =============================================================================
def etapa_wiebe():
    print("== Etapa 5: Double Wiebe ==")
    ok = True
    m1, t01, d1, m2, t02, d2, beta = (0.6, math.radians(-8.0),
                                      math.radians(15.0), 0.141,
                                      math.radians(2.83),
                                      math.radians(62.06), 0.063)
    th = np.linspace(-0.6, 0.6, 4001)
    x1r = mm.x1(th, m1, t01, d1)
    x2r = mm.x2(th, m2, t02, d2)
    xbr = mm.xb(th, m1, t01, d1, m2, t02, d2, beta)
    dx1r = mm.dx1dtheta(th, m1, t01, d1)
    dx2r = mm.dx2dtheta(th, m2, t02, d2)
    dxbr = mm.dxbdtheta(th, m1, t01, d1, m2, t02, d2, beta)

    wp = WiebeParameters(theta01=t01, delta1=d1, m1=m1, a1=mm.A1,
                         theta02=t02, delta2=d2, m2=m2, a2=mm.A2,
                         alpha=beta, mode="continuous")
    frac = double_burned_fraction(th, wp)
    ok &= check(5, "max|x1| ref x py", 0.0,
                float(np.max(np.abs(x1r - frac["x1"]))), ATOL_ALG)
    ok &= check(5, "max|x2| ref x py", 0.0,
                float(np.max(np.abs(x2r - frac["x2"]))), ATOL_ALG)
    ok &= check(5, "max|xb| ref x py", 0.0,
                float(np.max(np.abs(xbr - frac["xb"]))), ATOL_ALG)
    ok &= check(5, "max|dx1| ref x py", 0.0,
                float(np.max(np.abs(dx1r - frac["dx1"]))), 1e-12)
    ok &= check(5, "max|dx2| ref x py", 0.0,
                float(np.max(np.abs(dx2r - frac["dx2"]))), 1e-12)
    ok &= check(5, "max|dxb| ref x py", 0.0,
                float(np.max(np.abs(dxbr - frac["dxb"]))), 1e-12)

    # zeros antes do inicio de cada fase
    antes1 = th < t01
    antes2 = th < t02
    ok &= check(5, "x1 = 0 antes de theta01 (max)", 0.0,
                float(np.max(np.abs(frac["x1"][antes1]))), 0.0)
    ok &= check(5, "x2 = 0 antes de theta02 (max)", 0.0,
                float(np.max(np.abs(frac["x2"][antes2]))), 0.0)
    ok &= check(5, "dx1 = 0 antes de theta01 (max)", 0.0,
                float(np.max(np.abs(frac["dx1"][antes1]))), 0.0)
    ok &= check(5, "dx2 = 0 antes de theta02 (max)", 0.0,
                float(np.max(np.abs(frac["dx2"][antes2]))), 0.0)

    # derivadas por diferenca finita (interior das fases)
    h = 1e-6
    interior = (th > t01 + 0.05) & (th < t01 + d1 - 0.05)
    dx1_fd = (mm.x1(th[interior] + h, m1, t01, d1)
              - mm.x1(th[interior] - h, m1, t01, d1)) / (2 * h)
    err_fd1 = float(np.max(np.abs(dx1_fd - dx1r[interior])
                           / np.maximum(np.abs(dx1_fd), 1e-30)))
    ok &= check(5, "DF dx1/dtheta (erro relativo)", 0.0, err_fd1, 1e-6)
    interior2 = (th > t02 + 0.05) & (th < t02 + d2 - 0.05)
    dx2_fd = (mm.x2(th[interior2] + h, m2, t02, d2)
              - mm.x2(th[interior2] - h, m2, t02, d2)) / (2 * h)
    err_fd2 = float(np.max(np.abs(dx2_fd - dx2r[interior2])
                           / np.maximum(np.abs(dx2_fd), 1e-30)))
    ok &= check(5, "DF dx2/dtheta (erro relativo)", 0.0, err_fd2, 1e-6)

    # combinacao beta e reducao ao single
    ok &= check(5, "xb = beta*x1 + (1-beta)*x2 (max)", 0.0,
                float(np.max(np.abs(frac["xb"] - (beta * frac["x1"]
                                                  + (1 - beta) * frac["x2"])))),
                1e-14)
    w1 = WiebeParameters(theta01=t01, delta1=d1, m1=m1, a1=mm.A1,
                         theta02=t02, delta2=d2, m2=m2, a2=mm.A2,
                         alpha=1.0, mode="continuous")
    f1 = double_burned_fraction(th, w1)
    ok &= check(5, "beta=1 -> x = x1 (max)", 0.0,
                float(np.max(np.abs(f1["xb"] - f1["x1"]))), 1e-14)
    w0 = WiebeParameters(theta01=t01, delta1=d1, m1=m1, a1=mm.A1,
                         theta02=t02, delta2=d2, m2=m2, a2=mm.A2,
                         alpha=0.0, mode="continuous")
    f0 = double_burned_fraction(th, w0)
    ok &= check(5, "beta=0 -> x = x2 (max)", 0.0,
                float(np.max(np.abs(f0["xb"] - f0["x2"]))), 1e-14)

    # potencias fracionarias negativas: sem NaN, mask correta (m < 1)
    wsm = WiebeParameters(theta01=t01, delta1=d1, m1=0.3, a1=mm.A1,
                          theta02=t02, delta2=d2, m2=0.3, a2=mm.A2,
                          alpha=beta, mode="continuous")
    fsm = double_burned_fraction(th, wsm)
    nan_cnt = int(np.sum(~np.isfinite(fsm["xb"])))
    ok &= check(5, "NaN em z<0 com m=0.3 (contagem)", 0.0, float(nan_cnt), 0.0)
    print(f"   x1/x2/xb/dx identicos; DF dx1={err_fd1:.2e}, dx2={err_fd2:.2e}; "
          f"beta=1/0 -> single; sem NaN")
    return ok


# =============================================================================
# Etapa 6 — calor liberado
# =============================================================================
def etapa_calor(engine, theta):
    print("== Etapa 6: calor liberado Q = mcomb*PCI*x ==")
    ok = True
    check(6, "Q_total [kJ/ciclo]", mm.MCOMB * mm.PCI, engine.Q_total,
          RTOL_ALG * mm.MCOMB * mm.PCI)
    wp = WiebeParameters(theta01=math.radians(-8.0), delta1=math.radians(15.0),
                         m1=0.6, a1=mm.A1, theta02=math.radians(2.83),
                         delta2=math.radians(62.06), m2=0.141, a2=mm.A2,
                         alpha=0.063, mode="continuous")
    frac = double_burned_fraction(theta, wp)
    Q = engine.Q_total
    dQ1 = Q * wp.alpha * frac["dx1"]
    dQ2 = Q * (1.0 - wp.alpha) * frac["dx2"]
    dQ_tot = Q * frac["dxb"]
    ok &= check(6, "dQ1 + dQ2 = dQ_total (max)", 0.0,
                float(np.max(np.abs(dQ1 + dQ2 - dQ_tot))), 1e-12)
    # integral de dQ -> Q_total*x_inf = Q_total (cauda assintotica)
    Qint = float(np.trapezoid(dQ_tot, theta))
    ok &= check(6, "int dQ/dtheta (numérico) [kJ/ciclo]", Q, Qint, 1e-2)
    print(f"   Q_total = {Q:.9f} kJ/ciclo; conservacao e integral OK")
    return ok


# =============================================================================
# Etapa 7 — Hohenberg
# =============================================================================
def etapa_hohenberg(engine, theta):
    print("== Etapa 7: correlacao de Hohenberg ==")
    Rc = 15.34
    eng_h = EngineConfig(Rc=Rc)          # mesmo Rc dos dois lados
    P = np.linspace(138.2, 9000.0, 200)
    Tg = np.linspace(308.15, 2600.0, 200)
    sub = theta[:: max(1, theta.size // 200)][:200]
    h_py = hohenberg_h(sub, P, Tg, eng_h)
    Vs = mm.V(sub, Rc)
    h_ref = (130.0 * Vs ** (-0.06) * (P * 1e-2) ** 0.8 * Tg ** (-0.4)
             * (engine.Vp + 1.4) ** 0.8)
    ok = check(7, "max|h| ref x py", 0.0,
               float(np.max(np.abs(h_ref - h_py))), 1e-9)
    print(f"   h identico (max dif {np.max(np.abs(h_ref - h_py)):.2e})")
    return ok


# =============================================================================
# Etapas 8-11 — casos-ancora (MOD0d ref x pipeline Python)
# =============================================================================
def pipeline_erro(caso, theta, P_exp):
    """erro do notebook calculado sobre o P_sim do PIPELINE double_wiebe."""
    Rc, m1, t01, d1, m2, t02, d2, beta = caso
    eng = EngineConfig(Rc=float(Rc))
    wp = WiebeParameters(theta01=float(t01), delta1=float(d1), m1=float(m1),
                         a1=mm.A1, theta02=float(t02), delta2=float(d2),
                         m2=float(m2), a2=mm.A2, alpha=float(beta),
                         mode="continuous")
    sim = SimulationConfig(method="DOP853", rtol=1e-9, atol=1e-9)
    res = run_simulation(theta, P_exp, engine=eng, wiebe=wp, sim=sim)
    erro_nb = mm.erro_notebook(P_exp, res.P_sim)      # sqrt(SSres/(q-2))
    rmse_py = res.metrics["RMSE_kPa"]                 # sqrt(SSE/q)
    return res, erro_nb, rmse_py


def etapa_ancoras(theta, P_exp):
    print("== Etapas 8-11: casos-ancora ==")
    casos = [("A (caso obrigatorio)", CASE_A, CASE_A_ANCHOR),
             ("B (PSO ativo, k=190)", CASE_B, CASE_B_ANCHOR),
             ("C (vetor comentado, auxiliar)", CASE_C, CASE_C_ANCHOR)]
    resultados = {}
    ok_all = True
    ref_A = res_A_py = None
    for nome, caso, anchor in casos:
        ref = mm.mod0d(caso, theta, P_exp)
        res_py, erro_py, rmse_py = pipeline_erro(caso, theta, P_exp)
        if nome.startswith("A "):
            ref_A, res_A_py = ref, res_py
        d_ref = ref["erro"] - anchor
        d_py = erro_py - anchor
        ok = (abs(d_ref) <= TOL_ERRO) and (abs(d_py) <= TOL_ERRO)
        ok_all &= ok
        max_dP = float(np.max(np.abs(ref["P_sim"] - res_py.P_sim)))
        max_dTg = float(np.max(np.abs(ref["Tg"] - res_py.Tg)))
        rel_P = float(np.max(np.abs(ref["P_sim"] - res_py.P_sim)
                             / np.abs(ref["P_sim"])))
        CHECKS.append((8, f"{nome}: erro ref x ancora", anchor, ref["erro"],
                       TOL_ERRO, abs(d_ref) <= TOL_ERRO, "kPa"))
        CHECKS.append((8, f"{nome}: erro python x ancora", anchor, erro_py,
                       TOL_ERRO, abs(d_py) <= TOL_ERRO, "kPa"))
        CHECKS.append((8, f"{nome}: max|dP| ref x py", 0.0, max_dP, TOL_P,
                       max_dP <= TOL_P, "kPa"))
        CHECKS.append((8, f"{nome}: max|dTg| ref x py", 0.0, max_dTg, TOL_TG,
                       max_dTg <= TOL_TG, "K"))
        resultados[nome] = {
            "vetor": [float(v) for v in caso],
            "erro_ancora_mathematica": anchor,
            "erro_referencia_independente": ref["erro"],
            "erro_python_pipeline": erro_py,
            "erro_python_rmse_q": rmse_py,
            "delta_ref_ancora": d_ref,
            "delta_python_ancora": d_py,
            "max_dP_ref_vs_python_kPa": max_dP,
            "max_dTg_ref_vs_python_K": max_dTg,
            "erro_relativo_max_P": rel_P,
            "pass": bool(ok),
        }
        print(f"   {nome}: ancora={anchor:.3f}  ref={ref['erro']:.6f}  "
              f"python={erro_py:.6f}  d_ref={d_ref:+.6f}  d_py={d_py:+.6f}  "
              f"max|dP|={max_dP:.3e} kPa  max|dTg|={max_dTg:.3e} K  "
              f"{'OK' if ok else 'FALHOU'}")
    resultados["C (vetor comentado, auxiliar)"]["nota"] = (
        "Resultado 34.25858860927897 provem de celula COMENTADA do notebook "
        "(k=532), inconsistente com o MOD0d ativo: o mesmo vetor avaliado "
        "pelo MOD0d ativo (re-implementado) e pelo pipeline Python da "
        "100.5575/100.5574 kPa (diferenca entre as duas de 1.2e-4 kPa). "
        "Variants testados (calor desligado, beta no outro ramo, kp, a, Tw, "
        "T1, 503 pontos, permutacoes de ordem) nao reproduzem 34.2586. "
        "Evidencia de saida historica de estado anterior do notebook; teste "
        "auxiliar, nao faz parte dos criterios obrigatorios (A e B).")
    # convergencia do integrador (tolerancias 1e-9 / 1e-10 / 1e-11)
    ref9 = mm.mod0d(CASE_A, theta, P_exp, rtol=1e-9, atol=1e-9)["erro"]
    ref10 = mm.mod0d(CASE_A, theta, P_exp)["erro"]
    ref11 = mm.mod0d(CASE_A, theta, P_exp, rtol=1e-11, atol=1e-11)["erro"]
    conv = max(abs(ref9 - ref10), abs(ref10 - ref11))
    CHECKS.append((8, "convergencia rtol 1e-9/1e-10/1e-11 (max|d erro|)",
                   0.0, conv, 1e-3, conv <= 1e-3, "kPa"))
    print(f"   convergencia do integrador: max|d erro| = {conv:.3e} kPa")
    return resultados, ref_A, res_A_py, ok_all


# =============================================================================
# Etapa 9 — funcao de erro (q-2 do notebook x q do Python)
# =============================================================================
def etapa_funcao_erro(theta, P_exp, P_sim):
    print("== Etapa 9: funcao de erro ==")
    q = theta.size
    SSres = float(np.sum((P_exp - P_sim) ** 2))
    erro_nb = math.sqrt(SSres / (q - 2.0))
    rmse_py = math.sqrt(SSres / q)
    razao = erro_nb / rmse_py
    # o RMSE do pacote (metrics.rmse) deve bater com sqrt(SSE/q)
    from double_wiebe.metrics import rmse as rmse_pkg
    ok = check(9, "metrics.rmse (q) x sqrt(SSE/q)", rmse_py,
               float(rmse_pkg(P_exp, P_sim)), 1e-12)
    CHECKS.append((9, "razao erro(q-2)/rmse(q) = sqrt(q/(q-2))",
                   math.sqrt(q / (q - 2.0)), razao, 1e-12, True,
                   "divergencia documentada: notebook usa q-2"))
    print(f"   q={q}; erro_nb(q-2)={erro_nb:.6f}; rmse(q)={rmse_py:.6f}; "
          f"razao={razao:.9f} (esperado {math.sqrt(q/(q-2)):.9f})")
    return ok, erro_nb, rmse_py


# =============================================================================
# Etapa 12 — CSV ponto a ponto
# =============================================================================
def etapa_csv(theta, P_exp, ref, res_py):
    print("== Etapa 12: pointwise_comparison.csv ==")
    import csv as _csv
    th_deg = np.degrees(theta)
    dP = ref["P_sim"] - res_py.P_sim
    rel = dP / np.abs(ref["P_sim"]) * 100.0
    wp = CASE_A
    frac_py = double_burned_fraction(theta, WiebeParameters(
        theta01=wp[2], delta1=wp[3], m1=wp[1], a1=mm.A1,
        theta02=wp[5], delta2=wp[6], m2=wp[4], a2=mm.A2,
        alpha=wp[7], mode="continuous"))
    Rc = wp[0]
    cols = ["theta_rad", "theta_deg", "P_mathematica_kPa", "P_python_kPa",
            "erro_P_kPa", "erro_P_percentual", "Tg_K", "Qp_J",
            "x1", "x2", "x_total", "dx1", "dx2", "dx_total",
            "V_m3", "dV_dtheta_m3_rad", "As_m2"]
    with open(OUT / "pointwise_comparison.csv", "w", newline="",
              encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(cols)
        for i in range(theta.size):
            w.writerow([
                f"{theta[i]:.12g}", f"{np.degrees(theta[i]):.12g}",
                f"{ref['P_sim'][i]:.12g}", f"{res_py.P_sim[i]:.12g}",
                f"{dP[i]:.12g}", f"{rel[i]:.12g}",
                f"{ref['Tg'][i]:.12g}", f"{ref['Qp'][i]:.12g}",
                f"{frac_py['x1'][i]:.12g}", f"{frac_py['x2'][i]:.12g}",
                f"{frac_py['xb'][i]:.12g}", f"{frac_py['dx1'][i]:.12g}",
                f"{frac_py['dx2'][i]:.12g}", f"{frac_py['dxb'][i]:.12g}",
                f"{mm.V(theta[i], Rc):.12g}", f"{mm.dVdtheta(theta[i], Rc):.12g}",
                f"{mm.As(theta[i], Rc):.12g}",
            ])
    print(f"   {theta.size} linhas gravadas (caso A, obrigatorio)")
    return float(np.max(np.abs(dP))), float(np.max(np.abs(rel)))


# =============================================================================
# Etapa 13 — graficos
# =============================================================================
def etapa_graficos(theta, P_exp, ref, res_py):
    print("== Etapa 13: graficos ==")
    rad = theta
    # pressao
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(rad, P_exp, "+", color="red", markersize=4, label="Experimental")
    ax.plot(rad, ref["P_sim"], "--", color="black", linewidth=1.4,
            label="Mathematica (MOD0d)")
    ax.plot(rad, res_py.P_sim, "-", color="green", linewidth=1.2,
            label="Python (double_wiebe)")
    ax.set_xlabel("theta [rad]")
    ax.set_ylabel("Pressao [kPa]")
    ax.set_title("Pressao: experimental x Mathematica x Python (caso A)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "validation_pressure.png", dpi=150)
    plt.close(fig)
    # residuos
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(rad, P_exp - ref["P_sim"], "--", color="black", linewidth=1.2,
            label="Residuo Mathematica (P_exp - P_math)")
    ax.plot(rad, P_exp - res_py.P_sim, "-", color="green", linewidth=1.1,
            label="Residuo Python (P_exp - P_python)")
    ax.plot(rad, res_py.P_sim - ref["P_sim"], "-", color="blue",
            linewidth=1.0, label="Diferenca Python - Mathematica")
    ax.set_xlabel("theta [rad]")
    ax.set_ylabel("Residuo [kPa]")
    ax.set_title("Residuos (caso A)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "validation_residuals.png", dpi=150)
    plt.close(fig)
    # wiebe
    wp = CASE_A
    frac = double_burned_fraction(theta, WiebeParameters(
        theta01=wp[2], delta1=wp[3], m1=wp[1], a1=mm.A1,
        theta02=wp[5], delta2=wp[6], m2=wp[4], a2=mm.A2,
        alpha=wp[7], mode="continuous"))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(theta, mm.x1(theta, wp[1], wp[2], wp[3]), "--", color="black",
            linewidth=1.4, label="x1 Mathematica")
    ax.plot(theta, mm.x2(theta, wp[4], wp[5], wp[6]), "--", color="gray",
            linewidth=1.4, label="x2 Mathematica")
    ax.plot(theta, mm.xb(theta, wp[1], wp[2], wp[3], wp[4], wp[5], wp[6],
                         wp[7]), "--", color="black", linewidth=2.2,
            label="x total Mathematica")
    ax.plot(theta, frac["x1"], "-", color="green", linewidth=1.0,
            label="x1 Python")
    ax.plot(theta, frac["x2"], "-", color="limegreen", linewidth=1.0,
            label="x2 Python")
    ax.plot(theta, frac["xb"], "-", color="green", linewidth=1.8,
            label="x total Python")
    ax.set_xlabel("theta [rad]")
    ax.set_ylabel("Fracao queimada [-]")
    ax.set_title("Double Wiebe: Mathematica x Python (caso A)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "validation_wiebe.png", dpi=150)
    plt.close(fig)
    print("   3 PNGs gravados")


# =============================================================================
def main():
    print("=" * 72)
    print("VALIDACAO DE PARIDADE: Mathematica MOD0d x double_wiebe (Python)")
    print("=" * 72)
    theta, P_exp, dados = etapa_dados()
    etapa_constantes()
    engine = EngineConfig()
    etapa_geometria(engine, theta)
    etapa_wiebe()
    etapa_calor(engine, theta)
    etapa_hohenberg(engine, theta)
    ancoras, ref_A, res_A_py, ok_ancoras = etapa_ancoras(theta, P_exp)
    ok_err, erro_nb, rmse_py = etapa_funcao_erro(theta, P_exp, ref_A["P_sim"])
    max_dP, max_rel = etapa_csv(theta, P_exp, ref_A, res_A_py)
    etapa_graficos(theta, P_exp, ref_A, res_A_py)

    # matriz de rastreabilidade (CSV) e metricas (JSON)
    CHECKS.append((12, "max|dP| ponto a ponto (459)", 0.0, max_dP, TOL_P,
                   max_dP <= TOL_P, "kPa"))
    metricas = {
        "dados": dados,
        "tolerancias": {"erro_global_kPa": TOL_ERRO, "dP_kPa": TOL_P,
                        "dTg_K": TOL_TG, "erro_relativo_P": TOL_REL,
                        "rtol_algebrico": RTOL_ALG, "atol_algebrico": ATOL_ALG},
        "casos_ancora": ancoras,
        "funcao_erro": {
            "notebook": "sqrt(SSres/(q-2))",
            "python_metrics_rmse": "sqrt(SSE/q)",
            "q": int(theta.size),
            "erro_notebook_casoA": erro_nb,
            "rmse_python_casoA": rmse_py,
            "razao": erro_nb / rmse_py,
        },
        "ponto_a_ponto_casoA": {
            "max_dP_kPa": max_dP, "max_dP_percentual": max_rel,
            "max_dTg_K": float(np.max(np.abs(ref_A["Tg"] - res_A_py.Tg))),
        },
        "checks": [
            {"etapa": c[0], "item": c[1], "ref": c[2], "py": c[3],
             "tol": c[4], "ok": bool(c[5]), "obs": c[6]}
            for c in CHECKS
        ],
        "n_checks": len(CHECKS),
        "n_falhas": int(sum(1 for c in CHECKS if not c[5])),
        "limitacao": ("Sem Wolfram Engine no ambiente: a coluna "
                      "'Mathematica' ponto a ponto vem de uma "
                      "re-implementacao independente das equacoes extraidas "
                      "do notebook; a validacao externa usa as ancoras "
                      "numericas 100.538 / 55.381 / 34.2586 kPa."),
    }
    with open(OUT / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    falhas = [c for c in CHECKS if not c[5]]
    print("-" * 72)
    print(f"CHECKS: {len(CHECKS)} | falhas: {len(falhas)}")
    for c in falhas:
        print(f"  FALHA etapa {c[0]}: {c[1]}  ref={c[2]} py={c[3]} tol={c[4]}")
    return 0 if not falhas else 1


if __name__ == "__main__":
    sys.exit(main())