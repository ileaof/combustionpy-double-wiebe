# -*- coding: utf-8 -*-
"""
test_mathematica_parity.py
==========================
Testes de paridade entre o MOD0d do notebook Mathematica
(``Modelo_Double_Wiebe_v2.nb``) e o pacote Python ``double_wiebe``.

A referência é uma re-implementação INDEPENDENTE das equações extraídas do
notebook (``validation/mathematica_reference.py``), ancorada nos erros
globais produzidos pelo próprio Mathematica:

  - caso obrigatório: erro = 100.538 kPa  (tolerância ±0.5 kPa)
  - vetor ativo do PSO (k=190): erro = 55.381 kPa (tolerância ±0.5 kPa)
  - vetor comentado (k=532, célula comentada/histórica): 34.25858860927897
    NÃO é reproduzível pelo MOD0d ativo (ver validation_report.md) — o teste
    correspondente exige apenas que referência e Python coincidam.

Limitação declarada: sem Wolfram Engine no ambiente, a comparação ponto a
ponto usa a re-implementação independente (mesmas equações + DOP853).
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))

import mathematica_reference as mm  # noqa: E402
from double_wiebe.geometry import (  # noqa: E402
    cylinder_volume, dV_dtheta, heat_area, piston_disp)
from double_wiebe.metrics import rmse as rmse_pkg  # noqa: E402
from double_wiebe.models import EngineConfig, SimulationConfig, WiebeParameters  # noqa: E402
from double_wiebe.simulation import run_simulation  # noqa: E402
from double_wiebe.thermodynamics import hohenberg_h  # noqa: E402
from double_wiebe.wiebe import double_burned_fraction  # noqa: E402

DATA_PATH = Path(os.environ.get(
    "DW_PEXP",
    str(ROOT.parent / "P_exp-Carga-3_45%.txt")))

CASE_A = [15.34, 0.6, math.radians(-8.0), math.radians(15.0),
          0.141, math.radians(2.83), math.radians(62.06), 0.063]
CASE_A_ANCHOR = 100.538
CASE_B = [15.3496, 0.6, -0.193038, 0.436332,
          0.9, 0.0317214, 0.523599, 0.075285]
CASE_B_ANCHOR = 55.381
CASE_C = [15.340724668321869, 0.5999990714343, -0.13962634015954636,
          0.2617993877991494, 0.14162900274166848, 0.04937752133906362,
          1.082970478069876, 0.06333317846901508]
TOL_ERRO = 0.5
TOL_P = 1.0
TOL_TG = 1.0

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(), reason=f"dado experimental ausente: {DATA_PATH}")


def _pipeline(caso, theta, P_exp):
    Rc, m1, t01, d1, m2, t02, d2, beta = caso
    eng = EngineConfig(Rc=float(Rc))
    wp = WiebeParameters(theta01=float(t01), delta1=float(d1), m1=float(m1),
                         a1=mm.A1, theta02=float(t02), delta2=float(d2),
                         m2=float(m2), a2=mm.A2, alpha=float(beta),
                         mode="continuous")
    sim = SimulationConfig(method="DOP853", rtol=1e-9, atol=1e-9)
    res = run_simulation(theta, P_exp, engine=eng, wiebe=wp, sim=sim)
    return res, mm.erro_notebook(P_exp, res.P_sim)


@pytest.fixture(scope="module")
def dados():
    _, _, theta, P = mm.load_and_filter(str(DATA_PATH))
    return theta, P


# ---------------------------------------------------------------------------
# Etapa 2 — dados
# ---------------------------------------------------------------------------
def test_dados_filtro_459(dados):
    theta, _ = dados
    assert theta.size == 459
    assert abs(theta[0] - (-1.998401994)) < 1e-9
    assert abs(theta[-1] - 1.998401994) < 1e-9
    assert abs(float(np.mean(np.diff(theta))) - 0.008726646) < 1e-9


def test_dados_P1_bar_para_kPa(dados):
    _, P = dados
    assert abs(P[0] - 138.2) < 1e-9            # 1.382 bar * 100


# ---------------------------------------------------------------------------
# Etapas 3-4 — constantes e geometria
# ---------------------------------------------------------------------------
def test_constantes_motor():
    eng = EngineConfig()
    assert eng.bore == mm.D
    assert eng.stroke == mm.S
    assert eng.rod_length == mm.L
    assert eng.kappa == mm.KP
    assert eng.omega_rev_s == pytest.approx(mm.OMEGA, rel=1e-12)
    assert eng.A_p == pytest.approx(mm.ACIL, rel=1e-12)
    assert eng.Vp == pytest.approx(mm.VP, rel=1e-12)
    assert eng.Vd == pytest.approx(mm.VD, rel=1e-12)
    assert eng.m_fuel == mm.MCOMB
    assert eng.LHV == mm.PCI
    assert eng.T1 == mm.T1
    assert eng.Tw == mm.TW


def test_geometria_8_angulos(dados):
    theta, _ = dados
    Rc = 15.34
    ang = np.linspace(theta[0], theta[-1], 8)
    assert np.max(np.abs(cylinder_volume(ang, Rc, EngineConfig(Rc=Rc))
                         - mm.V(ang, Rc))) <= 1e-12
    assert np.max(np.abs(dV_dtheta(ang, Rc, EngineConfig(Rc=Rc))
                         - mm.dVdtheta(ang, Rc))) <= 1e-12
    assert np.max(np.abs(piston_disp(ang, Rc, EngineConfig(Rc=Rc))
                         - mm.y_piston(ang))) <= 1e-12
    assert np.max(np.abs(heat_area(ang, Rc, EngineConfig(Rc=Rc))
                         - mm.As(ang, Rc))) <= 1e-12


def test_geometria_diferencas_finitas():
    Rc = 15.34
    h = 1e-6
    ang = np.linspace(-1.5, 1.5, 17)
    ang = ang[np.abs(mm.dVdtheta(ang, Rc)) > 1e-8]   # exclui dV ~ 0 (theta=0)
    fd = (mm.V(ang + h, Rc) - mm.V(ang - h, Rc)) / (2 * h)
    err = np.max(np.abs(fd - mm.dVdtheta(ang, Rc)) / np.abs(fd))
    assert err < 1e-6
    # V no TDC = volume de folga = Vd/(Rc-1)
    assert mm.V(np.array([0.0]), Rc)[0] == pytest.approx(
        mm.VD / (Rc - 1.0), rel=1e-12)


# ---------------------------------------------------------------------------
# Etapa 5 — Double Wiebe
# ---------------------------------------------------------------------------
def test_wiebe_zeros_antes_do_inicio():
    t01, d1 = math.radians(-8.0), math.radians(15.0)
    t02, d2 = math.radians(2.83), math.radians(62.06)
    wp = WiebeParameters(theta01=t01, delta1=d1, m1=0.6, a1=mm.A1,
                         theta02=t02, delta2=d2, m2=0.141, a2=mm.A2,
                         alpha=0.063, mode="continuous")
    th = np.linspace(-0.5, 0.6, 2000)
    frac = double_burned_fraction(th, wp)
    assert np.max(np.abs(frac["x1"][th < t01])) == 0.0
    assert np.max(np.abs(frac["x2"][th < t02])) == 0.0
    assert np.max(np.abs(frac["dx1"][th < t01])) == 0.0
    assert np.max(np.abs(frac["dx2"][th < t02])) == 0.0


def test_wiebe_derivadas_diferencas_finitas():
    t01, d1 = math.radians(-8.0), math.radians(15.0)
    t02, d2 = math.radians(2.83), math.radians(62.06)
    wp = WiebeParameters(theta01=t01, delta1=d1, m1=0.6, a1=mm.A1,
                         theta02=t02, delta2=d2, m2=0.141, a2=mm.A2,
                         alpha=0.063, mode="continuous")
    th = np.linspace(-0.5, 0.6, 4001)
    frac = double_burned_fraction(th, wp)
    h = 1e-6
    for t0, d, dx, x, m in ((t01, d1, frac["dx1"], mm.x1, 0.6),
                            (t02, d2, frac["dx2"], mm.x2, 0.141)):
        # interior da fase com taxa significativa (a cauda assintótica tem
        # dx -> 0 e o erro relativo da DF explode por ruído de ponto flutuante)
        interior = (th > t0 + 0.05) & (th < t0 + d - 0.05) & (dx > 1e-6)
        fd = (x(th[interior] + h, m, t0, d)
              - x(th[interior] - h, m, t0, d)) / (2 * h)
        err = np.max(np.abs(fd - dx[interior])
                     / np.maximum(np.abs(fd), 1e-30))
        assert err < 1e-6


def test_wiebe_beta_e_reducao_single():
    t01, d1 = math.radians(-8.0), math.radians(15.0)
    t02, d2 = math.radians(2.83), math.radians(62.06)
    beta = 0.063
    th = np.linspace(-0.5, 0.6, 500)
    wp = WiebeParameters(theta01=t01, delta1=d1, m1=0.6, a1=mm.A1,
                         theta02=t02, delta2=d2, m2=0.141, a2=mm.A2,
                         alpha=beta, mode="continuous")
    frac = double_burned_fraction(th, wp)
    assert np.max(np.abs(frac["xb"]
                         - (beta * frac["x1"] + (1 - beta) * frac["x2"]))) \
        <= 1e-14
    wp1 = WiebeParameters(theta01=t01, delta1=d1, m1=0.6, a1=mm.A1,
                          theta02=t02, delta2=d2, m2=0.141, a2=mm.A2,
                          alpha=1.0, mode="continuous")
    assert np.max(np.abs(double_burned_fraction(th, wp1)["xb"]
                         - double_burned_fraction(th, wp1)["x1"])) <= 1e-14
    wp0 = WiebeParameters(theta01=t01, delta1=d1, m1=0.6, a1=mm.A1,
                          theta02=t02, delta2=d2, m2=0.141, a2=mm.A2,
                          alpha=0.0, mode="continuous")
    assert np.max(np.abs(double_burned_fraction(th, wp0)["xb"]
                         - double_burned_fraction(th, wp0)["x2"])) <= 1e-14


def test_wiebe_potencias_fracionarias_sem_nan():
    t01 = math.radians(-8.0)
    wp = WiebeParameters(theta01=t01, delta1=math.radians(15.0), m1=0.3,
                         a1=mm.A1, theta02=math.radians(2.83),
                         delta2=math.radians(62.06), m2=0.3, a2=mm.A2,
                         alpha=0.063, mode="continuous")
    th = np.linspace(-0.5, 0.6, 2000)
    frac = double_burned_fraction(th, wp)
    assert np.all(np.isfinite(frac["xb"])) and np.all(np.isfinite(frac["dxb"]))


# ---------------------------------------------------------------------------
# Etapas 6-7 — calor e Hohenberg
# ---------------------------------------------------------------------------
def test_hohenberg_identicos(dados):
    theta, _ = dados
    sub = theta[::45]
    P = np.linspace(138.2, 9000.0, sub.size)
    Tg = np.linspace(308.15, 2600.0, sub.size)
    eng = EngineConfig(Rc=15.34)
    h_py = hohenberg_h(sub, P, Tg, eng)
    h_ref = (130.0 * mm.V(sub, 15.34) ** (-0.06) * (P * 1e-2) ** 0.8
             * Tg ** (-0.4) * (eng.Vp + 1.4) ** 0.8)
    assert np.max(np.abs(h_ref - h_py)) <= 1e-9


# ---------------------------------------------------------------------------
# Etapas 8-11 — casos-âncora
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_ancora_caso_obrigatorio_100_538(dados):
    theta, P = dados
    ref = mm.mod0d(CASE_A, theta, P)
    assert abs(ref["erro"] - CASE_A_ANCHOR) <= TOL_ERRO
    _, erro_py = _pipeline(CASE_A, theta, P)
    assert abs(erro_py - CASE_A_ANCHOR) <= TOL_ERRO


@pytest.mark.slow
def test_ancora_pso_ativo_55_381(dados):
    theta, P = dados
    ref = mm.mod0d(CASE_B, theta, P)
    assert abs(ref["erro"] - CASE_B_ANCHOR) <= TOL_ERRO
    _, erro_py = _pipeline(CASE_B, theta, P)
    assert abs(erro_py - CASE_B_ANCHOR) <= TOL_ERRO


@pytest.mark.slow
def test_ponto_a_ponto_caso_A(dados):
    theta, P = dados
    ref = mm.mod0d(CASE_A, theta, P)
    res, _ = _pipeline(CASE_A, theta, P)
    assert np.max(np.abs(ref["P_sim"] - res.P_sim)) <= TOL_P
    assert np.max(np.abs(ref["Tg"] - res.Tg)) <= TOL_TG
    assert np.max(np.abs(ref["P_sim"] - res.P_sim) / np.abs(ref["P_sim"])) \
        <= 1e-3


@pytest.mark.slow
def test_vetor_comentado_ref_e_python_coincidem(dados):
    """O vetor comentado (k=532) NÃO reproduz 34.2586 com o MOD0d ativo
    (saída histórica de célula comentada — ver validation_report.md).
    Exige-se apenas que a referência independente e o Python coincidam."""
    theta, P = dados
    ref = mm.mod0d(CASE_C, theta, P)
    _, erro_py = _pipeline(CASE_C, theta, P)
    assert abs(ref["erro"] - erro_py) <= 1e-3
    assert abs(ref["erro"] - 34.25858860927897) > 0.5   # ressalva documentada


# ---------------------------------------------------------------------------
# Etapa 9 — função de erro (q-2 preservado)
# ---------------------------------------------------------------------------
def test_funcao_erro_divisor_q_menos_2(dados):
    theta, P = dados
    q = theta.size
    # solução sintética próxima do experimental: SSE conhecido
    P_sim = P * (1.0 + 1e-3)
    SSres = float(np.sum((P - P_sim) ** 2))
    erro = mm.erro_notebook(P, P_sim)
    assert erro == pytest.approx(math.sqrt(SSres / (q - 2.0)), rel=1e-12)
    # divergência documentada: RMSE do pacote usa divisor q
    assert rmse_pkg(P, P_sim) == pytest.approx(math.sqrt(SSres / q),
                                               rel=1e-12)
    assert erro / rmse_pkg(P, P_sim) == pytest.approx(
        math.sqrt(q / (q - 2.0)), rel=1e-12)