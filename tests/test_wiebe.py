# -*- coding: utf-8 -*-
"""
test_wiebe.py
=============
Funções de Wiebe: comportamento analítico das fases, combinação ponderada
pelo alpha, modos continuous/bounded e redução ao Single Wiebe.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from double_wiebe.models import WiebeParameters
from double_wiebe.wiebe import double_burned_fraction, phase_burned_fraction


THETA = np.linspace(-2.0, 3.0, 501)


# =============================================================================
# Fase individual
# =============================================================================
def test_fase_zero_antes_do_inicio():
    x, dx = phase_burned_fraction(THETA, theta0=0.5, delta=1.0, m=0.5,
                                  a=6.9078)
    assert np.all(x[THETA < 0.5] == 0.0)
    assert np.all(dx[THETA < 0.5] == 0.0)


def test_fase_monotona_e_limitada():
    x, dx = phase_burned_fraction(THETA, theta0=-0.1, delta=1.0, m=1.0,
                                  a=6.9078)
    assert np.all(np.diff(x) >= -1e-12)          # monotônica não decrescente
    assert np.all(x >= 0.0) and np.all(x <= 1.0)


def test_fase_no_inicio_exige_zero():
    # em theta = theta0, z = 0 ⇒ x = 0 (para m > 0)
    x, _ = phase_burned_fraction(np.array([0.0]), theta0=0.0, delta=1.0,
                                 m=1.0, a=6.9078)
    assert x[0] == pytest.approx(0.0, abs=1e-12)


def test_fase_contínua_atinge_1_assintoticamente():
    # em z = 1 (theta0 + delta): x = 1 - exp(-a); em z grande: x -> 1
    t = np.array([1.0, 10.0])
    x, _ = phase_burned_fraction(t, theta0=0.0, delta=1.0, m=1.0, a=6.9078)
    assert x[0] == pytest.approx(1.0 - math.exp(-6.9078), rel=1e-9)
    assert x[1] == pytest.approx(1.0, abs=1e-12)


def test_modo_bounded_tem_plato():
    # continuous continua crescendo; bounded satura em 1 - exp(-a)
    theta = np.array([0.5 + 1.0, 0.5 + 5.0])     # além de theta0 + delta
    xc, _ = phase_burned_fraction(theta, theta0=0.5, delta=1.0, m=1.0,
                                  a=6.9078, bounded=False)
    xb, dxb = phase_burned_fraction(theta, theta0=0.5, delta=1.0, m=1.0,
                                    a=6.9078, bounded=True)
    assert xb[0] == pytest.approx(xb[1], rel=1e-12)          # plateau
    assert dxb[1] == 0.0                                     # derivada nula
    assert xb[1] == pytest.approx(1.0 - math.exp(-6.9078), rel=1e-9)
    assert xc[1] > xb[1]                                     # contínuo cresce


# =============================================================================
# Derivada analítica × numérica
# =============================================================================
def test_derivada_analitica_concorda_com_numerica():
    th0, delta, m, a = 0.2, 0.8, 1.2, 6.9078
    h = 1e-5
    # pontos interiores (a diferença finita no contorno θ0 é meia-derivada)
    t = np.linspace(th0 + 20 * h, th0 + delta, 200)
    w = WiebeParameters(theta01=th0, delta1=delta, m1=m,
                        theta02=th0 + 10 * delta, delta2=delta, m2=m,
                        alpha=1.0)
    dx = double_burned_fraction(t, w)["dx1"]
    x_p, _ = phase_burned_fraction(t + h, th0, delta, m, 6.9078)
    x_m, _ = phase_burned_fraction(t - h, th0, delta, m, 6.9078)
    x_num = (x_p - x_m) / (2 * h)
    # tolerância de 2ª ordem em h e malha
    assert np.allclose(dx, x_num, rtol=1e-4, atol=1e-6)


# =============================================================================
# Double Wiebe: combinação ponderada
# =============================================================================
def test_combinacao_ponderada():
    w = WiebeParameters(alpha=0.30)
    frac = double_burned_fraction(THETA, w)
    esperado = 0.30 * frac["x1"] + 0.70 * frac["x2"]
    assert np.allclose(frac["xb"], esperado, atol=1e-12)


def test_alpha_zero_ou_um_reduz_a_uma_fase():
    w = WiebeParameters(alpha=1.0)
    frac = double_burned_fraction(THETA, w)
    assert np.allclose(frac["xb"], frac["x1"], atol=1e-12)
    w = WiebeParameters(alpha=0.0)
    frac = double_burned_fraction(THETA, w)
    assert np.allclose(frac["xb"], frac["x2"], atol=1e-12)


def test_reducao_ao_single_wiebe():
    """alpha = 1 (ou 0) com as duas fases idênticas reproduz a Single Wiebe:
    x = 1 - exp(-a * z^(m+1)), com x(θ0) = 0 e x monotônica."""
    w = WiebeParameters(theta01=-0.2, delta1=1.2, m1=0.5, a1=6.9078,
                        theta02=-0.2, delta2=1.2, m2=0.5, a2=6.9078,
                        alpha=1.0)
    frac = double_burned_fraction(THETA, w)
    z = (THETA - w.theta01) / w.delta1
    ref = np.where(z > 0, 1.0 - np.exp(-w.a1 * np.maximum(z, 0.0) ** 1.5), 0.0)
    assert np.allclose(frac["xb"], ref, atol=1e-12)


def test_xb_monotona_e_em_0_1():
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        w = WiebeParameters(alpha=alpha)
        frac = double_burned_fraction(THETA, w)
        assert np.all(np.diff(frac["xb"]) >= -1e-12)
        assert np.all(frac["xb"] >= 0.0) and np.all(frac["xb"] <= 1.0)


def test_validacoes():
    erros = WiebeParameters(delta1=-1.0).validate()
    assert erros and "delta1" in erros[0]
    assert WiebeParameters(alpha=1.5).validate()
    assert WiebeParameters(theta02=-5.0, theta01=0.0).validate()
    assert WiebeParameters().validate() == []