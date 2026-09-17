# -*- coding: utf-8 -*-
"""
wiebe.py
========
Funções de Wiebe do modelo Double Wiebe — duas fases ponderadas por alpha.

Fase j (j = 1 pré-misturada, j = 2 difusão):

    z_j = (theta - theta0_j) / delta_j

    theta <  theta0_j :  x_j = 0,  dx_j/dtheta = 0
    theta >= theta0_j :
        x_j        = 1 - exp[-a_j * z_j^(m_j + 1)]
        dx_j/dtheta = a_j*(m_j + 1)/delta_j * z_j^m_j * exp[-a_j*z_j^(m_j+1)]

Combinação:

    x_b        = alpha*x_1 + (1-alpha)*x_2
    dx_b/dtheta = alpha*dx_1 + (1-alpha)*dx_2

MODOS
-----
* ``continuous``: cada expressão continua após theta0 + delta (cauda
  assintótica, como no modelo Single Wiebe original).
* ``bounded``: após theta0 + delta a fase é mantida constante
  (x_j = 1 - exp(-a_j)) e sua derivada é zero (combustão encerrada).

Garantias numéricas: a soma ponderada de funções monotônicas não decrescentes
com pesos em [0, 1] é monotônica não decrescente; o resultado é ainda
satisfado (clipado) ao intervalo [0, 1] para eliminação de ruído de ponto
flutuante. Com m_j >= 0 (validado em WiebeParameters), 0^0 = 1 é tratado
corretamente e não há singularidades.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .models import WiebeParameters, MODE_BOUNDED


def phase_burned_fraction(
    theta,
    theta0: float,
    delta: float,
    m: float,
    a: float,
    bounded: bool = False,
):
    """Fração queimada de UMA fase e sua derivada.

    Retorna (x_j, dx_j/dtheta), arrays com o mesmo shape de ``theta``.

    No modo ``bounded``, para theta > theta0 + delta devolve a fase
    constante x_j = 1 - exp(-a) (valor no fim da duração) e derivada nula;
    há um pequeno degrau numérico em theta0 + delta, mas a fase permanece
    monotônica (o valor no fim do intervalo é exatamente o platô).
    """
    theta = np.asarray(theta, dtype=float)
    active = theta >= theta0
    z = (theta - theta0) / delta
    z_safe = np.where(active, np.maximum(z, 0.0), 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        zn = z_safe ** m                    # z^m   (0**0 = 1)
        znp1 = z_safe ** (m + 1.0)          # z^(m+1)
        expo = np.exp(-a * znp1)            # sobrefluxo -> 0.0 é aceitável
        x = np.where(active, 1.0 - expo, 0.0)
        dx = np.where(active, a * (m + 1.0) / delta * zn * expo, 0.0)

    if bounded:
        x_end = 1.0 - np.exp(-a)            # platô após theta0 + delta
        beyond = theta > theta0 + delta
        x = np.where(beyond, x_end, x)
        dx = np.where(beyond, 0.0, dx)

    # Satisfação numérica do intervalo físico (elimina -0.0 / 1.0000000002)
    x = np.clip(x, 0.0, 1.0)
    return x, dx


def double_burned_fraction(theta, params: WiebeParameters) -> Dict[str, np.ndarray]:
    """Avalia as duas fases e a combinação ponderada.

    Retorna um dicionário com:

        x1, x2, xb          frações queimadas (fase 1, fase 2, total)
        dx1, dx2, dxb       derivadas d/dtheta correspondentes [1/rad]
    """
    bounded = params.mode == MODE_BOUNDED
    x1, dx1 = phase_burned_fraction(
        theta, params.theta01, params.delta1, params.m1, params.a1, bounded)
    x2, dx2 = phase_burned_fraction(
        theta, params.theta02, params.delta2, params.m2, params.a2, bounded)
    xb = params.alpha * x1 + (1.0 - params.alpha) * x2
    dxb = params.alpha * dx1 + (1.0 - params.alpha) * dx2
    # Monotonicidade garantida: clip do intervalo físico (a soma ponderada de
    # fases monotônicas já é monotônica; o clip elimina apenas ruído de ponto
    # flutuante).
    xb = np.clip(xb, 0.0, 1.0)
    return {
        "x1": x1, "x2": x2, "xb": xb,
        "dx1": dx1, "dx2": dx2, "dxb": dxb,
    }