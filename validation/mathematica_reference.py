# -*- coding: utf-8 -*-
"""
mathematica_reference.py
========================
Re-implementação INDEPENDENTE do MOD0d do notebook
``Modelo_Double_Wiebe_v2.nb``, digitada diretamente das expressões extraídas
do notebook/PDF (sem reutilizar o código do pacote double_wiebe).

Serve de referência para o teste de paridade: se este módulo (equações do
Mathematica + solve_ivp DOP853) e o pacote double_wiebe coincidirem, e os
erros globais ancorarem nos valores do Mathematica (100.538 e 55.381 kPa),
o Python reproduz o notebook.

Ordem de parâmetros do notebook (MOD0d):
    Rc, m1, theta01, delta_theta1, m2, theta02, delta_theta2, beta
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

# Constantes do notebook (MOD0d)
D = 86.0 / 1000.0          # diâmetro do cilindro [m]
S = 70.0 / 1000.0          # curso do pistão [m]
N_CYL = 1.0                # número de cilindros
KP = 1.37                  # kappa [-]
L = 117.5 / 1000.0         # comprimento da biela [m]
OMEGA = 3396.20 / 60.0     # rotação [rev/s]
ACIL = np.pi * (D / 2.0) ** 2
VP = 2.0 * S * OMEGA
R_CRANK = S / 2.0
R_ROD = L / R_CRANK
VD = ACIL * S
MCOMB = 9.42754647351e-6
PCI = 39191.3
A1 = 6.9078
A2 = 6.9078
T1 = 273.15 + 35.0
TW = 440.0


# ---------------------------------------------------------------------------
# Dados experimentais (como no notebook: Import[Table] -> extrairIntervalo)
# ---------------------------------------------------------------------------
def load_and_filter(path: str):
    """tablePall -> tablePbar -> tableP (theta rad, P kPa = bar*10^2)."""
    raw = np.loadtxt(path, comments="#")
    raw = raw[np.isfinite(raw).all(axis=1)]
    theta_all, p_bar_all = raw[:, 0], raw[:, 1]
    # extrairIntervalo: -2 <= theta <= 2 (limite INCLUSIVE no notebook)
    sel = (theta_all >= -2.0) & (theta_all <= 2.0)
    theta = theta_all[sel]
    p_bar = p_bar_all[sel]
    order = np.argsort(theta)
    theta, p_bar = theta[order], p_bar[order]
    p_kpa = p_bar * 1e2
    return theta_all, p_bar_all, theta, p_kpa


# ---------------------------------------------------------------------------
# Geometria (notebook, verbatim)
# ---------------------------------------------------------------------------
def V(theta, Rc):
    return VD / (Rc - 1.0) + (VD / 2.0) * (
        R_ROD + 1.0 - np.cos(theta) - np.sqrt(R_ROD**2 - np.sin(theta) ** 2))


def dVdtheta(theta, Rc):
    return (VD * np.sin(theta) / 2.0) * (
        1.0 + np.cos(theta) / np.sqrt(R_ROD**2 - np.sin(theta) ** 2))


def y_piston(theta):
    return (L + R_CRANK - R_CRANK * np.cos(theta)
            - np.sqrt(L**2 - R_CRANK**2 * np.sin(theta) ** 2))


def As(theta, Rc):
    return (2.0 * np.pi * (D / 2.0) ** 2
            + np.pi * D * y_piston(theta)
            + np.pi * D * S / (Rc - 1.0))


# ---------------------------------------------------------------------------
# Double Wiebe (notebook, verbatim: If[theta >= theta0, ...], sem corte)
# ---------------------------------------------------------------------------
def x1(theta, m1, theta01, d1):
    z = (theta - theta01) / d1
    return np.where(theta >= theta01,
                    1.0 - np.exp(-A1 * np.power(np.maximum(z, 0.0), m1 + 1.0)),
                    0.0)


def x2(theta, m2, theta02, d2):
    z = (theta - theta02) / d2
    return np.where(theta >= theta02,
                    1.0 - np.exp(-A2 * np.power(np.maximum(z, 0.0), m2 + 1.0)),
                    0.0)


def xb(theta, m1, theta01, d1, m2, theta02, d2, beta):
    return beta * x1(theta, m1, theta01, d1) \
        + (1.0 - beta) * x2(theta, m2, theta02, d2)


def dx1dtheta(theta, m1, theta01, d1):
    out = np.zeros_like(np.asarray(theta, dtype=float))
    th = np.asarray(theta, dtype=float)
    act = th >= theta01
    z = (th[act] - theta01) / d1
    out[act] = (A1 * (m1 + 1.0) / d1 * np.power(z, m1)
                * np.exp(-A1 * np.power(z, m1 + 1.0)))
    return out


def dx2dtheta(theta, m2, theta02, d2):
    out = np.zeros_like(np.asarray(theta, dtype=float))
    th = np.asarray(theta, dtype=float)
    act = th >= theta02
    z = (th[act] - theta02) / d2
    out[act] = (A2 * (m2 + 1.0) / d2 * np.power(z, m2)
                * np.exp(-A2 * np.power(z, m2 + 1.0)))
    return out


def dxbdtheta(theta, m1, theta01, d1, m2, theta02, d2, beta):
    return beta * dx1dtheta(theta, m1, theta01, d1) \
        + (1.0 - beta) * dx2dtheta(theta, m2, theta02, d2)


# ---------------------------------------------------------------------------
# MOD0d: sistema de EDOs (NDSolve -> solve_ivp DOP853) e função de erro
# ---------------------------------------------------------------------------
def mod0d(vetor, theta_exp, P_exp, rtol=1e-10, atol=1e-12, dense=True):
    """MOD0d[Rc, m1, theta01, d1, m2, theta02, d2, beta] do notebook.

    Retorna dict com P_sim, Tg, Qp (nos ângulos experimentais), V1 e o erro
    ``sqrt(SSres/(q-2))`` — idêntico ao Mathematica.
    """
    Rc, m1, theta01, d1, m2, theta02, d2, beta = [float(v) for v in vetor]
    theta_i, theta_f = float(theta_exp[0]), float(theta_exp[-1])
    P1 = float(P_exp[0])
    V1 = float(V(theta_i, Rc))

    Q_total = MCOMB * PCI                                   # kJ/ciclo

    def rhs(th, y):
        P, Tg, _ = y
        dxd = float(dxbdtheta(th, m1, theta01, d1, m2, theta02, d2, beta))
        dQ = Q_total * dxd                                  # kJ/rad
        # P/Tg negativos so aparecem em passos de tentativa rejeitados do
        # integrador; expoentes fracionarios de base negativa virariam
        # complexos -> NaN (o NDSolve rejeita o passo na mesma situacao).
        with np.errstate(invalid="ignore"):
            h = (130.0 * float(V(th, Rc)) ** (-0.06)
                 * float(np.power(P * 1e-2, 0.8))
                 * float(np.power(float(Tg), -0.4))
                 * (VP + 1.4) ** 0.8)                       # W/(m²K)
        dQp = (h * float(As(th, Rc)) * (Tg - TW)) / (2.0 * np.pi * OMEGA)
        dP = (1.0 / float(V(th, Rc))) * (
            (KP - 1.0) * (dQ - dQp / 1000.0)
            - (float(dVdtheta(th, Rc)) * P * KP))
        dTg = (T1 / (P1 * V1)) * (float(V(th, Rc)) * dP
                                  + P * float(dVdtheta(th, Rc)))
        return [dP, dTg, dQp]

    sol = solve_ivp(rhs, (theta_i, theta_f), [P1, T1, 0.0],
                    method="DOP853", t_eval=theta_exp, rtol=rtol, atol=atol)
    P_sim, Tg, Qp = sol.y[0], sol.y[1], sol.y[2]
    SSres = float(np.sum((np.asarray(P_exp) - P_sim) ** 2))
    q = theta_exp.size
    erro = float(np.sqrt(SSres / (q - 2.0)))                # notebook, verbatim
    return {
        "P_sim": P_sim, "Tg": Tg, "Qp": Qp, "V1": V1, "erro": erro,
        "sol": sol, "theta": np.asarray(theta_exp, dtype=float),
        "P_exp": np.asarray(P_exp, dtype=float),
    }


def erro_notebook(P_exp, P_sim):
    """erro do notebook = sqrt(SSres/(q-2))."""
    P_exp = np.asarray(P_exp, dtype=float)
    P_sim = np.asarray(P_sim, dtype=float)
    q = P_exp.size
    return float(np.sqrt(np.sum((P_exp - P_sim) ** 2) / (q - 2.0)))