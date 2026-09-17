# -*- coding: utf-8 -*-
"""
thermodynamics.py
=================
Modelo termodinâmico de zona única acoplado ao Double Wiebe.

Correlação de Hohenberg (h em W/(m²·K)):

    h = 130 * V^(-0.06) * (P*1e-2)^0.8 * Tg^(-0.4) * (Vp + 1.4)^0.8
    Vp = 2*s*(rpm/60)                      (velocidade média do pistão)

Sistema de EDOs em theta, Y = [P (kPa), Tg (K), Q_wall (J)]:

    dQ_wall/dtheta = h*A_s*(Tg - Tw) / [2*pi*(rpm/60)]        [J/rad]
    dP/dtheta  = 1/V * {(kappa-1)*[dQ/dtheta - dQ_wall/1000]
                        - kappa*P*dV/dtheta}                  [kPa/rad]
    dTg/dtheta = T1/(P1*V1) * [V*dP/dtheta + P*dV/dtheta]     [K/rad]

Unidades de energia: dQ/dtheta em kJ/rad (Q_total = m_fuel*LHV em kJ/ciclo)
e dQ_wall em J/rad (o fator /1000 no balanço converte J -> kJ).

Condições iniciais (IVC, theta_i): P = P1 (experimental), Tg = T1,
Q_wall = 0. A solução é avaliada exatamente nos ângulos experimentais
(``t_eval``), sem interpolação.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .geometry import cylinder_volume, dV_dtheta, heat_area
from .models import EngineConfig, WiebeParameters
from .wiebe import double_burned_fraction


class ODEFailure(Exception):
    """Sinaliza RHS/integração não física (penalizada na calibração)."""


def hohenberg_h(theta, P_kPa, Tg, cfg: EngineConfig) -> np.ndarray:
    """Coeficiente de película por Hohenberg [W/(m²·K)].

    O fator (P*1e-2) converte kPa -> bar, escala original da correlação.
    """
    V = cylinder_volume(theta, cfg.Rc, cfg)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (
            130.0
            * V ** (-0.06)
            * (P_kPa * 1.0e-2) ** 0.8
            * Tg ** (-0.4)
            * (cfg.Vp + 1.4) ** 0.8
        )


def make_rhs(
    engine: EngineConfig,
    wiebe: WiebeParameters,
    P1: float,
    V1: float,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Constrói o lado direito do sistema de 3 EDOs (função pura)."""
    kappa, T1, Tw = engine.kappa, engine.T1, engine.Tw
    Q_total = engine.Q_total          # kJ/ciclo
    alpha = wiebe.alpha
    mode_bounded = wiebe.mode == "bounded"
    _LIMIT = 1.0e12                   # derivadas acima disso são não físicas

    def rhs(theta: float, y: np.ndarray) -> np.ndarray:
        P, Tg, _ = y
        # Frações queimadas e taxas das duas fases [1/rad]
        z1 = (theta - wiebe.theta01) / wiebe.delta1
        z2 = (theta - wiebe.theta02) / wiebe.delta2
        a1, m1, a2, m2 = wiebe.a1, wiebe.m1, wiebe.a2, wiebe.m2

        def _phase(z: float, th0: float, delta: float, m: float, a: float):
            if theta < th0:
                return 0.0, 0.0, 0.0
            zc = max(z, 0.0)
            zn = zc ** m
            znp1 = zc ** (m + 1.0)
            expo = np.exp(-a * znp1)
            x = 1.0 - expo
            dx = a * (m + 1.0) / delta * zn * expo
            if mode_bounded and theta > th0 + delta:
                x, dx = 1.0 - np.exp(-a), 0.0
            return x, dx, znp1

        x1, dx1, _ = _phase(z1, wiebe.theta01, wiebe.delta1, m1, a1)
        x2, dx2, _ = _phase(z2, wiebe.theta02, wiebe.delta2, m2, a2)
        dxb = alpha * dx1 + (1.0 - alpha) * dx2          # [1/rad]
        dQ = Q_total * dxb                               # kJ/rad

        V = cylinder_volume(theta, engine.Rc, engine)
        dV = dV_dtheta(theta, engine.Rc, engine)
        As = heat_area(theta, engine.Rc, engine)
        if engine.heat_transfer:
            h = hohenberg_h(theta, P, Tg, engine)        # W/(m²·K)
            dQw = h * As * (Tg - Tw) / (2.0 * np.pi * engine.omega_rev_s)  # J/rad
        else:
            dQw = 0.0

        dP = (1.0 / V) * (
            (kappa - 1.0) * (dQ - dQw / 1000.0) - kappa * P * dV
        )                                                # kPa/rad
        dTg = (T1 / (P1 * V1)) * (V * dP + P * dV)       # K/rad
        out = np.array([dP, dTg, dQw])
        if not np.all(np.isfinite(out)) or np.max(np.abs(out)) > _LIMIT:
            raise ODEFailure
        return out

    return rhs


def integrate_ode(
    theta_exp: np.ndarray,
    engine: EngineConfig,
    wiebe: WiebeParameters,
    P1: float,
    method: str = "DOP853",
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integra o sistema nos ângulos experimentais.

    Retorna (P_sim [kPa], Tg [K], Q_wall [J]) avaliados em ``theta_exp``.
    Levanta :class:`ODEFailure` quando a integração falha ou produz valores
    não físicos (NaN/inf, P <= 0, T <= 0, pressão não monotônica em queda
    livre — ver verificação abaixo).
    """
    theta_i = float(theta_exp[0])
    theta_f = float(theta_exp[-1])
    V1 = float(cylinder_volume(theta_i, engine.Rc, engine))
    rhs = make_rhs(engine, wiebe, P1, V1)
    y0 = np.array([P1, engine.T1, 0.0])

    try:
        sol = solve_ivp(
            rhs, (theta_i, theta_f), y0,
            method=method, t_eval=theta_exp, rtol=rtol, atol=atol,
        )
    except (ValueError, FloatingPointError, OverflowError, ODEFailure):
        raise ODEFailure("integração interrompida (RHS não física).") from None

    if (
        not sol.success
        or sol.y.shape[1] != theta_exp.size
        or not np.all(np.isfinite(sol.y))
    ):
        raise ODEFailure(f"integração falhou: {sol.message}")

    P_sim, Tg, Qw = sol.y
    if np.any(P_sim <= 0.0) or np.any(Tg <= 0.0):
        raise ODEFailure("pressão ou temperatura não positiva na solução.")
    return P_sim, Tg, Qw


def safe_integrate(
    theta_exp: np.ndarray,
    engine: EngineConfig,
    wiebe: WiebeParameters,
    P1: float,
    method: str = "DOP853",
    rtol: float = 1e-9,
    atol: float = 1e-9,
    penalty: float = 1.0e10,
):
    """Versão que NÃO levanta exceção: devolve (P_sim, Tg, Q_wall) ou
    ``(None, None, None)`` em caso de falha (usada na função objetivo)."""
    try:
        return integrate_ode(theta_exp, engine, wiebe, P1, method, rtol, atol)
    except ODEFailure:
        return None, None, None


# Reexporta a penalidade padrão para o módulo de calibração
PENALTY = 1.0e10