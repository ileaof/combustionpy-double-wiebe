# -*- coding: utf-8 -*-
"""
rk4_numpy.py — RK4 de passo fixo EM LOTE (backend ``cpu``).

Integra N candidatos de parâmetros simultaneamente: um único laço de
passos temporais com RHS vetorizado sobre a população. As equações são
EXATAMENTE as de ``thermodynamics.py``/``geometry.py`` (Double Wiebe +
Hohenberg + geometria biela-manivela); muda apenas o integrador
(solve_ivp/DOP853 adaptativo -> RK4 de passo fixo com ``substeps``
sub-passos por intervalo entre ângulos experimentais).

Candidato com RHS/estado não físico é marcado como falho (linha NaN),
reproduzindo a semântica de ODEFailure da referência serial (penalidade
na função objetivo).

Modo acelerado: após a otimização o MELHOR candidato é re-integrado com
solve_ivp (referência) e a diferença de objetivo é relatada — ver
``calibration.py``.
"""
from __future__ import annotations

import numpy as np

from ..models import EngineConfig

_LIMIT = 1.0e12            # mesmo limite de thermodynamics.make_rhs
_DTYPES = {"float64": np.float64, "float32": np.float32}


def batch_integrate_np(
    theta_exp: np.ndarray,
    P1: np.ndarray,
    cand: np.ndarray,
    engine: EngineConfig,
    sim=None,
    substeps: int = 4,
    precision: str = "float64",
) -> np.ndarray:
    """Integra N candidatos com RK4 de passo fixo (NumPy, vetorizado).

    Parameters
    ----------
    theta_exp : (n,) ângulos experimentais [rad] (saída exata nesses pontos)
    P1        : (N,) pressão no IVC de cada candidato [kPa] (ou escalar)
    cand      : (N,10) parâmetros na ordem PARAM_ORDER
                [Rc, theta01, delta1, m1, a1, theta02, delta2, m2, a2, alpha]
    engine    : EngineConfig (geometria/termodinâmica compartilhada)
    sim       : SimulationConfig (aceito por compatibilidade; o RK4 usa
                passo fixo e não usa método/rtol/atol do solve_ivp)
    substeps  : sub-passos RK4 por intervalo entre ângulos experimentais
    precision : "float64" | "float32" (dtype do estado da integração)

    Returns
    -------
    P_sim : (N, n) pressão simulada [kPa]; linha NaN = candidato falho
    """
    theta_exp = np.asarray(theta_exp, dtype=np.float64)
    cand = np.atleast_2d(np.asarray(cand, dtype=np.float64))
    N = cand.shape[0]
    Rc, th01, d1, m1, a1, th02, d2, m2, a2, alpha = \
        (cand[:, j] for j in range(10))
    P1 = np.asarray(P1, dtype=np.float64)
    if P1.ndim == 0:
        P1 = np.full(N, float(P1))

    dt = _DTYPES.get(precision)
    if dt is None:
        raise ValueError(f"precision deve ser uma de {sorted(_DTYPES)}; "
                         f"recebido '{precision}'.")

    Vd, R = engine.Vd, engine.R
    bore, stroke, rod = engine.bore, engine.stroke, engine.rod_length
    kappa, T1, Tw = engine.kappa, engine.T1, engine.Tw
    Q_total = engine.Q_total
    Vp_fac = (engine.Vp + 1.4) ** 0.8
    om2pi = 2.0 * np.pi * engine.omega_rev_s
    heat = engine.heat_transfer

    # --- constantes por candidato (dependem de Rc) -------------------------
    Vc = Vd / (Rc - 1.0)                                # volume de folga (N,)
    A_cyl = np.pi * bore * stroke / (Rc - 1.0)
    A_head = 2.0 * np.pi * (bore / 2.0) ** 2            # compartilhado
    th0 = float(theta_exp[0])
    s0, c0 = np.sin(th0), np.cos(th0)
    r_cr = stroke / 2.0
    y0 = rod + r_cr - r_cr * c0 - np.sqrt(rod ** 2 - r_cr ** 2 * s0 ** 2)
    V1 = Vc + (Vd / 2.0) * (R + 1.0 - c0 - np.sqrt(R ** 2 - s0 ** 2))
    dTg0 = T1 / (P1 * V1)                               # fator de dTg (N,)

    # --- estado e constantes no dtype da integração ------------------------
    Y = np.zeros((3, N), dtype=dt)
    Y[0], Y[1] = P1.astype(dt), np.full(N, T1, dtype=dt)
    Vc_d, A_cyl_d = Vc.astype(dt), A_cyl.astype(dt)
    th01_d, d1_d, m1_d, a1_d = (v.astype(dt) for v in (th01, d1, m1, a1))
    th02_d, d2_d, m2_d, a2_d = (v.astype(dt) for v in (th02, d2, m2, a2))
    alpha_d, P1_d = alpha.astype(dt), P1.astype(dt)
    dTg0_d = dTg0.astype(dt)

    def rhs(th: float, Y):
        P, Tg = Y[0], Y[1]
        s, c = np.sin(th), np.cos(th)
        rootR = np.sqrt(R * R - s * s)
        # V depende de Rc somente pela constante Vc; dV/dtheta é independente
        # de Rc (escalar compartilhado por toda a população)
        V = Vc_d + (Vd / 2.0) * (R + 1.0 - c - rootR)          # (N,)
        dV = (Vd * s / 2.0) * (1.0 + c / rootR)                # escalar
        y = rod + r_cr - r_cr * c - np.sqrt(rod ** 2 - r_cr ** 2 * s * s)
        As = A_head + np.pi * bore * y + A_cyl_d               # (N,)

        # Double Wiebe — mesma matemática de thermodynamics._phase
        dxs = []
        for (th0i, di, mi, ai) in ((th01_d, d1_d, m1_d, a1_d),
                                   (th02_d, d2_d, m2_d, a2_d)):
            ativo = th >= th0i
            zc = np.where(ativo, np.maximum((th - th0i) / di, dt(0.0)), dt(0.0))
            zn = zc ** mi
            expo = np.exp(-ai * zc * zn)
            dx = ai * (mi + 1.0) / di * zn * expo
            dxs.append(np.where(ativo, dx, dt(0.0)))
        dx1, dx2 = dxs
        dxb = alpha_d * dx1 + (1.0 - alpha_d) * dx2            # (N,)
        dQ = Q_total * dxb                                     # kJ/rad

        if heat:
            h = (130.0 * V ** (-0.06) * (P * 1.0e-2) ** 0.8
                 * Tg ** (-0.4) * Vp_fac)                      # W/(m²K)
            dQw = h * As * (Tg - Tw) / om2pi                   # J/rad
        else:
            dQw = np.zeros(N, dtype=dt)

        dP = (1.0 / V) * ((kappa - 1.0) * (dQ - dQw / 1000.0)
                          - kappa * P * dV)                    # kPa/rad
        dTg = dTg0_d * (V * dP + P * dV)                       # K/rad
        return dP, dTg, dQw

    def _ok(k):
        """k: (3, N) estágio do RK4 -> máscara de candidatos físicos."""
        return (np.isfinite(k).all(axis=0)
                & (np.abs(k) <= _LIMIT).all(axis=0))

    alive = np.ones(N, dtype=bool)
    P_out = np.empty((N, theta_exp.size), dtype=np.float64)
    P_out[:, 0] = P1

    for k in range(theta_exp.size - 1):
        a = float(theta_exp[k])
        h = (float(theta_exp[k + 1]) - a) / substeps
        for _ in range(substeps):
            with np.errstate(all="ignore"):
                k1 = np.stack(rhs(a, Y))
                alive &= _ok(k1)
                Y2 = np.where(alive, Y + (dt(h) / 2.0) * k1, Y)
                k2 = np.stack(rhs(a + h / 2.0, Y2))
                alive &= _ok(k2)
                # estagios 3 e 4 partem do estado Y (RK4 classico)
                Y3 = np.where(alive, Y + (dt(h) / 2.0) * k2, Y)
                k3 = np.stack(rhs(a + h / 2.0, Y3))
                alive &= _ok(k3)
                Y4 = np.where(alive, Y + dt(h) * k3, Y)
                k4 = np.stack(rhs(a + h, Y4))
                alive &= _ok(k4)
                Y = np.where(alive, Y + (dt(h) / 6.0)
                             * (k1 + 2.0 * k2 + 2.0 * k3 + k4), Y)
            a += h
        # P/T não positivos em ponto de saída -> falho (como na referência)
        alive &= (Y[0] > 0.0) & (Y[1] > 0.0)
        P_out[:, k + 1] = np.where(alive, Y[0].astype(np.float64), np.nan)

    return P_out


# nome público estável para `integrators.get_integrator`
batch_integrate = batch_integrate_np