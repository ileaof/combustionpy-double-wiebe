# -*- coding: utf-8 -*-
"""
rk4_numba.py — RK4 em lote compilado com Numba (opcional, extra ``.[cpu]``).

Mesma matemática de ``rk4_numpy.py``: o laço completo (candidatos x
sub-passos) é compilado com @njit(cache=True) e paralelizado com prange
(threads) quando solicitado. Equações idênticas à referência; NÃO altera
física, unidades nem limites.

Sem numba instalado, ``batch_integrate_numba`` cai automaticamente para a
versão NumPy (mesmos resultados, ganho menor). precision="float32" também
cai para a versão NumPy (o kernel compila em float64).
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange, set_num_threads
    _HAS_NUMBA = True
except ImportError:                                    # extra [cpu] ausente
    _HAS_NUMBA = False

from .rk4_numpy import batch_integrate_np


def _compile():
    """Compila o kernel (uma vez por processo). Retorna a função njit."""
    from numba import njit, prange

    @njit(cache=True, inline="always")
    def _rhs(th, P, Tg, Vc, A_cyl, A_head, Vd, R, bore, rod, r_cr,
             kappa, Tw, Q_total, Vp_fac, om2pi, heat,
             th01, d1, m1, a1, th02, d2, m2, a2, alpha, dTg0):
        # RHS de um candidato (escalares) — idêntico a thermodynamics.make_rhs
        s = math.sin(th)
        c = math.cos(th)
        rootR = math.sqrt(R * R - s * s)
        V = Vc + (Vd / 2.0) * (R + 1.0 - c - rootR)
        dV = (Vd * s / 2.0) * (1.0 + c / rootR)
        y = rod + r_cr - r_cr * c - math.sqrt(rod ** 2 - r_cr ** 2 * s * s)
        As = A_head + math.pi * bore * y + A_cyl

        # fase 1 (premixed)
        dx1 = 0.0
        if th >= th01:
            zc = (th - th01) / d1
            if zc < 0.0:
                zc = 0.0
            zn = zc ** m1
            expo = math.exp(-a1 * zc * zn)
            dx1 = a1 * (m1 + 1.0) / d1 * zn * expo
        # fase 2 (difusão)
        dx2 = 0.0
        if th >= th02:
            zc = (th - th02) / d2
            if zc < 0.0:
                zc = 0.0
            zn = zc ** m2
            expo = math.exp(-a2 * zc * zn)
            dx2 = a2 * (m2 + 1.0) / d2 * zn * expo

        dxb = alpha * dx1 + (1.0 - alpha) * dx2
        dQ = Q_total * dxb                                   # kJ/rad
        if heat:
            h = 130.0 * V ** (-0.06) * (P * 1.0e-2) ** 0.8 \
                * Tg ** (-0.4) * Vp_fac                      # W/(m²K)
            dQw = h * As * (Tg - Tw) / om2pi                 # J/rad
        else:
            dQw = 0.0
        dP = (1.0 / V) * ((kappa - 1.0) * (dQ - dQw / 1000.0)
                          - kappa * P * dV)                  # kPa/rad
        dTg = dTg0 * (V * dP + P * dV)                       # K/rad
        return dP, dTg, dQw

    @njit(cache=True, nogil=True)
    def _kernel(theta_exp, P1, cand, Vd, R, bore, stroke, rod, kappa, T1, Tw,
                Q_total, Vp_fac, om2pi, heat, substeps):
        N = cand.shape[0]
        n = theta_exp.size
        P_out = np.full((N, n), np.nan)
        r_cr = stroke / 2.0
        A_head = 2.0 * math.pi * (bore / 2.0) ** 2

        for i in prange(N):
            Rc = cand[i, 0]
            th01, d1, m1, a1 = cand[i, 1], cand[i, 2], cand[i, 3], cand[i, 4]
            th02, d2, m2, a2 = cand[i, 5], cand[i, 6], cand[i, 7], cand[i, 8]
            alpha = cand[i, 9]
            if Rc <= 1.0 or d1 <= 0.0 or d2 <= 0.0:
                continue
            Vc = Vd / (Rc - 1.0)
            A_cyl = math.pi * bore * stroke / (Rc - 1.0)
            th0 = theta_exp[0]
            s0 = math.sin(th0)
            c0 = math.cos(th0)
            y0 = rod + r_cr - r_cr * c0 \
                - math.sqrt(rod ** 2 - r_cr ** 2 * s0 ** 2)
            V1 = Vc + (Vd / 2.0) * (R + 1.0 - c0 - math.sqrt(R ** 2 - s0 ** 2))
            dTg0 = T1 / (P1[i] * V1)

            P = P1[i]
            Tg = T1
            alive = True
            P_out[i, 0] = P
            for k in range(n - 1):
                a = theta_exp[k]
                h = (theta_exp[k + 1] - a) / substeps
                for _ in range(substeps):
                    # RK4 clássico com verificação de física em cada estágio
                    dP, dT, dW = _rhs(a, P, Tg, Vc, A_cyl, A_head, Vd, R,
                                      bore, rod, r_cr, kappa, Tw, Q_total,
                                      Vp_fac, om2pi, heat, th01, d1, m1, a1,
                                      th02, d2, m2, a2, alpha, dTg0)
                    if (dP != dP or dT != dT or dW != dW
                            or abs(dP) > 1.0e12 or abs(dT) > 1.0e12
                            or abs(dW) > 1.0e12):
                        alive = False
                    if alive:
                        k1p, k1t, k1w = dP, dT, dW
                        dP, dT, dW = _rhs(a + h / 2.0, P + h / 2.0 * k1p,
                                          Tg + h / 2.0 * k1t, Vc, A_cyl,
                                          A_head, Vd, R, bore, rod, r_cr,
                                          kappa, Tw, Q_total, Vp_fac, om2pi,
                                          heat, th01, d1, m1, a1, th02, d2,
                                          m2, a2, alpha, dTg0)
                        if (dP != dP or dT != dT or dW != dW
                                or abs(dP) > 1.0e12 or abs(dT) > 1.0e12
                                or abs(dW) > 1.0e12):
                            alive = False
                    if alive:
                        k2p, k2t, k2w = dP, dT, dW
                        dP, dT, dW = _rhs(a + h / 2.0, P + h / 2.0 * k2p,
                                          Tg + h / 2.0 * k2t, Vc, A_cyl,
                                          A_head, Vd, R, bore, rod, r_cr,
                                          kappa, Tw, Q_total, Vp_fac, om2pi,
                                          heat, th01, d1, m1, a1, th02, d2,
                                          m2, a2, alpha, dTg0)
                        if (dP != dP or dT != dT or dW != dW
                                or abs(dP) > 1.0e12 or abs(dT) > 1.0e12
                                or abs(dW) > 1.0e12):
                            alive = False
                    if alive:
                        k3p, k3t, k3w = dP, dT, dW
                        dP, dT, dW = _rhs(a + h, P + h * k3p, Tg + h * k3t,
                                          Vc, A_cyl, A_head, Vd, R, bore,
                                          rod, r_cr, kappa, Tw, Q_total,
                                          Vp_fac, om2pi, heat, th01, d1,
                                          m1, a1, th02, d2, m2, a2, alpha,
                                          dTg0)
                        if (dP != dP or dT != dT or dW != dW
                                or abs(dP) > 1.0e12 or abs(dT) > 1.0e12
                                or abs(dW) > 1.0e12):
                            alive = False
                    if alive:
                        k4p, k4t, k4w = dP, dT, dW
                        P = P + (h / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p
                                             + k4p)
                        Tg = Tg + (h / 6.0) * (k1t + 2.0 * k2t + 2.0 * k3t
                                               + k4t)
                        a = a + h
                    else:
                        break
                    if (P <= 0.0 or Tg <= 0.0
                            or not (np.isfinite(P) and np.isfinite(Tg))):
                        alive = False
                        break
                if not alive:
                    break
                P_out[i, k + 1] = P
        return P_out

    return _kernel


_KERNEL = None


def batch_integrate_numba(
    theta_exp: np.ndarray,
    P1: np.ndarray,
    cand: np.ndarray,
    engine,
    sim=None,
    substeps: int = 4,
    precision: str = "float64",
    threads: int = 1,
) -> np.ndarray:
    """RK4 em lote compilado. Mesma assinatura de ``batch_integrate_np``.

    ``threads > 1`` ativa o prange (threads do numba) — usado em benchmarks;
    o backend ``cpu`` roda com 1 thread e a paralelização de população fica
    a cargo do backend escolhido (multiprocessing etc.).
    ``precision="float32"`` cai para a versão NumPy (kernel é float64).
    """
    global _KERNEL
    if precision != "float64":
        return batch_integrate_np(theta_exp, P1, cand, engine, sim,
                                  substeps, precision)
    try:
        import numba
    except ImportError:
        return batch_integrate_np(theta_exp, P1, cand, engine, sim,
                                  substeps, precision)
    if _KERNEL is None:
        _KERNEL = _compile()
    if threads > 1 and threads <= numba.config.NUMBA_NUM_THREADS:
        set_num_threads(threads)
    theta_exp = np.ascontiguousarray(theta_exp, dtype=np.float64)
    cand = np.ascontiguousarray(np.atleast_2d(cand), dtype=np.float64)
    P1 = np.asarray(P1, dtype=np.float64)
    if P1.ndim == 0:
        P1 = np.full(cand.shape[0], float(P1))
    return _KERNEL(theta_exp, P1, cand,
                   engine.Vd, engine.R, engine.bore, engine.stroke,
                   engine.rod_length, engine.kappa, engine.T1, engine.Tw,
                   engine.Q_total, (engine.Vp + 1.4) ** 0.8,
                   2.0 * math.pi * engine.omega_rev_s,
                   1 if engine.heat_transfer else 0,
                   int(substeps))