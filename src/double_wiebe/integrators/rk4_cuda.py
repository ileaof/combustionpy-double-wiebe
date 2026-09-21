# -*- coding: utf-8 -*-
"""
rk4_cuda.py — RK4 em lote na GPU (CUDA via CuPy, extra ``.[cuda]``).

Mesma matemática e mesma semântica de falha de ``rk4_numba.py`` (que é
bit-idêntico a ``rk4_numpy.py``): um thread CUDA integra UM candidato com
RK4 clássico de passo fixo e ``substeps`` sub-passos por intervalo
experimental; RHS/estado não físico -> linha NaN (PENALTY no objetivo).
Equações idênticas à referência; NÃO altera física, unidades nem limites.

Diferenças numéricas em relação à CPU vêm apenas das funções matemáticas
da GPU (sin/cos/exp/pow com erro de poucos ULP); a contração em FMA é
desligada (``-fmad=false``) para ficar o mais próximo possível da CPU.
A tolerância contra ``rk4_numpy`` é verificada em tests/test_cuda_backend.py
e o melhor candidato final é sempre re-integrado com solve_ivp.

``precision="float32"`` compila o mesmo kernel em float (muito mais rápido
em GPUs de consumo, cujo throughput FP64 é 1/32–1/64 do FP32).
"""
from __future__ import annotations

import math

import numpy as np

_THREADS = 128

_SOURCE = r"""
#define RL(x) ((real)(x))

__device__ __forceinline__ bool ok3(real a, real b, real c) {
    const real L = RL(1.0e12);
    return isfinite(a) && isfinite(b) && isfinite(c)
        && fabs(a) <= L && fabs(b) <= L && fabs(c) <= L;
}

__device__ __forceinline__ void rhs(
        real th, real P, real Tg, real Vc, real A_cyl, real A_head,
        real Vd, real R, real bore, real rod, real r_cr, real kappa,
        real Tw, real Q_total, real Vp_fac, real om2pi, int heat,
        real th01, real d1, real m1, real a1,
        real th02, real d2, real m2, real a2, real alpha, real dTg0,
        real* dP, real* dTg, real* dQw) {
    // RHS de um candidato — idêntico a thermodynamics.make_rhs/rk4_numba
    const real s = sin(th);
    const real c = cos(th);
    const real rootR = sqrt(R * R - s * s);
    const real V = Vc + (Vd / RL(2.0)) * (R + RL(1.0) - c - rootR);
    const real dV = (Vd * s / RL(2.0)) * (RL(1.0) + c / rootR);
    const real y = rod + r_cr - r_cr * c
        - sqrt(rod * rod - r_cr * r_cr * s * s);
    const real As = A_head + RL(3.141592653589793) * bore * y + A_cyl;

    real dx1 = RL(0.0);                       // fase 1 (premixed)
    if (th >= th01) {
        real zc = (th - th01) / d1;
        if (zc < RL(0.0)) zc = RL(0.0);
        const real zn = pow(zc, m1);
        dx1 = a1 * (m1 + RL(1.0)) / d1 * zn * exp(-a1 * zc * zn);
    }
    real dx2 = RL(0.0);                       // fase 2 (difusão)
    if (th >= th02) {
        real zc = (th - th02) / d2;
        if (zc < RL(0.0)) zc = RL(0.0);
        const real zn = pow(zc, m2);
        dx2 = a2 * (m2 + RL(1.0)) / d2 * zn * exp(-a2 * zc * zn);
    }
    const real dxb = alpha * dx1 + (RL(1.0) - alpha) * dx2;
    const real dQ = Q_total * dxb;                           // kJ/rad
    real qw = RL(0.0);
    if (heat) {
        const real h = RL(130.0) * pow(V, RL(-0.06))
            * pow(P * RL(1.0e-2), RL(0.8)) * pow(Tg, RL(-0.4)) * Vp_fac;
        qw = h * As * (Tg - Tw) / om2pi;                     // J/rad
    }
    *dP = (RL(1.0) / V) * ((kappa - RL(1.0)) * (dQ - qw / RL(1000.0))
                           - kappa * P * dV);                // kPa/rad
    *dTg = dTg0 * (V * *dP + P * dV);                        // K/rad
    *dQw = qw;
}

extern "C" __global__ void rk4_batch(
        const double* theta_exp, const real* P1, const real* cand,
        const int N, const int n,
        const real Vd, const real R, const real bore, const real stroke,
        const real rod, const real kappa, const real T1, const real Tw,
        const real Q_total, const real Vp_fac, const real om2pi,
        const int heat, const int substeps, double* P_out) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    double* row = P_out + (size_t)i * n;
    const real* p = cand + (size_t)i * 10;
    const real Rc = p[0];
    const real th01 = p[1], d1 = p[2], m1 = p[3], a1 = p[4];
    const real th02 = p[5], d2 = p[6], m2 = p[7], a2 = p[8];
    const real alpha = p[9];
    if (!(Rc > RL(1.0)) || !(d1 > RL(0.0)) || !(d2 > RL(0.0))) return;

    const real r_cr = stroke / RL(2.0);
    const real A_head = RL(2.0) * RL(3.141592653589793)
        * (bore / RL(2.0)) * (bore / RL(2.0));
    const real Vc = Vd / (Rc - RL(1.0));
    const real A_cyl = RL(3.141592653589793) * bore * stroke / (Rc - RL(1.0));
    const real th0 = (real)theta_exp[0];
    const real c0 = cos(th0), s0 = sin(th0);
    const real V1 = Vc + (Vd / RL(2.0))
        * (R + RL(1.0) - c0 - sqrt(R * R - s0 * s0));
    const real dTg0 = T1 / (P1[i] * V1);

    real P = P1[i], Tg = T1;
    row[0] = (double)P;
    for (int k = 0; k < n - 1; ++k) {
        real a = (real)theta_exp[k];
        const real h = ((real)theta_exp[k + 1] - a) / (real)substeps;
        const real h2 = h / RL(2.0);
        for (int s = 0; s < substeps; ++s) {
            real k1p, k1t, k1w, k2p, k2t, k2w, k3p, k3t, k3w, k4p, k4t, k4w;
            rhs(a, P, Tg, Vc, A_cyl, A_head, Vd, R, bore, rod, r_cr, kappa,
                Tw, Q_total, Vp_fac, om2pi, heat, th01, d1, m1, a1, th02,
                d2, m2, a2, alpha, dTg0, &k1p, &k1t, &k1w);
            if (!ok3(k1p, k1t, k1w)) return;
            rhs(a + h2, P + h2 * k1p, Tg + h2 * k1t, Vc, A_cyl, A_head, Vd,
                R, bore, rod, r_cr, kappa, Tw, Q_total, Vp_fac, om2pi, heat,
                th01, d1, m1, a1, th02, d2, m2, a2, alpha, dTg0,
                &k2p, &k2t, &k2w);
            if (!ok3(k2p, k2t, k2w)) return;
            rhs(a + h2, P + h2 * k2p, Tg + h2 * k2t, Vc, A_cyl, A_head, Vd,
                R, bore, rod, r_cr, kappa, Tw, Q_total, Vp_fac, om2pi, heat,
                th01, d1, m1, a1, th02, d2, m2, a2, alpha, dTg0,
                &k3p, &k3t, &k3w);
            if (!ok3(k3p, k3t, k3w)) return;
            rhs(a + h, P + h * k3p, Tg + h * k3t, Vc, A_cyl, A_head, Vd, R,
                bore, rod, r_cr, kappa, Tw, Q_total, Vp_fac, om2pi, heat,
                th01, d1, m1, a1, th02, d2, m2, a2, alpha, dTg0,
                &k4p, &k4t, &k4w);
            if (!ok3(k4p, k4t, k4w)) return;
            P = P + (h / RL(6.0)) * (k1p + RL(2.0) * k2p + RL(2.0) * k3p
                                     + k4p);
            Tg = Tg + (h / RL(6.0)) * (k1t + RL(2.0) * k2t + RL(2.0) * k3t
                                       + k4t);
            a = a + h;
            if (!(P > RL(0.0)) || !(Tg > RL(0.0))
                    || !isfinite(P) || !isfinite(Tg)) return;
        }
        row[k + 1] = (double)P;
    }
}
"""

_DTYPES = {"float64": (np.float64, "double"), "float32": (np.float32, "float")}
_KERNELS: dict = {}


def cuda_available() -> bool:
    """True se CuPy está instalado e há ao menos uma GPU CUDA utilizável."""
    try:
        import cupy
        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def _kernel(precision: str):
    """Compila (uma vez por processo e precisão) o kernel RK4 em lote."""
    if precision not in _KERNELS:
        import cupy as cp
        _, ctype = _DTYPES[precision]
        src = f"typedef {ctype} real;\n" + _SOURCE
        _KERNELS[precision] = cp.RawKernel(
            src, "rk4_batch", options=("-fmad=false",))
    return _KERNELS[precision]


def _engine_args(engine, dt):
    return tuple(dt(v) for v in (
        engine.Vd, engine.R, engine.bore, engine.stroke, engine.rod_length,
        engine.kappa, engine.T1, engine.Tw, engine.Q_total,
        (engine.Vp + 1.4) ** 0.8, 2.0 * math.pi * engine.omega_rev_s))


def batch_integrate_cuda_device(theta_d, P1, cand, engine, substeps=4,
                                precision="float64"):
    """Versão que mantém os dados na GPU.

    theta_d : cupy (n,) float64 com os ângulos experimentais
    P1      : escalar ou (N,) [kPa];  cand : (N,10) PARAM_ORDER
    Retorna cupy (N, n) float64 (linha NaN = candidato falho).
    """
    import cupy as cp

    if precision not in _DTYPES:
        raise ValueError(f"precision deve ser uma de {sorted(_DTYPES)}; "
                         f"recebido '{precision}'.")
    dt, _ = _DTYPES[precision]
    cand = np.ascontiguousarray(np.atleast_2d(cand), dtype=dt)
    N, n = cand.shape[0], int(theta_d.size)
    P1 = np.asarray(P1, dtype=dt)
    if P1.ndim == 0:
        P1 = np.full(N, P1, dtype=dt)
    P_out = cp.full((N, n), cp.nan, dtype=cp.float64)
    if N == 0:
        return P_out
    blocos = (N + _THREADS - 1) // _THREADS
    _kernel(precision)(
        (blocos,), (_THREADS,),
        (theta_d, cp.asarray(P1), cp.asarray(cand), np.int32(N), np.int32(n),
         *_engine_args(engine, dt), np.int32(1 if engine.heat_transfer else 0),
         np.int32(substeps), P_out))
    return P_out


def batch_integrate_cuda(
    theta_exp: np.ndarray,
    P1: np.ndarray,
    cand: np.ndarray,
    engine,
    sim=None,
    substeps: int = 4,
    precision: str = "float64",
) -> np.ndarray:
    """RK4 em lote na GPU. Mesma assinatura/saída de ``batch_integrate_np``
    (numpy (N, n) float64; linha NaN = candidato falho)."""
    import cupy as cp

    theta_d = cp.asarray(np.ascontiguousarray(theta_exp, dtype=np.float64))
    return cp.asnumpy(batch_integrate_cuda_device(
        theta_d, P1, cand, engine, substeps, precision))
