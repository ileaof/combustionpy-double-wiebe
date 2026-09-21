# -*- coding: utf-8 -*-
"""
test_cuda_backend.py — Validação do backend `cuda` (RK4 em lote na GPU).

Pulado automaticamente sem CuPy/GPU. Referência: rk4_numpy (mesmo
integrador na CPU) — a GPU difere apenas pelas funções matemáticas do
device (poucos ULP), então exige-se concordância relativa ~1e-12 em float64
e a MESMA máscara de candidatos falhos.
"""
from __future__ import annotations

import numpy as np
import pytest

from double_wiebe.integrators.rk4_cuda import cuda_available

pytestmark = pytest.mark.skipif(not cuda_available(),
                                reason="CuPy/GPU CUDA indisponível")

from double_wiebe.backends import (CPUBackend, CUDABackend,  # noqa: E402
                                   SerialBackend, select_backend)
from double_wiebe.calibration import PARAM_SPECS  # noqa: E402
from double_wiebe.models import (CalibrationConfig,  # noqa: E402
                                 DEFAULT_SELECTED, PARAM_ORDER)
from double_wiebe.thermodynamics import PENALTY  # noqa: E402


def _pop(S, seed, margem=0.0):
    rng = np.random.default_rng(seed)
    full0 = np.array([
        17.0, -6.54 * np.pi / 180, 25.0 * np.pi / 180, 0.5, 6.9078,
        10.0 * np.pi / 180, 55.0 * np.pi / 180, 1.0, 6.9078, 0.40,
    ])
    cols = []
    for n in DEFAULT_SELECTED:
        lo, hi = PARAM_SPECS[n]["lower"], PARAM_SPECS[n]["upper"]
        cols.append(rng.uniform(lo + margem * (hi - lo),
                                hi - margem * (hi - lo), S))
    X = np.tile(full0, (S, 1))
    X[:, [PARAM_ORDER.index(n) for n in DEFAULT_SELECTED]] = \
        np.column_stack(cols)
    return X


def test_rk4_cuda_matches_rk4_numpy(synthetic, engine, sim_cfg):
    theta, P, _, _ = synthetic
    from double_wiebe.integrators import batch_integrate_np, get_integrator
    X = _pop(64, 7)
    a = batch_integrate_np(theta, float(P[0]), X, engine, sim_cfg, 4)
    g = get_integrator("rk4_cuda")(theta, float(P[0]), X, engine, sim_cfg, 4)
    assert g.shape == a.shape and g.dtype == np.float64
    assert np.array_equal(np.isnan(a), np.isnan(g)), "máscara de falhas"
    ok = np.isfinite(a)
    np.testing.assert_allclose(g[ok], a[ok], rtol=1e-12, atol=0.0)


def test_rk4_cuda_failure_mask_matches_numpy(synthetic, engine, sim_cfg):
    """Candidatos não físicos (RHS explode, alpha fora de [0,1], P1 < 0)
    falham nos MESMOS candidatos em CPU e GPU."""
    theta, P, _, _ = synthetic
    from double_wiebe.integrators import batch_integrate_np
    from double_wiebe.integrators.rk4_cuda import batch_integrate_cuda
    X = _pop(6, 1)
    X[0, 3], X[0, 2] = -3.0, 1e-3
    X[1, 9], X[2, 9] = 50.0, -80.0
    X[3, 1], X[3, 3] = theta[40], -1.0
    P1 = np.full(6, float(P[0]))
    P1[4] = -1.0
    a = batch_integrate_np(theta, P1, X, engine, sim_cfg, 4)
    g = batch_integrate_cuda(theta, P1, X, engine, sim_cfg, 4)
    falhos = np.isnan(a).any(axis=1)
    assert falhos.sum() >= 3, "o caso de teste deveria gerar falhas"
    assert np.array_equal(falhos, np.isnan(g).any(axis=1))
    ok = np.isfinite(a) & np.isfinite(g)
    np.testing.assert_allclose(g[ok], a[ok], rtol=1e-12, atol=0.0)


def test_rk4_cuda_invalid_candidates_are_nan(synthetic, engine, sim_cfg):
    theta, P, _, _ = synthetic
    from double_wiebe.integrators.rk4_cuda import batch_integrate_cuda
    X = _pop(3, 1, margem=0.2)
    X[0, 0] = 1.0          # Rc <= 1
    X[1, 2] = 0.0          # delta1 <= 0
    g = batch_integrate_cuda(theta, float(P[0]), X, engine, sim_cfg, 4)
    assert np.isnan(g[0]).all() and np.isnan(g[1]).all()
    assert np.isfinite(g[2]).all()


def test_cuda_backend_matches_cpu(synthetic, engine, wiebe, sim_cfg):
    """Objetivo (RMSE + regularização + PENALTY) igual ao backend cpu,
    inclusive com candidatos falhos e batch_size que não divide S."""
    theta, P, _, _ = synthetic
    calib = CalibrationConfig()
    X = _pop(40, 99)
    f_cpu = CPUBackend("rk4_numpy").evaluate_population(
        X, theta, P, engine, wiebe, sim_cfg, calib)
    b = CUDABackend()
    f_gpu = b.evaluate_population(X, theta, P, engine, wiebe, sim_cfg,
                                  calib, batch_size=7)
    b.synchronize()
    assert f_gpu.shape == (40,) and np.isfinite(f_gpu).all()
    assert np.array_equal(f_cpu >= PENALTY, f_gpu >= PENALTY)
    np.testing.assert_allclose(f_gpu, f_cpu, rtol=1e-9, atol=1e-9)
    b.close()


def test_cuda_float32_close_to_serial(synthetic, engine, wiebe, sim_cfg):
    theta, P, _, _ = synthetic
    calib = CalibrationConfig()
    X = _pop(8, 123, margem=0.2)
    f_ser = SerialBackend().evaluate_population(X, theta, P, engine, wiebe,
                                                sim_cfg, calib)
    f32 = CUDABackend().evaluate_population(
        X, theta, P, engine, wiebe, sim_cfg, calib, precision="float32")
    ok = (f_ser < PENALTY) & (f32 < PENALTY)
    assert ok.sum() >= 6
    assert np.max(np.abs(f32[ok] - f_ser[ok])) < 5.0


def test_select_backend_cuda_and_interface(synthetic, engine, wiebe,
                                           sim_cfg):
    theta, P, _, _ = synthetic
    b = select_backend("cuda", fallback=False)
    assert isinstance(b, CUDABackend) and b.is_available()
    caps = b.capabilities()
    assert caps["gpus"] and caps["vetorizado"]
    P_sim, _, _ = b.simulate(theta, P, engine, wiebe, sim_cfg)
    assert np.isfinite(P_sim).all()


@pytest.mark.slow
def test_run_calibration_cuda(synthetic, sim_cfg):
    theta, P, engine, wiebe = synthetic
    from double_wiebe.calibration import run_calibration
    base = dict(method="pso", selected=["delta1", "alpha"], seed=42,
                pso_particles=8, pso_max_iter=12, tol=1e-10, polish=False)
    r_cpu = run_calibration(theta, P, engine, wiebe, sim_cfg,
                            CalibrationConfig(**base, backend="cpu",
                                              integrator="rk4_numpy"))
    r_gpu = run_calibration(theta, P, engine, wiebe, sim_cfg,
                            CalibrationConfig(**base, backend="cuda"))
    assert r_gpu["backend"] == "cuda"
    assert r_gpu["diferenca_integrador"] < 5.0
    assert r_gpu["rmse"] == pytest.approx(r_cpu["rmse"], rel=1e-6, abs=1e-6)
