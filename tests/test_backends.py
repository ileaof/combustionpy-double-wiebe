# -*- coding: utf-8 -*-
"""
test_backends.py — Validação numérica dos backends do plano HPC (regra 4).

Toda aceleração é comparada contra a referência serial (solve_ivp/DOP853):
equivalência exata para cpu-parallel (mesma implementação), tolerância
documentada para o RK4 em lote (modo acelerado) e bit-idêntico entre os
integradores de lote NumPy e Numba.
"""
from __future__ import annotations

import numpy as np
import pytest

from double_wiebe.backends import (BackendError, BackendNotAvailableError,
                                   CPUBackend, MultiprocessingBackend,
                                   SerialBackend, select_backend)
from double_wiebe.backends.base import regularize_batch
from double_wiebe.calibration import PARAM_SPECS, _regularization
from double_wiebe.models import DEFAULT_SELECTED, PARAM_ORDER
from double_wiebe.thermodynamics import PENALTY

# ---------------------------------------------------------------------------
# População de teste (dentro dos limites, margem de 20% das bordas)
# ---------------------------------------------------------------------------
N_CAND = 8


@pytest.fixture(scope="module")
def populacao():
    rng = np.random.default_rng(123)
    full0 = np.array([
        17.0, -6.54 * np.pi / 180, 25.0 * np.pi / 180, 0.5, 6.9078,
        10.0 * np.pi / 180, 55.0 * np.pi / 180, 1.0, 6.9078, 0.40,
    ])
    X_sel = np.column_stack([
        rng.uniform(PARAM_SPECS[n]["lower"] + 0.2 * (PARAM_SPECS[n]["upper"]
                                                     - PARAM_SPECS[n]["lower"]),
                    PARAM_SPECS[n]["upper"] - 0.2 * (PARAM_SPECS[n]["upper"]
                                                     - PARAM_SPECS[n]["lower"]),
                    N_CAND) for n in DEFAULT_SELECTED])
    idx = [PARAM_ORDER.index(n) for n in DEFAULT_SELECTED]
    X_full = np.tile(full0, (N_CAND, 1))
    X_full[:, idx] = X_sel
    return X_full


@pytest.fixture(scope="module")
def avaliacoes(synthetic, engine, wiebe, sim_cfg, populacao):
    """Valores objetivo por backend para a mesma população fixa."""
    theta, P, _, _ = synthetic
    from double_wiebe.models import CalibrationConfig
    calib = CalibrationConfig()
    X_full = populacao
    return {
        "serial": SerialBackend().evaluate_population(
            X_full, theta, P, engine, wiebe, sim_cfg, calib),
        "cpu_numpy": CPUBackend("rk4_numpy").evaluate_population(
            X_full, theta, P, engine, wiebe, sim_cfg, calib),
        "cpu_numba": CPUBackend("rk4_numba").evaluate_population(
            X_full, theta, P, engine, wiebe, sim_cfg, calib),
        "cpu_parallel": MultiprocessingBackend(workers=2).evaluate_population(
            X_full, theta, P, engine, wiebe, sim_cfg, calib),
    }, X_full


# ---------------------------------------------------------------------------
# 1. cpu-parallel == serial (idênticos: mesma implementação em processos)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_cpu_parallel_identical_to_serial(avaliacoes):
    vals, _ = avaliacoes
    f_ser, f_mp = vals["serial"], vals["cpu_parallel"]
    assert f_ser.shape == f_mp.shape == (N_CAND,)
    assert np.array_equal(f_ser, f_mp), (
        "cpu-parallel deve reproduzir a serial bit a bit")


# ---------------------------------------------------------------------------
# 2. RK4 em lote vs referência serial (tolerância documentada)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_cpu_rk4_within_tolerance(avaliacoes):
    vals, X_full = avaliacoes
    f_ser, f_cpu = vals["serial"], vals["cpu_numpy"]
    ok = (f_ser < PENALTY) & (f_cpu < PENALTY)
    assert ok.sum() >= N_CAND - 2, "demais candidatos falhos no fixture"
    # mesma máscara de penalidade OU diferença só em casos de borda
    df = np.abs(f_cpu[ok] - f_ser[ok])
    assert df.max() < 5.0, f"desvio máximo {df.max():.4f} kPa > 5 kPa"


@pytest.mark.slow
def test_batch_pressure_vs_solve_ivp(synthetic, engine, wiebe, sim_cfg,
                                     populacao):
    """max |P_sim(RK4 lote) − P_sim(solve_ivp)| <= 5 kPa por candidato."""
    theta, P, _, _ = synthetic
    from double_wiebe.models import CalibrationConfig
    from double_wiebe.simulation import evaluate_rmse
    from double_wiebe.backends.serial_backend import _row_to_configs
    from double_wiebe.integrators import batch_integrate_np
    calib = CalibrationConfig()
    P_sim = batch_integrate_np(theta, float(P[0]), populacao, engine, sim_cfg,
                               substeps=4)
    for i in range(populacao.shape[0]):
        if not np.isfinite(P_sim[i]).all():
            continue
        eng, wieb = _row_to_configs(populacao[i], engine, wiebe)
        P_ref, _, _ = (SerialBackend().simulate(theta, P, eng, wieb, sim_cfg))
        dv = np.nanmax(np.abs(P_sim[i] - P_ref))
        assert dv <= 5.0, f"candidato {i}: max|dP| = {dv:.4f} kPa"
        _ = evaluate_rmse, calib


# ---------------------------------------------------------------------------
# 3. rk4_numpy ≡ rk4_numba (bit-idênticos)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_rk4_numpy_equals_rk4_numba(synthetic, engine, sim_cfg, populacao):
    theta, P, _, _ = synthetic
    from double_wiebe.integrators import batch_integrate_np
    from double_wiebe.integrators.rk4_numba import batch_integrate_numba
    a = batch_integrate_np(theta, float(P[0]), populacao, engine, sim_cfg,
                           substeps=4)
    b = batch_integrate_numba(theta, float(P[0]), populacao, engine, sim_cfg,
                              substeps=4)
    assert np.array_equal(a, b), "NumPy e Numba devem ser bit-idênticos"


# ---------------------------------------------------------------------------
# 4. Regularização em lote == regularização escalar (por linha)
# ---------------------------------------------------------------------------
def test_regularize_batch_matches_scalar(synthetic, populacao):
    from double_wiebe.models import CalibrationConfig
    calib = CalibrationConfig()
    pen_batch = regularize_batch(populacao, calib)
    for i in range(populacao.shape[0]):
        assert pen_batch[i] == pytest.approx(
            _regularization(populacao[i], calib), abs=1e-12)


# ---------------------------------------------------------------------------
# 5. Falhas -> PENALTY (sem NaN) e populações de tamanho 1 / lotes ímpares
# ---------------------------------------------------------------------------
def test_failures_become_penalty(synthetic, engine, wiebe, sim_cfg):
    """Candidatos que falham na serial recebem PENALTY (sem NaN) em lote."""
    theta, P, _, _ = synthetic
    from double_wiebe.models import CalibrationConfig
    calib = CalibrationConfig()
    rng = np.random.default_rng(99)
    X_sel = np.column_stack([
        rng.uniform(PARAM_SPECS[n]["lower"], PARAM_SPECS[n]["upper"], 24)
        for n in DEFAULT_SELECTED])
    full0 = np.array([
        17.0, -6.54 * np.pi / 180, 25.0 * np.pi / 180, 0.5, 6.9078,
        10.0 * np.pi / 180, 55.0 * np.pi / 180, 1.0, 6.9078, 0.40,
    ])
    idx = [PARAM_ORDER.index(n) for n in DEFAULT_SELECTED]
    X = np.tile(full0, (24, 1))
    X[:, idx] = X_sel
    f_ser = SerialBackend().evaluate_population(X, theta, P, engine, wiebe,
                                                sim_cfg, calib)
    f_cpu = CPUBackend("rk4_numpy").evaluate_population(
        X, theta, P, engine, wiebe, sim_cfg, calib)
    assert np.isfinite(f_ser).all() and np.isfinite(f_cpu).all()
    falhos = f_ser >= PENALTY
    if not falhos.any():
        pytest.skip("nenhum candidato falhou na amostra (amostra benigna)")
    assert (f_cpu[falhos] >= PENALTY - 1.0).all()


def test_population_of_one(synthetic, engine, wiebe, sim_cfg, populacao):
    theta, P, _, _ = synthetic
    from double_wiebe.models import CalibrationConfig
    calib = CalibrationConfig()
    X = populacao[:1]
    f = CPUBackend("rk4_numpy").evaluate_population(
        X, theta, P, engine, wiebe, sim_cfg, calib, batch_size=3)
    f_ref = SerialBackend().evaluate_population(
        X, theta, P, engine, wiebe, sim_cfg, calib)
    assert f.shape == (1,)
    ok = (f_ref < PENALTY) & (f < PENALTY)
    if ok.any():
        assert abs(f[ok][0] - f_ref[ok][0]) < 5.0


@pytest.mark.slow
def test_batch_size_not_divisor(synthetic, engine, wiebe, sim_cfg, populacao):
    """batch_size que não divide S -> mesmo resultado do lote completo."""
    theta, P, _, _ = synthetic
    from double_wiebe.models import CalibrationConfig
    calib = CalibrationConfig()
    f_all = CPUBackend("rk4_numpy").evaluate_population(
        populacao, theta, P, engine, wiebe, sim_cfg, calib)
    f_blk = CPUBackend("rk4_numpy").evaluate_population(
        populacao, theta, P, engine, wiebe, sim_cfg, calib, batch_size=3)
    assert np.array_equal(f_all, f_blk)


# ---------------------------------------------------------------------------
# 6. Seleção de backend: inválido, fallback e reprodutibilidade
# ---------------------------------------------------------------------------
def test_select_backend_invalid():
    with pytest.raises(BackendError):
        select_backend("gpu-quantico")


def test_select_backend_valid():
    b = select_backend("serial")
    assert isinstance(b, SerialBackend)
    b = select_backend("cpu")
    assert isinstance(b, CPUBackend)


@pytest.mark.slow
def test_select_backend_auto():
    b = select_backend("auto")
    assert b is not None and b.is_available()


def test_fallback_unavailable(monkeypatch):
    from double_wiebe import backends as be
    class _Indisponivel(MultiprocessingBackend):
        def is_available(self):
            return False
    originais = dict(be._BACKEND_CLASSES)
    be._BACKEND_CLASSES["cpu-parallel"] = _Indisponivel
    try:
        with pytest.warns(UserWarning):
            b = select_backend("cpu-parallel")
        assert isinstance(b, SerialBackend)
        with pytest.raises(BackendNotAvailableError):
            select_backend("cpu-parallel", fallback=False)
    finally:
        be._BACKEND_CLASSES.clear()
        be._BACKEND_CLASSES.update(originais)


@pytest.mark.slow
def test_determinism_same_population(synthetic, engine, wiebe, sim_cfg,
                                     populacao):
    theta, P, _, _ = synthetic
    from double_wiebe.models import CalibrationConfig
    calib = CalibrationConfig()
    b = CPUBackend("rk4_numpy")
    f1 = b.evaluate_population(populacao, theta, P, engine, wiebe, sim_cfg,
                               calib)
    f2 = b.evaluate_population(populacao, theta, P, engine, wiebe, sim_cfg,
                               calib)
    assert np.array_equal(f1, f2)


# ---------------------------------------------------------------------------
# 7. float32 roda (modo acelerado) e devolve valores finitos
# ---------------------------------------------------------------------------
def test_precision_float32_runs(synthetic, engine, wiebe, sim_cfg, populacao):
    theta, P, _, _ = synthetic
    from double_wiebe.models import CalibrationConfig
    calib = CalibrationConfig()
    f = CPUBackend("rk4_numpy").evaluate_population(
        populacao, theta, P, engine, wiebe, sim_cfg, calib,
        precision="float32")
    assert f.shape == (populacao.shape[0],)
    assert np.isfinite(f).all()


# ---------------------------------------------------------------------------
# 8. Interface ComputeBackend: capabilities/simulate/synchronize/close
# ---------------------------------------------------------------------------
def test_backend_interface(synthetic, engine, wiebe, sim_cfg, populacao):
    theta, P, _, _ = synthetic
    from double_wiebe.models import CalibrationConfig
    calib = CalibrationConfig()
    for b in (SerialBackend(), CPUBackend("rk4_numpy"),
              MultiprocessingBackend(workers=2)):
        assert b.name and b.description
        assert isinstance(b.capabilities(), dict)
        b.synchronize()
        eng, wieb = (None, None)
        from double_wiebe.backends.serial_backend import _row_to_configs
        eng, wieb = _row_to_configs(populacao[0], engine, wiebe)
        P_sim, Tg, Qw = b.simulate(theta, P, eng, wieb, sim_cfg)
        assert np.isfinite(P_sim).all()
        b.close()          # no-op em serial/cpu; encerra pool no mp


# ---------------------------------------------------------------------------
# 9. Benchmark reproduzível (mesma entrada -> mesmas medições de erro)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_run_benchmark_reproducible(synthetic, engine, wiebe, sim_cfg):
    theta, P, _, _ = synthetic
    from double_wiebe.backends.benchmark import run_benchmark
    from double_wiebe.calibration import _full0
    from double_wiebe.models import CalibrationConfig
    calib = CalibrationConfig()
    full0 = _full0(engine, wiebe)
    linhas = run_benchmark(theta, P, engine, wiebe, sim_cfg, calib,
                           list(DEFAULT_SELECTED), full0,
                           tamanhos=(4, 10), rep=2, warmup=True)
    assert linhas, "benchmark deve produzir linhas"
    for l in linhas:
        assert np.isfinite(l["tempo_s"]) and l["tempo_s"] > 0.0
        if l["backend"] == "serial":
            assert l["speedup_vs_serial"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 10. Calibração com backends acelerados (regra 4: re-run serial no final)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_run_calibration_backends(synthetic, sim_cfg):
    theta, P, engine, wiebe = synthetic
    from double_wiebe.calibration import run_calibration
    from double_wiebe.models import CalibrationConfig
    base = dict(method="pso", selected=["delta1", "alpha"], seed=42,
                pso_particles=8, pso_max_iter=12, tol=1e-10, polish=False)
    r_serial = run_calibration(theta, P, engine, wiebe, sim_cfg,
                               CalibrationConfig(**base, backend="serial"))
    r_mp = run_calibration(theta, P, engine, wiebe, sim_cfg,
                           CalibrationConfig(**base, backend="cpu-parallel"))
    # PSO em lote com a implementação serial -> mesma trajetória, exato
    assert r_mp["rmse"] == pytest.approx(r_serial["rmse"], abs=0.0)
    for nome in r_serial["params"]:
        assert r_mp["params"][nome] == pytest.approx(
            r_serial["params"][nome], abs=0.0)
    assert r_mp["diferenca_integrador"] == pytest.approx(0.0, abs=1e-12)
    # backend cpu (RK4): diferença de integrador reportada e pequena
    r_cpu = run_calibration(theta, P, engine, wiebe, sim_cfg,
                            CalibrationConfig(**base, backend="cpu"))
    assert r_cpu["diferenca_integrador"] < 5.0
    assert r_cpu["backend"] == "cpu"
    assert r_cpu["tempo_s"] > 0.0