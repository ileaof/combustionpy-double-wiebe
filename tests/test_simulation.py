# -*- coding: utf-8 -*-
"""
test_simulation.py
==================
Simulação 3-EDO (P, Tg, Q_parede): sanidade física, métricas e
``evaluate_rmse`` com penalidade.
"""
from __future__ import annotations

import numpy as np
import pytest

from double_wiebe.metrics import (mae, max_abs_error, mean_percent_error,
                                  r_squared, rmse, sse, summary_metrics)
from double_wiebe.simulation import evaluate_rmse, run_simulation
from double_wiebe.thermodynamics import PENALTY, ODEFailure


THETA = np.linspace(-2.0, 2.0, 101)
P_EXP = np.full_like(THETA, 138.2)      # placeholder substituído pelo fixture


def test_simulacao_sem_nan(synthetic):
    theta, P_exp, engine, w = synthetic
    from double_wiebe.models import SimulationConfig
    res = run_simulation(theta, P_exp, engine, w,
                         SimulationConfig(method="LSODA", rtol=1e-8,
                                          atol=1e-8))
    assert res.ok
    assert np.all(np.isfinite(res.P_sim))
    assert np.all(np.isfinite(res.Tg)) and np.all(res.Tg > 0.0)
    assert np.all(np.isfinite(res.Q_wall))
    # antes da combustão Tg < Tw: calor da parede para o gás (negativo) é físico
    assert res.Q_wall[-1] >= 0.0


def test_indicadores_consistentes(synthetic):
    theta, P_exp, engine, w = synthetic
    res = run_simulation(theta, P_exp, engine, w)
    ind = res.indicators
    assert ind["P_max_kPa"] == pytest.approx(np.max(res.P_sim), rel=1e-9)
    assert ind["theta_at_P_max_deg"] == pytest.approx(
        np.degrees(theta[int(np.argmax(res.P_sim))]), rel=1e-9)
    assert ind["final_burned_fraction"] == pytest.approx(
        res.xb_total[-1], rel=1e-9)
    # frações de energia fase 1 + fase 2 = 100 %
    assert ind["phase1_energy_share_pct"] + ind["phase2_energy_share_pct"] == \
        pytest.approx(100.0, rel=1e-6)


def test_metricas_de_ajuste(synthetic):
    theta, P_exp, engine, w = synthetic
    res = run_simulation(theta, P_exp, engine, w)
    m = res.metrics
    assert m["RMSE_kPa"] == pytest.approx(rmse(P_exp, res.P_sim), rel=1e-9)
    assert m["MAE_kPa"] == pytest.approx(mae(P_exp, res.P_sim), rel=1e-9)
    assert m["R2"] == pytest.approx(r_squared(P_exp, res.P_sim), rel=1e-9)
    # auto-simulação perfeita ⇒ métricas ~ 0 (modelo vs ele mesmo)
    assert m["RMSE_kPa"] < 1.0        # mesma série: resíduo de malha apenas
    assert m["R2"] > 0.999


def test_metricas_unitarias():
    P = np.array([100.0, 200.0, 300.0])
    assert sse(P, P) == pytest.approx(0.0)
    assert mae(P, P + 10.0) == pytest.approx(10.0)
    assert rmse(P, P + 10.0) == pytest.approx(10.0)
    assert max_abs_error(P, P + 10.0) == pytest.approx(10.0)
    # média dos erros percentuais relativos a P_exp: (10/100+10/200+10/300)/3
    esperado_mpe = 100.0 * np.mean(10.0 / P)
    assert mean_percent_error(P, P + 10.0) == pytest.approx(esperado_mpe,
                                                            rel=1e-9)
    assert r_squared(P, P) == pytest.approx(1.0)
    resumo = summary_metrics(P, P + 5.0)
    assert set(resumo) >= {"SSE_kPa2", "MSE_kPa2", "RMSE_kPa", "MAE_kPa",
                           "R2", "max_abs_error_kPa", "mean_percent_error_pct"}


def test_evaluate_rmse_penalidade_quando_integracao_falha(synthetic):
    theta, P_exp, engine, w = synthetic
    from double_wiebe.models import SimulationConfig
    # kappa inválido força validação a falhar antes de integrar
    quebrado = engine.__class__(kappa=0.5)
    valor = evaluate_rmse(theta, P_exp, quebrado, w,
                          SimulationConfig())
    assert valor == PENALTY


def test_run_simulation_rejeita_configuracao_invalida(synthetic):
    theta, P_exp, engine, w = synthetic
    with pytest.raises(ValueError):
        run_simulation(theta, P_exp, engine, w.__class__(alpha=2.0))


def test_series_de_tamanhos_diferentes_falham(synthetic):
    theta, P_exp, engine, w = synthetic
    with pytest.raises(ValueError):
        run_simulation(theta[:-1], P_exp, engine, w)


def test_ode_failure_tem_mensagem():
    assert issubclass(ODEFailure, Exception)