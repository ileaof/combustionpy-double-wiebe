# -*- coding: utf-8 -*-
"""
test_calibration.py
===================
Calibração: reprodutibilidade com semente, respeito aos limites,
regularização e aplicação dos parâmetros calibrados.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from double_wiebe.calibration import (CalibrationError, PARAM_SPECS,
                                      _full0, _make_expand, _make_objective,
                                      _regularization, apply_calibrated,
                                      build_bounds, run_calibration)
from double_wiebe.simulation import evaluate_rmse
from double_wiebe.models import CalibrationConfig, PARAM_ORDER


# =============================================================================
# Limites e especificações
# =============================================================================
def test_param_specs_cobrem_param_order():
    assert set(PARAM_SPECS) == set(PARAM_ORDER)


def test_build_bounds_para_selecionados():
    sel, lower, upper = build_bounds(["Rc", "alpha"])
    assert sel == ["Rc", "alpha"]
    assert lower[0] < 17.0 < upper[0]          # default Rc dentro do intervalo
    assert lower[1] < 0.40 < upper[1]
    with pytest.raises(CalibrationError):
        build_bounds(["param_inexistente"])


def test_build_bounds_sem_selecao():
    with pytest.raises(CalibrationError):
        build_bounds([])


def test_build_bounds_limites_invalidos():
    with pytest.raises(CalibrationError):
        build_bounds(["Rc"], bounds={"Rc": (18.0, 15.0)})


# =============================================================================
# Calibração completa (rápida: 2 parâmetros livres, poucas iterações)
# =============================================================================
def test_calibracao_recupera_parametros(synthetic, quick_calib, sim_cfg):
    theta, P_exp, engine, w = synthetic
    # least-squares converge rápido (parte dos próprios valores padrão)
    calib = CalibrationConfig(method="least-squares",
                              selected=["delta1", "alpha"], seed=42,
                              maxiter=50, polish=False)
    resultado = run_calibration(theta, P_exp, engine, w, sim_cfg, calib)
    assert resultado["rmse"] < 1.0            # dado sintético do próprio modelo
    p = resultado["params"]
    assert p["alpha"] == pytest.approx(w.alpha, abs=0.02)
    assert p["delta1"] == pytest.approx(w.delta1, abs=0.02)


def test_calibracao_reprodutivel_com_semente(synthetic, quick_calib, sim_cfg):
    theta, P_exp, engine, w = synthetic
    r1 = run_calibration(theta, P_exp, engine, w, sim_cfg, quick_calib)
    r2 = run_calibration(theta, P_exp, engine, w, sim_cfg, quick_calib)
    assert r1["params"] == r2["params"]


def test_calibracao_respeita_limites(synthetic, sim_cfg):
    theta, P_exp, engine, w = synthetic
    calib = CalibrationConfig(selected=["alpha"], seed=1, maxiter=2,
                              popsize=3, polish=False)
    r = run_calibration(theta, P_exp, engine, w, sim_cfg, calib)
    spec = PARAM_SPECS["alpha"]
    assert spec["lower"] <= r["params"]["alpha"] <= spec["upper"]


def test_parametros_fixos_nao_mudam(synthetic, quick_calib, sim_cfg):
    theta, P_exp, engine, w = synthetic
    r = run_calibration(theta, P_exp, engine, w, sim_cfg, quick_calib)
    assert r["params"]["Rc"] == pytest.approx(engine.Rc)
    assert r["params"]["m2"] == pytest.approx(w.m2)


def test_aplicar_parametros_calibrados(synthetic, quick_calib, sim_cfg):
    theta, P_exp, engine, w = synthetic
    r = run_calibration(theta, P_exp, engine, w, sim_cfg, quick_calib)
    eng, wb = apply_calibrated(r, engine, w)
    assert wb.alpha == r["params"]["alpha"]
    assert wb.delta1 == r["params"]["delta1"]
    assert eng.Rc == engine.Rc                # não selecionado → inalterado


def test_selecao_invalida_falha(synthetic, sim_cfg):
    theta, P_exp, engine, w = synthetic
    calib = CalibrationConfig(selected=["nao_existe"])
    with pytest.raises(CalibrationError):
        run_calibration(theta, P_exp, engine, w, sim_cfg, calib)


def test_historico_objectivo(synthetic, quick_calib, sim_cfg):
    theta, P_exp, engine, w = synthetic
    r = run_calibration(theta, P_exp, engine, w, sim_cfg, quick_calib)
    assert len(r["objective_history"]) > 0
    assert min(r["objective_history"]) <= r["rmse"] + 1e-6


# =============================================================================
# Regularização
# =============================================================================
def _x_full(theta01=0.0, theta02=1.0, delta1=0.2, delta2=0.2, **kw):
    x = np.array([17.0, theta01, delta1, 0.5, 6.9078,
                  theta02, delta2, 1.0, 6.9078, 0.4])
    return x


def test_penalidade_ordem_das_fases(sim_cfg):
    calib = CalibrationConfig()
    x_ok = _x_full()                          # theta02 > theta01: coerente
    assert _regularization(x_ok, calib) == pytest.approx(0.0, abs=1e-12)
    x_mau = _x_full(theta01=0.5, theta02=0.0)  # theta02 < theta01
    assert _regularization(x_mau, calib) > 0.0


def test_penalidade_duracao_minima():
    calib = CalibrationConfig(delta_min_deg=5.0)
    x = _x_full(delta1=math.radians(1.0))     # abaixo do mínimo
    assert _regularization(x, calib) > 0.0


def test_penalidade_sobreposicao():
    calib = CalibrationConfig()
    # fase 2 começa antes de theta01 + delta1
    x = _x_full(theta01=0.0, theta02=0.05, delta1=0.2)
    assert _regularization(x, calib) > 0.0


def test_regularizacao_desativada_com_pesos_zero():
    calib = CalibrationConfig(w_order=0.0, w_overlap=0.0, w_min_duration=0.0)
    x = _x_full(theta01=0.5, theta02=0.0, delta1=0.0, delta2=0.0)
    assert _regularization(x, calib) == pytest.approx(0.0, abs=1e-12)

# =============================================================================
# Regressões: alinhamento do objetivo e histórico do DE
# =============================================================================
def test_objectivo_alinhado_com_param_order(synthetic, sim_cfg):
    """Regressão: _make_objective passava 'selected' com o vetor COMPLETO ao
    _params_from_vector, embaralhando os parâmetros (a1 virava theta02 etc.)."""
    theta, P_exp, engine, w = synthetic
    sel = ["theta02", "delta2", "m2", "alpha"]   # nomes além dos 4 primeiros
    calib = CalibrationConfig(w_order=0.0, w_overlap=0.0, w_min_duration=0.0,
                              selected=sel)
    sel, lower, upper = build_bounds(sel)
    full0 = _full0(engine, w)
    expand, idx = _make_expand(sel, full0)
    obj = _make_objective(theta, P_exp, engine, w, sim_cfg, calib, sel, expand)
    from double_wiebe.simulation import evaluate_rmse
    esperado = evaluate_rmse(theta, P_exp, engine, w, sim_cfg)
    assert np.isfinite(esperado) and esperado < 1e6
    assert obj(full0[idx]) == pytest.approx(esperado, abs=1e-9)


def test_objectivo_com_regularizacao_no_x0(synthetic, sim_cfg):
    theta, P_exp, engine, w = synthetic
    sel = ["Rc", "theta01", "delta1", "m1", "theta02", "delta2", "m2", "alpha"]
    calib = CalibrationConfig(selected=sel)
    sel, lower, upper = build_bounds(sel)
    full0 = _full0(engine, w)
    expand, idx = _make_expand(sel, full0)
    obj = _make_objective(theta, P_exp, engine, w, sim_cfg, calib, sel, expand)
    esperado = evaluate_rmse(theta, P_exp, engine, w, sim_cfg) \
        + _regularization(full0, calib)
    assert obj(full0[idx]) == pytest.approx(esperado, abs=1e-9)


def test_de_historico_sem_nan(synthetic, quick_calib, sim_cfg):
    """Regressão: no protocolo legado do DE (xk, convergence) o callback
    preenchia history_params com NaN."""
    theta, P_exp, engine, w = synthetic
    r = run_calibration(theta, P_exp, engine, w, sim_cfg, quick_calib)
    assert len(r["history_params"]) > 0
    for p in r["history_params"]:
        assert np.all(np.isfinite(p))
    assert all(np.isfinite(f) for f in r["objective_history"])
    assert all(np.isfinite(v) for v in r["params"].values())
