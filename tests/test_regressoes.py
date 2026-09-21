# -*- coding: utf-8 -*-
"""
test_regressoes.py — Regressões encontradas ao calibrar o ensaio real
(P_exp-Carga-3_45%, data/example_pressure.txt).

1. `--backend` (ou qualquer chave ausente do YAML) com `--config` gerava
   KeyError 'calibration.backend'.
2. O tratamento desse erro quebrava com UnboundLocalError.
3. least-squares relatava √(Σr²) em vez do RMSE (fator √n ≈ 21).
4. Após o polish aplicado, o RMSE relatado era o de antes do polish.
5. "diferença do integrador" saía sempre 0 (bloco duplicado).
6. least-squares com valor inicial fora dos limites: "x0 is infeasible".
7. GUI: limites angulares apareciam em radianos, sem unidade (o restante
   da GUI usa graus).
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from double_wiebe.calibration import run_calibration
from double_wiebe.cli import _carregar, app
from double_wiebe.models import load_config

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "data" / "example_pressure.txt"
CONFIG = RAIZ / "configs" / "example.yaml"


@pytest.fixture(scope="module")
def ensaio():
    return _carregar(DADOS, CONFIG, {})


def test_sobrescrita_de_chave_ausente_no_yaml():
    cfg = load_config(CONFIG, {"calibration.backend": "cpu",
                               "calibration.precision": "float32"})
    assert cfg["calibration"].backend == "cpu"
    assert cfg["calibration"].precision == "float32"
    # valores do YAML continuam valendo
    assert cfg["calibration"].popsize == 20


def test_cli_backend_com_config_e_erro_legivel(tmp_path):
    r = CliRunner().invoke(app, [
        "calibrate", "--data", str(DADOS), "--config", str(CONFIG),
        "--method", "least-squares", "--select", "alpha",
        "--backend", "cpu", "--output", str(tmp_path / "a"), "--quiet"])
    assert r.exit_code == 0, r.output
    r = CliRunner().invoke(app, [
        "calibrate", "--data", str(DADOS), "--config", str(CONFIG),
        "--set", "calibration.maxiter=abc", "--method", "pso",
        "--output", str(tmp_path / "b"), "--quiet"])
    assert r.exit_code != 0
    assert "UnboundLocalError" not in r.output


def test_least_squares_relata_rmse(ensaio):
    cfg, theta, P, _ = ensaio
    c = replace(cfg["calibration"], method="least-squares",
                selected=["alpha"])
    r = run_calibration(theta, P, cfg["engine"], cfg["wiebe"],
                        cfg["simulation"], c)
    from double_wiebe.calibration import apply_calibrated
    from double_wiebe.simulation import run_simulation
    eng, wb = apply_calibrated(r, cfg["engine"], cfg["wiebe"])
    sim = run_simulation(theta, P, eng, wb, cfg["simulation"])
    rmse_real = float(np.sqrt(np.mean((sim.P_sim - P) ** 2)))
    assert r["rmse"] == pytest.approx(rmse_real, rel=1e-6)


def test_polish_atualiza_rmse_relatado(ensaio):
    cfg, theta, P, _ = ensaio
    c = replace(cfg["calibration"], method="pso", selected=["delta2", "alpha"],
                pso_particles=6, pso_max_iter=4, seed=3, polish=True)
    r = run_calibration(theta, P, cfg["engine"], cfg["wiebe"],
                        cfg["simulation"], c)
    if r["polish"] and r["polish"].get("aplicado"):
        assert r["rmse"] == pytest.approx(r["polish"]["rmse"], rel=1e-12)


def test_diferenca_integrador_nao_e_zerada(ensaio):
    cfg, theta, P, _ = ensaio
    c = replace(cfg["calibration"], method="pso", selected=["alpha"],
                pso_particles=6, pso_max_iter=3, seed=1, polish=False,
                backend="cpu", integrator="rk4_numpy")
    r = run_calibration(theta, P, cfg["engine"], cfg["wiebe"],
                        cfg["simulation"], c)
    # RK4 de passo fixo × solve_ivp nunca coincidem exatamente
    assert r["diferenca_integrador"] > 0.0
    assert r["diferenca_integrador"] == pytest.approx(
        abs(r["rmse"] - r["rmse_integrador"]), abs=1e-9)


def test_least_squares_valor_inicial_fora_dos_limites(ensaio):
    cfg, theta, P, _ = ensaio
    c = replace(cfg["calibration"], method="least-squares",
                selected=["theta01", "alpha"])
    lim = {"theta01": (math.radians(-5.0), math.radians(5.0))}   # -6.54° fora
    r = run_calibration(theta, P, cfg["engine"], cfg["wiebe"],
                        cfg["simulation"], c, bounds=lim)
    assert lim["theta01"][0] <= r["params"]["theta01"] <= lim["theta01"][1]
    assert any("fora dos limites" in a for a in r["alerts"])


def test_gui_limites_angulares_em_graus():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(RAIZ / "src" / "double_wiebe" / "gui.py"),
                           default_timeout=120).run()
    lo = at.number_input(key="lim_lo_theta01_°")
    hi = at.number_input(key="lim_hi_delta2_°")
    assert "[°]" in lo.label and lo.value == pytest.approx(-30.0)
    assert "[°]" in hi.label and hi.value == pytest.approx(90.0)
    rc = at.number_input(key="lim_lo_Rc_-")
    assert "[°]" not in rc.label and rc.value == pytest.approx(14.0)
