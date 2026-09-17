# -*- coding: utf-8 -*-
"""
test_cli.py
===========
CLI (Typer): códigos de saída, sobrescritas --set, proteção de diretório e
equivalência CLI × núcleo (a GUI usa exatamente o mesmo núcleo científico).
"""
from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from double_wiebe.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def dados(tmp_path_factory):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    return root / "data" / "example_pressure.txt", root / "configs" / "example.yaml"


@pytest.fixture(scope="module")
def simulacao_basica(dados, tmp_path_factory):
    """Um `simulate` completo reutilizado por vários testes."""
    data, config = dados
    out = tmp_path_factory.mktemp("out_sim")
    r = runner.invoke(app, ["simulate", "--data", str(data),
                            "--config", str(config),
                            "--output", str(out), "--quiet"])
    return r, out


# =============================================================================
# example-config
# =============================================================================
def test_example_config_imprime_yaml():
    r = runner.invoke(app, ["example-config"])
    assert r.exit_code == 0
    assert "engine:" in r.output and "wiebe:" in r.output


def test_example_config_grava_e_protege(tmp_path):
    destino = tmp_path / "cfg.yaml"
    r = runner.invoke(app, ["example-config", str(destino)])
    assert r.exit_code == 0 and destino.exists()
    # segunda gravação sem --force deve falhar (exit 2)
    r = runner.invoke(app, ["example-config", str(destino)])
    assert r.exit_code == 2
    r = runner.invoke(app, ["example-config", str(destino), "--force"])
    assert r.exit_code == 0


# =============================================================================
# simulate
# =============================================================================
def test_simulate_exit_0_e_exportacoes(simulacao_basica):
    r, out = simulacao_basica
    assert r.exit_code == 0
    esperados = ["results.csv", "parameters.yaml", "metrics.json",
                 "pressure_comparison.png", "pressure_comparison.pdf",
                 "heat_release.png", "burned_fraction.png", "report.html"]
    for nome in esperados:
        assert (out / nome).exists(), f"faltou {nome}"


def test_results_csv_14_colunas(simulacao_basica):
    _, out = simulacao_basica
    import pandas as pd
    df = pd.read_csv(out / "results.csv", sep=";")
    assert df.shape[1] == 14
    assert list(df.columns) == [
        "theta_rad", "theta_deg", "pressure_exp_kPa", "pressure_sim_kPa",
        "residual_kPa", "temperature_K", "wall_heat_J", "volume_m3",
        "xb_phase1", "xb_phase2", "xb_total",
        "dQ1_dtheta_kJ_per_rad", "dQ2_dtheta_kJ_per_rad",
        "dQ_total_dtheta_kJ_per_rad"]
    assert np.all(np.isfinite(df.to_numpy(float)))       # sem NaN


def test_simulate_protege_diretorio_nao_vazio(simulacao_basica, dados):
    _, out = simulacao_basica
    data, config = dados
    r = runner.invoke(app, ["simulate", "--data", str(data),
                            "--config", str(config),
                            "--output", str(out), "--quiet"])
    assert r.exit_code == 2                              # exige --overwrite


def test_simulate_dados_inexistentes_exit_2():
    r = runner.invoke(app, ["simulate", "--data", "nao_existe.txt",
                            "--quiet"])
    assert r.exit_code == 2


def test_simulate_set_override(simulacao_basica, dados, tmp_path):
    """--set wiebe.alpha muda o parâmetro exportado (CLI = mesmo núcleo)."""
    _, out0 = simulacao_basica
    import yaml
    base = yaml.safe_load((out0 / "parameters.yaml").read_text("utf-8"))
    data, config = dados
    out = tmp_path / "out_set"
    r = runner.invoke(app, ["simulate", "--data", str(data),
                            "--config", str(config), "--output", str(out),
                            "--set", "wiebe.alpha=0.55", "--quiet"])
    assert r.exit_code == 0
    novo = yaml.safe_load((out / "parameters.yaml").read_text("utf-8"))
    assert novo["wiebe"]["alpha"] == pytest.approx(0.55)
    assert base["wiebe"]["alpha"] != pytest.approx(0.55)


def test_cli_igual_ao_nucleo_cientifico(simulacao_basica, dados):
    """Equivalência CLI × núcleo: results.csv reproduz run_simulation direto."""
    _, out = simulacao_basica
    data, config = dados
    import pandas as pd
    df = pd.read_csv(out / "results.csv", sep=";")

    from double_wiebe.data_processing import prepare_series, read_table
    from double_wiebe.models import load_config
    from double_wiebe.simulation import run_simulation
    cfg = load_config(str(config))
    dfr = read_table(data)
    theta, P, _ = prepare_series(
        dfr, angle_unit="radianos", pressure_unit="bar",
        theta_min=-2.0, theta_max=2.0)
    res = run_simulation(theta, P, cfg["engine"], cfg["wiebe"],
                         cfg["simulation"])
    assert np.allclose(df["pressure_sim_kPa"], res.P_sim, rtol=1e-6)


def test_metrics_json(simulacao_basica):
    _, out = simulacao_basica
    import json
    doc = json.loads((out / "metrics.json").read_text("utf-8"))
    assert set(doc["metrics"]) >= {"SSE_kPa2", "MSE_kPa2", "RMSE_kPa",
                                   "MAE_kPa", "R2", "max_abs_error_kPa",
                                   "mean_percent_error_pct"}
    assert "calibration" not in doc      # simulate simples não tem calibração


def test_report_html_conteudo(simulacao_basica):
    _, out = simulacao_basica
    html = (out / "report.html").read_text("utf-8")
    assert "Double Wiebe" in html
    assert "data:image/png;base64," in html


# =============================================================================
# calibrate (smoke: least-squares com 1 parâmetro — rápido)
# =============================================================================
def test_calibrate_smoke_exit_0(dados, tmp_path):
    data, config = dados
    out = tmp_path / "out_calib"
    r = runner.invoke(app, [
        "calibrate", "--data", str(data), "--config", str(config),
        "--method", "least-squares", "--select", "alpha",
        "--output", str(out), "--quiet"])
    assert r.exit_code == 0
    assert (out / "convergence.csv").exists()
    import yaml
    doc = yaml.safe_load((out / "parameters.yaml").read_text("utf-8"))
    assert 0.05 <= doc["wiebe"]["alpha"] <= 0.95


def test_calibrate_selecao_invalida_exit_2(dados, tmp_path):
    data, config = dados
    r = runner.invoke(app, [
        "calibrate", "--data", str(data), "--config", str(config),
        "--method", "least-squares", "--select", "param_falso",
        "--output", str(tmp_path / "x"), "--quiet"])
    assert r.exit_code == 2


# =============================================================================
# validate e plot
# =============================================================================
def test_validate_ok_exit_0(simulacao_basica, dados):
    data, config = dados
    r = runner.invoke(app, ["validate", "--data", str(data),
                            "--config", str(config)])
    assert r.exit_code == 0
    assert "configuração válida" in r.output


def test_validate_config_invalida_exit_2(dados):
    data, config = dados
    r = runner.invoke(app, ["validate", "--data", str(data),
                            "--config", str(config),
                            "--set", "engine.kappa=0.5"])
    assert r.exit_code == 2


def test_plot_regenera_graficos(simulacao_basica, tmp_path):
    _, out = simulacao_basica
    destino = tmp_path / "plots"
    r = runner.invoke(app, ["plot", str(out), "--out", str(destino)])
    assert r.exit_code == 0
    assert (destino / "pressure_comparison.png").exists()
    assert (destino / "burned_fraction.png").exists()