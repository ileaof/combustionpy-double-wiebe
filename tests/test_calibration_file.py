# -*- coding: utf-8 -*-
"""
test_calibration_file.py
========================
Salvar/abrir arquivo de calibração (JSON): round-trip fiel do resultado
e rejeição de arquivos inválidos.
"""
from __future__ import annotations

import io

import numpy as np
import pytest

from double_wiebe.models import PARAM_ORDER, EngineConfig, WiebeParameters
from double_wiebe.reporting import (calibration_json_bytes,
                                    read_calibration_json)


def _calibracao_exemplo() -> dict:
    eng, wieb = EngineConfig(), WiebeParameters()
    valores = list(np.linspace(1.0, 10.0, len(PARAM_ORDER)))
    return {
        "params": dict(zip(PARAM_ORDER, valores)),
        "params_initial": dict(zip(PARAM_ORDER,
                                   [v + 0.5 for v in valores])),
        "selected": ["theta01", "alpha"],
        "lower": {"theta01": -1.0, "alpha": 0.0},
        "upper": {"theta01": 1.0, "alpha": 1.0},
        "rmse": 74.79,
        "objective_history": [280.0, 74.79],
        "history_params": [np.asarray(valores) * 1.01,
                           np.asarray(valores)],
        "method": "PSO",
        "seed": 42,
        "iteracoes": 190,
        "message": "otimização concluída",
        "success": True,
        "cancelado": False,
        "parou_por_estagnacao": False,
        "polish": {"rmse": 74.79, "aplicado": True},
        "sensitivity": [{"param": "alpha", "delta_rmse_pct": 1.2,
                         "insensitive": False}],
        "alerts": ["⚠ teste"],
        "engine": dict(vars(eng)),
        "wiebe": dict(vars(wieb)),
        "theta_exp": np.linspace(-2.0, 2.0, 20),
        "P_exp": np.linspace(90.0, 7000.0, 20),
        "backend": "serial",
        "rmse_integrador": 74.79,
        "diferenca_integrador": 0.0,
        "tempo_s": 12.5,
    }


def test_calibracao_json_roundtrip():
    cal = _calibracao_exemplo()
    b = calibration_json_bytes(cal, data_name="dado.txt")
    aberto = read_calibration_json(io.BytesIO(b))
    c2 = aberto["calibracao"]

    assert c2["params"] == cal["params"]
    assert c2["params_initial"] == cal["params_initial"]
    assert c2["selected"] == cal["selected"]
    assert c2["rmse"] == pytest.approx(cal["rmse"])
    assert c2["method"] == "PSO"
    assert c2["seed"] == 42
    assert c2["iteracoes"] == 190
    assert c2["polish"] == cal["polish"]
    assert c2["sensitivity"] == cal["sensitivity"]
    assert c2["alerts"] == cal["alerts"]
    assert c2["backend"] == "serial"
    assert c2["tempo_s"] == pytest.approx(12.5)
    assert np.allclose(c2["objective_history"], cal["objective_history"])
    assert np.allclose(c2["history_params"][0], cal["history_params"][0])
    # configs de motor/Wiebe restauradas 1:1 (rastreabilidade)
    assert c2["engine"] == cal["engine"]
    assert c2["wiebe"] == cal["wiebe"]
    # dados embutidos
    assert np.allclose(aberto["theta"], cal["theta_exp"])
    assert np.allclose(aberto["pressao"], cal["P_exp"])
    assert aberto["motor"] == cal["engine"]
    assert aberto["wiebe"] == cal["wiebe"]
    assert aberto["arquivo_experimental"] == "dado.txt"


def test_calibracao_json_sem_dados_e_sem_iniciais():
    cal = _calibracao_exemplo()
    for chave in ("theta_exp", "P_exp", "params_initial",
                  "objective_history", "history_params"):
        cal.pop(chave)
    aberto = read_calibration_json(io.BytesIO(calibration_json_bytes(cal)))
    c2 = aberto["calibracao"]
    assert aberto["theta"] is None and aberto["pressao"] is None
    assert c2["params_initial"] == c2["params"]   # fallback
    assert c2["objective_history"] == []
    assert c2["history_params"] == []


@pytest.mark.parametrize("conteudo", [
    b'{"qualquer": 1}',
    b"nao e json",
    b'{"analise": {"formato": "double-wiebe-calibracao"}}',  # sem calibracao
])
def test_calibracao_json_rejeita_arquivo_invalido(conteudo):
    with pytest.raises(ValueError):
        read_calibration_json(io.BytesIO(conteudo))