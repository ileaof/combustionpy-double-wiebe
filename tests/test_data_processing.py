# -*- coding: utf-8 -*-
"""
test_data_processing.py
=======================
Leitura de tabelas experimentais, conversão de unidades, filtros e erros.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from double_wiebe.data_processing import (DataError, PRESSURE_FACTORS_KPA,
                                          prepare_series, read_table)


def _df(texto: str) -> pd.DataFrame:
    return read_table(io.StringIO(texto))


# =============================================================================
# read_table
# =============================================================================
def test_sep_virgula_auto():
    df = _df("0.0,100\n0.1,110\n0.2,120\n")
    assert df.shape == (3, 2)


def test_sep_ponto_virgula_auto():
    df = _df("0.0;100\n0.1;110\n0.2;120\n")
    assert df.shape == (3, 2)


def test_sep_whitespace_auto():
    df = _df("0.0 100\n0.1 110\n0.2 120\n")
    assert df.shape == (3, 2)


def test_tab_como_separador():
    df = read_table(io.StringIO("0.0\t100\n0.1\t110\n"), sep="\t")
    assert df.shape == (2, 2)


def test_csv_com_decimal_virgula():
    # separador explícito ';' (o auto-sniffer é ambíguo com decimal-vírgula)
    df = read_table(io.StringIO("0,1;120,5\n0,2;130,5\n0,3;140,5\n"), sep=";")
    assert df.iloc[1, 1] == pytest.approx(130.5)


def test_arquivo_vazio_falha():
    with pytest.raises(DataError):
        read_table(io.StringIO(""))


# =============================================================================
# prepare_series
# =============================================================================
def test_conversao_de_unidades():
    df = pd.DataFrame({"theta": [0.0, 1.0], "p_bar": [1.0, 2.0]})
    theta, P, _ = prepare_series(df, angle_unit="radianos",
                                 pressure_unit="bar", min_points=2)
    assert np.allclose(P, [100.0, 200.0])       # bar -> kPa


def test_conversao_graus_para_radianos():
    df = pd.DataFrame({"theta_deg": [0.0, 180.0], "p_kPa": [100.0, 100.0]})
    theta, _, _ = prepare_series(df, angle_unit="graus", min_points=2)
    assert theta[1] == pytest.approx(np.pi, rel=1e-6)


def test_unidades_desconhecidas_falham():
    df = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 2.0]})
    with pytest.raises(DataError):
        prepare_series(df, pressure_unit="psi")


def test_remocao_de_linhas_invalidas():
    df = pd.DataFrame({"a": [0.0, 1.0, np.nan, 2.0],
                       "b": [100.0, np.nan, 110.0, 120.0]})
    theta, P, resumo = prepare_series(df, min_points=2)
    assert resumo["n_obs"] == 2
    assert resumo["n_descartadas"] == 2


def test_linhas_invalidas_sem_remocao_falham():
    df = pd.DataFrame({"a": [0.0, np.nan, 2.0], "b": [1.0, 2.0, 3.0]})
    with pytest.raises(DataError):
        prepare_series(df, remove_invalid=False)


def test_filtro_de_intervalo():
    df = pd.DataFrame({"a": np.linspace(0, 2, 21), "b": np.full(21, 100.0)})
    theta, P, resumo = prepare_series(df, theta_min=0.5, theta_max=1.5)
    assert resumo["n_fora_intervalo"] > 0
    assert theta.min() >= 0.0 and theta.max() <= 2.0


def test_poucos_pontos_falham():
    df = pd.DataFrame({"a": [0.0, 1.0], "b": [100.0, 110.0]})
    with pytest.raises(DataError):
        prepare_series(df, min_points=10)


def test_pressao_nao_positiva_falha():
    df = pd.DataFrame({"a": np.linspace(0, 2, 11), "b": np.full(11, 0.0)})
    with pytest.raises(DataError):
        prepare_series(df)


def test_ordenacao_por_angulo():
    df = pd.DataFrame({"a": [2.0, 1.0, 0.0], "b": [3.0, 2.0, 1.0]})
    theta, P, _ = prepare_series(df, sort_by_angle=True, min_points=2)
    assert np.all(np.diff(theta) >= 0)
    assert P[0] == pytest.approx(1.0)


def test_colunas_fora_do_arquivo():
    df = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 2.0]})
    with pytest.raises(DataError):
        prepare_series(df, pressure_col=5)


def test_pressao_em_pa():
    df = pd.DataFrame({"a": [0.0, 1.0], "b": [138200.0, 138200.0]})
    _, P, _ = prepare_series(df, pressure_unit="Pa", min_points=2)
    assert np.allclose(P, [138.2, 138.2])


def test_fatores_registrados():
    assert PRESSURE_FACTORS_KPA["bar"] == 100.0
    assert PRESSURE_FACTORS_KPA["Pa"] == 0.001
    assert PRESSURE_FACTORS_KPA["kPa"] == 1.0