# -*- coding: utf-8 -*-
"""
data_processing.py
==================
Leitura flexível e preparação dos dados experimentais (ângulo, pressão).

Aceita ``.txt``, ``.csv`` e ``.tsv``. Converte internamente para as unidades
do modelo (ângulo em rad; pressão em kPa), valida, limpa e ordena. Nada é
modificado silenciosamente: cada remoção/ajuste é contado no resumo
(``resumo``) e erros bloqueantes levantam :class:`DataError`.
"""
from __future__ import annotations

import csv
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Fatores de conversão para as unidades internas do modelo
PRESSURE_FACTORS_KPA = {"Pa": 0.001, "kPa": 1.0, "bar": 100.0}
ANGLE_FACTORS_RAD = {"radianos": 1.0, "graus": np.pi / 180.0}


class DataError(Exception):
    """Erro de leitura/validação com mensagem amigável para CLI/GUI."""


def _peek_text(path_or_buffer) -> str:
    """Primeiros ~4 KB do conteúdo (caminho OU buffer), para detecção
    automática de separador/cabeçalho — sem consumir o buffer."""
    if hasattr(path_or_buffer, "read"):
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
        sample = path_or_buffer.read(4096)
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
        if isinstance(sample, bytes):
            sample = sample.decode("utf-8", errors="replace")
        return sample
    from pathlib import Path
    return Path(path_or_buffer).read_text(encoding="utf-8",
                                          errors="replace")[:4096]


def read_table(
    path_or_buffer,
    sep: str = "auto",
    has_header: Optional[bool] = None,
) -> pd.DataFrame:
    """Lê o arquivo experimental em um DataFrame numérico.

    Parâmetros
    ----------
    path_or_buffer : caminho ou buffer (StringIO/BytesIO) com o texto.
    sep : separador — "auto" (csv.Sniffer, fallback whitespace), "whitespace"
        ou um caractere fixo (",", ";", "\\t"...).
    has_header : None (auto-detecção: 1ª linha não numérica => cabeçalho),
        False (sem cabeçalho) ou True (descarta a 1ª linha).

    As células não numéricas viram NaN (o tratamento é feito em
    :func:`prepare_series`, que reporta quantas foram descartadas).
    """
    try:
        try:
            if sep == "auto":
                sample = _peek_text(path_or_buffer)
                try:
                    dialect_sep = csv.Sniffer().sniff(sample).delimiter
                except csv.Error:
                    dialect_sep = None
                # whitespace puro (" ") confunde o Sniffer -> pandas regex
                sep_used = r"\s+" if dialect_sep in (None, "", " ") else dialect_sep
            else:
                sep_used = r"\s+" if sep == "whitespace" else sep

            if has_header is True:
                header = 0
            elif has_header is False:
                header = None
            else:
                # auto: peek na 1ª linha — se contiver texto não numérico,
                # considera cabeçalho.
                peek = _peek_text(path_or_buffer)
                primeira = peek.splitlines()[0] if peek.strip() else ""
                campos = primeira.split(sep_used if sep_used != r"\s+" else None)
                header = 0 if any(
                    _nao_numerico(c) for c in campos if c.strip()
                ) else None
        except Exception:
            sep_used = r"\s+"
            header = None

        df = pd.read_csv(
            path_or_buffer, sep=sep_used, header=header,
            engine="python", comment="#", dtype=str,
        )
        for col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip().str.replace(",", ".", regex=False),
                errors="coerce",
            )
        df = df.dropna(axis=1, how="all")   # colunas totalmente vazias
        if df.shape[1] < 2:
            raise DataError(
                "Não foi possível identificar 2 colunas numéricas (ângulo e "
                "pressão). Verifique o separador/cabeçalho."
            )
        return df
    except pd.errors.EmptyDataError as e:
        raise DataError("O arquivo está vazio.") from e
    except UnicodeDecodeError as e:
        raise DataError("Não foi possível decodificar o arquivo como texto "
                        "(UTF-8/ASCII).") from e


def _nao_numerico(texto: str) -> bool:
    """True se o campo não pode ser interpretado como número."""
    try:
        float(str(texto).strip().replace(",", "."))
        return False
    except (ValueError, TypeError):
        return True


def prepare_series(
    df: pd.DataFrame,
    angle_col: int = 0,
    pressure_col: int = 1,
    angle_unit: str = "radianos",
    pressure_unit: str = "kPa",
    theta_min: Optional[float] = None,
    theta_max: Optional[float] = None,
    remove_invalid: bool = True,
    sort_by_angle: bool = True,
    min_points: int = 10,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Converte, filtra e valida as séries angulares de pressão.

    Retorna (theta_rad, P_kPa, resumo). Levanta :class:`DataError` com
    mensagem clara quando algo impede a análise. O resumo reporta todas as
    limpezas efetuadas (nada silencioso).
    """
    if angle_col >= df.shape[1] or pressure_col >= df.shape[1]:
        raise DataError(
            f"Coluna selecionada fora do arquivo (o arquivo tem "
            f"{df.shape[1]} colunas numéricas)."
        )

    theta = df.iloc[:, angle_col].to_numpy(dtype=float)
    P = df.iloc[:, pressure_col].to_numpy(dtype=float)

    if remove_invalid:
        valid = np.isfinite(theta) & np.isfinite(P)
        n_dropped = int((~valid).sum())
        theta, P = theta[valid].copy(), P[valid].copy()
    else:
        n_dropped = 0
        if not (np.isfinite(theta).all() and np.isfinite(P).all()):
            raise DataError(
                "O arquivo contém linhas não numéricas (NaN/inf). "
                "Ative 'Remover linhas inválidas'."
            )

    if theta.size == 0:
        raise DataError("Nenhuma observação numérica válida no arquivo.")

    if angle_unit not in ANGLE_FACTORS_RAD:
        raise DataError(f"Unidade angular desconhecida: {angle_unit} "
                        f"(opções: {list(ANGLE_FACTORS_RAD)}).")
    if pressure_unit not in PRESSURE_FACTORS_KPA:
        raise DataError(f"Unidade de pressão desconhecida: {pressure_unit} "
                        f"(opções: {list(PRESSURE_FACTORS_KPA)}).")
    theta = theta * ANGLE_FACTORS_RAD[angle_unit]
    P = P * PRESSURE_FACTORS_KPA[pressure_unit]

    mask = np.ones_like(theta, dtype=bool)
    if theta_min is not None:
        mask &= theta >= float(theta_min)
    if theta_max is not None:
        mask &= theta <= float(theta_max)
    n_filtered = int((~mask).sum())
    theta, P = theta[mask].copy(), P[mask].copy()
    if theta.size < min_points:
        raise DataError(
            f"Somente {theta.size} observações no intervalo angular "
            f"selecionado — mínimo exigido: {min_points}."
        )

    if sort_by_angle:
        order = np.argsort(theta)
        theta, P = theta[order], P[order]
    elif np.any(np.diff(theta) < 0):
        raise DataError(
            "Ângulos não estão ordenados. Ative 'Ordenar por ângulo'.")

    if not (np.isfinite(theta).all() and np.isfinite(P).all()):
        raise DataError("Séries contêm NaN/inf após o processamento.")
    if np.any(P <= 0):
        raise DataError("Pressões não positivas no conjunto (P <= 0).")

    resumo = {
        "n_obs": int(theta.size),
        "n_descartadas": n_dropped,
        "n_fora_intervalo": n_filtered,
        "theta_min": float(theta.min()),
        "theta_max": float(theta.max()),
        "P_min": float(P.min()),
        "P_max": float(P.max()),
        "P1": float(P[0]),   # pressão no IVC (primeira observação ordenada)
        "passo_medio": (float(np.mean(np.diff(np.sort(theta))))
                        if theta.size > 1 else 0.0),
    }
    return theta, P, resumo