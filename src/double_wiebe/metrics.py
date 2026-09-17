# -*- coding: utf-8 -*-
"""
metrics.py
==========
Métricas de ajuste e indicadores de simulação.

Métricas (P experimental vs P simulada, em kPa):

    SSE, MSE, RMSE, MAE, R², erro máximo absoluto, erro percentual médio

Indicadores (por simulação):

    pressão máxima; ângulo da pressão máxima; temperatura máxima;
    calor total liberado; calor perdido; fração queimada final;
    participação energética de cada fase.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np


def sse(P_exp, P_sim) -> float:
    """Soma dos erros quadráticos SSE = Σ (Pexp - Psim)² [kPa²]."""
    r = np.asarray(P_exp) - np.asarray(P_sim)
    return float(np.sum(r * r))


def mse(P_exp, P_sim) -> float:
    """Erro quadrático médio MSE = SSE/q [kPa²]."""
    q = np.asarray(P_exp).size
    return sse(P_exp, P_sim) / q


def rmse(P_exp, P_sim) -> float:
    """Raiz do erro quadrático médio RMSE = sqrt(MSE) [kPa]."""
    return float(np.sqrt(mse(P_exp, P_sim)))


def mae(P_exp, P_sim) -> float:
    """Erro absoluto médio MAE = mean(|Pexp - Psim|) [kPa]."""
    return float(np.mean(np.abs(np.asarray(P_exp) - np.asarray(P_sim))))


def r_squared(P_exp, P_sim) -> float:
    """Coeficiente de determinação R² = 1 - SSres/SStot [-]."""
    P_exp = np.asarray(P_exp, dtype=float)
    SStot = float(np.sum((P_exp - P_exp.mean()) ** 2))
    if SStot <= 0.0:
        return float("nan")
    SSres = float(np.sum((P_exp - np.asarray(P_sim)) ** 2))
    return 1.0 - SSres / SStot


def max_abs_error(P_exp, P_sim) -> float:
    """Erro máximo absoluto max|Pexp - Psim| [kPa]."""
    return float(np.max(np.abs(np.asarray(P_exp) - np.asarray(P_sim))))


def mean_percent_error(P_exp, P_sim) -> float:
    """Erro percentual médio = mean(|Pexp - Psim| / |Pexp|) * 100 [%]."""
    P_exp = np.asarray(P_exp, dtype=float)
    P_sim = np.asarray(P_sim, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        perc = np.abs((P_exp - P_sim) / P_exp) * 100.0
    return float(np.mean(perc[np.isfinite(perc)]))


def summary_metrics(P_exp, P_sim) -> Dict[str, float]:
    """Dicionário com todas as métricas de ajuste."""
    return {
        "SSE_kPa2": sse(P_exp, P_sim),
        "MSE_kPa2": mse(P_exp, P_sim),
        "RMSE_kPa": rmse(P_exp, P_sim),
        "MAE_kPa": mae(P_exp, P_sim),
        "R2": r_squared(P_exp, P_sim),
        "max_abs_error_kPa": max_abs_error(P_exp, P_sim),
        "mean_percent_error_pct": mean_percent_error(P_exp, P_sim),
    }


def simulation_indicators(
    theta: np.ndarray,
    P_exp: np.ndarray,
    P_sim: np.ndarray,
    Tg: np.ndarray,
    Q_wall: np.ndarray,
    xb_total: np.ndarray,
    engine,
    wiebe,
) -> Dict[str, float]:
    """Indicadores termodinâmicos da simulação.

    Inclui a participação energética de cada fase:

        E1 = alpha * Q_total * x1(theta_f)
        E2 = (1 - alpha) * Q_total * x2(theta_f)
        (x_j no último ângulo avaliado; no modo contínuo a cauda assintótica
        faz x_j -> ~1 - exp(-a_j) ≈ 0.999)
    """
    i_P = int(np.argmax(P_sim))
    i_T = int(np.argmax(Tg))
    Q_total = engine.Q_total                      # kJ/ciclo
    # Participação energética: fração final de cada fase (séries individuais)
    from .wiebe import double_burned_fraction
    frac = double_burned_fraction(theta, wiebe)
    E1 = wiebe.alpha * Q_total * float(frac["x1"][-1])
    E2 = (1.0 - wiebe.alpha) * Q_total * float(frac["x2"][-1])
    total = E1 + E2
    return {
        "P_max_kPa": float(np.max(P_sim)),
        "theta_at_P_max_rad": float(theta[i_P]),
        "theta_at_P_max_deg": float(math.degrees(theta[i_P])),
        "T_max_K": float(np.max(Tg)),
        "total_heat_released_kJ": total,
        "wall_heat_loss_J": float(Q_wall[-1]),
        "wall_heat_loss_kJ": float(Q_wall[-1] / 1000.0),
        "final_burned_fraction": float(xb_total[-1]),
        "phase1_energy_kJ": E1,
        "phase2_energy_kJ": E2,
        "phase1_energy_share_pct": 100.0 * E1 / total if total > 0 else float("nan"),
        "phase2_energy_share_pct": 100.0 * E2 / total if total > 0 else float("nan"),
    }