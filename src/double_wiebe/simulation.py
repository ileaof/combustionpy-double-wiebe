# -*- coding: utf-8 -*-
"""
simulation.py
=============
Orquestração da simulação Double Wiebe (núcleo usado por CLI e GUI).

Executa a integração das EDOs nos ângulos experimentais, completa com as
séries de Wiebe (fase 1, fase 2, total), taxas de liberação de calor, volume,
métricas e indicadores, e devolve um :class:`SimulationResult` imutável
(dataclass frozen) e serializável.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .geometry import cylinder_volume
from .metrics import simulation_indicators, summary_metrics
from .models import EngineConfig, SimulationConfig, WiebeParameters
from .thermodynamics import PENALTY, ODEFailure, integrate_ode
from .wiebe import double_burned_fraction


@dataclass(frozen=True)
class SimulationResult:
    """Resultado completo de uma simulação (imutável e serializável)."""
    theta: np.ndarray              # rad (ângulos experimentais)
    P_exp: np.ndarray              # kPa
    P_sim: np.ndarray              # kPa
    Tg: np.ndarray                 # K
    Q_wall: np.ndarray             # J (acumulado)
    xb1: np.ndarray                # fração queimada fase 1 [-]
    xb2: np.ndarray                # fração queimada fase 2 [-]
    xb_total: np.ndarray           # fração queimada total [-]
    dQ1: np.ndarray                # kJ/rad (fase 1)
    dQ2: np.ndarray                # kJ/rad (fase 2)
    dQ_total: np.ndarray           # kJ/rad (total)
    volume: np.ndarray             # m³
    metrics: Dict[str, float] = field(default_factory=dict)
    indicators: Dict[str, float] = field(default_factory=dict)
    engine: Dict = field(default_factory=dict)     # snapshot da configuração
    wiebe: Dict = field(default_factory=dict)
    simulation: Dict = field(default_factory=dict)
    mode: str = "continuous"
    data_summary: Dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.P_sim is not None and bool(np.all(np.isfinite(self.P_sim)))


def _snapshot(engine: EngineConfig, wiebe: WiebeParameters,
              sim: SimulationConfig) -> tuple:
    """Cópias serializáveis das configurações (para export/relatório)."""
    return (dict(vars(engine)), dict(vars(wiebe)),
            dict(vars(sim)))


def run_simulation(
    theta_exp: np.ndarray,
    P_exp: np.ndarray,
    engine: Optional[EngineConfig] = None,
    wiebe: Optional[WiebeParameters] = None,
    sim: Optional[SimulationConfig] = None,
) -> SimulationResult:
    """Roda a simulação completa e devolve :class:`SimulationResult`.

    Levanta :class:`~double_wiebe.thermodynamics.ODEFailure` (ou ValueError)
    quando a integração falha — CLI/GUI tratam e preservam o último
    resultado válido.
    """
    engine = engine or EngineConfig()
    wiebe = wiebe or WiebeParameters()
    sim = sim or SimulationConfig()

    erros = engine.validate() + wiebe.validate() + sim.validate()
    if erros:
        raise ValueError("Configuração inválida: " + " | ".join(erros))

    theta_exp = np.asarray(theta_exp, dtype=float)
    P_exp = np.asarray(P_exp, dtype=float)
    if theta_exp.size != P_exp.size or theta_exp.size < 2:
        raise ValueError("Séries experimentais devem ter o mesmo tamanho "
                         "(>= 2 pontos).")

    P1 = float(P_exp[0])
    P_sim, Tg, Q_wall = integrate_ode(
        theta_exp, engine, wiebe, P1,
        method=sim.method, rtol=sim.rtol, atol=sim.atol,
    )

    frac = double_burned_fraction(theta_exp, wiebe)
    Q_total = engine.Q_total                       # kJ/ciclo
    dQ1 = Q_total * wiebe.alpha * frac["dx1"]      # kJ/rad
    dQ2 = Q_total * (1.0 - wiebe.alpha) * frac["dx2"]
    dQ_total = Q_total * frac["dxb"]
    volume = cylinder_volume(theta_exp, engine.Rc, engine)

    metrics = summary_metrics(P_exp, P_sim)
    indicators = simulation_indicators(
        theta_exp, P_exp, P_sim, Tg, Q_wall, frac["xb"], engine, wiebe,
    )
    eng_s, wieb_s, sim_s = _snapshot(engine, wiebe, sim)

    return SimulationResult(
        theta=theta_exp, P_exp=P_exp, P_sim=P_sim, Tg=Tg, Q_wall=Q_wall,
        xb1=frac["x1"], xb2=frac["x2"], xb_total=frac["xb"],
        dQ1=dQ1, dQ2=dQ2, dQ_total=dQ_total, volume=volume,
        metrics=metrics, indicators=indicators,
        engine=eng_s, wiebe=wieb_s, simulation=sim_s,
        mode=wiebe.mode,
    )


def evaluate_rmse(
    theta_exp: np.ndarray,
    P_exp: np.ndarray,
    engine: EngineConfig,
    wiebe: WiebeParameters,
    sim: SimulationConfig,
    penalty: float = PENALTY,
) -> float:
    """RMSE puro para calibração — devolve ``penalty`` em caso de falha
    (nunca levanta exceção, nunca interrompe a otimização)."""
    from .metrics import rmse
    try:
        P1 = float(P_exp[0])
        P_sim, _, _ = integrate_ode(
            theta_exp, engine, wiebe, P1,
            method=sim.method, rtol=sim.rtol, atol=sim.atol,
        )
    except (ODEFailure, ValueError, OverflowError, FloatingPointError):
        return float(penalty)
    if not np.all(np.isfinite(P_sim)):      # NaN/inf sem exceção → penalidade
        return float(penalty)
    return rmse(P_exp, P_sim)