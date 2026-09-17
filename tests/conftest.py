# -*- coding: utf-8 -*-
"""Fixtures compartilhadas da suíte de testes do Double Wiebe."""
from __future__ import annotations

import numpy as np
import pytest

from double_wiebe.models import (CalibrationConfig, EngineConfig,
                                 SimulationConfig, WiebeParameters)
from double_wiebe.simulation import run_simulation

REPO = None  # preenchido em repo_root()


def repo_root() -> str:
    """Raiz do projeto (pasta que contém data/ e configs/)."""
    from pathlib import Path
    return str(Path(__file__).resolve().parents[1])


@pytest.fixture(scope="session")
def engine() -> EngineConfig:
    return EngineConfig()


@pytest.fixture(scope="session")
def wiebe() -> WiebeParameters:
    return WiebeParameters()


@pytest.fixture(scope="session")
def sim_cfg() -> SimulationConfig:
    # tolerância mais frouxa nos testes (velocidade), núcleo idêntico
    return SimulationConfig(method="LSODA", rtol=1e-8, atol=1e-8)


@pytest.fixture(scope="session")
def synthetic(sim_cfg):
    """Série sintética gerada pelo próprio modelo (θ em rad, P em kPa).

    O "experimental" é a saída do modelo com os parâmetros padrão — a
    calibração deve então recuperar os parâmetros verdadeiros com RMSE ~ 0.
    """
    theta = np.linspace(-2.0, 2.0, 101)
    engine = EngineConfig()
    w = WiebeParameters()
    res = run_simulation(theta, np.full_like(theta, 138.2), engine, w, sim_cfg)
    return theta, res.P_sim.copy(), engine, w


@pytest.fixture(scope="session")
def quick_calib() -> CalibrationConfig:
    """Configuração de calibração rápida (poucos parâmetros e iterações)."""
    return CalibrationConfig(
        method="differential-evolution",
        selected=["delta1", "alpha"],
        seed=42, maxiter=4, popsize=4, tol=1e-10, polish=False,
        w_order=1e4, w_overlap=1e4, w_min_duration=1e4, delta_min_deg=5.0,
    )