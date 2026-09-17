# -*- coding: utf-8 -*-
"""
double_wiebe
============
Núcleo científico do modelo Double Wiebe — duas fases de Wiebe acopladas ao
modelo termodinâmico de zona única (geometria biela-manivela, Hohenberg,
sistema de 3 EDOs), com calibração (DE / least-squares / PSO), CLI (Typer) e
GUI (Streamlit).

Double Wiebe Combustion Analysis
--------------------------------
This combustion simulation employs a double Wiebe function and extends the
single Wiebe model developed as part of L. Queiroz's M.Sc. thesis under the
supervision of Prof. I. L. Ferreira.

Toda a física vive neste pacote: a CLI (`cli.py`) e a GUI (`gui.py`) são
apenas capas de interação e chamam exatamente as mesmas funções.
"""
from __future__ import annotations

__version__ = "1.0.0"

DESCRIPTION_EN = (
    "Double Wiebe Combustion Analysis\n"
    "\n"
    "This combustion simulation employs a double Wiebe function and extends "
    "the single Wiebe model developed as part of L. Queiroz's M.Sc. thesis "
    "under the supervision of Prof. I. L. Ferreira."
)

from . import models  # noqa: F401
from .models import (  # noqa: F401
    EngineConfig,
    SimulationConfig,
    WiebeParameters,
    load_config,
)
from .wiebe import double_burned_fraction, phase_burned_fraction  # noqa: F401
from .simulation import run_simulation, SimulationResult  # noqa: F401
from .metrics import summary_metrics  # noqa: F401
from .calibration import run_calibration, PARAM_SPECS  # noqa: F401
from .data_processing import read_table, prepare_series  # noqa: F401

__all__ = [
    "EngineConfig", "SimulationConfig", "WiebeParameters", "load_config",
    "double_burned_fraction", "phase_burned_fraction", "run_simulation",
    "SimulationResult", "summary_metrics", "run_calibration", "PARAM_SPECS",
    "read_table", "prepare_series", "DESCRIPTION_EN", "__version__",
]