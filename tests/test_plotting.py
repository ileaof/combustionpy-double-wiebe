# -*- coding: utf-8 -*-
"""
test_plotting.py
================
Gráficos interativos (Plotly): o diagrama P–V deve mostrar os pontos
experimentais além da curva do modelo.
"""
from __future__ import annotations

import numpy as np

from double_wiebe.plotting import fig_pv
from double_wiebe.simulation import run_simulation


def test_fig_pv_mostra_pontos_experimentais(synthetic, engine, wiebe, sim_cfg):
    theta, P_exp, _, _ = synthetic
    res = run_simulation(theta, P_exp, engine, wiebe, sim_cfg)
    for unidade, fator in (("kPa", 1.0), ("bar", 0.01)):
        fig = fig_pv(res, unidade)
        assert len(fig.data) == 2
        modelo, exp = fig.data[0], fig.data[1]
        assert modelo.mode == "lines"
        assert exp.mode == "markers"
        assert np.allclose(exp.y, res.P_exp * fator)
        assert np.allclose(exp.x, res.volume)