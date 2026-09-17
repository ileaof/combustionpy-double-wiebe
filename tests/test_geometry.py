# -*- coding: utf-8 -*-
"""
test_geometry.py
================
Cinemática biela-manivela: volume, derivada do volume e área de calor.
"""
from __future__ import annotations

import numpy as np
import pytest

from double_wiebe.geometry import (cylinder_volume, dV_dtheta, heat_area,
                                   piston_disp)
from double_wiebe.models import EngineConfig


@pytest.fixture
def engine() -> EngineConfig:
    return EngineConfig()      # 86/70/117.5 mm, Rc = 17


THETA = np.linspace(-np.pi, np.pi, 721)


def test_volume_displaced(engine):
    Vd_esperado = np.pi * (0.086 / 2) ** 2 * 0.070
    assert engine.Vd == pytest.approx(Vd_esperado, rel=1e-12)
    # Vc = Vd/(Rc-1)
    assert engine.Vc == pytest.approx(Vd_esperado / 16.0, rel=1e-12)


def test_volume_nos_extremos_de_curso(engine):
    # PMI (θ = π): V = Vc + Vd ; PMS (θ = 0): V = Vc
    assert cylinder_volume(np.array([np.pi]), engine.Rc, engine)[0] == \
        pytest.approx(engine.Vc + engine.Vd, rel=1e-9)
    assert cylinder_volume(np.array([0.0]), engine.Rc, engine) == \
        pytest.approx(engine.Vc, rel=1e-9)
    # razão de compressão V(PMI)/V(PMS) = Rc
    razao = cylinder_volume(np.array([np.pi]), engine.Rc, engine)[0] / \
        cylinder_volume(np.array([0.0]), engine.Rc, engine)[0]
    assert razao == pytest.approx(17.0, rel=1e-9)


def test_deslocamento_do_pistao(engine):
    y = piston_disp(np.array([0.0, np.pi]), engine.Rc, engine)
    assert y[0] == pytest.approx(0.0, abs=1e-12)      # PMS
    assert y[1] == pytest.approx(engine.stroke, rel=1e-9)  # PMI


def test_dv_dtheta_vs_numerica(engine):
    h = 1e-6
    t = np.linspace(0.3, 2.5, 100)
    analit = dV_dtheta(t, engine.Rc, engine)
    numer = (cylinder_volume(t + h, engine.Rc, engine)
             - cylinder_volume(t - h, engine.Rc, engine)) / (2 * h)
    assert np.allclose(analit, numer, rtol=1e-6, atol=1e-12)


def test_area_de_calor(engine):
    # A_s = 2*A_p + A_p * (y(θ))' parcial — no mínimo 2*A_p (cabeça+fundo)
    A = heat_area(THETA, engine.Rc, engine)
    assert np.all(A >= 2.0 * engine.A_p - 1e-12)
    assert np.all(np.isfinite(A))
    # simetria: A(-θ) = A(θ)
    assert np.allclose(A, heat_area(-THETA, engine.Rc, engine), atol=1e-12)