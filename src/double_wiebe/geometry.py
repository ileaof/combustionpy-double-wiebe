# -*- coding: utf-8 -*-
"""
geometry.py
===========
Geometria biela-manivela do motor (volume, derivada, deslocamento do pistão
e área de transferência de calor).

Unidades: theta em rad, comprimentos em m, volume em m³.

    A_p   = pi*(d/2)²                    área do pistão
    r     = s/2                          raio da manivela
    R     = l/r                          razão biela/manivela
    Vd    = A_p*s                        volume deslocado
    Vc    = Vd/(Rc - 1)                  volume de folga

    V(theta)     = Vc + (Vd/2)*[R + 1 - cos(theta) - sqrt(R² - sin²(theta))]
    dV/dtheta    = (Vd*sin(theta)/2)*[1 + cos(theta)/sqrt(R² - sin²(theta))]
    y(theta)     = l + r - r*cos(theta) - sqrt(l² - r²*sin²(theta))
    A_s(theta)   = 2*pi*(d/2)² + pi*d*y(theta) + pi*d*s/(Rc - 1)
"""
from __future__ import annotations

import numpy as np

from .models import EngineConfig


def piston_disp(theta, Rc: float, cfg: EngineConfig) -> np.ndarray:
    """y(theta): deslocamento do pistão a partir do PMS [m]."""
    l, r = cfg.rod_length, cfg.r_crank
    return l + r - r * np.cos(theta) - np.sqrt(l**2 - r**2 * np.sin(theta) ** 2)


def cylinder_volume(theta, Rc: float, cfg: EngineConfig) -> np.ndarray:
    """V(theta): volume instantâneo do cilindro [m³]."""
    Vd, R = cfg.Vd, cfg.R
    return Vd / (Rc - 1.0) + (Vd / 2.0) * (
        R + 1.0 - np.cos(theta) - np.sqrt(R**2 - np.sin(theta) ** 2)
    )


def dV_dtheta(theta, Rc: float, cfg: EngineConfig) -> np.ndarray:
    """dV/dtheta [m³/rad]."""
    Vd, R = cfg.Vd, cfg.R
    return (Vd * np.sin(theta) / 2.0) * (
        1.0 + np.cos(theta) / np.sqrt(R**2 - np.sin(theta) ** 2)
    )


def heat_area(theta, Rc: float, cfg: EngineConfig) -> np.ndarray:
    """A_s(theta): área de transferência de calor [m²]."""
    return (
        2.0 * np.pi * (cfg.bore / 2.0) ** 2
        + np.pi * cfg.bore * piston_disp(theta, Rc, cfg)
        + np.pi * cfg.bore * cfg.stroke / (Rc - 1.0)
    )