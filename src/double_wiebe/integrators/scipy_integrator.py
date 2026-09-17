# -*- coding: utf-8 -*-
"""
scipy_integrator.py — Implementação de referência (backend ``serial``).

Reexporta as funções originais de ``thermodynamics.py`` (solve_ivp
adaptativo, t_eval = ângulos experimentais, sem interpolação). Este é o
caminho numérico de referência contra o qual TODA otimização é validada.
"""
from ..thermodynamics import (ODEFailure, integrate_ode,  # noqa: F401
                              safe_integrate)