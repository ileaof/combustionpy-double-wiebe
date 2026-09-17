# -*- coding: utf-8 -*-
"""
benchmark.py — Benchmark reproduzível dos backends (plano HPC, regra 7).

Compara serial / cpu (RK4 lote NumPy) / cpu (Numba) / cpu-parallel para
tamanhos de população variados, com warm-up, média e desvio-padrão, erro
máximo contra a referência serial e gravação opcional em CSV. Também fornece
a seleção automática (`select_backend("auto")`) por mini-benchmark.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..models import PARAM_ORDER
from .serial_backend import (CPUBackend, MultiprocessingBackend,
                             SerialBackend)

_TAMANHOS = (1, 20, 100, 1000)


def _populacao(S: int, nomes, seed: int = 42) -> np.ndarray:
    """População sintética dentro dos limites padrão dos parâmetros."""
    from ..calibration import PARAM_SPECS
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(PARAM_SPECS[n]["lower"], PARAM_SPECS[n]["upper"], S)
        for n in nomes])


def _expand(X_sel: np.ndarray, idx, full0: np.ndarray) -> np.ndarray:
    out = np.tile(full0, (X_sel.shape[0], 1))
    out[:, idx] = X_sel
    return out


def run_benchmark(theta, P_exp, engine, wiebe, sim, calib,
                  selected, full0,
                  tamanhos=_TAMANHOS, rep=3, warmup=True,
                  csv_path: Optional[str] = None,
                  substeps: int = 4) -> List[dict]:
    """Mede cada backend disponível para cada tamanho de população.

    Retorna linhas (dict): N, backend, tempo médio (s), desvio (s),
    ms por candidato, speedup vs serial, max |erro| vs serial (kPa,
    candidatos válidos nos dois lados) e igualdade das penalidades.
    """
    idx = [PARAM_ORDER.index(n) for n in selected]
    backends = {"serial": SerialBackend()}
    backends["cpu (rk4 numpy)"] = CPUBackend(integrator="rk4_numpy")
    try:
        import numba                                    # opcional [cpu]
        del numba
        backends["cpu (rk4 numba)"] = CPUBackend(integrator="rk4_numba")
    except ImportError:
        pass
    mpb = MultiprocessingBackend()
    if mpb.is_available():
        backends["cpu-parallel"] = mpb

    linhas: List[dict] = []
    for S in tamanhos:
        X_sel = _populacao(S, list(selected))
        X_full = _expand(X_sel, idx, full0)
        tempos, refs, desvios = {}, {}, {}
        for nome, b in backends.items():
            ts: List[float] = []
            f = None
            for r in range(rep + 1):
                t0 = time.perf_counter()
                f = b.evaluate_population(X_full, theta, P_exp, engine,
                                          wiebe, sim, calib, substeps=substeps)
                dt = time.perf_counter() - t0
                if warmup and r == 0:
                    continue                     # descarta (aquecimento)
                ts.append(dt)
            tempos[nome] = float(np.mean(ts))
            desvios[nome] = float(np.std(ts))
            refs[nome] = f
            if nome == "serial":
                f_serial = f            # referência de erro deste tamanho
        for nome, t in tempos.items():
            ok = (refs[nome] < 1e9) & (f_serial < 1e9)
            linha = {
                "N": S,
                "backend": nome,
                "tempo_s": t,
                "desvio_s": desvios[nome],
                "ms_por_candidato": t / S * 1000.0,
                "speedup_vs_serial": (tempos["serial"] / t)
                if tempos.get("serial") else np.nan,
                "max_erro_vs_serial_kPa": (
                    float(np.max(np.abs(refs[nome][ok] - f_serial[ok])))
                    if ok.any() else np.nan),
                "penalidades_iguais": bool(
                    ((refs[nome] >= 1e9) == (f_serial >= 1e9)).all()),
            }
            linhas.append(linha)
    if csv_path:
        p = Path(csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
            w.writeheader()
            w.writerows(linhas)
    return linhas


def print_benchmark(linhas) -> str:
    """Formata a tabela de benchmark em texto."""
    out = [f"{'N':>6} {'backend':<18} {'tempo (s)':>10} {'+-s':>8} "
           f"{'ms/cand':>9} {'speedup':>8} {'max err kPa':>12} {'pen ok':>7}"]
    out.append("-" * 86)
    for l in linhas:
        sp = l["speedup_vs_serial"]
        err = l["max_erro_vs_serial_kPa"]
        out.append(
            f"{l['N']:>6} {l['backend']:<18} {l['tempo_s']:>10.4f} "
            f"{l['desvio_s']:>8.4f} {l['ms_por_candidato']:>9.2f} "
            f"{('%0.1fx' % sp) if np.isfinite(sp) else '-':>8} "
            f"{('%0.4f' % err) if np.isfinite(err) else '-':>12} "
            f"{str(l['penalidades_iguais']):>7}")
    return "\n".join(out)


def choose_backend(workers: Optional[int] = None, substeps: int = 4,
                   n_test: int = 60):
    """Mini-benchmark com população pequena: escolhe o mais rápido.
    Retorna a INSTÂNCIA pronta para uso (pool já aquecido)."""
    from ..models import (CalibrationConfig, DEFAULT_SELECTED, EngineConfig,
                          SimulationConfig, WiebeParameters)

    # dado sintético leve (grade uniforme) — só para medir tempo relativo
    theta = np.linspace(-2.0, 2.0, 120)
    P = np.exp(1.37 * (2.0 - theta)) * 138.2          # forma plausível
    engine, wiebe = EngineConfig(), WiebeParameters()
    sim, calib = SimulationConfig(), CalibrationConfig()
    full0 = np.array([engine.Rc, wiebe.theta01, wiebe.delta1, wiebe.m1,
                      wiebe.a1, wiebe.theta02, wiebe.delta2, wiebe.m2,
                      wiebe.a2, wiebe.alpha])
    idx = [PARAM_ORDER.index(n) for n in DEFAULT_SELECTED]
    X_sel = _populacao(n_test, list(DEFAULT_SELECTED))
    X_full = _expand(X_sel, idx, full0)

    candidatos = {"cpu (rk4 numba)": CPUBackend(integrator="rk4_numba"),
                  "cpu (rk4 numpy)": CPUBackend(integrator="rk4_numpy")}
    try:
        import numba                                   # opcional
        del numba
        candidatos.pop("cpu (rk4 numpy)")
    except ImportError:
        candidatos.pop("cpu (rk4 numba)", None)
    mpb = MultiprocessingBackend(workers=workers)
    if mpb.is_available():
        candidatos["cpu-parallel"] = mpb

    melhor, t_melhor = None, np.inf
    for nome, b in candidatos.items():
        t0 = time.perf_counter()
        b.evaluate_population(X_full, theta, P, engine, wiebe, sim, calib,
                              substeps=substeps)
        t = time.perf_counter() - t0
        if t < t_melhor:
            melhor, t_melhor = b, t
    return melhor if melhor is not None else SerialBackend()