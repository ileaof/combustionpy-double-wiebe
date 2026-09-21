# -*- coding: utf-8 -*-
"""
calibration.py
==============
Calibração do modelo Double Wiebe.

Parâmetros calibráveis (individualmente selecionáveis):

    Rc, theta01, delta1, m1, a1, theta02, delta2, m2, a2, alpha

Métodos:

    * ``differential-evolution`` — scipy.optimize.differential_evolution
      (busca global, população/iterações/tolerância/semente configuráveis);
    * ``pso`` — enxame de partículas sem inércia (x += 2r1*(pbest-x) +
      2r2*(gbest-x), clamp nos limites, parada por estagnação), opcional;
    * ``least-squares`` — refinamento local (trust-region, vetor de
      resíduos P_sim - P_exp) a partir dos valores iniciais.

Após a busca global (DE/PSO) é oferecido um refinamento least-squares
opcional. A função objetivo principal é o RMSE entre pressão experimental
e simulada, mais penalidades de regularização configuráveis:

    * theta02 >= theta01            (ordem das fases)
    * sobreposição forte das fases  (fase 2 começa antes de theta01+delta1)
    * durações mínimas              (delta_j >= delta_min)
    * ridge (opcional)              (afasta os parâmetros das bordas)

Ao final é gerada uma ANÁLISE DE SENSIBILIDADE (perturbação ±1% em cada
parâmetro livre) e alertas de identificabilidade: quando a perturbação
praticamente não altera o RMSE, ou quando o ótimo fica preso na borda do
domínio, o resultado traz avisos explícitos — a recomendação do projeto é
NÃO calibrar os 10 parâmetros simultaneamente sem inspecionar esses avisos.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .models import (
    PARAM_ORDER, CalibrationConfig, EngineConfig, SimulationConfig,
    WiebeParameters,
)
from .simulation import evaluate_rmse
from .thermodynamics import PENALTY, ODEFailure, integrate_ode

ProgressCallback = Callable[[int, float, np.ndarray], None]
CancelCheck = Callable[[], bool]


# =============================================================================
# Registro de parâmetros (nome -> rótulo, unidade, limites padrão)
# =============================================================================
PARAM_SPECS: Dict[str, Dict] = {
    "Rc":      {"label": "Razão de compressão",      "unit": "-",
                "lower": 14.0, "upper": 20.0},
    "theta01": {"label": "Início fase 1",            "unit": "rad",
                "lower": math.radians(-30.0), "upper": math.radians(10.0)},
    "delta1":  {"label": "Duração fase 1",           "unit": "rad",
                "lower": math.radians(2.0), "upper": math.radians(40.0)},
    "m1":      {"label": "Forma fase 1",             "unit": "-",
                "lower": 0.05, "upper": 3.0},
    "a1":      {"label": "Eficiência fase 1",        "unit": "-",
                "lower": 1.0, "upper": 10.0},
    "theta02": {"label": "Início fase 2",            "unit": "rad",
                "lower": math.radians(-20.0), "upper": math.radians(40.0)},
    "delta2":  {"label": "Duração fase 2",           "unit": "rad",
                "lower": math.radians(5.0), "upper": math.radians(90.0)},
    "m2":      {"label": "Forma fase 2",             "unit": "-",
                "lower": 0.05, "upper": 3.0},
    "a2":      {"label": "Eficiência fase 2",        "unit": "-",
                "lower": 1.0, "upper": 10.0},
    "alpha":   {"label": "Fração de energia fase 1", "unit": "-",
                "lower": 0.05, "upper": 0.95},
}

PARAM_UNITS = {k: v["unit"] for k, v in PARAM_SPECS.items()}


class CalibrationError(Exception):
    """Erro de configuração da calibração (limites, seleção, etc.)."""


def build_bounds(
    selected: List[str],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Monta vetores (lower, upper) para os parâmetros selecionados.

    ``bounds`` sobrescreve limites individuais: {"Rc": (15.0, 17.0), ...}.
    """
    if not selected:
        raise CalibrationError("Nenhum parâmetro selecionado para calibrar.")
    desconhecidos = [s for s in selected if s not in PARAM_SPECS]
    if desconhecidos:
        raise CalibrationError(
            f"Parâmetros desconhecidos: {desconhecidos}. "
            f"Válidos: {PARAM_ORDER}.")
    bounds = bounds or {}
    lower, upper = [], []
    for nome in selected:
        lo, hi = bounds.get(nome, (PARAM_SPECS[nome]["lower"],
                                   PARAM_SPECS[nome]["upper"]))
        lo, hi = float(lo), float(hi)
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            raise CalibrationError(
                f"Limites inválidos para {nome}: ({lo}, {hi}) — é exigido "
                "lower < upper.")
        lower.append(lo)
        upper.append(hi)
    return list(selected), np.array(lower), np.array(upper)


def _params_from_vector(
    x_sel: np.ndarray, selected: List[str],
    base_engine: EngineConfig, base_wiebe: WiebeParameters,
) -> Tuple[EngineConfig, WiebeParameters]:
    """Reconstrói (engine, wiebe) com o subvetor otimizado aplicado."""
    wiebe = WiebeParameters(**vars(base_wiebe))
    engine = EngineConfig(**vars(base_engine))
    for valor, nome in zip(x_sel, selected):
        if nome == "Rc":
            engine.Rc = float(valor)
        elif nome in ("theta01", "delta1", "m1", "a1",
                      "theta02", "delta2", "m2", "a2", "alpha"):
            setattr(wiebe, nome, float(valor))
        else:  # pragma: no cover — build_bounds já valida os nomes
            raise CalibrationError(f"Parâmetro desconhecido '{nome}'.")
    return engine, wiebe


def _regularization(x_full: np.ndarray, reg_cfg: CalibrationConfig) -> float:
    """Penalidades de regularização (objetivo = RMSE + regularização).

    ``x_full`` é o vetor completo dos 10 parâmetros na ordem PARAM_ORDER.
    """
    (_, theta01, delta1, _m1, _a1,
     theta02, delta2, _m2, _a2, _alpha) = x_full
    delta_min = math.radians(reg_cfg.delta_min_deg)
    pen = 0.0
    pen += reg_cfg.w_order * max(0.0, theta01 - theta02) ** 2
    pen += reg_cfg.w_overlap * max(0.0, (theta01 + delta1) - theta02) ** 2
    pen += reg_cfg.w_min_duration * (
        max(0.0, delta_min - delta1) ** 2 + max(0.0, delta_min - delta2) ** 2)
    if reg_cfg.w_ridge > 0.0:
        for nome, valor in zip(PARAM_ORDER, x_full):
            lo, hi = PARAM_SPECS[nome]["lower"], PARAM_SPECS[nome]["upper"]
            span = hi - lo
            margem = min(valor - lo, hi - valor) / span
            pen += reg_cfg.w_ridge * (max(0.0, 0.02 - margem) / 0.02) ** 2
    return float(pen)


def _full0(engine: EngineConfig, wiebe: WiebeParameters) -> np.ndarray:
    """Vetor completo inicial dos 10 parâmetros (ordem PARAM_ORDER)."""
    return np.array([
        engine.Rc, wiebe.theta01, wiebe.delta1, wiebe.m1, wiebe.a1,
        wiebe.theta02, wiebe.delta2, wiebe.m2, wiebe.a2, wiebe.alpha,
    ])


def _make_expand(selected: List[str], full0: np.ndarray):
    """Cria a expansão subvetor -> vetor completo e o índice de posições."""
    idx = [PARAM_ORDER.index(n) for n in selected]

    def expand(x_sel: np.ndarray) -> np.ndarray:
        x_full = full0.copy()
        x_full[idx] = np.asarray(x_sel, dtype=float)
        return x_full

    return expand, idx


def _make_objective(
    theta_exp, P_exp, engine: EngineConfig, wiebe: WiebeParameters,
    sim: SimulationConfig, calib: CalibrationConfig, selected: List[str],
    expand: Callable[[np.ndarray], np.ndarray],
):
    """Função objetivo (RMSE + regularização) sobre o subvetor."""

    def objective(x_sel: np.ndarray) -> float:
        x_full = expand(x_sel)
        # x_full está na ordem canônica PARAM_ORDER (expand preenche full0):
        # passar 'selected' aqui desalinhava o zip e embaralhava os parâmetros
        eng, wieb = _params_from_vector(x_full, PARAM_ORDER, engine, wiebe)
        valor = evaluate_rmse(theta_exp, P_exp, eng, wieb, sim)
        if valor >= PENALTY:
            return valor
        return valor + _regularization(x_full, calib)

    return objective


# =============================================================================
# Buscas
# =============================================================================
def _run_pso(objective, lower, upper, calib, progress_callback, cancel_check,
             eval_batch=None):
    """PSO sem inércia: x += 2r1*(pbest-x) + 2r2*(gbest-x), clamp, parada
    por estagnação (pso_max_stall repetições com mudança < tol).

    ``eval_batch``: quando o backend acelera avaliações em lote, recebe
    (S, nvar) -> (S,); senão avalia partícula a partícula (serial)."""
    rng = np.random.default_rng(calib.seed)
    li, ls = np.asarray(lower), np.asarray(upper)
    nvar = li.size

    def _eval(X):
        if eval_batch is not None:
            return np.asarray(eval_batch(X), dtype=float)
        return np.array([objective(X[j]) for j in range(X.shape[0])])

    X = li + rng.random((calib.pso_particles, nvar)) * (ls - li)
    fX = _eval(X)
    P_best, f_best = X.copy(), fX.copy()
    jbest = int(np.argmin(f_best))
    fbest = float(f_best[jbest])
    kr, xv_old = 0, P_best[jbest].copy()
    history: List[float] = []
    history_params: List[np.ndarray] = []
    cancelado = False
    while kr < calib.pso_max_stall and len(history) < calib.pso_max_iter:
        r1 = rng.random((calib.pso_particles, nvar))
        r2 = rng.random((calib.pso_particles, nvar))
        X = X + calib.pso_beta * r1 * (P_best - X) \
            + calib.pso_beta * r2 * (P_best[jbest] - X)
        X = np.clip(X, li, ls)
        fX = _eval(X)
        improved = fX < f_best
        P_best[improved], f_best[improved] = X[improved], fX[improved]
        jnew = int(np.argmin(f_best))
        if f_best[jnew] < fbest:
            fbest, jbest = float(f_best[jnew]), jnew
        ermax = float(np.max(np.abs(P_best[jbest] - xv_old)))
        xv_old = P_best[jbest].copy()
        kr = kr + 1 if ermax <= calib.tol else 0
        history.append(fbest)
        history_params.append(P_best[jbest].copy())
        if progress_callback is not None:
            progress_callback(len(history), fbest, P_best[jbest].copy())
        if cancel_check is not None and cancel_check():
            cancelado = True
            break
    return {
        "x": P_best[jbest].copy(), "fun": fbest, "nit": len(history),
        "history": history, "history_params": history_params,
        "parou_por_repeticao": kr >= calib.pso_max_stall,
        "cancelado": cancelado, "success": True,
        "message": ("parou por estagnação" if kr >= calib.pso_max_stall
                    else "máximo de iterações alcançado"),
    }


def _run_de(objective, lower, upper, calib, progress_callback, cancel_check,
            eval_batch=None):
    """scipy.optimize.differential_evolution com progresso e cancelamento.

    Com backend acelerado (``eval_batch``) usa ``vectorized=True``: a
    população inteira de cada geração é avaliada em lote pelo backend
    (updating="deferred", exigido pelo scipy nesse modo)."""
    from scipy.optimize import differential_evolution

    history: List[float] = []
    history_params: List[np.ndarray] = []

    def callback(*args, **_):
        # Protocolos do scipy conforme a versão:
        #   novo (>= 1.9): callback(intermediate_result) — 1 arg com .x/.fun
        #   legado:        callback(xk, convergence)     — args[0] É o xk
        ir = args[0] if args else None
        if ir is not None and hasattr(ir, "fun"):
            f = float(ir.fun)
            x = np.asarray(ir.x, dtype=float)
        elif ir is not None and hasattr(ir, "x"):
            f = objective(ir.x)
            x = np.asarray(ir.x, dtype=float)
        else:
            # protocolo legado: args[0] é o ndarray xk (convergência em args[1])
            x = None if ir is None else np.asarray(ir, dtype=float)
            f = objective(x) if x is not None else float("inf")
        history.append(f)
        history_params.append(x if x is not None else np.full(len(lower), np.nan))
        if progress_callback is not None:
            progress_callback(len(history), f, history_params[-1])
        if cancel_check is not None and cancel_check():
            return True          # solicita parada do DE
        return False

    if eval_batch is not None:
        def obj_vec(X):
            # scipy passa X com forma (n_params, S); avaliamos em lote
            return np.asarray(eval_batch(np.asarray(X).T), dtype=float)

        result = differential_evolution(
            obj_vec, list(zip(np.asarray(lower), np.asarray(upper))),
            seed=calib.seed, maxiter=calib.maxiter, popsize=calib.popsize,
            tol=calib.tol, polish=False, updating="deferred",
            vectorized=True, callback=callback,
        )
    else:
        result = differential_evolution(
            objective, list(zip(np.asarray(lower), np.asarray(upper))),
            seed=calib.seed, maxiter=calib.maxiter, popsize=calib.popsize,
            tol=calib.tol, polish=False, updating="immediate",
            callback=callback,
        )
    return {
        "x": result.x.copy(), "fun": float(result.fun), "nit": int(result.nit),
        "history": history, "history_params": history_params,
        "cancelado": bool(not result.success
                          and "stopped" in str(result.message).lower()),
        "success": bool(result.success), "message": str(result.message),
    }


def _run_least_squares(
    theta_exp, P_exp, engine, wiebe, sim, calib, selected, lower, upper, x0,
):
    """least-squares (trust-region reflexive) com vetor de resíduos
    real: r_i = P_sim(theta_i) - P_exp(theta_i) [kPa]."""
    from scipy.optimize import least_squares

    def residual(x_sel: np.ndarray) -> np.ndarray:
        eng, wieb = _params_from_vector(x_sel, selected, engine, wiebe)
        try:
            P_sim, _, _ = integrate_ode(
                theta_exp, eng, wieb, float(P_exp[0]),
                method=sim.method, rtol=sim.rtol, atol=sim.atol,
            )
            if not np.all(np.isfinite(P_sim)):   # NaN/inf sem exceção
                raise ODEFailure("solução não finita")
            return P_sim - P_exp
        except (ODEFailure, ValueError, OverflowError, FloatingPointError):
            return np.full(theta_exp.size, math.sqrt(PENALTY / theta_exp.size))

    result = least_squares(
        residual, np.asarray(x0, dtype=float),
        bounds=(np.asarray(lower), np.asarray(upper)),
        method="trf", xtol=calib.tol, ftol=calib.tol, gtol=calib.tol,
        max_nfev=max(calib.maxiter, 1) * 10,
    )
    # least_squares minimiza cost = 0.5*Σr²  =>  RMSE = √(2·cost / n)
    return {
        "x": result.x.copy(),
        "fun": float(np.sqrt(2.0 * result.cost / theta_exp.size)),
        "nit": int(result.nfev), "history": [], "history_params": [],
        "cancelado": False, "success": bool(result.success),
        "message": str(result.message),
    }


def _refine_least_squares(
    theta_exp, P_exp, engine, wiebe, sim, selected, x_full, idx,
    lower_full, upper_full,
):
    """Refinamento local (least-squares, resíduos completos) dos parâmetros
    selecionados a partir do ótimo da busca global."""
    from scipy.optimize import least_squares

    def residual(x_sel: np.ndarray) -> np.ndarray:
        full = x_full.copy()
        full[idx] = x_sel
        eng, wieb = _params_from_vector(full, PARAM_ORDER, engine, wiebe)
        try:
            P_sim, _, _ = integrate_ode(
                theta_exp, eng, wieb, float(P_exp[0]),
                method=sim.method, rtol=sim.rtol, atol=sim.atol,
            )
            if not np.all(np.isfinite(P_sim)):   # NaN/inf sem exceção
                raise ODEFailure("solução não finita")
            return P_sim - P_exp
        except (ODEFailure, ValueError, OverflowError, FloatingPointError):
            return np.full(theta_exp.size, math.sqrt(PENALTY / theta_exp.size))

    result = least_squares(
        residual, x_full[idx],
        bounds=(np.asarray(lower_full)[idx], np.asarray(upper_full)[idx]),
        method="trf", xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=500,
    )
    return result.x.copy()


def _sensitivity(objective, x_sel, rmse_best, eval_batch=None) -> List[Dict]:
    """Sensibilidade por parâmetro: perturbação ±1% no valor ótimo.
    Com ``eval_batch``, avalia os 2n candidatos em lote (backend)."""
    linhas: List[Dict] = []
    limiar = 0.001 * abs(rmse_best) if rmse_best > 0 else 1e-12
    if eval_batch is not None:
        deltas = [0.01 * abs(x_sel[i]) if x_sel[i] != 0.0 else 1e-6
                  for i in range(x_sel.size)]
        cands = np.array([_perturb(x_sel, i, d * s)
                          for i, d in enumerate(deltas)
                          for s in (+1.0, -1.0)])
        fv = np.asarray(eval_batch(cands), dtype=float)
        for i in range(x_sel.size):
            d_rmse = max(abs(fv[2 * i] - rmse_best),
                         abs(fv[2 * i + 1] - rmse_best))
            linhas.append({
                "delta_rmse": float(d_rmse),
                "insensitive": bool(d_rmse < limiar),
            })
        return linhas
    for i in range(x_sel.size):
        delta = 0.01 * abs(x_sel[i]) if x_sel[i] != 0.0 else 1e-6
        f_plus = objective(_perturb(x_sel, i, +delta))
        f_minus = objective(_perturb(x_sel, i, -delta))
        d_rmse = max(abs(f_plus - rmse_best), abs(f_minus - rmse_best))
        linhas.append({
            "delta_rmse": float(d_rmse),
            "insensitive": bool(d_rmse < limiar),
        })
    return linhas


def _perturb(x: np.ndarray, i: int, delta: float) -> np.ndarray:
    y = x.copy()
    y[i] = y[i] + delta
    return y


# =============================================================================
# Orquestração
# =============================================================================
def run_calibration(
    theta_exp: np.ndarray,
    P_exp: np.ndarray,
    engine: Optional[EngineConfig] = None,
    wiebe: Optional[WiebeParameters] = None,
    sim: Optional[SimulationConfig] = None,
    calib: Optional[CalibrationConfig] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> Dict:
    """Calibração completa do Double Wiebe.

    Retorna dicionário com:

        params, params_initial     dicts nome -> valor (os 10 parâmetros)
        selected, lower, upper     seleção e limites efetivos
        rmse                       RMSE ótimo (kPa)
        history, history_params    convergência
        method, seed, iteracoes, message, success, cancelado
        polish                     refinamento least-squares (ou None)
        sensitivity                tabela de sensibilidade
        alerts                     avisos de borda/identificabilidade
    """
    engine = engine or EngineConfig()
    wiebe = wiebe or WiebeParameters()
    sim = sim or SimulationConfig()
    calib = calib or CalibrationConfig()

    erros = engine.validate() + wiebe.validate() + sim.validate() \
        + calib.validate()
    if erros:
        raise CalibrationError("Configuração inválida: " + " | ".join(erros))

    theta_exp = np.asarray(theta_exp, dtype=float)
    P_exp = np.asarray(P_exp, dtype=float)

    selected, lower, upper = build_bounds(calib.selected, bounds)
    full0 = _full0(engine, wiebe)
    expand, idx = _make_expand(selected, full0)
    objective = _make_objective(
        theta_exp, P_exp, engine, wiebe, sim, calib, selected, expand)

    # --- backend de desempenho (plano HPC) ---------------------------------
    # serial: caminho original (referência). cpu / cpu-parallel / auto:
    # avaliações de POPULAÇÃO em lote (RK4 vetorizado ou processos); o
    # melhor candidato final é SEMPRE re-integrado com solve_ivp abaixo.
    backend = None
    eval_batch = None
    import time as _time
    t_inicio = _time.perf_counter()
    if calib.backend not in ("serial", None):
        from .backends import select_backend
        backend = select_backend(calib.backend, workers=calib.workers,
                                 integrator=calib.integrator)
        idx_sel = [PARAM_ORDER.index(n) for n in selected]

        def eval_batch(X_sel: np.ndarray) -> np.ndarray:
            X_sel = np.atleast_2d(np.asarray(X_sel, dtype=float))
            X_full = np.tile(full0, (X_sel.shape[0], 1))
            X_full[:, idx_sel] = X_sel
            return backend.evaluate_population(
                X_full, theta_exp, P_exp, engine, wiebe, sim, calib,
                precision=calib.precision, substeps=calib.substeps,
                batch_size=calib.batch_size)

    # Limites efetivos no vetor completo (sobrescritas por parâmetro)
    lower_full = np.array([PARAM_SPECS[n]["lower"] for n in PARAM_ORDER])
    upper_full = np.array([PARAM_SPECS[n]["upper"] for n in PARAM_ORDER])
    for j, nome in enumerate(PARAM_ORDER):
        if bounds and nome in bounds:
            lower_full[j], upper_full[j] = float(bounds[nome][0]), float(
                bounds[nome][1])

    iniciais_ajustados: List[str] = []
    if calib.method == "pso":
        resultado = _run_pso(objective, lower, upper, calib,
                             progress_callback, cancel_check,
                             eval_batch=eval_batch)
    elif calib.method == "differential-evolution":
        resultado = _run_de(objective, lower, upper, calib,
                            progress_callback, cancel_check,
                            eval_batch=eval_batch)
    elif calib.method == "least-squares":
        # ponto inicial = valores atuais; se estiverem fora dos limites
        # escolhidos, são trazidos para dentro (o scipy recusaria x0)
        x0 = np.clip(full0[idx], lower, upper)
        iniciais_ajustados = [n for n, a, b in zip(selected, full0[idx], x0)
                              if a != b]
        resultado = _run_least_squares(
            theta_exp, P_exp, engine, wiebe, sim, calib, selected,
            lower, upper, x0)
    else:  # pragma: no cover — validado em CalibrationConfig
        raise CalibrationError(f"Método desconhecido: {calib.method}")

    x_sel = np.asarray(resultado["x"], dtype=float)
    x_full = expand(x_sel)

    # --- validação numérica (regra 4): re-run do melhor candidato com a
    # referência serial (solve_ivp), SEMPRE ------------------------------
    rmse_integrador = float(resultado["fun"])
    rmse_best = float(objective(x_sel))
    # RK4 em lote × solve_ivp no mesmo candidato (antes do refinamento)
    dif_integrador = abs(rmse_best - rmse_integrador)
    if backend is not None:
        resultado["fun"] = rmse_best

    try:
        # Refinamento least-squares (opcional, após DE/PSO)
        polish: Optional[Dict] = None
        if (calib.polish and calib.method in ("differential-evolution", "pso")
                and not resultado.get("cancelado")):
            try:
                x_ref = _refine_least_squares(
                    theta_exp, P_exp, engine, wiebe, sim, selected,
                    x_full, idx, lower_full, upper_full)
                f_ref = objective(x_ref)
                if f_ref < resultado["fun"]:
                    polish = {"rmse": float(f_ref), "aplicado": True}
                    x_sel, x_full = np.asarray(x_ref), expand(x_ref)
                    resultado["fun"] = float(f_ref)   # relata o refinado
                else:
                    polish = {"rmse": float(f_ref), "aplicado": False}
            except Exception as e:                # refinamento é opcional
                polish = {"erro": str(e), "aplicado": False}

        rmse_best = float(resultado["fun"])
        eng_cal, wieb_cal = _params_from_vector(
            x_full, PARAM_ORDER, engine, wiebe)

        # Sensibilidade (pula quando cancelado ou penalidade)
        sensitivity: List[Dict] = []
        if not resultado.get("cancelado") and rmse_best < PENALTY:
            linhas = _sensitivity(objective, x_sel, rmse_best,
                                  eval_batch=eval_batch)
            sensitivity = [
                {"param": nome, **lin} for nome, lin in zip(selected, linhas)
            ]

        # Alertas
        alertas: List[str] = []
        if iniciais_ajustados:
            alertas.append(
                "⚠ Valor inicial fora dos limites escolhidos, levado para a "
                f"borda antes do least-squares: {', '.join(iniciais_ajustados)}.")
        for i, nome in enumerate(selected):
            lo, hi = lower[i], upper[i]
            span = hi - lo
            if x_sel[i] <= lo + 1e-6 * span or x_sel[i] >= hi - 1e-6 * span:
                alertas.append(
                    f"⚠ {nome} calibrado na borda do domínio ({x_sel[i]:.4g}); "
                    f"limites ({lo:.4g}, {hi:.4g}) — ótimo possivelmente mal "
                    "condicionado.")
        insens = [s["param"] for s in sensitivity if s.get("insensitive")]
        if insens:
            alertas.append(
                "⚠ Identificabilidade: parâmetro(s) " + ", ".join(insens) +
                " com sensibilidade desprezível — combinações diferentes "
                "produzem erros praticamente equivalentes. Considere fixar "
                "estes parâmetros ou usar mais dados.")
        if rmse_best >= PENALTY:
            alertas.append("⚠ Calibração terminou em penalidade (nenhuma "
                           "combinação testada produziu integração válida).")
        if calib.w_order > 0 and wieb_cal.theta02 < wieb_cal.theta01:
            alertas.append("⚠ theta02 < theta01 no resultado (restrição "
                           "violada mesmo com regularização).")

        tempo_s = _time.perf_counter() - t_inicio
        return {
            "params": dict(zip(PARAM_ORDER, map(float, x_full))),
            "params_initial": dict(zip(PARAM_ORDER, map(float, full0))),
            "selected": selected,
            "lower": dict(zip(selected, map(float, lower))),
            "upper": dict(zip(selected, map(float, upper))),
            "rmse": rmse_best,
            "objective_history": list(resultado["history"]),
            "history_params": [np.asarray(h).copy()
                               for h in resultado["history_params"]],
            "method": calib.method,
            "seed": calib.seed,
            "iteracoes": int(resultado["nit"]),
            "message": str(resultado["message"]),
            "success": bool(resultado["success"]),
            "cancelado": bool(resultado.get("cancelado", False)),
            "parou_por_estagnacao": bool(
                resultado.get("parou_por_repeticao", False)),
            "polish": polish,
            "sensitivity": sensitivity,
            "alerts": alertas,
            "engine": dict(vars(eng_cal)),
            "wiebe": dict(vars(wieb_cal)),
            "theta_exp": theta_exp,
            "P_exp": P_exp,
            # --- HPC (plano de aceleração) ---
            "backend": calib.backend or "serial",
            "rmse_integrador": rmse_integrador,
            "diferenca_integrador": dif_integrador,
            "tempo_s": tempo_s,
        }
    finally:
        if backend is not None:
            backend.close()


def apply_calibrated(
    calib_result: Dict,
    base_engine: EngineConfig,
    base_wiebe: WiebeParameters,
) -> Tuple[EngineConfig, WiebeParameters]:
    """Aplica os parâmetros calibrados a (engine, wiebe) — usado pela CLI/GUI
    para rodar a simulação final com o resultado da calibração."""
    engine = EngineConfig(**vars(base_engine))
    wiebe = WiebeParameters(**vars(base_wiebe))
    engine.Rc = float(calib_result["params"]["Rc"])
    for nome in ("theta01", "delta1", "m1", "a1",
                 "theta02", "delta2", "m2", "a2", "alpha"):
        setattr(wiebe, nome, float(calib_result["params"][nome]))
    return engine, wiebe