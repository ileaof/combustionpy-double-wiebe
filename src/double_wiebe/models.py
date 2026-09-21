# -*- coding: utf-8 -*-
"""
models.py
=========
Modelos de configuração do projeto Double Wiebe (dataclasses + validação) e
carga/gravação de arquivos YAML.

Nenhuma equação vive aqui — apenas parâmetros, unidades, validação física e
serialização. As equações estão em `wiebe.py`, `geometry.py` e
`thermodynamics.py`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ----------------------------------------------------------------------------
# Modos da função de Wiebe
# ----------------------------------------------------------------------------
MODE_CONTINUOUS = "continuous"   # expressão continua após theta0 + delta
MODE_BOUNDED = "bounded"         # fase constante após theta0 + delta
WIEBE_MODES = (MODE_CONTINUOUS, MODE_BOUNDED)

# Valor padrão da constante de eficiência (mesma do Single Wiebe)
A_WIEBE_DEFAULT = 6.9078

# Parâmetros calibráveis (ordem canônica usada em calibration.py/reporting)
PARAM_ORDER = [
    "Rc", "theta01", "delta1", "m1", "a1",
    "theta02", "delta2", "m2", "a2", "alpha",
]
DEFAULT_SELECTED = [
    "Rc", "theta01", "delta1", "m1", "theta02", "delta2", "m2", "alpha",
]


# =============================================================================
# Motor: geometria + operação + combustível + termodinâmica
# =============================================================================
@dataclass
class EngineConfig:
    """Parâmetros físicos do motor (valores iniciais configuráveis).

    Unidades: comprimentos em m, rpm em rotações por minuto, massa em
    kg/ciclo, PCI em kJ/kg, temperaturas em K.
    """
    # Geometria
    bore: float = 0.086            # diâmetro do cilindro d [m]      (86 mm)
    stroke: float = 0.070          # curso do pistão s [m]           (70 mm)
    rod_length: float = 0.1175     # comprimento da biela l [m]      (117.5 mm)
    n_cylinders: int = 1           # número de cilindros [-]
    # Operação
    rpm: float = 3396.20           # rotação [rpm]
    Rc: float = 17.0               # razão de compressão [-]
    # Combustível
    m_fuel: float = 9.42754647351e-6   # massa de combustível [kg/ciclo]
    LHV: float = 39191.3           # poder calorífico inferior PCI/LHV [kJ/kg]
    # Termodinâmica
    kappa: float = 1.37            # razão de calores específicos [-]
    T1: float = 308.15             # temperatura inicial (IVC) [K]
    Tw: float = 440.0              # temperatura da parede [K]
    # Transferência de calor
    heat_transfer: bool = True     # ativa a perda de calor (Hohenberg)

    # ------------------------------------------------------------------
    # Grandezas derivadas (somente leitura)
    # ------------------------------------------------------------------
    @property
    def A_p(self) -> float:
        """Área do pistão A_p = pi*(d/2)² [m²]."""
        return math.pi * (self.bore / 2.0) ** 2

    @property
    def r_crank(self) -> float:
        """Raio da manivela r = s/2 [m]."""
        return self.stroke / 2.0

    @property
    def R(self) -> float:
        """Razão biela/manivela R = l/r [-]."""
        return self.rod_length / self.r_crank

    @property
    def Vd(self) -> float:
        """Volume deslocado Vd = A_p * s [m³]."""
        return self.A_p * self.stroke

    @property
    def Vc(self) -> float:
        """Volume de folga Vc = Vd/(Rc - 1) [m³]."""
        return self.Vd / (self.Rc - 1.0)

    @property
    def omega_rev_s(self) -> float:
        """Rotação em revoluções por segundo [rev/s]."""
        return self.rpm / 60.0

    @property
    def Vp(self) -> float:
        """Velocidade média do pistão V_p = 2*s*(rpm/60) [m/s]."""
        return 2.0 * self.stroke * self.omega_rev_s

    @property
    def Q_total(self) -> float:
        """Calor total por ciclo Q_total = m_fuel * LHV [kJ/ciclo]."""
        return self.m_fuel * self.LHV

    def validate(self) -> List[str]:
        """Retorna lista de erros (strings) para valores não físicos."""
        erros: List[str] = []
        for nome, valor, vmin in (
            ("diâmetro do cilindro", self.bore, 1e-4),
            ("curso do pistão", self.stroke, 1e-4),
            ("comprimento da biela", self.rod_length, 1e-4),
            ("rotação", self.rpm, 1.0),
            ("massa de combustível", self.m_fuel, 0.0),
            ("PCI/LHV", self.LHV, 1.0),
        ):
            if not (valor > vmin):
                erros.append(f"{nome} deve ser > {vmin} (recebido {valor}).")
        if self.n_cylinders < 1:
            erros.append("número de cilindros deve ser >= 1.")
        if not (1.0 < self.Rc):
            erros.append(f"razão de compressão Rc deve ser > 1 (recebido {self.Rc}).")
        if not (1.0 < self.kappa < 2.0):
            erros.append(f"kappa deve estar em (1, 2) (recebido {self.kappa}).")
        if self.T1 <= 0.0 or self.Tw <= 0.0:
            erros.append("temperaturas (T1, Tw) devem ser positivas [K].")
        if self.rod_length <= self.r_crank:
            erros.append("biela deve ser mais longa que o raio da manivela "
                         "(l > s/2) para a geometria ser válida.")
        return erros


# =============================================================================
# Parâmetros do Double Wiebe
# =============================================================================
@dataclass
class WiebeParameters:
    """Parâmetros das duas fases de Wiebe.

    Fase 1: combustão pré-misturada (premixed).
    Fase 2: combustão controlada por difusão (diffusion-controlled).

    ``alpha`` é a fração de energia atribuída à fase 1 (a fase 2 recebe
    ``1 - alpha``). ``mode`` seleciona entre a expressão contínua
    (assintótica) e o modo limitado (fase constante após theta0 + delta).
    """
    theta01: float = math.radians(-6.54)   # início da fase 1 [rad]
    delta1: float = math.radians(25.0)     # duração da fase 1 [rad]
    m1: float = 0.5                        # fator de forma da fase 1 [-]
    a1: float = A_WIEBE_DEFAULT            # eficiência da fase 1 [-]
    theta02: float = math.radians(10.0)    # início da fase 2 [rad]
    delta2: float = math.radians(55.0)     # duração da fase 2 [rad]
    m2: float = 1.0                        # fator de forma da fase 2 [-]
    a2: float = A_WIEBE_DEFAULT            # eficiência da fase 2 [-]
    alpha: float = 0.40                    # fração de energia da fase 1 [-]
    mode: str = MODE_CONTINUOUS            # "continuous" | "bounded"

    def validate(self) -> List[str]:
        """Aplica as restrições físicas do modelo Double Wiebe."""
        erros: List[str] = []
        if self.mode not in WIEBE_MODES:
            erros.append(f"mode deve ser um de {WIEBE_MODES} "
                         f"(recebido '{self.mode}').")
        for nome, valor in (("delta1", self.delta1), ("delta2", self.delta2)):
            if not (valor > 0.0):
                erros.append(f"{nome} deve ser > 0 [rad] (recebido {valor}).")
        for nome, valor in (("m1", self.m1), ("m2", self.m2)):
            if not (valor >= 0.0):
                erros.append(f"{nome} deve ser >= 0 (recebido {valor}).")
        for nome, valor in (("a1", self.a1), ("a2", self.a2)):
            if not (valor > 0.0):
                erros.append(f"{nome} deve ser > 0 (recebido {valor}).")
        if not (0.0 <= self.alpha <= 1.0):
            erros.append(f"alpha deve estar em [0, 1] (recebido {self.alpha}).")
        if self.theta02 < self.theta01:
            erros.append(
                "theta02 deve ser >= theta01 (a fase de difusão normalmente "
                "começa após a pré-misturada).")
        return erros

    def as_dict(self) -> Dict[str, Any]:
        """Parâmetros como dicionário (com ângulos também em graus)."""
        d = asdict(self)
        d["theta01_deg"] = math.degrees(self.theta01)
        d["theta02_deg"] = math.degrees(self.theta02)
        d["delta1_deg"] = math.degrees(self.delta1)
        d["delta2_deg"] = math.degrees(self.delta2)
        return d


# =============================================================================
# Configuração numérica da simulação
# =============================================================================
@dataclass
class SimulationConfig:
    """Opções do integrador (scipy.integrate.solve_ivp)."""
    method: str = "DOP853"      # DOP853 | LSODA | RK45
    rtol: float = 1e-9
    atol: float = 1e-9

    ALLOWED_METHODS = ("DOP853", "LSODA", "RK45")

    def validate(self) -> List[str]:
        erros: List[str] = []
        if self.method not in self.ALLOWED_METHODS:
            erros.append(f"método deve ser um de {self.ALLOWED_METHODS} "
                         f"(recebido '{self.method}').")
        if not (self.rtol > 0.0 and self.atol > 0.0):
            erros.append("rtol e atol devem ser positivos.")
        return erros


# =============================================================================
# Configuração da calibração
# =============================================================================
@dataclass
class CalibrationConfig:
    """Opções de calibração.

    method: "differential-evolution" | "pso" | "least-squares"
    selected: parâmetros livres (os demais ficam fixos nos valores iniciais).
    bounds: limites por parâmetro (nome -> (inf, sup)).
    regularization: pesos das penalidades (ver calibration.py).
    """
    method: str = "differential-evolution"
    selected: List[str] = field(default_factory=lambda: list(DEFAULT_SELECTED))
    seed: Optional[int] = None
    maxiter: int = 200
    popsize: int = 20
    tol: float = 1e-10
    polish: bool = True            # least-squares após a busca global
    # PSO (opcional)
    pso_particles: int = 20
    pso_beta: float = 2.0
    pso_max_iter: int = 300
    pso_max_stall: int = 10
    # Regularização (pesos; 0 desativa)
    w_order: float = 1.0e4         # theta02 >= theta01
    w_overlap: float = 1.0e4       # sobreposição forte das fases
    w_min_duration: float = 1.0e4  # durações excessivamente pequenas
    delta_min_deg: float = 5.0     # duração mínima física [graus]
    w_ridge: float = 0.0           # regularização ridge (afastar dos limites)
    # Desempenho (plano HPC)
    backend: str = "serial"        # serial | cpu | cpu-parallel | cuda | auto
    integrator: str = "auto"       # auto | scipy | rk4_numpy | rk4_numba
    workers: Optional[int] = None  # processos (cpu-parallel); None = auto
    batch_size: int = 0            # candidatos por lote (0 = tudo de uma vez)
    precision: str = "float64"     # float64 | float32 (só no modo acelerado)
    substeps: int = 4              # sub-passos do RK4 em lote (modo acelerado)

    METHODS = ("differential-evolution", "pso", "least-squares")
    BACKENDS = ("serial", "cpu", "cpu-parallel", "cuda", "auto")
    PRECISIONS = ("float64", "float32")
    INTEGRATORS = ("auto", "scipy", "rk4_numpy", "rk4_numba")

    def validate(self) -> List[str]:
        erros: List[str] = []
        if self.method not in self.METHODS:
            erros.append(f"method deve ser um de {self.METHODS} "
                         f"(recebido '{self.method}').")
        if self.maxiter < 1 or self.popsize < 2:
            erros.append("maxiter >= 1 e popsize >= 2 são exigidos.")
        if self.delta_min_deg <= 0.0:
            erros.append("delta_min_deg deve ser > 0.")
        if any(w < 0.0 for w in (self.w_order, self.w_overlap,
                                 self.w_min_duration, self.w_ridge)):
            erros.append("pesos de regularização devem ser >= 0.")
        if self.backend not in self.BACKENDS:
            erros.append(f"backend deve ser um de {self.BACKENDS} "
                         f"(recebido '{self.backend}').")
        if self.integrator not in self.INTEGRATORS:
            erros.append(f"integrator deve ser um de {self.INTEGRATORS} "
                         f"(recebido '{self.integrator}').")
        if self.precision not in self.PRECISIONS:
            erros.append(f"precision deve ser uma de {self.PRECISIONS} "
                         f"(recebido '{self.precision}').")
        if self.workers is not None and self.workers < 1:
            erros.append("workers deve ser >= 1 (ou None para automático).")
        if self.substeps < 1:
            erros.append("substeps deve ser >= 1.")
        return erros


# =============================================================================
# YAML
# =============================================================================
def _config_to_dict(engine: EngineConfig, wiebe: WiebeParameters,
                    simulation: SimulationConfig,
                    calibration: CalibrationConfig) -> Dict[str, Any]:
    """Estrutura de dicionário completa (seções engine/wiebe/simulation/
    calibration) com ângulos em GRAUS e radianos (graus é o formato do YAML,
    mais legível)."""
    return {
        "engine": {
            "bore_mm": engine.bore * 1000.0,
            "stroke_mm": engine.stroke * 1000.0,
            "rod_length_mm": engine.rod_length * 1000.0,
            "n_cylinders": engine.n_cylinders,
            "rpm": engine.rpm,
            "Rc": engine.Rc,
            "m_fuel_kg_per_cycle": engine.m_fuel,
            "LHV_kJ_per_kg": engine.LHV,
            "kappa": engine.kappa,
            "T1_K": engine.T1,
            "Tw_K": engine.Tw,
            "heat_transfer": engine.heat_transfer,
        },
        "wiebe": {
            "mode": wiebe.mode,
            "theta01_deg": math.degrees(wiebe.theta01),
            "delta1_deg": math.degrees(wiebe.delta1),
            "m1": wiebe.m1,
            "a1": wiebe.a1,
            "theta02_deg": math.degrees(wiebe.theta02),
            "delta2_deg": math.degrees(wiebe.delta2),
            "m2": wiebe.m2,
            "a2": wiebe.a2,
            "alpha": wiebe.alpha,
        },
        "simulation": {
            "method": simulation.method,
            "rtol": simulation.rtol,
            "atol": simulation.atol,
        },
        "calibration": {
            "method": calibration.method,
            "selected": list(calibration.selected),
            "seed": calibration.seed,
            "maxiter": calibration.maxiter,
            "popsize": calibration.popsize,
            "tol": calibration.tol,
            "polish": calibration.polish,
            "pso_particles": calibration.pso_particles,
            "pso_beta": calibration.pso_beta,
            "pso_max_iter": calibration.pso_max_iter,
            "pso_max_stall": calibration.pso_max_stall,
            "regularization": {
                "w_order": calibration.w_order,
                "w_overlap": calibration.w_overlap,
                "w_min_duration": calibration.w_min_duration,
                "delta_min_deg": calibration.delta_min_deg,
                "w_ridge": calibration.w_ridge,
            },
            # desempenho (plano HPC) — chaves planas para os overrides
            # da CLI (--set calibration.backend=cpu)
            "backend": calibration.backend,
            "integrator": calibration.integrator,
            "workers": calibration.workers,
            "batch_size": calibration.batch_size,
            "precision": calibration.precision,
            "substeps": calibration.substeps,
        },
    }


def dump_config_yaml(engine: EngineConfig, wiebe: WiebeParameters,
                     simulation: SimulationConfig,
                     calibration: CalibrationConfig) -> str:
    """Serializa a configuração completa em texto YAML."""
    return yaml.safe_dump(
        _config_to_dict(engine, wiebe, simulation, calibration),
        sort_keys=False, allow_unicode=True,
    )


def save_config_yaml(path: str | Path, engine: EngineConfig,
                     wiebe: WiebeParameters, simulation: SimulationConfig,
                     calibration: CalibrationConfig) -> Path:
    """Grava o YAML completo em `path` (usado pelo `example-config` e pela
    exportação de parâmetros)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        dump_config_yaml(engine, wiebe, simulation, calibration),
        encoding="utf-8",
    )
    return p


def _apply_overrides(d: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Aplica sobrescritas "secao.chave=valor" ao dicionário de configuração."""
    if not overrides:
        return d
    for chave, valor in overrides.items():
        secao, _, campo = chave.partition(".")
        if not campo:
            raise KeyError(
                f"Sobrescrita inválida '{chave}': use 'secao.chave=valor' "
                "(ex.: wiebe.alpha).")
        if secao not in d or not isinstance(d[secao], dict):
            raise KeyError(f"Seção desconhecida '{secao}' (em '{chave}').")
        if campo not in d[secao]:
            raise KeyError(f"Chave desconhecida '{secao}.{campo}'.")
        d[secao][campo] = valor
    return d


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Lê o YAML de configuração (ou usa os defaults) e devolve o dicionário
    de seções. `overrides` aplica pares "secao.chave" -> valor por cima.

    Os ângulos do YAML estão em graus; aqui são convertidos para radianos
    nas chaves esperadas pelos dataclasses.
    """
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Arquivo YAML inválido: {path}")
    else:
        raw = _config_to_dict(EngineConfig(), WiebeParameters(),
                              SimulationConfig(), CalibrationConfig())
    raw = _apply_overrides(raw, overrides)

    eng = raw.get("engine", {})
    wb = raw.get("wiebe", {})
    sim = raw.get("simulation", {})
    cal = raw.get("calibration", {})
    reg = cal.get("regularization", {})

    engine = EngineConfig(
        bore=float(eng.get("bore_mm", 86.0)) / 1000.0,
        stroke=float(eng.get("stroke_mm", 70.0)) / 1000.0,
        rod_length=float(eng.get("rod_length_mm", 117.5)) / 1000.0,
        n_cylinders=int(eng.get("n_cylinders", 1)),
        rpm=float(eng.get("rpm", 3396.20)),
        Rc=float(eng.get("Rc", 17.0)),
        m_fuel=float(eng.get("m_fuel_kg_per_cycle", 9.42754647351e-6)),
        LHV=float(eng.get("LHV_kJ_per_kg", 39191.3)),
        kappa=float(eng.get("kappa", 1.37)),
        T1=float(eng.get("T1_K", 308.15)),
        Tw=float(eng.get("Tw_K", 440.0)),
        heat_transfer=bool(eng.get("heat_transfer", True)),
    )
    wiebe = WiebeParameters(
        theta01=math.radians(float(wb.get("theta01_deg", math.degrees(
            math.radians(-6.54))))),
        delta1=math.radians(float(wb.get("delta1_deg", 25.0))),
        m1=float(wb.get("m1", 0.5)),
        a1=float(wb.get("a1", A_WIEBE_DEFAULT)),
        theta02=math.radians(float(wb.get("theta02_deg", 10.0))),
        delta2=math.radians(float(wb.get("delta2_deg", 55.0))),
        m2=float(wb.get("m2", 1.0)),
        a2=float(wb.get("a2", A_WIEBE_DEFAULT)),
        alpha=float(wb.get("alpha", 0.40)),
        mode=str(wb.get("mode", MODE_CONTINUOUS)),
    )
    simulation = SimulationConfig(
        method=str(sim.get("method", "DOP853")),
        rtol=float(sim.get("rtol", 1e-9)),
        atol=float(sim.get("atol", 1e-9)),
    )
    calibration = CalibrationConfig(
        method=str(cal.get("method", "differential-evolution")),
        selected=list(cal.get("selected", DEFAULT_SELECTED)),
        seed=(int(cal["seed"]) if cal.get("seed") is not None else None),
        maxiter=int(cal.get("maxiter", 200)),
        popsize=int(cal.get("popsize", 20)),
        tol=float(cal.get("tol", 1e-10)),
        polish=bool(cal.get("polish", True)),
        pso_particles=int(cal.get("pso_particles", 20)),
        pso_beta=float(cal.get("pso_beta", 2.0)),
        pso_max_iter=int(cal.get("pso_max_iter", 300)),
        pso_max_stall=int(cal.get("pso_max_stall", 10)),
        w_order=float(reg.get("w_order", 1.0e4)),
        w_overlap=float(reg.get("w_overlap", 1.0e4)),
        w_min_duration=float(reg.get("w_min_duration", 1.0e4)),
        delta_min_deg=float(reg.get("delta_min_deg", 5.0)),
        w_ridge=float(reg.get("w_ridge", 0.0)),
        backend=str(cal.get("backend", "serial") or "serial"),
        integrator=str(cal.get("integrator", "auto") or "auto"),
        workers=(int(cal["workers"])
                 if cal.get("workers") is not None else None),
        batch_size=int(cal.get("batch_size", 0)),
        precision=str(cal.get("precision", "float64") or "float64"),
        substeps=int(cal.get("substeps", 4)),
    )
    return {
        "engine": engine,
        "wiebe": wiebe,
        "simulation": simulation,
        "calibration": calibration,
        "raw": raw,
    }