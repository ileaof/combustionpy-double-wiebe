# -*- coding: utf-8 -*-
"""
cli.py
======
Interface de linha de comando do Double Wiebe (Typer).

Comando instalável (pelo `pip install -e .`)::

    double-wiebe {simulate,calibrate,validate,plot,gui,example-config}

Convenções:
  * códigos de saída: 0 = ok; 1 = falha de execução; 2 = entrada inválida;
  * ``--quiet`` suprime o resumo; ``--verbose`` imprime detalhes;
  * sobrescritas de configuração: ``--set secao.chave=valor`` (repetível);
  * nenhum diretório de saída é sobrescrito sem ``--overwrite``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(
    name="double-wiebe",
    help="Double Wiebe Combustion Analysis — simulação e calibração do "
         "modelo Double Wiebe (núcleo científico único, usado também pela "
         "GUI Streamlit).",
    add_completion=False,
    no_args_is_help=True,
)

EXIT_OK, EXIT_FAIL, EXIT_BADINPUT = 0, 1, 2


# =============================================================================
# Utilidades
# =============================================================================
def _parse_sets(sets: List[str]) -> dict:
    """Converte ['wiebe.alpha=0.4', ...] em {'wiebe.alpha': 0.4}."""
    overrides: dict = {}
    for item in sets or []:
        chave, _, valor = item.partition("=")
        chave = chave.strip()
        if not chave or not valor:
            raise typer.BadParameter(
                f"Sobrescrita inválida '{item}' — use secao.chave=valor.")
        try:
            valor_conv = json.loads(valor)
        except json.JSONDecodeError:
            valor_conv = valor           # string crua (ex.: 'bounded')
        overrides[chave] = valor_conv
    return overrides


def _carregar(data_path: Optional[Path], config_path: Optional[Path],
              overrides: Optional[dict] = None,
              ang_unit: Optional[str] = None,
              press_unit: Optional[str] = None,
              theta_min: Optional[float] = None,
              theta_max: Optional[float] = None):
    """Carrega YAML + sobrescritas + dados experimentais (núcleo único).

    Opções de dados não informadas na CLI caem primeiro na seção ``data``
    do YAML e depois nos defaults (radianos/kPa, separador auto).
    """
    from .models import load_config
    from .data_processing import DataError, prepare_series, read_table

    cfg = load_config(str(config_path) if config_path else None, overrides)
    if data_path is None:
        raise typer.BadParameter("Informe --data com o arquivo experimental.")
    if not Path(data_path).exists():
        typer.secho(f"Arquivo de dados não encontrado: {data_path}",
                    fg="red", err=True)
        raise typer.Exit(EXIT_BADINPUT)

    data_cfg = cfg["raw"].get("data", {}) or {}
    opts = {
        "angle_col": int(data_cfg.get("angle_col", 0)),
        "pressure_col": int(data_cfg.get("pressure_col", 1)),
        "angle_unit": ang_unit or data_cfg.get("angle_unit", "radianos"),
        "pressure_unit": press_unit or data_cfg.get("pressure_unit", "kPa"),
        "theta_min": (theta_min if theta_min is not None
                      else data_cfg.get("theta_min")),
        "theta_max": (theta_max if theta_max is not None
                      else data_cfg.get("theta_max")),
        "sep": data_cfg.get("sep", "auto"),
        "has_header": data_cfg.get("has_header"),
    }
    if opts["has_header"] is True:
        opts["has_header"] = "header"
    elif opts["has_header"] is False:
        opts["has_header"] = "none"
    elif opts["has_header"] in ("none", "header"):
        pass
    else:
        opts["has_header"] = None
    try:
        df = read_table(data_path, sep=opts["sep"],
                        has_header=opts["has_header"])
        theta, P, resumo = prepare_series(
            df,
            angle_col=opts["angle_col"],
            pressure_col=opts["pressure_col"],
            angle_unit=opts["angle_unit"],
            pressure_unit=opts["pressure_unit"],
            theta_min=opts["theta_min"],
            theta_max=opts["theta_max"],
        )
    except DataError as e:
        typer.secho(f"Dados inválidos: {e}", fg="red", err=True)
        raise typer.Exit(EXIT_BADINPUT)
    return cfg, theta, P, resumo


def _output_dir(output: Path, overwrite: bool) -> Path:
    """Prepara o diretório de saída (nunca sobrescreve sem permissão)."""
    if output.exists() and any(output.iterdir()) and not overwrite:
        typer.secho(
            f"Diretório de saída não vazio: {output}\n"
            "Use --overwrite para sobrescrever ou escolha outro diretório.",
            fg="red", err=True)
        raise typer.Exit(EXIT_BADINPUT)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _imprimir_resumo(res, calibration: Optional[dict], verbose: bool) -> None:
    """Resumo legível da simulação/calibração no stdout."""
    m = res.metrics
    i = res.indicators
    typer.secho("─" * 64, fg="bright_black")
    typer.secho("Double Wiebe — resumo da execução", fg="cyan", bold=True)
    typer.secho(f"  RMSE                : {m['RMSE_kPa']:.6g} kPa")
    typer.secho(f"  MAE                 : {m['MAE_kPa']:.6g} kPa")
    typer.secho(f"  R²                  : {m['R2']:.6f}")
    typer.secho(f"  erro máx. absoluto  : {m['max_abs_error_kPa']:.6g} kPa")
    typer.secho(f"  P máx               : {i['P_max_kPa']:.6g} kPa "
                f"@ θ = {i['theta_at_P_max_deg']:.4g}°")
    typer.secho(f"  T máx               : {i['T_max_K']:.6g} K")
    typer.secho(f"  x_b final           : {i['final_burned_fraction']:.6f}")
    typer.secho(f"  energia fase 1 / 2  : "
                f"{i['phase1_energy_share_pct']:.1f}% / "
                f"{i['phase2_energy_share_pct']:.1f}%")
    typer.secho(f"  calor perdido       : {i['wall_heat_loss_kJ']:.6g} kJ")
    typer.secho(f"  modo Wiebe          : {res.mode}")
    if calibration is not None:
        typer.secho("─" * 64, fg="bright_black")
        typer.secho(
            f"  calibração          : {calibration['method']} | "
            f"semente {calibration['seed']} | "
            f"{calibration['iteracoes']} iterações", fg="cyan")
        if calibration.get("backend") and calibration["backend"] != "serial":
            typer.secho(
                f"  backend             : {calibration['backend']} | "
                f"tempo {calibration.get('tempo_s', float('nan')):.2f} s | "
                f"diferença do integrador (re-run serial): "
                f"{calibration.get('diferenca_integrador', float('nan')):.3e}",
                fg="cyan")
        for nome, valor in calibration["params"].items():
            ini = calibration["params_initial"][nome]
            marcado = "*" if nome in calibration["selected"] else " "
            typer.secho(f"   {marcado} {nome:<8s}: {ini:.6g} -> {valor:.6g}")
        for alerta in calibration.get("alerts", []):
            typer.secho(f"  {alerta}", fg="yellow")
    if verbose:
        typer.secho("─" * 64, fg="bright_black")
        typer.echo("Parâmetros do motor:")
        for k, v in res.engine.items():
            typer.echo(f"    {k} = {v}")
        typer.echo("Parâmetros do Wiebe:")
        for k, v in res.wiebe.items():
            typer.echo(f"    {k} = {v}")


# =============================================================================
# double-wiebe simulate
# =============================================================================
@app.command()
def simulate(
    data: Optional[Path] = typer.Option(
        None, "--data", help="Arquivo experimental (.txt/.csv/.tsv)."),
    config: Optional[Path] = typer.Option(
        None, "--config", help="Arquivo YAML de configuração."),
    output: Path = typer.Option(
        Path("outputs/simulacao"), "--output", "-o",
        help="Diretório de saída dos resultados."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Permite sobrescrever o diretório de saída."),
    set_opt: List[str] = typer.Option(
        [], "--set", help="Sobrescreve configuração: secao.chave=valor "
        "(ex.: wiebe.alpha=0.4; repetível)."),
    ang_unit: Optional[str] = typer.Option(
        None, "--ang-unit", help="Unidade do ângulo: radianos|graus "
        "(default: seção data do YAML ou radianos)."),
    press_unit: Optional[str] = typer.Option(
        None, "--press-unit", help="Unidade da pressão: Pa|kPa|bar "
        "(default: seção data do YAML ou kPa)."),
    theta_min: Optional[float] = typer.Option(
        None, "--theta-min", help="Filtro: ângulo mínimo [rad] "
        "(default: seção data do YAML)."),
    theta_max: Optional[float] = typer.Option(
        None, "--theta-max", help="Filtro: ângulo máximo [rad] "
        "(default: seção data do YAML)."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Modo silencioso."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Modo detalhado."),
):
    """Simula o modelo Double Wiebe com os parâmetros configurados."""
    try:
        cfg, theta, P, resumo = _carregar(
            data, config, _parse_sets(set_opt),
            ang_unit=ang_unit, press_unit=press_unit,
            theta_min=theta_min, theta_max=theta_max)
        from .simulation import run_simulation
        res = run_simulation(theta, P, cfg["engine"], cfg["wiebe"],
                             cfg["simulation"])
        out = _output_dir(output, overwrite)
        from .reporting import export_all
        arquivos = export_all(res, out)
    except typer.Exit:
        raise
    except ValueError as e:
        typer.secho(f"Configuração inválida: {e}", fg="red", err=True)
        raise typer.Exit(EXIT_BADINPUT)
    except Exception as e:                      # falha de integração, etc.
        typer.secho(f"Falha na simulação: {e}", fg="red", err=True)
        raise typer.Exit(EXIT_FAIL)

    if not quiet:
        _imprimir_resumo(res, None, verbose)
        typer.secho("  arquivos gerados:", fg="bright_black")
        for a in arquivos:
            typer.echo(f"    {a}")
    raise typer.Exit(EXIT_OK)


# =============================================================================
# double-wiebe calibrate
# =============================================================================
@app.command()
def calibrate(
    data: Optional[Path] = typer.Option(
        None, "--data", help="Arquivo experimental (.txt/.csv/.tsv)."),
    config: Optional[Path] = typer.Option(
        None, "--config", help="Arquivo YAML de configuração."),
    output: Path = typer.Option(
        Path("outputs/calibracao"), "--output", help="Diretório de saída."),
    overwrite: bool = typer.Option(False, "--overwrite",
                                   help="Sobrescreve o diretório de saída."),
    method: Optional[str] = typer.Option(
        None, "--method", help="differential-evolution | pso | least-squares "
        "(sobrescreve o YAML)."),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Semente aleatória (reprodutibilidade)."),
    maxiter: Optional[int] = typer.Option(
        None, "--maxiter", help="Máximo de iterações da busca global."),
    popsize: Optional[int] = typer.Option(
        None, "--popsize", help="Tamanho da população (DE)."),
    polish: Optional[bool] = typer.Option(
        None, "--polish/--no-polish", help="Refinamento least-squares final."),
    select: Optional[str] = typer.Option(
        None, "--select", help="Parâmetros livres separados por vírgula "
        "(ex.: Rc,m1,alpha)."),
    # --- desempenho (plano HPC) -------------------------------------------
    backend: Optional[str] = typer.Option(
        None, "--backend", help="serial | cpu | cpu-parallel | cuda | auto "
        "(default: serial)."),
    integrator: Optional[str] = typer.Option(
        None, "--integrator", help="Integrador do backend cpu: auto | "
        "rk4_numpy | rk4_numba | scipy (default: auto). O backend cuda "
        "usa sempre rk4_cuda."),
    workers: Optional[int] = typer.Option(
        None, "--workers", help="Processos do cpu-parallel (default: "
        "núcleos-1)."),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", help="Candidatos por lote dos backends cpu/cuda "
        "(0 = todos de uma vez)."),
    precision: Optional[str] = typer.Option(
        None, "--precision", help="float64 | float32 (só no modo acelerado)."),
    substeps: Optional[int] = typer.Option(
        None, "--substeps", help="Sub-passos do RK4 em lote (default: 4)."),
    benchmark: bool = typer.Option(
        False, "--benchmark", help="Executa o benchmark dos backends antes "
        "da calibração e grava benchmark_hpc.csv na saída."),
    profile: bool = typer.Option(
        False, "--profile", help="Perfil (cProfile) da calibração; grava "
        "profile_calibracao.pstats na saída."),
    # -----------------------------------------------------------------------
    set_opt: List[str] = typer.Option(
        [], "--set", help="Sobrescritas secao.chave=valor (repetível)."),
    ang_unit: Optional[str] = typer.Option(None, "--ang-unit"),
    press_unit: Optional[str] = typer.Option(None, "--press-unit"),
    theta_min: Optional[float] = typer.Option(None, "--theta-min"),
    theta_max: Optional[float] = typer.Option(None, "--theta-max"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Calibra os parâmetros selecionados (RMSE + regularização) e roda a
    simulação final com o resultado calibrado."""
    try:
        overrides = _parse_sets(set_opt)
        if method is not None:
            overrides["calibration.method"] = method
        if seed is not None:
            overrides["calibration.seed"] = seed
        if maxiter is not None:
            overrides["calibration.maxiter"] = maxiter
        if popsize is not None:
            overrides["calibration.popsize"] = popsize
        if polish is not None:
            overrides["calibration.polish"] = polish
        if select is not None:
            overrides["calibration.selected"] = [
                s.strip() for s in select.split(",") if s.strip()]
        if backend is not None:
            overrides["calibration.backend"] = backend
        if integrator is not None:
            overrides["calibration.integrator"] = integrator
        if workers is not None:
            overrides["calibration.workers"] = workers
        if batch_size is not None:
            overrides["calibration.batch_size"] = batch_size
        if precision is not None:
            overrides["calibration.precision"] = precision
        if substeps is not None:
            overrides["calibration.substeps"] = substeps
        cfg, theta, P, resumo = _carregar(
            data, config, overrides,
            ang_unit=ang_unit, press_unit=press_unit,
            theta_min=theta_min, theta_max=theta_max)
        out = _output_dir(output, overwrite)

        from .calibration import CalibrationError, run_calibration

        # Benchmark reproduzível antes da calibração (regra 7 do plano HPC)
        if benchmark:
            from .calibration import _full0
            from .backends.benchmark import print_benchmark, run_benchmark
            selected = list(cfg["calibration"].selected)
            full0 = _full0(cfg["engine"], cfg["wiebe"])
            linhas = run_benchmark(
                theta, P, cfg["engine"], cfg["wiebe"], cfg["simulation"],
                cfg["calibration"], selected, full0,
                csv_path=str(out / "benchmark_hpc.csv"),
                substeps=cfg["calibration"].substeps)
            typer.secho("Benchmark dos backends (warm-up + 3 réplicas):",
                        fg="cyan")
            typer.echo(print_benchmark(linhas))
            typer.echo(f"  CSV: {out / 'benchmark_hpc.csv'}")

        if profile:
            import cProfile
            profiler = cProfile.Profile()

            profiler.enable()
            calibracao = run_calibration(
                theta, P, cfg["engine"], cfg["wiebe"], cfg["simulation"],
                cfg["calibration"])
            profiler.disable()
            stats_path = out / "profile_calibracao.pstats"
            profiler.dump_stats(str(stats_path))
            import pstats
            st = pstats.Stats(str(stats_path))
            st.sort_stats("cumulative").print_stats(15)
        else:
            calibracao = run_calibration(
                theta, P, cfg["engine"], cfg["wiebe"], cfg["simulation"],
                cfg["calibration"])
    except typer.Exit:
        raise
    except (CalibrationError, ValueError) as e:
        typer.secho(f"Calibração inválida: {e}", fg="red", err=True)
        raise typer.Exit(EXIT_BADINPUT)
    except Exception as e:
        typer.secho(f"Falha na calibração: {e}", fg="red", err=True)
        raise typer.Exit(EXIT_FAIL)

    try:
        # Simulação final com os parâmetros calibrados (mesmo núcleo)
        from .calibration import apply_calibrated
        from .simulation import run_simulation
        eng, wieb = apply_calibrated(calibracao, cfg["engine"], cfg["wiebe"])
        res = run_simulation(theta, P, eng, wieb, cfg["simulation"])
        from .reporting import export_all
        arquivos = export_all(res, out, calibracao)
    except ValueError as e:
        typer.secho(f"Parâmetros calibrados inválidos: {e}", fg="red", err=True)
        raise typer.Exit(EXIT_FAIL)

    if not quiet:
        _imprimir_resumo(res, calibracao, verbose)
        typer.secho("  arquivos gerados:", fg="bright_black")
        for a in arquivos:
            typer.echo(f"    {a}")
    raise typer.Exit(EXIT_OK)


# =============================================================================
# double-wiebe devices
# =============================================================================
@app.command()
def devices():
    """Mostra o hardware detectado (CPU, numba, CUDA, OpenCL)."""
    from .backends import get_available_backends, hardware_report
    typer.echo(hardware_report())
    disponiveis = ", ".join(sorted(get_available_backends()))
    typer.secho(f"\nBackends disponíveis: {disponiveis}", fg="cyan")
    raise typer.Exit(EXIT_OK)


# =============================================================================
# double-wiebe benchmark
# =============================================================================
@app.command()
def benchmark(
    data: Optional[Path] = typer.Option(
        None, "--data", help="Arquivo experimental (.txt/.csv/.tsv). Sem "
        "--data usa uma população sintética (tempos relativos)."),
    config: Optional[Path] = typer.Option(
        None, "--config", help="Arquivo YAML de configuração."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="CSV de saída (default: sem CSV)."),
    overwrite: bool = typer.Option(False, "--overwrite"),
    set_opt: List[str] = typer.Option([], "--set"),
    tamanhos: str = typer.Option(
        "1,20,100,1000", "--tamanhos",
        help="Tamanhos de população, separados por vírgula."),
    rep: int = typer.Option(3, "--rep", help="Réplicas por medida (>= 2)."),
    ang_unit: Optional[str] = typer.Option(None, "--ang-unit"),
    press_unit: Optional[str] = typer.Option(None, "--press-unit"),
):
    """Compara os backends (serial, cpu RK4 NumPy/Numba, cpu-parallel, cuda) com
    warm-up, média e desvio — benchmark reproduzível (regra 7 do plano HPC)."""
    try:
        from .backends.benchmark import print_benchmark, run_benchmark
        from .calibration import _full0
        from .models import DEFAULT_SELECTED

        cfg = None
        if data is not None:
            cfg, theta, P, _ = _carregar(
                data, config, _parse_sets(set_opt), ang_unit=ang_unit,
                press_unit=press_unit)
            engine, wiebe, sim, calib = (cfg["engine"], cfg["wiebe"],
                                         cfg["simulation"],
                                         cfg["calibration"])
            selected = list(cfg["calibration"].selected)
            full0 = _full0(engine, wiebe)
        else:
            # sem dados: sintético leve, só para medir tempo relativo
            import numpy as np
            from .models import (CalibrationConfig, EngineConfig,
                                 SimulationConfig, WiebeParameters)
            theta = np.linspace(-2.0, 2.0, 240)
            P = np.exp(1.37 * (2.0 - theta)) * 138.2
            engine, wiebe = EngineConfig(), WiebeParameters()
            sim, calib = SimulationConfig(), CalibrationConfig()
            selected = list(DEFAULT_SELECTED)
            full0 = _full0(engine, wiebe)

        if output is not None:
            p_out = Path(output)
            p_out.parent.mkdir(parents=True, exist_ok=True)
            csv_path = (str(p_out) if p_out.suffix
                        else str(p_out / "benchmark_hpc.csv"))
        else:
            csv_path = None
        linhas = run_benchmark(
            theta, P, engine, wiebe, sim, calib, selected, full0,
            tamanhos=tuple(int(t) for t in tamanhos.split(",") if t.strip()),
            rep=max(2, rep), csv_path=csv_path,
            substeps=cfg["calibration"].substeps
            if cfg is not None else 4)
        typer.echo(print_benchmark(linhas))
        if csv_path:
            typer.secho(f"CSV: {csv_path}", fg="green")
    except typer.Exit:
        raise
    except Exception as e:
        typer.secho(f"Falha no benchmark: {e}", fg="red", err=True)
        raise typer.Exit(EXIT_FAIL)
    raise typer.Exit(EXIT_OK)


# =============================================================================
# double-wiebe validate
# =============================================================================
@app.command()
def validate(
    data: Optional[Path] = typer.Option(
        None, "--data", help="Arquivo experimental para validar."),
    config: Optional[Path] = typer.Option(
        None, "--config", help="Arquivo YAML para validar."),
    set_opt: List[str] = typer.Option([], "--set"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Valida a configuração e os dados SEM executar a simulação."""
    problemas: List[str] = []
    try:
        from .models import load_config
        cfg = load_config(str(config) if config else None, _parse_sets(set_opt))
        for secao in ("engine", "wiebe", "simulation", "calibration"):
            erros = cfg[secao].validate()
            for e in erros:
                problemas.append(f"[{secao}] {e}")
    except Exception as e:
        problemas.append(f"[config] {e}")
        cfg = None

    if data is not None and cfg is not None and problemas == []:
        try:
            from .data_processing import read_table, prepare_series
            df = read_table(data)
            theta, P, resumo = prepare_series(df)
            typer.secho(f"dados ok: {resumo['n_obs']} observações "
                        f"(θ ∈ [{resumo['theta_min']:.4g}, "
                        f"{resumo['theta_max']:.4g}] rad; "
                        f"P ∈ [{resumo['P_min']:.4g}, {resumo['P_max']:.4g}] "
                        "kPa)", fg="green")
            if verbose:
                for k, v in resumo.items():
                    typer.echo(f"    {k} = {v}")
        except Exception as e:
            problemas.append(f"[data] {e}")
    elif data is None:
        typer.echo("nenhum --data informado (validação de dados ignorada).")

    if problemas:
        for p in problemas:
            typer.secho(f"✗ {p}", fg="red")
        raise typer.Exit(EXIT_BADINPUT)
    typer.secho("✓ configuração válida — nenhuma simulação executada",
                fg="green")
    raise typer.Exit(EXIT_OK)


# =============================================================================
# double-wiebe plot
# =============================================================================
@app.command()
def plot(
    run_dir: Path = typer.Argument(
        ..., help="Diretório de um resultado (contém results.csv e "
        "parameters.yaml)."),
    outdir: Optional[Path] = typer.Option(
        None, "--out", help="Diretório dos gráficos (default: o próprio "
        "run_dir)."),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """Regenera os gráficos estáticos a partir de um resultado exportado."""
    try:
        results_csv = Path(run_dir) / "results.csv"
        params_yaml = Path(run_dir) / "parameters.yaml"
        if not results_csv.exists() or not params_yaml.exists():
            typer.secho("Diretório sem results.csv/parameters.yaml: "
                        f"{run_dir}", fg="red", err=True)
            raise typer.Exit(EXIT_BADINPUT)
        from .data_processing import read_table, prepare_series
        df = read_table(results_csv, sep=";")
        theta, P, _ = prepare_series(df, angle_col=0, pressure_col=2,
                                     angle_unit="radianos",
                                     pressure_unit="kPa", min_points=2)

        # Recria o resultado a partir das colunas exportadas
        from .simulation import SimulationResult
        col = {c: i for i, c in enumerate(df.columns)}
        def serie(nome): return df.iloc[:, col[nome]].to_numpy(float)
        res = SimulationResult(
            theta=theta, P_exp=serie("pressure_exp_kPa"),
            P_sim=serie("pressure_sim_kPa"), Tg=serie("temperature_K"),
            Q_wall=serie("wall_heat_J"), xb1=serie("xb_phase1"),
            xb2=serie("xb_phase2"), xb_total=serie("xb_total"),
            dQ1=serie("dQ1_dtheta_kJ_per_rad"),
            dQ2=serie("dQ2_dtheta_kJ_per_rad"),
            dQ_total=serie("dQ_total_dtheta_kJ_per_rad"),
            volume=serie("volume_m3"),
        )
        # RMSE recalculado (metrics ausentes no CSV)
        from .metrics import summary_metrics
        res.metrics.update(summary_metrics(res.P_exp, res.P_sim))

        import yaml
        params = yaml.safe_load(params_yaml.read_text(encoding="utf-8"))
        res.wiebe.update(params.get("wiebe", {}))

        out = Path(outdir) if outdir else Path(run_dir)
        if out.exists() and any(out.iterdir()) and not overwrite:
            typer.secho(f"Diretório não vazio: {out} (use --overwrite)",
                        fg="red", err=True)
            raise typer.Exit(EXIT_BADINPUT)
        from .reporting import static_plots
        gerados = static_plots(res, out)
    except typer.Exit:
        raise
    except Exception as e:
        typer.secho(f"Falha ao gerar gráficos: {e}", fg="red", err=True)
        raise typer.Exit(EXIT_FAIL)
    for nome, caminho in gerados.items():
        typer.echo(f"  {caminho}")
    raise typer.Exit(EXIT_OK)


# =============================================================================
# double-wiebe example-config
# =============================================================================
@app.command("example-config")
def example_config(
    output: Optional[Path] = typer.Argument(
        None, help="Arquivo YAML de destino (opcional; sem ele imprime na "
        "tela)."),
    force: bool = typer.Option(False, "--force", help="Sobrescreve o arquivo "
                               "de destino se já existir."),
):
    """Imprime (ou grava) o YAML de configuração de exemplo."""
    from .models import (CalibrationConfig, EngineConfig, SimulationConfig,
                         WiebeParameters, dump_config_yaml)
    texto = dump_config_yaml(EngineConfig(), WiebeParameters(),
                             SimulationConfig(), CalibrationConfig())
    if output is None:
        typer.echo(texto)
        raise typer.Exit(EXIT_OK)
    output = Path(output)
    if output.exists() and not force:
        typer.secho(f"Arquivo já existe: {output} (use --force)",
                    fg="red", err=True)
        raise typer.Exit(EXIT_BADINPUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(texto, encoding="utf-8")
    typer.secho(f"Configuração gravada em {output}", fg="green")
    raise typer.Exit(EXIT_OK)


# =============================================================================
# double-wiebe gui
# =============================================================================
@app.command()
def gui(
    port: int = typer.Option(8501, "--port", help="Porta do Streamlit."),
    headless: bool = typer.Option(False, "--headless",
                                  help="Não abre o navegador."),
):
    """Inicia a interface gráfica (Streamlit)."""
    from importlib.util import find_spec
    if find_spec("streamlit") is None:
        typer.secho("Streamlit não instalado. Execute:\n"
                    "  pip install streamlit", fg="red", err=True)
        raise typer.Exit(EXIT_BADINPUT)
    gui_path = Path(__file__).with_name("gui.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(gui_path),
           "--server.port", str(port),
           "--browser.gatherUsageStats", "false"]
    if headless:
        cmd.append("--server.headless=true")
    typer.secho(f"Iniciando a GUI em http://localhost:{port} ...",
                fg="cyan")
    try:
        return_code = subprocess.call(cmd)
    except KeyboardInterrupt:
        return_code = EXIT_OK
    raise typer.Exit(EXIT_OK if return_code == 0 else EXIT_FAIL)


def main() -> None:
    """Ponto de entrada do console script (double-wiebe)."""
    try:
        app()
    except SystemExit as e:
        sys.exit(e.code)


if __name__ == "__main__":  # python -m double_wiebe.cli
    main()