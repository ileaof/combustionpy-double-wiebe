# -*- coding: utf-8 -*-
"""
reporting.py
============
Exportações do Double Wiebe (idênticas na CLI e na GUI — ambas chamam estas
funções sobre um :class:`~double_wiebe.simulation.SimulationResult`):

    results.csv               tabela completa (14 colunas, ';' decimal ponto)
    parameters.yaml           parâmetros do motor + Wiebe + simulação
    metrics.json              métricas de ajuste + indicadores
    convergence.csv           histórico da calibração
    pressure_comparison.png / .pdf
    heat_release.png
    burned_fraction.png
    report.html               relatório autocontido (imagens embutidas)
"""
from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

RESULTS_CSV_COLUMNS = [
    "theta_rad", "theta_deg",
    "pressure_exp_kPa", "pressure_sim_kPa", "residual_kPa",
    "temperature_K", "wall_heat_J", "volume_m3",
    "xb_phase1", "xb_phase2", "xb_total",
    "dQ1_dtheta_kJ_per_rad", "dQ2_dtheta_kJ_per_rad",
    "dQ_total_dtheta_kJ_per_rad",
]


# =============================================================================
# DataFrame central (também usado pela GUI para prévia de tabela)
# =============================================================================
def results_dataframe(res) -> pd.DataFrame:
    """DataFrame com as 14 colunas do results.csv."""
    return pd.DataFrame({
        "theta_rad": res.theta,
        "theta_deg": np.degrees(res.theta),
        "pressure_exp_kPa": res.P_exp,
        "pressure_sim_kPa": res.P_sim,
        "residual_kPa": res.P_exp - res.P_sim,
        "temperature_K": res.Tg,
        "wall_heat_J": res.Q_wall,
        "volume_m3": res.volume,
        "xb_phase1": res.xb1,
        "xb_phase2": res.xb2,
        "xb_total": res.xb_total,
        "dQ1_dtheta_kJ_per_rad": res.dQ1,
        "dQ2_dtheta_kJ_per_rad": res.dQ2,
        "dQ_total_dtheta_kJ_per_rad": res.dQ_total,
    })


def results_csv_bytes(res) -> bytes:
    """results.csv em bytes (CSV ';' — abre direto no Excel pt-BR)."""
    return results_dataframe(res).to_csv(index=False, sep=";",
                                         float_format="%.9g").encode("utf-8")


def results_csv_path(res, outdir: str | Path) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.csv"
    path.write_bytes(results_csv_bytes(res))
    return path


# =============================================================================
# parameters.yaml
# =============================================================================
def parameters_yaml_bytes(res) -> bytes:
    """parameters.yaml — parâmetros usados na simulação (do snapshot)."""
    import yaml
    return yaml.safe_dump(
        {"engine": res.engine, "wiebe": res.wiebe, "simulation": res.simulation},
        sort_keys=False, allow_unicode=True,
    ).encode("utf-8")


def parameters_yaml_path(res, outdir: str | Path) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "parameters.yaml"
    path.write_bytes(parameters_yaml_bytes(res))
    return path


# =============================================================================
# metrics.json
# =============================================================================
def metrics_dict(res, calibration: Optional[Dict] = None) -> Dict:
    """Dicionário completo de métricas/indicadores (para metrics.json)."""
    doc: Dict = {
        "metrics": res.metrics,
        "indicators": res.indicators,
        "wiebe_mode": res.mode,
        "engine": res.engine,
        "wiebe": res.wiebe,
        "simulation": res.simulation,
    }
    if calibration is not None:
        doc["calibration"] = {
            "method": calibration.get("method"),
            "seed": calibration.get("seed"),
            "iteracoes": calibration.get("iteracoes"),
            "rmse": calibration.get("rmse"),
            "message": calibration.get("message"),
            "selected": calibration.get("selected"),
            "params_initial": calibration.get("params_initial"),
            "params_calibrated": calibration.get("params"),
            "polish": calibration.get("polish"),
            "sensitivity": calibration.get("sensitivity"),
            "alerts": calibration.get("alerts"),
        }
    return doc


def metrics_json_bytes(res, calibration: Optional[Dict] = None) -> bytes:
    return json.dumps(metrics_dict(res, calibration), indent=2,
                      ensure_ascii=False).encode("utf-8")


def metrics_json_path(res, outdir: str | Path,
                      calibration: Optional[Dict] = None) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    path.write_bytes(metrics_json_bytes(res, calibration))
    return path


# =============================================================================
# Arquivo de calibração (salvar / abrir)
# =============================================================================
_CALIB_FORMATO = "double-wiebe-calibracao"
_CALIB_VERSAO = 1

# Chaves do calib_result serializadas em seções próprias do arquivo
_CALIB_CHAVES_SECAO = ("theta_exp", "P_exp", "engine", "wiebe",
                       "history_params", "objective_history")


def _json_safe(obj):
    """Converte tipos NumPy/containers para tipos JSON nativos (recursivo)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    return obj if obj is None or isinstance(obj, str) else str(obj)


def calibration_json_bytes(calibracao: Dict, data_name: str = "") -> bytes:
    """Serializa o resultado da calibração num JSON reaberto por
    `read_calibration_json`.

    O arquivo embute: os dados experimentais usados (θ em rad, P em kPa), as
    configurações de motor e Wiebe da busca e o resultado completo (parâmetros,
    histórico, sensibilidade, alertas, HPC). Ângulos em radianos — unidade
    interna do pacote.
    """
    doc = {
        "analise": {
            "titulo": "Double Wiebe Combustion Analysis — Calibração",
            "formato": _CALIB_FORMATO,
            "versao": _CALIB_VERSAO,
            "data_hora": datetime.now().isoformat(timespec="seconds"),
            "arquivo_experimental": data_name,
        },
        "dados": {
            "theta_rad": _json_safe(calibracao.get("theta_exp")),
            "pressao_kPa": _json_safe(calibracao.get("P_exp")),
        },
        "motor": _json_safe(calibracao.get("engine") or {}),
        "wiebe": _json_safe(calibracao.get("wiebe") or {}),
        "calibracao": _json_safe({k: calibracao[k] for k in calibracao
                                  if k not in _CALIB_CHAVES_SECAO}),
        "historico": {
            "rmse_kPa": _json_safe(calibracao.get("objective_history")),
            "parametros": _json_safe(calibracao.get("history_params")),
        },
    }
    return json.dumps(doc, indent=2, ensure_ascii=False).encode("utf-8")


def read_calibration_json(arquivo) -> Dict:
    """Lê um arquivo salvo por `calibration_json_bytes` e devolve um
    dicionário com: 'calibracao' (o calib_result reconstruído no formato
    consumido pela CLI/GUI), 'theta'/'pressao' (np.ndarray ou None — só
    restaura o par completo), 'motor', 'wiebe' e 'arquivo_experimental'.
    Lança ValueError em arquivo inválido.
    """
    try:
        texto = arquivo.read() if hasattr(arquivo, "read") else str(arquivo)
        if isinstance(texto, bytes):
            texto = texto.decode("utf-8")
        doc = json.loads(texto)
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"não é um JSON válido ({e}).") from e
    if (not isinstance(doc, dict)
            or (doc.get("analise") or {}).get("formato") != _CALIB_FORMATO):
        raise ValueError('esperado um arquivo salvo pela GUI '
                         f'("formato": "{_CALIB_FORMATO}").')
    c = doc.get("calibracao") or {}
    if "params" not in c or "rmse" not in c:
        raise ValueError('seção "calibracao" incompleta '
                         "(params/rmse ausentes).")
    params = {k: float(v) for k, v in c["params"].items()}
    if not all(np.isfinite(v) for v in params.values()):
        raise ValueError("parâmetros não finitos no arquivo.")

    hist = doc.get("historico") or {}
    calib = {
        "params": params,
        "params_initial": {k: float(v) for k, v in
                           (c.get("params_initial") or params).items()},
        "selected": list(c.get("selected") or []),
        "rmse": float(c["rmse"]),
        "objective_history": [float(h) for h in (hist.get("rmse_kPa") or [])],
        "history_params": [np.asarray(p, dtype=float)
                           for p in (hist.get("parametros") or [])],
        "method": str(c.get("method", "?")),
        "seed": c.get("seed"),
        "iteracoes": int(c.get("iteracoes", 0)),
        "message": str(c.get("message", "")),
        "success": bool(c.get("success", True)),
        "cancelado": False,
        "parou_por_estagnacao": bool(c.get("parou_por_estagnacao", False)),
        "polish": c.get("polish"),
        "sensitivity": list(c.get("sensitivity") or []),
        "alerts": list(c.get("alerts") or []),
        "engine": {k: v for k, v in (doc.get("motor") or {}).items()},
        "wiebe": {k: v for k, v in (doc.get("wiebe") or {}).items()},
        "theta_exp": None,
        "P_exp": None,
        "backend": str(c.get("backend", "serial")),
    }
    for chave in ("rmse_integrador", "diferenca_integrador", "tempo_s"):
        if c.get(chave) is not None:
            calib[chave] = float(c[chave])

    dados = doc.get("dados") or {}
    theta = (np.asarray(dados["theta_rad"], dtype=float)
             if dados.get("theta_rad") else None)
    pressao = (np.asarray(dados["pressao_kPa"], dtype=float)
               if dados.get("pressao_kPa") else None)
    if theta is None or pressao is None or theta.size != pressao.size:
        theta = pressao = None

    return {
        "calibracao": calib,
        "theta": theta,
        "pressao": pressao,
        "motor": dict(doc.get("motor") or {}),
        "wiebe": dict(doc.get("wiebe") or {}),
        "arquivo_experimental": str((doc.get("analise") or {})
                                    .get("arquivo_experimental") or ""),
    }


# =============================================================================
# convergence.csv
# =============================================================================
def convergence_csv_bytes(calibration: Dict) -> bytes:
    """Histórico da função objetivo (RMSE por iteração) + parâmetros."""
    hist = calibration.get("objective_history") or []
    hp = calibration.get("history_params") or []
    colunas = ["iteration", "rmse_kPa"] + list(calibration.get(
        "params", {}).keys())
    linhas: List[str] = [";".join(colunas)]
    for i, f in enumerate(hist):
        vals = [f"{i + 1}", f"{f:.9g}"]
        if i < len(hp):
            vals += [f"{float(v):.9g}" for v in np.asarray(hp[i])]
        linhas.append(";".join(vals))
    return ("\n".join(linhas) + "\n").encode("utf-8")


def convergence_csv_path(calibration: Dict, outdir: str | Path) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "convergence.csv"
    path.write_bytes(convergence_csv_bytes(calibration))
    return path


# =============================================================================
# Plots estáticos e PDF
# =============================================================================
def static_plots(res, outdir: str | Path) -> Dict[str, Path]:
    """Gera os 3 PNGs estáticos + o PDF (páginas única, via PdfPages)."""
    from matplotlib.backends.backend_pdf import PdfPages

    from .plotting import (static_burned_fraction, static_heat_release,
                           static_pressure_comparison, _plt)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    png_pressure = static_pressure_comparison(res, str(out / "pressure_comparison.png"))
    png_heat = static_heat_release(res, str(out / "heat_release.png"))
    png_burned = static_burned_fraction(res, str(out / "burned_fraction.png"))

    # PDF com as três figuras (redesenho direto no PdfPages)
    pdf_path = out / "pressure_comparison.pdf"
    plt = _plt()
    figs = []
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(res.theta, res.P_exp, "r+", ms=4, label="Experimental")
    ax.plot(res.theta, res.P_sim, "g-", lw=1.5, label="Modelo (Double Wiebe)")
    ax.set_xlabel("Ângulo do virabrequim [rad]")
    ax.set_ylabel("Pressão no cilindro [kPa]")
    ax.legend()
    figs.append(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(res.theta, res.dQ1, color="#1f77b4", lw=1.4, label="dQ₁/dθ")
    ax.plot(res.theta, res.dQ2, color="#ff7f0e", lw=1.4, label="dQ₂/dθ")
    ax.plot(res.theta, res.dQ_total, color="#2ca02c", lw=2.0, label="total")
    ax.set_xlabel("Ângulo do virabrequim [rad]")
    ax.set_ylabel("Taxa de liberação de calor [kJ/rad]")
    ax.legend()
    figs.append(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(res.theta, res.xb1, color="#1f77b4", lw=1.4, label="x₁")
    ax.plot(res.theta, res.xb2, color="#ff7f0e", lw=1.4, label="x₂")
    ax.plot(res.theta, res.xb_total, color="#2ca02c", lw=2.0, label="x_b")
    ax.set_xlabel("Ângulo do virabrequim [rad]")
    ax.set_ylabel("Fração queimada [-]")
    ax.legend()
    figs.append(fig)
    with PdfPages(str(pdf_path)) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return {
        "pressure_comparison.png": png_pressure,
        "pressure_comparison.pdf": str(pdf_path),
        "heat_release.png": png_heat,
        "burned_fraction.png": png_burned,
    }


# =============================================================================
# Relatório HTML
# =============================================================================
def _tabela(rows: List[List], floats: bool = True) -> str:
    """Tabela HTML simples (dados próprios do relatório, não entrada de
    usuário — sem risco de injeção)."""
    html = ["<table border='1' cellpadding='4' cellspacing='0' "
            "style='border-collapse:collapse'>"]
    for row in rows:
        html.append("<tr>")
        for cel in row:
            valor = f"{cel:.6g}" if (floats and isinstance(cel, float)) else str(cel)
            html.append(f"<td>{valor}</td>")
        html.append("</tr>")
    html.append("</table>")
    return "".join(html)


def _png_b64(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def report_html_bytes(
    res,
    calibration: Optional[Dict] = None,
    figures_png: Optional[Dict[str, bytes]] = None,
) -> bytes:
    """Relatório HTML autocontido (CSS inline, imagens embutidas em base64).
    Apenas valores numéricos/strings do próprio relatório são interpolados;
    nenhum HTML de usuário é inserido."""
    import html as html_mod

    esc = lambda s: html_mod.escape(str(s))  # noqa: E731

    linhas = []
    linhas.append("<h1>Double Wiebe Combustion Analysis — Relatório</h1>")
    linhas.append("<p><i>This combustion simulation employs a double Wiebe "
                  "function and extends the single Wiebe model developed as "
                  "part of L. Queiroz's M.Sc. thesis under the supervision "
                  "of Prof. I. L. Ferreira.</i></p>")
    linhas.append(f"<p>Modo do Wiebe: <b>{esc(res.mode)}</b></p>")

    linhas.append("<h2>Parâmetros</h2>")
    linhas.append(_tabela([["Motor", "Valor"], *[[k, v]
                  for k, v in res.engine.items()]]))
    linhas.append("<br>")
    linhas.append(_tabela([["Wiebe", "Valor"], *[[k, v]
                  for k, v in res.wiebe.items()]]))

    linhas.append("<h2>Métricas de ajuste</h2>")
    linhas.append(_tabela([["Métrica", "Valor"], *[[k, v]
                  for k, v in res.metrics.items()]]))
    linhas.append("<h2>Indicadores termodinâmicos</h2>")
    linhas.append(_tabela([["Indicador", "Valor"], *[[k, v]
                  for k, v in res.indicators.items()]]))

    if calibration is not None:
        linhas.append("<h2>Calibração</h2>")
        linhas.append(f"<p>Método: <b>{esc(calibration.get('method'))}</b> | "
                      f"semente: <b>{esc(calibration.get('seed'))}</b> | "
                      f"iterações: <b>{esc(calibration.get('iteracoes'))}"
                      f"</b> | RMSE: <b>"
                      f"{float(calibration.get('rmse', float('nan'))):.6g}"
                      f" kPa</b></p>")
        tabela = [["Parâmetro", "Inicial", "Calibrado"]]
        p0 = calibration.get("params_initial", {})
        p1 = calibration.get("params", {})
        for nome, v1 in p1.items():
            tabela.append([nome, p0.get(nome, float("nan")), v1])
        linhas.append(_tabela(tabela))
        sens = calibration.get("sensitivity") or []
        if sens:
            linhas.append("<h3>Sensibilidade (ΔRMSE para ±1%)</h3>")
            linhas.append(_tabela([["Parâmetro", "ΔRMSE [kPa]", "Insensível"]]
                          + [[s["param"], s["delta_rmse"],
                              "SIM" if s["insensitive"] else "não"]
                             for s in sens]))
        for alerta in calibration.get("alerts") or []:
            linhas.append(f"<p style='color:#a15c00'>⚠ {esc(alerta)}</p>")

    if figures_png is not None:
        linhas.append("<h2>Gráficos</h2>")
        for nome, png in figures_png.items():
            linhas.append(f"<h3>{esc(nome)}</h3>")
            linhas.append(f"<img src='{_png_b64(png)}' style='max-width:100%'>")

    html_doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Double Wiebe — Relatório</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;"
        "color:#222}table{font-size:13px}h2{border-bottom:1px solid #ccc}"
        "</style></head><body>" + "".join(linhas) + "</body></html>"
    )
    return html_doc.encode("utf-8")


def report_html_path(res, outdir: str | Path,
                     calibration: Optional[Dict] = None,
                     figures_png: Optional[Dict[str, bytes]] = None) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.html"
    path.write_bytes(report_html_bytes(res, calibration, figures_png))
    return path


def export_all(
    res,
    outdir: str | Path,
    calibration: Optional[Dict] = None,
) -> List[str]:
    """Exporta TODOS os arquivos de resultado para `outdir`.

    Nunca sobrescreve um diretório com conteúdo sem decisão do chamador —
    a CLI/GUI verificam antes e passam ``overwrite=True`` explicitamente
    (aqui os arquivos individuais são sobrescritos; a decisão de pasta
    nova/antiga é da camada de interação).
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    arquivos: List[str] = []
    arquivos.append(str(results_csv_path(res, out)))
    arquivos.append(str(parameters_yaml_path(res, out)))
    arquivos.append(str(metrics_json_path(res, out, calibration)))
    if calibration is not None:
        arquivos.append(str(convergence_csv_path(calibration, out)))
    estaticos = static_plots(res, out)
    arquivos.extend(estaticos.values())
    # PNGs em bytes para embutir no relatório
    pngs: Dict[str, bytes] = {}
    for nome in ("pressure_comparison.png", "heat_release.png",
                 "burned_fraction.png"):
        pngs[nome] = (out / nome).read_bytes()
    arquivos.append(str(report_html_path(res, out, calibration, pngs)))
    return arquivos