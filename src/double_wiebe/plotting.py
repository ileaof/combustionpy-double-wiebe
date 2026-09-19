# -*- coding: utf-8 -*-
"""
plotting.py
===========
Gráficos do Double Wiebe: interativos (Plotly, GUI) e estáticos
(Matplotlib, somente exportações PNG/PDF).

Estilo do gráfico principal (mesma família do modelo Single Wiebe):
dados experimentais como marcadores "+" vermelhos; modelo como linha verde;
linhas verticais nos inícios das duas fases; regiões sombreadas com as
durações; estrela no pico de pressão. Eixo horizontal alternável entre
graus e radianos; pressão entre kPa e bar.
"""
from __future__ import annotations

import math
import numpy as np

# Paleta (mesma família da GUI do Single Wiebe)
C_EXP = "#d62728"      # vermelho — experimental
C_SIM = "#2ca02c"      # verde — modelo
C_F1 = "#1f77b4"       # azul — fase 1
C_F2 = "#ff7f0e"       # laranja — fase 2
TEMPLATE = "plotly_white"

X_LABEL = {"rad": "Ângulo do virabrequim [rad]",
           "deg": "Ângulo do virabrequim [graus]"}
P_LABEL = {"kPa": "Pressão no cilindro [kPa]", "bar": "Pressão no cilindro [bar]"}


def _eixo(theta: np.ndarray, ang_deg: bool) -> np.ndarray:
    """Converte o eixo horizontal para graus quando selecionado."""
    return np.degrees(theta) if ang_deg else theta


def fig_pressure(res, ang_deg: bool = False, p_unit: str = "kPa"):
    """Gráfico principal: experimental ('+' vermelho) x modelo (linha verde),
    linhas verticais em theta01/theta02, faixas sombreadas das durações e
    estrela no pico de pressão."""
    import plotly.graph_objects as go

    x = _eixo(res.theta, ang_deg)
    fator = 1.0 if p_unit == "kPa" else 0.01
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=res.P_exp * fator, mode="markers",
        marker=dict(symbol="cross-thin", size=6, color=C_EXP,
                    line=dict(width=1.4, color=C_EXP)),
        name="Experimental"))
    fig.add_trace(go.Scatter(
        x=x, y=res.P_sim * fator, mode="lines", line=dict(color=C_SIM, width=2),
        name="Modelo (Double Wiebe)"))

    # Inícios das fases e durações
    x1 = _eixo(np.array([res.wiebe["theta01"]]), ang_deg)[0]
    x2 = _eixo(np.array([res.wiebe["theta02"]]), ang_deg)[0]
    d1 = res.wiebe["delta1"] if not ang_deg else math.degrees(res.wiebe["delta1"])
    d2 = res.wiebe["delta2"] if not ang_deg else math.degrees(res.wiebe["delta2"])
    ymax = float(np.max(res.P_sim) * fator)
    fig.add_vline(x=x1, line=dict(color=C_F1, dash="dash", width=1),
                  annotation_text="theta01")
    fig.add_vline(x=x2, line=dict(color=C_F2, dash="dash", width=1),
                  annotation_text="theta02")
    fig.add_vrect(x0=x1, x1=x1 + d1, fillcolor=C_F1, opacity=0.08,
                  line_width=0, annotation_text="fase 1")
    fig.add_vrect(x0=x2, x1=x2 + d2, fillcolor=C_F2, opacity=0.08,
                  line_width=0, annotation_text="fase 2")

    # Pico de pressão
    i_max = int(np.argmax(res.P_sim))
    fig.add_trace(go.Scatter(
        x=[x[i_max]], y=[res.P_sim[i_max] * fator], mode="markers",
        marker=dict(symbol="star", size=13, color="#9467bd"),
        name=f"P máx = {res.P_sim[i_max] * fator:.0f} {p_unit}"))

    fig.update_layout(
        template=TEMPLATE, legend=dict(orientation="h", y=1.12),
        xaxis_title=X_LABEL["deg" if ang_deg else "rad"],
        yaxis_title=P_LABEL[p_unit],
        title="Pressão simulada × experimental",
    )
    return fig


def fig_residual(res, ang_deg: bool = False):
    """Resíduo (experimental - simulado) em kPa."""
    import plotly.graph_objects as go

    x = _eixo(res.theta, ang_deg)
    fig = go.Figure(go.Scatter(
        x=x, y=res.P_exp - res.P_sim, mode="lines",
        line=dict(color=C_EXP, width=1.2), name="Resíduo"))
    fig.add_hline(y=0.0, line=dict(color="#444", dash="dot"))
    fig.update_layout(
        template=TEMPLATE,
        xaxis_title=X_LABEL["deg" if ang_deg else "rad"],
        yaxis_title="Resíduo [kPa]",
        title="Resíduo da pressão (experimental − simulada)",
    )
    return fig


def fig_burned(res, ang_deg: bool = False):
    """Frações queimadas das duas fases e total."""
    import plotly.graph_objects as go

    x = _eixo(res.theta, ang_deg)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=res.xb1, mode="lines",
                             line=dict(color=C_F1, width=1.8), name="x₁ (pré-misturada)"))
    fig.add_trace(go.Scatter(x=x, y=res.xb2, mode="lines",
                             line=dict(color=C_F2, width=1.8), name="x₂ (difusão)"))
    fig.add_trace(go.Scatter(x=x, y=res.xb_total, mode="lines",
                             line=dict(color=C_SIM, width=2.4), name="x_b total"))
    fig.update_layout(
        template=TEMPLATE,
        xaxis_title=X_LABEL["deg" if ang_deg else "rad"],
        yaxis_title="Fração queimada [-]",
        yaxis_range=[0, 1.02],
        title="Fração queimada por fase e total",
    )
    return fig


def fig_heat_release(res, ang_deg: bool = False):
    """Taxas de liberação de calor das duas fases e total [kJ/rad]."""
    import plotly.graph_objects as go

    x = _eixo(res.theta, ang_deg)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=res.dQ1, mode="lines",
                             line=dict(color=C_F1, width=1.8), name="dQ₁/dθ"))
    fig.add_trace(go.Scatter(x=x, y=res.dQ2, mode="lines",
                             line=dict(color=C_F2, width=1.8), name="dQ₂/dθ"))
    fig.add_trace(go.Scatter(x=x, y=res.dQ_total, mode="lines",
                             line=dict(color=C_SIM, width=2.4), name="dQ/dθ total"))
    fig.update_layout(
        template=TEMPLATE,
        xaxis_title=X_LABEL["deg" if ang_deg else "rad"],
        yaxis_title="Taxa de liberação de calor [kJ/rad]",
        title="Taxa de liberação de calor por fase e total",
    )
    return fig


def fig_temperature(res, ang_deg: bool = False):
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(
        x=_eixo(res.theta, ang_deg), y=res.Tg, mode="lines",
        line=dict(color="#9467bd", width=1.8), name="T_gas"))
    fig.update_layout(
        template=TEMPLATE,
        xaxis_title=X_LABEL["deg" if ang_deg else "rad"],
        yaxis_title="Temperatura do gás [K]",
        title="Temperatura do gás",
    )
    return fig


def fig_heat_loss(res, ang_deg: bool = False):
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(
        x=_eixo(res.theta, ang_deg), y=res.Q_wall, mode="lines",
        line=dict(color="#17becf", width=1.8), name="Q_parede"))
    fig.update_layout(
        template=TEMPLATE,
        xaxis_title=X_LABEL["deg" if ang_deg else "rad"],
        yaxis_title="Calor perdido acumulado [J]",
        title="Calor perdido para as paredes (Hohenberg)",
    )
    return fig


def fig_volume(res, ang_deg: bool = False):
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(
        x=_eixo(res.theta, ang_deg), y=res.volume, mode="lines",
        line=dict(color="#8c564b", width=1.8), name="V(θ)"))
    fig.update_layout(
        template=TEMPLATE,
        xaxis_title=X_LABEL["deg" if ang_deg else "rad"],
        yaxis_title="Volume do cilindro [m³]",
        title="Volume do cilindro",
    )
    return fig


def fig_pv(res, p_unit: str = "kPa"):
    """Diagrama P–V: modelo (linha) + pontos experimentais ('+' vermelho)."""
    import plotly.graph_objects as go

    fator = 1.0 if p_unit == "kPa" else 0.01
    fig = go.Figure(go.Scattergl(
        x=res.volume, y=res.P_sim * fator, mode="lines",
        line=dict(color=C_SIM, width=1.6), name="Modelo (P–V)"))
    # experimental: mesmos ângulos de t_eval → mesmo vetor de volume
    fig.add_trace(go.Scattergl(
        x=res.volume, y=res.P_exp * fator, mode="markers",
        marker=dict(symbol="cross-thin", size=6, color=C_EXP,
                    line=dict(width=1.4, color=C_EXP)),
        name="Experimental"))
    fig.update_layout(
        template=TEMPLATE,
        xaxis_title="Volume [m³]",
        yaxis_title=P_LABEL[p_unit],
        title="Diagrama P–V",
    )
    return fig


def fig_convergence(history, title="Convergência da calibração"):
    """Histórico da função objetivo (RMSE, kPa) por iteração."""
    import plotly.graph_objects as go

    fig = go.Figure()
    if history:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(history) + 1)), y=list(history),
            mode="lines+markers", line=dict(color="#1f77b4", width=1.6),
            marker=dict(size=4), name="RMSE"))
    fig.update_layout(
        template=TEMPLATE, xaxis_title="Iteração",
        yaxis_title="RMSE [kPa]", yaxis_type="log", title=title,
    )
    return fig


# =============================================================================
# Exportações estáticas (Matplotlib, Agg)
# =============================================================================
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def static_pressure_comparison(res, path: str) -> str:
    """pressure_comparison.png/pdf — experimental '+' vermelho + modelo verde."""
    plt = _plt()
    th = res.theta
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(th, res.P_exp, "r+", ms=4, label="Experimental")
    ax.plot(th, res.P_sim, "g-", lw=1.5, label="Modelo (Double Wiebe)")
    for th0, d, cor, rot in (
        (res.wiebe["theta01"], res.wiebe["delta1"], C_F1, "fase 1"),
        (res.wiebe["theta02"], res.wiebe["delta2"], C_F2, "fase 2"),
    ):
        ax.axvline(th0, color=cor, ls="--", lw=0.9)
        ax.axvspan(th0, th0 + d, color=cor, alpha=0.08)
    ax.set_xlabel("Ângulo do virabrequim [rad]")
    ax.set_ylabel("Pressão no cilindro [kPa]")
    ax.set_title(f"Pressão simulada × experimental  |  RMSE = "
                 f"{res.metrics['RMSE_kPa']:.3f} kPa")
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def static_heat_release(res, path: str) -> str:
    """heat_release.png — taxas das duas fases e total."""
    plt = _plt()
    th = res.theta
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(th, res.dQ1, color=C_F1, lw=1.4, label="dQ₁/dθ (pré-misturada)")
    ax.plot(th, res.dQ2, color=C_F2, lw=1.4, label="dQ₂/dθ (difusão)")
    ax.plot(th, res.dQ_total, color=C_SIM, lw=2.0, label="total")
    ax.set_xlabel("Ângulo do virabrequim [rad]")
    ax.set_ylabel("Taxa de liberação de calor [kJ/rad]")
    ax.set_title("Taxa de liberação de calor (Double Wiebe)")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def static_burned_fraction(res, path: str) -> str:
    """burned_fraction.png — frações das fases e total."""
    plt = _plt()
    th = res.theta
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(th, res.xb1, color=C_F1, lw=1.4, label="x₁ (pré-misturada)")
    ax.plot(th, res.xb2, color=C_F2, lw=1.4, label="x₂ (difusão)")
    ax.plot(th, res.xb_total, color=C_SIM, lw=2.0, label="x_b total")
    ax.set_xlabel("Ângulo do virabrequim [rad]")
    ax.set_ylabel("Fração queimada [-]")
    ax.set_title("Fração queimada (Double Wiebe)")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path