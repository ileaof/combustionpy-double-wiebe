# -*- coding: utf-8 -*-
"""
gui.py
======
Interface gráfica (Streamlit) do Double Wiebe.

Executar com::

    double-wiebe gui          # ou: streamlit run src/double_wiebe/gui.py

A GUI não contém equações: toda a física/válidação vive no núcleo
(`models`, `wiebe`, `geometry`, `thermodynamics`, `simulation`,
`calibration`, `metrics`, `reporting`) — exatamente o mesmo núcleo chamado
pela CLI.
"""
from __future__ import annotations

import io
import math
import threading
import time
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

from double_wiebe.calibration import PARAM_SPECS, apply_calibrated
from double_wiebe.data_processing import (ANGLE_FACTORS_RAD, PRESSURE_FACTORS_KPA,
                              DataError, read_table)
from double_wiebe import DESCRIPTION_EN
from double_wiebe.models import (PARAM_ORDER, CalibrationConfig, EngineConfig,
                     SimulationConfig, WiebeParameters)
from double_wiebe.plotting import (fig_burned, fig_convergence, fig_heat_loss,
                       fig_heat_release, fig_pressure, fig_pv, fig_residual,
                       fig_temperature, fig_volume)
from double_wiebe.reporting import (convergence_csv_bytes, metrics_json_bytes,
                        parameters_yaml_bytes, report_html_bytes,
                        results_csv_bytes, results_dataframe)
from double_wiebe.simulation import run_simulation
from double_wiebe.wiebe import double_burned_fraction

st.set_page_config(
    page_title="Double Wiebe Combustion Analysis",
    page_icon="🔥",
    layout="wide",
)


# =============================================================================
# Estado
# =============================================================================
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("data_theta", None)
    ss.setdefault("data_press", None)
    ss.setdefault("data_name", None)
    ss.setdefault("data_summary", None)
    ss.setdefault("data_preview", None)
    ss.setdefault("engine_cfg", EngineConfig())
    ss.setdefault("wiebe_cfg", WiebeParameters())
    ss.setdefault("sim_cfg", SimulationConfig())
    ss.setdefault("result", None)          # último resultado VÁLIDO
    ss.setdefault("calib_result", None)
    ss.setdefault("calib_running", False)
    ss.setdefault("calib_progress", None)  # dict compartilhado com a thread
    ss.setdefault("calib_thread", None)
    ss.setdefault("ang_deg", False)
    ss.setdefault("p_unit", "kPa")
    ss.setdefault("results_rerun_after_calib", False)


def _tem_dados() -> bool:
    return st.session_state.data_theta is not None


def _config_snapshot():
    """Cópia atual dos parâmetros das abas (fonte única: session_state)."""
    ss = st.session_state
    return ss.engine_cfg, ss.wiebe_cfg, ss.sim_cfg


@st.cache_data(show_spinner=False)
def _ler_upload(nome: str, bytes_arquivo: bytes, sep: str,
                header: str) -> pd.DataFrame:
    """Leitura em cache (determinística: mesma entrada → mesmo DataFrame)."""
    if header == "none":
        return read_table(io.BytesIO(bytes_arquivo), sep=sep, has_header=False)
    if header == "header":
        return read_table(io.BytesIO(bytes_arquivo), sep=sep, has_header=True)
    return read_table(io.BytesIO(bytes_arquivo), sep=sep, has_header=None)


# =============================================================================
# Worker da calibração (thread separada — a GUI permanece responsiva)
# =============================================================================
def _calib_worker(theta, P, engine, wiebe, sim, calib, bounds,
                  progress: dict) -> None:
    def cb(iteracao, melhor, params):
        progress["iter"] = iteracao
        progress["best"] = melhor
        progress["params"] = params
        progress["n"] = theta.size

    def cancel():
        return bool(progress.get("cancel"))

    try:
        from double_wiebe.calibration import run_calibration
        resultado = run_calibration(
            theta, P, engine, wiebe, sim, calib,
            bounds=bounds or None, progress_callback=cb, cancel_check=cancel)
        progress["result"] = resultado
        progress["done"] = True
    except Exception as e:                       # noqa: BLE001
        progress["error"] = str(e)
        progress["done"] = True


# =============================================================================
# Cabeçalho
# =============================================================================
st.title("🔥 Double Wiebe Combustion Analysis")
st.caption(DESCRIPTION_EN)
st.divider()

_init_state()

TAB_DADOS, TAB_MOTOR, TAB_WIEBE, TAB_SIM, TAB_CALIB, TAB_RES, TAB_EXPORT = \
    st.tabs(["Experimental Data", "Engine Setup", "Double Wiebe",
             "Simulation", "Calibration", "Results", "Export"])

# =============================================================================
# 1. Experimental Data
# =============================================================================
with TAB_DADOS:
    st.subheader("Dados experimentais")
    st.caption("Formatos .txt/.csv/.tsv — o núcleo converte para "
               "radianos e kPa internamente.")

    c1, c2, c3 = st.columns(3)
    with c1:
        upload = st.file_uploader("Arquivo de pressão", type=[
            "txt", "csv", "tsv"], key="upload_dados")
    with c2:
        sep = st.selectbox("Separador", ["auto", "whitespace", ",", ";", "\\t"],
                           key="sep_dados")
        sep_real = "\\t" if sep == "\\t" else sep
    with c3:
        header_opt = st.selectbox("Cabeçalho", ["auto", "none", "header"],
                                  key="header_dados")

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        ang_unit = st.selectbox("Unidade do ângulo",
                                list(ANGLE_FACTORS_RAD), key="ang_unit_dados")
    with c5:
        # "bar" primeiro: é a unidade do dado de exemplo e dos dados de
        # referência do projeto (PRESSURE_FACTORS_KPA define os fatores).
        press_unit = st.selectbox("Unidade da pressão",
                                  ["bar", "kPa", "Pa"], key="press_unit_dados")
    with c6:
        theta_min = st.number_input("θ mín [rad]", value=-2.0,
                                    key="thmin_dados")
    with c7:
        theta_max = st.number_input("θ máx [rad]", value=2.0,
                                    key="thmax_dados")

    ex1, ex2 = st.columns([1, 2])
    with ex1:
        carregar_exemplo = st.button("Carregar dados de exemplo",
                                     key="btn_exemplo",
                                     help="Carrega data/example_pressure.txt "
                                     "do projeto (θ em rad, P em bar).")
    with ex2:
        st.caption("Sem arquivo? Use o botão ao lado para ver a prévia "
                   "funcionando.")

    if carregar_exemplo:
        from pathlib import Path
        exemplo = Path(__file__).resolve().parents[2] / "data" / \
            "example_pressure.txt"
        if exemplo.exists():
            bytes_exemplo = exemplo.read_bytes()
            df_ex = _ler_upload(exemplo.name, bytes_exemplo, "auto", "auto")
            try:
                from double_wiebe.data_processing import prepare_series
                theta_e, P_e, res_e = prepare_series(
                    df_ex, angle_unit="radianos", pressure_unit="bar",
                    theta_min=-2.0, theta_max=2.0)
                st.session_state.data_theta = theta_e
                st.session_state.data_press = P_e
                st.session_state.data_name = exemplo.name
                st.session_state.data_summary = res_e
                st.session_state.data_preview = df_ex.head(10)
                st.success(f"Dados de exemplo carregados: {exemplo.name} "
                           f"({res_e['n_obs']} observações).")
            except DataError as e:
                st.error(f"Erro nos dados de exemplo: {e}")
        else:
            st.error("Arquivo de exemplo não encontrado "
                     "(data/example_pressure.txt).")

    # --- Fontes de dados: apenas LEM e armazenam em session_state -------
    if upload is not None:
        try:
            df = _ler_upload(upload.name, upload.getvalue(), sep_real,
                             header_opt)
            n_cols = df.shape[1]
            ca, cp = st.columns(2)
            with ca:
                angle_col = st.number_input(
                    "Coluna do ângulo (0-based)", 0, n_cols - 1, 0,
                    key="angle_col_dados")
            with cp:
                pressure_col = st.number_input(
                    "Coluna da pressão (0-based)", 0, n_cols - 1, 1,
                    key="press_col_dados")

            from double_wiebe.data_processing import prepare_series
            theta, P, resumo = prepare_series(
                df, angle_col=int(angle_col), pressure_col=int(pressure_col),
                angle_unit=ang_unit, pressure_unit=press_unit,
                theta_min=theta_min, theta_max=theta_max)
            st.session_state.data_theta = theta
            st.session_state.data_press = P
            st.session_state.data_name = upload.name
            st.session_state.data_summary = resumo
            st.session_state.data_preview = df.head(15)

            # plausibilidade: P no IVC fora da faixa típica sugere unidade errada
            if resumo["P1"] < 5.0:
                st.warning(f"P no IVC = {resumo['P1']:.3g} kPa — muito baixa. "
                           "Confira a **Unidade da pressão** (dados em bar "
                           "ou kPa interpretados como Pa ficam ~1000× "
                           "menores).")
            elif resumo["P1"] > 20000.0:
                st.warning(f"P no IVC = {resumo['P1']:.3g} kPa — muito alta. "
                           "Confira a **Unidade da pressão** (talvez Pa "
                           "esteja selecionado como kPa, ou vice-versa).")
        except DataError as e:
            st.error(f"Erro nos dados: {e}")
            st.session_state.data_theta = None
        except Exception as e:                   # noqa: BLE001
            st.error(f"Falha ao ler o arquivo: {e}")
            st.session_state.data_theta = None

    # --- Renderização comum: roda TODA vez que há dados ativos ----------
    # (antes o gráfico existia só no instante do upload; ao trocar de aba
    #  ou re-renderizar, ele desaparecia)
    if st.session_state.data_theta is not None:
        resumo = st.session_state.data_summary
        st.info(f"Dados ativos: **{st.session_state.data_name}** — "
                f"{resumo['n_obs']} observações.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Observações", resumo["n_obs"])
        m2.metric("Descartadas (inválidas)", resumo["n_descartadas"])
        m3.metric("Fora do intervalo", resumo["n_fora_intervalo"])
        m4.metric("P no IVC", f"{resumo['P1']:.1f} kPa")

        import plotly.graph_objects as go
        # marcador PREENCHIDO: símbolos abertos ("cross-thin") não são
        # desenhados pelo plotly.js 3.x sem marker.line explícito
        fig_pre = go.Figure(go.Scatter(
            x=st.session_state.data_theta, y=st.session_state.data_press,
            mode="lines+markers",
            line=dict(color="#d62728", width=1.2),
            marker=dict(size=4, color="#d62728"),
            name="experimental"))
        fig_pre.update_layout(
            template="plotly_white", height=360,
            xaxis_title="Ângulo [rad]", yaxis_title="Pressão [kPa]",
            title="Prévia: pressão × ângulo (após conversão e filtro)")
        st.plotly_chart(fig_pre, use_container_width=True,
                        key="chart_previa_dados")
        if st.session_state.data_preview is not None:
            with st.expander("Prévia da tabela (primeiras 15 linhas)"):
                st.dataframe(st.session_state.data_preview, width="stretch")
    else:
        st.info("Carregue um arquivo para começar — ou clique em "
                "**Carregar dados de exemplo**.")

# =============================================================================
# 2. Engine Setup
# =============================================================================
with TAB_MOTOR:
    st.subheader("Configuração do motor")
    g1, g2, g3 = st.columns(3)
    with st.form("form_motor"):
        st.markdown("**Geometria**")
        gc1, gc2, gc3, gc4 = st.columns(4)
        with gc1:
            bore_mm = st.number_input("Diâmetro do cilindro [mm]",
                                      value=st.session_state.engine_cfg.bore * 1000,
                                      min_value=1.0, key="bore")
        with gc2:
            stroke_mm = st.number_input("Curso [mm]",
                                        value=st.session_state.engine_cfg.stroke * 1000,
                                        min_value=1.0, key="stroke")
        with gc3:
            rod_mm = st.number_input("Biela [mm]",
                                     value=st.session_state.engine_cfg.rod_length * 1000,
                                     min_value=1.0, key="rod")
        with gc4:
            n_cyl = st.number_input("Cilindros",
                                    value=st.session_state.engine_cfg.n_cylinders,
                                    min_value=1, key="ncyl")
        st.markdown("**Operação**")
        oc1, oc2 = st.columns(2)
        with oc1:
            rpm = st.number_input("Rotação [rpm]",
                                  value=st.session_state.engine_cfg.rpm,
                                  min_value=1.0, key="rpm")
        with oc2:
            Rc = st.number_input("Razão de compressão [-]",
                                 value=st.session_state.engine_cfg.Rc,
                                 min_value=1.1, key="Rc")
        st.markdown("**Combustível**")
        fc1, fc2 = st.columns(2)
        with fc1:
            m_fuel = st.number_input("Massa de combustível [kg/ciclo]",
                                     value=st.session_state.engine_cfg.m_fuel,
                                     min_value=0.0,
                                     format="%.12e", key="mfuel")
        with fc2:
            LHV = st.number_input("PCI/LHV [kJ/kg]",
                                  value=st.session_state.engine_cfg.LHV,
                                  min_value=1.0, key="lhv")
        st.markdown("**Termodinâmica e transferência de calor**")
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            kappa = st.number_input("κ [-]",
                                    value=st.session_state.engine_cfg.kappa,
                                    min_value=1.01, max_value=1.99, key="kappa")
        with tc2:
            T1 = st.number_input("T inicial [K]",
                                 value=st.session_state.engine_cfg.T1,
                                 min_value=1.0, key="T1")
        with tc3:
            Tw = st.number_input("T parede [K]",
                                 value=st.session_state.engine_cfg.Tw,
                                 min_value=1.0, key="Tw")
        heat_transfer = st.checkbox("Transferência de calor (Hohenberg)",
                                    value=st.session_state.engine_cfg.heat_transfer,
                                    key="heat_transfer")
        if st.form_submit_button("Aplicar configuração"):
            cfg = EngineConfig(
                bore=bore_mm / 1000.0, stroke=stroke_mm / 1000.0,
                rod_length=rod_mm / 1000.0, n_cylinders=int(n_cyl),
                rpm=rpm, Rc=Rc, m_fuel=m_fuel, LHV=LHV, kappa=kappa,
                T1=T1, Tw=Tw, heat_transfer=heat_transfer)
            erros = cfg.validate()
            if erros:
                for e in erros:
                    st.error(e)
            else:
                st.session_state.engine_cfg = cfg
                st.success("Configuração aplicada.")

    cfg_now: EngineConfig = st.session_state.engine_cfg
    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("A_p", f"{cfg_now.A_p:.4e} m²")
    d2.metric("Vd", f"{cfg_now.Vd:.4e} m³")
    d3.metric("Vc", f"{cfg_now.Vc:.4e} m³")
    d4.metric("R = l/r", f"{cfg_now.R:.2f}")
    d5.metric("Vp", f"{cfg_now.Vp:.2f} m/s")
    d6.metric("Q_total", f"{cfg_now.Q_total:.4f} kJ")

# =============================================================================
# 3. Double Wiebe (prévia instantânea)
# =============================================================================
with TAB_WIEBE:
    st.subheader("Parâmetros das duas fases")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Premixed phase**")
        theta01_deg = st.number_input(
            "theta01 [°]", value=math.degrees(st.session_state.wiebe_cfg.theta01),
            key="theta01", help="Início da fase 1 (pré-misturada).")
        delta1_deg = st.number_input(
            "delta1 [°]", value=math.degrees(st.session_state.wiebe_cfg.delta1),
            min_value=0.1, key="delta1", help="Duração da fase 1.")
        m1 = st.number_input("m1 [-]", value=st.session_state.wiebe_cfg.m1,
                             min_value=0.0, key="m1",
                             help="Fator de forma da fase 1.")
        a1 = st.number_input("a1 [-]", value=st.session_state.wiebe_cfg.a1,
                             min_value=0.1, key="a1",
                             help="Constante de eficiência da fase 1.")
        alpha = st.slider("alpha (fração de energia da fase 1) [-]", 0.0, 1.0,
                          st.session_state.wiebe_cfg.alpha, 0.01, key="alpha")
        st.caption(f"1 − alpha = {1.0 - alpha:.2f} (energia da fase 2)")
    with colB:
        st.markdown("**Diffusion-controlled phase**")
        theta02_deg = st.number_input(
            "theta02 [°]", value=math.degrees(st.session_state.wiebe_cfg.theta02),
            key="theta02", help="Início da fase 2 (difusão).")
        delta2_deg = st.number_input(
            "delta2 [°]", value=math.degrees(st.session_state.wiebe_cfg.delta2),
            min_value=0.1, key="delta2", help="Duração da fase 2.")
        m2 = st.number_input("m2 [-]", value=st.session_state.wiebe_cfg.m2,
                             min_value=0.0, key="m2")
        a2 = st.number_input("a2 [-]", value=st.session_state.wiebe_cfg.a2,
                             min_value=0.1, key="a2")
        mode = st.selectbox("Modo", ["continuous", "bounded"], key="mode",
                            help="continuous: cauda assintótica; bounded: "
                            "fase constante após theta0 + delta.")

    # Aplica imediatamente (prévia instantânea) e valida
    w_new = WiebeParameters(
        theta01=math.radians(theta01_deg), delta1=math.radians(delta1_deg),
        m1=m1, a1=a1, theta02=math.radians(theta02_deg),
        delta2=math.radians(delta2_deg), m2=m2, a2=a2, alpha=alpha,
        mode=mode)
    erros_w = w_new.validate()
    if erros_w:
        for e in erros_w:
            st.warning(e)
    else:
        st.session_state.wiebe_cfg = w_new

    # Prévia instantânea das funções (independente de dados carregados)
    theta_prev = np.linspace(-0.6, 2.2, 400)
    frac = double_burned_fraction(theta_prev, w_new)
    import plotly.graph_objects as go
    fig_prev = go.Figure()
    fig_prev.add_trace(go.Scatter(
        x=theta_prev, y=frac["x1"], mode="lines", name="x₁ (pré-misturada)",
        line=dict(color="#1f77b4", width=2)))
    fig_prev.add_trace(go.Scatter(
        x=theta_prev, y=frac["x2"], mode="lines", name="x₂ (difusão)",
        line=dict(color="#ff7f0e", width=2)))
    fig_prev.add_trace(go.Scatter(
        x=theta_prev, y=frac["xb"], mode="lines", name="x_b total",
        line=dict(color="#2ca02c", width=3)))
    fig_prev.update_layout(
        template="plotly_white", height=380,
        xaxis_title="Ângulo [rad]", yaxis_title="Fração queimada [-]",
        title="Prévia instantânea das funções de Wiebe (soma ponderada)")
    st.plotly_chart(fig_prev, use_container_width=True)

# =============================================================================
# 4. Simulation
# =============================================================================
with TAB_SIM:
    st.subheader("Simulação")
    with st.form("form_simulacao"):
        s1, s2, s3 = st.columns(3)
        with s1:
            method = st.selectbox("Integrador",
                                  SimulationConfig.ALLOWED_METHODS,
                                  index=SimulationConfig.ALLOWED_METHODS.index(
                                      st.session_state.sim_cfg.method),
                                  key="metodo")
        with s2:
            rtol = st.number_input("rtol", value=st.session_state.sim_cfg.rtol,
                                   format="%.1e", key="rtol")
        with s3:
            atol = st.number_input("atol", value=st.session_state.sim_cfg.atol,
                                   format="%.1e", key="atol")
        b1, b2, b3 = st.columns([1, 1, 1])
        rodar = st.form_submit_button("Run simulation", type="primary")
        resetar = st.form_submit_button("Reset parameters")
        exemplo = st.form_submit_button("Load example")

    if resetar:
        st.session_state.sim_cfg = SimulationConfig()
        st.session_state.engine_cfg = EngineConfig()
        st.session_state.wiebe_cfg = WiebeParameters()
        st.rerun()
    if exemplo:
        st.session_state.engine_cfg = EngineConfig()   # defaults do enunciado
        st.session_state.wiebe_cfg = WiebeParameters()
        st.session_state.sim_cfg = SimulationConfig()
        st.rerun()

    if rodar:
        st.session_state.sim_cfg = SimulationConfig(
            method=method, rtol=rtol, atol=atol)
        if not _tem_dados():
            st.error("Carregue os dados experimentais na aba "
                     "**Experimental Data** antes de simular.")
        else:
            erros = (st.session_state.engine_cfg.validate()
                     + st.session_state.wiebe_cfg.validate()
                     + st.session_state.sim_cfg.validate())
            if erros:
                for e in erros:
                    st.error(e)
            else:
                try:
                    res = run_simulation(
                        st.session_state.data_theta,
                        st.session_state.data_press,
                        st.session_state.engine_cfg,
                        st.session_state.wiebe_cfg,
                        st.session_state.sim_cfg)
                    st.session_state.result = res      # preserva o último válido
                    st.session_state.calib_result = None
                    st.success(
                        f"Simulação concluída — RMSE = "
                        f"{res.metrics['RMSE_kPa']:.4g} kPa")
                except Exception as e:               # noqa: BLE001
                    st.error(f"A simulação falhou — resultado anterior "
                             f"preservado. Detalhe: {e}")

    if (r := st.session_state.result) is not None:
        st.markdown("**Último resultado válido:**")
        m = r.metrics
        i = r.indicators
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("RMSE [kPa]", f"{m['RMSE_kPa']:.4g}")
        k2.metric("R²", f"{m['R2']:.4f}")
        k3.metric("P máx [kPa]", f"{i['P_max_kPa']:.0f}")
        k4.metric("T máx [K]", f"{i['T_max_K']:.0f}")

# =============================================================================
# 5. Calibration
# =============================================================================
with TAB_CALIB:
    st.subheader("Calibração")
    st.caption("Selecione os parâmetros livres. A função objetivo é o RMSE "
               "(com regularização opcional). Ao final, uma análise de "
               "sensibilidade alerta para problemas de identificabilidade.")

    with st.form("form_calib"):
        selecionados = st.multiselect(
            "Parâmetros livres (os demais ficam fixos)",
            PARAM_ORDER, default=st.session_state.calib_result[
                "selected"] if st.session_state.calib_result
            else ["Rc", "theta01", "delta1", "m1", "theta02", "delta2", "m2",
                  "alpha"],
            key="calib_sel")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metodo_cal = st.selectbox("Método", ["differential-evolution",
                                                 "pso", "least-squares"],
                                      key="calib_metodo")
        with c2:
            seed_cal = st.number_input("Semente (0 = aleatória)", 0, 2**31 - 1,
                                       42, key="calib_seed")
        with c3:
            popsize_cal = st.number_input("População (DE)", 4, 200, 20,
                                          key="calib_popsize")
        with c4:
            maxiter_cal = st.number_input("Iterações máx.", 5, 5000, 200,
                                          key="calib_maxiter")
        c5, c6 = st.columns(2)
        with c5:
            tol_cal = st.number_input("Tolerância", 1e-12, 1e-1, 1e-10,
                                      format="%.1e", key="calib_tol")
        with c6:
            polish_cal = st.checkbox("Refinamento least-squares final",
                                     value=True, key="calib_polish")
        with st.expander("Limites por parâmetro (selecionados)"):
            limites_ui: Dict[str, tuple] = {}
            for nome in selecionados:
                spec = PARAM_SPECS[nome]
                lc1, lc2 = st.columns(2)
                with lc1:
                    lo = st.number_input(f"{nome} mín", value=float(spec["lower"]),
                                         key=f"lim_lo_{nome}")
                with lc2:
                    hi = st.number_input(f"{nome} máx", value=float(spec["upper"]),
                                         key=f"lim_hi_{nome}")
                limites_ui[nome] = (lo, hi)
        iniciar = st.form_submit_button("Start calibration", type="primary")
        cancelar = st.form_submit_button("Cancel calibration",
                                         disabled=not st.session_state.calib_running)

    if cancelar and st.session_state.calib_progress is not None:
        st.session_state.calib_progress["cancel"] = True
        st.warning("Cancelamento solicitado — aguardando a iteração atual...")

    if iniciar:
        if not _tem_dados():
            st.error("Carregue os dados experimentais primeiro.")
        elif st.session_state.calib_running:
            st.error("Já existe uma calibração em andamento nesta sessão.")
        elif not selecionados:
            st.error("Selecione pelo menos um parâmetro para calibrar.")
        else:
            limites = {}
            for nome, (lo, hi) in limites_ui.items():
                if lo >= hi:
                    st.error(f"Limites inválidos para {nome}: ({lo}, {hi}).")
                    break
                limites[nome] = (lo, hi)
            else:
                calib = CalibrationConfig(
                    method=metodo_cal, selected=selecionados,
                    seed=(int(seed_cal) if seed_cal else None),
                    maxiter=int(maxiter_cal), popsize=int(popsize_cal),
                    tol=tol_cal, polish=polish_cal)
                progress = {"cancel": False}
                st.session_state.calib_progress = progress
                st.session_state.calib_running = True
                st.session_state.calib_thread = threading.Thread(
                    target=_calib_worker,
                    args=(st.session_state.data_theta,
                          st.session_state.data_press,
                          st.session_state.engine_cfg,
                          st.session_state.wiebe_cfg,
                          st.session_state.sim_cfg, calib,
                          limites or None, progress),
                    daemon=True)
                st.session_state.calib_thread.start()
                st.rerun()

    # ------------------------------------------------------------------
    # Progresso/resultado da calibração (thread separada)
    # ------------------------------------------------------------------
    progress = st.session_state.calib_progress
    if st.session_state.calib_running and progress is not None:
        if progress.get("done"):
            st.session_state.calib_running = False
            if "error" in progress:
                st.error(f"A calibração falhou: {progress['error']}")
            else:
                st.session_state.calib_result = progress["result"]
                if progress["result"].get("cancelado"):
                    st.warning("Calibração cancelada — resultados "
                               "anteriores preservados.")
                else:
                    st.success("Calibração concluída.")
        else:
            k1, k2 = st.columns([1, 2])
            k1.metric("Iteração", progress.get("iter", 0))
            k2.metric("Melhor RMSE [kPa]",
                      f"{progress.get('best', float('nan')):.6g}")
            maxit = int(maxiter_cal) if maxiter_cal else 200
            frac = min(0.99, float(progress.get("iter", 0)) / max(1, maxit))
            st.progress(frac, text=f"Calibração em andamento — iteração "
                        f"{progress.get('iter', 0)}/{maxit} "
                        "(thread separada).")
            time.sleep(1.0)     # POLLING: evita re-render em loop fechado
            st.rerun()          # (sem essa pausa a interface congela)

    if (cal := st.session_state.calib_result) is not None:
        st.markdown("**Resultado da calibração** "
                    f"(método **{cal['method']}**, semente **{cal['seed']}**, "
                    f"{cal['iteracoes']} iterações, RMSE = "
                    f"**{cal['rmse']:.6g} kPa**):")
        tabela = pd.DataFrame({
            "Parâmetro": list(cal["params"]),
            "Inicial": [cal["params_initial"][n] for n in cal["params"]],
            "Calibrado": [cal["params"][n] for n in cal["params"]],
            "Livre": ["✓" if n in cal["selected"] else "fixo"
                      for n in cal["params"]],
        })
        st.dataframe(tabela, width="stretch")
        for alerta in cal.get("alerts", []):
            st.warning(alerta)
        sens = cal.get("sensitivity") or []
        if sens:
            with st.expander("Análise de sensibilidade (±1% por parâmetro)"):
                st.dataframe(pd.DataFrame(sens), width="stretch")
        if st.button("Aplicar parâmetros calibrados e rodar a simulação",
                     key="aplicar_calib"):
            eng, wieb = apply_calibrated(cal, st.session_state.engine_cfg,
                                         st.session_state.wiebe_cfg)
            try:
                st.session_state.result = run_simulation(
                    st.session_state.data_theta, st.session_state.data_press,
                    eng, wieb, st.session_state.sim_cfg)
                st.success("Simulação executada com os parâmetros calibrados — "
                           "veja a aba Results.")
            except Exception as e:                   # noqa: BLE001
                st.error(f"Falha: {e}")

# =============================================================================
# 6. Results
# =============================================================================
with TAB_RES:
    st.subheader("Resultados")
    if st.session_state.result is None:
        st.info("Execute uma simulação (aba **Simulation**).")
    else:
        r = st.session_state.result
        rc1, rc2 = st.columns(2)
        with rc1:
            ang_deg = st.toggle("Eixo em graus",
                                key="res_ang", value=st.session_state.ang_deg)
        with rc2:
            p_unit = st.selectbox("Pressão", ["kPa", "bar"], key="res_punit")
        st.session_state.ang_deg = ang_deg

        ind = r.indicators
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("RMSE [kPa]", f"{r.metrics['RMSE_kPa']:.4g}")
        k2.metric("R²", f"{r.metrics['R2']:.4f}")
        k3.metric("P máx [kPa]", f"{ind['P_max_kPa']:.0f} @ "
                                 f"{ind['theta_at_P_max_deg']:.2f}°")
        k4.metric("x_b final", f"{ind['final_burned_fraction']:.4f}")
        k5, k6, k7, k8 = st.columns(4)
        k5.metric("T máx [K]", f"{ind['T_max_K']:.0f}")
        k6.metric("Calor liberado [kJ]",
                  f"{ind['total_heat_released_kJ']:.4f}")
        k7.metric("Calor perdido [kJ]", f"{ind['wall_heat_loss_kJ']:.4f}")
        k8.metric("Energia fase1/fase2 [%]",
                  f"{ind['phase1_energy_share_pct']:.1f} / "
                  f"{ind['phase2_energy_share_pct']:.1f}")

        st.plotly_chart(fig_pressure(r, ang_deg, p_unit),
                        use_container_width=True)
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(fig_burned(r, ang_deg), use_container_width=True)
            st.plotly_chart(fig_temperature(r, ang_deg),
                            use_container_width=True)
            st.plotly_chart(fig_volume(r, ang_deg), use_container_width=True)
        with g2:
            st.plotly_chart(fig_heat_release(r, ang_deg),
                            use_container_width=True)
            st.plotly_chart(fig_residual(r, ang_deg), use_container_width=True)
            st.plotly_chart(fig_heat_loss(r, ang_deg), use_container_width=True)
        st.plotly_chart(fig_pv(r, p_unit), use_container_width=True)
        cal_hist = (st.session_state.calib_result or {}).get(
            "objective_history")
        if cal_hist:
            st.plotly_chart(fig_convergence(cal_hist),
                            use_container_width=True)
        with st.expander("Tabela de resíduos"):
            df_res = pd.DataFrame({
                "theta_rad": r.theta, "theta_deg": np.degrees(r.theta),
                "P_exp [kPa]": r.P_exp, "P_sim [kPa]": r.P_sim,
                "residual [kPa]": r.P_exp - r.P_sim,
                "residual [%]": 100.0 * (r.P_exp - r.P_sim) / r.P_exp,
            })
            st.dataframe(df_res, width="stretch", height=320)

# =============================================================================
# 7. Export
# =============================================================================
with TAB_EXPORT:
    st.subheader("Exportação")
    if st.session_state.result is None:
        st.info("Nenhum resultado para exportar — rode uma simulação.")
    else:
        r = st.session_state.result
        cal_atual = st.session_state.calib_result
        e1, e2 = st.columns(2)
        with e1:
            st.download_button("results.csv", data=results_csv_bytes(r),
                               file_name="results.csv", mime="text/csv",
                               key="dl_csv")
            st.download_button("parameters.yaml",
                               data=parameters_yaml_bytes(r),
                               file_name="parameters.yaml",
                               mime="application/yaml", key="dl_yaml")
            st.download_button("metrics.json",
                               data=metrics_json_bytes(r, cal_atual),
                               file_name="metrics.json",
                               mime="application/json", key="dl_json")
            if cal_atual is not None:
                st.download_button("convergence.csv",
                                   data=convergence_csv_bytes(cal_atual),
                                   file_name="convergence.csv",
                                   mime="text/csv", key="dl_conv")
        with e2:
            from double_wiebe.plotting import (static_burned_fraction,
                                   static_heat_release,
                                   static_pressure_comparison)
            if st.button("Gerar gráficos estáticos (PNG)",
                         key="btn_static"):
                buf = io.BytesIO()
                static_pressure_comparison(r, buf)
                buf.seek(0)
                pngs = {"pressure_comparison.png": buf.getvalue()}
                buf = io.BytesIO()
                static_heat_release(r, buf)
                buf.seek(0)
                pngs["heat_release.png"] = buf.getvalue()
                buf = io.BytesIO()
                static_burned_fraction(r, buf)
                buf.seek(0)
                pngs["burned_fraction.png"] = buf.getvalue()
                st.session_state.static_pngs = pngs
            if st.session_state.get("static_pngs"):
                for nome, dados in st.session_state.static_pngs.items():
                    st.download_button(nome, data=dados, file_name=nome,
                                       mime="image/png",
                                       key=f"dl_{nome}")
            if st.button("PDF (3 figuras)", key="btn_pdf"):
                from matplotlib.backends.backend_pdf import PdfPages
                pdf_buf = io.BytesIO()
                with PdfPages(pdf_buf) as pdf:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(9, 5.5))
                    ax.plot(r.theta, r.P_exp, "r+", ms=4, label="Experimental")
                    ax.plot(r.theta, r.P_sim, "g-", lw=1.5,
                            label="Modelo (Double Wiebe)")
                    ax.set_xlabel("Ângulo do virabrequim [rad]")
                    ax.set_ylabel("Pressão no cilindro [kPa]")
                    ax.legend()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    fig, ax = plt.subplots(figsize=(9, 4.5))
                    ax.plot(r.theta, r.dQ1, color="#1f77b4", lw=1.4,
                            label="dQ₁/dθ")
                    ax.plot(r.theta, r.dQ2, color="#ff7f0e", lw=1.4,
                            label="dQ₂/dθ")
                    ax.plot(r.theta, r.dQ_total, color="#2ca02c", lw=2.0,
                            label="total")
                    ax.set_xlabel("Ângulo do virabrequim [rad]")
                    ax.set_ylabel("Taxa de liberação de calor [kJ/rad]")
                    ax.legend()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    fig, ax = plt.subplots(figsize=(9, 4.5))
                    ax.plot(r.theta, r.xb1, color="#1f77b4", lw=1.4, label="x₁")
                    ax.plot(r.theta, r.xb2, color="#ff7f0e", lw=1.4, label="x₂")
                    ax.plot(r.theta, r.xb_total, color="#2ca02c", lw=2.0,
                            label="x_b")
                    ax.set_xlabel("Ângulo do virabrequim [rad]")
                    ax.set_ylabel("Fração queimada [-]")
                    ax.legend()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                pdf_buf.seek(0)
                st.session_state.pdf_bytes = pdf_buf.getvalue()
            if st.session_state.get("pdf_bytes"):
                st.download_button("pressure_comparison.pdf",
                                   data=st.session_state.pdf_bytes,
                                   file_name="pressure_comparison.pdf",
                                   mime="application/pdf", key="dl_pdf")
            if st.button("Relatório HTML (autocontido)", key="btn_html"):
                pngs = st.session_state.get("static_pngs", {})
                st.session_state.html_bytes = report_html_bytes(
                    r, cal_atual, figures_png=pngs or None)
            if st.session_state.get("html_bytes"):
                st.download_button("report.html",
                                   data=st.session_state.html_bytes,
                                   file_name="report.html",
                                   mime="text/html", key="dl_html")
        with st.expander("Prévia do results.csv (primeiras linhas)"):
            st.dataframe(results_dataframe(r).head(10),
                         use_container_width=True)