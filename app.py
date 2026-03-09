"""
Censo Energy Hedge Optimizer — Streamlit Orchestrator

Dunne coördinatielaag: alle businesslogica zit in core/,
alle UI-componenten in ui/. Dit bestand plakt ze aan elkaar.
"""

import pandas as pd
import streamlit as st

from core.data_processor import (
    load_aggregated_csv,
    prepare_dataframe,
    process_raw_connections,
)
from core.epex_client import fetch_epex_prices
from core.financial import (
    apply_hedge_columns,
    compute_financial_summary,
    compute_hedge_results,
    compute_quarterly_table,
)
from core.models import DataLoadError, EPEXFetchError
from ui import sidebar as ui_sidebar
from ui import state as ui_state
from ui import tabs as ui_tabs
from ui import theme as ui_theme

# ─────────────────────────────────────────────────────────────────────────
# Caching wrappers — houden core/ vrij van Streamlit
# ─────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def _cached_process_raw(file):
    """Verwerk ruwe meetdata met Streamlit caching + progress bar."""
    progress = st.progress(0, text="Analyseren aansluitingen...")

    def _on_progress(frac: float, msg: str) -> None:
        progress.progress(frac, text=msg)

    result = process_raw_connections(file, on_progress=_on_progress)
    progress.empty()
    return result


@st.cache_data(show_spinner="Spotprijzen downloaden via ENTSO-E API...")
def _cached_fetch_epex(api_key: str, start_date, end_date):
    return fetch_epex_prices(api_key, start_date, end_date)


# ─────────────────────────────────────────────────────────────────────────
# Pagina-instellingen
# ─────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Censo Energy Optimizer", layout="wide")
ui_theme.apply_censo_style()
ui_theme.render_hero_title()
ui_theme.render_how_it_works()
ui_state.init_state()

# ─────────────────────────────────────────────────────────────────────────
# Dynamische zijbalk (upload boven als er nog geen bestand is)
# ─────────────────────────────────────────────────────────────────────────

has_file = st.session_state.get("file_uploader_key") is not None
if has_file:
    c_config = st.sidebar.container()
    st.sidebar.markdown("---")
    c_input = st.sidebar.container()
else:
    c_input = st.sidebar.container()
    c_config = st.sidebar.container()

# ─────────────────────────────────────────────────────────────────────────
# 1. Data inlezen
# ─────────────────────────────────────────────────────────────────────────

uploaded_file, input_mode = ui_sidebar.render_upload_section(c_input)
df_hedge = None

if uploaded_file is not None:
    if input_mode == "Ruwe Aansluitingen (CSV)":
        try:
            df_agg, mapping = _cached_process_raw(uploaded_file)
            with st.expander("Analyse afgerond _", expanded=True):
                c1, c2, c3 = st.columns(3)
                counts = pd.Series(mapping.values()).value_counts()
                c1.metric("Consumers", counts.get("Consumer", 0))
                c2.metric("Prosumers", counts.get("Prosumer", 0))
                c3.metric("Producers", counts.get("Producer", 0))

            df_hedge = df_agg.reset_index()
            cols = list(df_hedge.columns)
            cols[0] = "Date"
            df_hedge.columns = cols
        except (DataLoadError, Exception) as e:
            st.error(f"Er ging iets mis met het verwerken: {e}")
            st.stop()
    else:
        try:
            df_hedge = load_aggregated_csv(uploaded_file)
        except (DataLoadError, Exception) as e:
            st.error(f"Fout bij inlezen bestand: {e}")
            st.stop()

# ─────────────────────────────────────────────────────────────────────────
# 2. Hoofdpijplijn (alleen als er data is)
# ─────────────────────────────────────────────────────────────────────────

if df_hedge is not None:
    df = prepare_dataframe(df_hedge)

    # --- Basisinstellingen ---
    profile_choice, strategy_period = ui_sidebar.render_settings(c_config)

    # --- Scenario's ---
    vol_multiplier, epex_multiplier = ui_sidebar.render_scenarios(c_config)

    p_mw_col = "Active_Profile_MW"
    df[p_mw_col] = df[f"{profile_choice}_MW"] * (1 + vol_multiplier)

    # --- Contractprijzen ---
    df = ui_sidebar.render_prices(c_config, df, strategy_period)

    # --- EPEX ophalen ---
    epex_loaded = False
    if "ENTSOE_API_KEY" not in st.secrets:
        with c_config:
            st.error("⚠️ ENTSO-E API Key ontbreekt in .streamlit/secrets.toml")
    else:
        try:
            df_epex = _cached_fetch_epex(
                st.secrets["ENTSOE_API_KEY"],
                df["Date"].min(),
                df["Date"].max(),
            )
            df["Date_Hour"] = df["Date"].dt.floor("H")
            df_epex = df_epex.copy()
            df_epex["EPEX_EUR_MWh"] = df_epex["EPEX_EUR_MWh"] * (1 + epex_multiplier)
            df = pd.merge(
                df, df_epex[["Date_Hour", "EPEX_EUR_MWh"]], on="Date_Hour", how="left"
            )
            epex_loaded = True
        except EPEXFetchError as e:
            with c_config:
                st.warning(f"EPEX-prijzen konden niet worden geladen: {e}")

    if not epex_loaded:
        df["EPEX_EUR_MWh"] = 0.0

    # --- Strategie knoppen ---
    ui_sidebar.render_strategy_buttons(c_config, df, p_mw_col, strategy_period)

    # --- Finetunen sliders ---
    df = ui_sidebar.render_hedge_sliders(c_config, df, p_mw_col, strategy_period)

    # ─────────────────────────────────────────────────────────────────────
    # 3. Berekeningen (pure core functies)
    # ─────────────────────────────────────────────────────────────────────

    df = apply_hedge_columns(df, p_mw_col)
    results = compute_hedge_results(df)
    financial = compute_financial_summary(df, epex_loaded)
    q_stats = compute_quarterly_table(df, epex_loaded)

    # ─────────────────────────────────────────────────────────────────────
    # 4. Visualisaties
    # ─────────────────────────────────────────────────────────────────────

    st.markdown("<br>", unsafe_allow_html=True)
    tab_main, tab_vol, tab_eco, tab_charts = st.tabs(
        ["Samenvatting", "Jouw volume flow", "Kengetallen", "Seizoenen"]
    )

    with tab_main:
        ui_tabs.render_summary_tab(df, financial, results)

    with tab_vol:
        ui_tabs.render_volume_tab(df, results)

    with tab_eco:
        ui_tabs.render_economics_tab(financial, epex_loaded)

    with tab_charts:
        ui_tabs.render_seasonal_tab(df, p_mw_col, q_stats, epex_loaded)

else:
    st.info("Upload hiernaast een bestand om de magie te starten.")
