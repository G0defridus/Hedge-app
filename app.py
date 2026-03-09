"""
Censo Energy Hedge Optimizer — Streamlit Orchestrator

3-tab layout:
  Tab 1: Data & Categorisatie — upload, configuratie, EPEX
  Tab 2: Strategieën — vergelijkingstabel (3 producten × 4 optimalisaties)
  Tab 3: Resultaten — finetuning + visualisaties

Dunne coördinatielaag: alle businesslogica zit in core/,
alle UI-componenten in ui/.
"""

import pandas as pd
import streamlit as st

from core.data_processor import (
    load_aggregated_csv,
    prepare_dataframe,
    process_raw_connections,
)
from core.epex_client import fetch_epex_prices
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
# Sidebar — alleen upload
# ─────────────────────────────────────────────────────────────────────────

uploaded_file, input_mode = ui_sidebar.render_upload_section()

# ─────────────────────────────────────────────────────────────────────────
# 1. Data inlezen
# ─────────────────────────────────────────────────────────────────────────

df_hedge = None
category_counts = None

if uploaded_file is not None:
    if input_mode == "Ruwe Aansluitingen (CSV)":
        try:
            df_agg, mapping = _cached_process_raw(uploaded_file)
            category_counts = pd.Series(mapping.values()).value_counts()
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

    # ─── 3 Hoofdtabs ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    tab_data, tab_strat, tab_results = st.tabs(
        ["📊 Data & Configuratie", "⚡ Strategieën", "💰 Resultaten"]
    )

    # ─── Tab 1: Data & Configuratie ──────────────────────────────────
    with tab_data:
        # Categorisatie-overzicht
        if category_counts is not None:
            st.markdown("### Data-analyse _")
            with st.expander("Categorisatie afgerond", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Consumers", category_counts.get("Consumer", 0))
                c2.metric("Prosumers", category_counts.get("Prosumer", 0))
                c3.metric("Producers", category_counts.get("Producer", 0))
            st.markdown("---")

        tab_config = ui_tabs.render_tab_data(df)
        profile_choice = tab_config["profile_choice"]
        strategy_period = tab_config["strategy_period"]
        vol_multiplier = tab_config["vol_multiplier"]
        epex_multiplier = tab_config["epex_multiplier"]
        df = tab_config["df"]

    # ─── Verrijking (tussen tabs, op basis van Tab 1 config) ─────────

    # Actief profiel instellen
    p_mw_col = "Active_Profile_MW"
    df[p_mw_col] = df[f"{profile_choice}_MW"] * (1 + vol_multiplier)

    # EPEX ophalen (automatisch)
    epex_loaded = False
    if "ENTSOE_API_KEY" not in st.secrets:
        st.sidebar.warning("⚠️ ENTSO-E API Key ontbreekt")
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
            st.sidebar.warning(f"EPEX niet geladen: {e}")

    if not epex_loaded:
        df["EPEX_EUR_MWh"] = 0.0

    # EPEX status in sidebar
    if epex_loaded:
        st.sidebar.success("✓ EPEX spotprijzen geladen")
    else:
        st.sidebar.info("EPEX niet beschikbaar")

    # ─── Tab 2: Strategieën ──────────────────────────────────────────
    with tab_strat:
        # Bepaal beschikbare categorieën
        available_cats = [
            c for c in ["Consumer", "Prosumer", "Producer"]
            if f"{c}_MW" in df.columns
        ]
        ui_tabs.render_tab_strategies(df, available_cats, epex_loaded)

    # ─── Tab 3: Resultaten ───────────────────────────────────────────
    with tab_results:
        ui_tabs.render_tab_results(df, p_mw_col, strategy_period, epex_loaded)

else:
    st.info("Upload hiernaast een bestand om de magie te starten.")
