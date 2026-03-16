"""
Censo Energy Hedge Optimizer — Streamlit Orchestrator

Flow:
  1. Upload: sidebar toont upload-widget, main toont welkomstbericht
  2. Na upload: sidebar wordt config-paneel
  3. Main: seizoensgrafieken + kwartaaltabel (100% SPOT baseline)
  4. Daaronder: 4 optimalisatie-kaarten
  5. Na selectie: grafieken updaten met hedge-overlay, resultaten verschijnen

Dunne coordinatielaag: alle businesslogica zit in core/,
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
from core.financial import apply_hedge_columns, compute_quarterly_table
from core.models import DataLoadError, EPEXFetchError
from core.scenario_engine import compute_all_scenarios
from ui import sidebar as ui_sidebar
from ui import state as ui_state
from ui import tabs as ui_tabs
from ui import theme as ui_theme
from ui.comparison_table import render_comparison_table

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
ui_state.init_state()

# ─────────────────────────────────────────────────────────────────────────
# Fase 1: Upload (als er nog geen data is)
# ─────────────────────────────────────────────────────────────────────────

if "df_prepared" not in st.session_state:
    ui_theme.render_how_it_works()

    uploaded_file, input_mode = ui_sidebar.render_upload_sidebar()

    if uploaded_file is not None:
        # Verwerk het bestand
        if input_mode == "Ruwe Aansluitingen (CSV)":
            try:
                df_agg, mapping = _cached_process_raw(uploaded_file)
                category_counts = pd.Series(mapping.values()).value_counts()
                df_hedge = df_agg.reset_index()
                cols = list(df_hedge.columns)
                cols[0] = "Date"
                df_hedge.columns = cols
                st.session_state["category_counts"] = category_counts
            except (DataLoadError, Exception) as e:
                st.error(f"Er ging iets mis met het verwerken: {e}")
                st.stop()
        else:
            try:
                df_hedge = load_aggregated_csv(uploaded_file)
            except (DataLoadError, Exception) as e:
                st.error(f"Fout bij inlezen bestand: {e}")
                st.stop()

        # Voorbereiden en opslaan
        st.session_state["df_prepared"] = prepare_dataframe(df_hedge)
        st.rerun()
    else:
        st.info("Upload hiernaast een bestand om de magie te starten.")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────
# Fase 2: Data geladen — config sidebar + main content
# ─────────────────────────────────────────────────────────────────────────

df = st.session_state["df_prepared"]

# Config sidebar
config = ui_sidebar.render_config_sidebar(df)
category = config["category"]
strategy_period = config["strategy_period"]
vol_multiplier = config["vol_multiplier"]
epex_multiplier = config["epex_multiplier"]
df = config["df"]  # df met Price_Base/Price_Peak kolommen

# ─── Actief profiel instellen ─────────────────────────────────────────

p_mw_col = "Active_Profile_MW"
df[p_mw_col] = df[f"{category}_MW"] * (1 + vol_multiplier)

# ─── EPEX ophalen (automatisch) ──────────────────────────────────────

epex_loaded = False
try:
    has_api_key = "ENTSOE_API_KEY" in st.secrets
except FileNotFoundError:
    has_api_key = False

if not has_api_key:
    st.sidebar.warning("ENTSO-E API Key ontbreekt")
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
    st.sidebar.success("EPEX spotprijzen geladen")
else:
    st.sidebar.info("EPEX niet beschikbaar")

# ─── Categorisatie-overzicht (alleen bij ruwe data) ───────────────────

category_counts = st.session_state.get("category_counts")
if category_counts is not None:
    with st.expander("Categorisatie afgerond", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Consumers", category_counts.get("Consumer", 0))
        c2.metric("Prosumers", category_counts.get("Prosumer", 0))
        c3.metric("Producers", category_counts.get("Producer", 0))

# ─── Profiel-DataFrame voorbereiden ───────────────────────────────────

df_calc = df.copy()
df_calc[f"{category}_MW"] = df_calc[p_mw_col]

# ─── 100% SPOT baseline berekening ───────────────────────────────────
# Start met geen hedge (Base=0, Peak=0) als er geen selectie is

selected = ui_state.get_selected_scenario(category)

if selected:
    # Na selectie: seizoensgrafieken worden getoond in render_results_section
    pass
else:
    # Baseline: 100% SPOT (geen hedge)
    df_baseline = df.copy()
    df_baseline["Hedge_Base_MW"] = 0.0
    df_baseline["Hedge_Peak_MW"] = 0.0
    df_baseline["Current_Hedge_MW"] = 0.0
    df_baseline = apply_hedge_columns(df_baseline, p_mw_col)
    q_stats_baseline = compute_quarterly_table(df_baseline, epex_loaded)

    ui_tabs.render_seasonal_preview(
        df_baseline, p_mw_col, q_stats_baseline, has_hedge=False
    )

# ─── Vergelijkingstabel (4 optimalisatie-kaarten) ─────────────────────

st.markdown("---")
st.markdown("### Kies je strategie _")

if strategy_period == "Per Jaar":
    price_base = df_calc["Price_Base"].mean()
    price_peak = df_calc["Price_Peak"].mean()

    with st.spinner(f"Scenario's berekenen voor {category}..."):
        cat_scenarios = compute_all_scenarios(
            df_calc, category, price_base, price_peak, epex_loaded
        )
    render_comparison_table(cat_scenarios)

else:
    # Per Kwartaal: jaaroverzicht + kwartaalbreakdown
    price_base = df_calc["Price_Base"].mean()
    price_peak = df_calc["Price_Peak"].mean()

    st.markdown("#### Jaaroverzicht")
    with st.spinner(f"Scenario's berekenen voor {category} (jaar)..."):
        cat_scenarios_yr = compute_all_scenarios(
            df_calc, category, price_base, price_peak, epex_loaded
        )
    render_comparison_table(cat_scenarios_yr)

    st.markdown("---")
    st.markdown("#### Kwartaal detail")

    q_tabs = st.tabs(["Q1", "Q2", "Q3", "Q4"])
    for idx, q in enumerate([1, 2, 3, 4]):
        with q_tabs[idx]:
            df_q = df_calc[df_calc["Quarter"] == q]
            if len(df_q) == 0:
                st.info(f"Geen data voor Q{q}.")
                continue

            price_base_q = df_q["Price_Base"].mean()
            price_peak_q = df_q["Price_Peak"].mean()

            with st.spinner(f"Q{q} berekenen..."):
                cat_scenarios_q = compute_all_scenarios(
                    df_q, category, price_base_q, price_peak_q, epex_loaded
                )
            render_comparison_table(
                cat_scenarios_q, key_prefix=f"q{q}_", selectable=False
            )

# ─── Resultaten (alleen als scenario geselecteerd) ────────────────────

if ui_state.has_scenario_selected(category):
    st.markdown("---")
    ui_tabs.render_results_section(df, p_mw_col, strategy_period, epex_loaded)
