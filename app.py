import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- IMPORT ONZE EIGEN MODULES ---
from data_processor import process_raw_connections
from epex_api import fetch_epex_prices
from endex_pricing import get_default_price
from hedge_optimizer import find_optimal_mw

# Pagina instellingen
st.set_page_config(page_title="Energy Hedge Optimizer 8.8", layout="wide")
st.title("⚡ Energy Hedge Optimizer 8.8 (Modulair)")

# --- DOCUMENTATIE BLOK ---
with st.expander("📘 Lees mij: Achtergrond en Methodiek (Klik om te openen)", expanded=False):
    st.markdown("""
    ### 1. Van Ruwe Data naar Profiel
    Het model analyseert slimme meter data om te bepalen of een aansluiting een **Consumer**, **Prosumer** of **Producer**.
    
    ### 2. Hedge Strategie
    We kopen in op de groothandelsmarkt in blokken van **0,1 MW** (Base en Peak).
    
    ### 3. Financiële Waardering (Blokken & Spot)
    Via contractprijzen en EPEX Spotmarkt bepalen we de daadwerkelijke Integrale Kostprijs.
    """)

# --- DYNAMISCHE ZIJBALK LOGICA ---
has_file = st.session_state.get('file_uploader_key') is not None

if has_file:
    c_config = st.sidebar.container()
    st.sidebar.markdown("---")
    c_input = st.sidebar.container()
    header_cfg = 1
    header_input = 5
else:
    c_input = st.sidebar.container()
    c_config = st.sidebar.container()
    header_input = 1
    header_cfg = 2 

# --- DATA INPUT (IN CONTAINER) ---
with c_input:
    st.header(f"{header_input}. Data Input" + (" (Ander Bestand)" if has_file else ""))
    input_mode = st.radio("Input Type", ["Ruwe Aansluitingen (CSV)", "Reeds Geaggregeerd (CSV)"], key="input_mode")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="file_uploader_key")

df_hedge = None

if uploaded_file is not None:
    if input_mode == "Ruwe Aansluitingen (CSV)":
        try:
            df_agg, mapping = process_raw_connections(uploaded_file)
            with st.expander("ℹ️ Resultaat Analyse & Categorisatie", expanded=True):
                c1, c2, c3 = st.columns(3)
                counts = pd.Series(mapping.values()).value_counts()
                c1.metric("Consumers", counts.get('Consumer', 0))
                c2.metric("Prosumers", counts.get('Prosumer', 0))
                c3.metric("Producers", counts.get('Producer', 0))
            
            df_hedge = df_agg.reset_index()
            cols = list(df_hedge.columns)
            cols[0] = 'Date'
            df_hedge.columns = cols
        except Exception as e:
            st.error(f"Fout bij verwerken ruwe data: {e}")
            st.stop()
    else: 
        try:
            df_hedge = pd.read_csv(uploaded_file, sep=';', decimal=',')
            if 'Date' not in df_hedge.columns:
                for c in ['Datum', 'Tijd', 'Time', 'date', 'time']:
                    if c in df_hedge.columns:
                        df_hedge.rename(columns={c: 'Date'}, inplace=True)
                        break
            if 'Date' not in df_hedge.columns:
                cols = list(df_hedge.columns); cols[0] = 'Date'; df_hedge.columns = cols

            for col in ['Consumer', 'Prosumer', 'Producer', 'Total']:
                if col in df_hedge.columns:
                    df_hedge[col] = pd.to_numeric(df_hedge[col].astype(str).str.replace(',', '.'), errors='coerce')
            
            if 'Total' not in df_hedge.columns:
                cols_to_sum = [c for c in ['Consumer', 'Prosumer', 'Producer'] if c in df_hedge.columns]
                df_hedge['Total'] = df_hedge[cols_to_sum].sum(axis=1) if cols_to_sum else 0.0
            
            df_hedge['Date'] = pd.to_datetime(df_hedge['Date'], dayfirst=True)
        except Exception as e:
            st.error(f"Fout bij inlezen bestand: {e}")
            st.stop()

# --- HEDGE LOGICA & BEREKENINGEN ---
if df_hedge is not None:
    df = df_hedge.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    df = df.sort_values('Date').drop_duplicates(subset='Date', keep='first')
    df = df.set_index('Date').asfreq('15min').ffill().reset_index()

    for col in ['Consumer', 'Prosumer', 'Producer', 'Total']:
        if col in df.columns: df[f'{col}_MW'] = (df[col] * 4) / 1000
        else: df[f'{col}_MW'] = 0.0
    
    df['is_peak'] = (df['Date'].dt.weekday < 5) & (df['Date'].dt.hour >= 8) & (df['Date'].dt.hour < 20)
    df['Quarter'] = df['Date'].dt.quarter

    with c_config:
        st.header(f"{header_cfg}. Hedge Configuratie")
        profile_choice = st.selectbox("Kies Profiel", ["Consumer", "Prosumer", "Producer", "Total"])
        strategy_period = st.radio("Periode", ["Per Jaar", "Per Kwartaal"])
        p_mw_col = f'{profile_choice}_MW'

        if 'slider_values' not in st.session_state: st.session_state['slider_values'] = {}

        # --- STRATEGIE BLOK ---
        st.markdown("---")
        st.header(f"{header_cfg + 1}. Kies Strategie")

        def apply_strategy(strat_name, custom_pct=None):
            periods = [0] if strategy_period == "Per Jaar" else [1, 2, 3, 4]
            for q in periods:
                sub_df = df if q == 0 else df[df['Quarter'] == q]
                if strat_name == "5%_sell": b, p = find_optimal_mw(sub_df, p_mw_col, target_over_pct_limit=5.0)
                elif strat_name == "10%_cov": b, p = find_optimal_mw(sub_df, p_mw_col, percent_volume_target=10)
                elif strat_name == "100%_cov": b, p = find_optimal_mw(sub_df, p_mw_col, percent_volume_target=100)
                elif strat_name == "custom_cov": b, p = find_optimal_mw(sub_df, p_mw_col, percent_volume_target=custom_pct)
                
                st.session_state[f'slider_b_yr' if q == 0 else f'slider_b_q{q}'] = float(b)
                st.session_state[f'slider_p_yr' if q == 0 else f'slider_p_q{q}'] = float(p)

        def on_custom_hedge_change():
            apply_strategy("custom_cov", custom_pct=st.session_state.custom_hedge_pct)

        col1, col2, col3 = st.columns(3)
        if col1.button("🎯 Max 5% Sell"): apply_strategy("5%_sell")
        if col2.button("📉 10% Hedge"): apply_strategy("10%_cov")
        if col3.button("⚖️ 100% Hedge"): apply_strategy("100%_cov")

        st.slider(
            "% Hedge op Volume", 
            min_value=0, max_value=150, value=100, step=1, 
            key="custom_hedge_pct", 
            on_change=on_custom_hedge_change,
            help="Versleep deze slider om de dekking aan te passen. De Base & Peak volumes (MW) worden direct geüpdatet."
        )

        # --- Sliders MW (Fine-Tuning) ---
        st.markdown("---")
        st.subheader(f"{header_cfg + 2}. Fine-tuning Volumes (MW)")
        
        df['Hedge_Base_MW'] = 0.0
        df['Hedge_Peak_MW'] = 0.0
        
        curr_min = df[p_mw_col].min()
        curr_max = df[p_mw_col].max()
        slider_min = float(np.floor(curr_min * 1.5 - 1))
        slider_max = float(np.ceil(curr_max * 1.5 + 1))
        if slider_max < slider_min: slider_max = slider_min + 10.0

        if strategy_period == "Per Jaar":
            if 'slider_b_yr' not in st.session_state:
                def_b, def_p = find_optimal_mw(df, p_mw_col, percent_volume_target=100)
                st.session_state['slider_b_yr'] = float(def_b); st.session_state['slider_p_yr'] = float(def_p)
                
            b_yr = st.slider("Base (Jaar)", slider_min, slider_max, key="slider_b_yr", step=0.1)
            p_yr = st.slider("Peak (Jaar)", slider_min, slider_max, key="slider_p_yr", step=0.1)
            
            df['Hedge_Base_MW'] = b_yr
            df['Hedge_Peak_MW'] = p_yr * df['is_peak']
        else:
            for q in [1, 2, 3, 4]:
                st.markdown(f"**Kwartaal {q}**")
                q_mask = df['Quarter'] == q
                if f'slider_b_q{q}' not in st.session_state:
                    def_b, def_p = find_optimal_mw(df[q_mask], p_mw_col, percent_volume_target=100)
                    st.session_state[f'slider_b_q{q}'] = float(def_b); st.session_state[f'slider_p_q{q}'] = float(def_p)
                
                sc1, sc2 = st.columns(2)
                b_q = sc1.slider(f"Q{q} Base", slider_min, slider_max, key=f"slider_b_q{q}", step=0.1)
                p_q = sc2.slider(f"Q{q} Peak", slider_min, slider_max, key=f"slider_p_q{q}", step=0.1)
                
                df.loc[q_mask, 'Hedge_Base_MW'] = b_q
                df.loc[q_mask, 'Hedge_Peak_MW'] = p_q * df.loc[q_mask, 'is_peak']
                
        df['Current_Hedge_MW'] = df['Hedge_Base_MW'] + df['Hedge_Peak_MW']

        # --- CONTRACTPRIJZEN BLOKKEN ---
        st.markdown("---")
        st.subheader(f"{header_cfg + 3}. Contractprijzen (Inkoopblokken)")
        
        df['Price_Base'] = 0.0
        df['Price_Peak'] = 0.0
        
        if strategy_period == "Per Jaar":
            cp1, cp2 = st.columns(2)
            def_b, def_p = get_default_price("Jaar")
            pr_b = cp1.number_input("Base Prijs (€/MWh)", value=def_b, step=1.0)
            pr_p = cp2.number_input("Peak Prijs (€/MWh)", value=def_p, step=1.0)
            df['Price_Base'] = pr_b
            df['Price_Peak'] = pr_p
        else:
            for q in [1, 2, 3, 4]:
                st.markdown(f"**Prijzen Q{q}**")
                cp1, cp2 = st.columns(2)
                def_b, def_p = get_default_price("Kwartaal", q)
                pr_b = cp1.number_input(f"Q{q} Base (€/MWh)", value=def_b, step=1.0, key=f"pr_b_q{q}")
                pr_p = cp2.number_input(f"Q{q} Peak (€/MWh)", value=def_p, step=1.0, key=f"pr_p_q{q}")
                q_mask = df['Quarter'] == q
                df.loc[q_mask, 'Price_Base'] = pr_b
                df.loc[q_mask, 'Price_Peak'] = pr_p

    # --- ENTSO-E INPUT (AUTOMATISCH) ---
    epex_loaded = False
    
    if "ENTSOE_API_KEY" not in st.secrets:
        st.sidebar.error("⚠️ Systeemconfiguratiefout: ENTSO-E API Key ontbreekt in de server instellingen (.streamlit/secrets.toml).")
    else:
        api_key = st.secrets["ENTSOE_API_KEY"]
        start_dt = df['Date'].min()
        end_dt = df['Date'].max()
        
        with st.spinner("Prijzen downloaden via ENTSO-E API..."):
            df_epex = fetch_epex_prices(api_key, start_dt, end_dt)
        
        if isinstance(df_epex, str):
            st.sidebar.error(f"⚠️ Fout bij ophalen EPEX: Controleer de API Key en de verbinding. Details: {df_epex}")
        else:
            df['Date_Hour'] = df['Date'].dt.floor('H')
            df = pd.merge(df, df_epex[['Date_Hour', 'EPEX_EUR_MWh']], on='Date_Hour', how='left')
            epex_loaded = True
            st.sidebar.success("✅ EPEX Prijzen automatisch gekoppeld!")

    if not epex_loaded:
        df['EPEX_EUR_MWh'] = 0.0

    # --- RESULTATEN BEREKENEN ---
    df['Profile_MWh'] = df[p_mw_col] * 0.25
    df['Hedge_MWh'] = df['Current_Hedge_MW'] * 0.25
    df['Used_Hedge_MWh'] = np.minimum(df['Hedge_MWh'], df['Profile_MWh']) 
    df['Over_Hedge_MWh'] = np.maximum(0, df['Hedge_MWh'] - df['Profile_MWh'])
    df['Under_Hedge_MWh'] = np.maximum(0, df['Profile_MWh'] - df['Hedge_MWh'])

    # Blok kosten = (MW * 0.25) * Prijs
    df['Cost_Hedge_Base_EUR'] = (df['Hedge_Base_MW'] * 0.25) * df['Price_Base']
    df['Cost_Hedge_Peak_EUR'] = (df['Hedge_Peak_MW'] * 0.25) * df['Price_Peak']
    df['Cost_Hedge_Total_EUR'] = df['Cost_Hedge_Base_EUR'] + df['Cost_Hedge_Peak_EUR']
    
    tot_hedge_eur = df['Cost_Hedge_Total_EUR'].sum()

    # Spotmarkt kosten
    df['Cost_Buy_EUR'] = df['Under_Hedge_MWh'] * df['EPEX_EUR_MWh'] if epex_loaded else 0.0
    df['Rev_Sell_EUR'] = df['Over_Hedge_MWh'] * df['EPEX_EUR_MWh'] if epex_loaded else 0.0
    df['Net_Spot_EUR'] = df['Rev_Sell_EUR'] - df['Cost_Buy_EUR'] if epex_loaded else 0.0

    total_prof = df['Profile_MWh'].sum()
    total_hedge = df['Hedge_MWh'].sum()
    total_over = df['Over_Hedge_MWh'].sum()
    total_under = df['Under_Hedge_MWh'].sum()
    denom = total_prof if total_prof != 0 else 1.0
    
    # UI METRICS VOLUMES
    st.markdown("### 📊 Volume Balans")
    k1, k2, k3, k4, k5 = st.columns(5)
    
    pct_hedge_eff = (df['Used_Hedge_MWh'].sum() / denom) * 100
    pct_hedge_tot = (total_hedge / denom) * 100
    
    k1.metric("Effectieve Dekking", f"{pct_hedge_eff:.1f}%", help="Percentage van het verbruik dat direct in hetzelfde kwartier is afgedekt.")
    k2.metric("Totale Hedge %", f"{pct_hedge_tot:.1f}%", help="Totaal ingekocht volume gedeeld door het totale verbruik.")
    k3.metric("Totaal Verbruik", f"{total_prof:,.0f} MWh")
    k4.metric("Teveel (Over)", f"{total_over:,.0f} MWh", f"{(total_over/denom)*100:.1f}%", delta_color="inverse")
    k5.metric("Tekort (Under)", f"{total_under:,.0f} MWh", f"{(total_under/denom)*100:.1f}%", delta_color="inverse")

    # UI METRICS FINANCIEEL
    if epex_loaded:
        st.markdown("### 💶 Financiële Waardering Totaal (Blokken + Spotmarkt)")
        f1, f2, f3, f4 = st.columns(4)
        
        net_spot_eur = df['Net_Spot_EUR'].sum()
        tot_energy_cost = tot_hedge_eur - net_spot_eur 
        avg_cost = tot_energy_cost / denom if denom > 0 else 0
        
        f1.metric("Kosten Inkoopblokken", f"€ {tot_hedge_eur:,.0f}")
        
        net_color = "normal" if net_spot_eur >= 0 else "inverse"
        f2.metric("Netto Spot Resultaat", f"€ {net_spot_eur:,.0f}", delta="Winst" if net_spot_eur >=0 else "Verlies", delta_color=net_color)
        
        f3.metric("Totale Kosten (Netto)", f"€ {tot_energy_cost:,.0f}")
        
        if avg_cost < 0:
            f4.metric("Integrale Opbrengst (Winst)", f"€ {abs(avg_cost):.2f} / MWh", delta="Verdienste", delta_color="normal")
        else:
            f4.metric("Integrale Kostprijs", f"€ {avg_cost:.2f} / MWh")
    else:
        st.markdown("### 💶 Financiële Waardering (Alleen Inkoopblokken)")
        f1, f2, f3 = st.columns(3)
        f1.metric("Kosten Inkoopblokken", f"€ {tot_hedge_eur:,.0f}")
        gem_blok = tot_hedge_eur / df['Hedge_MWh'].sum() if df['Hedge_MWh'].sum() > 0 else 0
        f2.metric("Gemiddelde Blokprijs", f"€ {gem_blok:.2f} / MWh")

    st.markdown("---")
    st.subheader("🔎 Seizoensanalyse (4 Representatieve Weken)")
    weeks = [
        {"name": "Februari", "start": "2025-02-03", "end": "2025-02-09"},
        {"name": "Mei",       "start": "2025-05-05", "end": "2025-05-11"},
        {"name": "Augustus",  "start": "2025-08-04", "end": "2025-08-10"},
        {"name": "November", "start": "2025-11-03", "end": "2025-11-09"}
    ]
    cols_chart = st.columns(2) + st.columns(2)
    for i, week in enumerate(weeks):
        with cols_chart[i]:
            st.caption(week['name'])
            if df['Date'].min() > pd.Timestamp(week['end']) or df['Date'].max() < pd.Timestamp(week['start']):
                st.info("Geen data")
                continue
            mask = (df['Date'] >= week['start']) & (df['Date'] <= pd.Timestamp(week['end']) + pd.Timedelta(days=1))
            chart_data = df.loc[mask].melt(id_vars=['Date'], value_vars=[p_mw_col, 'Current_Hedge_MW'], var_name='Type', value_name='MW')
            chart_data['Type'] = chart_data['Type'].replace({p_mw_col: 'Verbruik', 'Current_Hedge_MW': 'Hedge'})
            c = alt.Chart(chart_data).mark_line(interpolate='step-after').encode(
                x=alt.X('Date:T', axis=alt.Axis(format='%a %H:%M', title=None)),
                y=alt.Y('MW:Q', title=None), color=alt.Color('Type:N', legend=alt.Legend(orient='bottom', title=None))
            ).properties(height=180)
            st.altair_chart(c, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Kwartaal Balans")
    
    q_stats = df.groupby('Quarter').apply(lambda x: pd.Series({
        'Verbruik (MWh)': x['Profile_MWh'].sum(),
        'Hedge %': (x['Hedge_MWh'].sum() / x['Profile_MWh'].sum() * 100) if x['Profile_MWh'].sum() != 0 else 0,
        'Dekking %': (x['Used_Hedge_MWh'].sum() / x['Profile_MWh'].sum() * 100) if x['Profile_MWh'].sum() != 0 else 0,
        'Teveel (MWh)': x['Over_Hedge_MWh'].sum(),
        'Tekort (MWh)': x['Under_Hedge_MWh'].sum(),
        'Blokken Kosten (€)': x['Cost_Hedge_Total_EUR'].sum(),
        'EPEX Gem. (€/MWh)': x['EPEX_EUR_MWh'].mean() if epex_loaded else 0,
        'Netto Spot (€)': x['Net_Spot_EUR'].sum() if epex_loaded else 0,
        'Totale Kosten (€)': (x['Cost_Hedge_Total_EUR'].sum() - x['Net_Spot_EUR'].sum()) if epex_loaded else x['Cost_Hedge_Total_EUR'].sum(),
        'Integrale Prijs (€/MWh)': ((x['Cost_Hedge_Total_EUR'].sum() - x['Net_Spot_EUR'].sum()) / x['Profile_MWh'].sum()) if epex_loaded and x['Profile_MWh'].sum() != 0 else (x['Cost_Hedge_Total_EUR'].sum() / x['Profile_MWh'].sum() if x['Profile_MWh'].sum() != 0 else 0)
    }))

    format_dict = {
        'Verbruik (MWh)': "{:,.0f}", 'Hedge %': "{:.1f}%", 'Dekking %': "{:.1f}%", 'Teveel (MWh)': "{:,.0f}", 'Tekort (MWh)': "{:,.0f}",
        'Blokken Kosten (€)': "€ {:,.0f}", 'Totale Kosten (€)': "€ {:,.0f}", 'Integrale Prijs (€/MWh)': "€ {:.2f}"
    }
    if epex_loaded:
        format_dict.update({'EPEX Gem. (€/MWh)': "€ {:.2f}", 'Netto Spot (€)': "€ {:,.0f}"})
    else:
        q_stats = q_stats.drop(columns=['EPEX Gem. (€/MWh)', 'Netto Spot (€)'])

    st.dataframe(q_stats.style.format(format_dict), use_container_width=True)

    csv_dl = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Detail Data (CSV)", csv_dl, "hedge_resultaten.csv", "text/csv")
else:
    st.info("👆 Upload hiernaast een CSV-bestand om te beginnen.")
