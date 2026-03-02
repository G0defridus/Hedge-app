import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- IMPORT ONZE EIGEN MODULES ---
from data_processor import process_raw_connections
from epex_api import fetch_epex_prices
from endex_pricing import get_default_price
from hedge_optimizer import find_optimal_mw, optimize_advanced

# --- PAGINA INSTELLINGEN & CENSO HUISSTIJL (CSS) ---
st.set_page_config(page_title="Censo Energy Optimizer", layout="wide")

censo_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend+Deca:wght@400;500;700&family=Montserrat:ital,wght@1,400;1,500;1,700&display=swap');

html, body, [class*="css"] {
    font-family: 'Lexend Deca', sans-serif !important;
}
h1, h2, h3, h4 {
    font-family: 'Lexend Deca', sans-serif !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
}
/* Knoppen in Censo Gold */
.stButton>button {
    background-color: #fab517 !important;
    color: #000000 !important; 
    border: none !important;
    border-radius: 4px;
    font-weight: 500;
}
.stButton>button:hover {
    background-color: #d99d12 !important; 
}
/* Tabbladen styling */
.stTabs [data-baseweb="tab-list"] { gap: 2rem; }
.stTabs [data-baseweb="tab"] {
    height: 3rem;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 0px;
    font-family: 'Lexend Deca', sans-serif;
    font-weight: 500;
}
.stTabs [aria-selected="true"] { border-bottom: 3px solid #e8327c !important; }
</style>
"""
st.markdown(censo_css, unsafe_allow_html=True)

st.markdown("""
<div style="font-size: 2.8rem; font-weight: 700; margin-bottom: 1rem; margin-top: -1rem; font-family: 'Lexend Deca', sans-serif;">
    De energie-strategie. <span style="color: #e8327c;">Maar dan simpel</span> <span style="color: #fab517;">_</span>
</div>
""", unsafe_allow_html=True)

# --- DOCUMENTATIE BLOK ---
with st.expander("Hoe het werkt _", expanded=False):
    st.markdown("""
    **Samen zorgen we dat het goed voelt.**
    We analyseren jouw data en berekenen direct de impact op de spotmarkt.
    Nieuw in deze versie is de intelligente solver: laat het wiskundige model berekenen wat jouw absoluut goedkoopste (of minst risicovolle) inkoopstrategie is.
    """)

# --- DYNAMISCHE ZIJBALK LOGICA ---
has_file = st.session_state.get('file_uploader_key') is not None

if has_file:
    c_config = st.sidebar.container()
    st.sidebar.markdown("---")
    c_input = st.sidebar.container()
else:
    c_input = st.sidebar.container()
    c_config = st.sidebar.container()

# --- DATA INPUT ---
with c_input:
    st.header("1. Upload je data _" if not has_file else "Ander bestand _")
    input_mode = st.radio("Kies het type bestand", ["Ruwe Aansluitingen (CSV)", "Reeds Geaggregeerd (CSV)"], key="input_mode")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="file_uploader_key")

df_hedge = None

if uploaded_file is not None:
    if input_mode == "Ruwe Aansluitingen (CSV)":
        try:
            df_agg, mapping = process_raw_connections(uploaded_file)
            with st.expander("Analyse afgerond _", expanded=True):
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
            st.error(f"Er ging iets mis met het verwerken: {e}")
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

# --- HOOFDLOGICA ---
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
        st.header("2. Basisinstellingen _")
        profile_choice = st.selectbox("Welk profiel bekijken we?", ["Consumer", "Prosumer", "Producer", "Total"])
        strategy_period = st.radio("Contractperiode", ["Per Jaar", "Per Kwartaal"])
        
        st.markdown("---")
        st.header("3. Speel met scenario's _")
        vol_multiplier = st.slider("Verwachte groei of zon", min_value=-50, max_value=50, value=0, step=5, format="%d%%") / 100.0
        epex_multiplier = st.slider("Spotprijzen (EPEX)", min_value=-100, max_value=200, value=0, step=10, format="%d%%") / 100.0

        p_mw_col = 'Active_Profile_MW'
        df[p_mw_col] = df[f'{profile_choice}_MW'] * (1 + vol_multiplier)

        # ---------------------------------------------------------
        # NIEUWE VOLGORDE: Eerst prijzen & EPEX ophalen, dan pas strategie!
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("4. Jouw contractprijzen _")
        
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
                pr_b = cp1.number_input(f"Q{q} Base", value=def_b, step=1.0, key=f"pr_b_q{q}")
                pr_p = cp2.number_input(f"Q{q} Peak", value=def_p, step=1.0, key=f"pr_p_q{q}")
                q_mask = df['Quarter'] == q
                df.loc[q_mask, 'Price_Base'] = pr_b
                df.loc[q_mask, 'Price_Peak'] = pr_p

        # EPEX OPHALEN
        epex_loaded = False
        if "ENTSOE_API_KEY" not in st.secrets:
            st.error("⚠️ ENTSO-E API Key ontbreekt in .streamlit/secrets.toml")
        else:
            api_key = st.secrets["ENTSOE_API_KEY"]
            start_dt = df['Date'].min()
            end_dt = df['Date'].max()
            with st.spinner("Spotprijzen ophalen voor berekeningen..."):
                df_epex = fetch_epex_prices(api_key, start_dt, end_dt)
            if not isinstance(df_epex, str):
                df['Date_Hour'] = df['Date'].dt.floor('H')
                df_epex['EPEX_EUR_MWh'] = df_epex['EPEX_EUR_MWh'] * (1 + epex_multiplier)
                df = pd.merge(df, df_epex[['Date_Hour', 'EPEX_EUR_MWh']], on='Date_Hour', how='left')
                epex_loaded = True
        
        if not epex_loaded:
            df['EPEX_EUR_MWh'] = 0.0

        # ---------------------------------------------------------
        # STRATEGIE BLOK (Nu inclusief Algoritmes!)
        # ---------------------------------------------------------
        st.markdown("---")
        st.header("5. Kies je aanpak _")

        if 'slider_values' not in st.session_state: st.session_state['slider_values'] = {}

        def apply_strategy(strat_name, custom_pct=None):
            periods = [0] if strategy_period == "Per Jaar" else [1, 2, 3, 4]
            for q in periods:
                sub_df = df if q == 0 else df[df['Quarter'] == q]
                pb = sub_df['Price_Base'].iloc[0]
                pp = sub_df['Price_Peak'].iloc[0]
                
                if strat_name in ["least_cost", "value_risk"]:
                    b, p = optimize_advanced(sub_df, p_mw_col, pb, pp, strategy=strat_name)
                elif strat_name == "5%_sell": b, p = find_optimal_mw(sub_df, p_mw_col, target_over_pct_limit=5.0)
                elif strat_name == "10%_cov": b, p = find_optimal_mw(sub_df, p_mw_col, percent_volume_target=10)
                elif strat_name == "100%_cov": b, p = find_optimal_mw(sub_df, p_mw_col, percent_volume_target=100)
                elif strat_name == "custom_cov": b, p = find_optimal_mw(sub_df, p_mw_col, percent_volume_target=custom_pct)
                
                st.session_state[f'slider_b_yr' if q == 0 else f'slider_b_q{q}'] = float(b)
                st.session_state[f'slider_p_yr' if q == 0 else f'slider_p_q{q}'] = float(p)

        def on_custom_hedge_change():
            apply_strategy("custom_cov", custom_pct=st.session_state.custom_hedge_pct)

        # Rijen met knoppen verdeeld over Volume en Waarde
        st.markdown("**Slimme Optimalisatie (Algoritmes):**")
        s1, s2 = st.columns(2)
        if s1.button("📉 Zoek Laagste Kostprijs", help="Het model test duizenden combinaties om de laagste integrale prijs te vinden (inclusief spotmarkt)."): 
            apply_strategy("least_cost")
        if s2.button("⚖️ Zoek Minste Risico (Value Hedge)", help="Minimaliseert de financiële schommelingen en dekt piekprijzen agressiever af."): 
            apply_strategy("value_risk")

        st.markdown("**Volume-gebaseerd:**")
        v1, v2, v3 = st.columns(3)
        if v1.button("100% Volume"): apply_strategy("100%_cov")
        if v2.button("Subtiel (10%)"): apply_strategy("10%_cov")
        if v3.button("Max 5% over"): apply_strategy("5%_sell")

        st.slider("Of kies exact percentage", min_value=0, max_value=150, value=100, step=1, key="custom_hedge_pct", on_change=on_custom_hedge_change)

        st.markdown("---")
        st.subheader("6. Finetunen in MW _")
        
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

    # --- RESULTATEN BEREKENEN ---
    df['Profile_MWh'] = df[p_mw_col] * 0.25
    df['Hedge_MWh'] = df['Current_Hedge_MW'] * 0.25
    
    df['Over_Hedge_MWh'] = np.maximum(0, df['Hedge_MWh'] - df['Profile_MWh'])
    df['Under_Hedge_MWh'] = np.maximum(0, df['Profile_MWh'] - df['Hedge_MWh'])

    df['Used_Hedge_MWh_Abs'] = np.where(
        np.sign(df['Profile_MWh']) == np.sign(df['Hedge_MWh']),
        np.minimum(df['Profile_MWh'].abs(), df['Hedge_MWh'].abs()),
        0.0
    )

    df['Cost_Hedge_Base_EUR'] = (df['Hedge_Base_MW'] * 0.25) * df['Price_Base']
    df['Cost_Hedge_Peak_EUR'] = (df['Hedge_Peak_MW'] * 0.25) * df['Price_Peak']
    df['Cost_Hedge_Total_EUR'] = df['Cost_Hedge_Base_EUR'] + df['Cost_Hedge_Peak_EUR']
    
    tot_hedge_eur = df['Cost_Hedge_Total_EUR'].sum()

    df['Cost_Buy_EUR'] = df['Under_Hedge_MWh'] * df['EPEX_EUR_MWh'] if epex_loaded else 0.0
    df['Rev_Sell_EUR'] = df['Over_Hedge_MWh'] * df['EPEX_EUR_MWh'] if epex_loaded else 0.0
    df['Net_Spot_EUR'] = df['Rev_Sell_EUR'] - df['Cost_Buy_EUR'] if epex_loaded else 0.0

    total_prof = df['Profile_MWh'].sum()
    total_prof_abs = abs(total_prof)
    total_hedge_abs = df['Hedge_MWh'].abs().sum()
    total_over = df['Over_Hedge_MWh'].sum()
    total_under = df['Under_Hedge_MWh'].sum()
    denom = total_prof_abs if total_prof_abs != 0 else 1.0
    
    pct_hedge_eff = (df['Used_Hedge_MWh_Abs'].sum() / denom) * 100
    
    net_spot_eur = df['Net_Spot_EUR'].sum()
    tot_energy_cost = tot_hedge_eur - net_spot_eur 
    avg_cost = tot_energy_cost / denom if denom > 0 else 0

    # --- TABBLADEN STRUCTUUR (VISUALISATIES) ---
    st.markdown("<br>", unsafe_allow_html=True)
    tab_main, tab_vol, tab_eco, tab_charts = st.tabs([
        "Samenvatting", 
        "Jouw volume flow", 
        "Kengetallen", 
        "Seizoenen"
    ])

    # TAB 1: HOOFDOVERZICHT
    with tab_main:
        st.markdown("### Helder overzicht _")
        m1, m2, m3 = st.columns(3)
        
        if avg_cost < 0:
            m1.metric("Jouw opbrengst (Winst)", f"€ {abs(avg_cost):.2f} / MWh", "Verdienmodel", delta_color="normal")
        else:
            m1.metric("Jouw kostprijs", f"€ {avg_cost:.2f} / MWh", "Kosten", delta_color="inverse")
            
        cost_label = "Totale verdienste (Netto)" if tot_energy_cost < 0 else "Totale kosten (Netto)"
        m2.metric(cost_label, f"€ {abs(tot_energy_cost):,.0f}")
        m3.metric("Direct afgedekt", f"{pct_hedge_eff:.1f}%")

        st.markdown("---")
        st.markdown("### De cijfers door het jaar heen _")
        
        df['Maand_Naam'] = df['Date'].dt.strftime('%Y-%m')
        
        monthly_agg = df.groupby('Maand_Naam')[['Cost_Hedge_Total_EUR', 'Cost_Buy_EUR', 'Rev_Sell_EUR']].sum().reset_index()
        monthly_agg['Rev_Sell_EUR'] = -monthly_agg['Rev_Sell_EUR'] 
        
        monthly_melt = monthly_agg.melt(id_vars=['Maand_Naam'], value_vars=['Cost_Hedge_Total_EUR', 'Cost_Buy_EUR', 'Rev_Sell_EUR'], var_name='Kostenpost', value_name='Euro')
        
        monthly_melt['Kostenpost'] = monthly_melt['Kostenpost'].replace({
            'Cost_Hedge_Total_EUR': '1. Vaste inkoop',
            'Cost_Buy_EUR': '2. Spot inkoop (tekort)',
            'Rev_Sell_EUR': '3. Spot verkoop (overschot)'
        })

        cashflow_chart = alt.Chart(monthly_melt).mark_bar().encode(
            x=alt.X('Maand_Naam:N', title='Maand'),  
            y=alt.Y('sum(Euro):Q', title='Bedrag in Euro'),
            color=alt.Color('Kostenpost:N', 
                            scale=alt.Scale(
                                domain=['1. Vaste inkoop', '2. Spot inkoop (tekort)', '3. Spot verkoop (overschot)'],
                                range=['#9e9e9e', '#e8327c', '#fab517'] 
                            ), legend=alt.Legend(title="", orient="bottom")),
            tooltip=[
                alt.Tooltip('Maand_Naam:N', title='Maand'), 
                alt.Tooltip('Kostenpost:N', title='Post'), 
                alt.Tooltip('sum(Euro):Q', title='Bedrag (€)', format=',.0f')
            ]
        ).properties(height=400)
        
        st.altair_chart(cashflow_chart, use_container_width=True)

    # TAB 2: VOLUME BALANS (ECHTE WATERVAL GRAFIEK)
    with tab_vol:
        st.markdown("### Jouw verbruik in balans _")
        v1, v2, v3 = st.columns(3)
        prof_label = "Jouw totale opwek" if total_prof < 0 else "Jouw totale verbruik"
        v1.metric(prof_label, f"{total_prof_abs:,.0f} MWh")
        v2.metric("Ingekocht via blokken", f"{total_hedge_abs:,.0f} MWh")
        v3.metric("Direct afgedekt", f"{pct_hedge_eff:.1f}%")

        st.markdown("<br><br><b>De flow van jouw stroom _</b>", unsafe_allow_html=True)
        st.info("Een échte watervalgrafiek: we starten met de totale behoefte, trekken de ingekochte blokken eraf, en het restant werk je weg op de spotmarkt zodat we precies op 0 (balans) uitkomen.")
        
        wf_data = []
        wf_data.append({"Stap": f"1. {prof_label}", "Start": 0, "Eind": total_prof_abs, "Volume": total_prof_abs, "Kleur": "Profiel"})
        eind_hedge = total_prof_abs - total_hedge_abs
        wf_data.append({"Stap": "2. Ingekocht via blokken", "Start": total_prof_abs, "Eind": eind_hedge, "Volume": -total_hedge_abs, "Kleur": "Hedge"})
        if total_under > 0:
            wf_data.append({"Stap": "3. Spot inkoop (tekort)", "Start": eind_hedge, "Eind": 0, "Volume": total_under, "Kleur": "Tekort"})
        if total_over > 0:
            wf_data.append({"Stap": "4. Spot verkoop (overschot)", "Start": eind_hedge, "Eind": 0, "Volume": -total_over, "Kleur": "Overschot"})
            
        wf_df = pd.DataFrame(wf_data)
        
        waterfall_chart = alt.Chart(wf_df).mark_bar(size=60).encode(
            x=alt.X('Stap:O', title='', sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Start:Q', title='Volume (MWh)'),
            y2='Eind:Q',
            color=alt.Color('Kleur:N', scale=alt.Scale(
                domain=['Profiel', 'Hedge', 'Tekort', 'Overschot'],
                range=['#9e9e9e', '#fab517', '#e8327c', '#000000'] 
            ), legend=None),
            tooltip=[alt.Tooltip('Stap:N'), alt.Tooltip('Volume:Q', title='Volume Change (MWh)', format=',.0f')]
        ).properties(height=400)
        
        st.altair_chart(waterfall_chart, use_container_width=True)

    # TAB 3: UNIT ECONOMICS (Diepgaande KPI's)
    with tab_eco:
        st.markdown("### De cijfers per MWh _")
        if not epex_loaded:
            st.info("Spotprijzen konden niet worden geladen. Sommige velden staan op €0.")
        
        gem_spot_inkoop = df['Cost_Buy_EUR'].sum() / total_under if total_under > 0 else 0
        gem_spot_verkoop = df['Rev_Sell_EUR'].sum() / total_over if total_over > 0 else 0
        gem_vast_gebruikt = tot_hedge_eur / df['Used_Hedge_MWh_Abs'].sum() if df['Used_Hedge_MWh_Abs'].sum() > 0 else 0
        
        avg_epex_base = df['EPEX_EUR_MWh'].mean()
        capture_price = (df['Profile_MWh'].abs() * df['EPEX_EUR_MWh']).sum() / denom if denom > 0 else 0
        capture_diff = capture_price - avg_epex_base

        u1, u2 = st.columns(2)
        u1.metric("Kosten per benutte MWh", f"€ {gem_vast_gebruikt:.2f}", help="De kosten van de inkoopblokken verdeeld over uitsluitend het volume dat je *daadwerkelijk* zelf hebt afgestreept (inclusief weggegooid overschot).")
        u2.metric("Waarde van jouw profiel (Capture Price)", f"€ {capture_price:.2f}", f"€ {capture_diff:.2f} t.o.v. de markt", delta_color="normal" if capture_diff > 0 else "inverse", help="De échte gemiddelde waarde van jouw stroomprofiel op de spotmarkt.")

        st.markdown("<br>", unsafe_allow_html=True)
        u3, u4 = st.columns(2)
        u3.metric("Wat betaalde je voor tekorten?", f"€ {gem_spot_inkoop:.2f}")
        u4.metric("Wat leverde je overschot op?", f"€ {gem_spot_verkoop:.2f}")

    # TAB 4: DETAIL GRAFIEKEN & TABELLEN
    with tab_charts:
        st.markdown("### Seizoenen in de praktijk _")
        weeks = [
            {"name": "Typische winterweek", "start": "2025-02-03", "end": "2025-02-09"},
            {"name": "Typische lenteweek",  "start": "2025-05-05", "end": "2025-05-11"},
            {"name": "Typische zomerweek",  "start": "2025-08-04", "end": "2025-08-10"},
            {"name": "Typische herfstweek", "start": "2025-11-03", "end": "2025-11-09"}
        ]
        cols_chart = st.columns(2) + st.columns(2)
        for i, week in enumerate(weeks):
            with cols_chart[i]:
                st.caption(week['name'])
                if df['Date'].min() > pd.Timestamp(week['end']) or df['Date'].max() < pd.Timestamp(week['start']):
                    st.info("Geen data beschikbaar voor deze periode.")
                    continue
                
                mask = (df['Date'] >= week['start']) & (df['Date'] <= pd.Timestamp(week['end']) + pd.Timedelta(days=1))
                plot_df = df.loc[mask, ['Date', p_mw_col, 'Current_Hedge_MW']].copy()
                plot_df.rename(columns={p_mw_col: 'Jouw profiel', 'Current_Hedge_MW': 'Inkoopblok'}, inplace=True)
                
                chart_data = plot_df.melt(id_vars=['Date'], var_name='Type', value_name='MW')
                
                c = alt.Chart(chart_data).mark_line(interpolate='step-after').encode(
                    x=alt.X('Date:T', axis=alt.Axis(format='%a %H:%M', title=None)),
                    y=alt.Y('MW:Q', title=None), 
                    color=alt.Color('Type:N', 
                                    scale=alt.Scale(domain=['Jouw profiel', 'Inkoopblok'], range=['#808080', '#fab517']), 
                                    legend=alt.Legend(orient='bottom', title=None))
                ).properties(height=180)
                st.altair_chart(c, use_container_width=True)

        st.markdown("---")
        st.markdown("### Kwartaaloverzicht _")
        
        q_stats = df.groupby('Quarter').apply(lambda x: pd.Series({
            'Volume (MWh)': abs(x['Profile_MWh'].sum()),
            'Afgedekt (%)': (x['Used_Hedge_MWh_Abs'].sum() / x['Profile_MWh'].abs().sum() * 100) if x['Profile_MWh'].sum() != 0 else 0,
            'Verkocht (MWh)': x['Over_Hedge_MWh'].sum(),
            'Ingekocht (MWh)': x['Under_Hedge_MWh'].sum(),
            'Spot resultaat (€)': x['Net_Spot_EUR'].sum() if epex_loaded else 0,
            'Totale kosten (€)': (x['Cost_Hedge_Total_EUR'].sum() - x['Net_Spot_EUR'].sum()) if epex_loaded else x['Cost_Hedge_Total_EUR'].sum(),
            'Jouw prijs (€/MWh)': ((x['Cost_Hedge_Total_EUR'].sum() - x['Net_Spot_EUR'].sum()) / x['Profile_MWh'].abs().sum()) if epex_loaded and x['Profile_MWh'].sum() != 0 else (x['Cost_Hedge_Total_EUR'].sum() / x['Profile_MWh'].abs().sum() if x['Profile_MWh'].sum() != 0 else 0)
        }))

        format_dict = {
            'Volume (MWh)': "{:,.0f}", 'Afgedekt (%)': "{:.1f}%", 'Verkocht (MWh)': "{:,.0f}", 'Ingekocht (MWh)': "{:,.0f}",
            'Totale kosten (€)': "€ {:,.0f}", 'Jouw prijs (€/MWh)': "€ {:.2f}"
        }
        if epex_loaded: format_dict.update({'Spot resultaat (€)': "€ {:,.0f}"})
        else: q_stats = q_stats.drop(columns=['Spot resultaat (€)'])

        st.dataframe(q_stats.style.format(format_dict), use_container_width=True)

        csv_dl = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download je analyse (CSV)", csv_dl, "censo_analyse.csv", "text/csv")
else:
    st.info("Upload hiernaast een bestand om de magie te starten.")
