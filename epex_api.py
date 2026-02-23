import streamlit as st
import pandas as pd

@st.cache_data(show_spinner="Prijzen downloaden via ENTSO-E API...")
def fetch_epex_prices(api_key, start_date, end_date):
    try:
        from entsoe import EntsoePandasClient
        client = EntsoePandasClient(api_key=api_key)
        start = pd.Timestamp(start_date).tz_localize('Europe/Amsterdam')
        end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize('Europe/Amsterdam')
        
        ts = client.query_day_ahead_prices('NL', start=start, end=end)
        df_epex = ts.to_frame('EPEX_EUR_MWh')
        
        df_epex['Date_Hour'] = df_epex.index.tz_localize(None)
        df_epex = df_epex.drop_duplicates(subset='Date_Hour', keep='first')
        return df_epex
    except Exception as e:
        return str(e)
