import streamlit as st
import pandas as pd
import os
import glob

@st.cache_data
def get_default_price(period="Jaar", q=None):
    b_val, p_val = 80.0, 95.0
    if period == "Kwartaal":
        if q == 1: b_val, p_val = 90.0, 110.0
        elif q == 2: b_val, p_val = 65.0, 75.0
        elif q == 3: b_val, p_val = 70.0, 80.0
        elif q == 4: b_val, p_val = 85.0, 105.0

    try:
        if period == "Jaar":
            files = glob.glob("**/*jaar*endex*.csv", recursive=True) + glob.glob("**/*endex*jaar*.csv", recursive=True)
        else:
            files = glob.glob("**/*kwartaal*endex*.csv", recursive=True) + glob.glob("**/*endex*kwartaal*.csv", recursive=True)
            
        if not files: 
            files = [f for f in os.listdir('.') if period.lower() in f.lower() and 'endex' in f.lower() and f.endswith('.csv')]
            
        if files:
            with open(files[0], 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            skip = 0
            if len(lines) > 1 and "Date" in lines[1] and "Base" in lines[1]: skip = 1
                
            df_px = pd.read_csv(files[0], sep=';', skiprows=skip)
            
            for c in df_px.columns:
                if df_px[c].dtype == object:
                    df_px[c] = df_px[c].astype(str).str.replace(',', '.')

            base_cols = [c for c in df_px.columns if 'base' in c.lower()]
            peak_cols = [c for c in df_px.columns if 'p16' in c.lower() or 'peak' in c.lower()]
            
            if base_cols and peak_cols:
                found_b, found_p = False, False
                for bc in reversed(base_cols):
                    s = pd.to_numeric(df_px[bc], errors='coerce').dropna()
                    if len(s) > 10: 
                        b_val = float(s.iloc[-1])
                        found_b = True; break
                        
                for pc in reversed(peak_cols):
                    s = pd.to_numeric(df_px[pc], errors='coerce').dropna()
                    if len(s) > 10:
                        p_val = float(s.iloc[-1])
                        found_p = True; break
                        
                if period == "Kwartaal" and found_b and found_p:
                    gem_b = (90+65+70+85)/4; gem_p = (110+75+80+105)/4
                    if q == 1: b_val *= (90/gem_b); p_val *= (110/gem_p)
                    elif q == 2: b_val *= (65/gem_b); p_val *= (75/gem_p)
                    elif q == 3: b_val *= (70/gem_b); p_val *= (80/gem_p)
                    elif q == 4: b_val *= (85/gem_b); p_val *= (105/gem_p)
                    
    except Exception: pass 
    return round(b_val, 2), round(p_val, 2)
