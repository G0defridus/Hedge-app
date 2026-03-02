import pandas as pd
import numpy as np

def calculate_metrics(sub_df, base, peak_add, p_mw_col):
    hedge = base + (sub_df['is_peak'] * peak_add)
    prof = sub_df[p_mw_col]
    vol_prof = prof.sum() * 0.25
    
    diff = hedge - prof
    over_hedge_mwh = diff[diff > 0].sum() * 0.25
    
    over_pct = (over_hedge_mwh / abs(vol_prof) * 100) if vol_prof != 0 else 0
    return over_pct

def find_optimal_mw(sub_df, p_mw_col, target_over_pct_limit=None, percent_volume_target=None):
    if percent_volume_target is not None:
        off_peak_mean = sub_df.loc[~sub_df['is_peak'], p_mw_col].mean()
        peak_mean = sub_df.loc[sub_df['is_peak'], p_mw_col].mean()
        b = round(off_peak_mean * (percent_volume_target / 100.0), 1) if not pd.isna(off_peak_mean) else 0.0
        p_add = round((peak_mean * (percent_volume_target / 100.0)) - b, 1) if not pd.isna(peak_mean) else 0.0
        return b, p_add

    best_b, best_p = 0.0, 0.0
    for pct in range(150, 0, -1): 
        b_try, p_try = find_optimal_mw(sub_df, p_mw_col, percent_volume_target=pct)
        over_pct = calculate_metrics(sub_df, b_try, p_try, p_mw_col)
        if target_over_pct_limit is not None and over_pct <= target_over_pct_limit:
            best_b, best_p = b_try, p_try
            break 
    return best_b, best_p

def optimize_advanced(sub_df, p_mw_col, price_base, price_peak, strategy="least_cost"):
    """
    Slim algoritme dat duizenden combinaties test om het financiële optimum te vinden.
    """
    if 'EPEX_EUR_MWh' not in sub_df.columns or sub_df['EPEX_EUR_MWh'].sum() == 0:
        return find_optimal_mw(sub_df, p_mw_col, percent_volume_target=100)
        
    prof_min = sub_df[p_mw_col].min()
    prof_max = sub_df[p_mw_col].max()
    
    # We zoeken in zowel positieve als negatieve richting (voor producers)
    bound_min = min(0, prof_min * 1.5)
    bound_max = max(0, prof_max * 1.5)
    
    steps = 40 # 40x40 = 1600 scenario's per kwartaal berekenen
    b_vals = np.linspace(bound_min, bound_max, steps)
    p_vals = np.linspace(bound_min, bound_max, steps)
    B, P = np.meshgrid(b_vals, p_vals)
    B_flat = B.flatten()
    P_flat = P.flatten()
    
    prof_mwh = sub_df[p_mw_col].values * 0.25
    epex = sub_df['EPEX_EUR_MWh'].values
    is_peak = sub_df['is_peak'].values
    
    best_val = float('inf')
    best_b, best_p = 0.0, 0.0
    
    # Razendsnelle matrix-loop
    for i in range(len(B_flat)):
        b = B_flat[i]
        p = P_flat[i]
        hedge_mwh = (b + is_peak * p) * 0.25
        
        over = np.maximum(0, hedge_mwh - prof_mwh)
        under = np.maximum(0, prof_mwh - hedge_mwh)
        
        spot_cost = np.sum(under * epex) - np.sum(over * epex)
        hedge_cost_hourly = (b * 0.25 * price_base) + (is_peak * p * 0.25 * price_peak)
        
        if strategy == "least_cost":
            # Doel: Wat levert netto de laagste integrale kosten op?
            total_cost = np.sum(hedge_cost_hourly) + spot_cost
            if total_cost < best_val:
                best_val = total_cost
                best_b, best_p = b, p
                
        elif strategy == "value_risk":
            # Doel: Value-at-Risk minimaliseren (Zorgen dat de kosten/opbrengsten per uur zo vlak mogelijk zijn)
            hourly_cost = hedge_cost_hourly + (under * epex) - (over * epex)
            var_cost = np.var(hourly_cost)
            if var_cost < best_val:
                best_val = var_cost
                best_b, best_p = b, p

    return round(best_b, 1), round(best_p, 1)
