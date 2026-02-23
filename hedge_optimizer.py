import pandas as pd
import numpy as np

def calculate_metrics(sub_df, base, peak_add, p_mw_col):
    hedge = base + (sub_df['is_peak'] * peak_add)
    prof = sub_df[p_mw_col]
    vol_prof = prof.sum() * 0.25
    
    # Over_hedge_mwh: Voor zowel producer als consumer is dit "Hedge - Profiel > 0"
    diff = hedge - prof
    over_hedge_mwh = diff[diff > 0].sum() * 0.25
    
    # Gebruik de absolute waarde van het profiel, anders faalt de strategie bij producers
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
