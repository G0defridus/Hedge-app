"""
Hedge-optimalisatie strategieën.

Twee benaderingen:
  1. Volume-gebaseerd  → find_optimal_position()
  2. Financieel         → optimize_financial()

Geen Streamlit-afhankelijkheden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from core.models import HedgePosition


# ═══════════════════════════════════════════════════════════════════════════
# Interne helpers
# ═══════════════════════════════════════════════════════════════════════════

def _calculate_over_hedge_pct(
    sub_df: pd.DataFrame,
    base_mw: float,
    peak_add_mw: float,
    profile_col: str,
) -> float:
    """Bereken het over-hedge percentage voor een gegeven Base/Peak combinatie."""
    hedge = base_mw + (sub_df["is_peak"] * peak_add_mw)
    profile = sub_df[profile_col]
    vol_profile = profile.sum() * cfg.MWH_FACTOR

    diff = hedge - profile
    over_hedge_mwh = diff[diff > 0].sum() * cfg.MWH_FACTOR

    if vol_profile == 0:
        return 0.0
    return (over_hedge_mwh / abs(vol_profile)) * 100


def _position_from_volume_pct(
    sub_df: pd.DataFrame,
    profile_col: str,
    pct: float,
) -> HedgePosition:
    """Bereken Base/Peak MW voor een gewenst volumepercentage."""
    off_peak_mean = sub_df.loc[~sub_df["is_peak"], profile_col].mean()
    peak_mean = sub_df.loc[sub_df["is_peak"], profile_col].mean()

    base = round(off_peak_mean * (pct / 100.0), 1) if not pd.isna(off_peak_mean) else 0.0
    peak_add = round((peak_mean * (pct / 100.0)) - base, 1) if not pd.isna(peak_mean) else 0.0

    return HedgePosition(base_mw=base, peak_add_mw=peak_add)


# ═══════════════════════════════════════════════════════════════════════════
# Publieke API — Volume-gebaseerd
# ═══════════════════════════════════════════════════════════════════════════

def find_optimal_position(
    sub_df: pd.DataFrame,
    profile_col: str,
    *,
    target_over_pct_limit: float | None = None,
    percent_volume_target: float | None = None,
) -> HedgePosition:
    """Zoek de optimale Base/Peak MW via volumetrische benadering.

    Parameters
    ----------
    sub_df : DataFrame
        Bevat ``profile_col`` en ``is_peak`` kolommen.
    profile_col : str
        Naam van de actieve profiel-MW kolom.
    target_over_pct_limit : float, optional
        Maximaal toegestaan over-hedge percentage (bijv. 5.0 voor "max 5% sell").
    percent_volume_target : float, optional
        Exact volumepercentage (bijv. 100 voor "100% dekking").

    Returns
    -------
    HedgePosition
    """
    # Directe volumetrische berekening
    if percent_volume_target is not None:
        return _position_from_volume_pct(sub_df, profile_col, percent_volume_target)

    # Zoek maximale inkoop met over-hedge ≤ limiet
    # Eenvoudige for-loop (vervangt oude recursie-als-loop)
    best = HedgePosition(0.0, 0.0)

    for pct in range(cfg.VOLUME_SEARCH_MAX_PCT, 0, -1):
        candidate = _position_from_volume_pct(sub_df, profile_col, float(pct))
        over_pct = _calculate_over_hedge_pct(
            sub_df, candidate.base_mw, candidate.peak_add_mw, profile_col
        )
        if target_over_pct_limit is not None and over_pct <= target_over_pct_limit:
            best = candidate
            break

    return best


# ═══════════════════════════════════════════════════════════════════════════
# Publieke API — Financieel (Grid Search)
# ═══════════════════════════════════════════════════════════════════════════

def optimize_financial(
    sub_df: pd.DataFrame,
    profile_col: str,
    price_base: float,
    price_peak: float,
    strategy: str = "least_cost",
) -> HedgePosition:
    """Zoek de financieel optimale Base/Peak via grid search.

    Parameters
    ----------
    sub_df : DataFrame
        Bevat ``profile_col``, ``is_peak`` en ``EPEX_EUR_MWh``.
    profile_col : str
        Naam van de actieve profiel-MW kolom.
    price_base, price_peak : float
        Contractprijzen in €/MWh.
    strategy : str
        ``"least_cost"`` → minimaliseer totale kosten.
        ``"value_risk"`` → minimaliseer variantie (schommelingen).

    Returns
    -------
    HedgePosition
    """
    # Geen EPEX-data? Val terug op 100% volume
    if "EPEX_EUR_MWh" not in sub_df.columns or sub_df["EPEX_EUR_MWh"].sum() == 0:
        return find_optimal_position(sub_df, profile_col, percent_volume_target=100)

    # Zoekgrenzen
    prof_min = sub_df[profile_col].min()
    prof_max = sub_df[profile_col].max()
    bound_min = min(0, prof_min * 1.5)
    bound_max = max(0, prof_max * 1.5)

    steps = cfg.GRID_SEARCH_STEPS
    b_vals = np.linspace(bound_min, bound_max, steps)
    p_vals = np.linspace(bound_min, bound_max, steps)
    B, P = np.meshgrid(b_vals, p_vals)
    B_flat = B.flatten()
    P_flat = P.flatten()

    # Vectorized data
    prof_mwh = sub_df[profile_col].values * cfg.MWH_FACTOR
    epex = sub_df["EPEX_EUR_MWh"].values
    is_peak = sub_df["is_peak"].values.astype(float)

    best_val = float("inf")
    best_b, best_p = 0.0, 0.0

    for i in range(len(B_flat)):
        b = B_flat[i]
        p = P_flat[i]

        hedge_mwh = (b + is_peak * p) * cfg.MWH_FACTOR

        over = np.maximum(0, hedge_mwh - prof_mwh)
        under = np.maximum(0, prof_mwh - hedge_mwh)

        spot_cost = np.sum(under * epex) - np.sum(over * epex)
        hedge_cost_hourly = (b * cfg.MWH_FACTOR * price_base) + (
            is_peak * p * cfg.MWH_FACTOR * price_peak
        )

        if strategy == "least_cost":
            total_cost = np.sum(hedge_cost_hourly) + spot_cost
            if total_cost < best_val:
                best_val = total_cost
                best_b, best_p = b, p

        elif strategy == "value_risk":
            hourly_cost = hedge_cost_hourly + (under * epex) - (over * epex)
            var_cost = np.var(hourly_cost)
            if var_cost < best_val:
                best_val = var_cost
                best_b, best_p = b, p

    return HedgePosition(base_mw=round(best_b, 1), peak_add_mw=round(best_p, 1))
