"""
Financiële berekeningen — kosten, opbrengsten, KPI's.

Puur functioneel, geen Streamlit-afhankelijkheden.
Alle berekeningen werken op het hoofd-DataFrame nadat hedge-kolommen zijn toegepast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from core.models import FinancialSummary, HedgeResults


def apply_hedge_columns(df: pd.DataFrame, profile_col: str) -> pd.DataFrame:
    """Voeg alle berekende kolommen toe aan het DataFrame.

    Verwacht dat de volgende kolommen bestaan:
        - ``profile_col`` (actief profiel in MW)
        - ``Current_Hedge_MW``, ``Hedge_Base_MW``, ``Hedge_Peak_MW``
        - ``Price_Base``, ``Price_Peak``
        - ``EPEX_EUR_MWh``

    Returns een kopie met de berekende kolommen.
    """
    df = df.copy()

    df["Profile_MWh"] = df[profile_col] * cfg.MWH_FACTOR
    df["Hedge_MWh"] = df["Current_Hedge_MW"] * cfg.MWH_FACTOR

    df["Over_Hedge_MWh"] = np.maximum(0, df["Hedge_MWh"] - df["Profile_MWh"])
    df["Under_Hedge_MWh"] = np.maximum(0, df["Profile_MWh"] - df["Hedge_MWh"])

    df["Used_Hedge_MWh_Abs"] = np.where(
        np.sign(df["Profile_MWh"]) == np.sign(df["Hedge_MWh"]),
        np.minimum(df["Profile_MWh"].abs(), df["Hedge_MWh"].abs()),
        0.0,
    )

    # Kosten inkoopblokken
    df["Cost_Hedge_Base_EUR"] = (df["Hedge_Base_MW"] * cfg.MWH_FACTOR) * df["Price_Base"]
    df["Cost_Hedge_Peak_EUR"] = (df["Hedge_Peak_MW"] * cfg.MWH_FACTOR) * df["Price_Peak"]
    df["Cost_Hedge_Total_EUR"] = df["Cost_Hedge_Base_EUR"] + df["Cost_Hedge_Peak_EUR"]

    # Spotmarkt
    has_epex = df["EPEX_EUR_MWh"].sum() != 0
    if has_epex:
        df["Cost_Buy_EUR"] = df["Under_Hedge_MWh"] * df["EPEX_EUR_MWh"]
        df["Rev_Sell_EUR"] = df["Over_Hedge_MWh"] * df["EPEX_EUR_MWh"]
        df["Net_Spot_EUR"] = df["Rev_Sell_EUR"] - df["Cost_Buy_EUR"]
    else:
        df["Cost_Buy_EUR"] = 0.0
        df["Rev_Sell_EUR"] = 0.0
        df["Net_Spot_EUR"] = 0.0

    return df


def compute_hedge_results(df: pd.DataFrame) -> HedgeResults:
    """Bereken de volumetrische resultaten uit het (berekende) DataFrame."""
    total_prof = df["Profile_MWh"].sum()
    total_prof_abs = abs(total_prof)
    denom = total_prof_abs if total_prof_abs != 0 else 1.0

    total_hedge_abs = df["Hedge_MWh"].abs().sum()
    total_over = df["Over_Hedge_MWh"].sum()
    total_under = df["Under_Hedge_MWh"].sum()
    used_hedge = df["Used_Hedge_MWh_Abs"].sum()
    hedge_eff = (used_hedge / denom) * 100

    return HedgeResults(
        profile_mwh=total_prof,
        hedge_mwh=total_hedge_abs,
        over_hedge_mwh=total_over,
        under_hedge_mwh=total_under,
        used_hedge_mwh=used_hedge,
        hedge_efficiency_pct=hedge_eff,
    )


def compute_financial_summary(
    df: pd.DataFrame,
    epex_loaded: bool,
) -> FinancialSummary:
    """Bereken het volledige financiële overzicht."""
    total_prof = df["Profile_MWh"].sum()
    total_prof_abs = abs(total_prof)
    denom = total_prof_abs if total_prof_abs != 0 else 1.0

    tot_hedge_eur = df["Cost_Hedge_Total_EUR"].sum()
    net_spot_eur = df["Net_Spot_EUR"].sum()
    tot_energy_cost = tot_hedge_eur - net_spot_eur
    avg_cost = tot_energy_cost / denom if denom > 0 else 0

    total_over = df["Over_Hedge_MWh"].sum()
    total_under = df["Under_Hedge_MWh"].sum()
    used_hedge = df["Used_Hedge_MWh_Abs"].sum()

    # Unit economics
    cost_per_used = tot_hedge_eur / used_hedge if used_hedge > 0 else 0
    avg_spot_buy = df["Cost_Buy_EUR"].sum() / total_under if total_under > 0 else 0
    avg_spot_sell = df["Rev_Sell_EUR"].sum() / total_over if total_over > 0 else 0

    avg_epex_base = df["EPEX_EUR_MWh"].mean()
    capture_price = (
        (df["Profile_MWh"].abs() * df["EPEX_EUR_MWh"]).sum() / denom
        if denom > 0
        else 0
    )
    capture_diff = capture_price - avg_epex_base

    return FinancialSummary(
        total_hedge_cost=tot_hedge_eur,
        net_spot_result=net_spot_eur,
        total_energy_cost=tot_energy_cost,
        avg_cost_per_mwh=avg_cost,
        cost_per_used_mwh=cost_per_used,
        capture_price=capture_price,
        capture_diff=capture_diff,
        avg_spot_buy=avg_spot_buy,
        avg_spot_sell=avg_spot_sell,
    )


def compute_quarterly_table(
    df: pd.DataFrame,
    epex_loaded: bool,
) -> pd.DataFrame:
    """Genereer het kwartaaloverzicht als DataFrame."""

    def _quarter_stats(x: pd.DataFrame) -> pd.Series:
        prof_abs = x["Profile_MWh"].abs().sum()
        prof_sum = x["Profile_MWh"].sum()
        denom = prof_abs if prof_sum != 0 else 1.0

        hedge_cost = x["Cost_Hedge_Total_EUR"].sum()
        net_spot = x["Net_Spot_EUR"].sum() if epex_loaded else 0
        total_cost = hedge_cost - net_spot if epex_loaded else hedge_cost
        price = total_cost / denom if prof_sum != 0 else 0

        return pd.Series(
            {
                "Volume (MWh)": prof_abs,
                "Afgedekt (%)": (x["Used_Hedge_MWh_Abs"].sum() / denom * 100)
                if prof_sum != 0
                else 0,
                "Verkocht (MWh)": x["Over_Hedge_MWh"].sum(),
                "Ingekocht (MWh)": x["Under_Hedge_MWh"].sum(),
                "Spot resultaat (€)": net_spot,
                "Totale kosten (€)": total_cost,
                "Jouw prijs (€/MWh)": price,
            }
        )

    q_stats = df.groupby("Quarter").apply(_quarter_stats)

    if not epex_loaded:
        q_stats = q_stats.drop(columns=["Spot resultaat (€)"])

    return q_stats
