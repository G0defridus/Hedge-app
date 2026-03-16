"""
Scenario Engine — berekent 4 optimalisatie-strategieën voor één categorie.

Strategieën:
  - Value Hedge: prijsgewogen inkoop
  - Maximum Hedge: maximale inkoop, max 5% terugverkoop
  - Minimum Hedge: ~10% afdekking
  - Least Cost: grid search goedkoopste mix

Geen Streamlit-afhankelijkheden.
"""

from __future__ import annotations

import pandas as pd

import config as cfg
from core.financial import (
    apply_hedge_columns,
    compute_financial_summary,
    compute_hedge_results,
)
from core.models import (
    CategoryScenarios,
    HedgePosition,
    ScenarioResult,
)
from core.optimizer import compute_value_hedge, find_optimal_position, optimize_financial


# ═══════════════════════════════════════════════════════════════════════════
# Intern: één scenario doorrekenen
# ═══════════════════════════════════════════════════════════════════════════

def _compute_position(
    sub_df: pd.DataFrame,
    profile_col: str,
    optimization: str,
    price_base: float,
    price_peak: float,
) -> HedgePosition:
    """Bepaal de HedgePosition voor een optimalisatie-strategie."""
    if optimization == "value_hedge":
        return compute_value_hedge(sub_df, profile_col)
    elif optimization == "max_hedge":
        return find_optimal_position(sub_df, profile_col, target_over_pct_limit=5.0)
    elif optimization == "min_hedge":
        return find_optimal_position(
            sub_df, profile_col, percent_volume_target=cfg.MIN_HEDGE_VOLUME_PCT
        )
    elif optimization == "least_cost":
        return optimize_financial(sub_df, profile_col, price_base, price_peak, "least_cost")
    return HedgePosition(0.0, 0.0)


def _apply_position_to_df(
    df: pd.DataFrame,
    profile_col: str,
    position: HedgePosition,
) -> pd.DataFrame:
    """Pas een HedgePosition toe op een DataFrame-kopie en bereken resultaten."""
    df = df.copy()
    df["Hedge_Base_MW"] = position.base_mw
    df["Hedge_Peak_MW"] = position.peak_add_mw * df["is_peak"]
    df["Current_Hedge_MW"] = df["Hedge_Base_MW"] + df["Hedge_Peak_MW"]
    return apply_hedge_columns(df, profile_col)


# ═══════════════════════════════════════════════════════════════════════════
# Publieke API
# ═══════════════════════════════════════════════════════════════════════════

def compute_single_scenario(
    df: pd.DataFrame,
    profile_col: str,
    optimization: str,
    price_base: float,
    price_peak: float,
    epex_loaded: bool,
) -> ScenarioResult:
    """Bereken één optimalisatie-scenario."""
    position = _compute_position(df, profile_col, optimization, price_base, price_peak)
    df_calc = _apply_position_to_df(df, profile_col, position)
    results = compute_hedge_results(df_calc)
    financial = compute_financial_summary(df_calc, epex_loaded)

    return ScenarioResult(
        optimization=optimization,
        position=position,
        results=results,
        financial=financial,
    )


def compute_all_scenarios(
    df: pd.DataFrame,
    category: str,
    price_base: float,
    price_peak: float,
    epex_loaded: bool,
) -> CategoryScenarios:
    """Bereken alle 4 optimalisaties voor één categorie.

    Parameters
    ----------
    df : DataFrame
        Hoofd-DataFrame met profiel-MW kolommen, is_peak, EPEX, prijzen.
    category : str
        "Consumer", "Prosumer" of "Producer".
    price_base, price_peak : float
        Contractprijzen in €/MWh.
    epex_loaded : bool
        Of EPEX-data beschikbaar is.

    Returns
    -------
    CategoryScenarios
        Bevat 4 ScenarioResult objecten.
    """
    profile_col = f"{category}_MW"

    df = df.copy()
    df["Active_Profile_MW"] = df[profile_col]

    scenarios: list[ScenarioResult] = []

    for opt_cfg in cfg.OPTIMIZATIONS:
        result = compute_single_scenario(
            df, "Active_Profile_MW", opt_cfg["key"],
            price_base, price_peak, epex_loaded,
        )
        scenarios.append(result)

    return CategoryScenarios(category=category, scenarios=scenarios)


def get_scenario(
    cat_scenarios: CategoryScenarios,
    optimization: str,
) -> ScenarioResult | None:
    """Haal één specifiek scenario op uit CategoryScenarios."""
    for s in cat_scenarios.scenarios:
        if s.optimization == optimization:
            return s
    return None
