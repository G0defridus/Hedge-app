"""
Scenario Engine — batch-berekening van de 3×4 productvergelijkingsmatrix.

Berekent voor één categorie alle combinaties van:
  - 3 producten (Max Zekerheid, Minimale Hedge, Flex)
  - 4 optimalisaties (Laagste kosten, Minste risico, Max 5% verkoop, 100% volume)

Geen Streamlit-afhankelijkheden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from core.financial import (
    apply_hedge_columns,
    compute_financial_summary,
    compute_hedge_results,
)
from core.models import (
    CategoryScenarios,
    FinancialSummary,
    HedgePosition,
    HedgeResults,
    ScenarioResult,
)
from core.optimizer import compute_value_hedge, find_optimal_position, optimize_financial


# ═══════════════════════════════════════════════════════════════════════════
# Intern: één scenario doorrekenen
# ═══════════════════════════════════════════════════════════════════════════

def _compute_position(
    sub_df: pd.DataFrame,
    profile_col: str,
    product: str,
    optimization: str,
    price_base: float,
    price_peak: float,
) -> tuple[HedgePosition, bool]:
    """Bepaal de HedgePosition voor een product×optimalisatie combinatie.

    Returns (position, applicable).
    """
    # Flex → altijd 0 MW
    if product == "flex":
        return HedgePosition(0.0, 0.0), True

    # Minimale Hedge → vast percentage, optimalisatie heeft geen effect
    if product == "min_hedge":
        pos = find_optimal_position(
            sub_df, profile_col, percent_volume_target=cfg.MIN_HEDGE_VOLUME_PCT
        )
        # Alleen relevant voor least_cost rij (eerste), rest is herhaling
        return pos, optimization == "least_cost"

    # Max Zekerheid → alle 4 optimalisaties zijn relevant
    if optimization == "least_cost":
        return optimize_financial(sub_df, profile_col, price_base, price_peak, "least_cost"), True
    elif optimization == "value_risk":
        return compute_value_hedge(sub_df, profile_col), True
    elif optimization == "max_5pct":
        return find_optimal_position(sub_df, profile_col, target_over_pct_limit=5.0), True
    elif optimization == "100vol":
        return find_optimal_position(sub_df, profile_col, percent_volume_target=100), True

    return HedgePosition(0.0, 0.0), False


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
    product: str,
    optimization: str,
    price_base: float,
    price_peak: float,
    epex_loaded: bool,
) -> ScenarioResult:
    """Bereken één cel van de vergelijkingsmatrix."""
    position, applicable = _compute_position(
        df, profile_col, product, optimization, price_base, price_peak
    )

    if not applicable:
        return ScenarioResult(
            product=product,
            optimization=optimization,
            applicable=False,
        )

    df_calc = _apply_position_to_df(df, profile_col, position)
    results = compute_hedge_results(df_calc)
    financial = compute_financial_summary(df_calc, epex_loaded)

    return ScenarioResult(
        product=product,
        optimization=optimization,
        position=position,
        results=results,
        financial=financial,
        applicable=True,
    )


def compute_all_scenarios(
    df: pd.DataFrame,
    category: str,
    price_base: float,
    price_peak: float,
    epex_loaded: bool,
) -> CategoryScenarios:
    """Bereken de volledige 3×4 matrix voor één categorie.

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
        Bevat tot 12 ScenarioResult objecten.
    """
    profile_col = f"{category}_MW"

    # Voorbereiden: Active_Profile_MW instellen voor deze categorie
    df = df.copy()
    df["Active_Profile_MW"] = df[profile_col]

    scenarios: list[ScenarioResult] = []

    # Cache voor min_hedge (zelfde positie voor alle optimalisaties)
    min_hedge_result: ScenarioResult | None = None

    for product_cfg in cfg.PRODUCTS:
        product = product_cfg["key"]

        for opt_cfg in cfg.OPTIMIZATIONS:
            optimization = opt_cfg["key"]

            # Min hedge: hergebruik eerste berekening
            if product == "min_hedge" and min_hedge_result is not None:
                scenarios.append(
                    ScenarioResult(
                        product=product,
                        optimization=optimization,
                        position=min_hedge_result.position,
                        results=min_hedge_result.results,
                        financial=min_hedge_result.financial,
                        applicable=False,  # Toon als n.v.t. behalve eerste rij
                    )
                )
                continue

            # Flex: hergebruik na eerste berekening
            if product == "flex" and any(
                s.product == "flex" and s.applicable for s in scenarios
            ):
                first_flex = next(s for s in scenarios if s.product == "flex" and s.applicable)
                scenarios.append(
                    ScenarioResult(
                        product=product,
                        optimization=optimization,
                        position=first_flex.position,
                        results=first_flex.results,
                        financial=first_flex.financial,
                        applicable=False,
                    )
                )
                continue

            result = compute_single_scenario(
                df, "Active_Profile_MW", product, optimization,
                price_base, price_peak, epex_loaded,
            )
            scenarios.append(result)

            # Cache min_hedge resultaat
            if product == "min_hedge" and result.applicable:
                min_hedge_result = result

    return CategoryScenarios(category=category, scenarios=scenarios)


def get_scenario(
    cat_scenarios: CategoryScenarios,
    product: str,
    optimization: str,
) -> ScenarioResult | None:
    """Haal één specifiek scenario op uit CategoryScenarios."""
    for s in cat_scenarios.scenarios:
        if s.product == product and s.optimization == optimization:
            return s
    return None
