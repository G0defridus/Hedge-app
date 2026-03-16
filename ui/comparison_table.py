"""
Optimalisatie-kaarten — rendert 4 horizontale kaarten voor de strategiekeuze.

Elke kaart toont: label, beschrijving, prijs/MWh, hedge%, en een selecteer-knop.
Geselecteerde kaart krijgt een goud-rand.
"""

from __future__ import annotations

import streamlit as st

import config as cfg
from core.models import CategoryScenarios, ScenarioResult
from core.scenario_engine import get_scenario
from ui import state as ui_state


def _render_card(
    scenario: ScenarioResult,
    opt_cfg: dict,
    category: str,
    col,
    *,
    key_prefix: str = "",
    selectable: bool = True,
) -> None:
    """Render één optimalisatie-kaart."""
    fin = scenario.financial
    res = scenario.results

    # Check of dit scenario geselecteerd is
    if selectable:
        sel = ui_state.get_selected_scenario(category)
        is_selected = (
            sel is not None
            and sel["optimization"] == scenario.optimization
        )
    else:
        is_selected = False

    css_class = "scenario-cell selected" if is_selected else "scenario-cell"

    # Prijs weergave
    avg = fin.avg_cost_per_mwh
    if avg < 0:
        price_str = f"€ {abs(avg):.2f}"
        price_label = "opbrengst/MWh"
    else:
        price_str = f"€ {avg:.2f}"
        price_label = "kosten/MWh"

    hedge_str = f"{res.hedge_efficiency_pct:.0f}% afgedekt"

    col.markdown(
        f'<div class="{css_class}">'
        f'<div class="cell-title">{opt_cfg["label"]}</div>'
        f'<div class="cell-desc">{opt_cfg["desc"]}</div>'
        f'<div class="cell-price">{price_str}</div>'
        f'<div class="cell-hedge">{hedge_str}</div>'
        f'<div class="cell-label">{price_label}</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Selecteer-knop
    if selectable:
        btn_label = "✓ Geselecteerd" if is_selected else "Selecteer"
        btn_key = f"sel_{key_prefix}{category}_{scenario.optimization}"
        if col.button(btn_label, key=btn_key, use_container_width=True):
            ui_state.set_selected_scenario(category, scenario.optimization)
            ui_state.set_hedge_position(scenario.position)
            st.rerun()


def render_comparison_table(
    cat_scenarios: CategoryScenarios,
    *,
    key_prefix: str = "",
    selectable: bool = True,
) -> None:
    """Render 4 optimalisatie-kaarten naast elkaar.

    Parameters
    ----------
    cat_scenarios : CategoryScenarios
        De berekende scenario's (4 optimalisaties).
    key_prefix : str
        Prefix voor widget keys (bijv. "q1_" voor kwartaal-specifieke tabellen).
    selectable : bool
        Als True, toon Selecteer-knoppen en selectie-highlighting.
    """
    category = cat_scenarios.category

    cols = st.columns(len(cfg.OPTIMIZATIONS))

    for i, opt_cfg in enumerate(cfg.OPTIMIZATIONS):
        scenario = get_scenario(cat_scenarios, opt_cfg["key"])
        if scenario is not None:
            _render_card(
                scenario, opt_cfg, category, cols[i],
                key_prefix=key_prefix, selectable=selectable,
            )
        else:
            cols[i].markdown("—")
