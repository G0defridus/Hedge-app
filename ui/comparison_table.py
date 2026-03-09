"""
Vergelijkingstabel — rendert de 3 producten × 4 optimalisaties matrix.

Elke cel toont: €/MWh, hedge%, en een [Selecteer]-knop.
"""

from __future__ import annotations

import streamlit as st

import config as cfg
from core.models import CategoryScenarios, ScenarioResult
from core.scenario_engine import get_scenario
from ui import state as ui_state


def _render_cell(
    scenario: ScenarioResult | None,
    category: str,
    col,
) -> None:
    """Render één cel van de vergelijkingstabel."""
    if scenario is None:
        col.markdown("—")
        return

    if not scenario.applicable:
        col.markdown(
            '<div class="scenario-cell not-applicable">'
            '<div class="cell-price">—</div>'
            '<div class="cell-hedge">n.v.t.</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        return

    fin = scenario.financial
    res = scenario.results

    # Check of dit scenario momenteel geselecteerd is
    sel = ui_state.get_selected_scenario(category)
    is_selected = (
        sel is not None
        and sel["product"] == scenario.product
        and sel["optimization"] == scenario.optimization
    )
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
        f'<div class="cell-price">{price_str}</div>'
        f'<div class="cell-hedge">{hedge_str}</div>'
        f'<div class="cell-label">{price_label}</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Selecteer-knop
    btn_label = "✓ Geselecteerd" if is_selected else "Selecteer"
    btn_key = f"sel_{category}_{scenario.product}_{scenario.optimization}"
    if col.button(btn_label, key=btn_key, use_container_width=True):
        ui_state.set_selected_scenario(
            category, scenario.product, scenario.optimization
        )
        ui_state.set_hedge_position(scenario.position)
        st.rerun()


def render_comparison_table(cat_scenarios: CategoryScenarios) -> None:
    """Render de volledige vergelijkingstabel voor één categorie.

    Layout: 4 rijen (optimalisaties) × 3 kolommen (producten).
    """
    category = cat_scenarios.category

    # Kolomheaders
    header_cols = st.columns([1.5] + [1] * len(cfg.PRODUCTS))
    header_cols[0].markdown("**Strategie**")
    for i, prod in enumerate(cfg.PRODUCTS):
        header_cols[i + 1].markdown(f"**{prod['label']}**")
        header_cols[i + 1].caption(prod["desc"])

    st.markdown("---")

    # Rijen: één per optimalisatie
    for opt in cfg.OPTIMIZATIONS:
        row_cols = st.columns([1.5] + [1] * len(cfg.PRODUCTS))
        row_cols[0].markdown(f"**{opt['label']}**")

        for j, prod in enumerate(cfg.PRODUCTS):
            scenario = get_scenario(cat_scenarios, prod["key"], opt["key"])
            _render_cell(scenario, category, row_cols[j + 1])

        st.markdown("<br>", unsafe_allow_html=True)
