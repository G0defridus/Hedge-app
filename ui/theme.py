"""
Censo huisstijl — CSS, Altair chart helpers en herbruikbare UI-elementen.
"""

from __future__ import annotations

import altair as alt
import streamlit as st

import config as cfg

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CENSO_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend+Deca:wght@400;500;700&family=Montserrat:ital,wght@1,400;1,500;1,700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Lexend Deca', sans-serif !important;
}}
h1, h2, h3, h4 {{
    font-family: 'Lexend Deca', sans-serif !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
}}
/* Knoppen in Censo Gold */
.stButton>button {{
    background-color: {cfg.COLORS['gold']} !important;
    color: {cfg.COLORS['black']} !important;
    border: none !important;
    border-radius: 4px;
    font-weight: 500;
}}
.stButton>button:hover {{
    background-color: {cfg.COLORS['gold_hover']} !important;
}}
/* Tabbladen styling */
.stTabs [data-baseweb="tab-list"] {{ gap: 2rem; }}
.stTabs [data-baseweb="tab"] {{
    height: 3rem;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 0px;
    font-family: 'Lexend Deca', sans-serif;
    font-weight: 500;
}}
.stTabs [aria-selected="true"] {{ border-bottom: 3px solid {cfg.COLORS['ruby']} !important; }}

/* ─── Vergelijkingstabel cellen ─── */
div[data-testid="stVerticalBlock"] .scenario-cell {{
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    transition: border-color 0.2s, box-shadow 0.2s;
    background: {cfg.COLORS['white']};
    min-height: 120px;
}}
div[data-testid="stVerticalBlock"] .scenario-cell:hover {{
    border-color: {cfg.COLORS['gold']};
    box-shadow: 0 2px 8px rgba(250, 181, 23, 0.25);
}}
div[data-testid="stVerticalBlock"] .scenario-cell.selected {{
    border-color: {cfg.COLORS['gold']};
    box-shadow: 0 0 0 2px {cfg.COLORS['gold']};
}}
div[data-testid="stVerticalBlock"] .scenario-cell.not-applicable {{
    opacity: 0.45;
    pointer-events: none;
}}
.scenario-cell .cell-price {{
    font-size: 1.4rem;
    font-weight: 700;
    color: {cfg.COLORS['black']};
    margin: 0.2rem 0;
}}
.scenario-cell .cell-hedge {{
    font-size: 0.95rem;
    color: {cfg.COLORS['dark_grey']};
}}
.scenario-cell .cell-label {{
    font-size: 0.8rem;
    color: {cfg.COLORS['grey']};
    margin-top: 0.3rem;
}}
/* Streepje onder categorie sub-tabs */
.stTabs .stTabs [aria-selected="true"] {{
    border-bottom-color: {cfg.COLORS['gold']} !important;
}}
</style>
"""


def apply_censo_style() -> None:
    """Inject Censo CSS in de Streamlit pagina."""
    st.markdown(CENSO_CSS, unsafe_allow_html=True)


def render_hero_title() -> None:
    """Toon de hoofdtitel met Censo styling."""
    st.markdown(
        f"""
        <div style="font-size: 2.8rem; font-weight: 700; margin-bottom: 1rem;
                     margin-top: -1rem; font-family: 'Lexend Deca', sans-serif;">
            De energie-strategie.
            <span style="color: {cfg.COLORS['ruby']};">Maar dan simpel</span>
            <span style="color: {cfg.COLORS['gold']};">_</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_how_it_works() -> None:
    """Toon het uitklapbare help-menu."""
    with st.expander("Hoe het werkt _", expanded=False):
        st.markdown(
            "**Samen zorgen we dat het goed voelt.**\n\n"
            "We analyseren jouw data en berekenen direct de impact op de spotmarkt. "
            "Nieuw in deze versie is de intelligente solver: laat het wiskundige model "
            "berekenen wat jouw absoluut goedkoopste (of minst risicovolle) "
            "inkoopstrategie is."
        )


# ---------------------------------------------------------------------------
# Altair helpers
# ---------------------------------------------------------------------------

def censo_color_scale(
    domain: list[str],
    colors: list[str] | None = None,
) -> alt.Scale:
    """Bouw een Altair kleurschaal met Censo-kleuren.

    Parameters
    ----------
    domain : list[str]
        De categorieën.
    colors : list[str], optional
        Hex-kleurcodes per categorie.  Standaard: grey, ruby, gold.
    """
    if colors is None:
        palette = [
            cfg.COLORS["grey"],
            cfg.COLORS["ruby"],
            cfg.COLORS["gold"],
        ]
        colors = (palette * ((len(domain) // len(palette)) + 1))[: len(domain)]

    return alt.Scale(domain=domain, range=colors)


# Standaard kwartaaloverzicht formaat
QUARTERLY_FORMAT = {
    "Volume (MWh)": "{:,.0f}",
    "Afgedekt (%)": "{:.1f}%",
    "Verkocht (MWh)": "{:,.0f}",
    "Ingekocht (MWh)": "{:,.0f}",
    "Totale kosten (€)": "€ {:,.0f}",
    "Jouw prijs (€/MWh)": "€ {:.2f}",
    "Spot resultaat (€)": "€ {:,.0f}",
}
