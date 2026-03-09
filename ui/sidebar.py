"""
Sidebar rendering — alle zijbalk-componenten op één plek.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from core.models import HedgePosition, Period
from core.optimizer import find_optimal_position, optimize_financial
from core.pricing import get_default_price
from ui import state as ui_state


# ═══════════════════════════════════════════════════════════════════════════
# 1. Upload
# ═══════════════════════════════════════════════════════════════════════════

def render_upload_section(container) -> tuple[Optional[object], str]:
    """Toon upload-widget en input mode selector.

    Returns
    -------
    uploaded_file : UploadedFile | None
    input_mode : str
    """
    has_file = st.session_state.get("file_uploader_key") is not None
    with container:
        st.header("1. Upload je data _" if not has_file else "Ander bestand _")
        input_mode = st.radio(
            "Kies het type bestand",
            ["Ruwe Aansluitingen (CSV)", "Reeds Geaggregeerd (CSV)"],
            key="input_mode",
        )
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="file_uploader_key")
    return uploaded_file, input_mode


# ═══════════════════════════════════════════════════════════════════════════
# 2. Basis-instellingen
# ═══════════════════════════════════════════════════════════════════════════

def render_settings(container) -> tuple[str, str]:
    """Toon profiel- en periodekeuze.

    Returns
    -------
    profile_choice : str
    strategy_period : str
    """
    with container:
        st.header("2. Basisinstellingen _")
        profile_choice = st.selectbox(
            "Welk profiel bekijken we?", cfg.PROFILE_CHOICES
        )
        strategy_period = st.radio("Contractperiode", ["Per Jaar", "Per Kwartaal"])
    return profile_choice, strategy_period


# ═══════════════════════════════════════════════════════════════════════════
# 3. Scenario's
# ═══════════════════════════════════════════════════════════════════════════

def render_scenarios(container) -> tuple[float, float]:
    """Toon scenario-sliders.

    Returns
    -------
    vol_multiplier, epex_multiplier : float
    """
    with container:
        st.markdown("---")
        st.header("3. Speel met scenario's _")
        vol_pct = st.slider(
            "Verwachte groei of zon",
            min_value=-50, max_value=50, value=0, step=5, format="%d%%",
        )
        epex_pct = st.slider(
            "Spotprijzen (EPEX)",
            min_value=-100, max_value=200, value=0, step=10, format="%d%%",
        )
    return vol_pct / 100.0, epex_pct / 100.0


# ═══════════════════════════════════════════════════════════════════════════
# 4. Contractprijzen
# ═══════════════════════════════════════════════════════════════════════════

def render_prices(
    container,
    df: pd.DataFrame,
    strategy_period: str,
) -> pd.DataFrame:
    """Toon prijs-inputs en schrijf Price_Base / Price_Peak naar het DataFrame.

    Returns het DataFrame met prijskolommen.
    """
    df = df.copy()
    df["Price_Base"] = 0.0
    df["Price_Peak"] = 0.0

    with container:
        st.markdown("---")
        st.subheader("4. Jouw contractprijzen _")

        if strategy_period == "Per Jaar":
            cp1, cp2 = st.columns(2)
            def_b, def_p = get_default_price("Jaar")
            pr_b = cp1.number_input("Base Prijs (€/MWh)", value=def_b, step=1.0)
            pr_p = cp2.number_input("Peak Prijs (€/MWh)", value=def_p, step=1.0)
            df["Price_Base"] = pr_b
            df["Price_Peak"] = pr_p
        else:
            for q in [1, 2, 3, 4]:
                st.markdown(f"**Prijzen Q{q}**")
                cp1, cp2 = st.columns(2)
                def_b, def_p = get_default_price("Kwartaal", q)
                pr_b = cp1.number_input(
                    f"Q{q} Base", value=def_b, step=1.0, key=f"pr_b_q{q}"
                )
                pr_p = cp2.number_input(
                    f"Q{q} Peak", value=def_p, step=1.0, key=f"pr_p_q{q}"
                )
                q_mask = df["Quarter"] == q
                df.loc[q_mask, "Price_Base"] = pr_b
                df.loc[q_mask, "Price_Peak"] = pr_p

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5. Strategie-knoppen
# ═══════════════════════════════════════════════════════════════════════════

def _build_periods(
    df: pd.DataFrame, strategy_period: str
) -> list[tuple[int | None, pd.DataFrame]]:
    """Bouw lijst van (quarter, sub_df) tuples — elimineert Jaar/Kwartaal branching."""
    if strategy_period == "Per Jaar":
        return [(None, df)]
    return [(q, df[df["Quarter"] == q]) for q in [1, 2, 3, 4]]


def render_strategy_buttons(
    container,
    df: pd.DataFrame,
    p_mw_col: str,
    strategy_period: str,
) -> None:
    """Toon de strategie-knoppen en pas state aan bij klik."""

    def _apply_strategy(strat_name: str, custom_pct: float | None = None) -> None:
        periods = _build_periods(df, strategy_period)
        for quarter, sub_df in periods:
            pb = sub_df["Price_Base"].iloc[0]
            pp = sub_df["Price_Peak"].iloc[0]

            if strat_name in ("least_cost", "value_risk"):
                pos = optimize_financial(sub_df, p_mw_col, pb, pp, strategy=strat_name)
            elif strat_name == "5%_sell":
                pos = find_optimal_position(sub_df, p_mw_col, target_over_pct_limit=5.0)
            elif strat_name == "10%_cov":
                pos = find_optimal_position(sub_df, p_mw_col, percent_volume_target=10)
            elif strat_name == "100%_cov":
                pos = find_optimal_position(sub_df, p_mw_col, percent_volume_target=100)
            elif strat_name == "custom_cov":
                pos = find_optimal_position(
                    sub_df, p_mw_col, percent_volume_target=custom_pct
                )
            else:
                pos = HedgePosition()

            ui_state.set_hedge_position(pos, quarter)

    def _on_custom_change() -> None:
        _apply_strategy("custom_cov", custom_pct=st.session_state.custom_hedge_pct)

    with container:
        st.markdown("---")
        st.header("5. Kies je aanpak _")

        st.markdown("**Slimme Optimalisatie (Algoritmes):**")
        s1, s2 = st.columns(2)
        if s1.button(
            "📉 Zoek Laagste Kostprijs",
            help="Het model test duizenden combinaties om de laagste integrale prijs te vinden (inclusief spotmarkt).",
        ):
            _apply_strategy("least_cost")
        if s2.button(
            "⚖️ Zoek Minste Risico (Value Hedge)",
            help="Minimaliseert de financiële schommelingen en dekt piekprijzen agressiever af.",
        ):
            _apply_strategy("value_risk")

        st.markdown("**Volume-gebaseerd:**")
        v1, v2, v3 = st.columns(3)
        if v1.button("100% Volume"):
            _apply_strategy("100%_cov")
        if v2.button("Subtiel (10%)"):
            _apply_strategy("10%_cov")
        if v3.button("Max 5% over"):
            _apply_strategy("5%_sell")

        st.slider(
            "Of kies exact percentage",
            min_value=0, max_value=150, value=100, step=1,
            key="custom_hedge_pct",
            on_change=_on_custom_change,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 6. Finetunen sliders
# ═══════════════════════════════════════════════════════════════════════════

def render_hedge_sliders(
    container,
    df: pd.DataFrame,
    p_mw_col: str,
    strategy_period: str,
) -> pd.DataFrame:
    """Toon de Base/Peak MW sliders en schrijf hedge-kolommen naar het DataFrame.

    Returns het DataFrame met Hedge_Base_MW, Hedge_Peak_MW, Current_Hedge_MW.
    """
    df = df.copy()
    df["Hedge_Base_MW"] = 0.0
    df["Hedge_Peak_MW"] = 0.0

    curr_min = df[p_mw_col].min()
    curr_max = df[p_mw_col].max()
    slider_min = float(np.floor(curr_min * 1.5 - 1))
    slider_max = float(np.ceil(curr_max * 1.5 + 1))
    if slider_max < slider_min:
        slider_max = slider_min + 10.0

    with container:
        st.markdown("---")
        st.subheader("6. Finetunen in MW _")

        periods = _build_periods(df, strategy_period)

        for quarter, sub_df in periods:
            suffix = "yr" if quarter is None else f"q{quarter}"
            b_key = f"slider_b_{suffix}"
            p_key = f"slider_p_{suffix}"

            # Initialiseer op 100% als er nog geen waarde is
            if b_key not in st.session_state or st.session_state[b_key] == 0.0:
                default_pos = find_optimal_position(
                    sub_df, p_mw_col, percent_volume_target=100
                )
                st.session_state[b_key] = float(default_pos.base_mw)
                st.session_state[p_key] = float(default_pos.peak_add_mw)

            if quarter is not None:
                st.markdown(f"**Kwartaal {quarter}**")

            if quarter is None:
                b_val = st.slider(
                    "Base (Jaar)", slider_min, slider_max, key=b_key, step=0.1
                )
                p_val = st.slider(
                    "Peak (Jaar)", slider_min, slider_max, key=p_key, step=0.1
                )
                df["Hedge_Base_MW"] = b_val
                df["Hedge_Peak_MW"] = p_val * df["is_peak"]
            else:
                sc1, sc2 = st.columns(2)
                b_val = sc1.slider(
                    f"Q{quarter} Base", slider_min, slider_max, key=b_key, step=0.1
                )
                p_val = sc2.slider(
                    f"Q{quarter} Peak", slider_min, slider_max, key=p_key, step=0.1
                )
                q_mask = df["Quarter"] == quarter
                df.loc[q_mask, "Hedge_Base_MW"] = b_val
                df.loc[q_mask, "Hedge_Peak_MW"] = p_val * df.loc[q_mask, "is_peak"]

    df["Current_Hedge_MW"] = df["Hedge_Base_MW"] + df["Hedge_Peak_MW"]
    return df
