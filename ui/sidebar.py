"""
Sidebar rendering — twee fases: upload en configuratie.

Fase 1 (geen data): upload-widget + bestandstype selector
Fase 2 (data geladen): reset-knop, categorie, periode, prijzen, scenario-sliders
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import streamlit as st

import config as cfg
from core.pricing import contract_year_from_df, get_default_price, get_marktdata_source


# ═══════════════════════════════════════════════════════════════════════════
# Fase 1: Upload
# ═══════════════════════════════════════════════════════════════════════════

def render_upload_sidebar() -> tuple[Optional[object], str]:
    """Upload-fase: toon upload-widget en input mode selector in de sidebar.

    Returns
    -------
    uploaded_file : UploadedFile | None
    input_mode : str
    """
    st.sidebar.header("Upload je data _")

    input_mode = st.sidebar.radio(
        "Kies het type bestand",
        ["Ruwe Aansluitingen (CSV)", "Reeds Geaggregeerd (CSV)"],
        key="input_mode",
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV", type=["csv"], key="file_uploader_key"
    )

    st.sidebar.info("Upload een CSV om te starten.")

    return uploaded_file, input_mode


# ═══════════════════════════════════════════════════════════════════════════
# Fase 2: Configuratie
# ═══════════════════════════════════════════════════════════════════════════

def render_config_sidebar(df: pd.DataFrame) -> dict[str, Any]:
    """Config-fase: categorie, periode, prijzen, scenario-sliders.

    Parameters
    ----------
    df : DataFrame
        Het voorbereide DataFrame (uit session_state).

    Returns
    -------
    dict
        Keys: category, strategy_period, vol_multiplier, epex_multiplier, df
        (df is verrijkt met Price_Base en Price_Peak kolommen)
    """
    # ─── Reset-knop ─────────────────────────────────────────────────
    if st.sidebar.button("Opnieuw beginnen", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.sidebar.markdown("---")

    # ─── Categorie ──────────────────────────────────────────────────
    available_cats = [c for c in cfg.CATEGORY_CHOICES if f"{c}_MW" in df.columns]
    if not available_cats:
        available_cats = ["Consumer"]

    # Zorg dat huidige selectie geldig is
    current = st.session_state.get("selected_category", available_cats[0])
    if current not in available_cats:
        st.session_state["selected_category"] = available_cats[0]

    category = st.sidebar.radio(
        "Categorie",
        available_cats,
        key="selected_category",
    )

    st.sidebar.markdown("---")

    # ─── Contractperiode ────────────────────────────────────────────
    strategy_period = st.sidebar.radio(
        "Contractperiode",
        ["Per Jaar", "Per Kwartaal"],
        key="strategy_period",
        horizontal=True,
    )

    st.sidebar.markdown("---")

    # ─── Contractprijzen ────────────────────────────────────────────
    st.sidebar.markdown("**Contractprijzen**")
    df = df.copy()
    df["Price_Base"] = 0.0
    df["Price_Peak"] = 0.0

    cal_year = contract_year_from_df(df)

    if strategy_period == "Per Jaar":
        def_b, def_p = get_default_price("Jaar", contract_year=cal_year)
        c1, c2 = st.sidebar.columns(2)
        pr_b = c1.number_input(
            "Base €/MWh", value=def_b, step=1.0, key="price_base_yr"
        )
        pr_p = c2.number_input(
            "Peak €/MWh", value=def_p, step=1.0, key="price_peak_yr"
        )
        df["Price_Base"] = pr_b
        df["Price_Peak"] = pr_p
    else:
        for q in [1, 2, 3, 4]:
            def_b, def_p = get_default_price("Kwartaal", q, contract_year=cal_year)
            c1, c2 = st.sidebar.columns(2)
            pr_b = c1.number_input(
                f"Q{q} Base", value=def_b, step=1.0, key=f"pr_b_q{q}"
            )
            pr_p = c2.number_input(
                f"Q{q} Peak", value=def_p, step=1.0, key=f"pr_p_q{q}"
            )
            q_mask = df["Quarter"] == q
            df.loc[q_mask, "Price_Base"] = pr_b
            df.loc[q_mask, "Price_Peak"] = pr_p

    # Bron-indicator
    source = get_marktdata_source()
    if source:
        st.sidebar.caption(
            f"Bron: {source['contract']} — "
            f"Base €{source['base_value']:.2f} / Peak €{source['peak_value']:.2f} "
            f"({source['trade_date']})"
        )

    st.sidebar.markdown("---")

    # ─── Scenario-sliders ───────────────────────────────────────────
    st.sidebar.markdown("**Scenario's**")
    vol_pct = st.sidebar.slider(
        "Verwachte groei of zon",
        min_value=-50, max_value=50, value=0, step=5, format="%d%%",
    )
    epex_pct = st.sidebar.slider(
        "Spotprijzen (EPEX)",
        min_value=-100, max_value=200, value=0, step=10, format="%d%%",
    )

    return {
        "category": category,
        "strategy_period": strategy_period,
        "vol_multiplier": vol_pct / 100.0,
        "epex_multiplier": epex_pct / 100.0,
        "df": df,
    }
