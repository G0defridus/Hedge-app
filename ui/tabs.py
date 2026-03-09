"""
Tab renderers — 3 hoofdtabs + 4 resultaat-subtabs.

Tab 1: Data & Categorisatie
Tab 2: Strategieën (vergelijkingstabel)
Tab 3: Resultaten (finetuning + visualisaties)
"""

from __future__ import annotations

import numpy as np
import altair as alt
import pandas as pd
import streamlit as st

import config as cfg
from core.financial import (
    apply_hedge_columns,
    compute_financial_summary,
    compute_hedge_results,
    compute_quarterly_table,
)
from core.models import FinancialSummary, HedgePosition, HedgeResults
from core.optimizer import find_optimal_position
from core.pricing import get_default_price
from core.scenario_engine import compute_all_scenarios
from ui import state as ui_state
from ui.comparison_table import render_comparison_table
from ui.theme import censo_color_scale, QUARTERLY_FORMAT


# ═══════════════════════════════════════════════════════════════════════════
# Tab 1: Data & Categorisatie
# ═══════════════════════════════════════════════════════════════════════════

def render_tab_data(df: pd.DataFrame) -> dict:
    """Configuratie-tab: profiel, prijzen, scenario's.

    Returns
    -------
    dict met keys: profile_choice, strategy_period, vol_multiplier,
                   epex_multiplier, df (verrijkt met prijzen + scenario-sliders)
    """
    st.markdown("### Configureer je analyse _")

    # --- Profiel & Periode ---
    col_prof, col_period = st.columns(2)
    with col_prof:
        profile_choice = st.selectbox(
            "Welk profiel bekijken we?", cfg.PROFILE_CHOICES
        )
    with col_period:
        strategy_period = st.radio(
            "Contractperiode", ["Per Jaar", "Per Kwartaal"], horizontal=True
        )

    st.markdown("---")

    # --- Contractprijzen ---
    st.markdown("#### Contractprijzen _")
    df = df.copy()
    df["Price_Base"] = 0.0
    df["Price_Peak"] = 0.0

    if strategy_period == "Per Jaar":
        cp1, cp2 = st.columns(2)
        def_b, def_p = get_default_price("Jaar")
        pr_b = cp1.number_input("Base Prijs (€/MWh)", value=def_b, step=1.0)
        pr_p = cp2.number_input("Peak Prijs (€/MWh)", value=def_p, step=1.0)
        df["Price_Base"] = pr_b
        df["Price_Peak"] = pr_p
    else:
        price_cols = st.columns(4)
        for idx, q in enumerate([1, 2, 3, 4]):
            with price_cols[idx]:
                st.markdown(f"**Q{q}**")
                def_b, def_p = get_default_price("Kwartaal", q)
                pr_b = st.number_input(
                    f"Base", value=def_b, step=1.0, key=f"pr_b_q{q}"
                )
                pr_p = st.number_input(
                    f"Peak", value=def_p, step=1.0, key=f"pr_p_q{q}"
                )
                q_mask = df["Quarter"] == q
                df.loc[q_mask, "Price_Base"] = pr_b
                df.loc[q_mask, "Price_Peak"] = pr_p

    st.markdown("---")

    # --- Scenario-sliders ---
    st.markdown("#### Speel met scenario's _")
    sc1, sc2 = st.columns(2)
    with sc1:
        vol_pct = st.slider(
            "Verwachte groei of zon",
            min_value=-50, max_value=50, value=0, step=5, format="%d%%",
        )
    with sc2:
        epex_pct = st.slider(
            "Spotprijzen (EPEX)",
            min_value=-100, max_value=200, value=0, step=10, format="%d%%",
        )

    return {
        "profile_choice": profile_choice,
        "strategy_period": strategy_period,
        "vol_multiplier": vol_pct / 100.0,
        "epex_multiplier": epex_pct / 100.0,
        "df": df,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tab 2: Strategieën
# ═══════════════════════════════════════════════════════════════════════════

def render_tab_strategies(
    df: pd.DataFrame,
    available_categories: list[str],
    epex_loaded: bool,
) -> None:
    """Vergelijkingstabel per categorie — het middelpunt van de app."""
    st.markdown("### Vergelijk strategieën _")
    st.info(
        "Hieronder zie je per categorie alle combinaties van inkoopproducten "
        "en optimalisatiestrategieën. Klik **Selecteer** om een strategie "
        "over te nemen naar het Resultaten-tabblad."
    )

    if not available_categories:
        st.warning("Geen categorieën beschikbaar. Upload data in Tab 1.")
        return

    cat_tabs = st.tabs(available_categories)

    for i, category in enumerate(available_categories):
        with cat_tabs[i]:
            # Haal gemiddelde prijzen voor de compute
            price_base = df["Price_Base"].mean()
            price_peak = df["Price_Peak"].mean()

            with st.spinner(f"Scenario's berekenen voor {category}..."):
                cat_scenarios = compute_all_scenarios(
                    df, category, price_base, price_peak, epex_loaded,
                )

            render_comparison_table(cat_scenarios)


# ═══════════════════════════════════════════════════════════════════════════
# Tab 3: Resultaten
# ═══════════════════════════════════════════════════════════════════════════

def render_tab_results(
    df: pd.DataFrame,
    p_mw_col: str,
    strategy_period: str,
    epex_loaded: bool,
) -> None:
    """Resultaten-tab: finetuning sliders + 4 sub-tabs met visualisaties."""

    # --- Finetuning sliders ---
    st.markdown("### Finetunen _")

    sel_info = _render_selection_info()

    df = _render_hedge_sliders(df, p_mw_col, strategy_period)

    # --- Berekeningen ---
    df = apply_hedge_columns(df, p_mw_col)
    results = compute_hedge_results(df)
    financial = compute_financial_summary(df, epex_loaded)
    q_stats = compute_quarterly_table(df, epex_loaded)

    # --- 4 Sub-tabs ---
    st.markdown("---")
    tab_main, tab_vol, tab_eco, tab_charts = st.tabs(
        ["Samenvatting", "Jouw volume flow", "Kengetallen", "Seizoenen"]
    )

    with tab_main:
        _render_summary_subtab(df, financial, results)

    with tab_vol:
        _render_volume_subtab(df, results)

    with tab_eco:
        _render_economics_subtab(financial, epex_loaded)

    with tab_charts:
        _render_seasonal_subtab(df, p_mw_col, q_stats, epex_loaded)


# ---------------------------------------------------------------------------
# Helpers voor Tab 3
# ---------------------------------------------------------------------------

def _render_selection_info() -> None:
    """Toon welk scenario geselecteerd is (vanuit Tab 2)."""
    selected = st.session_state.get("selected_scenarios", {})
    if selected:
        for cat, sel in selected.items():
            prod_label = next(
                (p["label"] for p in cfg.PRODUCTS if p["key"] == sel["product"]),
                sel["product"],
            )
            opt_label = next(
                (o["label"] for o in cfg.OPTIMIZATIONS if o["key"] == sel["optimization"]),
                sel["optimization"],
            )
            st.success(
                f"**{cat}**: {prod_label} — {opt_label} "
                f"(vanuit Strategieën-tab)"
            )
    else:
        st.info(
            "Selecteer een strategie in de Strategieën-tab, "
            "of pas de sliders hieronder handmatig aan."
        )


def _build_periods(
    df: pd.DataFrame, strategy_period: str
) -> list[tuple[int | None, pd.DataFrame]]:
    """Bouw lijst van (quarter, sub_df) tuples."""
    if strategy_period == "Per Jaar":
        return [(None, df)]
    return [(q, df[df["Quarter"] == q]) for q in [1, 2, 3, 4]]


def _render_hedge_sliders(
    df: pd.DataFrame,
    p_mw_col: str,
    strategy_period: str,
) -> pd.DataFrame:
    """Toon de Base/Peak MW sliders en schrijf hedge-kolommen naar het DataFrame."""
    df = df.copy()
    df["Hedge_Base_MW"] = 0.0
    df["Hedge_Peak_MW"] = 0.0

    curr_min = df[p_mw_col].min()
    curr_max = df[p_mw_col].max()
    slider_min = float(np.floor(curr_min * 1.5 - 1))
    slider_max = float(np.ceil(curr_max * 1.5 + 1))
    if slider_max < slider_min:
        slider_max = slider_min + 10.0

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
            sc1, sc2 = st.columns(2)
            b_val = sc1.slider(
                "Base MW (Jaar)", slider_min, slider_max, key=b_key, step=0.1
            )
            p_val = sc2.slider(
                "Peak MW (Jaar)", slider_min, slider_max, key=p_key, step=0.1
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


# ═══════════════════════════════════════════════════════════════════════════
# Sub-tab renderers (bestaande logica, verplaatst)
# ═══════════════════════════════════════════════════════════════════════════

def _render_summary_subtab(
    df: pd.DataFrame,
    financial: FinancialSummary,
    results: HedgeResults,
) -> None:
    """Hoofdoverzicht met KPI's en maandelijkse cashflow-grafiek."""
    st.markdown("### Helder overzicht _")

    m1, m2, m3 = st.columns(3)

    avg = financial.avg_cost_per_mwh
    if avg < 0:
        m1.metric(
            "Jouw opbrengst (Winst)",
            f"€ {abs(avg):.2f} / MWh",
            "Verdienmodel",
            delta_color="normal",
        )
    else:
        m1.metric(
            "Jouw kostprijs",
            f"€ {avg:.2f} / MWh",
            "Kosten",
            delta_color="inverse",
        )

    tot = financial.total_energy_cost
    cost_label = "Totale verdienste (Netto)" if tot < 0 else "Totale kosten (Netto)"
    m2.metric(cost_label, f"€ {abs(tot):,.0f}")
    m3.metric("Direct afgedekt", f"{results.hedge_efficiency_pct:.1f}%")

    st.markdown("---")
    st.markdown("### De cijfers door het jaar heen _")

    # Maandelijkse cashflow
    df_plot = df.copy()
    df_plot["Maand_Naam"] = df_plot["Date"].dt.strftime("%Y-%m")

    monthly = (
        df_plot.groupby("Maand_Naam")[
            ["Cost_Hedge_Total_EUR", "Cost_Buy_EUR", "Rev_Sell_EUR"]
        ]
        .sum()
        .reset_index()
    )
    monthly["Rev_Sell_EUR"] = -monthly["Rev_Sell_EUR"]

    monthly_melt = monthly.melt(
        id_vars=["Maand_Naam"],
        value_vars=["Cost_Hedge_Total_EUR", "Cost_Buy_EUR", "Rev_Sell_EUR"],
        var_name="Kostenpost",
        value_name="Euro",
    )
    monthly_melt["Kostenpost"] = monthly_melt["Kostenpost"].replace(
        {
            "Cost_Hedge_Total_EUR": "1. Vaste inkoop",
            "Cost_Buy_EUR": "2. Spot inkoop (tekort)",
            "Rev_Sell_EUR": "3. Spot verkoop (overschot)",
        }
    )

    domain = ["1. Vaste inkoop", "2. Spot inkoop (tekort)", "3. Spot verkoop (overschot)"]
    colors = [cfg.COLORS["grey"], cfg.COLORS["ruby"], cfg.COLORS["gold"]]

    chart = (
        alt.Chart(monthly_melt)
        .mark_bar()
        .encode(
            x=alt.X("Maand_Naam:N", title="Maand"),
            y=alt.Y("sum(Euro):Q", title="Bedrag in Euro"),
            color=alt.Color(
                "Kostenpost:N",
                scale=alt.Scale(domain=domain, range=colors),
                legend=alt.Legend(title="", orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("Maand_Naam:N", title="Maand"),
                alt.Tooltip("Kostenpost:N", title="Post"),
                alt.Tooltip("sum(Euro):Q", title="Bedrag (€)", format=",.0f"),
            ],
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_volume_subtab(
    df: pd.DataFrame,
    results: HedgeResults,
) -> None:
    """Watervalgrafiek: profiel → hedge → spotmarkt → balans."""
    st.markdown("### Jouw verbruik in balans _")

    total_prof_abs = abs(results.profile_mwh)
    prof_label = (
        "Jouw totale opwek" if results.profile_mwh < 0 else "Jouw totale verbruik"
    )

    v1, v2, v3 = st.columns(3)
    v1.metric(prof_label, f"{total_prof_abs:,.0f} MWh")
    v2.metric("Ingekocht via blokken", f"{results.hedge_mwh:,.0f} MWh")
    v3.metric("Direct afgedekt", f"{results.hedge_efficiency_pct:.1f}%")

    st.markdown("<br><br><b>De flow van jouw stroom _</b>", unsafe_allow_html=True)
    st.info(
        "Een échte watervalgrafiek: we starten met de totale behoefte, "
        "trekken de ingekochte blokken eraf, en het restant werk je weg "
        "op de spotmarkt zodat we precies op 0 (balans) uitkomen."
    )

    wf_data = []
    wf_data.append(
        {
            "Stap": f"1. {prof_label}",
            "Start": 0,
            "Eind": total_prof_abs,
            "Volume": total_prof_abs,
            "Kleur": "Profiel",
        }
    )
    eind_hedge = total_prof_abs - results.hedge_mwh
    wf_data.append(
        {
            "Stap": "2. Ingekocht via blokken",
            "Start": total_prof_abs,
            "Eind": eind_hedge,
            "Volume": -results.hedge_mwh,
            "Kleur": "Hedge",
        }
    )
    if results.under_hedge_mwh > 0:
        wf_data.append(
            {
                "Stap": "3. Spot inkoop (tekort)",
                "Start": eind_hedge,
                "Eind": 0,
                "Volume": results.under_hedge_mwh,
                "Kleur": "Tekort",
            }
        )
    if results.over_hedge_mwh > 0:
        wf_data.append(
            {
                "Stap": "4. Spot verkoop (overschot)",
                "Start": eind_hedge,
                "Eind": 0,
                "Volume": -results.over_hedge_mwh,
                "Kleur": "Overschot",
            }
        )

    wf_df = pd.DataFrame(wf_data)

    waterfall = (
        alt.Chart(wf_df)
        .mark_bar(size=60)
        .encode(
            x=alt.X("Stap:O", title="", sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Start:Q", title="Volume (MWh)"),
            y2="Eind:Q",
            color=alt.Color(
                "Kleur:N",
                scale=alt.Scale(
                    domain=["Profiel", "Hedge", "Tekort", "Overschot"],
                    range=[
                        cfg.COLORS["grey"],
                        cfg.COLORS["gold"],
                        cfg.COLORS["ruby"],
                        cfg.COLORS["black"],
                    ],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Stap:N"),
                alt.Tooltip("Volume:Q", title="Volume Change (MWh)", format=",.0f"),
            ],
        )
        .properties(height=400)
    )
    st.altair_chart(waterfall, use_container_width=True)


def _render_economics_subtab(
    financial: FinancialSummary,
    epex_loaded: bool,
) -> None:
    """Diepgaande KPI's per MWh."""
    st.markdown("### De cijfers per MWh _")

    if not epex_loaded:
        st.info("Spotprijzen konden niet worden geladen. Sommige velden staan op €0.")

    u1, u2 = st.columns(2)
    u1.metric(
        "Kosten per benutte MWh",
        f"€ {financial.cost_per_used_mwh:.2f}",
        help=(
            "De kosten van de inkoopblokken verdeeld over uitsluitend het volume "
            "dat je *daadwerkelijk* zelf hebt afgestreept (inclusief weggegooid overschot)."
        ),
    )
    u2.metric(
        "Waarde van jouw profiel (Capture Price)",
        f"€ {financial.capture_price:.2f}",
        f"€ {financial.capture_diff:.2f} t.o.v. de markt",
        delta_color="normal" if financial.capture_diff > 0 else "inverse",
        help=(
            "De échte gemiddelde waarde van jouw stroomprofiel op de spotmarkt."
        ),
    )

    st.markdown("<br>", unsafe_allow_html=True)
    u3, u4 = st.columns(2)
    u3.metric("Wat betaalde je voor tekorten?", f"€ {financial.avg_spot_buy:.2f}")
    u4.metric("Wat leverde je overschot op?", f"€ {financial.avg_spot_sell:.2f}")


def _render_seasonal_subtab(
    df: pd.DataFrame,
    p_mw_col: str,
    q_stats: pd.DataFrame,
    epex_loaded: bool,
) -> None:
    """Seizoensgrafieken + kwartaaltabel + downloadknop."""
    st.markdown("### Seizoenen in de praktijk _")

    # Bepaal het jaar uit de data
    data_year = df["Date"].dt.year.mode().iloc[0]

    cols_chart = st.columns(2) + st.columns(2)

    for i, week_cfg in enumerate(cfg.SEASONAL_WEEKS):
        with cols_chart[i]:
            st.caption(week_cfg["name"])

            try:
                week_start = pd.Timestamp.fromisocalendar(
                    data_year, week_cfg["iso_week"], 1
                )
                week_end = week_start + pd.Timedelta(days=6)
            except (ValueError, AttributeError):
                st.info("Kan weekdatum niet bepalen.")
                continue

            if df["Date"].min() > week_end or df["Date"].max() < week_start:
                st.info("Geen data beschikbaar voor deze periode.")
                continue

            mask = (df["Date"] >= week_start) & (
                df["Date"] <= week_end + pd.Timedelta(days=1)
            )
            plot_df = df.loc[mask, ["Date", p_mw_col, "Current_Hedge_MW"]].copy()
            plot_df.rename(
                columns={p_mw_col: "Jouw profiel", "Current_Hedge_MW": "Inkoopblok"},
                inplace=True,
            )

            chart_data = plot_df.melt(
                id_vars=["Date"], var_name="Type", value_name="MW"
            )
            c = (
                alt.Chart(chart_data)
                .mark_line(interpolate="step-after")
                .encode(
                    x=alt.X(
                        "Date:T",
                        axis=alt.Axis(format="%a %H:%M", title=None),
                    ),
                    y=alt.Y("MW:Q", title=None),
                    color=alt.Color(
                        "Type:N",
                        scale=alt.Scale(
                            domain=["Jouw profiel", "Inkoopblok"],
                            range=[cfg.COLORS["dark_grey"], cfg.COLORS["gold"]],
                        ),
                        legend=alt.Legend(orient="bottom", title=None),
                    ),
                )
                .properties(height=180)
            )
            st.altair_chart(c, use_container_width=True)

    # --- Kwartaaloverzicht ---
    st.markdown("---")
    st.markdown("### Kwartaaloverzicht _")

    format_dict = {k: v for k, v in QUARTERLY_FORMAT.items() if k in q_stats.columns}
    st.dataframe(q_stats.style.format(format_dict), use_container_width=True)

    # --- Download ---
    csv_dl = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download je analyse (CSV)", csv_dl, "censo_analyse.csv", "text/csv"
    )
