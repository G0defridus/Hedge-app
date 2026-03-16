"""
Resultaat-renderers — standalone functies voor de resultaten-sectie.

render_seasonal_preview() kan ook standalone worden aangeroepen (100% SPOT baseline).
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
from core.excel_export import build_export_workbook
from core.models import FinancialSummary, HedgeResults
from core.optimizer import find_optimal_position
from ui import state as ui_state
from ui.theme import QUARTERLY_FORMAT


# ═══════════════════════════════════════════════════════════════════════════
# Seizoensgrafieken + kwartaaltabel (standalone, vóór selectie)
# ═══════════════════════════════════════════════════════════════════════════

def render_seasonal_preview(
    df: pd.DataFrame,
    p_mw_col: str,
    q_stats: pd.DataFrame,
    *,
    has_hedge: bool = False,
) -> None:
    """Seizoensgrafieken + kwartaaltabel.

    Parameters
    ----------
    df : DataFrame
        Het DataFrame met profiel en (optioneel) hedge-kolommen.
    p_mw_col : str
        Kolomnaam voor het actieve profiel in MW.
    q_stats : DataFrame
        Output van ``compute_quarterly_table()``.
    has_hedge : bool
        Als True, toon ook de hedge-overlay in de grafieken.
    """
    st.markdown("### Seizoenen in de praktijk _")

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

            if has_hedge and "Current_Hedge_MW" in df.columns:
                plot_df = df.loc[mask, ["Date", p_mw_col, "Current_Hedge_MW"]].copy()
                plot_df.rename(
                    columns={p_mw_col: "Jouw profiel", "Current_Hedge_MW": "Inkoopblok"},
                    inplace=True,
                )
                domain = ["Jouw profiel", "Inkoopblok"]
                colors = [cfg.COLORS["dark_grey"], cfg.COLORS["gold"]]
            else:
                plot_df = df.loc[mask, ["Date", p_mw_col]].copy()
                plot_df.rename(columns={p_mw_col: "Jouw profiel"}, inplace=True)
                domain = ["Jouw profiel"]
                colors = [cfg.COLORS["dark_grey"]]

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
                        scale=alt.Scale(domain=domain, range=colors),
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


# ═══════════════════════════════════════════════════════════════════════════
# Hoofdfunctie — rendert de volledige resultaten-sectie (na selectie)
# ═══════════════════════════════════════════════════════════════════════════

def render_results_section(
    df: pd.DataFrame,
    p_mw_col: str,
    strategy_period: str,
    epex_loaded: bool,
) -> None:
    """Render de complete resultaten-sectie onder de optimalisatie-kaarten."""
    render_selection_info()

    df = render_hedge_sliders(df, p_mw_col, strategy_period)

    # Berekeningen
    df = apply_hedge_columns(df, p_mw_col)
    results = compute_hedge_results(df)
    financial = compute_financial_summary(df, epex_loaded)
    q_stats = compute_quarterly_table(df, epex_loaded)

    # Seizoensgrafieken met hedge-overlay
    render_seasonal_preview(df, p_mw_col, q_stats, has_hedge=True)

    # Excel download
    selected = st.session_state.get("selected_scenarios", {})
    category = next(iter(selected), "")
    sel = selected.get(category, {})
    optimization_key = sel.get("optimization", "")

    excel_buf = build_export_workbook(
        df, category, optimization_key,
        strategy_period, epex_loaded, q_stats,
    )
    st.download_button(
        "Download berekening (Excel)",
        excel_buf,
        "censo_hedge_berekening.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # 3 Sub-tabs
    st.markdown("---")
    tab_main, tab_vol, tab_eco = st.tabs(
        ["Samenvatting", "Jouw volume flow", "Kengetallen"]
    )

    with tab_main:
        render_summary(df, financial, results)

    with tab_vol:
        render_volume_flow(df, results)

    with tab_eco:
        render_economics(financial, epex_loaded)


# ═══════════════════════════════════════════════════════════════════════════
# Selectie-info
# ═══════════════════════════════════════════════════════════════════════════

def render_selection_info() -> None:
    """Toon welk scenario geselecteerd is."""
    selected = st.session_state.get("selected_scenarios", {})
    if selected:
        for cat, sel in selected.items():
            opt_label = next(
                (o["label"] for o in cfg.OPTIMIZATIONS if o["key"] == sel["optimization"]),
                sel["optimization"],
            )
            st.success(f"**{cat}**: {opt_label}")


# ═══════════════════════════════════════════════════════════════════════════
# Finetuning sliders
# ═══════════════════════════════════════════════════════════════════════════

def render_hedge_sliders(
    df: pd.DataFrame,
    p_mw_col: str,
    strategy_period: str,
) -> pd.DataFrame:
    """Toon Base/Peak MW sliders en schrijf hedge-kolommen naar het DataFrame."""
    st.markdown("### Finetunen _")

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


def _build_periods(
    df: pd.DataFrame, strategy_period: str
) -> list[tuple[int | None, pd.DataFrame]]:
    """Bouw lijst van (quarter, sub_df) tuples."""
    if strategy_period == "Per Jaar":
        return [(None, df)]
    return [(q, df[df["Quarter"] == q]) for q in [1, 2, 3, 4]]


# ═══════════════════════════════════════════════════════════════════════════
# Samenvatting
# ═══════════════════════════════════════════════════════════════════════════

def render_summary(
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


# ═══════════════════════════════════════════════════════════════════════════
# Volume flow (watervalgrafiek)
# ═══════════════════════════════════════════════════════════════════════════

def render_volume_flow(
    df: pd.DataFrame,
    results: HedgeResults,
) -> None:
    """Watervalgrafiek: profiel -> hedge -> spotmarkt -> balans."""
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
        "Een watervalgrafiek: we starten met de totale behoefte, "
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
                alt.Tooltip("Volume:Q", title="Volume (MWh)", format=",.0f"),
            ],
        )
        .properties(height=400)
    )
    st.altair_chart(waterfall, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Kengetallen (per MWh)
# ═══════════════════════════════════════════════════════════════════════════

def render_economics(
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
            "De gemiddelde waarde van jouw stroomprofiel op de spotmarkt."
        ),
    )

    st.markdown("<br>", unsafe_allow_html=True)
    u3, u4 = st.columns(2)
    u3.metric("Wat betaalde je voor tekorten?", f"€ {financial.avg_spot_buy:.2f}")
    u4.metric("Wat leverde je overschot op?", f"€ {financial.avg_spot_sell:.2f}")
