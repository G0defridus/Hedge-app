"""
Tab renderers — alle 4 tabbladen.

Elke functie ontvangt het berekende DataFrame + resultaat-dataclasses
en rendert één tabblad.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import config as cfg
from core.models import FinancialSummary, HedgeResults
from ui.theme import censo_color_scale, QUARTERLY_FORMAT


# ═══════════════════════════════════════════════════════════════════════════
# Tab 1: Samenvatting
# ═══════════════════════════════════════════════════════════════════════════

def render_summary_tab(
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
# Tab 2: Volume flow (waterval)
# ═══════════════════════════════════════════════════════════════════════════

def render_volume_tab(
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


# ═══════════════════════════════════════════════════════════════════════════
# Tab 3: Kengetallen (Unit Economics)
# ═══════════════════════════════════════════════════════════════════════════

def render_economics_tab(
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


# ═══════════════════════════════════════════════════════════════════════════
# Tab 4: Seizoenen + Kwartaaloverzicht
# ═══════════════════════════════════════════════════════════════════════════

def render_seasonal_tab(
    df: pd.DataFrame,
    p_mw_col: str,
    q_stats: pd.DataFrame,
    epex_loaded: bool,
) -> None:
    """Seizoensgrafieken + kwartaaltabel + downloadknop."""
    st.markdown("### Seizoenen in de praktijk _")

    # Bepaal het jaar uit de data (dynamisch ipv hardcoded 2025)
    data_year = df["Date"].dt.year.mode().iloc[0]

    cols_chart = st.columns(2) + st.columns(2)

    for i, week_cfg in enumerate(cfg.SEASONAL_WEEKS):
        with cols_chart[i]:
            st.caption(week_cfg["name"])

            # Zoek de maandag van de ISO-week in het datajaar
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
