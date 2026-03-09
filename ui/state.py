"""
Gecentraliseerd session state beheer.

Alle slider-keys en hun standaardwaarden op één plek.
Voorkomt verspreid ``if key not in st.session_state`` door de hele codebase.
"""

from __future__ import annotations

import streamlit as st

from core.models import HedgePosition


# ---------------------------------------------------------------------------
# Schema: alle state keys met defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, float | int] = {
    "slider_b_yr": 0.0,
    "slider_p_yr": 0.0,
    "slider_b_q1": 0.0,
    "slider_p_q1": 0.0,
    "slider_b_q2": 0.0,
    "slider_p_q2": 0.0,
    "slider_b_q3": 0.0,
    "slider_p_q3": 0.0,
    "slider_b_q4": 0.0,
    "slider_p_q4": 0.0,
    "custom_hedge_pct": 100,
}


def init_state() -> None:
    """Initialiseer alle state keys als ze nog niet bestaan."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Getters / Setters per periode
# ---------------------------------------------------------------------------

def _suffix(quarter: int | None) -> str:
    return "yr" if quarter is None else f"q{quarter}"


def get_hedge_position(quarter: int | None = None) -> HedgePosition:
    """Lees de huidige slider-waarden voor een periode."""
    s = _suffix(quarter)
    return HedgePosition(
        base_mw=st.session_state.get(f"slider_b_{s}", 0.0),
        peak_add_mw=st.session_state.get(f"slider_p_{s}", 0.0),
    )


def set_hedge_position(pos: HedgePosition, quarter: int | None = None) -> None:
    """Schrijf een HedgePosition naar de slider state."""
    s = _suffix(quarter)
    st.session_state[f"slider_b_{s}"] = float(pos.base_mw)
    st.session_state[f"slider_p_{s}"] = float(pos.peak_add_mw)
