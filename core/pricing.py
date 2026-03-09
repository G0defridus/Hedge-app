"""
Contractprijzen — standaardwaarden + optioneel CSV-bestand met Endex-prijzen.

Geen Streamlit-afhankelijkheden.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Optional

import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)


def _find_pricing_file(period: str) -> Optional[str]:
    """Zoek een CSV-bestand met Endex-prijzen.

    Parameters
    ----------
    period : str
        ``"Jaar"`` of ``"Kwartaal"``.

    Returns
    -------
    str | None
        Pad naar het eerst gevonden bestand, of None.
    """
    keyword = "jaar" if period == "Jaar" else "kwartaal"
    patterns = [
        f"**/*{keyword}*endex*.csv",
        f"**/*endex*{keyword}*.csv",
    ]
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            return files[0]

    # Fallback: alleen huidige directory
    for f in os.listdir("."):
        if keyword in f.lower() and "endex" in f.lower() and f.endswith(".csv"):
            return f

    return None


def _parse_pricing_csv(filepath: str) -> pd.DataFrame:
    """Lees en parse een Endex-prijzen CSV.

    Verwacht een CSV met kolommen die 'base' en 'peak'/'p16' bevatten.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()

    skip = 0
    if len(lines) > 1 and "Date" in lines[1] and "Base" in lines[1]:
        skip = 1

    df = pd.read_csv(filepath, sep=";", skiprows=skip)

    # Komma → punt conversie
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", ".")

    return df


def _extract_base_peak(df: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
    """Extraheer de laatste geldige Base- en Peak-prijs uit het DataFrame."""
    base_cols = [c for c in df.columns if "base" in c.lower()]
    peak_cols = [c for c in df.columns if "p16" in c.lower() or "peak" in c.lower()]

    if not base_cols or not peak_cols:
        return None, None

    base_val: Optional[float] = None
    for bc in reversed(base_cols):
        s = pd.to_numeric(df[bc], errors="coerce").dropna()
        if len(s) > 10:
            base_val = float(s.iloc[-1])
            break

    peak_val: Optional[float] = None
    for pc in reversed(peak_cols):
        s = pd.to_numeric(df[pc], errors="coerce").dropna()
        if len(s) > 10:
            peak_val = float(s.iloc[-1])
            break

    return base_val, peak_val


def _apply_quarterly_weight(
    base: float, peak: float, quarter: int
) -> tuple[float, float]:
    """Schaal jaarlijkse CSV-prijs naar kwartaalniveau met seizoensverhoudingen."""
    avg_base = sum(cfg.QUARTERLY_BASE_WEIGHTS.values()) / 4
    avg_peak = sum(cfg.QUARTERLY_PEAK_WEIGHTS.values()) / 4

    b_weight = cfg.QUARTERLY_BASE_WEIGHTS.get(quarter, avg_base)
    p_weight = cfg.QUARTERLY_PEAK_WEIGHTS.get(quarter, avg_peak)

    return base * (b_weight / avg_base), peak * (p_weight / avg_peak)


# ═══════════════════════════════════════════════════════════════════════════
# Publieke API
# ═══════════════════════════════════════════════════════════════════════════

def get_default_price(
    period: str = "Jaar",
    quarter: Optional[int] = None,
) -> tuple[float, float]:
    """Geef de standaard Base/Peak-prijs (€/MWh) voor een periode.

    Probeert eerst een Endex CSV te laden; valt terug op config-defaults.

    Parameters
    ----------
    period : str
        ``"Jaar"`` of ``"Kwartaal"``.
    quarter : int, optional
        Kwartaal (1–4), alleen nodig als ``period == "Kwartaal"``.

    Returns
    -------
    tuple[float, float]
        ``(base_price, peak_price)`` in €/MWh.
    """
    # Stap 1: defaults uit config
    if period == "Kwartaal" and quarter is not None:
        key = f"Q{quarter}"
        defaults = cfg.DEFAULT_PRICES.get(key, cfg.DEFAULT_PRICES["year"])
    else:
        defaults = cfg.DEFAULT_PRICES["year"]

    b_val = defaults["base"]
    p_val = defaults["peak"]

    # Stap 2: probeer CSV te laden
    try:
        filepath = _find_pricing_file(period)
        if filepath is None:
            logger.debug("Geen Endex CSV gevonden voor period=%s — gebruik defaults.", period)
            return round(b_val, 2), round(p_val, 2)

        logger.info("Endex CSV gevonden: %s", filepath)
        df = _parse_pricing_csv(filepath)
        csv_base, csv_peak = _extract_base_peak(df)

        if csv_base is not None and csv_peak is not None:
            b_val, p_val = csv_base, csv_peak

            if period == "Kwartaal" and quarter is not None:
                b_val, p_val = _apply_quarterly_weight(b_val, p_val, quarter)
        else:
            logger.warning(
                "CSV gevonden maar geen geldige Base/Peak kolommen — gebruik defaults."
            )

    except Exception:
        logger.warning(
            "Fout bij laden Endex CSV — gebruik standaardprijzen.", exc_info=True
        )

    return round(b_val, 2), round(p_val, 2)
