"""
Contractprijzen — live marktdata via censo_marktdata, CSV-fallback, of defaults.

Prioriteit:
  1. censo_marktdata (Profiteia API + SQLite cache)
  2. Lokale Endex CSV-bestanden
  3. Hardcoded defaults uit config.py

Geen Streamlit-afhankelijkheden.
"""

from __future__ import annotations

import glob
import logging
import os
import sys
from typing import Optional

import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)

# censo_marktdata staat buiten deze repo — pad toevoegen voor import
_MARKTDATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Marketdata dashboard")
if os.path.isdir(_MARKTDATA_DIR) and _MARKTDATA_DIR not in sys.path:
    sys.path.insert(0, os.path.normpath(_MARKTDATA_DIR))


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
# censo_marktdata integratie
# ═══════════════════════════════════════════════════════════════════════════

# Cache om get_latest_prices() niet bij elke sidebar-render opnieuw aan te roepen
_marktdata_cache: dict | None = None
_marktdata_source: dict | None = None  # {"base_key": ..., "peak_key": ..., "trade_date": ...}


def _fetch_marktdata_prices(
    contract_year: int,
    period: str = "Jaar",
    quarter: Optional[int] = None,
) -> tuple[Optional[float], Optional[float]]:
    """Haal laatste Base/Peak prijs op via censo_marktdata.

    Parameters
    ----------
    contract_year : int
        Het Cal-jaar (bijv. 26 voor Cal 26). Wordt afgeleid uit profieljaar.
    period : str
        ``"Jaar"`` of ``"Kwartaal"``.
    quarter : int, optional
        Kwartaal voor seizoensschaling.

    Returns
    -------
    tuple[float | None, float | None]
    """
    global _marktdata_cache, _marktdata_source

    try:
        from censo_marktdata import get_latest_prices  # noqa: F811

        if _marktdata_cache is None:
            _marktdata_cache = get_latest_prices()

        cal_label = f"Cal {contract_year}"
        base_key = f"Elektriciteit Base ({cal_label})"
        peak_key = f"Elektriciteit Piek ({cal_label})"

        base_entry = _marktdata_cache.get(base_key)
        peak_entry = _marktdata_cache.get(peak_key)

        if base_entry is None or peak_entry is None:
            logger.debug("Contract %s niet gevonden in marktdata.", cal_label)
            return None, None

        b = float(base_entry["value"])
        p = float(peak_entry["value"])

        _marktdata_source = {
            "contract": cal_label,
            "base_value": b,
            "peak_value": p,
            "trade_date": base_entry.get("trade_date", ""),
        }

        if period == "Kwartaal" and quarter is not None:
            b, p = _apply_quarterly_weight(b, p, quarter)

        return round(b, 2), round(p, 2)

    except ImportError:
        logger.debug("censo_marktdata module niet beschikbaar.")
    except Exception:
        logger.debug("Fout bij ophalen marktdata.", exc_info=True)

    return None, None


def get_marktdata_source() -> Optional[dict]:
    """Geef metadata over de laatst opgehaalde marktdata-prijzen.

    Returns
    -------
    dict | None
        Keys: contract, base_value, peak_value, trade_date
    """
    return _marktdata_source


def contract_year_from_df(df: pd.DataFrame) -> int:
    """Bepaal het Cal-contractjaar op basis van het profieljaar in de data.

    Cal-contract = profieljaar (bijv. profiel 2026 → Cal 26).
    Geeft het tweecijferige jaar terug (26, niet 2026).
    """
    if "Date" in df.columns:
        year = int(df["Date"].dt.year.mode().iloc[0])
    else:
        year = 2026  # fallback
    return year % 100


# ═══════════════════════════════════════════════════════════════════════════
# Publieke API
# ═══════════════════════════════════════════════════════════════════════════

def get_default_price(
    period: str = "Jaar",
    quarter: Optional[int] = None,
    contract_year: Optional[int] = None,
) -> tuple[float, float]:
    """Geef de standaard Base/Peak-prijs (€/MWh) voor een periode.

    Prioriteit:
      1. censo_marktdata (live forward curves)
      2. Lokale Endex CSV
      3. Hardcoded defaults uit config.py

    Parameters
    ----------
    period : str
        ``"Jaar"`` of ``"Kwartaal"``.
    quarter : int, optional
        Kwartaal (1–4), alleen nodig als ``period == "Kwartaal"``.
    contract_year : int, optional
        Cal-contractjaar (bijv. 26). Als None, wordt marktdata overgeslagen.

    Returns
    -------
    tuple[float, float]
        ``(base_price, peak_price)`` in €/MWh.
    """
    # Stap 1: probeer censo_marktdata
    if contract_year is not None:
        mb, mp = _fetch_marktdata_prices(contract_year, period, quarter)
        if mb is not None and mp is not None:
            return mb, mp

    # Stap 2: defaults uit config (fallback)
    if period == "Kwartaal" and quarter is not None:
        key = f"Q{quarter}"
        defaults = cfg.DEFAULT_PRICES.get(key, cfg.DEFAULT_PRICES["year"])
    else:
        defaults = cfg.DEFAULT_PRICES["year"]

    b_val = defaults["base"]
    p_val = defaults["peak"]

    # Stap 3: probeer CSV
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
