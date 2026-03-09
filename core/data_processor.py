"""
Data-inlees en categorisatie — volledig onafhankelijk van Streamlit.

Twee ingangspunten:
  - process_raw_connections()  → ruwe 15-min meetdata → geaggregeerd DataFrame
  - load_aggregated_csv()      → reeds geaggregeerd CSV → gevalideerd DataFrame
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

import config as cfg
from core.models import DataLoadError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias voor een optionele progress-callback: (fractie, tekst) → None
# ---------------------------------------------------------------------------
ProgressCallback = Optional[Callable[[float, str], None]]


# ═══════════════════════════════════════════════════════════════════════════
# Interne helpers
# ═══════════════════════════════════════════════════════════════════════════

def _calculate_winter_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Bereken het gemiddelde kwartierverbruik per tijdstip in de wintermaanden."""
    winter_mask = df.index.month.isin([1, 11, 12])
    winter_df = df[winter_mask]
    if winter_df.empty:
        return df.groupby(df.index.time).mean()
    return winter_df.groupby(winter_df.index.time).mean()


def _estimate_gross_solar(
    df: pd.DataFrame,
    col: str,
    winter_profile: pd.DataFrame,
) -> pd.Series:
    """Schat de bruto zonne-opwek achter de meter voor één aansluiting."""
    w_prof = winter_profile[col]

    # Nacht-baseline (uren uit config)
    night_mask = np.zeros(len(df), dtype=bool)
    for start_h, end_h in cfg.NIGHT_HOUR_RANGES:
        night_mask |= (df.index.hour >= start_h) & (df.index.hour < end_h)

    daily_night_avg = df.loc[night_mask, col].resample("D").mean()

    night_times = [
        t
        for t in w_prof.index
        if any(s <= t.hour < e for s, e in cfg.NIGHT_HOUR_RANGES)
    ]
    base_night_avg = w_prof.loc[night_times].mean()
    if base_night_avg < cfg.NIGHT_BASELINE_FLOOR:
        base_night_avg = cfg.NIGHT_BASELINE_FLOOR

    # Dagelijkse schaalfactor
    daily_scaling = (daily_night_avg / base_night_avg).clip(
        cfg.SCALING_CLIP_MIN, cfg.SCALING_CLIP_MAX
    )
    dates = pd.Series(df.index.normalize(), index=df.index)
    scaling_series = dates.map(daily_scaling).ffill().bfill()

    # Verwacht basisverbruik
    base_load = pd.Series(
        df.index.map(lambda x: w_prof.loc[x.time()]),  # noqa: B023
        index=df.index,
    )
    expected_load = base_load * scaling_series

    # Bruto opwek = verwacht verbruik − daadwerkelijke meting (als positief, overdag)
    solar_behind_meter = expected_load - df[col]
    daylight_mask = (df.index.hour >= cfg.DAYLIGHT_START_HOUR) & (
        df.index.hour <= cfg.DAYLIGHT_END_HOUR
    )
    solar_behind_meter = solar_behind_meter.where(daylight_mask, 0).clip(lower=0)

    actual_export = df[col].clip(upper=0).abs()
    return solar_behind_meter + actual_export


def _build_ideal_solar_curve() -> np.ndarray:
    """Ideale Gauss-zonnecurve (24 uurwaarden) voor correlatie-check."""
    hours = np.arange(24)
    curve = np.exp(-((hours - 13) ** 2) / (2 * 2.5**2))
    curve[hours < 6] = 0
    curve[hours > 21] = 0
    return curve


def _categorize_connections(
    df: pd.DataFrame,
    gross_production_df: pd.DataFrame,
    estimated_volumes: dict[str, float],
) -> dict[str, str]:
    """Bepaal Consumer / Prosumer / Producer per aansluiting."""
    solar_curve_ideal = _build_ideal_solar_curve()

    # Stap 1: eerste indeling op volume
    categories: dict[str, str] = {}
    for col in df.columns:
        gross_vol = estimated_volumes[col]
        if gross_vol > cfg.GROSS_SOLAR_THRESHOLD_KWH:
            total_import = df[col][df[col] > 0].sum()
            if total_import < cfg.PRODUCER_IMPORT_RATIO * gross_vol:
                categories[col] = "Producer"
            else:
                categories[col] = "Prosumer"
        else:
            categories[col] = "Consumer"

    # Stap 2: Prosumer-correlatie met ideale zonnecurve
    final: dict[str, str] = {}
    for col in df.columns:
        cat = categories[col]
        if cat == "Prosumer":
            daily_avg = gross_production_df[col].groupby(
                gross_production_df[col].index.hour
            ).mean()
            daily_avg = daily_avg.reindex(range(24), fill_value=0)
            corr = 0.0
            if np.std(daily_avg) > 0 and np.std(solar_curve_ideal) > 0:
                corr = float(np.corrcoef(daily_avg, solar_curve_ideal)[0, 1])
            final[col] = "Prosumer" if corr >= cfg.SOLAR_CORRELATION_MIN else "Consumer"
        else:
            final[col] = cat

    return final


# ═══════════════════════════════════════════════════════════════════════════
# Publieke API
# ═══════════════════════════════════════════════════════════════════════════

def process_raw_connections(
    file,
    on_progress: ProgressCallback = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Verwerk ruwe 15-min meetdata → geaggregeerd profiel-DataFrame + mapping.

    Parameters
    ----------
    file : str | Path | file-like
        CSV-bestand met tijdreeks per aansluiting (sep=';', decimal=',').
    on_progress : callable, optional
        Callback ``(fraction: float, message: str) → None`` voor UI-feedback.

    Returns
    -------
    agg_df : pd.DataFrame
        Kolommen: Consumer, Prosumer, Producer, Total.  Index = DatetimeIndex.
    mapping : dict[str, str]
        Aansluiting → categorie.

    Raises
    ------
    DataLoadError
        Bij ongeldige invoer of een leeg bestand.
    """
    # --- Inlezen ---
    try:
        df = pd.read_csv(file, sep=";", decimal=",", index_col=0, parse_dates=True, dayfirst=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Eerste kolom is geen datum")
    except Exception:
        try:
            if hasattr(file, "seek"):
                file.seek(0)
            df = pd.read_csv(file, sep=";", decimal=",")
            df["Date"] = pd.to_datetime(df.iloc[:, 0], dayfirst=True)
            df = df.set_index("Date")
        except Exception as exc:
            raise DataLoadError(f"Kan het bestand niet inlezen: {exc}") from exc

    df = df.select_dtypes(include=[np.number])
    if df.empty:
        raise DataLoadError("Geen numerieke kolommen gevonden in het bestand.")

    winter_profile = _calculate_winter_profile(df)

    # --- Bruto opwek per aansluiting ---
    connection_cols = df.columns.tolist()
    total_cols = len(connection_cols)
    estimated_volumes: dict[str, float] = {}
    gross_prod_dict: dict[str, pd.Series] = {}

    for i, col in enumerate(connection_cols):
        gross_series = _estimate_gross_solar(df, col, winter_profile)
        gross_prod_dict[col] = gross_series
        estimated_volumes[col] = float(gross_series.sum())

        if on_progress and i % max(1, total_cols // 10) == 0:
            on_progress((i + 1) / total_cols, f"Analyseren: {col}")

    gross_production_df = pd.DataFrame(gross_prod_dict, index=df.index)

    # --- Categorisatie ---
    mapping = _categorize_connections(df, gross_production_df, estimated_volumes)

    cat_consumer = [c for c, cat in mapping.items() if cat == "Consumer"]
    cat_prosumer = [c for c, cat in mapping.items() if cat == "Prosumer"]
    cat_producer = [c for c, cat in mapping.items() if cat == "Producer"]

    agg_df = pd.DataFrame(index=df.index)
    agg_df["Consumer"] = df[cat_consumer].sum(axis=1) if cat_consumer else 0.0
    agg_df["Prosumer"] = df[cat_prosumer].sum(axis=1) if cat_prosumer else 0.0
    agg_df["Producer"] = df[cat_producer].sum(axis=1) if cat_producer else 0.0
    agg_df["Total"] = agg_df["Consumer"] + agg_df["Prosumer"] + agg_df["Producer"]

    return agg_df, mapping


def load_aggregated_csv(file) -> pd.DataFrame:
    """Lees een reeds geaggregeerd CSV-bestand in en valideer het.

    Returns een DataFrame met kolom 'Date' (datetime) en numerieke profielkolommen.

    Raises
    ------
    DataLoadError
        Bij ontbrekende datum-kolom of ongeldige data.
    """
    try:
        df = pd.read_csv(file, sep=";", decimal=",")
    except Exception as exc:
        raise DataLoadError(f"Kan CSV niet inlezen: {exc}") from exc

    # --- Zoek datum-kolom ---
    if "Date" not in df.columns:
        for alias in cfg.DATE_COLUMN_ALIASES:
            if alias in df.columns:
                df.rename(columns={alias: "Date"}, inplace=True)
                break
        if "Date" not in df.columns:
            cols = list(df.columns)
            cols[0] = "Date"
            df.columns = cols

    # --- Numerieke conversie (komma → punt) ---
    for col in ["Consumer", "Prosumer", "Producer", "Total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "."), errors="coerce"
            )

    if "Total" not in df.columns:
        sum_cols = [c for c in ["Consumer", "Prosumer", "Producer"] if c in df.columns]
        df["Total"] = df[sum_cols].sum(axis=1) if sum_cols else 0.0

    try:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    except Exception as exc:
        raise DataLoadError(f"Kan datumkolom niet converteren: {exc}") from exc

    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standaard feature engineering op het hoofd-DataFrame.

    - Sorteren & dedupliceren op datum
    - 15-min frequentie afdwingen
    - MW-kolommen toevoegen
    - Peak-flag en kwartaal toevoegen
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="first")
    df = df.set_index("Date").asfreq(f"{cfg.INTERVAL_MINUTES}min").ffill().reset_index()

    for col in cfg.PROFILE_CHOICES:
        if col in df.columns:
            # kWh per kwartier → MW:  waarde * 4 / 1000
            df[f"{col}_MW"] = (df[col] * 4) / 1000
        else:
            df[f"{col}_MW"] = 0.0

    df["is_peak"] = (
        df["Date"].dt.weekday.isin(cfg.PEAK_WEEKDAYS)
        & (df["Date"].dt.hour >= cfg.PEAK_HOUR_START)
        & (df["Date"].dt.hour < cfg.PEAK_HOUR_END)
    )
    df["Quarter"] = df["Date"].dt.quarter

    return df
