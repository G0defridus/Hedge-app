"""
ENTSO-E API client voor EPEX Day-Ahead spotprijzen (zone NL).

Geen Streamlit-afhankelijkheden — caching en spinners worden door de UI-laag afgehandeld.
"""

from __future__ import annotations

import logging

import pandas as pd

from core.models import EPEXFetchError

logger = logging.getLogger(__name__)


def fetch_epex_prices(
    api_key: str,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
) -> pd.DataFrame:
    """Haal uurlijkse Day-Ahead prijzen op via de ENTSO-E Transparency Platform.

    Parameters
    ----------
    api_key : str
        Geldige ENTSO-E API key.
    start_date, end_date : Timestamp-achtig
        Begin- en einddatum van de gewenste range.

    Returns
    -------
    pd.DataFrame
        Kolommen: ``Date_Hour`` (tz-naïef, Europe/Amsterdam) en ``EPEX_EUR_MWh``.

    Raises
    ------
    EPEXFetchError
        Bij een API-fout, netwerk-probleem of ongeldige key.
    """
    try:
        from entsoe import EntsoePandasClient
    except ImportError as exc:
        raise EPEXFetchError(
            "De entsoe-py library is niet geïnstalleerd. "
            "Voeg 'entsoe-py' toe aan requirements.txt."
        ) from exc

    try:
        client = EntsoePandasClient(api_key=api_key)

        start = pd.Timestamp(start_date)
        if start.tzinfo is None:
            start = start.tz_localize("Europe/Amsterdam")

        end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        if end.tzinfo is None:
            end = end.tz_localize("Europe/Amsterdam")

        ts = client.query_day_ahead_prices("NL", start=start, end=end)
        df_epex = ts.to_frame("EPEX_EUR_MWh")

        # Maak tz-naïef voor eenvoudige merge met het hoofd-DataFrame
        df_epex["Date_Hour"] = df_epex.index.tz_localize(None)
        df_epex = df_epex.drop_duplicates(subset="Date_Hour", keep="first")

        logger.info(
            "EPEX prijzen opgehaald: %s → %s (%d uurwaarden)",
            start.date(),
            end.date(),
            len(df_epex),
        )
        return df_epex

    except EPEXFetchError:
        raise
    except Exception as exc:
        raise EPEXFetchError(f"Kon EPEX-prijzen niet ophalen: {exc}") from exc
