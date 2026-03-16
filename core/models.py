"""
Typed data structures — vervangen losse tuples en verspreide variabelen.
Alle modules communiceren via deze dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Hedge positie (Base + Peak MW per periode)
# ---------------------------------------------------------------------------
@dataclass
class HedgePosition:
    """Eén combinatie van Base- en Peak-inkoop in MW."""

    base_mw: float = 0.0
    peak_add_mw: float = 0.0


# ---------------------------------------------------------------------------
# Volumetrische resultaten
# ---------------------------------------------------------------------------
@dataclass
class HedgeResults:
    """Samenvatting van de volumebalans na toepassing van de hedge."""

    profile_mwh: float = 0.0
    hedge_mwh: float = 0.0
    over_hedge_mwh: float = 0.0
    under_hedge_mwh: float = 0.0
    used_hedge_mwh: float = 0.0
    hedge_efficiency_pct: float = 0.0


# ---------------------------------------------------------------------------
# Financieel overzicht
# ---------------------------------------------------------------------------
@dataclass
class FinancialSummary:
    """Totale financiële uitkomst van de hedge-strategie."""

    total_hedge_cost: float = 0.0
    net_spot_result: float = 0.0
    total_energy_cost: float = 0.0
    avg_cost_per_mwh: float = 0.0

    # Unit economics
    cost_per_used_mwh: float = 0.0
    capture_price: float = 0.0
    capture_diff: float = 0.0
    avg_spot_buy: float = 0.0
    avg_spot_sell: float = 0.0


# ---------------------------------------------------------------------------
# Periodes (elimineert alle Jaar/Kwartaal branching)
# ---------------------------------------------------------------------------
@dataclass
class Period:
    """Eén tijdperiode — 'Jaar' of 'Q1'…'Q4'.

    ``mask`` is een boolean Series die op het hoofd-DataFrame past.
    """

    label: str                          # "Jaar" of "Q1"/"Q2"/"Q3"/"Q4"
    quarter: Optional[int] = None       # None voor jaar, 1-4 voor kwartalen
    mask: Optional[pd.Series] = field(default=None, repr=False)

    @property
    def is_yearly(self) -> bool:
        return self.quarter is None


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class DataLoadError(Exception):
    """Fout bij het inlezen of verwerken van meetdata."""


class EPEXFetchError(Exception):
    """Fout bij het ophalen van EPEX spotprijzen."""


class PricingError(Exception):
    """Fout bij het laden van contractprijzen."""


# ---------------------------------------------------------------------------
# Scenario-vergelijking (4 optimalisaties)
# ---------------------------------------------------------------------------
@dataclass
class ScenarioResult:
    """Resultaat van één optimalisatie-strategie."""

    optimization: str     # "value_hedge" | "max_hedge" | "min_hedge" | "least_cost"
    position: HedgePosition = field(default_factory=HedgePosition)
    results: HedgeResults = field(default_factory=HedgeResults)
    financial: FinancialSummary = field(default_factory=FinancialSummary)


@dataclass
class CategoryScenarios:
    """Alle scenario's voor één categorie (Consumer/Prosumer/Producer)."""

    category: str
    scenarios: list[ScenarioResult] = field(default_factory=list)
