"""
Centrale configuratie: alle constanten, drempels, kleuren en standaardwaarden.
Eén bron van waarheid — niets is hardcoded in andere modules.
"""

# ---------------------------------------------------------------------------
# Tijdresolutie
# ---------------------------------------------------------------------------
INTERVAL_MINUTES = 15
MWH_FACTOR = INTERVAL_MINUTES / 60  # 0.25 — conversie kW→kWh of MW→MWh per interval

# ---------------------------------------------------------------------------
# Categorisatie drempels (data_processor)
# ---------------------------------------------------------------------------
GROSS_SOLAR_THRESHOLD_KWH = 1000       # Onder dit niveau → Consumer
PRODUCER_IMPORT_RATIO = 0.20           # Netto import < 20% van bruto opwek → Producer
SOLAR_CORRELATION_MIN = 0.85           # Pearson r ≥ 0.85 tegen ideale zonnecurve
DAYLIGHT_START_HOUR = 8
DAYLIGHT_END_HOUR = 20
NIGHT_HOUR_RANGES = ((0, 6), (23, 24)) # Nachturen voor baseline-berekening
NIGHT_BASELINE_FLOOR = 0.05            # Minimale nacht-baseline (voorkomt deling door 0)
SCALING_CLIP_MIN = 0.2
SCALING_CLIP_MAX = 5.0

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
GRID_SEARCH_STEPS = 40                 # 40×40 = 1.600 scenario's per periode
VOLUME_SEARCH_MAX_PCT = 150            # Zoekbereik: 150% → 1%

# ---------------------------------------------------------------------------
# Standaard contractprijzen (€/MWh) — fallback als er geen CSV is
# ---------------------------------------------------------------------------
DEFAULT_PRICES = {
    "year": {"base": 80.0, "peak": 95.0},
    "Q1":   {"base": 90.0, "peak": 110.0},
    "Q2":   {"base": 65.0, "peak": 75.0},
    "Q3":   {"base": 70.0, "peak": 80.0},
    "Q4":   {"base": 85.0, "peak": 105.0},
}

# Seizoensverhoudingen per kwartaal (voor schaling van jaarlijkse CSV-prijzen)
QUARTERLY_BASE_WEIGHTS = {1: 90, 2: 65, 3: 70, 4: 85}
QUARTERLY_PEAK_WEIGHTS = {1: 110, 2: 75, 3: 80, 4: 105}

# ---------------------------------------------------------------------------
# Peak-definitie (Endex NL: Ma-Vr 08:00–20:00)
# ---------------------------------------------------------------------------
PEAK_WEEKDAYS = range(0, 5)  # maandag=0 t/m vrijdag=4
PEAK_HOUR_START = 8
PEAK_HOUR_END = 20

# ---------------------------------------------------------------------------
# Censo huisstijl — kleuren
# ---------------------------------------------------------------------------
COLORS = {
    "black":      "#000000",
    "gold":       "#fab517",
    "gold_hover": "#d99d12",
    "ruby":       "#e8327c",
    "white":      "#ffffff",
    "grey":       "#9e9e9e",
    "dark_grey":  "#808080",
}

# ---------------------------------------------------------------------------
# Seizoensgrafieken — dynamisch op basis van ISO-weeknummer
# ---------------------------------------------------------------------------
SEASONAL_WEEKS = [
    {"name": "Typische winterweek", "iso_week": 6},
    {"name": "Typische lenteweek",  "iso_week": 19},
    {"name": "Typische zomerweek",  "iso_week": 32},
    {"name": "Typische herfstweek", "iso_week": 45},
]

# ---------------------------------------------------------------------------
# Profielkeuzes
# ---------------------------------------------------------------------------
PROFILE_CHOICES = ["Consumer", "Prosumer", "Producer", "Total"]

# ---------------------------------------------------------------------------
# Date-kolom aliassen bij inlezen
# ---------------------------------------------------------------------------
DATE_COLUMN_ALIASES = ["Datum", "Tijd", "Time", "date", "time"]

# ---------------------------------------------------------------------------
# Verkoopproducten (3 varianten)
# ---------------------------------------------------------------------------
PRODUCTS = [
    {"key": "max_zekerheid", "label": "Max Zekerheid",
     "desc": "Koop zoveel mogelijk in via vaste ENDEX-blokken"},
    {"key": "min_hedge", "label": "Minimale Hedge",
     "desc": "Dek slechts ~10% van je volume af, rest via spot"},
    {"key": "flex", "label": "Volledig Flex",
     "desc": "Geen vaste inkoop — 100% op de spotmarkt"},
]

MIN_HEDGE_VOLUME_PCT = 10  # Doelpercentage voor "Minimale Hedge"

# ---------------------------------------------------------------------------
# Optimalisatie-strategieën (4 opties)
# ---------------------------------------------------------------------------
OPTIMIZATIONS = [
    {"key": "least_cost",
     "label": "Laagste kosten",
     "desc": "Test 1.600 combinaties en kiest de goedkoopste mix van vaste inkoop + spotmarkt."},
    {"key": "value_risk",
     "label": "Value Hedge",
     "desc": "Weegt je inkoop naar prijsniveau: meer MW in dure uren, minder in goedkope. Elimineert spotexposure bij 100% dekking."},
    {"key": "max_5pct",
     "label": "Max 5% ENDEX verkoop",
     "desc": "Koop zoveel mogelijk in, maar maximaal 5% terugverkopen op de spotmarkt."},
    {"key": "100vol",
     "label": "100% volume",
     "desc": "Dek het gemiddelde profiel exact af: Base = off-peak gemiddelde, Peak = piekgemiddelde."},
]

# Categorieën voor de vergelijkingstabel (zonder 'Total')
CATEGORY_CHOICES = ["Consumer", "Prosumer", "Producer"]
