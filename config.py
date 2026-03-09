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
