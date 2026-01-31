# live/config_v3.py
"""
Configurazione ufficiale Cerbero Coscienza v3 PRO.
Tutte le soglie e i parametri di rischio / classi segnali
sono centralizzati qui.
"""

# === Universe simboli & TF ===
SYMBOLS = [
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCHF", "USDCAD",
    "EURJPY", "AUDJPY", "CADJPY", "GBPJPY",
    "XAUUSD", "XAGUSD", "LIGHTCMDUSD",
    "DOLLARIDXUSD",
    "BTCUSD", "ETHUSD",
]

TIMEFRAMES = ["5m", "15m", "4h", "1d"]

# === Classi segnali ===
# strength = |p_blend - 0.5|
STRENGTH_MIN = 0.02      # sotto è rumore → nessun segnale
STRENGTH_B   = 0.05      # B da 0.05 a 0.10
STRENGTH_A   = 0.10      # A da 0.10 in su
STRENGTH_A_PLUS = 0.20   # A+ super élite (possiamo alzarlo/abbassarlo dopo)

# Per ora: operiamo SOLO con A e A+
ENABLE_CLASS_B = False   # prima settimana: solo A / A+

# === Rischio account tiers ===
# equity in USDC
SMALL_ACCOUNT_MAX        = 300.0   # < 300 → small boost forte
INTERMEDIATE_ACCOUNT_MAX = 600.0   # 300–600 → intermedio
# >= 600 → standard

# Rischio base per asset class (in % dell'equity)
RISK_BASE_FX                 = 0.25 / 100.0
RISK_BASE_FX_INTERMEDIATE    = 0.35 / 100.0
RISK_BASE_FX_SMALL           = 0.50 / 100.0

RISK_BASE_CRYPTO             = 0.35 / 100.0
RISK_BASE_CRYPTO_INTERMEDIATE = 0.45 / 100.0
RISK_BASE_CRYPTO_SMALL       = 0.75 / 100.0  # max 1%, partiamo più conservativi

# JPY / oro (asset più volatili) leggermente più bassi
RISK_BASE_JPY                = 0.20 / 100.0
RISK_BASE_JPY_INTERMEDIATE   = 0.30 / 100.0
RISK_BASE_JPY_SMALL          = 0.50 / 100.0

RISK_BASE_METALS             = 0.20 / 100.0
RISK_BASE_METALS_INTERMEDIATE = 0.30 / 100.0
RISK_BASE_METALS_SMALL       = 0.50 / 100.0

# Hard cap per trade (qualsiasi asset)
# Hard cap per trade (tier-aware, qualsiasi asset)
  # NOTA: questi cap sono pensati per consentire AGGRESSIVE x2 senza clamp “random”.
MAX_RISK_PER_TRADE_SMALL        = 2.00 / 100.0   # 2.00% cap per trade (SMALL)
MAX_RISK_PER_TRADE_INTERMEDIATE = 1.00 / 100.0   # 1.00% cap per trade (INTERMEDIATE)
MAX_RISK_PER_TRADE_STANDARD     = 0.70 / 100.0   # 0.70% cap per trade (STANDARD)
# Cap assoluto (safety net)
MAX_RISK_PER_TRADE = MAX_RISK_PER_TRADE_STANDARD   # alias safety-net (fallback)

# Tier-aware cap per trade (usato dal risk engine)
MAX_RISK_PER_TRADE_BY_TIER = {
    'small': MAX_RISK_PER_TRADE_SMALL,
    'intermediate': MAX_RISK_PER_TRADE_INTERMEDIATE,
    'standard': MAX_RISK_PER_TRADE_STANDARD,
}

# Max rischio totale aperto
MAX_TOTAL_OPEN_RISK = 4.0 / 100.0   # tetto 4%

# === Drawdown / Aggressività ===
DD_SOFT     = 4.0 / 100.0   # drawdown 7gg per entrare in CONSERVATIVA
DD_HARD_DAY = 5.0 / 100.0   # per considerare lockdown hard/manual review

# === Mapping asset class (per rischi diversi) ===
ASSET_CLASS = {
    "EURUSD": "fx",
    "GBPUSD": "fx",
    "AUDUSD": "fx",
    "USDJPY": "jpy",
    "USDCHF": "fx",
    "USDCAD": "fx",
    "EURJPY": "jpy",
    "AUDJPY": "jpy",
    "CADJPY": "jpy",
    "GBPJPY": "jpy",

    "XAUUSD": "metal",
    "XAGUSD": "metal",
    "LIGHTCMDUSD": "commodity",

    "DOLLARIDXUSD": "index",  # trattato come FX “macro” nel risk engine

    "BTCUSD": "crypto",
    "ETHUSD": "crypto",
}

# === Flag generali ===
TELEGRAM_ALERTS_ENABLED = True
