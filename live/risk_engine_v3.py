# live/risk_engine_v3.py
"""
Risk Engine v3 per Cerbero Coscienza.

- Usa i parametri definiti in live.config_v3:
  - tier di equity (small / intermediate / standard)
  - rischio base per asset class (fx / jpy / metal / crypto / commodity / index)
  - MAX_RISK_PER_TRADE, MAX_TOTAL_OPEN_RISK

- Espone una funzione principale:
    compute_position_size(...)

  pensata per essere robusta:
  accetta sia parametri nominati (symbol=..., equity=..., sl_points=..., current_open_risk_pct=..., clazz=...)
  sia i primi 3 come posizionali.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from live.config_v3 import (
    SMALL_ACCOUNT_MAX,
    INTERMEDIATE_ACCOUNT_MAX,
    RISK_BASE_FX,
    RISK_BASE_FX_INTERMEDIATE,
    RISK_BASE_FX_SMALL,
    RISK_BASE_CRYPTO,
    RISK_BASE_CRYPTO_INTERMEDIATE,
    RISK_BASE_CRYPTO_SMALL,
    RISK_BASE_JPY,
    RISK_BASE_JPY_INTERMEDIATE,
    RISK_BASE_JPY_SMALL,
    RISK_BASE_METALS,
    RISK_BASE_METALS_INTERMEDIATE,
    RISK_BASE_METALS_SMALL,
    MAX_RISK_PER_TRADE,
    MAX_RISK_PER_TRADE_BY_TIER,
    MAX_TOTAL_OPEN_RISK,
    ASSET_CLASS,
)


@dataclass
class PositionSizingResult:
    symbol: str
    equity: float
    risk_pct: float          # percentuale di equity allocata su questo trade (0–0.01…)
    risk_amount: float       # USD (USDC) a rischio
    sl_distance: float       # distanza monetaria usata per il calcolo
    size: float              # "units" / notional = risk_amount / sl_distance
    skip: bool               # True se non apriamo il trade
    reason: str              # motivazione (utile per logging)


def _get_equity_tier(equity: float) -> str:
    """
    Restituisce: 'small', 'intermediate', 'standard'
    """
    if equity < SMALL_ACCOUNT_MAX:
        return "small"
    elif equity < INTERMEDIATE_ACCOUNT_MAX:
        return "intermediate"
    else:
        return "standard"


def _base_risk_for_asset(asset_class: str, tier: str) -> float:
    """
    Ritorna il rischio base (in % dell'equity, es. 0.0025) per asset class e tier.
    """
    ac = asset_class.lower()

    # FX + indici → usiamo lo stesso schema FX
    if ac in ("fx", "index"):
        if tier == "small":
            return RISK_BASE_FX_SMALL
        elif tier == "intermediate":
            return RISK_BASE_FX_INTERMEDIATE
        else:
            return RISK_BASE_FX

    # JPY cross
    if ac == "jpy":
        if tier == "small":
            return RISK_BASE_JPY_SMALL
        elif tier == "intermediate":
            return RISK_BASE_JPY_INTERMEDIATE
        else:
            return RISK_BASE_JPY

    # Metalli
    if ac == "metal":
        if tier == "small":
            return RISK_BASE_METALS_SMALL
        elif tier == "intermediate":
            return RISK_BASE_METALS_INTERMEDIATE
        else:
            return RISK_BASE_METALS

    # Crypto
    if ac == "crypto":
        if tier == "small":
            return RISK_BASE_CRYPTO_SMALL
        elif tier == "intermediate":
            return RISK_BASE_CRYPTO_INTERMEDIATE
        else:
            return RISK_BASE_CRYPTO

    # Commodity generica (es. LIGHTCMDUSD) → per ora usiamo schema FX
    if ac == "commodity":
        if tier == "small":
            return RISK_BASE_FX_SMALL
        elif tier == "intermediate":
            return RISK_BASE_FX_INTERMEDIATE
        else:
            return RISK_BASE_FX

    # Fallback ultra-sicuro
    if tier == "small":
        return RISK_BASE_FX_SMALL
    elif tier == "intermediate":
        return RISK_BASE_FX_INTERMEDIATE
    else:
        return RISK_BASE_FX


def compute_position_size(
    *args,
    **kwargs,
) -> PositionSizingResult:
    """
    Funzione principale di sizing.

    Interpreta gli argomenti in modo robusto:

    Posizionali (se usati):
      0 -> symbol
      1 -> equity
      2 -> sl_distance (o sl_points)

    Keyword supportate:
      symbol: str
      equity: float
      sl_distance: float     # distanza monetaria tra entry e stop
      sl_points: float       # alias di sl_distance
      current_open_risk_pct: float  # somma risk% trade aperti (0–0.04, ecc.)
      clazz: str | None      # "A" / "A+" / "B" (se vuoi in futuro modulare il rischio)

    Ritorna un PositionSizingResult. Se skip=True, size=0.
    """
    # --- parsing robusto degli argomenti --- #
    symbol = kwargs.get("symbol")
    equity = kwargs.get("equity")
    sl_distance = kwargs.get("sl_distance", kwargs.get("sl_points"))
    current_open_risk_pct = kwargs.get("current_open_risk_pct", 0.0)
    aggressiveness = kwargs.get("aggressiveness", "NORMAL")
    clazz: Optional[str] = kwargs.get("clazz")

    # fallback ai posizionali, se mancanti
    if symbol is None and len(args) > 0:
        symbol = args[0]
    if equity is None and len(args) > 1:
        equity = args[1]
    if sl_distance is None and len(args) > 2:
        sl_distance = args[2]

    symbol = str(symbol) if symbol is not None else "UNKNOWN"
    try:
        equity = float(equity)
    except Exception:
        equity = 0.0
    try:
        sl_distance = float(sl_distance)
    except Exception:
        sl_distance = 0.0
    try:
        current_open_risk_pct = float(current_open_risk_pct)
    except Exception:
        current_open_risk_pct = 0.0

    # --- controlli base --- #
    if equity <= 0:
        return PositionSizingResult(
            symbol=symbol,
            equity=equity,
            risk_pct=0.0,
            risk_amount=0.0,
            sl_distance=sl_distance,
            size=0.0,
            skip=True,
            reason="Equity non valida o zero",
        )

    if sl_distance <= 0:
        return PositionSizingResult(
            symbol=symbol,
            equity=equity,
            risk_pct=0.0,
            risk_amount=0.0,
            sl_distance=sl_distance,
            size=0.0,
            skip=True,
            reason="SL distance <= 0 (stop mancante o non valido)",
        )

    if current_open_risk_pct >= MAX_TOTAL_OPEN_RISK:
        return PositionSizingResult(
            symbol=symbol,
            equity=equity,
            risk_pct=0.0,
            risk_amount=0.0,
            sl_distance=sl_distance,
            size=0.0,
            skip=True,
            reason="Rischio totale già al limite (MAX_TOTAL_OPEN_RISK)",
        )

    # --- determina asset class e tier --- #
    asset_class = ASSET_CLASS.get(symbol.upper(), "fx")
    tier = _get_equity_tier(equity)
    base_risk_pct = _base_risk_for_asset(asset_class, tier)

    # --- modulazione per classe segnale (A / A+ / B) --- #
    # Per ora: A = 1.0, A+ = 1.3, B = 0.7 (se mai la useremo)
    class_mult = 1.0
    if clazz == "A+":
        class_mult = 1.3
    elif clazz == "B":
        class_mult = 0.7


    # --- aggressiveness (NORMAL / AGGRESSIVE) ---
    ag = str(aggressiveness).upper() if aggressiveness is not None else "NORMAL"
    ag_mult = 2.0 if ag in ("AGGRESSIVE", "2", "X2", "TRUE", "1") else 1.0

    risk_pct = base_risk_pct * class_mult * ag_mult
    # clamp all'hard cap per trade (tier-aware)
    try:
        cap = float(MAX_RISK_PER_TRADE_BY_TIER.get(tier, MAX_RISK_PER_TRADE))
    except Exception:
        cap = float(MAX_RISK_PER_TRADE)
    if risk_pct > cap:
        risk_pct = cap

    # controlla che somma open_risk + questo trade non superi il limite totale
    if current_open_risk_pct + risk_pct > MAX_TOTAL_OPEN_RISK:
        return PositionSizingResult(
            symbol=symbol,
            equity=equity,
            risk_pct=0.0,
            risk_amount=0.0,
            sl_distance=sl_distance,
            size=0.0,
            skip=True,
            reason="Aprendo questo trade supereremmo MAX_TOTAL_OPEN_RISK",
        )

    # --- calcolo size --- #
    risk_amount = equity * risk_pct          # es. 0.003 * 10,000 = 30 USDC
    size = risk_amount / sl_distance        # notional = risk / SL

    return PositionSizingResult(
        symbol=symbol,
        equity=equity,
        risk_pct=risk_pct,
        risk_amount=risk_amount,
        sl_distance=sl_distance,
        size=size,
        skip=False,
        reason="OK",
    )
