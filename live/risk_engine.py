# live/risk_engine.py
"""
Risk Engine per Cerbero Coscienza v3 PRO.

Calcola il rischio per singolo trade in base a:
- equity (small / intermedio / standard),
- asset class del simbolo,
- classe del segnale (A, A_PLUS, B),
- rischio totale già aperto sul conto.

Per ora non calcola ancora la size in lotti/contratti; ritorna
solo la PERCENTUALE DI RISCHIO sull'equity (es. 0.003 = 0.3%).
"""

from dataclasses import dataclass
from typing import Literal, Tuple

from live.config_v3 import (
    ASSET_CLASS,
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
    MAX_TOTAL_OPEN_RISK,
)

Tier = Literal["small", "intermediate", "standard"]
SignalClass = Literal["A_PLUS", "A", "B"]  # B per futuro, ora quasi disabilitata


@dataclass
class RiskDecision:
    risk_pct: float          # es. 0.003 = 0.3% dell'equity
    allowed: bool            # False = non aprire trade
    reason: str              # descrizione umana (log / debug)


def get_account_tier(equity: float) -> Tier:
    """
    Determina il tier dell'account in base all'equity.
    """
    if equity < SMALL_ACCOUNT_MAX:
        return "small"
    elif equity < INTERMEDIATE_ACCOUNT_MAX:
        return "intermediate"
    else:
        return "standard"


def get_asset_class(symbol: str) -> str:
    """
    Restituisce la asset class di un simbolo (fx, jpy, metal, crypto, index, commodity, ...).
    Di default, se non mappato, considera 'fx'.
    """
    return ASSET_CLASS.get(symbol, "fx")


def base_risk_for(symbol: str, equity: float) -> float:
    """
    Rischio base (prima della classe segnale) per un dato simbolo + equity.
    """
    tier = get_account_tier(equity)
    ac = get_asset_class(symbol)

    # FX-like (fx, index, commodity, dollar index, ecc.)
    if ac in ("fx", "index", "commodity"):
        if tier == "small":
            return RISK_BASE_FX_SMALL
        elif tier == "intermediate":
            return RISK_BASE_FX_INTERMEDIATE
        else:
            return RISK_BASE_FX

    # JPY pair (più nervosi)
    if ac == "jpy":
        if tier == "small":
            return RISK_BASE_JPY_SMALL
        elif tier == "intermediate":
            return RISK_BASE_JPY_INTERMEDIATE
        else:
            return RISK_BASE_JPY

    # Metalli (oro/argento)
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

    # Fallback: tratta come FX
    if tier == "small":
        return RISK_BASE_FX_SMALL
    elif tier == "intermediate":
        return RISK_BASE_FX_INTERMEDIATE
    else:
        return RISK_BASE_FX


def class_multiplier(signal_class: SignalClass) -> float:
    """
    Moltiplicatore in base alla classe del segnale.
    Per ora:
    - A_PLUS: leggermente sopra il base (1.2x),
    - A: base (1.0x),
    - B: ridotto (0.7x).
    """
    if signal_class == "A_PLUS":
        return 1.2
    if signal_class == "B":
        return 0.7
    return 1.0  # A


def compute_risk_for_trade(
    equity: float,
    symbol: str,
    signal_class: SignalClass,
    total_open_risk: float,
) -> RiskDecision:
    """
    Calcola la percentuale di rischio da assegnare a un nuovo trade.

    Parametri:
    - equity: equity account (USDC).
    - symbol: es. 'EURUSD'.
    - signal_class: 'A', 'A_PLUS', 'B'.
    - total_open_risk: somma dei rischi % di tutti i trade aperti (es. 0.025 = 2.5%).

    Ritorna:
    - RiskDecision(risk_pct, allowed, reason)
    """

    # 1) Se siamo già oltre il cap di rischio totale → no nuovi trade
    if total_open_risk >= MAX_TOTAL_OPEN_RISK:
        return RiskDecision(
            risk_pct=0.0,
            allowed=False,
            reason=f"Total open risk {total_open_risk:.4f} >= cap {MAX_TOTAL_OPEN_RISK:.4f}",
        )

    # 2) Rischio base per simbolo + tier
    base_risk = base_risk_for(symbol, equity)

    # 3) Applica moltiplicatore in base alla classe del segnale
    mult = class_multiplier(signal_class)
    risk = base_risk * mult

    # 4) Applica hard cap per trade
    if risk > MAX_RISK_PER_TRADE:
        risk = MAX_RISK_PER_TRADE

    # 5) Non superare il cap totale aperto (somma)
    room = MAX_TOTAL_OPEN_RISK - total_open_risk
    if risk > room:
        # se lo spazio residuo è troppo piccolo (< metà del base) → meglio non aprire
        if room < base_risk * 0.5:
            return RiskDecision(
                risk_pct=0.0,
                allowed=False,
                reason=f"Room residuo {room:.4f} troppo basso per aprire nuovo trade",
            )
        risk = room

    # 6) Controllo finale: niente rischio negativo o ridicolo
    if risk <= 0.0:
        return RiskDecision(
            risk_pct=0.0,
            allowed=False,
            reason="Rischio calcolato <= 0",
        )

    return RiskDecision(
        risk_pct=risk,
        allowed=True,
        reason=f"OK (tier={get_account_tier(equity)}, asset_class={get_asset_class(symbol)}, class={signal_class})",
    )
