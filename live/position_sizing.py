# live/position_sizing.py

from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

from live.config_risk import (
    compute_risk_per_trade,
    Aggressiveness,
)


# Tipo semplice per asset_class (per ora ci basta sapere se è crypto o no)
AssetClass = Literal["fx", "index", "crypto", "other"]


class SignalClass(Enum):
    """
    Classi di qualità del segnale.
    Per la prima settimana useremo SOLO:
      - A
      - A_PLUS
    (eventuale B_PLUS potrà essere aggiunta in futuro).
    """
    A = auto()
    A_PLUS = auto()
    # B_PLUS = auto()  # se in futuro vorrai riattivarla


# Moltiplicatori di rischio per classe del segnale.
# Questi agiscono SOPRA al rischio base (profilo + aggressività),
# ma restano SEMPRE limitati dall'hard_cap_per_trade.
SIGNAL_CLASS_RISK_MULTIPLIER = {
    SignalClass.A: 1.0,       # usa il rischio base
    SignalClass.A_PLUS: 1.2,  # spinge un po' di più sui segnali élite
    # SignalClass.B_PLUS: 0.8,  # esempio futuro: segnali B+ con rischio ridotto
}


@dataclass
class PositionSizingInput:
    equity: float                   # equity attuale in USDC
    entry_price: float              # prezzo di ingresso
    stop_price: float               # prezzo dello stop loss
    value_per_unit: float           # valore in USDC di 1 unità del simbolo (es. 1 coin, 1 contratto, 1 lotto)
    asset_class: AssetClass         # "fx" / "index" / "crypto" / "other"
    aggressiveness: Aggressiveness  # NORMAL / NORMAL / AGGRESSIVE
    signal_class: SignalClass       # A / A_PLUS (per ora)


@dataclass
class PositionSizingResult:
    size: float                     # quantità da tradare (in "unità" corrispondenti a value_per_unit)
    risk_pct: float                 # rischio effettivo usato (frazione di equity, es. 0.0025 = 0.25%)
    risk_amount: float              # rischio in USDC
    stop_distance: float            # distanza prezzo (entry - stop) in unità di prezzo
    stop_distance_monetary: float   # distanza monetaria TOTALE (in USDC, già moltiplicata per la size)


def is_crypto_asset(asset_class: AssetClass) -> bool:
    """
    Utility semplice per capire se l'asset è crypto.
    """
    return asset_class == "crypto"


def compute_position_size(params: PositionSizingInput) -> PositionSizingResult:
    """
    Calcola la size del trade in base a:
      - equity
      - distanza dello stop
      - asset class (crypto vs non crypto)
      - stato di aggressività
      - classe del segnale (A / A_PLUS)

    La pipeline è:
      1) otteniamo il rischio base per trade (profilo account + aggressività)
      2) applichiamo un moltiplicatore in base alla classe del segnale
      3) ci assicuriamo di non superare l'hard cap per trade
      4) calcoliamo la size a partire dal rischio in USDC e dalla distanza dello stop
    """

    equity = params.equity

    # 1) Calcoliamo il rischio per trade in % (frazione di equity) dal profilo + aggressività
    base_risk_pct, hard_cap = compute_risk_per_trade(
        equity=equity,
        is_crypto=is_crypto_asset(params.asset_class),
        aggressiveness=params.aggressiveness,
    )

    # 2) Applichiamo il moltiplicatore della classe del segnale (A / A_PLUS)
    class_multiplier = SIGNAL_CLASS_RISK_MULTIPLIER.get(params.signal_class, 1.0)
    raw_risk_pct = base_risk_pct * class_multiplier

    # 3) Non superare mai l'hard cap definito dal profilo
    risk_pct = min(raw_risk_pct, hard_cap)

    # 4) Calcoliamo la distanza dello stop
    stop_distance = abs(params.entry_price - params.stop_price)

    if stop_distance <= 0:
        raise ValueError("stop_distance <= 0: controlla entry_price e stop_price")

    # 5) Convertiamo in distanza monetaria per 1 unità (es. 1 coin, 1 contratto, ecc.)
    stop_distance_monetary_per_unit = stop_distance * params.value_per_unit

    if stop_distance_monetary_per_unit <= 0:
        raise ValueError("stop_distance_monetary_per_unit <= 0: controlla value_per_unit")

    # 6) Rischio in USDC per trade
    risk_amount = equity * risk_pct

    # 7) Size = rischio / distanza monetaria per unità
    raw_size = risk_amount / stop_distance_monetary_per_unit

    if raw_size <= 0:
        raise ValueError("raw_size <= 0: qualcosa non torna nel calcolo della size")

    size = raw_size  # gli arrotondamenti specifici del broker/smart contract saranno applicati a valle

    return PositionSizingResult(
        size=size,
        risk_pct=risk_pct,
        risk_amount=risk_amount,
        stop_distance=stop_distance,
        stop_distance_monetary=stop_distance_monetary_per_unit * size,
    )
