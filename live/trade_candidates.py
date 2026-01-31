# live/trade_candidates.py

from dataclasses import dataclass
from typing import Optional

from live.config_risk import Aggressiveness
from live.position_sizing import (
    AssetClass,
    SignalClass,
    PositionSizingInput,
    PositionSizingResult,
    compute_position_size,
)
from live.signal_classification import (
    ClassifiedSignal,
    classify_signal_from_p,
    SignalQuality,
)


@dataclass
class TradeCandidate:
    """
    Rappresenta un'operazione POTENZIALE generata dalla Coscienza,
    prima del ranking e prima della decisione finale di invio al Coordinatore.
    """
    symbol: str
    timeframe: str

    # info sul segnale
    direction: str                  # "LONG" / "SHORT"
    p_blend: float                  # probabilità grezza del radar
    strength: float                 # |p_blend - 0.5|
    signal_quality: SignalQuality   # A / A_PLUS (per ora)
    signal_class: SignalClass       # A / A_PLUS

    # info sul pricing
    entry_price: float
    stop_price: float

    # info sul sizing
    size: float                     # quantità da tradare
    risk_pct: float                 # frazione di equity usata (es. 0.0025 = 0.25%)
    risk_amount: float              # rischio in USDC
    stop_distance: float            # distanza entry-stop in prezzo
    stop_distance_monetary: float   # rischio monetario totale (distanza * size * value_per_unit)

    # meta
    asset_class: AssetClass
    aggressiveness: Aggressiveness
    equity: float                   # equity usata per il calcolo (snapshot)


def build_trade_candidate(
    *,
    symbol: str,
    timeframe: str,
    equity: float,
    p_blend: float,
    entry_price: float,
    stop_price: float,
    value_per_unit: float,
    asset_class: AssetClass,
    aggressiveness: Aggressiveness,
) -> Optional[TradeCandidate]:
    """
    Costruisce un TradeCandidate a partire da:
      - output del radar (p_blend)
      - info di pricing (entry/stop)
      - info di rischio (equity, asset_class, aggressiveness)

    Passi:
      1) classifica il segnale (A / A_PLUS o NONE)
      2) se NONE -> ritorna None (nessun trade)
      3) calcola la size con il modulo di position sizing
      4) ritorna un TradeCandidate pronto per il ranking
    """

    # 1) Classifichiamo il segnale dal p_blend
    classified: ClassifiedSignal = classify_signal_from_p(p_blend)

    if classified.quality == SignalQuality.NONE or classified.signal_class is None:
        # segnale troppo debole o non valido -> nessun trade
        return None

    # 2) Prepariamo l'input per il calcolo della size
    sizing_input = PositionSizingInput(
        equity=equity,
        entry_price=entry_price,
        stop_price=stop_price,
        value_per_unit=value_per_unit,
        asset_class=asset_class,
        aggressiveness=aggressiveness,
        signal_class=classified.signal_class,
    )

    # 3) Calcoliamo la size
    sizing_result: PositionSizingResult = compute_position_size(sizing_input)

    # 4) Costruiamo il TradeCandidate
    candidate = TradeCandidate(
        symbol=symbol,
        timeframe=timeframe,
        direction=classified.direction,
        p_blend=classified.p_blend,
        strength=classified.strength,
        signal_quality=classified.quality,
        signal_class=classified.signal_class,
        entry_price=entry_price,
        stop_price=stop_price,
        size=sizing_result.size,
        risk_pct=sizing_result.risk_pct,
        risk_amount=sizing_result.risk_amount,
        stop_distance=sizing_result.stop_distance,
        stop_distance_monetary=sizing_result.stop_distance_monetary,
        asset_class=asset_class,
        aggressiveness=aggressiveness,
        equity=equity,
    )

    return candidate
