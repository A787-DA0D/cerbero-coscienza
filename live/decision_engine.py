# live/decision_engine.py

from dataclasses import dataclass
from typing import List

from live.config_risk import Aggressiveness
from live.position_sizing import AssetClass
from live.trade_candidates import TradeCandidate, build_trade_candidate
from live.ranking import RankedTrade, rank_trade_candidates


@dataclass
class RawSignalInput:
    """
    Input "grezzo" per un simbolo/timeframe in un singolo giro di live loop.
    Qui arrivano i dati dal radar + feed prezzi.
    """
    symbol: str
    timeframe: str

    p_blend: float            # probabilità blend del radar (0..1)
    entry_price: float        # prezzo di ingresso che useresti se apri ora
    stop_price: float         # prezzo dello stop loss suggerito (ATR/swing)
    value_per_unit: float     # valore in USDC di 1 unità (es. 1 coin, 1 contratto, ecc.)
    asset_class: AssetClass   # "fx" / "index" / "crypto" / "other"

    # placeholder che in futuro verranno da moduli dedicati:
    mtf_score: float = 1.0            # 1.0 = neutro, >1 meglio, <1 peggio
    market_quality_score: float = 1.0 # 1.0 = neutro, >1 meglio, <1 peggio


def propose_trades_for_batch(
    *,
    signals: List[RawSignalInput],
    equity: float,
    aggressiveness: Aggressiveness,
) -> List[RankedTrade]:
    """
    Prende una lista di RawSignalInput (uno per simbolo/timeframe che sta generando un segnale)
    e restituisce una lista di RankedTrade ordinata dal migliore al peggiore.

    Passi:
      1) converte ogni RawSignalInput in TradeCandidate (usando build_trade_candidate)
         - se il segnale non è almeno A/A+ -> viene scartato (None)
      2) calcola un rank su tutti i candidati validi
      3) ritorna la lista dei RankedTrade ordinati per score decrescente

    NOTA:
      Qui NON applichiamo ancora i limiti di portafoglio (max rischio totale, cluster, ecc.),
      che verranno gestiti in uno strato successivo (Portfolio Risk Engine).
    """

    candidates: List[TradeCandidate] = []
    mtf_scores: List[float] = []
    market_scores: List[float] = []

    for s in signals:
        candidate = build_trade_candidate(
            symbol=s.symbol,
            timeframe=s.timeframe,
            equity=equity,
            p_blend=s.p_blend,
            entry_price=s.entry_price,
            stop_price=s.stop_price,
            value_per_unit=s.value_per_unit,
            asset_class=s.asset_class,
            aggressiveness=aggressiveness,
        )

        if candidate is None:
            # segnale troppo debole o non valido -> lo ignoriamo
            continue

        candidates.append(candidate)
        mtf_scores.append(s.mtf_score)
        market_scores.append(s.market_quality_score)

    if not candidates:
        return []

    # ranking vero e proprio
    ranked = rank_trade_candidates(
        candidates=candidates,
        mtf_scores=mtf_scores,
        market_quality_scores=market_scores,
    )

    return ranked
