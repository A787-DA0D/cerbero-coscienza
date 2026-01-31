# live/ranking.py

from dataclasses import dataclass
from typing import List

from live.trade_candidates import TradeCandidate
from live.position_sizing import SignalClass
from live.signal_classification import SignalQuality


@dataclass
class RankedTrade:
    """
    TradeCandidate + punteggio di ranking e motivazioni.
    """
    candidate: TradeCandidate
    score: float
    reason_tags: List[str]


def _base_score_for_class(signal_class: SignalClass) -> float:
    """
    Bonus di base per la classe del segnale.
    Per ora:
      - A_PLUS > A
    """
    if signal_class == SignalClass.A_PLUS:
        return 20.0
    elif signal_class == SignalClass.A:
        return 10.0
    else:
        return 0.0


def compute_trade_score(
    candidate: TradeCandidate,
    mtf_score: float = 1.0,
    market_quality_score: float = 1.0,
) -> RankedTrade:
    """
    Calcola uno score complessivo per un TradeCandidate.

    Per ora usiamo:
      - strength del segnale
      - classe (A / A_PLUS)
      - placeholder per:
          * mtf_score (multi-timeframe)
          * market_quality_score (spread, volatilità, anomalie, news, ecc.)

    In futuro:
      - mtf_score sarà calcolato da un modulo MTF dedicato
      - market_quality_score da un modulo di 'market health' / anomalie / news
    """

    reasons: List[str] = []

    # 1) partiamo dalla forza del segnale (strength)
    #    strength è in [0, 0.5] ma realisticamente 0.05–0.20
    strength_component = candidate.strength * 100.0  # porta in scala 0–50 circa
    reasons.append(f"strength={candidate.strength:.3f}")

    # 2) aggiungiamo un bonus per la classe del segnale
    class_bonus = _base_score_for_class(candidate.signal_class)
    reasons.append(f"class={candidate.signal_class.name}")

    # 3) MTF e qualità mercato come moltiplicatori (per ora 1.0, verranno gestiti fuori)
    #    Se mtf_score < 1, il segnale è penalizzato; se > 1, è premiato.
    #    Idem per market_quality_score.
    reasons.append(f"mtf_score={mtf_score:.2f}")
    reasons.append(f"market_quality={market_quality_score:.2f}")

    raw_score = (strength_component + class_bonus)
    adjusted_score = raw_score * mtf_score * market_quality_score

    return RankedTrade(
        candidate=candidate,
        score=adjusted_score,
        reason_tags=reasons,
    )


def rank_trade_candidates(
    candidates: List[TradeCandidate],
    mtf_scores: List[float] | None = None,
    market_quality_scores: List[float] | None = None,
) -> List[RankedTrade]:
    """
    Prende una lista di TradeCandidate e restituisce una lista di RankedTrade
    ordinata per score decrescente.

    mtf_scores e market_quality_scores (se forniti) devono avere
    la stessa lunghezza di candidates e verranno applicati 1:1.
    """

    ranked: List[RankedTrade] = []

    if mtf_scores is None:
        mtf_scores = [1.0] * len(candidates)
    if market_quality_scores is None:
        market_quality_scores = [1.0] * len(candidates)

    for cand, mtf_s, mq_s in zip(candidates, mtf_scores, market_quality_scores):
        ranked_trade = compute_trade_score(
            candidate=cand,
            mtf_score=mtf_s,
            market_quality_score=mq_s,
        )
        ranked.append(ranked_trade)

    # ordiniamo dal migliore al peggiore
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked
