# live/trade_intent_v3.py
"""
Costruisce il payload ufficiale che la Coscienza v3 manda al Coordinator.

Idea chiave:
- Qui dentro NON facciamo logica di trading.
- Prendiamo in input:
  - decisione della Coscienza (side, strength, class, motivazioni),
  - info di rischio (size, notional, risk_pct, SL/TP concettuali),
  - contesto (MTF, volatilità, news),
  - probabilità dei 3 modelli (grafico/tecnico/contestuale + blend),
  e restituiamo un dict serializzabile in JSON, pronto per il Coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class TradeIntentCore:
    symbol: str
    timeframe: str
    side: str  # "long" | "short"
    price_entry: float
    position_size: float
    notional_value: float
    leverage: float
    risk_pct: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    sl_distance: Optional[float]
    tp_distance: Optional[float]
    timestamp_utc: str  # ISO format UTC


@dataclass
class TradeIntentSignalMeta:
    signal_strength: float
    signal_class: str        # "A", "A+", "B", ecc.
    probs: Dict[str, float]  # p_grafico/tecnico/contestuale/blend
    reasons: List[str]


@dataclass
class TradeIntentContextMeta:
    mtf_status: Dict[str, Any]
    context_status: Dict[str, Any]
    econ_block_info: Optional[Dict[str, Any]]


def build_trade_intent(
    *,
    symbol: str,
    tf: str,
    side: str,
    entry_price: float,
    position_size: float,
    notional_value: float,
    leverage: float,
    risk_pct: float,
    sl_price: Optional[float],
    tp_price: Optional[float],
    probs: Dict[str, float],
    strength: float,
    signal_class: str,
    mtf_status: Dict[str, Any],
    context_status: Dict[str, Any],
    econ_block_info: Optional[Dict[str, Any]] = None,
    reasons: Optional[List[str]] = None,
    now_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Costruisce il dict finale da mandare al Coordinator.

    Tutti i campi numerici sono float semplici.
    I tempi sono in ISO UTC.
    Le sottosezioni sono:
      - core
      - signal
      - context
    """

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    reasons = reasons or []

    sl_distance = None
    tp_distance = None
    if sl_price is not None:
        sl_distance = abs(entry_price - sl_price)
    if tp_price is not None:
        tp_distance = abs(entry_price - tp_price)

    core = TradeIntentCore(
        symbol=symbol,
        timeframe=tf,
        side=side,
        price_entry=float(entry_price),
        position_size=float(position_size),
        notional_value=float(notional_value),
        leverage=float(leverage),
        risk_pct=float(risk_pct),
        stop_loss_price=float(sl_price) if sl_price is not None else None,
        take_profit_price=float(tp_price) if tp_price is not None else None,
        sl_distance=float(sl_distance) if sl_distance is not None else None,
        tp_distance=float(tp_distance) if tp_distance is not None else None,
        timestamp_utc=now_utc.isoformat(),
    )

    signal_meta = TradeIntentSignalMeta(
        signal_strength=float(strength),
        signal_class=str(signal_class),
        probs={k: float(v) for k, v in probs.items()},
        reasons=[str(r) for r in reasons],
    )

    ctx_meta = TradeIntentContextMeta(
        mtf_status=mtf_status,
        context_status=context_status,
        econ_block_info=econ_block_info,
    )

    # Dict finale piatto ma organizzato in sezioni
    return {
        "core": asdict(core),
        "signal": asdict(signal_meta),
        "context": asdict(ctx_meta),
        # spazio per future estensioni (es. "account", "session_id", ecc.)
    }
