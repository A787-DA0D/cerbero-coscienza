# live/intent_normalizer.py
"""
Adapter ufficiale: RawSignal (Coscienza) -> TradeIntent FLAT (Coordinator)

Micro-spec vincolanti:
1) symbol canonico interno: "EURUSD", "BTCUSD", "XAUUSD" (NO slash)
   - se arriva "EUR/USD" o "BTC/USD" viene normalizzato qui
2) signal_id: UUID v4 generato QUI (non nei modelli)
3) confidence_score: STANDARD UNICO = strength
   strength = abs(p_blend - 0.5) * 2    range 0..1
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import uuid


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        # --- Anti-confusione SL/TP: salva sia human(%) che dec ---
        try:
            if isinstance(meta, dict):
                if "sl_pct" in meta and "sl_pct_dec" not in meta:
                    _sl_dec = float(meta["sl_pct"])  # decimale (es. 0.0025)
                    meta["sl_pct_dec"] = _sl_dec
                    meta.setdefault("sl_pct_human", _sl_dec * 100.0)  # umano (es. 0.25)
                if "tp_pct" in meta and "tp_pct_dec" not in meta:
                    _tp_dec = float(meta["tp_pct"])  # decimale (es. 0.0060)
                    meta["tp_pct_dec"] = _tp_dec
                    meta.setdefault("tp_pct_human", _tp_dec * 100.0)  # umano (es. 0.60)
        except Exception:
            pass

        # META_HUMAN_DEC_WIRING: aggiunge campi duplicati anti-confusione (zero logica)

        try:

            _m = (None.get('meta') if isinstance(None, dict) else None) or {}

            if isinstance(_m, dict):

                # sl

                if _m.get('sl_pct_dec') is None and _m.get('sl_pct_human') is not None:

                    try: _m['sl_pct_dec'] = float(_m.get('sl_pct_human')) / 100.0

                    except Exception: pass

                if _m.get('sl_pct_human') is None and _m.get('sl_pct_dec') is not None:

                    try: _m['sl_pct_human'] = float(_m.get('sl_pct_dec')) * 100.0

                    except Exception: pass

                # tp

                if _m.get('tp_pct_dec') is None and _m.get('tp_pct_human') is not None:

                    try: _m['tp_pct_dec'] = float(_m.get('tp_pct_human')) / 100.0

                    except Exception: pass

                if _m.get('tp_pct_human') is None and _m.get('tp_pct_dec') is not None:

                    try: _m['tp_pct_human'] = float(_m.get('tp_pct_dec')) * 100.0

                    except Exception: pass

                None['meta'] = _m

        except Exception:

            pass


        return None
    try:
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _pick(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _normalize_symbol_to_canonical(sym: str) -> str:
    """
    Canonico: EURUSD / BTCUSD / XAUUSD
    Converte:
      - "EUR/USD", "EUR-USD", "EUR_USD" -> "EURUSD"
      - "BTCUSDT" -> "BTCUSD"
    """
    s = (sym or "").strip().upper()
    s = s.replace("/", "").replace("-", "").replace("_", "").replace(" ", "")
    if s.endswith("USDT"):
        s = s[:-4] + "USD"
    return s


def _normalize_direction(raw: Any) -> str:
    """
    Must be 'LONG' or 'SHORT'
    Accetta: long/short, buy/sell, 1/-1
    """
    if raw is None:
        raise ValueError("direction missing")

    if isinstance(raw, (int, float)):
        return "LONG" if float(raw) > 0 else "SHORT"

    s = str(raw).strip().upper()
    if s in ("LONG", "BUY"):
        return "LONG"
    if s in ("SHORT", "SELL"):
        return "SHORT"

    raise ValueError(f"invalid direction: {raw}")


def _compute_strength_from_p_blend(p_blend: float) -> float:
    # strength = abs(p_blend - 0.5) * 2, clamp 0..1
    strength = abs(p_blend - 0.5) * 2.0
    if strength < 0.0:
        strength = 0.0
    if strength > 1.0:
        strength = 1.0
    return strength


def normalize_intent_for_coordinator(raw_signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Output: TradeIntent FLAT schema (obbligatorio)
    """
    if not isinstance(raw_signal, dict):
        raise ValueError("raw_signal must be a dict")

    # symbol
    symbol_raw = _pick(raw_signal, "symbol", "pair", "market", "instrument")
    if not symbol_raw:
        raise ValueError("symbol missing")
    symbol = _normalize_symbol_to_canonical(str(symbol_raw))

    # direction
    direction_raw = _pick(raw_signal, "direction", "side", "signal_direction", "trade_direction")
    direction = _normalize_direction(direction_raw)

    # signal_id (sempre generato qui se non presente)
    signal_id_raw = _pick(raw_signal, "signal_id")
    signal_id = str(signal_id_raw) if signal_id_raw else str(uuid.uuid4())

    # risk_notional_usd (MANDATORY)
    size_raw = _pick(
        raw_signal,
        "risk_notional_usd",
        "risk_notional_usd",
        "risk_risk_notional_usd",
        "position_risk_notional_usd",
        # fallback dal vecchio schema (se qualcuno passa ancora intent legacy)
        "position_size",
    )
    risk_notional_usd = _to_float(size_raw)
    if risk_notional_usd is None:
        raise ValueError("risk_notional_usd missing/invalid")

    # leverage (MANDATORY)
    lev_raw = _pick(raw_signal, "leverage", "lev", "dyn_leverage")
    leverage = _to_int(lev_raw)
    if leverage is None:
        raise ValueError("leverage missing/invalid")

    # order_type (per ora solo MARKET)
    order_type = str(_pick(raw_signal, "order_type", "execution_type") or "MARKET").upper()
    if order_type != "MARKET":
        order_type = "MARKET"

    # dry_run supportato sempre
    dry_run_val = _pick(raw_signal, "dry_run", "is_dry_run", "paper", "paper_trade")
    dry_run = bool(dry_run_val) if dry_run_val is not None else True

    # SL mandatory
    sl_raw = _pick(raw_signal, "sl_price", "stop_loss", "sl", "stop_price", "stop_loss_price")
    sl_price = _to_float(sl_raw)
    if sl_price is None:
        raise ValueError("sl_price missing/invalid")

    # TP optional
    tp_raw = _pick(raw_signal, "tp_price", "take_profit", "tp", "limit_price", "take_profit_price")
    tp_price = _to_float(tp_raw)  # can be None

    # strategy_origin
    strategy_origin = str(_pick(raw_signal, "strategy_origin", "strategy", "model", "origin") or "UNKNOWN")

    # confidence_score = strength (standard unico)
    p_blend = None

    # caso 1: raw_signal contiene probs dict
    probs = _pick(raw_signal, "probs")
    if isinstance(probs, dict):
        p_blend = _to_float(_pick(probs, "p_blend", "blend", "pBlend", "p_final", "p"))

    # caso 2: raw_signal contiene p_blend diretto
    if p_blend is None:
        p_blend = _to_float(_pick(raw_signal, "p_blend", "blend", "p_final"))

    # calcolo strength standard
    if p_blend is not None:
        confidence_score = _compute_strength_from_p_blend(float(p_blend))
    else:
        # fallback: usa strength già presente (solo se manca p_blend)
        strength_raw = _pick(raw_signal, "strength", "signal_strength")
        confidence_score = _to_float(strength_raw) or 0.0
        if confidence_score < 0.0:
            confidence_score = 0.0
        if confidence_score > 1.0:
            confidence_score = 1.0


    # ------------------------------
    # META passthrough + risk pct normalization
    # Executor expects sl_pct/tp_pct as DECIMALS:
    #   0.0025 == 0.25%
    # Risk policy / humans often provide 0.25..0.60 meaning 0.25%..0.60%
    # We normalize here ONCE to avoid double conversion elsewhere.
    # ------------------------------
    meta_in = _pick(raw_signal, "meta") if isinstance(_pick(raw_signal, "meta"), dict) else {}

    def _pct_to_decimal(x: Optional[float]) -> Optional[float]:
        if x is None:
            # META_HUMAN_DEC_WIRING_IN_NORMALIZE_FN: anti-confusione (zero logica)
            try:
                _m = (None.get('meta') if isinstance(None, dict) else None) or {}
                # META_HUMAN_DEC_2LINES_FINAL: aggiunge sl/tp *_human senza cambiare logica
                try:
                    if isinstance(_m, dict):
                        if _m.get('sl_pct_dec') is not None and _m.get('sl_pct_human') is None:
                            try: _m['sl_pct_human'] = float(_m.get('sl_pct_dec')) * 100.0
                            except Exception: pass
                        if _m.get('tp_pct_dec') is not None and _m.get('tp_pct_human') is None:
                            try: _m['tp_pct_human'] = float(_m.get('tp_pct_dec')) * 100.0
                            except Exception: pass
                except Exception:
                    pass

                if isinstance(_m, dict):
                    if _m.get('sl_pct_human') is None and _m.get('sl_pct_dec') is not None:
                        try: _m['sl_pct_human'] = float(_m.get('sl_pct_dec')) * 100.0
                        except Exception: pass
                    if _m.get('tp_pct_human') is None and _m.get('tp_pct_dec') is not None:
                        try: _m['tp_pct_human'] = float(_m.get('tp_pct_dec')) * 100.0
                        except Exception: pass
                    if _m.get('sl_pct_dec') is None and _m.get('sl_pct_human') is not None:
                        try: _m['sl_pct_dec'] = float(_m.get('sl_pct_human')) / 100.0
                        except Exception: pass
                    if _m.get('tp_pct_dec') is None and _m.get('tp_pct_human') is not None:
                        try: _m['tp_pct_dec'] = float(_m.get('tp_pct_human')) / 100.0
                        except Exception: pass
                    None['meta'] = _m
            except Exception:
                pass


            return None
        try:
            xf = float(x)
        except Exception:
            return None
        if xf <= 0:
            return None
        # heuristic: values > 0.05 are almost surely percent-like (0.25..0.60) -> /100
        return xf / 100.0 if xf > 0.05 else xf

    sl_pct_raw = _to_float(_pick(meta_in, "sl_pct", "stop_loss_pct", "slPercent", "sl_pct_dec"))
    tp_pct_raw = _to_float(_pick(meta_in, "tp_pct", "take_profit_pct", "tpPercent", "tp_pct_dec"))

    # also allow top-level keys (legacy callers)
    if sl_pct_raw is None:
        sl_pct_raw = _to_float(_pick(raw_signal, "sl_pct", "stop_loss_pct", "slPercent", "sl_pct_dec"))
    if tp_pct_raw is None:
        tp_pct_raw = _to_float(_pick(raw_signal, "tp_pct", "take_profit_pct", "tpPercent", "tp_pct_dec"))

    sl_pct = _pct_to_decimal(sl_pct_raw)
    tp_pct = _pct_to_decimal(tp_pct_raw)

    meta_out: Dict[str, Any] = {}
    if isinstance(meta_in, dict):
        meta_out.update(meta_in)
    if sl_pct is not None:
        meta_out["sl_pct"] = float(sl_pct)
    if tp_pct is not None:
        meta_out["tp_pct"] = float(tp_pct)

    return {
        # IDENTITÀ
        "symbol": symbol,
        "direction": direction,
        "signal_id": signal_id,

        # RISK & SIZE
        "risk_notional_usd": float(risk_notional_usd),
        "leverage": int(leverage),

        # ESECUZIONE
        "order_type": order_type,
        "dry_run": bool(dry_run),

        # RISK MANAGEMENT
        "sl_price": float(sl_price),
        "tp_price": (float(tp_price) if tp_price is not None else None),

        # METADATA STRATEGICA
        "strategy_origin": strategy_origin,
        "confidence_score": float(confidence_score),
        "meta": meta_out if meta_out else None,
    }
