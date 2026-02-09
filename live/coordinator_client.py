# live/coordinator_client.py
"""
Client HTTP della Coscienza verso il Coordinator.

Da oggi:
- payload CANONICO = intent_normalizer.normalize_intent_for_coordinator(raw_signal)
- + campi compatibilità per l'infrastruttura attuale:
  tenant_email, timeframe, strength_class, risk_pct, ts_signal

Niente più builder v2 "povero".
"""

import os
import json
from typing import Any, Dict

import requests

from live.intent_normalizer import normalize_intent_for_coordinator

COORDINATOR_BASE_URL = os.environ.get(
    "COORDINATOR_BASE_URL",
    "https://coordinatore-365935921345.europe-west8.run.app",
).rstrip("/")

COORDINATOR_API_KEY = os.environ.get("COORDINATOR_API_KEY", "")

SESSION = requests.Session()
SESSION.headers.update({"Content-Type": "application/json"})


def _strength_class_from_sig_class(sig_class: str) -> str:
    s = (sig_class or "A").strip().upper()
    if s in ("A+", "APLUS", "A_PLUS"):
        return "A_PLUS"
    if s in ("A", "B", "C", "D"):
        return s
    return "A"


def _to_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        if v > 0:
            return v
    except Exception:
        pass
    return float(default)


def _build_payload(raw_signal: Dict[str, Any]) -> Dict[str, Any]:
    # 1) CANONICO istituzionale
    payload = normalize_intent_for_coordinator(raw_signal)

    if payload.get("sl_price") is None:
        raise ValueError("normalize_intent_for_coordinator produced missing sl_price (BUG)")

    raw_signal = dict(raw_signal or {})
    raw_signal["sl_price"] = payload.get("sl_price")
    raw_signal["tp_price"] = payload.get("tp_price")
    raw_signal["risk_notional_usd"] = payload.get("risk_notional_usd")
    raw_signal["leverage"] = payload.get("leverage")

    tf = (raw_signal.get("tf") or raw_signal.get("timeframe") or "15m")
    tenant_email = raw_signal.get("tenant_email") or raw_signal.get("user_id") or None
    if not tenant_email:
        tenant_email = (os.environ.get("CERBERO_TENANT_EMAIL", "") or "").strip() or None
    if not tenant_email:
        raise ValueError("tenant_email missing")

    sig_class = raw_signal.get("sig_class") or "A"
    strength_class = _strength_class_from_sig_class(str(sig_class))

    ts_signal = raw_signal.get("ts_signal")
    if not ts_signal:
        raise ValueError("raw_signal missing ts_signal")

    risk_pct = _to_float(raw_signal.get("risk_pct"), default=None)
    if risk_pct is None:
        raise ValueError("raw_signal missing risk_pct")

    payload.update({
        "tenant_email": tenant_email,
        "user_id": tenant_email,
        "timeframe": str(tf).lower(),
        "strength_class": strength_class,
        "risk_pct": float(risk_pct),
        "ts_signal": ts_signal,
    })

    meta_out = dict(payload.get("meta") or {})
    meta_out.update({
        "strategy_origin": raw_signal.get("strategy_origin"),
        "p_blend": raw_signal.get("p_blend"),
        "strength": raw_signal.get("strength"),
        "reasons": raw_signal.get("reasons"),
        "tp": raw_signal.get("tp_price"),
        "sl": raw_signal.get("sl_price"),
        "tp_r": raw_signal.get("tp_r"),
    })

    def _as_float(x):
        try:
            return float(x)
        except Exception:
            return None

    # --- ENSURE SL/TP PCT DECIMALS IN META (both keys, always) ---
    # priority: existing meta (from normalizer) -> raw_signal *_dec -> raw_signal *_human -> raw_signal sl_pct/tp_pct (human?) safe-convert
    def _ensure_pct(meta_key_dec: str, meta_key: str, raw_dec_key: str, raw_human_key: str):
        v = _as_float(meta_out.get(meta_key_dec))
        if v is None:
            v = _as_float(meta_out.get(meta_key))
        if v is None:
            v = _as_float(raw_signal.get(raw_dec_key))
        if v is None:
            vh = _as_float(raw_signal.get(raw_human_key))
            if vh is not None:
                v = vh / 100.0
        # last resort: raw_signal has sl_pct/tp_pct but might be human (e.g. 0.25) or decimal (0.0025)
        if v is None:
            vv = _as_float(raw_signal.get(meta_key))
            if vv is not None:
                v = vv / 100.0 if vv > 0.05 else vv  # heuristic: 0.25 is human; 0.0025 is decimal

        if v is not None:
            meta_out[meta_key_dec] = float(v)
            meta_out[meta_key] = float(v)

    _ensure_pct("sl_pct_dec", "sl_pct", "sl_pct_dec", "sl_pct_human")
    _ensure_pct("tp_pct_dec", "tp_pct", "tp_pct_dec", "tp_pct_human")

    payload["meta"] = meta_out
    return payload

def send_trade_intent(raw_signal: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    url = f"{COORDINATOR_BASE_URL}/v1/trade-intent"
    payload = _build_payload(raw_signal)
    print("[COORDINATOR] SENT_PAYLOAD", json.dumps(payload)[:2000])

    headers = {"Content-Type": "application/json"}
    if COORDINATOR_API_KEY:
        headers["X-API-Key"] = COORDINATOR_API_KEY

    try:
        resp = SESSION.post(url, headers=headers, data=json.dumps(payload), timeout=15)
        if not resp.ok:
            raise RuntimeError(f"Coordinator error {resp.status_code}: {resp.text} | sent={payload}")
        data = resp.json()
        print(f"[COORDINATOR] OK {data}")
        return data
    except Exception as e:
        print(f"[COORDINATOR] ERROR sending intent: {e}")
        if not dry_run:
            raise
        return {"status": "ERROR", "error": str(e), "sent": payload}
