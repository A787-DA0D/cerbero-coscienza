# live/alerts.py
"""
Gestione degli alert Telegram per Cerbero.

- Usa TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID presi da variabili d'ambiente.
- Logga sempre su stdout quello che manda.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

import requests


TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")  # deve essere un ID numerico (stringa)


def _build_message(kind: str, payload: Dict[str, Any]) -> str:
    """
    Costruisce un messaggio testo leggibile per Telegram.
    """
    base = payload.get("msg")
    if not base:
        base = json.dumps(payload, ensure_ascii=False, indent=2)

    return f"📡 Cerbero alert [{kind}]\n\n{base}"


def send_telegram_alert(kind: str, payload: Dict[str, Any]) -> None:
    """
    Invia un alert Telegram (se TOKEN e CHAT_ID sono configurati).

    kind: tipo di alert, es. "test", "trade_intent", "error", ecc.
    payload: dict con i dettagli (almeno "msg" se vogliamo un testo pulito).
    """
    ts = datetime.now(timezone.utc).isoformat()
    envelope = {
        "ts": ts,
        "kind": kind,
        "payload": payload,
    }

    # Log sempre in stdout (come già vedevi prima)
    print("📣 TELEGRAM ALERT:", json.dumps(envelope, ensure_ascii=False))

    token = TELEGRAM_BOT_TOKEN
    chat_id = TELEGRAM_CHAT_ID

    # Se non sono configurati, esce in silenzio (ma abbiamo già loggato sopra)
    if not token or not chat_id:
        print("⚠️ TELEGRAM non configurato (manca TOKEN o CHAT_ID).")
        return

    text = _build_message(kind, payload)

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
            },
            timeout=5,
        )
        if resp.status_code != 200:
            print(f"⚠️ Errore Telegram: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"⚠️ Eccezione Telegram: {e}")
