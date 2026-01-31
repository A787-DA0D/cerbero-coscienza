# live/mtf_filters.py
"""
Filtri Multi-TimeFrame (MTF) per Cerbero Coscienza v3 PRO.

Obiettivo:
- Dare alla Coscienza un "giudizio" su cosa sta facendo il Daily e il 4H,
  e dire se un segnale LONG/SHORT su 5m/15m è coerente oppure no.

Per ora usiamo una logica semplice basata su:
- posizione del close rispetto a ema20 / ema50,
- distanza dalle ema.

In futuro potremo raffinarla (usare trend/choppy, ecc.).
"""

from __future__ import annotations

from typing import Tuple, Dict, Optional

import pandas as pd


RAW_BUCKET = "cerbero-data-processed-gns"


def _feat_path(sym: str, tf: str) -> str:
    return f"gs://{RAW_BUCKET}/features_{tf}/{sym}.parquet"


def _load_last_bar(sym: str, tf: str) -> Optional[pd.Series]:
    """
    Legge l'ultima riga delle feature per (sym, tf).
    Ritorna una Series o None se c'è un problema.
    """
    path = _feat_path(sym, tf)
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        row = df.iloc[-1].copy()
        return row
    except Exception as e:
        print(f"⚠️  MTF: errore nel leggere {path}: {e}")
        return None


def _compute_bias_from_bar(row: pd.Series) -> str:
    """
    Calcola un bias molto semplice da una barra Daily/4H:

    - Se ema20 e ema50 esistono:
        - se close > ema20 > ema50 → 'bull'
        - se close < ema20 < ema50 → 'bear'
        - altrimenti → 'neutral'
    - Se non ci sono ema*:
        - prova ad usare 'trend'/'choppy' se presenti,
        - altrimenti ritorna 'neutral'.
    """
    close = row.get("close", None)
    ema20 = row.get("ema20", None)
    ema50 = row.get("ema50", None)

    try:
        if close is not None and ema20 is not None and ema50 is not None:
            if close > ema20 > ema50:
                return "bull"
            if close < ema20 < ema50:
                return "bear"
            return "neutral"
    except Exception:
        pass

    # fallback: se esiste una colonna 'trend' con valori tipo 1/-1/0
    trend = row.get("trend", None)
    if trend is not None:
        try:
            t = float(trend)
            if t > 0:
                return "bull"
            if t < 0:
                return "bear"
        except Exception:
            pass

    return "neutral"


def get_mtf_state(sym: str) -> Dict[str, str]:
    """
    Ritorna un dizionario con il bias Daily e 4H per un simbolo.

    Esempio:
        {
            "daily_bias": "bull",
            "h4_bias": "neutral"
        }
    """
    out = {
        "daily_bias": "unknown",
        "h4_bias": "unknown",
    }

    # Daily (1d)
    row_1d = _load_last_bar(sym, "1d")
    if row_1d is not None:
        out["daily_bias"] = _compute_bias_from_bar(row_1d)

    # 4H (4h)
    row_4h = _load_last_bar(sym, "4h")
    if row_4h is not None:
        out["h4_bias"] = _compute_bias_from_bar(row_4h)

    return out


def check_mtf_ok(sym: str, direction: str) -> Tuple[bool, Dict[str, str]]:
    """
    Decide se un segnale LONG/SHORT sui TF bassi (5m,15m) è coerente con il MTF.

    Regole semplici (prima versione):

    - Se daily_bias è 'bull':
        - LONG → ok (a meno che h4_bias sia 'bear' -> conflitto forte)
        - SHORT → conflitto (ok solo se in futuro lo tratteremo come 'contro-trend')
    - Se daily_bias è 'bear':
        - SHORT → ok (a meno che h4_bias sia 'bull')
        - LONG → conflitto
    - Se daily_bias è 'neutral':
        - ci affidiamo di più a h4_bias:
            - se h4_bias allineato alla direzione → ok
            - altrimenti → borderline (per ora lo consideriamo non ok).

    Ritorna:
        (mtf_ok: bool, info: dict)
    """
    info = get_mtf_state(sym)
    daily = info.get("daily_bias", "unknown")
    h4 = info.get("h4_bias", "unknown")

    direction = direction.upper()

    # default: neutrale ma prudente
    mtf_ok = True
    reason = ""

    if daily == "bull":
        if direction == "LONG":
            # LONG in bull daily: ok, ma se 4H è bear segnaliamo conflitto
            if h4 == "bear":
                mtf_ok = False
                reason = "daily bull ma 4H bear (conflitto forte)"
            else:
                mtf_ok = True
                reason = "daily bull, LONG coerente"
        elif direction == "SHORT":
            mtf_ok = False
            reason = "daily bull ma segnale SHORT (contro-trend)"
        else:
            reason = "direzione sconosciuta"

    elif daily == "bear":
        if direction == "SHORT":
            if h4 == "bull":
                mtf_ok = False
                reason = "daily bear ma 4H bull (conflitto forte)"
            else:
                mtf_ok = True
                reason = "daily bear, SHORT coerente"
        elif direction == "LONG":
            mtf_ok = False
            reason = "daily bear ma segnale LONG (contro-trend)"
        else:
            reason = "direzione sconosciuta"

    else:  # daily == 'neutral' o 'unknown'
        # Qui guardiamo di più il 4H
        if h4 == "bull" and direction == "LONG":
            mtf_ok = True
            reason = "daily neutro, 4H bull, LONG coerente"
        elif h4 == "bear" and direction == "SHORT":
            mtf_ok = True
            reason = "daily neutro, 4H bear, SHORT coerente"
        elif h4 in ("unknown", "neutral"):
            # non abbiamo info forti → per ora ok ma con motivo
            mtf_ok = True
            reason = "MTF poco chiaro (daily/h4 neutri), nessun veto"
        else:
            # h4 contro direzione
            mtf_ok = False
            reason = "daily neutro ma 4H contro la direzione"

    info["mtf_ok_reason"] = reason
    info["mtf_ok_flag"] = "ok" if mtf_ok else "block"

    return mtf_ok, info
