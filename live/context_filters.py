# live/context_filters.py
"""
Filtri di contesto per Cerbero Coscienza v3:

- compute_context_status(symbol, tf, row)
    → legge feature tipo atr_z, bb_width_z, bb_squeeze, ecc. (se esistono)
      e costruisce un piccolo stato di regime/volatilità.

- check_economic_block(symbol, now_utc, window_minutes=30)
    → controlla se, entro +/- window_minutes, ci sono eventi macro "bomba"
      (rate decision + NFP) nel CSV live/econ_events_bombs.csv.

- format_news_events(events)
    → formatta la lista di eventi per log / telegram.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Set

import numpy as np
import pandas as pd

# Path locale del CSV con gli eventi bomba (rate decision + NFP)
_ECO_CSV_PATH = os.path.join(os.path.dirname(__file__), "econ_events_bombs.csv")

# cache in memoria
_ECO_DF_CACHE: Optional[pd.DataFrame] = None


# ---------------------- ECONOMIC EVENTS ---------------------- #

def _load_econ_events() -> pd.DataFrame:
    """
    Carica il CSV degli eventi macro “bomba”.

    Campo richiesto minimo:
        - ts_utc (datetime ISO string)
        - currency (es. USD, EUR, GBP, JPY…)
        - kind (es. RATE_DECISION, NFP)
        - event (stringa descrittiva)
    """
    global _ECO_DF_CACHE

    if _ECO_DF_CACHE is not None:
        return _ECO_DF_CACHE

    if not os.path.exists(_ECO_CSV_PATH):
        _ECO_DF_CACHE = pd.DataFrame(
            columns=["ts_utc", "currency", "kind", "event"]
        )
        return _ECO_DF_CACHE

    df = pd.read_csv(_ECO_CSV_PATH)

    if "ts_utc" not in df.columns:
        df["ts_utc"] = pd.NaT
    else:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")

    # normalizziamo le colonne
    for col in ["currency", "kind", "event"]:
        if col not in df.columns:
            df[col] = ""

    # filtra righe senza timestamp
    df = df.loc[~df["ts_utc"].isna()].copy()
    _ECO_DF_CACHE = df
    return _ECO_DF_CACHE


def _symbol_to_currencies(symbol: str) -> List[str]:
    """
    Mappa simbolo → lista di currency rilevanti per news.

    Esempi:
      - EURUSD -> ["EUR","USD"]
      - XAUUSD -> ["USD"]
      - DOLLARIDXUSD -> ["USD"]
      - BTCUSD, ETHUSD -> ["USD"] (per NFP e FOMC)
    """
    s = symbol.upper()

    # cross FX classici
    if len(s) == 6 and s.isalpha():
        return [s[0:3], s[3:6]]

    # mappa manuale per i nostri 16
    if s in ("XAUUSD", "XAGUSD", "DOLLARIDXUSD", "BTCUSD", "ETHUSD"):
        return ["USD"]

    if s in ("USA500IDXUSD", "USATECHIDXUSD"):
        return ["USD"]

    if s in ("DEUIDXEUR", "JPNIDXJPY"):
        return [s[-3:]]

    # default: prova a prendere le ultime 3 come currency
    if len(s) >= 3:
        return [s[-3:]]
    return []


def check_economic_block(
    symbol: str,
    now_utc: datetime,
    window_minutes: int = 30,
) -> Dict[str, Any]:
    """
    Ritorna se il simbolo deve essere bloccato per news macro “bomba”
    entro +/- window_minutes dall'orario corrente.

    Usa solo eventi importanti (RATE_DECISION + NFP)
    che abbiamo messo nel CSV.
    """
    df = _load_econ_events()
    if df.empty:
        return {
            "blocked": False,
            "hits": [],
            "reason": "no_econ_file",
        }

    cur_list: Set[str] = set(_symbol_to_currencies(symbol))
    if not cur_list:
        return {
            "blocked": False,
            "hits": [],
            "reason": "no_relevant_currency",
        }

    t0 = now_utc.replace(tzinfo=timezone.utc)
    delta = timedelta(minutes=window_minutes)

    # filtro time window
    mask_time = (df["ts_utc"] >= t0 - delta) & (df["ts_utc"] <= t0 + delta)
    df_win = df.loc[mask_time].copy()
    if df_win.empty:
        return {
            "blocked": False,
            "hits": [],
            "reason": "no_events_in_window",
        }

    # filtro currency
    mask_cur = df_win["currency"].isin(cur_list)
    df_win = df_win.loc[mask_cur].copy()
    if df_win.empty:
        return {
            "blocked": False,
            "hits": [],
            "reason": "no_currency_match",
        }

    # ci sono eventi rilevanti
    hits: List[Dict[str, Any]] = []
    for _, r in df_win.iterrows():
        hits.append(
            {
                "ts_utc": r["ts_utc"],
                "currency": str(r.get("currency", "")),
                "kind": str(r.get("kind", "")),
                "event": str(r.get("event", "")),
            }
        )

    return {
        "blocked": True,
        "hits": hits,
        "reason": "econ_block_window",
    }


def format_news_events(events: List[Dict[str, Any]]) -> str:
    """
    Rende leggibile una lista di eventi macro.
    """
    if not events:
        return "Nessun evento macro rilevante."
    lines = []
    for ev in events:
        ts = ev.get("ts_utc")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except Exception:
                pass
        if isinstance(ts, datetime):
            ts_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        else:
            ts_str = "N/A"
        cur = ev.get("currency", "")
        kind = ev.get("kind", "")
        name = ev.get("event", "")
        lines.append(f"- {ts_str} | {cur} | {kind} | {name}")
    return "\n".join(lines)


# ---------------------- CONTEXT STATUS ---------------------- #

def compute_context_status(symbol: str, tf: str, row: pd.Series) -> Dict[str, Any]:
    """
    Calcola uno stato semplice di contesto a partire dall'ultima riga di feature.

    Non blocca quasi mai (ok=True di default), serve per dare “consapevolezza”
    alla Coscienza:
      - regime: trend / choppy / unknown
      - volatility: low / normal / high
      - squeeze: True/False (se bb_squeeze presente)
    """
    out: Dict[str, Any] = {
        "ok": True,
        "regime": "unknown",
        "volatility": "unknown",
        "squeeze": False,
        "details": {},
    }

    # estraiamo valori se esistono
    atr_z = float(row.get("atr_z", np.nan)) if "atr_z" in row.index else np.nan
    bb_width_z = float(row.get("bb_width_z", np.nan)) if "bb_width_z" in row.index else np.nan
    bb_squeeze = bool(row.get("bb_squeeze", False)) if "bb_squeeze" in row.index else False
    trend_flag = row.get("trend", np.nan)
    choppy_flag = row.get("choppy", np.nan)

    # volatilità
    if not np.isnan(atr_z):
        if atr_z < -0.5:
            vol = "low"
        elif atr_z > 1.0:
            vol = "high"
        else:
            vol = "normal"
        out["volatility"] = vol

    # regime trend/choppy
    if not pd.isna(trend_flag) and trend_flag:
        out["regime"] = "trend"
    elif not pd.isna(choppy_flag) and choppy_flag:
        out["regime"] = "choppy"

    out["squeeze"] = bool(bb_squeeze)

    # in teoria potremmo usare queste info per mettere ok=False in casi estremi,
    # per ora manteniamo ok=True e usiamo solo a livello informativo.
    out["details"] = {
        "atr_z": atr_z,
        "bb_width_z": bb_width_z,
        "bb_squeeze": bb_squeeze,
        "trend_flag": trend_flag,
        "choppy_flag": choppy_flag,
    }

    return out
