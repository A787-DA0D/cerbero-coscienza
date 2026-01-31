# training/mtf_logic.py
"""
Logica Multi-Timeframe (MTF) per Cerbero Coscienza v3.

Questa versione:
- Usa le feature storiche 1D e 4H (features_1d / features_4h su GCS),
- Usa i modelli radar v3 tramite live.loader_v3.load_radar_config,
- Calcola p_blend su 1D e 4H,
- Mappa in bias ("long" / "short" / "sideways"),
- Ritorna un dizionario completo:
    {
      "symbol": ...,
      "p_blend_1d": ...,
      "p_blend_4h": ...,
      "bias_1d": ...,
      "bias_4h": ...,
      "bias": ...,
      "ok": True/False,
      "detail": "string di debug"
    }

Questa è una logica stabile che può andare in produzione; in futuro
potremo solo affinare le soglie di bias senza cambiare firma.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from live.loader_v3 import load_radar_config

# Bucket dove vivono le feature già generate da 02_features.py
FEATURE_BUCKET = "cerbero-data-processed-gns"

# Cache in RAM per non rileggere continuamente i parquet
_FEATURE_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}


def _load_features(symbol: str, tf: str) -> pd.DataFrame:
    """
    Carica e cache-a le feature per (symbol, tf) da GCS:
    gs://cerbero-data-processed-gns/features_<tf>/<symbol>.parquet

    tf ∈ {"1d","4h"} in questo modulo.
    """
    key = (symbol, tf)
    if key in _FEATURE_CACHE:
        return _FEATURE_CACHE[key]

    path = f"gs://{FEATURE_BUCKET}/features_{tf}/{symbol}.parquet"
    df = pd.read_parquet(path)

    # Normalizza indice datetime e ordina
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~df.index.isna()].sort_index()

    _FEATURE_CACHE[key] = df
    return df


def _get_last_row(symbol: str, tf: str, now_utc: datetime | None) -> pd.Series | None:
    """
    Restituisce l'ultima riga disponibile per (symbol, tf),
    opzionalmente filtrata a "<= now_utc" se passato.
    """
    df = _load_features(symbol, tf)
    if df.empty:
        return None

    if now_utc is not None:
        df2 = df.loc[df.index <= now_utc]
        if df2.empty:
            return None
        return df2.iloc[-1]
    else:
        return df.iloc[-1]


def _map_bias(p_blend: float, up: float = 0.55, down: float = 0.45) -> str:
    """
    Mappa una probabilità blend in un bias direzionale.
    - p >= up   → "long"
    - p <= down → "short"
    - altrimenti "sideways"
    """
    if p_blend >= up:
        return "long"
    if p_blend <= down:
        return "short"
    return "sideways"


def compute_mtf_status(
    symbol: str,
    now_utc: datetime | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Calcola lo stato MTF "reale" per il simbolo, usando:
    - radar v3 su 1D e 4H,
    - features_1d / features_4h.

    Parametri
    ---------
    symbol : str
        Simbolo (es. "EURUSD").
    now_utc : datetime | None
        Timestamp corrente in UTC (usato per scegliere la barra più recente).
    kwargs : Any
        Placeholder per futuri parametri (non usati ora).

    Ritorno
    -------
    dict con chiavi:
        - symbol
        - p_blend_1d
        - p_blend_4h
        - bias_1d ("long"/"short"/"sideways" o "unknown")
        - bias_4h (idem)
        - bias (bias combinato)
        - ok (True/False → MTF accetta o meno nuovi trade)
        - detail (stringa di log / debug)
    """
    ts_str = now_utc.isoformat() if now_utc is not None else "n/a"

    # Default "neutro ma ok", nel caso di problemi
    default = {
        "symbol": symbol,
        "p_blend_1d": None,
        "p_blend_4h": None,
        "bias_1d": "unknown",
        "bias_4h": "unknown",
        "bias": "neutral",
        "ok": True,
        "detail": f"MTF v3: fallback neutral (errore dati/modelli) @ {ts_str}",
    }

    try:
        # 1) Carica config radar per 1D e 4H
        cfg_1d = load_radar_config(symbol, "1d")
        cfg_4h = load_radar_config(symbol, "4h")

        # 2) Prendi ultima barra di feature per 1D e 4H
        row_1d = _get_last_row(symbol, "1d", now_utc)
        row_4h = _get_last_row(symbol, "4h", now_utc)

        if row_1d is None or row_4h is None:
            return default | {
                "detail": f"MTF v3: missing features for {symbol} (1d/4h) @ {ts_str}"
            }

        # 3) Calcola p_blend su 1D e 4H
        probs_1d = cfg_1d.predict_probas(row_1d)
        probs_4h = cfg_4h.predict_probas(row_4h)

        p1 = float(probs_1d.get("blend", 0.5))
        p4 = float(probs_4h.get("blend", 0.5))

        # 4) Mappa in bias direzionali
        bias_1d = _map_bias(p1)
        bias_4h = _map_bias(p4)

        # 5) Combina in bias complessivo + ok
        # Regola:
        #   - se 1D long e 4H short → conflitto forte → ok=False
        #   - se 1D short e 4H long → conflitto forte → ok=False
        #   - altrimenti ok=True, bias combinato:
        #       preferisci 1D, 4H come conferma
        if (bias_1d == "long" and bias_4h == "short") or (
            bias_1d == "short" and bias_4h == "long"
        ):
            ok = False
            bias = "conflict"
            detail = (
                f"MTF v3: conflict 1D={bias_1d}, 4H={bias_4h} "
                f"p1d={p1:.3f}, p4h={p4:.3f} @ {ts_str}"
            )
        else:
            ok = True
            # Se 1D ha direzione, usala; se 1D sideways, guarda 4H
            if bias_1d in ("long", "short"):
                bias = bias_1d
            elif bias_4h in ("long", "short"):
                bias = bias_4h
            else:
                bias = "sideways"

            detail = (
                f"MTF v3: ok, bias={bias} "
                f"(1D={bias_1d} p={p1:.3f}, 4H={bias_4h} p={p4:.3f}) @ {ts_str}"
            )

        return {
            "symbol": symbol,
            "p_blend_1d": p1,
            "p_blend_4h": p4,
            "bias_1d": bias_1d,
            "bias_4h": bias_4h,
            "bias": bias,
            "ok": ok,
            "detail": detail,
        }

    except Exception as e:
        # Mai bloccare tutto per errore MTF: log e lascia ok=True
        return default | {
            "detail": f"MTF v3: exception {e!r} @ {ts_str}"
        }
