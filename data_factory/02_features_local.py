# data_factory/02_features_local.py
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# === PATHS LOCALI ===
BASE_DIR = os.path.expanduser("~/cerbero-coscienza")
RAW_DIR = os.path.join(BASE_DIR, "local_raw", "ask_parquet")
OUT_BASE = os.path.join(BASE_DIR, "local_features")

# timeframes che vogliamo mantenere in locale
TF_RULES: Dict[str, str] = {
    "5m": "5min",
    "15m": "15min",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
}

# Simboli (devono combaciare con quelli che ingestiamo)
SYMBOLS: List[str] = [
    "EURUSD","GBPUSD","AUDUSD","USDJPY","USDCHF","USDCAD",
    "EURJPY","AUDJPY","CADJPY","GBPJPY",
    "XAUUSD","XAGUSD","LIGHTCMDUSD",
    "BTCUSD","ETHUSD",
]

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _read_raw(sym: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"{sym}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"RAW missing: {path}")
    df = pd.read_parquet(path)
    # index deve essere datetime tz-aware
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.loc[~df.index.isna()].sort_index()
    # keep OHLCV
    keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    df = df[keep].copy()
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    rs = up.ewm(alpha=1/period, adjust=False).mean() / (dn.ewm(alpha=1/period, adjust=False).mean() + 1e-12)
    return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]; low = df["low"]; close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def _bb(close: pd.Series, period: int = 20, k: float = 2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    up = ma + k*sd
    dn = ma - k*sd
    width = (up - dn) / (ma.abs() + 1e-12)
    pos = (close - dn) / ((up - dn) + 1e-12)
    return ma, up, dn, width, pos

def _zscore(s: pd.Series, window: int = 200) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - mu) / (sd + 1e-12)

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature set *minimo ma solido* per far ripartire il loop:
    - EMA20/EMA50 + slope + distance
    - RSI14
    - ATR14 + ATR z-score
    - BB width/pos + BB width z-score + squeeze
    - trend/choppy (semplici)
    - hour/dow
    """
    out = df.copy()

    out["ema20"] = _ema(out["close"], 20)
    out["ema50"] = _ema(out["close"], 50)
    out["ema20_slope"] = out["ema20"].diff()
    out["ema50_slope"] = out["ema50"].diff()
    out["ema_distance"] = (out["ema20"] - out["ema50"]) / (out["close"].abs() + 1e-12)

    out["rsi14"] = _rsi(out["close"], 14)

    out["atr14"] = _atr(out, 14)
    out["atr_z"] = _zscore(out["atr14"], 200)

    ma, up, dn, width, pos = _bb(out["close"], 20, 2.0)
    out["bb_width"] = width
    out["bb_pos"] = pos
    out["bb_width_z"] = _zscore(out["bb_width"].fillna(0.0), 200)

    # squeeze: bb_width molto bassa rispetto alla sua media
    out["bb_squeeze"] = (out["bb_width_z"] < -0.75).astype(int)

    # regime semplice: trend se |ema_distance| alto, choppy se basso
    out["trend"] = (out["ema_distance"].abs() > 0.002).astype(int)
    out["choppy"] = (out["ema_distance"].abs() < 0.0008).astype(int)

    out["hour"] = out.index.hour
    out["dow"] = out.index.dayofweek

    # pulizia finale
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna()

    return out

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    if "volume" in df.columns:
        v = df["volume"].resample(rule).sum()
        out = pd.concat([o,h,l,c,v], axis=1)
        out.columns = ["open","high","low","close","volume"]
    else:
        out = pd.concat([o,h,l,c], axis=1)
        out.columns = ["open","high","low","close"]
    out = out.dropna()
    return out

def main(argv: List[str]) -> int:
    os.makedirs(OUT_BASE, exist_ok=True)
    for tf in TF_RULES.keys():
        os.makedirs(os.path.join(OUT_BASE, f"features_{tf}"), exist_ok=True)

    print(f"[{_now_utc()}] 02_features_local START raw_dir={RAW_DIR} out_base={OUT_BASE}")

    for sym in SYMBOLS:
        try:
            raw = _read_raw(sym)
        except Exception as e:
            print(f"[{_now_utc()}] FAIL read_raw {sym}: {e}", file=sys.stderr)
            continue

        # 5m base (TwelveData già 5min, ma lo normalizziamo comunque)
        try:
            base_5m = _resample_ohlcv(raw, TF_RULES["5m"])
        except Exception as e:
            print(f"[{_now_utc()}] FAIL resample 5m {sym}: {e}", file=sys.stderr)
            continue

        # per ogni tf: resample da 5m -> compute features -> write parquet
        for tf, rule in TF_RULES.items():
            try:
                if tf == "5m":
                    ohlcv = base_5m
                else:
                    ohlcv = _resample_ohlcv(base_5m, rule)

                feats = compute_features(ohlcv)

                out_path = os.path.join(OUT_BASE, f"features_{tf}", f"{sym}.parquet")
                feats.to_parquet(out_path, index=True)
                print(f"[{_now_utc()}] OK features {sym} [{tf}] rows={len(feats)} -> {out_path}")
            except Exception as e:
                print(f"[{_now_utc()}] FAIL features {sym} [{tf}]: {e}", file=sys.stderr)

    print(f"[{_now_utc()}] 02_features_local DONE")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
