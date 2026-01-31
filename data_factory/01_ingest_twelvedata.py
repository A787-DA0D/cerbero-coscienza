# data_factory/01_ingest_twelvedata.py
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

# === CONFIG ===
API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
BASE_URL = "https://api.twelvedata.com/time_series"

# RAW locale (sostituisce gs://.../ask_parquet/)
RAW_DIR = Path.home() / "cerbero-coscienza" / "local_raw" / "ask_parquet"

# Mappa Cerbero symbol -> Twelve Data symbol
SYMBOL_MAP: Dict[str, str] = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "AUDUSD": "AUD/USD",
    "USDJPY": "USD/JPY",
    "USDCHF": "USD/CHF",
    "USDCAD": "USD/CAD",
    "EURJPY": "EUR/JPY",
    "AUDJPY": "AUD/JPY",
    "CADJPY": "CAD/JPY",
    "GBPJPY": "GBP/JPY",
    "XAUUSD": "XAU/USD",
    "XAGUSD": "XAG/USD",
    "LIGHTCMDUSD": "CL",     # Crude Oil (se non va, poi lo sistemiamo)
    "BTCUSD": "BTC/USD",
    "ETHUSD": "ETH/USD",
}

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch_time_series(symbol_td: str, interval: str, outputsize: int = 2000) -> pd.DataFrame:
    params = {
        "symbol": symbol_td,
        "interval": interval,
        "outputsize": outputsize,
        "format": "JSON",
        "apikey": API_KEY,
        "timezone": "UTC",
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(f"TwelveData error: {data}")

    values = data.get("values")
    if not values:
        raise RuntimeError(f"No 'values' in response: {data}")

    df = pd.DataFrame(values)
    if "datetime" not in df.columns:
        raise RuntimeError(f"Missing datetime col: {df.columns}")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")

    # cast numerici
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna()
    return df

def write_parquet_local_atomic(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=True)
    tmp_path.replace(out_path)  # atomic rename su stesso filesystem

def main(argv: List[str]) -> int:
    if not API_KEY:
        print("ERROR: TWELVE_DATA_API_KEY missing (source secrets/twelvedata.env).", file=sys.stderr)
        return 2

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    symbols = list(SYMBOL_MAP.keys())

    print(f"[{_now_utc()}] TwelveData ingest start symbols={len(symbols)} raw_dir={RAW_DIR}")

    for sym in symbols:
        td_sym = SYMBOL_MAP[sym]
        try:
            # RAW base: 5min (come prima)
            df5 = fetch_time_series(td_sym, "5min", outputsize=5000)
            out_file = RAW_DIR / f"{sym}.parquet"
            write_parquet_local_atomic(df5, out_file)
            print(f"[{_now_utc()}] OK RAW {sym} -> {out_file} rows={len(df5)}")
        except Exception as e:
            print(f"[{_now_utc()}] FAIL RAW {sym}: {e}", file=sys.stderr)

        time.sleep(0.25)  # throttle gentile

    print(f"[{_now_utc()}] TwelveData ingest done")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
