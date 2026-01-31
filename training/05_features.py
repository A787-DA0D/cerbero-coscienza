import argparse, io, math
import numpy as np
import pandas as pd
from google.cloud import storage

BUCKET = "cerbero-data-processed-gns"
SRC_PREFIX = "features_merged"     # output dello step 04
OUT_PREFIX = "features_final"      # output di questo step

# --------- indicatori base (no dipendenze esterne) ---------
def ema(s, span):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def sma(s, n):
    return s.rolling(n, min_periods=n).mean()

def rsi(close, n=14):
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(n, min_periods=n).mean()
    roll_down = pd.Series(down, index=close.index).rolling(n, min_periods=n).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def stoch(high, low, close, k=14, d=3):
    lowest = low.rolling(k, min_periods=k).min()
    highest = high.rolling(k, min_periods=k).max()
    fast_k = 100 * (close - lowest) / (highest - lowest)
    slow_d = fast_k.rolling(d, min_periods=d).mean()
    return fast_k, slow_d

def macd(close, f=12, s=26, sig=9):
    f_ema = ema(close, f)
    s_ema = ema(close, s)
    line = f_ema - s_ema
    signal = ema(line, sig)
    hist = line - signal
    return line, signal, hist

def boll(close, n=20, mult=2.0):
    m = sma(close, n)
    std = close.rolling(n, min_periods=n).std()
    upper = m + mult * std
    lower = m - mult * std
    width = (upper - lower) / m
    return m, upper, lower, width

# --------- I/O helpers ---------
def load_merged(client, sym):
    blob = client.bucket(BUCKET).blob(f"{SRC_PREFIX}/{sym}.parquet")
    b = blob.download_as_bytes()
    df = pd.read_parquet(io.BytesIO(b))
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def save_df(client, sym, df):
    out = client.bucket(BUCKET).blob(f"{OUT_PREFIX}/{sym}.parquet")
    out.upload_from_string(df.to_parquet(), content_type="application/octet-stream")
    print(f"✅ {sym}: features {len(df):,} righe → gs://{BUCKET}/{OUT_PREFIX}/{sym}.parquet")

# --------- pipeline features ---------
def build_features(sym):
    client = storage.Client()
    x = load_merged(client, sym)

    # colonne base 5m
    close = x["close_5m"]; high=x["high_5m"]; low=x["low_5m"]; open_=x["open_5m"]

    # 1) Trend (SMA/EMA)
    x["sma_20_5m"] = sma(close, 20)
    x["ema_20_5m"] = ema(close, 20)
    x["ema_50_5m"] = ema(close, 50)

    # 2) Momentum (RSI, Stoch)
    x["rsi_14_5m"] = rsi(close, 14)
    x["stoch_k_14_5m"], x["stoch_d_3_5m"] = stoch(high, low, close, 14, 3)

    # 3) MACD
    x["macd_line_5m"], x["macd_signal_5m"], x["macd_hist_5m"] = macd(close, 12, 26, 9)

    # 4) Volatilità (Bollinger)
    x["bb_mid_20_5m"], x["bb_up_20_5m"], x["bb_dn_20_5m"], x["bb_width_20_5m"] = boll(close, 20, 2.0)

    # 5) Estratti multi-TF (close/ema/rsi su 15m/4h/1d)
    for tf in ("15m","4h","1d"):
        for col in ("close","ema_20","rsi_14"):
            # Se non esiste (es. ema_20_15m) proviamo a calcolarla dal set del TF
            if f"{col}_{tf}" not in x.columns:
                if f"close_{tf}" in x.columns and col == "ema_20":
                    x[f"ema_20_{tf}"] = ema(x[f"close_{tf}"], 20)
                elif f"close_{tf}" in x.columns and col == "rsi_14":
                    x[f"rsi_14_{tf}"] = rsi(x[f"close_{tf}"], 14)
            # niente else: se manca, resterà NaN e verrà droppato sotto

    # Pulizia: rimuovi righe con NaN nelle colonne principali
    core_cols = [
        "sma_20_5m","ema_20_5m","ema_50_5m","rsi_14_5m","stoch_k_14_5m","stoch_d_3_5m",
        "macd_line_5m","macd_signal_5m","macd_hist_5m","bb_width_20_5m",
        "close_5m","close_15m","close_4h","close_1d"
    ]
    keep = [c for c in core_cols if c in x.columns]
    x = x.dropna(subset=keep)

    # Target provvisorio per Radar (direzione a 5 barre ~ 25m)
    x["ret_fwd_5"] = close.pct_change(5).shift(-5)
    x["y_up"] = (x["ret_fwd_5"] > 0).astype(int)

    # Salva
    save_df(client, sym, x)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    args = ap.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    for s in symbols:
        try:
            build_features(s)
        except Exception as e:
            print(f"❌ Errore su {s}: {e}")
