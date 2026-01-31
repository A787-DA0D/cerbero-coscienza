# training/02_features.py
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd

# === Bucket I/O (immutati, ma noi scriviamo SOLO locale) ===
RAW_BUCKET = "cerbero-data-processed-gns"   # input: ask_parquet/<SYM>.parquet
OUT_BUCKET = "cerbero-data-processed-gns"   # output: features_<tf>/<SYM>.parquet (uri logico)

# === Timeframe mapping (no deprecation) ===
# aggiungo 1h perché serve spesso (e tu lo vuoi)
TFS  = ["5m", "15m", "1h", "4h", "1d"]
RULE = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}

# === Import moduli esistenti ===
from features.strategies_pro import add_indicator_features
from training.contextual_signals import add_context_signals


# ============================================================
# I/O helper
# ============================================================
def read_parquet_any(path: str) -> pd.DataFrame:
    """Legge parquet da path locale o gs:// usando pandas (se hai permessi)."""
    return pd.read_parquet(path)


def write_parquet_local(df: pd.DataFrame, path: str) -> None:
    """
    Scrive SEMPRE in locale.
    Se path è gs://bucket/prefix/file.parquet -> salva in ./local_features/prefix/file.parquet
    """
    if path.startswith("gs://"):
        rel = path[len("gs://"):]  # bucket/...
        parts = rel.split("/", 1)
        rel_path = parts[1] if len(parts) == 2 else rel
        local_path = os.path.join("local_features", rel_path)
    else:
        local_path = path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    df.to_parquet(local_path, index=True)
    print(f"[LOCAL] wrote {local_path}")


# ============================================================
# Resampling OHLCV
# ============================================================
def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Garantisce colonne OHLCV e tipi numerici puliti."""
    x = df.copy()

    # colonne minime richieste
    required = ["open", "high", "low", "close"]
    for c in required:
        if c not in x.columns:
            raise ValueError(f"RAW missing required col '{c}'")

    if "volume" not in x.columns:
        x["volume"] = 0.0

    # cast numerico robusto
    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    # pulizia inf/nan
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])

    return x


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLCV con aggregazioni corrette.
    rule: '5min','15min','1h','4h','1D'
    """
    x = df.copy()

    x.index = pd.to_datetime(x.index, utc=True, errors="coerce")
    x = x.loc[~x.index.isna()].sort_index()

    # rimuovi duplicati timestamp (tieni ultimo)
    x = x[~x.index.duplicated(keep="last")]

    x = _ensure_ohlcv(x)

    o = x["open"].resample(rule).first()
    h = x["high"].resample(rule).max()
    l = x["low"].resample(rule).min()
    c = x["close"].resample(rule).last()
    v = x["volume"].resample(rule).sum()

    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()

    # garantisci numerico
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna()
    return out


# ============================================================
# Regime / Contestuale avanzato (già esistente)
# ============================================================
def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature di regime/volatilità/tempo:
      atr_z, bb_width_z, trend, choppy, hour, dow
    """
    df = df.copy()

    required = ["atr14", "bb_width", "ema50", "ema50_slope", "high", "low"]
    if not all(col in df.columns for col in required):
        return df

    long_roll = 200
    choppy_roll = 50

    atr = df["atr14"]
    atr_ma = atr.rolling(long_roll, min_periods=long_roll // 2).mean()
    atr_std = atr.rolling(long_roll, min_periods=long_roll // 2).std()
    df["atr_z"] = (atr - atr_ma) / (atr_std + 1e-6)

    bbw = df["bb_width"]
    bbw_ma = bbw.rolling(long_roll, min_periods=long_roll // 2).mean()
    bbw_std = bbw.rolling(long_roll, min_periods=long_roll // 2).std()
    df["bb_width_z"] = (bbw - bbw_ma) / (bbw_std + 1e-6)

    ema50 = df["ema50"]
    ema_std = ema50.rolling(long_roll, min_periods=long_roll // 2).std()
    df["trend"] = df["ema50_slope"] / (ema_std + 1e-6)

    high_roll = df["high"].rolling(choppy_roll, min_periods=choppy_roll // 2).max()
    low_roll = df["low"].rolling(choppy_roll, min_periods=choppy_roll // 2).min()
    raw_choppy = (df["atr14"] * choppy_roll) / ((high_roll - low_roll).abs() + 1e-6)

    ch_ma = raw_choppy.rolling(long_roll, min_periods=long_roll // 2).mean()
    ch_std = raw_choppy.rolling(long_roll, min_periods=long_roll // 2).std()
    df["choppy"] = (raw_choppy - ch_ma) / (ch_std + 1e-6)

    if isinstance(df.index, pd.DatetimeIndex):
        df["hour"] = df.index.hour
        df["dow"] = df.index.dayofweek

    return df


# ============================================================
# ICT / SMC – STRUTTURA (Swing, BOS, MSS)
# ============================================================
def add_swing_points(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    df = df.copy()
    if not {"high", "low"}.issubset(df.columns):
        return df

    h = df["high"]
    l = df["low"]

    swing_high = pd.Series(True, index=df.index)
    swing_low = pd.Series(True, index=df.index)

    for i in range(1, left + 1):
        swing_high &= h > h.shift(i)
        swing_low &= l < l.shift(i)

    for i in range(1, right + 1):
        swing_high &= h >= h.shift(-i)
        swing_low &= l <= l.shift(-i)

    df["swing_high"] = swing_high.fillna(False)
    df["swing_low"] = swing_low.fillna(False)
    return df


def add_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not {"high", "low", "close"}.issubset(df.columns):
        return df

    if "swing_high" not in df.columns or "swing_low" not in df.columns:
        df = add_swing_points(df)

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    swing_high = df["swing_high"].to_numpy(dtype=bool)
    swing_low = df["swing_low"].to_numpy(dtype=bool)

    n = len(df)
    bos_bull = np.zeros(n, dtype=bool)
    bos_bear = np.zeros(n, dtype=bool)
    mss_bull = np.zeros(n, dtype=bool)
    mss_bear = np.zeros(n, dtype=bool)
    ms_trend = np.zeros(n, dtype=int)

    last_swing_high = np.nan
    last_swing_low = np.nan

    for i in range(n):
        if swing_high[i]:
            last_swing_high = high[i]
        if swing_low[i]:
            last_swing_low = low[i]

        prev_trend = ms_trend[i - 1] if i > 0 else 0
        new_trend = prev_trend

        if not np.isnan(last_swing_high) and close[i] > last_swing_high:
            bos_bull[i] = True
            new_trend = 1
        elif not np.isnan(last_swing_low) and close[i] < last_swing_low:
            bos_bear[i] = True
            new_trend = -1

        if bos_bull[i] and prev_trend == -1:
            mss_bull[i] = True
        if bos_bear[i] and prev_trend == 1:
            mss_bear[i] = True

        ms_trend[i] = new_trend

    df["bos_bull"] = bos_bull
    df["bos_bear"] = bos_bear
    df["mss_bull"] = mss_bull
    df["mss_bear"] = mss_bear
    df["ms_trend"] = ms_trend
    return df


# ============================================================
# ICT – PD ARRAYS (FVG, Order Block, Breaker)
# ============================================================
def _compute_fvg_state(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not {"high", "low", "close"}.issubset(df.columns):
        return df

    high = df["high"]
    low = df["low"]
    close = df["close"]

    h1 = high.shift(1)
    l1 = low.shift(1)
    h3 = high.shift(-1)
    l3 = low.shift(-1)

    bull_cond = l3 > h1
    bear_cond = h3 < l1

    bull_gap_low = np.where(bull_cond, h1, np.nan)
    bull_gap_high = np.where(bull_cond, l3, np.nan)
    bear_gap_low = np.where(bear_cond, h3, np.nan)
    bear_gap_high = np.where(bear_cond, l1, np.nan)

    n = len(df)
    in_bull = np.zeros(n, dtype=bool)
    in_bear = np.zeros(n, dtype=bool)

    active_bull_low = np.nan
    active_bull_high = np.nan
    active_bear_low = np.nan
    active_bear_high = np.nan

    h_arr = high.to_numpy()
    l_arr = low.to_numpy()
    c_arr = close.to_numpy()

    bull_gap_low_arr = np.asarray(bull_gap_low, dtype=float)
    bull_gap_high_arr = np.asarray(bull_gap_high, dtype=float)
    bear_gap_low_arr = np.asarray(bear_gap_low, dtype=float)
    bear_gap_high_arr = np.asarray(bear_gap_high, dtype=float)

    bull_cond_arr = bull_cond.to_numpy(dtype=bool)
    bear_cond_arr = bear_cond.to_numpy(dtype=bool)

    for i in range(n):
        if bull_cond_arr[i]:
            active_bull_low = bull_gap_low_arr[i]
            active_bull_high = bull_gap_high_arr[i]
        if bear_cond_arr[i]:
            active_bear_low = bear_gap_low_arr[i]
            active_bear_high = bear_gap_high_arr[i]

        if not np.isnan(active_bull_low):
            if (l_arr[i] <= active_bull_low) and (h_arr[i] >= active_bull_high):
                active_bull_low = np.nan
                active_bull_high = np.nan

        if not np.isnan(active_bear_low):
            if (l_arr[i] <= active_bear_low) and (h_arr[i] >= active_bear_high):
                active_bear_low = np.nan
                active_bear_high = np.nan

        if not np.isnan(active_bull_low) and not np.isnan(active_bull_high):
            in_bull[i] = (c_arr[i] >= active_bull_low) and (c_arr[i] <= active_bull_high)

        if not np.isnan(active_bear_low) and not np.isnan(active_bear_high):
            in_bear[i] = (c_arr[i] >= active_bear_low) and (c_arr[i] <= active_bear_high)

    df["fvg_bull"] = bull_cond.fillna(False)
    df["fvg_bear"] = bear_cond.fillna(False)
    df["fvg_bull_gap_low"] = bull_gap_low
    df["fvg_bull_gap_high"] = bull_gap_high
    df["fvg_bear_gap_low"] = bear_gap_low
    df["fvg_bear_gap_high"] = bear_gap_high
    df["in_fvg_bull"] = in_bull
    df["in_fvg_bear"] = in_bear
    return df


def _tag_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not {"open", "close", "high", "low"}.issubset(df.columns):
        return df
    if "bos_bull" not in df.columns or "bos_bear" not in df.columns:
        df = add_market_structure(df)

    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    bos_bull = df["bos_bull"].to_numpy(dtype=bool)
    bos_bear = df["bos_bear"].to_numpy(dtype=bool)

    n = len(df)
    ob_bull = np.zeros(n, dtype=bool)
    ob_bear = np.zeros(n, dtype=bool)
    ob_bull_price = np.full(n, np.nan)
    ob_bear_price = np.full(n, np.nan)

    last_bear_idx = None
    last_bull_idx = None

    for i in range(n):
        if c[i] < o[i]:
            last_bear_idx = i
        elif c[i] > o[i]:
            last_bull_idx = i

        if bos_bull[i] and last_bear_idx is not None:
            ob_bull[last_bear_idx] = True
            ob_bull_price[last_bear_idx] = o[last_bear_idx]
        if bos_bear[i] and last_bull_idx is not None:
            ob_bear[last_bull_idx] = True
            ob_bear_price[last_bull_idx] = o[last_bull_idx]

    df["ob_bull"] = ob_bull
    df["ob_bear"] = ob_bear
    df["ob_bull_open"] = ob_bull_price
    df["ob_bear_open"] = ob_bear_price
    return df


def _tag_breaker_blocks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not {"ob_bull", "ob_bear", "ob_bull_open", "ob_bear_open", "high", "low", "close"}.issubset(df.columns):
        return df

    ob_bull = df["ob_bull"].to_numpy(dtype=bool)
    ob_bear = df["ob_bear"].to_numpy(dtype=bool)
    ob_bull_open = df["ob_bull_open"].to_numpy()
    ob_bear_open = df["ob_bear_open"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    n = len(df)
    breaker_bull = np.zeros(n, dtype=bool)
    breaker_bear = np.zeros(n, dtype=bool)

    active_bear_ob_level = np.nan
    active_bull_ob_level = np.nan

    for i in range(n):
        if ob_bear[i] and not np.isnan(ob_bear_open[i]):
            active_bear_ob_level = ob_bear_open[i]
        if ob_bull[i] and not np.isnan(ob_bull_open[i]):
            active_bull_ob_level = ob_bull_open[i]

        if not np.isnan(active_bear_ob_level):
            if close[i] > active_bear_ob_level and low[i] < active_bear_ob_level:
                breaker_bull[i] = True
                active_bear_ob_level = np.nan

        if not np.isnan(active_bull_ob_level):
            if close[i] < active_bull_ob_level and high[i] > active_bull_ob_level:
                breaker_bear[i] = True
                active_bull_ob_level = np.nan

    df["breaker_bull"] = breaker_bull
    df["breaker_bear"] = breaker_bear
    return df


def add_pd_array_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _compute_fvg_state(df)
    df = _tag_order_blocks(df)
    df = _tag_breaker_blocks(df)
    return df


# ============================================================
# ICT – PREMIUM vs DISCOUNT
# ============================================================
def add_premium_discount(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not {"high", "low", "close"}.issubset(df.columns):
        return df

    if "swing_high" not in df.columns or "swing_low" not in df.columns:
        df = add_swing_points(df)

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    sh = df["swing_high"].to_numpy(dtype=bool)
    sl = df["swing_low"].to_numpy(dtype=bool)

    n = len(df)
    range_high = np.full(n, np.nan)
    range_low = np.full(n, np.nan)

    last_high = np.nan
    last_low = np.nan

    for i in range(n):
        if sh[i]:
            last_high = high[i]
        if sl[i]:
            last_low = low[i]
        range_high[i] = last_high
        range_low[i] = last_low

    rng = range_high - range_low
    pos = (close - range_low) / (rng + 1e-9)
    valid = (~np.isnan(range_high)) & (~np.isnan(range_low)) & (rng > 0)

    premium = np.zeros(n, dtype=bool)
    discount = np.zeros(n, dtype=bool)
    premium[valid & (pos > 0.5)] = True
    discount[valid & (pos < 0.5)] = True

    df["range_high"] = range_high
    df["range_low"] = range_low
    df["range_pos"] = np.where(valid, pos, np.nan)
    df["is_premium"] = premium
    df["is_discount"] = discount
    return df


# ============================================================
# ICT – LIQUIDITY POOLS
# ============================================================
def add_liquidity_pools(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    df = df.copy()
    if not {"high", "low", "close"}.issubset(df.columns):
        return df

    eq_tol = df["close"].abs() * 0.0002
    eq_high = (df["high"] - df["high"].shift(1)).abs() <= eq_tol
    eq_low = (df["low"] - df["low"].shift(1)).abs() <= eq_tol

    df["eq_high"] = eq_high.fillna(False)
    df["eq_low"] = eq_low.fillna(False)

    df["bsl_level"] = df["high"].shift(1).rolling(lookback, min_periods=1).max()
    df["ssl_level"] = df["low"].shift(1).rolling(lookback, min_periods=1).min()

    close = df["close"]
    df["hit_bsl"] = (close >= df["bsl_level"]) & df["bsl_level"].notna()
    df["hit_ssl"] = (close <= df["ssl_level"]) & df["ssl_level"].notna()
    return df


# ============================================================
# ICT – TEMPO (Kill Zones & AMD)
# ============================================================
def add_time_ict_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    idx = df.index
    hour = idx.hour

    df["killzone_london"] = (hour >= 7) & (hour < 10)
    df["killzone_ny"] = (hour >= 12) & (hour < 15)

    day = idx.normalize()
    df["midnight_open"] = df.groupby(day)["open"].transform("first")

    amd_phase = np.full(len(df), "", dtype=object)
    amd_phase[(hour >= 0) & (hour < 7)] = "A"
    amd_phase[(hour >= 7) & (hour < 12)] = "M"
    amd_phase[(hour >= 12)] = "D"
    df["amd_A"] = (amd_phase == "A")
    df["amd_M"] = (amd_phase == "M")
    df["amd_D"] = (amd_phase == "D")
    df["amd_disp"] = df["close"] - df["midnight_open"]
    return df


def add_ict_smc_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_swing_points(df)
    df = add_market_structure(df)
    df = add_pd_array_features(df)
    df = add_premium_discount(df)
    df = add_liquidity_pools(df)
    df = add_time_ict_features(df)
    return df


# ============================================================
# Pipeline per simbolo
# ============================================================
def run_for_symbol(sym: str) -> None:
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # repo root
    LOCAL_RAW_DIR = os.path.join(BASE_DIR, "local_raw", "ask_parquet")
    src_local = os.path.join(LOCAL_RAW_DIR, f"{sym}.parquet")

    if not os.path.exists(src_local):
        raise FileNotFoundError(f"RAW mancante: {src_local} (prima esegui ingest twelvedata)")

    # 1) leggo RAW
    df = read_parquet_any(src_local)

    # 2) pulizia indice
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.loc[~df.index.isna()].sort_index()

    # 3) ciclo TF → resample → indicatori → context → regime → ICT/SMC → write
    for tf in TFS:
        rule = RULE[tf]
        x = resample_ohlcv(df, rule)

        # Indicatori tecnici PRO
        x = add_indicator_features(x)

        # Segnali contestuali base
        try:
            x = add_context_signals(x, symbol=sym, tf=tf)
        except TypeError:
            x = add_context_signals(x)

        # Regime / volatilità / tempo
        x = add_regime_features(x)

        # ICT / SMC / IPDA
        x = add_ict_smc_features(x)

        # 4) scrivo SOLO in locale (così GCS non c'entra nulla)
        out_dir = os.path.join(BASE_DIR, "local_features", f"features_{tf}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{sym}.parquet")
        x.to_parquet(out_path, index=True)

        print(f"[LOCAL] wrote {out_path}")
        print(f"✅ {sym} [{tf}] features OK  cols={len(x.columns)} rows={len(x)}")



# ============================================================
# Main
# ============================================================
def _parse_symbols(s: str) -> List[str]:
    out = [x.strip().upper() for x in s.split(",")]
    return [x for x in out if x]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help='Es: "EURUSD" oppure "EURUSD,USDJPY"')
    args = ap.parse_args()

    symbols = _parse_symbols(args.symbols)
    for sym in symbols:
        run_for_symbol(sym)

    print("🏁 Done.")
