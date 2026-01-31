# training/contextual_signals.py
import pandas as pd
import numpy as np

def add_context_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # momentum bruto (candele forti)
    body = (out["close"] - out["open"]).abs()
    range_ = (out["high"] - out["low"]).replace(0, np.nan)
    out["candle_body_ratio"] = (body / range_).fillna(0)
    out["strong_bull"] = ((out["close"] > out["open"]) & (out["candle_body_ratio"] > 0.7)).astype(int)
    out["strong_bear"] = ((out["close"] < out["open"]) & (out["candle_body_ratio"] > 0.7)).astype(int)

    # “session squeeze” sul range intraday
    win = 48
    intraday_comp = (out["high"].rolling(win).max() - out["low"].rolling(win).min()) / out["close"].rolling(win).mean()
    out["session_squeeze"] = (intraday_comp < intraday_comp.rolling(500).quantile(0.2)).astype(int)

    # proxy “London/NY open” per TF corti (5m/15m) usando ore locali UTC
    # (se index è tz-naive supponiamo UTC)
    hrs = out.index.hour
    out["is_london_open"] = ((hrs >= 7) & (hrs <= 9)).astype(int)     # 07–09 UTC ~ 08–10 CET
    out["is_ny_open"]     = ((hrs >= 13) & (hrs <= 15)).astype(int)   # 13–15 UTC

    # reazioni a open (breakout dopo compressione)
    out["open_breakout"] = ((out["is_london_open"] | out["is_ny_open"]) & (out["range_break_up"] | out["range_break_dn"])).astype(int)

    # inside bar continuation “pulito”
    out["inside_continuation"] = ((out["inside_bar"] == 1) & (out["bb_squeeze"] == 1)).astype(int)

    return out.replace([np.inf, -np.inf], np.nan).fillna(0)
