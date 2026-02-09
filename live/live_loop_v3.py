# live/live_loop_v3.py
from __future__ import annotations

"""
Cerbero Coscienza v3 – live loop ufficiale (Radar + IPDA LSTM).

- Radar v3 (grafico/tecnico/contestuale) fa il timing.
- IPDA LSTM (5m/15m) fa da filtro direzionale (Generale) con diritto di veto.

Regole:
- Se IPDA strong bull => accetto SOLO LONG.
- Se IPDA strong bear => accetto SOLO SHORT.
- Se IPDA neutral => radar libero.
- MTF (1D/4H) è solo filtro: NON tradare su 1D/4H.

ARCHITETTURA (VINCOLANTE):
- La Coscienza genera RawSignal (NO campi tecnici di esecuzione specifici broker)
- normalize_intent_for_coordinator() (in coordinator_client) produce TradeIntent FLAT
- Il Coordinator valida/clampa/protegge
"""

# --- Soppressione warning fastidiosi per dry-run ---
import warnings


# =========================
# MACRO_PRICE_FEED_TWELVEDATA (NO parquet / NO features)
# =========================
import os, time
import requests

_TD_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
_TD_QUOTE_URL = "https://api.twelvedata.com/quote"

_macro_cache = {}  # key -> (ts, value)

def _td_quote(symbol_td: str) -> float | None:
    if not _TD_API_KEY:
        return None
    try:
        r = requests.get(_TD_QUOTE_URL, params={"symbol": symbol_td, "apikey": _TD_API_KEY}, timeout=10)
        j = r.json()
        if isinstance(j, dict) and j.get("status") == "error":
            return None
        px = j.get("price") or j.get("close")
        return float(px) if px is not None else None
    except Exception:
        return None

def _macro_get(key: str, ttl_sec: int, fetch_fn):
    now = time.time()
    ts, val = _macro_cache.get(key, (0.0, None))
    if val is not None and (now - ts) < ttl_sec:
        return val
    val = fetch_fn()
    _macro_cache[key] = (now, val)
    return val

def get_macro_prices() -> dict:
    # DOLLARIDXUSD is macro-only; use TwelveData USDX (or DX) as proxy
    usdx = _macro_get("USDX", ttl_sec=60, fetch_fn=lambda: _td_quote("USDX"))
    return {
        "DOLLARIDXUSD": usdx
    }
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

import argparse
import time
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from live.config_v3 import SYMBOLS, TIMEFRAMES, TELEGRAM_ALERTS_ENABLED
from live.risk_policy import load_risk_policy
from live.loader_v3 import load_radar_config
from live.signal_classifier_v3 import classify_signal
from live.risk_engine_v3 import compute_position_size
from live.context_filters import compute_context_status, check_economic_block
from training.mtf_logic import compute_mtf_status
from live.coordinator_client import send_trade_intent
from live.alerts import send_telegram_alert
from live.numpy_compat import *  # shim numpy._core.numeric
from live.ipda_lstm import IPDALSTM

FEATURES_BUCKET = "cerbero-data-processed-gns"

# IMPORTANTISSIMO: ordine feature LSTM = stesso ordine usato nel dataset (84 features)
LSTM_FEATURE_COLS: List[str] = [
    "open","high","low","close","volume",
    "ema20","ema50","ema20_slope","ema50_slope","ema_distance",
    "rsi14","stoch_k","stoch_d","bb_width","bb_pos","bb_squeeze",
    "atr14","atr_spike","rsi_bear_div","rsi_bull_div",
    "inside_bar","inside_break_up","inside_break_dn",
    "flag_setup","flag_break","pullback_long","pullback_short",
    "range_high_n","range_low_n","range_break_up","range_break_dn",
    "candle_body_ratio","strong_bull","strong_bear","session_squeeze",
    "is_london_open","is_ny_open","open_breakout","inside_continuation",
    "atr_z","bb_width_z","trend","choppy","hour","dow",
    "swing_high","swing_low","bos_bull","bos_bear","mss_bull","mss_bear","ms_trend",
    "fvg_bull","fvg_bear","fvg_bull_gap_low","fvg_bull_gap_high","fvg_bear_gap_low","fvg_bear_gap_high",
    "in_fvg_bull","in_fvg_bear",
    "ob_bull","ob_bear","ob_bull_open","ob_bear_open",
    "breaker_bull","breaker_bear",
    "range_high","range_low","range_pos","is_premium","is_discount",
    "eq_high","eq_low","bsl_level","ssl_level","hit_bsl","hit_ssl",
    "killzone_london","killzone_ny","midnight_open",
    "amd_A","amd_M","amd_D","amd_disp",
]

SEQ_LEN = 128  # come training

TENANT_EMAIL = os.environ.get("CERBERO_TENANT_EMAIL", "").strip()

IPDA_MODEL_PATHS = {
    "5m":  "models/coscienza_v3_lstm_5m_BEST_LOSS.pt",
    "15m": "models/coscienza_v3_lstm_15m_BEST_LOSS.pt",
}


def _log(msg: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    print(f"[{now}] {msg}")



def _env_flag(name: str) -> bool:
    v = (os.environ.get(name, "") or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _cooldown_sec() -> int:
    try:
        return int((os.environ.get("CERBERO_SYMBOL_COOLDOWN_SEC", "") or "").strip() or "900")
    except Exception:
        return 900

def _state_path() -> Path:
    Path("logs").mkdir(parents=True, exist_ok=True)
    return Path("logs") / "live_state.json"

def _load_state() -> dict:
    sp = _state_path()
    if not sp.exists():
        return {}
    try:
        return json.loads(sp.read_text() or "{}")
    except Exception:
        return {}

def _save_state(st: dict) -> None:
    sp = _state_path()
    try:
        sp.write_text(json.dumps(st, indent=2, sort_keys=True))
    except Exception:
        pass

def _check_loop_control_or_skip() -> tuple[bool, str]:
    if _env_flag("CERBERO_EMERGENCY_STOP"):
        return False, "EMERGENCY_STOP"
    if _env_flag("CERBERO_PAUSED"):
        return False, "PAUSED"
    return True, "OK"

def _cooldown_allows(symbol: str, now_ts: float) -> tuple[bool, float]:
    cd = float(_cooldown_sec())
    if cd <= 0:
        return True, 0.0
    st = _load_state()
    last = float(st.get("last_intent_ts", {}).get(symbol, 0.0) or 0.0)
    remaining = (last + cd) - now_ts
    if remaining > 0:
        return False, remaining
    return True, 0.0

def _mark_intent_sent(symbol: str, now_ts: float) -> None:
    st = _load_state()
    st.setdefault("last_intent_ts", {})
    st["last_intent_ts"][symbol] = float(now_ts)
    _save_state(st)


def _load_features_df(symbol: str, tf: str) -> Optional[pd.DataFrame]:
    if symbol == "DOLLARIDXUSD":
        return None

    local_path = f"{Path.home()}/cerbero-coscienza/local_features/features_{tf}/{symbol}.parquet"
    try:
        df = pd.read_parquet(local_path)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        _log(f"⚠️  LOCAL read failed {symbol} [{tf}] {local_path}: {e}")

    path = f"gs://{FEATURES_BUCKET}/features_{tf}/{symbol}.parquet"
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        _log(f"⚠️  Impossibile leggere features per {symbol} [{tf}] da {path}: {e}")
        return None

    if df is None or df.empty:
        _log(f"⚠️  Nessuna riga di features per {symbol} [{tf}]")
        return None

    return df


def _build_lstm_seq(df: pd.DataFrame) -> Optional[np.ndarray]:
    if df is None or df.empty:
        return None

    tail = df.tail(SEQ_LEN)
    if len(tail) < SEQ_LEN:
        return None

    x = tail.copy()
    for c in LSTM_FEATURE_COLS:
        if c not in x.columns:
            x[c] = 0.0

    x = x[LSTM_FEATURE_COLS].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return x.to_numpy(dtype=np.float32)


def _radar_side_from_p(p_blend: float) -> str:
    return "long" if float(p_blend) >= 0.5 else "short"


def _compute_strength(p_blend: float) -> float:
    s = abs(float(p_blend) - 0.5) * 2.0
    if s < 0.0:
        s = 0.0
    if s > 1.0:
        s = 1.0
    return s


def _ipda_gate(*, ipda_5m, ipda_15m, radar_side: str) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    if ipda_5m is None or ipda_15m is None:
        reasons.append("ipda_missing")
        return True, reasons

    if ipda_15m.strong:
        bias = ipda_15m.bias
        reasons.append(f"ipda15m_{bias}_{ipda_15m.p_up:.3f}")
        strong = True
    elif ipda_5m.strong:
        bias = ipda_5m.bias
        reasons.append(f"ipda5m_{bias}_{ipda_5m.p_up:.3f}")
        strong = True
    else:
        reasons.append(f"ipda_neutral_5m={ipda_5m.p_up:.3f}_15m={ipda_15m.p_up:.3f}")
        return True, reasons

    if strong and bias == "bull" and radar_side != "long":
        reasons.append("veto_ipda_bull_blocks_short")
        return False, reasons
    if strong and bias == "bear" and radar_side != "short":
        reasons.append("veto_ipda_bear_blocks_long")
        return False, reasons

    return True, reasons


def _compute_sl_tp_from_row(row: pd.Series, side: str, entry_price: float) -> Tuple[Optional[float], Optional[float]]:
    """
    TP v1 istituzionale:
    - SL come prima (swing/level-based + ATR fallback)
    - TP calcolato con RR minimo su distanza SL (default 1.5R)
    - clamp minimo distanza TP per evitare target troppo vicini
    NOTE: broker-aware stop-level/freeze-level verrà gestito in step successivo.
    """
    side = (side or "").lower()
    entry_price = float(entry_price)

    # --- read ATR ---
    atr = None
    if "atr14" in row.index:
        try:
            atr = float(row.get("atr14"))
            if not (atr > 0) or atr != atr:  # nan check
                atr = None
        except Exception:
            atr = None

    # --- read levels ---
    ssl_level = None
    bsl_level = None
    range_low = None
    range_high = None

    def _get_float(k: str):
        try:
            if k in row.index and not pd.isna(row[k]):
                v = float(row[k])
                if np.isfinite(v):
                    return v
        except Exception:
            return None
        return None

    ssl_level = _get_float("ssl_level")
    bsl_level = _get_float("bsl_level")
    range_low  = _get_float("range_low")
    range_high = _get_float("range_high")

    buf = (atr * 0.2) if (atr is not None) else 0.0

    # --- helpers ---
    def _min_tp_distance() -> float:
        # clamp anti-TP troppo vicino: usa ATR se disponibile, altrimenti un minimo percentuale
        # (0.05% di entry) -> evita target minuscoli sui prezzi "grandi"
        pct_min = abs(entry_price) * 0.0005
        atr_min = (atr * 0.3) if (atr is not None) else 0.0
        return max(pct_min, atr_min)

    RR = float(os.environ.get("CERBERO_TP_RR", "1.5"))

    if side == "long":
        # SL
        candidates = [x for x in [ssl_level, range_low] if x is not None]
        if candidates:
            sl = min(candidates) - buf
        elif atr is not None:
            sl = entry_price - 1.5 * atr
        else:
            sl = entry_price * 0.995

        sl = float(sl)
        sl_dist = entry_price - sl
        if not (sl_dist > 0):
            return sl, None  # qualcosa di strano, non inventiamo

        # TP = entry + RR * (entry - SL)
        tp = entry_price + RR * sl_dist

        # clamp distanza minima
        if (tp - entry_price) < _min_tp_distance():
            tp = entry_price + _min_tp_distance()

        return float(sl), float(tp)

    if side == "short":
        # SL
        candidates = [x for x in [bsl_level, range_high] if x is not None]
        if candidates:
            sl = max(candidates) + buf
        elif atr is not None:
            sl = entry_price + 1.5 * atr
        else:
            sl = entry_price * 1.005

        sl = float(sl)
        sl_dist = sl - entry_price
        if not (sl_dist > 0):
            return sl, None

        # TP = entry - RR * (SL - entry)
        tp = entry_price - RR * sl_dist

        # clamp distanza minima
        if (entry_price - tp) < _min_tp_distance():
            tp = entry_price - _min_tp_distance()

        return float(sl), float(tp)

    return None, None


def _get_reference_price_from_row(row: pd.Series) -> Optional[float]:
    for k in ("close", "Close", "c", "price_close"):
        if k in row.index and not pd.isna(row.get(k)):
            try:
                v = float(row.get(k))
                if np.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass
    return None


def _evaluate_symbol_tf(symbol: str, tf: str, radar_cfg, ipda: IPDALSTM, equity: float, dry_run: bool, policy: Dict[str, Any]) -> None:
    now_utc = datetime.now(timezone.utc)

    df = _load_features_df(symbol, tf)
    if df is None:
        return
    row = df.iloc[-1]

    probs = radar_cfg.predict_probas(row)
    p_blend = float(probs.get("blend", probs.get("p_blend", 0.5)))
    strength = _compute_strength(p_blend)
    radar_side = _radar_side_from_p(p_blend)

    ipda_5m = None
    ipda_15m = None
    try:
        df5 = _load_features_df(symbol, "5m")
        seq5 = _build_lstm_seq(df5) if df5 is not None else None
        if seq5 is not None:
            ipda_5m = ipda.predict("5m", seq5)

        df15 = _load_features_df(symbol, "15m")
        seq15 = _build_lstm_seq(df15) if df15 is not None else None
        if seq15 is not None:
            ipda_15m = ipda.predict("15m", seq15)
    except Exception as e:
        _log(f"⚠️  IPDA LSTM error {symbol}: {e}")

    allowed, ipda_reasons = _ipda_gate(ipda_5m=ipda_5m, ipda_15m=ipda_15m, radar_side=radar_side)
    if not allowed:
        _log(f"SKIP {symbol} [{tf}] reason=ipda_veto radar_side={radar_side} ipda={'/'.join(ipda_reasons)}")
        return

    try:
        mtf_status: Dict[str, Any] = compute_mtf_status(symbol, now_utc)
    except TypeError:
        mtf_status = compute_mtf_status(symbol)
    mtf_ok = bool(mtf_status.get("ok", True))

    context_status: Dict[str, Any] = compute_context_status(symbol, tf, row)
    econ_block: Dict[str, Any] = check_economic_block(symbol, now_utc)

    thresholds = radar_cfg.thresholds
    decision = classify_signal(
        symbol=symbol,
        tf=tf,
        p_blend=p_blend,
        strength=strength,
        thresholds=thresholds,
        mtf_ok=mtf_ok,
        context_status=context_status,
        econ_block=econ_block,
        probs=probs,
    )

    if not bool(decision.get("ok", False)):
        _log(
            f"SKIP {symbol} [{tf}] reason={decision.get('reason')} "
            f"p={float(p_blend):.3f} strength={float(strength):.3f} mtf_ok={bool(mtf_ok)} "
            f"news_blocked={bool(econ_block.get('blocked', False))}"
        )
        return

    sig_class = decision.get("class", "A")
    strength_class = ("A_PLUS" if str(sig_class).strip().upper() in ("A+","APLUS","A_PLUS") else str(sig_class).strip().upper())
    reasons: List[str] = [decision.get("reason", "ok")] + ipda_reasons
    side = radar_side

    mark_price = _get_reference_price_from_row(row)
    if mark_price is None:
        _log(f"⚠️  Nessun prezzo reference (close) per {symbol}, salto trade.")
        return

    sl_price, tp_price = _compute_sl_tp_from_row(row=row, side=side, entry_price=float(mark_price))

    # --- Policy-driven SL clamp + TP_R ---
    sl_min = float(policy.get("sl_pct_min", 0.25))
    sl_max = float(policy.get("sl_pct_max", 0.60))
    r_map = policy.get("r_multiple_by_class", {}) or {}
    tp_r = float((r_map.get(strength_class, {}) or {}).get("tp_r", 1.4))

    # calcola SL% rispetto al prezzo reference (mark_price)
    sl_pct = abs((float(mark_price) - float(sl_price)) / float(mark_price)) * 100.0

    # se SL strutturale troppo largo -> reject
    if sl_pct > sl_max:
        _log(f"SKIP {symbol} [{tf}] reason=sl_pct_gt_max sl_pct={sl_pct:.3f}% max={sl_max:.3f}%")
        return

    # clamp al minimo (se troppo stretto)
    if sl_pct < sl_min:
        # ricostruisci sl_price per rispettare sl_min
        adj = (sl_min / 100.0) * float(mark_price)
        sl_price = float(mark_price) - adj if side == "long" else float(mark_price) + adj
        sl_pct = sl_min

    # TP: usa R-multiple sulla distanza SL
    tp_dist = (sl_pct / 100.0) * float(mark_price) * tp_r
    tp_price = float(mark_price) + tp_dist if side == "long" else float(mark_price) - tp_dist

    # TP% (umana) = SL% * R
    tp_pct = (sl_pct * tp_r)
    if sl_price is None:
        _log(f"⚠️  Stop loss non disponibile per {symbol} [{tf}] -> skip")
        return

    sl_distance = abs(float(mark_price) - float(sl_price))

    risk_info = compute_position_size(
        symbol=symbol,
        equity=equity,
        sl_distance=sl_distance,
        clazz=sig_class,
        current_open_risk_pct=0.0,
        aggressiveness=(os.environ.get("CERBERO_AGGRESSIVENESS", "NORMAL") or "NORMAL").strip().upper(),
    )
    if getattr(risk_info, "skip", True):
        _log(f"⚠️  Risk engine ha rifiutato trade {symbol} {side} [{tf}]: {risk_info.reason}")
        return

    position_size = float(risk_info.size)
    risk_pct = float(risk_info.risk_pct)
    leverage = 1.0

    entry_price = float(mark_price)
    notional_value = float(entry_price) * float(position_size)

    raw_signal: Dict[str, Any] = {
        "symbol": symbol,
        "direction": ("LONG" if side == "long" else "SHORT"),
        "risk_notional_usd": float(notional_value),
        "leverage": int(leverage),
        "order_type": "MARKET",
        "dry_run": bool(dry_run),
        "sl_price": float(sl_price),
        "tp_price": (float(tp_price) if tp_price is not None else None),

        # broker-agnostic risk targets (percentuali "umane")
        "sl_pct_human": float(sl_pct),
        "sl_pct_dec": float(sl_pct) / 100.0,
        "tp_pct_human": float(tp_pct),
        "tp_pct_dec": float(tp_pct) / 100.0,
        "tp_r": float(tp_r),
        "strategy_origin": f"IPDA_{tf.upper()}",
        "probs": dict(probs),
        "p_blend": float(p_blend),
        "strength": float(strength),
        "tf": tf,
        "sig_class": sig_class,
        "risk_pct": risk_pct,
        "entry_price": entry_price,
        "position_size": position_size,
        "notional_value": notional_value,
        "mtf_status": mtf_status,
        "context_status": context_status,
        "econ_block_info": econ_block,
        "reasons": reasons,
        "ts_signal": now_utc.isoformat(),
        "user_id": (TENANT_EMAIL or None),
        "tenant_email": (TENANT_EMAIL or None),
    }

    result = send_trade_intent(raw_signal, dry_run=dry_run)

    if TELEGRAM_ALERTS_ENABLED and not dry_run:
        try:
            msg = (
                f"🚀 Cerbero trade\n"
                f"{symbol} {('LONG' if side=='long' else 'SHORT')} [{tf}]\n"
                f"class={sig_class}, p={p_blend:.3f}, strength={strength:.3f}\n"
                f"risk={risk_pct*100:.2f}% equity={equity:.2f}\n"
                f"notional={notional_value:.2f}, lev={leverage:.1f}x\n"
                f"reasons: {', '.join(reasons[:3])}"
            )
            send_telegram_alert(kind="trade", payload={"msg": msg, "raw_signal": raw_signal, "coordinator": result})
        except Exception as e:
            _log(f"⚠️  Errore nell'invio alert Telegram trade: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--equity", type=float, default=10_000.0)
    args = ap.parse_args()

    dry_run = bool(args.dry_run)
    equity = float(args.equity)

    policy = load_risk_policy()

    ipda = IPDALSTM(model_paths=IPDA_MODEL_PATHS)
    trade_tfs = [tf for tf in TIMEFRAMES if tf in ("5m", "15m")]

    _log(f"== Cerbero Coscienza v3 – live loop (IPDA+Radar) dry_run={dry_run}, equity={equity}, trade_tfs={trade_tfs} ==")

    for sym in SYMBOLS:
        if sym == "DOLLARIDXUSD":
            continue
        for tf in trade_tfs:
            try:
                radar_cfg = load_radar_config(sym, tf)
            except Exception as e:
                _log(f"⚠️  Errore nel caricare RadarConfig per {sym} [{tf}]: {e}")
                continue

            if not radar_cfg.models:
                _log(f"⚠️  Nessun radar disponibile per {sym} [{tf}] (salto).")
                continue

            _evaluate_symbol_tf(sym, tf, radar_cfg, ipda, equity, dry_run, policy)

    _log("== Fine passata live_loop_v3 ==")


if __name__ == "__main__":
    main()
