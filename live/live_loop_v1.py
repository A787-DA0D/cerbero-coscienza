# live/live_loop_v1.py
import os, time, json, io, warnings, joblib
import numpy as np
import pandas as pd
import requests
from google.cloud import storage

# === Config ===
RAW_BUCKET   = "cerbero-data-processed-gns"     # features_{tf}/{SYM}.parquet
MODEL_BUCKET = "cerbero-models-gns"             # radar_v1/{SYM}/{tf}/...
MODEL_PREFIX = "radar_v1"

# env
COORDINATOR_URL = os.getenv("COORDINATOR_URL", "").rstrip("/")
COORDINATOR_KEY = os.getenv("COORDINATOR_API_KEY", "")
DRY_RUN         = os.getenv("DRY_RUN", "1") == "1"  # default: non invia

POLL_SEC   = float(os.getenv("LIVE_POLL_SEC", "30"))  # ogni 30s
LOOKBACK_N = int(os.getenv("LIVE_LOOKBACK_N", "500")) # righe features da usare

warnings.filterwarnings("ignore")

# --- utils GCS ---
_gcs = storage.Client()

def gcs_blob_text(bucket, path):
    b = _gcs.bucket(bucket).blob(path)
    return b.download_as_text()

def gcs_blob_bytes(bucket, path):
    b = _gcs.bucket(bucket).blob(path)
    return b.download_as_bytes()

def load_parquet_gcs(bucket, path):
    data = gcs_blob_bytes(bucket, path)
    return pd.read_parquet(io.BytesIO(data))

# --- loader modelli + pesi/soglie ---
def load_models_weights_thr(symbol, tf):
    base = f"{MODEL_PREFIX}/{symbol}/{tf}"
    models = {}
    for kind in ("grafico","tecnico","contestuale"):
        models[kind] = joblib.load(io.BytesIO(gcs_blob_bytes(MODEL_BUCKET, f"{base}/{kind}.joblib")))
    weights     = json.loads(gcs_blob_text(MODEL_BUCKET, f"{base}/weights.json"))["weights"]
    thresholds  = json.loads(gcs_blob_text(MODEL_BUCKET, f"{base}/thresholds.json"))["thresholds"]
    return models, weights, thresholds

def latest_features(symbol, tf):
    feat_path = f"features_{tf}/{symbol}.parquet"
    df = load_parquet_gcs(RAW_BUCKET, feat_path)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~df.index.isna()].sort_index()
    if len(df) > LOOKBACK_N:
        df = df.iloc[-LOOKBACK_N:]
    # X = df con solo colonne numeriche (esclude target/segnali passati se presenti)
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
    return df, num

def blend_probs(p, w):
    # p = dict(kind -> prob), w = dict(kind -> weight)  (sommatoria pesi = 1)
    return float(sum(p[k]*w.get(k,0.0) for k in ("grafico","tecnico","contestuale")))

def decide_action(prob_blend, thr_blend):
    if prob_blend >= thr_blend:
        return "BUY"      # semplificazione: solo lato long per v1
    return "HOLD"

def post_signal(payload):
    if not COORDINATOR_URL:
        print("⚠️  COORDINATOR_URL non impostata: skip POST")
        return False, "no-url"
    url = f"{COORDINATOR_URL}/api/signal"
    headers = {"Content-Type":"application/json"}
    if COORDINATOR_KEY:
        headers["Authorization"] = f"Bearer {COORDINATOR_KEY}"
    if DRY_RUN:
        print("🧪 DRY_RUN ON — non invio. Payload →", json.dumps(payload))
        return True, "dry-run"
    r = requests.post(url, headers=headers, json=payload, timeout=8)
    ok = 200 <= r.status_code < 300
    return ok, f"status={r.status_code} body={r.text[:200]}"

def run_once(symbol, tf):
    # 1) carica modelli/pesi/soglie
    models, weights, thr = load_models_weights_thr(symbol, tf)
    # 2) features ultime N
    df_raw, X = latest_features(symbol, tf)
    # 3) proba su ultima riga
    last_idx = X.index[-1]
    x_last = X.iloc[[-1]]

    probs = {}
    for kind, pipe in models.items():
        # pipe è sklearn Pipeline con clf predict_proba
        try:
            pr = pipe.predict_proba(x_last)[:,1].item()
        except Exception:
            # fallback: decision_function -> sigmoid
            dfc = pipe.decision_function(x_last).item()
            pr = 1/(1+np.exp(-dfc))
        probs[kind] = float(pr)

    p_blend = blend_probs(probs, weights)
    action = decide_action(p_blend, thr["blend"])

    payload = {
        "symbol": symbol,
        "tf": tf,
        "ts": last_idx.isoformat(),
        "probs": {"grafico": probs["grafico"], "tecnico": probs["tecnico"], "contestuale": probs["contestuale"], "blend": p_blend},
        "weights": weights,
        "thresholds": thr,
        "action": action,
        "version": "radar_v1"
    }

    ok, info = post_signal(payload)
    print(f"[{symbol} {tf} @ {last_idx}] blend={p_blend:.3f} thr={thr['blend']:.3f} → {action}  ({'POST OK' if ok else 'POST FAIL'}: {info})")

def live_loop(symbols, tfs):
    print(f"▶️  live_loop avviato | symbols={symbols} tfs={tfs} poll={POLL_SEC}s dry_run={DRY_RUN}")
    while True:
        t0 = time.time()
        for s in symbols:
            for tf in tfs:
                try:
                    run_once(s, tf)
                except Exception as e:
                    print(f"⚠️  errore {s} {tf}: {e}")
        dt = time.time() - t0
        sleep_s = max(1.0, POLL_SEC - dt)
        time.sleep(sleep_s)

if __name__ == "__main__":
    # simboli/timeframe da env (default: EURUSD 5m)
    symbols = [s.strip() for s in os.getenv("LIVE_SYMBOLS","EURUSD").split(",") if s.strip()]
    tfs     = [t.strip() for t in os.getenv("LIVE_TFS","5m").split(",") if t.strip()]
    live_loop(symbols, tfs)
