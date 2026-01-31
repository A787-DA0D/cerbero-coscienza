# training/06_train_radar.py
import argparse, json, io
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from google.cloud import storage

RAW_BUCKET = "cerbero-data-processed-gns"
MODEL_BUCKET = "cerbero-models-gns"
MODEL_PREFIX = "radar_v1"   # lasciamo v1 come cartella (cambiano solo le label)
TFS = ["5m","15m","4h","1d"]

# === import Target v2 ===
from targets.target_v2 import make_labels_v2

_client = storage.Client()

def read_parquet_gcs(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def write_bytes_to_gcs(bucket: str, blob_name: str, data: bytes, content_type="application/octet-stream"):
    b = _client.bucket(bucket).blob(blob_name)
    b.upload_from_file(io.BytesIO(data), size=len(data), content_type=content_type)

def save_json_gcs(bucket: str, blob_name: str, obj: dict):
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    write_bytes_to_gcs(bucket, blob_name, data, content_type="application/json")

def feat_path(sym: str, tf: str) -> str:
    return f"gs://{RAW_BUCKET}/features_{tf}/{sym}.parquet"

def patterns_path(sym: str, tf: str) -> str:
    # opzionale: se hai patterns_* per TF
    return f"gs://{RAW_BUCKET}/patterns_{tf}/{sym}.parquet"

def load_features_with_patterns(sym: str, tf: str) -> pd.DataFrame:
    # Base features
    X = read_parquet_gcs(feat_path(sym, tf))
    # Merge (se esistono i patterns, ok; altrimenti solo features)
    try:
        P = read_parquet_gcs(patterns_path(sym, tf))
        # allinea su index
        X = X.join(P, how="left")
    except Exception:
        pass
    # pulizia index
    X.index = pd.to_datetime(X.index, errors="coerce")
    X = X.loc[~X.index.isna()].sort_index()
    return X

def build_pipeline(random_state: int = 42) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                solver="saga",
                penalty="l2",
                max_iter=10000,
                n_jobs=-1,
                class_weight="balanced",
                random_state=random_state
            ))
        ]
    )
    return pipe

def time_series_cv_auc(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Tuple[float, np.ndarray]:
    """
    CV temporale (expanding window). Ritorna media AUC e pred out-of-fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.full(len(X), np.nan, dtype=float)

    Xv = X.values
    yv = y.values

    for tr_idx, te_idx in tscv.split(Xv):
        X_tr, X_te = Xv[tr_idx], Xv[te_idx]
        y_tr, y_te = yv[tr_idx], yv[te_idx]
        mdl = pipe
        mdl.fit(X_tr, y_tr)
        p = mdl.predict_proba(X_te)[:,1]
        oof[te_idx] = p

    mask = ~np.isnan(oof)
    auc = roc_auc_score(yv[mask], oof[mask]) if mask.any() else np.nan
    return auc, oof

def train_one_kind(sym: str, tf: str, kind: str, random_state: int = 42) -> Tuple[float, dict]:
    """
    kind ∈ {"grafico","tecnico","contestuale"}
    Seleziona sottoinsiemi di colonne per kind; poi addestra con y_v2.
    """
    X_all = load_features_with_patterns(sym, tf)

    # --- selezione colonne by kind ---
    cols = []
    if kind == "grafico":
        # cose di price action/patterns/bollinger/stoch ecc.
        cols = [c for c in X_all.columns if any(k in c.lower() for k in [
            "pattern", "bb_", "stoch", "inside", "flag", "pennant", "ema",
            "momentum", "squeeze", "range", "breakout", "pullback",
            "strong_bull", "strong_bear", "candle_body_ratio"
        ])] + [c for c in X_all.columns if c.lower() in ["open","high","low","close","volume"]]
    elif kind == "tecnico":
        cols = [c for c in X_all.columns if any(k in c.lower() for k in [
            "rsi", "macd", "ema", "adx", "atr", "roc", "williams", "cci", "dmi", "div"
        ])]
    elif kind == "contestuale":
        cols = [c for c in X_all.columns if any(k in c.lower() for k in [
            "session", "is_", "atr_z", "bb_width_z", "trend", "choppy", "hour", "dow"
        ])]
    else:
        raise ValueError(f"kind sconosciuto: {kind}")

    # fallback: se vuoto, usa tutte
    if not cols:
        cols = list(X_all.columns)

    X = X_all[cols].copy()

    # --- LABEL v2 denoise (ATR) ---
    y = make_labels_v2(X_all, tf=tf)  # usa close/ATR del dataset completo

    # allinea X e y e droppa neutri
    df_xy = X.join(y, how="inner")
    df_xy = df_xy.loc[~df_xy["y_v2"].isna()].copy()

    if len(df_xy) < 500:
        raise RuntimeError(f"poche righe dopo filtro neutri: {sym} [{tf}] {kind} → {len(df_xy)}")

    y_fin = df_xy["y_v2"].astype(int)
    X_fin = df_xy.drop(columns=["y_v2"])

    # === NUOVO: pulizia NaN / inf sulle feature ===
    X_fin = X_fin.replace([np.inf, -np.inf], np.nan)
    mask_good = ~X_fin.isna().any(axis=1)

    X_fin = X_fin.loc[mask_good]
    y_fin = y_fin.loc[mask_good]

    if len(X_fin) < 500:
        raise RuntimeError(f"poche righe dopo pulizia NaN/inf: {sym} [{tf}] {kind} → {len(X_fin)}")

    pipe = build_pipeline()
    auc, oof = time_series_cv_auc(pipe, X_fin, y_fin, n_splits=5)

    # fit finale su tutto e serializza
    pipe.fit(X_fin.values, y_fin.values)

    meta = {
        "symbol": sym,
        "tf": tf,
        "kind": kind,
        "auc": float(auc),
        "rows": int(len(X_fin)),
        "cols": list(X_fin.columns),
    }
    return auc, {"model": pipe, "meta": meta}

def save_model(sym: str, tf: str, kind: str, model_obj: dict):
    # joblib compatibile via cloudpickle in bytes
    import cloudpickle
    model_blob = f"{MODEL_PREFIX}/{sym}/{tf}/{kind}.joblib"
    json_blob  = f"{MODEL_PREFIX}/{sym}/{tf}/{kind}.json"

    by = cloudpickle.dumps(model_obj["model"])
    write_bytes_to_gcs(MODEL_BUCKET, model_blob, by, content_type="application/octet-stream")
    save_json_gcs(MODEL_BUCKET, json_blob, model_obj["meta"])

def run_for_symbol(sym: str, tfs: List[str]):
    kinds = ["grafico","tecnico","contestuale"]
    for tf in tfs:
        for kind in kinds:
            try:
                auc, obj = train_one_kind(sym, tf, kind)
                save_model(sym, tf, kind, obj)
                print(f"  ✅ {sym} [{tf}] {kind}: AUC={auc:.3f}  → gs://{MODEL_BUCKET}/{MODEL_PREFIX}/{sym}/{tf}/{kind}.joblib")
            except Exception as e:
                print(f"  ⚠️  {sym} [{tf}] {kind} errore: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help='Es: "EURUSD" oppure "EURUSD,USDJPY"')
    ap.add_argument("--tfs", default="5m,15m,4h,1d")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]

    print(f"Simboli: {symbols}")
    print(f"Timeframes: {tfs}")
    for s in symbols:
        print(f"\n=== {s} ===")
        run_for_symbol(s, tfs)
