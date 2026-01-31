# training/08_fit_blend_weights.py
import argparse, io, json, math
from typing import Dict, List
from dataclasses import dataclass
from google.cloud import storage

# === CONFIG ===
MODELS_BUCKET = "cerbero-models-gns"
BASE_PREFIX   = "radar_v1"
SYMBOLS = [
    "AUDJPY","AUDUSD","BTCUSD","CADJPY","DEUIDXEUR","DOLLARIDXUSD","ETHUSD",
    "EURJPY","EURUSD","GBPJPY","GBPUSD","JPNIDXJPY","LIGHTCMDUSD","USA500IDXUSD",
    "USATECHIDXUSD","USDCAD","USDCHF","USDJPY","XAGUSD","XAUUSD"
]
TFS   = ["5m","15m","4h","1d"]
KINDS = ["grafico","tecnico","contestuale"]

_client = storage.Client()

def _parse_gs(path: str):
    assert path.startswith("gs://")
    _, rest = path.split("gs://", 1)
    bucket, blob = rest.split("/", 1)
    return bucket, blob

def gcs_read_text(path: str) -> str:
    bkt, blob = _parse_gs(path)
    return _client.bucket(bkt).blob(blob).download_as_text()

def gcs_write_bytes(path: str, data: bytes):
    bkt, blob = _parse_gs(path)
    _client.bucket(bkt).blob(blob).upload_from_file(io.BytesIO(data), rewind=True)

def gcs_write_json(path: str, obj: dict):
    gcs_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2).encode())

def try_read_auc(sym: str, tf: str, kind: str):
    """
    Tenta di leggere l'AUC da:
      gs://cerbero-models-gns/radar_v1/{sym}/{tf}/{kind}.json
    Il JSON generato dal training contiene 'auc': float
    """
    path = f"gs://{MODELS_BUCKET}/{BASE_PREFIX}/{sym}/{tf}/{kind}.json"
    try:
        txt = gcs_read_text(path)
        data = json.loads(txt)
        # field name robust
        for k in ("auc","AUC","roc_auc","rocAuc"):
            if k in data and isinstance(data[k], (int,float)):
                return float(data[k])
        # fallback: se non trova, prova 'metrics': {'auc':...}
        if "metrics" in data and isinstance(data["metrics"], dict):
            for k in ("auc","AUC","roc_auc","rocAuc"):
                if k in data["metrics"]:
                    return float(data["metrics"][k])
        return None
    except Exception as e:
        return None

def safe_weight(a: float) -> float:
    """
    Converte un AUC in 'forza' positiva da pesare:
      - rimuove il baseline 0.5
      - tronca a >= 0.001 per non azzerare completamente nessun canale
    """
    if a is None: 
        return 0.001
    return max(a - 0.5, 0.001)

def normalize(ws: Dict[str, float]) -> Dict[str, float]:
    s = sum(ws.values()) or 1.0
    return {k: (v / s) for k, v in ws.items()}

def main(save: bool):
    rows = []  # per CSV report
    tf_fallback_accum = {tf: {k: 0.0 for k in KINDS} for tf in TFS}
    tf_fallback_count = {tf: 0 for tf in TFS}

    for sym in SYMBOLS:
        for tf in TFS:
            aucs = {k: try_read_auc(sym, tf, k) for k in KINDS}
            raw  = {k: safe_weight(aucs[k]) for k in KINDS}
            w    = normalize(raw)

            # salva pesi per symbol/tf
            weights_obj = {
                "symbol": sym,
                "tf": tf,
                "weights": w,
                "aucs": aucs,
                "note": "pesi calcolati proporzionali ad (auc-0.5), min 0.001; normalizzati a somma=1"
            }
            if save:
                out_path = f"gs://{MODELS_BUCKET}/{BASE_PREFIX}/{sym}/{tf}/weights.json"
                gcs_write_json(out_path, weights_obj)

            rows.append((sym, tf, w["grafico"], w["tecnico"], w["contestuale"]))

            # accumulo per fallback tf
            for k in KINDS:
                tf_fallback_accum[tf][k] += w[k]
            tf_fallback_count[tf] += 1

    # Fallback medi per TF
    tf_fallback = {}
    for tf in TFS:
        n = max(tf_fallback_count[tf], 1)
        avg = {k: tf_fallback_accum[tf][k] / n for k in KINDS}
        tf_fallback[tf] = normalize(avg)

    if save:
        # report CSV
        csv_lines = ["symbol,tf,w_grafico,w_tecnico,w_contestuale"]
        for (sym, tf, wg, wt, wc) in rows:
            csv_lines.append(f"{sym},{tf},{wg},{wt},{wc}")
        csv_data = ("\n".join(csv_lines) + "\n").encode()
        gcs_write_bytes(f"gs://{MODELS_BUCKET}/{BASE_PREFIX}/reports/blend_weights.csv", csv_data)

        # fallback per TF
        gcs_write_json(f"gs://{MODELS_BUCKET}/{BASE_PREFIX}/reports/blend_fallback_by_tf.json", tf_fallback)

    # stampa preview
    print("— Esempio pesi (prime 10 righe) —")
    for r in rows[:10]:
        print(f"{r[0]} {r[1]}  w_grafico={r[2]:.3f}  w_tecnico={r[3]:.3f}  w_contestuale={r[4]:.3f}")
    print("\n— Fallback medi per TF —")
    for tf in TFS:
        w = tf_fallback[tf]
        print(f"{tf} → {w}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="Scrive pesi e report su GCS")
    args = ap.parse_args()
    main(save=args.save)
