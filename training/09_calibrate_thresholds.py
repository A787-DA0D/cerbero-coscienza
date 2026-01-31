# training/09_calibrate_thresholds.py
import argparse, io, json
from typing import Dict
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

def try_read_json(path: str):
    try:
        return json.loads(gcs_read_text(path))
    except Exception:
        return None

def clamp(x, lo, hi): 
    return max(lo, min(hi, x))

def base_threshold_for_tf(tf: str) -> float:
    # Base prudente; verrà leggermente adattata dalla "forza blend"
    return {"5m":0.60, "15m":0.60, "4h":0.58, "1d":0.57}.get(tf, 0.60)

def main(save: bool):
    # carica fallback pesi per TF (se presente), altrimenti default = uniformi
    fb = try_read_json(f"gs://{MODELS_BUCKET}/{BASE_PREFIX}/reports/blend_fallback_by_tf.json") or {}
    fb = {tf: fb.get(tf, {"grafico":1/3,"tecnico":1/3,"contestuale":1/3}) for tf in TFS}

    csv_lines = ["symbol,tf,thr_blend,thr_grafico,thr_tecnico,thr_contestuale,strength"]
    for sym in SYMBOLS:
        for tf in TFS:
            # leggi AUC per i 3 canali
            aucs = {}
            for k in KINDS:
                p = f"gs://{MODELS_BUCKET}/{BASE_PREFIX}/{sym}/{tf}/{k}.json"
                j = try_read_json(p) or {}
                val = None
                for kk in ("auc","AUC","roc_auc","rocAuc"):
                    if kk in j and isinstance(j[kk], (int,float)):
                        val = float(j[kk]); break
                if val is None and isinstance(j.get("metrics"), dict):
                    for kk in ("auc","AUC","roc_auc","rocAuc"):
                        if kk in j["metrics"]:
                            val = float(j["metrics"][kk]); break
                aucs[k] = val if isinstance(val,(int,float)) else 0.5

            # leggi pesi specifici; se non esiste -> fallback tf
            wpath = f"gs://{MODELS_BUCKET}/{BASE_PREFIX}/{sym}/{tf}/weights.json"
            wj = try_read_json(wpath)
            if wj and isinstance(wj.get("weights"), dict):
                w = {k: float(wj["weights"].get(k, 1/3)) for k in KINDS}
            else:
                w = fb[tf]

            # forza blend = somma_k w_k * max(auc_k-0.5, 0)
            strength = sum( w[k] * max(aucs[k]-0.5, 0.0) for k in KINDS )

            # threshold blend = base - piccolo bonus in funzione della strength
            base_thr = base_threshold_for_tf(tf)
            thr_blend = clamp(base_thr - min(0.05, strength*0.20), 0.52, 0.65)

            # thresholds per singolo canale: base + piccolo aggiustamento vs (auc-0.5)
            thr_k = {}
            for k in KINDS:
                adj = max(aucs[k]-0.5, 0.0) * 0.15  # 0 → 0, 0.05 → 0.0075
                thr_k[k] = clamp(base_thr - adj, 0.52, 0.65)

            if save:
                out = {
                    "symbol": sym,
                    "tf": tf,
                    "thresholds": {
                        "blend": thr_blend,
                        "grafico": thr_k["grafico"],
                        "tecnico": thr_k["tecnico"],
                        "contestuale": thr_k["contestuale"],
                    },
                    "aucs": aucs,
                    "weights_used": w,
                    "strength": strength,
                    "note": "Heuristic thresholds; base per TF adattata con forza blend (AUC) e pesi."
                }
                gcs_write_json(f"gs://{MODELS_BUCKET}/{BASE_PREFIX}/{sym}/{tf}/thresholds.json", out)

            csv_lines.append(f"{sym},{tf},{thr_blend},{thr_k['grafico']},{thr_k['tecnico']},{thr_k['contestuale']},{strength}")

    if save:
        gcs_write_bytes(f"gs://{MODELS_BUCKET}/{BASE_PREFIX}/reports/thresholds.csv",
                        ("\n".join(csv_lines) + "\n").encode())
    print("✅ Soglie generate (anteprima):")
    for ln in csv_lines[:8]:
        print(ln)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="Scrive thresholds su GCS e CSV di report")
    args = ap.parse_args()
    main(save=args.save)
