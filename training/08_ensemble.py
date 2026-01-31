# training/08_ensemble.py
import os, io, json, argparse
import pandas as pd
from urllib.parse import urlparse

# Config
REPORT_CSV = "gs://cerbero-models-gns/radar_v1/reports/validation_summary.csv"
OUT_DIR    = "gs://cerbero-models-gns/radar_v1/ensemble/weights"
KINDS      = ["grafico","tecnico","contestuale"]
TFS        = ["5m","15m","4h","1d"]

def _split_gs(uri: str):
    assert uri.startswith("gs://")
    path = uri[5:]
    bucket, key = path.split("/", 1)
    return bucket, key

def _gsutil_read_csv(gs_uri: str) -> pd.DataFrame:
    # usa pandas che sa leggere GCS con gcsfs già presente nell'env
    return pd.read_csv(gs_uri)

def _gsutil_write_json(obj: dict, gs_uri: str):
    bucket, key = _split_gs(gs_uri)
    # scriviamo usando gsutil via pipe (robusto e semplice)
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        tmp = f.name
    subprocess.check_call(["gsutil", "cp", tmp, gs_uri])
    os.remove(tmp)

def _gsutil_write_csv(df: pd.DataFrame, gs_uri: str):
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        df.to_csv(f.name, index=False)
        tmp = f.name
    subprocess.check_call(["gsutil", "cp", tmp, gs_uri])
    os.remove(tmp)

def auc_to_weight(auc: float) -> float:
    # sposta e tronca: solo extra rispetto a random
    # se auc<=0.5, peso 0; altrimenti (auc-0.5)
    return max(auc - 0.5, 0.0)

def compute_weights(df: pd.DataFrame):
    """
    df: columns = [symbol, tf, kind, auc]
    Ritorna:
      - per-symbol/TF weights JSON salvati
      - fallback globali per TF
      - un csv riepilogo con i pesi effettivi
    """
    out_rows = []
    # Fallback globali (media per TF tra simboli)
    global_w = {}
    for tf in TFS:
        sub = df[df["tf"]==tf]
        w = {k:auc_to_weight(sub[sub["kind"]==k]["auc"].mean()) for k in KINDS}
        s = sum(w.values())
        if s==0:
            w = {k:1/3 for k in KINDS}
        else:
            w = {k: v/s for k,v in w.items()}
        global_w[tf] = w

    # Pesi per simbolo/TF
    for sym in sorted(df["symbol"].unique()):
        for tf in TFS:
            sub = df[(df["symbol"]==sym) & (df["tf"]==tf)]
            if sub.empty:
                # niente modelli per quel TF → salta
                continue
            w_raw = {k: auc_to_weight(sub[sub["kind"]==k]["auc"].values[0]) for k in KINDS}
            s = sum(w_raw.values())
            if s==0:
                # usa fallback globale
                w = global_w[tf]
            else:
                w = {k: v/s for k,v in w_raw.items()}
            out_rows.append({"symbol":sym,"tf":tf, **{f"w_{k}":w[k] for k in KINDS}})
            # salva JSON per simbolo/tf
            dst = f"{OUT_DIR}/{sym}/{tf}.json"
            _gsutil_write_json({"symbol":sym,"tf":tf,"weights":w}, dst)

    # salva anche i fallback globali
    _gsutil_write_json({"weights_per_tf":global_w}, f"{OUT_DIR}/global_fallback.json")
    # e un CSV riassuntivo
    summary = pd.DataFrame(out_rows).sort_values(["symbol","tf"])
    _gsutil_write_csv(summary, f"{OUT_DIR}/weights_summary.csv")
    return summary, global_w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=REPORT_CSV, help="Path al validation_summary.csv su GCS")
    args = ap.parse_args()

    df = _gsutil_read_csv(args.report)
    # sanity
    need = {"symbol","tf","kind","auc"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise SystemExit(f"validation_summary.csv mancano colonne: {missing}")

    summary, global_w = compute_weights(df)
    print("— Weights per symbol/tf (prime righe) —")
    print(summary.head(12).to_string(index=False))
    print("\n— Global fallback per TF —")
    for tf in TFS:
        print(tf, "→", global_w[tf])

if __name__ == "__main__":
    main()
