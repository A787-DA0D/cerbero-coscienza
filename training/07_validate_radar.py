# training/07_validate_radar.py
import os, json, argparse
import pandas as pd
from google.cloud import storage

from live.config_v3 import SYMBOLS as SYMBOLS_V3  # usa solo i simboli ufficiali v3

BUCKET = "cerbero-models-gns"
PREFIX = "radar_v1"
REPORT_DIR_LOCAL = "reports"
REPORT_PREFIX_GCS = f"{PREFIX}/reports"  # gs://cerbero-models-gns/radar_v1/reports/...


def _get_client():
    return storage.Client()


def _iter_metric_json():
    """
    Scansiona gs://cerbero-models-gns/radar_v1/*/*/*.json (escludendo /reports/)
    e restituisce record {symbol, tf, kind, auc}.
    """
    client = _get_client()
    blobs = client.list_blobs(BUCKET, prefix=f"{PREFIX}/")
    for b in blobs:
        name = b.name  # es: radar_v1/EURUSD/5m/tecnico.json
        if not name.endswith(".json"):
            continue
        if f"{PREFIX}/reports/" in name:
            continue
        parts = name.split("/")
        # atteso: [radar_v1, SYMBOL, TF, FILENAME.json]
        if len(parts) != 4:
            continue
        _, symbol, tf, filename = parts
        kind = filename.replace(".json", "")

        # filtra subito simboli fuori dall'universo v3 (indici, vecchi test, ecc.)
        if symbol not in SYMBOLS_V3:
            continue

        try:
            txt = b.download_as_text()
            data = json.loads(txt)
            # prova varie chiavi possibili
            auc = None
            for k in ("auc", "AUC", "roc_auc"):
                if k in data:
                    auc = data[k]
                    break
            if auc is None:
                continue
            yield {"symbol": symbol, "tf": tf, "kind": kind, "auc": float(auc)}
        except Exception as e:
            print(f"⚠️  errore leggendo {name}: {e}")


def main(save: bool):
    rows = list(_iter_metric_json())
    if not rows:
        print("❌ Nessuna metrica trovata nei JSON per i simboli v3. Verifica che i .json siano nel bucket.")
        print("   SYMBOLS v3:", SYMBOLS_V3)
        return

    df = pd.DataFrame(rows).sort_values(["symbol", "tf", "kind"])
    print("\n— Preview (solo simboli v3) —")
    print(df.head(20).to_string(index=False))

    print("\n— Medie AUC per TF/Kind (solo universo v3) —")
    print(df.groupby(["tf", "kind"])["auc"].mean().round(3))

    if save:
        os.makedirs(REPORT_DIR_LOCAL, exist_ok=True)
        csv_path = os.path.join(REPORT_DIR_LOCAL, "validation_summary.csv")
        json_path = os.path.join(REPORT_DIR_LOCAL, "validation_summary.json")
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)

        client = _get_client()
        bucket = client.bucket(BUCKET)
        for local in (csv_path, json_path):
            blob = bucket.blob(f"{REPORT_PREFIX_GCS}/{os.path.basename(local)}")
            blob.upload_from_filename(local)
        print(f"\n✅ Report salvati in gs://{BUCKET}/{REPORT_PREFIX_GCS}/")
        print("   - validation_summary.csv")
        print("   - validation_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="salva CSV/JSON anche su GCS")
    args = ap.parse_args()
    main(save=args.save)
