import argparse, io
import pandas as pd
from google.cloud import storage

BUCKET = "cerbero-data-processed-gns"
SRC_PREFIX = "features_resampled"        # dove ha scritto 03_resample.py
OUT_PREFIX = "features_merged"           # output di questo step

# helper: scarica parquet da GCS -> DataFrame indicizzato su datetime
def load_df(client, path):
    blob = client.bucket(BUCKET).blob(path)
    b = blob.download_as_bytes()
    df = pd.read_parquet(io.BytesIO(b))
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def merge_for_symbol(sym: str):
    client = storage.Client()

    # carica i 4 timeframe
    df5   = load_df(client, f"{SRC_PREFIX}/{sym}_5T.parquet")
    df15  = load_df(client, f"{SRC_PREFIX}/{sym}_15T.parquet")
    df4h  = load_df(client, f"{SRC_PREFIX}/{sym}_4H.parquet")
    df1d  = load_df(client, f"{SRC_PREFIX}/{sym}_1D.parquet")

    # rinomina colonne con suffissi per chiarezza
    def add_suf(df, suf):
        return df.rename(columns={c: f"{c}_{suf}" for c in df.columns})

    df5  = add_suf(df5,  "5m")
    df15 = add_suf(df15, "15m")
    df4h = add_suf(df4h, "4h")
    df1d = add_suf(df1d, "1d")

    # base = 5m; facciamo asof merge “backward” con tolleranze coerenti
    base = df5.reset_index().rename(columns={"index":"ts"})
    def asof_join(left, right, tol):
        r = right.reset_index().rename(columns={"index":"ts"}).sort_values("ts")
        m = pd.merge_asof(left.sort_values("ts"), r, on="ts",
                          direction="backward", tolerance=pd.Timedelta(tol))
        return m

    m = base
    m = asof_join(m, df15, tol="15min")
    m = asof_join(m, df4h, tol="4H")
    m = asof_join(m, df1d, tol="1D")
    m = m.set_index("ts").dropna(how="any")  # tieni solo righe complete

    # salva su GCS
    out_blob = client.bucket(BUCKET).blob(f"{OUT_PREFIX}/{sym}.parquet")
    out_blob.upload_from_string(m.to_parquet(), content_type="application/octet-stream")

    print(f"✅ {sym}: merged {len(m):,} righe → gs://{BUCKET}/{OUT_PREFIX}/{sym}.parquet")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    args = ap.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    for s in symbols:
        try:
            merge_for_symbol(s)
        except Exception as e:
            print(f"❌ Errore su {s}: {e}")
