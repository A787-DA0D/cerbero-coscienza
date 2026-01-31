import argparse
import pandas as pd
from google.cloud import storage
import io

RAW_BUCKET = "cerbero-data-processed-gns"
OUT_PREFIX = "features_resampled"

def resample_features(symbol: str, rule: str):
    client = storage.Client()

    src_blob = client.bucket(RAW_BUCKET).blob(f"features_parquet/{symbol}.parquet")
    df = pd.read_parquet(io.BytesIO(src_blob.download_as_bytes()))

    # Conversione a datetime
    df.index = pd.to_datetime(df.index)

    # Resampling (5M, 15M, 4H, 1D)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_resampled = df.resample(rule).agg(agg).dropna()

    # Upload
    out_blob = client.bucket(RAW_BUCKET).blob(f"{OUT_PREFIX}/{symbol}_{rule}.parquet")
    out_blob.upload_from_string(df_resampled.to_parquet(), content_type="application/octet-stream")

    print(f"✅ {symbol} ({rule}) salvato ({len(df_resampled)} righe)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", required=True)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    rules = ["5T", "15T", "4H", "1D"]

    for sym in symbols:
        for rule in rules:
            try:
                resample_features(sym, rule)
            except Exception as e:
                print(f"❌ Errore su {sym} [{rule}]: {e}")
