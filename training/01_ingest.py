# 01_ingest.py — solo ASK → Parquet (1-min OHLCV) su GCS
# Usage:
#   python training/01_ingest.py --symbols "AUDJPY EURUSD"      # elenco esplicito
#   python training/01_ingest.py                                 # auto-scan (tutti i simboli trovati)

import io
import re
import sys
import argparse
from typing import List

import pandas as pd
from google.cloud import storage

RAW_BUCKET = "cerbero-data-raw-gns"
OUT_BUCKET = "cerbero-data-processed-gns"
OUT_PREFIX = "ask_parquet"  # gs://cerbero-data-processed-gns/ask_parquet/<SYM>.parquet

client = storage.Client()
b_raw = client.bucket(RAW_BUCKET)
b_out = client.bucket(OUT_BUCKET)

def read_tolerant_csv(data: bytes) -> pd.DataFrame:
    """
    Parser robusto per file tipo:
    2020.01.02 23:00:00,75,848,75,849,75,84,75,841,26,9   -> ricompone coppie in float
    Header: Time (EET),Open,High,Low,Close,Volume \r
    Fallback: gestisce anche separatori a spazi/tab con virgole come decimali.
    """
    s = io.StringIO(data.decode("utf-8", "ignore"))
    rows = []
    for raw in s:
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("time"):
            continue

        if "," in line:
            parts = [p.strip() for p in line.split(",")]
            # ci aspettiamo: ts + 10 token (5 numeri come "int,dec")
            if len(parts) >= 11:
                ts_str = parts[0]
                rest = parts[1:]
                pair10 = rest[:10]
                nums = []
                ok = True
                for i in range(0, 10, 2):
                    a, b = pair10[i], pair10[i + 1]
                    if not re.fullmatch(r"^-?\d+$", a) or not re.fullmatch(r"^\d+$", b):
                        ok = False
                        break
                    nums.append(float(f"{a}.{b}"))
                if ok and len(nums) == 5:
                    ts = pd.to_datetime(ts_str, errors="coerce", format="%Y.%m.%d %H:%M:%S")
                    if not pd.isna(ts):
                        o, h, l, c, v = nums
                        rows.append([ts, o, h, l, c, v])
                    continue  # riga gestita

        # fallback: spazi/tab e/o virgole come decimali
        line2 = line.replace(",", ".")
        toks = line2.split()
        if len(toks) < 6:
            continue

        if re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", toks[0]) and len(toks) >= 7 and re.fullmatch(r"\d{2}:\d{2}:\d{2}", toks[1]):
            ts_str = toks[0] + " " + toks[1]
            vals = toks[2:]
        else:
            ts_str = toks[0]
            vals = toks[1:]

        nums = []
        for t in vals:
            try:
                nums.append(float(t))
            except:
                pass
            if len(nums) == 5:
                break
        if len(nums) == 5:
            ts = pd.to_datetime(ts_str, errors="coerce", format="%Y.%m.%d %H:%M:%S")
            if not pd.isna(ts):
                o, h, l, c, v = nums
                rows.append([ts, o, h, l, c, v])

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    if not df.empty:
        df = df.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        df = df.set_index("ts")
    return df

def to_min_grid(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~idx.isna()].copy()
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = pd.Index(idx).floor("min")
    df = (
        df.groupby(level=0)
          .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
          .dropna(subset=["open", "high", "low", "close"])
          .sort_index()
    )
    df.index.name = "ts"
    return df

def list_ask_blobs_for_symbol(sym: str) -> List[str]:
    # oggetti tipo: "<SYM>_1 Min_Ask_YYYY.MM.DD_YYYY.MM.DD.csv"
    blobs = client.list_blobs(RAW_BUCKET, prefix=f"{sym}_1 Min_Ask_")
    return [b.name for b in blobs if b.name.endswith(".csv")]

def ingest_symbol(sym: str) -> bool:
    print(f"\n=== {sym} ===")
    blob_names = list_ask_blobs_for_symbol(sym)
    if not blob_names:
        raise RuntimeError(f"Nessun file Ask trovato per {sym}")

    frames = []
    for name in sorted(blob_names):
        blob = b_raw.blob(name)
        data = blob.download_as_bytes()
        df = read_tolerant_csv(data)
        if df.empty:
            print(f"⚠️  File vuoto/sporco: {name}")
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError(f"Nessun CSV valido per {sym} (Ask)")

    dfa = pd.concat(frames, axis=0).sort_index()
    dfa = dfa[~dfa.index.duplicated(keep="last")]
    dfa = to_min_grid(dfa)
    if dfa.empty:
        raise RuntimeError(f"Dati vuoti dopo griglia 1-min per {sym}")

    # salva Parquet
    out_path = f"{OUT_PREFIX}/{sym}.parquet"
    out_blob = b_out.blob(out_path)
    buf = io.BytesIO()
    dfa.to_parquet(buf, index=True)
    out_blob.upload_from_string(buf.getvalue(), content_type="application/octet-stream")
    print(f"✅ Salvato: gs://{OUT_BUCKET}/{out_path}  (righe={len(dfa)})")
    return True

def auto_detect_symbols() -> List[str]:
    # prendi tutti i blob e estrai il prefisso prima di "_1 Min_Ask_"
    syms = set()
    for blob in client.list_blobs(RAW_BUCKET):
        name = blob.name
        if "_1 Min_Ask_" in name and name.endswith(".csv"):
            sym = name.split("_1 Min_Ask_")[0]
            if sym:
                syms.add(sym.upper())
    return sorted(syms)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="", help='Lista simboli "AUDJPY EURUSD" oppure vuoto per auto-scan')
    args = ap.parse_args()

    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.replace(",", " ").split()]
    else:
        symbols = auto_detect_symbols()

    print("Simboli target:", symbols)
    failed = []
    for s in symbols:
        try:
            ingest_symbol(s)
        except Exception as e:
            print(f"❌ Errore su {s}: {e}")
            failed.append(s)

    if failed:
        print("\n⚠️ Terminato con errori su:", failed)
        sys.exit(2)
    print("\n✅ Ingest completato senza errori.")
