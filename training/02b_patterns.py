# training/02b_patterns.py
import argparse
import io
from typing import List
import numpy as np
import pandas as pd
from google.cloud import storage

# === CONFIG ===
RAW_BUCKET = "cerbero-data-processed-gns"
OUT_BUCKET = "cerbero-data-processed-gns"

# Timeframes e mapping (nome umano -> regola pandas e suffisso file)
TFS_ALL = ["5m", "15m", "4h", "1d"]
RULE = {"5m": "5T", "15m": "15T", "4h": "4H", "1d": "1D"}  # per i file resampled

# === GCS helpers (senza gcsfs) ===
_client = storage.Client()

def _parse_gs_path(path: str):
    assert path.startswith("gs://"), f"Path non GCS: {path}"
    _, rest = path.split("gs://", 1)
    bucket, blob = rest.split("/", 1)
    return bucket, blob

def _gcs_download_bytes(bucket: str, blob_name: str) -> bytes:
    b = _client.bucket(bucket).blob(blob_name)
    return b.download_as_bytes()  # solleva se non esiste

def _gcs_upload_bytes(bucket: str, blob_name: str, data: bytes, content_type="application/octet-stream"):
    b = _client.bucket(bucket).blob(blob_name)
    b.upload_from_file(io.BytesIO(data), size=len(data), content_type=content_type)

def read_parquet_gcs(path: str) -> pd.DataFrame:
    bucket, blob = _parse_gs_path(path)
    raw = _gcs_download_bytes(bucket, blob)
    return pd.read_parquet(io.BytesIO(raw), engine="pyarrow")

def write_parquet_gcs(df: pd.DataFrame, path: str):
    bucket, blob = _parse_gs_path(path)
    bio = io.BytesIO()
    df.to_parquet(bio, engine="pyarrow", index=True)
    _gcs_upload_bytes(bucket, blob, bio.getvalue(), content_type="application/octet-stream")

# === Risoluzione path resampled (allinea al layout REALE del bucket) ===
def _resolve_resampled_path(sym: str, tf: str) -> List[str]:
    """
    Cerca i resample di (sym, tf) nei layout seguenti (in ordine):
    1) gs://cerbero-data-processed-gns/features_resampled/SYMBOL_RULE.parquet   <-- quello che hai
    2) gs://cerbero-data-processed-gns/resampled_tf/SYMBOL.parquet              <-- fallback vecchio
    3) gs://cerbero-data-processed-gns/resampled_RULE/SYMBOL.parquet            <-- fallback vecchio
    """
    rule = RULE[tf]  # es. 5m -> 5T
    return [
        f"gs://{RAW_BUCKET}/features_resampled/{sym}_{rule}.parquet",
        f"gs://{RAW_BUCKET}/resampled_{tf}/{sym}.parquet",
        f"gs://{RAW_BUCKET}/resampled_{rule}/{sym}.parquet",
    ]

# === Pattern base su candele OHLC ===
def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge colonne binarie con alcuni pattern classici:
    - engulfing_bull / engulfing_bear
    - hammer / shooting_star
    - doji
    - inside_bar / outside_bar
    Richiede colonne: open, high, low, close
    """
    req = {"open", "high", "low", "close"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Colonne OHLC mancanti: {missing}")

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    po = o.shift(1)
    ph = h.shift(1)
    pl = l.shift(1)
    pc = c.shift(1)

    body = (c - o).abs()
    prev_body = (pc - po).abs()
    rng = (h - l).replace(0, np.nan)

    # Engulfing
    engulf_bull = (pc < po) & (c > o) & (o <= pc) & (c >= po) & (body > prev_body)
    engulf_bear = (pc > po) & (c < o) & (o >= pc) & (c <= po) & (body > prev_body)

    # Hammer / Shooting star
    upper = h - np.maximum(c, o)
    lower = np.minimum(c, o) - l
    hammer = (lower >= 2 * body) & (upper <= 0.5 * body)
    shooting = (upper >= 2 * body) & (lower <= 0.5 * body)

    # Doji: corpo piccolo rispetto all'intervallo
    doji = body <= 0.1 * rng

    # Inside / Outside bar
    inside = (h <= ph) & (l >= pl)
    outside = (h >= ph) & (l <= pl)

    out = df.copy()
    out["pattern_engulfing_bull"] = engulf_bull.astype("int8")
    out["pattern_engulfing_bear"] = engulf_bear.astype("int8")
    out["pattern_hammer"] = hammer.astype("int8")
    out["pattern_shooting_star"] = shooting.astype("int8")
    out["pattern_doji"] = doji.fillna(False).astype("int8")
    out["pattern_inside_bar"] = inside.fillna(False).astype("int8")
    out["pattern_outside_bar"] = outside.fillna(False).astype("int8")

    return out

def process_symbol_tf(sym: str, tf: str) -> str:
    """
    Carica il resample corretto per (sym, tf), calcola pattern e salva:
      gs://cerbero-data-processed-gns/patterns_<tf>/<sym>.parquet
    Ritorna il path di output.
    """
    candidates = _resolve_resampled_path(sym, tf)

    src = None
    for cand in candidates:
        try:
            # test leggibilità
            _ = read_parquet_gcs(cand).head(1)
            src = cand
            break
        except Exception:
            continue

    if src is None:
        raise RuntimeError(f"Resampled mancante per {sym} [{tf}]. Provati: {candidates}")

    df = read_parquet_gcs(src)
    # indice datetime ordinato e deduplicato
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # rinomina eventuali colonne in minuscolo per coerenza
    df.columns = [c.lower() for c in df.columns]

    # assicurati che le 4 OHLC ci siano (alcune pipeline chiamano 'close_5m' ecc.)
    # se trovate forme tipo 'open_5m', le mappiamo alle canoniche
    def pick(name: str):
        if name in df.columns:
            return df[name]
        # fallback: cerca prefissi open_, high_, ...
        for col in df.columns:
            if col.startswith(name + "_"):
                return df[col]
        raise KeyError(name)

    # costruisci un df OHLC minimale
    dfo = pd.DataFrame(
        {
            "open": pick("open"),
            "high": pick("high"),
            "low": pick("low"),
            "close": pick("close"),
        },
        index=df.index,
    )

    out = add_candle_patterns(dfo)

    dst = f"gs://{OUT_BUCKET}/patterns_{tf}/{sym}.parquet"
    write_parquet_gcs(out, dst)
    return dst

def run(symbols: List[str], tfs: List[str]):
    errors = []
    for sym in symbols:
        print(f"\n=== {sym} ===")
        for tf in tfs:
            try:
                dst = process_symbol_tf(sym, tf)
                print(f"✅ {sym} [{tf}] patterns → {dst}  (cols={len(out_cols(sym, tf)) if False else 'n/a'})")
            except Exception as e:
                print(f"⚠️  {sym} [{tf}] errore: {e}")
                errors.append((sym, tf, str(e)))

    if errors:
        print("\n⚠️ Terminato con errori su:")
        for sym, tf, msg in errors:
            print(f" - {sym} [{tf}] → {msg}")
    else:
        print("\n🏁 Patterns completati senza errori.")

# (stub per compatibilità con la print sopra; non risolviamo davvero le colonne)
def out_cols(sym: str, tf: str):
    return []

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="", help="Lista separata da virgole (es. 'EURUSD,XAUUSD')")
    ap.add_argument("--tfs", type=str, default="5m,15m,4h,1d", help="Lista separata da virgole (default: tutti)")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else []
    if not symbols:
        raise SystemExit("Devi specificare --symbols (es. --symbols 'EURUSD,XAUUSD')")

    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]
    for t in tfs:
        if t not in TFS_ALL:
            raise SystemExit(f"Timeframe non valido: {t}. Validi: {TFS_ALL}")

    return symbols, tfs

if __name__ == "__main__":
    syms, tfs = parse_args()
    print("Simboli:", syms)
    print("Timeframes:", tfs)
    run(syms, tfs)
