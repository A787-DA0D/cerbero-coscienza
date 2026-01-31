import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

# Dove leggere le feature ICT (output del 02_features v3)
DEFAULT_FEATURES_DIR = "local_features"
DEFAULT_OUT_DIR = "datasets"

# Timeframe supportati (devono combaciare con 02_features)
VALID_TFS = ["5m", "15m", "4h", "1d"]

# Parametri sequenza / target (puoi ajustarli se vuoi)
SEQ_LEN = 128          # lunghezza finestra temporale
PRED_HORIZON = 4       # numero di barre future da guardare
MIN_RET_ABS = 0.0      # (al momento non usata, potremo usarla dopo)


def load_symbol_tf_features(
    features_dir: str,
    symbol: str,
    tf: str,
) -> pd.DataFrame:
    """
    Legge il parquet delle feature per un dato symbol+tf.
    Usa la struttura:
        {features_dir}/features_<tf>/<symbol>.parquet
    """
    path = os.path.join(features_dir, f"features_{tf}", f"{symbol}.parquet")
    print(f"[LOAD] {symbol} [{tf}] <- {path} (exists={os.path.exists(path)})")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features file not found: {path}")

    df = pd.read_parquet(path)
    # Assicuro indice temporale ordinato
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~df.index.isna()].sort_index()

    # Non facciamo dropna aggressivo qui.
    df["symbol"] = symbol

    print(f"[LOAD] {symbol} [{tf}] df shape = {df.shape}")
    return df


def build_sequences_for_symbol(
    df: pd.DataFrame,
    tf: str,
    seq_len: int = SEQ_LEN,
    pred_horizon: int = PRED_HORIZON,
    min_ret_abs: float = MIN_RET_ABS,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Costruisce:
      X: [num_samples, seq_len, num_features]
      y: [num_samples] (0 = down, 1 = up)
    per un singolo symbol e un singolo timeframe.

    Versione semplificata: ogni barra (tranne le ultime pred_horizon)
    contribuisce con una etichetta UP/DOWN basata sul close futuro.
    """
    df = df.copy()

    if "close" not in df.columns:
        raise ValueError("Column 'close' not found in DataFrame.")

    # Seleziono solo colonne numeriche per X
    num_df = df.select_dtypes(include=[np.number]).copy()
    # Rimpiazzo NaN / inf con 0.0 per evitare problemi
    num_df = num_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    feature_names = list(num_df.columns)
    values = num_df.to_numpy(dtype=np.float32)

    close = df["close"].to_numpy(dtype=np.float32)
    n = len(df)

    # Calcolo il close futuro a pred_horizon barre
    future_close = np.full_like(close, np.nan, dtype=np.float32)
    if n > pred_horizon:
        future_close[:-pred_horizon] = close[pred_horizon:]

    # Rendimento relativo
    denom = np.where(close != 0.0, close, 1.0)
    ret = (future_close - close) / denom

    # Label: 0 (down) / 1 (up)
    labels = np.zeros(n, dtype=np.int64)
    labels[ret > 0.0] = 1

    max_start = n - seq_len - pred_horizon
    print(f"[SEQ] TF={tf} df_len={n}, seq_len={seq_len}, pred_horizon={pred_horizon}, max_start={max_start}")
    if max_start <= 0:
        print(f"[SEQ] TF={tf} not enough data to build any sequence.")
        return (
            np.empty((0, seq_len, len(feature_names)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            feature_names,
        )

    samples_x = []
    samples_y = []

    for start in range(max_start):
        end = start + seq_len
        y_idx = end - 1  # etichetta sulla barra finale della finestra

        if np.isnan(ret[y_idx]):
            continue

        x_seq = values[start:end, :]
        y_val = labels[y_idx]

        samples_x.append(x_seq)
        samples_y.append(y_val)

    if not samples_x:
        print(f"[SEQ] TF={tf} produced 0 samples (all ret NaN?)")
        return (
            np.empty((0, seq_len, len(feature_names)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            feature_names,
        )

    X = np.stack(samples_x, axis=0)
    y = np.array(samples_y, dtype=np.int64)

    print(
        f"[SEQ] TF={tf}: produced samples={X.shape[0]}, "
        f"seq_len={seq_len}, num_features={X.shape[2]}"
    )

    return X, y, feature_names


def build_dataset(
    symbols: List[str],
    tf: str,
    features_dir: str = DEFAULT_FEATURES_DIR,
    out_dir: str = DEFAULT_OUT_DIR,
) -> str:
    """
    Costruisce un dataset unico concatenando tutti i simboli per il TF scelto.
    Restituisce il path del file .npz creato, oppure stringa vuota se vuoto.
    """
    os.makedirs(out_dir, exist_ok=True)

    all_X = []
    all_y = []
    final_feature_names: List[str] = []

    for sym in symbols:
        print(f"\n🔧 Building sequences for {sym} [{tf}] ...")
        try:
            df = load_symbol_tf_features(features_dir, sym, tf)
        except FileNotFoundError as e:
            print(f"⚠️ {e}")
            continue

        X_sym, y_sym, feature_names = build_sequences_for_symbol(df, tf)

        print(f"[RESULT] {sym} [{tf}] → X_sym.shape={X_sym.shape}, y_sym.shape={y_sym.shape}")

        if X_sym.shape[0] == 0:
            print(f"⚠️ No samples for {sym} [{tf}] (X_sym is empty)")
            continue

        if not final_feature_names:
            final_feature_names = feature_names
        else:
            if feature_names != final_feature_names:
                raise ValueError(f"Feature mismatch for symbol {sym}: {feature_names} != {final_feature_names}")

        all_X.append(X_sym)
        all_y.append(y_sym)

    if not all_X:
        print("❌ No samples produced for any symbol; dataset is empty. NOT saving dataset.")
        return ""

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"\n📦 Final dataset for TF={tf}:")
    print(f"   X shape = {X.shape}  (samples, seq_len, num_features)")
    print(f"   y shape = {y.shape}")
    print(f"   num_features = {len(final_feature_names)}")

    out_path = os.path.join(out_dir, f"dataset_{tf}.npz")
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        feature_names=np.array(final_feature_names),
        seq_len=np.array([SEQ_LEN]),
        pred_horizon=np.array([PRED_HORIZON]),
    )
    print(f"\n💾 Saved dataset → {out_path}")

    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--symbols",
        required=True,
        help='Lista simboli, es: "EURUSD,GBPUSD,USDJPY"',
    )
    ap.add_argument(
        "--tf",
        required=True,
        choices=VALID_TFS,
        help="Timeframe da usare (deve combaciare con le feature esistenti)",
    )
    ap.add_argument(
        "--features_dir",
        default=DEFAULT_FEATURES_DIR,
        help="Cartella base delle feature (default: local_features)",
    )
    ap.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help="Cartella output per i dataset (default: datasets)",
    )
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    out_path = build_dataset(
        symbols=symbols,
        tf=args.tf,
        features_dir=args.features_dir,
        out_dir=args.out_dir,
    )

    if out_path:
        print("🏁 Dataset build done.")
    else:
        print("🏁 Dataset build ended with EMPTY dataset (no file saved).")


if __name__ == "__main__":
    main()
