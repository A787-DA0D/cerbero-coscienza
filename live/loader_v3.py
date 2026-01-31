# live/loader_v3.py
"""
Loader v3 per Cerbero Coscienza.

- Carica i modelli radar (grafico / tecnico / contestuale) da GCS.
- Legge i meta JSON per sapere quali colonne usare.
- Carica pesi (weights.json) e soglie (thresholds.json).
- Fornisce funzioni per calcolare p_grafico / p_tecnico / p_contestuale / p_blend
  dato un vettore di feature (ultima barra).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import io
import json
import numpy as np
import pandas as pd
from google.cloud import storage
from joblib import load as joblib_load

from live.config_v3 import SYMBOLS, TIMEFRAMES

MODEL_BUCKET = "cerbero-models-gns"
MODEL_PREFIX = "radar_v1"   # cartella modelli (v1 come path, ma ora contenuto v3)

_client = storage.Client()


def _download_bytes(bucket_name: str, blob_name: str) -> bytes:
    bkt = _client.bucket(bucket_name)
    blob = bkt.blob(blob_name)
    return blob.download_as_bytes()


def _download_json(bucket_name: str, blob_name: str) -> dict:
    data = _download_bytes(bucket_name, blob_name)
    return json.loads(data.decode("utf-8"))


def _download_joblib(bucket_name: str, blob_name: str):
    data = _download_bytes(bucket_name, blob_name)
    bio = io.BytesIO(data)
    return joblib_load(bio)


@dataclass
class SingleModel:
    kind: str                      # "grafico" | "tecnico" | "contestuale"
    model: object                  # sklearn Pipeline
    cols: list                     # ordine colonne usate in training


@dataclass
class RadarConfig:
    symbol: str
    tf: str
    models: Dict[str, SingleModel]  # key: kind
    weights: Dict[str, float]       # w_grafico / w_tecnico / w_contestuale
    thresholds: Dict[str, float]    # thr_* (blend, grafico, tecnico, contestuale, strength)

    def predict_probas(self, row: pd.Series) -> Dict[str, float]:
        """
        row: Serie con TUTTE le feature disponibili per quel (symbol, tf).
        Ritorna:
            {
              "grafico": p_g,
              "tecnico": p_t,
              "contestuale": p_c,
              "blend": p_blend
            }
        """
        out = {}
        for kind, sm in self.models.items():
            # allinea le feature alla lista di colonne del modello
            x = _prepare_features(row, sm.cols)
            p = sm.model.predict_proba(x)[:, 1][0]
            out[kind] = float(p)

        # blend lineare con i pesi
        wg = self.weights.get("grafico", 1 / 3)
        wt = self.weights.get("tecnico", 1 / 3)
        wc = self.weights.get("contestuale", 1 / 3)

        p_blend = (
            out.get("grafico", 0.0) * wg +
            out.get("tecnico", 0.0) * wt +
            out.get("contestuale", 0.0) * wc
        )
        out["blend"] = float(p_blend)
        return out


def _prepare_features(row: pd.Series, cols_expected: list) -> np.ndarray:
    """
    Allinea la Serie 'row' all'ordine di colonne usato in training.
    Se mancano colonne, le crea a 0.0; se ci sono extra, vengono ignorate.
    Ritorna un array 2D shape (1, n_features).
    """
    # Garantiamo float
    row = row.astype(float)

    # Costruiamo una nuova Serie con le sole colonne attese (o 0 se mancano)
    data = {}
    for c in cols_expected:
        if c in row.index:
            v = row[c]
            if pd.isna(v) or np.isinf(v):
                v = 0.0
        else:
            v = 0.0
        data[c] = float(v)

    arr = np.array([data[c] for c in cols_expected], dtype=float).reshape(1, -1)
    return arr


def load_single_model(symbol: str, tf: str, kind: str) -> SingleModel:
    """
    Carica un singolo modello (grafico/tecnico/contestuale) + meta cols.
    """
    base = f"{MODEL_PREFIX}/{symbol}/{tf}/{kind}"

    # modello
    mdl_blob = f"{base}.joblib"
    model = _download_joblib(MODEL_BUCKET, mdl_blob)

    # meta json (con elenco colonne)
    json_blob = f"{base}.json"
    meta = _download_json(MODEL_BUCKET, json_blob)
    cols = meta.get("cols", [])

    return SingleModel(kind=kind, model=model, cols=cols)


def load_weights(symbol: str, tf: str) -> Dict[str, float]:
    """
    Carica i pesi di blend (grafico/tecnico/contestuale) da weights.json.
    """
    blob_name = f"{MODEL_PREFIX}/{symbol}/{tf}/weights.json"
    try:
        data = _download_json(MODEL_BUCKET, blob_name)
        # ci aspettiamo un dict tipo {"grafico": 0.4, "tecnico": 0.3, "contestuale": 0.3}
        return {
            "grafico": float(data.get("grafico", 1 / 3)),
            "tecnico": float(data.get("tecnico", 1 / 3)),
            "contestuale": float(data.get("contestuale", 1 / 3)),
        }
    except Exception:
        # fallback: pesi uguali
        return {"grafico": 1 / 3, "tecnico": 1 / 3, "contestuale": 1 / 3}


def load_thresholds(symbol: str, tf: str) -> Dict[str, float]:
    """
    Carica le soglie da thresholds.json.
    """
    blob_name = f"{MODEL_PREFIX}/{symbol}/{tf}/thresholds.json"
    try:
        data = _download_json(MODEL_BUCKET, blob_name)
        # lasciamo pass-through (thr_blend, thr_grafico, ecc.)
        return {k: float(v) for k, v in data.items()}
    except Exception:
        # fallback: soglia blend 0.55, il resto ignorato
        return {"thr_blend": 0.55}


def load_radar_config(symbol: str, tf: str) -> RadarConfig:
    """
    Carica TUTTO per un singolo (symbol, tf):
    - 3 modelli (grafico/tecnico/contestuale)
    - pesi
    - soglie
    """
    kinds = ["grafico", "tecnico", "contestuale"]
    models = {}

    for k in kinds:
        try:
            sm = load_single_model(symbol, tf, k)
            models[k] = sm
        except Exception as e:
            # se qualcosa va storto, il modello mancante semplicemente non verrà usato
            print(f"⚠️  Errore nel caricare {symbol} [{tf}] {k}: {e}")

    weights = load_weights(symbol, tf)
    thresholds = load_thresholds(symbol, tf)

    return RadarConfig(
        symbol=symbol,
        tf=tf,
        models=models,
        weights=weights,
        thresholds=thresholds,
    )


def load_all_models_v3(symbols=None, tfs=None) -> Dict[tuple, RadarConfig]:
    """
    Carica tutti i RadarConfig per (symbol, tf) dell'universo v3.

    Ritorna un dict:
        chiave: (symbol, tf)
        valore: RadarConfig
    """
    if symbols is None:
        symbols = SYMBOLS
    if tfs is None:
        tfs = TIMEFRAMES

    configs: Dict[tuple, RadarConfig] = {}

    for sym in symbols:
        for tf in tfs:
            try:
                cfg = load_radar_config(sym, tf)
                if not cfg.models:
                    print(f"⚠️  Nessun modello valido per {sym} [{tf}], salto.")
                    continue
                configs[(sym, tf)] = cfg
            except Exception as e:
                print(f"⚠️  Errore nel caricare config per {sym} [{tf}]: {e}")

    print(f"✅ load_all_models_v3: caricati {len(configs)} config Radar v3.")
    return configs
