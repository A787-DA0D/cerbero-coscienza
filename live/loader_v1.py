# live/loader_v1.py
"""
Loader v1 per i modelli radar live.

- Carica i modelli grafico/tecnico/contestuale da GCS (joblib).
- Carica pesi (weights.json) e soglie (thresholds.json) per simbolo/timeframe.
- Espone funzioni helper per il live_loop:
    - load_radar_bundle(symbols, tfs)
    - load_radar_models(symbols, tfs)
    - load_models(symbols, tfs)

Gestisce il mismatch 39 feature vs n_feature_in_ del modello
usando model.feature_names_in_ quando disponibile.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Any

import joblib
import pandas as pd
from google.cloud import storage

# === Config GCS ===
MODEL_BUCKET = "cerbero-models-gns"
RADAR_PREFIX = "radar_v1"

_storage_client: storage.Client | None = None


def _get_storage_client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


def _gcs_path(symbol: str, tf: str, filename: str) -> str:
    """Path relativo dentro al bucket per un certo file del radar."""
    return f"{RADAR_PREFIX}/{symbol}/{tf}/{filename}"


def _download_to_temp(symbol: str, tf: str, filename: str) -> str:
    """
    Scarica un file da GCS in /tmp e ritorna il path locale.
    """
    client = _get_storage_client()
    bucket = client.bucket(MODEL_BUCKET)
    blob_path = _gcs_path(symbol, tf, filename)
    blob = bucket.blob(blob_path)

    # estensione solo per estetica
    _, ext = os.path.splitext(filename)
    fd, tmp_path = tempfile.mkstemp(prefix="radar_", suffix=ext or ".bin")
    os.close(fd)

    blob.download_to_filename(tmp_path)
    return tmp_path


@dataclass
class RadarModel:
    """
    Wrapper semplice per un modello sklearn (Pipeline) pre-addestrato
    che espone predict_proba(row).
    """

    symbol: str
    tf: str
    kind: str  # "grafico" | "tecnico" | "contestuale"
    model: Any

    @classmethod
    def from_gcs(cls, symbol: str, tf: str, kind: str) -> "RadarModel":
        """
        Carica il modello <kind>.joblib da GCS.
        """
        local_path = _download_to_temp(symbol, tf, f"{kind}.joblib")
        mdl = joblib.load(local_path)
        return cls(symbol=symbol, tf=tf, kind=kind, model=mdl)

    def predict_proba(self, row: pd.Series) -> float:
        """
        row: Series con tutte le feature disponibili (es. 39 colonne).
        Riallinea alle colonne usate in training usando
        model.feature_names_in_ se presente.
        """
        # 1) una sola riga → DataFrame (1 x n_col)
        X = row.to_frame().T

        # 2) Se il modello conosce le colonne di training, usiamole
        feat_names = getattr(self.model, "feature_names_in_", None)
        if feat_names is not None:
            cols: List[str] = list(feat_names)

            # in caso manchi qualche colonna, la creiamo a 0.0
            for c in cols:
                if c not in X.columns:
                    X[c] = 0.0

            # ordina/filtra esattamente come in training
            X = X[cols]

        # 3) Probabilità della classe positiva
        proba = float(self.model.predict_proba(X)[0, 1])
        return proba


@dataclass
class RadarConfig:
    """
    Configurazione completa per (symbol, tf):
      - modelli grafico/tecnico/contestuale
      - pesi di blend
      - soglie (thresholds) e strength
    """

    symbol: str
    tf: str
    models: Dict[str, RadarModel]
    weights: Dict[str, float]
    thresholds: Dict[str, float]
    strength: float = 0.0


def _load_json_from_gcs(symbol: str, tf: str, filename: str) -> Dict[str, Any]:
    """
    Legge un JSON (weights.json / thresholds.json) da GCS.
    Se non esiste, ritorna {}.
    """
    client = _get_storage_client()
    bucket = client.bucket(MODEL_BUCKET)
    blob_path = _gcs_path(symbol, tf, filename)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        return {}

    text = blob.download_as_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # in caso di file corrotto o vuoto
        return {}


def load_radar_config(symbol: str, tf: str) -> RadarConfig:
    """
    Carica tutto ciò che serve per un singolo (symbol, tf):
      - modelli grafico/tecnico/contestuale
      - pesi
      - soglie + strength
    """
    # Modelli
    models = {
        kind: RadarModel.from_gcs(symbol, tf, kind)
        for kind in ("grafico", "tecnico", "contestuale")
    }

    # Pesi
    weights_data = _load_json_from_gcs(symbol, tf, "weights.json")
    weights = weights_data.get("weights", {})

    # Soglie
    thr_data = _load_json_from_gcs(symbol, tf, "thresholds.json")
    thresholds = thr_data.get("thresholds", {})
    strength = float(thr_data.get("strength", 0.0))

    return RadarConfig(
        symbol=symbol,
        tf=tf,
        models=models,
        weights=weights,
        thresholds=thresholds,
        strength=strength,
    )


def load_radar_bundle(symbols: List[str], tfs: List[str]) -> Dict[str, Dict[str, RadarConfig]]:
    """
    Ritorna una struttura:
    {
        "EURUSD": {
            "5m": RadarConfig(...),
            "15m": RadarConfig(...),
            ...
        },
        "BTCUSD": {
            ...
        },
        ...
    }
    """
    out: Dict[str, Dict[str, RadarConfig]] = {}
    for s in symbols:
        out[s] = {}
        for tf in tfs:
            try:
                out[s][tf] = load_radar_config(s, tf)
            except Exception as e:
                # In live è meglio loggare l'errore ma non esplodere.
                # Qui lasciamo che il chiamante decida cosa fare;
                # puoi anche stampare se vuoi debug:
                # print(f"⚠️ errore load_radar_config({s}, {tf}): {e}")
                raise
    return out


# Alias per compatibilità con eventuali import nel live_loop
def load_radar_models(symbols: List[str], tfs: List[str]) -> Dict[str, Dict[str, RadarConfig]]:
    return load_radar_bundle(symbols, tfs)


def load_models(symbols: List[str], tfs: List[str]) -> Dict[str, Dict[str, RadarConfig]]:
    return load_radar_bundle(symbols, tfs)
