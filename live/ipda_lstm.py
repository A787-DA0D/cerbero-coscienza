# live/ipda_lstm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CoscienzaLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


@dataclass
class IPDAResult:
    bias: str            # "bull" | "bear" | "neutral"
    p_up: float
    strong: bool         # True se bias “forte”


class IPDALSTM:
    """
    Carica 1 modello per timeframe e produce:
    - p_up = P(classe 1)
    - bias bull/bear/neutral
    """

    def __init__(self, model_paths: Dict[str, str], bull_thr: float = 0.55, bear_thr: float = 0.45):
        self.model_paths = model_paths
        self.bull_thr = float(bull_thr)
        self.bear_thr = float(bear_thr)
        self._cache: Dict[str, Tuple[nn.Module, int]] = {}  # tf -> (model, input_dim)

    def _load(self, tf: str) -> Tuple[nn.Module, int]:
        if tf in self._cache:
            return self._cache[tf]

        path = self.model_paths.get(tf)
        if not path:
            raise FileNotFoundError(f"No LSTM path configured for tf={tf}")

        ckpt = torch.load(path, map_location=DEVICE)
        input_dim = int(ckpt.get("input_dim", 0))
        if input_dim <= 0:
            raise ValueError(f"Invalid input_dim in checkpoint: {path}")

        model = CoscienzaLSTM(input_dim=input_dim).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        self._cache[tf] = (model, input_dim)
        return model, input_dim

    def predict(self, tf: str, X_seq: np.ndarray) -> IPDAResult:
        """
        X_seq: shape (T, F) oppure (1, T, F)
        """
        model, input_dim = self._load(tf)

        if X_seq.ndim == 2:
            X_seq = X_seq[None, :, :]
        if X_seq.shape[-1] != input_dim:
            raise ValueError(f"LSTM input_dim mismatch tf={tf}: got {X_seq.shape[-1]} expected {input_dim}")

        x = torch.from_numpy(X_seq.astype(np.float32)).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0, 1].item()

        if prob >= self.bull_thr:
            return IPDAResult(bias="bull", p_up=float(prob), strong=True)
        if prob <= self.bear_thr:
            return IPDAResult(bias="bear", p_up=float(prob), strong=True)
        return IPDAResult(bias="neutral", p_up=float(prob), strong=False)
