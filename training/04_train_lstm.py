# training/04_train_lstm.py
import argparse
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: [N, T, F]
        y: [N]
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
        # x: [B, T, F]
        out, _ = self.lstm(x)        # out: [B, T, H]
        last = out[:, -1, :]         # [B, H]
        logits = self.fc(last)       # [B, C]
        return logits


def load_dataset_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    return X, y


def train_model(
    dataset_path: str,
    tf: str,
    batch_size: int = 256,
    lr: float = 1e-3,
    epochs: int = 10,
    val_split: float = 0.1,
    out_dir: str = "models",
):
    os.makedirs(out_dir, exist_ok=True)

    print(f"📂 Loading dataset from {dataset_path}")
    X, y = load_dataset_npz(dataset_path)
    print(f"   X: {X.shape}, y: {y.shape}")

    dataset = SequenceDataset(X, y)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    input_dim = X.shape[-1]
    model = CoscienzaLSTM(input_dim=input_dim).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n🚀 Starting training on {DEVICE} (TF={tf})")
    print(f"   Train samples: {train_size}, Val samples: {val_size}")
    print(f"   Input dim: {input_dim}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_X.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # Val
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                logits = model(batch_X)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_X.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_samples += batch_X.size(0)

        val_loss /= max(1, val_samples)
        val_acc = val_correct / max(1, val_samples)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.5f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.5f} val_acc={val_acc:.4f}"
        )

    out_path = os.path.join(out_dir, f"coscienza_v3_lstm_{tf}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "tf": tf,
        },
        out_path,
    )
    print(f"\n💾 Saved model → {out_path}")
    print("🏁 Training done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_path",
        required=True,
        help="Path al file .npz (es: datasets/dataset_5m.npz)",
    )
    ap.add_argument(
        "--tf",
        required=True,
        help="Timeframe (solo per naming modello, es: 5m)",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (default: 256)",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Numero epoche (default: 10)",
    )
    ap.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Frazione di dati per validazione (default: 0.1)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="models",
        help="Cartella in cui salvare il modello (default: models)",
    )
    args = ap.parse_args()

    train_model(
        dataset_path=args.dataset_path,
        tf=args.tf,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        val_split=args.val_split,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
