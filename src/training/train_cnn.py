"""Entrenamiento de la CNN (fase: Modelado).

Función de pérdida: entropía cruzada ponderada
    L = -(1/N) sum_i  w_{y_i} log softmax(z_i)_{y_i},   w_k = N / (K N_k)
Optimizador: Adam (lr con decaimiento coseno) + weight decay (regularización L2).
Aumento de datos (solo train): giros, rotaciones de 90°, cambios de brillo y
saturación -> el modelo ve variaciones plausibles de la misma fruta.
Parada temprana: se guarda el modelo con mejor F1 macro en validación.

Hiperparámetros a explorar (documentar en experiments/results):
    lr in {1e-3, 3e-4}, batch in {16, 32}, dropout in {0.3, 0.5}, widths.
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src import config
from src.models.cnn import SmallCNN, count_params
from src.utils.helpers import set_seed

MEAN = np.array([0.5, 0.5, 0.5], np.float32)
STD = np.array([0.25, 0.25, 0.25], np.float32)


def to_tensor(crop_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 -> tensor float normalizado C x H x W."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (config.IMG_SIZE, config.IMG_SIZE))
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy())


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1, :]
    img = np.rot90(img, k=rng.integers(4)).copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    mask = img.max(axis=2) > 0
    hsv[..., 1] *= rng.uniform(0.85, 1.15)
    hsv[..., 2] *= rng.uniform(0.8, 1.2)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    out[~mask] = 0  # el fondo sigue negro
    return out


class CropDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        self.paths = [config.ROOT / p for p in df["crop"]]
        self.y = df["quality"].map({c: i for i, c in enumerate(config.QUALITY_CLASSES)}).to_numpy()
        self.train = train
        self.rng = np.random.default_rng(config.SEED)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(str(self.paths[i]))
        if self.train:
            img = augment(img, self.rng)
        return to_tensor(img), int(self.y[i])


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu())
            ys.append(y)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return f1_score(y, p, average="macro"), float((y == p).mean())


def main(epochs: int = 80, lr: float = 1e-3, batch: int = 32, dropout: float = 0.3,
         patience: int = 15, weight_decay: float = 1e-4) -> dict:
    set_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    df = pd.read_csv(config.SPLITS_CSV)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    dl_tr = DataLoader(CropDataset(tr, True), batch_size=batch, shuffle=True)
    dl_va = DataLoader(CropDataset(va, False), batch_size=batch)
    # F1 de train sobre una submuestra fija (sin aumento) para vigilar sobreajuste
    tr_eval = tr.sample(min(len(tr), 2000), random_state=config.SEED)
    dl_tr_eval = DataLoader(CropDataset(tr_eval, False), batch_size=128)

    K = len(config.QUALITY_CLASSES)
    counts = np.bincount(dl_tr.dataset.y, minlength=K).astype(np.float32)
    weights = torch.tensor(counts.sum() / (K * np.maximum(counts, 1)), device=device)

    model = SmallCNN(K, dropout=dropout).to(device)
    print(f"[cnn] parámetros entrenables: {count_params(model):,}  dispositivo: {device}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    history, best_f1, wait = [], -1.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        sched.step()
        f1_tr, acc_tr = evaluate(model, dl_tr_eval, device)
        f1_va, acc_va = evaluate(model, dl_va, device)
        history.append({"epoch": ep, "loss": tot / n, "f1_train": f1_tr, "acc_train": acc_tr,
                        "f1_val": f1_va, "acc_val": acc_va})
        print(f"  ep {ep:02d} loss={tot / n:.4f} f1_tr={f1_tr:.3f} f1_val={f1_va:.3f}")
        if f1_va > best_f1:
            best_f1, wait = f1_va, 0
            torch.save({"state_dict": model.state_dict(), "dropout": dropout,
                        "classes": config.QUALITY_CLASSES}, config.CHECKPOINTS_DIR / "cnn.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"[cnn] parada temprana en la época {ep}")
                break

    pd.DataFrame(history).to_csv(config.LOGS_DIR / "cnn_history.csv", index=False)
    info = {"lr": lr, "batch": batch, "dropout": dropout, "weight_decay": weight_decay,
            "best_f1_val": best_f1, "epochs_run": len(history)}
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "cnn_config.json").write_text(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    a = ap.parse_args()
    main(a.epochs, a.lr, a.batch, a.dropout)
