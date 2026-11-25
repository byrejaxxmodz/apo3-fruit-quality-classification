"""Búsqueda de hiperparámetros de la CNN (fase: Modelado).

La consigna pide ajustar hiperparámetros. Se explora una rejilla sobre la tasa
de aprendizaje, el dropout y el ancho de las capas; cada configuración se
entrena con el mismo presupuesto (mismas épocas y parada temprana) y se compara
por **F1 macro en validación**. La configuración ganadora se guarda como el
modelo final `cnn.pt`, de modo que `evaluate.py` reporte ya la CNN ajustada.

Rejilla (por defecto 2 x 2 x 2 = 8 configuraciones):
    lr      in {1e-3, 3e-4}
    dropout in {0.3, 0.5}
    widths  in {(32,64,128), (16,32,64)}   # capacidad del extractor
Fijos: batch=32, weight_decay=1e-4, Adam + coseno, entropía cruzada ponderada.

Eficiencia: los recortes (128x128) se precargan una sola vez en RAM (~0.5 GB)
para no leer disco en cada época; el aumento de datos se aplica al vuelo. Así
una época cuesta ~11 s en MPS y la rejilla completa cabe en minutos.

Salidas en experiments/:
    results/cnn_busqueda.csv : una fila por configuración (F1 val, época, épocas)
    results/cnn_config.json  : configuración ganadora + toda la rejilla
    checkpoints/cnn.pt       : modelo final (la mejor configuración)
    logs/cnn_history.csv     : curvas de la configuración ganadora
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src import config
from src.models.cnn import SmallCNN, count_params
from src.training.train_cnn import augment, to_tensor
from src.utils.helpers import set_seed

# Rejilla de hiperparámetros
LRS = [1e-3, 3e-4]
DROPOUTS = [0.3, 0.5]
WIDTHS = [(32, 64, 128), (16, 32, 64)]
BATCH = 32
WEIGHT_DECAY = 1e-4


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CachedCrops(Dataset):
    """Recortes ya decodificados en RAM; el aumento se aplica al vuelo."""

    def __init__(self, imgs: list[np.ndarray], y: np.ndarray, train: bool):
        self.imgs, self.y, self.train = imgs, y, train
        self.rng = np.random.default_rng(config.SEED)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i]
        if self.train:
            img = augment(img, self.rng)
        return to_tensor(img), int(self.y[i])


def _load_split(df: pd.DataFrame):
    imgs = [cv2.imread(str(config.ROOT / p)) for p in df["crop"]]
    y = df["quality"].map({c: i for i, c in enumerate(config.QUALITY_CLASSES)}).to_numpy()
    return imgs, y


def _f1(model, loader, device) -> float:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu())
            ys.append(y)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return f1_score(y, p, average="macro")


def train_one(cfg, data, device, epochs, patience):
    """Entrena una configuración; devuelve (mejor_f1_val, mejor_época, historia, estado)."""
    set_seed(config.SEED)
    (tr_imgs, tr_y), (va_imgs, va_y), (ev_imgs, ev_y) = data
    dl_tr = DataLoader(CachedCrops(tr_imgs, tr_y, True), batch_size=cfg["batch"], shuffle=True)
    dl_va = DataLoader(CachedCrops(va_imgs, va_y, False), batch_size=128)
    dl_ev = DataLoader(CachedCrops(ev_imgs, ev_y, False), batch_size=128)

    K = len(config.QUALITY_CLASSES)
    counts = np.bincount(tr_y, minlength=K).astype(np.float32)
    weights = torch.tensor(counts.sum() / (K * np.maximum(counts, 1)), device=device)

    model = SmallCNN(K, widths=cfg["widths"], dropout=cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    history, best_f1, best_ep, best_state, wait = [], -1.0, 0, None, 0
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
        f1_va, f1_tr = _f1(model, dl_va, device), _f1(model, dl_ev, device)
        history.append({"epoch": ep, "loss": tot / n, "f1_train": f1_tr, "f1_val": f1_va})
        if f1_va > best_f1:
            best_f1, best_ep, wait = f1_va, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    return best_f1, best_ep, history, best_state


def main(epochs: int = 35, patience: int = 8, solo_mejor: bool = False) -> pd.DataFrame:
    """Con `solo_mejor` se reentrena únicamente la configuración ganadora que ya
    está en cnn_config.json (por ejemplo, después de corregir la partición) y se
    conserva la rejilla anterior en vez de repetir las 8 corridas."""
    device = _device()
    previo = None
    if solo_mejor:
        previo = json.loads((config.RESULTS_DIR / "cnn_config.json").read_text())
    df = pd.read_csv(config.SPLITS_CSV)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    ev = tr.sample(min(len(tr), 2000), random_state=config.SEED)  # F1 train sin aumento

    print(f"[búsqueda] dispositivo: {device}  | precargando recortes en RAM...")
    t0 = time.time()
    data = (_load_split(tr), _load_split(va), _load_split(ev))
    print(f"[búsqueda] precarga: {time.time() - t0:.1f}s  "
          f"(train={len(tr)}, val={len(va)})")

    grid = [{"lr": lr, "dropout": dr, "widths": w, "batch": BATCH, "weight_decay": WEIGHT_DECAY}
            for lr, dr, w in itertools.product(LRS, DROPOUTS, WIDTHS)]
    if previo:
        m = previo["mejor"]
        grid = [{**m, "widths": tuple(m["widths"])}]

    rows, best = [], None
    for i, cfg in enumerate(grid, 1):
        t = time.time()
        f1_va, ep, hist, state = train_one(cfg, data, device, epochs, patience)
        params = count_params(SmallCNN(len(config.QUALITY_CLASSES), widths=cfg["widths"],
                                       dropout=cfg["dropout"]))
        row = {"lr": cfg["lr"], "dropout": cfg["dropout"], "widths": str(cfg["widths"]),
               "params": params, "f1_val": round(f1_va, 4), "mejor_epoca": ep,
               "epocas_corridas": len(hist), "segundos": round(time.time() - t, 1)}
        rows.append(row)
        print(f"  [{i}/{len(grid)}] lr={cfg['lr']} dropout={cfg['dropout']} "
              f"widths={cfg['widths']} -> F1_val={f1_va:.4f} (ep {ep}, {row['segundos']}s)")
        if best is None or f1_va > best["f1_val"]:
            best = {"cfg": cfg, "f1_val": f1_va, "history": hist, "state": state}

    res = pd.DataFrame(rows).sort_values("f1_val", ascending=False).reset_index(drop=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not previo:
        res.to_csv(config.RESULTS_DIR / "cnn_busqueda.csv", index=False)

    # Promover la mejor configuración como modelo final (el que usa evaluate.py)
    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state"], "dropout": best["cfg"]["dropout"],
                "widths": best["cfg"]["widths"], "classes": config.QUALITY_CLASSES},
               config.CHECKPOINTS_DIR / "cnn.pt")
    pd.DataFrame(best["history"]).to_csv(config.LOGS_DIR / "cnn_history.csv", index=False)
    info = {"mejor": {"lr": best["cfg"]["lr"], "dropout": best["cfg"]["dropout"],
                      "widths": list(best["cfg"]["widths"]), "batch": BATCH,
                      "weight_decay": WEIGHT_DECAY},
            "best_f1_val": round(best["f1_val"], 4),
            "epochs": epochs, "patience": patience,
            "rejilla": rows}
    if previo:
        info["rejilla"] = previo["rejilla"]
        info["nota"] = ("La rejilla se evaluó con la partición anterior; con la partición actual "
                        "solo se reentrenó la configuración ganadora (best_f1_val es de esa corrida).")
    (config.RESULTS_DIR / "cnn_config.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))

    print("\n=== Rejilla (ordenada por F1 de validación) ===")
    print(res.to_string(index=False))
    print(f"\nMejor: {info['mejor']}  ->  F1_val={best['f1_val']:.4f}")
    print("Guardado como experiments/checkpoints/cnn.pt (modelo final).")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--solo-mejor", action="store_true",
                    help="reentrenar solo la configuración ganadora de cnn_config.json")
    a = ap.parse_args()
    main(a.epochs, a.patience, a.solo_mejor)
