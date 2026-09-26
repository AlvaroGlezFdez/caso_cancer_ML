"""Fase 4 — Red Neuronal MLP con BatchNorm, Dropout y ajuste de umbral.

Implementación en PyTorch (TensorFlow no instalable en Windows por límite de Long Path).
Arquitectura idéntica al plan original.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 256
VAL_SPLIT = 0.2
POS_WEIGHT = 4.18
LR = 1e-3
ES_PATIENCE = 12
LR_PATIENCE = 6
LR_FACTOR = 0.5


def set_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_splits():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv").values.astype("float32")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv").values.astype("float32")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns").values.astype("float32")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns").values.astype("float32")
    return X_train, X_test, y_train, y_test


class MLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.20),

            nn.Linear(32, 1),  # logits — sigmoide se aplica fuera
        )

    def forward(self, x):
        return self.net(x)


def epoch_pass(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, n = 0.0, 0
    all_probs, all_targets = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_targets.append(yb.detach().cpu().numpy())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    avg_loss = total_loss / n
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = float("nan")
    return avg_loss, auc, probs, targets


def find_best_threshold(y_val, y_proba_val) -> tuple[float, float]:
    thresholds = np.arange(0.10, 0.91, 0.01)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_proba_val >= t).astype(int)
        f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def plot_history(history, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Binary crossentropy (BCEWithLogits)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_auc"], label="train")
    axes[1].plot(history["val_auc"], label="val")
    axes[1].set_title("AUC-ROC por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("AUC")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    set_seed()
    OUTPUTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train_full, X_test, y_train_full, y_test = load_splits()
    print(f"Train: {X_train_full.shape}    Test: {X_test.shape}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VAL_SPLIT,
        stratify=y_train_full,
        random_state=RANDOM_STATE,
    )
    print(f"Tr interno: {X_tr.shape}    Val interno: {X_val.shape}")

    g = torch.Generator()
    g.manual_seed(RANDOM_STATE)

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True, generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=512, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=512, shuffle=False,
    )

    model = MLP(n_features=X_tr.shape[1]).to(device)
    print(model)

    pos_weight = torch.tensor([POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE,
    )

    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    last_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_auc, _, _ = epoch_pass(model, tr_loader, criterion, optimizer, device, train=True)
        val_loss, val_auc, _, _ = epoch_pass(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(tr_auc)
        history["val_auc"].append(val_auc)

        print(f"Epoch {epoch:3d}  tr_loss={tr_loss:.4f} val_loss={val_loss:.4f}  "
              f"tr_auc={tr_auc:.4f} val_auc={val_auc:.4f}  lr={cur_lr:.2e}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= ES_PATIENCE:
                print(f"EarlyStopping en epoch {epoch} (sin mejora en {ES_PATIENCE} epochs).")
                last_epoch = epoch
                break
        last_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Mejor val_loss: {best_val_loss:.4f} restaurado.")

    _, _, val_probs, val_targets = epoch_pass(model, val_loader, criterion, optimizer, device, train=False)
    best_t, best_f1_val = find_best_threshold(val_targets, val_probs)
    print(f"\nUmbral óptimo (val): {best_t:.2f}    F1 val: {best_f1_val:.4f}")

    _, _, test_probs, test_targets = epoch_pass(model, test_loader, criterion, optimizer, device, train=False)
    test_pred = (test_probs >= best_t).astype(int)

    precision = precision_score(test_targets, test_pred, pos_label=1)
    recall = recall_score(test_targets, test_pred, pos_label=1)
    f1 = f1_score(test_targets, test_pred, pos_label=1)
    auc = roc_auc_score(test_targets, test_probs)
    acc = accuracy_score(test_targets, test_pred)
    cm = confusion_matrix(test_targets, test_pred)

    print("\n--- MLP (test) ---")
    print(f"Umbral aplicado    : {best_t:.2f}")
    print(f"Precision (cancer=1): {precision:.4f}")
    print(f"Recall    (cancer=1): {recall:.4f}")
    print(f"F1        (cancer=1): {f1:.4f}")
    print(f"AUC-ROC            : {auc:.4f}")
    print(f"Accuracy           : {acc:.4f}")
    print("Matriz de confusión [[TN FP] [FN TP]]:")
    print(cm)
    print(f"Épocas entrenadas  : {last_epoch}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": X_tr.shape[1],
        "best_threshold": best_t,
    }, OUTPUTS_DIR / "model_mlp.pt")

    plot_history(history, OUTPUTS_DIR / "curvas_mlp.png")

    res_path = OUTPUTS_DIR / "resultados_ml.csv"
    df = pd.read_csv(res_path)
    df = df[df["Modelo"] != "MLP"]
    new_row = pd.DataFrame([{
        "Modelo": "MLP",
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC_ROC": auc,
        "Accuracy": acc,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(res_path, index=False)

    print(f"\nResultados actualizados en {res_path}")
    print(f"Modelo guardado en {OUTPUTS_DIR / 'model_mlp.pt'}")
    print(f"Curvas guardadas en {OUTPUTS_DIR / 'curvas_mlp.png'}")


if __name__ == "__main__":
    main()
