"""Fase 5 — Visualizaciones finales y comparativa global."""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

PREVALENCE = 0.1929
MLP_THRESHOLD = 0.37
RANDOM_STATE = 42

MODELS_INFO = {
    "Random Forest v1": {
        "color": "#1f77b4",
        "Precision": 0.6305, "Recall": 0.2379, "F1": 0.3455, "AUC": 0.7962,
        "pkl": "model_randomforest.pkl", "data": "v1",
    },
    "XGBoost v1": {
        "color": "#ff7f0e",
        "Precision": 0.4393, "Recall": 0.5568, "F1": 0.4911, "AUC": 0.7708,
        "pkl": "model_xgboost.pkl", "data": "v1",
    },
    "SVM v2": {
        "color": "#2ca02c",
        "Precision": 0.4100, "Recall": 0.7024, "F1": 0.5178, "AUC": 0.7979,
        "pkl": "model_svm_v2.pkl", "data": "v2",
    },
    "MLP v2": {
        "color": "#d62728",
        "Precision": 0.3383, "Recall": 0.8440, "F1": 0.4830, "AUC": 0.8150,
        "pkl": "model_mlp_v2.pt", "data": "v2",
    },
}


# ---------- MLP definition (idéntica a la usada en entrenamiento) ----------

class MLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.25),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.25),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_mlp(path: Path, n_features: int) -> MLP:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MLP(n_features)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict_mlp_proba(model: MLP, X: pd.DataFrame) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(X.values.astype("float32"))
        return torch.sigmoid(model(x).squeeze(1)).cpu().numpy()


# ---------- LOAD DATA & PROBABILITIES ----------

def load_test_data():
    X_test_v1 = pd.read_csv(DATA_DIR / "X_test.csv")
    X_test_v2 = pd.read_csv(DATA_DIR / "X_test_v2.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns").values
    return X_test_v1, X_test_v2, y_test


def get_probas(X_v1, X_v2, y):
    probas = {}
    for name, info in MODELS_INFO.items():
        path = OUTPUTS_DIR / info["pkl"]
        X = X_v2 if info["data"] == "v2" else X_v1
        if info["pkl"].endswith(".pt"):
            model = load_mlp(path, n_features=X.shape[1])
            probas[name] = predict_mlp_proba(model, X)
        else:
            model = joblib.load(path)
            probas[name] = model.predict_proba(X)[:, 1]
    return probas


# ---------- GRÁFICO 1: BARRAS ----------

def plot_bars(path: Path) -> None:
    metrics = ["Precision", "Recall", "F1", "AUC"]
    models = list(MODELS_INFO.keys())
    x = np.arange(len(models))
    width = 0.20

    fig, ax = plt.subplots(figsize=(11, 6))
    metric_colors = ["#4C72B0", "#DD8452", "#55A467", "#8172B2"]
    for i, m in enumerate(metrics):
        vals = [MODELS_INFO[mdl][m] for mdl in models]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=m, color=metric_colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Valor")
    ax.set_title("Comparativa de métricas por modelo (test set)")
    ax.legend(loc="upper left", ncols=4)
    ax.grid(axis="y", alpha=0.3)

    mlp_idx = models.index("MLP v2")
    ax.annotate(
        "Mejor modelo",
        xy=(mlp_idx, 0.95),
        xytext=(mlp_idx, 0.92),
        ha="center",
        fontsize=11, fontweight="bold", color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5),
    )

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


# ---------- GRÁFICO 2: CURVAS ROC ----------

def plot_roc(probas, y, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for name, info in MODELS_INFO.items():
        fpr, tpr, _ = roc_curve(y, probas[name])
        auc = roc_auc_score(y, probas[name])
        plt.plot(fpr, tpr, color=info["color"], lw=2,
                 label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Aleatorio (AUC=0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curvas ROC — Comparativa de los 4 modelos finales")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


# ---------- GRÁFICO 3: PRECISION-RECALL ----------

def plot_pr(probas, y, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for name, info in MODELS_INFO.items():
        p, r, _ = precision_recall_curve(y, probas[name])
        plt.plot(r, p, color=info["color"], lw=2, label=name)
    plt.axhline(PREVALENCE, color="gray", ls="--", alpha=0.6,
                label=f"Baseline (prevalencia={PREVALENCE:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curvas Precision-Recall — Comparativa")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


# ---------- GRÁFICO 4: MATRIZ DE CONFUSIÓN MLP v2 ----------

def plot_confusion_mlp(probas_mlp, y, path: Path) -> None:
    y_pred = (probas_mlp >= MLP_THRESHOLD).astype(int)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    labels = np.array([
        [f"TN\n{tn}", f"FP\n{fp}"],
        [f"FN\n{fn}", f"TP\n{tp}"],
    ])
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, labels[i, j], ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")

    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred. Sano", "Pred. Cáncer"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Real Sano", "Real Cáncer"])
    ax.set_title(f"Matriz de confusión — MLP v2 (umbral={MLP_THRESHOLD})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


# ---------- GRÁFICO 5: CURVAS DE ENTRENAMIENTO MLP v2 ----------

def retrain_mlp_capture_curves(X_train_v2: pd.DataFrame, y_train_v2: np.ndarray):
    """Re-entrena la MLP brevemente capturando loss y F2 por época en train y val."""
    np.random.seed(RANDOM_STATE); torch.manual_seed(RANDOM_STATE)
    X = X_train_v2.values.astype("float32")
    y = y_train_v2.astype("float32")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE,
    )
    g = torch.Generator(); g.manual_seed(RANDOM_STATE)
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                           batch_size=256, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=512, shuffle=False)

    device = torch.device("cpu")
    model = MLP(X_tr.shape[1]).to(device)
    pos_weight = torch.tensor([4.18], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)

    hist = {"train_loss": [], "val_loss": [], "train_f2": [], "val_f2": []}
    best_val_loss = float("inf"); best_state = None; no_improve = 0

    for epoch in range(1, 101):
        # train
        model.train()
        tot, n, probs, tgts = 0.0, 0, [], []
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot += loss.item() * xb.size(0); n += xb.size(0)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            tgts.append(yb.detach().cpu().numpy())
        tr_loss = tot / n
        tr_probs = np.concatenate(probs); tr_tgt = np.concatenate(tgts)
        tr_f2 = fbeta_score(tr_tgt, (tr_probs >= MLP_THRESHOLD).astype(int),
                            beta=2.0, pos_label=1, zero_division=0)

        # val
        model.eval(); tot, n, probs, tgts = 0.0, 0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb).squeeze(1)
                loss = criterion(logits, yb)
                tot += loss.item() * xb.size(0); n += xb.size(0)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                tgts.append(yb.cpu().numpy())
        val_loss = tot / n
        val_probs = np.concatenate(probs); val_tgt = np.concatenate(tgts)
        val_f2 = fbeta_score(val_tgt, (val_probs >= MLP_THRESHOLD).astype(int),
                             beta=2.0, pos_label=1, zero_division=0)

        hist["train_loss"].append(tr_loss); hist["val_loss"].append(val_loss)
        hist["train_f2"].append(tr_f2); hist["val_f2"].append(val_f2)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 12:
                break

    return hist


def plot_training_curves(hist, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(hist["train_loss"]) + 1)

    axes[0].plot(epochs, hist["train_loss"], label="train")
    axes[0].plot(epochs, hist["val_loss"], label="val")
    axes[0].set_title("Loss por época (BCE con pos_weight)")
    axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, hist["train_f2"], label="train")
    axes[1].plot(epochs, hist["val_f2"], label="val")
    axes[1].set_title(f"F2-score por época (umbral={MLP_THRESHOLD})")
    axes[1].set_xlabel("Época"); axes[1].set_ylabel("F2")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


# ---------- MAIN ----------

def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    X_test_v1, X_test_v2, y_test = load_test_data()
    print(f"X_test_v1: {X_test_v1.shape}    X_test_v2: {X_test_v2.shape}    y_test: {y_test.shape}")

    print("Calculando probabilidades de cada modelo sobre su test set...")
    probas = get_probas(X_test_v1, X_test_v2, y_test)

    print("Generando gráfico 1: barras comparativas...")
    plot_bars(OUTPUTS_DIR / "comparativa_barras.png")

    print("Generando gráfico 2: curvas ROC...")
    plot_roc(probas, y_test, OUTPUTS_DIR / "curvas_roc_todos.png")

    print("Generando gráfico 3: precision-recall...")
    plot_pr(probas, y_test, OUTPUTS_DIR / "precision_recall.png")

    print("Generando gráfico 4: matriz de confusión MLP v2...")
    plot_confusion_mlp(probas["MLP v2"], y_test, OUTPUTS_DIR / "confusion_mlp_v2.png")

    print("Generando gráfico 5: curvas de entrenamiento MLP v2 (reentreno breve)...")
    X_train_v2 = pd.read_csv(DATA_DIR / "X_train_v2.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns").values
    hist = retrain_mlp_capture_curves(X_train_v2, y_train)
    plot_training_curves(hist, OUTPUTS_DIR / "curvas_mlp_v2.png")

    # tabla resumen
    rows = []
    for name, info in MODELS_INFO.items():
        rows.append({
            "Modelo": name,
            "Precision": info["Precision"],
            "Recall": info["Recall"],
            "F1": info["F1"],
            "AUC_ROC": info["AUC"],
        })
    pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "tabla_resumen_final.csv", index=False)

    print("\nTodos los gráficos y tabla guardados en:")
    for p in [
        "comparativa_barras.png", "curvas_roc_todos.png", "precision_recall.png",
        "confusion_mlp_v2.png", "curvas_mlp_v2.png", "tabla_resumen_final.csv",
    ]:
        print(f"  {OUTPUTS_DIR / p}")


if __name__ == "__main__":
    main()
