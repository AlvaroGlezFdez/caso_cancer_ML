"""Mejora de modelos v2: feature engineering + F2 threshold + ensemble suave."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

RANDOM_STATE = 42
ACTIVIDAD_MAP = {"Baja": 0, "Moderada": 1, "Alta": 2}
TARGET = "cancer"

MUTACIONES = ["mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS",
              "mut_PIK3CA", "mut_ALK", "mut_BRAF"]
CONTINUOUS_BASE = ["glucosa", "colesterol", "trigliceridos", "hemoglobina",
                   "leucocitos", "plaquetas", "creatinina", "edad"]
BINARY_ORDINAL_BASE = MUTACIONES + ["fumador", "actividad_fisica"]

CONTINUOUS_V2 = CONTINUOUS_BASE + ["n_mutaciones"]
BINARY_ORDINAL_V2 = BINARY_ORDINAL_BASE + ["glucosa_alta", "hemoglobina_baja", "fumador_mayor55"]


# ---------- PASO 1: FEATURE ENGINEERING ----------

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["actividad_fisica"] = df["actividad_fisica"].map(ACTIVIDAD_MAP).astype(int)
    df["n_mutaciones"] = df[MUTACIONES].sum(axis=1).astype(int)
    df["glucosa_alta"] = (df["glucosa"] > 130).astype(int)
    df["hemoglobina_baja"] = (df["hemoglobina"] < 11).astype(int)
    df["fumador_mayor55"] = (df["fumador"].astype(int) * (df["edad"] > 55).astype(int)).astype(int)
    return df


def make_v2_splits():
    df = pd.read_csv(DATA_DIR / "dataset_merged.csv")
    df = feature_engineering(df)
    X = df[CONTINUOUS_V2 + BINARY_ORDINAL_V2].copy()
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE,
    )
    scaler = StandardScaler()
    X_train[CONTINUOUS_V2] = scaler.fit_transform(X_train[CONTINUOUS_V2])
    X_test[CONTINUOUS_V2] = scaler.transform(X_test[CONTINUOUS_V2])

    X_train.to_csv(DATA_DIR / "X_train_v2.csv", index=False)
    X_test.to_csv(DATA_DIR / "X_test_v2.csv", index=False)
    joblib.dump(scaler, OUTPUTS_DIR / "scaler_v2.pkl")
    return X_train, X_test, y_train, y_test


# ---------- PASO 2: MODELOS CLÁSICOS ----------

SCALE_POS_WEIGHT = 40357 / 9644


def train_classical(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=200, scale_pos_weight=SCALE_POS_WEIGHT,
        random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
    )
    svm = SVC(kernel="rbf", class_weight="balanced",
              probability=True, random_state=RANDOM_STATE)

    print("\n[v2] Entrenando RandomForest...")
    rf.fit(X_train, y_train)
    print("[v2] Entrenando XGBoost...")
    xgb.fit(X_train, y_train)
    print("[v2] Entrenando SVM (kernel RBF — puede tardar varios minutos)...")
    svm.fit(X_train, y_train)
    return {"RandomForest": rf, "XGBoost": xgb, "SVM": svm}


# ---------- PASO 2 bis: MLP ----------

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


def epoch_pass(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    probs, targets = [], []
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
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            targets.append(yb.detach().cpu().numpy())
    return total_loss / n, np.concatenate(probs), np.concatenate(targets)


def train_mlp(X_train, y_train):
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cpu")

    X_train = X_train.values.astype("float32")
    y_train = y_train.values.astype("float32")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE,
    )

    g = torch.Generator(); g.manual_seed(RANDOM_STATE)
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                           batch_size=256, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=512, shuffle=False)

    model = MLP(X_tr.shape[1]).to(device)
    pos_weight = torch.tensor([4.18], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)

    best_val_loss = float("inf"); best_state = None; no_improve = 0
    for epoch in range(1, 101):
        tr_loss, _, _ = epoch_pass(model, tr_loader, criterion, optimizer, device, True)
        val_loss, _, _ = epoch_pass(model, val_loader, criterion, optimizer, device, False)
        scheduler.step(val_loss)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 12:
                print(f"[v2 MLP] EarlyStopping en epoch {epoch} (mejor val_loss={best_val_loss:.4f})")
                break
    model.load_state_dict(best_state)

    _, val_probs, val_targets = epoch_pass(model, val_loader, criterion, optimizer, device, False)
    return model, val_probs, val_targets


def predict_mlp(model, X):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X.values.astype("float32"))
        return torch.sigmoid(model(x).squeeze(1)).cpu().numpy()


# ---------- UMBRAL F2 ----------

def find_best_threshold_fbeta(y_val, proba_val, beta: float = 2.0) -> tuple[float, float]:
    thresholds = np.arange(0.10, 0.91, 0.01)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        y_pred = (proba_val >= t).astype(int)
        score = fbeta_score(y_val, y_pred, beta=beta, pos_label=1, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, best_score


# ---------- MÉTRICAS ----------

def metrics(name, y_true, y_pred, y_proba, threshold: float | None = None):
    p = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    r = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n--- {name} ---")
    if threshold is not None:
        print(f"Umbral             : {threshold:.2f}")
    print(f"Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}  F2: {f2:.4f}  AUC: {auc:.4f}  Acc: {acc:.4f}")
    print(f"CM [[TN FP][FN TP]]: {cm.tolist()}")
    return {"Modelo": name, "Precision": p, "Recall": r, "F1": f1, "F2": f2,
            "AUC_ROC": auc, "Accuracy": acc, "Threshold": threshold}


# ---------- MAIN ----------

def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    print("=" * 70)
    print("PASO 1 — Feature engineering + split v2")
    print("=" * 70)
    X_train, X_test, y_train, y_test = make_v2_splits()
    print(f"X_train_v2: {X_train.shape}    X_test_v2: {X_test.shape}")
    print(f"Features v2 ({X_train.shape[1]}): {list(X_train.columns)}")

    print("\n" + "=" * 70)
    print("PASO 2 — Reentrenamiento clásicos sobre v2 (umbral default 0.5)")
    print("=" * 70)
    classical = train_classical(X_train, y_train)

    results_v2 = []
    test_probas = {}
    for name, model in classical.items():
        joblib.dump(model, OUTPUTS_DIR / f"model_{name.lower()}_v2.pkl")
        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        test_probas[name] = proba
        results_v2.append(metrics(f"{name} v2", y_test, pred, proba, threshold=0.5))

    print("\n" + "=" * 70)
    print("PASO 2 bis — MLP v2 con umbral F2")
    print("=" * 70)
    mlp_model, val_probs, val_targets = train_mlp(X_train, y_train)
    best_t_mlp, best_f2_val = find_best_threshold_fbeta(val_targets, val_probs, beta=2.0)
    print(f"[v2 MLP] Umbral óptimo por F2 (val): {best_t_mlp:.2f}   F2 val={best_f2_val:.4f}")

    mlp_test_proba = predict_mlp(mlp_model, X_test)
    test_probas["MLP"] = mlp_test_proba
    mlp_test_pred = (mlp_test_proba >= best_t_mlp).astype(int)
    results_v2.append(metrics("MLP v2", y_test, mlp_test_pred, mlp_test_proba, threshold=best_t_mlp))

    torch.save({"model_state_dict": mlp_model.state_dict(),
                "n_features": X_train.shape[1],
                "best_threshold": best_t_mlp},
               OUTPUTS_DIR / "model_mlp_v2.pt")

    print("\n" + "=" * 70)
    print("PASO 3 — Ensemble suave (MLP + XGBoost + SVM)")
    print("=" * 70)
    ens_test_proba = (test_probas["MLP"] + test_probas["XGBoost"] + test_probas["SVM"]) / 3.0

    # Umbral F2 para el ensemble: lo derivamos de un split val nuevo aplicando los 3 modelos
    X_tr_inner, X_val_inner, y_tr_inner, y_val_inner = train_test_split(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE,
    )
    val_proba_xgb = classical["XGBoost"].predict_proba(X_val_inner)[:, 1]
    val_proba_svm = classical["SVM"].predict_proba(X_val_inner)[:, 1]
    val_proba_mlp = predict_mlp(mlp_model, X_val_inner)
    ens_val_proba = (val_proba_mlp + val_proba_xgb + val_proba_svm) / 3.0
    best_t_ens, best_f2_ens = find_best_threshold_fbeta(y_val_inner.values, ens_val_proba, beta=2.0)
    print(f"[Ensemble] Umbral óptimo por F2 (val): {best_t_ens:.2f}   F2 val={best_f2_ens:.4f}")

    ens_test_pred = (ens_test_proba >= best_t_ens).astype(int)
    results_v2.append(metrics("Ensemble v2 (MLP+XGB+SVM)", y_test, ens_test_pred, ens_test_proba,
                              threshold=best_t_ens))

    # ---------- TABLA COMPARATIVA v1 vs v2 ----------
    print("\n" + "=" * 70)
    print("COMPARATIVA v1 vs v2")
    print("=" * 70)
    df_v1 = pd.read_csv(OUTPUTS_DIR / "resultados_ml.csv")
    df_v2 = pd.DataFrame(results_v2)

    cols_print = ["Modelo", "Precision", "Recall", "F1", "AUC_ROC"]
    # Añadir F2 retrospectivamente a v1 no es posible sin las probas; lo dejamos en blanco
    df_v1_display = df_v1[cols_print].copy()
    df_v1_display["F2"] = np.nan
    df_v1_display["Versión"] = "v1"

    df_v2_display = df_v2[["Modelo", "Precision", "Recall", "F1", "AUC_ROC", "F2"]].copy()
    df_v2_display["Versión"] = "v2"

    cmp = pd.concat([df_v1_display, df_v2_display], ignore_index=True)
    cmp = cmp[["Versión", "Modelo", "Precision", "Recall", "F1", "F2", "AUC_ROC"]]
    print(cmp.to_string(index=False))

    df_v2.to_csv(OUTPUTS_DIR / "resultados_v2.csv", index=False)
    print(f"\nResultados v2 guardados en: {OUTPUTS_DIR / 'resultados_v2.csv'}")


if __name__ == "__main__":
    main()
