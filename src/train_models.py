"""Fase 3 — Entrenamiento y evaluación de modelos ML clásicos."""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

RANDOM_STATE = 42
SCALE_POS_WEIGHT = 40357 / 9644  # ≈ 4.18 (neg/pos en dataset completo)


def load_splits():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def build_models() -> dict:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            scale_pos_weight=SCALE_POS_WEIGHT,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- {name} ---")
    print(f"Precision (cancer=1): {precision:.4f}")
    print(f"Recall    (cancer=1): {recall:.4f}")
    print(f"F1        (cancer=1): {f1:.4f}")
    print(f"AUC-ROC            : {auc:.4f}")
    print(f"Accuracy           : {acc:.4f}")
    print("Matriz de confusión [[TN FP] [FN TP]]:")
    print(cm)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return {
        "Modelo": name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC_ROC": auc,
        "Accuracy": acc,
        "fpr": fpr,
        "tpr": tpr,
    }


def plot_roc(results: list[dict], path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for r in results:
        plt.plot(r["fpr"], r["tpr"], label=f"{r['Modelo']} (AUC={r['AUC_ROC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Azar")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curvas ROC — Modelos ML clásicos")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = load_splits()
    print(f"Train: {X_train.shape}    Test: {X_test.shape}")
    print(f"scale_pos_weight XGBoost = {SCALE_POS_WEIGHT:.4f}")

    models = build_models()
    results = []
    for name, model in models.items():
        print(f"\nEntrenando {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, OUTPUTS_DIR / f"model_{name.lower()}.pkl")
        results.append(evaluate(name, model, X_test, y_test))

    df = pd.DataFrame([
        {k: r[k] for k in ["Modelo", "Precision", "Recall", "F1", "AUC_ROC", "Accuracy"]}
        for r in results
    ])
    df.to_csv(OUTPUTS_DIR / "resultados_ml.csv", index=False)

    plot_roc(results, OUTPUTS_DIR / "curvas_roc_ml.png")

    print("\n" + "=" * 60)
    print("RESUMEN COMPARATIVO")
    print("=" * 60)
    print(df.to_string(index=False))

    best_f1 = df.loc[df["F1"].idxmax(), "Modelo"]
    best_auc = df.loc[df["AUC_ROC"].idxmax(), "Modelo"]
    print(f"\nMejor F1 (cancer=1): {best_f1}")
    print(f"Mejor AUC-ROC:       {best_auc}")
    print(f"\nGuardado en: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
