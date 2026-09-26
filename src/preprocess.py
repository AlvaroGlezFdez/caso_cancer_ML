"""Preprocesado: codificación, escalado y split train/test estratificado."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

CONTINUOUS = [
    "glucosa", "colesterol", "trigliceridos", "hemoglobina",
    "leucocitos", "plaquetas", "creatinina", "edad",
]
BINARY_ORDINAL = [
    "mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS",
    "mut_PIK3CA", "mut_ALK", "mut_BRAF",
    "fumador", "actividad_fisica",
]
ACTIVIDAD_MAP = {"Baja": 0, "Moderada": 1, "Alta": 2}
TARGET = "cancer"
RANDOM_STATE = 42


def load() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "dataset_merged.csv")


def encode_actividad(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["actividad_fisica"] = df["actividad_fisica"].map(ACTIVIDAD_MAP)
    if df["actividad_fisica"].isna().any():
        bad = df.loc[df["actividad_fisica"].isna()]
        raise ValueError(
            f"Valores no mapeados en actividad_fisica: {bad.shape[0]} filas"
        )
    df["actividad_fisica"] = df["actividad_fisica"].astype(int)
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET]
    X = df.drop(columns=["paciente_id", TARGET])
    X = X[CONTINUOUS + BINARY_ORDINAL]
    return X, y


def scale_continuous(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[CONTINUOUS] = scaler.fit_transform(X_train[CONTINUOUS])
    X_test[CONTINUOUS] = scaler.transform(X_test[CONTINUOUS])
    return X_train, X_test, scaler


def summarize(X_train, X_test, y_train, y_test) -> None:
    print("=" * 60)
    print("RESUMEN DEL PREPROCESADO")
    print("=" * 60)
    print(f"\nX_train: {X_train.shape}    y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}    y_test:  {y_test.shape}")

    def dist(name, y):
        counts = y.value_counts().sort_index()
        total = len(y)
        print(f"\nDistribución target en {name}:")
        for v, c in counts.items():
            print(f"  cancer={v}: {c} ({c/total*100:.2f}%)")

    dist("train", y_train)
    dist("test", y_test)
    print("=" * 60)


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    df = load()
    df = encode_actividad(df)
    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train, X_test, scaler = scale_continuous(X_train, X_test)

    X_train.to_csv(DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(DATA_DIR / "y_test.csv", index=False)
    joblib.dump(scaler, OUTPUTS_DIR / "scaler.pkl")

    summarize(X_train, X_test, y_train, y_test)
    print(f"\nArchivos guardados en: {DATA_DIR}")
    print(f"Scaler guardado en:    {OUTPUTS_DIR / 'scaler.pkl'}")


if __name__ == "__main__":
    main()
