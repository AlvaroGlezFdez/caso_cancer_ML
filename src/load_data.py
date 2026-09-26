"""Carga, merge y selección de features para el proyecto Cancer ML."""
from pathlib import Path
from functools import reduce
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CSV_FILES = [
    "CASOCANCER_01_BIOQUIMICOS.csv",
    "CASOCANCER_02_CLINICOS.csv",
    "CASOCANCER_03_GENETICOS.csv",
    "CASOCANCER_04_ECONOMICOS.csv",
    "CASOCANCER_05_GENERALES.csv",
    "CASOCANCER_06_SOCIODEMOGRAFICOS.csv",
]

FEATURES = [
    "glucosa", "colesterol", "trigliceridos", "hemoglobina",
    "leucocitos", "plaquetas", "creatinina",
    "mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS",
    "mut_PIK3CA", "mut_ALK", "mut_BRAF",
    "fumador", "actividad_fisica",
    "edad",
]
TARGET = "cancer"


def load_and_merge() -> pd.DataFrame:
    dfs = [pd.read_csv(DATA_DIR / f) for f in CSV_FILES]
    merged = reduce(
        lambda left, right: pd.merge(left, right, on="paciente_id", how="left"),
        dfs,
    )
    cols = ["paciente_id"] + FEATURES + [TARGET]
    return merged[cols]


def summarize(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("RESUMEN DEL DATASET MERGED")
    print("=" * 60)
    print(f"\nShape final: {df.shape}")

    print("\n% de nulos por columna:")
    nulls_pct = (df.isna().mean() * 100).round(2)
    for col, pct in nulls_pct.items():
        print(f"  {col:<25s} {pct:>6.2f}%")

    print(f"\nDistribución del target ({TARGET}):")
    counts = df[TARGET].value_counts().sort_index()
    total = len(df)
    for val, count in counts.items():
        print(f"  {TARGET}={val}: {count} ({count/total*100:.2f}%)")
    print("=" * 60)


def main() -> pd.DataFrame:
    df = load_and_merge()
    summarize(df)
    out_path = DATA_DIR / "dataset_merged.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDataset guardado en: {out_path}")
    return df


if __name__ == "__main__":
    main()
