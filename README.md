# Cancer Diagnosis — ML vs. Neural Network Pipeline

End-to-end machine-learning pipeline that evaluates whether multi-source clinical
data is enough to anticipate a cancer diagnosis, comparing three classical ML
models against a multilayer neural network (MLP). Built as an academic case study
for the *Artificial Intelligence* course (B.Sc. in Mathematical Engineering, UAX).

> **Note on the data:** the dataset is **synthetic** (50,001 records generated with
> realistic epidemiological criteria for the course). It contains no real patient
> data, and the raw CSVs are not redistributed here. The schema is documented in
> [`data/metadata.md`](metadata.md).

## Problem

A hospital wants to know if its data can flag likely cancer cases as a **first-line
screening filter**. Because a missed cancer (false negative) is far costlier than a
false alarm, the priority metric is **recall on the positive class**, not accuracy.
The target is imbalanced: only **19.3%** of patients are positive.

## Approach

- **Data:** 6 source tables (biochemical, genetic, clinical, lifestyle, economic,
  sociodemographic) joined by `paciente_id` → 50,001 records.
- **Feature selection with justification:** kept causal predictors (blood markers,
  oncogenic mutations, smoking, physical activity, age) and **deliberately excluded
  leakage variables** — economic/outcome fields (cost, hospital days, survival) and
  comorbidities that correlate with the target by construction. This step is the
  backbone of the study.
- **Preprocessing:** categorical encoding, `StandardScaler`, stratified 80/20 split.
- **Models:** Random Forest, XGBoost, SVM, and an **MLP built in PyTorch**.
- **Imbalance handling & threshold tuning:** class weighting, and the decision
  threshold chosen on the **validation** set (0.57) — never on test — to avoid data
  leakage.

## Neural network architecture

`21 inputs → 128 → 64 → 32 → 1 (sigmoid)`, ~46,913 parameters, with
**BatchNorm + Dropout** on each hidden layer, **Early Stopping** and
**ReduceLROnPlateau**.

## Results (test set, positive class)

| Model          | Precision | Recall   | F1   | AUC-ROC |
|----------------|:---------:|:--------:|:----:|:-------:|
| Random Forest  | 0.63      | 0.24     | 0.35 | 0.80    |
| XGBoost        | 0.44      | 0.56     | 0.49 | 0.77    |
| SVM            | 0.41      | 0.70     | 0.52 | 0.80    |
| **MLP (chosen)** | 0.34    | **0.84** | 0.48 | 0.82    |

The **MLP** was selected because it detects **84% of real cancer cases** — far above
the classical models — which is what matters in a screening context. The trade-off
is lower precision (more false alarms), acceptable here because no diagnosis is made
on the model alone: flagged patients go to a confirmatory test.

Plots are in [`outputs/`](outputs/): model comparison, ROC curves, precision-recall,
MLP loss/accuracy curves and the confusion matrix.

## Repository structure

```
├── src/                  # pipeline code
│   ├── load_data.py      # load + merge the 6 sources, feature selection
│   ├── preprocess.py     # encoding, scaling, stratified split
│   ├── train_models.py   # classical models (RF, XGBoost, SVM)
│   ├── improve_models.py # tuned versions of the classical models
│   ├── train_mlp.py      # PyTorch MLP + threshold tuning
│   └── compare_models.py # final comparison, metrics and plots
├── outputs/              # result plots + summary table (CSV)
├── slides/               # 5-slide viability study (PDF)
└── data/                 # dataset schema (synthetic data, not included)
```

## Tech stack

Python · PyTorch · scikit-learn · XGBoost · pandas · NumPy · Matplotlib

## How to run

```bash
pip install -r requirements.txt
# place the course CSVs in data/  (see data/metadata.md for the schema)
python src/load_data.py
python src/preprocess.py
python src/train_models.py
python src/train_mlp.py
python src/compare_models.py
```

---
*Academic project · B.Sc. in Mathematical Engineering, Universidad Alfonso X el Sabio (UAX) · 2025–2026.*
