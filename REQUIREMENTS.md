# REQUIREMENTS.md — Diabetes Pattern Recognition System (FIXED)

---

## MATLAB Version

| Requirement | Specification |
|-------------|--------------|
| Minimum | MATLAB R2020b (9.9) |
| Recommended | MATLAB R2021a+ (for native `shapley()` in Phase 5) |

---

## Required Toolboxes

### 1. Statistics and Machine Learning Toolbox — MANDATORY

Used for: `fitcknn`, `fitcnb`, `fitcsvm`, `fitglm`, `TreeBagger`, `cvpartition`, `perfcurve`, `tsne`, `zscore`, `shapley`

```matlab
% Verify:
fitcknn([1;2;3;4], [0;0;1;1], 'NumNeighbors', 1);
```

### 2. Bioinformatics Toolbox — OPTIONAL (auto-fallback)

Used for: `knnimpute` in Phase 1.

If not installed, the system **automatically** uses a manual k-NN imputation implementation. No action needed.

### 3. Global Optimization Toolbox — OPTIONAL (auto-fallback)

Used for: `ga()` in Phase 3.

If not installed, the system **automatically** switches to `Phase3_GA_Fallback.m` (custom GA implementation). No action needed.

---

## Dataset

| Property | Value |
|----------|-------|
| File | `data/diabetes.csv` |
| Source | Pima Indians Diabetes Database (UCI / Kaggle) |
| Rows | 768 |
| Columns | 9 (8 features + 1 label) |

**Columns**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome

If missing: run `generate_sample_data` for a synthetic version, or download from https://www.kaggle.com/uciml/pima-indians-diabetes-database

---

## Hardware

- RAM: 8 GB minimum (16 GB recommended for SHAP)
- Display: 1920×1080 for App Designer GUI

---

## Quick Verification

```matlab
cd('path/to/Diabetes_PR_Project')
setup    % Runs all checks automatically
```

The setup script verifies all toolboxes, functions, dataset, and directories. It clearly reports what is available and what will use fallbacks.

---

## Auto-Fallback Summary

| Component | Primary | Fallback (automatic) |
|-----------|---------|---------------------|
| k-NN Imputation | `knnimpute` (Bioinformatics Toolbox) | `manual_knn_impute` (built-in to Phase 1) |
| Genetic Algorithm | `ga()` (Global Optimization Toolbox) | `Phase3_GA_Fallback.m` (custom implementation) |
| SHAP Values | `shapley()` (R2021a+) | `computePermutationSHAP` (built-in to Phase 5) |

All fallbacks activate automatically via try-catch — no manual code changes needed.

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `Undefined function 'fitcknn'` | Install Statistics and Machine Learning Toolbox (MANDATORY) |
| `Dataset file not found` | Run `generate_sample_data` or download CSV from Kaggle |
| `Out of memory during SHAP` | Reduce `n_shap_samples` in Phase5 (line: `n_shap_samples = min(50, ...)`) |
| App Designer crashes | Run `main_pipeline` first to generate `results/trained_system.mat` |
| Figures not saving | Run `setup` first to create `results/` directory |
