# REQUIREMENTS.md — Beyond the Black Box (Diabetes PR System)

This document outlines the software, hardware, and dataset requirements needed to successfully run the **Leakage-Proof, Explainable Pattern Recognition Framework for Clinical Diabetes Prediction**.

---

## 💻 1. Software Requirements

### MATLAB Version
| Requirement     | Specification                                                                        |
|-----------------|--------------------------------------------------------------------------------------|
| **Minimum**     | MATLAB R2020b (9.9)                                                                  |
| **Recommended** | MATLAB R2021a or newer (Required for the native `shapley()` XAI function in Phase 5) |

### Required MATLAB Toolboxes
The system is built with a **"Zero-Crash Fallback Architecture"**. However, for the optimal (Run B) experience, the following toolboxes are used:

#### 🔴 MANDATORY TOOLBOX
* **Statistics and Machine Learning Toolbox**
  * *Why we need it:* It provides the core algorithms for Phase 2 and Phase 4 (`fitcknn`, `fitcnb`, `fitcsvm`, `fitglm`, `TreeBagger`, `cvpartition`, `perfcurve`, `tsne`, `zscore`, `shapley`).
  * *How to verify:* Type `ver` in your MATLAB command window and look for it in the list.

#### 🟢 OPTIONAL TOOLBOXES (With Auto-Fallbacks)
* **Bioinformatics Toolbox**
  * *Why we need it:* Used for `knnimpute` in Phase 1 (Data Pre-processing).
  * *Fallback:* If not installed, the system automatically uses our custom `manual_knn_impute` function.
* **Global Optimization Toolbox**
  * *Why we need it:* Used for the Genetic Algorithm `ga()` in Phase 3 (Feature Selection).
  * *Fallback:* If not installed, the system automatically switches to `Phase3_GA_Fallback.m` (a custom-built genetic algorithm).

---

## 📊 2. Dataset Requirements

The system requires the **Pima Indians Diabetes Database (PIDD)** to run. 

| Property          | Value                                                |
|-------------------|------------------------------------------------------|
| **File Path**     | `data/diabetes.csv` (Must be exactly in this folder) |
| **Total Rows**    | 768 patients                                         |
| **Total Columns** | 9 (8 medical features + 1 outcome label)             |

**Required Exact Column Names:**
1. `Pregnancies`
2. `Glucose`
3. `BloodPressure`
4. `SkinThickness`
5. `Insulin`
6. `BMI`
7. `DiabetesPedigreeFunction`
8. `Age`
9. `Outcome` (0 = Healthy, 1 = Diabetic)

*Note: If you do not have the dataset, you can download the original CSV from the [UCI Machine Learning Repository or Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database).*

---

## 🖥️ 3. Hardware Requirements

* **RAM (Memory):** Minimum 8 GB. (16 GB is highly recommended for faster SHAP values computation in Phase 5).
* **Storage:** Less than 50 MB for code and data. (The `results/` folder will generate around 10-20 MB of plots and model files).
* **Display:** A resolution of 1920×1080 is recommended to properly view the Interactive Clinical App Designer GUI (Phase 6).

---

## ⚙️ 4. The Auto-Fallback System (Zero-Crash Guarantee)

To ensure that the project runs on any university or personal computer without crashing due to missing licenses, we implemented an automatic fallback mechanism.

| Pipeline Phase             | Primary Method (Requires Toolbox) | Automatic Fallback Method (No Toolbox) |
|----------------------------|-----------------------------------|-----------------------------------------------------------------|
| **Phase 1: Imputation**    | `knnimpute` (Bioinformatics)      | `manual_knn_impute` (Custom Euclidean distance k-NN)            |
| **Phase 3: Selection**     | `ga()` (Global Optimization)      | `Phase3_GA_Fallback.m` (Custom evolutionary algorithm)          |
| **Phase 5: Explainability**| `shapley()` (MATLAB R2021a+)      | `computePermutationSHAP` (Custom random permutation XAI)        |
| **Phase 4: Mahalanobis**   | Standard matrix inversion         | Regularized backslash operator (`\`) for near-singular matrices |

**All fallbacks activate automatically via `try-catch` blocks — you do not need to change any code.**

---

## 🚀 5. Quick Verification & Execution

To verify your system and run the project, execute these exact steps in the MATLAB Command Window:

```matlab
% Step 1: Navigate to the project folder
cd('path/to/Diabetes_PR_Project')

% Step 2: Run the setup script (Checks toolboxes, creates folders, and sets paths)
setup    

% Step 3: Run the full AI pipeline (Trains models, applies GA, generates SHAP)
% Note: This may take 5 to 10 minutes depending on your CPU.
main_pipeline

% Step 4: Open the Interactive Clinical App
Phase6_AppDesigner
````

-----

## 🛠️ 6. Troubleshooting Guide

If you encounter any issues, check the table below:

| Error / Issue                      | Cause & Solution                                                                                                                                                                         |
|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`Undefined function 'fitcknn'`** | **Cause:** Statistics and Machine Learning Toolbox is missing.<br>**Solution:** This toolbox is strictly mandatory. You must install it via the MATLAB Add-On Explorer.                  |
| **`Dataset file not found`**       | **Cause:** The `diabetes.csv` file is missing or in the wrong folder.<br>**Solution:** Ensure the file is placed exactly inside the `data/` folder.                                      |
| **`Out of memory` during Phase 5** | **Cause:** SHAP value computation is memory-intensive.<br>**Solution:** Open `Phase5_Explainability.m` and reduce the `n_shap_samples` limit from 50 to 20 or 10.                        |
| **App Designer GUI crashes**       | **Cause:** Missing the trained models file.<br>**Solution:** You must successfully run `main_pipeline.m` first to generate the `results/trained_system.mat` file before opening the app. |
| **Figures are not saving**         | **Cause:** Missing directories.<br>**Solution:** Always run `setup.m` first. It automatically creates the `results/` and `results/figures/` folders.                                     |

-----

*If you pass the `setup.m` check without any red errors, your system is 100% ready to run the project.*

`````