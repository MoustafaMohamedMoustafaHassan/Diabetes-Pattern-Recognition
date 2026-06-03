# Beyond the Black Box: A Leakage-Proof, Explainable Pattern Recognition Framework
**Clinical Diabetes Prediction System**


---


---

## 👥 Meet the Auther
* **Moustafa Yehia Hassan** *(Project Lead)*: Designed and coded the overall system architecture, including the Zero-Leakage Firewall, Geometric Analysis, Genetic Algorithm, Weighted Ensemble Classification, SHAP Explainability integration, and the Interactive GUI.
* **Shasikumara Rajasegeran**: Proposed the initial medical domain (Diabetes), drafted introductory concepts, and assisted with GUI testing and data leakdge proof.
* **Kirrthana Shanmugavelu**: Contributed to the baseline literature review and initial dataset statistics review.

---

## 🧠 Project Philosophy & Core Objectives
This is not just another machine learning classification script. It is a **Clinical Triage System** designed to solve three major problems in medical AI:

1. **Data Leakage:** We built a strict "Zero-Leakage Firewall." Test and validation data are never used to fill in missing values (imputation) or scale the data (normalization). Everything is strictly learned from the training set only.
2. **The Black-Box Problem:** Doctors need to know *why* a decision was made. Our system uses **SHAP (SHapley Additive exPlanations)** to provide a clear, mathematical reason for every prediction.
3. **Model Fragility:** Instead of relying on one algorithm, we built a **Weighted Ensemble** that combines 5 different models. If one model has a weak spot, the others cover it.

---

## ⚙️ System Architecture (The 6 Phases)
The project runs automatically through a 6-phase pipeline:

### Phase 1: Strict Data Pre-processing
* Filters out biologically impossible data (like blood glucose = 0).
* Splits the data strictly into 70% Training / 10% Validation / 20% Testing.
* Applies **k-NN Imputation** and Z-score normalization *only* using training data to prevent leakage.

### Phase 2: Exploratory Geometric Analysis
* Analyzes the mathematical shape of the data using Covariance Matrices and t-SNE.
* Proves why using **Mahalanobis Distance** is better than standard Euclidean distance for this specific medical data.

### Phase 3: Genetic Algorithm (GA) Feature Selection
* Uses an evolutionary algorithm to find the best combination of biomarkers.
* Successfully reduced the required medical features from 8 down to 5 (a 37.5% reduction) while keeping the system highly accurate.

### Phase 4: Multi-Model Ensemble Engine
Combines 5 powerful algorithms using a "Weighted Soft-Voting" system:
1. **k-NN** (using Mahalanobis distance)
2. **Naive Bayes** (for stable probabilities)
3. **SVM** (with RBF Kernel for complex patterns)
4. **Random Forest** (100 trees)
5. **Logistic Regression** (for a clear linear baseline)

### Phase 5: SHAP Explainable AI (XAI)
* Analyzes the final model to show the most important medical features globally.
* Generates individual "Waterfall plots" to explain specific patient cases (e.g., why a specific healthy patient was flagged as high-risk).

### Phase 6: Interactive Clinical App (GUI)
* A fully working interface built with MATLAB App Designer.
* Allows doctors to input new patient data, instantly see the risk assessment, and view the feature explanations in real-time.

---

## 🚀 Quick Start Guide (How to Run)

The code is engineered to run smoothly and handle missing MATLAB toolboxes automatically (Fallback Mechanisms). Just open MATLAB, navigate to the project folder, and run:

```matlab
% 1. Check your environment and set up folders (Run this first!)
setup              

% 2. Run the complete 6-phase AI pipeline (Takes 5-10 minutes)
main_pipeline      

% 3. Launch the Interactive Clinical App for testing
Phase6_AppDesigner 
```
*Note: If your MATLAB is missing the `Bioinformatics` or `Global Optimization` toolboxes, the `setup` script will automatically switch to custom-built fallback codes so the program never crashes.*

---

## 📊 Key Results
* **Ensemble Test Accuracy:** ~75.8% (We sacrificed artificial "high accuracy" to ensure zero data leakage and real-world reliability).
* **AUC-ROC (Discrimination Power):** **0.842** (Excellent ability to separate healthy from diabetic patients).
* **Specificity:** **89.0%** (The system is very good at correctly clearing healthy people, which is perfect for reducing false alarms in hospitals).
* **Top Medical Features (via SHAP):** Glucose, BMI, and Age.

---

## 📋 Assignment Compliance Matrix

| Requirement                    | Status | Implementation Details                                                   |
|--------------------------------|--------|--------------------------------------------------------------------------|
| **1.2 Goal**                   | ✅    | Built a complete Type 2 Diabetes classification system.                   |
| **1.3 Data Split**             | ✅    | Strict stratified split: 70% Train / 10% Validation / 20% Test.           |
| **1.4 Pre-processing**         | ✅    | Zero-leakage k-NN imputation + Z-score scaling.                           |
| **1.5 Feature Extraction**     | ✅    | Genetic Algorithm + Covariance analysis + t-SNE.                          |
| **1.6 Model Selection**        | ✅    | Integrated 5 models (k-NN, NB, SVM, RF, LR).                              |
| **1.7 Training**               | ✅    | Cross-validation used, with ensemble weights tuned on the validation set. |
| **1.8 Evaluation**             | ✅    | Computed Accuracy, Sensitivity, Specificity, Precision, F1, and AUC.      |
| **1.9 Deployment**             | ✅    | Built an interactive GUI using MATLAB App Designer.                       |
| **1.10 Optimization**          | ✅    | Optimized features via GA and tuned hyperparameters (e.g., k in k-NN).    |
| **1.11 Documentation**         | ✅    | Comprehensive A+ level academic report included.                          |
| **2.0 Demo**                   | ✅    | The `Phase6_AppDesigner` application is ready for the live presentation.  |

---

## 🛠️ Prerequisites & System Requirements
To run this project smoothly, you will need:
* **MATLAB**: Version R2021a or newer is recommended.
* **Required Toolboxes** (Don't worry, the code will auto-switch to custom fallbacks if you don't have them):
  * Statistics and Machine Learning Toolbox
  * Bioinformatics Toolbox
  * Global Optimization Toolbox

## 📁 Folder Structure
Here is how the project files are organized:

```text
Diabetes_PR_Project/
│
├── data/
│   └── diabetes.csv                # The original Pima Indians Diabetes dataset
│
├── src/                            # Main source code files
│   ├── setup.m                     # Environment setup and toolbox checker
│   ├── main_pipeline.m             # The main script that runs Phases 1 to 5
│   ├── Phase1_Preprocessing.m      # Data cleaning, k-NN imputation, Z-score scaling
│   ├── Phase2_GeometricAnalysis.m  # Covariance and t-SNE analysis
│   ├── Phase3_GeneticAlgorithm.m   # Feature selection using GA
│   ├── Phase3_GA_Fallback.m        # Custom GA (runs automatically if toolbox is missing)
│   ├── Phase4_Classification.m     # 5-Model Ensemble training and evaluation
│   ├── Phase5_Explainability.m     # SHAP values calculation and plotting
│   └── Phase6_AppDesigner.m        # The Interactive Clinical GUI
│
├── results/                        # Generated outputs (Auto-created by setup.m)
│   ├── figures/                    # Saved plots and charts (ROC, SHAP, etc.)
│   └── trained_system.mat          # Saved model and parameters used by the GUI
│
└── README.md                       # This documentation file
```

## 💻 Installation & Setup
1. **Download the project**: Clone this repository or download it as a ZIP file and extract it to your computer.
2. **Open MATLAB**: Launch your MATLAB application.
3. **Set Current Folder**: In MATLAB, navigate to the extracted project folder so it becomes your active "Current Folder".

## ▶️ Step-by-Step Execution (How to Run)
Follow these simple steps in the MATLAB Command Window to run the project:

**Step 1: Initialize the Environment**
```matlab
% This creates necessary folders and checks if you have the required toolboxes.
setup
```

**Step 2: Train the AI Models (Phases 1 to 5)**
```matlab
% This runs the entire data processing, training, testing, and SHAP pipeline.
% It will automatically save the trained system and generate all analysis graphs.
main_pipeline
```
*Wait for the script to finish (it may take 5 to 10 minutes depending on your PC). It will print the accuracy and AUC results in the Command Window and save all visual charts in the `results/figures/` folder.*

**Step 3: Launch the Clinical App (Phase 6)**
```matlab
% This opens the interactive graphical interface.
Phase6_AppDesigner
```
*Once the app opens, you can type in patient biomarker numbers (like Glucose and BMI) and click "Predict Risk" to see the ensemble's decision and the SHAP explanation.*

## 🐛 Troubleshooting
* **Error: "File not found"**: Make sure you have set MATLAB's "Current Folder" to the main project directory before running any scripts.
* **Error: "Out of memory"**: If the Genetic Algorithm or SHAP analysis takes too much memory, try closing other heavy applications or restarting MATLAB.
* **Warning: "Toolbox missing"**: Do not worry. The `setup.m` script is designed to detect missing toolboxes and will automatically run our custom-built fallback functions without crashing.

---
*If you have any questions or face any issues running the code, please feel free to reach out or open an issue.*
```
