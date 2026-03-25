%% =========================================================================
%  ADVANCED PATTERN RECOGNITION SYSTEM FOR DIABETES DIAGNOSIS
%  =========================================================================
%  Project: Intelligent Diagnostic Platform for Type 2 Diabetes
%  Method:  Multi-dimensional Feature Space Analysis with Weighted Ensemble
%
%  Mathematical Foundation:
%  -----------------------
%  This system operates on the principle that disease states create
%  distinguishable geometric structures in high-dimensional feature space.
%  By combining Mahalanobis distance (which accounts for covariance
%  structure) with probabilistic reasoning (Naive Bayes) and evolutionary
%  feature selection (Genetic Algorithm), we achieve robust pattern
%  discrimination that mimics expert clinical reasoning.
%
%  References:
%  [1] Smith, J.W., et al. (1988). "Using the ADAP Learning Algorithm to
%      Forecast the Onset of Diabetes Mellitus." SCAMC Proceedings.
%  [2] De Maesschalck, R., et al. (2000). "The Mahalanobis Distance."
%      Chemometrics and Intelligent Laboratory Systems, 50(1), 1-18.
%  [3] Goldberg, D.E. (1989). "Genetic Algorithms in Search, Optimization
%      and Machine Learning." Addison-Wesley.
%  [4] Lundberg, S.M. & Lee, S.I. (2017). "A Unified Approach to
%      Interpreting Model Predictions." NeurIPS.
%
%  Dataset: Pima Indians Diabetes Database (UCI Repository)
%  Author:  Advanced PR Research Team
%  Date:    2025
%  =========================================================================

clc; clear; close all;
fprintf('=============================================================\n');
fprintf('  ADVANCED PATTERN RECOGNITION SYSTEM - DIABETES DIAGNOSIS\n');
fprintf('  Multi-Dimensional Biomarker Pattern Analysis Platform\n');
fprintf('=============================================================\n\n');

%% ===================== CONFIGURATION =====================
% All hyperparameters are centralized here for reproducibility
config = struct();
config.dataFile         = 'data/diabetes.csv';
config.knn_k            = 5;              % k-NN neighborhood size
config.knn_distance     = 'mahalanobis';  % Distance metric for classification
config.ga_popSize       = 50;             % GA population size
config.ga_maxGen        = 100;            % GA maximum generations
config.ga_crossFrac     = 0.8;            % GA crossover fraction
config.cv_folds         = 10;             % Cross-validation folds
config.knn_impute_k     = 5;              % k-NN neighbors for imputation
config.random_seed      = 42;             % Fixed seed for reproducibility
config.test_ratio       = 0.2;            % Fraction of data held out for testing
config.val_ratio        = 0.125;          % Fraction of train+val held out for validation
                                          % (0.125 * 80% = 10% of total dataset)
                                          % Final split: 70% train, 10% val, 20% test

% Feature names for interpretability and axis labelling
config.featureNames = {'Pregnancies', 'Glucose', 'BloodPressure', ...
                       'SkinThickness', 'Insulin', 'BMI', ...
                       'DiabetesPedigree', 'Age'};

% Columns where a value of zero is biologically impossible and indicates
% a missing measurement rather than a true observation
config.zeroIsMissing = [2, 3, 4, 5, 6]; % Glucose, BP, SkinThickness, Insulin, BMI

% Initialize the random number generator for a fully reproducible run
rng(config.random_seed);

fprintf('[CONFIG] Random seed: %d\n', config.random_seed);
fprintf('[CONFIG] Cross-validation folds: %d\n', config.cv_folds);
fprintf('[CONFIG] Distance metric: %s\n', config.knn_distance);
fprintf('[CONFIG] Data split: 70%% train / 10%% validation / 20%% test\n\n');

%% ===================== PHASE 1: DATA PRE-PROCESSING =====================
fprintf('=============================================================\n');
fprintf('  PHASE 1: Data Pre-processing & Feature Space Preparation\n');
fprintf('=============================================================\n\n');

% Phase 1 returns leakage-free, normalized partitions. Imputation and
% normalization statistics are derived from the training set only and
% stored in imputation_stats for use by Phase 6.
[X_train, X_val, X_test, y_train, y_val, y_test, X_normalized, y_all, ...
 X_raw, X_imputed, imputation_stats] = Phase1_DataPreprocessing(config);

%% ===================== PHASE 2: EXPLORATORY ANALYSIS =====================
fprintf('\n=============================================================\n');
fprintf('  PHASE 2: Exploratory Pattern Analysis (Geometric View)\n');
fprintf('=============================================================\n\n');

% Exploratory analysis operates on the training partition so that class
% geometry, PCA projections, and distribution plots reflect only the data
% that the models are permitted to observe during learning.
Phase2_ExploratoryAnalysis(X_train, y_train, config);

%% ===================== PHASE 3: GENETIC ALGORITHM =====================
fprintf('\n=============================================================\n');
fprintf('  PHASE 3: Genetic Algorithm Feature Selection\n');
fprintf('=============================================================\n\n');

% Attempt the toolbox-based GA implementation; if the Global Optimization
% Toolbox is unavailable, fall back to the self-contained custom GA.
try
    [selected_features, ga_results] = Phase3_GeneticAlgorithm(...
        X_train, y_train, config);
catch ME
    fprintf('  [WARNING] Phase3_GeneticAlgorithm failed: %s\n', ME.message);
    fprintf('  [WARNING] Switching to custom GA fallback (no toolbox required)...\n\n');
    [selected_features, ga_results] = Phase3_GA_Fallback(...
        X_train, y_train, config);
end

%% ===================== PHASE 4: CLASSIFICATION ENGINE =====================
fprintf('\n=============================================================\n');
fprintf('  PHASE 4: Classification Engine (Weighted Ensemble)\n');
fprintf('=============================================================\n\n');

[ensemble_results, knn_model, nb_model, svm_model, rf_model_cls, lr_model] = ...
    Phase4_ClassificationEngine(...
    X_train, y_train, X_val, y_val, X_test, y_test, selected_features, config);

%% ===================== PHASE 5: SHAP ANALYSIS =====================
fprintf('\n=============================================================\n');
fprintf('  PHASE 5: Explainability Analysis (SHAP Values)\n');
fprintf('=============================================================\n\n');

shap_results = Phase5_SHAPAnalysis(X_train, y_train, X_test, y_test, ...
    selected_features, config);

%% ===================== FINAL REPORT =====================
fprintf('\n=============================================================\n');
fprintf('  FINAL SYSTEM REPORT\n');
fprintf('=============================================================\n\n');

fprintf('--- Data Summary ---\n');
fprintf('  Total samples:      %d\n', size(X_raw, 1));
fprintf('  Training samples:   %d\n', size(X_train, 1));
fprintf('  Validation samples: %d\n', size(X_val, 1));
fprintf('  Testing samples:    %d\n', size(X_test, 1));
fprintf('  Original features:  %d\n', size(X_raw, 2));
fprintf('  Selected features:  %d ', length(selected_features));
fprintf('(%s)\n', strjoin(config.featureNames(selected_features), ', '));

fprintf('\n--- Ensemble Performance (Test Set) ---\n');
fprintf('  Accuracy:    %.2f%%\n', ensemble_results.accuracy * 100);
fprintf('  Sensitivity: %.2f%%\n', ensemble_results.sensitivity * 100);
fprintf('  Specificity: %.2f%%\n', ensemble_results.specificity * 100);
fprintf('  F1-Score:    %.4f\n',   ensemble_results.f1_score);
fprintf('  AUC:         %.4f\n',   ensemble_results.auc);

fprintf('\n--- Individual Model Performance (Test Set) ---\n');
fprintf('  k-NN (Mahalanobis) Accuracy:  %.2f%%\n', ...
    ensemble_results.knn_accuracy * 100);
fprintf('  Naive Bayes Accuracy:         %.2f%%\n', ...
    ensemble_results.nb_accuracy * 100);
fprintf('  SVM (RBF) Accuracy:           %.2f%%\n', ...
    ensemble_results.svm_accuracy * 100);
fprintf('  Random Forest Accuracy:       %.2f%%\n', ...
    ensemble_results.rf_accuracy * 100);
fprintf('  Logistic Regression Accuracy: %.2f%%\n', ...
    ensemble_results.lr_accuracy * 100);

fprintf('\n=============================================================\n');
fprintf('  All figures saved to results/ directory\n');
fprintf('  To launch interactive demo: run Phase6_AppDesigner\n');
fprintf('=============================================================\n');

%% Save Workspace for the App Designer
% The .mat file bundles all trained models, normalization parameters,
% raw and imputed data, and SHAP results so that Phase 6 can normalize
% new patient inputs, compute population percentiles, and run inference
% without re-executing the full pipeline.
save('results/trained_system.mat', ...
     'knn_model', 'nb_model', 'svm_model', 'rf_model_cls', 'lr_model', ...
     'selected_features', 'config', 'ensemble_results', ...
     'X_train', 'y_train', 'X_val', 'y_val', ...
     'X_raw', 'X_imputed', 'y_all', ...
     'imputation_stats', 'shap_results');
fprintf('\n[SAVED] Trained system saved to results/trained_system.mat\n');
fprintf('  -> Includes: raw data, normalization params (mu, sigma),\n');
fprintf('     imputed data, all 5 trained models, and SHAP results.\n');
