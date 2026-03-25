function [ensemble_results, knn_model, nb_model, svm_model, rf_model_cls, lr_model] = ...
    Phase4_ClassificationEngine(...
    X_train, y_train, X_val, y_val, X_test, y_test, selected_features, config)
%% =========================================================================
%  PHASE 4: CLASSIFICATION ENGINE - MULTI-MODEL WEIGHTED ENSEMBLE
%  =========================================================================
%  Mathematical Basis:
%  -------------------
%  Our ensemble combines five complementary classifiers:
%
%  1. k-NN with Mahalanobis Distance:
%     Decision rule: y_hat = argmax_c SUM_{x_j in N_k(x)} I(y_j = c)
%     Mahalanobis distance accounts for the COVARIANCE structure of data.
%
%  2. Naive Bayes (Gaussian):
%     Posterior: P(c|x) proportional to P(c) * PROD_i P(x_i|c)
%     Provides calibrated probability estimates for medical decisions.
%
%  3. Support Vector Machine (RBF Kernel):
%     Finds the maximum-margin hyperplane in a kernel-induced feature space.
%     The RBF kernel maps data to infinite-dimensional space, enabling
%     non-linear decision boundaries.
%
%  4. Random Forest:
%     Ensemble of decision trees trained on bootstrap samples with random
%     feature subsets. Captures non-linear interactions between features.
%
%  5. Logistic Regression:
%     Linear classifier: P(y=1|x) = sigma(w'x + b)
%     Provides well-calibrated probabilistic outputs and serves as
%     a strong interpretable baseline.
%
%  6. Weighted Voting Ensemble:
%     Final prediction = argmax_c SUM_m w_m * P_m(c|x)
%     Weights are proportional to validation set accuracy.
%
%  References:
%  [1] Cover, T.M. & Hart, P.E. (1967). "Nearest Neighbor Pattern 
%      Classification." IEEE IT, 13(1), 21-27.
%  [2] Hand, D.J. & Yu, K. (2001). "Idiot's Bayes - Not So Stupid 
%      After All?" International Statistical Review, 69(3), 385-398.
%  [3] Cortes, C. & Vapnik, V. (1995). "Support-vector networks."
%      Machine Learning, 20(3), 273-297.
%  [4] Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32.
%  [5] Kittler, J., et al. (1998). "On Combining Classifiers." IEEE 
%      TPAMI, 20(3), 226-239.
%  =========================================================================

fprintf('[PHASE 4] Building classification engine (5 models)...\n');

% Use only GA-selected features
X_train_sel = X_train(:, selected_features);
X_val_sel   = X_val(:, selected_features);
X_test_sel  = X_test(:, selected_features);

sel_names = config.featureNames(selected_features);
fprintf('  -> Using %d selected features: {%s}\n', ...
    length(selected_features), strjoin(sel_names, ', '));

%% Step 4.0: Hyperparameter Tuning on Validation Set
% -------------------------------------------------------------------------
% Use the validation set (not test set) to tune hyperparameters.
% This prevents information leakage from the test set.
% -------------------------------------------------------------------------
fprintf('\n[PHASE 4] Tuning hyperparameters on validation set...\n');

% Tune k for k-NN
k_values = [3, 5, 7, 9, 11, 15];
best_k = config.knn_k;
best_val_acc = 0;

for ki = 1:length(k_values)
    k_try = k_values(ki);
    try
        mdl_try = fitcknn(X_train_sel, y_train, ...
            'NumNeighbors', k_try, 'Distance', 'mahalanobis', ...
            'Standardize', false);
        pred_try = predict(mdl_try, X_val_sel);
        val_acc = mean(pred_try == y_val);
    catch
        % Mahalanobis might fail for some k, try with regularization
        X0 = X_train_sel(y_train == 0, :);
        X1 = X_train_sel(y_train == 1, :);
        n0 = size(X0, 1); n1 = size(X1, 1);
        Sigma_p = (n0 * cov(X0) + n1 * cov(X1)) / (n0 + n1);
        lam = 0.01 * trace(Sigma_p) / size(X_train_sel, 2);
        Sigma_r = Sigma_p + lam * eye(size(X_train_sel, 2));
        mdl_try = fitcknn(X_train_sel, y_train, ...
            'NumNeighbors', k_try, 'Distance', 'mahalanobis', ...
            'Cov', Sigma_r, 'Standardize', false);
        pred_try = predict(mdl_try, X_val_sel);
        val_acc = mean(pred_try == y_val);
    end
    fprintf('  -> k=%d: Val Accuracy = %.2f%%\n', k_try, val_acc*100);
    if val_acc > best_val_acc
        best_val_acc = val_acc;
        best_k = k_try;
    end
end

fprintf('  -> Best k = %d (Val Accuracy = %.2f%%)\n', best_k, best_val_acc*100);

%% Step 4.1: Train k-NN with Mahalanobis Distance
% -------------------------------------------------------------------------
fprintf('\n[PHASE 4] Training k-NN (k=%d, Distance=Mahalanobis)...\n', best_k);

try
    knn_model = fitcknn(X_train_sel, y_train, ...
        'NumNeighbors', best_k, ...
        'Distance', 'mahalanobis', ...
        'Standardize', false);
    [~] = predict(knn_model, X_test_sel(1,:));
    fprintf('  -> k-NN with Mahalanobis distance trained successfully.\n');
    distance_used = 'mahalanobis';
    
catch ME
    fprintf('  -> Direct Mahalanobis failed: %s\n', ME.message);
    fprintf('  -> Applying covariance regularization...\n');
    
    X0 = X_train_sel(y_train == 0, :);
    X1 = X_train_sel(y_train == 1, :);
    n0 = size(X0, 1); n1 = size(X1, 1);
    Sigma_pooled = (n0 * cov(X0) + n1 * cov(X1)) / (n0 + n1);
    lambda = 0.01 * trace(Sigma_pooled) / size(X_train_sel, 2);
    Sigma_reg = Sigma_pooled + lambda * eye(size(X_train_sel, 2));
    
    knn_model = fitcknn(X_train_sel, y_train, ...
        'NumNeighbors', best_k, ...
        'Distance', 'mahalanobis', ...
        'Cov', Sigma_reg, ...
        'Standardize', false);
    
    fprintf('  -> k-NN with regularized Mahalanobis trained successfully.\n');
    distance_used = 'mahalanobis (regularized)';
end

% k-NN Cross-Validation Performance
cv_knn = crossval(knn_model, 'KFold', config.cv_folds);
knn_cv_error = kfoldLoss(cv_knn);
knn_cv_accuracy = 1 - knn_cv_error;
fprintf('  -> k-NN %d-fold CV Accuracy: %.2f%%\n', config.cv_folds, knn_cv_accuracy*100);

% k-NN Test Set Predictions
[knn_pred, knn_scores] = predict(knn_model, X_test_sel);
knn_test_accuracy = mean(knn_pred == y_test);
fprintf('  -> k-NN Test Accuracy: %.2f%%\n', knn_test_accuracy*100);

%% Step 4.2: Train Naive Bayes
% -------------------------------------------------------------------------
fprintf('\n[PHASE 4] Training Naive Bayes (Gaussian)...\n');

nb_model = fitcnb(X_train_sel, y_train, ...
    'DistributionNames', 'normal', ...
    'Prior', 'empirical');

cv_nb = crossval(nb_model, 'KFold', config.cv_folds);
nb_cv_error = kfoldLoss(cv_nb);
nb_cv_accuracy = 1 - nb_cv_error;
fprintf('  -> NB %d-fold CV Accuracy: %.2f%%\n', config.cv_folds, nb_cv_accuracy*100);

[nb_pred, nb_scores] = predict(nb_model, X_test_sel);
nb_test_accuracy = mean(nb_pred == y_test);
fprintf('  -> NB Test Accuracy: %.2f%%\n', nb_test_accuracy*100);

%% Step 4.3: Train Support Vector Machine (RBF Kernel)
% -------------------------------------------------------------------------
% The SVM finds the maximum-margin hyperplane in kernel-induced space.
% RBF kernel: K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)
% We enable probability estimates via Platt scaling for ensemble fusion.
% -------------------------------------------------------------------------
fprintf('\n[PHASE 4] Training SVM (RBF Kernel)...\n');

svm_model = fitcsvm(X_train_sel, y_train, ...
    'KernelFunction', 'rbf', ...
    'Standardize', false, ...
    'ClassNames', [0, 1], ...
    'BoxConstraint', 1, ...
    'KernelScale', 'auto');

% Fit posterior probability model (Platt scaling) for score calibration
svm_model = fitPosterior(svm_model);

cv_svm = crossval(svm_model, 'KFold', config.cv_folds);
svm_cv_error = kfoldLoss(cv_svm);
svm_cv_accuracy = 1 - svm_cv_error;
fprintf('  -> SVM %d-fold CV Accuracy: %.2f%%\n', config.cv_folds, svm_cv_accuracy*100);

[svm_pred, svm_scores] = predict(svm_model, X_test_sel);
svm_test_accuracy = mean(svm_pred == y_test);
fprintf('  -> SVM Test Accuracy: %.2f%%\n', svm_test_accuracy*100);

%% Step 4.4: Train Random Forest
% -------------------------------------------------------------------------
% Random Forest: ensemble of N decision trees, each trained on a bootstrap
% sample with sqrt(d) random feature subset. The majority vote gives the
% prediction; posterior probabilities come from the proportion of trees.
%
% Reference: Breiman, L. (2001). "Random Forests." ML, 45(1), 5-32.
% -------------------------------------------------------------------------
fprintf('\n[PHASE 4] Training Random Forest (100 trees)...\n');

rf_model_cls = TreeBagger(100, X_train_sel, y_train, ...
    'Method', 'classification', ...
    'OOBPrediction', 'on', ...
    'MinLeafSize', 5);

oob_error = oobError(rf_model_cls);
rf_cv_accuracy = 1 - oob_error(end);  % OOB is an unbiased CV estimate
fprintf('  -> RF OOB Accuracy: %.2f%%\n', rf_cv_accuracy*100);

[rf_pred_cell, rf_scores] = predict(rf_model_cls, X_test_sel);
rf_pred = str2double(rf_pred_cell);
rf_test_accuracy = mean(rf_pred == y_test);
fprintf('  -> RF Test Accuracy: %.2f%%\n', rf_test_accuracy*100);

%% Step 4.5: Train Logistic Regression
% -------------------------------------------------------------------------
% Logistic Regression: P(y=1|x) = 1 / (1 + exp(-(w'x + b)))
% The linear model provides well-calibrated probabilities and serves as
% a strong interpretable baseline. Using fitglm with binomial distribution.
%
% Reference: Hosmer, D.W. & Lemeshow, S. (2000). "Applied Logistic 
%            Regression." Wiley.
% -------------------------------------------------------------------------
fprintf('\n[PHASE 4] Training Logistic Regression...\n');

lr_model = fitglm(X_train_sel, y_train, ...
    'Distribution', 'binomial', ...
    'Link', 'logit');

% LR predictions
lr_prob = predict(lr_model, X_test_sel);
lr_pred = double(lr_prob >= 0.5);
lr_scores = [1 - lr_prob, lr_prob];  % [P(0), P(1)]
lr_test_accuracy = mean(lr_pred == y_test);
fprintf('  -> LR Test Accuracy: %.2f%%\n', lr_test_accuracy*100);

% LR CV accuracy (manual k-fold)
cv_lr = cvpartition(y_train, 'KFold', config.cv_folds);
lr_cv_errors = zeros(config.cv_folds, 1);
for fold = 1:config.cv_folds
    train_fold = training(cv_lr, fold);
    test_fold = test(cv_lr, fold);
    mdl_fold = fitglm(X_train_sel(train_fold,:), y_train(train_fold), ...
        'Distribution', 'binomial', 'Link', 'logit');
    pred_fold = predict(mdl_fold, X_train_sel(test_fold,:));
    lr_cv_errors(fold) = mean((pred_fold >= 0.5) ~= y_train(test_fold));
end
lr_cv_accuracy = 1 - mean(lr_cv_errors);
fprintf('  -> LR %d-fold CV Accuracy: %.2f%%\n', config.cv_folds, lr_cv_accuracy*100);

%% Step 4.6: Validation Set Model Selection
% -------------------------------------------------------------------------
% Use the validation set to compute ensemble weights. This is the proper
% way to tune the ensemble: weights are based on validation accuracy, NOT
% training or CV accuracy. The test set remains completely untouched during
% weight computation.
% -------------------------------------------------------------------------
fprintf('\n[PHASE 4] Computing ensemble weights from validation set...\n');

% Validation predictions for each model
[~, knn_val_scores] = predict(knn_model, X_val_sel);
[~, nb_val_scores] = predict(nb_model, X_val_sel);
[~, svm_val_scores] = predict(svm_model, X_val_sel);
[rf_val_cell, rf_val_scores] = predict(rf_model_cls, X_val_sel);
lr_val_prob = predict(lr_model, X_val_sel);
lr_val_scores = [1 - lr_val_prob, lr_val_prob];

knn_val_acc = mean(predict(knn_model, X_val_sel) == y_val);
nb_val_acc = mean(predict(nb_model, X_val_sel) == y_val);
svm_val_acc = mean(predict(svm_model, X_val_sel) == y_val);
rf_val_acc = mean(str2double(rf_val_cell) == y_val);
lr_val_acc = mean((lr_val_prob >= 0.5) == y_val);

total_val_acc = knn_val_acc + nb_val_acc + svm_val_acc + rf_val_acc + lr_val_acc;
w_knn = knn_val_acc / total_val_acc;
w_nb  = nb_val_acc / total_val_acc;
w_svm = svm_val_acc / total_val_acc;
w_rf  = rf_val_acc / total_val_acc;
w_lr  = lr_val_acc / total_val_acc;

fprintf('  -> Validation Accuracies:\n');
fprintf('     k-NN: %.2f%% (w=%.3f)\n', knn_val_acc*100, w_knn);
fprintf('     NB:   %.2f%% (w=%.3f)\n', nb_val_acc*100, w_nb);
fprintf('     SVM:  %.2f%% (w=%.3f)\n', svm_val_acc*100, w_svm);
fprintf('     RF:   %.2f%% (w=%.3f)\n', rf_val_acc*100, w_rf);
fprintf('     LR:   %.2f%% (w=%.3f)\n', lr_val_acc*100, w_lr);

%% Step 4.7: Weighted Voting Ensemble on Test Set
% -------------------------------------------------------------------------
fprintf('\n[PHASE 4] Building weighted voting ensemble (5 models)...\n');

% Ensemble scoring on test set
ensemble_scores = w_knn * knn_scores + w_nb * nb_scores + ...
                  w_svm * svm_scores + w_rf * rf_scores + ...
                  w_lr * lr_scores;

[~, ensemble_pred_idx] = max(ensemble_scores, [], 2);
ensemble_pred = ensemble_pred_idx - 1;  % Convert from {1,2} to {0,1}

ensemble_accuracy = mean(ensemble_pred == y_test);
fprintf('  -> Ensemble Test Accuracy: %.2f%%\n', ensemble_accuracy*100);

%% Step 4.8: Comprehensive Performance Metrics
% -------------------------------------------------------------------------
fprintf('\n[PHASE 4] Computing comprehensive metrics...\n');

% Confusion Matrix
TP = sum(ensemble_pred == 1 & y_test == 1);
TN = sum(ensemble_pred == 0 & y_test == 0);
FP = sum(ensemble_pred == 1 & y_test == 0);
FN = sum(ensemble_pred == 0 & y_test == 1);

sensitivity = TP / (TP + FN);
specificity = TN / (TN + FP);
precision_val = TP / (TP + FP);
f1_score = 2 * precision_val * sensitivity / (precision_val + sensitivity);
mcc = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));

if isnan(f1_score), f1_score = 0; end
if isnan(mcc), mcc = 0; end

% ROC Curve and AUC
[roc_X, roc_Y, ~, auc] = perfcurve(y_test, ensemble_scores(:,2), 1);
[roc_X_knn, roc_Y_knn, ~, auc_knn] = perfcurve(y_test, knn_scores(:,2), 1);
[roc_X_nb, roc_Y_nb, ~, auc_nb] = perfcurve(y_test, nb_scores(:,2), 1);
[roc_X_svm, roc_Y_svm, ~, auc_svm] = perfcurve(y_test, svm_scores(:,2), 1);
[roc_X_rf, roc_Y_rf, ~, auc_rf] = perfcurve(y_test, rf_scores(:,2), 1);
[roc_X_lr, roc_Y_lr, ~, auc_lr] = perfcurve(y_test, lr_scores(:,2), 1);

fprintf('\n============================================\n');
fprintf('  ENSEMBLE CLASSIFICATION RESULTS\n');
fprintf('============================================\n');
fprintf('  Confusion Matrix:\n');
fprintf('                  Predicted\n');
fprintf('                  Neg    Pos\n');
fprintf('  Actual Neg:   %4d   %4d\n', TN, FP);
fprintf('  Actual Pos:   %4d   %4d\n', FN, TP);
fprintf('  ---------------------\n');
fprintf('  Accuracy:     %.4f (%.2f%%)\n', ensemble_accuracy, ensemble_accuracy*100);
fprintf('  Sensitivity:  %.4f (Detect %% of diabetics)\n', sensitivity);
fprintf('  Specificity:  %.4f (Detect %% of healthy)\n', specificity);
fprintf('  Precision:    %.4f\n', precision_val);
fprintf('  F1-Score:     %.4f\n', f1_score);
fprintf('  MCC:          %.4f\n', mcc);
fprintf('  AUC:          %.4f\n', auc);
fprintf('============================================\n');

% Store all results
ensemble_results = struct();
ensemble_results.accuracy = ensemble_accuracy;
ensemble_results.sensitivity = sensitivity;
ensemble_results.specificity = specificity;
ensemble_results.precision = precision_val;
ensemble_results.f1_score = f1_score;
ensemble_results.mcc = mcc;
ensemble_results.auc = auc;
ensemble_results.knn_accuracy = knn_test_accuracy;
ensemble_results.nb_accuracy = nb_test_accuracy;
ensemble_results.svm_accuracy = svm_test_accuracy;
ensemble_results.rf_accuracy = rf_test_accuracy;
ensemble_results.lr_accuracy = lr_test_accuracy;
ensemble_results.knn_cv_accuracy = knn_cv_accuracy;
ensemble_results.nb_cv_accuracy = nb_cv_accuracy;
ensemble_results.svm_cv_accuracy = svm_cv_accuracy;
ensemble_results.rf_cv_accuracy = rf_cv_accuracy;
ensemble_results.lr_cv_accuracy = lr_cv_accuracy;
ensemble_results.weights = [w_knn, w_nb, w_svm, w_rf, w_lr];
ensemble_results.distance_used = distance_used;
ensemble_results.best_k = best_k;
ensemble_results.ensemble_pred = ensemble_pred;
ensemble_results.ensemble_scores = ensemble_scores;
ensemble_results.confusion = [TN, FP; FN, TP];

%% Step 4.9: Generate Classification Report Figures
% -------------------------------------------------------------------------

% ----- FIGURE 1: Confusion Matrix & ROC Curves -----
figure('Name', 'Phase 4: Classification Results', ...
       'Position', [50, 50, 1600, 600], 'Color', 'w');

% Subplot 1: Confusion Matrix Heatmap
subplot(1,3,1);
cm = [TN, FP; FN, TP];
imagesc(cm);
colormap(gca, [linspace(1, 0.2, 64)', linspace(1, 0.5, 64)', linspace(1, 0.8, 64)']);
colorbar;

labels = {'TN', 'FP'; 'FN', 'TP'};
for i = 1:2
    for j = 1:2
        text(j, i, sprintf('%s\n%d', labels{i,j}, cm(i,j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 14, ...
            'FontWeight', 'bold');
    end
end

set(gca, 'XTick', [1,2], 'YTick', [1,2], ...
    'XTickLabel', {'Predicted Healthy', 'Predicted Diabetic'}, ...
    'YTickLabel', {'Actual Healthy', 'Actual Diabetic'}, ...
    'FontSize', 11);
title('Confusion Matrix', 'FontSize', 14, 'FontWeight', 'bold');

% Subplot 2: ROC Curves (All 6 models)
subplot(1,3,2);
plot(roc_X, roc_Y, 'b-', 'LineWidth', 2.5, ...
    'DisplayName', sprintf('Ensemble (AUC=%.3f)', auc));
hold on;
plot(roc_X_knn, roc_Y_knn, 'r--', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('k-NN (AUC=%.3f)', auc_knn));
plot(roc_X_nb, roc_Y_nb, 'g--', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('NB (AUC=%.3f)', auc_nb));
plot(roc_X_svm, roc_Y_svm, 'm--', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('SVM (AUC=%.3f)', auc_svm));
plot(roc_X_rf, roc_Y_rf, 'c--', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('RF (AUC=%.3f)', auc_rf));
plot(roc_X_lr, roc_Y_lr, 'k--', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('LR (AUC=%.3f)', auc_lr));
plot([0,1], [0,1], 'k:', 'LineWidth', 1, 'DisplayName', 'Random');
legend('show', 'Location', 'southeast', 'FontSize', 9);
xlabel('False Positive Rate (1 - Specificity)', 'FontSize', 12);
ylabel('True Positive Rate (Sensitivity)', 'FontSize', 12);
title('ROC Curves Comparison', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% Subplot 3: Performance Metrics Bar Chart
subplot(1,3,3);
metrics = [ensemble_accuracy, sensitivity, specificity, precision_val, f1_score, auc];
metric_names = {'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'AUC'};
b = bar(metrics, 0.6);
b.FaceColor = 'flat';
colors = parula(6);
b.CData = colors;
set(gca, 'XTickLabel', metric_names, 'XTickLabelRotation', 30, 'FontSize', 10);
ylabel('Score', 'FontSize', 12);
title('Ensemble Performance Metrics', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0, 1.1]);
grid on;
for i = 1:6
    text(i, metrics(i)+0.03, sprintf('%.3f', metrics(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
end

sgtitle('PHASE 4: Classification Engine Performance Report', ...
    'FontSize', 16, 'FontWeight', 'bold');

saveas(gcf, 'results/Phase4_Classification_Results.png');
saveas(gcf, 'results/Phase4_Classification_Results.fig');

% ----- FIGURE 2: Model Comparison -----
figure('Name', 'Phase 4: Model Comparison', ...
       'Position', [100, 100, 1400, 500], 'Color', 'w');

% Subplot 1: Accuracy comparison (all 5 + ensemble)
subplot(1,2,1);
model_test_acc = [knn_test_accuracy, nb_test_accuracy, svm_test_accuracy, ...
                  rf_test_accuracy, lr_test_accuracy, ensemble_accuracy] * 100;
model_cv_acc = [knn_cv_accuracy, nb_cv_accuracy, svm_cv_accuracy, ...
                rf_cv_accuracy, lr_cv_accuracy, ...
                sum([knn_cv_accuracy, nb_cv_accuracy, svm_cv_accuracy, ...
                     rf_cv_accuracy, lr_cv_accuracy] .* ...
                    [w_knn, w_nb, w_svm, w_rf, w_lr])] * 100;
model_data = [model_cv_acc; model_test_acc]';
b_comp = bar(model_data, 'grouped');
b_comp(1).FaceColor = [0.5, 0.7, 0.9];
b_comp(2).FaceColor = [0.2, 0.4, 0.8];
model_labels = {'k-NN', 'NB', 'SVM', 'RF', 'LR', 'Ensemble'};
set(gca, 'XTickLabel', model_labels, 'FontSize', 10, 'XTickLabelRotation', 15);
ylabel('Accuracy (%)', 'FontSize', 12);
legend('CV Accuracy', 'Test Accuracy', 'Location', 'southeast');
title('Model Accuracy Comparison', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

% Subplot 2: Prediction score distributions
subplot(1,2,2);
histogram(ensemble_scores(y_test==0, 2), 20, ...
    'FaceColor', [0.2, 0.6, 0.86], 'FaceAlpha', 0.7, ...
    'Normalization', 'pdf', 'DisplayName', 'Healthy');
hold on;
histogram(ensemble_scores(y_test==1, 2), 20, ...
    'FaceColor', [0.9, 0.3, 0.3], 'FaceAlpha', 0.7, ...
    'Normalization', 'pdf', 'DisplayName', 'Diabetic');
xline(0.5, '--k', 'LineWidth', 2, 'DisplayName', 'Threshold');
legend('show', 'Location', 'best');
xlabel('P(Diabetic | Features)', 'FontSize', 12);
ylabel('Probability Density', 'FontSize', 12);
title('Ensemble Score Distributions', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

sgtitle('PHASE 4: Model Comparison Analysis', ...
    'FontSize', 16, 'FontWeight', 'bold');

saveas(gcf, 'results/Phase4_Model_Comparison.png');
saveas(gcf, 'results/Phase4_Model_Comparison.fig');

fprintf('\n[PHASE 4] COMPLETE - Classification engine built and evaluated.\n');
fprintf('  -> 5 models trained: k-NN, NB, SVM, RF, LR\n');
fprintf('  -> Ensemble weights computed from validation set\n');
fprintf('  -> 2 report figures saved to results/\n');

end
