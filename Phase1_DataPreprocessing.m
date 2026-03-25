function [X_train, X_val, X_test, y_train, y_val, y_test, X_normalized, y_all, ...
          X_raw, X_imputed, imputation_stats] = Phase1_DataPreprocessing(config)
%% =========================================================================
%  PHASE 1: DATA PRE-PROCESSING & FEATURE SPACE PREPARATION
%  =========================================================================
%  Mathematical Basis:
%  -------------------
%  In Pattern Recognition, the quality of the feature space directly
%  determines the quality of the decision boundary. This phase constructs
%  a leakage-free feature space through the following ordered stages:
%
%  1. Data Integrity: Missing values (encoded as zeros in physiological
%     measures) are replaced with NaN markers before any set is formed.
%     The training partition is then imputed using k-NN (Troyanskaya
%     et al., 2001), which preserves the local manifold structure.
%     Validation and test partitions are imputed using the column-wise
%     medians derived exclusively from valid training observations,
%     ensuring that no information from held-out sets contaminates
%     the learned statistics.
%
%  2. Feature Scaling: Z-score normalization transforms each feature x_i
%     using parameters (mu_i, sigma_i) computed solely on the training set:
%         z_i = (x_i - mu_i) / sigma_i
%     The same parameters are applied to validation and test sets, which
%     mirrors real deployment conditions where population statistics are
%     unavailable at inference time. This also ensures that the
%     Mahalanobis distance computation in Phase 4 operates on a properly
%     conditioned feature space, preventing features with large numeric
%     ranges (e.g., Insulin: 0-846) from dominating features with small
%     ranges (e.g., DiabetesPedigree: 0.078-2.42).
%
%  3. Stratified Splitting: The dataset is partitioned before imputation
%     and normalization. Stratification maintains the class ratio across
%     all three sets, which is critical for imbalanced medical datasets.
%
%  References:
%  [1] Troyanskaya, O., et al. (2001). "Missing value estimation methods
%      for DNA microarrays." Bioinformatics, 17(6), 520-525.
%  [2] Jain, A.K., et al. (2000). "Statistical Pattern Recognition:
%      A Review." IEEE TPAMI, 22(1), 4-37.
%  =========================================================================

%% Step 1.1: Load the Raw Dataset
fprintf('[PHASE 1] Loading dataset from: %s\n', config.dataFile);

% Verify the data file exists, with a fallback to the working directory
if ~isfile(config.dataFile)
    if isfile('diabetes.csv')
        config.dataFile = 'diabetes.csv';
    else
        error('Dataset file not found. Please place diabetes.csv in the data/ directory.');
    end
end

% Read the CSV file into a table and extract numeric arrays
data   = readtable(config.dataFile);
fprintf('  -> Loaded %d samples with %d columns\n', height(data), width(data));

X_raw = table2array(data(:, 1:8));  % Feature matrix: 768 x 8
y_all = table2array(data(:, 9));    % Label vector:   768 x 1

% Report the class distribution of the full dataset
n_healthy  = sum(y_all == 0);
n_diabetic = sum(y_all == 1);
fprintf('  -> Class distribution: %d Healthy (%.1f%%) | %d Diabetic (%.1f%%)\n', ...
    n_healthy, n_healthy/length(y_all)*100, ...
    n_diabetic, n_diabetic/length(y_all)*100);
fprintf('  -> Imbalance ratio: 1:%.2f\n', n_healthy/n_diabetic);

%% Step 1.2: Detect and Mark Biologically Impossible Zeros as NaN
% -------------------------------------------------------------------------
% Medical Context: A living person cannot have Glucose=0, BloodPressure=0,
% BMI=0, etc. These zeros are data-entry artifacts representing missing
% measurements. Replacing them with NaN before the dataset is split ensures
% that the missingness structure of the original data is preserved intact
% across all three partitions.
% -------------------------------------------------------------------------
fprintf('\n[PHASE 1] Detecting biologically impossible zero values:\n');

imputation_stats = struct();
zero_counts = zeros(1, length(config.zeroIsMissing));

X_with_nan = X_raw;  % Work on a copy to preserve X_raw for reporting
for i = 1:length(config.zeroIsMissing)
    col      = config.zeroIsMissing(i);
    zero_mask = X_with_nan(:, col) == 0;
    zero_counts(i) = sum(zero_mask);
    fprintf('  -> %s: %d zeros detected (%.1f%% of data)\n', ...
        config.featureNames{col}, zero_counts(i), ...
        zero_counts(i)/size(X_with_nan, 1)*100);
    X_with_nan(zero_mask, col) = NaN;
end

imputation_stats.zero_counts   = zero_counts;
imputation_stats.zero_features = config.zeroIsMissing;

%% Step 1.3: Stratified Train-Validation-Test Split
% -------------------------------------------------------------------------
% The split is performed on the NaN-marked raw data, before any imputation
% or normalization. This is the foundational requirement for a leakage-free
% pipeline: all statistics used to preprocess the data (imputation values,
% normalization parameters) must be derived exclusively from the training
% partition and then applied forward to the held-out partitions.
%
% Split proportions:
%   - 70% Training   — used for all model fitting
%   - 10% Validation — used for hyperparameter selection
%   - 20% Testing    — used for the final unbiased evaluation
%
% Stratified sampling ensures the class ratio is preserved across all three
% sets, which is critical for imbalanced medical datasets.
% -------------------------------------------------------------------------
fprintf('\n[PHASE 1] Performing stratified train/validation/test split (70/10/20)...\n');

% Stage A: Hold out 20% as the test set, stratified on class labels
cv1          = cvpartition(y_all, 'HoldOut', config.test_ratio, 'Stratify', true);
test_idx     = test(cv1);
trainval_idx = training(cv1);

% Stage B: From the remaining 80%, hold out 12.5% as validation.
%          12.5% of 80% equals 10% of the total dataset.
y_trainval      = y_all(trainval_idx);
cv2             = cvpartition(y_trainval, 'HoldOut', config.val_ratio, 'Stratify', true);
val_idx_local   = test(cv2);
train_idx_local = training(cv2);

% Map local trainval indices back to global positions in the full dataset
global_trainval_indices = find(trainval_idx);
train_idx = false(size(y_all));
val_idx   = false(size(y_all));
train_idx(global_trainval_indices(train_idx_local)) = true;
val_idx(global_trainval_indices(val_idx_local))     = true;

% Partition the NaN-marked feature matrix and the label vector
X_train_with_nan = X_with_nan(train_idx, :);
X_val_with_nan   = X_with_nan(val_idx,   :);
X_test_with_nan  = X_with_nan(test_idx,  :);
y_train = y_all(train_idx);
y_val   = y_all(val_idx);
y_test  = y_all(test_idx);

% Confirm that stratification preserved the class ratio across all sets
fprintf('  -> Training set:   %d samples (Healthy: %d, Diabetic: %d)\n', ...
    length(y_train), sum(y_train==0), sum(y_train==1));
fprintf('  -> Validation set: %d samples (Healthy: %d, Diabetic: %d)\n', ...
    length(y_val), sum(y_val==0), sum(y_val==1));
fprintf('  -> Testing set:    %d samples (Healthy: %d, Diabetic: %d)\n', ...
    length(y_test), sum(y_test==0), sum(y_test==1));
fprintf('  -> Train ratio: %.2f | Val ratio: %.2f | Test ratio: %.2f\n', ...
    sum(y_train==0)/sum(y_train==1), ...
    sum(y_val==0)/sum(y_val==1), ...
    sum(y_test==0)/sum(y_test==1));

%% Step 1.4: k-NN Imputation on the Training Partition
% -------------------------------------------------------------------------
% Mathematical Basis: k-NN imputation estimates each missing value x_{i,j}
% as the mean of the corresponding column across the k nearest neighbors
% of row i, where proximity is measured using the non-missing features:
%
%   x_{i,j} = (1/k) * SUM_{m in N_k(i)} x_{m,j}
%
% Neighbors are identified within the training set only, so the imputed
% values capture the local manifold structure of the training distribution
% without any influence from validation or test observations.
% -------------------------------------------------------------------------
fprintf('\n[PHASE 1] Performing k-NN imputation on training set (k=%d)...\n', ...
    config.knn_impute_k);

% Attempt the Bioinformatics Toolbox implementation; fall back to the
% manual implementation if the toolbox is not available.
try
    % knnimpute operates column-wise on a (features x samples) matrix,
    % so the input is transposed and the result is transposed back.
    X_train_imputed = knnimpute(X_train_with_nan', config.knn_impute_k)';
    fprintf('  -> Imputation complete using knnimpute (Bioinformatics Toolbox).\n');
catch ME
    fprintf('  -> knnimpute not available: %s\n', ME.message);
    fprintf('  -> Using manual k-NN imputation fallback...\n');
    X_train_imputed = manual_knn_impute(X_train_with_nan, config.knn_impute_k);
    fprintf('  -> Manual k-NN imputation complete.\n');
end

% Verify completeness; apply a column-median safety net for any residual NaN
remaining_nan = sum(isnan(X_train_imputed(:)));
fprintf('  -> Remaining NaN values after k-NN imputation: %d\n', remaining_nan);

if remaining_nan > 0
    fprintf('  -> Applying column-median safety net for residual NaN values...\n');
    for col = 1:size(X_train_imputed, 2)
        nan_mask = isnan(X_train_imputed(:, col));
        if any(nan_mask)
            X_train_imputed(nan_mask, col) = ...
                median(X_train_imputed(~nan_mask, col));
        end
    end
end

% Record per-feature imputation statistics from the training set
for i = 1:length(config.zeroIsMissing)
    col = config.zeroIsMissing(i);
    imputation_stats.imputed_means(i) = mean(X_train_imputed(:, col));
    imputation_stats.imputed_stds(i)  = std(X_train_imputed(:, col));
end

%% Step 1.5: Median Imputation for Validation and Test Partitions
% -------------------------------------------------------------------------
% Validation and test observations are filled using the column-wise medians
% computed from the valid (non-NaN) entries of the TRAINING partition.
% This approach:
%   (a) Guarantees zero information leakage from held-out sets.
%   (b) Is robust to outliers that can distort mean-based estimates.
%   (c) Is straightforward to replicate at deployment time, requiring only
%       the stored train_medians vector from imputation_stats.
%
% Formally, for each feature column j:
%   median_j = median({ x_{i,j} : i in TrainSet, x_{i,j} is not NaN })
%
% Any NaN at position (i, j) in X_val or X_test is then replaced with
% median_j.
% -------------------------------------------------------------------------
fprintf('\n[PHASE 1] Imputing validation and test sets with training-set medians...\n');

% Compute the median of each feature from valid training observations only
n_features    = size(X_train_with_nan, 2);
train_medians = zeros(1, n_features);
for col = 1:n_features
    valid_vals = X_train_with_nan(~isnan(X_train_with_nan(:, col)), col);
    if isempty(valid_vals)
        train_medians(col) = 0;  % Degenerate fallback; should not occur in practice
    else
        train_medians(col) = median(valid_vals);
    end
end

% Apply training medians to fill NaNs in the validation partition
X_val_imputed = X_val_with_nan;
for col = 1:n_features
    nan_mask = isnan(X_val_imputed(:, col));
    if any(nan_mask)
        X_val_imputed(nan_mask, col) = train_medians(col);
    end
end

% Apply training medians to fill NaNs in the test partition
X_test_imputed = X_test_with_nan;
for col = 1:n_features
    nan_mask = isnan(X_test_imputed(:, col));
    if any(nan_mask)
        X_test_imputed(nan_mask, col) = train_medians(col);
    end
end

fprintf('  -> Validation and test imputation complete using training medians.\n');

% Persist the median vector so Phase 6 can impute new patient inputs
% using the identical transformation applied during training
imputation_stats.train_medians = train_medians;

%% Step 1.6: Z-Score Normalization Using Training Statistics
% -------------------------------------------------------------------------
% Mathematical Basis: Z-score normalization transforms feature x to:
%   z = (x - mu) / sigma
% This is essential because the Mahalanobis distance D_M is defined as:
%   D_M(x, mu) = sqrt((x-mu)' * Sigma^{-1} * (x-mu))
% where Sigma is the covariance matrix. Pre-normalizing ensures numerical
% stability in the matrix inversion step.
%
% The parameters (mu, sigma) are derived from the training set alone and
% then applied uniformly to validation and test partitions. This replicates
% the real-world scenario in which population statistics are estimated from
% labeled training data and frozen before the model encounters new inputs.
%
% The stored (mu, sigma) enable Phase 6 (App Designer) to apply the
% identical transformation to arbitrary new patient data at inference time.
% -------------------------------------------------------------------------
fprintf('\n[PHASE 1] Applying Z-score normalization using training statistics...\n');

% Estimate normalization parameters from the training partition only
[X_train, mu_norm, sigma_norm] = zscore(X_train_imputed);

% Apply the same training-derived parameters to validation and test sets
X_val  = (X_val_imputed  - mu_norm) ./ sigma_norm;
X_test = (X_test_imputed - mu_norm) ./ sigma_norm;

% Persist normalization parameters for downstream phases and deployment
imputation_stats.mu    = mu_norm;    % Per-feature mean (training partition, original scale)
imputation_stats.sigma = sigma_norm; % Per-feature std  (training partition, original scale)

% Display per-feature statistics on the normalized training set
fprintf('  -> Feature statistics after normalization (training set):\n');
fprintf('  %-20s  Mean       Std\n', 'Feature');
fprintf('  %-20s  ---------  ---------\n', '-------');
for i = 1:8
    fprintf('  %-20s  %+.6f  %.6f\n', config.featureNames{i}, ...
        mean(X_train(:, i)), std(X_train(:, i)));
end
fprintf('\n  -> Normalization parameters (mu, sigma) saved for deployment.\n');
fprintf('  -> Phase 6 will use these to normalize new patient inputs.\n');

% Assign output aliases used by the pipeline and reporting figures.
% X_normalized holds the normalized training feature matrix.
% X_imputed holds the original-scale imputed training matrix (used below
% for the pre/post-imputation distribution plot).
X_normalized = X_train;
X_imputed    = X_train_imputed;

%% Step 1.7: Generate Pre-processing Report Figure
% -------------------------------------------------------------------------
% Visual documentation of the pre-processing pipeline. All plots use
% training-set data so that the figures faithfully represent the statistics
% that were used during model learning.
% -------------------------------------------------------------------------
figure('Name', 'Phase 1: Data Pre-processing Report', ...
       'Position', [50, 50, 1400, 900], 'Color', 'w');

% --- Subplot 1: Class Distribution (full dataset) ---
subplot(2, 3, 1);
bar_data = [n_healthy, n_diabetic];
b = bar(bar_data, 0.6);
b.FaceColor = 'flat';
b.CData(1, :) = [0.2, 0.6, 0.86];  % Blue for healthy
b.CData(2, :) = [0.9, 0.3, 0.3];   % Red for diabetic
set(gca, 'XTickLabel', {'Healthy (0)', 'Diabetic (1)'}, 'FontSize', 11);
ylabel('Number of Samples', 'FontSize', 12);
title('Class Distribution', 'FontSize', 13, 'FontWeight', 'bold');
text(1, n_healthy + 15, sprintf('%d (%.1f%%)', n_healthy, n_healthy/768*100), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(2, n_diabetic + 15, sprintf('%d (%.1f%%)', n_diabetic, n_diabetic/768*100), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on;

% --- Subplot 2: Missing Data Pattern (full dataset, before splitting) ---
subplot(2, 3, 2);
missing_features = config.featureNames(config.zeroIsMissing);
bar(zero_counts, 0.6, 'FaceColor', [0.85, 0.55, 0.13]);
set(gca, 'XTickLabel', missing_features, 'XTickLabelRotation', 45, 'FontSize', 10);
ylabel('Count of Missing (Zero) Values', 'FontSize', 11);
title('Missing Data Pattern', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

% --- Subplot 3: Glucose Distribution Before vs After Imputation ---
% The pre-imputation histogram uses all non-zero glucose values from the
% original dataset; the post-imputation histogram uses the training set.
subplot(2, 3, 3);
histogram(data.Glucose(data.Glucose > 0), 30, ...
    'FaceColor', [0.3, 0.7, 0.4], 'FaceAlpha', 0.7, ...
    'DisplayName', 'Original (non-zero)');
hold on;
histogram(X_imputed(:, 2), 30, ...
    'FaceColor', [0.2, 0.4, 0.8], 'FaceAlpha', 0.5, ...
    'DisplayName', 'After Imputation (train)');
legend('show', 'Location', 'best');
xlabel('Glucose Level', 'FontSize', 11);
ylabel('Frequency', 'FontSize', 11);
title('Glucose: Before vs After Imputation', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

% --- Subplot 4: Box plots of normalized training features ---
subplot(2, 3, 4);
boxplot(X_normalized, 'Labels', config.featureNames, ...
    'LabelOrientation', 'inline');
ylabel('Normalized Value (Z-score)', 'FontSize', 11);
title('Feature Distributions — Training Set (Post-Normalization)', ...
    'FontSize', 13, 'FontWeight', 'bold');
grid on;

% --- Subplot 5: Feature correlation heatmap (normalized training set) ---
subplot(2, 3, 5);
corr_matrix = corr(X_normalized);
imagesc(corr_matrix);
colorbar;
colormap(gca, 'parula');
set(gca, 'XTick', 1:8, 'YTick', 1:8, ...
    'XTickLabel', config.featureNames, ...
    'YTickLabel', config.featureNames, ...
    'XTickLabelRotation', 45, 'FontSize', 9);
title('Feature Correlation Matrix (Training Set)', 'FontSize', 13, 'FontWeight', 'bold');
for i = 1:8
    for j = 1:8
        text(j, i, sprintf('%.2f', corr_matrix(i, j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7, ...
            'Color', abs(corr_matrix(i, j)) > 0.5 * [1, 1, 1]);
    end
end

% --- Subplot 6: Class composition of each partition ---
subplot(2, 3, 6);
split_data = [sum(y_train==0), sum(y_train==1); ...
              sum(y_val==0),   sum(y_val==1);   ...
              sum(y_test==0),  sum(y_test==1)];
b2 = bar(split_data, 'grouped');
b2(1).FaceColor = [0.2, 0.6, 0.86];
b2(2).FaceColor = [0.9, 0.3, 0.3];
set(gca, 'XTickLabel', {'Training Set', 'Validation Set', 'Testing Set'}, 'FontSize', 11);
ylabel('Number of Samples', 'FontSize', 11);
legend('Healthy', 'Diabetic', 'Location', 'best');
title('Stratified Train / Val / Test Split', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

sgtitle('PHASE 1: Data Pre-processing Report', 'FontSize', 16, 'FontWeight', 'bold');

% Save the report figure to the results directory
saveas(gcf, 'results/Phase1_Preprocessing_Report.png');
saveas(gcf, 'results/Phase1_Preprocessing_Report.fig');
fprintf('\n[PHASE 1] Report figure saved to results/\n');
fprintf('[PHASE 1] COMPLETE - Feature space prepared.\n');

end

%% ===================== HELPER FUNCTION =====================
function X_imputed = manual_knn_impute(X, k)
% MANUAL_KNN_IMPUTE  k-NN imputation without the Bioinformatics Toolbox.
% -------------------------------------------------------------------------
% For each missing value at position (row, col):
%   1. Find the k nearest neighbors of that row using the Euclidean distance
%      computed over all columns that are non-NaN for both the query row and
%      each candidate neighbor (mutually available features).
%   2. Impute the missing value as the mean of the target column across
%      those k neighbors.
%
% If no valid neighbors exist, the column median is used as a safe fallback.
% -------------------------------------------------------------------------
    [~, n_cols] = size(X);
    X_imputed   = X;

    for col = 1:n_cols
        nan_rows = find(isnan(X(:, col)));
        if isempty(nan_rows)
            continue;
        end

        for r = 1:length(nan_rows)
            row = nan_rows(r);

            % Columns available (non-NaN) in this row for distance computation
            available_cols = find(~isnan(X(row, :)));
            if isempty(available_cols)
                % Entire row is NaN; fall back to column median
                X_imputed(row, col) = median(X(~isnan(X(:, col)), col));
                continue;
            end

            % Candidate rows must be non-NaN in the target column
            candidate_rows = setdiff(find(~isnan(X(:, col))), row);

            if isempty(candidate_rows)
                X_imputed(row, col) = median(X(~isnan(X(:, col)), col));
                continue;
            end

            % Compute Euclidean distances over mutually available columns
            dists = zeros(length(candidate_rows), 1);
            for c = 1:length(candidate_rows)
                cr           = candidate_rows(c);
                mutual_cols  = available_cols(~isnan(X(cr, available_cols)));
                if isempty(mutual_cols)
                    dists(c) = Inf;
                else
                    dists(c) = sqrt(sum((X(row, mutual_cols) - X(cr, mutual_cols)).^2));
                end
            end

            % Select the k nearest neighbors and average their target-column values
            [~, sort_idx]  = sort(dists);
            k_actual       = min(k, length(sort_idx));
            nn_indices     = candidate_rows(sort_idx(1:k_actual));
            X_imputed(row, col) = mean(X(nn_indices, col));
        end
    end
end
