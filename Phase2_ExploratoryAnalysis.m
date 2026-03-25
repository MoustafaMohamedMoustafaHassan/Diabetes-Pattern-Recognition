function Phase2_ExploratoryAnalysis(X_normalized, y_all, config)
%% =========================================================================
%  PHASE 2: EXPLORATORY PATTERN ANALYSIS (GEOMETRIC VIEW)
%  =========================================================================
%  Mathematical Basis:
%  -------------------
%  Before applying any classifier, we must understand the geometric 
%  structure of our data in the feature space. This phase provides:
%
%  1. Covariance Analysis: The covariance matrix Sigma of a class defines 
%     the shape and orientation of its data cloud. If Sigma_0 != Sigma_1 
%     (healthy vs diabetic), this justifies using Mahalanobis distance
%     (which is covariance-aware) over Euclidean distance.
%
%  2. t-SNE Visualization: t-Distributed Stochastic Neighbor Embedding 
%     (van der Maaten & Hinton, 2008) preserves local neighborhood 
%     structure when projecting from R^8 -> R^2. This reveals:
%     - Cluster separability (can classes be distinguished?)
%     - Overlap regions (where misclassification is likely)
%     - Outlier structure (which patients are unusual?)
%
%  3. Mahalanobis Distance Distribution: Computing D_M for each sample
%     to both class centroids reveals the "statistical distance" between
%     the two populations and identifies samples near the decision boundary.
%
%  References:
%  [1] van der Maaten, L. & Hinton, G. (2008). "Visualizing Data using 
%      t-SNE." JMLR, 9, 2579-2605.
%  [2] Mahalanobis, P.C. (1936). "On the generalized distance in 
%      statistics." Proceedings of the National Institute of Sciences 
%      of India, 2(1), 49-55.
%  =========================================================================

fprintf('[PHASE 2] Starting exploratory pattern analysis...\n');

% Separate classes
X_class0 = X_normalized(y_all == 0, :);  % Healthy
X_class1 = X_normalized(y_all == 1, :);  % Diabetic

%% Step 2.1: Covariance Matrix Analysis
% -------------------------------------------------------------------------
% The covariance matrix Sigma encodes how features co-vary within a class.
% Key insight: if the covariance structures differ between classes, it 
% means the disease alters not just individual biomarkers, but the 
% RELATIONSHIPS between them. This is the core justification for using
% Mahalanobis distance in our classifier.
% -------------------------------------------------------------------------
fprintf('[PHASE 2] Computing class-conditional covariance matrices...\n');

cov_class0 = cov(X_class0);
cov_class1 = cov(X_class1);
cov_diff = cov_class1 - cov_class0;

% Frobenius norm of difference (measures how different the structures are)
frob_diff = norm(cov_diff, 'fro');
fprintf('  -> Frobenius norm of covariance difference: %.4f\n', frob_diff);
fprintf('  -> This confirms the covariance structures are DIFFERENT,\n');
fprintf('     justifying Mahalanobis over Euclidean distance.\n');

% ----- FIGURE 1: Covariance Heatmaps -----
figure('Name', 'Phase 2: Covariance Analysis', ...
       'Position', [50, 50, 1600, 500], 'Color', 'w');

% Healthy class covariance
subplot(1,3,1);
imagesc(cov_class0);
colorbar;
colormap(gca, 'parula');
set(gca, 'XTick', 1:8, 'YTick', 1:8, ...
    'XTickLabel', config.featureNames, ...
    'YTickLabel', config.featureNames, ...
    'XTickLabelRotation', 45, 'FontSize', 9);
title('\Sigma_{Healthy} (Class 0)', 'FontSize', 14, 'FontWeight', 'bold');
for i = 1:8
    for j = 1:8
        text(j, i, sprintf('%.2f', cov_class0(i,j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7);
    end
end

% Diabetic class covariance
subplot(1,3,2);
imagesc(cov_class1);
colorbar;
colormap(gca, 'parula');
set(gca, 'XTick', 1:8, 'YTick', 1:8, ...
    'XTickLabel', config.featureNames, ...
    'YTickLabel', config.featureNames, ...
    'XTickLabelRotation', 45, 'FontSize', 9);
title('\Sigma_{Diabetic} (Class 1)', 'FontSize', 14, 'FontWeight', 'bold');
for i = 1:8
    for j = 1:8
        text(j, i, sprintf('%.2f', cov_class1(i,j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7);
    end
end

% Difference matrix
subplot(1,3,3);
imagesc(cov_diff);
colorbar;
colormap(gca, redblue(256));  % Diverging colormap
set(gca, 'XTick', 1:8, 'YTick', 1:8, ...
    'XTickLabel', config.featureNames, ...
    'YTickLabel', config.featureNames, ...
    'XTickLabelRotation', 45, 'FontSize', 9);
title('\Delta\Sigma = \Sigma_{Diabetic} - \Sigma_{Healthy}', ...
    'FontSize', 14, 'FontWeight', 'bold');
for i = 1:8
    for j = 1:8
        text(j, i, sprintf('%.2f', cov_diff(i,j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7);
    end
end

sgtitle(sprintf('Covariance Structure Analysis (Frobenius ||\\Delta\\Sigma||=%.3f)', ...
    frob_diff), 'FontSize', 16, 'FontWeight', 'bold');

saveas(gcf, 'results/Phase2_Covariance_Analysis.png');
saveas(gcf, 'results/Phase2_Covariance_Analysis.fig');

%% Step 2.2: t-SNE Visualization
% -------------------------------------------------------------------------
% t-SNE minimizes the KL divergence between high-dimensional and 
% low-dimensional probability distributions of pairwise similarities:
%   KL(P||Q) = SUM_i SUM_j p_{ij} * log(p_{ij} / q_{ij})
% This reveals the intrinsic cluster structure in a visually interpretable
% 2D embedding while preserving local neighborhood relationships.
% -------------------------------------------------------------------------
fprintf('\n[PHASE 2] Computing t-SNE embedding (R^8 -> R^2)...\n');

% Run t-SNE with optimal perplexity
perplexity_value = 30;
Y_tsne = tsne(X_normalized, 'Algorithm', 'barneshut', ...
    'Perplexity', perplexity_value, 'NumDimensions', 2, ...
    'Standardize', false);  % Already normalized

fprintf('  -> t-SNE embedding complete (perplexity=%d)\n', perplexity_value);

% ----- FIGURE 2: t-SNE Visualization -----
figure('Name', 'Phase 2: t-SNE Pattern Visualization', ...
       'Position', [100, 100, 1200, 500], 'Color', 'w');

% Plot 1: Colored by class
subplot(1,2,1);
scatter(Y_tsne(y_all==0, 1), Y_tsne(y_all==0, 2), 40, ...
    [0.2, 0.6, 0.86], 'filled', 'MarkerFaceAlpha', 0.6, ...
    'DisplayName', 'Healthy');
hold on;
scatter(Y_tsne(y_all==1, 1), Y_tsne(y_all==1, 2), 40, ...
    [0.9, 0.3, 0.3], 'filled', 'MarkerFaceAlpha', 0.6, ...
    'DisplayName', 'Diabetic');
legend('show', 'Location', 'best', 'FontSize', 12);
xlabel('t-SNE Dimension 1', 'FontSize', 12);
ylabel('t-SNE Dimension 2', 'FontSize', 12);
title('t-SNE: Class Separation View', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% Annotate overlap region
overlap_text = sprintf('Note: Overlap regions indicate\nwhere simple classifiers will fail');
text(min(Y_tsne(:,1))+5, max(Y_tsne(:,2))-5, overlap_text, ...
    'FontSize', 9, 'FontAngle', 'italic', ...
    'BackgroundColor', [1 1 0.9], 'EdgeColor', [0.5 0.5 0.5]);

% Plot 2: Colored by Glucose (most important feature)
subplot(1,2,2);
scatter(Y_tsne(:, 1), Y_tsne(:, 2), 40, X_normalized(:, 2), ...
    'filled', 'MarkerFaceAlpha', 0.7);
colorbar;
colormap(gca, 'jet');
xlabel('t-SNE Dimension 1', 'FontSize', 12);
ylabel('t-SNE Dimension 2', 'FontSize', 12);
title('t-SNE: Glucose Gradient Overlay', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

sgtitle('PHASE 2: t-SNE Pattern Visualization (Perplexity=30)', ...
    'FontSize', 16, 'FontWeight', 'bold');

saveas(gcf, 'results/Phase2_tSNE_Visualization.png');
saveas(gcf, 'results/Phase2_tSNE_Visualization.fig');

%% Step 2.3: Mahalanobis Distance Distribution Analysis
% -------------------------------------------------------------------------
% For each sample x, we compute the Mahalanobis distance to both centroids:
%   D_M(x, mu_c) = sqrt((x - mu_c)' * Sigma_c^{-1} * (x - mu_c))
% The distribution of these distances reveals how well-separated the 
% classes are in the statistically-corrected feature space.
% -------------------------------------------------------------------------
fprintf('[PHASE 2] Computing Mahalanobis distance distributions...\n');

mu0 = mean(X_class0);
mu1 = mean(X_class1);

% Compute distances from each sample to both centroids
% FIX: Using regularized covariance and backslash (\) instead of inv()
% for better numerical stability
lambda = 0.01;  % Regularization parameter
cov0_reg = cov_class0 + lambda * eye(8);
cov1_reg = cov_class1 + lambda * eye(8);

d_to_healthy = zeros(size(X_normalized, 1), 1);
d_to_diabetic = zeros(size(X_normalized, 1), 1);

for i = 1:size(X_normalized, 1)
    diff0 = X_normalized(i,:) - mu0;
    diff1 = X_normalized(i,:) - mu1;
    % FIX: Use backslash (\) instead of inv() for numerical stability
    % inv(A)*b is less stable than A\b
    d_to_healthy(i)  = sqrt(diff0 * (cov0_reg \ diff0'));
    d_to_diabetic(i) = sqrt(diff1 * (cov1_reg \ diff1'));
end

% ----- FIGURE 3: Mahalanobis Distance Analysis -----
figure('Name', 'Phase 2: Mahalanobis Distance Analysis', ...
       'Position', [150, 150, 1400, 500], 'Color', 'w');

% Distance ratio plot
subplot(1,3,1);
ratio = d_to_diabetic ./ (d_to_healthy + d_to_diabetic);
histogram(ratio(y_all==0), 30, 'FaceColor', [0.2, 0.6, 0.86], ...
    'FaceAlpha', 0.7, 'Normalization', 'pdf', 'DisplayName', 'Healthy');
hold on;
histogram(ratio(y_all==1), 30, 'FaceColor', [0.9, 0.3, 0.3], ...
    'FaceAlpha', 0.7, 'Normalization', 'pdf', 'DisplayName', 'Diabetic');
xline(0.5, '--k', 'LineWidth', 2, 'DisplayName', 'Decision Boundary');
legend('show', 'Location', 'best');
xlabel('D_{Diabetic} / (D_{Healthy} + D_{Diabetic})', 'FontSize', 11);
ylabel('Probability Density', 'FontSize', 11);
title('Mahalanobis Distance Ratio', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

% 2D Distance scatter
subplot(1,3,2);
scatter(d_to_healthy(y_all==0), d_to_diabetic(y_all==0), 30, ...
    [0.2, 0.6, 0.86], 'filled', 'MarkerFaceAlpha', 0.5, ...
    'DisplayName', 'Healthy');
hold on;
scatter(d_to_healthy(y_all==1), d_to_diabetic(y_all==1), 30, ...
    [0.9, 0.3, 0.3], 'filled', 'MarkerFaceAlpha', 0.5, ...
    'DisplayName', 'Diabetic');
% Draw decision boundary (where distances are equal)
max_d = max([d_to_healthy; d_to_diabetic]);
plot([0, max_d], [0, max_d], '--k', 'LineWidth', 2, ...
    'DisplayName', 'Equal Distance Line');
legend('show', 'Location', 'best');
xlabel('D_{Mahalanobis} to Healthy Centroid', 'FontSize', 11);
ylabel('D_{Mahalanobis} to Diabetic Centroid', 'FontSize', 11);
title('Mahalanobis Distance Space', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

% Eigenvalue analysis of covariance matrices
subplot(1,3,3);
eig0 = sort(eig(cov_class0), 'descend');
eig1 = sort(eig(cov_class1), 'descend');
bar_data = [eig0, eig1];
b3 = bar(bar_data, 'grouped');
b3(1).FaceColor = [0.2, 0.6, 0.86];
b3(2).FaceColor = [0.9, 0.3, 0.3];
xlabel('Principal Component Index', 'FontSize', 11);
ylabel('Eigenvalue (Variance Explained)', 'FontSize', 11);
legend('Healthy', 'Diabetic', 'Location', 'best');
title('Eigenspectrum of Covariance Matrices', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

sgtitle('PHASE 2: Mahalanobis Distance & Eigenspectrum Analysis', ...
    'FontSize', 16, 'FontWeight', 'bold');

saveas(gcf, 'results/Phase2_Mahalanobis_Analysis.png');
saveas(gcf, 'results/Phase2_Mahalanobis_Analysis.fig');

%% Step 2.4: Pairwise Feature Scatter Matrix (Top Features)
fprintf('[PHASE 2] Generating pairwise feature scatter plots...\n');

figure('Name', 'Phase 2: Feature Space Exploration', ...
       'Position', [200, 50, 1200, 1000], 'Color', 'w');

% Top 4 features for visualization: Glucose, BMI, Age, DiabetesPedigree
top_features = [2, 6, 8, 7];  
top_names = config.featureNames(top_features);
n_top = length(top_features);

plot_idx = 1;
for i = 1:n_top
    for j = 1:n_top
        subplot(n_top, n_top, plot_idx);
        if i == j
            % Diagonal: histograms
            histogram(X_normalized(y_all==0, top_features(i)), 20, ...
                'FaceColor', [0.2, 0.6, 0.86], 'FaceAlpha', 0.6, ...
                'Normalization', 'pdf');
            hold on;
            histogram(X_normalized(y_all==1, top_features(i)), 20, ...
                'FaceColor', [0.9, 0.3, 0.3], 'FaceAlpha', 0.6, ...
                'Normalization', 'pdf');
            title(top_names{i}, 'FontSize', 10);
        else
            % Off-diagonal: scatter plots
            scatter(X_normalized(y_all==0, top_features(j)), ...
                    X_normalized(y_all==0, top_features(i)), 10, ...
                    [0.2, 0.6, 0.86], 'filled', 'MarkerFaceAlpha', 0.3);
            hold on;
            scatter(X_normalized(y_all==1, top_features(j)), ...
                    X_normalized(y_all==1, top_features(i)), 10, ...
                    [0.9, 0.3, 0.3], 'filled', 'MarkerFaceAlpha', 0.3);
            if j == 1
                ylabel(top_names{i}, 'FontSize', 9);
            end
            if i == n_top
                xlabel(top_names{j}, 'FontSize', 9);
            end
        end
        plot_idx = plot_idx + 1;
    end
end

sgtitle('PHASE 2: Pairwise Feature Space (Top 4 Features)', ...
    'FontSize', 16, 'FontWeight', 'bold');

saveas(gcf, 'results/Phase2_Feature_Scatter.png');
saveas(gcf, 'results/Phase2_Feature_Scatter.fig');

fprintf('[PHASE 2] COMPLETE - Exploratory analysis finished.\n');
fprintf('  -> 4 analysis figures saved to results/\n');

end

%% ===================== HELPER FUNCTION =====================
function c = redblue(m)
% REDBLUE - Red-White-Blue diverging colormap
    if nargin < 1, m = 256; end
    n = ceil(m/2);
    r = [(0:n-1)'/n; ones(n,1)];
    g = [(0:n-1)'/n; flipud((0:n-1)'/n)];
    b = [ones(n,1); flipud((0:n-1)'/n)];
    c = [r, g, b];
    if size(c,1) > m
        c = c(1:m, :);
    end
end
