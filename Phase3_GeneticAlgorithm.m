function [selected_features, ga_results] = Phase3_GeneticAlgorithm(X_train, y_train, config)
%% =========================================================================
%  PHASE 3: GENETIC ALGORITHM FEATURE SELECTION
%  =========================================================================
%  Mathematical Basis:
%  -------------------
%  Feature selection is a combinatorial optimization problem. With 8 
%  features, there are 2^8 - 1 = 255 possible feature subsets. The Genetic
%  Algorithm (GA) efficiently searches this space by:
%
%  1. Encoding: Each solution (chromosome) is a binary vector b in {0,1}^8
%     where b_i = 1 means feature i is selected.
%
%  2. Fitness Function: f(b) = CrossValidationError(X[:, b==1], y)
%     We minimize the k-NN classification error using k-fold CV on the
%     selected feature subset. A small penalty alpha*sum(b) is added to
%     favor simpler models (Occam's Razor).
%
%  3. Evolution: GA applies selection (tournament), crossover (single-point),
%     and mutation (bit-flip) to evolve better feature subsets.
%
%  4. Convergence: The process terminates when the best fitness stagnates
%     for a specified number of generations or max generations is reached.
%
%  Why GA over Filter Methods?
%  - GA evaluates feature INTERACTIONS, not just individual relevance.
%  - Example: Insulin alone may be weak, but Insulin + Glucose together
%    may be very strong. GA can discover such synergies.
%
%  References:
%  [1] Goldberg, D.E. (1989). "Genetic Algorithms in Search, Optimization
%      and Machine Learning." Addison-Wesley.
%  [2] Siedlecki, W. & Sklansky, J. (1989). "A Note on Genetic Algorithms
%      for Large-Scale Feature Selection." Pattern Recognition Letters.
%  =========================================================================

fprintf('[PHASE 3] Initializing Genetic Algorithm for feature selection...\n');

%% Step 3.1: Define GA Parameters
% -------------------------------------------------------------------------
% The GA operates on binary chromosomes of length 8 (one bit per feature).
% We use MATLAB's ga() with integer constraints (binary is a special case).
% -------------------------------------------------------------------------

nFeatures = size(X_train, 2);  % 8 features

% Store training data as persistent for fitness function access
% (Using nested function approach)
fprintf('  -> Search space: 2^%d - 1 = %d possible subsets\n', ...
    nFeatures, 2^nFeatures - 1);
fprintf('  -> Population size: %d\n', config.ga_popSize);
fprintf('  -> Max generations: %d\n', config.ga_maxGen);

%% Step 3.2: Define Fitness Function
% -------------------------------------------------------------------------
% The fitness function evaluates a binary chromosome by:
% 1. Extracting selected features
% 2. Training k-NN with 5-fold CV on the subset
% 3. Returning: error_rate + 0.01 * num_selected_features
% The small penalty (0.01) prefers simpler models when accuracy is similar.
% -------------------------------------------------------------------------

% Create the fitness function handle
fitnessFcn = @(chromosome) gaFitnessEval(chromosome, X_train, y_train, config);

%% Step 3.3: Configure and Run GA
% -------------------------------------------------------------------------
% GA Options:
% - Binary encoding via integer constraints (lower=0, upper=1)
% - Tournament selection with size 4
% - Single-point crossover
% - Bit-flip mutation
% - Elite count: 2 (best 2 survive unchanged)
% -------------------------------------------------------------------------

% GA options
ga_options = optimoptions('ga', ...
    'PopulationSize', config.ga_popSize, ...
    'MaxGenerations', config.ga_maxGen, ...
    'CrossoverFraction', config.ga_crossFrac, ...
    'EliteCount', 2, ...
    'SelectionFcn', @selectiontournament, ...
    'CrossoverFcn', @crossoversinglepoint, ...
    'MutationFcn', {@mutationuniform, 0.1}, ...
    'FitnessScalingFcn', @fitscalingrank, ...
    'Display', 'iter', ...
    'PlotFcn', {@gaplotbestf, @gaplotselection}, ...
    'UseParallel', false, ...
    'FunctionTolerance', 1e-6, ...
    'MaxStallGenerations', 20);

% Define bounds (binary: 0 or 1 for each feature)
lb = zeros(1, nFeatures);  % Lower bound
ub = ones(1, nFeatures);   % Upper bound
intCon = 1:nFeatures;      % All variables are integers

fprintf('\n[PHASE 3] Running Genetic Algorithm...\n');
fprintf('  -> This may take a few minutes. Evolving optimal feature subset...\n\n');

% Run GA
[best_chromosome, best_fitness, exitflag, output] = ga(...
    fitnessFcn, nFeatures, [], [], [], [], lb, ub, [], intCon, ga_options);

%% Step 3.4: Extract and Report Results
% -------------------------------------------------------------------------
% Convert the best binary chromosome to feature indices
% -------------------------------------------------------------------------

% Round to ensure binary (GA may produce near-0 or near-1 values)
best_chromosome = round(best_chromosome);

% Ensure at least 2 features are selected
if sum(best_chromosome) < 2
    % If GA selected too few, add the top 2 by univariate importance
    fprintf('  [WARNING] GA selected < 2 features. Adding top features...\n');
    % Compute univariate AUC for each feature
    aucs = zeros(1, nFeatures);
    for f = 1:nFeatures
        [~, ~, ~, aucs(f)] = perfcurve(y_train, X_train(:,f), 1);
    end
    [~, sorted_idx] = sort(aucs, 'descend');
    best_chromosome(sorted_idx(1:2)) = 1;
end

selected_features = find(best_chromosome == 1);
selected_names = config.featureNames(selected_features);

fprintf('\n============================================\n');
fprintf('  GENETIC ALGORITHM RESULTS\n');
fprintf('============================================\n');
fprintf('  Exit flag: %d (%s)\n', exitflag, output.message);
fprintf('  Generations completed: %d\n', output.generations);
fprintf('  Best fitness (error + penalty): %.4f\n', best_fitness);
fprintf('  Selected features (%d/%d):\n', length(selected_features), nFeatures);
for i = 1:length(selected_features)
    fprintf('    [%d] %s\n', selected_features(i), selected_names{i});
end
fprintf('============================================\n');

% Store results
ga_results = struct();
ga_results.best_chromosome = best_chromosome;
ga_results.best_fitness = best_fitness;
ga_results.selected_features = selected_features;
ga_results.selected_names = selected_names;
ga_results.generations = output.generations;
ga_results.exitflag = exitflag;

%% Step 3.5: Feature Importance Analysis (Post-GA)
% -------------------------------------------------------------------------
% To validate GA results, we also compute individual feature importance 
% using permutation importance: shuffle each feature and measure accuracy drop
% -------------------------------------------------------------------------
fprintf('\n[PHASE 3] Computing permutation importance for validation...\n');

% Train a baseline k-NN on all features
knn_all = fitcknn(X_train, y_train, 'NumNeighbors', config.knn_k, ...
    'Distance', 'euclidean');  % Use euclidean for speed here
baseline_loss = loss(knn_all, X_train, y_train);

perm_importance = zeros(1, nFeatures);
n_perm = 10;  % Number of permutation repetitions

for f = 1:nFeatures
    perm_losses = zeros(1, n_perm);
    for rep = 1:n_perm
        X_perm = X_train;
        X_perm(:, f) = X_train(randperm(size(X_train,1)), f);
        perm_losses(rep) = loss(knn_all, X_perm, y_train);
    end
    perm_importance(f) = mean(perm_losses) - baseline_loss;
end

%% Step 3.6: Generate GA Report Figures
% -------------------------------------------------------------------------

% ----- FIGURE 1: Feature Selection Summary -----
figure('Name', 'Phase 3: GA Feature Selection', ...
       'Position', [100, 100, 1400, 600], 'Color', 'w');

% Subplot 1: Selected vs Rejected features
subplot(1,3,1);
colors = zeros(nFeatures, 3);
for i = 1:nFeatures
    if best_chromosome(i) == 1
        colors(i,:) = [0.2, 0.7, 0.3];  % Green for selected
    else
        colors(i,:) = [0.8, 0.2, 0.2];  % Red for rejected
    end
end
b = barh(1:nFeatures, best_chromosome, 0.5);
b.FaceColor = 'flat';
b.CData = colors;
set(gca, 'YTick', 1:nFeatures, 'YTickLabel', config.featureNames, 'FontSize', 10);
xlabel('Selected (1) / Rejected (0)', 'FontSize', 11);
title('GA Feature Selection', 'FontSize', 13, 'FontWeight', 'bold');
xlim([-0.1, 1.5]);
grid on;

% Add labels
for i = 1:nFeatures
    if best_chromosome(i) == 1
        text(1.05, i, 'SELECTED', 'Color', [0, 0.5, 0], ...
            'FontWeight', 'bold', 'FontSize', 9);
    else
        text(0.05, i, 'REJECTED', 'Color', [0.6, 0, 0], ...
            'FontSize', 9);
    end
end

% Subplot 2: Permutation Importance
subplot(1,3,2);
[sorted_imp, sort_idx] = sort(perm_importance, 'descend');
colors_imp = zeros(nFeatures, 3);
for i = 1:nFeatures
    if best_chromosome(sort_idx(i)) == 1
        colors_imp(i,:) = [0.2, 0.7, 0.3];
    else
        colors_imp(i,:) = [0.7, 0.7, 0.7];
    end
end
b2 = barh(1:nFeatures, sorted_imp, 0.6);
b2.FaceColor = 'flat';
b2.CData = colors_imp;
set(gca, 'YTick', 1:nFeatures, 'YTickLabel', config.featureNames(sort_idx), ...
    'FontSize', 10);
xlabel('Permutation Importance (Δ Error)', 'FontSize', 11);
title('Feature Importance Validation', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

% Subplot 3: Feature subset comparison
subplot(1,3,3);
% Compare: All features vs GA-selected vs Top-3 (Glucose, BMI, Age)
subsets = {1:8, selected_features, [2, 6, 8]};
subset_names = {'All 8 Features', 'GA-Selected', 'Domain Expert (Glu+BMI+Age)'};
subset_errors = zeros(1, 3);

for s = 1:3
    feat_idx = subsets{s};
    cv_model = fitcknn(X_train(:, feat_idx), y_train, ...
        'NumNeighbors', config.knn_k, 'Distance', 'euclidean');
    cv_loss = crossval('mcr', X_train(:, feat_idx), y_train, ...
        'Predfun', @(xtr, ytr, xte) predict(fitcknn(xtr, ytr, ...
        'NumNeighbors', config.knn_k, 'Distance', 'euclidean'), xte), ...
        'KFold', 5);
    subset_errors(s) = cv_loss;
end

b3 = bar(1:3, (1-subset_errors)*100, 0.6);
b3.FaceColor = 'flat';
b3.CData = [0.5, 0.5, 0.5; 0.2, 0.7, 0.3; 0.2, 0.5, 0.8];
set(gca, 'XTickLabel', subset_names, 'XTickLabelRotation', 15, 'FontSize', 10);
ylabel('Accuracy (%)', 'FontSize', 11);
title('Feature Subset Comparison', 'FontSize', 13, 'FontWeight', 'bold');
ylim([60, 100]);
grid on;

% Add accuracy text on bars
for i = 1:3
    text(i, (1-subset_errors(i))*100 + 1, ...
        sprintf('%.1f%%', (1-subset_errors(i))*100), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
end

sgtitle('PHASE 3: Genetic Algorithm Feature Selection Report', ...
    'FontSize', 16, 'FontWeight', 'bold');

saveas(gcf, 'results/Phase3_GA_FeatureSelection.png');
saveas(gcf, 'results/Phase3_GA_FeatureSelection.fig');

% Save GA convergence figure (from GA's own plot)
if ishandle(findobj('Type', 'figure', 'Name', 'Genetic Algorithm'))
    saveas(findobj('Type', 'figure', 'Name', 'Genetic Algorithm'), ...
        'results/Phase3_GA_Convergence.png');
end

fprintf('\n[PHASE 3] COMPLETE - Feature selection finished.\n');
fprintf('  -> Best feature subset: {%s}\n', strjoin(selected_names, ', '));

end

%% ===================== FITNESS FUNCTION =====================
function error = gaFitnessEval(chromosome, X_train, y_train, config)
% GAFITNESS - Evaluates a binary chromosome for GA feature selection
% -------------------------------------------------------------------------
% Mathematical Basis:
% The fitness of a feature subset S is defined as:
%   f(S) = CV_Error(X[:, S], y; k-NN) + alpha * |S| / d
% where:
%   - CV_Error is the 5-fold cross-validation misclassification rate
%   - alpha = 0.01 is the complexity penalty coefficient
%   - |S| is the number of selected features
%   - d is the total number of features
%
% This penalizes complex models slightly, implementing Occam's Razor.
% -------------------------------------------------------------------------

    % Round to binary
    chromosome = round(chromosome);
    selected = find(chromosome == 1);
    
    % Penalize empty feature sets heavily
    if isempty(selected)
        error = 1.0;  % Maximum error
        return;
    end
    
    % Extract selected features
    X_subset = X_train(:, selected);
    
    % 5-fold cross-validation with k-NN
    try
        knn_model = fitcknn(X_subset, y_train, ...
            'NumNeighbors', config.knn_k, ...
            'Distance', 'euclidean');  % Euclidean for GA speed
        cv_model = crossval(knn_model, 'KFold', 5);
        cv_error = kfoldLoss(cv_model);
    catch
        cv_error = 1.0;  % If training fails, assign max error
    end
    
    % Complexity penalty (Occam's Razor)
    alpha = 0.01;
    complexity_penalty = alpha * length(selected) / size(X_train, 2);
    
    % Total fitness (minimize)
    error = cv_error + complexity_penalty;
end
