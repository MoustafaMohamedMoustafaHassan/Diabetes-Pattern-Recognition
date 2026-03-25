function shap_results = Phase5_SHAPAnalysis(X_train, y_train, X_test, y_test, ...
    selected_features, config)
%% =========================================================================
%  PHASE 5: EXPLAINABILITY ANALYSIS (SHAP VALUES)
%  =========================================================================
%  Mathematical Basis:
%  -------------------
%  SHAP (SHapley Additive exPlanations) values are based on cooperative 
%  game theory. For each prediction, SHAP assigns each feature a 
%  contribution phi_i such that:
%
%    f(x) = E[f(X)] + SUM_i phi_i(x)
%
%  where:
%    phi_i = SUM_{S subset of F\{i}} [|S|!(|F|-|S|-1)!/|F|!] * 
%            [f(S union {i}) - f(S)]
%
%  This is the Shapley value from game theory, where:
%  - F = set of all features
%  - S = a subset of features
%  - f(S) = model prediction using only features in S
%  - phi_i = the marginal contribution of feature i, averaged over all
%    possible feature orderings
%
%  Properties (guaranteed by Shapley axioms):
%  1. Efficiency: SUM_i phi_i = f(x) - E[f(X)]
%  2. Symmetry: Equal features get equal attribution
%  3. Linearity: Additive across model components
%  4. Null Player: Irrelevant features get phi = 0
%
%  Medical Significance:
%  A SHAP plot tells the clinician: "This patient was classified as 
%  high-risk because Glucose contributed +0.3 to the risk score while 
%  Age contributed -0.1, meaning the glucose pattern dominated the age 
%  pattern in this specific case."
%
%  References:
%  [1] Lundberg, S.M. & Lee, S.I. (2017). "A Unified Approach to 
%      Interpreting Model Predictions." NeurIPS.
%  [2] Shapley, L.S. (1953). "A Value for n-Person Games." Annals of 
%      Mathematics Studies, 28, 307-317.
%  =========================================================================

fprintf('[PHASE 5] Computing SHAP values for model explainability...\n');

% Extract selected features
X_train_sel = X_train(:, selected_features);
X_test_sel  = X_test(:, selected_features);
sel_names = config.featureNames(selected_features);
nFeatures_sel = length(selected_features);

%% Step 5.1: Train a Black-Box Model for SHAP
fprintf('[PHASE 5] Training ensemble model for SHAP computation...\n');

rf_model = TreeBagger(100, X_train_sel, y_train, ...
    'Method', 'classification', ...
    'OOBPrediction', 'on', ...
    'MinLeafSize', 5, ...
    'OOBPredictorImportance', 'on');

oob_error = oobError(rf_model);
fprintf('  -> Random Forest OOB error: %.4f\n', oob_error(end));

rf_importance = rf_model.OOBPermutedPredictorDeltaError;
fprintf('  -> RF feature importance computed.\n');

%% Step 5.2: Compute SHAP Values
fprintf('[PHASE 5] Computing SHAP values (this may take a moment)...\n');

n_shap_samples = min(50, size(X_test_sel, 1));

try
    % Try MATLAB's built-in shapley (R2021a+)
    predictFcn = @(X) safeRFPredict(rf_model, X);
explainer = shapley(predictFcn, X_train_sel, 'Method', 'interventional');
    
    shap_values = zeros(n_shap_samples, nFeatures_sel);
    
    for i = 1:n_shap_samples
        queryPoint = X_test_sel(i, :);
        result = fit(explainer, queryPoint);
        shap_values(i, :) = result.ShapleyValues.ShapleyValue';
    end
    
    fprintf('  -> SHAP values computed using MATLAB shapley() for %d samples.\n', ...
        n_shap_samples);
    shap_method = 'MATLAB shapley()';
    
catch ME
    fprintf('  -> MATLAB shapley() not available: %s\n', ME.message);
    fprintf('  -> Computing approximate SHAP via permutation method...\n');
    
    shap_values = computePermutationSHAP(X_train_sel, y_train, ...
        X_test_sel(1:n_shap_samples, :), rf_model, 10);
    shap_method = 'Permutation-based approximation';
    
    fprintf('  -> Approximate SHAP values computed for %d samples.\n', ...
        n_shap_samples);
end

%% Step 5.3: Global SHAP Analysis
fprintf('\n[PHASE 5] Global SHAP Feature Importance:\n');

global_shap = mean(abs(shap_values), 1);
[sorted_shap, sort_idx] = sort(global_shap, 'descend');

for i = 1:nFeatures_sel
    fprintf('  %d. %-20s: %.4f\n', i, sel_names{sort_idx(i)}, sorted_shap(i));
end

%% Step 5.4: Store SHAP Results
shap_results = struct();
shap_results.shap_values = shap_values;
shap_results.global_importance = global_shap;
shap_results.sort_order = sort_idx;
shap_results.method = shap_method;
shap_results.rf_importance = rf_importance;
shap_results.n_samples = n_shap_samples;
shap_results.feature_names = sel_names;

%% Step 5.5: Generate SHAP Report Figures

% ----- FIGURE 1: Global SHAP Summary -----
figure('Name', 'Phase 5: SHAP Global Analysis', ...
       'Position', [50, 50, 1500, 600], 'Color', 'w');

% Subplot 1: Mean |SHAP| bar chart
subplot(1,3,1);
barh(1:nFeatures_sel, global_shap(sort_idx(end:-1:1)), 0.6, ...
    'FaceColor', [0.2, 0.5, 0.8]);
set(gca, 'YTick', 1:nFeatures_sel, ...
    'YTickLabel', sel_names(sort_idx(end:-1:1)), 'FontSize', 11);
xlabel('Mean |SHAP Value|', 'FontSize', 12);
title('Global Feature Importance (SHAP)', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

% Subplot 2: SHAP Beeswarm Plot
subplot(1,3,2);
hold on;
for f = 1:nFeatures_sel
    f_idx = sort_idx(nFeatures_sel - f + 1);
    sv = shap_values(:, f_idx);
    fv = X_test_sel(1:n_shap_samples, f_idx);
    jitter = (rand(n_shap_samples, 1) - 0.5) * 0.3;
    scatter(sv, f + jitter, 20, fv, 'filled', 'MarkerFaceAlpha', 0.6);
end
colormap(gca, 'jet');
cb = colorbar;
cb.Label.String = 'Feature Value (normalized)';
set(gca, 'YTick', 1:nFeatures_sel, ...
    'YTickLabel', sel_names(sort_idx(end:-1:1)), 'FontSize', 10);
xlabel('SHAP Value (Impact on Prediction)', 'FontSize', 11);
xline(0, '--k', 'LineWidth', 1);
title('SHAP Beeswarm Plot', 'FontSize', 13, 'FontWeight', 'bold');
grid on;
ylim([0.5, nFeatures_sel + 0.5]);

% Subplot 3: RF Feature Importance Comparison
subplot(1,3,3);
[~, rf_sort_idx] = sort(rf_importance, 'descend');
barh(1:nFeatures_sel, rf_importance(rf_sort_idx(end:-1:1)), 0.6, ...
    'FaceColor', [0.8, 0.4, 0.2]);
set(gca, 'YTick', 1:nFeatures_sel, ...
    'YTickLabel', sel_names(rf_sort_idx(end:-1:1)), 'FontSize', 11);
xlabel('OOB Permutation Importance', 'FontSize', 12);
title('RF Feature Importance (Validation)', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

sgtitle(sprintf('PHASE 5: SHAP Explainability Report (%s)', shap_method), ...
    'FontSize', 16, 'FontWeight', 'bold');

saveas(gcf, 'results/Phase5_SHAP_Global.png');
saveas(gcf, 'results/Phase5_SHAP_Global.fig');

% ----- FIGURE 2: Individual Patient Explanations -----
figure('Name', 'Phase 5: Individual SHAP Explanations', ...
       'Position', [100, 100, 1400, 700], 'Color', 'w');

% FIX: Safely get RF predictions using helper function
rf_pred_probs = safeRFPredict(rf_model, X_test_sel(1:n_shap_samples, :));

% Find interesting cases
correct_diabetic = find(y_test(1:n_shap_samples) == 1 & ...
    rf_pred_probs > 0.5, 1, 'first');
correct_healthy = find(y_test(1:n_shap_samples) == 0 & ...
    rf_pred_probs <= 0.5, 1, 'first');
misclass = find((y_test(1:n_shap_samples) == 1 & rf_pred_probs <= 0.5) | ...
    (y_test(1:n_shap_samples) == 0 & rf_pred_probs > 0.5));

cases = [];
case_titles = {};

if ~isempty(correct_diabetic)
    cases = [cases, correct_diabetic];
    case_titles{end+1} = 'Correctly Identified: DIABETIC';
end
if ~isempty(correct_healthy)
    cases = [cases, correct_healthy];
    case_titles{end+1} = 'Correctly Identified: HEALTHY';
end
if length(misclass) >= 1
    cases = [cases, misclass(1)];
    if y_test(misclass(1)) == 1
        case_titles{end+1} = 'MISSED: Diabetic classified as Healthy';
    else
        case_titles{end+1} = 'FALSE ALARM: Healthy classified as Diabetic';
    end
end
if length(misclass) >= 2
    cases = [cases, misclass(2)];
    if y_test(misclass(2)) == 1
        case_titles{end+1} = 'MISSED: Diabetic classified as Healthy';
    else
        case_titles{end+1} = 'FALSE ALARM: Healthy classified as Diabetic';
    end
end

% Fill remaining slots
while length(cases) < 4
    new_case = randi(n_shap_samples);
    if ~ismember(new_case, cases)
        cases = [cases, new_case];
        if y_test(new_case) == 1
            case_titles{end+1} = sprintf('Patient #%d (Diabetic)', new_case);
        else
            case_titles{end+1} = sprintf('Patient #%d (Healthy)', new_case);
        end
    end
end

% Plot individual SHAP waterfall for each case
for p = 1:min(4, length(cases))
    subplot(2,2,p);
    
    case_idx = cases(p);
    sv = shap_values(case_idx, :);
    
    [~, sv_sort] = sort(abs(sv), 'descend');
    sv_sorted = sv(sv_sort);
    names_sorted = sel_names(sv_sort);
    
    colors = zeros(nFeatures_sel, 3);
    for f = 1:nFeatures_sel
        if sv_sorted(f) > 0
            colors(f, :) = [0.9, 0.3, 0.3];
        else
            colors(f, :) = [0.2, 0.6, 0.86];
        end
    end
    
    bh = barh(1:nFeatures_sel, sv_sorted(end:-1:1), 0.6);
    bh.FaceColor = 'flat';
    bh.CData = colors(end:-1:1, :);
    set(gca, 'YTick', 1:nFeatures_sel, ...
        'YTickLabel', names_sorted(end:-1:1), 'FontSize', 9);
    xlabel('SHAP Value', 'FontSize', 10);
    xline(0, '--k');
    
    for f = 1:nFeatures_sel
        f_display = nFeatures_sel - f + 1;
        val = X_test_sel(case_idx, sv_sort(f));
        text(max(abs(sv_sorted))*1.1*sign(sv_sorted(f)), f_display, ...
            sprintf('(%.2f)', val), 'FontSize', 8, ...
            'HorizontalAlignment', 'center');
    end
    
    title(case_titles{p}, 'FontSize', 11, 'FontWeight', 'bold');
    grid on;
end

sgtitle('PHASE 5: Individual Patient SHAP Explanations', ...
    'FontSize', 16, 'FontWeight', 'bold');

saveas(gcf, 'results/Phase5_SHAP_Individual.png');
saveas(gcf, 'results/Phase5_SHAP_Individual.fig');

% ----- FIGURE 3: SHAP Dependence Plots -----
figure('Name', 'Phase 5: SHAP Dependence Plots', ...
       'Position', [150, 50, 1200, 500], 'Color', 'w');

for f = 1:min(2, nFeatures_sel)
    subplot(1,2,f);
    f_idx = sort_idx(f);
    
    scatter(X_test_sel(1:n_shap_samples, f_idx), ...
            shap_values(:, f_idx), 40, ...
            y_test(1:n_shap_samples), 'filled', 'MarkerFaceAlpha', 0.7);
    
    colormap(gca, [0.2, 0.6, 0.86; 0.9, 0.3, 0.3]);
    cb = colorbar;
    cb.Ticks = [0, 1];
    cb.TickLabels = {'Healthy', 'Diabetic'};
    
    hold on;
    p_coeff = polyfit(X_test_sel(1:n_shap_samples, f_idx), ...
                     shap_values(:, f_idx), 2);
    x_range = linspace(min(X_test_sel(1:n_shap_samples, f_idx)), ...
                       max(X_test_sel(1:n_shap_samples, f_idx)), 100);
    plot(x_range, polyval(p_coeff, x_range), 'k--', 'LineWidth', 2);
    
    xlabel(sprintf('%s (Normalized)', sel_names{f_idx}), 'FontSize', 12);
    ylabel('SHAP Value', 'FontSize', 12);
    title(sprintf('SHAP Dependence: %s', sel_names{f_idx}), ...
        'FontSize', 13, 'FontWeight', 'bold');
    grid on;
end

sgtitle('PHASE 5: SHAP Feature Dependence Analysis', ...
    'FontSize', 16, 'FontWeight', 'bold');

saveas(gcf, 'results/Phase5_SHAP_Dependence.png');
saveas(gcf, 'results/Phase5_SHAP_Dependence.fig');

fprintf('\n[PHASE 5] COMPLETE - Explainability analysis finished.\n');
fprintf('  -> 3 SHAP report figures saved to results/\n');
fprintf('  -> Method used: %s\n', shap_method);

end

%% ===================== HELPER FUNCTIONS =====================
function probs = safeRFPredict(rf_model, X)
% SAFERFPREDICT - Safely get P(class=1) from TreeBagger predictions
% Handles the cell-array output format of TreeBagger.predict robustly.
    [~, scores] = predict(rf_model, X);
    probs = scores(:, 2);  % P(class=1)
end

function shap_values = computePermutationSHAP(X_train, y_train, X_query, model, n_perm)
% COMPUTEPERMUTATIONSHAP - Approximate SHAP values via permutation method
% FIX: Improved robustness of str2double handling for TreeBagger output

    n_samples = size(X_query, 1);
    n_features = size(X_query, 2);
    shap_values = zeros(n_samples, n_features);
    
    for q = 1:n_samples
        x = X_query(q, :);
        
        for perm = 1:n_perm
            order = randperm(n_features);
            
            % Random background sample
            bg_idx = randi(size(X_train, 1));
            x_with = X_train(bg_idx, :);
            x_without = x_with;
            
            for pos = 1:n_features
                f = order(pos);
                
                x_with(f) = x(f);
                
                % FIX: Use safer score extraction
                [~, score_with] = predict(model, x_with);
                [~, score_without] = predict(model, x_without);
                
                pw = score_with(2);  % P(class=1) - scores are numeric from TreeBagger
                pwo = score_without(2);
                
                if isnan(pw), pw = 0.5; end
                if isnan(pwo), pwo = 0.5; end
                
                shap_values(q, f) = shap_values(q, f) + (pw - pwo);
                
                x_without(f) = x(f);
            end
        end
        
        shap_values(q, :) = shap_values(q, :) / n_perm;
    end
end
