function Phase6_AppDesigner()
%% =========================================================================
%  PHASE 6: INTERACTIVE CLINICAL DIAGNOSTIC APPLICATION
%  =========================================================================
%  Purpose:
%  --------
%  This application provides a clinical interface for the Pattern 
%  Recognition system. It allows healthcare professionals to:
%  1. Input patient biomarker data
%  2. Receive instant risk classification (At Risk / Normal)
%  3. View the SHAP-based explanation of why the system made its decision
%  4. Visualize the patient's position in the feature space
%
%  Design Philosophy:
%  -----------------
%  The interface follows Evidence-Based Design principles:
%  - Traffic-light color coding (Green=Safe, Red=Risk)
%  - Clear numerical confidence levels
%  - Transparent feature attribution (SHAP waterfall)
%  - Comparison to population statistics
%
%  Technical Implementation:
%  - Built with MATLAB App Designer (uifigure + uicomponents)
%  - Loads pre-trained models from results/trained_system.mat
%  - Processes input through the same pipeline as training data
%
%  FIX Bug #1 & #2: This version correctly normalizes new patient data
%  using the ORIGINAL-SCALE normalization parameters (mu, sigma) stored
%  during Phase 1 training. The previous version erroneously used 
%  statistics from already-normalized data, producing meaningless results.
%  =========================================================================

%% Load Pre-trained System
fprintf('[PHASE 6] Loading trained models...\n');

if ~isfile('results/trained_system.mat')
    error(['Trained system not found. Please run main_pipeline.m first ' ...
           'to train the models and save them.']);
end

S = load('results/trained_system.mat');
knn_model = S.knn_model;
nb_model = S.nb_model;
selected_features = S.selected_features;
config = S.config;
ensemble_results = S.ensemble_results;
X_train = S.X_train;
y_train = S.y_train;

% FIX Bug #1: Load normalization parameters (original-scale mu and sigma)
% and the imputed (pre-normalization) data for correct percentile computation
imputation_stats = S.imputation_stats;
norm_mu = imputation_stats.mu;       % Mean of each feature in ORIGINAL scale
norm_sigma = imputation_stats.sigma; % Std of each feature in ORIGINAL scale

% FIX: Load raw/imputed data for population percentile comparison
if isfield(S, 'X_imputed')
    X_imputed = S.X_imputed;
else
    % Fallback: if X_imputed not saved, use X_raw
    X_imputed = S.X_raw;
end

% Load additional models if available
svm_model = [];
rf_model_cls = [];
lr_model = [];
if isfield(S, 'svm_model'), svm_model = S.svm_model; end
if isfield(S, 'rf_model_cls'), rf_model_cls = S.rf_model_cls; end
if isfield(S, 'lr_model'), lr_model = S.lr_model; end

if isfield(S, 'shap_results')
    shap_results = S.shap_results;
else
    shap_results = [];
end

sel_names = config.featureNames(selected_features);
n_models = 2;  % At minimum: kNN + NB
if ~isempty(svm_model), n_models = n_models + 1; end
if ~isempty(rf_model_cls), n_models = n_models + 1; end
if ~isempty(lr_model), n_models = n_models + 1; end

fprintf('  -> Models loaded successfully (%d classifiers).\n', n_models);
fprintf('  -> Selected features: {%s}\n', strjoin(sel_names, ', '));
fprintf('  -> Ensemble AUC: %.3f\n', ensemble_results.auc);
fprintf('  -> Normalization params loaded (original-scale mu, sigma).\n');

%% Create the Application Figure
fig = uifigure('Name', 'Diabetes Pattern Recognition - Clinical Diagnostic Tool', ...
    'Position', [100, 50, 1400, 800], ...
    'Color', [0.95, 0.97, 1.0], ...
    'Resize', 'on');

% ===== TITLE PANEL =====
titlePanel = uipanel(fig, 'Position', [10, 740, 1380, 55], ...
    'BackgroundColor', [0.15, 0.25, 0.45], ...
    'BorderType', 'none');
uilabel(titlePanel, 'Position', [20, 5, 800, 40], ...
    'Text', 'Advanced Pattern Recognition: Diabetes Risk Assessment', ...
    'FontSize', 20, 'FontWeight', 'bold', 'FontColor', 'white');
uilabel(titlePanel, 'Position', [850, 10, 500, 30], ...
    'Text', sprintf('Ensemble Accuracy: %.1f%% | AUC: %.3f | Models: %d', ...
    ensemble_results.accuracy*100, ensemble_results.auc, n_models), ...
    'FontSize', 13, 'FontColor', [0.7, 0.9, 1.0], ...
    'HorizontalAlignment', 'right');

% ===== INPUT PANEL (Left Side) =====
inputPanel = uipanel(fig, 'Title', 'Patient Biomarker Input', ...
    'Position', [10, 280, 400, 455], ...
    'FontSize', 14, 'FontWeight', 'bold', ...
    'BackgroundColor', [1, 1, 1]);

featureLabels = {'Pregnancies', 'Glucose (mg/dL)', 'Blood Pressure (mmHg)', ...
                 'Skin Thickness (mm)', 'Insulin (mu U/ml)', ...
                 'BMI (kg/m2)', 'Diabetes Pedigree', 'Age (years)'};
defaultValues = {'1', '120', '70', '20', '80', '25.0', '0.5', '35'};
normalRanges  = {'0-17', '70-140', '60-90', '10-50', '15-276', ...
                 '18.5-35', '0.08-2.42', '21-81'};

inputFields = cell(1, 8);
yPos = 385;
for i = 1:8
    uilabel(inputPanel, 'Position', [15, yPos, 200, 22], ...
        'Text', featureLabels{i}, 'FontSize', 11, 'FontWeight', 'bold');
    
    inputFields{i} = uieditfield(inputPanel, 'numeric', ...
        'Position', [220, yPos, 80, 25], ...
        'Value', str2double(defaultValues{i}), ...
        'FontSize', 11);
    
    uilabel(inputPanel, 'Position', [310, yPos, 80, 22], ...
        'Text', sprintf('[%s]', normalRanges{i}), ...
        'FontSize', 9, 'FontColor', [0.5, 0.5, 0.5]);
    
    yPos = yPos - 45;
end

% ===== ANALYZE BUTTON =====
analyzeBtn = uibutton(inputPanel, 'push', ...
    'Text', 'Analyze Clinical Pattern', ...
    'Position', [40, 15, 320, 50], ...
    'FontSize', 15, 'FontWeight', 'bold', ...
    'BackgroundColor', [0.2, 0.5, 0.8], ...
    'FontColor', 'white');

% ===== RESULT PANEL (Right Side - Top) =====
resultPanel = uipanel(fig, 'Title', 'Diagnostic Result', ...
    'Position', [420, 500, 560, 235], ...
    'FontSize', 14, 'FontWeight', 'bold', ...
    'BackgroundColor', [1, 1, 1]);

resultLabel = uilabel(resultPanel, 'Position', [20, 130, 520, 60], ...
    'Text', 'Awaiting patient data...', ...
    'FontSize', 28, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', ...
    'FontColor', [0.5, 0.5, 0.5]);

confidenceLabel = uilabel(resultPanel, 'Position', [20, 90, 520, 30], ...
    'Text', '', 'FontSize', 14, ...
    'HorizontalAlignment', 'center');

detailLabel = uilabel(resultPanel, 'Position', [20, 10, 520, 80], ...
    'Text', ['Enter patient biomarker values and click ' ...
             '"Analyze Clinical Pattern" to begin diagnosis.'], ...
    'FontSize', 11, 'WordWrap', 'on', ...
    'FontColor', [0.4, 0.4, 0.4]);

% ===== SHAP PLOT PANEL =====
shapPanel = uipanel(fig, 'Title', 'Feature Attribution (SHAP Analysis)', ...
    'Position', [420, 280, 560, 215], ...
    'FontSize', 14, 'FontWeight', 'bold', ...
    'BackgroundColor', [1, 1, 1]);

shapAxes = uiaxes(shapPanel, 'Position', [10, 10, 540, 175]);
title(shapAxes, 'Feature contributions will appear here', 'FontSize', 11);

% ===== POPULATION COMPARISON PANEL =====
popPanel = uipanel(fig, 'Title', 'Patient vs Population Comparison', ...
    'Position', [10, 10, 970, 265], ...
    'FontSize', 14, 'FontWeight', 'bold', ...
    'BackgroundColor', [1, 1, 1]);

popAxes = uiaxes(popPanel, 'Position', [10, 10, 950, 225]);
title(popAxes, 'Population comparison will appear after analysis', 'FontSize', 11);

% ===== MODEL INFO PANEL =====
infoPanel = uipanel(fig, 'Title', 'System Information', ...
    'Position', [990, 280, 400, 455], ...
    'FontSize', 14, 'FontWeight', 'bold', ...
    'BackgroundColor', [1, 1, 1]);

% Build info text with available models
modelInfoStr = sprintf([...
    'System Architecture:\n' ...
    '---------------------\n' ...
    'Classifier 1: k-NN (k=%d)\n' ...
    '  Distance: %s\n' ...
    '  CV Accuracy: %.1f%%\n\n' ...
    'Classifier 2: Naive Bayes\n' ...
    '  Distribution: Gaussian\n' ...
    '  CV Accuracy: %.1f%%\n'], ...
    ensemble_results.best_k, ensemble_results.distance_used, ...
    ensemble_results.knn_cv_accuracy*100, ...
    ensemble_results.nb_cv_accuracy*100);

if ~isempty(svm_model)
    modelInfoStr = [modelInfoStr, sprintf([...
        '\nClassifier 3: SVM (RBF)\n' ...
        '  CV Accuracy: %.1f%%\n'], ensemble_results.svm_cv_accuracy*100)];
end
if ~isempty(rf_model_cls)
    modelInfoStr = [modelInfoStr, sprintf([...
        '\nClassifier 4: Random Forest\n' ...
        '  CV Accuracy: %.1f%%\n'], ensemble_results.rf_cv_accuracy*100)];
end
if ~isempty(lr_model)
    modelInfoStr = [modelInfoStr, sprintf([...
        '\nClassifier 5: Logistic Regression\n' ...
        '  CV Accuracy: %.1f%%\n'], ensemble_results.lr_cv_accuracy*100)];
end

modelInfoStr = [modelInfoStr, sprintf([...
    '\nEnsemble: Weighted Voting\n' ...
    'Selected Features (%d/%d):\n' ...
    '  %s\n\n' ...
    'Split: 70/10/20 (Train/Val/Test)\n' ...
    'Feature Selection: Genetic Algorithm\n'], ...
    length(selected_features), 8, ...
    strjoin(sel_names, ', '))];

uitextarea(infoPanel, 'Position', [10, 10, 380, 410], ...
    'Value', strsplit(modelInfoStr, '\n'), ...
    'FontSize', 11, 'Editable', 'off', ...
    'FontName', 'Consolas', ...
    'BackgroundColor', [0.98, 0.98, 0.98]);

% ===== SAMPLE BUTTONS =====
samplePanel = uipanel(fig, 'Title', 'Quick Test Samples', ...
    'Position', [990, 10, 400, 265], ...
    'FontSize', 14, 'FontWeight', 'bold', ...
    'BackgroundColor', [1, 1, 1]);

uilabel(samplePanel, 'Position', [10, 210, 380, 20], ...
    'Text', 'Load pre-defined patient profiles:', ...
    'FontSize', 11, 'FontColor', [0.4, 0.4, 0.4]);

highRiskBtn = uibutton(samplePanel, 'push', ...
    'Text', 'High Risk Profile (Glucose=180, BMI=38)', ...
    'Position', [15, 170, 370, 35], ...
    'FontSize', 11, 'BackgroundColor', [1, 0.85, 0.85]);

lowRiskBtn = uibutton(samplePanel, 'push', ...
    'Text', 'Low Risk Profile (Glucose=90, BMI=22)', ...
    'Position', [15, 125, 370, 35], ...
    'FontSize', 11, 'BackgroundColor', [0.85, 1, 0.85]);

borderBtn = uibutton(samplePanel, 'push', ...
    'Text', 'Borderline Profile (Glucose=130, BMI=30)', ...
    'Position', [15, 80, 370, 35], ...
    'FontSize', 11, 'BackgroundColor', [1, 1, 0.85]);

youngBtn = uibutton(samplePanel, 'push', ...
    'Text', 'Young + Family History (Age=25, Pedigree=1.8)', ...
    'Position', [15, 35, 370, 35], ...
    'FontSize', 11, 'BackgroundColor', [0.9, 0.85, 1]);

%% ===== CALLBACK FUNCTIONS =====
analyzeBtn.ButtonPushedFcn = @(~, ~) analyzePatient();
highRiskBtn.ButtonPushedFcn = @(~, ~) loadSample([6, 180, 85, 35, 200, 38, 1.2, 55]);
lowRiskBtn.ButtonPushedFcn  = @(~, ~) loadSample([1, 90, 65, 15, 60, 22, 0.2, 28]);
borderBtn.ButtonPushedFcn   = @(~, ~) loadSample([3, 130, 75, 25, 120, 30, 0.7, 42]);
youngBtn.ButtonPushedFcn    = @(~, ~) loadSample([0, 105, 68, 18, 90, 24, 1.8, 25]);

%% ===== NESTED FUNCTIONS =====

    function loadSample(values)
        for idx = 1:8
            inputFields{idx}.Value = values(idx);
        end
        analyzePatient();
    end

    function analyzePatient()
        % Collect raw input values (original scale)
        patient_raw = zeros(1, 8);
        for idx = 1:8
            patient_raw(idx) = inputFields{idx}.Value;
        end
        
        % Validate inputs
        if any(patient_raw < 0)
            resultLabel.Text = 'Invalid Input';
            resultLabel.FontColor = [0.8, 0, 0];
            detailLabel.Text = 'All values must be non-negative.';
            return;
        end
        
        % ====================================================================
        % FIX Bug #1 & #2: CORRECT NORMALIZATION
        % ====================================================================
        % We normalize the raw patient data using the ORIGINAL-SCALE parameters
        % (mu, sigma) that were computed from the imputed training data in 
        % Phase 1 BEFORE z-scoring. These are stored in imputation_stats.
        %
        % Z-score formula: z_i = (x_i - mu_i) / sigma_i
        %
        % Previously, the code used mean(X_train) and std(X_train) where 
        % X_train was already normalized (mean~0, std~1), so raw patient 
        % values were passed directly to classifiers — completely wrong.
        % ====================================================================
        
        % Normalize ALL 8 features using original-scale parameters
        patient_normalized_all = (patient_raw - norm_mu) ./ norm_sigma;
        
        % Select only the GA-selected features for prediction
        patient_normalized = patient_normalized_all(selected_features);
        
        % ---- k-NN Prediction ----
        [knn_pred, knn_scores] = predict(knn_model, patient_normalized);
        
        % ---- Naive Bayes Prediction ----
        [nb_pred, nb_scores] = predict(nb_model, patient_normalized);
        
        % ---- Weighted Ensemble ----
        w = ensemble_results.weights;
        ens_scores = w(1) * knn_scores + w(2) * nb_scores;
        
        % Add SVM, RF, LR if available
        model_idx = 3;
        if ~isempty(svm_model) && model_idx <= length(w)
            [~, svm_sc] = predict(svm_model, patient_normalized);
            ens_scores = ens_scores + w(model_idx) * svm_sc;
            model_idx = model_idx + 1;
        end
        if ~isempty(rf_model_cls) && model_idx <= length(w)
            [~, rf_sc] = predict(rf_model_cls, patient_normalized);
            ens_scores = ens_scores + w(model_idx) * rf_sc;
            model_idx = model_idx + 1;
        end
        if ~isempty(lr_model) && model_idx <= length(w)
            lr_prob = predict(lr_model, patient_normalized);
            lr_sc = [1 - lr_prob, lr_prob];
            ens_scores = ens_scores + w(model_idx) * lr_sc;
        end
        
        if ens_scores(2) > 0.5
            prediction = 1;
            risk_label = 'AT RISK FOR DIABETES';
            risk_color = [0.85, 0.15, 0.15];
            panel_color = [1, 0.92, 0.92];
        else
            prediction = 0;
            risk_label = 'NORMAL (Low Risk)';
            risk_color = [0.1, 0.6, 0.2];
            panel_color = [0.92, 1, 0.92];
        end
        
        % Update result display
        resultPanel.BackgroundColor = panel_color;
        resultLabel.Text = risk_label;
        resultLabel.FontColor = risk_color;
        
        confidence = max(ens_scores) * 100;
        confidenceLabel.Text = sprintf('Confidence: %.1f%% | P(Diabetic)=%.3f | P(Healthy)=%.3f', ...
            confidence, ens_scores(2), ens_scores(1));
        
        detailLabel.Text = sprintf(['k-NN vote: %s (%.1f%%) | NB vote: %s (%.1f%%)\n' ...
            'Agreement: %s'], ...
            iif(knn_pred == 1, 'Diabetic', 'Healthy'), max(knn_scores)*100, ...
            iif(nb_pred == 1, 'Diabetic', 'Healthy'), max(nb_scores)*100, ...
            iif(knn_pred == nb_pred, 'YES', 'NO (using weighted score)'));
        
        % ---- SHAP Waterfall Plot ----
        cla(shapAxes);
        
        % Compute feature contributions via perturbation
        base_value = mean(y_train);
        contributions = zeros(1, length(selected_features));
        for f = 1:length(selected_features)
            perturbed = patient_normalized;
            perturbed(f) = 0;  % Set to mean (z-score = 0)
            [~, pert_scores_knn] = predict(knn_model, perturbed);
            [~, pert_scores_nb] = predict(nb_model, perturbed);
            pert_ens = w(1) * pert_scores_knn + w(2) * pert_scores_nb;
            contributions(f) = ens_scores(2) - pert_ens(2);
        end
        
        [~, sort_order] = sort(abs(contributions), 'descend');
        sorted_contrib = contributions(sort_order);
        sorted_names = sel_names(sort_order);
        
        colors_shap = zeros(length(sorted_contrib), 3);
        for f = 1:length(sorted_contrib)
            if sorted_contrib(f) > 0
                colors_shap(f,:) = [0.9, 0.3, 0.3];
            else
                colors_shap(f,:) = [0.2, 0.6, 0.86];
            end
        end
        
        bh = barh(shapAxes, 1:length(sorted_contrib), sorted_contrib(end:-1:1), 0.6);
        bh.FaceColor = 'flat';
        bh.CData = colors_shap(end:-1:1, :);
        set(shapAxes, 'YTick', 1:length(sorted_contrib), ...
            'YTickLabel', sorted_names(end:-1:1), 'FontSize', 9);
        xlabel(shapAxes, 'Contribution to P(Diabetic)', 'FontSize', 10);
        xline(shapAxes, 0, '--k');
        title(shapAxes, sprintf('Feature Attribution (Base=%.3f, Pred=%.3f)', ...
            base_value, ens_scores(2)), 'FontSize', 11, 'FontWeight', 'bold');
        grid(shapAxes, 'on');
        
        % ====================================================================
        % FIX Bug #1: CORRECT POPULATION PERCENTILE COMPUTATION
        % ====================================================================
        % Use X_imputed (original-scale, pre-normalization data) for 
        % population comparison. Previously used X_train which was already
        % z-scored, making percentiles meaningless.
        % ====================================================================
        cla(popAxes);
        
        all_names = config.featureNames;
        patient_percentiles = zeros(1, 8);
        for f = 1:8
            % FIX: Compare raw patient values against ORIGINAL-SCALE population
            all_vals = sort(X_imputed(:, f));
            patient_percentiles(f) = sum(all_vals <= patient_raw(f)) / length(all_vals) * 100;
        end
        
        patient_percentiles = max(0, min(100, patient_percentiles));
        
        b_pop = bar(popAxes, 1:8, patient_percentiles, 0.6);
        b_pop.FaceColor = 'flat';
        for f = 1:8
            if patient_percentiles(f) > 75
                b_pop.CData(f,:) = [0.9, 0.3, 0.3];
            elseif patient_percentiles(f) > 50
                b_pop.CData(f,:) = [1, 0.7, 0.2];
            else
                b_pop.CData(f,:) = [0.2, 0.7, 0.4];
            end
        end
        
        set(popAxes, 'XTick', 1:8, 'XTickLabel', all_names, ...
            'XTickLabelRotation', 25, 'FontSize', 10);
        ylabel(popAxes, 'Percentile in Training Population (%)', 'FontSize', 11);
        title(popAxes, 'Patient Biomarker Percentiles vs Training Population', ...
            'FontSize', 12, 'FontWeight', 'bold');
        yline(popAxes, 50, '--', 'Median', 'LineWidth', 1.5, 'FontSize', 10);
        yline(popAxes, 75, ':r', '75th Percentile', 'LineWidth', 1, 'FontSize', 9);
        ylim(popAxes, [0, 105]);
        grid(popAxes, 'on');
        
        for f = 1:8
            text(popAxes, f, patient_percentiles(f) + 3, ...
                sprintf('%.0f%%', patient_percentiles(f)), ...
                'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
                'FontSize', 9);
        end
        
        fprintf('[APP] Patient analyzed: %s (P(Diabetic)=%.3f)\n', ...
            risk_label, ens_scores(2));
    end

fprintf('[PHASE 6] Application launched successfully!\n');
fprintf('  -> Enter patient data and click "Analyze Clinical Pattern"\n');

end

%% ===== UTILITY FUNCTIONS =====
function result = iif(condition, true_val, false_val)
% Inline if-else function
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
