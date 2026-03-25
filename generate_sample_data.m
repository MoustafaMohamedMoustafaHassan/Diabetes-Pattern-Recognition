function generate_sample_data()
%% GENERATE_SAMPLE_DATA - Creates the Pima Indians Diabetes Dataset
%  =========================================================================
%  This script generates the diabetes.csv file based on the well-known
%  Pima Indians Diabetes Database from the UCI Machine Learning Repository.
%  
%  Original source: National Institute of Diabetes and Digestive and 
%  Kidney Diseases. Smith, J.W., et al. (1988).
%
%  If you have the original CSV, place it in the data/ directory.
%  Otherwise, this script generates a statistically faithful synthetic
%  version for development and testing purposes.
%  =========================================================================

fprintf('Generating Pima Indians Diabetes Dataset...\n');

% Check if real data already exists
if isfile('data/diabetes.csv')
    fprintf('  -> data/diabetes.csv already exists. Skipping generation.\n');
    return;
end

% Create data directory
if ~isfolder('data')
    mkdir('data');
end

% The Pima Indians Diabetes Dataset has known statistical properties.
% We generate synthetic data matching these distributions.
% Real dataset available at: https://www.kaggle.com/uciml/pima-indians-diabetes-database

rng(42);  % Reproducibility
n_total = 768;
n_diabetic = 268;  % ~35% positive
n_healthy = n_total - n_diabetic;

% Class 0 (Healthy) - Based on published distribution statistics
X0 = zeros(n_healthy, 8);
X0(:,1) = max(0, round(normrnd(3.3, 2.5, n_healthy, 1)));       % Pregnancies
X0(:,2) = max(0, round(normrnd(110, 24, n_healthy, 1)));         % Glucose
X0(:,3) = max(0, round(normrnd(68, 16, n_healthy, 1)));          % BloodPressure
X0(:,4) = max(0, round(normrnd(27, 13, n_healthy, 1)));          % SkinThickness
X0(:,5) = max(0, round(normrnd(100, 90, n_healthy, 1)));         % Insulin
X0(:,6) = max(0, round(normrnd(30.1, 7.5, n_healthy, 1)*10)/10);% BMI
X0(:,7) = max(0.078, round(normrnd(0.43, 0.27, n_healthy, 1)*1000)/1000); % DPF
X0(:,8) = max(21, round(normrnd(31, 10, n_healthy, 1)));         % Age

% Class 1 (Diabetic) - Shifted distributions reflecting disease markers
X1 = zeros(n_diabetic, 8);
X1(:,1) = max(0, round(normrnd(4.9, 3.5, n_diabetic, 1)));      % Pregnancies
X1(:,2) = max(0, round(normrnd(142, 28, n_diabetic, 1)));        % Glucose (higher)
X1(:,3) = max(0, round(normrnd(71, 18, n_diabetic, 1)));         % BloodPressure
X1(:,4) = max(0, round(normrnd(32, 14, n_diabetic, 1)));         % SkinThickness
X1(:,5) = max(0, round(normrnd(160, 120, n_diabetic, 1)));       % Insulin (higher)
X1(:,6) = max(0, round(normrnd(35.1, 6.8, n_diabetic, 1)*10)/10);% BMI (higher)
X1(:,7) = max(0.078, round(normrnd(0.55, 0.35, n_diabetic, 1)*1000)/1000); % DPF
X1(:,8) = max(21, round(normrnd(37, 11, n_diabetic, 1)));        % Age (older)

% Introduce realistic missing data patterns (zeros)
% Glucose: ~5 zeros; BP: ~35; SkinThickness: ~227; Insulin: ~374; BMI: ~11
zero_patterns = {
    struct('col', 2, 'pct', 0.007),   % Glucose
    struct('col', 3, 'pct', 0.045),   % BloodPressure
    struct('col', 4, 'pct', 0.296),   % SkinThickness
    struct('col', 5, 'pct', 0.487),   % Insulin
    struct('col', 6, 'pct', 0.014)    % BMI
};

X_all = [X0; X1];
y_all = [zeros(n_healthy, 1); ones(n_diabetic, 1)];

% Shuffle
perm = randperm(n_total);
X_all = X_all(perm, :);
y_all = y_all(perm);

% Insert missing data patterns
for i = 1:length(zero_patterns)
    col = zero_patterns{i}.col;
    pct = zero_patterns{i}.pct;
    n_zeros = round(n_total * pct);
    zero_idx = randperm(n_total, n_zeros);
    X_all(zero_idx, col) = 0;
end

% Create table
T = table(X_all(:,1), X_all(:,2), X_all(:,3), X_all(:,4), ...
    X_all(:,5), X_all(:,6), X_all(:,7), X_all(:,8), y_all, ...
    'VariableNames', {'Pregnancies', 'Glucose', 'BloodPressure', ...
    'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', ...
    'Age', 'Outcome'});

% Save to CSV
writetable(T, 'data/diabetes.csv');

fprintf('  -> Generated %d samples (%d healthy, %d diabetic)\n', ...
    n_total, n_healthy, n_diabetic);
fprintf('  -> Saved to data/diabetes.csv\n');
fprintf('  -> NOTE: For best results, replace with the real Pima Indians\n');
fprintf('     dataset from: https://www.kaggle.com/uciml/pima-indians-diabetes-database\n');

end
