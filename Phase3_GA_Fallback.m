function [selected_features, ga_results] = Phase3_GA_Fallback(X_train, y_train, config)
%% =========================================================================
%  PHASE 3 (FALLBACK): CUSTOM GENETIC ALGORITHM FOR FEATURE SELECTION
%  =========================================================================
%  This is a standalone GA implementation that does NOT require the 
%  Global Optimization Toolbox. Use this if ga() is unavailable.
%
%  Replace the call in main_pipeline.m:
%    [selected_features, ga_results] = Phase3_GeneticAlgorithm(...)
%  With:
%    [selected_features, ga_results] = Phase3_GA_Fallback(...)
%  =========================================================================

fprintf('[PHASE 3 - FALLBACK] Running custom Genetic Algorithm...\n');

nFeatures = size(X_train, 2);
popSize = config.ga_popSize;
maxGen = config.ga_maxGen;
crossFrac = config.ga_crossFrac;
mutRate = 0.1;
eliteCount = 2;

%% Initialize Population (Random binary chromosomes)
population = randi([0, 1], popSize, nFeatures);

% Ensure no empty chromosomes
for i = 1:popSize
    if sum(population(i,:)) == 0
        population(i, randi(nFeatures)) = 1;
    end
end

%% Evaluate Initial Population
fitness = zeros(popSize, 1);
for i = 1:popSize
    fitness(i) = evaluateFitness(population(i,:), X_train, y_train, config);
end

% Track best fitness across generations
best_fitness_history = zeros(maxGen, 1);
mean_fitness_history = zeros(maxGen, 1);
stall_count = 0;
best_overall_fitness = min(fitness);
best_overall_chromosome = population(find(fitness == best_overall_fitness, 1), :);

fprintf('  -> Initial best fitness: %.4f\n', best_overall_fitness);

%% Evolution Loop
for gen = 1:maxGen
    new_population = zeros(popSize, nFeatures);
    
    % 1. Elitism: Keep top individuals
    [~, elite_idx] = sort(fitness);
    for i = 1:eliteCount
        new_population(i, :) = population(elite_idx(i), :);
    end
    
    % 2. Selection + Crossover + Mutation
    for i = (eliteCount + 1):2:popSize
        % Tournament selection (size 3)
        parent1 = tournamentSelect(population, fitness, 3);
        parent2 = tournamentSelect(population, fitness, 3);
        
        % Single-point crossover
        if rand() < crossFrac
            cp = randi([1, nFeatures-1]);
            child1 = [parent1(1:cp), parent2(cp+1:end)];
            child2 = [parent2(1:cp), parent1(cp+1:end)];
        else
            child1 = parent1;
            child2 = parent2;
        end
        
        % Bit-flip mutation
        for j = 1:nFeatures
            if rand() < mutRate
                child1(j) = 1 - child1(j);
            end
            if rand() < mutRate
                child2(j) = 1 - child2(j);
            end
        end
        
        % Ensure non-empty
        if sum(child1) == 0, child1(randi(nFeatures)) = 1; end
        if sum(child2) == 0, child2(randi(nFeatures)) = 1; end
        
        new_population(i, :) = child1;
        if i + 1 <= popSize
            new_population(i + 1, :) = child2;
        end
    end
    
    population = new_population;
    
    % Evaluate new population
    for i = 1:popSize
        fitness(i) = evaluateFitness(population(i,:), X_train, y_train, config);
    end
    
    % Track progress
    gen_best = min(fitness);
    best_fitness_history(gen) = gen_best;
    mean_fitness_history(gen) = mean(fitness);
    
    if gen_best < best_overall_fitness
        best_overall_fitness = gen_best;
        best_overall_chromosome = population(find(fitness == gen_best, 1), :);
        stall_count = 0;
    else
        stall_count = stall_count + 1;
    end
    
    % Display progress every 10 generations
    if mod(gen, 10) == 0 || gen == 1
        fprintf('  Gen %3d: Best=%.4f | Mean=%.4f | Stall=%d\n', ...
            gen, gen_best, mean(fitness), stall_count);
    end
    
    % Early stopping
    if stall_count >= 20
        fprintf('  -> Converged at generation %d (stall limit reached)\n', gen);
        break;
    end
end

%% Extract Results
selected_features = find(best_overall_chromosome == 1);
selected_names = config.featureNames(selected_features);

fprintf('\n============================================\n');
fprintf('  CUSTOM GA RESULTS (FALLBACK)\n');
fprintf('============================================\n');
fprintf('  Generations completed: %d\n', gen);
fprintf('  Best fitness: %.4f\n', best_overall_fitness);
fprintf('  Selected features (%d/%d):\n', length(selected_features), nFeatures);
for i = 1:length(selected_features)
    fprintf('    [%d] %s\n', selected_features(i), selected_names{i});
end
fprintf('============================================\n');

ga_results = struct();
ga_results.best_chromosome = best_overall_chromosome;
ga_results.best_fitness = best_overall_fitness;
ga_results.selected_features = selected_features;
ga_results.selected_names = selected_names;
ga_results.generations = gen;
ga_results.fitness_history = best_fitness_history(1:gen);

%% Plot GA Convergence
figure('Name', 'Phase 3: GA Convergence (Fallback)', ...
       'Position', [100, 100, 800, 400], 'Color', 'w');

plot(1:gen, best_fitness_history(1:gen), 'b-', 'LineWidth', 2, ...
    'DisplayName', 'Best Fitness');
hold on;
plot(1:gen, mean_fitness_history(1:gen), 'r--', 'LineWidth', 1.5, ...
    'DisplayName', 'Mean Fitness');
xlabel('Generation', 'FontSize', 12);
ylabel('Fitness (Error + Penalty)', 'FontSize', 12);
title('Genetic Algorithm Convergence', 'FontSize', 14, 'FontWeight', 'bold');
legend('show', 'Location', 'best');
grid on;

saveas(gcf, 'results/Phase3_GA_Convergence_Fallback.png');

fprintf('\n[PHASE 3 - FALLBACK] COMPLETE.\n');

end

%% Helper Functions
function fitness = evaluateFitness(chromosome, X_train, y_train, config)
    selected = find(chromosome == 1);
    if isempty(selected)
        fitness = 1.0;
        return;
    end
    
    X_subset = X_train(:, selected);
    
    try
        knn = fitcknn(X_subset, y_train, ...
            'NumNeighbors', config.knn_k, 'Distance', 'euclidean');
        cv = crossval(knn, 'KFold', 5);
        cv_error = kfoldLoss(cv);
    catch
        cv_error = 1.0;
    end
    
    alpha = 0.01;
    fitness = cv_error + alpha * length(selected) / size(X_train, 2);
end

function parent = tournamentSelect(population, fitness, tournSize)
    candidates = randi(size(population, 1), 1, tournSize);
    [~, best_idx] = min(fitness(candidates));
    parent = population(candidates(best_idx), :);
end
