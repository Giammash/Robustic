%% DTW Analysis with Real EMG Data
% This script loads exported templates/recordings and demonstrates
% the DTW classification algorithm on real data.
%
% Prerequisites:
%   Run: python export_templates_for_matlab.py
%   This creates data/templates.mat with real EMG templates


clear; clc; close all;

%% LOAD DATA
fprintf('=== DTW Analysis with Real EMG Data ===\n\n');

data_file = 'data/templates.mat';

if ~exist(data_file, 'file')
    error(['Data file not found: %s\n' ...
           'Please run: python export_templates_for_matlab.py'], data_file);
end

data = load(data_file);
fprintf('Loaded data from: %s\n', data_file);

% Extract parameters
Fs = double(data.Fs);
window_size = double(data.window_size);
window_shift = double(data.window_shift);

fprintf('  Sampling rate: %d Hz\n', Fs);
fprintf('  Window size: %d samples (%.1f ms)\n', window_size, window_size/Fs*1000);
fprintf('  Window shift: %d samples (%.1f ms)\n', window_shift, window_shift/Fs*1000);

%% PROCESS TEMPLATES
% Feature-extracted templates are already in the file
% Shape: (n_templates, n_windows, n_channels)

templates_open = data.templates_open_features;
templates_closed = data.templates_closed_features;

% Handle different possible shapes
if ndim(templates_open) == 3
    n_open = size(templates_open, 1);
    n_closed = size(templates_closed, 1);
    [~, n_windows, n_channels] = size(templates_open);
else
    % Single template case
    n_open = 1;
    n_closed = 1;
    templates_open = reshape(templates_open, 1, size(templates_open, 1), size(templates_open, 2));
    templates_closed = reshape(templates_closed, 1, size(templates_closed, 1), size(templates_closed, 2));
    [~, n_windows, n_channels] = size(templates_open);
end

fprintf('\nTemplates loaded:\n');
fprintf('  OPEN:   %d templates, %d windows x %d channels\n', n_open, n_windows, n_channels);
fprintf('  CLOSED: %d templates, %d windows x %d channels\n', n_closed, n_windows, n_channels);

%% COMPUTE INTRA-CLASS DISTANCES (CLOSED vs CLOSED)
fprintf('\n=== Computing Intra-Class Distances (CLOSED) ===\n');

intra_distances_closed = [];
for i = 1:n_closed
    for j = i+1:n_closed
        t1 = squeeze(templates_closed(i, :, :));
        t2 = squeeze(templates_closed(j, :, :));
        [dist, ~, ~] = dtw_cosine(t1, t2);
        intra_distances_closed = [intra_distances_closed, dist];
        fprintf('  CLOSED_%d vs CLOSED_%d: %.4f\n', i, j, dist);
    end
end

if ~isempty(intra_distances_closed)
    mean_intra_closed = mean(intra_distances_closed);
    std_intra_closed = std(intra_distances_closed);
    fprintf('  Mean: %.4f, Std: %.4f\n', mean_intra_closed, std_intra_closed);
else
    mean_intra_closed = 0;
    std_intra_closed = 0;
end

%% COMPUTE INTRA-CLASS DISTANCES (OPEN vs OPEN)
fprintf('\n=== Computing Intra-Class Distances (OPEN) ===\n');

intra_distances_open = [];
for i = 1:n_open
    for j = i+1:n_open
        t1 = squeeze(templates_open(i, :, :));
        t2 = squeeze(templates_open(j, :, :));
        [dist, ~, ~] = dtw_cosine(t1, t2);
        intra_distances_open = [intra_distances_open, dist];
        fprintf('  OPEN_%d vs OPEN_%d: %.4f\n', i, j, dist);
    end
end

if ~isempty(intra_distances_open)
    mean_intra_open = mean(intra_distances_open);
    std_intra_open = std(intra_distances_open);
    fprintf('  Mean: %.4f, Std: %.4f\n', mean_intra_open, std_intra_open);
else
    mean_intra_open = 0;
    std_intra_open = 0;
end

%% COMPUTE CROSS-CLASS DISTANCES (OPEN vs CLOSED)
fprintf('\n=== Computing Cross-Class Distances (OPEN vs CLOSED) ===\n');

cross_distances = [];
for i = 1:n_open
    for j = 1:n_closed
        t1 = squeeze(templates_open(i, :, :));
        t2 = squeeze(templates_closed(j, :, :));
        [dist, ~, ~] = dtw_cosine(t1, t2);
        cross_distances = [cross_distances, dist];
    end
end

mean_cross = mean(cross_distances);
std_cross = std(cross_distances);
fprintf('  Mean: %.4f, Std: %.4f\n', mean_cross, std_cross);
fprintf('  Min: %.4f, Max: %.4f\n', min(cross_distances), max(cross_distances));

%% COMPUTE THRESHOLDS
fprintf('\n=== Threshold Computation ===\n');

s = 1.0;  % Number of standard deviations

threshold_closed = mean_intra_closed + s * std_intra_closed;
threshold_open = mean_intra_open + s * std_intra_open;

fprintf('  Threshold CLOSED (s=%.1f): %.4f\n', s, threshold_closed);
fprintf('  Threshold OPEN (s=%.1f):   %.4f\n', s, threshold_open);

%% EVALUATE SEPARABILITY
fprintf('\n=== Class Separability Analysis ===\n');

% Separation ratio: how far apart are the classes?
separation_ratio = mean_cross / max(mean_intra_closed, mean_intra_open);
fprintf('  Separation ratio (cross/intra): %.2f\n', separation_ratio);

if separation_ratio > 2
    fprintf('  --> GOOD separability\n');
elseif separation_ratio > 1.5
    fprintf('  --> MODERATE separability\n');
else
    fprintf('  --> POOR separability (classes overlap)\n');
end

%% VISUALIZATION
fprintf('\nGenerating plots...\n');

figure('Position', [100, 100, 1200, 800]);

% Plot 1: Distance distributions
subplot(2, 2, 1);
if ~isempty(intra_distances_closed)
    histogram(intra_distances_closed, 10, 'FaceColor', 'b', 'FaceAlpha', 0.5);
    hold on;
end
if ~isempty(intra_distances_open)
    histogram(intra_distances_open, 10, 'FaceColor', 'g', 'FaceAlpha', 0.5);
end
histogram(cross_distances, 15, 'FaceColor', 'r', 'FaceAlpha', 0.5);
xlabel('DTW Distance');
ylabel('Count');
title('Distance Distributions');
legend('Intra-CLOSED', 'Intra-OPEN', 'Cross-class', 'Location', 'best');
grid on;

% Plot 2: Box plot comparison
subplot(2, 2, 2);
all_data = [intra_distances_closed(:); intra_distances_open(:); cross_distances(:)];
groups = [ones(length(intra_distances_closed), 1); ...
          2*ones(length(intra_distances_open), 1); ...
          3*ones(length(cross_distances), 1)];
if ~isempty(all_data)
    boxplot(all_data, groups, 'Labels', {'Intra-CLOSED', 'Intra-OPEN', 'Cross-class'});
    ylabel('DTW Distance');
    title('Distance Comparison');
    grid on;
end

% Plot 3: Feature visualization (first template of each class)
subplot(2, 2, 3);
if n_closed > 0
    imagesc(squeeze(templates_closed(1, :, :))');
    colorbar;
    xlabel('Window');
    ylabel('Channel');
    title('CLOSED Template 1 (Features)');
end

% Plot 4: Summary statistics
subplot(2, 2, 4);
stats = [mean_intra_closed, mean_intra_open, mean_cross; ...
         std_intra_closed, std_intra_open, std_cross];
bar(stats');
set(gca, 'XTickLabel', {'Intra-CLOSED', 'Intra-OPEN', 'Cross-class'});
legend('Mean', 'Std', 'Location', 'best');
ylabel('Distance');
title('Summary Statistics');
grid on;

sgtitle('DTW Classification Analysis - Real Data', 'FontSize', 14, 'FontWeight', 'bold');

%% SUMMARY
fprintf('\n=== Summary ===\n');
fprintf('Intra-class (CLOSED): mean=%.4f, std=%.4f, threshold=%.4f\n', ...
    mean_intra_closed, std_intra_closed, threshold_closed);
fprintf('Intra-class (OPEN):   mean=%.4f, std=%.4f, threshold=%.4f\n', ...
    mean_intra_open, std_intra_open, threshold_open);
fprintf('Cross-class:          mean=%.4f, std=%.4f\n', mean_cross, std_cross);
fprintf('Separation ratio:     %.2f\n', separation_ratio);

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function d = cosine_distance(v1, v2)
    dot_product = dot(v1, v2);
    norm1 = norm(v1);
    norm2 = norm(v2);
    d = 1 - dot_product / (norm1 * norm2 + 1e-8);
end

function [alignment_cost, cost_matrix, path] = dtw_cosine(template1, template2)
    [N, ~] = size(template1);
    [M, ~] = size(template2);

    cost_matrix = inf(N + 1, M + 1);
    cost_matrix(1, 1) = 0;
    traceback = zeros(N, M);

    for i = 1:N
        for j = 1:M
            dist = cosine_distance(template1(i, :), template2(j, :));
            candidates = [cost_matrix(i, j), cost_matrix(i, j + 1), cost_matrix(i + 1, j)];
            [min_cost, direction] = min(candidates);
            cost_matrix(i + 1, j + 1) = dist + min_cost;
            traceback(i, j) = direction;
        end
    end

    alignment_cost = cost_matrix(N + 1, M + 1);
    path = reconstruct_path(traceback, N, M);
end

function path = reconstruct_path(traceback, N, M)
    path = [N, M];
    i = N; j = M;
    while i > 1 || j > 1
        if i == 1
            j = j - 1;
        elseif j == 1
            i = i - 1;
        else
            direction = traceback(i, j);
            if direction == 1, i = i - 1; j = j - 1;
            elseif direction == 2, i = i - 1;
            else, j = j - 1;
            end
        end
        path = [i, j; path];
    end
end
