%% DTW Offline Analysis and Visualization
% This script performs comprehensive DTW analysis similar to dtw_offline_test.py
%
% Features:
% 1. Load recording and templates
% 2. Select a 1-second window interactively
% 3. Compute DTW with full diagnostics
% 4. Visualize cost matrix, warping path, alignment
% 5. Per-channel distance contribution
% 6. Onset vs Hold region analysis
% 7. Noise sensitivity analysis
% 8. Artifact simulation
%


clear; clc; close all;

%% CONFIGURATION
Fs = 2000;                    % Sampling frequency (Hz)
window_size = 192;            % ~96ms window for feature extraction
window_shift = 64;            % ~32ms shift
template_duration_s = 1.0;    % Template duration
template_samples = template_duration_s * Fs;

fprintf('=== DTW Offline Analysis Tool ===\n\n');

%% LOAD DATA
% Check if exported data exists
data_file = 'data/templates.mat';

if exist(data_file, 'file')
    fprintf('Loading data from: %s\n', data_file);
    data = load(data_file);

    templates_open = data.templates_open_features;
    templates_closed = data.templates_closed_features;

    if ndims(templates_open) == 3
        n_open = size(templates_open, 1);
        n_closed = size(templates_closed, 1);
    else
        n_open = 1;
        n_closed = 1;
    end

    fprintf('  OPEN templates: %d\n', n_open);
    fprintf('  CLOSED templates: %d\n', n_closed);

    use_real_data = true;
else
    fprintf('No exported data found. Using synthetic data for demo.\n');
    [templates_open, templates_closed, test_emg, test_gt] = generate_synthetic_data(Fs, template_samples, window_size, window_shift);
    n_open = size(templates_open, 1);
    n_closed = size(templates_closed, 1);
    use_real_data = false;
end

%% GENERATE OR LOAD TEST RECORDING
if ~use_real_data
    fprintf('\nUsing synthetic test recording.\n');
    recording_emg = test_emg;
    recording_gt = test_gt;
else
    % Check for recording file
    recording_file = 'data/recording.mat';
    if exist(recording_file, 'file')
        rec = load(recording_file);
        recording_emg = rec.emg;
        recording_gt = rec.gt;
    else
        fprintf('\nNo recording file found. Generating synthetic recording.\n');
        [~, ~, recording_emg, recording_gt] = generate_synthetic_data(Fs, template_samples, window_size, window_shift);
    end
end

%% INTERACTIVE WINDOW SELECTION
fprintf('\n[1] Interactive Window Selection\n');
fprintf('    Click on the plot to select a 1-second window start position.\n');

[selected_emg, start_time] = interactive_window_selection(recording_emg, recording_gt, Fs, template_samples);

if isempty(selected_emg)
    fprintf('No window selected. Using default (first second).\n');
    selected_emg = recording_emg(:, 1:template_samples);
    start_time = 0;
end

%% EXTRACT FEATURES FROM SELECTED WINDOW
fprintf('\n[2] Extracting features from selected window...\n');
test_features = extract_wl_features(selected_emg, window_size, window_shift);
fprintf('    Test features shape: %d windows x %d channels\n', size(test_features));

%% COMPARE WITH ALL TEMPLATES
fprintf('\n[3] Computing DTW distances to all templates...\n');

% Compare with OPEN templates
fprintf('\n--- OPEN Templates ---\n');
open_distances = zeros(n_open, 1);
for i = 1:n_open
    if ndims(templates_open) == 3
        template = squeeze(templates_open(i, :, :));
    else
        template = templates_open;
    end
    [open_distances(i), ~, ~] = dtw_cosine(test_features, template);
    fprintf('  Template %d: DTW = %.4f\n', i, open_distances(i));
end
fprintf('  Mean: %.4f, Min: %.4f (Template %d)\n', mean(open_distances), min(open_distances), find(open_distances == min(open_distances), 1));

% Compare with CLOSED templates
fprintf('\n--- CLOSED Templates ---\n');
closed_distances = zeros(n_closed, 1);
for i = 1:n_closed
    if ndims(templates_closed) == 3
        template = squeeze(templates_closed(i, :, :));
    else
        template = templates_closed;
    end
    [closed_distances(i), ~, ~] = dtw_cosine(test_features, template);
    fprintf('  Template %d: DTW = %.4f\n', i, closed_distances(i));
end
fprintf('  Mean: %.4f, Min: %.4f (Template %d)\n', mean(closed_distances), min(closed_distances), find(closed_distances == min(closed_distances), 1));

%% CLASSIFICATION RESULT
fprintf('\n=== CLASSIFICATION RESULT ===\n');
min_open = min(open_distances);
min_closed = min(closed_distances);
fprintf('  Min distance to OPEN:   %.4f\n', min_open);
fprintf('  Min distance to CLOSED: %.4f\n', min_closed);
if min_open < min_closed
    fprintf('  Predicted class: OPEN\n');
else
    fprintf('  Predicted class: CLOSED\n');
end
fprintf('  Confidence margin: %.4f\n', abs(min_open - min_closed));

%% DETAILED DTW VISUALIZATION - Best Match
fprintf('\n[4] Detailed DTW Visualization...\n');

% Select best matching template for detailed analysis
if min_open <= min_closed
    best_idx = find(open_distances == min_open, 1);
    if ndims(templates_open) == 3
        best_template = squeeze(templates_open(best_idx, :, :));
    else
        best_template = templates_open;
    end
    best_class = 'OPEN';
else
    best_idx = find(closed_distances == min_closed, 1);
    if ndims(templates_closed) == 3
        best_template = squeeze(templates_closed(best_idx, :, :));
    else
        best_template = templates_closed;
    end
    best_class = 'CLOSED';
end

% Full DTW analysis
[alignment_cost, cost_matrix, local_cost, path] = dtw_with_diagnostics(test_features, best_template);

% Per-channel distances
per_channel_dist = compute_per_channel_distances(test_features, best_template, path);

% Region analysis (onset vs hold)
region_stats = compute_region_distances(test_features, best_template, path, 0.3);

%% VISUALIZATION - Multi-panel Figure
figure('Position', [50, 50, 1600, 900], 'Name', 'DTW Detailed Analysis');

% 1. Accumulated Cost Matrix with Path
subplot(2, 3, 1);
imagesc(cost_matrix');
hold on;
path_i = cellfun(@(x) x(1), num2cell(path, 2));
path_j = cellfun(@(x) x(2), num2cell(path, 2));
plot(path_i, path_j, 'r-', 'LineWidth', 2);
colorbar;
xlabel('Test (windows)');
ylabel('Template (windows)');
title(sprintf('Accumulated Cost Matrix\nTotal DTW: %.4f', alignment_cost));
axis xy;

% 2. Local Cost Matrix
subplot(2, 3, 2);
imagesc(local_cost');
hold on;
plot(path_i, path_j, 'c-', 'LineWidth', 2);
colorbar;
xlabel('Test (windows)');
ylabel('Template (windows)');
title('Local Cost Matrix (Cosine Distance)');
axis xy;

% 3. Aligned Signals
subplot(2, 3, 3);
test_mean = mean(test_features, 2);
template_mean = mean(best_template, 2);

plot(test_mean, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Test');
hold on;
plot(template_mean, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Template');

% Show aligned version
aligned_test = test_mean(path_i);
aligned_template = template_mean(path_j);
plot(aligned_test, 'b--', 'LineWidth', 1, 'DisplayName', 'Test (aligned)');
plot(aligned_template, 'r--', 'LineWidth', 1, 'DisplayName', 'Template (aligned)');

xlabel('Index');
ylabel('Mean feature value');
title('Signal Alignment');
legend('Location', 'best');
grid on;

% 4. Per-Channel Distance
subplot(2, 3, 4);
n_ch = length(per_channel_dist);
bar(1:n_ch, per_channel_dist, 'FaceColor', [0.3, 0.6, 0.9]);
xlabel('Channel');
ylabel('Mean cosine distance');
title('Per-Channel Distance Contribution');
grid on;

% Mark top contributors
[~, sorted_idx] = sort(per_channel_dist, 'descend');
hold on;
for i = 1:min(3, n_ch)
    ch = sorted_idx(i);
    bar(ch, per_channel_dist(ch), 'FaceColor', 'r');
    text(ch, per_channel_dist(ch) + 0.01, sprintf('Ch%d', ch), 'HorizontalAlignment', 'center', 'FontSize', 8);
end

% 5. Onset vs Hold Analysis
subplot(2, 3, 5);
regions = categorical({'Onset (first 30%)', 'Hold (last 70%)'});
regions = reordercats(regions, {'Onset (first 30%)', 'Hold (last 70%)'});
distances = [region_stats.onset_mean, region_stats.hold_mean];
b = bar(regions, distances);
b.FaceColor = 'flat';
b.CData(1,:) = [1, 0.6, 0];  % Orange
b.CData(2,:) = [0, 0.5, 1];  % Blue
ylabel('Mean local distance');
title(sprintf('Distance by Region\nOnset: %d steps, Hold: %d steps', region_stats.onset_count, region_stats.hold_count));

% Add value labels
for i = 1:2
    text(i, distances(i) + 0.01, sprintf('%.4f', distances(i)), 'HorizontalAlignment', 'center');
end

% 6. Distance Summary
subplot(2, 3, 6);
bar_data = [open_distances; closed_distances];
bar_colors = [repmat([0, 0.7, 0], n_open, 1); repmat([0.8, 0, 0], n_closed, 1)];
b = bar(bar_data);
b.FaceColor = 'flat';
b.CData = bar_colors;
hold on;
xline(n_open + 0.5, 'k--', 'LineWidth', 2);
xlabel('Template index');
ylabel('DTW Distance');
title('All Template Distances');
legend_labels = [repmat({'OPEN'}, n_open, 1); repmat({'CLOSED'}, n_closed, 1)];

% Mark best matches
[~, best_open] = min(open_distances);
[~, best_closed] = min(closed_distances);
plot(best_open, open_distances(best_open), 'g^', 'MarkerSize', 12, 'LineWidth', 2);
plot(n_open + best_closed, closed_distances(best_closed), 'rv', 'MarkerSize', 12, 'LineWidth', 2);

sgtitle(sprintf('DTW Analysis - Best Match: %s Template %d (Distance: %.4f)', best_class, best_idx, alignment_cost), 'FontSize', 14, 'FontWeight', 'bold');

%% WARPING VISUALIZATION
figure('Position', [100, 100, 1400, 500], 'Name', 'Warping Alignment');

% 1. Warping connections
subplot(1, 3, 1);
offset = max(abs(template_mean)) * 1.5;
plot(test_mean, 'b-', 'LineWidth', 2, 'DisplayName', 'Test');
hold on;
plot(template_mean - offset, 'r-', 'LineWidth', 2, 'DisplayName', 'Template');

% Draw warping lines (subsampled)
step = max(1, floor(length(path) / 30));
for k = 1:step:length(path)
    i = path(k, 1);
    j = path(k, 2);
    plot([i, j], [test_mean(i), template_mean(j) - offset], 'g-', 'Alpha', 0.3);
end

xlabel('Window index');
ylabel('Feature value');
title('DTW Warping Alignment');
legend('Test', 'Template', 'Location', 'best');
grid on;

% 2. Warping function
subplot(1, 3, 2);
plot(path_i, 'b-', 'LineWidth', 2, 'DisplayName', 'Test index');
hold on;
plot(path_j, 'r-', 'LineWidth', 2, 'DisplayName', 'Template index');
plot([1, length(path)], [1, max(path_i)], 'k--', 'DisplayName', 'Diagonal (no warp)');
fill([1:length(path), length(path):-1:1], [path_i', fliplr(path_j')], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
xlabel('Alignment step');
ylabel('Original index');
title('Warping Function');
legend('Location', 'best');
grid on;

% 3. Local cost along path
subplot(1, 3, 3);
local_costs_along_path = zeros(length(path), 1);
for k = 1:length(path)
    i = path(k, 1);
    j = path(k, 2);
    local_costs_along_path(k) = local_cost(i, j);
end

plot(local_costs_along_path, 'purple', 'LineWidth', 2);
hold on;
fill([1:length(path), length(path), 1], [local_costs_along_path', 0, 0], 'purple', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
yline(mean(local_costs_along_path), 'r--', 'LineWidth', 2);

% Highlight high-cost regions
threshold = mean(local_costs_along_path) + std(local_costs_along_path);
high_cost = local_costs_along_path > threshold;
scatter(find(high_cost), local_costs_along_path(high_cost), 50, 'r', 'filled');

xlabel('Alignment step');
ylabel('Local cosine distance');
title(sprintf('Cost Along Path (Mean: %.4f)', mean(local_costs_along_path)));
legend('Local cost', '', sprintf('Mean: %.4f', mean(local_costs_along_path)), 'High cost', 'Location', 'best');
grid on;

sgtitle('Warping Analysis', 'FontSize', 14, 'FontWeight', 'bold');

%% NOISE SENSITIVITY ANALYSIS
fprintf('\n[5] Noise Sensitivity Analysis...\n');

figure('Position', [150, 150, 1000, 400], 'Name', 'Noise Sensitivity');

noise_levels = linspace(0, 0.5, 15);
n_trials = 5;
distances_noise = zeros(length(noise_levels), n_trials);

for nl = 1:length(noise_levels)
    for trial = 1:n_trials
        noise = randn(size(test_features)) * noise_levels(nl) * std(test_features(:));
        noisy_features = test_features + noise;
        [distances_noise(nl, trial), ~, ~] = dtw_cosine(noisy_features, best_template);
    end
end

mean_dist = mean(distances_noise, 2);
std_dist = std(distances_noise, 0, 2);

subplot(1, 2, 1);
plot(noise_levels, mean_dist, 'b-', 'LineWidth', 2);
hold on;
fill([noise_levels, fliplr(noise_levels)], [mean_dist' - std_dist', fliplr(mean_dist' + std_dist')], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
yline(alignment_cost, 'g--', 'LineWidth', 2);
xlabel('Noise level (fraction of std)');
ylabel('DTW distance');
title('Effect of Noise on DTW Distance');
legend('Mean', '\pm1 std', sprintf('Baseline: %.4f', alignment_cost), 'Location', 'best');
grid on;

subplot(1, 2, 2);
relative_change = (mean_dist - alignment_cost) / alignment_cost * 100;
plot(noise_levels, relative_change, 'r-', 'LineWidth', 2);
hold on;
fill([noise_levels, fliplr(noise_levels)], ...
     [relative_change' - std_dist'/alignment_cost*100, fliplr(relative_change' + std_dist'/alignment_cost*100)], ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
yline(0, 'g--', 'LineWidth', 2);
xlabel('Noise level (fraction of std)');
ylabel('Distance change (%)');
title('Relative Distance Change');
grid on;

sgtitle('Noise Sensitivity Analysis', 'FontSize', 14, 'FontWeight', 'bold');

%% ARTIFACT SIMULATION
fprintf('\n[6] Artifact Simulation...\n');

figure('Position', [200, 200, 800, 500], 'Name', 'Artifact Simulation');

artifact_names = {'Baseline', 'Spike (5x)', 'Baseline shift', 'Amplitude decay', 'Channel dropout', 'Noise burst'};
artifact_distances = zeros(6, 1);

% Baseline
artifact_distances(1) = alignment_cost;

% Spike artifact
spike_features = test_features;
spike_pos = floor(size(test_features, 1) / 2);
spike_features(spike_pos:spike_pos+1, :) = spike_features(spike_pos:spike_pos+1, :) * 5;
[artifact_distances(2), ~, ~] = dtw_cosine(spike_features, best_template);

% Baseline shift
shift_features = test_features + std(test_features(:)) * 0.5;
[artifact_distances(3), ~, ~] = dtw_cosine(shift_features, best_template);

% Amplitude decay
decay_features = test_features .* linspace(1, 0.5, size(test_features, 1))';
[artifact_distances(4), ~, ~] = dtw_cosine(decay_features, best_template);

% Channel dropout
dropout_features = test_features;
dropout_features(:, 1:2) = 0;
[artifact_distances(5), ~, ~] = dtw_cosine(dropout_features, best_template);

% Noise burst
burst_features = test_features;
burst_start = floor(size(test_features, 1) / 3);
burst_end = burst_start + floor(size(test_features, 1) / 6);
burst_features(burst_start:burst_end, :) = burst_features(burst_start:burst_end, :) + randn(burst_end-burst_start+1, size(test_features, 2)) * std(test_features(:));
[artifact_distances(6), ~, ~] = dtw_cosine(burst_features, best_template);

% Plot
colors = zeros(6, 3);
for i = 1:6
    pct_change = (artifact_distances(i) - alignment_cost) / alignment_cost * 100;
    if pct_change > 20
        colors(i, :) = [0.8, 0, 0];  % Red
    elseif pct_change > 10
        colors(i, :) = [1, 0.6, 0];  % Orange
    else
        colors(i, :) = [0, 0.7, 0];  % Green
    end
end
colors(1, :) = [0, 0, 0.8];  % Blue for baseline

b = bar(artifact_distances);
b.FaceColor = 'flat';
b.CData = colors;

hold on;
yline(alignment_cost, 'b--', 'LineWidth', 2);

set(gca, 'XTickLabel', artifact_names, 'XTickLabelRotation', 15);
ylabel('DTW Distance');
title(sprintf('Effect of Artifacts on DTW Distance\n(Green: <10%%, Orange: 10-20%%, Red: >20%% change)'));

% Add percentage change labels
for i = 2:6
    pct_change = (artifact_distances(i) - alignment_cost) / alignment_cost * 100;
    text(i, artifact_distances(i) + 0.02, sprintf('%+.1f%%', pct_change), 'HorizontalAlignment', 'center', 'FontSize', 9);
end

grid on;

%% SUMMARY
fprintf('\n=== Analysis Complete ===\n');
fprintf('Results:\n');
fprintf('  Best match: %s Template %d (Distance: %.4f)\n', best_class, best_idx, alignment_cost);
fprintf('  OPEN templates - Mean: %.4f, Min: %.4f\n', mean(open_distances), min(open_distances));
fprintf('  CLOSED templates - Mean: %.4f, Min: %.4f\n', mean(closed_distances), min(closed_distances));
fprintf('  Prediction: %s (margin: %.4f)\n', best_class, abs(min_open - min_closed));

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function [selected_emg, start_time] = interactive_window_selection(emg, gt, Fs, window_samples)
    % Interactive window selection from recording

    [n_ch, n_samples] = size(emg);
    t = (0:n_samples-1) / Fs;
    window_duration = window_samples / Fs;

    % Normalize for display
    emg_display = emg(1, :) / (max(abs(emg(1, :))) + 1e-10);

    fig = figure('Position', [100, 100, 1200, 500], 'Name', 'Select Window');
    ax = axes(fig);

    plot(ax, t, emg_display, 'b-', 'LineWidth', 0.5);
    hold(ax, 'on');

    if ~isempty(gt)
        gt_norm = gt(:)' / (max(abs(gt(:))) + 1e-10);
        fill(ax, [t(1:length(gt_norm)), fliplr(t(1:length(gt_norm)))], ...
             [-ones(1, length(gt_norm)), gt_norm], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end

    xlabel(ax, 'Time (s)');
    ylabel(ax, 'Normalized amplitude');
    title(ax, sprintf('Click to select %0.1f second window start', window_duration));
    grid(ax, 'on');
    ylim(ax, [-1.2, 1.2]);

    % Wait for click
    [x, ~] = ginput(1);

    if isempty(x)
        selected_emg = [];
        start_time = 0;
        close(fig);
        return;
    end

    start_time = max(0, min(x, (n_samples - window_samples) / Fs));
    start_idx = round(start_time * Fs) + 1;
    end_idx = start_idx + window_samples - 1;

    % Draw selection rectangle
    rectangle(ax, 'Position', [start_time, -1.2, window_duration, 2.4], ...
              'EdgeColor', 'r', 'LineWidth', 2, 'FaceColor', [1, 0, 0, 0.1]);
    title(ax, sprintf('Selected: %.2f s to %.2f s', start_time, start_time + window_duration));
    drawnow;
    pause(1);

    selected_emg = emg(:, start_idx:end_idx);
    close(fig);
end

function features = extract_wl_features(emg, window_size, window_shift)
    [n_ch, n_samples] = size(emg);
    n_windows = floor((n_samples - window_size) / window_shift) + 1;
    features = zeros(n_windows, n_ch);

    for w = 1:n_windows
        start_idx = (w - 1) * window_shift + 1;
        end_idx = start_idx + window_size - 1;
        window = emg(:, start_idx:end_idx);

        for ch = 1:n_ch
            features(w, ch) = sum(abs(diff(window(ch, :))));
        end
    end
end

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
    cost_matrix = cost_matrix(2:end, 2:end);
    path = reconstruct_path(traceback, N, M);
end

function [alignment_cost, cost_matrix, local_cost, path] = dtw_with_diagnostics(template1, template2)
    [N, ~] = size(template1);
    [M, ~] = size(template2);

    cost_matrix = inf(N + 1, M + 1);
    cost_matrix(1, 1) = 0;
    local_cost = zeros(N, M);
    traceback = zeros(N, M);

    for i = 1:N
        for j = 1:M
            dist = cosine_distance(template1(i, :), template2(j, :));
            local_cost(i, j) = dist;
            candidates = [cost_matrix(i, j), cost_matrix(i, j + 1), cost_matrix(i + 1, j)];
            [min_cost, direction] = min(candidates);
            cost_matrix(i + 1, j + 1) = dist + min_cost;
            traceback(i, j) = direction;
        end
    end

    alignment_cost = cost_matrix(N + 1, M + 1);
    cost_matrix = cost_matrix(2:end, 2:end);
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

function per_channel_dist = compute_per_channel_distances(t1, t2, path)
    n_ch = size(t1, 2);
    per_channel_dist = zeros(n_ch, 1);

    for ch = 1:n_ch
        total_dist = 0;
        for k = 1:size(path, 1)
            i = path(k, 1);
            j = path(k, 2);
            v1 = t1(i, ch);
            v2 = t2(j, ch);
            dist = 1 - (v1 * v2) / (abs(v1) * abs(v2) + 1e-8);
            total_dist = total_dist + dist;
        end
        per_channel_dist(ch) = total_dist / size(path, 1);
    end
end

function stats = compute_region_distances(t1, t2, path, onset_fraction)
    n_windows = size(t1, 1);
    onset_boundary = floor(n_windows * onset_fraction);

    onset_dists = [];
    hold_dists = [];

    for k = 1:size(path, 1)
        i = path(k, 1);
        j = path(k, 2);
        dist = cosine_distance(t1(i, :), t2(j, :));

        if i <= onset_boundary
            onset_dists = [onset_dists; dist];
        else
            hold_dists = [hold_dists; dist];
        end
    end

    stats.onset_mean = mean(onset_dists);
    stats.onset_count = length(onset_dists);
    stats.hold_mean = mean(hold_dists);
    stats.hold_count = length(hold_dists);
end

function [templates_open, templates_closed, test_emg, test_gt] = generate_synthetic_data(Fs, template_samples, window_size, window_shift)
    % Generate synthetic data for demo

    n_ch = 8;
    n_templates = 5;
    t = (0:template_samples-1) / Fs;

    rng(42);

    % CLOSED templates (strong activation)
    templates_closed_raw = cell(n_templates, 1);
    for i = 1:n_templates
        emg = randn(n_ch, template_samples) * 30;
        activation = exp(-((t - 0.5).^2) / (0.04 + 0.01*randn));
        for ch = 1:n_ch
            emg(ch, :) = emg(ch, :) + activation * (150 + 40*randn);
        end
        templates_closed_raw{i} = emg;
    end

    % OPEN templates (weak activation)
    templates_open_raw = cell(n_templates, 1);
    for i = 1:n_templates
        emg = randn(n_ch, template_samples) * 30;
        activation = exp(-((t - 0.5).^2) / 0.02) * 0.2;
        for ch = 1:n_ch
            emg(ch, :) = emg(ch, :) + activation * (60 + 20*randn);
        end
        templates_open_raw{i} = emg;
    end

    % Extract features
    templates_closed = zeros(n_templates, floor((template_samples - window_size) / window_shift) + 1, n_ch);
    templates_open = zeros(n_templates, floor((template_samples - window_size) / window_shift) + 1, n_ch);

    for i = 1:n_templates
        templates_closed(i, :, :) = extract_wl_features(templates_closed_raw{i}, window_size, window_shift);
        templates_open(i, :, :) = extract_wl_features(templates_open_raw{i}, window_size, window_shift);
    end

    % Test recording (3 seconds: open -> closed -> open)
    test_duration_s = 3;
    test_samples = test_duration_s * Fs;
    t_test = (0:test_samples-1) / Fs;

    test_emg = randn(n_ch, test_samples) * 30;
    test_gt = zeros(1, test_samples);

    % Add CLOSED activation in middle second
    closed_start = Fs;
    closed_end = 2 * Fs;
    test_gt(closed_start:closed_end) = 1;

    activation_test = exp(-((t_test - 1.5).^2) / 0.1);
    for ch = 1:n_ch
        test_emg(ch, :) = test_emg(ch, :) + activation_test * (150 + 40*randn);
    end
end
