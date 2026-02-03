%% DTW with Cosine Distance - Core Algorithm Demo
% This script demonstrates the Dynamic Time Warping (DTW) algorithm
% used in the MindMove/Robustic EMG classification framework.
%
% The algorithm compares two EMG templates (time series of feature vectors)
% and computes a similarity score. Lower distance = more similar.
%
% Author: MindMove Framework Demo
% Date: 2024

clear; clc; close all;

%% PARAMETERS (matching the Python framework)
Fs = 2000;                    % Sampling frequency (Hz)
template_duration = 1.0;      % Template duration (seconds)
window_size = 192;            % samples, ~96ms window
window_shift = 64;            % ~32ms shift
n_channels = 8;               % Number of EMG channels (simplified)

n_samples = template_duration * Fs;

%% GENERATE SYNTHETIC EMG DATA
fprintf('=== DTW with Cosine Distance Demo ===\n\n');
fprintf('Generating synthetic EMG templates...\n');

t = (0:n_samples-1) / Fs;

% Template 1: CLOSED gesture (hand closing = muscle activation)
rng(42);  % For reproducibility
emg_closed_1 = randn(n_channels, n_samples) * 50;  % baseline noise
activation = exp(-((t - 0.5).^2) / 0.05);          % Gaussian activation burst
for ch = 1:n_channels
    emg_closed_1(ch, :) = emg_closed_1(ch, :) + activation * (200*ch + 50*randn);
end

% Template 2: Another CLOSED gesture (similar pattern, slight variation)
emg_closed_2 = randn(n_channels, n_samples) * 50;
activation2 = exp(-((t - 0.52).^2) / 0.055);       % Slightly shifted
for ch = 1:n_channels
    emg_closed_2(ch, :) = emg_closed_2(ch, :) + activation2 * (190*ch + 50*randn);
end

% Template 3: OPEN gesture (low activation / different pattern)
emg_open = randn(n_channels, n_samples) * 50;
activation_open = exp(-((t - 0.5).^2) / 0.03);  
for ch = 1:n_channels
    emg_open(ch, :) = emg_open(ch, :) + activation_open * (200/ch + 30*randn);
end

%% PROCESS TEMPLATES: Windowing + Feature Extraction
fprintf('Extracting features (RMS)...\n');

% Segment into windows and extract features
features_closed_1 = extract_rms_features(emg_closed_1, window_size, window_shift);
features_closed_2 = extract_rms_features(emg_closed_2, window_size, window_shift);
features_open = extract_rms_features(emg_open, window_size, window_shift);

fprintf('  Template shape: %d windows x %d channels\n', size(features_closed_1));

%% COMPUTE DTW DISTANCES
fprintf('\nComputing DTW distances...\n');

% Same class comparison (CLOSED vs CLOSED)
[dist_same, cost_matrix_same, path_same] = dtw_cosine(features_closed_1, features_closed_2);
fprintf('  CLOSED_1 vs CLOSED_2 (same class):     %.4f\n', dist_same);

% Different class comparison (CLOSED vs OPEN)
[dist_diff1, cost_matrix_diff, path_diff] = dtw_cosine(features_closed_1, features_open);
fprintf('  CLOSED_1 vs OPEN (different class):    %.4f\n', dist_diff1);

[dist_diff2, ~, ~] = dtw_cosine(features_closed_2, features_open);
fprintf('  CLOSED_2 vs OPEN (different class):    %.4f\n', dist_diff2);

%% THRESHOLD COMPUTATION
fprintf('\n=== Threshold Computation ===\n');

% In practice: threshold = mean(intra-class distances) + s * std
% Here we only have 2 CLOSED templates, so we estimate

mean_intra = dist_same;
std_intra = dist_same * 0.2;  % Estimated
s = 1.0;  % Number of standard deviations

threshold = mean_intra + s * std_intra;
fprintf('  Intra-class mean distance: %.4f\n', mean_intra);
fprintf('  Threshold (s=%.1f):         %.4f\n', s, threshold);

%% CLASSIFICATION LOGIC
fprintf('\n=== Classification Demo ===\n');

test_distance = dist_diff1;  % Using OPEN template as "test"
fprintf('  Test signal distance to CLOSED templates: %.4f\n', test_distance);
fprintf('  Threshold: %.4f\n', threshold);

if test_distance < threshold
    fprintf('  Result: CLOSED (distance < threshold)\n');
else
    fprintf('  Result: NOT CLOSED (distance >= threshold)\n');
end

%% VISUALIZATION
fprintf('\nGenerating plots...\n');

figure('Position', [100, 100, 1400, 800]);

% Plot 1: Raw EMG signals
subplot(2, 3, 1);
plot(t, emg_closed_1(1, :), 'b', 'LineWidth', 0.5);
hold on;
plot(t, emg_closed_2(1, :) + 500, 'r', 'LineWidth', 0.5);
plot(t, emg_open(1, :) + 1000, 'g', 'LineWidth', 0.5);
xlabel('Time (s)');
ylabel('Amplitude (uV)');
title('Raw EMG (Channel 1)');
legend('CLOSED 1', 'CLOSED 2', 'OPEN', 'Location', 'best');
grid on;

% Plot 2: Feature vectors (Waveform Length)
subplot(2, 3, 2);
imagesc(features_closed_1');
colorbar;
xlabel('Window');
ylabel('Channel');
title('Features: CLOSED Template 1 (WL)');

% Plot 3: Cost matrix for same-class comparison
subplot(2, 3, 3);
imagesc(cost_matrix_same(2:end, 2:end));
hold on;
plot(path_same(:, 2), path_same(:, 1), 'r-', 'LineWidth', 2);
colorbar;
xlabel('Template 2 (windows)');
ylabel('Template 1 (windows)');
title(sprintf('DTW Cost Matrix (CLOSED vs CLOSED)\nDistance = %.4f', dist_same));

% Plot 4: Cost matrix for different-class comparison
subplot(2, 3, 4);
imagesc(cost_matrix_diff(2:end, 2:end));
hold on;
plot(path_diff(:, 2), path_diff(:, 1), 'r-', 'LineWidth', 2);
colorbar;
xlabel('OPEN template (windows)');
ylabel('CLOSED template (windows)');
title(sprintf('DTW Cost Matrix (CLOSED vs OPEN)\nDistance = %.4f', dist_diff1));

% Plot 5: Distance comparison bar chart
subplot(2, 3, 5);
distances = [dist_same, dist_diff1, dist_diff2];
b = bar(distances);
hold on;
yline(threshold, 'r--', 'LineWidth', 2);
set(gca, 'XTickLabel', {'CLOSED1-CLOSED2', 'CLOSED1-OPEN', 'CLOSED2-OPEN'});
ylabel('DTW Distance');
title('Distance Comparison');
legend('Distance', 'Threshold', 'Location', 'best');
grid on;

% Plot 6: Cosine distance explanation
subplot(2, 3, 6);
theta = linspace(0, pi, 100);
cos_dist = 1 - cos(theta);
plot(theta * 180/pi, cos_dist, 'b-', 'LineWidth', 2);
xlabel('Angle between vectors (degrees)');
ylabel('Cosine Distance');
title('Cosine Distance Function');
grid on;
text(45, 0.3, 'd = 1 - cos(\theta)', 'FontSize', 12);
text(5, 0.15, 'Identical', 'FontSize', 10, 'Color', 'g');
text(85, 1.15, 'Perpendicular', 'FontSize', 10);
text(155, 1.85, 'Opposite', 'FontSize', 10, 'Color', 'r');

sgtitle('DTW with Cosine Distance - MindMove Framework Demo', 'FontSize', 14, 'FontWeight', 'bold');

%% SUMMARY
fprintf('\n=== Summary ===\n');
fprintf('The DTW algorithm with cosine distance:\n');
fprintf('1. Segments EMG into overlapping windows (~96ms each)\n');
fprintf('2. Extracts features (Waveform Length) from each window\n');
fprintf('3. Computes cosine distance between corresponding windows\n');
fprintf('4. Uses dynamic programming to find optimal alignment\n');
fprintf('5. Accumulates distances along the alignment path\n');
fprintf('6. Lower total distance = more similar templates\n');
fprintf('\nKey insight: Cosine distance is DIMENSIONLESS (no units)\n');
fprintf('It measures the angle between feature vectors, not magnitude.\n');
fprintf('\nDone! Check the figure for visualization.\n');

%% ========================================================================
%  LOCAL FUNCTIONS (MATLAB R2016b+ required for local functions in scripts)
%  ========================================================================

function features = extract_wl_features(emg, window_size, window_shift)
    % Extract Waveform Length features from EMG using sliding windows
    %
    % Input:
    %   emg: nch x n_samples matrix
    %   window_size: samples per window
    %   window_shift: samples to shift between windows
    %
    % Output:
    %   features: n_windows x nch matrix of WL features

    [nch, n_samples] = size(emg);
    n_windows = floor((n_samples - window_size) / window_shift) + 1;
    features = zeros(n_windows, nch);

    for w = 1:n_windows
        start_idx = (w - 1) * window_shift + 1;
        end_idx = start_idx + window_size - 1;
        window = emg(:, start_idx:end_idx);

        % Waveform Length = sum of absolute differences
        for ch = 1:nch
            features(w, ch) = sum(abs(diff(window(ch, :))));
        end
    end
end

function features = extract_rms_features(emg, window_size, window_shift)
    % Extract Waveform Length features from EMG using sliding windows
    %
    % Input:
    %   emg: nch x n_samples matrix
    %   window_size: samples per window
    %   window_shift: samples to shift between windows
    %
    % Output:
    %   features: n_windows x nch matrix of WL features

    [nch, n_samples] = size(emg);
    n_windows = floor((n_samples - window_size) / window_shift) + 1;
    features = zeros(n_windows, nch);

    for w = 1:n_windows
        start_idx = (w - 1) * window_shift + 1;
        end_idx = start_idx + window_size - 1;
        window = emg(:, start_idx:end_idx);

        % Waveform Length = sum of absolute differences
        for ch = 1:nch
            features(w, ch) = sqrt(mean(window(ch, :).^2));
        end
    end
end

function d = cosine_distance(v1, v2)
    % Compute cosine distance between two vectors
    % d = 1 - cos(theta) = 1 - (v1 . v2) / (|v1| * |v2|)
    %
    % Range: [0, 2]
    %   0 = identical direction
    %   1 = perpendicular (90 degrees)
    %   2 = opposite direction (180 degrees)

    dot_product = dot(v1, v2);
    norm1 = norm(v1);
    norm2 = norm(v2);

    % Add small epsilon to avoid division by zero
    d = 1 - dot_product / (norm1 * norm2 + 1e-8);
end

function [alignment_cost, cost_matrix, path] = dtw_cosine(template1, template2)
    % Dynamic Time Warping with Cosine Distance
    %
    % This is the CORE ALGORITHM of the MindMove framework.
    %
    % Inputs:
    %   template1: N x nch matrix (N time windows, nch channels/features)
    %   template2: M x nch matrix (M time windows, nch channels/features)
    %
    % Outputs:
    %   alignment_cost: scalar DTW distance (LOWER = MORE SIMILAR)
    %   cost_matrix: accumulated cost matrix (for visualization)
    %   path: optimal warping path

    [N, ~] = size(template1);
    [M, ~] = size(template2);

    % Initialize cost matrix with infinity
    % Size is (N+1) x (M+1) to handle boundary conditions
    cost_matrix = inf(N + 1, M + 1);
    cost_matrix(1, 1) = 0;

    % Traceback matrix for path reconstruction (optional, for visualization)
    traceback = zeros(N, M);

    % =====================================================================
    % MAIN DTW LOOP - Dynamic Programming
    % =====================================================================
    for i = 1:N
        for j = 1:M
            % -----------------------------------------------------------------
            % Step 1: Compute LOCAL COST (cosine distance between windows)
            % -----------------------------------------------------------------
            % template1(i, :) = feature vector at window i of template 1
            % template2(j, :) = feature vector at window j of template 2
            dist = cosine_distance(template1(i, :), template2(j, :));

            % -----------------------------------------------------------------
            % Step 2: DTW RECURRENCE RELATION
            % cost(i,j) = local_cost + min(cost from three neighbors)
            % -----------------------------------------------------------------
            % Three possible predecessors:
            %   - Diagonal (i-1, j-1): "match" - both templates advance
            %   - Vertical (i-1, j):   "insertion" - template1 advances
            %   - Horizontal (i, j-1): "deletion" - template2 advances
            candidates = [
                cost_matrix(i, j),     ... % diagonal
                cost_matrix(i, j + 1),  ... % vertical
                cost_matrix(i + 1, j)   % horizontal
            ];

            [min_cost, direction] = min(candidates);
            cost_matrix(i + 1, j + 1) = dist + min_cost;
            traceback(i, j) = direction;
        end
    end

    % =====================================================================
    % FINAL RESULT: Total accumulated cost at (N, M)
    % =====================================================================
    alignment_cost = cost_matrix(N + 1, M + 1);

    % Reconstruct path for visualization
    path = reconstruct_path_local(traceback, N, M);
end

function path = reconstruct_path_local(traceback, N, M)
    % Reconstruct optimal warping path from traceback matrix
    path = [N, M];
    i = N;
    j = M;

    while i > 1 || j > 1
        if i == 1
            j = j - 1;
        elseif j == 1
            i = i - 1;
        else
            direction = traceback(i, j);
            if direction == 1      % diagonal
                i = i - 1;
                j = j - 1;
            elseif direction == 2  % vertical
                i = i - 1;
            else                   % horizontal
                j = j - 1;
            end
        end
        path = [i, j; path];
    end
end
