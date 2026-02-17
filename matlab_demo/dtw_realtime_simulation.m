%% DTW Real-Time Simulation
% Simulates real-time DTW classification using a sliding 1-second buffer
% with 50ms update interval, exactly like the Python framework.
%
% Usage:
%   1. Export model and recording: python export_templates_for_matlab.py
%   2. Update file paths below
%   3. Run this script

clear; clc; close all;

%% configuration

% Path to exported model (.mat file)
% model_file = 'model_MindMove_Model_sd_20260203_131149_protocol_test_alberto_96_32_wl_avg3_mv.mat';
% model_file = 'model_MindMove_Model_sd_20260203_125755_protocol_test_alberto_96_32_rms_avg3_mv.mat';
model_file = 'model_MindMove_Model_sd_20260206_121332_test_patient.mat';

% Path to exported recording (.mat file)
% recording_file = 'recording_recordings_MindMove_GuidedRecording_sd_20260203_124415596790_Validation_Alberto.mat';
recording_file = 'recording_recordings_MindMove_GuidedRecording_sd_20260206_115904408711_guided_1cycles.mat';
% recording_file = 'recording_recordings_MindMove_GuidedRecording_sd_20260206_120739441579_guided_4cycles.mat';
%%  STEP 1: LOAD MODEL

model = load(model_file);

% Extract key parameters
Fs = double(model.Fs);                          % Sampling frequency (Hz)
window_size = double(model.window_size);        % Window size (samples)
window_shift = double(model.window_shift);      % Window shift (samples)

% Extract templates (already feature-extracted)
% Shape: (n_templates, n_windows, n_channels)
templates_open = model.templates_open_features;
templates_closed = model.templates_closed_features;

% Get dimensions
[n_open, n_windows_template, n_channels] = size(templates_open);
n_closed = size(templates_closed, 1);

% Extract thresholds
% threshold_open = model.threshold_base_open;
% threshold_closed = model.threshold_base_closed;
threshold_open = 2.0;
threshold_closed = 2.0;

% Distance aggregation method from model
if isfield(model, 'distance_aggregation')
    distance_aggregation = char(model.distance_aggregation);
else
    distance_aggregation = 'average';
end

fprintf('  Model loaded: %s\n', model_file);
fprintf('  Distance aggregation: %s\n', distance_aggregation);
%% LOAD RECORDING
fprintf('Loading recording...\n');
recording = load(recording_file);

emg = recording.emg;%(:,1:25*Fs); % prende solo i primi 20 secondi
% emg = recording.emg;
gt = recording.gt;%(1:25*Fs);
[~, n_samples] = size(emg);

fprintf('  Recording: %.2f seconds\n', n_samples / Fs);

%% REAL-TIME SIMULATION PARAMETERS
buffer_duration_s = 1.0;
update_interval_s = 0.05;

buffer_samples = round(buffer_duration_s * Fs);
update_samples = round(update_interval_s * Fs);

n_updates = floor((n_samples - buffer_samples) / update_samples);

fprintf('  Buffer: %.0f ms, Update: %.0f ms\n', buffer_duration_s*1000, update_interval_s*1000);
fprintf('  Total updates: %d\n', n_updates);

%% RUN SIMULATION
fprintf('\nRunning real-time simulation...\n');

time_points = zeros(n_updates, 1);
dist_open_over_time = zeros(n_updates, 1);
dist_closed_over_time = zeros(n_updates, 1);
gt_over_time = zeros(n_updates, 1);

for k = 1:n_updates
    buffer_end = buffer_samples + (k - 1) * update_samples;
    buffer_start = buffer_end - buffer_samples + 1;

    buffer_emg = emg(:, buffer_start:buffer_end);

    time_points(k) = buffer_end / Fs;
    gt_over_time(k) = gt(buffer_end);

    buffer_features = extract_wl_features(buffer_emg, window_size, window_shift);
    % buffer_features = extract_rms_features(buffer_emg, window_size, window_shift);


    distances_open = zeros(n_open, 1);
    for t = 1:n_open
        template = squeeze(templates_open(t, :, :));
        [distances_open(t), ~, ~] = dtw_cosine(buffer_features, template);
    end
    dist_open_over_time(k) = aggregate_distances(distances_open, distance_aggregation);

    distances_closed = zeros(n_closed, 1);
    for t = 1:n_closed
        template = squeeze(templates_closed(t, :, :));
        [distances_closed(t), ~, ~] = dtw_cosine(buffer_features, template);
    end
    dist_closed_over_time(k) = aggregate_distances(distances_closed, distance_aggregation);

    if mod(k, 50) == 0
        fprintf('  Progress: %d/%d (%.1f%%)\n', k, n_updates, k/n_updates*100);
    end
end

fprintf('Simulation complete.\n');

%% VISUALIZATION
% figure('Position', [100, 100, 1200, 800]);
% 
ch_to_plot = 11; % cambiare canale se necessario
% ch_to_plot = input('Enter channel number to plot: ');
% for ch_to_plot = 2:16
    figure('Position', [100, 100, 1200, 800]);

    subplot(3, 1, 1);
    t_emg = (0:n_samples-1) / Fs;
    yyaxis left;
    plot(t_emg, emg(ch_to_plot, :), 'b', 'LineWidth', 0.5);
    ylabel(sprintf('EMG Ch %d (uV)', ch_to_plot));
    yyaxis right;
    plot(t_emg, gt, 'r', 'LineWidth', 1.5);
    ylabel('Ground Truth');
    ylim([-0.1, 1.1]);
    xlabel('Time (s)');
    title(sprintf('EMG Channel %d with Ground Truth', ch_to_plot));
    legend('EMG', 'GT (0=OPEN, 1=CLOSED)', 'Location', 'best');
    grid on;

    subplot(3, 1, 2);
    plot(time_points, dist_open_over_time, 'b', 'LineWidth', 1.5);
    hold on;
    yline(threshold_open, 'b--', 'LineWidth', 1.5);
    ylabel('Distance');
    xlabel('Time (s)');
    title(sprintf('Distance to OPEN Templates (threshold = %.4f)', threshold_open));
    legend('Avg Distance', 'Threshold', 'Location', 'best');
    grid on;

    subplot(3, 1, 3);
    plot(time_points, dist_closed_over_time, 'r', 'LineWidth', 1.5);
    hold on;
    yline(threshold_closed, 'r--', 'LineWidth', 1.5);
    ylabel('Distance');
    xlabel('Time (s)');
    title(sprintf('Distance to CLOSED Templates (threshold = %.4f)', threshold_closed));
    legend('Avg Distance', 'Threshold', 'Location', 'best');
    grid on;

    sgtitle('DTW Real-Time Simulation', 'FontSize', 14, 'FontWeight', 'bold');

% end

%% LOCAL FUNCTIONS

function d = aggregate_distances(distances, method)
    % Aggregate template distances using the specified method.
    % Matches Python: algorithm.py compute_distance_from_training_set_online()
    switch method
        case 'minimum'
            d = min(distances);
        case 'avg_3_smallest'
            n_smallest = min(3, length(distances));
            sorted_d = sort(distances);
            d = mean(sorted_d(1:n_smallest));
        otherwise  % 'average'
            d = mean(distances);
    end
end

function features = extract_rms_features(emg, window_size, window_shift)
    [nch, n_samples] = size(emg);
    n_windows = floor((n_samples - window_size) / window_shift) + 1;
    features = zeros(n_windows, nch);
    for w = 1:n_windows
        start_idx = (w - 1) * window_shift + 1;
        end_idx = start_idx + window_size - 1;
        window = emg(:, start_idx:end_idx);
        for ch = 1:nch
            features(w, ch) = sqrt(mean(window(ch, :).^2));
        end
    end
end

function features = extract_wl_features(emg, window_size, window_shift)
    [nch, n_samples] = size(emg);
    n_windows = floor((n_samples - window_size) / window_shift) + 1;
    features = zeros(n_windows, nch);
    for w = 1:n_windows
        start_idx = (w - 1) * window_shift + 1;
        end_idx = start_idx + window_size - 1;
        window = emg(:, start_idx:end_idx);
        for ch = 1:nch
            features(w, ch) = sum(abs(diff(window(ch, :))));
        end
    end
end

