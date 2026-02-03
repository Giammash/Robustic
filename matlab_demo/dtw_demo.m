%% DTW Step-by-Step Analysis
% This script loads an exported model and recording, then runs
% DTW classification step-by-step.
%
%
% Update the file paths below for model and test recording and run the script 

clear; clc; close all;

%% configuration

% Path to exported model (.mat file)
% model_file = 'data/model_MindMove_Model_sd_20260202_165447_SD_test_rms.mat';
% model_file = 'data/model_MindMove_Model_mp_20260129_183359_test_2_3.mat';
model_file = 'data/model_MindMove_Model_20260126_141327_test_01010.mat';

% Path to exported recording (.mat file)
recording_file = 'data/recording_27.01.2026_MindMove_GuidedRecording_mp_20260128_154438427220_guided_2cycles.mat';

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
threshold_open = model.threshold_base_open;
threshold_closed = model.threshold_base_closed;

fprintf('  Model loaded: %s\n', model_file);

%%  STEP 2: LOAD RECORDING

recording = load(recording_file);

% Extract EMG and ground truth
emg = recording.emg;           % Shape: (n_channels, n_samples)
gt = recording.gt;             % grounfd truth (n_samples,) - values 0 to 1

[emg_channels, n_samples] = size(emg);
duration_s = n_samples / Fs;

fprintf('  Recording loaded: %s\n', recording_file);

%%  STEP 3: EXTRACT FEATURES FROM RECORDING

% We need to extract the same features used in the model
% Default is RMS (Root Mean Square)

% Calculate number of windows
n_windows_recording = floor((n_samples - window_size) / window_shift) + 1;

fprintf('  Extracting RMS features...\n');
fprintf('  Number of windows: %d\n', n_windows_recording);

% Pre-allocate feature matrix
% Shape: (n_windows, n_channels)
recording_features = zeros(n_windows_recording, emg_channels);

% === BREAKPOINT HERE to see windowing ===
for w = 1:n_windows_recording
    % Calculate window boundaries
    start_idx = (w - 1) * window_shift + 1;
    end_idx = start_idx + window_size - 1;

    % Extract window from all channels
    window_data = emg(:, start_idx:end_idx);  % (n_channels, window_size)

    % Compute RMS (or wl) for each channel
    for ch = 1:emg_channels
        % recording_features(w, ch) = sqrt(mean(window_data(ch, :).^2));
        recording_features(w, ch) = sum(abs(diff(window_data(ch, :))));
    end
end

fprintf('  Features extracted: %d windows x %d channels\n', size(recording_features));

%%  STEP 4: SELECT TEST SEGMENT
fprintf('\n=== STEP 4: Select Test Segment ===\n');

% Template duration in windows
template_duration_windows = n_windows_template;


% Define test positions (in windows)
test_positions = round(linspace(1, n_windows_recording - template_duration_windows, 10));

fprintf('  Template duration: %d windows\n', template_duration_windows);
fprintf('  Testing %d positions along the recording\n', length(test_positions));

%%  STEP 5: DTW COMPUTATION (CORE ALGORITHM)

fprintf('\n=== STEP 5: DTW Distance Computation ===\n');

% Storage for results
distances_to_open = zeros(length(test_positions), n_open);
distances_to_closed = zeros(length(test_positions), n_closed);
avg_distance_open = zeros(length(test_positions), 1);
avg_distance_closed = zeros(length(test_positions), 1);
predicted_state = cell(length(test_positions), 1);
gt_at_position = zeros(length(test_positions), 1);

% === MAIN LOOP: Test each position ===
for pos_idx = 1:length(test_positions)
    start_window = test_positions(pos_idx);
    end_window = start_window + template_duration_windows - 1;

    % Extract test segment
    test_segment = recording_features(start_window:end_window, :);

    % Get ground truth at this position (middle of segment)
    middle_sample = ((start_window + end_window) / 2) * window_shift;
    gt_idx = min(round(middle_sample), length(gt));
    gt_at_position(pos_idx) = gt(gt_idx);

    % Compute DTW distance to each OPEN template
    for t = 1:n_open
        template = squeeze(templates_open(t, :, :));  % (n_windows, n_channels)

        [dist, ~, ~] = dtw_cosine(test_segment, template);

        distances_to_open(pos_idx, t) = dist;
    end

    % Compute DTW distance to each CLOSED template
    for t = 1:n_closed
        template = squeeze(templates_closed(t, :, :));

        [dist, ~, ~] = dtw_cosine(test_segment, template);

        distances_to_closed(pos_idx, t) = dist;
    end

    % Aggregate distances (average)
    avg_distance_open(pos_idx) = mean(distances_to_open(pos_idx, :));
    avg_distance_closed(pos_idx) = mean(distances_to_closed(pos_idx, :));

    % Classification decision
    % Current state assumed CLOSED, check if should transition to OPEN
    if avg_distance_open(pos_idx) < threshold_open
        predicted_state{pos_idx} = 'OPEN';
    else
        predicted_state{pos_idx} = 'CLOSED';
    end

    fprintf('  Position %2d: D_open=%.4f, D_closed=%.4f -> %s (GT=%.2f)\n', ...
        pos_idx, avg_distance_open(pos_idx), avg_distance_closed(pos_idx), ...
        predicted_state{pos_idx}, gt_at_position(pos_idx));
end

