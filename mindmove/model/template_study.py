"""
Template Study — Automatic Template Positioning via Cross-Cycle DTW Matching.

Standalone CLI tool (like offline_test.py) that:
1. Loads a guided recording and extracts cycles
2. Lets the user pick a reference window position in one cycle (visually)
3. Automatically finds the best-matching 1s window in every other cycle using DTW
4. Computes intra-class consistency metrics (separability ratio, silhouette score)
5. Constrains the search: CLOSED window between close cue and open cue,
   OPEN window after open cue

Usage:
    python -m mindmove.model.template_study
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from itertools import combinations

from mindmove.config import config
from mindmove.model.core.algorithm import compute_dtw, compute_dtw_multivariate
from mindmove.model.core.windowing import sliding_window
from mindmove.model.core.features.features_registry import FEATURES
from mindmove.model.templates.template_manager import TemplateManager

# ── Configuration ────────────────────────────────────────────────────────────

# Mode: "auto" = DTW auto-search only, "compare" = user picks vs DTW picks,
#       "onset" = automatic onset detection placement
MODE = "onset"

RECORDING_FILE = "MindMove_GuidedRecording_sd_20260206_120536761870_guided_16cycles.pkl"
# RECORDING_FILE = "MindMove_GuidedRecording_sd_20260206_120739441579_guided_4cycles.pkl"

PATIENT_DIR = "data/recordings/patient S1"
FEATURE = "rms"                                 # Single-feature mode
FEATURES_MULTI = ["rms", "ssi"]     # Multi-feature mode (set to None to disable)
TEMPLATE_DURATION_S = 1.0
SEARCH_STEP_S = 0.05       # 50 ms sliding step for search
WINDOW_LENGTH = 192         # Feature window size (samples)
WINDOW_INCREMENT = 64       # Feature window step (samples)
PRE_CLOSE_S = 2.0           # Seconds before close transition to include in cycle
POST_OPEN_S = 2.0           # Seconds after open transition to include in cycle
PLOT_RAW_EMG = True         # True: plot raw EMG channels, False: plot feature-extracted signal

# Onset detection parameters
ONSET_ENVELOPE_WINDOW_S = 0.05   # 50ms RMS envelope window
ONSET_BASELINE_DURATION_S = 0.5  # Duration after cue to estimate baseline (patient hasn't reacted)
ONSET_THRESHOLD_K = 3.0          # Onset = baseline_mean + k * baseline_std
ONSET_MIN_SUSTAINED_S = 0.1     # Onset must be sustained for this long (100ms)
ONSET_PRE_FRACTION = 0.2         # Fraction of template duration before onset (20% = 200ms for 1s)
ONSET_METHOD = "threshold"       # "threshold" or "cusum"
# CUSUM parameters (in units of baseline std)
CUSUM_DRIFT = 0.5                # Allowable drift before accumulating
CUSUM_H = 5.0                    # Decision threshold for cumulative sum
# TKEO transition detector parameters
ONSET_ANTICIPATORY_S = 0.5      # Search extends this far before the audio cue
TKEO_MIN_PEAK_RATIO = 3.0       # Peak/median noise ratio required for a valid detection
TKEO_CONTRIBUTION_FRACTION = 0.3  # Channels above this fraction of max contribution → 'channels_fired'

# Signal quality parameters
DEAD_CHANNEL_EPSILON = 1e-6       # Std threshold for dead channel
ARTIFACT_CROSS_CH_FACTOR = 5.0    # Channel RMS vs median of others
ARTIFACT_WITHIN_CH_FACTOR = 10.0  # Channel RMS in template window vs baseline


# ── Core functions ───────────────────────────────────────────────────────────

def load_and_extract_cycles(recording_path: str) -> list:
    """Load .pkl recording, extract cycles via TemplateManager."""
    with open(recording_path, "rb") as f:
        recording = pickle.load(f)

    # Infer channel count from recording
    emg = recording.get("emg", recording.get("biosignal"))
    if emg is not None:
        n_ch = emg.shape[0]
        if n_ch != config.num_channels:
            print(f"[INFO] Adjusting config.num_channels: {config.num_channels} -> {n_ch}")
            config.num_channels = n_ch
            config.active_channels = [i for i in range(n_ch) if i not in config.dead_channels]

    tm = TemplateManager()
    cycles = tm.extract_complete_cycles(recording, pre_close_s=PRE_CLOSE_S, post_open_s=POST_OPEN_S)
    return cycles


def extract_features_from_segment(emg_segment: np.ndarray, feature_name: str = FEATURE) -> np.ndarray:
    """
    Raw EMG (n_ch, n_samples) -> sliding_window -> feature -> (n_windows, n_ch).
    """
    windowed = sliding_window(emg_segment, WINDOW_LENGTH, WINDOW_INCREMENT)
    # windowed shape: (n_windows, n_ch, window_length)
    feature_fn = FEATURES[feature_name]["function"]
    features = feature_fn(windowed)
    # features shape: (n_windows, n_ch)
    return features


def extract_multi_features_from_segment(emg_segment: np.ndarray, feature_names: list) -> list:
    """
    Raw EMG (n_ch, n_samples) -> sliding_window -> multiple features.

    Returns:
        list of np.ndarray, each (n_windows, n_ch) — one per feature
    """
    windowed = sliding_window(emg_segment, WINDOW_LENGTH, WINDOW_INCREMENT)
    result = []
    for fname in feature_names:
        feature_fn = FEATURES[fname]["function"]
        result.append(feature_fn(windowed))
    return result


def find_best_match(
    reference_features: np.ndarray,
    cycle_emg: np.ndarray,
    search_start: int,
    search_end: int,
    feature_name: str = FEATURE,
    step_samples: int = None,
) -> tuple:
    """
    Slide a 1s window in [search_start, search_end] of cycle_emg,
    compute DTW to reference, return (best_pos, best_dist, positions, dist_curve).
    """
    if step_samples is None:
        step_samples = int(SEARCH_STEP_S * config.FSAMP)

    template_samples = int(TEMPLATE_DURATION_S * config.FSAMP)

    # Clamp search bounds
    search_start = max(0, search_start)
    search_end = min(cycle_emg.shape[1] - template_samples, search_end)

    if search_start > search_end:
        return (search_start, float('inf'), [], np.array([]))

    positions = list(range(search_start, search_end + 1, step_samples))
    if not positions:
        return (search_start, float('inf'), [], np.array([]))

    distances = np.full(len(positions), np.inf)

    for i, pos in enumerate(positions):
        segment = cycle_emg[:, pos:pos + template_samples]
        if segment.shape[1] < template_samples:
            continue
        candidate_features = extract_features_from_segment(segment, feature_name)
        distances[i] = compute_dtw(reference_features, candidate_features)

    best_idx = np.argmin(distances)
    return (positions[best_idx], distances[best_idx], positions, distances)


def find_best_match_multi(
    reference_feature_arrays: list,
    cycle_emg: np.ndarray,
    search_start: int,
    search_end: int,
    feature_names: list,
    step_samples: int = None,
) -> tuple:
    """
    Multivariate version: slide a 1s window, compute DTW with per-feature
    cosine distances summed into one cost matrix.

    Returns:
        (best_pos, best_dist, positions, total_dists, per_feature_dists)
        where per_feature_dists is (n_positions, n_features)
    """
    if step_samples is None:
        step_samples = int(SEARCH_STEP_S * config.FSAMP)

    template_samples = int(TEMPLATE_DURATION_S * config.FSAMP)
    n_features = len(feature_names)

    search_start = max(0, search_start)
    search_end = min(cycle_emg.shape[1] - template_samples, search_end)

    if search_start > search_end:
        return (search_start, float('inf'), [], np.array([]), np.empty((0, n_features)))

    positions = list(range(search_start, search_end + 1, step_samples))
    if not positions:
        return (search_start, float('inf'), [], np.array([]), np.empty((0, n_features)))

    total_dists = np.full(len(positions), np.inf)
    per_feature_dists = np.full((len(positions), n_features), np.inf)

    for i, pos in enumerate(positions):
        segment = cycle_emg[:, pos:pos + template_samples]
        if segment.shape[1] < template_samples:
            continue
        cand_arrays = extract_multi_features_from_segment(segment, feature_names)
        result = compute_dtw_multivariate(reference_feature_arrays, cand_arrays)
        total_dists[i] = result['total']
        per_feature_dists[i] = result['per_feature']

    best_idx = np.argmin(total_dists)
    return (positions[best_idx], total_dists[best_idx], positions, total_dists, per_feature_dists)


# ── Onset detection ─────────────────────────────────────────────────────────

def compute_rms_envelope(emg: np.ndarray, window_s: float = ONSET_ENVELOPE_WINDOW_S) -> tuple:
    """
    Compute per-channel RMS envelope and max-across-channels envelope.

    Args:
        emg: (n_channels, n_samples)
        window_s: RMS window duration in seconds

    Returns:
        per_channel_env: (n_channels, n_env_points) per-channel RMS envelopes
        max_env: (n_env_points,) max across channels at each time point
        env_time: (n_env_points,) time in seconds of each envelope point
    """
    window_samples = int(window_s * config.FSAMP)
    step_samples = window_samples // 2  # 50% overlap
    n_ch, n_samples = emg.shape

    n_points = max(1, (n_samples - window_samples) // step_samples + 1)
    per_channel_env = np.zeros((n_ch, n_points))

    for i in range(n_points):
        start = i * step_samples
        end = start + window_samples
        segment = emg[:, start:end]
        per_channel_env[:, i] = np.sqrt(np.mean(segment ** 2, axis=1))

    max_env = np.max(per_channel_env, axis=0)
    env_time = (np.arange(n_points) * step_samples + window_samples / 2) / config.FSAMP

    return per_channel_env, max_env, env_time


def detect_onset_per_channel(
    emg: np.ndarray,
    search_start: int,
    search_end: int,
    baseline_start: int = None,
    k: float = ONSET_THRESHOLD_K,
) -> dict:
    """
    Detect EMG onset independently per channel within [search_start, search_end].

    Each channel gets its own baseline and threshold. Returns per-channel onset
    times and a combined "earliest onset among channels that fired".

    Args:
        emg: (n_channels, n_samples) full cycle EMG
        search_start: sample index to start searching
        search_end: sample index to stop searching
        baseline_start: sample index for baseline estimation start
        k: threshold multiplier

    Returns:
        dict with:
            per_channel_onset: list of onset_sample per channel (None if not detected)
            channels_fired: list of channel indices that detected onset
            earliest_onset: sample of earliest detected onset (None if none)
            envelope_data: dict with per_channel_env, env_time, per_channel_baseline,
                           per_channel_threshold, search_start_s, search_end_s
    """
    per_ch_env, max_env, env_time = compute_rms_envelope(emg)
    n_ch = per_ch_env.shape[0]

    # Convert sample indices to envelope indices
    env_step = int(ONSET_ENVELOPE_WINDOW_S * config.FSAMP) // 2
    search_start_env = max(0, search_start // env_step)
    search_end_env = min(len(env_time), search_end // env_step)

    # Baseline region
    if baseline_start is None:
        baseline_start = search_start
    baseline_start_env = max(0, baseline_start // env_step)
    baseline_duration_env = int(ONSET_BASELINE_DURATION_S / (ONSET_ENVELOPE_WINDOW_S / 2))
    baseline_end_env = min(baseline_start_env + baseline_duration_env, search_end_env)
    if baseline_end_env <= baseline_start_env:
        baseline_end_env = min(baseline_start_env + 5, len(env_time))

    min_sustained_env = max(1, int(ONSET_MIN_SUSTAINED_S / (ONSET_ENVELOPE_WINDOW_S / 2)))

    per_channel_onset = []
    per_channel_baseline = []
    per_channel_threshold = []

    for ch in range(n_ch):
        ch_env = per_ch_env[ch]
        baseline_seg = ch_env[baseline_start_env:baseline_end_env]
        bl_mean = np.mean(baseline_seg)
        bl_std = np.std(baseline_seg)
        thr = bl_mean + k * bl_std
        per_channel_baseline.append(bl_mean)
        per_channel_threshold.append(thr)

        # Sustained crossing detection for this channel
        above = ch_env[search_start_env:search_end_env] > thr
        onset_env_idx = None
        consecutive = 0
        for i in range(len(above)):
            if above[i]:
                consecutive += 1
                if consecutive >= min_sustained_env:
                    onset_env_idx = search_start_env + i - min_sustained_env + 1
                    break
            else:
                consecutive = 0

        if onset_env_idx is not None:
            per_channel_onset.append(int(env_time[onset_env_idx] * config.FSAMP))
        else:
            per_channel_onset.append(None)

    channels_fired = [ch for ch in range(n_ch) if per_channel_onset[ch] is not None]
    fired_onsets = [per_channel_onset[ch] for ch in channels_fired]
    mean_onset = int(np.mean(fired_onsets)) if fired_onsets else None

    envelope_data = {
        "per_channel_env": per_ch_env,
        "max_env": max_env,
        "env_time": env_time,
        "per_channel_baseline": per_channel_baseline,
        "per_channel_threshold": per_channel_threshold,
        "search_start_s": search_start / config.FSAMP,
        "search_end_s": search_end / config.FSAMP,
    }

    return {
        "per_channel_onset": per_channel_onset,
        "channels_fired": channels_fired,
        "earliest_onset": mean_onset,
        "envelope_data": envelope_data,
    }


def detect_onset_per_channel_cusum(
    emg: np.ndarray,
    search_start: int,
    search_end: int,
    baseline_start: int = None,
    drift: float = CUSUM_DRIFT,
    h: float = CUSUM_H,
) -> dict:
    """
    Detect EMG onset per channel using CUSUM (Cumulative Sum) change-point detection.

    Unlike threshold crossing, CUSUM detects sustained shifts in signal level
    regardless of the starting level. Works for:
    - Onset from rest (classic case)
    - Onset on top of a previous activation (change from one level to another)
    - Subtle but sustained changes that never cross a fixed threshold

    The algorithm accumulates deviations above the expected level (baseline mean + drift).
    When the cumulative sum exceeds h * baseline_std, a change point is declared.

    Args:
        emg: (n_channels, n_samples) full cycle EMG
        search_start: sample index to start searching
        search_end: sample index to stop searching
        baseline_start: sample index for baseline estimation start
        drift: allowable drift before accumulating (in baseline std units)
        h: decision threshold (in baseline std units)

    Returns:
        Same structure as detect_onset_per_channel.
    """
    per_ch_env, max_env, env_time = compute_rms_envelope(emg)
    n_ch = per_ch_env.shape[0]

    env_step = int(ONSET_ENVELOPE_WINDOW_S * config.FSAMP) // 2
    search_start_env = max(0, search_start // env_step)
    search_end_env = min(len(env_time), search_end // env_step)

    if baseline_start is None:
        baseline_start = search_start
    baseline_start_env = max(0, baseline_start // env_step)
    baseline_duration_env = int(ONSET_BASELINE_DURATION_S / (ONSET_ENVELOPE_WINDOW_S / 2))
    baseline_end_env = min(baseline_start_env + baseline_duration_env, search_end_env)
    if baseline_end_env <= baseline_start_env:
        baseline_end_env = min(baseline_start_env + 5, len(env_time))

    per_channel_onset = []
    per_channel_baseline = []
    per_channel_threshold = []
    per_channel_cusum = []  # Store CUSUM traces for diagnostics

    for ch in range(n_ch):
        ch_env = per_ch_env[ch]
        baseline_seg = ch_env[baseline_start_env:baseline_end_env]
        bl_mean = np.mean(baseline_seg)
        bl_std = np.std(baseline_seg)
        if bl_std == 0:
            bl_std = 1e-10
        per_channel_baseline.append(bl_mean)
        per_channel_threshold.append(h * bl_std)  # For display purposes

        # CUSUM: accumulate positive deviations from (baseline_mean + drift*std)
        search_signal = ch_env[search_start_env:search_end_env]
        allowance = bl_mean + drift * bl_std
        decision_level = h * bl_std

        cusum = np.zeros(len(search_signal))
        onset_env_idx = None
        s_pos = 0.0  # Cumulative sum (positive direction)

        for i in range(len(search_signal)):
            s_pos = max(0, s_pos + (search_signal[i] - allowance))
            cusum[i] = s_pos
            if s_pos > decision_level and onset_env_idx is None:
                # Walk back to find where CUSUM started rising from zero
                start_idx = i
                while start_idx > 0 and cusum[start_idx - 1] > 0:
                    start_idx -= 1
                onset_env_idx = search_start_env + start_idx

        per_channel_cusum.append(cusum)

        if onset_env_idx is not None:
            per_channel_onset.append(int(env_time[onset_env_idx] * config.FSAMP))
        else:
            per_channel_onset.append(None)

    channels_fired = [ch for ch in range(n_ch) if per_channel_onset[ch] is not None]
    fired_onsets = [per_channel_onset[ch] for ch in channels_fired]
    mean_onset = int(np.mean(fired_onsets)) if fired_onsets else None

    envelope_data = {
        "per_channel_env": per_ch_env,
        "max_env": max_env,
        "env_time": env_time,
        "per_channel_baseline": per_channel_baseline,
        "per_channel_threshold": per_channel_threshold,
        "per_channel_cusum": per_channel_cusum,
        "search_start_s": search_start / config.FSAMP,
        "search_end_s": search_end / config.FSAMP,
        "search_start_env": search_start_env,
    }

    return {
        "per_channel_onset": per_channel_onset,
        "channels_fired": channels_fired,
        "earliest_onset": mean_onset,
        "envelope_data": envelope_data,
    }


def detect_transition_onset(
    emg: np.ndarray,
    search_start: int,
    search_end: int,
    min_peak_ratio: float = TKEO_MIN_PEAK_RATIO,
    contribution_fraction: float = TKEO_CONTRIBUTION_FRACTION,
) -> dict:
    """
    Detect the onset of a state transition using |d(TKEO_envelope)/dt| summed
    across channels.

    Direction-agnostic: channels activating (rising TKEO) and channels
    deactivating (falling TKEO) both produce a large absolute derivative at the
    same moment and therefore both contribute to the detection. This handles:

    - CLOSED onset from an open posture: extensors deactivate + flexors activate
    - OPEN onset from a closed posture:  flexors deactivate + extensors activate
    - Any mix of simultaneously-changing channels

    Silent channels throughout the cycle contribute near-zero derivative and
    self-exclude without any explicit dead-channel logic.

    Args:
        emg: (n_channels, n_samples) full cycle EMG (already filtered)
        search_start: first sample index of the search window
        search_end: last sample index of the search window
        min_peak_ratio: peak / median(combined) must exceed this ratio.
            Prevents detecting noise as a transition when no clear event exists.
        contribution_fraction: channels whose |d(TKEO_env)| at the peak exceeds
            this fraction of the maximum are labelled 'channels_fired'.

    Returns:
        dict with keys:
            earliest_onset: sample index of the detected transition (None if not found)
            channels_fired: list of channel indices most active at the transition
            channel_contributions: (n_ch,) array of each channel's contribution at peak
            envelope_data: dict with 'combined', 'env_time', 'search_start_s', 'search_end_s'
    """
    n_ch, n_samples = emg.shape
    _empty = {
        "earliest_onset": None,
        "channels_fired": [],
        "channel_contributions": np.zeros(n_ch),
        "envelope_data": {
            "combined": np.array([]),
            "env_time": np.array([]),
            "search_start_s": search_start / config.FSAMP,
            "search_end_s": search_end / config.FSAMP,
        },
    }

    if search_end <= search_start or n_samples < 3:
        return _empty

    # --- Step 1: TKEO per channel ---
    # psi[n] = x[n]^2 - x[n-1]*x[n+1]  (instantaneous energy, ~5ms effective window)
    psi = emg[:, 1:-1] ** 2 - emg[:, :-2] * emg[:, 2:]  # (n_ch, n_samples-2)
    psi = np.abs(psi)  # TKEO is theoretically non-negative; abs guards against noise

    # --- Step 2: RMS envelope of TKEO per channel (same grid as amplitude detector) ---
    psi_env, _, env_time = compute_rms_envelope(psi)  # (n_ch, n_env)

    if psi_env.shape[1] < 2:
        return _empty

    # --- Step 3: |d(TKEO_envelope)/dt| per channel ---
    d_psi = np.abs(np.diff(psi_env, axis=1))  # (n_ch, n_env-1)

    # --- Step 4: Sum across channels ---
    # Both rising and falling channels contribute positively to the same peak
    combined = np.sum(d_psi, axis=0)  # (n_env-1,)

    # --- Step 5: Map search window to envelope indices ---
    env_step = int(ONSET_ENVELOPE_WINDOW_S * config.FSAMP) // 2
    search_start_env = max(0, search_start // env_step)
    search_end_env = min(combined.shape[0], search_end // env_step)

    # Align env_time with d_psi (diff reduces length by 1; use midpoint convention)
    d_env_time = env_time[:-1] + (env_time[1] - env_time[0]) / 2 if len(env_time) > 1 else env_time

    envelope_data = {
        "combined": combined,
        "env_time": d_env_time,
        "search_start_s": search_start / config.FSAMP,
        "search_end_s": search_end / config.FSAMP,
    }

    if search_end_env <= search_start_env:
        return {**_empty, "envelope_data": envelope_data}

    # --- Step 6: Find peak within search window ---
    window = combined[search_start_env:search_end_env]
    peak_idx_local = int(np.argmax(window))
    peak_val = float(window[peak_idx_local])

    # --- Step 7: Quality gate — peak must stand out against noise floor ---
    noise_floor = float(np.median(window))
    if noise_floor <= 0 or peak_val < noise_floor * min_peak_ratio:
        return {**_empty, "envelope_data": envelope_data}

    peak_env_idx = search_start_env + peak_idx_local
    onset_sample = int(d_env_time[min(peak_env_idx, len(d_env_time) - 1)] * config.FSAMP)

    # --- Step 8: Per-channel contribution at the peak (±2 envelope bins) ---
    hw = 2
    peak_lo = max(0, peak_env_idx - hw)
    peak_hi = min(d_psi.shape[1], peak_env_idx + hw + 1)
    per_channel_contribution = np.mean(d_psi[:, peak_lo:peak_hi], axis=1)  # (n_ch,)

    max_contrib = float(np.max(per_channel_contribution))
    if max_contrib > 0:
        channels_fired = [
            ch for ch in range(n_ch)
            if per_channel_contribution[ch] >= max_contrib * contribution_fraction
        ]
    else:
        channels_fired = []

    return {
        "earliest_onset": onset_sample,
        "channels_fired": channels_fired,
        "channel_contributions": per_channel_contribution,
        "envelope_data": envelope_data,
    }


def place_template_at_onset(onset_sample: int, cycle_emg_length: int) -> int:
    """
    Place template window so that ~20% is pre-onset (preparation) and ~80% is activation.

    Returns:
        window_start_sample
    """
    template_samples = int(TEMPLATE_DURATION_S * config.FSAMP)
    pre_samples = int(ONSET_PRE_FRACTION * template_samples)
    start = onset_sample - pre_samples
    # Clamp to valid range
    start = max(0, min(start, cycle_emg_length - template_samples))
    return start


def detect_dead_channels(emg: np.ndarray, epsilon: float = DEAD_CHANNEL_EPSILON) -> list:
    """
    Detect dead (flat-line) channels based on standard deviation.

    Args:
        emg: (n_channels, n_samples)
        epsilon: std threshold below which a channel is considered dead

    Returns:
        List of 0-indexed dead channel indices.
    """
    stds = np.std(emg, axis=1)
    return [int(ch) for ch in range(emg.shape[0]) if stds[ch] < epsilon]


def detect_artifact_channels(
    emg: np.ndarray,
    template_start: int,
    template_end: int,
    baseline_start: int,
    baseline_end: int,
    cross_channel_factor: float = ARTIFACT_CROSS_CH_FACTOR,
    within_channel_factor: float = ARTIFACT_WITHIN_CH_FACTOR,
    adjacent_factor: float = 3.0,
) -> dict:
    """
    Detect artifact channels by comparing RMS in the template window
    against (a) the median of other channels, (b) a baseline window, and
    (c) adjacent (neighbouring) channels.

    All three checks must trigger to flag a channel as artifact. The adjacent
    check prevents false positives on channels with low baseline but genuine
    activation that is comparable to their physical neighbours.

    Args:
        emg: (n_channels, n_samples)
        template_start: start sample of the template window
        template_end: end sample of the template window
        baseline_start: start sample of the baseline window
        baseline_end: end sample of the baseline window
        cross_channel_factor: flag if channel RMS > factor × median of others
        within_channel_factor: flag if template RMS > factor × baseline RMS
        adjacent_factor: flag if channel RMS > factor × mean RMS of adjacent channels

    Returns:
        dict with:
            artifact_channels: list of 0-indexed flagged channels
            channel_rms: per-channel RMS in template window
            baseline_rms: per-channel RMS in baseline window
            cross_channel_median: median RMS across channels in template window
    """
    n_ch = emg.shape[0]

    # RMS in template window
    template_seg = emg[:, template_start:template_end]
    channel_rms = np.sqrt(np.mean(template_seg ** 2, axis=1))

    # RMS in baseline window
    baseline_seg = emg[:, baseline_start:baseline_end]
    baseline_rms = np.sqrt(np.mean(baseline_seg ** 2, axis=1))

    artifact_channels = []
    cross_channel_median = np.median(channel_rms)

    for ch in range(n_ch):
        # Cross-channel check: channel RMS vs median of all OTHER channels
        other_rms = np.concatenate([channel_rms[:ch], channel_rms[ch+1:]])
        other_median = np.median(other_rms) if len(other_rms) > 0 else 0
        cross_flagged = other_median > 0 and channel_rms[ch] > cross_channel_factor * other_median

        # Within-channel check: template RMS vs baseline RMS
        within_flagged = baseline_rms[ch] > 0 and channel_rms[ch] > within_channel_factor * baseline_rms[ch]

        # Adjacent channel check: channel RMS vs mean RMS of immediate neighbours
        # A real activation shows up on neighbouring channels too; an artifact is isolated
        neighbours = []
        if ch > 0:
            neighbours.append(channel_rms[ch - 1])
        if ch < n_ch - 1:
            neighbours.append(channel_rms[ch + 1])
        if neighbours:
            adj_mean = np.mean(neighbours)
            adjacent_flagged = adj_mean > 0 and channel_rms[ch] > adjacent_factor * adj_mean
        else:
            adjacent_flagged = False

        # All three checks must trigger
        if cross_flagged and within_flagged and adjacent_flagged:
            artifact_channels.append(ch)

    return {
        "artifact_channels": artifact_channels,
        "channel_rms": channel_rms,
        "baseline_rms": baseline_rms,
        "cross_channel_median": cross_channel_median,
    }


def plot_onset_detection(
    cycles: list,
    closed_results: dict,
    open_results: dict,
    closed_positions: dict,
    open_positions: dict,
):
    """
    Diagnostic plot: for each cycle, show per-channel RMS envelopes with detected
    onsets marked per channel. Two columns: CLOSED (left), OPEN (right).
    Channels that fired are colored, others are gray.
    """
    n_cycles = len(cycles)
    fig, axes = plt.subplots(n_cycles, 2, figsize=(18, 4 * n_cycles), sharex=False)
    if n_cycles == 1:
        axes = axes.reshape(1, 2)

    template_duration_s = TEMPLATE_DURATION_S
    cmap = plt.cm.tab20

    for idx, cycle in enumerate(cycles):
        cn = cycle["cycle_number"]
        n_ch = cycle["emg"].shape[0]

        for col, (class_name, results, positions) in enumerate([
            ("CLOSED", closed_results, closed_positions),
            ("OPEN", open_results, open_positions),
        ]):
            ax = axes[idx, col]
            result = results.get(cn)
            if result is None:
                ax.set_title(f"Cycle {cn} — {class_name}: no data")
                continue

            env_data = result["envelope_data"]
            per_ch_env = env_data["per_channel_env"]
            env_time = env_data["env_time"]
            channels_fired = result["channels_fired"]
            per_ch_onset = result["per_channel_onset"]

            # Plot each channel's envelope
            for ch in range(n_ch):
                if ch in channels_fired:
                    color_ch = cmap(ch % 20)
                    ax.plot(env_time, per_ch_env[ch], color=color_ch, linewidth=1.2,
                            alpha=0.9, label=f"CH{ch+1}")
                    # Mark onset with vertical tick
                    onset_s = per_ch_onset[ch] / config.FSAMP
                    ax.axvline(onset_s, color=color_ch, linewidth=1, alpha=0.5, linestyle=":")
                else:
                    ax.plot(env_time, per_ch_env[ch], color="lightgray", linewidth=0.5, alpha=0.4)

            # Template window
            if cn in positions:
                pos_s = positions[cn] / config.FSAMP
                y_hi = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else np.max(per_ch_env) * 1.1
                ax.set_ylim(bottom=0)
                class_color = "#FF5722" if class_name == "CLOSED" else "#2196F3"
                rect = Rectangle((pos_s, 0), template_duration_s, y_hi,
                                  facecolor=class_color, alpha=0.12, edgecolor=class_color, linewidth=2)
                ax.add_patch(rect)

            # Cue marker
            if class_name == "CLOSED" and cycle.get("close_cue_idx") is not None:
                ax.axvline(cycle["close_cue_idx"] / config.FSAMP, color='#FF5722',
                           linestyle='--', linewidth=1.5, alpha=0.6)
            if class_name == "OPEN" and cycle.get("open_cue_idx") is not None:
                ax.axvline(cycle["open_cue_idx"] / config.FSAMP, color='#2196F3',
                           linestyle='--', linewidth=1.5, alpha=0.6)

            n_fired = len(channels_fired)
            fired_str = ",".join(str(ch+1) for ch in channels_fired) if channels_fired else "none"
            earliest = result["earliest_onset"]
            onset_str = f"{earliest/config.FSAMP:.2f}s" if earliest is not None else "N/A"
            ax.set_title(f"Cycle {cn} — {class_name}: {n_fired} ch fired [{fired_str}], earliest={onset_str}",
                         fontsize=8)
            ax.set_ylabel("RMS", fontsize=8)
            if idx == 0 and channels_fired:
                ax.legend(fontsize=5, loc="upper right", ncol=2)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(f"Per-Channel Onset Detection (k={ONSET_THRESHOLD_K}, "
                 f"sustained={ONSET_MIN_SUSTAINED_S}s, baseline={ONSET_BASELINE_DURATION_S}s)",
                 fontsize=11)
    fig.tight_layout()
    return fig


def compute_template_metrics(templates_open: list, templates_closed: list) -> dict:
    """
    Compute full distance matrix, separability ratio, silhouette scores.

    Args:
        templates_open: list of feature arrays (n_windows, n_ch) for OPEN class
        templates_closed: list of feature arrays (n_windows, n_ch) for CLOSED class

    Returns:
        dict with distance_matrix, labels, separability_ratio, silhouette_score,
        per_template_silhouette, intra_open, intra_closed, inter
    """
    all_templates = templates_closed + templates_open
    n_closed = len(templates_closed)
    n_open = len(templates_open)
    n_total = n_closed + n_open
    labels = ["CLOSED"] * n_closed + ["OPEN"] * n_open

    # Full pairwise distance matrix
    dist_matrix = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(i + 1, n_total):
            d = compute_dtw(all_templates[i], all_templates[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Intra-class distances
    intra_closed_dists = []
    for i, j in combinations(range(n_closed), 2):
        intra_closed_dists.append(dist_matrix[i, j])

    intra_open_dists = []
    for i, j in combinations(range(n_closed, n_total), 2):
        intra_open_dists.append(dist_matrix[i, j])

    # Inter-class distances
    inter_dists = []
    for i in range(n_closed):
        for j in range(n_closed, n_total):
            inter_dists.append(dist_matrix[i, j])

    intra_closed_mean = np.mean(intra_closed_dists) if intra_closed_dists else 0
    intra_closed_std = np.std(intra_closed_dists) if intra_closed_dists else 0
    intra_open_mean = np.mean(intra_open_dists) if intra_open_dists else 0
    intra_open_std = np.std(intra_open_dists) if intra_open_dists else 0
    inter_mean = np.mean(inter_dists) if inter_dists else 0
    inter_std = np.std(inter_dists) if inter_dists else 0

    intra_mean = np.mean(intra_closed_dists + intra_open_dists) if (intra_closed_dists + intra_open_dists) else 0
    separability_ratio = inter_mean / intra_mean if intra_mean > 0 else float('inf')

    # Per-template silhouette: s(i) = (b(i) - a(i)) / max(a(i), b(i))
    silhouettes = np.zeros(n_total)
    for i in range(n_total):
        same_class = [j for j in range(n_total) if labels[j] == labels[i] and j != i]
        other_class = [j for j in range(n_total) if labels[j] != labels[i]]

        a_i = np.mean([dist_matrix[i, j] for j in same_class]) if same_class else 0
        b_i = np.mean([dist_matrix[i, j] for j in other_class]) if other_class else 0

        denom = max(a_i, b_i)
        silhouettes[i] = (b_i - a_i) / denom if denom > 0 else 0

    return {
        "distance_matrix": dist_matrix,
        "labels": labels,
        "separability_ratio": separability_ratio,
        "silhouette_score": np.mean(silhouettes),
        "per_template_silhouette": silhouettes,
        "intra_closed": {"mean": intra_closed_mean, "std": intra_closed_std},
        "intra_open": {"mean": intra_open_mean, "std": intra_open_std},
        "inter": {"mean": inter_mean, "std": inter_std},
    }


def compute_per_class_channels(onset_info: list, recordings: list, dead_channels: list = None) -> dict:
    """
    Assign EMG channels to OPEN or CLOSED class based on onset firing patterns.

    For channels that fire for both classes, compare mean RMS amplitude during
    GT=CLOSED vs GT=OPEN periods across all recordings and assign to the class
    with higher amplitude.

    Parameters
    ----------
    onset_info : list of dict
        Per-cycle onset info, each with 'closed_channels_fired' and 'open_channels_fired'.
    recordings : list of dict
        Recording dicts with 'emg' (n_channels, n_samples) and 'gt' (n_samples,).
    dead_channels : list of int, optional
        0-indexed dead channel indices to exclude.

    Returns
    -------
    dict with keys:
        "closed": list of 0-indexed channels assigned to CLOSED
        "open": list of 0-indexed channels assigned to OPEN
        "unassigned": list of channels that never fired
        "details": per-channel info (firing counts, amplitude ratios)
    """
    if dead_channels is None:
        dead_channels = []

    # Step 1: Aggregate firing counts across cycles
    closed_counts = {}  # channel -> number of cycles it fired for CLOSED
    open_counts = {}    # channel -> number of cycles it fired for OPEN

    for info in onset_info:
        for ch in info.get("closed_channels_fired", []):
            closed_counts[ch] = closed_counts.get(ch, 0) + 1
        for ch in info.get("open_channels_fired", []):
            open_counts[ch] = open_counts.get(ch, 0) + 1

    all_fired = set(closed_counts.keys()) | set(open_counts.keys())
    closed_only = set(closed_counts.keys()) - set(open_counts.keys())
    open_only = set(open_counts.keys()) - set(closed_counts.keys())
    both = set(closed_counts.keys()) & set(open_counts.keys())

    # Step 2: Resolve conflicts using RMS amplitude
    details = {}
    resolved_closed = set(closed_only)
    resolved_open = set(open_only)

    if both and recordings:
        # Compute per-channel mean RMS during GT=CLOSED vs GT=OPEN periods
        closed_rms_sums = {}
        open_rms_sums = {}
        closed_sample_counts = {}
        open_sample_counts = {}

        for rec in recordings:
            emg = rec.get("emg", rec.get("biosignal"))
            gt = rec.get("gt", rec.get("ground_truth"))
            if emg is None or gt is None:
                continue
            if hasattr(gt, 'flatten'):
                gt = gt.flatten()
            gt = np.array(gt)

            n_ch = emg.shape[0]
            # GT: 1=CLOSED, 0=OPEN
            closed_mask = gt == 1
            open_mask = gt == 0

            for ch in both:
                if ch >= n_ch:
                    continue
                # RMS during CLOSED periods
                if np.any(closed_mask):
                    closed_seg = emg[ch, closed_mask]
                    rms_c = np.sqrt(np.mean(closed_seg ** 2))
                    closed_rms_sums[ch] = closed_rms_sums.get(ch, 0.0) + rms_c
                    closed_sample_counts[ch] = closed_sample_counts.get(ch, 0) + 1
                # RMS during OPEN periods
                if np.any(open_mask):
                    open_seg = emg[ch, open_mask]
                    rms_o = np.sqrt(np.mean(open_seg ** 2))
                    open_rms_sums[ch] = open_rms_sums.get(ch, 0.0) + rms_o
                    open_sample_counts[ch] = open_sample_counts.get(ch, 0) + 1

        for ch in both:
            mean_rms_closed = (closed_rms_sums.get(ch, 0.0) /
                               closed_sample_counts.get(ch, 1))
            mean_rms_open = (open_rms_sums.get(ch, 0.0) /
                             open_sample_counts.get(ch, 1))
            ratio = mean_rms_closed / mean_rms_open if mean_rms_open > 0 else float('inf')

            details[ch] = {
                "closed_fire_count": closed_counts.get(ch, 0),
                "open_fire_count": open_counts.get(ch, 0),
                "mean_rms_closed": mean_rms_closed,
                "mean_rms_open": mean_rms_open,
                "ratio_closed_over_open": ratio,
            }

            if mean_rms_closed >= mean_rms_open:
                resolved_closed.add(ch)
            else:
                resolved_open.add(ch)
    elif both:
        # No recordings available — assign by higher firing count
        for ch in both:
            if closed_counts.get(ch, 0) >= open_counts.get(ch, 0):
                resolved_closed.add(ch)
            else:
                resolved_open.add(ch)

    # Add details for non-conflict channels too
    for ch in closed_only:
        details[ch] = {
            "closed_fire_count": closed_counts.get(ch, 0),
            "open_fire_count": 0,
            "assignment": "closed_only",
        }
    for ch in open_only:
        details[ch] = {
            "open_fire_count": open_counts.get(ch, 0),
            "closed_fire_count": 0,
            "assignment": "open_only",
        }

    # Step 3: Exclude dead channels
    resolved_closed -= set(dead_channels)
    resolved_open -= set(dead_channels)

    # Determine unassigned channels (total channels from first recording)
    n_ch_total = 0
    if recordings:
        emg = recordings[0].get("emg", recordings[0].get("biosignal"))
        if emg is not None:
            n_ch_total = emg.shape[0]
    unassigned = [ch for ch in range(n_ch_total)
                  if ch not in resolved_closed and ch not in resolved_open
                  and ch not in dead_channels]

    return {
        "closed": sorted(resolved_closed),
        "open": sorted(resolved_open),
        "unassigned": unassigned,
        "details": details,
    }


def analyze_template_quality(
    metrics: dict,
    n_closed: int,
    n_open: int,
) -> dict:
    """
    Comprehensive template quality analysis with outlier detection and grading.

    Takes the output of compute_template_metrics_with_aggregation and produces:
    - Per-template outlier flags from multiple methods
    - Consensus outlier list (flagged by >= 2 methods)
    - Quality grade out of 30 (separation + consistency + robustness, 10 each)

    Parameters
    ----------
    metrics : dict
        Output of compute_template_metrics_with_aggregation.
    n_closed : int
        Number of CLOSED templates.
    n_open : int
        Number of OPEN templates.

    Returns
    -------
    dict with:
        "closed_analysis": list of per-template dicts
        "open_analysis": list of per-template dicts
        "grade": dict with total (0-30), separation (0-10), consistency (0-10), robustness (0-10)
        "summary": str human-readable summary
    """
    ic = metrics["intra_closed"]
    io = metrics["intra_open"]
    ec = metrics["inter_closed_to_open"]
    eo = metrics["inter_open_to_closed"]
    dist_matrix = metrics["distance_matrix"]

    def _analyze_class(intra, inter, n_templates, class_name, class_offset):
        """Analyze one class. Returns list of per-template analysis dicts."""
        if n_templates == 0:
            return []

        intra_vals = intra["per_template"]
        inter_vals = inter["per_template"]

        # --- Method 1: Z-score (intra > mean + 2*std) ---
        z_threshold = intra["mean"] + 2 * intra["std"] if intra["std"] > 0 else float('inf')

        # --- Method 2: IQR ---
        if len(intra_vals) >= 4:
            q1 = np.percentile(intra_vals, 25)
            q3 = np.percentile(intra_vals, 75)
            iqr = q3 - q1
            iqr_threshold = q3 + 1.5 * iqr
        else:
            iqr_threshold = float('inf')

        # --- Method 3: Silhouette per template ---
        # s(i) = (b(i) - a(i)) / max(a(i), b(i))
        # a(i) = mean distance to same class, b(i) = mean distance to other class
        # Using the full distance matrix for mean (not aggregated)
        if class_name == "CLOSED":
            same_indices = list(range(n_closed))
            other_indices = list(range(n_closed, n_closed + n_open))
        else:
            same_indices = list(range(n_closed, n_closed + n_open))
            other_indices = list(range(n_closed))

        silhouettes = []
        for local_idx in range(n_templates):
            global_idx = class_offset + local_idx
            same_others = [j for j in same_indices if j != global_idx]
            a_i = np.mean([dist_matrix[global_idx, j] for j in same_others]) if same_others else 0
            b_i = np.mean([dist_matrix[global_idx, j] for j in other_indices]) if other_indices else 0
            denom = max(a_i, b_i)
            sil = (b_i - a_i) / denom if denom > 0 else 0
            silhouettes.append(sil)

        # --- Method 4: Gap crosser (intra >= inter) ---
        # --- Method 5: Relative margin ---
        # margin = (inter - intra) / inter → negative = bad

        results = []
        for idx in range(n_templates):
            intra_val = float(intra_vals[idx])
            inter_val = float(inter_vals[idx])
            margin = inter_val - intra_val
            rel_margin = margin / inter_val if inter_val > 0 else 0

            flags = []
            if intra_val > z_threshold:
                flags.append("Z-SCORE")
            if intra_val > iqr_threshold:
                flags.append("IQR")
            if silhouettes[idx] < 0:
                flags.append("SILHOUETTE<0")
            if intra_val >= inter_val:
                flags.append("GAP-CROSSER")
            # Also flag if intra exceeds the mid-gap threshold
            midgap = (intra["mean"] + intra["std"] + inter["mean"] - inter["std"]) / 2
            if intra_val > midgap:
                flags.append("ABOVE-MIDGAP")

            results.append({
                "index": idx,
                "label": f"{'C' if class_name == 'CLOSED' else 'O'}{idx+1}",
                "intra": intra_val,
                "inter": inter_val,
                "margin": margin,
                "rel_margin": rel_margin,
                "silhouette": silhouettes[idx],
                "flags": flags,
                "n_flags": len(flags),
                "is_outlier": len(flags) >= 2,  # consensus: 2+ methods agree
            })

        return results

    closed_analysis = _analyze_class(ic, ec, n_closed, "CLOSED", 0)
    open_analysis = _analyze_class(io, eo, n_open, "OPEN", n_closed)

    # ── Quality Grade (0-30) ──

    # --- Separation score (0-10) ---
    # Based on normalized gap: gap / mean_inter
    # gap = (inter_mean - inter_std) - (intra_mean + intra_std)
    ic_upper = ic["mean"] + ic["std"]
    io_upper = io["mean"] + io["std"]
    ec_lower = ec["mean"] - ec["std"]
    eo_lower = eo["mean"] - eo["std"]

    gap_closed = ec_lower - ic_upper
    gap_open = eo_lower - io_upper

    # Normalized gap: fraction of inter-class distance that is "gap"
    norm_gap_c = gap_closed / ec["mean"] if ec["mean"] > 0 else 0
    norm_gap_o = gap_open / eo["mean"] if eo["mean"] > 0 else 0
    avg_norm_gap = (norm_gap_c + norm_gap_o) / 2

    # Map to 0-10: negative gap → 0, gap = 50% of inter → 10
    separation_score = max(0, min(10, avg_norm_gap * 20))

    # --- Consistency score (0-10) ---
    # Based on coefficient of variation (std/mean) of intra-class distances
    # Lower CV = more consistent. Also penalize for outliers.
    cv_c = ic["std"] / ic["mean"] if ic["mean"] > 0 else 1.0
    cv_o = io["std"] / io["mean"] if io["mean"] > 0 else 1.0
    avg_cv = (cv_c + cv_o) / 2

    # Map to 0-10: CV=0 → 10 (perfect), CV >= 1 → 0
    consistency_base = max(0, min(10, 10 * (1 - avg_cv)))

    # Penalty for outlier templates (consensus outliers)
    n_outliers_c = sum(1 for t in closed_analysis if t["is_outlier"])
    n_outliers_o = sum(1 for t in open_analysis if t["is_outlier"])
    n_total = n_closed + n_open
    outlier_fraction = (n_outliers_c + n_outliers_o) / n_total if n_total > 0 else 0
    consistency_score = max(0, consistency_base * (1 - outlier_fraction))

    # --- Robustness score (0-10) ---
    # Based on worst-case template: minimum margin across all templates
    all_margins = ([t["rel_margin"] for t in closed_analysis] +
                   [t["rel_margin"] for t in open_analysis])
    if all_margins:
        worst_margin = min(all_margins)
        median_margin = float(np.median(all_margins))
    else:
        worst_margin = 0
        median_margin = 0

    # Penalty for gap crossers
    n_crossers = sum(1 for t in closed_analysis + open_analysis if "GAP-CROSSER" in t["flags"])

    # Map: worst_margin < 0 → heavy penalty, median_margin > 0.5 → good
    robustness_base = max(0, min(10, median_margin * 15))
    # Each crosser costs 2 points
    robustness_score = max(0, robustness_base - n_crossers * 2)

    total_grade = separation_score + consistency_score + robustness_score

    grade = {
        "total": round(total_grade, 1),
        "separation": round(separation_score, 1),
        "consistency": round(consistency_score, 1),
        "robustness": round(robustness_score, 1),
        "details": {
            "gap_closed": gap_closed,
            "gap_open": gap_open,
            "norm_gap_avg": avg_norm_gap,
            "cv_closed": cv_c,
            "cv_open": cv_o,
            "n_outliers": n_outliers_c + n_outliers_o,
            "n_crossers": n_crossers,
            "worst_margin": worst_margin,
            "median_margin": median_margin,
        }
    }

    # ── Build summary ──
    def _template_report(analysis, class_name):
        lines = []
        # Sort by n_flags descending, then by margin ascending
        sorted_a = sorted(analysis, key=lambda t: (-t["n_flags"], t["margin"]))
        for t in sorted_a:
            if t["n_flags"] == 0:
                continue
            flag_str = ", ".join(t["flags"])
            lines.append(
                f"  {t['label']}  intra={t['intra']:.4f}  inter={t['inter']:.4f}  "
                f"margin={t['margin']:.4f}  sil={t['silhouette']:.3f}  "
                f"[{flag_str}]"
            )
        if not lines:
            lines.append(f"  No flagged templates in {class_name}.")
        return "\n".join(lines)

    summary = (
        f"GRADE: {total_grade:.1f}/30  "
        f"(Separation: {separation_score:.1f}  "
        f"Consistency: {consistency_score:.1f}  "
        f"Robustness: {robustness_score:.1f})\n"
        f"\n"
        f"Flagged CLOSED templates:\n"
        f"{_template_report(closed_analysis, 'CLOSED')}\n"
        f"\n"
        f"Flagged OPEN templates:\n"
        f"{_template_report(open_analysis, 'OPEN')}\n"
        f"\n"
        f"Outlier methods: Z-SCORE (>mean+2std), IQR (>Q3+1.5*IQR), "
        f"SILHOUETTE<0, GAP-CROSSER (intra>=inter), ABOVE-MIDGAP\n"
        f"Consensus outlier = flagged by 2+ methods  |  "
        f"Outliers: {n_outliers_c + n_outliers_o}/{n_total}  "
        f"Crossers: {n_crossers}/{n_total}"
    )

    return {
        "closed_analysis": closed_analysis,
        "open_analysis": open_analysis,
        "grade": grade,
        "summary": summary,
    }


def compute_template_metrics_with_aggregation(
    templates_closed: list,
    templates_open: list,
    feature_name: str = "rms",
    window_length: int = 192,
    window_increment: int = 64,
    distance_aggregation: str = "avg_3_smallest",
    active_channels: list = None,
) -> dict:
    """Compute pairwise DTW distances with feature extraction and aggregation.

    Takes raw EMG templates (n_ch, n_samples), extracts features internally,
    computes full distance matrix for heatmap, and computes intra/inter-class
    distances using the specified aggregation method (matching online prediction).

    Args:
        templates_closed: list of raw EMG arrays (n_ch, n_samples) for CLOSED
        templates_open: list of raw EMG arrays (n_ch, n_samples) for OPEN
        feature_name: feature to extract (key from FEATURES registry)
        window_length: feature window size in samples
        window_increment: feature window step in samples
        distance_aggregation: "avg_3_smallest", "minimum", or "average"
        active_channels: list of 0-indexed active channel indices (None = all)

    Returns:
        dict with: distance_matrix, labels, intra_closed, intra_open,
        inter_closed_to_open, inter_open_to_closed (each with mean/std/per_template)
    """
    feature_fn = FEATURES[feature_name]["function"]

    def _extract(raw_template):
        windowed = sliding_window(raw_template, window_length, window_increment)
        return feature_fn(windowed)  # (n_windows, n_ch)

    # Extract features from all templates
    closed_features = [_extract(t) for t in templates_closed]
    open_features = [_extract(t) for t in templates_open]
    all_features = closed_features + open_features

    n_closed = len(closed_features)
    n_open = len(open_features)
    n_total = n_closed + n_open
    labels = ["CLOSED"] * n_closed + ["OPEN"] * n_open

    # Full pairwise distance matrix (for heatmap)
    dist_matrix = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(i + 1, n_total):
            d = compute_dtw(all_features[i], all_features[j], active_channels=active_channels)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    def _aggregate(distances):
        """Aggregate distances using the configured method."""
        if len(distances) == 0:
            return 0.0
        if distance_aggregation == "minimum":
            return np.min(distances)
        elif distance_aggregation == "avg_3_smallest":
            n_smallest = min(3, len(distances))
            return np.mean(np.sort(distances)[:n_smallest])
        else:  # "average"
            return np.mean(distances)

    # Intra-class with aggregation: for each template, aggregate distances to
    # all OTHER templates in same class → one value per template
    def _intra_class(indices):
        per_template = []
        for i in indices:
            others = [j for j in indices if j != i]
            if not others:
                per_template.append(0.0)
                continue
            dists = np.array([dist_matrix[i, j] for j in others])
            per_template.append(_aggregate(dists))
        per_template = np.array(per_template)
        return {
            "mean": float(np.mean(per_template)) if len(per_template) else 0.0,
            "std": float(np.std(per_template)) if len(per_template) else 0.0,
            "per_template": per_template,
        }

    # Inter-class with aggregation: for each template in class A, aggregate
    # distances to all templates in class B → one value per template
    def _inter_class(source_indices, target_indices):
        per_template = []
        for i in source_indices:
            dists = np.array([dist_matrix[i, j] for j in target_indices])
            per_template.append(_aggregate(dists))
        per_template = np.array(per_template)
        return {
            "mean": float(np.mean(per_template)) if len(per_template) else 0.0,
            "std": float(np.std(per_template)) if len(per_template) else 0.0,
            "per_template": per_template,
        }

    closed_indices = list(range(n_closed))
    open_indices = list(range(n_closed, n_total))

    return {
        "distance_matrix": dist_matrix,
        "labels": labels,
        "intra_closed": _intra_class(closed_indices),
        "intra_open": _intra_class(open_indices),
        "inter_closed_to_open": _inter_class(closed_indices, open_indices),
        "inter_open_to_closed": _inter_class(open_indices, closed_indices),
    }


# ── Visualization ────────────────────────────────────────────────────────────

def _compute_feature_signal(emg: np.ndarray, feature_name: str = FEATURE) -> tuple:
    """Compute feature-extracted signal for each channel via sliding window.

    Args:
        emg: (n_ch, n_samples)
        feature_name: feature to extract (default: RMS)

    Returns:
        features: (n_ch, n_windows) feature value per channel per window
        time_centers: (n_windows,) time in seconds of each window center
    """
    windowed = sliding_window(emg, WINDOW_LENGTH, WINDOW_INCREMENT)
    # windowed: (n_windows, n_ch, window_length)
    feature_fn = FEATURES[feature_name]["function"]
    features = feature_fn(windowed)
    # features: (n_windows, n_ch) → transpose to (n_ch, n_windows)
    features = features.T

    n_windows = features.shape[1]
    # Window centers in seconds
    time_centers = (np.arange(n_windows) * WINDOW_INCREMENT + WINDOW_LENGTH / 2) / config.FSAMP

    return features, time_centers


def _plot_stacked_features(ax, emg: np.ndarray, gt=None, cycle=None, colors=None):
    """Plot per-channel feature-extracted signal stacked vertically.

    Args:
        ax: matplotlib axis
        emg: (n_ch, n_samples)
        gt: optional GT signal (n_samples,)
        cycle: optional cycle dict for cue markers
        colors: optional list of colors per channel

    Returns:
        n_ch: number of channels (for y-axis sizing)
    """
    features, time_centers = _compute_feature_signal(emg)
    n_ch = features.shape[0]

    # Normalize each channel to [0, 1] for stacking
    ch_maxes = features.max(axis=1, keepdims=True)
    ch_maxes[ch_maxes == 0] = 1
    features_norm = features / ch_maxes

    # Stack: channel 0 at top
    spacing = 1.0
    if colors is None:
        cmap = plt.cm.tab20
        colors = [cmap(i % 20) for i in range(n_ch)]

    for ch in range(n_ch):
        offset = (n_ch - 1 - ch) * spacing
        ax.plot(time_centers, features_norm[ch] * 0.8 + offset, color=colors[ch],
                linewidth=0.7, alpha=0.85)

    # GT shading across full height
    if gt is not None:
        n_samples = len(gt)
        gt_time = np.arange(n_samples) / config.FSAMP
        y_max = n_ch * spacing
        gt_scaled = gt * y_max
        ax.fill_between(gt_time, 0, gt_scaled, alpha=0.08, color="red")

    # Cue markers
    if cycle is not None:
        if cycle.get("close_cue_idx") is not None:
            ax.axvline(cycle["close_cue_idx"] / config.FSAMP, color="red",
                        linestyle="--", linewidth=0.8, alpha=0.7)
        if cycle.get("open_cue_idx") is not None:
            ax.axvline(cycle["open_cue_idx"] / config.FSAMP, color="blue",
                        linestyle="--", linewidth=0.8, alpha=0.7)

    # Channel labels
    ax.set_ylim(-0.1, n_ch * spacing)
    ax.set_yticks([(n_ch - 1 - ch) * spacing + 0.4 for ch in range(n_ch)])
    ax.set_yticklabels([str(ch + 1) for ch in range(n_ch)], fontsize=6)
    ax.set_xlim(time_centers[0], time_centers[-1])

    return n_ch


def _plot_stacked_raw(ax, emg: np.ndarray, gt=None, cycle=None, colors=None):
    """Plot raw EMG channels stacked vertically.

    Args:
        ax: matplotlib axis
        emg: (n_ch, n_samples)
        gt: optional GT signal (n_samples,)
        cycle: optional cycle dict for cue markers
        colors: optional list of colors per channel

    Returns:
        n_ch: number of channels
    """
    n_ch, n_samples = emg.shape
    time_s = np.arange(n_samples) / config.FSAMP

    # Normalize each channel to [-0.5, 0.5] for stacking
    ch_maxes = np.max(np.abs(emg), axis=1, keepdims=True)
    ch_maxes[ch_maxes == 0] = 1
    emg_norm = emg / ch_maxes

    spacing = 1.0
    if colors is None:
        cmap = plt.cm.tab20
        colors = [cmap(i % 20) for i in range(n_ch)]

    for ch in range(n_ch):
        offset = (n_ch - 1 - ch) * spacing + 0.5
        ax.plot(time_s, emg_norm[ch] * 0.4 + offset, color=colors[ch],
                linewidth=0.3, alpha=0.8)

    # GT shading
    if gt is not None:
        y_max = n_ch * spacing
        gt_scaled = gt * y_max
        ax.fill_between(time_s, 0, gt_scaled, alpha=0.08, color="red")

    # Cue markers
    if cycle is not None:
        if cycle.get("close_cue_idx") is not None:
            ax.axvline(cycle["close_cue_idx"] / config.FSAMP, color="red",
                        linestyle="--", linewidth=0.8, alpha=0.7)
        if cycle.get("open_cue_idx") is not None:
            ax.axvline(cycle["open_cue_idx"] / config.FSAMP, color="blue",
                        linestyle="--", linewidth=0.8, alpha=0.7)

    # Channel labels
    ax.set_ylim(-0.1, n_ch * spacing)
    ax.set_yticks([(n_ch - 1 - ch) * spacing + 0.5 for ch in range(n_ch)])
    ax.set_yticklabels([str(ch + 1) for ch in range(n_ch)], fontsize=6)
    ax.set_xlim(time_s[0], time_s[-1])

    return n_ch


def _plot_stacked(ax, emg, **kwargs):
    """Dispatch to raw or feature-extracted plot based on PLOT_RAW_EMG config."""
    if PLOT_RAW_EMG:
        return _plot_stacked_raw(ax, emg, **kwargs)
    else:
        return _plot_stacked_features(ax, emg, **kwargs)


def plot_cycle_overview(cycles: list, title: str = "Cycle Overview"):
    """
    Plot all cycles with per-channel feature signal stacked, GT overlay, cue markers.
    Returns the figure for interactive use.
    """
    n_cycles = len(cycles)
    n_ch = cycles[0]["emg"].shape[0] if cycles else 16
    per_cycle_height = max(4, n_ch * 0.35)
    fig, axes = plt.subplots(n_cycles, 1, figsize=(16, per_cycle_height * n_cycles), sharex=False)
    if n_cycles == 1:
        axes = [axes]

    for idx, cycle in enumerate(cycles):
        ax = axes[idx]
        _plot_stacked(ax, cycle["emg"], gt=cycle["gt"], cycle=cycle)

        ax.set_ylabel(f"Cycle {cycle['cycle_number']}", fontsize=9)
        if idx == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color="red", linestyle="--", label="Close cue"),
                Line2D([0], [0], color="blue", linestyle="--", label="Open cue"),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_cycle_overview_with_windows(
    cycles: list,
    closed_positions: dict,
    open_positions: dict,
    title: str = "Cycle Overview — Selected Windows",
):
    """Plot all cycles with selected template windows highlighted."""
    n_cycles = len(cycles)
    n_ch = cycles[0]["emg"].shape[0] if cycles else 16
    per_cycle_height = max(4, n_ch * 0.35)
    fig, axes = plt.subplots(n_cycles, 1, figsize=(16, per_cycle_height * n_cycles), sharex=False)
    if n_cycles == 1:
        axes = [axes]

    template_duration_s = TEMPLATE_DURATION_S

    for idx, cycle in enumerate(cycles):
        ax = axes[idx]
        cn = cycle["cycle_number"]

        n_ch = _plot_stacked(ax, cycle["emg"], gt=cycle["gt"], cycle=cycle)
        y_max = n_ch

        # CLOSED window highlight (orange)
        if cn in closed_positions:
            pos_s = closed_positions[cn] / config.FSAMP
            rect = Rectangle((pos_s, 0), template_duration_s, y_max,
                              facecolor="orange", alpha=0.25, edgecolor="orange", linewidth=1.5)
            ax.add_patch(rect)

        # OPEN window highlight (blue)
        if cn in open_positions:
            pos_s = open_positions[cn] / config.FSAMP
            rect = Rectangle((pos_s, 0), template_duration_s, y_max,
                              facecolor="dodgerblue", alpha=0.25, edgecolor="dodgerblue", linewidth=1.5)
            ax.add_patch(rect)

        ax.set_ylabel(f"C{cn}", fontsize=9)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_search_curves(
    cycles: list,
    closed_results: dict,
    open_results: dict,
):
    """DTW distance vs window position for each cycle's search."""
    cycle_nums = sorted(set(closed_results.keys()) | set(open_results.keys()))
    n = len(cycle_nums)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.2 * n), sharex=False)
    if n == 1:
        axes = axes.reshape(1, 2)

    # Build cycle lookup by number
    cycle_by_num = {c["cycle_number"]: c for c in cycles}

    for row, cn in enumerate(cycle_nums):
        cycle = cycle_by_num.get(cn, {})

        # CLOSED search curve
        ax_c = axes[row, 0]
        if cn in closed_results:
            best_pos, best_dist, positions, dists = closed_results[cn]
            if len(positions) > 0:
                pos_s = np.array(positions) / config.FSAMP
                ax_c.plot(pos_s, dists, color="orange", linewidth=1)
                best_s = best_pos / config.FSAMP
                ax_c.scatter([best_s], [best_dist], color="red", zorder=5, s=30)
                ax_c.set_title(f"Cycle {cn} — CLOSED (d={best_dist:.2f})", fontsize=9)
            else:
                ax_c.set_title(f"Cycle {cn} — CLOSED (reference)", fontsize=9)
        # Audio cue marker
        if cycle.get("close_cue_idx") is not None:
            ax_c.axvline(cycle["close_cue_idx"] / config.FSAMP, color="red",
                         linestyle="--", linewidth=1, alpha=0.6, label="Close cue")
        ax_c.set_ylabel("DTW dist", fontsize=8)

        # OPEN search curve
        ax_o = axes[row, 1]
        if cn in open_results:
            best_pos, best_dist, positions, dists = open_results[cn]
            if len(positions) > 0:
                pos_s = np.array(positions) / config.FSAMP
                ax_o.plot(pos_s, dists, color="dodgerblue", linewidth=1)
                best_s = best_pos / config.FSAMP
                ax_o.scatter([best_s], [best_dist], color="blue", zorder=5, s=30)
                ax_o.set_title(f"Cycle {cn} — OPEN (d={best_dist:.2f})", fontsize=9)
            else:
                ax_o.set_title(f"Cycle {cn} — OPEN (reference)", fontsize=9)
        # Audio cue marker
        if cycle.get("open_cue_idx") is not None:
            ax_o.axvline(cycle["open_cue_idx"] / config.FSAMP, color="blue",
                         linestyle="--", linewidth=1, alpha=0.6, label="Open cue")
        ax_o.set_ylabel("DTW dist", fontsize=8)

    axes[-1, 0].set_xlabel("Window start (s)")
    axes[-1, 1].set_xlabel("Window start (s)")
    fig.suptitle("Search Curves — DTW Distance vs Window Position", fontsize=12)
    fig.tight_layout()
    return fig


def plot_distance_matrix(metrics: dict, vmin: float = None, vmax: float = None, ax=None):
    """Heatmap of all-vs-all DTW distances.

    Args:
        metrics: dict from compute_template_metrics
        vmin, vmax: optional shared color scale limits
        ax: optional matplotlib axis (creates new figure if None)
    """
    dist_matrix = metrics["distance_matrix"]
    labels = metrics["labels"]
    n = len(labels)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure
    im = ax.imshow(dist_matrix, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="DTW Distance")

    # Tick labels
    # Per-class numbering: C1,C2,... O1,O2,...
    class_counters = {}
    tick_labels = []
    for i in range(n):
        cls = labels[i][0]
        class_counters[cls] = class_counters.get(cls, 0) + 1
        tick_labels.append(f"{cls}{class_counters[cls]}")
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=7)

    # Draw class boundary
    n_closed = labels.count("CLOSED")
    ax.axhline(n_closed - 0.5, color="white", linewidth=2)
    ax.axvline(n_closed - 0.5, color="white", linewidth=2)

    ax.set_title("Template Distance Matrix (CLOSED | OPEN)")
    fig.tight_layout()
    return fig


def print_metrics(metrics: dict):
    """Print metrics summary table."""
    print("\n" + "=" * 60)
    print("TEMPLATE STUDY — METRICS SUMMARY")
    print("=" * 60)

    ic = metrics["intra_closed"]
    io = metrics["intra_open"]
    inter = metrics["inter"]

    print(f"\n  Intra-class CLOSED:  mean = {ic['mean']:.3f}  std = {ic['std']:.3f}")
    print(f"  Intra-class OPEN:    mean = {io['mean']:.3f}  std = {io['std']:.3f}")
    print(f"  Inter-class:         mean = {inter['mean']:.3f}  std = {inter['std']:.3f}")
    print(f"\n  Separability ratio (inter/intra):  {metrics['separability_ratio']:.3f}")
    print(f"  Mean silhouette score:             {metrics['silhouette_score']:.3f}")

    print("\n  Per-template silhouette scores:")
    labels = metrics["labels"]
    sils = metrics["per_template_silhouette"]
    for i, (label, s) in enumerate(zip(labels, sils)):
        marker = " ***" if s < 0 else ""
        print(f"    {label:7s} #{i+1:2d}:  {s:+.3f}{marker}")

    print("=" * 60)


# ── Training-style cycle plot (matches CycleReviewWidget) ───────────────────

def _plot_cycle_training_style(ax, emg: np.ndarray, gt: np.ndarray, cycle: dict = None):
    """
    Plot a single cycle matching the training protocol's CycleReviewWidget style.
    Raw EMG as black stacked traces, GT as red bottom trace, state labels, cue markers.

    Returns:
        (y_min, y_max, offset_step): axis limits and offset for window placement
    """
    n_channels, n_samples = emg.shape
    time_axis = np.arange(n_samples) / config.FSAMP

    # Vertical offset from median peak-to-peak (same as CycleReviewWidget)
    all_ranges = [np.ptp(emg[ch, :]) for ch in range(n_channels)]
    offset_step = np.median(all_ranges) * 1.5 if np.median(all_ranges) > 0 else 1.0

    # Stacked EMG channels (CH1 at top)
    yticks = []
    ytick_labels = []
    for ch in range(n_channels):
        offset = (n_channels - ch) * offset_step
        ax.plot(time_axis, emg[ch, :] + offset, 'k-', linewidth=0.5, alpha=0.9)
        yticks.append(offset)
        ytick_labels.append(f"CH{ch + 1}")

    # GT as bottom red trace
    gt_scaled = gt * offset_step * 0.8
    ax.plot(time_axis, gt_scaled, 'r-', linewidth=1.5, alpha=0.7, label='GT')
    yticks.append(0)
    ytick_labels.append("GT")

    # State labels on GT regions
    gt_binary = (gt > 0.5).astype(int)
    gt_diff = np.diff(gt_binary, prepend=gt_binary[0])
    transitions = np.where(gt_diff != 0)[0]
    region_starts = np.concatenate([[0], transitions])
    region_ends = np.concatenate([transitions, [n_samples]])
    gt_label_y = offset_step * 0.9
    for rs, re in zip(region_starts, region_ends):
        if re - rs < config.FSAMP * 0.3:
            continue
        mid_time = (rs + re) / 2 / config.FSAMP
        region_val = gt_binary[min(rs + (re - rs) // 2, n_samples - 1)]
        state_label = "CLOSED" if region_val == 1 else "OPEN"
        state_color = '#FF5722' if region_val == 1 else '#2196F3'
        ax.text(mid_time, gt_label_y, state_label,
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color=state_color, alpha=0.8)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=7)

    # Audio cue markers
    if cycle is not None:
        if cycle.get("close_cue_idx") is not None:
            t = cycle["close_cue_idx"] / config.FSAMP
            ax.axvline(t, color='#FF5722', linestyle='--', linewidth=1.5, alpha=0.7, label='Close cue')
        if cycle.get("open_cue_idx") is not None:
            t = cycle["open_cue_idx"] / config.FSAMP
            ax.axvline(t, color='#2196F3', linestyle='--', linewidth=1.5, alpha=0.7, label='Open cue')

    y_min = -offset_step * 0.5
    y_max = (n_channels + 1) * offset_step
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, n_samples / config.FSAMP)
    ax.legend(loc='upper left', fontsize=7, framealpha=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    return y_min, y_max, offset_step


# ── Interactive reference selection ──────────────────────────────────────────

class DraggableWindow:
    """A draggable rectangle for selecting a template window on a matplotlib axis."""

    def __init__(self, ax, start_s: float, duration_s: float, min_s: float, max_s: float,
                 color: str, label: str):
        self.ax = ax
        self.start_s = start_s
        self.duration_s = duration_s
        self.min_s = min_s
        self.max_s = max_s
        self.color = color
        self.label = label

        y_lo, y_hi = ax.get_ylim()
        self.rect = Rectangle(
            (start_s, y_lo), duration_s, y_hi - y_lo,
            alpha=0.3, facecolor=color, edgecolor=color, linewidth=2,
        )
        ax.add_patch(self.rect)
        self.start_line = ax.axvline(start_s, color=color, linewidth=2)
        self.end_line = ax.axvline(start_s + duration_s, color=color, linewidth=2)

        self.dragging = False
        self._drag_offset = 0.0

    def set_position(self, start_s: float):
        start_s = max(self.min_s, min(start_s, self.max_s))
        self.start_s = start_s
        self.rect.set_x(start_s)
        self.start_line.set_xdata([start_s, start_s])
        self.end_line.set_xdata([start_s + self.duration_s, start_s + self.duration_s])

    def contains(self, x: float) -> bool:
        return self.start_s <= x <= self.start_s + self.duration_s


def select_reference_window(cycles: list, class_name: str) -> tuple:
    """
    Interactive: user picks a cycle, then drags a window to set position.
    Close the figure window (or press Enter in terminal) to confirm.

    Args:
        cycles: list of cycle dicts
        class_name: "CLOSED" or "OPEN"

    Returns:
        (reference_cycle_number, window_start_sample)
    """
    print(f"\n--- Select REFERENCE window for {class_name} class ---")

    # Show cycle overview
    fig = plot_cycle_overview(cycles, title=f"Select reference for {class_name}")

    # Ask user for cycle number
    cycle_numbers = [c["cycle_number"] for c in cycles]
    print(f"Available cycles: {cycle_numbers}")

    while True:
        try:
            ref_num = int(input(f"Enter reference cycle number for {class_name}: "))
            if ref_num in cycle_numbers:
                break
            print(f"Invalid. Choose from {cycle_numbers}")
        except ValueError:
            print("Enter a number.")

    plt.close(fig)

    ref_cycle = next(c for c in cycles if c["cycle_number"] == ref_num)

    # Determine valid region
    template_samples = int(TEMPLATE_DURATION_S * config.FSAMP)
    if class_name == "CLOSED":
        region_start = ref_cycle.get("close_cue_idx") or ref_cycle["close_start_idx"]
        region_end = (ref_cycle.get("open_cue_idx") or ref_cycle["open_start_idx"]) - template_samples
    else:
        region_start = ref_cycle.get("open_cue_idx") or ref_cycle["open_start_idx"]
        region_end = ref_cycle["emg"].shape[1] - template_samples

    region_start_s = region_start / config.FSAMP
    region_end_s = region_end / config.FSAMP

    # Plot the selected cycle (training-style)
    emg = ref_cycle["emg"]
    n_ch = emg.shape[0]
    fig_height = max(6, n_ch * 0.5)
    fig2, ax = plt.subplots(figsize=(16, fig_height))

    _plot_cycle_training_style(ax, emg, ref_cycle["gt"], ref_cycle)

    # Initial window at region start
    color = "#FF5722" if class_name == "CLOSED" else "#2196F3"
    window = DraggableWindow(ax, region_start_s, TEMPLATE_DURATION_S, region_start_s, region_end_s,
                              color, class_name)

    ax.set_title(f"Cycle {ref_num} — Drag the {class_name} window, then close figure to confirm",
                 fontsize=11)
    ax.set_xlabel("Time (s)")

    # Wire up drag events
    dragging_state = {"window": None, "offset": 0.0}

    def on_press(event):
        if event.inaxes != ax or event.xdata is None:
            return
        if window.contains(event.xdata):
            dragging_state["window"] = window
            dragging_state["offset"] = event.xdata - window.start_s

    def on_move(event):
        if dragging_state["window"] is None or event.xdata is None:
            return
        new_start = event.xdata - dragging_state["offset"]
        dragging_state["window"].set_position(new_start)
        fig2.canvas.draw_idle()

    def on_release(event):
        dragging_state["window"] = None

    fig2.canvas.mpl_connect("button_press_event", on_press)
    fig2.canvas.mpl_connect("motion_notify_event", on_move)
    fig2.canvas.mpl_connect("button_release_event", on_release)

    print(f"Drag the window within [{region_start_s:.2f}s, {region_end_s:.2f}s]. Close the figure to confirm.")

    plt.show()  # blocks until user closes the figure

    window_start_sample = int(window.start_s * config.FSAMP)
    print(f"  -> Reference: Cycle {ref_num}, window start = {window.start_s:.3f}s (sample {window_start_sample})")
    return ref_num, window_start_sample


def select_window_from_cycle(cycle: dict, class_name: str, dtw_pos: int = None) -> int:
    """
    Interactive: user drags a window on a specific cycle to select a template.
    Optionally shows the DTW-selected position as a semi-transparent overlay.

    Args:
        cycle: cycle dict with emg, gt, cue indices
        class_name: "CLOSED" or "OPEN"
        dtw_pos: optional DTW-selected position (samples) to show as reference

    Returns:
        window_start_sample
    """
    cn = cycle["cycle_number"]
    template_samples = int(TEMPLATE_DURATION_S * config.FSAMP)

    if class_name == "CLOSED":
        region_start = cycle.get("close_cue_idx") or cycle["close_start_idx"]
        region_end = (cycle.get("open_cue_idx") or cycle["open_start_idx"]) - template_samples
    else:
        region_start = cycle.get("open_cue_idx") or cycle["open_start_idx"]
        region_end = cycle["emg"].shape[1] - template_samples

    region_start_s = region_start / config.FSAMP
    region_end_s = region_end / config.FSAMP

    emg = cycle["emg"]
    n_ch = emg.shape[0]
    fig_height = max(6, n_ch * 0.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    y_min, y_max, offset_step = _plot_cycle_training_style(ax, emg, cycle["gt"], cycle)

    # Show DTW pick as semi-transparent overlay
    if dtw_pos is not None:
        dtw_pos_s = dtw_pos / config.FSAMP
        dtw_rect = Rectangle((dtw_pos_s, y_min), TEMPLATE_DURATION_S, y_max - y_min,
                              facecolor="gray", alpha=0.2, edgecolor="gray",
                              linewidth=1.5, linestyle="--")
        ax.add_patch(dtw_rect)
        ax.text(dtw_pos_s + TEMPLATE_DURATION_S / 2, y_max * 0.95, "DTW",
                ha="center", va="top", fontsize=9, color="gray", fontweight="bold")

    # User's draggable window — start at region midpoint
    initial_s = (region_start_s + region_end_s) / 2
    color = "#FF5722" if class_name == "CLOSED" else "#2196F3"
    window = DraggableWindow(ax, initial_s, TEMPLATE_DURATION_S, region_start_s, region_end_s,
                              color, class_name)

    ax.set_title(f"Cycle {cn} — Drag YOUR {class_name} window, then close to confirm",
                 fontsize=11)
    ax.set_xlabel("Time (s)")

    dragging_state = {"window": None, "offset": 0.0}

    def on_press(event):
        if event.inaxes != ax or event.xdata is None:
            return
        if window.contains(event.xdata):
            dragging_state["window"] = window
            dragging_state["offset"] = event.xdata - window.start_s

    def on_move(event):
        if dragging_state["window"] is None or event.xdata is None:
            return
        new_start = event.xdata - dragging_state["offset"]
        dragging_state["window"].set_position(new_start)
        fig.canvas.draw_idle()

    def on_release(event):
        dragging_state["window"] = None

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)

    print(f"  Cycle {cn} {class_name}: Drag window in [{region_start_s:.2f}s, {region_end_s:.2f}s]. "
          f"Close figure to confirm.")

    plt.show()

    window_start_sample = int(window.start_s * config.FSAMP)
    print(f"    -> Your pick: {window.start_s:.3f}s (sample {window_start_sample})")
    return window_start_sample


# ── Comparison visualization ────────────────────────────────────────────────

def plot_comparison_overview(
    cycles: list,
    dtw_closed_positions: dict,
    dtw_open_positions: dict,
    user_closed_positions: dict,
    user_open_positions: dict,
    title: str = "Comparison: User (solid) vs DTW (dashed)",
):
    """Plot all cycles with both user and DTW selected windows overlaid."""
    n_cycles = len(cycles)
    n_ch = cycles[0]["emg"].shape[0] if cycles else 16
    per_cycle_height = max(4, n_ch * 0.35)
    fig, axes = plt.subplots(n_cycles, 1, figsize=(16, per_cycle_height * n_cycles), sharex=False)
    if n_cycles == 1:
        axes = [axes]

    template_duration_s = TEMPLATE_DURATION_S

    for idx, cycle in enumerate(cycles):
        ax = axes[idx]
        cn = cycle["cycle_number"]
        n_ch = _plot_stacked(ax, cycle["emg"], gt=cycle["gt"], cycle=cycle)
        y_max = n_ch

        # DTW picks (dashed border, no fill)
        if cn in dtw_closed_positions:
            pos_s = dtw_closed_positions[cn] / config.FSAMP
            rect = Rectangle((pos_s, 0), template_duration_s, y_max,
                              facecolor="none", alpha=0.8, edgecolor="orange",
                              linewidth=2, linestyle="--")
            ax.add_patch(rect)
        if cn in dtw_open_positions:
            pos_s = dtw_open_positions[cn] / config.FSAMP
            rect = Rectangle((pos_s, 0), template_duration_s, y_max,
                              facecolor="none", alpha=0.8, edgecolor="dodgerblue",
                              linewidth=2, linestyle="--")
            ax.add_patch(rect)

        # User picks (solid fill)
        if cn in user_closed_positions:
            pos_s = user_closed_positions[cn] / config.FSAMP
            rect = Rectangle((pos_s, 0), template_duration_s, y_max,
                              facecolor="orange", alpha=0.25, edgecolor="orange", linewidth=1.5)
            ax.add_patch(rect)
        if cn in user_open_positions:
            pos_s = user_open_positions[cn] / config.FSAMP
            rect = Rectangle((pos_s, 0), template_duration_s, y_max,
                              facecolor="dodgerblue", alpha=0.25, edgecolor="dodgerblue", linewidth=1.5)
            ax.add_patch(rect)

        ax.set_ylabel(f"C{cn}", fontsize=9)

    axes[-1].set_xlabel("Time (s)")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor="orange", alpha=0.25, edgecolor="orange", label="User CLOSED"),
        Rectangle((0, 0), 1, 1, facecolor="dodgerblue", alpha=0.25, edgecolor="dodgerblue", label="User OPEN"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="orange", linestyle="--", linewidth=2, label="DTW CLOSED"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="dodgerblue", linestyle="--", linewidth=2, label="DTW OPEN"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=7)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def print_comparison_summary(
    cycles: list,
    dtw_closed_positions: dict,
    dtw_open_positions: dict,
    user_closed_positions: dict,
    user_open_positions: dict,
    dtw_metrics: dict,
    user_metrics: dict,
):
    """Print side-by-side comparison of DTW vs user picks."""
    print("\n" + "=" * 70)
    print("COMPARISON: USER PICKS vs DTW PICKS")
    print("=" * 70)

    # Per-cycle position comparison
    print(f"\n  {'Cycle':<8} {'Class':<8} {'User pos (s)':<14} {'DTW pos (s)':<14} {'Diff (ms)':<12}")
    print(f"  {'-'*56}")

    diffs_closed = []
    diffs_open = []

    for cycle in cycles:
        cn = cycle["cycle_number"]
        if cn in user_closed_positions and cn in dtw_closed_positions:
            user_s = user_closed_positions[cn] / config.FSAMP
            dtw_s = dtw_closed_positions[cn] / config.FSAMP
            diff_ms = (user_s - dtw_s) * 1000
            diffs_closed.append(abs(diff_ms))
            print(f"  {cn:<8} {'CLOSED':<8} {user_s:<14.3f} {dtw_s:<14.3f} {diff_ms:<+12.0f}")
        if cn in user_open_positions and cn in dtw_open_positions:
            user_s = user_open_positions[cn] / config.FSAMP
            dtw_s = dtw_open_positions[cn] / config.FSAMP
            diff_ms = (user_s - dtw_s) * 1000
            diffs_open.append(abs(diff_ms))
            print(f"  {cn:<8} {'OPEN':<8} {user_s:<14.3f} {dtw_s:<14.3f} {diff_ms:<+12.0f}")

    print(f"\n  Mean absolute difference:")
    if diffs_closed:
        print(f"    CLOSED: {np.mean(diffs_closed):.0f} ms  (std: {np.std(diffs_closed):.0f} ms)")
    if diffs_open:
        print(f"    OPEN:   {np.mean(diffs_open):.0f} ms  (std: {np.std(diffs_open):.0f} ms)")

    # Metrics comparison
    print(f"\n  {'Metric':<30} {'User':<12} {'DTW':<12}")
    print(f"  {'-'*54}")
    print(f"  {'Separability ratio':<30} {user_metrics['separability_ratio']:<12.3f} {dtw_metrics['separability_ratio']:<12.3f}")
    print(f"  {'Mean silhouette':<30} {user_metrics['silhouette_score']:<12.3f} {dtw_metrics['silhouette_score']:<12.3f}")
    print(f"  {'Intra CLOSED mean':<30} {user_metrics['intra_closed']['mean']:<12.3f} {dtw_metrics['intra_closed']['mean']:<12.3f}")
    print(f"  {'Intra OPEN mean':<30} {user_metrics['intra_open']['mean']:<12.3f} {dtw_metrics['intra_open']['mean']:<12.3f}")
    print(f"  {'Inter-class mean':<30} {user_metrics['inter']['mean']:<12.3f} {dtw_metrics['inter']['mean']:<12.3f}")

    # Verdict
    user_better = user_metrics['separability_ratio'] > dtw_metrics['separability_ratio']
    print(f"\n  Verdict: {'USER' if user_better else 'DTW'} picks have higher separability ratio")
    print("=" * 70)


# ── Main ─────────────────────────────────────────────────────────────────────

def plot_search_curves_multi(
    cycles: list,
    closed_results: dict,
    open_results: dict,
    feature_names: list,
):
    """Search curves with per-feature distance breakdown.

    closed/open_results values:
        (best_pos, best_dist, positions, total_dists, per_feature_dists)
    """
    cycle_nums = sorted(set(closed_results.keys()) | set(open_results.keys()))
    n = len(cycle_nums)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.5 * n), sharex=False)
    if n == 1:
        axes = axes.reshape(1, 2)

    cycle_by_num = {c["cycle_number"]: c for c in cycles}
    feat_colors = plt.cm.Set2(np.linspace(0, 1, len(feature_names)))

    for row, cn in enumerate(cycle_nums):
        cycle = cycle_by_num.get(cn, {})

        # CLOSED
        ax_c = axes[row, 0]
        if cn in closed_results:
            result = closed_results[cn]
            best_pos, best_dist = result[0], result[1]
            positions, total_dists = result[2], result[3]
            per_feat = result[4] if len(result) > 4 else None
            if len(positions) > 0:
                pos_s = np.array(positions) / config.FSAMP
                ax_c.plot(pos_s, total_dists, color="orange", linewidth=1.5, label="total")
                if per_feat is not None and per_feat.shape[0] > 0:
                    for f_idx, fname in enumerate(feature_names):
                        ax_c.plot(pos_s, per_feat[:, f_idx], color=feat_colors[f_idx],
                                  linewidth=0.8, alpha=0.7, linestyle="--", label=fname)
                best_s = best_pos / config.FSAMP
                ax_c.scatter([best_s], [best_dist], color="red", zorder=5, s=30)
                ax_c.set_title(f"Cycle {cn} — CLOSED (d={best_dist:.2f})", fontsize=9)
                if row == 0:
                    ax_c.legend(fontsize=6, loc="upper right")
            else:
                ax_c.set_title(f"Cycle {cn} — CLOSED (reference)", fontsize=9)
        if cycle.get("close_cue_idx") is not None:
            ax_c.axvline(cycle["close_cue_idx"] / config.FSAMP, color="red",
                         linestyle="--", linewidth=1, alpha=0.6)
        ax_c.set_ylabel("DTW dist", fontsize=8)

        # OPEN
        ax_o = axes[row, 1]
        if cn in open_results:
            result = open_results[cn]
            best_pos, best_dist = result[0], result[1]
            positions, total_dists = result[2], result[3]
            per_feat = result[4] if len(result) > 4 else None
            if len(positions) > 0:
                pos_s = np.array(positions) / config.FSAMP
                ax_o.plot(pos_s, total_dists, color="dodgerblue", linewidth=1.5, label="total")
                if per_feat is not None and per_feat.shape[0] > 0:
                    for f_idx, fname in enumerate(feature_names):
                        ax_o.plot(pos_s, per_feat[:, f_idx], color=feat_colors[f_idx],
                                  linewidth=0.8, alpha=0.7, linestyle="--", label=fname)
                best_s = best_pos / config.FSAMP
                ax_o.scatter([best_s], [best_dist], color="blue", zorder=5, s=30)
                ax_o.set_title(f"Cycle {cn} — OPEN (d={best_dist:.2f})", fontsize=9)
                if row == 0:
                    ax_o.legend(fontsize=6, loc="upper right")
            else:
                ax_o.set_title(f"Cycle {cn} — OPEN (reference)", fontsize=9)
        if cycle.get("open_cue_idx") is not None:
            ax_o.axvline(cycle["open_cue_idx"] / config.FSAMP, color="blue",
                         linestyle="--", linewidth=1, alpha=0.6)
        ax_o.set_ylabel("DTW dist", fontsize=8)

    axes[-1, 0].set_xlabel("Window start (s)")
    axes[-1, 1].set_xlabel("Window start (s)")
    fig.suptitle("Search Curves — Multivariate DTW (per-feature breakdown)", fontsize=12)
    fig.tight_layout()
    return fig


def main():
    # 1. Load recording and extract cycles
    recording_path = Path(PATIENT_DIR) / RECORDING_FILE
    print(f"Loading recording: {recording_path}")
    cycles = load_and_extract_cycles(str(recording_path))

    if not cycles:
        print("ERROR: No cycles extracted.")
        return

    multivariate = FEATURES_MULTI is not None and len(FEATURES_MULTI) > 1
    if multivariate:
        print(f"Mode: MULTIVARIATE DTW with features: {FEATURES_MULTI}")
    else:
        print(f"Mode: SINGLE-FEATURE DTW with feature: {FEATURE}")

    print(f"Extracted {len(cycles)} cycles")
    for c in cycles:
        close_cue = f"{c['close_cue_idx']/config.FSAMP:.2f}s" if c.get('close_cue_idx') is not None else "N/A"
        open_cue = f"{c['open_cue_idx']/config.FSAMP:.2f}s" if c.get('open_cue_idx') is not None else "N/A"
        print(f"  Cycle {c['cycle_number']}: {c['duration_s']:.1f}s, "
              f"close@{c['close_start_idx']/config.FSAMP:.2f}s, "
              f"open@{c['open_start_idx']/config.FSAMP:.2f}s, "
              f"close_cue={close_cue}, open_cue={open_cue}")

    template_samples = int(TEMPLATE_DURATION_S * config.FSAMP)
    step_samples = int(SEARCH_STEP_S * config.FSAMP)

    # 2. Select CLOSED reference
    ref_closed_num, ref_closed_pos = select_reference_window(cycles, "CLOSED")
    ref_closed_cycle = next(c for c in cycles if c["cycle_number"] == ref_closed_num)
    ref_closed_segment = ref_closed_cycle["emg"][:, ref_closed_pos:ref_closed_pos + template_samples]

    # 3. Select OPEN reference
    ref_open_num, ref_open_pos = select_reference_window(cycles, "OPEN")
    ref_open_cycle = next(c for c in cycles if c["cycle_number"] == ref_open_num)
    ref_open_segment = ref_open_cycle["emg"][:, ref_open_pos:ref_open_pos + template_samples]

    # Extract reference features
    if multivariate:
        ref_closed_multi = extract_multi_features_from_segment(ref_closed_segment, FEATURES_MULTI)
        ref_open_multi = extract_multi_features_from_segment(ref_open_segment, FEATURES_MULTI)
    # Always extract single-feature for metrics (uses FEATURE)
    ref_closed_features = extract_features_from_segment(ref_closed_segment)
    ref_open_features = extract_features_from_segment(ref_open_segment)

    # 4. Search all cycles for best-matching windows
    print("\n--- Searching for best-matching windows across all cycles ---")

    closed_search_results = {}
    open_search_results = {}
    closed_templates = []
    open_templates = []
    closed_positions = {}
    open_positions = {}

    for cycle in cycles:
        cn = cycle["cycle_number"]
        emg = cycle["emg"]

        closed_search_start = cycle.get("close_cue_idx") or cycle["close_start_idx"]
        closed_search_end = (cycle.get("open_cue_idx") or cycle["open_start_idx"]) - template_samples
        open_search_start = cycle.get("open_cue_idx") or cycle["open_start_idx"]
        open_search_end = emg.shape[1] - template_samples

        # ── CLOSED ──
        if cn == ref_closed_num:
            closed_positions[cn] = ref_closed_pos
            closed_templates.append(ref_closed_features)
            if multivariate:
                closed_search_results[cn] = (ref_closed_pos, 0.0, [], np.array([]), np.empty((0, len(FEATURES_MULTI))))
            else:
                closed_search_results[cn] = (ref_closed_pos, 0.0, [], np.array([]))
            print(f"  Cycle {cn} CLOSED: reference (pos={ref_closed_pos/config.FSAMP:.3f}s)")
        else:
            if multivariate:
                best_pos, best_dist, positions, total_dists, pf_dists = find_best_match_multi(
                    ref_closed_multi, emg, closed_search_start, closed_search_end,
                    FEATURES_MULTI, step_samples=step_samples,
                )
                closed_search_results[cn] = (best_pos, best_dist, positions, total_dists, pf_dists)
            else:
                best_pos, best_dist, positions, dists = find_best_match(
                    ref_closed_features, emg, closed_search_start, closed_search_end,
                    step_samples=step_samples,
                )
                closed_search_results[cn] = (best_pos, best_dist, positions, dists)
            closed_positions[cn] = best_pos
            segment = emg[:, best_pos:best_pos + template_samples]
            closed_templates.append(extract_features_from_segment(segment))
            print(f"  Cycle {cn} CLOSED: pos={best_pos/config.FSAMP:.3f}s, dist={best_dist:.3f}")

        # ── OPEN ──
        if cn == ref_open_num:
            open_positions[cn] = ref_open_pos
            open_templates.append(ref_open_features)
            if multivariate:
                open_search_results[cn] = (ref_open_pos, 0.0, [], np.array([]), np.empty((0, len(FEATURES_MULTI))))
            else:
                open_search_results[cn] = (ref_open_pos, 0.0, [], np.array([]))
            print(f"  Cycle {cn} OPEN:   reference (pos={ref_open_pos/config.FSAMP:.3f}s)")
        else:
            if multivariate:
                best_pos, best_dist, positions, total_dists, pf_dists = find_best_match_multi(
                    ref_open_multi, emg, open_search_start, open_search_end,
                    FEATURES_MULTI, step_samples=step_samples,
                )
                open_search_results[cn] = (best_pos, best_dist, positions, total_dists, pf_dists)
            else:
                best_pos, best_dist, positions, dists = find_best_match(
                    ref_open_features, emg, open_search_start, open_search_end,
                    step_samples=step_samples,
                )
                open_search_results[cn] = (best_pos, best_dist, positions, dists)
            open_positions[cn] = best_pos
            segment = emg[:, best_pos:best_pos + template_samples]
            open_templates.append(extract_features_from_segment(segment))
            print(f"  Cycle {cn} OPEN:   pos={best_pos/config.FSAMP:.3f}s, dist={best_dist:.3f}")

    # 5. Compute metrics (always single-feature based for consistency)
    print("\n--- Computing template metrics ---")
    metrics = compute_template_metrics(open_templates, closed_templates)
    print_metrics(metrics)

    # 6. Visualize
    fig1 = plot_cycle_overview_with_windows(cycles, closed_positions, open_positions)
    if multivariate:
        fig2 = plot_search_curves_multi(cycles, closed_search_results, open_search_results, FEATURES_MULTI)
    else:
        fig2 = plot_search_curves(cycles, closed_search_results, open_search_results)
    fig3 = plot_distance_matrix(metrics)

    plt.show()


def main_compare():
    """
    Comparison mode: user manually picks templates from all cycles while DTW
    auto-finds best matches. Then compare both sets side-by-side.

    Flow:
    1. Select reference window for CLOSED and OPEN (cycle 1 typically)
    2. DTW auto-finds best match in all other cycles (background)
    3. User manually selects windows from all other cycles (seeing DTW pick as hint)
    4. Compare: positions, metrics, visualization
    """
    # 1. Load recording and extract cycles
    recording_path = Path(PATIENT_DIR) / RECORDING_FILE
    print(f"Loading recording: {recording_path}")
    cycles = load_and_extract_cycles(str(recording_path))

    if not cycles:
        print("ERROR: No cycles extracted.")
        return

    print(f"Mode: COMPARE (user vs DTW)")
    print(f"Feature for DTW search: {FEATURE}")
    print(f"Extracted {len(cycles)} cycles")
    for c in cycles:
        close_cue = f"{c['close_cue_idx']/config.FSAMP:.2f}s" if c.get('close_cue_idx') is not None else "N/A"
        open_cue = f"{c['open_cue_idx']/config.FSAMP:.2f}s" if c.get('open_cue_idx') is not None else "N/A"
        print(f"  Cycle {c['cycle_number']}: {c['duration_s']:.1f}s, "
              f"close_cue={close_cue}, open_cue={open_cue}")

    template_samples = int(TEMPLATE_DURATION_S * config.FSAMP)
    step_samples = int(SEARCH_STEP_S * config.FSAMP)

    # 2. Select CLOSED reference
    ref_closed_num, ref_closed_pos = select_reference_window(cycles, "CLOSED")
    ref_closed_cycle = next(c for c in cycles if c["cycle_number"] == ref_closed_num)
    ref_closed_segment = ref_closed_cycle["emg"][:, ref_closed_pos:ref_closed_pos + template_samples]
    ref_closed_features = extract_features_from_segment(ref_closed_segment)

    # 3. Select OPEN reference
    ref_open_num, ref_open_pos = select_reference_window(cycles, "OPEN")
    ref_open_cycle = next(c for c in cycles if c["cycle_number"] == ref_open_num)
    ref_open_segment = ref_open_cycle["emg"][:, ref_open_pos:ref_open_pos + template_samples]
    ref_open_features = extract_features_from_segment(ref_open_segment)

    # 4. DTW auto-search all cycles
    print("\n--- DTW auto-search across all cycles ---")

    dtw_closed_positions = {}
    dtw_open_positions = {}
    dtw_closed_templates = []
    dtw_open_templates = []

    for cycle in cycles:
        cn = cycle["cycle_number"]
        emg = cycle["emg"]

        closed_search_start = cycle.get("close_cue_idx") or cycle["close_start_idx"]
        closed_search_end = (cycle.get("open_cue_idx") or cycle["open_start_idx"]) - template_samples
        open_search_start = cycle.get("open_cue_idx") or cycle["open_start_idx"]
        open_search_end = emg.shape[1] - template_samples

        # CLOSED
        if cn == ref_closed_num:
            dtw_closed_positions[cn] = ref_closed_pos
            dtw_closed_templates.append(ref_closed_features)
            print(f"  Cycle {cn} CLOSED: reference (pos={ref_closed_pos/config.FSAMP:.3f}s)")
        else:
            best_pos, best_dist, _, _ = find_best_match(
                ref_closed_features, emg, closed_search_start, closed_search_end,
                step_samples=step_samples,
            )
            dtw_closed_positions[cn] = best_pos
            segment = emg[:, best_pos:best_pos + template_samples]
            dtw_closed_templates.append(extract_features_from_segment(segment))
            print(f"  Cycle {cn} CLOSED: pos={best_pos/config.FSAMP:.3f}s, dist={best_dist:.3f}")

        # OPEN
        if cn == ref_open_num:
            dtw_open_positions[cn] = ref_open_pos
            dtw_open_templates.append(ref_open_features)
            print(f"  Cycle {cn} OPEN:   reference (pos={ref_open_pos/config.FSAMP:.3f}s)")
        else:
            best_pos, best_dist, _, _ = find_best_match(
                ref_open_features, emg, open_search_start, open_search_end,
                step_samples=step_samples,
            )
            dtw_open_positions[cn] = best_pos
            segment = emg[:, best_pos:best_pos + template_samples]
            dtw_open_templates.append(extract_features_from_segment(segment))
            print(f"  Cycle {cn} OPEN:   pos={best_pos/config.FSAMP:.3f}s, dist={best_dist:.3f}")

    # 5. User manual selection from all non-reference cycles
    print("\n--- YOUR TURN: Select templates from each cycle ---")
    print("  (The gray overlay shows where DTW would pick)\n")

    user_closed_positions = {}
    user_open_positions = {}
    user_closed_templates = []
    user_open_templates = []

    for cycle in cycles:
        cn = cycle["cycle_number"]
        emg = cycle["emg"]

        # CLOSED
        if cn == ref_closed_num:
            user_closed_positions[cn] = ref_closed_pos
            user_closed_templates.append(ref_closed_features)
        else:
            user_pos = select_window_from_cycle(
                cycle, "CLOSED", dtw_pos=dtw_closed_positions[cn]
            )
            user_closed_positions[cn] = user_pos
            segment = emg[:, user_pos:user_pos + template_samples]
            user_closed_templates.append(extract_features_from_segment(segment))

        # OPEN
        if cn == ref_open_num:
            user_open_positions[cn] = ref_open_pos
            user_open_templates.append(ref_open_features)
        else:
            user_pos = select_window_from_cycle(
                cycle, "OPEN", dtw_pos=dtw_open_positions[cn]
            )
            user_open_positions[cn] = user_pos
            segment = emg[:, user_pos:user_pos + template_samples]
            user_open_templates.append(extract_features_from_segment(segment))

    # 6. Compute metrics for both sets
    print("\n--- Computing metrics ---")

    print("\n  [DTW picks]")
    dtw_metrics = compute_template_metrics(dtw_open_templates, dtw_closed_templates)
    print_metrics(dtw_metrics)

    print("\n  [User picks]")
    user_metrics = compute_template_metrics(user_open_templates, user_closed_templates)
    print_metrics(user_metrics)

    # 7. Print comparison
    print_comparison_summary(
        cycles,
        dtw_closed_positions, dtw_open_positions,
        user_closed_positions, user_open_positions,
        dtw_metrics, user_metrics,
    )

    # 8. Visualize
    fig1 = plot_comparison_overview(
        cycles,
        dtw_closed_positions, dtw_open_positions,
        user_closed_positions, user_open_positions,
    )

    # Shared color scale for both distance matrices
    vmin = min(dtw_metrics["distance_matrix"].min(), user_metrics["distance_matrix"].min())
    vmax = max(dtw_metrics["distance_matrix"].max(), user_metrics["distance_matrix"].max())

    fig_dm, (ax_dtw, ax_user) = plt.subplots(1, 2, figsize=(16, 7))
    plot_distance_matrix(dtw_metrics, vmin=vmin, vmax=vmax, ax=ax_dtw)
    ax_dtw.set_title("DTW Picks")
    plot_distance_matrix(user_metrics, vmin=vmin, vmax=vmax, ax=ax_user)
    ax_user.set_title("User Picks")
    fig_dm.suptitle("Distance Matrices — Shared Scale", fontsize=13)
    fig_dm.tight_layout()

    plt.show()


def main_onset():
    """
    Onset detection mode: automatically detect EMG onset per channel in each cycle,
    place template windows at earliest onset - 20% preparation.
    Shows diagnostic plots so user can evaluate placement quality.
    """
    recording_path = Path(PATIENT_DIR) / RECORDING_FILE
    print(f"Loading recording: {recording_path}")
    cycles = load_and_extract_cycles(str(recording_path))

    if not cycles:
        print("ERROR: No cycles extracted.")
        return

    print(f"Mode: PER-CHANNEL ONSET DETECTION (fully automatic)")
    print(f"Method: {ONSET_METHOD}")
    if ONSET_METHOD == "cusum":
        print(f"Parameters: drift={CUSUM_DRIFT}, h={CUSUM_H}, "
              f"baseline={ONSET_BASELINE_DURATION_S}s, pre_fraction={ONSET_PRE_FRACTION}")
    else:
        print(f"Parameters: k={ONSET_THRESHOLD_K}, sustained={ONSET_MIN_SUSTAINED_S}s, "
              f"baseline={ONSET_BASELINE_DURATION_S}s, pre_fraction={ONSET_PRE_FRACTION}")
    print(f"Extracted {len(cycles)} cycles\n")

    template_samples = int(TEMPLATE_DURATION_S * config.FSAMP)

    closed_results = {}
    open_results = {}
    closed_positions = {}
    open_positions = {}
    closed_templates = []
    open_templates = []

    for cycle in cycles:
        cn = cycle["cycle_number"]
        emg = cycle["emg"]
        n_samples = emg.shape[1]

        closed_search_start = cycle.get("close_cue_idx") or cycle["close_start_idx"]
        closed_search_end = (cycle.get("open_cue_idx") or cycle["open_start_idx"]) - template_samples
        open_search_start = cycle.get("open_cue_idx") or cycle["open_start_idx"]
        open_search_end = n_samples - template_samples

        # Select detection function
        if ONSET_METHOD == "cusum":
            _detect = detect_onset_per_channel_cusum
        else:
            _detect = detect_onset_per_channel

        # Baseline from BEFORE the cue (pre-cue period captures the previous state)
        baseline_samples = int(ONSET_BASELINE_DURATION_S * config.FSAMP)
        closed_baseline_start = max(0, closed_search_start - baseline_samples)
        open_baseline_start = max(0, open_search_start - baseline_samples)

        # Dead channel detection (whole cycle)
        dead_chs = detect_dead_channels(emg)
        if dead_chs:
            print(f"  Cycle {cn} DEAD channels: {[ch+1 for ch in dead_chs]}")

        # ── CLOSED onset ──
        result = _detect(emg, closed_search_start, closed_search_end,
                         baseline_start=closed_baseline_start)
        closed_results[cn] = result
        earliest = result["earliest_onset"]
        channels_fired = result["channels_fired"]

        if earliest is not None:
            pos = place_template_at_onset(earliest, n_samples)
            closed_positions[cn] = pos
            segment = emg[:, pos:pos + template_samples]
            closed_templates.append(extract_features_from_segment(segment))
            fired_str = ",".join(str(ch+1) for ch in channels_fired)
            print(f"  Cycle {cn} CLOSED: onset={earliest/config.FSAMP:.3f}s, "
                  f"window={pos/config.FSAMP:.3f}s-{(pos+template_samples)/config.FSAMP:.3f}s, "
                  f"channels=[{fired_str}]")

            # Artifact detection for CLOSED template
            art_result = detect_artifact_channels(
                emg, pos, pos + template_samples,
                closed_baseline_start, closed_search_start,
            )
            if art_result["artifact_channels"]:
                art_str = ",".join(str(ch+1) for ch in art_result["artifact_channels"])
                print(f"           CLOSED artifacts: CH[{art_str}]")
        else:
            print(f"  Cycle {cn} CLOSED: NO ONSET DETECTED")

        # ── OPEN onset ──
        result = _detect(emg, open_search_start, open_search_end,
                         baseline_start=open_baseline_start)
        open_results[cn] = result
        earliest = result["earliest_onset"]
        channels_fired = result["channels_fired"]

        if earliest is not None:
            pos = place_template_at_onset(earliest, n_samples)
            open_positions[cn] = pos
            segment = emg[:, pos:pos + template_samples]
            open_templates.append(extract_features_from_segment(segment))
            fired_str = ",".join(str(ch+1) for ch in channels_fired)
            print(f"  Cycle {cn} OPEN:   onset={earliest/config.FSAMP:.3f}s, "
                  f"window={pos/config.FSAMP:.3f}s-{(pos+template_samples)/config.FSAMP:.3f}s, "
                  f"channels=[{fired_str}]")

            # Artifact detection for OPEN template
            art_result = detect_artifact_channels(
                emg, pos, pos + template_samples,
                open_baseline_start, open_search_start,
            )
            if art_result["artifact_channels"]:
                art_str = ",".join(str(ch+1) for ch in art_result["artifact_channels"])
                print(f"           OPEN artifacts: CH[{art_str}]")
        else:
            print(f"  Cycle {cn} OPEN:   NO ONSET DETECTED")

    # Cross-cycle channel consistency
    print("\n--- Channel consistency across cycles ---")
    n_ch = cycles[0]["emg"].shape[0]
    for class_name, results in [("CLOSED", closed_results), ("OPEN", open_results)]:
        ch_counts = np.zeros(n_ch, dtype=int)
        for cn, result in results.items():
            for ch in result["channels_fired"]:
                ch_counts[ch] += 1
        print(f"\n  {class_name} — channels that fired (count / {len(cycles)} cycles):")
        for ch in range(n_ch):
            if ch_counts[ch] > 0:
                bar = "#" * ch_counts[ch]
                print(f"    CH{ch+1:>2}: {ch_counts[ch]:>2} {bar}")

    # Summary
    n_closed_ok = sum(1 for r in closed_results.values() if r["earliest_onset"] is not None)
    n_open_ok = sum(1 for r in open_results.values() if r["earliest_onset"] is not None)
    print(f"\n  Detected: {n_closed_ok}/{len(cycles)} CLOSED onsets, "
          f"{n_open_ok}/{len(cycles)} OPEN onsets")

    # Compute metrics if we have enough templates
    if len(closed_templates) >= 2 and len(open_templates) >= 2:
        print("\n--- Computing template metrics ---")
        metrics = compute_template_metrics(open_templates, closed_templates)
        print_metrics(metrics)
    else:
        metrics = None
        print("\n  Not enough templates for metrics (need >= 2 per class)")

    # Diagnostic plots
    fig1 = plot_onset_detection(
        cycles, closed_results, open_results,
        closed_positions, open_positions,
    )

    if closed_positions or open_positions:
        fig2 = plot_cycle_overview_with_windows(
            cycles, closed_positions, open_positions,
            title="Onset Detection — Placed Windows",
        )

    if metrics is not None:
        fig3 = plot_distance_matrix(metrics)

    plt.show()


if __name__ == "__main__":
    if MODE == "compare":
        main_compare()
    elif MODE == "onset":
        main_onset()
    else:
        main()
