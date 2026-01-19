from datetime import datetime
import numpy as np
import os
import pickle
from mindmove.config import config

def load_templates_recordings(DATA_FOLDER):
    """Load all EMG templates recordings from pickle files."""

    data_files = sorted(f for f in os.listdir(DATA_FOLDER) if f.endswith('.pkl'))

    # stores all recordings in a list (already pre-procesed)
    data_list = [] 

    for file_name in data_files:
        file_path = os.path.join(DATA_FOLDER, file_name)
        with open(file_path, 'rb') as file:
            recording = pickle.load(file)

            # pre-processing
            emg = recording["biosignal"]
            emg = np.concatenate(emg.T, axis=0).T
            emg = emg[:config.num_channels, :]
            # emg = np.delete(emg, config.dead_channels-1, axis=0)  # delete dead channels          
            
            data_list.append(emg)
    
    return data_list

def load_ground_truth_recordings(DATA_FOLDER):
    """Load ground truth recorded with the keyboard in pickle files."""

    data_files = sorted(f for f in os.listdir(DATA_FOLDER) if f.endswith('.pkl'))
    gt_list = []

    for file_name in data_files:
            file_path = os.path.join(DATA_FOLDER, file_name)
            with open(file_path, 'rb') as file:
                gt = pickle.load(file)
                gt_list.append(gt)
    
    return gt_list

def load_emg_and_gt(DATA_FOLDER_EMG, DATA_FOLDER_GT):
    data_files_emg = sorted(f for f in os.listdir(DATA_FOLDER_EMG) if f.endswith('.pkl'))
    data_files_gt = sorted(f for f in os.listdir(DATA_FOLDER_GT) if f.endswith('.pkl'))

    data_list = []
    gt_list = []

    for filename_emg, filename_gt in zip(data_files_emg, data_files_gt):
        # compute starting time for emg
        base = os.path.basename(filename_emg)
        parts = base.split("_")
        date_str = parts[2]  # YYYYMMDD
        time_str = parts[3]  # HHMMSSfff
        emg_start = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S%f")
        
        # open emg and gt files
        file_path_emg = os.path.join(DATA_FOLDER_EMG, filename_emg)
        file_path_gt = os.path.join(DATA_FOLDER_GT, filename_gt)
        with open(file_path_gt, 'rb') as file:
            gt = pickle.load(file)
            offset = (emg_start - gt['start_datetime']).total_seconds() # offset in seconds
            # print(offset)
            gt['key_events'] = [(t - offset, state) for t, state in gt['key_events']]
            gt_list.append(gt)

        with open(file_path_emg, 'rb') as file:
            recording = pickle.load(file)
            # pre-processing
            emg = recording["biosignal"]
            emg = np.concatenate(emg.T, axis=0).T
            emg = emg[:config.num_channels, :]
            # emg = np.delete(emg, config.dead_channels-1, axis=0)  # delete dead channels          
            data_list.append(emg)

    if len(data_list) == 1:
        return emg, gt
    else:
        return data_list, gt_list
    
def convert_gt_to_binary(gt_dict, nsamp):
    """
    Convert GT from key_events dict into a full binary vector.
    """
    gt_binary = np.zeros(nsamp, dtype=int)
    key_events = gt_dict["key_events"]
    event_idx = 0
    current_state = 0

    for i in range(nsamp):
        t = i / config.FSAMP
        while event_idx < len(key_events) and key_events[event_idx][0] <= t:
            current_state = key_events[event_idx][1]
            event_idx += 1
        gt_binary[i] = current_state

    return gt_binary

def activation_extractor(emg, gt, activations_list, non_activations_list):
    """
    the function appends to activations and non_activations all
    the segments where there is one and there is not

    """

    # if activations_list is None:
    #     activations_list = []
    # if non_activations_list is None:
    #     non_activations_list = []
        
    nch, nsamp = emg.shape



    gt_binary = convert_gt_to_binary(gt,nsamp)

    gt = np.asarray(gt_binary)
    

    # Identify transitions
    diffs = np.diff(gt, prepend=gt[0])

    starts_1 = np.where(diffs == 1)[0]          # rising edges
    ends_1   = np.where(diffs == -1)[0] - 1     # falling edges

    # If GT ends in a 1, close last segment
    if gt[-1] == 1:
        ends_1 = np.append(ends_1, len(gt)-1)

    # Now do the same for non-activations (segments where gt == 0)

    # Invert GT
    gt_inv = 1 - gt
    diffs0 = np.diff(gt_inv, prepend=gt_inv[0])

    starts_0 = np.where(diffs0 == 1)[0]
    ends_0   = np.where(diffs0 == -1)[0] - 1

    if gt[0] == 0 and starts_0[0] != 0:
        starts_0 = np.insert(starts_0, 0, 0)

    if gt[-1] == 0:
        ends_0 = np.append(ends_0, len(gt)-1)

    # Extract and append EMG segments
    for s, e in zip(starts_1, ends_1):
        # print(s)
        # print(e)
        activations_list.append(emg[:,s:e+1])

    for s, e in zip(starts_0, ends_0):
        # print(s)
        # print(e)
        non_activations_list.append(emg[:,s:e+1])

    return gt_binary, activations_list, non_activations_list        


def extract_intervals(timestamps, values):
    intervals = []
    current_start = None

    for t, v in zip(timestamps, values):
        if v == 1 and current_start is None:
            # Activation begins
            current_start = t
        elif v == 0 and current_start is not None:
            # Activation ends
            intervals.append((current_start, t))
            current_start = None

    # If recording ended while still active
    if current_start is not None:
        intervals.append((current_start, timestamps[-1]))

    return intervals


# ============ Kinematics-based Ground Truth Functions ============

def convert_kinematics_to_binary(
    kinematics: np.ndarray,
    n_samples_emg: int = None,
    fsamp_kin: int = None
) -> np.ndarray:
    """
    Convert virtual hand kinematics to binary activation signal.

    Uses mean of all kinematic dimensions as activation measure.
    Higher values = more flexion = closed hand.

    Args:
        kinematics: (n_dims, n_samples) kinematic data from virtual hand
        n_samples_emg: Number of EMG samples to match (for upsampling)
        fsamp_kin: Kinematics sampling frequency (default from config)

    Returns:
        Binary vector (0=open, 1=closed). If n_samples_emg provided,
        upsampled to EMG rate.
    """
    if fsamp_kin is None:
        fsamp_kin = config.KINEMATICS_FSAMP

    # Use mean of joint values as activation measure
    mean_activation = np.mean(kinematics, axis=0)

    # Threshold at midpoint between min and max
    threshold = (np.max(mean_activation) + np.min(mean_activation)) / 2
    binary_kin = (mean_activation > threshold).astype(int)

    # Upsample to EMG rate if requested
    if n_samples_emg is not None:
        from scipy.ndimage import zoom
        zoom_factor = n_samples_emg / len(binary_kin)
        binary_emg = zoom(binary_kin.astype(float), zoom_factor, order=0)
        binary_emg = (binary_emg > 0.5).astype(int)
        return binary_emg[:n_samples_emg]

    return binary_kin


def extract_activation_segments_from_binary(
    emg: np.ndarray,
    gt_binary: np.ndarray,
    min_duration_s: float = 1.5,
    onset_offset_s: float = 0.0
) -> list:
    """
    Extract EMG segments where GT=1, with optional onset offset.

    Args:
        emg: (n_channels, n_samples)
        gt_binary: Binary ground truth at EMG sample rate
        min_duration_s: Minimum activation duration to keep
        onset_offset_s: Seconds to include before GT transition
                       (negative = before GT=1, positive = after)

    Returns:
        List of EMG segments (each is n_channels x n_samples)
    """
    # Find rising/falling edges
    diffs = np.diff(gt_binary, prepend=gt_binary[0])
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    # Handle edge cases
    if len(ends) == 0 or (len(starts) > 0 and len(ends) > 0 and ends[0] < starts[0]):
        starts = np.insert(starts, 0, 0)
    if gt_binary[-1] == 1:
        ends = np.append(ends, len(gt_binary))

    # Ensure same length
    n_segments = min(len(starts), len(ends))
    starts = starts[:n_segments]
    ends = ends[:n_segments]

    # Extract segments
    segments = []
    min_samples = int(min_duration_s * config.FSAMP)
    offset_samples = int(onset_offset_s * config.FSAMP)

    for start, end in zip(starts, ends):
        duration = end - start
        if duration >= min_samples:
            # Apply onset offset (negative = start earlier)
            actual_start = max(0, start + offset_samples)
            segments.append(emg[:, actual_start:end])

    return segments


def load_mindmove_recording(filepath: str) -> dict:
    """
    Load a MindMove recording file.

    Args:
        filepath: Path to .pkl recording file

    Returns:
        Recording dict with keys: emg, kinematics, timings_emg, timings_kinematics, label, task
    """
    with open(filepath, 'rb') as f:
        recording = pickle.load(f)

    # Validate required keys
    required_keys = ["emg", "kinematics", "timings_emg", "timings_kinematics", "label", "task"]
    if not all(key in recording for key in required_keys):
        raise ValueError(f"Recording missing required keys. Found: {list(recording.keys())}")

    return recording
