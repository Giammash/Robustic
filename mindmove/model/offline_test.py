"""
Offline testing script for DTW model.

Simulates real-time processing of test recordings using the buffer-based DTW computation.
Compares performance between custom DTW and tslearn optimized DTW.
"""
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat

from mindmove.config import config
from mindmove.model.core.algorithm import compute_dtw, compute_distance_from_training_set_online, compute_spatial_similarity, GPUDTW_AVAILABLE
from mindmove.model.core.windowing import sliding_window
from mindmove.model.core.filtering import apply_rtfiltering
from mindmove.model.core.features.features_registry import FEATURES


def compute_metrics(results: dict, gt_data: np.ndarray, gt_mode: str = None) -> dict:
    """
    Compute classification metrics from simulation results and ground truth.

    Returns dict with: accuracy, per-class precision/recall/F1, response delay,
    false transitions, and stability metrics.
    """
    timestamps = results['timestamps']
    predictions = results['predictions']
    n_preds = len(predictions)

    if gt_data is None or n_preds == 0:
        return None

    # Convert predictions to numeric: CLOSED=1, OPEN=0
    pred_numeric = np.array([1 if p == "CLOSED" else 0 for p in predictions])

    # Align GT to prediction timestamps
    gt_1d = gt_data.flatten() if gt_data.ndim > 1 else gt_data.copy()
    if len(gt_1d) == n_preds:
        gt_aligned = (gt_1d > 0.5).astype(int)
    else:
        # Resample GT to match prediction count
        gt_time = np.linspace(timestamps[0], timestamps[-1], len(gt_1d))
        gt_interp = np.interp(timestamps, gt_time, gt_1d)
        gt_aligned = (gt_interp > 0.5).astype(int)

    # --- Basic classification metrics ---
    correct = (pred_numeric == gt_aligned).sum()
    accuracy = correct / n_preds

    # Per-class: CLOSED=1 (positive), OPEN=0 (negative)
    tp = ((pred_numeric == 1) & (gt_aligned == 1)).sum()  # true CLOSED
    fp = ((pred_numeric == 1) & (gt_aligned == 0)).sum()  # false CLOSED
    tn = ((pred_numeric == 0) & (gt_aligned == 0)).sum()  # true OPEN
    fn = ((pred_numeric == 0) & (gt_aligned == 1)).sum()  # false OPEN (missed close)

    prec_closed = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_closed = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_closed = 2 * prec_closed * rec_closed / (prec_closed + rec_closed) if (prec_closed + rec_closed) > 0 else 0.0

    prec_open = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_open = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_open = 2 * prec_open * rec_open / (prec_open + rec_open) if (prec_open + rec_open) > 0 else 0.0

    # --- Transition analysis ---
    # Find GT transitions
    gt_transitions = []
    for i in range(1, len(gt_aligned)):
        if gt_aligned[i] != gt_aligned[i-1]:
            gt_transitions.append((i, "CLOSED" if gt_aligned[i] == 1 else "OPEN"))

    # Find prediction transitions
    pred_transitions = []
    for i in range(1, len(pred_numeric)):
        if pred_numeric[i] != pred_numeric[i-1]:
            pred_transitions.append((i, "CLOSED" if pred_numeric[i] == 1 else "OPEN"))

    # Response delay: for each GT transition, find nearest matching pred transition
    dt = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 0.05  # DTW interval
    response_delays = []
    matched_pred_transitions = set()
    tolerance_samples = int(10.0 / dt)  # 10 second search window

    for gt_idx, gt_dir in gt_transitions:
        best_delay = None
        best_pred_idx = None
        for pi, (pred_idx, pred_dir) in enumerate(pred_transitions):
            if pred_dir == gt_dir and pi not in matched_pred_transitions:
                delay = (pred_idx - gt_idx) * dt
                if -2.0 <= delay <= 10.0:  # allow 2s early, 10s late
                    if best_delay is None or abs(delay) < abs(best_delay):
                        best_delay = delay
                        best_pred_idx = pi
        if best_delay is not None:
            response_delays.append(best_delay)
            matched_pred_transitions.add(best_pred_idx)

    # False transitions = pred transitions not matched to any GT transition
    n_false_transitions = len(pred_transitions) - len(matched_pred_transitions)

    # --- Time in correct state ---
    time_correct_pct = 100.0 * accuracy

    # --- Stability ---
    n_gt_transitions = len(gt_transitions)
    n_pred_transitions = len(pred_transitions)

    metrics = {
        'accuracy': accuracy,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'precision_closed': prec_closed,
        'recall_closed': rec_closed,
        'f1_closed': f1_closed,
        'precision_open': prec_open,
        'recall_open': rec_open,
        'f1_open': f1_open,
        'f1_macro': (f1_closed + f1_open) / 2,
        'n_gt_transitions': n_gt_transitions,
        'n_pred_transitions': n_pred_transitions,
        'n_false_transitions': n_false_transitions,
        'n_matched_transitions': len(response_delays),
        'mean_response_delay_s': np.mean(response_delays) if response_delays else float('nan'),
        'median_response_delay_s': np.median(response_delays) if response_delays else float('nan'),
        'std_response_delay_s': np.std(response_delays) if response_delays else float('nan'),
        'time_correct_pct': time_correct_pct,
    }
    return metrics


def print_metrics(metrics: dict, label: str = ""):
    """Print metrics in a readable format."""
    if metrics is None:
        print("  No GT available for metrics.")
        return
    if label:
        print(f"\n  --- {label} ---")
    print(f"  Accuracy:          {metrics['accuracy']:.1%}  ({metrics['time_correct_pct']:.1f}% time in correct state)")
    print(f"  CLOSED  P={metrics['precision_closed']:.2f}  R={metrics['recall_closed']:.2f}  F1={metrics['f1_closed']:.2f}  (TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']})")
    print(f"  OPEN    P={metrics['precision_open']:.2f}  R={metrics['recall_open']:.2f}  F1={metrics['f1_open']:.2f}  (TN={metrics['tn']})")
    print(f"  F1 macro:          {metrics['f1_macro']:.3f}")
    print(f"  Transitions:       {metrics['n_pred_transitions']} pred vs {metrics['n_gt_transitions']} GT")
    print(f"  Matched:           {metrics['n_matched_transitions']}/{metrics['n_gt_transitions']} GT transitions detected")
    print(f"  False transitions: {metrics['n_false_transitions']}")
    if not np.isnan(metrics['mean_response_delay_s']):
        print(f"  Response delay:    {metrics['mean_response_delay_s']*1000:.0f}ms mean, "
              f"{metrics['median_response_delay_s']*1000:.0f}ms median "
              f"(+/- {metrics['std_response_delay_s']*1000:.0f}ms)")


def print_sweep_table(sweep_results: list):
    """Print a comparison table for parameter sweep results."""
    if not sweep_results:
        return

    print(f"\n{'='*130}")
    print(f"PARAMETER SWEEP RESULTS")
    print(f"{'='*130}")
    header = (f"{'Feature':<6} {'Aggreg':<14} {'Smooth':<12} {'ThOpen':>7} {'ThClos':>7} "
              f"{'Acc':>6} {'F1mac':>6} {'F1_CL':>6} {'F1_OP':>6} "
              f"{'Prec_C':>6} {'Rec_C':>6} "
              f"{'Tr_P':>5} {'Tr_GT':>5} {'Match':>5} {'False':>5} "
              f"{'Delay':>7}")
    print(header)
    print(f"{'-'*130}")

    # Sort by F1 macro descending
    for r in sorted(sweep_results, key=lambda x: x['metrics']['f1_macro'], reverse=True):
        m = r['metrics']
        delay_str = f"{m['mean_response_delay_s']*1000:.0f}ms" if not np.isnan(m['mean_response_delay_s']) else "N/A"
        print(f"{r['feature']:<6} {r['aggregation']:<14} {r['smoothing']:<12} "
              f"{r['th_open']:>7.3f} {r['th_closed']:>7.3f} "
              f"{m['accuracy']:>5.1%} {m['f1_macro']:>6.3f} {m['f1_closed']:>6.3f} {m['f1_open']:>6.3f} "
              f"{m['precision_closed']:>6.2f} {m['recall_closed']:>6.2f} "
              f"{m['n_pred_transitions']:>5} {m['n_gt_transitions']:>5} {m['n_matched_transitions']:>5} {m['n_false_transitions']:>5} "
              f"{delay_str:>7}")
    print(f"{'='*130}")


def load_templates(templates_path: str) -> list:
    """Load pre-computed feature templates from pickle file."""
    with open(templates_path, 'rb') as f:
        templates = pickle.load(f)
    return templates


def load_model(model_path: str) -> dict:
    """Load trained DTW model."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def load_test_recording(recording_path: str) -> np.ndarray:
    """
    Load test recording and convert to (n_channels, n_samples) format.

    Supports multiple formats:
    - MindMove keyboard format: {emg, gt, ...}
    - MindMove virtual hand format: {emg, kinematics, ...}
    - VHI format: {biosignal, device_information, ...}
    - MAT files with biosignal data
    """
    recording_path = Path(recording_path)
    file_ext = recording_path.suffix.lower()

    if file_ext == '.pkl':
        with open(recording_path, 'rb') as f:
            data = pickle.load(f)

        # Detect format based on keys
        if 'emg' in data:
            # MindMove format (keyboard or virtual hand)
            emg_data = data['emg']
            n_emg_channels = emg_data.shape[0]
            sampling_freq = config.FSAMP

            # Get ground truth if available
            if 'gt' in data:
                gt_data = data['gt']
                gt_mode = "keyboard"
            elif 'kinematics' in data and hasattr(data['kinematics'], 'size') and data['kinematics'].size > 0:
                gt_data = data['kinematics']
                gt_mode = "virtual_hand"
            elif 'predictions' in data and data['predictions']:
                # Prediction file: use stored predictions as reference GT
                gt_data = np.array(data['predictions'], dtype=float)
                gt_mode = "predictions"
            else:
                gt_data = None
                gt_mode = "none"

            # Add GT to data for later use
            data['_gt_data'] = gt_data
            data['_gt_mode'] = gt_mode

            print(f"Loaded recording: {recording_path}")
            print(f"  Format: MindMove ({gt_mode})")
            print(f"  EMG data shape: {emg_data.shape}")
            print(f"  Duration: {emg_data.shape[1] / sampling_freq:.1f} seconds")
            if gt_data is not None:
                gt_shape = gt_data.shape if hasattr(gt_data, 'shape') else len(gt_data)
                print(f"  GT data shape: {gt_shape}")

            return emg_data, data

        elif 'biosignal' in data:
            # VHI format
            biosignal = data['biosignal']
            device_info = data['device_information']
            n_emg_channels = device_info['number_of_biosignal_channels']
            sampling_freq = device_info['sampling_frequency']

            # Take only EMG channels and reshape
            emg_data = biosignal[:n_emg_channels, :, :]
            emg_data = np.concatenate(emg_data.T, axis=0).T

            print(f"Loaded recording: {recording_path}")
            print(f"  Format: VHI")
            print(f"  Original shape: {biosignal.shape}")
            print(f"  EMG data shape: {emg_data.shape}")
            print(f"  Duration: {emg_data.shape[1] / sampling_freq:.1f} seconds")

            return emg_data, data

        else:
            raise ValueError(f"Unknown pickle format. Keys: {list(data.keys())}")

    elif file_ext == '.mat':
        mat_data = loadmat(str(recording_path), squeeze_me=True)
        biosignal = mat_data['biosignal']
        # Handle device_information from .mat (may be a structured array)
        if 'device_information' in mat_data:
            device_info = mat_data['device_information']
            # If it's a structured numpy array, convert to dict
            if hasattr(device_info, 'dtype') and device_info.dtype.names:
                device_info = {name: device_info[name].item() for name in device_info.dtype.names}
            elif isinstance(device_info, np.ndarray) and device_info.size == 1:
                device_info = device_info.item()
            n_emg_channels = device_info.get('number_of_biosignal_channels', 32)
            sampling_freq = device_info.get('sampling_frequency', config.FSAMP)
        else:
            # Fallback to config defaults if device_information not present
            n_emg_channels = config.num_channels
            sampling_freq = config.FSAMP
            device_info = {'number_of_biosignal_channels': n_emg_channels, 'sampling_frequency': sampling_freq}
        data = {'biosignal': biosignal, 'device_information': device_info}

        # Take only EMG channels (first 32) and reshape to (n_channels, n_samples)
        emg_data = biosignal[:n_emg_channels, :, :]  # (32, 18, 11000)
        emg_data = np.concatenate(emg_data.T, axis=0).T  # (32, 198000)

        print(f"Loaded recording: {recording_path}")
        print(f"  Format: MAT")
        print(f"  Original shape: {biosignal.shape}")
        print(f"  EMG data shape: {emg_data.shape}")
        print(f"  Duration: {emg_data.shape[1] / sampling_freq:.1f} seconds")

        return emg_data, data
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .pkl or .mat")


def simulate_realtime_dtw(
    emg_data: np.ndarray,
    templates_open: list,
    templates_closed: list,
    threshold_open: float,
    threshold_closed: float,
    feature_name: str = 'wl',
    verbose: bool = True,
    distance_aggregation: str = 'average',
    refractory_period_s: float = None,
    spatial_ref_open: dict = None,
    spatial_ref_closed: dict = None,
    spatial_threshold: float = 0.5,
    use_spatial_correction: bool = False,
    spatial_mode: str = "off",
    spatial_sharpness: float = 3.0,
    spatial_relu_baseline: float = 0.2,
    initial_state: str = "CLOSED",
    decision_mode: str = "threshold",
    decision_nn_weights: dict = None,
    decision_catboost_model: dict = None,
    spatial_similarity_mode: str = "mean",
    spatial_n_best: int = 3,
) -> dict:
    """
    Simulate real-time DTW processing using the same buffer logic as the online model.

    Args:
        emg_data: EMG data array (n_channels, n_samples)
        templates_open: List of open hand templates
        templates_closed: List of closed hand templates
        threshold_open: Threshold for open detection
        threshold_closed: Threshold for closed detection
        feature_name: Feature to extract (default 'wl')
        verbose: Print progress

    Returns:
        Dictionary with results including distances and timing info
    """
    if refractory_period_s is None:
        refractory_period_s = 0.0

    # Initialize ML decision models if applicable
    decision_nn = None
    decision_catboost = None
    if decision_mode == "nn" and decision_nn_weights is not None:
        from mindmove.model.core.decision_network import DecisionNetworkInference
        decision_nn = DecisionNetworkInference(decision_nn_weights)
    elif decision_mode == "catboost" and decision_catboost_model is not None:
        from mindmove.model.core.decision_network import CatBoostDecisionInference
        decision_catboost = CatBoostDecisionInference(decision_catboost_model)

    n_channels, n_samples = emg_data.shape

    # Buffer configuration (same as Model class)
    buffer_length = config.template_nsamp  # 1 second = 2000 samples
    increment_dtw = config.increment_dtw_samples  # 100 samples (50ms)
    window_length = config.window_length  # 192 samples
    increment = config.increment  # 64 samples

    # Initialize buffer
    emg_buffer = np.zeros((n_channels, buffer_length))

    # Results storage
    D_open_list = []
    D_closed_list = []
    predictions = []
    dtw_times = []
    timestamps = []
    sim_open_list = []
    sim_closed_list = []
    spatial_blocked_list = []  # True/False per tick
    ml_prob_list = []  # p(CLOSED) per tick from NN/CatBoost, None in threshold mode
    D_open_corrected_list = []   # spatially-corrected D_open (scaling/contrast only, else None)
    D_closed_corrected_list = [] # spatially-corrected D_closed (scaling/contrast only, else None)

    # Filtered signal reconstruction (only the "new" 50ms from each filtered buffer)
    filtered_signal_chunks = []

    # State machine
    current_state = initial_state
    last_predictions = []

    # Refractory period
    last_state_change_time = -refractory_period_s  # allow immediate first change
    in_refractory = False

    # Feature function
    feature_fn = FEATURES[feature_name]["function"]

    # Simulate packet arrival (simulate 50ms chunks = 100 samples at 2000Hz)
    packet_size = increment_dtw
    n_packets = n_samples // packet_size

    samples_since_last_dtw = 0

    if verbose:
        print(f"\nStarting offline simulation...")
        print(f"  Total samples: {n_samples}")
        print(f"  Packet size: {packet_size} samples ({packet_size/config.FSAMP*1000:.1f}ms)")
        print(f"  Number of packets: {n_packets}")
        print(f"  Buffer length: {buffer_length} samples ({buffer_length/config.FSAMP:.1f}s)")
        print(f"  DTW increment: {increment_dtw} samples ({increment_dtw/config.FSAMP*1000:.1f}ms)")
        print(f"  Using tslearn DTW: {config.USE_TSLEARN_DTW}")
        if refractory_period_s > 0:
            print(f"  Refractory period: {refractory_period_s:.1f}s")
        print()

    total_dtw_time = 0
    dtw_count = 0

    for i in range(n_packets):
        # Get new packet
        start_idx = i * packet_size
        end_idx = start_idx + packet_size
        new_samples = emg_data[:, start_idx:end_idx]

        # Update buffer (sliding window)
        emg_buffer = np.roll(emg_buffer, -packet_size, axis=1)
        emg_buffer[:, -packet_size:] = new_samples
        samples_since_last_dtw += packet_size

        # Check if we should compute DTW
        if samples_since_last_dtw >= increment_dtw and i >= (buffer_length // packet_size):
            samples_since_last_dtw = 0

            # Apply filtering
            if config.ENABLE_FILTERING:
                emg_filtered = apply_rtfiltering(emg_buffer)
                # Store only the LAST packet_size samples (the "new" 50ms)
                # These are the samples that just entered the buffer and got filtered
                # with 1900 samples of "history" before them
                filtered_signal_chunks.append(emg_filtered[:, -packet_size:].copy())
            else:
                emg_filtered = emg_buffer.copy()
                # Store raw samples for consistency
                filtered_signal_chunks.append(emg_buffer[:, -packet_size:].copy())

            # Extract features
            windowed = sliding_window(emg_filtered, window_length, increment)
            features = feature_fn(windowed)

            # Compute DTW distances
            dtw_start = time.perf_counter()

            D_open = compute_distance_from_training_set_online(features, templates_open, distance_aggregation=distance_aggregation)
            D_closed = compute_distance_from_training_set_online(features, templates_closed, distance_aggregation=distance_aggregation)

            dtw_end = time.perf_counter()
            dtw_time_ms = (dtw_end - dtw_start) * 1000

            total_dtw_time += dtw_time_ms
            dtw_count += 1

            # Store results
            D_open_list.append(D_open)
            D_closed_list.append(D_closed)
            dtw_times.append(dtw_time_ms)
            timestamps.append(end_idx / config.FSAMP)

            # State machine logic
            current_time_s = end_idx / config.FSAMP

            # Spatial similarity computation (always compute both if refs available)
            sim_open_val = None
            sim_closed_val = None
            spatial_blocked = False

            has_spatial_refs = spatial_ref_open is not None and spatial_ref_closed is not None
            if has_spatial_refs:
                buf = emg_filtered if config.ENABLE_FILTERING else emg_buffer
                _per_tpl_open   = (spatial_ref_open.get("per_template_rms",   None)
                                   if spatial_similarity_mode == "per_template" else None)
                _per_tpl_closed = (spatial_ref_closed.get("per_template_rms", None)
                                   if spatial_similarity_mode == "per_template" else None)
                sim_open_val = compute_spatial_similarity(
                    buf, spatial_ref_open["ref_profile"], spatial_ref_open["weights"],
                    per_template_rms=_per_tpl_open, n_best=spatial_n_best,
                )
                sim_closed_val = compute_spatial_similarity(
                    buf, spatial_ref_closed["ref_profile"], spatial_ref_closed["weights"],
                    per_template_rms=_per_tpl_closed, n_best=spatial_n_best,
                )

            p_closed = None
            D_open_corr = None
            D_closed_corr = None
            if decision_mode == "nn" and decision_nn is not None:
                # --- Neural Network decision ---
                if decision_nn.has_spatial and sim_open_val is not None:
                    nn_features = np.array([D_open, D_closed, sim_open_val, sim_closed_val], dtype=np.float32)
                else:
                    nn_features = np.array([D_open, D_closed], dtype=np.float32)
                p_closed = float(decision_nn.predict(nn_features))
                triggered_state = "CLOSED" if p_closed > 0.5 else "OPEN"
            elif decision_mode == "catboost" and decision_catboost is not None:
                if getattr(decision_catboost, 'transition_mode', False):
                    # --- CatBoost transition-mode: state-conditioned features ---
                    if current_state == "OPEN":
                        sim_t = sim_closed_val  # check CLOSED target
                        p_transition = decision_catboost.predict_transition(D_closed, sim_t)
                        triggered_state = "CLOSED" if p_transition > 0.5 else "OPEN"
                        p_closed = p_transition
                    else:
                        sim_t = sim_open_val  # check OPEN target
                        p_transition = decision_catboost.predict_transition(D_open, sim_t)
                        triggered_state = "OPEN" if p_transition > 0.5 else "CLOSED"
                        p_closed = 1.0 - p_transition  # invert: high = CLOSED maintained
                else:
                    # --- CatBoost posture-classifier (legacy) ---
                    if decision_catboost.has_spatial and sim_open_val is not None:
                        cb_features = np.array([D_open, D_closed, sim_open_val, sim_closed_val], dtype=np.float32)
                    else:
                        cb_features = np.array([D_open, D_closed], dtype=np.float32)
                    p_closed = float(decision_catboost.predict(cb_features))
                    triggered_state = "CLOSED" if p_closed > 0.5 else "OPEN"
            else:
                # --- Threshold decision ---
                if current_state == "OPEN":
                    triggered_state = "CLOSED" if D_closed < threshold_closed else "OPEN"
                else:
                    triggered_state = "OPEN" if D_open < threshold_open else "CLOSED"

                # Spatial correction (threshold mode only)
                if spatial_mode != "off" and has_spatial_refs:
                    k = spatial_sharpness
                    if spatial_mode == "gate":
                        if current_state == "OPEN":
                            if triggered_state == "CLOSED" and sim_closed_val < spatial_threshold:
                                spatial_blocked = True
                                triggered_state = "OPEN"
                        else:
                            if triggered_state == "OPEN" and sim_open_val < spatial_threshold:
                                spatial_blocked = True
                                triggered_state = "CLOSED"
                    elif spatial_mode == "scaling":
                        sim_c = max(sim_closed_val, 0.1)
                        sim_o = max(sim_open_val, 0.1)
                        D_closed_corr = D_closed / (sim_c ** k)
                        D_open_corr   = D_open   / (sim_o ** k)
                        if current_state == "OPEN":
                            triggered_state = "CLOSED" if D_closed_corr < threshold_closed else "OPEN"
                        else:
                            triggered_state = "OPEN" if D_open_corr < threshold_open else "CLOSED"
                    elif spatial_mode == "contrast":
                        sim_c = max(sim_closed_val, 0.1)
                        sim_o = max(sim_open_val, 0.1)
                        D_closed_corr = D_closed * ((sim_o / sim_c) ** k)
                        D_open_corr   = D_open   * ((sim_c / sim_o) ** k)
                        if current_state == "OPEN":
                            triggered_state = "CLOSED" if D_closed_corr < threshold_closed else "OPEN"
                        else:
                            triggered_state = "OPEN" if D_open_corr < threshold_open else "CLOSED"
                    elif spatial_mode in ("relu_scaling", "relu_contrast"):
                        b = spatial_relu_baseline
                        t = max(spatial_threshold, 1e-6)
                        k = max(spatial_sharpness, 0.1)
                        def _f(s): return 1.0 if s >= t else b ** ((1.0 - s / t) ** (1.0 / k))
                        f_c = _f(sim_closed_val)
                        f_o = _f(sim_open_val)
                        if spatial_mode == "relu_scaling":
                            D_closed_corr = D_closed / max(f_c, 0.01)
                            D_open_corr   = D_open   / max(f_o, 0.01)
                        else:  # relu_contrast
                            D_closed_corr = D_closed * f_o / max(f_c, 0.01)
                            D_open_corr   = D_open   * f_c / max(f_o, 0.01)
                        if current_state == "OPEN":
                            triggered_state = "CLOSED" if D_closed_corr < threshold_closed else "OPEN"
                        else:
                            triggered_state = "OPEN" if D_open_corr < threshold_open else "CLOSED"

            sim_open_list.append(sim_open_val)
            sim_closed_list.append(sim_closed_val)
            spatial_blocked_list.append(spatial_blocked)
            ml_prob_list.append(p_closed)
            D_open_corrected_list.append(D_open_corr)
            D_closed_corrected_list.append(D_closed_corr)

            # Refractory period check
            if in_refractory:
                if (current_time_s - last_state_change_time) >= refractory_period_s:
                    in_refractory = False
                    last_predictions = []  # Reset smoothing buffer

            previous_state = current_state

            # Only apply smoothing if not in refractory period
            if not in_refractory:
                if config.POST_PREDICTION_SMOOTHING == "NONE":
                    current_state = triggered_state
                elif config.POST_PREDICTION_SMOOTHING == "MAJORITY VOTE":
                    last_predictions.append(triggered_state)
                    if len(last_predictions) > config.SMOOTHING_WINDOW:
                        last_predictions.pop(0)
                        if last_predictions.count("CLOSED") > last_predictions.count("OPEN"):
                            current_state = "CLOSED"
                        else:
                            current_state = "OPEN"
                        last_predictions = []
                elif config.POST_PREDICTION_SMOOTHING == "5 CONSECUTIVE":
                    last_predictions.append(triggered_state)
                    if len(last_predictions) > config.SMOOTHING_WINDOW:
                        last_predictions.pop(0)
                    if len(last_predictions) >= config.SMOOTHING_WINDOW:
                        if all(p == last_predictions[-1] for p in last_predictions):
                            current_state = last_predictions[-1]

            # Start refractory on state change
            if current_state != previous_state:
                last_state_change_time = current_time_s
                in_refractory = True
                last_predictions = []

            predictions.append(current_state)

            if verbose and dtw_count % 20 == 0:
                spatial_str = ""
                if use_spatial_correction:
                    sim_val = sim_closed_val if sim_closed_val is not None else sim_open_val
                    if sim_val is not None:
                        spatial_str = f" | S={sim_val:.4f}"
                    if spatial_blocked:
                        spatial_str += " [BLOCKED]"
                print(f"  [{dtw_count:4d}] t={end_idx/config.FSAMP:6.2f}s | "
                      f"D_open={D_open:.4f} D_closed={D_closed:.4f} | "
                      f"State={current_state:6s} | DTW time={dtw_time_ms:.2f}ms{spatial_str}")

    # Summary statistics
    avg_dtw_time = total_dtw_time / dtw_count if dtw_count > 0 else 0

    # Reconstruct filtered signal from chunks
    if filtered_signal_chunks:
        filtered_signal_reconstructed = np.concatenate(filtered_signal_chunks, axis=1)
        # Calculate time offset: first chunk starts after buffer is full (1 second)
        filtered_start_time = buffer_length / config.FSAMP
    else:
        filtered_signal_reconstructed = None
        filtered_start_time = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"  Total DTW computations: {dtw_count}")
        print(f"  Average DTW time: {avg_dtw_time:.2f} ms")
        print(f"  Min DTW time: {min(dtw_times):.2f} ms")
        print(f"  Max DTW time: {max(dtw_times):.2f} ms")
        print(f"  Total computation time: {total_dtw_time:.2f} ms")
        print(f"  Using tslearn: {config.USE_TSLEARN_DTW}")
        print(f"  Filtering enabled: {config.ENABLE_FILTERING}")
        if filtered_signal_reconstructed is not None:
            print(f"  Reconstructed filtered signal: {filtered_signal_reconstructed.shape}")
            print(f"    - Starts at: {filtered_start_time:.2f}s (after buffer fill)")
            print(f"    - Duration: {filtered_signal_reconstructed.shape[1]/config.FSAMP:.2f}s")
        print(f"{'='*60}\n")

    return {
        'D_open': np.array(D_open_list),
        'D_closed': np.array(D_closed_list),
        'predictions': predictions,
        'dtw_times': np.array(dtw_times),
        'timestamps': np.array(timestamps),
        'avg_dtw_time': avg_dtw_time,
        'threshold_open': threshold_open,
        'threshold_closed': threshold_closed,
        # Reconstructed filtered signal (only from DTW computation points)
        'filtered_signal': filtered_signal_reconstructed,
        'filtered_start_time': filtered_start_time,
        # Spatial correction data
        'sim_open': sim_open_list,
        'sim_closed': sim_closed_list,
        'spatial_blocked': spatial_blocked_list,
        'spatial_threshold': spatial_threshold if spatial_mode != "off" else None,
        'spatial_mode': spatial_mode,
        # ML probability (p(CLOSED) per tick, None values when threshold mode)
        'ml_probabilities': ml_prob_list,
        # Spatially-corrected distances (scaling/contrast modes only, else None per tick)
        'D_open_corrected': D_open_corrected_list,
        'D_closed_corrected': D_closed_corrected_list,
    }


def apply_state_machine(
    timestamps: np.ndarray,
    D_open: np.ndarray,
    D_closed: np.ndarray,
    threshold_open: float,
    threshold_closed: float,
    refractory_period_s: float = 0.0,
    initial_state: str = "CLOSED",
) -> list:
    """
    Re-run the state machine on pre-computed distances.

    This is cheap (no DTW), so it can be re-run with different smoothing/refractory
    settings without re-computing distances.
    """
    current_state = initial_state
    last_predictions = []
    last_state_change_time = -refractory_period_s
    in_refractory = False
    predictions = []

    for i in range(len(timestamps)):
        t = timestamps[i]

        if current_state == "OPEN":
            triggered_state = "CLOSED" if D_closed[i] < threshold_closed else "OPEN"
        else:
            triggered_state = "OPEN" if D_open[i] < threshold_open else "CLOSED"

        # Refractory check
        if in_refractory:
            if (t - last_state_change_time) >= refractory_period_s:
                in_refractory = False
                last_predictions = []

        previous_state = current_state

        if not in_refractory:
            if config.POST_PREDICTION_SMOOTHING == "NONE":
                current_state = triggered_state
            elif config.POST_PREDICTION_SMOOTHING == "MAJORITY VOTE":
                last_predictions.append(triggered_state)
                if len(last_predictions) > config.SMOOTHING_WINDOW:
                    last_predictions.pop(0)
                    if last_predictions.count("CLOSED") > last_predictions.count("OPEN"):
                        current_state = "CLOSED"
                    else:
                        current_state = "OPEN"
                    last_predictions = []
            elif config.POST_PREDICTION_SMOOTHING == "5 CONSECUTIVE":
                last_predictions.append(triggered_state)
                if len(last_predictions) > config.SMOOTHING_WINDOW:
                    last_predictions.pop(0)
                if len(last_predictions) >= config.SMOOTHING_WINDOW:
                    if all(p == last_predictions[-1] for p in last_predictions):
                        current_state = last_predictions[-1]

        if current_state != previous_state:
            last_state_change_time = t
            in_refractory = True
            last_predictions = []

        predictions.append(current_state)

    return predictions


def _resample_gt_to_emg(gt_data, n_samples):
    """Resample GT data to match EMG sample count using nearest-neighbor."""
    if gt_data is None:
        return None
    gt_1d = gt_data.flatten() if gt_data.ndim > 1 else gt_data
    if len(gt_1d) == n_samples:
        return gt_1d
    if len(gt_1d) == 0:
        return None
    # Nearest-neighbor resampling
    gt_time = np.linspace(0, 1, len(gt_1d))
    emg_time = np.linspace(0, 1, n_samples)
    indices = np.searchsorted(gt_time, emg_time, side='right') - 1
    indices = np.clip(indices, 0, len(gt_1d) - 1)
    return gt_1d[indices]


def _plot_stacked_emg(ax, time_axis, emg_data, gt_resampled=None, time_emg_full=None, title="EMG"):
    """Plot all EMG channels stacked vertically with optional GT shading."""
    n_channels = emg_data.shape[0]
    # Compute offset from median peak-to-peak across channels
    ptps = np.ptp(emg_data, axis=1)
    offset_step = np.median(ptps) * 1.2 if np.median(ptps) > 0 else 1.0

    yticks = []
    ytick_labels = []
    for ch in range(n_channels):
        offset = ch * offset_step
        ax.plot(time_axis, emg_data[ch, :] + offset, linewidth=0.4, alpha=0.8)
        yticks.append(offset)
        ytick_labels.append(f"CH{ch+1}")

    # GT shading over full y range
    if gt_resampled is not None and time_emg_full is not None:
        y_min = -offset_step * 0.5
        y_max = (n_channels - 1) * offset_step + offset_step * 0.5
        ax.fill_between(time_emg_full, y_min, y_max, where=gt_resampled > 0.5,
                        alpha=0.12, color='green', label='GT=CLOSED')

    ax.set_yticks(yticks[::max(1, n_channels // 8)])  # Show subset of labels if many channels
    ax.set_yticklabels([ytick_labels[i] for i in range(0, n_channels, max(1, n_channels // 8))])
    ax.set_ylabel("EMG Channels")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if gt_resampled is not None:
        ax.legend(loc='upper right')


def plot_distance_results(
    emg_data: np.ndarray,
    results: dict,
    title: str = "DTW Distance Analysis",
    save_path: str = None,
    show_filtered: bool = True,
    gt_data: np.ndarray = None,
    time_range: tuple = None,
    gt_mode: str = None,
):
    """
    Plot DTW distances over time with stacked EMG channels.

    Args:
        emg_data: Original EMG data (n_channels, n_samples)
        results: Results dictionary from simulate_realtime_dtw
        title: Plot title
        save_path: Optional path to save figure
        show_filtered: If True, show reconstructed real-time filtered signal
        gt_data: Ground truth data (optional, for overlay on prediction plot)
        time_range: Optional (start_s, end_s) tuple to set xlim on all subplots
        gt_mode: GT data mode ("keyboard", "virtual_hand", "predictions", "none")
    """
    timestamps = results['timestamps']
    D_open = results['D_open']
    D_closed = results['D_closed']
    threshold_open = results['threshold_open']
    threshold_closed = results['threshold_closed']
    predictions = results['predictions']

    # Get reconstructed filtered signal from results
    filtered_signal = results.get('filtered_signal')
    filtered_start_time = results.get('filtered_start_time', 0)

    # Time axis for EMG
    n_channels, n_samples = emg_data.shape
    time_emg = np.arange(n_samples) / config.FSAMP

    # Resample GT to EMG sample count (handles mismatched lengths)
    gt_resampled = _resample_gt_to_emg(gt_data, n_samples)

    # Slice data to time_range to avoid plotting millions of invisible samples
    if time_range is not None:
        s_idx = max(0, int(time_range[0] * config.FSAMP))
        e_idx = min(n_samples, int(time_range[1] * config.FSAMP))
        emg_data = emg_data[:, s_idx:e_idx]
        time_emg = time_emg[s_idx:e_idx]
        if gt_resampled is not None:
            gt_resampled = gt_resampled[s_idx:e_idx]
        n_samples = emg_data.shape[1]
        if filtered_signal is not None:
            fs_start = filtered_start_time
            fs_end = fs_start + filtered_signal.shape[1] / config.FSAMP
            if time_range[0] < fs_end and time_range[1] > fs_start:
                fs_s = max(0, int((time_range[0] - fs_start) * config.FSAMP))
                fs_e = min(filtered_signal.shape[1], int((time_range[1] - fs_start) * config.FSAMP))
                filtered_signal = filtered_signal[:, fs_s:fs_e]
                filtered_start_time = max(time_range[0], fs_start)
            else:
                filtered_signal = None

    # For predictions-mode GT, also create a timestamps-aligned version
    gt_on_timestamps = None
    if gt_data is not None and gt_mode == "predictions":
        gt_1d = gt_data.flatten() if gt_data.ndim > 1 else gt_data
        # GT from predictions file has same count as DTW outputs
        if len(gt_1d) == len(timestamps):
            gt_on_timestamps = gt_1d
        elif len(gt_1d) > 0:
            # Resample to timestamps count
            gt_time = np.linspace(timestamps[0], timestamps[-1], len(gt_1d))
            gt_on_timestamps = np.interp(timestamps, gt_time, gt_1d)

    # Check if we have filtered signal to show (only if filtering is enabled)
    has_filtered = show_filtered and filtered_signal is not None and config.ENABLE_FILTERING

    # Create figure with height ratios: EMG tall, distances medium, prediction compact
    # EMG=4, filtered EMG=4, distance=2, prediction=1
    if has_filtered:
        height_ratios = [4, 4, 2, 2, 1]
    else:
        height_ratios = [4, 2, 2, 1]
    n_plots = len(height_ratios)
    total_height = sum(height_ratios) * 1.5
    fig, axs = plt.subplots(n_plots, 1, figsize=(16, total_height), sharex=True,
                            gridspec_kw={'height_ratios': height_ratios})

    plot_idx = 0

    # Plot 1: Raw EMG (all channels stacked) with GT overlay
    _plot_stacked_emg(
        axs[plot_idx], time_emg, emg_data,
        gt_resampled=gt_resampled, time_emg_full=time_emg,
        title=f"{title} (RAW)"
    )
    plot_idx += 1

    # Plot 2: Reconstructed real-time filtered EMG (if available)
    if has_filtered:
        n_filtered_samples = filtered_signal.shape[1]
        time_filtered = filtered_start_time + np.arange(n_filtered_samples) / config.FSAMP
        # Resample GT for filtered time range
        gt_filtered = None
        if gt_resampled is not None:
            start_idx = int(filtered_start_time * config.FSAMP)
            end_idx = start_idx + n_filtered_samples
            if end_idx <= len(gt_resampled):
                gt_filtered = gt_resampled[start_idx:end_idx]
        _plot_stacked_emg(
            axs[plot_idx], time_filtered, filtered_signal,
            gt_resampled=gt_filtered, time_emg_full=time_filtered,
            title=f"REAL-TIME FILTERED - starts at {filtered_start_time:.1f}s"
        )
        plot_idx += 1

    # Distance to OPEN templates
    ax = axs[plot_idx]
    ax.plot(timestamps, D_open, 'g-', linewidth=1, label='Distance to OPEN')
    # Threshold: time-varying (array) or constant (scalar)
    thresholds_open_ot = results.get('thresholds_open_over_time')
    if thresholds_open_ot is not None and len(thresholds_open_ot) == len(timestamps):
        ax.plot(timestamps, thresholds_open_ot, 'r--', linewidth=1.5, label='Threshold')
    else:
        ax.axhline(threshold_open, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold_open:.3f})')
    ax.set_ylabel("DTW Distance")
    ax.set_title("Distance to OPEN templates")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # Distance to CLOSED templates
    ax = axs[plot_idx]
    ax.plot(timestamps, D_closed, 'orange', linewidth=1, label='Distance to CLOSED')
    thresholds_closed_ot = results.get('thresholds_closed_over_time')
    if thresholds_closed_ot is not None and len(thresholds_closed_ot) == len(timestamps):
        ax.plot(timestamps, thresholds_closed_ot, 'r--', linewidth=1.5, label='Threshold')
    else:
        ax.axhline(threshold_closed, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold_closed:.3f})')
    ax.set_ylabel("DTW Distance")
    ax.set_title("Distance to CLOSED templates")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # Predictions with GT overlay
    ax = axs[plot_idx]
    pred_numeric = [1 if p == "CLOSED" else 0 for p in predictions]
    ax.step(timestamps, pred_numeric, 'purple', linewidth=2, where='post', label='Predicted')
    # Add GT overlay — use timestamps-aligned version if available, else resampled
    if gt_on_timestamps is not None:
        ax.step(timestamps, gt_on_timestamps, 'green', linewidth=1.5, where='post', alpha=0.7, label='Ground Truth')
        ax.legend(loc='upper right')
    elif gt_resampled is not None:
        ax.step(time_emg, gt_resampled, 'green', linewidth=1.5, where='post', alpha=0.7, label='Ground Truth')
        ax.legend(loc='upper right')
    ax.set_ylabel("State")
    ax.set_xlabel("Time (s)")
    ax.set_title("Predicted State vs Ground Truth (0=OPEN, 1=CLOSED)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['OPEN', 'CLOSED'])
    ax.grid(True, alpha=0.3)

    # Apply time range (xlim) if specified
    if time_range is not None:
        for ax in axs:
            ax.set_xlim(time_range[0], time_range[1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_distance_results_segmented(
    emg_data: np.ndarray,
    results: dict,
    title: str = "DTW Distance Analysis",
    save_path: str = None,
    show_filtered: bool = True,
    gt_data: np.ndarray = None,
    gt_mode: str = None,
    segment_duration_s: float = 50,
):
    """
    Plot DTW results, splitting long recordings into multiple figures of segment_duration_s each.

    For short recordings (<= segment_duration_s), produces a single figure.
    For longer recordings, produces one figure per segment with xlim-based windowing.
    """
    n_samples = emg_data.shape[1]
    total_duration = n_samples / config.FSAMP

    if total_duration <= segment_duration_s:
        # Short recording — single plot
        return [plot_distance_results(
            emg_data, results, title=title, save_path=save_path,
            show_filtered=show_filtered, gt_data=gt_data, gt_mode=gt_mode,
        )]

    # Long recording — split into segments
    figs = []
    n_segments = int(np.ceil(total_duration / segment_duration_s))
    save_base = Path(save_path).stem if save_path else None
    save_dir = Path(save_path).parent if save_path else None

    for seg_idx in range(n_segments):
        t_start = seg_idx * segment_duration_s
        t_end = min((seg_idx + 1) * segment_duration_s, total_duration)
        seg_title = f"{title} [{t_start:.0f}s - {t_end:.0f}s]"

        seg_save = None
        if save_path:
            seg_save = str(save_dir / f"{save_base}_seg{seg_idx+1}.png")

        fig = plot_distance_results(
            emg_data, results, title=seg_title, save_path=seg_save,
            show_filtered=show_filtered, gt_data=gt_data, gt_mode=gt_mode,
            time_range=(t_start, t_end),
        )
        figs.append(fig)

    print(f"  Generated {n_segments} segment plots ({segment_duration_s}s each)")
    return figs


def run_feature_comparison(
    emg_data: np.ndarray,
    raw_templates_open: list,
    raw_templates_closed: list,
    feature_names: list = None,
    distance_aggregation: str = 'average',
    title: str = "Feature Comparison",
    save_dir: str = None,
    gt_data: np.ndarray = None,
):
    """
    Run offline test with every specified feature and compare results side by side.

    Uses raw EMG templates (n_channels, n_samples) and re-extracts features for each
    feature type. Computes intra-class thresholds (mean + 1*std) for each feature.

    Args:
        emg_data: EMG recording (n_channels, n_samples)
        raw_templates_open: List of raw EMG templates for OPEN (each: n_channels, 2000)
        raw_templates_closed: List of raw EMG templates for CLOSED (each: n_channels, 2000)
        feature_names: List of feature keys to test (default: all time-domain features)
        distance_aggregation: Distance aggregation method
        title: Plot title prefix
        save_dir: Directory to save plots (optional)
        gt_data: Ground truth array (optional)

    Returns:
        Dictionary mapping feature_name -> results dict
    """
    from mindmove.model.core.algorithm import compute_threshold

    if feature_names is None:
        feature_names = list(FEATURES.keys())

    window_length = config.window_length
    increment = config.increment

    all_feature_results = {}

    for feat_name in feature_names:
        print(f"\n{'='*70}")
        print(f"Feature: {feat_name} ({FEATURES[feat_name]['name']})")
        print(f"{'='*70}")

        feat_fn = FEATURES[feat_name]["function"]

        # Extract features from raw templates
        feat_templates_open = []
        for raw_t in raw_templates_open:
            windowed = sliding_window(raw_t, window_length, increment)
            feat_templates_open.append(feat_fn(windowed))

        feat_templates_closed = []
        for raw_t in raw_templates_closed:
            windowed = sliding_window(raw_t, window_length, increment)
            feat_templates_closed.append(feat_fn(windowed))

        print(f"  Template shape: {feat_templates_open[0].shape}")
        print(f"  Templates: {len(feat_templates_open)} OPEN, {len(feat_templates_closed)} CLOSED")

        # Compute intra-class thresholds (mean + 1*std)
        mean_open, std_open, threshold_open = compute_threshold(
            feat_templates_open, s=1, verbose=False
        )
        mean_closed, std_closed, threshold_closed = compute_threshold(
            feat_templates_closed, s=1, verbose=False
        )
        print(f"  Threshold OPEN: {threshold_open:.4f} (mean={mean_open:.4f}, std={std_open:.4f})")
        print(f"  Threshold CLOSED: {threshold_closed:.4f} (mean={mean_closed:.4f}, std={std_closed:.4f})")

        # Run simulation
        results = simulate_realtime_dtw(
            emg_data, feat_templates_open, feat_templates_closed,
            threshold_open, threshold_closed, feat_name,
            verbose=False,
            distance_aggregation=distance_aggregation,
        )

        # Stats
        preds = results['predictions']
        n_open = sum(1 for p in preds if p == 'OPEN')
        n_closed = sum(1 for p in preds if p == 'CLOSED')
        transitions = sum(1 for j in range(1, len(preds)) if preds[j] != preds[j-1])

        print(f"  Predictions: {n_open} OPEN ({100*n_open/len(preds):.1f}%), "
              f"{n_closed} CLOSED ({100*n_closed/len(preds):.1f}%)")
        print(f"  Transitions: {transitions}")

        all_feature_results[feat_name] = {
            'results': results,
            'threshold_open': threshold_open,
            'threshold_closed': threshold_closed,
            'mean_open': mean_open,
            'std_open': std_open,
            'mean_closed': mean_closed,
            'std_closed': std_closed,
            'n_open': n_open,
            'n_closed': n_closed,
            'transitions': transitions,
        }

        # Plot for each feature
        plot_title = f"{title} | {FEATURES[feat_name]['name']} ({feat_name})"
        save_path = None
        if save_dir:
            save_path = str(Path(save_dir) / f"feature_{feat_name}.png")

        plot_distance_results_segmented(
            emg_data, results, title=plot_title,
            save_path=save_path,
            show_filtered=False,
            gt_data=gt_data,
            segment_duration_s=SEGMENT_DURATION_S,
        )

    # Print summary table
    print(f"\n{'='*90}")
    print(f"FEATURE COMPARISON SUMMARY")
    print(f"{'='*90}")
    print(f"{'Feature':<20} {'Th_open':>8} {'Th_closed':>10} {'%OPEN':>7} {'%CLOSED':>8} {'Trans':>6}")
    print(f"{'-'*90}")

    if gt_data is not None:
        gt_1d = gt_data.flatten() if gt_data.ndim > 1 else gt_data
        gt_open_pct = 100 * np.sum(gt_1d < 0.5) / len(gt_1d)
        print(f"{'GT':<20} {'':>8} {'':>10} {gt_open_pct:>6.1f}% {100-gt_open_pct:>7.1f}%")
        print(f"{'-'*90}")

    for feat_name, fdata in all_feature_results.items():
        total = fdata['n_open'] + fdata['n_closed']
        pct_open = 100 * fdata['n_open'] / total
        pct_closed = 100 * fdata['n_closed'] / total
        print(f"{feat_name:<20} {fdata['threshold_open']:>8.3f} {fdata['threshold_closed']:>10.3f} "
              f"{pct_open:>6.1f}% {pct_closed:>7.1f}% {fdata['transitions']:>6}")
    print(f"{'='*90}")

    return all_feature_results


def compare_dtw_implementations(
    emg_data: np.ndarray,
    templates_open: list,
    templates_closed: list,
    threshold_open: float,
    threshold_closed: float,
    feature_name: str = 'wl',
    include_gpudtw: bool = False
) -> dict:
    """
    Compare DTW implementations performance.

    Args:
        include_gpudtw: If True and GPUDTW is available, include GPU comparison.

    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*70)
    print("COMPARING DTW IMPLEMENTATIONS")
    print("="*70)

    results = {}
    test_num = 1
    total_tests = 2 + (1 if include_gpudtw and GPUDTW_AVAILABLE else 0)

    # Test with numba (fastest CPU)
    config.USE_NUMBA_DTW = True
    config.USE_TSLEARN_DTW = False
    config.USE_GPUDTW = False
    print(f"\n[{test_num}/{total_tests}] Testing with NUMBA DTW (JIT-compiled, cosine)...")
    results['numba'] = simulate_realtime_dtw(
        emg_data, templates_open, templates_closed,
        threshold_open, threshold_closed, feature_name,
        verbose=True
    )
    test_num += 1

    # Test with original implementation
    config.USE_NUMBA_DTW = False
    config.USE_TSLEARN_DTW = False
    config.USE_GPUDTW = False
    print(f"\n[{test_num}/{total_tests}] Testing with ORIGINAL DTW (pure Python, cosine)...")
    results['original'] = simulate_realtime_dtw(
        emg_data, templates_open, templates_closed,
        threshold_open, threshold_closed, feature_name,
        verbose=True
    )
    test_num += 1

    # Test with GPUDTW if requested and available
    if include_gpudtw:
        if GPUDTW_AVAILABLE:
            config.USE_NUMBA_DTW = False
            config.USE_TSLEARN_DTW = False
            config.USE_GPUDTW = True
            print(f"\n[{test_num}/{total_tests}] Testing with GPUDTW (GPU-accelerated, Euclidean)...")
            results['gpudtw'] = simulate_realtime_dtw(
                emg_data, templates_open, templates_closed,
                threshold_open, threshold_closed, feature_name,
                verbose=True
            )
        else:
            print("\n[SKIPPED] GPUDTW not available (install with: pip install GPUDTW)")

    # Comparison summary
    speedup_numba = results['original']['avg_dtw_time'] / results['numba']['avg_dtw_time']

    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"  Original DTW avg time: {results['original']['avg_dtw_time']:.2f} ms")
    print(f"  Numba DTW avg time:    {results['numba']['avg_dtw_time']:.2f} ms")
    print(f"  Numba speedup:         {speedup_numba:.1f}x vs Original")

    if 'gpudtw' in results:
        speedup_gpu = results['original']['avg_dtw_time'] / results['gpudtw']['avg_dtw_time']
        print(f"  GPUDTW avg time:       {results['gpudtw']['avg_dtw_time']:.2f} ms")
        print(f"  GPUDTW speedup:        {speedup_gpu:.1f}x vs Original")
        results['speedup_gpu'] = speedup_gpu

    print("="*70 + "\n")

    # Reset to numba (fastest CPU)
    config.USE_NUMBA_DTW = True
    config.USE_GPUDTW = False

    results['speedup_numba'] = speedup_numba
    return results


def run_offline_test_single(
    emg_data: np.ndarray,
    recording_path: Path,
    templates_open: list,
    templates_closed: list,
    threshold_open: float,
    threshold_closed: float,
    feature_name: str,
    save_dir: Path,
    verbose: bool = True,
    gt_data: np.ndarray = None,
    distance_aggregation: str = 'average',
) -> dict:
    """
    Run offline test on a single recording.
    """
    folder_name = recording_path.parent.name
    recording_name = recording_path.stem

    if verbose:
        print(f"\n{'='*70}")
        print(f"Processing: {folder_name}/{recording_name}")
        print(f"{'='*70}")

    # Run simulation
    results = simulate_realtime_dtw(
        emg_data, templates_open, templates_closed,
        threshold_open, threshold_closed, feature_name,
        verbose=verbose,
        distance_aggregation=distance_aggregation,
    )

    # Plot and save
    plot_title = f"Folder: {folder_name} | {recording_name}"
    save_path = save_dir / f"{folder_name}_{recording_name}.png"

    plot_distance_results_segmented(
        emg_data, results, title=plot_title,
        save_path=str(save_path),
        gt_data=gt_data,
        segment_duration_s=SEGMENT_DURATION_S,
    )

    return results


def run_offline_test_folder(
    test_folder: str,
    model_path: str = None,
    show_plot: bool = False
):
    """
    Run offline DTW testing on ALL recordings in a folder.

    Args:
        test_folder: Name of folder inside data/tests/ (e.g., "test", "test open")
        model_path: Path to trained model (default: data/models/dtw_model_0.pkl)
        show_plot: If True, display plots (blocks execution)
    """
    base_path = Path(__file__).parent.parent.parent  # mindmove-framework-main

    # Model path
    if model_path is None:
        model_path = base_path / "data" / "models" / "dtw_model_0.pkl"

    # Test folder path - check multiple locations
    test_dir = base_path / "data" / "tests" / test_folder
    if not test_dir.exists():
        # Also check data/recordings/ directly
        test_dir = base_path / "data" / test_folder
        if not test_dir.exists():
            raise FileNotFoundError(f"Test folder not found in data/tests/{test_folder} or data/{test_folder}")

    # Find all recordings (.pkl and .mat files)
    test_files = sorted(list(test_dir.glob("*.pkl")) + list(test_dir.glob("*.mat")))
    if not test_files:
        raise FileNotFoundError(f"No .pkl or .mat recordings found in {test_dir}")

    print("="*70)
    print("OFFLINE DTW MODEL TEST - FOLDER MODE")
    print("="*70)
    print(f"Test folder: {test_folder}")
    print(f"Recordings found: {len(test_files)}")
    print(f"Model: {model_path}")
    print("="*70)

    # Load model
    model_data = load_model(str(model_path))
    templates_open = model_data['open_templates']
    templates_closed = model_data['closed_templates']
    threshold_open = model_data['threshold_base_open'] #* 2
    threshold_closed = model_data['threshold_base_closed'] #* 2
    feature_name = model_data['feature_name']
    distance_aggregation = model_data.get('distance_aggregation', 'average')

    # Apply smoothing from model if available
    smoothing = model_data.get('smoothing_method', None)
    if smoothing:
        config.POST_PREDICTION_SMOOTHING = smoothing

    print(f"\nModel loaded:")
    print(f"  Open templates: {len(templates_open)}")
    print(f"  Closed templates: {len(templates_closed)}")
    print(f"  Threshold OPEN: {threshold_open:.4f}")
    print(f"  Threshold CLOSED: {threshold_closed:.4f}")
    print(f"  Feature: {feature_name}")
    print(f"  Distance aggregation: {distance_aggregation}")
    print(f"  Smoothing: {config.POST_PREDICTION_SMOOTHING}")

    # Create output directory
    save_dir = base_path / "data" / "predictions" / test_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPlots will be saved to: {save_dir}")

    # Process each recording
    all_results = []
    total_avg_time = 0

    for i, recording_path in enumerate(test_files):
        print(f"\n[{i+1}/{len(test_files)}] Loading {recording_path.name}...")

        try:
            emg_data, data = load_test_recording(str(recording_path))

            # Extract GT data if available
            gt_data = data.get('_gt_data', None)

            results = run_offline_test_single(
                emg_data, recording_path,
                templates_open, templates_closed,
                threshold_open, threshold_closed,
                feature_name, save_dir,
                verbose=True,
                gt_data=gt_data,
                distance_aggregation=distance_aggregation,
            )

            all_results.append({
                'recording': recording_path.name,
                'avg_dtw_time': results['avg_dtw_time'],
                'results': results
            })
            total_avg_time += results['avg_dtw_time']

        except Exception as e:
            print(f"  ERROR processing {recording_path.name}: {e}")

    # Final summary
    print("\n" + "="*70)
    print("FOLDER PROCESSING COMPLETE")
    print("="*70)
    print(f"  Recordings processed: {len(all_results)}/{len(test_files)}")
    if all_results:
        print(f"  Average DTW time across all: {total_avg_time/len(all_results):.2f} ms")
    print(f"  Plots saved to: {save_dir}")
    print("="*70)

    if show_plot:
        plt.show()

    return all_results


def build_results_from_prediction(data: dict) -> dict:
    """
    Build a results dict from a saved prediction .pkl file for plotting.

    For new recordings (made with updated code), per-step thresholds are already
    stored in distance_history. For older recordings, reconstructs them from
    unity_output (context-dependent: CLOSED state → OPEN threshold, vice versa).

    Returns dict compatible with plot_distance_results().
    """
    history = data['distance_history']

    hist_timestamps = np.array(history['timestamps'])
    D_open = np.array(history['D_open'])
    D_closed = np.array(history['D_closed'])
    states = history['states']
    scalar_th_open = history['threshold_open']
    scalar_th_closed = history['threshold_closed']

    if len(hist_timestamps) == 0:
        return None

    # Normalize history timestamps to start from 0
    hist_t0 = hist_timestamps[0]
    hist_timestamps = hist_timestamps - hist_t0

    # Check if per-step thresholds are already saved (new format)
    if 'thresholds_open_over_time' in history and history['thresholds_open_over_time'] is not None:
        thresholds_open = np.array(history['thresholds_open_over_time'])
        thresholds_closed = np.array(history['thresholds_closed_over_time'])
        print("  Using saved per-step thresholds from distance_history.")
    elif 'unity_output' in data and data['unity_output']:
        # Reconstruct from unity_output (old format)
        print("  Reconstructing per-step thresholds from unity_output...")
        unity_output = data['unity_output']
        uo_times = np.array([e['timestamp'] for e in unity_output])
        uo_states = np.array([e['state'] for e in unity_output])
        uo_thresholds = np.array([e['threshold'] for e in unity_output])

        # Normalize unity timestamps to same epoch
        uo_t0 = uo_times[0]
        uo_times = uo_times - uo_t0

        # When state==1.0 (CLOSED), threshold is for OPEN class
        # When state==0.0 (OPEN), threshold is for CLOSED class
        thresholds_open = np.full(len(hist_timestamps), scalar_th_open)
        thresholds_closed = np.full(len(hist_timestamps), scalar_th_closed)

        for i, t in enumerate(hist_timestamps):
            idx = np.searchsorted(uo_times, t)
            idx = min(idx, len(uo_times) - 1)
            if idx > 0 and abs(uo_times[idx - 1] - t) < abs(uo_times[idx] - t):
                idx = idx - 1

            if uo_states[idx] == 1.0:  # CLOSED → threshold is for OPEN
                thresholds_open[i] = uo_thresholds[idx]
            else:  # OPEN → threshold is for CLOSED
                thresholds_closed[i] = uo_thresholds[idx]

        # Forward-fill gaps
        last_open = scalar_th_open
        last_closed = scalar_th_closed
        for i in range(len(hist_timestamps)):
            if thresholds_open[i] != scalar_th_open:
                last_open = thresholds_open[i]
            else:
                thresholds_open[i] = last_open
            if thresholds_closed[i] != scalar_th_closed:
                last_closed = thresholds_closed[i]
            else:
                thresholds_closed[i] = last_closed
    else:
        # No per-step data available, use scalar thresholds
        thresholds_open = None
        thresholds_closed = None

    predictions = ["CLOSED" if s == "CLOSED" or s == 1.0 else "OPEN" for s in states]

    result = {
        'timestamps': hist_timestamps,
        'D_open': D_open,
        'D_closed': D_closed,
        'predictions': predictions,
        'threshold_open': scalar_th_open,
        'threshold_closed': scalar_th_closed,
    }
    if thresholds_open is not None:
        result['thresholds_open_over_time'] = thresholds_open
        result['thresholds_closed_over_time'] = thresholds_closed
    return result


# =============================================================================
# CONFIGURATION - EDIT THIS SECTION TO RUN YOUR TESTS
# =============================================================================

# --- Mode selection ---
# "folder"   : run offline test on all recordings in a folder using a trained model
# "features" : compare all time-domain features on a single recording
# "single"   : run a single feature with manual DTW thresholds
# "sweep"    : parameter sweep — try many combinations, output metrics table
# "replay"   : load a saved prediction .pkl and plot with actual thresholds used
MODE = "replay"

# --- Common settings ---
USE_NUMBA = True
USE_TSLEARN = False
USE_GPUDTW = False

# --- Folder mode settings ---
TEST_FOLDER = "aperture"
MODEL_FILE = "MindMove_Model_20260121_133757_test_average_5_no_outliers.pkl"

# --- Feature comparison / single feature mode settings ---
# Patient S1 data paths (relative to project root)
PATIENT_DIR = "data/recordings/patient S1"
# TEMPLATES_FILE = "templates_sd_20260206_121311_guided_4cycles.pkl"
TEMPLATES_FILE = "templates_mp_20260212_105426_guided_16cycles.pkl"
# Recording to test on (pick one):
#   guided_1cycles  = unseen (never used for templates)
#   guided_16cycles = 16 closing cycles
#   guided_4cycles  = 4 grasping cycles (template source)
# TEST_RECORDING = "MindMove_GuidedRecording_sd_20260206_120739441579_guided_4cycles.pkl"
# TEST_RECORDING = "MindMove_GuidedRecording_sd_20260206_120536761870_guided_16cycles.pkl"
# TEST_RECORDING = "MindMove_GuidedRecording_sd_20260206_115904408711_guided_1cycles.pkl"
TEST_RECORDING = "MindMove_Predictions_20260206_123131324533_None.pkl"



# Distance aggregation: "average", "minimum", "avg_3_smallest"
DISTANCE_AGGREGATION = "avg_3_smallest"
# DISTANCE_AGGREGATION = "minimum"

# Post-prediction smoothing: "NONE", "MAJORITY VOTE", "5 CONSECUTIVE"
SMOOTHING = "5 CONSECUTIVE"
SMOOTHING_WINDOW = 5

# --- Dead channels (0-indexed, range 0-15 for differential mode) ---
# Channels to exclude from DTW computation. Set to [] to use all.
# Example: [13, 14, 15] excludes CH14, CH15, CH16 (1-indexed)
# Example: [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15] keeps only CH11 (0-indexed=10)
DEAD_CHANNELS = []

# --- Sweep mode settings ---
# Each list entry is tried in combination with the others (grid search)
SWEEP_FEATURES = ["rms", "wl", "mav"]
SWEEP_AGGREGATIONS = ["avg_3_smallest", "average", "minimum"]
SWEEP_SMOOTHINGS = [("5 CONSECUTIVE", 5), ("MAJORITY VOTE", 5), ("NONE", 1)]
# Threshold multipliers (applied to intra-class std: threshold = mean + s * std)
SWEEP_THRESHOLD_S = [0.5, 1.0, 1.5, 2.0]
# Set to True to also generate plots for the top N results
SWEEP_PLOT_TOP_N = 3

# --- Plot segmentation ---
# Split long recordings into segments of this duration (seconds) for plotting
SEGMENT_DURATION_S = 100
# Time range to analyze: None = auto-prompt if >60s, or tuple (start_s, end_s)
TIME_RANGE_S = None

# --- Single feature mode settings ---
# Feature to use (from features_registry: rms, mav, wl, wamp, iemg, ssi, etc.)
SINGLE_FEATURE = "rms"
# DTW thresholds: set to None for automatic (intra-class mean + 1*std), or a manual value
THRESHOLD_OPEN = None    # e.g. 2.5
THRESHOLD_CLOSED = None  # e.g. 2.5

# --- Replay mode settings ---
# Path to a saved prediction .pkl file (relative to PATIENT_DIR)
PREDICTION_FILE = "MindMove_Predictions_20260206_123131324533_None.pkl"
# First model (cached as baseline). Results are cached after first run.
REPLAY_MODEL_FILE = "MindMove_Model_sd_20260206_121332_test_patient.pkl"
# Second model for comparison: set to a templates file (raw EMG templates)
# or None to just show the first model. Uses SINGLE_FEATURE and DISTANCE_AGGREGATION.
REPLAY_COMPARE_TEMPLATES = "templates_mp_20260212_105426_guided_16cycles.pkl"
# REPLAY_COMPARE_TEMPLATES = None

# --- Refractory period ---
# Minimum time (seconds) between state changes, matching the online model.
# Set to 0 to disable.
REFRACTORY_PERIOD_S = 1.0

# =============================================================================


if __name__ == "__main__":
    import sys
    from pathlib import Path

    base_path = Path(__file__).parent.parent.parent

    # Apply DTW configuration
    config.USE_NUMBA_DTW = USE_NUMBA
    config.USE_TSLEARN_DTW = USE_TSLEARN
    config.USE_GPUDTW = USE_GPUDTW
    config.POST_PREDICTION_SMOOTHING = SMOOTHING
    config.SMOOTHING_WINDOW = SMOOTHING_WINDOW
    config.ENABLE_FILTERING = False  # Recordings are already filtered

    # Override dead channels from config.py (don't touch config.py)
    config.dead_channels = DEAD_CHANNELS

    show = "--no-show" not in sys.argv

    def apply_dead_channels():
        """Recompute active_channels from DEAD_CHANNELS after num_channels is set."""
        config.dead_channels = DEAD_CHANNELS
        config.active_channels = [i for i in range(config.num_channels) if i not in DEAD_CHANNELS]
        if DEAD_CHANNELS:
            print(f"  Dead channels (0-indexed): {DEAD_CHANNELS}")
            print(f"  Active channels: {len(config.active_channels)}/{config.num_channels} -> {config.active_channels}")

    if MODE == "folder":
        model_path = base_path / "data" / "models" / MODEL_FILE
        print(f"\n  Mode: Folder test")
        print(f"  Test folder: {TEST_FOLDER}")
        print(f"  Model: {MODEL_FILE}")
        apply_dead_channels()
        run_offline_test_folder(TEST_FOLDER, model_path=str(model_path), show_plot=show)

    elif MODE in ("features", "single"):
        patient_dir = base_path / PATIENT_DIR
        templates_path = patient_dir / TEMPLATES_FILE
        recording_path = patient_dir / TEST_RECORDING

        print("\n" + "="*70)
        print(f"{'FEATURE COMPARISON' if MODE == 'features' else 'SINGLE FEATURE'} MODE")
        print("="*70)
        print(f"  Recording: {TEST_RECORDING}")
        print(f"  Templates: {TEMPLATES_FILE}")
        print(f"  Distance aggregation: {DISTANCE_AGGREGATION}")
        print(f"  Smoothing: {SMOOTHING} (window={SMOOTHING_WINDOW})")
        if MODE == "single":
            print(f"  Feature: {SINGLE_FEATURE}")
            print(f"  Threshold OPEN: {THRESHOLD_OPEN if THRESHOLD_OPEN is not None else 'auto'}")
            print(f"  Threshold CLOSED: {THRESHOLD_CLOSED if THRESHOLD_CLOSED is not None else 'auto'}")
        print("="*70)

        # Load raw templates
        with open(templates_path, 'rb') as f:
            templates_data = pickle.load(f)
        raw_templates_open = templates_data['templates_open']
        raw_templates_closed = templates_data['templates_closed']
        print(f"\n  Raw templates: {len(raw_templates_open)} OPEN, {len(raw_templates_closed)} CLOSED")
        print(f"  Template shape: {raw_templates_open[0].shape}")

        # Set differential mode from templates (check top-level, metadata dict, or infer from shape)
        is_diff = (
            templates_data.get('differential_mode', False)
            or templates_data.get('metadata', {}).get('differential_mode', False)
            or raw_templates_open[0].shape[0] <= 16  # infer from template shape (n_channels, n_samples)
        )
        if is_diff:
            config.ENABLE_DIFFERENTIAL_MODE = True
            config.num_channels = 16
            print(f"  Differential mode detected (16 channels)")

        # Apply dead channels (must be after num_channels is set)
        config.dead_channels = DEAD_CHANNELS
        config.active_channels = [i for i in range(config.num_channels) if i not in DEAD_CHANNELS]
        if DEAD_CHANNELS:
            print(f"  Dead channels (0-indexed): {DEAD_CHANNELS}")
            print(f"  Active channels: {len(config.active_channels)}/{config.num_channels} -> {config.active_channels}")

        # Load recording
        emg_data, data = load_test_recording(str(recording_path))
        gt_data = data.get('_gt_data', None)

        # Time range selection for long recordings
        duration_s = emg_data.shape[1] / config.FSAMP
        if TIME_RANGE_S is not None:
            # Use configured time range
            t_start, t_end = TIME_RANGE_S
            t_start = max(0, t_start)
            t_end = min(duration_s, t_end)
            start_idx = int(t_start * config.FSAMP)
            end_idx = int(t_end * config.FSAMP)
            emg_data = emg_data[:, start_idx:end_idx]
            if gt_data is not None:
                gt_1d = gt_data.flatten() if gt_data.ndim > 1 else gt_data
                if len(gt_1d) == data['emg'].shape[1]:
                    gt_data = gt_1d[start_idx:end_idx]
                else:
                    # Resample GT indices proportionally
                    gt_start = int(t_start / duration_s * len(gt_1d))
                    gt_end = int(t_end / duration_s * len(gt_1d))
                    gt_data = gt_1d[gt_start:gt_end]
            print(f"\n  Cropped to time range: {t_start:.1f}s - {t_end:.1f}s ({emg_data.shape[1]} samples)")
        elif duration_s > 60:
            # Prompt user for time range
            print(f"\n  Recording is {duration_s:.1f}s long ({duration_s/60:.1f} min).")
            print(f"  Enter time range as 'start-end' in seconds (e.g. '0-120'),")
            user_input = input(f"  or press Enter for full recording: ").strip()
            if user_input:
                try:
                    parts = user_input.split('-')
                    t_start = max(0, float(parts[0]))
                    t_end = min(duration_s, float(parts[1]))
                    start_idx = int(t_start * config.FSAMP)
                    end_idx = int(t_end * config.FSAMP)
                    emg_data = emg_data[:, start_idx:end_idx]
                    if gt_data is not None:
                        gt_1d = gt_data.flatten() if gt_data.ndim > 1 else gt_data
                        if len(gt_1d) == data['emg'].shape[1]:
                            gt_data = gt_1d[start_idx:end_idx]
                        else:
                            gt_start = int(t_start / duration_s * len(gt_1d))
                            gt_end = int(t_end / duration_s * len(gt_1d))
                            gt_data = gt_1d[gt_start:gt_end]
                    print(f"  Cropped to: {t_start:.1f}s - {t_end:.1f}s ({emg_data.shape[1]} samples)")
                except (ValueError, IndexError):
                    print(f"  Invalid input, using full recording.")

        # Output directory
        save_dir = base_path / "data" / "predictions" / "feature_comparison"
        save_dir.mkdir(parents=True, exist_ok=True)

        if MODE == "features":
            # Run comparison across all features
            results = run_feature_comparison(
                emg_data, raw_templates_open, raw_templates_closed,
                distance_aggregation=DISTANCE_AGGREGATION,
                title=f"Patient S1 - {Path(TEST_RECORDING).stem}",
                save_dir=str(save_dir),
                gt_data=gt_data,
            )

        elif MODE == "single":
            from mindmove.model.core.algorithm import compute_threshold

            feat_fn = FEATURES[SINGLE_FEATURE]["function"]
            feat_display = FEATURES[SINGLE_FEATURE]["name"]
            window_length = config.window_length
            increment = config.increment

            # Extract features from raw templates
            feat_templates_open = []
            for raw_t in raw_templates_open:
                windowed = sliding_window(raw_t, window_length, increment)
                feat_templates_open.append(feat_fn(windowed))

            feat_templates_closed = []
            for raw_t in raw_templates_closed:
                windowed = sliding_window(raw_t, window_length, increment)
                feat_templates_closed.append(feat_fn(windowed))

            print(f"\n  Feature template shape: {feat_templates_open[0].shape}")

            # Compute or use manual thresholds
            mean_open, std_open, auto_th_open = compute_threshold(
                feat_templates_open, s=1, verbose=False
            )
            mean_closed, std_closed, auto_th_closed = compute_threshold(
                feat_templates_closed, s=1, verbose=False
            )

            th_open = THRESHOLD_OPEN if THRESHOLD_OPEN is not None else auto_th_open
            th_closed = THRESHOLD_CLOSED if THRESHOLD_CLOSED is not None else auto_th_closed

            print(f"  Auto thresholds:   OPEN={auto_th_open:.4f} (mean={mean_open:.4f}, std={std_open:.4f})")
            print(f"                     CLOSED={auto_th_closed:.4f} (mean={mean_closed:.4f}, std={std_closed:.4f})")
            print(f"  Using thresholds:  OPEN={th_open:.4f}, CLOSED={th_closed:.4f}")

            # Run simulation
            results = simulate_realtime_dtw(
                emg_data, feat_templates_open, feat_templates_closed,
                th_open, th_closed, SINGLE_FEATURE,
                verbose=True,
                distance_aggregation=DISTANCE_AGGREGATION,
            )

            # Plot
            save_path = str(save_dir / f"single_{SINGLE_FEATURE}.png")
            gt_mode = data.get('_gt_mode', None)
            plot_distance_results_segmented(
                emg_data, results,
                title=f"Patient S1 | {feat_display} ({SINGLE_FEATURE}) | th_open={th_open:.3f} th_closed={th_closed:.3f}",
                save_path=save_path,
                show_filtered=False,
                gt_data=gt_data,
                gt_mode=gt_mode,
                segment_duration_s=SEGMENT_DURATION_S,
            )

            # Metrics
            gt_mode = data.get('_gt_mode', None)
            metrics = compute_metrics(results, gt_data, gt_mode=gt_mode)
            print_metrics(metrics)

        if show:
            plt.show()

    elif MODE == "sweep":
        from mindmove.model.core.algorithm import compute_threshold
        from itertools import product

        patient_dir = base_path / PATIENT_DIR
        templates_path = patient_dir / TEMPLATES_FILE
        recording_path = patient_dir / TEST_RECORDING

        print("\n" + "="*70)
        print("PARAMETER SWEEP MODE")
        print("="*70)
        print(f"  Recording: {TEST_RECORDING}")
        print(f"  Templates: {TEMPLATES_FILE}")
        print(f"  Features:     {SWEEP_FEATURES}")
        print(f"  Aggregations: {SWEEP_AGGREGATIONS}")
        print(f"  Smoothings:   {[s[0] for s in SWEEP_SMOOTHINGS]}")
        print(f"  Threshold s:  {SWEEP_THRESHOLD_S}")

        # Load templates
        with open(templates_path, 'rb') as f:
            templates_data = pickle.load(f)
        raw_templates_open = templates_data['templates_open']
        raw_templates_closed = templates_data['templates_closed']

        # Set differential mode
        is_diff = (
            templates_data.get('differential_mode', False)
            or templates_data.get('metadata', {}).get('differential_mode', False)
            or raw_templates_open[0].shape[0] <= 16
        )
        if is_diff:
            config.ENABLE_DIFFERENTIAL_MODE = True
            config.num_channels = 16
        config.dead_channels = DEAD_CHANNELS
        config.active_channels = [i for i in range(config.num_channels) if i not in DEAD_CHANNELS]

        # Load recording
        emg_data, data = load_test_recording(str(recording_path))
        gt_data = data.get('_gt_data', None)
        gt_mode_str = data.get('_gt_mode', None)

        # Time range selection (same logic as single mode)
        duration_s = emg_data.shape[1] / config.FSAMP
        if TIME_RANGE_S is not None:
            t_start, t_end = TIME_RANGE_S
            t_start = max(0, t_start)
            t_end = min(duration_s, t_end)
            s_idx = int(t_start * config.FSAMP)
            e_idx = int(t_end * config.FSAMP)
            emg_data = emg_data[:, s_idx:e_idx]
            if gt_data is not None:
                gt_1d = gt_data.flatten() if gt_data.ndim > 1 else gt_data
                if len(gt_1d) == data['emg'].shape[1]:
                    gt_data = gt_1d[s_idx:e_idx]
                else:
                    gt_data = gt_1d[int(t_start/duration_s*len(gt_1d)):int(t_end/duration_s*len(gt_1d))]
            print(f"  Cropped to: {t_start:.1f}s - {t_end:.1f}s")
        elif duration_s > 60:
            print(f"\n  Recording is {duration_s:.1f}s long ({duration_s/60:.1f} min).")
            print(f"  Enter time range as 'start-end' in seconds (e.g. '0-120'),")
            user_input = input(f"  or press Enter for full recording: ").strip()
            if user_input:
                try:
                    parts = user_input.split('-')
                    t_start, t_end = max(0, float(parts[0])), min(duration_s, float(parts[1]))
                    s_idx, e_idx = int(t_start * config.FSAMP), int(t_end * config.FSAMP)
                    emg_data = emg_data[:, s_idx:e_idx]
                    if gt_data is not None:
                        gt_1d = gt_data.flatten() if gt_data.ndim > 1 else gt_data
                        if len(gt_1d) == data['emg'].shape[1]:
                            gt_data = gt_1d[s_idx:e_idx]
                        else:
                            gt_data = gt_1d[int(t_start/duration_s*len(gt_1d)):int(t_end/duration_s*len(gt_1d))]
                    print(f"  Cropped to: {t_start:.1f}s - {t_end:.1f}s")
                except (ValueError, IndexError):
                    print(f"  Invalid input, using full recording.")

        window_length = config.window_length
        increment = config.increment

        # Pre-extract features for each feature type (avoid re-computing)
        feature_templates = {}
        for feat_name in SWEEP_FEATURES:
            feat_fn = FEATURES[feat_name]["function"]
            ft_open = [feat_fn(sliding_window(t, window_length, increment)) for t in raw_templates_open]
            ft_closed = [feat_fn(sliding_window(t, window_length, increment)) for t in raw_templates_closed]
            feature_templates[feat_name] = (ft_open, ft_closed)

        # Build sweep combinations
        combos = list(product(SWEEP_FEATURES, SWEEP_AGGREGATIONS, SWEEP_SMOOTHINGS, SWEEP_THRESHOLD_S))
        n_combos = len(combos)
        print(f"\n  Total combinations: {n_combos}")
        print("="*70)

        sweep_results = []

        for i, (feat_name, aggregation, (smoothing, smooth_window), th_s) in enumerate(combos):
            # Set config
            config.POST_PREDICTION_SMOOTHING = smoothing
            config.SMOOTHING_WINDOW = smooth_window

            ft_open, ft_closed = feature_templates[feat_name]

            # Compute thresholds with this s value
            mean_o, std_o, _ = compute_threshold(ft_open, s=th_s, verbose=False)
            mean_c, std_c, _ = compute_threshold(ft_closed, s=th_s, verbose=False)
            th_open = mean_o + th_s * std_o
            th_closed = mean_c + th_s * std_c

            # Run simulation (quiet)
            results = simulate_realtime_dtw(
                emg_data, ft_open, ft_closed,
                th_open, th_closed, feat_name,
                verbose=False,
                distance_aggregation=aggregation,
            )

            # Compute metrics
            metrics = compute_metrics(results, gt_data, gt_mode=gt_mode_str)

            if metrics is not None:
                sweep_results.append({
                    'feature': feat_name,
                    'aggregation': aggregation,
                    'smoothing': smoothing,
                    'smooth_window': smooth_window,
                    'th_s': th_s,
                    'th_open': th_open,
                    'th_closed': th_closed,
                    'metrics': metrics,
                    'results': results,
                })

            if (i + 1) % 10 == 0 or i == n_combos - 1:
                print(f"  [{i+1}/{n_combos}] completed...")

        # Print results table
        print_sweep_table(sweep_results)

        # Generate plots for top N results
        if SWEEP_PLOT_TOP_N > 0 and sweep_results:
            save_dir = base_path / "data" / "predictions" / "sweep"
            save_dir.mkdir(parents=True, exist_ok=True)

            top_n = sorted(sweep_results, key=lambda x: x['metrics']['f1_macro'], reverse=True)[:SWEEP_PLOT_TOP_N]
            for rank, r in enumerate(top_n):
                label = f"rank{rank+1}_{r['feature']}_{r['aggregation']}_{r['smoothing']}_s{r['th_s']}"
                save_path = str(save_dir / f"{label}.png")
                plot_title = (f"#{rank+1} {r['feature']} | {r['aggregation']} | {r['smoothing']} | "
                              f"s={r['th_s']} | F1={r['metrics']['f1_macro']:.3f}")
                plot_distance_results_segmented(
                    emg_data, r['results'], title=plot_title,
                    save_path=save_path, show_filtered=False,
                    gt_data=gt_data, gt_mode=gt_mode_str,
                    segment_duration_s=SEGMENT_DURATION_S,
                )
                print_metrics(r['metrics'], label=f"#{rank+1} {label}")

        if show:
            plt.show()

    elif MODE == "replay":
        from mindmove.model.core.algorithm import compute_threshold

        patient_dir = base_path / PATIENT_DIR
        prediction_path = patient_dir / PREDICTION_FILE
        model_path = patient_dir / REPLAY_MODEL_FILE

        print("\n" + "="*70)
        print("REPLAY MODE — Re-simulate + Comparison")
        print("="*70)
        print(f"  Prediction file: {PREDICTION_FILE}")
        print(f"  First model:     {REPLAY_MODEL_FILE}")
        if REPLAY_COMPARE_TEMPLATES:
            print(f"  Compare with:    {REPLAY_COMPARE_TEMPLATES}")
        print(f"  Refractory:      {REFRACTORY_PERIOD_S}s")
        print("="*70)

        # Load first model
        model_data = load_model(str(model_path))
        templates_open_1 = model_data['open_templates']
        templates_closed_1 = model_data['closed_templates']
        feature_name_1 = model_data['feature_name']
        dist_agg_1 = model_data.get('distance_aggregation', 'average')
        th_open_1 = model_data['threshold_base_open']
        th_closed_1 = model_data['threshold_base_closed']

        smoothing = model_data.get('smoothing_method', None)
        if smoothing:
            config.POST_PREDICTION_SMOOTHING = smoothing
        smoothing_window = model_data.get('parameters', {}).get('smoothing_window', config.SMOOTHING_WINDOW)
        config.SMOOTHING_WINDOW = smoothing_window

        print(f"\n  [First model]")
        print(f"  Feature: {feature_name_1}, Aggregation: {dist_agg_1}")
        print(f"  Smoothing: {config.POST_PREDICTION_SMOOTHING} (window={config.SMOOTHING_WINDOW})")
        print(f"  Thresholds: OPEN={th_open_1:.4f}, CLOSED={th_closed_1:.4f}")
        print(f"  Templates: {len(templates_open_1)} OPEN, {len(templates_closed_1)} CLOSED")

        # Load prediction file for EMG
        with open(prediction_path, 'rb') as f:
            pred_data = pickle.load(f)

        emg_data = pred_data['emg']
        n_channels = emg_data.shape[0]

        # Set differential mode
        is_diff = model_data.get('differential_mode', False) or n_channels <= 16
        if is_diff:
            config.ENABLE_DIFFERENTIAL_MODE = True
            config.num_channels = 16

        model_dead = model_data.get('dead_channels', DEAD_CHANNELS)
        config.dead_channels = model_dead
        config.active_channels = [i for i in range(config.num_channels) if i not in model_dead]

        print(f"\n  EMG shape: {emg_data.shape}")
        print(f"  Duration: {emg_data.shape[1] / config.FSAMP:.1f}s")

        # Cache directory
        cache_dir = base_path / "data" / "predictions" / "replay"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # --- Helper: simulate or load cached distances ---
        def get_cached_distances(emg_data, templates_open, templates_closed,
                                 feature_name, distance_aggregation, cache_name):
            """Run DTW simulation or load from cache. Cache stores only distances."""
            cache_path = cache_dir / cache_name
            if cache_path.exists():
                print(f"  Loading cached distances from: {cache_name}")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)

            print(f"  Computing distances (will be cached to {cache_name})...")
            results = simulate_realtime_dtw(
                emg_data, templates_open, templates_closed,
                0.0, 0.0, feature_name,  # thresholds don't matter for distances
                verbose=True, distance_aggregation=distance_aggregation,
                refractory_period_s=0.0,  # no state machine in cache
            )
            # Cache only the expensive part (distances + timestamps)
            cache_data = {
                'D_open': results['D_open'],
                'D_closed': results['D_closed'],
                'timestamps': results['timestamps'],
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"  Cached to: {cache_name}")
            return cache_data

        # --- First model: get distances (cached) ---
        cache_name_1 = f"cache_{Path(PREDICTION_FILE).stem}__{Path(REPLAY_MODEL_FILE).stem}.pkl"
        cached_1 = get_cached_distances(
            emg_data, templates_open_1, templates_closed_1,
            feature_name_1, dist_agg_1, cache_name_1,
        )

        # Predictions for the first model come from the prediction file
        # (what was actually sent to the virtual hand — more accurate than re-simulating)
        original_preds = pred_data.get('predictions', [])
        # Resample to match distance_history timestamps count
        sim_ts = cached_1['timestamps']
        if len(original_preds) == len(sim_ts):
            preds_1 = ["CLOSED" if p == 1.0 else "OPEN" for p in original_preds]
        elif len(original_preds) > 0:
            # Resample: map each sim timestamp to nearest prediction
            pred_times = np.linspace(sim_ts[0], sim_ts[-1], len(original_preds))
            indices = np.searchsorted(pred_times, sim_ts, side='right') - 1
            indices = np.clip(indices, 0, len(original_preds) - 1)
            preds_1 = ["CLOSED" if original_preds[i] == 1.0 else "OPEN" for i in indices]
        else:
            preds_1 = ["CLOSED"] * len(sim_ts)

        results_1 = {
            **cached_1,
            'predictions': preds_1,
            'threshold_open': th_open_1,
            'threshold_closed': th_closed_1,
        }

        # Overlay actual thresholds from prediction's unity_output
        actual = build_results_from_prediction(pred_data)
        if actual is not None and 'thresholds_open_over_time' in actual:
            actual_ts = actual['timestamps']
            results_1['thresholds_open_over_time'] = np.interp(
                sim_ts, actual_ts, actual['thresholds_open_over_time']
            )
            results_1['thresholds_closed_over_time'] = np.interp(
                sim_ts, actual_ts, actual['thresholds_closed_over_time']
            )

        # --- Second model (comparison): get distances + run state machine ---
        results_2 = None
        compare_label = None
        if REPLAY_COMPARE_TEMPLATES:
            compare_path = patient_dir / REPLAY_COMPARE_TEMPLATES
            with open(compare_path, 'rb') as f:
                compare_data = pickle.load(f)
            raw_open_2 = compare_data['templates_open']
            raw_closed_2 = compare_data['templates_closed']

            feat_name_2 = SINGLE_FEATURE
            dist_agg_2 = DISTANCE_AGGREGATION
            feat_fn_2 = FEATURES[feat_name_2]["function"]
            window_length = config.window_length
            increment = config.increment

            # Extract features from raw templates
            ft_open_2 = [feat_fn_2(sliding_window(t, window_length, increment)) for t in raw_open_2]
            ft_closed_2 = [feat_fn_2(sliding_window(t, window_length, increment)) for t in raw_closed_2]

            # Compute thresholds (intra-class mean + 1*std)
            _, _, th_open_2 = compute_threshold(ft_open_2, s=1, verbose=False)
            _, _, th_closed_2 = compute_threshold(ft_closed_2, s=1, verbose=False)

            compare_label = Path(REPLAY_COMPARE_TEMPLATES).stem
            print(f"\n  [New model — {compare_label}]")
            print(f"  Feature: {feat_name_2}, Aggregation: {dist_agg_2}")
            print(f"  Thresholds: OPEN={th_open_2:.4f}, CLOSED={th_closed_2:.4f}")
            print(f"  Templates: {len(ft_open_2)} OPEN, {len(ft_closed_2)} CLOSED")

            cache_name_2 = f"cache_{Path(PREDICTION_FILE).stem}__{compare_label}_{feat_name_2}_{dist_agg_2}.pkl"
            cached_2 = get_cached_distances(
                emg_data, ft_open_2, ft_closed_2,
                feat_name_2, dist_agg_2, cache_name_2,
            )

            preds_2 = apply_state_machine(
                cached_2['timestamps'], cached_2['D_open'], cached_2['D_closed'],
                th_open_2, th_closed_2, refractory_period_s=REFRACTORY_PERIOD_S,
            )
            results_2 = {
                **cached_2,
                'predictions': preds_2,
                'threshold_open': th_open_2,
                'threshold_closed': th_closed_2,
            }

        # --- Time range selection ---
        duration_s = emg_data.shape[1] / config.FSAMP
        if TIME_RANGE_S is not None:
            t_start, t_end = max(0, TIME_RANGE_S[0]), min(duration_s, TIME_RANGE_S[1])
            s_idx, e_idx = int(t_start * config.FSAMP), int(t_end * config.FSAMP)
            emg_data = emg_data[:, s_idx:e_idx]
            print(f"  Cropped to: {t_start:.1f}s - {t_end:.1f}s")
        elif duration_s > 60:
            print(f"\n  Recording is {duration_s:.1f}s long ({duration_s/60:.1f} min).")
            user_input = input(f"  Enter time range as 'start-end' (s), or Enter for full: ").strip()
            if user_input:
                try:
                    parts = user_input.split('-')
                    t_start = max(0, float(parts[0]))
                    t_end = min(duration_s, float(parts[1]))
                    s_idx, e_idx = int(t_start * config.FSAMP), int(t_end * config.FSAMP)
                    emg_data = emg_data[:, s_idx:e_idx]
                    print(f"  Cropped to: {t_start:.1f}s - {t_end:.1f}s")
                except (ValueError, IndexError):
                    print(f"  Invalid input, using full recording.")

        # --- Comparison plot ---
        def plot_replay_comparison(emg_data, results_1, results_2, title, save_path,
                                   time_range=None, segment_duration_s=100):
            """Plot comparison between first model and new model."""
            n_ch, n_samp = emg_data.shape
            total_dur = n_samp / config.FSAMP
            time_emg = np.arange(n_samp) / config.FSAMP

            if total_dur <= segment_duration_s:
                segments = [(None, None)]
            else:
                n_seg = int(np.ceil(total_dur / segment_duration_s))
                t0_emg = time_emg[0]
                segments = [(t0_emg + i * segment_duration_s,
                             min(t0_emg + (i + 1) * segment_duration_s, t0_emg + total_dur))
                            for i in range(n_seg)]

            for seg_i, (seg_start, seg_end) in enumerate(segments):
                has_compare = results_2 is not None
                # 4 subplots: EMG, D_open, D_closed, predictions (overlaid)
                n_rows = 4
                height_ratios = [3, 2, 2, 1]
                total_h = sum(height_ratios) * 1.5
                fig, axs = plt.subplots(n_rows, 1, figsize=(16, total_h), sharex=True,
                                        gridspec_kw={'height_ratios': height_ratios})

                seg_title = title
                if seg_start is not None:
                    seg_title = f"{title} [{seg_start:.0f}s - {seg_end:.0f}s]"

                # Slice EMG for this segment
                if seg_start is not None:
                    s_i = max(0, int(seg_start * config.FSAMP) - int(time_emg[0] * config.FSAMP))
                    e_i = min(n_samp, int(seg_end * config.FSAMP) - int(time_emg[0] * config.FSAMP))
                    emg_seg = emg_data[:, s_i:e_i]
                    time_seg = time_emg[s_i:e_i]
                else:
                    emg_seg = emg_data
                    time_seg = time_emg

                # 1) EMG
                ax = axs[0]
                _plot_stacked_emg(ax, time_seg, emg_seg, title=seg_title)

                ts1 = results_1['timestamps']
                D_open_1 = results_1['D_open']
                D_closed_1 = results_1['D_closed']

                # 2) Distance to OPEN
                ax = axs[1]
                ax.plot(ts1, D_open_1, 'g-', linewidth=1, alpha=0.7, label='D_open (first model)')
                th_open_ot = results_1.get('thresholds_open_over_time')
                if th_open_ot is not None and len(th_open_ot) == len(ts1):
                    ax.plot(ts1, th_open_ot, 'g--', linewidth=1.5, alpha=0.5, label='Threshold (first model)')
                else:
                    ax.axhline(results_1['threshold_open'], color='g', ls='--', lw=1.5, alpha=0.5,
                               label=f'Threshold first ({results_1["threshold_open"]:.3f})')

                if has_compare:
                    ts2 = results_2['timestamps']
                    ax.plot(ts2, results_2['D_open'], 'b-', linewidth=1, label='D_open (new model)')
                    ax.axhline(results_2['threshold_open'], color='b', ls='--', lw=1.5,
                               label=f'Threshold new ({results_2["threshold_open"]:.3f})')

                ax.set_ylabel("DTW Distance")
                ax.set_title("Distance to OPEN templates")
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)

                # 3) Distance to CLOSED
                ax = axs[2]
                ax.plot(ts1, D_closed_1, color='orange', linewidth=1, alpha=0.7, label='D_closed (first model)')
                th_closed_ot = results_1.get('thresholds_closed_over_time')
                if th_closed_ot is not None and len(th_closed_ot) == len(ts1):
                    ax.plot(ts1, th_closed_ot, color='orange', ls='--', lw=1.5, alpha=0.5, label='Threshold (first model)')
                else:
                    ax.axhline(results_1['threshold_closed'], color='orange', ls='--', lw=1.5, alpha=0.5,
                               label=f'Threshold first ({results_1["threshold_closed"]:.3f})')

                if has_compare:
                    ax.plot(ts2, results_2['D_closed'], 'r-', linewidth=1, label='D_closed (new model)')
                    ax.axhline(results_2['threshold_closed'], color='r', ls='--', lw=1.5,
                               label=f'Threshold new ({results_2["threshold_closed"]:.3f})')

                ax.set_ylabel("DTW Distance")
                ax.set_title("Distance to CLOSED templates")
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)

                # 4) Predictions — both overlaid in one subplot
                ax = axs[3]
                pred_1 = [1 if p == "CLOSED" else 0 for p in results_1['predictions']]
                ax.step(ts1, pred_1, 'purple', linewidth=2, where='post', label='Prediction (first model)')
                if has_compare:
                    pred_2 = [1 if p == "CLOSED" else 0 for p in results_2['predictions']]
                    ax.step(ts2, pred_2, 'blue', linewidth=1.5, where='post', alpha=0.8, label='Prediction (new model)')
                ax.set_ylabel("State")
                ax.set_xlabel("Time (s)")
                ax.set_ylim(-0.1, 1.1)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['OPEN', 'CLOSED'])
                ax.set_title("Predicted State")
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)

                if seg_start is not None:
                    for a in axs:
                        a.set_xlim(seg_start, seg_end)

                plt.tight_layout()

                seg_suffix = f"_seg{seg_i+1}" if seg_start is not None else ""
                seg_save = save_path.replace('.png', f'{seg_suffix}.png') if save_path else None
                if seg_save:
                    plt.savefig(seg_save, dpi=150, bbox_inches='tight')
                    print(f"  Figure saved to: {seg_save}")

        # Generate plots
        if results_2:
            plot_save = str(cache_dir / f"compare_{Path(PREDICTION_FILE).stem}.png")
        else:
            plot_save = str(cache_dir / f"replay_{Path(PREDICTION_FILE).stem}__{Path(REPLAY_MODEL_FILE).stem}.png")

        plot_replay_comparison(
            emg_data, results_1, results_2,
            title=f"REPLAY — {Path(REPLAY_MODEL_FILE).stem}",
            save_path=plot_save,
            segment_duration_s=SEGMENT_DURATION_S,
        )

        # Print metrics
        print(f"\n  [First model] predictions: "
              f"{sum(1 for p in results_1['predictions'] if p == 'OPEN')} OPEN, "
              f"{sum(1 for p in results_1['predictions'] if p == 'CLOSED')} CLOSED, "
              f"{sum(1 for i in range(1, len(results_1['predictions'])) if results_1['predictions'][i] != results_1['predictions'][i-1])} transitions")
        if results_2:
            print(f"  [New model]   predictions: "
                  f"{sum(1 for p in results_2['predictions'] if p == 'OPEN')} OPEN, "
                  f"{sum(1 for p in results_2['predictions'] if p == 'CLOSED')} CLOSED, "
                  f"{sum(1 for i in range(1, len(results_2['predictions'])) if results_2['predictions'][i] != results_2['predictions'][i-1])} transitions")

        if show:
            plt.show()
