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
from mindmove.model.core.algorithm import compute_dtw, compute_distance_from_training_set_online, GPUDTW_AVAILABLE
from mindmove.model.core.windowing import sliding_window
from mindmove.model.core.filtering import apply_rtfiltering
from mindmove.model.core.features.features_registry import FEATURES


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
            elif 'kinematics' in data:
                gt_data = data['kinematics']
                gt_mode = "virtual_hand"
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
    verbose: bool = True
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

    # Filtered signal reconstruction (only the "new" 50ms from each filtered buffer)
    filtered_signal_chunks = []

    # State machine
    current_state = "CLOSED"
    last_predictions = []

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

            D_open = compute_distance_from_training_set_online(features, templates_open)
            D_closed = compute_distance_from_training_set_online(features, templates_closed)

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
            if current_state == "OPEN":
                triggered_state = "CLOSED" if D_closed < threshold_closed else "OPEN"
            else:
                triggered_state = "OPEN" if D_open < threshold_open else "CLOSED"

            # Majority vote smoothing
            if config.POST_PREDICTION_SMOOTHING != "NONE":
                last_predictions.append(triggered_state)
                if len(last_predictions) > config.SMOOTHING_WINDOW:
                    last_predictions.pop(0)
                    if config.POST_PREDICTION_SMOOTHING == "MAJORITY VOTE":
                        if last_predictions.count("CLOSED") > last_predictions.count("OPEN"):
                            current_state = "CLOSED"
                        else:
                            current_state = "OPEN"
                        last_predictions = []
            else:
                current_state = triggered_state

            predictions.append(current_state)

            if verbose and dtw_count % 20 == 0:
                print(f"  [{dtw_count:4d}] t={end_idx/config.FSAMP:6.2f}s | "
                      f"D_open={D_open:.4f} D_closed={D_closed:.4f} | "
                      f"State={current_state:6s} | DTW time={dtw_time_ms:.2f}ms")

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
    }


def plot_distance_results(
    emg_data: np.ndarray,
    results: dict,
    title: str = "DTW Distance Analysis",
    ch_to_plot: int = 0,
    save_path: str = None,
    show_filtered: bool = True,
    gt_data: np.ndarray = None
):
    """
    Plot DTW distances over time with EMG signal.

    Args:
        emg_data: Original EMG data (n_channels, n_samples)
        results: Results dictionary from simulate_realtime_dtw
        title: Plot title
        ch_to_plot: Channel to display
        save_path: Optional path to save figure
        show_filtered: If True, show reconstructed real-time filtered signal
        gt_data: Ground truth data (optional, for overlay on prediction plot)
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
    n_samples = emg_data.shape[1]
    time_emg = np.arange(n_samples) / config.FSAMP

    # Check if we have filtered signal to show (only if filtering is enabled)
    has_filtered = show_filtered and filtered_signal is not None and config.ENABLE_FILTERING

    # Create figure (5 subplots if showing filtered, 4 otherwise)
    n_plots = 5 if has_filtered else 4
    fig, axs = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots), sharex=True)

    plot_idx = 0

    # Plot 1: Raw EMG with GT overlay
    ax = axs[plot_idx]
    ax.plot(time_emg, emg_data[ch_to_plot, :], 'b-', alpha=0.7, linewidth=0.5)
    # Add GT overlay if available
    if gt_data is not None:
        gt_1d = gt_data.flatten() if gt_data.ndim > 1 else gt_data
        if len(gt_1d) == n_samples:
            # Scale GT to match EMG range for visualization
            emg_max = np.max(np.abs(emg_data[ch_to_plot, :]))
            ax.fill_between(time_emg, -emg_max, emg_max, where=gt_1d > 0.5,
                           alpha=0.15, color='green', label='GT=1')
    ax.set_ylabel(f"EMG CH{ch_to_plot+1} (µV)")
    ax.set_title(f"{title} - Channel {ch_to_plot+1} (RAW)")
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # Plot 2: Reconstructed real-time filtered EMG (if available)
    if has_filtered:
        ax = axs[plot_idx]
        # Time axis for filtered signal (starts after 1 second buffer fill)
        n_filtered_samples = filtered_signal.shape[1]
        time_filtered = filtered_start_time + np.arange(n_filtered_samples) / config.FSAMP
        ax.plot(time_filtered, filtered_signal[ch_to_plot, :], 'g-', alpha=0.7, linewidth=0.5)
        ax.set_ylabel(f"EMG CH{ch_to_plot+1} (µV)")
        filter_type = "notch only" if config.ENABLE_FILTERING else "none"
        ax.set_title(f"Channel {ch_to_plot+1} (REAL-TIME FILTERED: {filter_type}) - starts at {filtered_start_time:.1f}s")
        ax.set_xlim(time_emg[0], time_emg[-1])  # Match x-axis with raw signal
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Distance to OPEN templates
    ax = axs[plot_idx]
    ax.plot(timestamps, D_open, 'g-', linewidth=1, label='Distance to OPEN')
    ax.axhline(threshold_open, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold_open:.3f})')
    ax.set_ylabel("DTW Distance")
    ax.set_title("Distance to OPEN templates")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # Distance to CLOSED templates
    ax = axs[plot_idx]
    ax.plot(timestamps, D_closed, 'orange', linewidth=1, label='Distance to CLOSED')
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
    # Add GT overlay
    if gt_data is not None:
        gt_1d = gt_data.flatten() if gt_data.ndim > 1 else gt_data
        if len(gt_1d) == n_samples:
            ax.step(time_emg, gt_1d, 'green', linewidth=1.5, where='post', alpha=0.7, label='Ground Truth')
            ax.legend(loc='upper right')
    ax.set_ylabel("State")
    ax.set_xlabel("Time (s)")
    ax.set_title("Predicted State vs Ground Truth (0=OPEN, 1=CLOSED)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['OPEN', 'CLOSED'])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


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
    gt_data: np.ndarray = None
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
        verbose=verbose
    )

    # Plot and save
    plot_title = f"Folder: {folder_name} | {recording_name}"
    save_path = save_dir / f"{folder_name}_{recording_name}.png"

    plot_distance_results(
        emg_data, results, title=plot_title,
        save_path=str(save_path),
        gt_data=gt_data
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

    print(f"\nModel loaded:")
    print(f"  Open templates: {len(templates_open)}")
    print(f"  Closed templates: {len(templates_closed)}")
    print(f"  Threshold OPEN: {threshold_open:.4f}")
    print(f"  Threshold CLOSED: {threshold_closed:.4f}")
    print(f"  Feature: {feature_name}")

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
                gt_data=gt_data
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


# =============================================================================
# CONFIGURATION - EDIT THIS SECTION TO RUN YOUR TESTS
# =============================================================================

# Test folder (inside data/tests/)
# Available: "test", "test open", "test closed for closed detector", "test closed for open detector"
TEST_FOLDER = "aperture"

# Model file (inside data/models/)
# Use None for default (dtw_model_0.pkl), or specify a filename
MODEL_FILE = "MindMove_Model_20260121_133757_test_average_5_no_outliers.pkl"
# "dtw_model_0.pkl"  # e.g., "dtw_model_tslearn.pkl" for tslearn model

# DTW implementation (set ONE to True, others to False)
# - USE_NUMBA: fast numba (cosine distance) - use with cosine-trained models
# - USE_TSLEARN: tslearn (Euclidean distance) - use with tslearn-trained models
# - USE_GPUDTW: GPU-accelerated (Euclidean distance) - requires CUDA/OpenCL
USE_NUMBA = True
USE_TSLEARN = False
USE_GPUDTW = False

# =============================================================================


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Apply DTW configuration
    config.USE_NUMBA_DTW = USE_NUMBA
    config.USE_TSLEARN_DTW = USE_TSLEARN
    config.USE_GPUDTW = USE_GPUDTW

    # Build model path
    base_path = Path(__file__).parent.parent.parent
    model_path = base_path / "data" / "models" / MODEL_FILE

    # Check command line args
    show = "--no-show" not in sys.argv

    # Determine DTW implementation name for display
    if USE_GPUDTW:
        dtw_name = "GPUDTW (GPU, Euclidean)"
    elif USE_TSLEARN:
        dtw_name = "tslearn (Euclidean)"
    elif USE_NUMBA:
        dtw_name = "NUMBA (cosine)"
    else:
        dtw_name = "Original (cosine)"

    # Print configuration
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    print(f"  Test folder: {TEST_FOLDER}")
    print(f"  Model: {MODEL_FILE}")
    print(f"  DTW: {dtw_name}")
    print("="*70 + "\n")

    run_offline_test_folder(TEST_FOLDER, model_path=str(model_path), show_plot=show)
