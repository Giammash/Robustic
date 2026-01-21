"""
DTW Offline Testing and Visualization Script

This script allows you to:
1. Load a recording and a set of templates
2. Interactively select a 1-second window from the recording
3. Compute DTW between the selected window and templates
4. Visualize in detail:
   - Raw EMG comparison
   - Feature vectors after windowing
   - DTW cost matrix with optimal warping path
   - Aligned time series showing the warping
   - Per-channel distance contribution
   - Onset vs Hold region analysis

Author: MindMove Project
"""

import numpy as np
import pickle
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mindmove.config import config
from mindmove.model.core.windowing import sliding_window
from mindmove.model.core.features.features_registry import FEATURES


def dtw_with_path(t1: np.ndarray, t2: np.ndarray, active_channels: Optional[List[int]] = None) -> Tuple[float, np.ndarray, List[Tuple[int, int]]]:
    """
    Compute DTW with full diagnostic output.

    Returns the alignment cost, cost matrix, and optimal warping path.
    Uses cosine distance (same as the numba implementation).

    Args:
        t1: First template, shape (n_windows, n_channels)
        t2: Second template, shape (n_windows, n_channels)
        active_channels: List of channel indices to use. If None, uses config.active_channels

    Returns:
        alignment_cost: Final DTW distance
        cost_matrix: Full accumulated cost matrix (N+1, M+1)
        path: List of (i, j) tuples representing the optimal warping path
    """
    if active_channels is None:
        active_channels = config.active_channels

    N, nch = t1.shape
    M, _ = t2.shape

    # Cost matrix (N+1 x M+1 for boundary conditions)
    cost_mat = np.zeros((N + 1, M + 1))
    cost_mat[0, 1:] = np.inf
    cost_mat[1:, 0] = np.inf

    # Local cost matrix (for visualization)
    local_cost = np.zeros((N, M))

    # Traceback matrix: 0=diagonal, 1=vertical, 2=horizontal
    traceback_mat = np.zeros((N, M), dtype=int)

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # Cosine distance for active channels
            t1_vec = t1[i - 1, active_channels]
            t2_vec = t2[j - 1, active_channels]

            num = np.dot(t1_vec, t2_vec)
            den = (np.linalg.norm(t1_vec) * np.linalg.norm(t2_vec) + 1e-8)
            dist = 1 - num / den

            local_cost[i - 1, j - 1] = dist

            # DTW recurrence
            penalties = [
                cost_mat[i - 1, j - 1],  # diagonal (match)
                cost_mat[i - 1, j],       # vertical (insertion)
                cost_mat[i, j - 1],       # horizontal (deletion)
            ]
            i_penalty = np.argmin(penalties)
            cost_mat[i, j] = dist + penalties[i_penalty]
            traceback_mat[i - 1, j - 1] = i_penalty

    # Traceback to find optimal path
    path = []
    i, j = N - 1, M - 1
    path.append((i, j))

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            tb = traceback_mat[i, j]
            if tb == 0:  # diagonal
                i -= 1
                j -= 1
            elif tb == 1:  # vertical
                i -= 1
            else:  # horizontal
                j -= 1
        path.append((i, j))

    path = path[::-1]  # Reverse to get start-to-end order

    alignment_cost = cost_mat[N, M]

    return alignment_cost, cost_mat[1:, 1:], local_cost, path


def compute_per_channel_distances(t1: np.ndarray, t2: np.ndarray, path: List[Tuple[int, int]]) -> np.ndarray:
    """
    Compute the contribution of each channel to the total DTW distance.

    Args:
        t1: First template (n_windows, n_channels)
        t2: Second template (n_windows, n_channels)
        path: Warping path from DTW

    Returns:
        per_channel_dist: Array of shape (n_channels,) with each channel's contribution
    """
    n_channels = t1.shape[1]
    per_channel_dist = np.zeros(n_channels)

    for ch in range(n_channels):
        total_dist = 0
        for i, j in path:
            # Cosine distance for this channel
            v1 = t1[i, ch]
            v2 = t2[j, ch]
            num = v1 * v2
            den = (np.abs(v1) * np.abs(v2) + 1e-8)
            dist = 1 - num / den
            total_dist += dist
        per_channel_dist[ch] = total_dist / len(path)

    return per_channel_dist


def compute_region_distances(t1: np.ndarray, t2: np.ndarray, path: List[Tuple[int, int]],
                             active_channels: List[int], region_split: float = 0.3) -> dict:
    """
    Analyze DTW distance contribution from different time regions.

    Args:
        t1: First template (n_windows, n_channels)
        t2: Second template (n_windows, n_channels)
        path: Warping path
        active_channels: Channels to consider
        region_split: Fraction of template to consider as "onset" (default 0.3 = first 30%)

    Returns:
        Dictionary with onset_distance, hold_distance, and per-region statistics
    """
    n_windows = t1.shape[0]
    onset_boundary = int(n_windows * region_split)

    onset_distances = []
    hold_distances = []

    for i, j in path:
        # Cosine distance
        t1_vec = t1[i, active_channels]
        t2_vec = t2[j, active_channels]
        num = np.dot(t1_vec, t2_vec)
        den = (np.linalg.norm(t1_vec) * np.linalg.norm(t2_vec) + 1e-8)
        dist = 1 - num / den

        # Classify by position in t1 (the test signal)
        if i < onset_boundary:
            onset_distances.append(dist)
        else:
            hold_distances.append(dist)

    return {
        'onset_distance': np.sum(onset_distances) if onset_distances else 0,
        'onset_mean': np.mean(onset_distances) if onset_distances else 0,
        'onset_count': len(onset_distances),
        'hold_distance': np.sum(hold_distances) if hold_distances else 0,
        'hold_mean': np.mean(hold_distances) if hold_distances else 0,
        'hold_count': len(hold_distances),
        'onset_boundary': onset_boundary,
    }


class DTWVisualizer:
    """Interactive DTW visualization tool."""

    def __init__(self):
        self.recording = None
        self.templates = None
        self.template_metadata = None
        self.selected_window = None
        self.feature_name = 'wl'
        self.window_samples = int(0.096 * config.FSAMP)  # 96ms
        self.overlap_samples = int(0.032 * config.FSAMP)  # 32ms

    def load_recording(self, filepath: str) -> dict:
        """Load a recording file."""
        print(f"Loading recording: {filepath}")

        if filepath.endswith('.mat'):
            import scipy.io as sio
            mat_data = sio.loadmat(filepath)
            recording = {}
            for key in mat_data.keys():
                if not key.startswith('__'):
                    val = mat_data[key]
                    if val.dtype.kind in ('U', 'S'):
                        recording[key] = str(val.flat[0]) if val.size > 0 else ""
                    elif val.size == 1:
                        recording[key] = val.flat[0]
                    else:
                        recording[key] = val
        else:
            with open(filepath, 'rb') as f:
                recording = pickle.load(f)

        self.recording = recording

        # Detect format and extract EMG
        if 'emg' in recording:
            emg = recording['emg']
        elif 'biosignal' in recording:
            emg = recording['biosignal']
        else:
            raise ValueError(f"Unknown recording format. Keys: {list(recording.keys())}")

        print(f"  EMG shape: {emg.shape}")
        print(f"  Duration: {emg.shape[1] / config.FSAMP:.2f}s")

        return recording

    def load_templates(self, filepath: str) -> List[np.ndarray]:
        """Load templates file."""
        print(f"Loading templates: {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict) and 'templates' in data:
            templates = data['templates']
            self.template_metadata = data.get('metadata', {})
        elif isinstance(data, list):
            templates = data
            self.template_metadata = {}
        else:
            raise ValueError("Unknown template format")

        self.templates = templates
        print(f"  Loaded {len(templates)} templates")
        print(f"  Template shape: {templates[0].shape}")

        return templates

    def load_model(self, filepath: str) -> dict:
        """Load a trained model (already has feature-extracted templates)."""
        print(f"Loading model: {filepath}")

        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        print(f"  Open templates: {len(model['open_templates'])}")
        print(f"  Closed templates: {len(model['closed_templates'])}")
        print(f"  Feature: {model.get('feature_name', 'unknown')}")

        return model

    def interactive_window_selection(self, channel: int = 0) -> Optional[np.ndarray]:
        """
        Interactively select a 1-second window from the recording.

        Args:
            channel: Which channel to display for selection

        Returns:
            Selected EMG window (n_channels, n_samples) or None if cancelled
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import Button

        if self.recording is None:
            print("No recording loaded!")
            return None

        # Get EMG data
        if 'emg' in self.recording:
            emg = self.recording['emg']
        else:
            emg = self.recording['biosignal']

        n_channels, n_samples = emg.shape
        time_axis = np.arange(n_samples) / config.FSAMP
        window_duration = 1.0  # 1 second
        window_samples = int(window_duration * config.FSAMP)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        plt.subplots_adjust(bottom=0.2)

        # Plot EMG signal
        emg_signal = emg[channel, :]
        emg_normalized = emg_signal / (np.max(np.abs(emg_signal)) + 1e-10)
        ax.plot(time_axis, emg_normalized, 'b-', linewidth=0.5, label=f'EMG Ch{channel + 1}')

        # Plot GT if available
        if 'gt' in self.recording:
            gt = self.recording['gt']
            if gt.ndim > 1:
                gt = gt.flatten()
            ax.fill_between(time_axis[:len(gt)], -1, 1, where=gt > 0.5,
                           alpha=0.2, color='green', label='GT=1')
        elif 'kinematics' in self.recording:
            kin = self.recording['kinematics']
            if kin.ndim > 1:
                kin = kin[0, :]  # Take first row
            kin_norm = kin / (np.max(np.abs(kin)) + 1e-10)
            ax.plot(time_axis[:len(kin_norm)], kin_norm, 'g-', alpha=0.5, linewidth=0.5, label='Kinematics')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized Amplitude')
        ax.set_title(f'Click to select 1-second window start\nChannel {channel + 1}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.2, 1.2)

        # Selection state
        state = {'start_time': None, 'rect': None, 'vline': None, 'confirmed': False}

        def on_click(event):
            if event.inaxes != ax or event.button != 1:
                return

            click_time = event.xdata
            if click_time is None:
                return

            click_idx = int(click_time * config.FSAMP)
            if click_idx + window_samples > n_samples:
                print(f"Not enough samples after click position")
                return

            # Remove previous markers
            if state['rect'] is not None:
                state['rect'].remove()
            if state['vline'] is not None:
                state['vline'].remove()

            # Draw new markers
            state['vline'] = ax.axvline(x=click_time, color='red', linestyle='-', linewidth=2)
            state['rect'] = ax.add_patch(
                Rectangle((click_time, -1.2), window_duration, 2.4,
                          facecolor='red', alpha=0.2, edgecolor='red', linewidth=2)
            )

            state['start_time'] = click_time
            ax.set_title(f'Selected: {click_time:.2f}s to {click_time + window_duration:.2f}s\n'
                        f'Click "Confirm" or click again to adjust')
            fig.canvas.draw()

        def on_confirm(event):
            state['confirmed'] = True
            plt.close(fig)

        def on_cancel(event):
            state['confirmed'] = False
            state['start_time'] = None
            plt.close(fig)

        # Add buttons
        ax_confirm = plt.axes([0.7, 0.05, 0.1, 0.05])
        ax_cancel = plt.axes([0.81, 0.05, 0.1, 0.05])
        btn_confirm = Button(ax_confirm, 'Confirm')
        btn_cancel = Button(ax_cancel, 'Cancel')
        btn_confirm.on_clicked(on_confirm)
        btn_cancel.on_clicked(on_cancel)

        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

        if state['confirmed'] and state['start_time'] is not None:
            start_idx = int(state['start_time'] * config.FSAMP)
            end_idx = start_idx + window_samples
            self.selected_window = emg[:, start_idx:end_idx]
            print(f"Selected window: {state['start_time']:.2f}s to {state['start_time'] + window_duration:.2f}s")
            print(f"Window shape: {self.selected_window.shape}")
            return self.selected_window

        return None

    def extract_features(self, emg_window: np.ndarray) -> np.ndarray:
        """
        Extract features from an EMG window.

        Args:
            emg_window: EMG data (n_channels, n_samples)

        Returns:
            Feature array (n_windows, n_channels)
        """
        feature_fn = FEATURES[self.feature_name]['function']
        windowed = sliding_window(emg_window, self.window_samples, self.overlap_samples)
        features = feature_fn(windowed)
        return features

    def visualize_dtw_full(self, test_features: np.ndarray, template_features: np.ndarray,
                          template_idx: int = 0, active_channels: Optional[List[int]] = None):
        """
        Full DTW visualization with multiple panels.

        Args:
            test_features: Features from test window (n_windows, n_channels)
            template_features: Features from template (n_windows, n_channels)
            template_idx: Template index for display purposes
            active_channels: Channels to use for DTW
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        if active_channels is None:
            active_channels = config.active_channels

        # Compute DTW with full diagnostics
        alignment_cost, cost_matrix, local_cost, path = dtw_with_path(
            test_features, template_features, active_channels
        )

        # Compute per-channel distances
        per_channel_dist = compute_per_channel_distances(test_features, template_features, path)

        # Compute region distances
        region_stats = compute_region_distances(test_features, template_features, path, active_channels)

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))

        # 1. Cost matrix with warping path
        ax1 = fig.add_subplot(2, 3, 1)
        im1 = ax1.imshow(cost_matrix.T, origin='lower', aspect='auto', cmap='viridis')
        path_i, path_j = zip(*path)
        ax1.plot(path_i, path_j, 'r-', linewidth=2, label='Warping path')
        ax1.set_xlabel('Test window (windows)')
        ax1.set_ylabel('Template (windows)')
        ax1.set_title(f'Accumulated Cost Matrix\nTotal DTW: {alignment_cost:.4f}')
        plt.colorbar(im1, ax=ax1, label='Accumulated cost')
        ax1.legend()

        # 2. Local cost matrix
        ax2 = fig.add_subplot(2, 3, 2)
        im2 = ax2.imshow(local_cost.T, origin='lower', aspect='auto', cmap='hot')
        ax2.plot(path_i, path_j, 'cyan', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Test window (windows)')
        ax2.set_ylabel('Template (windows)')
        ax2.set_title('Local Cost Matrix (Cosine Distance)')
        plt.colorbar(im2, ax=ax2, label='Local cost')

        # 3. Aligned time series (mean across active channels)
        ax3 = fig.add_subplot(2, 3, 3)
        test_mean = np.mean(test_features[:, active_channels], axis=1)
        template_mean = np.mean(template_features[:, active_channels], axis=1)

        # Original signals
        ax3.plot(test_mean, 'b-', alpha=0.5, label='Test (original)')
        ax3.plot(template_mean, 'r-', alpha=0.5, label='Template (original)')

        # Aligned template (warped to test)
        aligned_template = np.array([template_mean[j] for i, j in path])
        aligned_test = np.array([test_mean[i] for i, j in path])
        ax3.plot(aligned_test, 'b--', linewidth=2, label='Test (aligned)')
        ax3.plot(aligned_template, 'r--', linewidth=2, label='Template (aligned)')

        ax3.set_xlabel('Aligned index')
        ax3.set_ylabel('Mean feature value')
        ax3.set_title('Time Series Alignment')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Per-channel distance contribution
        ax4 = fig.add_subplot(2, 3, 4)
        x_channels = np.arange(len(per_channel_dist))
        colors = ['green' if ch in active_channels else 'gray' for ch in x_channels]
        bars = ax4.bar(x_channels, per_channel_dist, color=colors, alpha=0.7)
        ax4.set_xlabel('Channel')
        ax4.set_ylabel('Mean cosine distance')
        ax4.set_title('Per-Channel Distance Contribution\n(Green = active, Gray = inactive)')
        ax4.set_xticks(x_channels[::4])
        ax4.set_xticklabels([str(x+1) for x in x_channels[::4]])

        # Highlight top contributors
        sorted_channels = np.argsort(per_channel_dist)[::-1]
        for i, ch in enumerate(sorted_channels[:3]):
            if ch in active_channels:
                bars[ch].set_color('red')
                ax4.annotate(f'Ch{ch+1}', (ch, per_channel_dist[ch]),
                           textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

        # 5. Onset vs Hold region analysis
        ax5 = fig.add_subplot(2, 3, 5)
        regions = ['Onset\n(first 30%)', 'Hold\n(last 70%)']
        distances = [region_stats['onset_mean'], region_stats['hold_mean']]
        counts = [region_stats['onset_count'], region_stats['hold_count']]

        bars = ax5.bar(regions, distances, color=['orange', 'blue'], alpha=0.7)
        ax5.set_ylabel('Mean local distance')
        ax5.set_title(f'Distance by Region\nOnset: {region_stats["onset_count"]} steps, Hold: {region_stats["hold_count"]} steps')

        for bar, dist, cnt in zip(bars, distances, counts):
            ax5.annotate(f'{dist:.4f}\n({cnt} steps)',
                        (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 5), ha='center')

        # 6. Feature comparison for a few channels
        ax6 = fig.add_subplot(2, 3, 6)
        n_show = min(4, len(active_channels))
        channels_to_show = active_channels[:n_show]

        for i, ch in enumerate(channels_to_show):
            offset = i * 2
            ax6.plot(test_features[:, ch] + offset, 'b-', alpha=0.7,
                    label=f'Test Ch{ch+1}' if i == 0 else '')
            ax6.plot(template_features[:, ch] + offset, 'r--', alpha=0.7,
                    label=f'Template Ch{ch+1}' if i == 0 else '')
            ax6.text(-1, offset, f'Ch{ch+1}', fontsize=8, va='center')

        ax6.set_xlabel('Window index')
        ax6.set_ylabel('Feature value (offset for visibility)')
        ax6.set_title(f'Feature Comparison\n(First {n_show} active channels)')
        ax6.legend(['Test', 'Template'])

        plt.suptitle(f'DTW Analysis - Template {template_idx + 1}\n'
                    f'Feature: {self.feature_name}, Active channels: {len(active_channels)}',
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return {
            'alignment_cost': alignment_cost,
            'cost_matrix': cost_matrix,
            'local_cost': local_cost,
            'path': path,
            'per_channel_dist': per_channel_dist,
            'region_stats': region_stats,
        }

    def compare_with_all_templates(self, test_features: np.ndarray,
                                   templates_features: List[np.ndarray],
                                   class_label: str = ""):
        """
        Compare test window with all templates and show summary.

        Args:
            test_features: Features from test window
            templates_features: List of template features
            class_label: Label for display
        """
        import matplotlib.pyplot as plt

        distances = []
        for i, template in enumerate(templates_features):
            cost, _, _, _ = dtw_with_path(test_features, template)
            distances.append(cost)
            print(f"  Template {i+1}: DTW = {cost:.4f}")

        distances = np.array(distances)

        # Plot summary
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Bar chart of distances
        ax1 = axes[0]
        x = np.arange(len(distances))
        colors = ['green' if d == distances.min() else 'blue' for d in distances]
        ax1.bar(x, distances, color=colors, alpha=0.7)
        ax1.axhline(y=np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.4f}')
        ax1.set_xlabel('Template index')
        ax1.set_ylabel('DTW distance')
        ax1.set_title(f'{class_label} Templates - DTW Distances')
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(i+1) for i in x])
        ax1.legend()

        # Histogram
        ax2 = axes[1]
        ax2.hist(distances, bins=10, color='blue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.4f}')
        ax2.axvline(x=np.min(distances), color='green', linestyle='-', label=f'Min: {np.min(distances):.4f}')
        ax2.set_xlabel('DTW distance')
        ax2.set_ylabel('Count')
        ax2.set_title('Distance Distribution')
        ax2.legend()

        plt.suptitle(f'Comparison with {len(templates_features)} {class_label} templates\n'
                    f'Best match: Template {np.argmin(distances) + 1}',
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return distances


def main():
    """Main interactive testing function."""
    import matplotlib
    matplotlib.use('TkAgg')  # Use interactive backend

    print("=" * 60)
    print("DTW Offline Testing and Visualization")
    print("=" * 60)

    visualizer = DTWVisualizer()

    # File selection
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()

    # 1. Load recording
    print("\n[1] Select a recording file...")
    recording_path = filedialog.askopenfilename(
        title="Select Recording",
        initialdir="data/recordings",
        filetypes=[("Pickle files", "*.pkl"), ("MAT files", "*.mat"), ("All files", "*.*")]
    )

    if not recording_path:
        print("No recording selected. Exiting.")
        return

    visualizer.load_recording(recording_path)

    # 2. Load model or templates
    print("\n[2] Select a model or template file...")
    model_path = filedialog.askopenfilename(
        title="Select Model or Templates",
        initialdir="data/models",
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
    )

    if not model_path:
        print("No model/templates selected. Exiting.")
        return

    # Detect if it's a model or raw templates
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict) and 'open_templates' in data:
        # It's a trained model (feature-extracted templates)
        model = data
        open_templates = model['open_templates']
        closed_templates = model['closed_templates']
        visualizer.feature_name = model.get('feature_name', 'wl')

        # Get window parameters from model
        params = model.get('parameters', {})
        visualizer.window_samples = params.get('window_samples', int(0.096 * config.FSAMP))
        visualizer.overlap_samples = params.get('overlap_samples', int(0.032 * config.FSAMP))

        templates_ready = True
        print(f"  Loaded trained model with {len(open_templates)} open, {len(closed_templates)} closed templates")
        print(f"  Feature: {visualizer.feature_name}")
    else:
        # Raw templates - need to extract features
        if isinstance(data, dict) and 'templates' in data:
            raw_templates = data['templates']
        else:
            raw_templates = data

        print(f"  Loaded {len(raw_templates)} raw templates - will extract features")
        templates_ready = False

        # Ask which class
        print("\nWhich class are these templates? (open/closed): ", end='')
        class_choice = input().strip().lower()
        if class_choice == 'open':
            open_templates = [visualizer.extract_features(t) for t in raw_templates]
            closed_templates = []
        else:
            closed_templates = [visualizer.extract_features(t) for t in raw_templates]
            open_templates = []
        templates_ready = True

    # 3. Interactive window selection
    print("\n[3] Select a 1-second window from the recording...")
    print("    (Click on the plot to set the window start)")

    channel = 0
    window = visualizer.interactive_window_selection(channel=channel)

    if window is None:
        print("No window selected. Exiting.")
        return

    # Extract features from selected window
    print(f"\nExtracting features ({visualizer.feature_name})...")
    test_features = visualizer.extract_features(window)
    print(f"  Test features shape: {test_features.shape}")

    # 4. Compare with templates
    if open_templates:
        print(f"\n[4] Comparing with OPEN templates...")
        open_distances = visualizer.compare_with_all_templates(test_features, open_templates, "OPEN")

        # Detailed view of best match
        best_open_idx = np.argmin(open_distances)
        print(f"\nDetailed DTW visualization for best OPEN match (Template {best_open_idx + 1})...")
        visualizer.visualize_dtw_full(test_features, open_templates[best_open_idx],
                                      template_idx=best_open_idx)

    if closed_templates:
        print(f"\n[5] Comparing with CLOSED templates...")
        closed_distances = visualizer.compare_with_all_templates(test_features, closed_templates, "CLOSED")

        # Detailed view of best match
        best_closed_idx = np.argmin(closed_distances)
        print(f"\nDetailed DTW visualization for best CLOSED match (Template {best_closed_idx + 1})...")
        visualizer.visualize_dtw_full(test_features, closed_templates[best_closed_idx],
                                      template_idx=best_closed_idx)

    # Summary
    if open_templates and closed_templates:
        min_open = np.min(open_distances)
        min_closed = np.min(closed_distances)

        print("\n" + "=" * 60)
        print("CLASSIFICATION RESULT")
        print("=" * 60)
        print(f"  Min distance to OPEN:   {min_open:.4f} (Template {np.argmin(open_distances) + 1})")
        print(f"  Min distance to CLOSED: {min_closed:.4f} (Template {np.argmin(closed_distances) + 1})")
        print(f"  Predicted class: {'OPEN' if min_open < min_closed else 'CLOSED'}")
        print(f"  Confidence margin: {abs(min_open - min_closed):.4f}")

    print("\n[Done] You can re-run to test different windows.")


if __name__ == "__main__":
    main()
