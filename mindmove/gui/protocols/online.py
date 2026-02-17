from __future__ import annotations
from typing import TYPE_CHECKING
from PySide6.QtCore import QObject, QThread, Signal
import numpy as np
from datetime import datetime
import pickle
import os
import time
from PySide6.QtWidgets import (
    QFileDialog, QSlider, QLabel, QHBoxLayout, QVBoxLayout, QGroupBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QPushButton, QMessageBox,
    QDialog, QWidget, QSpinBox
)
from PySide6.QtCore import Qt
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# MindMove imports
from mindmove.model.interface import MindMoveInterface

from mindmove.config import config


class SimulationWorker(QObject):
    """Worker class for running offline simulation (+ optional calibration) in a separate thread."""
    finished = Signal(dict, object)  # (simulation results, calibration_thresholds or None)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, emg_data, model, gt_data=None):
        super().__init__()
        self.emg_data = emg_data
        self.model = model
        self.gt_data = gt_data  # Optional: real GT for calibration

    def run(self):
        """Run offline simulation using the loaded model's parameters.
        If gt_data is provided, also compute calibration thresholds."""
        try:
            from mindmove.model.offline_test import simulate_realtime_dtw

            templates_open = self.model.templates_open
            templates_closed = self.model.templates_closed
            threshold_open = self.model.THRESHOLD_OPEN
            threshold_closed = self.model.THRESHOLD_CLOSED
            feature_name = self.model.feature_name
            distance_aggregation = self.model.distance_aggregation

            # Spatial correction settings
            use_spatial = getattr(self.model, 'use_spatial_correction', False)
            spatial_ref_open = getattr(self.model, 'spatial_ref_open', None)
            spatial_ref_closed = getattr(self.model, 'spatial_ref_closed', None)
            spatial_threshold = getattr(self.model, 'spatial_threshold', 0.5)

            # Sync config with model parameters so simulate_realtime_dtw uses correct values
            config.window_length = self.model.window_length
            config.increment = self.model.increment

            print(f"[OFFLINE TEST] Starting offline simulation...")
            print(f"  EMG shape: {self.emg_data.shape}")
            print(f"  Feature: {feature_name}")
            print(f"  Templates: {len(templates_open)} OPEN, {len(templates_closed)} CLOSED")
            print(f"  Thresholds: OPEN={threshold_open:.4f}, CLOSED={threshold_closed:.4f}")
            print(f"  Aggregation: {distance_aggregation}")
            print(f"  Window/increment: {config.window_length}/{config.increment} samples")
            if use_spatial:
                print(f"  Spatial correction: ON (threshold={spatial_threshold:.2f})")
            else:
                print(f"  Spatial correction: OFF")
            if self.gt_data is not None:
                print(f"  GT available: yes ({len(self.gt_data)} samples) — will compute calibration thresholds")

            results = simulate_realtime_dtw(
                emg_data=self.emg_data,
                templates_open=templates_open,
                templates_closed=templates_closed,
                threshold_open=threshold_open,
                threshold_closed=threshold_closed,
                feature_name=feature_name,
                verbose=True,
                distance_aggregation=distance_aggregation,
                spatial_ref_open=spatial_ref_open,
                spatial_ref_closed=spatial_ref_closed,
                spatial_threshold=spatial_threshold,
                use_spatial_correction=use_spatial,
            )

            # If GT available, compute calibration thresholds from the simulation distances
            calibration_thresholds = None
            if self.gt_data is not None:
                try:
                    from mindmove.model.core.algorithm import find_plateau_thresholds

                    gt = self.gt_data
                    if hasattr(gt, 'flatten'):
                        gt = gt.flatten()
                    gt = np.array(gt, dtype=float)

                    timestamps = results['timestamps']
                    D_open = results['D_open']
                    D_closed = results['D_closed']

                    # Interpolate GT to DTW timestamps
                    gt_time = np.arange(len(gt)) / config.FSAMP
                    gt_at_dtw = np.interp(timestamps, gt_time, gt)
                    gt_at_dtw = (gt_at_dtw > 0.5).astype(float)

                    print(f"\n[OFFLINE TEST] Computing calibration thresholds from GT...")
                    calibration_thresholds = find_plateau_thresholds(
                        D_open, D_closed, gt_at_dtw,
                        confidence_k=1.0,
                    )
                    print(f"  OPEN plateau: {calibration_thresholds['open_plateau']:.4f}, "
                          f"threshold: {calibration_thresholds['threshold_open']:.4f}")
                    print(f"  CLOSED plateau: {calibration_thresholds['closed_plateau']:.4f}, "
                          f"threshold: {calibration_thresholds['threshold_closed']:.4f}")
                except Exception as e:
                    print(f"[OFFLINE TEST] Warning: calibration threshold computation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    calibration_thresholds = None

            self.finished.emit(results, calibration_thresholds)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class OfflineTestReviewDialog(QDialog):
    """Unified dialog for reviewing offline test results.

    Subplots (all share x-axis):
    1. EMG — stacked channels (filtered assumed)
    2. Distance to OPEN + threshold (+ calibration lines if GT)
    3. Distance to CLOSED + threshold (+ calibration lines if GT)
    4. Spatial Similarity (only if spatial correction enabled)
    5. State Comparison — reference (GT or previous predictions) + new offline prediction
    """

    def __init__(self, emg_data, results, gt_data=None, ref_predictions=None,
                 calibration_thresholds=None, model_thresholds=None,
                 recording_type="unknown", parent=None):
        super().__init__(parent)
        self.emg_data = emg_data
        self.results = results
        self.gt_data = gt_data  # Real GT from guided recording (array at FSAMP rate)
        self.ref_predictions = ref_predictions  # Previous online predictions (numeric array at DTW rate)
        self.calibration_thresholds = calibration_thresholds  # dict with plateau/threshold or None
        self.model_thresholds = model_thresholds  # {"threshold_open": ..., "threshold_closed": ...}
        self.recording_type = recording_type

        self.setWindowTitle("Offline Test Review")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        self.setModal(False)

        self._setup_ui()
        self._update_plot()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        header_layout = QHBoxLayout()
        title = QLabel("Offline Test Review")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        # Stats label
        results = self.results
        n_dtw = len(results['timestamps'])
        duration = results['timestamps'][-1] if n_dtw > 0 else 0
        avg_time = results.get('avg_dtw_time', 0)

        stats_parts = [
            f"DTW steps: {n_dtw}",
            f"Duration: {duration:.1f}s",
            f"Avg DTW: {avg_time:.2f}ms",
            f"Model thr OPEN: {results['threshold_open']:.4f}  CLOSED: {results['threshold_closed']:.4f}",
        ]
        if self.calibration_thresholds:
            ct = self.calibration_thresholds
            stats_parts.append(
                f"Calib thr OPEN: {ct['threshold_open']:.4f}  CLOSED: {ct['threshold_closed']:.4f}"
            )

        stats = QLabel("  |  ".join(stats_parts))
        stats.setStyleSheet("font-family: monospace; font-size: 10px; color: #555;")
        stats.setWordWrap(True)
        header_layout.addWidget(stats)
        layout.addLayout(header_layout)

        self.figure = Figure(figsize=(14, 10), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self.canvas.setFocus()

    def _update_plot(self):
        self.figure.clear()

        emg = self.emg_data
        results = self.results
        timestamps = results['timestamps']
        D_open = results['D_open']
        D_closed = results['D_closed']
        threshold_open = results['threshold_open']
        threshold_closed = results['threshold_closed']
        predictions = results['predictions']

        n_ch, n_samples = emg.shape
        time_emg = np.arange(n_samples) / config.FSAMP

        has_spatial = results.get('spatial_threshold') is not None
        has_gt = self.gt_data is not None
        has_ref = self.ref_predictions is not None
        has_calibration = self.calibration_thresholds is not None

        # Determine subplot layout
        height_ratios = [3, 1.5, 1.5]  # EMG, D_open, D_closed
        if has_spatial:
            height_ratios.append(1.0)
        height_ratios.append(1.2)  # State comparison
        n_plots = len(height_ratios)

        axs = self.figure.subplots(n_plots, 1, sharex=True,
                                   gridspec_kw={'height_ratios': height_ratios})

        plot_idx = 0

        # --- Plot 1: Stacked EMG (all channels) ---
        ax = axs[plot_idx]
        self._plot_stacked_emg(ax, time_emg, emg, title=f"EMG ({n_ch} channels)")

        # If GT available, add GT overlay as background shading
        if has_gt:
            gt = self.gt_data
            if hasattr(gt, 'flatten'):
                gt = gt.flatten()
            total_height = (n_ch - 1) * 1.0 + 1
            gt_for_plot = gt[:len(time_emg)] if len(gt) >= len(time_emg) else np.pad(gt, (0, len(time_emg) - len(gt)))
            ax.fill_between(time_emg, -0.5, total_height - 0.5, where=gt_for_plot > 0.5,
                            alpha=0.1, color='green', label='GT=CLOSED')
            ax.legend(loc='upper right', fontsize=7)
        plot_idx += 1

        # --- Plot 2: Distance to OPEN ---
        ax = axs[plot_idx]
        ax.plot(timestamps, D_open, 'g-', linewidth=1, label='D_open')
        ax.axhline(threshold_open, color='r', linestyle='--', linewidth=2,
                   label=f'Model thr: {threshold_open:.4f}')
        if has_calibration:
            ct = self.calibration_thresholds
            ax.axhline(ct['open_plateau'], color='b', linestyle=':', linewidth=1.5,
                       label=f'Plateau: {ct["open_plateau"]:.4f}')
            ax.axhline(ct['threshold_open'], color='magenta', linestyle='-.', linewidth=1.5,
                       label=f'Calib thr: {ct["threshold_open"]:.4f}')
        ax.set_ylabel("DTW Distance")
        ax.set_title("Distance to OPEN templates")
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        # --- Plot 3: Distance to CLOSED ---
        ax = axs[plot_idx]
        ax.plot(timestamps, D_closed, 'orange', linewidth=1, label='D_closed')
        ax.axhline(threshold_closed, color='r', linestyle='--', linewidth=2,
                   label=f'Model thr: {threshold_closed:.4f}')
        if has_calibration:
            ct = self.calibration_thresholds
            ax.axhline(ct['closed_plateau'], color='b', linestyle=':', linewidth=1.5,
                       label=f'Plateau: {ct["closed_plateau"]:.4f}')
            ax.axhline(ct['threshold_closed'], color='magenta', linestyle='-.', linewidth=1.5,
                       label=f'Calib thr: {ct["threshold_closed"]:.4f}')
        ax.set_ylabel("DTW Distance")
        ax.set_title("Distance to CLOSED templates")
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        # --- Plot 4 (optional): Spatial Similarity ---
        if has_spatial:
            ax = axs[plot_idx]
            sim_open_data = results.get('sim_open', [])
            sim_closed_data = results.get('sim_closed', [])
            spatial_thr = results['spatial_threshold']

            if sim_open_data:
                sim_open_plot = [s if s is not None else float('nan') for s in sim_open_data]
                ax.plot(timestamps[:len(sim_open_plot)], sim_open_plot,
                        'g-', linewidth=1, alpha=0.8, label='Sim OPEN')
            if sim_closed_data:
                sim_closed_plot = [s if s is not None else float('nan') for s in sim_closed_data]
                ax.plot(timestamps[:len(sim_closed_plot)], sim_closed_plot,
                        'orange', linewidth=1, alpha=0.8, label='Sim CLOSED')

            ax.axhline(spatial_thr, color='r', linestyle='--', linewidth=1.5,
                        label=f'Spatial thr: {spatial_thr:.2f}')

            # Mark blocked transitions
            blocked = results.get('spatial_blocked', [])
            for i, b in enumerate(blocked):
                if b and i < len(timestamps):
                    ax.axvline(timestamps[i], color='red', alpha=0.3, linewidth=2)

            ax.set_ylabel("Spatial Sim")
            ax.set_title("Spatial Similarity (consistency-weighted)")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # --- Plot 5: State Comparison ---
        ax = axs[plot_idx]

        # New offline prediction (purple)
        pred_numeric = [1 if p == "CLOSED" else 0 for p in predictions]
        ax.step(timestamps, pred_numeric, 'purple', linewidth=2, where='post',
                label='Offline prediction')

        # Reference signal (green): GT or previous predictions
        if has_gt:
            gt = self.gt_data
            if hasattr(gt, 'flatten'):
                gt = gt.flatten()
            gt = np.array(gt, dtype=float)
            gt_time = np.arange(len(gt)) / config.FSAMP
            # Interpolate GT to DTW timestamps for overlay
            gt_at_dtw = np.interp(timestamps, gt_time, gt)
            ax.step(timestamps, gt_at_dtw, 'green', linewidth=1.5, where='post', alpha=0.7,
                    label='GT (guided)')
        elif has_ref:
            ref = np.array(self.ref_predictions, dtype=float)
            if len(ref) == len(timestamps):
                ax.step(timestamps, ref, 'green', linewidth=1.5, where='post', alpha=0.7,
                        label='Previous prediction')
            elif len(ref) > 0:
                ref_time = np.linspace(timestamps[0], timestamps[-1], len(ref))
                ref_interp = np.interp(timestamps, ref_time, ref)
                ax.step(timestamps, ref_interp, 'green', linewidth=1.5, where='post', alpha=0.7,
                        label='Previous prediction')

        ax.set_ylabel("State")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['OPEN', 'CLOSED'])
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_stacked_emg(self, ax, time_arr, emg_data, title=""):
        """Plot all EMG channels stacked vertically."""
        n_ch = emg_data.shape[0]
        spacing = 1.0
        channel_maxes = np.array([np.max(np.abs(emg_data[ch, :])) for ch in range(n_ch)])
        channel_maxes[channel_maxes == 0] = 1
        colors = plt.cm.tab20(np.linspace(0, 1, n_ch))
        for ch in range(n_ch):
            offset = ch * spacing
            normalized = emg_data[ch, :] / channel_maxes[ch] * 0.4
            ax.plot(time_arr, normalized + offset, color=colors[ch], linewidth=0.4, alpha=0.8)
        ax.set_yticks([ch * spacing for ch in range(n_ch)])
        ax.set_yticklabels([f'{ch+1}' for ch in range(n_ch)], fontsize=7)
        ax.set_ylabel("Channels")
        ax.set_ylim(-0.5, (n_ch - 1) * spacing + 0.5)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)


class OnlineSessionReviewDialog(QDialog):
    """
    Dialog for reviewing online prediction session with channel selection.

    Shows 3 subplots:
    - EMG signal (selectable channel)
    - State over time
    - Distance to templates with thresholds

    Supports arrow keys for channel switching.
    """

    def __init__(self, emg_signal, emg_time_axis, history, parent=None, unity_output=None):
        super().__init__(parent)
        self.emg_signal = emg_signal  # (n_channels, n_samples)
        self.emg_time_axis = emg_time_axis
        self.history = history
        self.unity_output = unity_output
        self.current_channel = 0
        self.n_channels = emg_signal.shape[0] if emg_signal is not None else 32

        self.setWindowTitle("Online Session Review")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        self.setModal(False)

        self._setup_ui()
        self._update_plot()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Header with channel selector
        header_layout = QHBoxLayout()

        title = QLabel("Online Prediction Session Review")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Channel selector
        header_layout.addWidget(QLabel("EMG Channel:"))
        self.channel_combo = QComboBox()
        for i in range(1, self.n_channels + 1):
            self.channel_combo.addItem(str(i))
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        self.channel_combo.setToolTip("Use Up/Down arrow keys to switch channels")
        header_layout.addWidget(self.channel_combo)

        # Instructions
        hint = QLabel("(Up/Down to change channel)")
        hint.setStyleSheet("color: #666; font-style: italic;")
        header_layout.addWidget(hint)

        layout.addLayout(header_layout)

        # Matplotlib figure with 3 subplots
        self.figure = Figure(figsize=(14, 10), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        layout.addWidget(self.canvas)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        # Set focus to canvas for key events
        self.canvas.setFocus()

    def keyPressEvent(self, event):
        """Handle arrow key presses for channel switching."""
        if event.key() == Qt.Key_Up:
            new_idx = max(0, self.channel_combo.currentIndex() - 1)
            self.channel_combo.setCurrentIndex(new_idx)
        elif event.key() == Qt.Key_Down:
            new_idx = min(self.n_channels - 1, self.channel_combo.currentIndex() + 1)
            self.channel_combo.setCurrentIndex(new_idx)
        else:
            super().keyPressEvent(event)

    def _on_channel_changed(self, index):
        self.current_channel = index
        self._update_plot()

    @staticmethod
    def _reconstruct_thresholds_from_unity(unity_output, history_timestamps, default_open, default_closed):
        """Reconstruct per-step threshold time series from unity_output.

        unity_output stores the threshold being checked at each step:
        - When state=OPEN -> threshold = THRESHOLD_CLOSED (checking closed templates)
        - When state=CLOSED -> threshold = THRESHOLD_OPEN (checking open templates)

        We forward-fill gaps (assume threshold stays constant until next known value).

        Returns:
            (thresholds_open, thresholds_closed) -- lists aligned with history_timestamps
        """
        n = len(history_timestamps)
        th_open = [None] * n
        th_closed = [None] * n

        # Build a lookup from unity_output timestamps
        # unity_output has much higher resolution (every EMG packet),
        # history_timestamps are every DTW step (~50ms)
        # Normalize unity timestamps to same base as history (which starts at 0)
        uo_times_raw = np.array([e['timestamp'] for e in unity_output])
        uo_t0 = uo_times_raw[0] if len(uo_times_raw) > 0 else 0
        uo_times = uo_times_raw - uo_t0
        uo_thresholds = [e['threshold'] for e in unity_output]
        uo_states = [e['state_name'] for e in unity_output]

        for i, t in enumerate(history_timestamps):
            # Find closest unity_output entry
            idx = np.searchsorted(uo_times, t)
            idx = min(idx, len(uo_times) - 1)

            state = uo_states[idx]
            thresh = uo_thresholds[idx]
            if thresh > 0:
                if state == "CLOSED":
                    th_open[i] = thresh
                else:
                    th_closed[i] = thresh

        # Forward-fill gaps
        last_open = default_open
        last_closed = default_closed
        for i in range(n):
            if th_open[i] is not None:
                last_open = th_open[i]
            th_open[i] = last_open
            if th_closed[i] is not None:
                last_closed = th_closed[i]
            th_closed[i] = last_closed

        return th_open, th_closed

    def _update_plot(self):
        """Update the plot with current channel selection.

        Shows 4 subplots like offline_test.py:
        1. Raw EMG (selected channel) with predicted state overlay
        2. Distance to OPEN templates with threshold
        3. Distance to CLOSED templates with threshold
        4. Predicted State (0=OPEN, 1=CLOSED)
        """
        self.figure.clear()

        if self.history is None:
            return

        timestamps = np.array(self.history["timestamps"])
        D_open = self.history["D_open"]
        D_closed = self.history["D_closed"]
        states = self.history["states"]
        threshold_open = self.history["threshold_open"]
        threshold_closed = self.history["threshold_closed"]
        state_transitions = self.history.get("state_transitions", [])

        if len(timestamps) == 0:
            return

        # Normalize timestamps
        t0 = timestamps[0]
        timestamps_rel = timestamps - t0

        # Reconstruct per-step thresholds from unity_output if not in history
        if not self.history.get("thresholds_open_over_time") and self.unity_output:
            th_open_ot, th_closed_ot = self._reconstruct_thresholds_from_unity(
                self.unity_output, timestamps, threshold_open, threshold_closed
            )
            self.history["thresholds_open_over_time"] = th_open_ot
            self.history["thresholds_closed_over_time"] = th_closed_ot

        # Create 4 subplots (like offline_test.py)
        axes = self.figure.subplots(4, 1, sharex=True)

        # --- Plot 1: EMG Signal (selected channel) with state overlay ---
        ax1 = axes[0]
        if self.emg_signal is not None and self.emg_time_axis is not None:
            emg_time_rel = self.emg_time_axis - t0
            channel_data = self.emg_signal[self.current_channel, :]

            # Add predicted state as background shading
            states_numeric = np.array([1 if s == "CLOSED" else 0 for s in states])
            emg_max = np.max(np.abs(channel_data)) if len(channel_data) > 0 else 1

            # Interpolate states to EMG time resolution for shading
            if len(timestamps_rel) > 1:
                from scipy.interpolate import interp1d
                state_interp = interp1d(timestamps_rel, states_numeric, kind='previous',
                                        bounds_error=False, fill_value=(states_numeric[0], states_numeric[-1]))
                states_at_emg = state_interp(emg_time_rel)
                ax1.fill_between(emg_time_rel, -emg_max, emg_max,
                                where=states_at_emg > 0.5, alpha=0.15, color='purple', label='CLOSED')

            ax1.plot(emg_time_rel, channel_data, 'b-', linewidth=0.5, alpha=0.8)

        ax1.set_ylabel(f"EMG Ch{self.current_channel + 1} (uV)", fontsize=10)
        ax1.set_title(f"Online Session - Channel {self.current_channel + 1}", fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)

        # --- Plot 2: Distance to OPEN templates ---
        ax2 = axes[1]
        # Filter out None values
        valid_d_open = [(t, d) for t, d in zip(timestamps_rel, D_open) if d is not None]
        if valid_d_open:
            t_open = [x[0] for x in valid_d_open]
            d_open = [x[1] for x in valid_d_open]
            ax2.plot(t_open, d_open, 'g-', linewidth=1, label='D_open')
        # Threshold over time (if available) or flat line
        thresholds_open_ot = self.history.get("thresholds_open_over_time")
        if thresholds_open_ot and len(thresholds_open_ot) == len(timestamps_rel):
            ax2.plot(timestamps_rel, thresholds_open_ot, 'r--', linewidth=1.5, label='Threshold')
        else:
            ax2.axhline(threshold_open, color='r', linestyle='--', linewidth=2,
                       label=f'Threshold ({threshold_open:.3f})')
        ax2.set_ylabel("DTW Distance", fontsize=10)
        ax2.set_title("Distance to OPEN templates", fontsize=10)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # --- Plot 3: Distance to CLOSED templates ---
        ax3 = axes[2]
        # Filter out None values
        valid_d_closed = [(t, d) for t, d in zip(timestamps_rel, D_closed) if d is not None]
        if valid_d_closed:
            t_closed = [x[0] for x in valid_d_closed]
            d_closed = [x[1] for x in valid_d_closed]
            ax3.plot(t_closed, d_closed, 'orange', linewidth=1, label='D_closed')
        # Threshold over time (if available) or flat line
        thresholds_closed_ot = self.history.get("thresholds_closed_over_time")
        if thresholds_closed_ot and len(thresholds_closed_ot) == len(timestamps_rel):
            ax3.plot(timestamps_rel, thresholds_closed_ot, 'r--', linewidth=1.5, label='Threshold')
        else:
            ax3.axhline(threshold_closed, color='r', linestyle='--', linewidth=2,
                       label=f'Threshold ({threshold_closed:.3f})')
        ax3.set_ylabel("DTW Distance", fontsize=10)
        ax3.set_title("Distance to CLOSED templates", fontsize=10)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)

        # --- Plot 4: Predicted State ---
        ax4 = axes[3]
        states_numeric = [1 if s == "CLOSED" else 0 for s in states]
        ax4.step(timestamps_rel, states_numeric, 'purple', linewidth=2, where='post', label='Predicted')
        ax4.fill_between(timestamps_rel, 0, states_numeric, step='post', alpha=0.3, color='purple')

        # Mark state transitions with vertical lines
        for trans_time, from_state, to_state in state_transitions:
            trans_t = trans_time - t0
            if 0 <= trans_t <= timestamps_rel[-1]:
                color = 'green' if to_state == "OPEN" else 'orange'
                ax4.axvline(x=trans_t, color=color, linestyle='-', linewidth=1.5, alpha=0.7)

        ax4.set_ylabel("State", fontsize=10)
        ax4.set_xlabel("Time (seconds)", fontsize=10)
        ax4.set_title("Predicted State (0=OPEN, 1=CLOSED)", fontsize=10)
        ax4.set_ylim(-0.1, 1.1)
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['OPEN', 'CLOSED'])
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper right', fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw()

def _detect_mode_from_data(data: dict) -> bool | None:
    """
    Detect differential mode from loaded data dict.

    Returns:
        True if SD (differential), False if MP (monopolar), None if unknown.
    """
    # Check direct field
    mode = data.get('differential_mode')
    if mode is not None:
        return bool(mode)

    # Check metadata
    metadata = data.get('metadata', {})
    if isinstance(metadata, dict):
        mode = metadata.get('differential_mode')
        if mode is not None:
            return bool(mode)

    # Infer from EMG shape
    emg = data.get('emg')
    if emg is not None and hasattr(emg, 'shape') and len(emg.shape) >= 1:
        return emg.shape[0] <= 16

    return None


if TYPE_CHECKING:
    from mindmove.gui.mindmove import MindMove


class OnlineProtocol(QObject):
    def __init__(self, parent: MindMove | None = ...) -> None:
        super().__init__(parent)

        self.main_window: MindMove = parent

        # Initialize Protocol UI
        self._setup_protocol_ui()

        # Model Interface
        self.model_interface: MindMoveInterface = MindMoveInterface(
            parent=self.main_window,
            use_diagnostic=config.DIAGNOSTIC_MODE
            )
        self.model_label: str = None

        # Offline test state
        self.offline_test_recording = None
        self.offline_test_emg = None
        self.offline_test_gt = None  # Real GT from guided recording
        self.offline_test_ref_predictions = None  # Previous online predictions (numeric)
        self.offline_test_recording_type = None  # "guided", "prediction", "emg_only"
        self.offline_test_calibration_thresholds = None  # Calibration results (if GT available)

        # Buffers
        self.emg_buffer: list[np.ndarray] = []
        self.kinematics_buffer: list[list[float]] = []
        self.emg_timings_buffer: list[float] = []
        self.kinematics_timings_buffer: list[float] = []
        self.predictions_buffer: list[list[float]] = []
        self.unity_output_buffer: list[dict] = []  # Store Unity outputs with timestamps

        # File management
        self.prediction_dir_path: str = "data/predictions/"
        self.model_dir_path: str = "data/models/"

    def _validate_differential_mode(self, loaded_mode: bool) -> bool:
        """
        Check if loaded data's differential mode matches the current app config.

        If mismatch, show dialog offering to switch. If user accepts, toggles the
        device UI button (which triggers the full config + filter chain).

        Args:
            loaded_mode: True if loaded data is SD (differential), False if MP.

        Returns:
            True if modes match (or user switched), False if user declined (abort load).
        """
        if loaded_mode == config.ENABLE_DIFFERENTIAL_MODE:
            return True

        loaded_name = "Single Differential (16ch)" if loaded_mode else "Monopolar (32ch)"
        current_name = "Single Differential (16ch)" if config.ENABLE_DIFFERENTIAL_MODE else "Monopolar (32ch)"

        result = QMessageBox.question(
            self.main_window,
            "Differential Mode Mismatch",
            f"This data was recorded in <b>{loaded_name}</b> mode,\n"
            f"but the app is currently in <b>{current_name}</b> mode.\n\n"
            f"Switch to <b>{loaded_name}</b> to match?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if result == QMessageBox.Yes:
            self.main_window.device.differential_mode_button.setChecked(loaded_mode)
            print(f"[MODE] Auto-switched to {'SD' if loaded_mode else 'MP'} to match loaded data")
            return True
        else:
            print(f"[MODE] User declined mode switch — load aborted")
            return False

    def online_emg_update(self, data: np.ndarray) -> None:
        # Extract EMG data (filtering is applied here if enabled in MuoviWidget)
        emg_data = self.main_window.device.extract_emg_data(data)
        # shape (32, nsamp)

        # Get prediction from model
        prediction = self.model_interface.predict(emg_data)

        # Get extended result with distance and threshold info
        result = self.model_interface.get_last_result()

        if result:
            # Convert binary state to 10-joint format for Unity VHI
            # OPEN (state=0.0): all fingers extended [0,0,0,0,0,0,0,0,0,0]
            # CLOSED (state=1.0): all fingers closed [1,1,1,1,1,1,1,1,1,1]
            state = result['state']  # 0.0 or 1.0
            joint_value = int(state)  # 0 or 1
            unity_data = [joint_value] * 10  # All 10 joints same value

            self.main_window.virtual_hand_interface.output_message_signal.emit(
                str(unity_data).encode("utf-8")
            )
        else:
            # Fallback: send all zeros (OPEN state)
            unity_data = [0] * 10
            result = {'state': 0.0, 'distance': 0.0, 'threshold': 0.0, 'state_name': 'OPEN'}
            self.main_window.virtual_hand_interface.output_message_signal.emit(
                str(unity_data).encode("utf-8")
            )

        if self.online_record_toggle_push_button.isChecked():
            # Store with timestamp for proper reconstruction (same format as record protocol)
            self.emg_buffer.append((time.time(), emg_data))
            self.predictions_buffer.append(prediction)
            # Store Unity output with full context
            self.unity_output_buffer.append({
                'timestamp': time.time(),
                'joints': unity_data,
                'state': result['state'],
                'state_name': result['state_name'],
                'distance': result['distance'],
                'threshold': result['threshold'],
            })
            # Note: emg_timings_buffer is now redundant but kept for compatibility
            self.emg_timings_buffer.append(time.time())

    def online_kinematics_update(self, data: np.ndarray) -> None:
        if self.online_record_toggle_push_button.isChecked():
            self.kinematics_buffer.append(data)
            self.kinematics_timings_buffer.append(time.time())

    def _toggle_recording(self):
        if self.online_record_toggle_push_button.isChecked():
            # Starting recording
            self.timings = []
            self.online_record_toggle_push_button.setText("Stop Recording")
            self.online_load_model_push_button.setEnabled(False)

            # Reset model history for new session
            self.model_interface.reset_history()

            # connect signals
            self.main_window.device.ready_read_signal.connect(self.online_emg_update)

            self.emg_buffer = []
            self.kinematics_buffer = []
            self.emg_timings_buffer = []
            self.kinematics_timings_buffer = []
            self.predictions_buffer = []
            self.unity_output_buffer = []

            print("\n" + "=" * 70)
            print("RECORDING STARTED - History reset")
            print("=" * 70 + "\n")
        else:
            # Stopping recording
            self.online_record_toggle_push_button.setText("Start Recording")
            self.online_load_model_push_button.setEnabled(True)
            self.main_window.device.ready_read_signal.disconnect(self.online_emg_update)

            # Print diagnostic summary if in diagnostic mode
            if config.DIAGNOSTIC_MODE:
                print("\n" + "="*70)
                print("STOPPING RECORDING - PRINTING DIAGNOSTIC SUMMARY")
                print("="*70)
                self.model_interface.print_summary()

            # Plot distance history (if enabled)
            if self.plot_at_end_checkbox.isChecked():
                self._plot_distance_history()

            self._save_data()

    def _plot_distance_history(self) -> None:
        """
        Plot distance history after stopping recording using Qt dialog.

        Creates a 3-subplot figure:
        - Top: EMG signal (selectable channel with arrow keys)
        - Middle: State over time (0=OPEN, 1=CLOSED)
        - Bottom: Distance to opposite-state templates with threshold
        """
        history = self.model_interface.get_distance_history()

        if history is None or len(history.get("timestamps", [])) == 0:
            print("[PLOT] No distance history to plot")
            return

        timestamps = np.array(history["timestamps"])
        state_transitions = history.get("state_transitions", [])

        # Reconstruct continuous EMG signal from buffer
        if self.emg_buffer:
            emg_signal = np.hstack([emg_data for _, emg_data in self.emg_buffer])
            n_channels, n_samples = emg_signal.shape

            # Create continuous time axis for EMG
            emg_time_axis = np.zeros(n_samples)
            sample_idx = 0
            for i, (pkt_time, pkt_data) in enumerate(self.emg_buffer):
                pkt_samples = pkt_data.shape[1]
                if i == 0:
                    pkt_duration = pkt_samples / config.FSAMP
                    emg_time_axis[sample_idx:sample_idx + pkt_samples] = np.linspace(
                        pkt_time - pkt_duration, pkt_time, pkt_samples, endpoint=False
                    )
                else:
                    prev_time = self.emg_buffer[i-1][0]
                    emg_time_axis[sample_idx:sample_idx + pkt_samples] = np.linspace(
                        prev_time, pkt_time, pkt_samples, endpoint=False
                    )
                sample_idx += pkt_samples
        else:
            emg_signal = None
            emg_time_axis = None

        # Total duration
        t0 = timestamps[0] if len(timestamps) > 0 else 0
        total_duration = timestamps[-1] - t0 if len(timestamps) > 0 else 0

        print(f"\n[PLOT] Opening session review dialog for {total_duration:.1f}s recording")
        print(f"       DTW computations: {len(timestamps)}")
        print(f"       EMG samples: {emg_signal.shape[1] if emg_signal is not None else 0}")
        print(f"       State transitions: {len(state_transitions)}")

        # Open the review dialog
        dialog = OnlineSessionReviewDialog(emg_signal, emg_time_axis, history, self.main_window,
                                              unity_output=self.unity_output_buffer)
        dialog.exec()

    def _save_data(self) -> None:
        # Reconstruct EMG as (32, total_samples) - same format as record protocol
        if self.emg_buffer:
            # emg_buffer is list of (timestamp, emg_data) tuples
            emg_signal = np.hstack([emg_data for _, emg_data in self.emg_buffer])
            emg_timings = np.array([timestamp for timestamp, _ in self.emg_buffer])
        else:
            emg_signal = np.array([])
            emg_timings = np.array([])

        save_pickle_dict = {
            "emg": emg_signal,  # Shape: (32, total_samples) - consistent with record protocol
            "kinematics": np.array(self.kinematics_buffer) if self.kinematics_buffer else np.array([]),
            "timings_emg": emg_timings,
            "timings_kinematics": np.array(self.kinematics_timings_buffer),
            "predictions": self.predictions_buffer,
            "label": self.model_label,
            # Include distance history for offline analysis
            "distance_history": self.model_interface.get_distance_history(),
            # Unity/VHI output with full context (timestamp, joints, state, distance, threshold)
            "unity_output": self.unity_output_buffer,
        }

        now = datetime.now()
        formatted_now = now.strftime("%Y%m%d_%H%M%S%f")
        file_name = (
            f"MindMove_Predictions_{formatted_now}_{self.model_label}.pkl"
        )

        if not os.path.exists(self.prediction_dir_path):
            os.makedirs(self.prediction_dir_path)

        with open(os.path.join(self.prediction_dir_path, file_name), "wb") as f:
            pickle.dump(save_pickle_dict, f)

        print(f"[SAVE] Saved prediction data to {file_name}")
        print(f"       EMG shape: {emg_signal.shape if len(emg_signal) > 0 else 'empty'}")

        # Reset buffers
        self.emg_buffer = []
        self.kinematics_buffer = []
        self.emg_timings_buffer = []
        self.kinematics_timings_buffer = []
        self.predictions_buffer = []
        self.unity_output_buffer = []

    def _load_model(self) -> None:
        if config.DIAGNOSTIC_MODE:
            # In diagnostic mode, don't need to select a file
            self.model_interface.load_model(None)  # Loads DiagnosticModel
            self.online_model_label.setText("DIAGNOSTIC MODE (no model file)")
            self._update_threshold_display()
            # Offline test not available in diagnostic mode
            self.run_offline_test_button.setEnabled(False)
            return

        # Normal model loading code
        if not os.path.exists(self.model_dir_path):
            os.makedirs(self.model_dir_path)

        dialog = QFileDialog(self.main_window)
        dialog.setFileMode(QFileDialog.ExistingFile)

        file_name = dialog.getOpenFileName(
            self.main_window,
            "Open Model",
            self.model_dir_path,
        )[0]

        if not file_name:
            return

        self.model_interface.load_model(file_name)

        # Validate differential mode
        model_mode = getattr(self.model_interface.model, 'differential_mode', None)
        if model_mode is not None:
            if not self._validate_differential_mode(model_mode):
                # User declined switch — unload model
                self.model_interface.model = None
                self.model_interface.model_is_loaded = False
                self.online_model_label.setText("Load aborted (mode mismatch)")
                return

        label = file_name.split("/")[-1].split("_")[-1].split(".")[0]
        self.online_model_label.setText(f"{label} loaded.")

        # Update spatial correction availability
        model = self.model_interface.model
        has_spatial = (model is not None
                       and model.spatial_ref_open is not None
                       and model.spatial_ref_closed is not None)
        self.spatial_correction_checkbox.setEnabled(has_spatial)
        if has_spatial:
            self.spatial_status_label.setText("(available)")
            self.spatial_status_label.setStyleSheet("color: #2a7;")
            # Sync threshold slider with model
            self.spatial_threshold_slider.setValue(int(model.spatial_threshold * 100))
        else:
            self.spatial_status_label.setText("(not available — legacy model)")
            self.spatial_status_label.setStyleSheet("color: #999;")
            self.spatial_correction_checkbox.setChecked(False)

        # Update threshold display after loading
        self._update_threshold_display()

        # Enable offline test button if recording is also loaded
        if self.offline_test_emg is not None:
            self.run_offline_test_button.setEnabled(True)

    def _update_threshold_display(self) -> None:
        """Update the threshold sliders and labels after model load."""
        thresholds = self.model_interface.get_current_thresholds()
        if thresholds and thresholds.get("threshold_open") is not None:
            # Get model statistics
            mean_open = thresholds.get("mean_open", 0)
            std_open = thresholds.get("std_open", 0.1)
            mean_closed = thresholds.get("mean_closed", 0)
            std_closed = thresholds.get("std_closed", 0.1)

            threshold_open = thresholds["threshold_open"]
            threshold_closed = thresholds["threshold_closed"]
            s_open = thresholds.get("s_open", 1.0)
            s_closed = thresholds.get("s_closed", 1.0)

            # Calculate max threshold for sliders (mean + 5*std to give enough range)
            self._max_threshold_open = mean_open + 5 * std_open
            self._max_threshold_closed = mean_closed + 5 * std_closed

            # Ensure max is at least as large as current threshold
            self._max_threshold_open = max(self._max_threshold_open, threshold_open * 1.5)
            self._max_threshold_closed = max(self._max_threshold_closed, threshold_closed * 1.5)

            # Update slider range labels
            self._open_max_label.setText(f"{self._max_threshold_open:.3f}")
            self._closed_max_label.setText(f"{self._max_threshold_closed:.3f}")

            # Update slider positions to match current threshold values
            self.threshold_slider_open.blockSignals(True)
            slider_val_open = int((threshold_open / self._max_threshold_open) * self._slider_resolution)
            self.threshold_slider_open.setValue(min(slider_val_open, self._slider_resolution))
            self.threshold_slider_open.blockSignals(False)

            self.threshold_slider_closed.blockSignals(True)
            slider_val_closed = int((threshold_closed / self._max_threshold_closed) * self._slider_resolution)
            self.threshold_slider_closed.setValue(min(slider_val_closed, self._slider_resolution))
            self.threshold_slider_closed.blockSignals(False)

            # Update labels with threshold and s values
            self.threshold_open_label.setText(
                f"OPEN: {threshold_open:.4f} (s={s_open:.2f})"
            )
            self.threshold_closed_label.setText(
                f"CLOSED: {threshold_closed:.4f} (s={s_closed:.2f})"
            )

            print(f"[THRESHOLD UI] OPEN: {threshold_open:.4f} (s={s_open:.2f}), range: 0 - {self._max_threshold_open:.4f}")
            print(f"[THRESHOLD UI] CLOSED: {threshold_closed:.4f} (s={s_closed:.2f}), range: 0 - {self._max_threshold_closed:.4f}")

        # Update preset dropdown
        self._update_preset_dropdown()

    def _update_preset_dropdown(self) -> None:
        """Update the preset dropdown based on loaded model's presets."""
        # Block signals while updating combo box
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()

        if self.model_interface.has_threshold_presets():
            presets = self.model_interface.get_available_presets()

            # Add placeholder
            self.preset_combo.addItem("-- Select Preset --", None)

            # Add presets in a logical order
            preset_order = ["mid_gap", "current", "cross_class", "safety_margin", "conservative"]
            for key in preset_order:
                if key in presets:
                    preset = presets[key]
                    name = preset.get("name", key)
                    self.preset_combo.addItem(name, key)

            # Add any other presets not in the standard order
            for key, preset in presets.items():
                if key not in preset_order:
                    name = preset.get("name", key)
                    self.preset_combo.addItem(name, key)

            self.preset_combo.setEnabled(True)
            self.preset_description_label.setText("Select a preset to auto-configure thresholds")
            self.preset_description_label.setVisible(True)
            print(f"[PRESET UI] Loaded {len(presets)} presets: {list(presets.keys())}")
        else:
            # Legacy model without presets
            self.preset_combo.addItem("No presets (legacy model)", None)
            self.preset_combo.setEnabled(False)
            self.preset_description_label.setText("This model was created without threshold presets. Use sliders for manual tuning.")
            self.preset_description_label.setVisible(True)
            print("[PRESET UI] Legacy model - no presets available")

        self.preset_combo.blockSignals(False)

    def _setup_protocol_ui(self) -> None:
        self.online_load_model_group_box = self.main_window.ui.onlineLoadModelGroupBox

        self.online_load_model_push_button = (
            self.main_window.ui.onlineLoadModelPushButton
        )
        self.online_load_model_push_button.clicked.connect(self._load_model)
        self.online_model_label = self.main_window.ui.onlineModelLabel
        self.online_model_label.setText("No model loaded!")

        self.online_commands_group_box = self.main_window.ui.onlineCommandsGroupBox
        self.online_record_toggle_push_button = (
            self.main_window.ui.onlineRecordTogglePushButton
        )
        self.online_record_toggle_push_button.clicked.connect(self._toggle_recording)

        # Add unified offline test UI (programmatically)
        self._setup_offline_test_ui()

        # Add threshold tuning slider (programmatically)
        self._setup_threshold_slider()

        # Add refractory period control (programmatically)
        self._setup_refractory_control()

        # Add spatial correction controls
        self._setup_spatial_correction_ui()

        # Add plot options checkbox
        self._setup_plot_options()

    def _setup_offline_test_ui(self) -> None:
        """Setup unified offline test UI (replaces separate calibration + simulation)."""
        self.offline_test_group_box = QGroupBox("Offline Test")
        layout = QVBoxLayout()

        # Recording selection row
        rec_layout = QHBoxLayout()
        self.offline_test_load_button = QPushButton("Load Recording")
        self.offline_test_load_button.clicked.connect(self._load_offline_test_recording)
        rec_layout.addWidget(self.offline_test_load_button)

        self.offline_test_recording_label = QLabel("No recording loaded")
        self.offline_test_recording_label.setStyleSheet("color: #666;")
        rec_layout.addWidget(self.offline_test_recording_label)
        rec_layout.addStretch()
        layout.addLayout(rec_layout)

        # Run button
        self.run_offline_test_button = QPushButton("Run Offline Test")
        self.run_offline_test_button.setEnabled(False)
        self.run_offline_test_button.clicked.connect(self._run_offline_test)
        layout.addWidget(self.run_offline_test_button)

        # Calibration results display (only visible when GT available and test completed)
        self.calibration_results_label = QLabel("")
        self.calibration_results_label.setStyleSheet(
            "font-family: monospace; color: #333333; background-color: #f5f5f5; padding: 8px; border-radius: 4px;"
        )
        self.calibration_results_label.setWordWrap(True)
        self.calibration_results_label.setVisible(False)
        layout.addWidget(self.calibration_results_label)

        # Apply calibration thresholds button (only visible when calibration available)
        self.apply_calibration_button = QPushButton("Apply Calibration Thresholds to Sliders")
        self.apply_calibration_button.setEnabled(False)
        self.apply_calibration_button.setVisible(False)
        self.apply_calibration_button.clicked.connect(self._apply_calibration)
        layout.addWidget(self.apply_calibration_button)

        self.offline_test_group_box.setLayout(layout)

        # Add to the online commands group box layout
        if self.online_commands_group_box.layout():
            self.online_commands_group_box.layout().addWidget(self.offline_test_group_box)

    def _load_offline_test_recording(self) -> None:
        """Load a recording for offline test (auto-detects type: guided/prediction/emg_only)."""
        # Try recordings dir first, then predictions dir
        start_dir = "data/recordings/"
        if not os.path.exists(start_dir):
            start_dir = "data/predictions/"
        if not os.path.exists(start_dir):
            start_dir = "."

        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Select Recording for Offline Test",
            start_dir,
            "Pickle files (*.pkl)"
        )
        if not file_path:
            return

        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            emg = data.get('emg', data.get('biosignal'))
            if emg is None or (hasattr(emg, 'size') and emg.size == 0):
                QMessageBox.warning(self.main_window, "Invalid Recording",
                                    "No EMG data found in this file.", QMessageBox.Ok)
                return

            # Validate differential mode
            rec_mode = _detect_mode_from_data(data)
            if rec_mode is not None:
                if not self._validate_differential_mode(rec_mode):
                    return  # User declined switch — abort load

            # Detect recording type
            gt = data.get('gt')
            predictions = data.get('predictions')

            if gt is not None:
                recording_type = "guided"
                gt_arr = gt
                if hasattr(gt_arr, 'flatten'):
                    gt_arr = gt_arr.flatten()
                self.offline_test_gt = np.array(gt_arr, dtype=float)
                self.offline_test_ref_predictions = None
            elif predictions is not None and len(predictions) > 0:
                recording_type = "prediction"
                self.offline_test_gt = None
                # Convert string predictions to numeric
                if isinstance(predictions[0], str):
                    self.offline_test_ref_predictions = np.array(
                        [1.0 if p == "CLOSED" else 0.0 for p in predictions]
                    )
                else:
                    self.offline_test_ref_predictions = np.array(predictions, dtype=float)
            else:
                recording_type = "emg_only"
                self.offline_test_gt = None
                self.offline_test_ref_predictions = None

            self.offline_test_recording = data
            self.offline_test_emg = emg
            self.offline_test_recording_type = recording_type
            self.offline_test_calibration_thresholds = None  # Reset previous calibration

            # Update UI
            n_ch, n_samples = emg.shape
            duration_s = n_samples / config.FSAMP
            basename = os.path.basename(file_path)

            type_labels = {
                "guided": "Guided (GT)",
                "prediction": "Prediction",
                "emg_only": "EMG only",
            }
            type_label = type_labels.get(recording_type, "Unknown")

            self.offline_test_recording_label.setText(
                f"{basename} ({n_ch}ch, {duration_s:.1f}s) — {type_label}"
            )
            self.offline_test_recording_label.setStyleSheet("color: #333;")

            # Hide previous calibration results
            self.calibration_results_label.setVisible(False)
            self.apply_calibration_button.setVisible(False)
            self.apply_calibration_button.setEnabled(False)

            # Enable run button only if model is loaded
            model_loaded = (self.model_interface is not None and
                            self.model_interface.model_is_loaded)
            self.run_offline_test_button.setEnabled(model_loaded)

            if not model_loaded:
                self.offline_test_recording_label.setText(
                    self.offline_test_recording_label.text() + " (load model first)"
                )

            print(f"[OFFLINE TEST] Loaded recording: {basename}")
            print(f"  EMG shape: {emg.shape}, duration: {duration_s:.1f}s")
            print(f"  Type: {recording_type}")
            if self.offline_test_gt is not None:
                print(f"  GT: {len(self.offline_test_gt)} samples")
            if self.offline_test_ref_predictions is not None:
                print(f"  Previous predictions: {len(self.offline_test_ref_predictions)} values")

        except Exception as e:
            QMessageBox.critical(self.main_window, "Error",
                                 f"Failed to load recording:\n{e}", QMessageBox.Ok)
            import traceback
            traceback.print_exc()

    def _run_offline_test(self) -> None:
        """Run offline test: simulation (always) + calibration thresholds (if GT available)."""
        if self.offline_test_emg is None:
            return
        if self.model_interface is None or self.model_interface.model is None:
            QMessageBox.warning(self.main_window, "No Model",
                                "Please load a model first.", QMessageBox.Ok)
            return

        self.run_offline_test_button.setEnabled(False)
        self.run_offline_test_button.setText("Running offline test...")

        model = self.model_interface.model

        self._offline_test_thread = QThread()
        self._offline_test_worker = SimulationWorker(
            self.offline_test_emg, model, gt_data=self.offline_test_gt
        )
        self._offline_test_worker.moveToThread(self._offline_test_thread)
        self._offline_test_thread.started.connect(self._offline_test_worker.run)
        self._offline_test_worker.finished.connect(self._on_offline_test_finished)
        self._offline_test_worker.error.connect(self._on_offline_test_error)
        self._offline_test_worker.finished.connect(self._offline_test_thread.quit)
        self._offline_test_worker.error.connect(self._offline_test_thread.quit)
        self._offline_test_thread.finished.connect(self._offline_test_thread.deleteLater)
        self._offline_test_thread.start()

    def _on_offline_test_finished(self, results: dict, calibration_thresholds) -> None:
        """Handle offline test completion."""
        self.run_offline_test_button.setEnabled(True)
        self.run_offline_test_button.setText("Run Offline Test")

        n_dtw = len(results['timestamps'])
        preds = results['predictions']
        n_closed = sum(1 for p in preds if p == "CLOSED")
        n_open = sum(1 for p in preds if p == "OPEN")
        avg_time = results.get('avg_dtw_time', 0)

        print(f"\n[OFFLINE TEST] Complete: {n_dtw} DTW steps, avg {avg_time:.2f}ms")
        print(f"  Predictions: {n_closed} CLOSED, {n_open} OPEN")

        # Store calibration thresholds if available
        self.offline_test_calibration_thresholds = calibration_thresholds

        # Show calibration results UI if GT was available
        if calibration_thresholds is not None:
            ct = calibration_thresholds
            self.calibration_results_label.setText(
                f"OPEN plateau:     {ct['open_plateau']:.4f}    "
                f"threshold: {ct['threshold_open']:.4f} (plateau + 1.0 x std)\n"
                f"CLOSED plateau:   {ct['closed_plateau']:.4f}    "
                f"threshold: {ct['threshold_closed']:.4f} (plateau + 1.0 x std)"
            )
            self.calibration_results_label.setStyleSheet(
                "font-family: monospace; color: #333333; background-color: #f5f5f5; padding: 8px; border-radius: 4px;"
            )
            self.calibration_results_label.setVisible(True)
            self.apply_calibration_button.setVisible(True)
            self.apply_calibration_button.setEnabled(True)
        else:
            self.calibration_results_label.setVisible(False)
            self.apply_calibration_button.setVisible(False)
            self.apply_calibration_button.setEnabled(False)

        # Get model thresholds for comparison in the review dialog
        model_thresholds = None
        thresholds_data = self.model_interface.get_current_thresholds()
        if thresholds_data and thresholds_data.get("threshold_open") is not None:
            model_thresholds = {
                "threshold_open": thresholds_data["threshold_open"],
                "threshold_closed": thresholds_data["threshold_closed"],
            }

        # Show review dialog
        self._offline_test_review_dialog = OfflineTestReviewDialog(
            emg_data=self.offline_test_emg,
            results=results,
            gt_data=self.offline_test_gt,
            ref_predictions=self.offline_test_ref_predictions,
            calibration_thresholds=calibration_thresholds,
            model_thresholds=model_thresholds,
            recording_type=self.offline_test_recording_type or "unknown",
            parent=self.main_window,
        )
        self._offline_test_review_dialog.show()

    def _on_offline_test_error(self, error_msg: str) -> None:
        """Handle offline test error."""
        self.run_offline_test_button.setEnabled(True)
        self.run_offline_test_button.setText("Run Offline Test")
        self.calibration_results_label.setText(f"Error: {error_msg}")
        self.calibration_results_label.setStyleSheet(
            "font-family: monospace; color: red; background-color: #fff0f0; padding: 8px; border-radius: 4px;"
        )
        self.calibration_results_label.setVisible(True)
        QMessageBox.critical(self.main_window, "Offline Test Error",
                             f"Offline test failed:\n{error_msg}", QMessageBox.Ok)
        print(f"[OFFLINE TEST ERROR] {error_msg}")

    def _apply_calibration(self) -> None:
        """Apply calibrated thresholds to sliders."""
        if not self.offline_test_calibration_thresholds:
            return

        threshold_open = self.offline_test_calibration_thresholds['threshold_open']
        threshold_closed = self.offline_test_calibration_thresholds['threshold_closed']

        # Set thresholds directly
        self.model_interface.set_threshold_open_direct(threshold_open)
        self.model_interface.set_threshold_closed_direct(threshold_closed)

        # Update slider display
        self._update_threshold_display()

        print(f"\n[CALIBRATION] Applied thresholds:")
        print(f"  OPEN:   {threshold_open:.4f}")
        print(f"  CLOSED: {threshold_closed:.4f}\n")

    def _setup_threshold_slider(self) -> None:
        """Setup threshold tuning sliders UI (direct threshold control)."""
        # Create a group box for threshold tuning
        self.threshold_group_box = QGroupBox("Threshold Tuning")

        # Store max threshold values (will be updated when model is loaded)
        self._max_threshold_open = 1.0  # Default, updated on model load
        self._max_threshold_closed = 1.0  # Default, updated on model load
        self._slider_resolution = 1000  # Slider units per threshold range

        main_layout = QVBoxLayout()

        # --- Threshold Preset Dropdown ---
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Threshold Preset:")
        preset_layout.addWidget(preset_label)

        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(200)
        self.preset_combo.addItem("-- Select Preset --", None)
        self.preset_combo.setEnabled(False)  # Disabled until model with presets is loaded
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        preset_layout.addWidget(self.preset_combo)

        preset_layout.addStretch()
        main_layout.addLayout(preset_layout)

        # Preset description label
        self.preset_description_label = QLabel("")
        self.preset_description_label.setStyleSheet(
            "color: #666; background-color: #f5f5f5; padding: 6px; border-radius: 4px; font-style: italic;"
        )
        self.preset_description_label.setWordWrap(True)
        self.preset_description_label.setVisible(False)  # Hidden until preset is selected
        main_layout.addWidget(self.preset_description_label)

        main_layout.addSpacing(10)

        # --- OPEN threshold slider ---
        open_label = QLabel("OPEN threshold (distance to open templates):")
        main_layout.addWidget(open_label)

        open_slider_layout = QHBoxLayout()
        self._open_min_label = QLabel("0")
        open_slider_layout.addWidget(self._open_min_label)

        self.threshold_slider_open = QSlider(Qt.Horizontal)
        self.threshold_slider_open.setMinimum(0)
        self.threshold_slider_open.setMaximum(self._slider_resolution)
        self.threshold_slider_open.setValue(self._slider_resolution // 2)  # Default to middle
        self.threshold_slider_open.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider_open.setTickInterval(self._slider_resolution // 10)
        self.threshold_slider_open.valueChanged.connect(self._on_threshold_open_changed)
        open_slider_layout.addWidget(self.threshold_slider_open)

        self._open_max_label = QLabel("1.0")
        open_slider_layout.addWidget(self._open_max_label)
        main_layout.addLayout(open_slider_layout)

        self.threshold_open_label = QLabel("Threshold OPEN: Not loaded")
        main_layout.addWidget(self.threshold_open_label)

        # --- CLOSED threshold slider ---
        closed_label = QLabel("CLOSED threshold (distance to closed templates):")
        main_layout.addWidget(closed_label)

        closed_slider_layout = QHBoxLayout()
        self._closed_min_label = QLabel("0")
        closed_slider_layout.addWidget(self._closed_min_label)

        self.threshold_slider_closed = QSlider(Qt.Horizontal)
        self.threshold_slider_closed.setMinimum(0)
        self.threshold_slider_closed.setMaximum(self._slider_resolution)
        self.threshold_slider_closed.setValue(self._slider_resolution // 2)  # Default to middle
        self.threshold_slider_closed.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider_closed.setTickInterval(self._slider_resolution // 10)
        self.threshold_slider_closed.valueChanged.connect(self._on_threshold_closed_changed)
        closed_slider_layout.addWidget(self.threshold_slider_closed)

        self._closed_max_label = QLabel("1.0")
        closed_slider_layout.addWidget(self._closed_max_label)
        main_layout.addLayout(closed_slider_layout)

        self.threshold_closed_label = QLabel("Threshold CLOSED: Not loaded")
        main_layout.addWidget(self.threshold_closed_label)

        self.threshold_group_box.setLayout(main_layout)

        # Add to the online commands group box layout
        if self.online_commands_group_box.layout():
            self.online_commands_group_box.layout().addWidget(self.threshold_group_box)

    def _setup_refractory_control(self) -> None:
        """Setup refractory period control UI."""
        # Create a group box for refractory period
        self.refractory_group_box = QGroupBox("Refractory Period")

        layout = QHBoxLayout()

        # Label
        label = QLabel("Period after state change (no transitions allowed):")
        layout.addWidget(label)

        # Spinbox for refractory period (0.0 to 5.0 seconds)
        self.refractory_spinbox = QDoubleSpinBox()
        self.refractory_spinbox.setMinimum(0.0)
        self.refractory_spinbox.setMaximum(5.0)
        self.refractory_spinbox.setSingleStep(0.1)
        self.refractory_spinbox.setValue(1.0)  # Default 1 second
        self.refractory_spinbox.setSuffix(" s")
        self.refractory_spinbox.setDecimals(1)
        self.refractory_spinbox.valueChanged.connect(self._on_refractory_changed)
        layout.addWidget(self.refractory_spinbox)

        self.refractory_group_box.setLayout(layout)

        # Add to the online commands group box layout
        if self.online_commands_group_box.layout():
            self.online_commands_group_box.layout().addWidget(self.refractory_group_box)

    def _on_refractory_changed(self, value: float) -> None:
        """Called when refractory period spinbox value changes."""
        self.model_interface.set_refractory_period(value)

    def _setup_spatial_correction_ui(self) -> None:
        """Setup spatial correction controls (consistency-weighted spatial match)."""
        self.spatial_group_box = QGroupBox("Spatial Correction")

        layout = QVBoxLayout()

        # Enable checkbox
        row1 = QHBoxLayout()
        self.spatial_correction_checkbox = QCheckBox("Enable spatial correction")
        self.spatial_correction_checkbox.setChecked(False)
        self.spatial_correction_checkbox.setToolTip(
            "When enabled, state transitions require both DTW distance < threshold\n"
            "AND spatial pattern similarity > spatial threshold.\n"
            "Helps reject co-contractions and false triggers."
        )
        self.spatial_correction_checkbox.stateChanged.connect(self._on_spatial_correction_toggled)
        row1.addWidget(self.spatial_correction_checkbox)

        self.spatial_status_label = QLabel("(not available)")
        self.spatial_status_label.setStyleSheet("color: #999;")
        row1.addWidget(self.spatial_status_label)
        row1.addStretch()
        layout.addLayout(row1)

        # Threshold slider
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Threshold:"))

        self.spatial_threshold_slider = QSlider(Qt.Horizontal)
        self.spatial_threshold_slider.setMinimum(0)
        self.spatial_threshold_slider.setMaximum(100)
        self.spatial_threshold_slider.setValue(50)  # Default 0.50
        self.spatial_threshold_slider.setToolTip(
            "Minimum spatial similarity required to allow a state transition.\n"
            "Higher = more strict (fewer false triggers, but may miss real ones).\n"
            "Lower = more permissive."
        )
        self.spatial_threshold_slider.valueChanged.connect(self._on_spatial_threshold_changed)
        row2.addWidget(self.spatial_threshold_slider)

        self.spatial_threshold_label = QLabel("0.50")
        self.spatial_threshold_label.setMinimumWidth(40)
        row2.addWidget(self.spatial_threshold_label)
        layout.addLayout(row2)

        self.spatial_group_box.setLayout(layout)

        # Add to the online commands group box layout
        if self.online_commands_group_box.layout():
            self.online_commands_group_box.layout().addWidget(self.spatial_group_box)

    def _on_spatial_correction_toggled(self, state: int) -> None:
        """Toggle spatial correction on/off."""
        enabled = state == Qt.Checked.value if hasattr(Qt.Checked, 'value') else state == 2
        if self.model_interface.model is not None:
            self.model_interface.model.use_spatial_correction = enabled
            status = "ON" if enabled else "OFF"
            print(f"[SPATIAL] Spatial correction: {status}")
        else:
            if enabled:
                self.spatial_correction_checkbox.setChecked(False)
                QMessageBox.information(self.main_window, "No Model",
                                        "Load a model first before enabling spatial correction.")

    def _on_spatial_threshold_changed(self, value: int) -> None:
        """Update spatial similarity threshold from slider."""
        threshold = value / 100.0
        self.spatial_threshold_label.setText(f"{threshold:.2f}")
        if self.model_interface.model is not None:
            self.model_interface.model.spatial_threshold = threshold
            print(f"[SPATIAL] Threshold: {threshold:.2f}")

    def _setup_plot_options(self) -> None:
        """Setup plot options UI."""
        # Create a group box for plot options
        self.plot_options_group_box = QGroupBox("Recording Options")

        layout = QHBoxLayout()

        # Checkbox for plotting at end
        self.plot_at_end_checkbox = QCheckBox("Show plot after recording")
        self.plot_at_end_checkbox.setChecked(True)  # Default: show plot
        self.plot_at_end_checkbox.setToolTip(
            "When enabled, displays distance history plot after stopping recording.\n"
            "Disable for long recordings to avoid popup."
        )
        layout.addWidget(self.plot_at_end_checkbox)

        layout.addStretch()

        self.plot_options_group_box.setLayout(layout)

        # Add to the online commands group box layout
        if self.online_commands_group_box.layout():
            self.online_commands_group_box.layout().addWidget(self.plot_options_group_box)

    def _on_threshold_open_changed(self, value: int) -> None:
        """Called when OPEN threshold slider is moved (direct threshold control)."""
        # Convert slider value to threshold (0 to max_threshold)
        threshold = (value / self._slider_resolution) * self._max_threshold_open

        # Update model threshold directly
        self.model_interface.set_threshold_open_direct(threshold)

        # Update label with threshold and computed s
        thresholds = self.model_interface.get_current_thresholds()
        if thresholds and thresholds.get("threshold_open") is not None:
            s_open = thresholds.get("s_open", 0)
            self.threshold_open_label.setText(
                f"Threshold OPEN: {threshold:.4f} (s={s_open:.2f})"
            )
        else:
            self.threshold_open_label.setText(f"Threshold OPEN: {threshold:.4f}")

    def _on_threshold_closed_changed(self, value: int) -> None:
        """Called when CLOSED threshold slider is moved (direct threshold control)."""
        # Convert slider value to threshold (0 to max_threshold)
        threshold = (value / self._slider_resolution) * self._max_threshold_closed

        # Update model threshold directly
        self.model_interface.set_threshold_closed_direct(threshold)

        # Update label with threshold and computed s
        thresholds = self.model_interface.get_current_thresholds()
        if thresholds and thresholds.get("threshold_closed") is not None:
            s_closed = thresholds.get("s_closed", 0)
            self.threshold_closed_label.setText(
                f"Threshold CLOSED: {threshold:.4f} (s={s_closed:.2f})"
            )
        else:
            self.threshold_closed_label.setText(f"Threshold CLOSED: {threshold:.4f}")

    def _on_preset_selected(self, index: int) -> None:
        """Called when a preset is selected from the dropdown."""
        preset_key = self.preset_combo.currentData()

        if preset_key is None:
            # "Select Preset" placeholder selected
            self.preset_description_label.setVisible(False)
            return

        # Apply the preset
        success = self.model_interface.apply_threshold_preset(preset_key)

        if success:
            # Update sliders to reflect the new threshold values
            self._update_threshold_display()

            # Show preset description
            presets = self.model_interface.get_available_presets()
            if presets and preset_key in presets:
                preset = presets[preset_key]
                description = preset.get("description", "")
                self.preset_description_label.setText(description)
                self.preset_description_label.setVisible(True)
        else:
            self.preset_description_label.setText("Failed to apply preset")
            self.preset_description_label.setVisible(True)
