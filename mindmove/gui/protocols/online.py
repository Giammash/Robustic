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


class RealtimePlaybackWorker(QObject):
    """Streams a recorded EMG file chunk-by-chunk at real-time (or scaled) speed.

    Each chunk is emitted via chunk_ready so the main thread can feed it to the
    model and the VisPy plot, exactly like live data.
    """
    chunk_ready = Signal(np.ndarray)  # (n_ch, chunk_size) pre-filtered EMG
    progress    = Signal(float)       # current playback time in seconds
    finished    = Signal()
    error       = Signal(str)

    def __init__(self, emg_data: np.ndarray, speed: float = 1.0, chunk_size: int = 64):
        super().__init__()
        self.emg_data   = emg_data          # (n_ch, n_total_samples) already filtered
        self.speed      = max(speed, 0.01)
        self.chunk_size = chunk_size
        self._stop      = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            n_ch, n_total = self.emg_data.shape
            sleep_s = self.chunk_size / config.FSAMP / self.speed

            i = 0
            while i < n_total and not self._stop:
                chunk = self.emg_data[:, i:i + self.chunk_size]
                if chunk.shape[1] == 0:
                    break
                self.chunk_ready.emit(chunk)
                self.progress.emit(i / config.FSAMP)
                i += self.chunk_size
                time.sleep(sleep_s)

            if not self._stop:
                self.finished.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class SimulationWorker(QObject):
    """Worker class for running offline simulation in a separate thread."""
    finished = Signal(dict, object)  # results dict + calibration_thresholds (dict or None)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, emg_data, model, initial_state="CLOSED", gt_data=None):
        super().__init__()
        self.emg_data = emg_data
        self.model = model
        self.initial_state = initial_state
        self.gt_data = gt_data  # Optional real GT (at FSAMP rate) for plateau calibration

    def run(self):
        """Run offline simulation using the loaded model's parameters."""
        try:
            from mindmove.model.offline_test import simulate_realtime_dtw

            templates_open = self.model.templates_open
            templates_closed = self.model.templates_closed

            if templates_open is None or templates_closed is None:
                self.error.emit(
                    "Model has no templates loaded (templates_open or templates_closed is None). "
                    "Please load a valid model file."
                )
                return
            threshold_open = self.model.THRESHOLD_OPEN
            threshold_closed = self.model.THRESHOLD_CLOSED
            feature_name = self.model.feature_name
            distance_aggregation = self.model.distance_aggregation

            # Spatial correction settings
            spatial_mode = getattr(self.model, 'spatial_mode', 'off')
            spatial_ref_open = getattr(self.model, 'spatial_ref_open', None)
            spatial_ref_closed = getattr(self.model, 'spatial_ref_closed', None)
            spatial_threshold = getattr(self.model, 'spatial_threshold', 0.5)
            spatial_sharpness = getattr(self.model, 'spatial_sharpness', 3.0)
            spatial_relu_baseline = getattr(self.model, 'spatial_relu_baseline', 0.2)
            spatial_similarity_mode = getattr(self.model, 'spatial_similarity_mode', 'mean')
            spatial_n_best = getattr(self.model, 'spatial_n_best', 3)
            spatial_coupled = getattr(self.model, 'spatial_coupled', False)

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
            spatial_info = ""
            if spatial_mode in ("gate", "relu_scaling", "relu_contrast", "relu_ext_scaling", "relu_ext_contrast"):
                spatial_info = f" (threshold={spatial_threshold:.2f}, b={spatial_relu_baseline:.2f}, k={spatial_sharpness:.1f})"
            elif spatial_mode in ("scaling", "contrast"):
                spatial_info = f" (k={spatial_sharpness:.1f})"
            print(f"  Spatial mode: {spatial_mode}{spatial_info}")
            print(f"  Initial state: {self.initial_state}")

            # Decision mode
            decision_mode = getattr(self.model, 'decision_mode', 'threshold')
            decision_nn_weights = None
            decision_catboost_model = None
            if decision_mode == "nn" and self.model.decision_nn is not None:
                decision_nn_weights = self.model.decision_nn.get_weights_dict()
                print(f"  Decision mode: nn (accuracy={self.model.decision_nn.accuracy:.1%})")
            elif decision_mode == "catboost" and self.model.decision_catboost is not None:
                decision_catboost_model = self.model.decision_catboost.get_model_dict()
                print(f"  Decision mode: catboost (accuracy={self.model.decision_catboost.accuracy:.1%})")
            else:
                print(f"  Decision mode: threshold")

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
                spatial_mode=spatial_mode,
                spatial_sharpness=spatial_sharpness,
                spatial_relu_baseline=spatial_relu_baseline,
                initial_state=self.initial_state,
                decision_mode=decision_mode,
                decision_nn_weights=decision_nn_weights,
                decision_catboost_model=decision_catboost_model,
                spatial_similarity_mode=spatial_similarity_mode,
                spatial_n_best=spatial_n_best,
                spatial_coupled=spatial_coupled,
            )

            # --- Optional calibration: find plateau thresholds from GT ---
            calibration_thresholds = None
            if self.gt_data is not None and len(results.get('D_open', [])) > 0:
                try:
                    from mindmove.model.core.algorithm import find_plateau_thresholds
                    timestamps = results['timestamps']
                    gt_arr = np.array(self.gt_data, dtype=float).flatten()
                    gt_time = np.arange(len(gt_arr)) / config.FSAMP
                    gt_at_dtw = np.interp(timestamps, gt_time, gt_arr)
                    gt_at_dtw = np.clip(np.round(gt_at_dtw), 0, 1)
                    calibration_thresholds = find_plateau_thresholds(
                        D_open=np.array(results['D_open']),
                        D_closed=np.array(results['D_closed']),
                        gt=gt_at_dtw,
                    )
                    print(f"[CALIBRATION] OPEN plateau={calibration_thresholds['open_plateau']:.4f}, "
                          f"thr={calibration_thresholds['threshold_open']:.4f}")
                    print(f"[CALIBRATION] CLOSED plateau={calibration_thresholds['closed_plateau']:.4f}, "
                          f"thr={calibration_thresholds['threshold_closed']:.4f}")
                except Exception as cal_err:
                    print(f"[CALIBRATION] Error computing calibration thresholds: {cal_err}")

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
                 recording_type="unknown", time_offset=0.0, calibration_thresholds=None,
                 parent=None):
        super().__init__(parent)
        self.emg_data = emg_data
        self.results = results
        self.gt_data = gt_data  # Real GT from guided recording (array at FSAMP rate)
        self.ref_predictions = ref_predictions  # Previous online predictions (numeric array at DTW rate)
        self.recording_type = recording_type
        self.time_offset = time_offset  # Seconds to add to all timestamps (for sliced recordings)
        self.calibration_thresholds = calibration_thresholds  # From find_plateau_thresholds (or None)

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
            f"Thr OPEN: {results['threshold_open']:.4f}  CLOSED: {results['threshold_closed']:.4f}",
        ]
        if self.calibration_thresholds is not None:
            cal = self.calibration_thresholds
            stats_parts.append(
                f"Cal OPEN: {cal['threshold_open']:.4f}  CLOSED: {cal['threshold_closed']:.4f}"
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
        time_emg = np.arange(n_samples) / config.FSAMP + self.time_offset
        timestamps = np.array(timestamps) + self.time_offset

        has_spatial = results.get('spatial_threshold') is not None
        has_gt = self.gt_data is not None
        has_ref = self.ref_predictions is not None
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

        # Corrected distances (scaling/contrast modes — None list otherwise)
        D_open_corr = results.get('D_open_corrected', [])
        D_closed_corr = results.get('D_closed_corrected', [])
        has_corrected = any(v is not None for v in D_open_corr)

        # --- Plot 2: Distance to OPEN ---
        ax = axs[plot_idx]
        ax.plot(timestamps, D_open, 'g-', linewidth=1, alpha=0.5 if has_corrected else 1.0,
                label='D_open (raw)')
        if has_corrected:
            corr_arr = np.array([v if v is not None else np.nan for v in D_open_corr])
            ax.plot(timestamps, corr_arr, color='darkgreen', linewidth=1.5,
                    linestyle='--', label='D_open (corrected)')
        ax.axhline(threshold_open, color='r', linestyle='--', linewidth=2,
                   label=f'Thr: {threshold_open:.4f}')
        if self.calibration_thresholds is not None:
            cal = self.calibration_thresholds
            ax.axhline(cal['open_plateau'], color='blue', linestyle=':', linewidth=1.5, alpha=0.8,
                       label=f"Cal plateau: {cal['open_plateau']:.4f}")
            ax.axhline(cal['threshold_open'], color='blue', linestyle='--', linewidth=1.5, alpha=0.8,
                       label=f"Cal thr: {cal['threshold_open']:.4f}")
        ax.set_ylabel("DTW Distance")
        ax.set_title("Distance to OPEN templates")
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        # --- Plot 3: Distance to CLOSED ---
        ax = axs[plot_idx]
        ax.plot(timestamps, D_closed, 'orange', linewidth=1, alpha=0.5 if has_corrected else 1.0,
                label='D_closed (raw)')
        if has_corrected:
            corr_arr = np.array([v if v is not None else np.nan for v in D_closed_corr])
            ax.plot(timestamps, corr_arr, color='darkorange', linewidth=1.5,
                    linestyle='--', label='D_closed (corrected)')
        ax.axhline(threshold_closed, color='r', linestyle='--', linewidth=2,
                   label=f'Thr: {threshold_closed:.4f}')
        if self.calibration_thresholds is not None:
            cal = self.calibration_thresholds
            ax.axhline(cal['closed_plateau'], color='blue', linestyle=':', linewidth=1.5, alpha=0.8,
                       label=f"Cal plateau: {cal['closed_plateau']:.4f}")
            ax.axhline(cal['threshold_closed'], color='blue', linestyle='--', linewidth=1.5, alpha=0.8,
                       label=f"Cal thr: {cal['threshold_closed']:.4f}")
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
            spatial_thr = results.get('spatial_threshold')
            spatial_mode = results.get('spatial_mode', 'off')

            if sim_open_data:
                sim_open_plot = [s if s is not None else float('nan') for s in sim_open_data]
                ax.plot(timestamps[:len(sim_open_plot)], sim_open_plot,
                        'g-', linewidth=1, alpha=0.8, label='Sim OPEN')
            if sim_closed_data:
                sim_closed_plot = [s if s is not None else float('nan') for s in sim_closed_data]
                ax.plot(timestamps[:len(sim_closed_plot)], sim_closed_plot,
                        'orange', linewidth=1, alpha=0.8, label='Sim CLOSED')

            # Show threshold line only for gate mode
            if spatial_mode == "gate" and spatial_thr is not None:
                ax.axhline(spatial_thr, color='r', linestyle='--', linewidth=1.5,
                            label=f'Gate thr: {spatial_thr:.2f}')

            # Mark blocked transitions (gate mode)
            blocked = results.get('spatial_blocked', [])
            for i, b in enumerate(blocked):
                if b and i < len(timestamps):
                    ax.axvline(timestamps[i], color='red', alpha=0.3, linewidth=2)

            mode_label = {"gate": "Gate", "scaling": "Distance Scaling", "contrast": "Contrast"}.get(spatial_mode, spatial_mode)
            ax.set_ylabel("Spatial Sim")
            ax.set_title(f"Spatial Similarity [{mode_label}]")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # --- Plot 5: State Comparison ---
        ax = axs[plot_idx]

        # ML probability overlay (grey, behind everything)
        ml_probs = results.get('ml_probabilities', [])
        if ml_probs and any(p is not None for p in ml_probs):
            prob_arr = np.array([p if p is not None else np.nan for p in ml_probs], dtype=float)
            ax.fill_between(timestamps, 0, prob_arr, step='post',
                            color='gray', alpha=0.25, label='ML probability')
            ax.plot(timestamps, prob_arr, color='gray', linewidth=0.8,
                    alpha=0.5, drawstyle='steps-post')

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
            gt_time = np.arange(len(gt)) / config.FSAMP + self.time_offset
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
        self._offline_test_calibration_thresholds = None  # From last GT-based run
        self._offline_test_dialogs: list = []  # Keep references so windows stay open
        self._playback_thread = None
        self._playback_worker = None

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

            # Reset model history for new session with selected initial state
            initial_state = self.initial_state_combo.currentText()
            self.model_interface.reset_history(initial_state=initial_state)

            # connect signals
            self.main_window.device.ready_read_signal.connect(self.online_emg_update)

            self.emg_buffer = []
            self.kinematics_buffer = []
            self.emg_timings_buffer = []
            self.kinematics_timings_buffer = []
            self.predictions_buffer = []
            self.unity_output_buffer = []

            print("\n" + "=" * 70)
            print(f"RECORDING STARTED - History reset (initial state: {initial_state})")
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
        # Always reset combo to "off" — new Model object always starts with spatial_mode="off",
        # so keeping the old combo value would desync UI from the model.
        self.spatial_mode_combo.blockSignals(True)
        self.spatial_mode_combo.setCurrentIndex(0)
        self.spatial_mode_combo.blockSignals(False)
        self.spatial_threshold_row.setVisible(False)
        self.spatial_sharpness_row.setVisible(False)
        self.spatial_relu_baseline_row.setVisible(False)
        self.spatial_coupling_row.setVisible(False)
        self.spatial_coupling_combo.blockSignals(True)
        self.spatial_coupling_combo.setCurrentIndex(0)  # reset to Global profile
        self.spatial_coupling_combo.blockSignals(False)
        # Disable coupled option if per_template_rms not available in this model
        has_per_tpl = False
        if model is not None:
            ref_o = getattr(model, 'spatial_ref_open', None)
            ref_c = getattr(model, 'spatial_ref_closed', None)
            has_per_tpl = (ref_o is not None and ref_o.get("per_template_rms") is not None
                           and ref_c is not None and ref_c.get("per_template_rms") is not None)
        from PySide6.QtGui import QStandardItemModel as _QStdModel
        _std = self.spatial_coupling_combo.model()
        if isinstance(_std, _QStdModel):
            _coupled_item = _std.item(1)
            if _coupled_item is not None:
                _coupled_item.setEnabled(has_per_tpl)
                if not has_per_tpl:
                    _coupled_item.setToolTip("Requires a model built with spatial profiles (per_template_rms).")

        self.spatial_mode_combo.setEnabled(has_spatial)
        if has_spatial:
            self.spatial_status_label.setText("(available)")
            self.spatial_status_label.setStyleSheet("color: #2a7;")
            self.spatial_threshold_slider.setValue(int(model.spatial_threshold * 100))
            self.spatial_sharpness_spinbox.setValue(model.spatial_sharpness)
        else:
            self.spatial_status_label.setText("(not available)")
            self.spatial_status_label.setStyleSheet("color: #999;")

        # Update ML model availability status
        has_catboost = (model is not None and model.decision_catboost is not None)
        has_nn = (model is not None and model.decision_nn is not None)
        parts = []
        if has_catboost:
            cb_mode = "transition" if getattr(model.decision_catboost, 'transition_mode', False) else "posture"
            parts.append(f"CatBoost {model.decision_catboost.accuracy:.1%} [{cb_mode}]")
        if has_nn:
            parts.append(f"NN {model.decision_nn.accuracy:.1%}")
        if parts:
            self.decision_model_status_label.setText("(" + ", ".join(parts) + ")")
            self.decision_model_status_label.setStyleSheet("color: #2a7;")
        else:
            self.decision_model_status_label.setText("(no ML models available)")
            self.decision_model_status_label.setStyleSheet("color: #999;")
            self.decision_model_combo.setCurrentIndex(0)  # fall back to threshold

        # Update threshold display after loading
        self._update_threshold_display()

        # Enable offline test buttons if recording is also loaded
        if self.offline_test_emg is not None:
            self.run_offline_test_button.setEnabled(True)
            self.play_realtime_button.setEnabled(True)

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

        # Add initial state selector next to recording toggle
        self._setup_initial_state_selector()

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

    def _setup_initial_state_selector(self) -> None:
        """Add initial state combo box to the commands group box."""
        state_widget = QWidget()
        state_layout = QHBoxLayout()
        state_layout.setContentsMargins(0, 0, 0, 0)
        state_layout.addWidget(QLabel("Initial State:"))
        self.initial_state_combo = QComboBox()
        self.initial_state_combo.addItems(["OPEN", "CLOSED"])
        self.initial_state_combo.setCurrentIndex(0)  # Default OPEN
        self.initial_state_combo.setFixedWidth(90)
        state_layout.addWidget(self.initial_state_combo)
        state_layout.addStretch()
        state_widget.setLayout(state_layout)

        # Add to the commands group box
        cmd_layout = self.online_commands_group_box.layout()
        if cmd_layout is not None:
            cmd_layout.addWidget(state_widget)

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

        # Time range row
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Range:"))
        self.offline_test_start_spinbox = QDoubleSpinBox()
        self.offline_test_start_spinbox.setMinimum(0.0)
        self.offline_test_start_spinbox.setMaximum(9999.0)
        self.offline_test_start_spinbox.setValue(0.0)
        self.offline_test_start_spinbox.setSuffix(" s")
        self.offline_test_start_spinbox.setDecimals(1)
        self.offline_test_start_spinbox.setFixedWidth(90)
        self.offline_test_start_spinbox.setToolTip("Start time of the recording slice to analyse.")
        self.offline_test_start_spinbox.setEnabled(False)
        range_layout.addWidget(self.offline_test_start_spinbox)

        range_layout.addWidget(QLabel("→"))

        self.offline_test_end_spinbox = QDoubleSpinBox()
        self.offline_test_end_spinbox.setMinimum(0.0)
        self.offline_test_end_spinbox.setMaximum(9999.0)
        self.offline_test_end_spinbox.setValue(0.0)
        self.offline_test_end_spinbox.setSuffix(" s")
        self.offline_test_end_spinbox.setDecimals(1)
        self.offline_test_end_spinbox.setFixedWidth(90)
        self.offline_test_end_spinbox.setToolTip("End time of the recording slice to analyse.")
        self.offline_test_end_spinbox.setEnabled(False)
        range_layout.addWidget(self.offline_test_end_spinbox)

        self.offline_test_duration_label = QLabel("")
        self.offline_test_duration_label.setStyleSheet("color: #888;")
        range_layout.addWidget(self.offline_test_duration_label)
        range_layout.addStretch()
        layout.addLayout(range_layout)

        # Run button + initial state selector
        run_layout = QHBoxLayout()
        self.run_offline_test_button = QPushButton("Run Offline Test")
        self.run_offline_test_button.setEnabled(False)
        self.run_offline_test_button.clicked.connect(self._run_offline_test)
        run_layout.addWidget(self.run_offline_test_button)

        run_layout.addWidget(QLabel("Start:"))
        self.offline_test_initial_state_combo = QComboBox()
        self.offline_test_initial_state_combo.addItems(["OPEN", "CLOSED"])
        self.offline_test_initial_state_combo.setCurrentIndex(0)  # Default OPEN
        self.offline_test_initial_state_combo.setFixedWidth(90)
        run_layout.addWidget(self.offline_test_initial_state_combo)
        run_layout.addStretch()
        layout.addLayout(run_layout)

        # Real-time playback row
        playback_layout = QHBoxLayout()
        self.play_realtime_button = QPushButton("▶ Play Real-time")
        self.play_realtime_button.setEnabled(False)
        self.play_realtime_button.setToolTip(
            "Stream the recording through the model at real-time speed.\n"
            "Updates the EMG plot and sends predictions to Unity, just like live data."
        )
        self.play_realtime_button.clicked.connect(self._run_realtime_playback)
        playback_layout.addWidget(self.play_realtime_button)

        self.stop_playback_button = QPushButton("■ Stop")
        self.stop_playback_button.setEnabled(False)
        self.stop_playback_button.clicked.connect(self._stop_realtime_playback)
        playback_layout.addWidget(self.stop_playback_button)

        playback_layout.addWidget(QLabel("Speed:"))
        self.playback_speed_spinbox = QDoubleSpinBox()
        self.playback_speed_spinbox.setMinimum(0.1)
        self.playback_speed_spinbox.setMaximum(10.0)
        self.playback_speed_spinbox.setSingleStep(0.5)
        self.playback_speed_spinbox.setValue(1.0)
        self.playback_speed_spinbox.setSuffix("×")
        self.playback_speed_spinbox.setDecimals(1)
        self.playback_speed_spinbox.setFixedWidth(75)
        playback_layout.addWidget(self.playback_speed_spinbox)

        self._playback_time_label = QLabel("")
        self._playback_time_label.setStyleSheet("color: #888; font-family: monospace;")
        playback_layout.addWidget(self._playback_time_label)
        playback_layout.addStretch()
        layout.addLayout(playback_layout)

        # Decision model selector
        decision_layout = QHBoxLayout()
        decision_layout.addWidget(QLabel("Decision Model:"))
        self.decision_model_combo = QComboBox()
        self.decision_model_combo.addItems(["Threshold", "CatBoost", "Neural Network"])
        self.decision_model_combo.setEnabled(True)
        self.decision_model_combo.setToolTip(
            "Threshold: manual sliders + optional spatial correction\n"
            "CatBoost: gradient boosting trained from templates (deterministic)\n"
            "Neural Network: small NN trained from templates (requires PyTorch)"
        )
        self.decision_model_combo.currentIndexChanged.connect(self._on_decision_model_changed)
        decision_layout.addWidget(self.decision_model_combo)

        self.decision_model_status_label = QLabel("")
        self.decision_model_status_label.setStyleSheet("color: #999;")
        decision_layout.addWidget(self.decision_model_status_label)
        decision_layout.addStretch()
        layout.addLayout(decision_layout)

        # Calibration results (only shown when GT is available and test completed)
        self.calibration_results_label = QLabel("")
        self.calibration_results_label.setStyleSheet(
            "color: #2a7; font-family: monospace; font-size: 10px; padding: 2px;"
        )
        self.calibration_results_label.setWordWrap(True)
        self.calibration_results_label.setVisible(False)
        layout.addWidget(self.calibration_results_label)

        self.apply_calibration_button = QPushButton("Apply Calibration Thresholds to Sliders")
        self.apply_calibration_button.setVisible(False)
        self.apply_calibration_button.clicked.connect(self._apply_calibration_thresholds)
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

            # Configure time range spinboxes
            self.offline_test_start_spinbox.setMaximum(duration_s)
            self.offline_test_end_spinbox.setMaximum(duration_s)
            self.offline_test_start_spinbox.setValue(0.0)
            self.offline_test_end_spinbox.setValue(duration_s)
            self.offline_test_duration_label.setText(f"of {duration_s:.1f}s total")
            self.offline_test_start_spinbox.setEnabled(True)
            self.offline_test_end_spinbox.setEnabled(True)

            # Enable run/play buttons only if model is loaded
            model_loaded = (self.model_interface is not None and
                            self.model_interface.model_is_loaded)
            self.run_offline_test_button.setEnabled(model_loaded)
            self.play_realtime_button.setEnabled(model_loaded)

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

        initial_state = self.offline_test_initial_state_combo.currentText()

        # --- Slice EMG and annotations by time range ---
        start_s = self.offline_test_start_spinbox.value()
        end_s = self.offline_test_end_spinbox.value()
        if end_s <= start_s:
            end_s = start_s + 1.0

        n_total = self.offline_test_emg.shape[1]
        start_sample = max(0, min(int(start_s * config.FSAMP), n_total - 1))
        end_sample = max(start_sample + 1, min(int(end_s * config.FSAMP), n_total))
        emg_slice = self.offline_test_emg[:, start_sample:end_sample]

        # Slice GT (at FSAMP rate)
        gt_slice = None
        if self.offline_test_gt is not None:
            gt_arr = self.offline_test_gt
            gt_slice = gt_arr[start_sample:min(end_sample, len(gt_arr))]

        # Slice ref_predictions (at DTW rate — proportional to total duration)
        ref_slice = None
        if self.offline_test_ref_predictions is not None:
            ref = self.offline_test_ref_predictions
            n_ref = len(ref)
            total_dur_s = n_total / config.FSAMP
            if total_dur_s > 0:
                ref_start = max(0, int((start_s / total_dur_s) * n_ref))
                ref_end = min(n_ref, int((end_s / total_dur_s) * n_ref))
                ref_slice = ref[ref_start:ref_end] if ref_end > ref_start else None

        # Store slices so _on_offline_test_finished can access them
        self._offline_test_emg_slice = emg_slice
        self._offline_test_gt_slice = gt_slice
        self._offline_test_ref_slice = ref_slice
        self._offline_test_start_s = start_s

        print(f"[OFFLINE TEST] Time range: {start_s:.1f}s – {end_s:.1f}s "
              f"({emg_slice.shape[1]} samples)")

        self._offline_test_thread = QThread()
        self._offline_test_worker = SimulationWorker(
            emg_slice, model, initial_state=initial_state, gt_data=gt_slice,
        )
        self._offline_test_worker.moveToThread(self._offline_test_thread)
        self._offline_test_thread.started.connect(self._offline_test_worker.run)
        self._offline_test_worker.finished.connect(self._on_offline_test_finished)
        self._offline_test_worker.error.connect(self._on_offline_test_error)
        self._offline_test_worker.finished.connect(lambda r, c: self._offline_test_thread.quit())
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

        # Store and display calibration thresholds if GT was available
        self._offline_test_calibration_thresholds = calibration_thresholds
        if calibration_thresholds is not None:
            open_p = calibration_thresholds['open_plateau']
            open_t = calibration_thresholds['threshold_open']
            closed_p = calibration_thresholds['closed_plateau']
            closed_t = calibration_thresholds['threshold_closed']
            self.calibration_results_label.setText(
                f"OPEN plateau: {open_p:.4f}  →  threshold: {open_t:.4f}\n"
                f"CLOSED plateau: {closed_p:.4f}  →  threshold: {closed_t:.4f}"
            )
            self.calibration_results_label.setVisible(True)
            self.apply_calibration_button.setVisible(True)
        else:
            self.calibration_results_label.setVisible(False)
            self.apply_calibration_button.setVisible(False)

        # Show review dialog (standalone — keep all open simultaneously)
        emg_for_dialog = getattr(self, '_offline_test_emg_slice', self.offline_test_emg)
        gt_for_dialog = getattr(self, '_offline_test_gt_slice', self.offline_test_gt)
        ref_for_dialog = getattr(self, '_offline_test_ref_slice', self.offline_test_ref_predictions)

        dialog = OfflineTestReviewDialog(
            emg_data=emg_for_dialog,
            results=results,
            gt_data=gt_for_dialog,
            ref_predictions=ref_for_dialog,
            recording_type=self.offline_test_recording_type or "unknown",
            time_offset=getattr(self, '_offline_test_start_s', 0.0),
            calibration_thresholds=calibration_thresholds,
            parent=None,  # No parent → truly standalone window
        )
        # Remove any dialogs that have been closed to free memory
        self._offline_test_dialogs = [d for d in self._offline_test_dialogs
                                       if not d.isHidden()]
        self._offline_test_dialogs.append(dialog)
        dialog.show()

    def _on_offline_test_error(self, error_msg: str) -> None:
        """Handle offline test error."""
        self.run_offline_test_button.setEnabled(True)
        self.run_offline_test_button.setText("Run Offline Test")
        QMessageBox.critical(self.main_window, "Offline Test Error",
                             f"Offline test failed:\n{error_msg}", QMessageBox.Ok)
        print(f"[OFFLINE TEST ERROR] {error_msg}")

    def _apply_calibration_thresholds(self) -> None:
        """Apply calibration thresholds (from GT plateau analysis) to the threshold sliders."""
        cal = self._offline_test_calibration_thresholds
        if cal is None:
            return
        threshold_open = cal['threshold_open']
        threshold_closed = cal['threshold_closed']
        self.model_interface.set_threshold_open_direct(threshold_open)
        self.model_interface.set_threshold_closed_direct(threshold_closed)
        self._update_threshold_display()
        print(f"[CALIBRATION] Applied: OPEN={threshold_open:.4f}, CLOSED={threshold_closed:.4f}")

    # ------------------------------------------------------------------
    # Real-time playback
    # ------------------------------------------------------------------

    def _run_realtime_playback(self) -> None:
        """Stream the loaded recording through the model at real-time speed."""
        if self.offline_test_emg is None:
            return
        if self.model_interface is None or self.model_interface.model is None:
            QMessageBox.warning(self.main_window, "No Model",
                                "Please load a model first.", QMessageBox.Ok)
            return

        # Stop any existing playback before starting a new one
        self._stop_realtime_playback()

        # Slice by time range (same logic as _run_offline_test)
        start_s = self.offline_test_start_spinbox.value()
        end_s   = self.offline_test_end_spinbox.value()
        if end_s <= start_s:
            end_s = start_s + 1.0
        n_total = self.offline_test_emg.shape[1]
        start_sample = max(0, min(int(start_s * config.FSAMP), n_total - 1))
        end_sample   = max(start_sample + 1, min(int(end_s * config.FSAMP), n_total))
        emg_slice = self.offline_test_emg[:, start_sample:end_sample]

        speed = self.playback_speed_spinbox.value()

        # Reset model state for fresh playback
        initial_state = self.offline_test_initial_state_combo.currentText()
        self.model_interface.reset_history(initial_state=initial_state)

        duration_s = emg_slice.shape[1] / config.FSAMP
        print(f"\n[PLAYBACK] Starting real-time playback")
        print(f"  EMG: {emg_slice.shape}, duration: {duration_s:.1f}s at {speed:.1f}× speed")
        print(f"  Initial state: {initial_state}")

        # Update UI
        self.play_realtime_button.setEnabled(False)
        self.run_offline_test_button.setEnabled(False)
        self.stop_playback_button.setEnabled(True)
        self._playback_time_label.setText("▶ 0.0s")

        self._playback_thread = QThread()
        self._playback_worker = RealtimePlaybackWorker(emg_slice, speed=speed)
        self._playback_worker.moveToThread(self._playback_thread)
        self._playback_thread.started.connect(self._playback_worker.run)
        self._playback_worker.chunk_ready.connect(self._on_playback_chunk)
        self._playback_worker.progress.connect(self._on_playback_progress)
        self._playback_worker.finished.connect(self._on_playback_finished)
        self._playback_worker.error.connect(self._on_playback_error)
        self._playback_worker.finished.connect(self._playback_thread.quit)
        self._playback_worker.error.connect(self._playback_thread.quit)
        self._playback_thread.finished.connect(self._playback_thread.deleteLater)
        self._playback_thread.start()

    def _disconnect_playback_signals(self) -> None:
        """Disconnect all playback worker signals to prevent stale updates."""
        if self._playback_worker is None:
            return
        for sig, slot in [
            (self._playback_worker.chunk_ready, self._on_playback_chunk),
            (self._playback_worker.progress,    self._on_playback_progress),
            (self._playback_worker.finished,    self._on_playback_finished),
            (self._playback_worker.error,       self._on_playback_error),
        ]:
            try:
                sig.disconnect(slot)
            except RuntimeError:
                pass  # Already disconnected

    def _on_playback_chunk(self, emg_chunk: np.ndarray) -> None:
        """Called in the main thread for each streamed EMG chunk."""
        # Update the VisPy real-time plot (same scaling as live mode)
        try:
            if (hasattr(self.main_window, 'plot')
                    and self.main_window.plot is not None
                    and self.main_window.plot_enabled_check_box.isChecked()):
                self.main_window.plot.set_plot_data(emg_chunk / 1000)
        except Exception:
            pass  # Plot update is best-effort; don't crash playback

        # Run model prediction (wrapped so exceptions don't leak into Qt event loop)
        try:
            self.model_interface.predict(emg_chunk)
            result = self.model_interface.get_last_result()
            if result:
                state = result['state']
                joint_value = int(state)
                unity_data = [joint_value] * 10
                self.main_window.virtual_hand_interface.output_message_signal.emit(
                    str(unity_data).encode("utf-8")
                )
        except Exception as e:
            print(f"[PLAYBACK] Prediction error: {e}")

    def _on_playback_progress(self, t: float) -> None:
        self._playback_time_label.setText(f"▶ {t:.1f}s")

    def _on_playback_finished(self) -> None:
        self._disconnect_playback_signals()
        self.play_realtime_button.setEnabled(True)
        self.stop_playback_button.setEnabled(False)
        self.run_offline_test_button.setEnabled(True)
        self._playback_time_label.setText("■ Done")
        print("[PLAYBACK] Finished")

    def _on_playback_error(self, error_msg: str) -> None:
        self._disconnect_playback_signals()
        self.play_realtime_button.setEnabled(True)
        self.stop_playback_button.setEnabled(False)
        self.run_offline_test_button.setEnabled(True)
        self._playback_time_label.setText("")
        QMessageBox.critical(self.main_window, "Playback Error",
                             f"Playback failed:\n{error_msg}", QMessageBox.Ok)
        print(f"[PLAYBACK ERROR] {error_msg}")

    def _stop_realtime_playback(self) -> None:
        # Disconnect signals first so no more chunks reach _on_playback_chunk
        # even if the worker thread is mid-sleep and hasn't noticed _stop yet
        self._disconnect_playback_signals()
        if self._playback_worker is not None:
            self._playback_worker.stop()
        if self._playback_thread is not None and self._playback_thread.isRunning():
            self._playback_thread.quit()
            self._playback_thread.wait(2000)  # Max 2s for thread to exit cleanly
        self._playback_worker = None
        self._playback_thread = None
        self.stop_playback_button.setEnabled(False)
        self._playback_time_label.setText("■ Stopped")
        print("[PLAYBACK] Stopped")

    def _on_decision_model_changed(self, index: int) -> None:
        """Handle decision model combo change."""
        model = self.model_interface.model if self.model_interface else None
        choice = self.decision_model_combo.currentText()

        if choice == "CatBoost":
            if model is None or model.decision_catboost is None:
                QMessageBox.information(self.main_window, "Not Available",
                    "No CatBoost model in the loaded model file.\n"
                    "Recreate the model with 'Decision Model: CatBoost' selected.")
                self.decision_model_combo.setCurrentIndex(0)
                return
            if model:
                model.decision_mode = "catboost"
            self.threshold_group_box.setEnabled(False)
            self.spatial_group_box.setEnabled(False)
            print(f"[DECISION] CatBoost (accuracy={model.decision_catboost.accuracy:.1%})")
        elif choice == "Neural Network":
            if model is None or model.decision_nn is None:
                QMessageBox.information(self.main_window, "Not Available",
                    "No Neural Network model in the loaded model file.\n"
                    "Recreate the model with 'Decision Model: Neural Network' selected.")
                self.decision_model_combo.setCurrentIndex(0)
                return
            if model:
                model.decision_mode = "nn"
            self.threshold_group_box.setEnabled(False)
            self.spatial_group_box.setEnabled(False)
            print(f"[DECISION] Neural Network (accuracy={model.decision_nn.accuracy:.1%})")
        else:  # Threshold
            if model:
                model.decision_mode = "threshold"
            self.threshold_group_box.setEnabled(True)
            self.spatial_group_box.setEnabled(True)
            print(f"[DECISION] Threshold mode")


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
        """Setup spatial correction controls with mode selector."""
        self.spatial_group_box = QGroupBox("Spatial Correction")

        layout = QVBoxLayout()

        # Mode selector
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Mode:"))
        self.spatial_mode_combo = QComboBox()
        self.spatial_mode_combo.addItem("Off", "off")
        self.spatial_mode_combo.addItem("Gate (threshold)", "gate")
        self.spatial_mode_combo.addItem("Distance Scaling", "scaling")
        self.spatial_mode_combo.addItem("Contrast", "contrast")
        self.spatial_mode_combo.addItem("ReLU Scaling", "relu_scaling")
        self.spatial_mode_combo.addItem("ReLU Contrast", "relu_contrast")
        self.spatial_mode_combo.addItem("ReLU Scaling (ext)", "relu_ext_scaling")
        self.spatial_mode_combo.addItem("ReLU Contrast (ext)", "relu_ext_contrast")
        self.spatial_mode_combo.setCurrentIndex(0)
        self.spatial_mode_combo.setToolTip(
            "Off: no spatial correction\n"
            "Gate: block transitions if spatial similarity < threshold\n"
            "Distance Scaling: D / sim^k — power-law penalty\n"
            "Contrast: D * (sim_current / sim_target)^k — ratio between classes\n"
            "ReLU Scaling: D / f(sim_target) — exponential penalty, saturates at 1 above threshold\n"
            "ReLU Contrast: D * f(sim_current) / f(sim_target) — combines ratio with ReLU saturation\n"
            "ReLU+ Scaling: like ReLU Scaling but f > 1 above threshold (rewards high similarity)\n"
            "ReLU+ Contrast: like ReLU Contrast but f > 1 above threshold (amplifies ratio)"
        )
        self.spatial_mode_combo.currentIndexChanged.connect(self._on_spatial_mode_changed)
        row1.addWidget(self.spatial_mode_combo)

        self.spatial_status_label = QLabel("(not available)")
        self.spatial_status_label.setStyleSheet("color: #999;")
        row1.addWidget(self.spatial_status_label)
        row1.addStretch()
        layout.addLayout(row1)

        # Threshold slider (only for gate mode)
        self.spatial_threshold_row = QWidget()
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.addWidget(QLabel("Threshold:"))

        self.spatial_threshold_slider = QSlider(Qt.Horizontal)
        self.spatial_threshold_slider.setMinimum(0)
        self.spatial_threshold_slider.setMaximum(100)
        self.spatial_threshold_slider.setValue(50)  # Default 0.50
        self.spatial_threshold_slider.setToolTip(
            "Minimum spatial similarity required to allow a state transition.\n"
            "Higher = more strict, Lower = more permissive."
        )
        self.spatial_threshold_slider.valueChanged.connect(self._on_spatial_threshold_changed)
        row2.addWidget(self.spatial_threshold_slider)

        self.spatial_threshold_label = QLabel("0.50")
        self.spatial_threshold_label.setMinimumWidth(40)
        row2.addWidget(self.spatial_threshold_label)
        self.spatial_threshold_row.setLayout(row2)
        self.spatial_threshold_row.setVisible(False)  # Hidden by default (shown for gate mode)
        layout.addWidget(self.spatial_threshold_row)

        # Sharpness spinbox (for scaling and contrast modes)
        self.spatial_sharpness_row = QWidget()
        row3 = QHBoxLayout()
        row3.setContentsMargins(0, 0, 0, 0)
        row3.addWidget(QLabel("Sharpness (k):"))
        self.spatial_sharpness_spinbox = QDoubleSpinBox()
        self.spatial_sharpness_spinbox.setMinimum(1.0)
        self.spatial_sharpness_spinbox.setMaximum(10.0)
        self.spatial_sharpness_spinbox.setSingleStep(0.5)
        self.spatial_sharpness_spinbox.setValue(3.0)  # Default k=3
        self.spatial_sharpness_spinbox.setDecimals(1)
        self.spatial_sharpness_spinbox.setToolTip(
            "Exponent for nonlinear spatial correction.\n"
            "k=1: linear (weak), k=3: moderate, k=5+: aggressive.\n"
            "Scaling: D / sim^k  |  Contrast: D * (sim_current/sim_target)^k"
        )
        self.spatial_sharpness_spinbox.valueChanged.connect(self._on_spatial_sharpness_changed)
        row3.addWidget(self.spatial_sharpness_spinbox)
        row3.addStretch()
        self.spatial_sharpness_row.setLayout(row3)
        self.spatial_sharpness_row.setVisible(False)  # Hidden by default
        layout.addWidget(self.spatial_sharpness_row)

        # Baseline for ReLU scaling (f(0) = baseline, f(threshold) = 1)
        self.spatial_relu_baseline_row = QWidget()
        row_rb = QHBoxLayout()
        row_rb.setContentsMargins(0, 0, 0, 0)
        row_rb.addWidget(QLabel("Baseline (b):"))
        self.spatial_relu_baseline_spinbox = QDoubleSpinBox()
        self.spatial_relu_baseline_spinbox.setMinimum(0.01)
        self.spatial_relu_baseline_spinbox.setMaximum(0.99)
        self.spatial_relu_baseline_spinbox.setSingleStep(0.05)
        self.spatial_relu_baseline_spinbox.setValue(0.3)
        self.spatial_relu_baseline_spinbox.setDecimals(2)
        self.spatial_relu_baseline_spinbox.setToolTip(
            "Minimum value of f(sim) at sim=0.\n"
            "f(sim) = b^(1 - sim/threshold): concave-up exponential.\n"
            "Lower b → more aggressive penalty for low similarities."
        )
        self.spatial_relu_baseline_spinbox.valueChanged.connect(self._on_spatial_relu_baseline_changed)
        row_rb.addWidget(self.spatial_relu_baseline_spinbox)
        row_rb.addStretch()
        self.spatial_relu_baseline_row.setLayout(row_rb)
        self.spatial_relu_baseline_row.setVisible(False)
        layout.addWidget(self.spatial_relu_baseline_row)

        # Spatial similarity profile mode (mean vs per-template top-k)
        self.spatial_profile_row = QWidget()
        row4 = QHBoxLayout()
        row4.setContentsMargins(0, 0, 0, 0)
        row4.addWidget(QLabel("Profile:"))
        self.spatial_profile_combo = QComboBox()
        self.spatial_profile_combo.addItem("Mean profile", "mean")
        self.spatial_profile_combo.addItem("Per-template top-k", "per_template")
        self.spatial_profile_combo.setToolTip(
            "Mean profile: similarity against the class mean spatial profile.\n"
            "Per-template top-k: compute similarity against each template, "
            "average the k highest (analogous to avg_3_smallest DTW)."
        )
        self.spatial_profile_combo.currentIndexChanged.connect(self._on_spatial_profile_mode_changed)
        row4.addWidget(self.spatial_profile_combo)

        self.spatial_nbest_spinbox = QSpinBox()
        self.spatial_nbest_spinbox.setMinimum(1)
        self.spatial_nbest_spinbox.setMaximum(20)
        self.spatial_nbest_spinbox.setValue(3)
        self.spatial_nbest_spinbox.setPrefix("k=")
        self.spatial_nbest_spinbox.setToolTip("Number of top similarities to average.")
        self.spatial_nbest_spinbox.setVisible(False)
        self.spatial_nbest_spinbox.valueChanged.connect(self._on_spatial_nbest_changed)
        row4.addWidget(self.spatial_nbest_spinbox)
        row4.addStretch()
        self.spatial_profile_row.setLayout(row4)
        layout.addWidget(self.spatial_profile_row)

        # Coupling row: Global profile vs Per-template coupled
        self.spatial_coupling_row = QWidget()
        row_coupling = QHBoxLayout()
        row_coupling.setContentsMargins(0, 0, 0, 0)
        row_coupling.addWidget(QLabel("Coupling:"))
        self.spatial_coupling_combo = QComboBox()
        self.spatial_coupling_combo.addItem("Global profile", "global")
        self.spatial_coupling_combo.addItem("Per-template coupled", "coupled")
        self.spatial_coupling_combo.setToolTip(
            "Global profile: spatial similarity vs the class mean profile (or top-k mean).\n"
            "Per-template coupled: each DTW distance is corrected by its own template's\n"
            "spatial similarity before aggregation. Contrast modes not supported."
        )
        self.spatial_coupling_combo.currentIndexChanged.connect(self._on_spatial_coupling_changed)
        row_coupling.addWidget(self.spatial_coupling_combo)

        self.spatial_coupling_note = QLabel("(Contrast modes fall back to Scaling)")
        self.spatial_coupling_note.setStyleSheet("color: #999; font-style: italic; font-size: 10px;")
        self.spatial_coupling_note.setVisible(False)
        row_coupling.addWidget(self.spatial_coupling_note)
        row_coupling.addStretch()
        self.spatial_coupling_row.setLayout(row_coupling)
        self.spatial_coupling_row.setVisible(False)  # Hidden when spatial mode is "off"
        layout.addWidget(self.spatial_coupling_row)

        self.spatial_group_box.setLayout(layout)

        # Add to the online commands group box layout
        if self.online_commands_group_box.layout():
            self.online_commands_group_box.layout().addWidget(self.spatial_group_box)

    def _on_spatial_mode_changed(self, index: int) -> None:
        """Handle spatial correction mode change."""
        mode = self.spatial_mode_combo.currentData()

        # Show/hide threshold slider (gate = block threshold; relu modes = saturation point)
        self.spatial_threshold_row.setVisible(mode in ("gate", "relu_scaling", "relu_contrast", "relu_ext_scaling", "relu_ext_contrast"))
        # Show/hide sharpness spinbox (power-law modes + relu modes)
        self.spatial_sharpness_row.setVisible(mode in ("scaling", "contrast", "relu_scaling", "relu_contrast", "relu_ext_scaling", "relu_ext_contrast"))
        # Show/hide baseline spinbox (relu modes only)
        self.spatial_relu_baseline_row.setVisible(mode in ("relu_scaling", "relu_contrast", "relu_ext_scaling", "relu_ext_contrast"))
        # Show coupling row only when spatial is active
        self.spatial_coupling_row.setVisible(mode != "off")
        # Update contrast-mode note visibility
        is_coupled = self.spatial_coupling_combo.currentData() == "coupled"
        is_contrast = mode in ("contrast", "relu_contrast", "relu_ext_contrast")
        self.spatial_coupling_note.setVisible(is_coupled and is_contrast)

        if self.model_interface.model is not None:
            model = self.model_interface.model
            if mode == "off":
                model.spatial_mode = "off"
                model.use_spatial_correction = False
            else:
                if model.spatial_ref_open is None or model.spatial_ref_closed is None:
                    self.spatial_mode_combo.setCurrentIndex(0)  # Reset to off
                    QMessageBox.information(self.main_window, "Not Available",
                                            "This model has no spatial profiles. Spatial correction unavailable.")
                    return
                model.spatial_mode = mode
                model.use_spatial_correction = True
            print(f"[SPATIAL] Mode: {mode}")
        else:
            if mode != "off":
                self.spatial_mode_combo.setCurrentIndex(0)
                QMessageBox.information(self.main_window, "No Model",
                                        "Load a model first.")

    def _on_spatial_threshold_changed(self, value: int) -> None:
        """Update spatial similarity threshold from slider."""
        threshold = value / 100.0
        self.spatial_threshold_label.setText(f"{threshold:.2f}")
        if self.model_interface.model is not None:
            self.model_interface.model.spatial_threshold = threshold
            print(f"[SPATIAL] Threshold: {threshold:.2f}")

    def _on_spatial_sharpness_changed(self, value: float) -> None:
        """Update spatial sharpness exponent."""
        if self.model_interface.model is not None:
            self.model_interface.model.spatial_sharpness = value
            print(f"[SPATIAL] Sharpness (k): {value:.1f}")

    def _on_spatial_relu_baseline_changed(self, value: float) -> None:
        """Update ReLU scaling baseline."""
        if self.model_interface.model is not None:
            self.model_interface.model.spatial_relu_baseline = value
            print(f"[SPATIAL] ReLU baseline (b): {value:.2f}")

    def _on_spatial_profile_mode_changed(self, index: int) -> None:
        """Switch between mean-profile and per-template spatial similarity."""
        mode = self.spatial_profile_combo.currentData()
        self.spatial_nbest_spinbox.setVisible(mode == "per_template")
        if self.model_interface.model is not None:
            self.model_interface.model.spatial_similarity_mode = mode
            print(f"[SPATIAL] Profile mode: {mode}")

    def _on_spatial_nbest_changed(self, value: int) -> None:
        """Update top-k for per-template spatial similarity."""
        if self.model_interface.model is not None:
            self.model_interface.model.spatial_n_best = value
            print(f"[SPATIAL] Per-template k: {value}")

    def _on_spatial_coupling_changed(self, index: int) -> None:
        """Switch between global profile and per-template coupled correction."""
        coupling = self.spatial_coupling_combo.currentData()
        is_coupled = coupling == "coupled"
        mode = self.spatial_mode_combo.currentData()
        is_contrast = mode in ("contrast", "relu_contrast", "relu_ext_contrast")
        self.spatial_coupling_note.setVisible(is_coupled and is_contrast)

        if self.model_interface.model is not None:
            model = self.model_interface.model
            if is_coupled:
                # Validate that per_template_rms is available
                ref_open = getattr(model, 'spatial_ref_open', None)
                ref_closed = getattr(model, 'spatial_ref_closed', None)
                has_per_tpl = (
                    ref_open is not None and ref_open.get("per_template_rms") is not None and
                    ref_closed is not None and ref_closed.get("per_template_rms") is not None
                )
                if not has_per_tpl:
                    self.spatial_coupling_combo.blockSignals(True)
                    self.spatial_coupling_combo.setCurrentIndex(0)  # revert to Global
                    self.spatial_coupling_combo.blockSignals(False)
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.information(
                        self.main_window, "Not Available",
                        "Per-template coupling requires a model built with spatial profiles.\n"
                        "The current model does not have per-template RMS data."
                    )
                    return
            model.spatial_coupled = is_coupled
            print(f"[SPATIAL] Coupling: {'per-template coupled' if is_coupled else 'global profile'}")

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
