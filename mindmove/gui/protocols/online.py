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
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# MindMove imports
from mindmove.model.interface import MindMoveInterface

from mindmove.config import config


class CalibrationWorker(QObject):
    """Worker class for running calibration in a separate thread."""
    finished = Signal(dict, dict)  # (result, thresholds)
    error = Signal(str)

    def __init__(self, recording, model):
        super().__init__()
        self.recording = recording
        self.model = model

    def run(self):
        """Run calibration computation in background thread."""
        try:
            from mindmove.model.core.algorithm import compute_calibration_distances, find_plateau_thresholds

            # Get data
            emg = self.recording['emg']
            gt = self.recording['gt']

            # Ensure gt is 1D array
            if hasattr(gt, 'flatten'):
                gt = gt.flatten()
            gt = np.array(gt)

            # Get model parameters
            templates_open = self.model.templates_open
            templates_closed = self.model.templates_closed
            feature_name = self.model.feature_name
            window_length = self.model.window_length
            increment = self.model.increment
            active_channels = self.model.active_channels
            distance_aggregation = self.model.distance_aggregation

            print(f"[CALIBRATION WORKER] Starting computation...")
            print(f"  Feature: {feature_name}")
            print(f"  Window/increment: {window_length}/{increment} samples")
            print(f"  Active channels: {len(active_channels)}")
            print(f"  Distance aggregation: {distance_aggregation}")

            # Compute continuous distances
            print("\n[CALIBRATION WORKER] Computing DTW distances over recording...")
            result = compute_calibration_distances(
                emg, gt,
                templates_open, templates_closed,
                feature_name=feature_name,
                window_length=window_length,
                increment=increment,
                active_channels=active_channels,
                distance_aggregation=distance_aggregation,
            )

            n_dtw = len(result['timestamps'])
            print(f"[CALIBRATION WORKER] Computed {n_dtw} DTW distances")

            # Find plateau thresholds
            print("\n[CALIBRATION WORKER] Finding plateau thresholds...")
            thresholds = find_plateau_thresholds(
                result['D_open'],
                result['D_closed'],
                result['gt_at_dtw'],
                confidence_k=1.0,
            )

            print(f"[CALIBRATION WORKER] Calibration complete")
            self.finished.emit(result, thresholds)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class OnlineSessionReviewDialog(QDialog):
    """
    Dialog for reviewing online prediction session with channel selection.

    Shows 3 subplots:
    - EMG signal (selectable channel)
    - State over time
    - Distance to templates with thresholds

    Supports arrow keys for channel switching.
    """

    def __init__(self, emg_signal, emg_time_axis, history, parent=None):
        super().__init__(parent)
        self.emg_signal = emg_signal  # (n_channels, n_samples)
        self.emg_time_axis = emg_time_axis
        self.history = history
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
        hint = QLabel("(↑↓ to change channel)")
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

        ax1.set_ylabel(f"EMG Ch{self.current_channel + 1} (µV)", fontsize=10)
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

        # Calibration state
        self.calibration_recording = None
        self.calibration_result = None
        self.calibration_thresholds = None

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
        dialog = OnlineSessionReviewDialog(emg_signal, emg_time_axis, history, self.main_window)
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
            # Calibration not available in diagnostic mode
            self.run_calibration_button.setEnabled(False)
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
        label = file_name.split("/")[-1].split("_")[-1].split(".")[0]
        self.online_model_label.setText(f"{label} loaded.")

        # Update threshold display after loading
        self._update_threshold_display()

        # Enable calibration button if recording is also loaded
        if self.calibration_recording is not None:
            self.run_calibration_button.setEnabled(True)

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
                f"Threshold OPEN: {threshold_open:.4f} (s={s_open:.2f})"
            )
            self.threshold_closed_label.setText(
                f"Threshold CLOSED: {threshold_closed:.4f} (s={s_closed:.2f})"
            )

            print(f"[THRESHOLD UI] OPEN range: 0 - {self._max_threshold_open:.4f}, current: {threshold_open:.4f}")
            print(f"[THRESHOLD UI] CLOSED range: 0 - {self._max_threshold_closed:.4f}, current: {threshold_closed:.4f}")

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
            preset_order = ["current", "cross_class", "safety_margin", "conservative"]
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

        # Add offline calibration UI (programmatically)
        self._setup_calibration_ui()

        # Add threshold tuning slider (programmatically)
        self._setup_threshold_slider()

        # Add refractory period control (programmatically)
        self._setup_refractory_control()

        # Add plot options checkbox
        self._setup_plot_options()

    def _setup_calibration_ui(self) -> None:
        """Setup offline threshold calibration UI."""
        self.calibration_group_box = QGroupBox("Threshold Offline Calibration")
        layout = QVBoxLayout()

        # Recording selection row
        rec_layout = QHBoxLayout()
        self.calibration_load_button = QPushButton("Load Recording")
        self.calibration_load_button.clicked.connect(self._load_calibration_recording)
        rec_layout.addWidget(self.calibration_load_button)

        self.calibration_recording_label = QLabel("No recording loaded")
        self.calibration_recording_label.setStyleSheet("color: #666;")
        rec_layout.addWidget(self.calibration_recording_label)
        rec_layout.addStretch()
        layout.addLayout(rec_layout)

        # Run calibration button
        self.run_calibration_button = QPushButton("Run Calibration")
        self.run_calibration_button.setEnabled(False)
        self.run_calibration_button.clicked.connect(self._run_calibration)
        layout.addWidget(self.run_calibration_button)

        # Results display
        self.calibration_results_label = QLabel("")
        self.calibration_results_label.setStyleSheet(
            "font-family: monospace; color: #333333; background-color: #f5f5f5; padding: 8px; border-radius: 4px;"
        )
        self.calibration_results_label.setWordWrap(True)
        self.calibration_results_label.setVisible(False)
        layout.addWidget(self.calibration_results_label)

        # Apply button
        self.apply_calibration_button = QPushButton("Apply to Sliders")
        self.apply_calibration_button.setEnabled(False)
        self.apply_calibration_button.clicked.connect(self._apply_calibration)
        layout.addWidget(self.apply_calibration_button)

        self.calibration_group_box.setLayout(layout)

        # Add to the online commands group box layout (after Load Model)
        if self.online_commands_group_box.layout():
            self.online_commands_group_box.layout().addWidget(self.calibration_group_box)

    def _load_calibration_recording(self) -> None:
        """Load a recording file for calibration."""
        recording_dir = "data/recordings/"
        if not os.path.exists(recording_dir):
            recording_dir = "."

        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Select Recording for Calibration",
            recording_dir,
            "Pickle Files (*.pkl)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'rb') as f:
                recording = pickle.load(f)

            # Verify it has required data
            if 'emg' not in recording:
                QMessageBox.warning(
                    self.main_window,
                    "Invalid Recording",
                    "Recording must contain 'emg' data."
                )
                return

            if 'gt' not in recording:
                QMessageBox.warning(
                    self.main_window,
                    "Missing Ground Truth",
                    "Recording must have 'gt' (ground truth) signal for calibration.\n\n"
                    "Use a recording from guided_record protocol that includes GT."
                )
                return

            self.calibration_recording = recording
            self.calibration_recording_path = file_path

            # Update UI
            filename = os.path.basename(file_path)
            emg_shape = recording['emg'].shape
            gt_shape = recording['gt'].shape if hasattr(recording['gt'], 'shape') else len(recording['gt'])
            duration = emg_shape[1] / config.FSAMP

            self.calibration_recording_label.setText(
                f"{filename} ({duration:.1f}s)"
            )
            self.calibration_recording_label.setStyleSheet("color: green;")

            # Enable run button only if model is also loaded
            if self.model_interface.model_is_loaded:
                self.run_calibration_button.setEnabled(True)

            print(f"[CALIBRATION] Loaded recording: {filename}")
            print(f"  EMG shape: {emg_shape}")
            print(f"  GT shape: {gt_shape}")
            print(f"  Duration: {duration:.1f}s")

        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "Error Loading Recording",
                f"Failed to load recording:\n{str(e)}"
            )

    def _run_calibration(self) -> None:
        """Run offline calibration to find plateau thresholds (in background thread)."""
        if not self.calibration_recording or not self.model_interface.model_is_loaded:
            return

        print("\n" + "=" * 70)
        print("RUNNING OFFLINE THRESHOLD CALIBRATION (Background Thread)")
        print("=" * 70)

        # Disable button and show progress
        self.run_calibration_button.setEnabled(False)
        self.run_calibration_button.setText("Calibrating...")
        self.calibration_results_label.setText("Computing DTW distances... Please wait.")
        self.calibration_results_label.setVisible(True)

        # Create worker and thread
        self._cal_thread = QThread()
        self._cal_worker = CalibrationWorker(
            self.calibration_recording,
            self.model_interface.model
        )
        self._cal_worker.moveToThread(self._cal_thread)

        # Connect signals
        self._cal_thread.started.connect(self._cal_worker.run)
        self._cal_worker.finished.connect(self._on_calibration_finished)
        self._cal_worker.error.connect(self._on_calibration_error)
        self._cal_worker.finished.connect(self._cal_thread.quit)
        self._cal_worker.error.connect(self._cal_thread.quit)
        self._cal_thread.finished.connect(self._cal_thread.deleteLater)

        # Start the thread
        self._cal_thread.start()

    def _on_calibration_finished(self, result: dict, thresholds: dict) -> None:
        """Handle calibration completion."""
        self.calibration_result = result
        self.calibration_thresholds = thresholds

        # Display results
        open_plateau = thresholds['open_plateau']
        closed_plateau = thresholds['closed_plateau']
        threshold_open = thresholds['threshold_open']
        threshold_closed = thresholds['threshold_closed']
        open_std = thresholds['open_std']
        closed_std = thresholds['closed_std']

        print(f"\n  OPEN plateau:   {open_plateau:.4f} (std={open_std:.4f})")
        print(f"  OPEN threshold: {threshold_open:.4f}")
        print(f"\n  CLOSED plateau: {closed_plateau:.4f} (std={closed_std:.4f})")
        print(f"  CLOSED threshold: {threshold_closed:.4f}")
        print("=" * 70 + "\n")

        # Update UI
        self.run_calibration_button.setEnabled(True)
        self.run_calibration_button.setText("Run Calibration")
        self.calibration_results_label.setText(
            f"OPEN plateau:     {open_plateau:.4f}\n"
            f"OPEN threshold:   {threshold_open:.4f} (plateau + 1.0 × std)\n\n"
            f"CLOSED plateau:   {closed_plateau:.4f}\n"
            f"CLOSED threshold: {threshold_closed:.4f} (plateau + 1.0 × std)"
        )
        self.calibration_results_label.setVisible(True)
        self.apply_calibration_button.setEnabled(True)

        # Plot calibration results
        self._plot_calibration_results()

    def _on_calibration_error(self, error_msg: str) -> None:
        """Handle calibration error."""
        self.run_calibration_button.setEnabled(True)
        self.run_calibration_button.setText("Run Calibration")
        self.calibration_results_label.setText(f"Error: {error_msg}")
        self.calibration_results_label.setStyleSheet(
            "font-family: monospace; color: red; background-color: #fff0f0; padding: 8px; border-radius: 4px;"
        )
        print(f"[CALIBRATION ERROR] {error_msg}")

    def _plot_calibration_results(self) -> None:
        """Plot calibration results showing distances and computed thresholds.

        For guided bidirectional recordings, also shows:
        - Audio cue markers (blue dashed vertical lines)
        - GT ramps (linear transitions 0→1 and 1→0)
        - Reaction time periods (cyan shaded regions)
        """
        if not self.calibration_result or not self.calibration_thresholds:
            return

        emg = self.calibration_recording['emg']
        gt = self.calibration_recording['gt']
        if hasattr(gt, 'flatten'):
            gt = gt.flatten()

        # Get calibration data
        timestamps = self.calibration_result['timestamps']
        D_open = self.calibration_result['D_open']
        D_closed = self.calibration_result['D_closed']
        gt_at_dtw = self.calibration_result['gt_at_dtw']

        # Thresholds
        threshold_open = self.calibration_thresholds['threshold_open']
        threshold_closed = self.calibration_thresholds['threshold_closed']
        plateau_open = self.calibration_thresholds['open_plateau']
        plateau_closed = self.calibration_thresholds['closed_plateau']

        # Time axis for EMG
        time_emg = np.arange(emg.shape[1]) / config.FSAMP

        # Check if this is a guided bidirectional recording
        is_guided = self.calibration_recording.get('gt_mode') == 'guided_animation'
        cycles = self.calibration_recording.get('cycles', [])
        timings_emg = self.calibration_recording.get('timings_emg')
        recording_start_time = timings_emg[0] if timings_emg is not None and len(timings_emg) > 0 else None

        # Extract cue times from cycles (convert absolute timestamps to seconds from start)
        close_cue_times = []
        open_cue_times = []
        reaction_times = []  # List of (cue_time, reaction_duration) tuples

        if is_guided and cycles and recording_start_time is not None:
            for cycle in cycles:
                close_cue_time = cycle.get('close_cue_time')
                open_cue_time = cycle.get('open_cue_time')
                timing_config = cycle.get('timing_config', {})
                reaction_time_s = timing_config.get('reaction_time_s', 0.2)

                if close_cue_time is not None:
                    cue_s = close_cue_time - recording_start_time
                    close_cue_times.append(cue_s)
                    reaction_times.append((cue_s, reaction_time_s, 'close'))

                if open_cue_time is not None:
                    cue_s = open_cue_time - recording_start_time
                    open_cue_times.append(cue_s)
                    reaction_times.append((cue_s, reaction_time_s, 'open'))

        # Create 4-subplot figure
        fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        # Plot 1: EMG with GT overlay
        ax = axs[0]
        ax.plot(time_emg, emg[0, :], 'b-', linewidth=0.5, alpha=0.7)
        emg_max = np.max(np.abs(emg[0, :])) if np.max(np.abs(emg[0, :])) > 0 else 1
        # Ensure gt has same length as time_emg for fill_between
        gt_for_plot = gt[:len(time_emg)] if len(gt) >= len(time_emg) else np.pad(gt, (0, len(time_emg) - len(gt)))
        ax.fill_between(time_emg, -emg_max, emg_max, where=gt_for_plot > 0.5,
                        alpha=0.15, color='green', label='GT=CLOSED')
        ax.set_ylabel("EMG Ch1 (µV)")
        ax.set_title("Offline Threshold Calibration" + (" (Guided Recording)" if is_guided else ""))
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot 2: D_open
        ax = axs[1]
        ax.plot(timestamps, D_open, 'g-', linewidth=1, label='D_open')
        ax.axhline(threshold_open, color='r', linestyle='--', linewidth=2,
                   label=f'Threshold: {threshold_open:.4f}')
        ax.axhline(plateau_open, color='b', linestyle=':', linewidth=2,
                   label=f'Plateau: {plateau_open:.4f}')
        ax.set_ylabel("DTW Distance")
        ax.set_title("Distance to OPEN templates (checked when hand CLOSED)")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot 3: D_closed
        ax = axs[2]
        ax.plot(timestamps, D_closed, 'orange', linewidth=1, label='D_closed')
        ax.axhline(threshold_closed, color='r', linestyle='--', linewidth=2,
                   label=f'Threshold: {threshold_closed:.4f}')
        ax.axhline(plateau_closed, color='b', linestyle=':', linewidth=2,
                   label=f'Plateau: {plateau_closed:.4f}')
        ax.set_ylabel("DTW Distance")
        ax.set_title("Distance to CLOSED templates (checked when hand OPEN)")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot 4: GT with cues and ramps
        ax = axs[3]

        # Plot GT - use line plot for ramps if guided, step for binary
        gt_len = min(len(gt), len(time_emg))
        if is_guided:
            # Guided recording: GT has linear ramps (0.0 → 1.0 during closing, 1.0 → 0.0 during opening)
            ax.plot(time_emg[:gt_len], gt[:gt_len], 'purple', linewidth=2, label='GT (with ramps)')

            # Add audio cue markers
            for i, cue_time in enumerate(close_cue_times):
                label = 'Close cue' if i == 0 else None
                ax.axvline(cue_time, color='#1976D2', linestyle='--', linewidth=1.5,
                           alpha=0.8, label=label)

            for i, cue_time in enumerate(open_cue_times):
                label = 'Open cue' if i == 0 else None
                ax.axvline(cue_time, color='#0288D1', linestyle='-.', linewidth=1.5,
                           alpha=0.8, label=label)

            # Add reaction time shading (between cue and GT transition start)
            for cue_time, reaction_s, cue_type in reaction_times:
                color = '#FFF3E0' if cue_type == 'close' else '#E3F2FD'  # Light orange or light blue
                ax.axvspan(cue_time, cue_time + reaction_s, alpha=0.3, color=color)

            ax.legend(loc='upper right', fontsize=8)
        else:
            # Non-guided recording: binary GT
            ax.step(time_emg[:gt_len], gt[:gt_len], 'purple', linewidth=2, where='post')

        ax.set_ylabel("GT State")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['OPEN', 'CLOSED'])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def _apply_calibration(self) -> None:
        """Apply calibrated thresholds to sliders."""
        if not self.calibration_thresholds:
            return

        threshold_open = self.calibration_thresholds['threshold_open']
        threshold_closed = self.calibration_thresholds['threshold_closed']

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


######################################

    # # Modify _load_model to handle diagnostic mode:
    # def _load_model(self) -> None:
    #     if config.DIAGNOSTIC_MODE:
    #         # In diagnostic mode, don't need to select a file
    #         self.model_interface.load_model(None)  # Loads DiagnosticModel
    #         self.online_model_label.setText("DIAGNOSTIC MODE (no model file)")
    #         return
        
    #     # Normal model loading code
    #     if not os.path.exists(self.model_dir_path):
    #         os.makedirs(self.model_dir_path)

    #     dialog = QFileDialog(self.main_window)
    #     dialog.setFileMode(QFileDialog.ExistingFile)

    #     file_name = dialog.getOpenFileName(
    #         self.main_window,
    #         "Open Model",
    #         self.model_dir_path,
    #     )[0]

    #     if not file_name:
    #         return

    #     self.model_interface.load_model(file_name)
    #     label = file_name.split("/")[-1].split("_")[-1].split(".")[0]
    #     self.online_model_label.setText(f"{label} loaded.")


    # # Optionally add a method to print diagnostic summary when stopping recording:
    # def _toggle_recording(self):
    #     if self.online_record_toggle_push_button.isChecked():
    #         self.timings = []
    #         self.online_record_toggle_push_button.setText("Stop Recording")
    #         self.online_load_model_push_button.setEnabled(False)

    #         # connect signals
    #         self.main_window.device.ready_read_signal.connect(self.online_emg_update)

    #         self.emg_buffer = []
    #         self.kinematics_buffer = []
    #         self.emg_timings_buffer = []
    #         self.kinematics_timings_buffer = []
    #         self.predictions_buffer = []
    #     else:
    #         self.online_record_toggle_push_button.setText("Start Recording")
    #         self.online_load_model_push_button.setEnabled(True)
    #         self.main_window.device.ready_read_signal.disconnect(self.online_emg_update)
            
    #         # Print diagnostic summary if in diagnostic mode
    #         if config.DIAGNOSTIC_MODE:
    #             print("\n" + "="*70)
    #             print("STOPPING RECORDING - PRINTING DIAGNOSTIC SUMMARY")
    #             print("="*70)
    #             self.model_interface.print_summary()
            
    #         self._save_data()