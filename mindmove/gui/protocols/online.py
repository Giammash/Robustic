from __future__ import annotations
from typing import TYPE_CHECKING
from PySide6.QtCore import QObject
import numpy as np
from datetime import datetime
import pickle
import os
import time
from PySide6.QtWidgets import QFileDialog, QSlider, QLabel, QHBoxLayout, QVBoxLayout, QGroupBox
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt

# MindMove imports
from mindmove.model.interface import MindMoveInterface

from mindmove.config import config

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

        # Buffers
        self.emg_buffer: list[np.ndarray] = []
        self.kinematics_buffer: list[list[float]] = []
        self.emg_timings_buffer: list[float] = []
        self.kinematics_timings_buffer: list[float] = []
        self.predictions_buffer: list[list[float]] = []

        # File management
        self.prediction_dir_path: str = "data/predictions/"
        self.model_dir_path: str = "data/models/"
        


    def online_emg_update(self, data: np.ndarray) -> None:
        # TODO: Implement online prediction in model interface and model class
        emg_data = self.main_window.device.extract_emg_data(data)
        # shape (32, nsamp)
        # forward to model interface: the Model.predict must handel buffer inside model

        prediction = self.model_interface.predict(emg_data)

        # Stream prediction values to the virtual hand interface
        self.main_window.virtual_hand_interface.output_message_signal.emit(
            str(prediction).encode("utf-8")
        )

        if self.online_record_toggle_push_button.isChecked():
            # Store with timestamp for proper reconstruction (same format as record protocol)
            self.emg_buffer.append((time.time(), emg_data))
            self.predictions_buffer.append(prediction)
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

            # Plot distance history
            self._plot_distance_history()

            self._save_data()

    def _plot_distance_history(self) -> None:
        """
        Plot distance history after stopping recording.

        Creates a 3-subplot figure:
        - Top: EMG signal reconstruction
        - Middle: State over time (0=OPEN, 1=CLOSED)
        - Bottom: Distance to opposite-state templates with threshold

        For long recordings (>100s), creates multiple plots.
        """
        history = self.model_interface.get_distance_history()

        if history is None or len(history.get("timestamps", [])) == 0:
            print("[PLOT] No distance history to plot")
            return

        timestamps = np.array(history["timestamps"])
        D_open = history["D_open"]
        D_closed = history["D_closed"]
        states = history["states"]
        threshold_open = history["threshold_open"]
        threshold_closed = history["threshold_closed"]
        state_transitions = history.get("state_transitions", [])
        s_open = history.get("s_open", 1.0)
        s_closed = history.get("s_closed", 1.0)

        # Reconstruct continuous EMG signal from buffer
        # emg_buffer is now list of (timestamp, emg_data) tuples
        if self.emg_buffer:
            emg_timestamps = np.array([t for t, _ in self.emg_buffer])
            emg_signal = np.hstack([emg_data for _, emg_data in self.emg_buffer])  # (32, total_samples)
            n_channels, n_samples = emg_signal.shape

            # Create continuous time axis for EMG
            # Each packet has a timestamp; reconstruct sample-level times
            emg_time_axis = np.zeros(n_samples)
            sample_idx = 0
            for i, (pkt_time, pkt_data) in enumerate(self.emg_buffer):
                pkt_samples = pkt_data.shape[1]
                if i == 0:
                    # First packet: samples end at pkt_time
                    pkt_duration = pkt_samples / config.FSAMP
                    emg_time_axis[sample_idx:sample_idx + pkt_samples] = np.linspace(
                        pkt_time - pkt_duration, pkt_time, pkt_samples, endpoint=False
                    )
                else:
                    # Subsequent packets: interpolate between timestamps
                    prev_time = self.emg_buffer[i-1][0]
                    emg_time_axis[sample_idx:sample_idx + pkt_samples] = np.linspace(
                        prev_time, pkt_time, pkt_samples, endpoint=False
                    )
                sample_idx += pkt_samples
        else:
            emg_signal = None
            emg_time_axis = None

        # Normalize timestamps to start from 0
        t0 = timestamps[0] if len(timestamps) > 0 else 0
        timestamps_rel = timestamps - t0

        if emg_time_axis is not None:
            emg_time_rel = emg_time_axis - t0

        # Total duration
        total_duration = timestamps_rel[-1] if len(timestamps_rel) > 0 else 0

        # Window size for plotting (100 seconds)
        window_size = 100.0
        n_windows = max(1, int(np.ceil(total_duration / window_size)))

        print(f"\n[PLOT] Creating {n_windows} plot(s) for {total_duration:.1f}s recording")
        print(f"       DTW computations: {len(timestamps)}")
        print(f"       EMG samples: {emg_signal.shape[1] if emg_signal is not None else 0}")
        print(f"       State transitions: {len(state_transitions)}")

        for window_idx in range(n_windows):
            t_start = window_idx * window_size
            t_end = min((window_idx + 1) * window_size, total_duration + 1)

            # Filter DTW data for this window
            mask = (timestamps_rel >= t_start) & (timestamps_rel < t_end)
            t_window = timestamps_rel[mask]

            if len(t_window) == 0:
                continue

            # Create figure with 3 subplots
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

            # --- Plot 1: EMG Signal ---
            ax1 = axes[0]
            if emg_signal is not None and emg_time_rel is not None:
                # Filter EMG samples for this time window
                emg_mask = (emg_time_rel >= t_start) & (emg_time_rel < t_end)
                if np.any(emg_mask):
                    emg_times_window = emg_time_rel[emg_mask]
                    emg_values_window = emg_signal[0, emg_mask]  # Channel 0
                    ax1.plot(emg_times_window, emg_values_window, 'b-', linewidth=0.5, alpha=0.7)

            ax1.set_ylabel("EMG Ch1 (µV)", fontsize=11)
            ax1.set_title(f"Online Session Analysis - Window {window_idx + 1}/{n_windows} "
                         f"[{t_start:.0f}s - {t_end:.0f}s]", fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # --- Plot 2: State Over Time ---
            ax2 = axes[1]
            states_window = [states[i] for i in range(len(states)) if mask[i]]
            states_numeric = [1 if s == "CLOSED" else 0 for s in states_window]

            if len(t_window) > 0 and len(states_numeric) > 0:
                ax2.step(t_window, states_numeric, 'purple', linewidth=2, where='post')
                ax2.fill_between(t_window, 0, states_numeric, step='post', alpha=0.3, color='purple')

            ax2.set_ylabel("State", fontsize=11)
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['OPEN', 'CLOSED'])
            ax2.grid(True, alpha=0.3)

            # Mark state transitions in this window
            for trans_time, from_state, to_state in state_transitions:
                trans_t = trans_time - t0
                if t_start <= trans_t < t_end:
                    color = 'green' if to_state == "OPEN" else 'orange'
                    ax2.axvline(x=trans_t, color=color, linestyle='-', linewidth=2, alpha=0.7)

            # --- Plot 3: Distance to Opposite-State Templates ---
            ax3 = axes[2]

            # For each point, plot the distance that was computed (opposite state)
            # When state=CLOSED, we computed D_open (checking if should open)
            # When state=OPEN, we computed D_closed (checking if should close)

            distances_to_plot = []
            thresholds_to_plot = []
            colors = []

            for i, (t, state) in enumerate(zip(timestamps_rel, states)):
                if not (t_start <= t < t_end):
                    continue

                if state == "CLOSED":
                    # Was computing D_open to check if should open
                    if D_open[i] is not None:
                        distances_to_plot.append((t, D_open[i]))
                        thresholds_to_plot.append((t, threshold_open))
                        colors.append('green')  # Green for D_open
                else:  # OPEN
                    # Was computing D_closed to check if should close
                    if D_closed[i] is not None:
                        distances_to_plot.append((t, D_closed[i]))
                        thresholds_to_plot.append((t, threshold_closed))
                        colors.append('red')  # Red for D_closed

            if distances_to_plot:
                t_dist = [d[0] for d in distances_to_plot]
                d_vals = [d[1] for d in distances_to_plot]
                t_thresh = [th[0] for th in thresholds_to_plot]
                th_vals = [th[1] for th in thresholds_to_plot]

                # Plot distances with color indicating which template set
                for i in range(len(t_dist)):
                    ax3.scatter(t_dist[i], d_vals[i], c=colors[i], s=15, alpha=0.7)

                # Plot threshold line (step function that changes with state)
                ax3.step(t_thresh, th_vals, 'k--', linewidth=2, where='post',
                        label='Threshold (adaptive)')

                # Add legend
                ax3.scatter([], [], c='green', s=30, label=f'D_open (T={threshold_open:.3f})')
                ax3.scatter([], [], c='red', s=30, label=f'D_closed (T={threshold_closed:.3f})')
                ax3.legend(loc='upper right', fontsize=9)

            ax3.set_xlabel("Time (seconds)", fontsize=11)
            ax3.set_ylabel("DTW Distance", fontsize=11)
            ax3.set_title(f"Distance to Opposite-State Templates (s_open={s_open:.2f}, s_closed={s_closed:.2f})",
                         fontsize=11)
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            # Use non-blocking show to avoid Qt event loop conflict
            plt.show(block=False)
            plt.pause(0.1)  # Small pause to ensure the window renders

        print(f"[PLOT] Done plotting {n_windows} window(s)")

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

    def _load_model(self) -> None:
        if config.DIAGNOSTIC_MODE:
            # In diagnostic mode, don't need to select a file
            self.model_interface.load_model(None)  # Loads DiagnosticModel
            self.online_model_label.setText("DIAGNOSTIC MODE (no model file)")
            self._update_threshold_display()
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

        # Add threshold tuning slider (programmatically)
        self._setup_threshold_slider()

    def _setup_threshold_slider(self) -> None:
        """Setup threshold tuning sliders UI (direct threshold control)."""
        # Create a group box for threshold tuning
        self.threshold_group_box = QGroupBox("Threshold Tuning")

        # Store max threshold values (will be updated when model is loaded)
        self._max_threshold_open = 1.0  # Default, updated on model load
        self._max_threshold_closed = 1.0  # Default, updated on model load
        self._slider_resolution = 1000  # Slider units per threshold range

        main_layout = QVBoxLayout()

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