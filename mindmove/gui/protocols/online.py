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
            self.emg_buffer.append(emg_data)
            self.predictions_buffer.append(prediction)
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
        """Plot distance history after stopping recording."""
        history = self.model_interface.get_distance_history()

        if history is None or len(history.get("timestamps", [])) == 0:
            print("[PLOT] No distance history to plot")
            return

        timestamps = history["timestamps"]
        D_open = history["D_open"]
        D_closed = history["D_closed"]
        states = history["states"]
        threshold_open = history["threshold_open"]
        threshold_closed = history["threshold_closed"]
        state_transitions = history.get("state_transitions", [])

        # Normalize transition timestamps
        t0 = timestamps[0] if timestamps else 0
        t0_abs = history["timestamps"][0] if history["timestamps"] else 0

        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # --- Plot 1: Distance to OPEN templates ---
        ax1 = axes[0]
        # Filter out None values for plotting
        t_open = [t for t, d in zip(timestamps, D_open) if d is not None]
        d_open_valid = [d for d in D_open if d is not None]

        if t_open and d_open_valid:
            ax1.scatter(t_open, d_open_valid, c='blue', s=10, alpha=0.7, label='D_open (when CLOSED)')
            ax1.plot(t_open, d_open_valid, 'b-', alpha=0.3, linewidth=0.5)

        # Threshold line
        if threshold_open is not None:
            ax1.axhline(y=threshold_open, color='blue', linestyle='--', linewidth=2,
                       label=f'Threshold OPEN: {threshold_open:.4f}')

        # Mark state transitions
        for trans_time, from_state, to_state in state_transitions:
            trans_t = trans_time - t0_abs
            if to_state == "OPEN":
                ax1.axvline(x=trans_t, color='green', linestyle='-', linewidth=2, alpha=0.7)

        ax1.set_ylabel("Distance to OPEN templates", fontsize=12)
        ax1.set_title("Distance to OPEN Templates (computed when state is CLOSED)", fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # --- Plot 2: Distance to CLOSED templates ---
        ax2 = axes[1]
        # Filter out None values for plotting
        t_closed = [t for t, d in zip(timestamps, D_closed) if d is not None]
        d_closed_valid = [d for d in D_closed if d is not None]

        if t_closed and d_closed_valid:
            ax2.scatter(t_closed, d_closed_valid, c='red', s=10, alpha=0.7, label='D_closed (when OPEN)')
            ax2.plot(t_closed, d_closed_valid, 'r-', alpha=0.3, linewidth=0.5)

        # Threshold line
        if threshold_closed is not None:
            ax2.axhline(y=threshold_closed, color='red', linestyle='--', linewidth=2,
                       label=f'Threshold CLOSED: {threshold_closed:.4f}')

        # Mark state transitions
        for trans_time, from_state, to_state in state_transitions:
            trans_t = trans_time - t0_abs
            if to_state == "CLOSED":
                ax2.axvline(x=trans_t, color='orange', linestyle='-', linewidth=2, alpha=0.7)

        ax2.set_xlabel("Time (seconds)", fontsize=12)
        ax2.set_ylabel("Distance to CLOSED templates", fontsize=12)
        ax2.set_title("Distance to CLOSED Templates (computed when state is OPEN)", fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Add info text
        s_value = history.get("threshold_s", 1.0)
        fig.suptitle(f"Online Session Distance Analysis (s = {s_value:.2f})", fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.show()

        print(f"\n[PLOT] Distance history plotted: {len(timestamps)} DTW computations")
        print(f"       State transitions: {len(state_transitions)}")

    def _save_data(self) -> None:
        # TODO: add code to save buffered data
        save_pickle_dict = {
            "emg": np.array(self.emg_buffer),
            "kinematics": np.array(self.kinematics_buffer),
            "timings_emg": np.array(self.emg_timings_buffer),
            "timings_kinematics": np.array(self.kinematics_timings_buffer),
            "label": np.array(self.model_label),
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
        """Update the threshold value label after model load or slider change."""
        thresholds = self.model_interface.get_current_thresholds()
        if thresholds and thresholds.get("threshold_open") is not None:
            s = thresholds.get("threshold_s", 1.0)
            # Update slider position to match model's s value
            self.threshold_slider.blockSignals(True)
            self.threshold_slider.setValue(int(s * 100))
            self.threshold_slider.blockSignals(False)

            self.threshold_value_label.setText(
                f"s = {s:.2f} | OPEN: {thresholds['threshold_open']:.4f} | CLOSED: {thresholds['threshold_closed']:.4f}"
            )

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
        """Setup threshold tuning slider UI."""
        # Create a group box for threshold tuning
        self.threshold_group_box = QGroupBox("Threshold Tuning (s value)")

        layout = QVBoxLayout()

        # Slider row
        slider_layout = QHBoxLayout()

        # Min label
        min_label = QLabel("0.5")
        slider_layout.addWidget(min_label)

        # Slider (0.5 to 3.0, mapped as 50 to 300 for integer slider)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(50)
        self.threshold_slider.setMaximum(300)
        self.threshold_slider.setValue(100)  # Default s=1.0
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(50)
        self.threshold_slider.valueChanged.connect(self._on_threshold_slider_changed)
        slider_layout.addWidget(self.threshold_slider)

        # Max label
        max_label = QLabel("3.0")
        slider_layout.addWidget(max_label)

        layout.addLayout(slider_layout)

        # Value display
        self.threshold_value_label = QLabel("s = 1.00 | Thresholds: Not loaded")
        layout.addWidget(self.threshold_value_label)

        self.threshold_group_box.setLayout(layout)

        # Add to the online commands group box layout
        if self.online_commands_group_box.layout():
            self.online_commands_group_box.layout().addWidget(self.threshold_group_box)

    def _on_threshold_slider_changed(self, value: int) -> None:
        """Called when threshold slider is moved."""
        s = value / 100.0  # Convert back to float (50-300 -> 0.5-3.0)

        # Update model thresholds
        self.model_interface.update_thresholds(s)

        # Update label
        thresholds = self.model_interface.get_current_thresholds()
        if thresholds and thresholds.get("threshold_open") is not None:
            self.threshold_value_label.setText(
                f"s = {s:.2f} | OPEN: {thresholds['threshold_open']:.4f} | CLOSED: {thresholds['threshold_closed']:.4f}"
            )
        else:
            self.threshold_value_label.setText(f"s = {s:.2f} | Thresholds: Not loaded")


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