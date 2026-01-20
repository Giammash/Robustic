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
        s_open = history.get("s_open", 1.0)
        s_closed = history.get("s_closed", 1.0)
        fig.suptitle(f"Online Session Distance Analysis (s_open={s_open:.2f}, s_closed={s_closed:.2f})",
                     fontsize=14, fontweight='bold')

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
        """Update the threshold labels after model load."""
        thresholds = self.model_interface.get_current_thresholds()
        if thresholds and thresholds.get("threshold_open") is not None:
            s_open = thresholds.get("s_open", 1.0)
            s_closed = thresholds.get("s_closed", 1.0)

            # Update slider positions to match model's s values
            self.threshold_slider_open.blockSignals(True)
            self.threshold_slider_open.setValue(int(s_open * 100))
            self.threshold_slider_open.blockSignals(False)

            self.threshold_slider_closed.blockSignals(True)
            self.threshold_slider_closed.setValue(int(s_closed * 100))
            self.threshold_slider_closed.blockSignals(False)

            # Update labels
            self.threshold_open_label.setText(
                f"s_open = {s_open:.2f} | Threshold: {thresholds['threshold_open']:.4f}"
            )
            self.threshold_closed_label.setText(
                f"s_closed = {s_closed:.2f} | Threshold: {thresholds['threshold_closed']:.4f}"
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
        """Setup threshold tuning sliders UI (separate for OPEN and CLOSED)."""
        # Create a group box for threshold tuning
        self.threshold_group_box = QGroupBox("Threshold Tuning")

        main_layout = QVBoxLayout()

        # --- OPEN threshold slider ---
        open_label = QLabel("s_open (OPEN threshold):")
        main_layout.addWidget(open_label)

        open_slider_layout = QHBoxLayout()
        open_slider_layout.addWidget(QLabel("0.0"))

        self.threshold_slider_open = QSlider(Qt.Horizontal)
        self.threshold_slider_open.setMinimum(0)      # s = 0.0
        self.threshold_slider_open.setMaximum(500)    # s = 5.0
        self.threshold_slider_open.setValue(100)      # Default s=1.0
        self.threshold_slider_open.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider_open.setTickInterval(100)  # Tick every 1.0
        self.threshold_slider_open.valueChanged.connect(self._on_threshold_open_changed)
        open_slider_layout.addWidget(self.threshold_slider_open)

        open_slider_layout.addWidget(QLabel("5.0"))
        main_layout.addLayout(open_slider_layout)

        self.threshold_open_label = QLabel("s_open = 1.00 | Threshold: Not loaded")
        main_layout.addWidget(self.threshold_open_label)

        # --- CLOSED threshold slider ---
        closed_label = QLabel("s_closed (CLOSED threshold):")
        main_layout.addWidget(closed_label)

        closed_slider_layout = QHBoxLayout()
        closed_slider_layout.addWidget(QLabel("0.0"))

        self.threshold_slider_closed = QSlider(Qt.Horizontal)
        self.threshold_slider_closed.setMinimum(0)    # s = 0.0
        self.threshold_slider_closed.setMaximum(500)  # s = 5.0
        self.threshold_slider_closed.setValue(100)    # Default s=1.0
        self.threshold_slider_closed.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider_closed.setTickInterval(100)  # Tick every 1.0
        self.threshold_slider_closed.valueChanged.connect(self._on_threshold_closed_changed)
        closed_slider_layout.addWidget(self.threshold_slider_closed)

        closed_slider_layout.addWidget(QLabel("5.0"))
        main_layout.addLayout(closed_slider_layout)

        self.threshold_closed_label = QLabel("s_closed = 1.00 | Threshold: Not loaded")
        main_layout.addWidget(self.threshold_closed_label)

        self.threshold_group_box.setLayout(main_layout)

        # Add to the online commands group box layout
        if self.online_commands_group_box.layout():
            self.online_commands_group_box.layout().addWidget(self.threshold_group_box)

    def _on_threshold_open_changed(self, value: int) -> None:
        """Called when OPEN threshold slider is moved."""
        s_open = value / 100.0  # Convert back to float (0-500 -> 0.0-5.0)

        # Update model threshold
        self.model_interface.update_threshold_open(s_open)

        # Update label
        thresholds = self.model_interface.get_current_thresholds()
        if thresholds and thresholds.get("threshold_open") is not None:
            self.threshold_open_label.setText(
                f"s_open = {s_open:.2f} | Threshold: {thresholds['threshold_open']:.4f}"
            )
        else:
            self.threshold_open_label.setText(f"s_open = {s_open:.2f} | Threshold: Not loaded")

    def _on_threshold_closed_changed(self, value: int) -> None:
        """Called when CLOSED threshold slider is moved."""
        s_closed = value / 100.0  # Convert back to float (0-500 -> 0.0-5.0)

        # Update model threshold
        self.model_interface.update_threshold_closed(s_closed)

        # Update label
        thresholds = self.model_interface.get_current_thresholds()
        if thresholds and thresholds.get("threshold_closed") is not None:
            self.threshold_closed_label.setText(
                f"s_closed = {s_closed:.2f} | Threshold: {thresholds['threshold_closed']:.4f}"
            )
        else:
            self.threshold_closed_label.setText(f"s_closed = {s_closed:.2f} | Threshold: Not loaded")


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