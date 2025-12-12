from __future__ import annotations
from typing import TYPE_CHECKING
from PySide6.QtCore import QObject
import numpy as np
from datetime import datetime
import pickle
import os
import time
from PySide6.QtWidgets import QFileDialog

# MindMove imports
from mindmove.model.interface import MindMoveInterface

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
            parent=self.main_window
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
        # Check for connections!
        if self.online_record_toggle_push_button.isChecked():
            self.timings = []
            self.online_record_toggle_push_button.setText("Stop Recording")
            self.online_load_model_push_button.setEnabled(False)
            self.main_window.device.ready_read_signal.connect(self.online_emg_update)

            self.emg_buffer = []
            self.kinematics_buffer = []
            self.emg_timings_buffer = []
            self.kinematics_timings_buffer = []
            self.predictions_buffer = []
        else:
            self.online_record_toggle_push_button.setText("Start Recording")
            self.online_load_model_push_button.setEnabled(True)
            self.main_window.device.ready_read_signal.disconnect(self.online_emg_update)
            self._save_data()

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
        if not os.path.exists(self.model_dir_path):
            os.makedirs(self.model_dir_path)

        dialog = QFileDialog(self.main_window)
        dialog.setFileMode(QFileDialog.ExistingFile)

        file_name = dialog.getOpenFileName(
            self.main_window,
            "Open Model",
            self.model_dir_path,
        )[0]

        # TODO: Load model using the model interface and model class
        self.model_interface.load_model(file_name)

        label = file_name.split("/")[-1].split("_")[-1].split(".")[0]
        self.online_model_label.setText(f"{label} loaded.")

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
