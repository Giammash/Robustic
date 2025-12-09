from __future__ import annotations
from typing import TYPE_CHECKING
from PySide6.QtCore import QObject
import time
import numpy as np
import pickle
import os
from datetime import datetime

if TYPE_CHECKING:
    from mindmove.gui.mindmove import MindMove


class RecordProtocol(QObject):
    def __init__(self, parent: MindMove | None = ...) -> None:
        super().__init__(parent)

        self.main_window = parent

        # Initialize Protocol UI
        self._setup_protocol_ui()

        # Initialize Protocol
        self.current_task: str = None
        self.emg_sampling_frequency: int = 2000
        self.kinematics_sampling_frequency: int = 60
        self.recording_time: int = self.record_duration_spin_box.value()
        self.emg_recording_time: int = int(
            self.recording_time * self.emg_sampling_frequency
        )
        self.kinematics_recording_time: int = int(
            self.recording_time * self.kinematics_sampling_frequency
        )
        self.emg_buffer: list[(int, np.ndarray)] = []
        self.kinematics_buffer: list[(int, np.ndarray)] = []

        self.has_finished_emg: bool = False
        self.has_finished_kinematics: bool = False

        self.start_time: float = None

        # File management:
        self.recording_dir_path: str = "data/recordings/"

    def emg_update(self, data: np.ndarray) -> None:
        self.emg_buffer.append((time.time(), data))
        current_samples = len(self.emg_buffer) * self.emg_buffer[0][1].shape[1]
        self._set_emg_progress_bar(current_samples)
        if current_samples >= self.emg_recording_time:
            print("EMG recording finished at: ", time.time() - self.start_time)
            self.has_finished_emg = True
            self.main_window.device.ready_read_signal.disconnect(self.emg_update)
            self.finished_recording()

    def kinematics_update(self, data: np.ndarray) -> None:
        self.kinematics_buffer.append((time.time(), data))
        current_samples = len(self.kinematics_buffer)
        self._set_kinematics_progress_bar(current_samples)
        if current_samples >= self.kinematics_recording_time:
            print("Kinematics recording finished at: ", time.time() - self.start_time)
            self.has_finished_kinematics = True
            self.main_window.virtual_hand_interface.input_message_signal.disconnect(
                self.kinematics_update
            )
            self.finished_recording()

    def _start_recording(self, checked: bool) -> None:
        if checked:
            if not self.main_window.device.device.is_streaming:
                print("Device is not connected!")
                self.record_toggle_push_button.setChecked(False)
                return

            if not self.main_window.virtual_hand_interface.is_streaming:
                print("Virtual Hand Interface is not connected!")
                self.record_toggle_push_button.setChecked(False)
                return

            self.main_window.virtual_hand_interface.input_message_signal.connect(
                self.kinematics_update
            )

            self.main_window.device.ready_read_signal.connect(self.emg_update)

            self.start_time = time.time()

            # Reset buffers
            self.emg_buffer = []
            self.kinematics_buffer = []

            # Set duration time
            self.recording_time: int = self.record_duration_spin_box.value()
            self.emg_recording_time: int = int(
                self.recording_time * self.emg_sampling_frequency
            )
            self.kinematics_recording_time: int = int(
                self.recording_time * self.kinematics_sampling_frequency
            )

            self.record_toggle_push_button.setText("Recording...")
            self.record_group_box.setEnabled(False)
            self.current_task: str = self.record_task_combo_box.currentText()

            self.has_finished_emg = False
            self.has_finished_kinematics = False

    def _set_emg_progress_bar(self, value: int) -> None:
        self.record_emg_progress_bar.setValue(value / self.emg_recording_time * 100)

    def _set_kinematics_progress_bar(self, value: int) -> None:
        self.record_kinematics_progress_bar.setValue(
            value / self.kinematics_recording_time * 100
        )

    def finished_recording(self) -> None:
        if not self.has_finished_kinematics:
            print("Kinematics recording not finished yet!")
            return

        if not self.has_finished_emg:
            print("EMG recording not finished yet!")
            return

        self.review_recording_stacked_widget.setCurrentIndex(1)
        self.record_toggle_push_button.setText("Finished Recording")
        self.review_recording_task_label.setText(self.current_task.capitalize())

        # Plot Kinematics Signal in Preview Window
        kinematics_signal = np.vstack([data for _, data in self.kinematics_buffer]).T
        self.review_recording_kinematics_plot_widget.refresh_plot()
        self.review_recording_kinematics_plot_widget.configure_lines_plot(
            display_time=self.recording_time,
            fs=self.kinematics_sampling_frequency,
            lines=kinematics_signal.shape[0],
        )
        self.review_recording_kinematics_plot_widget.set_plot_data(kinematics_signal)

        # Plot EMG Signal in Preview Window
        emg_signal = np.hstack([data for _, data in self.emg_buffer])[
            :, : self.emg_recording_time
        ]
        emg_signal = self.main_window.device.extract_emg_data(emg_signal)
        self.review_recording_emg_plot_widget.refresh_plot()
        self.review_recording_emg_plot_widget.configure_lines_plot(
            display_time=self.recording_time,
            fs=self.emg_sampling_frequency,
            lines=emg_signal.shape[0],
        )
        self.review_recording_emg_plot_widget.set_plot_data(emg_signal / 1000)

    def _accept_recording(self) -> None:
        print("Recording accepted")
        self.review_recording_stacked_widget.setCurrentIndex(0)
        self.record_toggle_push_button.setText("Start Recording")
        self.record_toggle_push_button.setChecked(False)
        self.record_group_box.setEnabled(True)

        # Save Recordings
        label = self.review_recording_label_line_edit.text()
        if not label:
            label = "default"

        emg_signal = self.main_window.device.extract_emg_data(
            np.hstack([data for _, data in self.emg_buffer])
        )[:, : self.emg_recording_time]

        save_pickle_dict = {
            "emg": emg_signal,
            "kinematics": np.vstack([data for _, data in self.kinematics_buffer]).T,
            "timings_emg": np.array([time_stamp for time_stamp, _ in self.emg_buffer]),
            "timings_kinematics": np.array(
                [time_stamp for time_stamp, _ in self.kinematics_buffer]
            ),
            "label": label,
            "task": self.current_task,
        }
        now = datetime.now()
        formatted_now = now.strftime("%Y%m%d_%H%M%S%f")
        file_name = f"MindMove_Recording_{formatted_now}_{self.current_task.lower()}_{label.lower()}.pkl"

        if not os.path.exists(self.recording_dir_path):
            os.makedirs(self.recording_dir_path)

        with open(os.path.join(self.recording_dir_path, file_name), "wb") as f:
            pickle.dump(save_pickle_dict, f)

        # Reset progress bars
        self.record_emg_progress_bar.setValue(0)
        self.record_kinematics_progress_bar.setValue(0)

        # Reset buffers
        self.emg_buffer = []
        self.kinematics_buffer = []

    def _reject_recording(self) -> None:
        self.review_recording_stacked_widget.setCurrentIndex(0)
        self.record_toggle_push_button.setText("Start Recording")
        self.record_toggle_push_button.setChecked(False)
        self.record_group_box.setEnabled(True)

        # Reset progress bars
        self.record_emg_progress_bar.setValue(0)
        self.record_kinematics_progress_bar.setValue(0)

    def _setup_protocol_ui(self) -> None:
        # Record UI
        self.record_group_box = self.main_window.ui.recordRecordingGroupBox
        self.record_task_combo_box = self.main_window.ui.recordTaskComboBox
        self.record_duration_spin_box = self.main_window.ui.recordDurationSpinBox
        self.record_toggle_push_button = self.main_window.ui.recordRecordPushButton
        self.record_toggle_push_button.toggled.connect(self._start_recording)
        self.record_emg_progress_bar = self.main_window.ui.recordEMGProgressBar
        self.record_emg_progress_bar.setValue(0)
        self.record_kinematics_progress_bar = (
            self.main_window.ui.recordKinematicsProgressBar
        )
        self.record_kinematics_progress_bar.setValue(0)

        # Review Recording UI
        self.review_recording_stacked_widget = (
            self.main_window.ui.recordReviewRecordingStackedWidget
        )
        self.review_recording_stacked_widget.setCurrentIndex(0)

        self.review_recording_task_label = self.main_window.ui.reviewRecordingTaskLabel
        self.review_recording_label_line_edit = (
            self.main_window.ui.reviewRecordingLabelLineEdit
        )
        self.review_recording_emg_plot_widget = (
            self.main_window.ui.reviewRecordingEMGPlotWidget
        )
        self.review_recording_kinematics_plot_widget = (
            self.main_window.ui.reviewRecordingKinematicsPlotWidget
        )
        self.review_recording_accept_push_button = (
            self.main_window.ui.reviewRecordingAcceptPushButton
        )
        self.review_recording_accept_push_button.clicked.connect(self._accept_recording)

        self.review_recording_reject_push_button = (
            self.main_window.ui.reviewRecordingRejectPushButton
        )
        self.review_recording_reject_push_button.clicked.connect(self._reject_recording)
