from __future__ import annotations
from typing import TYPE_CHECKING
from PySide6.QtCore import QObject, Qt, QEvent
from PySide6.QtWidgets import QComboBox, QLabel
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

        # Ground truth mode: "virtual_hand" or "keyboard"
        self.gt_mode: str = "keyboard"  # Default to keyboard for simplicity

        # Keyboard trigger state (press-and-hold behavior)
        self.keyboard_gt_state: int = 0  # 0 = NO ACTIVATION, 1 = ACTIVATION
        self.keyboard_gt_buffer: list = []  # (timestamp, state) pairs
        self._spacebar_enabled: bool = False  # Only enabled during recording
        self._spacebar_pressed: bool = False  # Track if spacebar is currently held

        # Setup keyboard event filter
        self._setup_keyboard_events()

    def _setup_keyboard_events(self) -> None:
        """Setup event filter for spacebar press/release detection."""
        self.main_window.installEventFilter(self)

    def eventFilter(self, obj, event: QEvent) -> bool:
        """Filter key events for spacebar press and release."""
        if not self._spacebar_enabled:
            return False

        if event.type() == QEvent.KeyPress and not event.isAutoRepeat():
            if event.key() == Qt.Key_Space:
                self._on_spacebar_pressed()
                return True

        elif event.type() == QEvent.KeyRelease and not event.isAutoRepeat():
            if event.key() == Qt.Key_Space:
                self._on_spacebar_released()
                return True

        return False

    def _on_spacebar_pressed(self) -> None:
        """Handle spacebar press - ACTIVATION BEGINS."""
        if self._spacebar_pressed:
            return  # Already pressed, ignore

        self._spacebar_pressed = True
        self.keyboard_gt_state = 1
        current_time = time.time()

        # Store the transition
        self.keyboard_gt_buffer.append((current_time, self.keyboard_gt_state))

        # Visual feedback
        elapsed = current_time - self.start_time
        print(f"\n>>> ACTIVATION BEGINS at t={elapsed:.2f}s <<<\n")

    def _on_spacebar_released(self) -> None:
        """Handle spacebar release - ACTIVATION STOPS."""
        if not self._spacebar_pressed:
            return  # Not pressed, ignore

        self._spacebar_pressed = False
        self.keyboard_gt_state = 0
        current_time = time.time()

        # Store the transition
        self.keyboard_gt_buffer.append((current_time, self.keyboard_gt_state))

        # Visual feedback
        elapsed = current_time - self.start_time
        print(f"\n<<< ACTIVATION STOPS at t={elapsed:.2f}s <<<\n")

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

            # Get GT mode from combo box
            self.gt_mode = self.gt_mode_combo_box.currentData()

            # Check virtual hand connection only if using virtual hand mode
            if self.gt_mode == "virtual_hand":
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
            self.keyboard_gt_buffer = []
            self.keyboard_gt_state = 0  # Start with OPEN

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

            # Enable spacebar for keyboard mode (press-and-hold)
            if self.gt_mode == "keyboard":
                self._spacebar_enabled = True
                self._spacebar_pressed = False
                self.has_finished_kinematics = True  # No kinematics to wait for
                print("\n" + "=" * 60)
                print("KEYBOARD MODE: Hold SPACEBAR for activation")
                print("  Press spacebar   → ACTIVATION BEGINS")
                print("  Release spacebar → ACTIVATION STOPS")
                print("Starting state: NO ACTIVATION (GT=0)")
                print("=" * 60 + "\n")

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

        # Disable spacebar
        self._spacebar_enabled = False
        self._spacebar_pressed = False

        self.review_recording_stacked_widget.setCurrentIndex(1)
        self.record_toggle_push_button.setText("Finished Recording")
        self.review_recording_task_label.setText(self.current_task.capitalize())

        # Plot GT/Kinematics Signal in Preview Window
        if self.gt_mode == "virtual_hand":
            kinematics_signal = np.vstack([data for _, data in self.kinematics_buffer]).T
            self.review_recording_kinematics_plot_widget.refresh_plot()
            self.review_recording_kinematics_plot_widget.configure_lines_plot(
                display_time=self.recording_time,
                fs=self.kinematics_sampling_frequency,
                lines=kinematics_signal.shape[0],
            )
            self.review_recording_kinematics_plot_widget.set_plot_data(kinematics_signal)
        else:
            # Keyboard mode: build GT signal from toggle events
            gt_signal = self._build_gt_signal_from_keyboard()
            self.review_recording_kinematics_plot_widget.refresh_plot()
            self.review_recording_kinematics_plot_widget.configure_lines_plot(
                display_time=self.recording_time,
                fs=self.emg_sampling_frequency,  # Same fs as EMG for GT
                lines=1,
            )
            self.review_recording_kinematics_plot_widget.set_plot_data(gt_signal.reshape(1, -1))

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

    def _build_gt_signal_from_keyboard(self) -> np.ndarray:
        """Build a GT signal from keyboard toggle events."""
        n_samples = self.emg_recording_time
        gt_signal = np.zeros(n_samples)

        if not self.keyboard_gt_buffer:
            return gt_signal

        # Convert toggle events to continuous signal
        current_state = 0  # Start with OPEN
        last_sample = 0

        for toggle_time, new_state in self.keyboard_gt_buffer:
            # Calculate sample index
            sample_idx = int((toggle_time - self.start_time) * self.emg_sampling_frequency)
            sample_idx = min(sample_idx, n_samples - 1)

            # Fill from last sample to this sample with previous state
            gt_signal[last_sample:sample_idx] = current_state
            current_state = new_state
            last_sample = sample_idx

        # Fill remaining samples with final state
        gt_signal[last_sample:] = current_state

        return gt_signal

    def _accept_recording(self) -> None:
        print("Recording accepted")
        self.review_recording_stacked_widget.setCurrentIndex(0)
        self.record_toggle_push_button.setText("Start Recording")
        self.record_toggle_push_button.setChecked(False)
        self.record_group_box.setEnabled(True)

        # Ensure spacebar is disabled
        self._spacebar_enabled = False
        self._spacebar_pressed = False

        # Save Recordings
        label = self.review_recording_label_line_edit.text()
        if not label:
            label = "default"

        emg_signal = self.main_window.device.extract_emg_data(
            np.hstack([data for _, data in self.emg_buffer])
        )[:, : self.emg_recording_time]

        # Build save dict based on GT mode
        if self.gt_mode == "virtual_hand":
            save_pickle_dict = {
                "emg": emg_signal,
                "kinematics": np.vstack([data for _, data in self.kinematics_buffer]).T,
                "timings_emg": np.array([time_stamp for time_stamp, _ in self.emg_buffer]),
                "timings_kinematics": np.array(
                    [time_stamp for time_stamp, _ in self.kinematics_buffer]
                ),
                "label": label,
                "task": self.current_task,
                "gt_mode": "virtual_hand",
            }
        else:
            # Keyboard mode: save GT signal directly
            gt_signal = self._build_gt_signal_from_keyboard()
            save_pickle_dict = {
                "emg": emg_signal,
                "gt": gt_signal,  # Binary GT signal at EMG sampling rate
                "keyboard_toggles": self.keyboard_gt_buffer,  # Raw toggle events
                "timings_emg": np.array([time_stamp for time_stamp, _ in self.emg_buffer]),
                "label": label,
                "task": self.current_task,
                "gt_mode": "keyboard",
            }

        now = datetime.now()
        formatted_now = now.strftime("%Y%m%d_%H%M%S%f")
        file_name = f"MindMove_Recording_{formatted_now}_{self.current_task.lower()}_{label.lower()}.pkl"

        if not os.path.exists(self.recording_dir_path):
            os.makedirs(self.recording_dir_path)

        with open(os.path.join(self.recording_dir_path, file_name), "wb") as f:
            pickle.dump(save_pickle_dict, f)

        print(f"Recording saved: {file_name}")
        print(f"  GT mode: {self.gt_mode}")
        print(f"  EMG shape: {emg_signal.shape}")
        if self.gt_mode == "keyboard":
            print(f"  GT toggles: {len(self.keyboard_gt_buffer)}")

        # Reset progress bars
        self.record_emg_progress_bar.setValue(0)
        self.record_kinematics_progress_bar.setValue(0)

        # Reset buffers
        self.emg_buffer = []
        self.kinematics_buffer = []
        self.keyboard_gt_buffer = []

    def _reject_recording(self) -> None:
        self.review_recording_stacked_widget.setCurrentIndex(0)
        self.record_toggle_push_button.setText("Start Recording")
        self.record_toggle_push_button.setChecked(False)
        self.record_group_box.setEnabled(True)

        # Disable spacebar
        self._spacebar_enabled = False
        self._spacebar_pressed = False

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

        # Add GT mode selector (programmatically)
        self._setup_gt_mode_selector()

        # Setup review recording UI
        self._setup_review_ui()

    def _setup_gt_mode_selector(self) -> None:
        """Setup GT mode selector combo box."""
        from PySide6.QtWidgets import QGridLayout

        # Create GT mode selector
        gt_mode_label = QLabel("Ground Truth Mode:")
        self.gt_mode_combo_box = QComboBox()
        self.gt_mode_combo_box.addItem("Keyboard (Hold spacebar)", "keyboard")
        self.gt_mode_combo_box.addItem("Virtual Hand Interface", "virtual_hand")

        # Add to record group box layout
        layout = self.record_group_box.layout()
        if layout and isinstance(layout, QGridLayout):
            # Add GT mode selector to the next available row
            next_row = layout.rowCount()
            layout.addWidget(gt_mode_label, next_row, 0)
            layout.addWidget(self.gt_mode_combo_box, next_row, 1)

    def _setup_review_ui(self) -> None:
        """Setup review recording UI."""
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
