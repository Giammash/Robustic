"""
Guided Template Recording Protocol

Patient-friendly template recording with VHI animation guide.
- VHI shows timed open→close→open animation
- Patient follows the virtual hand
- GT is automatically derived from VHI animation state
- Therapist controls timing between cycles
- Audio cues signal transitions
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, List
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QDoubleSpinBox, QCheckBox, QProgressBar,
    QFrame
)
import time
import numpy as np
import pickle
import os
from datetime import datetime

from mindmove.config import config

if TYPE_CHECKING:
    from mindmove.gui.mindmove import MindMove


class HandAnimationSequencer:
    """
    Controls VHI animation for guided template recording.

    Manages phases: HOLD_OPEN → CLOSING → HOLD_CLOSED → OPENING
    with smooth interpolation during transitions.
    """

    # Phase definitions
    PHASE_HOLD_OPEN = "HOLD_OPEN"
    PHASE_CLOSING = "CLOSING"
    PHASE_HOLD_CLOSED = "HOLD_CLOSED"
    PHASE_OPENING = "OPENING"

    def __init__(self):
        self.phases: List[Tuple[str, float, Optional[List[float]]]] = []
        self.current_phase_idx: int = 0
        self.phase_start_time: float = 0
        self.cycle_start_time: float = 0
        self.is_running: bool = False
        self._previous_phase: Optional[str] = None

        # Default timings
        self.hold_open_s: float = 2.0
        self.transition_s: float = 1.0
        self.hold_closed_s: float = 2.0

    def setup_cycle(self, hold_open_s: float, transition_s: float, hold_closed_s: float):
        """Define one open→close→open cycle with specified timings."""
        self.hold_open_s = hold_open_s
        self.transition_s = transition_s
        self.hold_closed_s = hold_closed_s

        # Open joints = 0, Closed joints = 1
        self.phases = [
            (self.PHASE_HOLD_OPEN, hold_open_s, [0.0] * 10),
            (self.PHASE_CLOSING, transition_s, None),  # Interpolate 0→1
            (self.PHASE_HOLD_CLOSED, hold_closed_s, [1.0] * 10),
            (self.PHASE_OPENING, transition_s, None),  # Interpolate 1→0
        ]

    def get_total_cycle_duration(self) -> float:
        """Get total duration of one cycle in seconds."""
        return self.hold_open_s + self.transition_s + self.hold_closed_s + self.transition_s

    def start_cycle(self):
        """Begin a new animation cycle."""
        self.current_phase_idx = 0
        self.cycle_start_time = time.time()
        self.phase_start_time = time.time()
        self.is_running = True
        self._previous_phase = None

    def stop(self):
        """Stop the animation."""
        self.is_running = False

    def get_current_state(self, current_time: float) -> Tuple[List[float], int, str, bool]:
        """
        Get current animation state.

        Args:
            current_time: Current timestamp

        Returns:
            Tuple of (joints[10], gt_value, phase_name, phase_changed)
            - joints: 10 joint values (0.0 = open, 1.0 = closed)
            - gt_value: Ground truth (0 = open state, 1 = closed state)
            - phase_name: Current phase name
            - phase_changed: True if phase just changed
        """
        if not self.is_running or not self.phases:
            return [0.0] * 10, 0, "IDLE", False

        elapsed_in_cycle = current_time - self.cycle_start_time

        # Find current phase based on elapsed time
        cumulative_time = 0.0
        for idx, (phase_name, duration, target_joints) in enumerate(self.phases):
            if elapsed_in_cycle < cumulative_time + duration:
                self.current_phase_idx = idx
                time_in_phase = elapsed_in_cycle - cumulative_time
                progress = time_in_phase / duration if duration > 0 else 1.0

                # Determine if phase changed
                phase_changed = (self._previous_phase != phase_name)
                self._previous_phase = phase_name

                # Calculate joint positions
                joints = self._interpolate_joints(phase_name, progress)

                # Determine GT value
                # GT=1 during CLOSING and HOLD_CLOSED (hand is closing/closed)
                # GT=0 during HOLD_OPEN and OPENING (hand is opening/open)
                if phase_name in [self.PHASE_CLOSING, self.PHASE_HOLD_CLOSED]:
                    gt_value = 1
                else:
                    gt_value = 0

                return joints, gt_value, phase_name, phase_changed

            cumulative_time += duration

        # Cycle complete
        self.is_running = False
        return [0.0] * 10, 0, "CYCLE_COMPLETE", True

    def _interpolate_joints(self, phase_name: str, progress: float) -> List[float]:
        """Interpolate joint positions for smooth transitions."""
        # Use smooth easing function (ease in-out)
        smooth_progress = self._ease_in_out(progress)

        if phase_name == self.PHASE_HOLD_OPEN:
            return [0.0] * 10
        elif phase_name == self.PHASE_CLOSING:
            # Interpolate from 0 to 1
            value = smooth_progress
            return [value] * 10
        elif phase_name == self.PHASE_HOLD_CLOSED:
            return [1.0] * 10
        elif phase_name == self.PHASE_OPENING:
            # Interpolate from 1 to 0
            value = 1.0 - smooth_progress
            return [value] * 10
        else:
            return [0.0] * 10

    @staticmethod
    def _ease_in_out(t: float) -> float:
        """Smooth ease-in-out function for natural animation."""
        if t < 0.5:
            return 2 * t * t
        else:
            return 1 - pow(-2 * t + 2, 2) / 2

    def is_cycle_complete(self) -> bool:
        """Check if current cycle has finished."""
        if not self.is_running:
            return True
        elapsed = time.time() - self.cycle_start_time
        return elapsed >= self.get_total_cycle_duration()

    def get_time_remaining_in_phase(self, current_time: float) -> float:
        """Get time remaining in current phase."""
        if not self.is_running or not self.phases:
            return 0.0

        elapsed_in_cycle = current_time - self.cycle_start_time
        cumulative_time = 0.0

        for phase_name, duration, _ in self.phases:
            if elapsed_in_cycle < cumulative_time + duration:
                time_in_phase = elapsed_in_cycle - cumulative_time
                return duration - time_in_phase
            cumulative_time += duration

        return 0.0


class AudioCueManager:
    """
    Manages audio feedback for guided recording.

    Uses simple beep sounds via winsound (Windows) or print fallback.
    """

    def __init__(self):
        self.enabled = True
        self._last_cue_time = 0
        self._min_cue_interval = 0.5  # Minimum seconds between cues

        # Try to import audio library
        try:
            import winsound
            self._winsound = winsound
            self._has_audio = True
        except ImportError:
            self._has_audio = False
            print("[AUDIO] winsound not available, using print fallback")

    def play(self, cue_name: str):
        """Play an audio cue."""
        if not self.enabled:
            return

        current_time = time.time()
        if current_time - self._last_cue_time < self._min_cue_interval:
            return
        self._last_cue_time = current_time

        if self._has_audio:
            try:
                if cue_name == "CLOSING":
                    # Low-pitched beep for closing
                    self._winsound.Beep(400, 200)
                elif cue_name == "OPENING":
                    # High-pitched beep for opening
                    self._winsound.Beep(800, 200)
                elif cue_name == "HOLD_OPEN":
                    # Short beep for hold start
                    self._winsound.Beep(600, 100)
                elif cue_name == "HOLD_CLOSED":
                    # Short beep for hold start
                    self._winsound.Beep(500, 100)
                elif cue_name == "CYCLE_COMPLETE":
                    # Success chime
                    self._winsound.Beep(600, 100)
                    self._winsound.Beep(800, 100)
                    self._winsound.Beep(1000, 200)
            except Exception as e:
                print(f"[AUDIO] Error playing sound: {e}")
        else:
            print(f"[AUDIO CUE] {cue_name}")


class GuidedRecordProtocol(QObject):
    """
    Protocol for patient-friendly guided template recording.

    Features:
    - VHI animation guide (open→close→open)
    - Configurable timing for each phase
    - Pause between cycles for therapist instructions
    - Audio cues for transitions
    - Automatic GT from animation state
    """

    # Signals
    cycle_completed = Signal(int)  # Emits cycle count
    recording_finished = Signal()

    def __init__(self, parent: MindMove | None = None) -> None:
        super().__init__(parent)

        self.main_window: MindMove = parent

        # Animation control
        self.sequencer = HandAnimationSequencer()
        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(33)  # ~30 Hz updates
        self.animation_timer.timeout.connect(self._update_animation)

        # Audio feedback
        self.audio_manager = AudioCueManager()

        # Recording state
        self.is_recording = False
        self.is_cycle_running = False
        self.waiting_for_next_cycle = True  # Start waiting

        # Recording buffers
        self.emg_buffer: List[Tuple[float, np.ndarray]] = []
        self.gt_buffer: List[Tuple[float, int]] = []
        self.cycle_boundaries: List[dict] = []

        # Timing
        self.recording_start_time: float = 0
        self.cycle_start_time: float = 0

        # Cycle tracking
        self.cycles_completed: int = 0

        # File management
        self.recording_dir_path: str = "data/recordings/"

        # Initialize UI
        self._setup_protocol_ui()

    def _setup_protocol_ui(self) -> None:
        """Create the UI for guided recording protocol."""
        # Create main widget container
        self.main_widget = QWidget()
        main_layout = QVBoxLayout(self.main_widget)

        # === Timing Configuration Group ===
        timing_group = QGroupBox("Timing Configuration")
        timing_layout = QGridLayout(timing_group)

        # Hold Open duration
        timing_layout.addWidget(QLabel("Hold Open:"), 0, 0)
        self.hold_open_spinbox = QDoubleSpinBox()
        self.hold_open_spinbox.setRange(0.5, 10.0)
        self.hold_open_spinbox.setSingleStep(0.5)
        self.hold_open_spinbox.setValue(2.0)
        self.hold_open_spinbox.setSuffix(" s")
        timing_layout.addWidget(self.hold_open_spinbox, 0, 1)

        # Transition duration
        timing_layout.addWidget(QLabel("Transition:"), 1, 0)
        self.transition_spinbox = QDoubleSpinBox()
        self.transition_spinbox.setRange(0.5, 5.0)
        self.transition_spinbox.setSingleStep(0.5)
        self.transition_spinbox.setValue(1.0)
        self.transition_spinbox.setSuffix(" s")
        timing_layout.addWidget(self.transition_spinbox, 1, 1)

        # Hold Closed duration
        timing_layout.addWidget(QLabel("Hold Closed:"), 2, 0)
        self.hold_closed_spinbox = QDoubleSpinBox()
        self.hold_closed_spinbox.setRange(0.5, 10.0)
        self.hold_closed_spinbox.setSingleStep(0.5)
        self.hold_closed_spinbox.setValue(2.0)
        self.hold_closed_spinbox.setSuffix(" s")
        timing_layout.addWidget(self.hold_closed_spinbox, 2, 1)

        # Audio cues checkbox
        self.audio_checkbox = QCheckBox("Audio cues enabled")
        self.audio_checkbox.setChecked(True)
        self.audio_checkbox.toggled.connect(self._on_audio_toggled)
        timing_layout.addWidget(self.audio_checkbox, 3, 0, 1, 2)

        main_layout.addWidget(timing_group)

        # === Recording Control Group ===
        control_group = QGroupBox("Recording Control")
        control_layout = QVBoxLayout(control_group)

        # Buttons row
        buttons_layout = QHBoxLayout()

        self.start_recording_button = QPushButton("Start Recording")
        self.start_recording_button.setCheckable(True)
        self.start_recording_button.toggled.connect(self._on_start_recording_toggled)
        buttons_layout.addWidget(self.start_recording_button)

        self.stop_save_button = QPushButton("Stop && Save")
        self.stop_save_button.setEnabled(False)
        self.stop_save_button.clicked.connect(self._on_stop_and_save)
        buttons_layout.addWidget(self.stop_save_button)

        control_layout.addLayout(buttons_layout)

        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("font-weight: bold; padding: 5px;")
        control_layout.addWidget(self.status_label)

        # Cycle counter
        self.cycle_label = QLabel("Cycles completed: 0")
        control_layout.addWidget(self.cycle_label)

        # Start Next Cycle button
        self.start_cycle_button = QPushButton("Start Next Cycle")
        self.start_cycle_button.setEnabled(False)
        self.start_cycle_button.clicked.connect(self._on_start_next_cycle)
        self.start_cycle_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        control_layout.addWidget(self.start_cycle_button)

        main_layout.addWidget(control_group)

        # === Phase Progress Group ===
        progress_group = QGroupBox("Current Phase")
        progress_layout = QVBoxLayout(progress_group)

        self.phase_label = QLabel("Phase: -")
        self.phase_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        progress_layout.addWidget(self.phase_label)

        self.phase_progress = QProgressBar()
        self.phase_progress.setRange(0, 100)
        self.phase_progress.setValue(0)
        progress_layout.addWidget(self.phase_progress)

        self.time_remaining_label = QLabel("Time remaining: -")
        progress_layout.addWidget(self.time_remaining_label)

        main_layout.addWidget(progress_group)

        # === Label for saving ===
        label_group = QGroupBox("Recording Label")
        label_layout = QHBoxLayout(label_group)

        label_layout.addWidget(QLabel("Label:"))
        from PySide6.QtWidgets import QLineEdit
        self.label_line_edit = QLineEdit()
        self.label_line_edit.setPlaceholderText("e.g., position_1, table_neutral")
        label_layout.addWidget(self.label_line_edit)

        main_layout.addWidget(label_group)

        # Spacer
        main_layout.addStretch()

    def get_widget(self) -> QWidget:
        """Return the main widget for this protocol."""
        return self.main_widget

    def _on_audio_toggled(self, checked: bool):
        """Handle audio checkbox toggle."""
        self.audio_manager.enabled = checked

    def _on_start_recording_toggled(self, checked: bool):
        """Handle start/stop recording button."""
        if checked:
            self._start_recording()
        else:
            # Just uncheck, don't stop (use Stop & Save for that)
            if self.is_recording:
                self.start_recording_button.setChecked(True)

    def _start_recording(self):
        """Start the recording session."""
        # Check device connection
        if not self.main_window.device.device.is_streaming:
            print("[GUIDED] Device is not streaming!")
            self.start_recording_button.setChecked(False)
            return

        # Check VHI connection
        if not self.main_window.virtual_hand_interface.is_streaming:
            print("[GUIDED] Virtual Hand Interface is not streaming!")
            print("[GUIDED] Please start VHI streaming first.")
            self.start_recording_button.setChecked(False)
            return

        # Setup sequencer with current timing settings
        self.sequencer.setup_cycle(
            hold_open_s=self.hold_open_spinbox.value(),
            transition_s=self.transition_spinbox.value(),
            hold_closed_s=self.hold_closed_spinbox.value()
        )

        # Reset state
        self.is_recording = True
        self.is_cycle_running = False
        self.waiting_for_next_cycle = True
        self.cycles_completed = 0

        # Reset buffers
        self.emg_buffer = []
        self.gt_buffer = []
        self.cycle_boundaries = []

        # Record start time
        self.recording_start_time = time.time()

        # Connect EMG signal
        self.main_window.device.ready_read_signal.connect(self._on_emg_data)

        # Start animation timer (for sending VHI state even when paused)
        self.animation_timer.start()

        # Update UI
        self.start_recording_button.setText("Recording...")
        self.stop_save_button.setEnabled(True)
        self.start_cycle_button.setEnabled(True)
        self._set_timing_controls_enabled(False)

        self.status_label.setText("Status: Recording - Press 'Start Next Cycle' to begin")
        self.cycle_label.setText("Cycles completed: 0")
        self.phase_label.setText("Phase: Waiting for cycle start")

        # Send open hand to VHI
        self._send_joints_to_vhi([0.0] * 10)

        print("\n" + "=" * 60)
        print("GUIDED RECORDING STARTED")
        print("  Predicted Hand becomes Control Hand (visual guide)")
        print(f"  Hold Open: {self.hold_open_spinbox.value()}s")
        print(f"  Transition: {self.transition_spinbox.value()}s")
        print(f"  Hold Closed: {self.hold_closed_spinbox.value()}s")
        print(f"  Cycle duration: {self.sequencer.get_total_cycle_duration()}s")
        print("Press 'Start Next Cycle' to begin first cycle")
        print("=" * 60 + "\n")

    def _on_stop_and_save(self):
        """Stop recording and save data."""
        if not self.is_recording:
            return

        # Stop animation
        self.sequencer.stop()
        self.animation_timer.stop()

        # Disconnect EMG signal
        try:
            self.main_window.device.ready_read_signal.disconnect(self._on_emg_data)
        except RuntimeError:
            pass  # Already disconnected

        self.is_recording = False
        self.is_cycle_running = False

        # Save data
        self._save_recording()

        # Update UI
        self.start_recording_button.setChecked(False)
        self.start_recording_button.setText("Start Recording")
        self.stop_save_button.setEnabled(False)
        self.start_cycle_button.setEnabled(False)
        self._set_timing_controls_enabled(True)

        self.status_label.setText("Status: Recording saved")
        self.phase_label.setText("Phase: -")
        self.phase_progress.setValue(0)

        # Send open hand to VHI
        self._send_joints_to_vhi([0.0] * 10)

    def _on_start_next_cycle(self):
        """Start the next animation cycle."""
        if not self.is_recording:
            return

        # Update sequencer timings in case they changed
        self.sequencer.setup_cycle(
            hold_open_s=self.hold_open_spinbox.value(),
            transition_s=self.transition_spinbox.value(),
            hold_closed_s=self.hold_closed_spinbox.value()
        )

        # Start new cycle
        self.cycle_start_time = time.time()
        self.sequencer.start_cycle()
        self.is_cycle_running = True
        self.waiting_for_next_cycle = False

        # Record cycle boundary
        self.cycle_boundaries.append({
            "start_time": self.cycle_start_time,
            "end_time": None,
            "cycle_number": self.cycles_completed + 1
        })

        # Update UI
        self.start_cycle_button.setEnabled(False)
        self.status_label.setText(f"Status: Cycle {self.cycles_completed + 1} running")

        print(f"\n[GUIDED] Starting cycle {self.cycles_completed + 1}")

    def _update_animation(self):
        """Called at 30Hz - update VHI and record GT."""
        if not self.is_recording:
            return

        current_time = time.time()

        if self.is_cycle_running and self.sequencer.is_running:
            # Get current animation state
            joints, gt_value, phase_name, phase_changed = self.sequencer.get_current_state(current_time)

            # Send joints to VHI
            self._send_joints_to_vhi(joints)

            # Record GT
            self.gt_buffer.append((current_time, gt_value))

            # Handle phase change
            if phase_changed:
                if phase_name != "CYCLE_COMPLETE":
                    self.audio_manager.play(phase_name)
                    print(f"[GUIDED] Phase: {phase_name}")

            # Update UI
            self._update_phase_ui(phase_name, current_time)

            # Check cycle completion
            if self.sequencer.is_cycle_complete():
                self._on_cycle_complete()

        elif self.waiting_for_next_cycle:
            # Keep VHI at open position while waiting
            self._send_joints_to_vhi([0.0] * 10)
            # Still record GT=0 while waiting
            self.gt_buffer.append((current_time, 0))

    def _on_cycle_complete(self):
        """Handle cycle completion."""
        self.cycles_completed += 1
        self.is_cycle_running = False
        self.waiting_for_next_cycle = True

        # Update cycle boundary
        if self.cycle_boundaries:
            self.cycle_boundaries[-1]["end_time"] = time.time()

        # Play completion sound
        self.audio_manager.play("CYCLE_COMPLETE")

        # Update UI
        self.start_cycle_button.setEnabled(True)
        self.cycle_label.setText(f"Cycles completed: {self.cycles_completed}")
        self.status_label.setText(f"Status: Cycle {self.cycles_completed} complete - Ready for next")
        self.phase_label.setText("Phase: Cycle complete - Position patient for next cycle")
        self.phase_progress.setValue(0)

        print(f"[GUIDED] Cycle {self.cycles_completed} complete")
        print("[GUIDED] Position patient and press 'Start Next Cycle'")

        self.cycle_completed.emit(self.cycles_completed)

    def _update_phase_ui(self, phase_name: str, current_time: float):
        """Update UI to show current phase."""
        self.phase_label.setText(f"Phase: {phase_name}")

        # Calculate progress within current phase
        time_remaining = self.sequencer.get_time_remaining_in_phase(current_time)
        phase_durations = {
            "HOLD_OPEN": self.hold_open_spinbox.value(),
            "CLOSING": self.transition_spinbox.value(),
            "HOLD_CLOSED": self.hold_closed_spinbox.value(),
            "OPENING": self.transition_spinbox.value(),
        }
        phase_duration = phase_durations.get(phase_name, 1.0)
        if phase_duration > 0:
            progress = int(100 * (1 - time_remaining / phase_duration))
            self.phase_progress.setValue(max(0, min(100, progress)))

        self.time_remaining_label.setText(f"Time remaining: {time_remaining:.1f}s")

    def _send_joints_to_vhi(self, joints: List[float]):
        """Send joint positions to Virtual Hand Interface."""
        if self.main_window.virtual_hand_interface.is_streaming:
            # Convert to int for cleaner transmission (0 or 1 for hold states)
            joint_values = [round(j, 2) for j in joints]
            self.main_window.virtual_hand_interface.output_message_signal.emit(
                str(joint_values).encode("utf-8")
            )

    def _on_emg_data(self, data: np.ndarray):
        """Handle incoming EMG data."""
        if self.is_recording:
            self.emg_buffer.append((time.time(), data))

    def _set_timing_controls_enabled(self, enabled: bool):
        """Enable/disable timing controls."""
        self.hold_open_spinbox.setEnabled(enabled)
        self.transition_spinbox.setEnabled(enabled)
        self.hold_closed_spinbox.setEnabled(enabled)

    def _save_recording(self):
        """Save the recording to file."""
        if not self.emg_buffer:
            print("[GUIDED] No EMG data to save!")
            return

        # Build EMG array
        emg_signal = self.main_window.device.extract_emg_data(
            np.hstack([data for _, data in self.emg_buffer])
        )
        emg_timestamps = np.array([t for t, _ in self.emg_buffer])

        # Build GT signal at EMG sample rate
        gt_signal = self._build_gt_signal_at_emg_rate(emg_signal.shape[1])

        # Get label
        label = self.label_line_edit.text().strip()
        if not label:
            label = f"guided_{self.cycles_completed}cycles"

        # Build save dictionary
        save_dict = {
            "emg": emg_signal,
            "gt": gt_signal,
            "gt_raw": self.gt_buffer,  # Raw (timestamp, value) pairs
            "timings_emg": emg_timestamps,
            "gt_mode": "guided_animation",
            "animation_config": {
                "hold_open_s": self.hold_open_spinbox.value(),
                "transition_s": self.transition_spinbox.value(),
                "hold_closed_s": self.hold_closed_spinbox.value(),
            },
            "cycles": self.cycle_boundaries,
            "cycles_completed": self.cycles_completed,
            "label": label,
            "task": "guided_open_close",
            "recording_duration_s": time.time() - self.recording_start_time,
        }

        # Save file
        if not os.path.exists(self.recording_dir_path):
            os.makedirs(self.recording_dir_path)

        now = datetime.now()
        formatted_now = now.strftime("%Y%m%d_%H%M%S%f")
        file_name = f"MindMove_GuidedRecording_{formatted_now}_{label}.pkl"

        with open(os.path.join(self.recording_dir_path, file_name), "wb") as f:
            pickle.dump(save_dict, f)

        print(f"\n[GUIDED] Recording saved: {file_name}")
        print(f"  EMG shape: {emg_signal.shape}")
        print(f"  GT shape: {gt_signal.shape}")
        print(f"  Cycles completed: {self.cycles_completed}")
        print(f"  Total duration: {save_dict['recording_duration_s']:.1f}s")

    def _build_gt_signal_at_emg_rate(self, n_emg_samples: int) -> np.ndarray:
        """
        Build GT signal at EMG sample rate from recorded GT buffer.

        The GT buffer contains (timestamp, gt_value) pairs at ~30Hz.
        This interpolates to EMG rate (2000Hz).
        """
        if not self.gt_buffer or not self.emg_buffer:
            return np.zeros(n_emg_samples)

        gt_signal = np.zeros(n_emg_samples)

        # Get EMG time range
        emg_start_time = self.emg_buffer[0][0]
        emg_end_time = self.emg_buffer[-1][0]
        emg_duration = emg_end_time - emg_start_time

        if emg_duration <= 0:
            return gt_signal

        # Convert GT timestamps to sample indices and fill
        current_gt = 0
        last_sample_idx = 0

        for gt_time, gt_value in self.gt_buffer:
            # Calculate sample index
            relative_time = gt_time - emg_start_time
            sample_idx = int((relative_time / emg_duration) * n_emg_samples)
            sample_idx = max(0, min(sample_idx, n_emg_samples - 1))

            # Fill from last sample to current with previous GT value
            if sample_idx > last_sample_idx:
                gt_signal[last_sample_idx:sample_idx] = current_gt

            current_gt = gt_value
            last_sample_idx = sample_idx

        # Fill remaining with final GT value
        gt_signal[last_sample_idx:] = current_gt

        return gt_signal
