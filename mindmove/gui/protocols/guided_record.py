"""
Guided Template Recording Protocol

Patient-friendly template recording with VHI animation guide.
- VHI shows timed open→close→open animation
- Patient follows the virtual hand
- GT is automatically derived from VHI animation state
- Therapist controls timing between cycles
- Audio cues signal transitions
- Post-recording review UI for manual template selection
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, List, Dict
from PySide6.QtCore import QObject, QTimer, Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QDoubleSpinBox, QCheckBox, QProgressBar,
    QFrame, QStackedWidget, QScrollArea, QSplitter, QMessageBox,
    QComboBox
)
import time
import numpy as np
import pickle
import os
from datetime import datetime

# Matplotlib imports for template review plots
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from mindmove.config import config
from mindmove.model.templates.template_manager import TemplateManager

if TYPE_CHECKING:
    from mindmove.gui.mindmove import MindMove


class HandAnimationSequencer:
    """
    Controls VHI animation for guided template recording.

    Manages phases with reaction time delays:
    HOLD_OPEN → [AUDIO "Close"] → REACTION_CLOSE → CLOSING → HOLD_CLOSED → [AUDIO "Open"] → REACTION_OPEN → OPENING → HOLD_OPEN_END

    Audio cues play at the end of HOLD_OPEN and HOLD_CLOSED.
    VHI movement and GT transition start after the reaction time delay.
    GT ramps linearly during CLOSING (0→1) and OPENING (1→0).
    """

    # Phase definitions
    PHASE_HOLD_OPEN = "HOLD_OPEN"
    PHASE_REACTION_CLOSE = "REACTION_CLOSE"  # After close audio cue, before movement
    PHASE_CLOSING = "CLOSING"
    PHASE_HOLD_CLOSED = "HOLD_CLOSED"
    PHASE_REACTION_OPEN = "REACTION_OPEN"  # After open audio cue, before movement
    PHASE_OPENING = "OPENING"
    PHASE_HOLD_OPEN_END = "HOLD_OPEN_END"  # Final hold open (same duration as first)
    PHASE_HOLD_CLOSED_END = "HOLD_CLOSED_END"  # Final hold closed for inverted mode

    def __init__(self):
        self.phases: List[Tuple[str, float]] = []
        self.current_phase_idx: int = 0
        self.phase_start_time: float = 0
        self.cycle_start_time: float = 0
        self.is_running: bool = False
        self._previous_phase: Optional[str] = None

        # Default timings
        self.hold_open_s: float = 2.0
        self.reaction_time_s: float = 0.2  # 200ms reaction time
        self.closing_s: float = 0.35  # Time for hand to close
        self.hold_closed_s: float = 2.0
        self.opening_s: float = 0.35  # Time for hand to open

        # Protocol mode: "standard" (open→close→open) or "inverted" (close→open→close)
        self.protocol_mode: str = "standard"

        # Audio cue timestamps (recorded during cycle for later visualization)
        self.close_cue_time: Optional[float] = None
        self.open_cue_time: Optional[float] = None

    def setup_cycle(
        self,
        hold_open_s: float,
        closing_s: float,
        hold_closed_s: float,
        opening_s: float,
        reaction_time_s: float = 0.2,
        protocol_mode: str = "standard"
    ):
        """
        Define one cycle with specified timings.

        Args:
            hold_open_s: Time before close audio cue (hand stays open)
            closing_s: Duration for hand to close (VHI movement + GT ramp)
            hold_closed_s: Time before open audio cue (hand stays closed)
            opening_s: Duration for hand to open (VHI movement + GT ramp)
            reaction_time_s: Delay between audio cue and movement start
            protocol_mode: "standard" (open→close→open) or "inverted" (close→open→close)
        """
        self.hold_open_s = hold_open_s
        self.reaction_time_s = reaction_time_s
        self.closing_s = closing_s
        self.hold_closed_s = hold_closed_s
        self.opening_s = opening_s
        self.protocol_mode = protocol_mode

        if protocol_mode == "inverted":
            # Inverted: start closed → open → close
            # Phase sequence: HOLD_CLOSED → REACTION_OPEN → OPENING → HOLD_OPEN → REACTION_CLOSE → CLOSING → HOLD_CLOSED_END
            self.phases = [
                (self.PHASE_HOLD_CLOSED, hold_closed_s),
                (self.PHASE_REACTION_OPEN, reaction_time_s),
                (self.PHASE_OPENING, opening_s),
                (self.PHASE_HOLD_OPEN, hold_open_s),
                (self.PHASE_REACTION_CLOSE, reaction_time_s),
                (self.PHASE_CLOSING, closing_s),
                (self.PHASE_HOLD_CLOSED_END, hold_closed_s),
            ]
        else:
            # Standard: start open → close → open
            # Phase sequence with durations (includes final HOLD_OPEN_END)
            self.phases = [
                (self.PHASE_HOLD_OPEN, hold_open_s),
                (self.PHASE_REACTION_CLOSE, reaction_time_s),
                (self.PHASE_CLOSING, closing_s),
                (self.PHASE_HOLD_CLOSED, hold_closed_s),
                (self.PHASE_REACTION_OPEN, reaction_time_s),
                (self.PHASE_OPENING, opening_s),
                (self.PHASE_HOLD_OPEN_END, hold_open_s),  # Same duration as first hold open
            ]

        # Reset audio cue times
        self.close_cue_time = None
        self.open_cue_time = None

    def get_total_cycle_duration(self) -> float:
        """Get total duration of one cycle in seconds."""
        return sum(duration for _, duration in self.phases)

    def get_close_cue_time_in_cycle(self) -> float:
        """Get time within cycle when close audio cue should play."""
        if self.protocol_mode == "inverted":
            # In inverted mode: HOLD_CLOSED → REACTION_OPEN → OPENING → HOLD_OPEN → [CLOSE CUE]
            return self.hold_closed_s + self.reaction_time_s + self.opening_s + self.hold_open_s
        else:
            # Standard: HOLD_OPEN → [CLOSE CUE]
            return self.hold_open_s

    def get_open_cue_time_in_cycle(self) -> float:
        """Get time within cycle when open audio cue should play."""
        if self.protocol_mode == "inverted":
            # In inverted mode: HOLD_CLOSED → [OPEN CUE]
            return self.hold_closed_s
        else:
            # Standard: HOLD_OPEN → REACTION_CLOSE → CLOSING → HOLD_CLOSED → [OPEN CUE]
            return self.hold_open_s + self.reaction_time_s + self.closing_s + self.hold_closed_s

    def start_cycle(self):
        """Begin a new animation cycle."""
        self.current_phase_idx = 0
        self.cycle_start_time = time.time()
        self.phase_start_time = time.time()
        self.is_running = True
        self._previous_phase = None
        self.close_cue_time = None
        self.open_cue_time = None

    def stop(self):
        """Stop the animation."""
        self.is_running = False

    def record_close_cue_time(self, timestamp: float):
        """Record when close audio cue was played."""
        self.close_cue_time = timestamp

    def record_open_cue_time(self, timestamp: float):
        """Record when open audio cue was played."""
        self.open_cue_time = timestamp

    def get_current_state(self, current_time: float) -> Tuple[List[float], float, str, bool]:
        """
        Get current animation state.

        Args:
            current_time: Current timestamp

        Returns:
            Tuple of (joints[10], gt_value, phase_name, phase_changed)
            - joints: 10 joint values (0.0 = open, 1.0 = closed)
            - gt_value: Ground truth (0.0-1.0, with linear ramps during transitions)
            - phase_name: Current phase name
            - phase_changed: True if phase just changed
        """
        if not self.is_running or not self.phases:
            return [0.0] * 10, 0.0, "IDLE", False

        elapsed_in_cycle = current_time - self.cycle_start_time

        # Find current phase based on elapsed time
        cumulative_time = 0.0
        for idx, (phase_name, duration) in enumerate(self.phases):
            if elapsed_in_cycle < cumulative_time + duration:
                self.current_phase_idx = idx
                time_in_phase = elapsed_in_cycle - cumulative_time
                progress = time_in_phase / duration if duration > 0 else 1.0

                # Determine if phase changed
                phase_changed = (self._previous_phase != phase_name)
                self._previous_phase = phase_name

                # Calculate joint positions and GT value based on phase
                joints, gt_value = self._get_phase_state(phase_name, progress)

                return joints, gt_value, phase_name, phase_changed

            cumulative_time += duration

        # Cycle complete
        self.is_running = False
        return [0.0] * 10, 0.0, "CYCLE_COMPLETE", True

    def _get_phase_state(self, phase_name: str, progress: float) -> Tuple[List[float], float]:
        """
        Get joint positions and GT value for a given phase.

        Returns:
            Tuple of (joints[10], gt_value)
        """
        if phase_name == self.PHASE_HOLD_OPEN:
            # Hand open, GT = 0
            return [0.0] * 10, 0.0

        elif phase_name == self.PHASE_REACTION_CLOSE:
            # After close cue, waiting for reaction time
            # Hand still open, GT still 0
            return [0.0] * 10, 0.0

        elif phase_name == self.PHASE_CLOSING:
            # Hand closing, GT ramps 0 → 1 linearly
            value = progress  # Linear 0 to 1
            return [value] * 10, value

        elif phase_name == self.PHASE_HOLD_CLOSED:
            # Hand closed, GT = 1
            return [1.0] * 10, 1.0

        elif phase_name == self.PHASE_REACTION_OPEN:
            # After open cue, waiting for reaction time
            # Hand still closed, GT still 1
            return [1.0] * 10, 1.0

        elif phase_name == self.PHASE_OPENING:
            # Hand opening, GT ramps 1 → 0 linearly
            value = 1.0 - progress  # Linear 1 to 0
            return [value] * 10, value

        elif phase_name == self.PHASE_HOLD_OPEN_END:
            # Final hold open, GT = 0
            return [0.0] * 10, 0.0

        elif phase_name == self.PHASE_HOLD_CLOSED_END:
            # Final hold closed (inverted mode), GT = 1
            return [1.0] * 10, 1.0

        else:
            return [0.0] * 10, 0.0

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

        for phase_name, duration in self.phases:
            if elapsed_in_cycle < cumulative_time + duration:
                time_in_phase = elapsed_in_cycle - cumulative_time
                return duration - time_in_phase
            cumulative_time += duration

        return 0.0

    def should_play_close_cue(self, current_time: float) -> bool:
        """Check if close audio cue should play now."""
        if not self.is_running or self.close_cue_time is not None:
            return False
        elapsed = current_time - self.cycle_start_time
        return elapsed >= self.hold_open_s and elapsed < self.hold_open_s + 0.05

    def should_play_open_cue(self, current_time: float) -> bool:
        """Check if open audio cue should play now."""
        if not self.is_running or self.open_cue_time is not None:
            return False
        elapsed = current_time - self.cycle_start_time
        open_cue_time = self.get_open_cue_time_in_cycle()
        return elapsed >= open_cue_time and elapsed < open_cue_time + 0.05


class AudioCueManager:
    """
    Manages audio feedback for guided recording.

    Uses simple beep sounds via winsound (Windows) or print fallback.
    Audio is played in a background thread to avoid blocking the animation loop.
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

        # Import threading for non-blocking audio
        import threading
        self._threading = threading

    def _play_beep_async(self, frequency: int, duration_ms: int):
        """Play a beep in a background thread (non-blocking)."""
        def _beep():
            try:
                self._winsound.Beep(frequency, duration_ms)
            except Exception as e:
                print(f"[AUDIO] Error playing sound: {e}")

        thread = self._threading.Thread(target=_beep, daemon=True)
        thread.start()

    def _play_chime_async(self):
        """Play completion chime in a background thread (non-blocking)."""
        def _chime():
            try:
                self._winsound.Beep(600, 100)
                self._winsound.Beep(800, 100)
                self._winsound.Beep(1000, 200)
            except Exception as e:
                print(f"[AUDIO] Error playing sound: {e}")

        thread = self._threading.Thread(target=_chime, daemon=True)
        thread.start()

    def play(self, cue_name: str):
        """
        Play an audio cue (non-blocking).

        Cue names:
        - CLOSE_CUE: Signal to start closing (low pitch)
        - OPEN_CUE: Signal to start opening (high pitch)
        - CYCLE_COMPLETE: Success chime at end of cycle
        """
        if not self.enabled:
            return

        current_time = time.time()
        if current_time - self._last_cue_time < self._min_cue_interval:
            return
        self._last_cue_time = current_time

        if self._has_audio:
            if cue_name == "CLOSE_CUE":
                # Low-pitched beep for "start closing" (non-blocking)
                self._play_beep_async(400, 300)
            elif cue_name == "OPEN_CUE":
                # High-pitched beep for "start opening" (non-blocking)
                self._play_beep_async(800, 300)
            elif cue_name == "CYCLE_COMPLETE":
                # Success chime (non-blocking)
                self._play_chime_async()
        else:
            print(f"[AUDIO CUE] {cue_name}")


class ActivationViewerWidget(QWidget):
    """
    Widget for viewing and selecting template windows from activation segments.

    Displays EMG activation plot with click-to-select 1-second window functionality.
    """

    # Signal emitted when template is accepted
    template_accepted = Signal(np.ndarray)  # Emits the template array

    def __init__(self, class_label: str, parent=None):
        super().__init__(parent)
        self.class_label = class_label  # "open" or "closed"
        self.activations: List[np.ndarray] = []
        self.current_index: int = 0
        self.selected_start_sample: Optional[int] = None
        self.template_duration_samples: int = int(1.0 * config.FSAMP)  # 1 second
        self.accepted_templates: List[np.ndarray] = []

        self._setup_ui()

    def _setup_ui(self):
        """Create the viewer UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header with class label and count
        header_layout = QHBoxLayout()
        self.header_label = QLabel(f"{self.class_label.upper()} Activations: 0")
        self.header_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self.header_label)

        self.accepted_label = QLabel("Accepted: 0")
        self.accepted_label.setStyleSheet("color: green;")
        header_layout.addWidget(self.accepted_label)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        # Matplotlib figure for EMG plot
        self.figure = Figure(figsize=(8, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(200)
        self.ax = self.figure.add_subplot(111)
        self.figure.tight_layout()

        # Connect click event
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

        layout.addWidget(self.canvas)

        # Info label
        self.info_label = QLabel("Click on plot to select 1-second template start point")
        self.info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.info_label)

        # Navigation and action buttons
        button_layout = QHBoxLayout()

        self.prev_button = QPushButton("◀ Prev")
        self.prev_button.clicked.connect(self._go_prev)
        button_layout.addWidget(self.prev_button)

        self.index_label = QLabel("0 / 0")
        self.index_label.setAlignment(Qt.AlignCenter)
        self.index_label.setMinimumWidth(60)
        button_layout.addWidget(self.index_label)

        self.next_button = QPushButton("Next ▶")
        self.next_button.clicked.connect(self._go_next)
        button_layout.addWidget(self.next_button)

        button_layout.addSpacing(20)

        self.accept_button = QPushButton("✓ Accept Template")
        self.accept_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.accept_button.clicked.connect(self._accept_current)
        self.accept_button.setEnabled(False)
        button_layout.addWidget(self.accept_button)

        self.skip_button = QPushButton("Skip")
        self.skip_button.clicked.connect(self._go_next)
        button_layout.addWidget(self.skip_button)

        layout.addLayout(button_layout)

    def set_activations(self, activations: List[np.ndarray]):
        """Set the list of activations to review."""
        self.activations = activations
        self.current_index = 0
        self.selected_start_sample = None
        self.accepted_templates = []

        self.header_label.setText(f"{self.class_label.upper()} Activations: {len(activations)}")
        self._update_display()

    def _update_display(self):
        """Update the plot and navigation controls."""
        n_activations = len(self.activations)

        # Update navigation
        self.index_label.setText(f"{self.current_index + 1} / {n_activations}" if n_activations > 0 else "0 / 0")
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < n_activations - 1)
        self.skip_button.setEnabled(self.current_index < n_activations - 1)

        # Update accepted count
        self.accepted_label.setText(f"Accepted: {len(self.accepted_templates)}")

        # Clear and redraw plot
        self.ax.clear()

        if n_activations == 0 or self.current_index >= n_activations:
            self.ax.text(0.5, 0.5, "No activations to display",
                        ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
            self.accept_button.setEnabled(False)
            return

        activation = self.activations[self.current_index]
        n_samples = activation.shape[1]
        duration_s = n_samples / config.FSAMP

        # Plot all channels (or subset for clarity)
        time_axis = np.arange(n_samples) / config.FSAMP

        # Plot mean envelope across channels for clarity
        envelope = np.mean(np.abs(activation), axis=0)
        self.ax.plot(time_axis, envelope, 'b-', linewidth=0.8, alpha=0.8, label='Mean |EMG|')

        # Mark the selected template window
        if self.selected_start_sample is not None:
            start_s = self.selected_start_sample / config.FSAMP
            end_s = (self.selected_start_sample + self.template_duration_samples) / config.FSAMP

            # Shade the selected region
            self.ax.axvspan(start_s, end_s, alpha=0.3, color='green', label='Selected 1s')
            self.ax.axvline(start_s, color='green', linestyle='--', linewidth=2)
            self.ax.axvline(end_s, color='green', linestyle='--', linewidth=2)

            self.info_label.setText(f"Selected: {start_s:.2f}s - {end_s:.2f}s (click to change)")
            self.accept_button.setEnabled(True)
        else:
            self.info_label.setText("Click on plot to select 1-second template start point")
            self.accept_button.setEnabled(False)

        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title(f'{self.class_label.upper()} Activation {self.current_index + 1} ({duration_s:.2f}s)')
        self.ax.set_xlim(0, duration_s)
        self.ax.legend(loc='upper right', fontsize=8)
        self.ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def _on_canvas_click(self, event):
        """Handle click on canvas to select template start."""
        if event.inaxes != self.ax:
            return

        if len(self.activations) == 0 or self.current_index >= len(self.activations):
            return

        activation = self.activations[self.current_index]
        n_samples = activation.shape[1]

        # Convert click position to sample index
        click_time = event.xdata
        if click_time is None:
            return

        click_sample = int(click_time * config.FSAMP)

        # Ensure the 1-second window fits within the activation
        max_start = n_samples - self.template_duration_samples
        if max_start < 0:
            self.info_label.setText("Activation too short for 1-second template!")
            return

        # Clamp to valid range
        self.selected_start_sample = max(0, min(click_sample, max_start))

        self._update_display()

    def _go_prev(self):
        """Go to previous activation."""
        if self.current_index > 0:
            self.current_index -= 1
            self.selected_start_sample = None
            self._update_display()

    def _go_next(self):
        """Go to next activation."""
        if self.current_index < len(self.activations) - 1:
            self.current_index += 1
            self.selected_start_sample = None
            self._update_display()

    def _accept_current(self):
        """Accept the current template selection."""
        if self.selected_start_sample is None:
            return

        if self.current_index >= len(self.activations):
            return

        activation = self.activations[self.current_index]

        # Extract the selected 1-second template
        start = self.selected_start_sample
        end = start + self.template_duration_samples

        if end > activation.shape[1]:
            return

        template = activation[:, start:end]
        self.accepted_templates.append(template)

        # Emit signal
        self.template_accepted.emit(template)

        # Update display
        self.accepted_label.setText(f"Accepted: {len(self.accepted_templates)}")

        # Move to next
        if self.current_index < len(self.activations) - 1:
            self.current_index += 1
            self.selected_start_sample = None
            self._update_display()
        else:
            self.info_label.setText("All activations reviewed!")

    def get_accepted_templates(self) -> List[np.ndarray]:
        """Return list of accepted templates."""
        return self.accepted_templates

    def clear(self):
        """Clear all data."""
        self.activations = []
        self.current_index = 0
        self.selected_start_sample = None
        self.accepted_templates = []
        self._update_display()


class GuidedRecordProtocol(QObject):
    """
    Protocol for patient-friendly guided template recording.

    Features:
    - VHI animation guide (open→close→open)
    - Configurable timing for each phase
    - Pause between cycles for therapist instructions
    - Audio cues for transitions
    - Automatic GT from animation state
    - Post-recording review UI for template extraction
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
        self.animation_timer.setInterval(10)  # ~100 Hz updates for smooth VHI movement
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

        # Template management
        self.template_manager = TemplateManager()
        self.last_recording: Optional[dict] = None
        self.last_recording_path: Optional[str] = None

        # Initialize UI
        self._setup_protocol_ui()

    def _setup_protocol_ui(self) -> None:
        """Create the UI for guided recording protocol."""
        # Create main widget container with stacked widget for recording/review views
        self.main_widget = QWidget()
        main_layout = QVBoxLayout(self.main_widget)

        # Stacked widget to switch between recording and review views
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # === Recording View (index 0) ===
        self.recording_view = QWidget()
        recording_layout = QVBoxLayout(self.recording_view)

        # === Protocol Mode Group ===
        protocol_group = QGroupBox("Protocol Mode")
        protocol_layout = QHBoxLayout(protocol_group)

        protocol_layout.addWidget(QLabel("Animation:"))
        self.protocol_combo = QComboBox()
        self.protocol_combo.addItems([
            "Standard (Open \u2192 Close \u2192 Open)",
            "Inverted (Close \u2192 Open \u2192 Close)"
        ])
        self.protocol_combo.setToolTip(
            "Standard: For patients with open hand at rest\n"
            "Inverted: For patients with closed hand at rest"
        )
        protocol_layout.addWidget(self.protocol_combo)
        protocol_layout.addStretch()

        recording_layout.addWidget(protocol_group)

        # === Timing Configuration Group ===
        timing_group = QGroupBox("Timing Configuration")
        timing_layout = QGridLayout(timing_group)

        # Hold Open duration (time before close audio cue)
        timing_layout.addWidget(QLabel("Hold Open:"), 0, 0)
        self.hold_open_spinbox = QDoubleSpinBox()
        self.hold_open_spinbox.setRange(0.5, 10.0)
        self.hold_open_spinbox.setSingleStep(0.5)
        self.hold_open_spinbox.setValue(2.0)
        self.hold_open_spinbox.setSuffix(" s")
        self.hold_open_spinbox.setToolTip("Time before 'Close' audio cue plays")
        timing_layout.addWidget(self.hold_open_spinbox, 0, 1)

        # Closing transition duration (VHI hand closing time)
        timing_layout.addWidget(QLabel("Closing:"), 1, 0)
        self.closing_transition_spinbox = QDoubleSpinBox()
        self.closing_transition_spinbox.setRange(0.1, 2.0)
        self.closing_transition_spinbox.setSingleStep(0.05)
        self.closing_transition_spinbox.setValue(0.35)  # 350ms for onset capture
        self.closing_transition_spinbox.setSuffix(" s")
        self.closing_transition_spinbox.setToolTip("Duration for VHI hand to close (GT ramps 0→1)")
        timing_layout.addWidget(self.closing_transition_spinbox, 1, 1)

        # Hold Closed duration (time before open audio cue)
        timing_layout.addWidget(QLabel("Hold Closed:"), 2, 0)
        self.hold_closed_spinbox = QDoubleSpinBox()
        self.hold_closed_spinbox.setRange(0.5, 10.0)
        self.hold_closed_spinbox.setSingleStep(0.5)
        self.hold_closed_spinbox.setValue(2.0)
        self.hold_closed_spinbox.setSuffix(" s")
        self.hold_closed_spinbox.setToolTip("Time before 'Open' audio cue plays")
        timing_layout.addWidget(self.hold_closed_spinbox, 2, 1)

        # Opening transition duration (VHI hand opening time)
        timing_layout.addWidget(QLabel("Opening:"), 3, 0)
        self.opening_transition_spinbox = QDoubleSpinBox()
        self.opening_transition_spinbox.setRange(0.1, 2.0)
        self.opening_transition_spinbox.setSingleStep(0.05)
        self.opening_transition_spinbox.setValue(0.35)  # 350ms for onset capture
        self.opening_transition_spinbox.setSuffix(" s")
        self.opening_transition_spinbox.setToolTip("Duration for VHI hand to open (GT ramps 1→0)")
        timing_layout.addWidget(self.opening_transition_spinbox, 3, 1)

        # Reaction time (delay between audio cue and VHI movement)
        timing_layout.addWidget(QLabel("Reaction Time:"), 4, 0)
        self.reaction_time_combo = QComboBox()
        self.reaction_time_combo.addItems([
            "100 ms", "150 ms", "200 ms", "250 ms", "300 ms", "350 ms", "400 ms"
        ])
        self.reaction_time_combo.setCurrentText("200 ms")  # Default 200ms
        self.reaction_time_combo.setToolTip("Delay between audio cue and VHI hand movement start")
        timing_layout.addWidget(self.reaction_time_combo, 4, 1)

        # Audio cues checkbox
        self.audio_checkbox = QCheckBox("Audio cues enabled (Close / Open)")
        self.audio_checkbox.setChecked(True)
        self.audio_checkbox.toggled.connect(self._on_audio_toggled)
        timing_layout.addWidget(self.audio_checkbox, 5, 0, 1, 2)

        # Add timing group to recording layout
        recording_layout.addWidget(timing_group)

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

        recording_layout.addWidget(control_group)

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

        recording_layout.addWidget(progress_group)

        # === Label for saving ===
        label_group = QGroupBox("Recording Label")
        label_layout = QHBoxLayout(label_group)

        label_layout.addWidget(QLabel("Label:"))
        from PySide6.QtWidgets import QLineEdit
        self.label_line_edit = QLineEdit()
        self.label_line_edit.setPlaceholderText("e.g., position_1, table_neutral")
        label_layout.addWidget(self.label_line_edit)

        recording_layout.addWidget(label_group)

        # Spacer
        recording_layout.addStretch()

        self.stacked_widget.addWidget(self.recording_view)

        # === Review & Extract View (index 1) ===
        self._setup_review_ui()

    def _setup_review_ui(self) -> None:
        """Create the review & extract templates UI."""
        self.review_view = QWidget()
        review_layout = QVBoxLayout(self.review_view)

        # Header with back button
        header_layout = QHBoxLayout()

        self.back_to_recording_button = QPushButton("← Back to Recording")
        self.back_to_recording_button.clicked.connect(self._switch_to_recording_view)
        header_layout.addWidget(self.back_to_recording_button)

        header_layout.addStretch()

        self.review_title = QLabel("Review & Extract Templates")
        self.review_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        header_layout.addWidget(self.review_title)

        header_layout.addStretch()

        review_layout.addLayout(header_layout)

        # Splitter for CLOSED and OPEN viewers
        splitter = QSplitter(Qt.Vertical)

        # CLOSED activations viewer
        closed_group = QGroupBox("CLOSED Templates (from closing movements)")
        closed_layout = QVBoxLayout(closed_group)
        self.closed_viewer = ActivationViewerWidget("closed")
        self.closed_viewer.template_accepted.connect(self._on_template_accepted)
        closed_layout.addWidget(self.closed_viewer)
        splitter.addWidget(closed_group)

        # OPEN activations viewer
        open_group = QGroupBox("OPEN Templates (from opening movements)")
        open_layout = QVBoxLayout(open_group)
        self.open_viewer = ActivationViewerWidget("open")
        self.open_viewer.template_accepted.connect(self._on_template_accepted)
        open_layout.addWidget(self.open_viewer)
        splitter.addWidget(open_group)

        review_layout.addWidget(splitter, stretch=1)

        # Bottom actions
        actions_layout = QHBoxLayout()

        self.save_status_label = QLabel("")
        self.save_status_label.setStyleSheet("color: #666;")
        actions_layout.addWidget(self.save_status_label)

        actions_layout.addStretch()

        self.save_all_button = QPushButton("Save All Accepted Templates")
        self.save_all_button.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px 20px; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.save_all_button.clicked.connect(self._save_all_templates)
        actions_layout.addWidget(self.save_all_button)

        review_layout.addLayout(actions_layout)

        self.stacked_widget.addWidget(self.review_view)

    def _switch_to_recording_view(self):
        """Switch back to recording view."""
        self.stacked_widget.setCurrentIndex(0)

    def _switch_to_review_view(self):
        """Switch to review & extract view."""
        self.stacked_widget.setCurrentIndex(1)

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
        protocol_mode = self._get_protocol_mode()
        self.sequencer.setup_cycle(
            hold_open_s=self.hold_open_spinbox.value(),
            closing_s=self.closing_transition_spinbox.value(),
            hold_closed_s=self.hold_closed_spinbox.value(),
            opening_s=self.opening_transition_spinbox.value(),
            reaction_time_s=self._get_reaction_time_s(),
            protocol_mode=protocol_mode
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

        # Send initial hand position to VHI based on protocol mode
        if protocol_mode == "inverted":
            self._send_joints_to_vhi([1.0] * 10)  # Start closed
        else:
            self._send_joints_to_vhi([0.0] * 10)  # Start open

        print("\n" + "=" * 60)
        print("GUIDED RECORDING STARTED")
        print(f"  Protocol Mode: {protocol_mode.upper()}")
        print("  Predicted Hand becomes Control Hand (visual guide)")
        print(f"  Hold Open: {self.hold_open_spinbox.value()}s")
        print(f"  Closing: {self.closing_transition_spinbox.value()}s")
        print(f"  Hold Closed: {self.hold_closed_spinbox.value()}s")
        print(f"  Opening: {self.opening_transition_spinbox.value()}s")
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
        recording_path = self._save_recording()

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

        # Note: Template extraction is done in the separate "Bidirectional Training" tab
        # Recording saved - user can now load it in the training protocol

    def _get_reaction_time_s(self) -> float:
        """Get reaction time in seconds from the combo box."""
        text = self.reaction_time_combo.currentText()  # e.g., "200 ms"
        ms_value = int(text.split()[0])  # Extract number
        return ms_value / 1000.0

    def _get_protocol_mode(self) -> str:
        """Get protocol mode from the combo box."""
        # Index 0 = "Standard", Index 1 = "Inverted"
        return "inverted" if self.protocol_combo.currentIndex() == 1 else "standard"

    def _on_start_next_cycle(self):
        """Start the next animation cycle."""
        if not self.is_recording:
            return

        # Update sequencer timings in case they changed
        protocol_mode = self._get_protocol_mode()
        self.sequencer.setup_cycle(
            hold_open_s=self.hold_open_spinbox.value(),
            closing_s=self.closing_transition_spinbox.value(),
            hold_closed_s=self.hold_closed_spinbox.value(),
            opening_s=self.opening_transition_spinbox.value(),
            reaction_time_s=self._get_reaction_time_s(),
            protocol_mode=protocol_mode
        )

        # Start new cycle
        self.cycle_start_time = time.time()
        self.sequencer.start_cycle()
        self.is_cycle_running = True
        self.waiting_for_next_cycle = False

        # Record cycle boundary with timing config
        self.cycle_boundaries.append({
            "start_time": self.cycle_start_time,
            "end_time": None,
            "cycle_number": self.cycles_completed + 1,
            "close_cue_time": None,  # Will be filled when cue plays
            "open_cue_time": None,   # Will be filled when cue plays
            "timing_config": {
                "hold_open_s": self.hold_open_spinbox.value(),
                "closing_s": self.closing_transition_spinbox.value(),
                "hold_closed_s": self.hold_closed_spinbox.value(),
                "opening_s": self.opening_transition_spinbox.value(),
                "reaction_time_s": self._get_reaction_time_s(),
            },
            "protocol_mode": protocol_mode,
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

            # Record GT (now a float 0.0-1.0 with linear transitions)
            self.gt_buffer.append((current_time, gt_value))

            # Handle phase change - play audio cues at the right moments
            if phase_changed:
                if phase_name == self.sequencer.PHASE_REACTION_CLOSE:
                    # Play CLOSE audio cue and record timestamp
                    self.audio_manager.play("CLOSE_CUE")
                    self.sequencer.record_close_cue_time(current_time)
                    if self.cycle_boundaries:
                        self.cycle_boundaries[-1]["close_cue_time"] = current_time
                    print(f"[GUIDED] AUDIO: Close cue played")

                elif phase_name == self.sequencer.PHASE_REACTION_OPEN:
                    # Play OPEN audio cue and record timestamp
                    self.audio_manager.play("OPEN_CUE")
                    self.sequencer.record_open_cue_time(current_time)
                    if self.cycle_boundaries:
                        self.cycle_boundaries[-1]["open_cue_time"] = current_time
                    print(f"[GUIDED] AUDIO: Open cue played")

                if phase_name != "CYCLE_COMPLETE":
                    print(f"[GUIDED] Phase: {phase_name}")

            # Update UI
            self._update_phase_ui(phase_name, current_time)

            # Check cycle completion
            if self.sequencer.is_cycle_complete():
                self._on_cycle_complete()

        elif self.waiting_for_next_cycle:
            # Keep VHI at starting position while waiting (based on protocol mode)
            if self.sequencer.protocol_mode == "inverted":
                self._send_joints_to_vhi([1.0] * 10)  # Closed position
                self.gt_buffer.append((current_time, 1.0))
            else:
                self._send_joints_to_vhi([0.0] * 10)  # Open position
                self.gt_buffer.append((current_time, 0.0))

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
        # User-friendly phase names (context-aware for inverted mode)
        is_inverted = self.sequencer.protocol_mode == "inverted"
        if is_inverted:
            display_names = {
                "HOLD_CLOSED": "Hold Closed (waiting for open cue)",
                "REACTION_OPEN": "Reaction Time (open cue played)",
                "OPENING": "Opening (VHI hand opening)",
                "HOLD_OPEN": "Hold Open (waiting for close cue)",
                "REACTION_CLOSE": "Reaction Time (close cue played)",
                "CLOSING": "Closing (VHI hand closing)",
                "HOLD_CLOSED_END": "Hold Closed (cycle ending)",
            }
        else:
            display_names = {
                "HOLD_OPEN": "Hold Open (waiting for close cue)",
                "REACTION_CLOSE": "Reaction Time (close cue played)",
                "CLOSING": "Closing (VHI hand closing)",
                "HOLD_CLOSED": "Hold Closed (waiting for open cue)",
                "REACTION_OPEN": "Reaction Time (open cue played)",
                "OPENING": "Opening (VHI hand opening)",
                "HOLD_OPEN_END": "Hold Open (cycle ending)",
            }
        display_name = display_names.get(phase_name, phase_name)
        self.phase_label.setText(f"Phase: {display_name}")

        # Calculate progress within current phase
        time_remaining = self.sequencer.get_time_remaining_in_phase(current_time)
        reaction_time = self._get_reaction_time_s()
        phase_durations = {
            "HOLD_OPEN": self.hold_open_spinbox.value(),
            "REACTION_CLOSE": reaction_time,
            "CLOSING": self.closing_transition_spinbox.value(),
            "HOLD_CLOSED": self.hold_closed_spinbox.value(),
            "REACTION_OPEN": reaction_time,
            "OPENING": self.opening_transition_spinbox.value(),
            "HOLD_OPEN_END": self.hold_open_spinbox.value(),
            "HOLD_CLOSED_END": self.hold_closed_spinbox.value(),
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
        """Enable/disable timing and protocol controls."""
        self.hold_open_spinbox.setEnabled(enabled)
        self.closing_transition_spinbox.setEnabled(enabled)
        self.hold_closed_spinbox.setEnabled(enabled)
        self.opening_transition_spinbox.setEnabled(enabled)
        self.reaction_time_combo.setEnabled(enabled)
        self.protocol_combo.setEnabled(enabled)

    def _save_recording(self) -> Optional[str]:
        """Save the recording to file."""
        if not self.emg_buffer:
            print("[GUIDED] No EMG data to save!")
            return None

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

        # Get mode suffix from device
        mode_suffix = self.main_window.device.get_mode_suffix()
        differential_mode = config.ENABLE_DIFFERENTIAL_MODE

        # Build save dictionary
        save_dict = {
            "emg": emg_signal,
            "gt": gt_signal,
            "gt_raw": self.gt_buffer,  # Raw (timestamp, gt_value) pairs - gt_value is float 0.0-1.0
            "timings_emg": emg_timestamps,
            "gt_mode": "guided_animation",
            "animation_config": {
                "hold_open_s": self.hold_open_spinbox.value(),
                "closing_s": self.closing_transition_spinbox.value(),
                "hold_closed_s": self.hold_closed_spinbox.value(),
                "opening_s": self.opening_transition_spinbox.value(),
                "reaction_time_s": self._get_reaction_time_s(),
                "protocol_mode": self._get_protocol_mode(),
            },
            "cycles": self.cycle_boundaries,  # Includes close_cue_time and open_cue_time per cycle
            "cycles_completed": self.cycles_completed,
            "label": label,
            "task": f"guided_{self._get_protocol_mode()}",
            "recording_duration_s": time.time() - self.recording_start_time,
            "differential_mode": differential_mode,
        }

        # Save file
        if not os.path.exists(self.recording_dir_path):
            os.makedirs(self.recording_dir_path)

        now = datetime.now()
        formatted_now = now.strftime("%Y%m%d_%H%M%S%f")
        file_name = f"MindMove_GuidedRecording{mode_suffix}{formatted_now}_{label}.pkl"
        file_path = os.path.join(self.recording_dir_path, file_name)

        with open(file_path, "wb") as f:
            pickle.dump(save_dict, f)

        # Store for review
        self.last_recording = save_dict
        self.last_recording_path = file_path

        print(f"\n[GUIDED] Recording saved: {file_name}")
        print(f"  EMG shape: {emg_signal.shape}")
        print(f"  GT shape: {gt_signal.shape}")
        print(f"  Cycles completed: {self.cycles_completed}")
        print(f"  Total duration: {save_dict['recording_duration_s']:.1f}s")

        return file_path

    def _populate_review_ui(self):
        """Extract activations and populate the review UI."""
        if self.last_recording is None:
            print("[GUIDED] No recording to review!")
            return

        # Clear previous data
        self.template_manager.clear_all()
        self.closed_viewer.clear()
        self.open_viewer.clear()

        # Extract bidirectional activations
        print("\n[GUIDED] Extracting OPEN and CLOSED activations...")
        activations = self.template_manager.extract_activations_bidirectional(
            self.last_recording,
            min_duration_s=config.MIN_ACTIVATION_DURATION_S
        )

        # Populate viewers
        self.closed_viewer.set_activations(activations["closed"])
        self.open_viewer.set_activations(activations["open"])

        # Update save status
        self._update_save_status()

        print(f"[GUIDED] Review UI populated:")
        print(f"  CLOSED activations: {len(activations['closed'])}")
        print(f"  OPEN activations: {len(activations['open'])}")

    def _update_save_status(self):
        """Update the save status label."""
        n_closed = len(self.closed_viewer.get_accepted_templates())
        n_open = len(self.open_viewer.get_accepted_templates())
        self.save_status_label.setText(
            f"Ready to save: {n_closed} CLOSED, {n_open} OPEN templates"
        )

    def _on_template_accepted(self, template: np.ndarray):
        """Called when a template is accepted in either viewer."""
        self._update_save_status()

    def _save_all_templates(self):
        """Save all accepted templates to disk."""
        closed_templates = self.closed_viewer.get_accepted_templates()
        open_templates = self.open_viewer.get_accepted_templates()

        if not closed_templates and not open_templates:
            QMessageBox.warning(
                self.main_widget,
                "No Templates",
                "No templates have been accepted. Please review activations and accept templates before saving."
            )
            return

        # Set templates in template manager
        self.template_manager.templates["closed"] = closed_templates
        self.template_manager.templates["open"] = open_templates

        # Get label for template set name
        label = self.label_line_edit.text().strip()
        template_set_name = label if label else None

        saved_paths = []

        # Save CLOSED templates
        if closed_templates:
            path = self.template_manager.save_templates("closed", template_set_name)
            saved_paths.append(path)
            print(f"[GUIDED] Saved {len(closed_templates)} CLOSED templates to {path}")

        # Save OPEN templates
        if open_templates:
            path = self.template_manager.save_templates("open", template_set_name)
            saved_paths.append(path)
            print(f"[GUIDED] Saved {len(open_templates)} OPEN templates to {path}")

        # Show confirmation
        msg = f"Templates saved successfully!\n\n"
        msg += f"CLOSED templates: {len(closed_templates)}\n"
        msg += f"OPEN templates: {len(open_templates)}\n\n"
        msg += "Saved to:\n" + "\n".join(saved_paths)

        QMessageBox.information(
            self.main_widget,
            "Templates Saved",
            msg
        )

        self.save_status_label.setText(
            f"Saved: {len(closed_templates)} CLOSED, {len(open_templates)} OPEN templates"
        )
        self.save_status_label.setStyleSheet("color: green; font-weight: bold;")

    def _build_gt_signal_at_emg_rate(self, n_emg_samples: int) -> np.ndarray:
        """
        Build GT signal at EMG sample rate from recorded GT buffer.

        The GT buffer contains (timestamp, gt_value) pairs at ~30Hz.
        GT values are floats (0.0-1.0) with linear ramps during transitions.
        This uses linear interpolation to upsample to EMG rate (2000Hz).
        """
        if not self.gt_buffer or not self.emg_buffer:
            return np.zeros(n_emg_samples)

        gt_signal = np.zeros(n_emg_samples, dtype=np.float32)

        # Get EMG time range
        emg_start_time = self.emg_buffer[0][0]
        emg_end_time = self.emg_buffer[-1][0]
        emg_duration = emg_end_time - emg_start_time

        if emg_duration <= 0:
            return gt_signal

        # Convert GT buffer to arrays for interpolation
        gt_times = np.array([t for t, _ in self.gt_buffer])
        gt_values = np.array([v for _, v in self.gt_buffer], dtype=np.float32)

        # Create EMG time axis
        emg_times = np.linspace(emg_start_time, emg_end_time, n_emg_samples)

        # Linearly interpolate GT values to EMG sample rate
        gt_signal = np.interp(emg_times, gt_times, gt_values)

        return gt_signal
