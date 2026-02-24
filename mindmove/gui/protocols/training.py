from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Dict, Tuple

from PySide6.QtCore import QObject, Signal, QThread, Qt
from PySide6.QtWidgets import (
    QFileDialog, QMessageBox, QListWidgetItem, QLabel, QLineEdit,
    QComboBox, QPushButton, QSpinBox, QDoubleSpinBox, QGroupBox,
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QWidget, QListWidget,
    QScrollArea,
)
import pickle
from datetime import datetime
import numpy as np
import os

# Matplotlib imports for review UI
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# MindMove imports
from mindmove.model.interface import MindMoveInterface
from mindmove.model.templates.template_manager import TemplateManager
from mindmove.config import config

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


def _infer_mode_from_templates(templates: list) -> bool:
    """
    Infer differential mode from template arrays.
    Templates are (n_channels, n_samples) — if n_channels <= 16 it's SD.
    """
    if templates and hasattr(templates[0], 'shape'):
        return templates[0].shape[0] <= 16
    return config.ENABLE_DIFFERENTIAL_MODE  # fallback


if TYPE_CHECKING:
    from mindmove.gui.mindmove import MindMove


class PyQtThread(QThread):
    has_finished_signal = Signal()
    progress_bar_signal = Signal(int)

    def __init__(self, target, parent=None) -> None:
        super(PyQtThread, self).__init__(parent)

        self.t = target

    def run(self):
        self.t()
        self.has_finished_signal.emit()

    def quit(self) -> None:
        self.exit(0)


class ActivationReviewWidget(QWidget):
    """
    Widget for reviewing and selecting template windows from activation segments.
    Used in the Guided Recording Review Dialog.
    """

    template_accepted = Signal(np.ndarray)

    def __init__(self, class_label: str, parent=None):
        super().__init__(parent)
        self.class_label = class_label
        self.activations: List[np.ndarray] = []
        self.gt_signals: List[np.ndarray] = []  # GT for each activation
        self.current_index: int = 0
        self.selected_start_sample: Optional[int] = None
        self.template_duration_samples: int = int(1.0 * config.FSAMP)
        self.accepted_templates: List[np.ndarray] = []
        self.accepted_indices: set = set()  # Track which activations have been accepted
        self.current_channel: int = 0  # 0-indexed internally

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Enable focus for keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

        # Header with channel selector
        header_layout = QHBoxLayout()
        self.header_label = QLabel(f"{self.class_label.upper()} Activations: 0")
        self.header_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self.header_label)

        self.accepted_label = QLabel("Accepted: 0")
        self.accepted_label.setStyleSheet("color: green;")
        header_layout.addWidget(self.accepted_label)

        header_layout.addStretch()

        # Channel selector (1-indexed for display)
        header_layout.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.setFixedWidth(60)
        for i in range(1, 33):  # 1-32 for user
            self.channel_combo.addItem(str(i))
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        self.channel_combo.setToolTip("Use Up/Down arrows to switch channels, Left/Right for navigation")
        header_layout.addWidget(self.channel_combo)

        # Keyboard hint
        hint = QLabel("(↑↓ ch, ←→ nav)")
        hint.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
        header_layout.addWidget(hint)

        layout.addLayout(header_layout)

        # Matplotlib figure
        self.figure = Figure(figsize=(8, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(200)
        self.ax = self.figure.add_subplot(111)
        self.figure.tight_layout()
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        layout.addWidget(self.canvas)

        # Info label
        self.info_label = QLabel("Click on plot to select 1-second template start point")
        self.info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.info_label)

        # Buttons
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

    def _on_channel_changed(self, index: int):
        """Handle channel selection change."""
        self.current_channel = index  # combo index is 0-based, matches our internal 0-indexed channel
        self._update_display()

    def keyPressEvent(self, event):
        """Handle arrow key presses for fast channel switching."""
        if event.key() == Qt.Key_Up:
            new_idx = max(0, self.channel_combo.currentIndex() - 1)
            self.channel_combo.setCurrentIndex(new_idx)
            event.accept()
        elif event.key() == Qt.Key_Down:
            new_idx = min(31, self.channel_combo.currentIndex() + 1)
            self.channel_combo.setCurrentIndex(new_idx)
            event.accept()
        elif event.key() == Qt.Key_Left:
            self._go_prev()
            event.accept()
        elif event.key() == Qt.Key_Right:
            self._go_next()
            event.accept()
        else:
            super().keyPressEvent(event)

    def set_activations(self, activations: List[np.ndarray], gt_signals: Optional[List[np.ndarray]] = None):
        """Set activations and optionally their GT signals."""
        self.activations = activations
        self.gt_signals = gt_signals if gt_signals else []
        self.current_index = 0
        self.selected_start_sample = None
        self.accepted_templates = []
        self.accepted_indices = set()
        self.header_label.setText(f"{self.class_label.upper()} Activations: {len(activations)}")
        self._update_display()

    def _update_display(self):
        n_activations = len(self.activations)

        self.index_label.setText(f"{self.current_index + 1} / {n_activations}" if n_activations > 0 else "0 / 0")
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < n_activations - 1)
        self.skip_button.setEnabled(self.current_index < n_activations - 1)
        self.accepted_label.setText(f"Accepted: {len(self.accepted_templates)}")

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
        time_axis = np.arange(n_samples) / config.FSAMP

        # Get raw EMG for selected channel
        channel = min(self.current_channel, activation.shape[0] - 1)
        emg_signal = activation[channel, :]
        max_val = np.max(np.abs(emg_signal)) if np.max(np.abs(emg_signal)) > 0 else 1

        # Plot GT overlay if available (as overlapping line scaled to EMG amplitude)
        if self.current_index < len(self.gt_signals) and len(self.gt_signals[self.current_index]) > 0:
            gt = self.gt_signals[self.current_index]
            gt_time = np.arange(len(gt)) / config.FSAMP

            # Scale GT to match EMG amplitude for visibility
            gt_scaled = gt * max_val * 0.9
            self.ax.plot(gt_time, gt_scaled, 'r-', linewidth=1.5, alpha=0.7, label='GT')

        # Plot raw EMG signal for selected channel
        self.ax.plot(time_axis, emg_signal, 'b-', linewidth=0.5, alpha=0.9, label=f'EMG Ch{channel + 1}')

        # Mark selected region
        if self.selected_start_sample is not None:
            start_s = self.selected_start_sample / config.FSAMP
            end_s = (self.selected_start_sample + self.template_duration_samples) / config.FSAMP
            self.ax.axvspan(start_s, end_s, alpha=0.3, color='green', label='Selected 1s')
            self.ax.axvline(start_s, color='green', linestyle='--', linewidth=2)
            self.ax.axvline(end_s, color='green', linestyle='--', linewidth=2)
            self.info_label.setText(f"Selected: {start_s:.2f}s - {end_s:.2f}s (click to change)")
            # Only enable accept if this activation hasn't been accepted yet
            self.accept_button.setEnabled(self.current_index not in self.accepted_indices)
        else:
            self.info_label.setText("Click on plot to select 1-second template start point")
            self.accept_button.setEnabled(False)

        # Show if already accepted
        if self.current_index in self.accepted_indices:
            self.info_label.setText(f"Already accepted! Use Prev/Next to navigate.")
            self.accept_button.setEnabled(False)

        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude (µV)')
        self.ax.set_title(f'{self.class_label.upper()} Activation {self.current_index + 1} ({duration_s:.2f}s)')
        self.ax.set_xlim(0, duration_s)
        self.ax.legend(loc='upper right', fontsize=8)
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def _on_canvas_click(self, event):
        # Grab focus for keyboard events
        self.setFocus()
        if event.inaxes != self.ax:
            return
        if len(self.activations) == 0 or self.current_index >= len(self.activations):
            return
        # Don't allow selection if already accepted
        if self.current_index in self.accepted_indices:
            return

        activation = self.activations[self.current_index]
        n_samples = activation.shape[1]

        click_time = event.xdata
        if click_time is None:
            return

        click_sample = int(click_time * config.FSAMP)
        max_start = n_samples - self.template_duration_samples
        if max_start < 0:
            self.info_label.setText("Activation too short for 1-second template!")
            return

        self.selected_start_sample = max(0, min(click_sample, max_start))
        self._update_display()

    def _go_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.selected_start_sample = None
            self._update_display()

    def _go_next(self):
        if self.current_index < len(self.activations) - 1:
            self.current_index += 1
            self.selected_start_sample = None
            self._update_display()

    def _accept_current(self):
        if self.selected_start_sample is None or self.current_index >= len(self.activations):
            return
        # Prevent duplicate acceptance
        if self.current_index in self.accepted_indices:
            return

        activation = self.activations[self.current_index]
        start = self.selected_start_sample
        end = start + self.template_duration_samples

        if end > activation.shape[1]:
            return

        template = activation[:, start:end]
        self.accepted_templates.append(template)
        self.accepted_indices.add(self.current_index)  # Mark as accepted
        self.template_accepted.emit(template)
        self.accepted_label.setText(f"Accepted: {len(self.accepted_templates)}")

        # Disable accept button for this activation
        self.accept_button.setEnabled(False)

        if self.current_index < len(self.activations) - 1:
            self.current_index += 1
            self.selected_start_sample = None
            self._update_display()
        else:
            self.info_label.setText("All activations reviewed!")

    def get_accepted_templates(self) -> List[np.ndarray]:
        return self.accepted_templates

    def clear(self):
        self.activations = []
        self.gt_signals = []
        self.current_index = 0
        self.selected_start_sample = None
        self.accepted_templates = []
        self.accepted_indices = set()
        self._update_display()


class DraggableWindow:
    """A draggable rectangle representing a 1-second template window."""

    def __init__(self, ax, start_sample: int, duration_samples: int, color: str, label: str, fsamp: float):
        self.ax = ax
        self.start_sample = start_sample
        self.duration_samples = duration_samples
        self.color = color
        self.label = label
        self.fsamp = fsamp

        # Convert to time for display
        start_time = start_sample / fsamp
        duration_time = duration_samples / fsamp

        # Create rectangle patch (will be drawn in time coordinates)
        self.rect = Rectangle(
            (start_time, -1e6), duration_time, 2e6,
            alpha=0.3, facecolor=color, edgecolor=color, linewidth=2
        )
        ax.add_patch(self.rect)

        # Create start/end line markers
        self.start_line = ax.axvline(start_time, color=color, linewidth=2, linestyle='-')
        self.end_line = ax.axvline(start_time + duration_time, color=color, linewidth=2, linestyle='-')

        # Dragging state
        self.dragging = False

    def set_position(self, start_sample: int):
        """Set the position of the template window (in samples)."""
        self.start_sample = start_sample
        start_time = start_sample / self.fsamp
        duration_time = self.duration_samples / self.fsamp
        self.rect.set_x(start_time)
        self.start_line.set_xdata([start_time, start_time])
        self.end_line.set_xdata([start_time + duration_time, start_time + duration_time])

    def contains(self, time_s: float) -> bool:
        """Check if time coordinate is within the window."""
        start_time = self.start_sample / self.fsamp
        end_time = (self.start_sample + self.duration_samples) / self.fsamp
        return start_time <= time_s <= end_time

    def get_sample_range(self) -> Tuple[int, int]:
        """Return (start_sample, end_sample)."""
        return self.start_sample, self.start_sample + self.duration_samples


class CycleReviewWidget(QWidget):
    """
    Widget for reviewing complete cycles and selecting both CLOSED and OPEN templates.

    Shows one complete cycle at a time with:
    - Audio cue markers (blue vertical lines)
    - GT signal as background shading
    - Draggable template windows for CLOSED (orange) and OPEN (blue)

    User drags the windows to desired positions, then accepts both templates.
    """

    # Signal emitted when both templates from a cycle are accepted
    templates_accepted = Signal(np.ndarray, np.ndarray)  # (closed_template, open_template)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cycles: List[Dict] = []
        self.current_cycle_idx: int = 0

        # Template duration
        self.template_duration_samples: int = int(1.0 * config.FSAMP)

        # Draggable windows
        self.closed_window: Optional[DraggableWindow] = None
        self.open_window: Optional[DraggableWindow] = None

        # Mouse dragging state
        self._dragging_window: Optional[DraggableWindow] = None
        self._drag_offset: float = 0

        # Accepted templates - keyed by cycle index to prevent duplicates
        self.accepted_closed_templates: Dict[int, np.ndarray] = {}
        self.accepted_open_templates: Dict[int, np.ndarray] = {}

        # Saved window positions per cycle - preserves positions when changing channels
        self._saved_closed_positions: Dict[int, int] = {}  # cycle_idx -> start_sample
        self._saved_open_positions: Dict[int, int] = {}    # cycle_idx -> start_sample

        # Onset detection info - per-cycle channel activation data
        self._onset_info: list = []  # list of dicts with channels_fired per cycle

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Enable focus for keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

        # Header with cycle info and channel selector
        header_layout = QHBoxLayout()

        self.cycle_label = QLabel("Cycle 0 / 0")
        self.cycle_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(self.cycle_label)

        header_layout.addStretch()

        # Accepted counts
        self.accepted_label = QLabel("Accepted: 0 CLOSED, 0 OPEN")
        self.accepted_label.setStyleSheet("color: green;")
        header_layout.addWidget(self.accepted_label)

        header_layout.addSpacing(20)

        # Keyboard hint
        hint = QLabel("(←→ nav)")
        hint.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
        header_layout.addWidget(hint)

        layout.addLayout(header_layout)

        # Instructions label
        self.instruction_label = QLabel(
            "Drag the orange (CLOSED) and blue (OPEN) windows to select 1-second templates. "
            "Click 'Accept' when satisfied."
        )
        self.instruction_label.setStyleSheet(
            "color: #666; background-color: #f5f5f5; padding: 8px; border-radius: 4px;"
        )
        self.instruction_label.setWordWrap(True)
        layout.addWidget(self.instruction_label)

        # Matplotlib figure - wider and taller for stacked all-channels view
        self.figure = Figure(figsize=(14, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(500)
        self.ax = self.figure.add_subplot(111)
        self.figure.tight_layout()

        # Connect mouse events for dragging
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

        layout.addWidget(self.canvas)

        # Info label for selection feedback
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.info_label)

        # Navigation and action buttons
        button_layout = QHBoxLayout()

        self.prev_button = QPushButton("◀ Prev Cycle")
        self.prev_button.clicked.connect(self._go_prev)
        button_layout.addWidget(self.prev_button)

        self.cycle_index_label = QLabel("0 / 0")
        self.cycle_index_label.setAlignment(Qt.AlignCenter)
        self.cycle_index_label.setMinimumWidth(80)
        button_layout.addWidget(self.cycle_index_label)

        self.next_button = QPushButton("Next Cycle ▶")
        self.next_button.clicked.connect(self._go_next)
        button_layout.addWidget(self.next_button)

        button_layout.addSpacing(30)

        # Reset windows button
        self.reset_button = QPushButton("Reset Windows")
        self.reset_button.clicked.connect(self._reset_windows)
        button_layout.addWidget(self.reset_button)

        # Accept buttons
        self.accept_button = QPushButton("✓ Accept Both")
        self.accept_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 12px; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.accept_button.clicked.connect(self._accept_templates)
        button_layout.addWidget(self.accept_button)

        # Accept Only CLOSED button
        self.accept_closed_btn = QPushButton("Accept CLOSED")
        self.accept_closed_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 6px 12px; }"
        )
        self.accept_closed_btn.clicked.connect(self._accept_closed_only)
        button_layout.addWidget(self.accept_closed_btn)

        # Accept Only OPEN button
        self.accept_open_btn = QPushButton("Accept OPEN")
        self.accept_open_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 6px 12px; }"
        )
        self.accept_open_btn.clicked.connect(self._accept_open_only)
        button_layout.addWidget(self.accept_open_btn)

        self.skip_button = QPushButton("Skip")
        self.skip_button.clicked.connect(self._go_next)
        button_layout.addWidget(self.skip_button)

        layout.addLayout(button_layout)

    def set_cycles(self, cycles: List[Dict], recording_name: str = None):
        """Set the list of cycles to review.

        Args:
            cycles: List of cycle dictionaries
            recording_name: Optional name to display in the plot title
        """
        self.cycles = cycles
        self.current_cycle_idx = 0
        self.accepted_closed_templates = {}  # Keyed by cycle index
        self.accepted_open_templates = {}    # Keyed by cycle index
        self._saved_closed_positions = {}    # Clear saved positions for new data
        self._saved_open_positions = {}      # Clear saved positions for new data
        self.recording_name = recording_name  # Store for display in title

        self._update_display()

    def keyPressEvent(self, event):
        """Handle arrow key presses for cycle navigation."""
        if event.key() == Qt.Key_Left:
            self._go_prev()
            event.accept()
        elif event.key() == Qt.Key_Right:
            self._go_next()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _save_window_positions(self):
        """Save current window positions for the current cycle."""
        if self.closed_window is not None:
            self._saved_closed_positions[self.current_cycle_idx] = self.closed_window.start_sample
        if self.open_window is not None:
            self._saved_open_positions[self.current_cycle_idx] = self.open_window.start_sample

    def _reset_windows(self):
        """Reset window positions to defaults for current cycle."""
        # Clear saved positions for current cycle to force defaults
        self._saved_closed_positions.pop(self.current_cycle_idx, None)
        self._saved_open_positions.pop(self.current_cycle_idx, None)
        self._update_display()

    def _update_display(self):
        """Update the plot and UI elements."""
        n_cycles = len(self.cycles)

        # Update navigation
        self.cycle_label.setText(f"Cycle {self.current_cycle_idx + 1} / {n_cycles}" if n_cycles > 0 else "No cycles")
        self.cycle_index_label.setText(f"{self.current_cycle_idx + 1} / {n_cycles}" if n_cycles > 0 else "0 / 0")
        self.prev_button.setEnabled(self.current_cycle_idx > 0)
        self.next_button.setEnabled(self.current_cycle_idx < n_cycles - 1)
        self.skip_button.setEnabled(self.current_cycle_idx < n_cycles - 1)

        # Update accepted counts (show separate CLOSED and OPEN counts)
        n_closed = len(self.accepted_closed_templates)
        n_open = len(self.accepted_open_templates)
        self.accepted_label.setText(f"Accepted: {n_closed} CLOSED, {n_open} OPEN")

        # Check what's already accepted for current cycle
        has_closed = self.current_cycle_idx in self.accepted_closed_templates
        has_open = self.current_cycle_idx in self.accepted_open_templates

        # Update button styles based on accepted status
        if has_closed and has_open:
            self.accept_button.setText("✓ Update Both")
            self.accept_button.setStyleSheet(
                "QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 6px 12px; }"
            )
        else:
            self.accept_button.setText("✓ Accept Both")
            self.accept_button.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 12px; }"
                "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
            )

        # Update individual button styles
        if has_closed:
            self.accept_closed_btn.setStyleSheet(
                "QPushButton { background-color: #E65100; color: white; font-weight: bold; padding: 6px 12px; }"
            )
            self.accept_closed_btn.setText("Update CLOSED")
        else:
            self.accept_closed_btn.setStyleSheet(
                "QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 6px 12px; }"
            )
            self.accept_closed_btn.setText("Accept CLOSED")

        if has_open:
            self.accept_open_btn.setStyleSheet(
                "QPushButton { background-color: #1565C0; color: white; font-weight: bold; padding: 6px 12px; }"
            )
            self.accept_open_btn.setText("Update OPEN")
        else:
            self.accept_open_btn.setStyleSheet(
                "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 6px 12px; }"
            )
            self.accept_open_btn.setText("Accept OPEN")

        # Clear and redraw plot - recreate axes to avoid stale state
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.closed_window = None
        self.open_window = None

        if n_cycles == 0 or self.current_cycle_idx >= n_cycles:
            self.ax.text(0.5, 0.5, "No cycles to display",
                        ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
            self.accept_button.setEnabled(False)
            return

        cycle = self.cycles[self.current_cycle_idx]
        emg = cycle['emg']
        gt = cycle['gt']
        close_start_idx = cycle.get('close_start_idx', 0)
        open_start_idx = cycle.get('open_start_idx', 0)

        # Audio cue times (if available from guided recording)
        close_cue_idx = cycle.get('close_cue_idx')
        open_cue_idx = cycle.get('open_cue_idx')

        n_channels = emg.shape[0]
        n_samples = emg.shape[1]
        time_axis = np.arange(n_samples) / config.FSAMP

        # Compute vertical offset for stacked channels from median peak-to-peak
        all_ranges = [np.ptp(emg[ch, :]) for ch in range(n_channels)]
        offset_step = np.median(all_ranges) * 1.5 if np.median(all_ranges) > 0 else 1.0

        # Get onset channel info for this cycle (if available)
        closed_channels_fired = set()
        open_channels_fired = set()
        dead_channels = set()
        artifact_channels_closed = set()
        artifact_channels_open = set()
        if self._onset_info and self.current_cycle_idx < len(self._onset_info):
            info = self._onset_info[self.current_cycle_idx]
            closed_channels_fired = set(info.get("closed_channels_fired", []))
            open_channels_fired = set(info.get("open_channels_fired", []))
            dead_channels = set(info.get("dead_channels", []))
            artifact_channels_closed = set(info.get("artifact_channels_closed", []))
            artifact_channels_open = set(info.get("artifact_channels_open", []))
        artifact_channels = artifact_channels_closed | artifact_channels_open

        # Plot all EMG channels stacked with vertical offset
        yticks = []
        ytick_labels = []
        for ch in range(n_channels):
            offset = (n_channels - ch) * offset_step  # CH1 at top, leave slot 0 for GT

            # Dead channels: gray dashed
            if ch in dead_channels:
                color, lw, alpha = '#999999', 0.6, 0.5
                linestyle = '--'
                suffix = " [DEAD]"
            # Artifact channels: red dashed (overrides onset coloring)
            elif ch in artifact_channels:
                color, lw, alpha = '#D32F2F', 0.8, 0.9
                linestyle = '--'
                # Build suffix combining onset + artifact info
                onset_suffix = ""
                if ch in closed_channels_fired and ch in open_channels_fired:
                    onset_suffix = " [C+O]"
                elif ch in closed_channels_fired:
                    onset_suffix = " [C]"
                elif ch in open_channels_fired:
                    onset_suffix = " [O]"
                suffix = f"{onset_suffix} [ART]"
            # Highlight channels detected by onset detection
            elif ch in closed_channels_fired and ch in open_channels_fired:
                color, lw, alpha = '#9C27B0', 0.8, 0.95  # Purple = both
                linestyle = '-'
                suffix = " [C+O]"
            elif ch in closed_channels_fired:
                color, lw, alpha = '#FF5722', 0.8, 0.95  # Orange = CLOSED flexor
                linestyle = '-'
                suffix = " [C]"
            elif ch in open_channels_fired:
                color, lw, alpha = '#2196F3', 0.8, 0.95  # Blue = OPEN extensor
                linestyle = '-'
                suffix = " [O]"
            else:
                color, lw, alpha = 'k', 0.5, 0.9
                linestyle = '-'
                suffix = ""

            self.ax.plot(time_axis, emg[ch, :] + offset, color=color, linewidth=lw,
                         alpha=alpha, linestyle=linestyle)
            yticks.append(offset)
            ytick_labels.append(f"CH{ch + 1}{suffix}")

        # Plot GT as bottom trace (slot below all channels)
        gt_scaled = gt * offset_step * 0.8
        self.ax.plot(time_axis, gt_scaled, 'r-', linewidth=1.5, alpha=0.7, label='GT')
        yticks.append(0)
        ytick_labels.append("GT")

        # Add OPEN/CLOSED text labels on constant GT regions
        # GT=0 → OPEN, GT=1 → CLOSED (regardless of protocol mode)
        gt_binary = (gt > 0.5).astype(int)
        gt_diff = np.diff(gt_binary, prepend=gt_binary[0])
        # Find constant regions by detecting transitions
        transitions = np.where(gt_diff != 0)[0]
        region_starts = np.concatenate([[0], transitions])
        region_ends = np.concatenate([transitions, [n_samples]])
        gt_label_y = offset_step * 0.9  # Just above GT trace max
        for rs, re in zip(region_starts, region_ends):
            if re - rs < config.FSAMP * 0.3:
                continue  # Skip very short regions (transitions)
            mid_time = (rs + re) / 2 / config.FSAMP
            region_val = gt_binary[min(rs + (re - rs) // 2, n_samples - 1)]
            state_label = "CLOSED" if region_val == 1 else "OPEN"
            state_color = '#FF5722' if region_val == 1 else '#2196F3'
            self.ax.text(mid_time, gt_label_y, state_label,
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color=state_color, alpha=0.8)

        self.ax.set_yticks(yticks)
        self.ax.set_yticklabels(ytick_labels, fontsize=7)

        # Mark audio cue times with distinct colors
        if close_cue_idx is not None:
            close_cue_time = close_cue_idx / config.FSAMP
            self.ax.axvline(close_cue_time, color='#FF5722', linestyle='--', linewidth=1.5,
                           alpha=0.7, label='Close cue')

        if open_cue_idx is not None:
            open_cue_time = open_cue_idx / config.FSAMP
            self.ax.axvline(open_cue_time, color='#2196F3', linestyle='--', linewidth=1.5,
                           alpha=0.7, label='Open cue')

        # Calculate window positions (use saved positions if available, otherwise defaults)
        max_start = n_samples - self.template_duration_samples

        # Detect protocol mode for this cycle
        protocol_mode = cycle.get('protocol_mode', 'standard')

        # CLOSED window position
        if self.current_cycle_idx in self._saved_closed_positions:
            closed_idx = self._saved_closed_positions[self.current_cycle_idx]
        else:
            # Default: use close_cue_idx or close_start_idx if available
            closed_idx = close_cue_idx if close_cue_idx is not None else close_start_idx
            if closed_idx is None or closed_idx == 0:
                if protocol_mode == "inverted":
                    closed_idx = 3 * n_samples // 4
                else:
                    closed_idx = n_samples // 4
        closed_idx = max(0, min(closed_idx, max_start))

        # OPEN window position
        if self.current_cycle_idx in self._saved_open_positions:
            open_idx = self._saved_open_positions[self.current_cycle_idx]
        else:
            # Default: use open_cue_idx or open_start_idx if available
            open_idx = open_cue_idx if open_cue_idx is not None else open_start_idx
            if open_idx is None or open_idx == 0:
                if protocol_mode == "inverted":
                    open_idx = n_samples // 4
                else:
                    open_idx = 3 * n_samples // 4
        open_idx = max(0, min(open_idx, max_start))

        # Create draggable windows at calculated positions
        self.closed_window = DraggableWindow(
            self.ax, closed_idx, self.template_duration_samples,
            '#FF5722', 'CLOSED', config.FSAMP
        )
        self.open_window = DraggableWindow(
            self.ax, open_idx, self.template_duration_samples,
            '#2196F3', 'OPEN', config.FSAMP
        )

        # Labels and formatting
        self.ax.set_xlabel('Time (s)')
        status = " [ACCEPTED]" if (has_closed and has_open) else (" [CLOSED]" if has_closed else (" [OPEN]" if has_open else ""))
        mode_indicator = " [INV]" if protocol_mode == "inverted" else ""
        # Use per-cycle source recording name (set when multiple recordings loaded)
        # Fall back to widget-level recording_name for single-recording use
        cycle_rec = self.cycles[self.current_cycle_idx].get('source_recording') if self.cycles else None
        _rec = cycle_rec or (self.recording_name if hasattr(self, 'recording_name') else None)
        rec_name = f'{_rec} - ' if _rec else ''

        # Onset detection status indicator
        onset_status = ""
        if self._onset_info and self.current_cycle_idx < len(self._onset_info):
            info = self._onset_info[self.current_cycle_idx]
            has_closed_onset = "closed_channels_fired" in info and info["closed_channels_fired"]
            has_open_onset = "open_channels_fired" in info and info["open_channels_fired"]
            if has_closed_onset and has_open_onset:
                onset_status = " [ONSET: C+O]"
            elif has_closed_onset:
                onset_status = " [ONSET: C only]"
            elif has_open_onset:
                onset_status = " [ONSET: O only]"
            else:
                onset_status = " [NO ONSET]"

        # Signal quality indicator (dead + artifact channels)
        quality_status = ""
        if dead_channels or artifact_channels:
            parts = []
            if dead_channels:
                parts.append("DEAD:" + ",".join(f"CH{ch+1}" for ch in sorted(dead_channels)))
            if artifact_channels:
                parts.append("ART:" + ",".join(f"CH{ch+1}" for ch in sorted(artifact_channels)))
            quality_status = " [" + " ".join(parts) + "]"

        self.ax.set_title(f'{rec_name}Cycle {self.current_cycle_idx + 1}{mode_indicator}{onset_status}{quality_status}{status}: Drag windows to select templates')
        self.ax.set_xlim(0, n_samples / config.FSAMP)

        # Set y-axis limits to encompass all stacked channels + GT
        y_min = -offset_step * 0.5
        y_max = (n_channels + 1) * offset_step
        self.ax.set_ylim(y_min, y_max)

        self.ax.legend(loc='upper left', fontsize=7, framealpha=0.8)
        self.ax.grid(True, alpha=0.3, axis='x')

        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.canvas.update()

        # Enable accept button
        self.accept_button.setEnabled(True)

        # Update info label
        self._update_info_label()

    def _update_info_label(self):
        """Update the info label with selection status."""
        parts = []
        if self.closed_window is not None:
            start_s = self.closed_window.start_sample / config.FSAMP
            end_s = (self.closed_window.start_sample + self.template_duration_samples) / config.FSAMP
            parts.append(f"CLOSED: {start_s:.2f}s - {end_s:.2f}s")
        if self.open_window is not None:
            start_s = self.open_window.start_sample / config.FSAMP
            end_s = (self.open_window.start_sample + self.template_duration_samples) / config.FSAMP
            parts.append(f"OPEN: {start_s:.2f}s - {end_s:.2f}s")

        if parts:
            self.info_label.setText("Selected: " + " | ".join(parts))
        else:
            self.info_label.setText("Drag windows to select template regions")

    def _on_mouse_press(self, event):
        """Handle mouse press for dragging."""
        # Grab focus for keyboard events
        self.setFocus()
        if event.inaxes != self.ax or event.xdata is None:
            return

        click_time = event.xdata

        # Check if clicking on a window (check closed first, then open)
        if self.closed_window and self.closed_window.contains(click_time):
            self._dragging_window = self.closed_window
            start_time = self.closed_window.start_sample / config.FSAMP
            self._drag_offset = click_time - start_time
        elif self.open_window and self.open_window.contains(click_time):
            self._dragging_window = self.open_window
            start_time = self.open_window.start_sample / config.FSAMP
            self._drag_offset = click_time - start_time

    def _on_mouse_release(self, event):
        """Handle mouse release."""
        if self._dragging_window is not None:
            self._update_info_label()
        self._dragging_window = None

    def _on_mouse_move(self, event):
        """Handle mouse move for dragging."""
        if self._dragging_window is None or event.xdata is None:
            return

        if self.current_cycle_idx >= len(self.cycles):
            return

        cycle = self.cycles[self.current_cycle_idx]
        n_samples = cycle['emg'].shape[1]
        max_start_sample = n_samples - self.template_duration_samples

        # Calculate new position in time, then convert to samples
        new_start_time = event.xdata - self._drag_offset
        new_start_sample = int(new_start_time * config.FSAMP)

        # Clamp to valid range
        new_start_sample = max(0, min(new_start_sample, max_start_sample))

        self._dragging_window.set_position(new_start_sample)
        self.canvas.draw()

    def _go_prev(self):
        if self.current_cycle_idx > 0:
            self._save_window_positions()  # Preserve positions when navigating
            self.current_cycle_idx -= 1
            self._update_display()

    def _go_next(self):
        if self.current_cycle_idx < len(self.cycles) - 1:
            self._save_window_positions()  # Preserve positions when navigating
            self.current_cycle_idx += 1
            self._update_display()

    def _accept_templates(self):
        """Accept both selected templates (overwrites if cycle already accepted)."""
        if self.closed_window is None or self.open_window is None:
            return
        if self.current_cycle_idx >= len(self.cycles):
            return

        cycle = self.cycles[self.current_cycle_idx]
        emg = cycle['emg']
        cycle_idx = self.current_cycle_idx

        # Check if this is an overwrite
        is_overwrite = cycle_idx in self.accepted_closed_templates

        # Extract CLOSED template - store by cycle index (overwrites if exists)
        closed_start, closed_end = self.closed_window.get_sample_range()
        closed_end = min(closed_end, emg.shape[1])
        if closed_end > closed_start:
            closed_template = emg[:, closed_start:closed_end]
            self.accepted_closed_templates[cycle_idx] = closed_template

        # Extract OPEN template - store by cycle index (overwrites if exists)
        open_start, open_end = self.open_window.get_sample_range()
        open_end = min(open_end, emg.shape[1])
        if open_end > open_start:
            open_template = emg[:, open_start:open_end]
            self.accepted_open_templates[cycle_idx] = open_template

        # Emit signal
        if closed_end > closed_start and open_end > open_start:
            self.templates_accepted.emit(
                self.accepted_closed_templates[cycle_idx],
                self.accepted_open_templates[cycle_idx]
            )

        # Update display to show accepted status
        action = "Updated" if is_overwrite else "Accepted"
        self.info_label.setText(f"{action} templates from cycle {cycle_idx + 1}")
        self._update_display()

        # Auto-advance to next unaccepted cycle if available
        if not is_overwrite:
            for next_idx in range(self.current_cycle_idx + 1, len(self.cycles)):
                if next_idx not in self.accepted_closed_templates:
                    self.current_cycle_idx = next_idx
                    self._update_display()
                    return
            # All cycles reviewed
            if len(self.accepted_closed_templates) == len(self.cycles):
                self.info_label.setText("All cycles accepted! Click 'Save All' when ready.")

    def _accept_closed_only(self):
        """Accept only the CLOSED template from current cycle."""
        if self.closed_window is None:
            return
        if self.current_cycle_idx >= len(self.cycles):
            return

        cycle = self.cycles[self.current_cycle_idx]
        emg = cycle['emg']
        cycle_idx = self.current_cycle_idx

        is_overwrite = cycle_idx in self.accepted_closed_templates

        # Extract CLOSED template only
        closed_start, closed_end = self.closed_window.get_sample_range()
        closed_end = min(closed_end, emg.shape[1])
        if closed_end > closed_start:
            closed_template = emg[:, closed_start:closed_end]
            self.accepted_closed_templates[cycle_idx] = closed_template

        action = "Updated" if is_overwrite else "Accepted"
        self.info_label.setText(f"{action} CLOSED template from cycle {cycle_idx + 1}")
        self._update_display()

        # Auto-advance
        if not is_overwrite:
            self._auto_advance()

    def _accept_open_only(self):
        """Accept only the OPEN template from current cycle."""
        if self.open_window is None:
            return
        if self.current_cycle_idx >= len(self.cycles):
            return

        cycle = self.cycles[self.current_cycle_idx]
        emg = cycle['emg']
        cycle_idx = self.current_cycle_idx

        is_overwrite = cycle_idx in self.accepted_open_templates

        # Extract OPEN template only
        open_start, open_end = self.open_window.get_sample_range()
        open_end = min(open_end, emg.shape[1])
        if open_end > open_start:
            open_template = emg[:, open_start:open_end]
            self.accepted_open_templates[cycle_idx] = open_template

        action = "Updated" if is_overwrite else "Accepted"
        self.info_label.setText(f"{action} OPEN template from cycle {cycle_idx + 1}")
        self._update_display()

        # Auto-advance
        if not is_overwrite:
            self._auto_advance()

    def _auto_advance(self):
        """Auto-advance to next cycle that doesn't have both templates accepted."""
        for next_idx in range(self.current_cycle_idx + 1, len(self.cycles)):
            # Advance if either template is missing
            has_closed = next_idx in self.accepted_closed_templates
            has_open = next_idx in self.accepted_open_templates
            if not (has_closed and has_open):
                self.current_cycle_idx = next_idx
                self._update_display()
                return

    def get_accepted_templates(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return (closed_templates, open_templates) as lists."""
        # Convert dicts to lists (sorted by cycle index for consistency)
        closed_list = [self.accepted_closed_templates[k] for k in sorted(self.accepted_closed_templates.keys())]
        open_list = [self.accepted_open_templates[k] for k in sorted(self.accepted_open_templates.keys())]
        return closed_list, open_list


class GuidedRecordingReviewDialog(QDialog):
    """
    Dialog for reviewing and extracting templates from guided recordings.
    Shows complete cycles one at a time for unified template selection.
    """

    def __init__(self, recordings: List[dict], template_manager: TemplateManager,
                 parent=None, onset_detection: bool = False, onset_info: list = None,
                 onset_method: str = "amplitude"):
        super().__init__(parent)
        if isinstance(recordings, dict):
            self.recordings = [recordings]
        else:
            self.recordings = recordings
        self.template_manager = template_manager
        self.saved = False
        self._onset_detection = onset_detection
        self._onset_info = onset_info  # list of dicts with channels_fired and positions per cycle
        self._onset_method = onset_method  # "amplitude" or "tkeo"

        n_recordings = len(self.recordings)
        title_suffix = " (Onset Detection)" if onset_detection else ""
        self.setWindowTitle(f"Review & Extract Templates ({n_recordings} recording{'s' if n_recordings > 1 else ''}){title_suffix}")
        self.setMinimumSize(1400, 800)
        self.resize(1500, 850)  # Start with a larger size
        self.setModal(True)

        self._setup_ui()
        self._extract_and_populate()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Review & Extract Templates - Cycle View")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        info = QLabel(
            "Each cycle shows: HOLD OPEN → CLOSING → HOLD CLOSED → OPENING → OPEN\n"
            "Select a 1-second CLOSED template, then a 1-second OPEN template for each cycle."
        )
        info.setStyleSheet("color: #666;")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)

        # Cycle viewer
        self.cycle_viewer = CycleReviewWidget()
        self.cycle_viewer.templates_accepted.connect(self._on_templates_accepted)
        layout.addWidget(self.cycle_viewer, stretch=1)

        # Onset controls (only shown in onset detection mode)
        if self._onset_detection:
            onset_ctrl_layout = QHBoxLayout()
            method_label_text = {
                "amplitude": "Amplitude Threshold (mean + k·std)",
                "tkeo": "TKEO (|d/dt| energy)",
            }.get(self._onset_method, self._onset_method)
            onset_ctrl_layout.addWidget(QLabel(f"Method: {method_label_text}"))
            rerun_btn = QPushButton("Re-run Detection")
            rerun_btn.clicked.connect(self._rerun_onset_detection)
            onset_ctrl_layout.addWidget(rerun_btn)
            onset_ctrl_layout.addStretch()
            layout.addLayout(onset_ctrl_layout)

        # Status and buttons
        status_layout = QHBoxLayout()

        self.status_label = QLabel("Ready to save: 0 CLOSED, 0 OPEN templates")
        self.status_label.setStyleSheet("color: #666;")
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        self.save_button = QPushButton("Save All Accepted Templates")
        self.save_button.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px 20px; }"
        )
        self.save_button.clicked.connect(self._save_templates)
        status_layout.addWidget(self.save_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        status_layout.addWidget(cancel_button)

        layout.addLayout(status_layout)

    def _extract_and_populate(self):
        """Extract complete cycles from all recordings."""
        n_recordings = len(self.recordings)
        print(f"\n[REVIEW] Extracting complete cycles from {n_recordings} recording(s)...")

        all_cycles = []

        for i, recording in enumerate(self.recordings):
            print(f"  Processing recording {i+1}/{n_recordings}...")
            cycles = self.template_manager.extract_complete_cycles(
                recording,
                pre_close_s=1.0,  # 1 second before closing starts
                post_open_s=2.5   # 2.5 seconds after opening to include HOLD_OPEN_END
            )
            # Tag each cycle with its source recording name for display in the viewer
            rec_label = recording.get('label', f'Recording {i+1}')
            for c in cycles:
                c['source_recording'] = rec_label
            all_cycles.extend(cycles)

        self.cycle_viewer.set_cycles(all_cycles)
        print(f"[REVIEW] Total: {len(all_cycles)} complete cycles")

        # Pre-set window positions and channel info from onset detection
        if self._onset_detection and all_cycles:
            if self._onset_info:
                # Use pre-computed onset info from automatic extraction
                self._apply_onset_info(all_cycles)
            else:
                # Run onset detection now (for manual mode with onset)
                self._run_onset_detection(all_cycles)

        self._update_status()

    def _rerun_onset_detection(self):
        """Re-run onset detection with the current method (set by template type selection)."""
        cycles = self.cycle_viewer.cycles
        if cycles:
            self._run_onset_detection(cycles)

    def _apply_onset_info(self, cycles: list):
        """Apply pre-computed onset detection results to cycle viewer.

        onset_info positions are stored as offsets from the GT transition index
        (closed_pos = window_start - cue_closed, open_pos = window_start - cue_open).
        Re-apply using the viewer cycle's own GT transition indices so the positions
        are correct regardless of which pre_close_s was used during extraction.
        """
        template_samples = int(self.template_manager.template_duration_s * config.FSAMP)

        for idx in range(min(len(cycles), len(self._onset_info))):
            info = self._onset_info[idx]
            cycle = cycles[idx]
            n_samples = cycle["emg"].shape[1]

            # Viewer cycle GT transition indices
            viewer_cue_closed = (cycle.get("close_cue_idx") or
                                 cycle.get("close_start_idx", 0))
            viewer_cue_open   = (cycle.get("open_cue_idx") or
                                 cycle.get("open_start_idx", 0))

            if info["closed_pos"] is not None:
                pos = viewer_cue_closed + info["closed_pos"]
                pos = max(0, min(pos, n_samples - template_samples))
                self.cycle_viewer._saved_closed_positions[idx] = pos
            if info["open_pos"] is not None:
                pos = viewer_cue_open + info["open_pos"]
                pos = max(0, min(pos, n_samples - template_samples))
                self.cycle_viewer._saved_open_positions[idx] = pos

        # Store channel info for highlighting
        self.cycle_viewer._onset_info = self._onset_info
        self.cycle_viewer._update_display()

    def _run_onset_detection(self, cycles: list):
        """Run onset detection (amplitude threshold or TKEO) and pre-set window positions."""
        from mindmove.model.template_study import (
            detect_onset_per_channel,
            detect_transition_onset,
            place_template_at_onset,
            ONSET_ANTICIPATORY_S,
            ONSET_MAX_POST_CUE_S,
            ONSET_BASELINE_DURATION_S,
        )

        method = getattr(self, "_onset_method", "amplitude")
        print(f"\n[ONSET] Running detection method: {method}")

        template_samples = int(self.template_manager.template_duration_s * config.FSAMP)
        anticipatory_samples = int(ONSET_ANTICIPATORY_S * config.FSAMP)
        post_cue_samples = int(ONSET_MAX_POST_CUE_S * config.FSAMP)
        baseline_samples = int(ONSET_BASELINE_DURATION_S * config.FSAMP)

        n_closed_detected = 0
        n_open_detected = 0
        n_total = len(cycles)
        onset_info = []

        for idx, cycle in enumerate(cycles):
            emg = cycle["emg"]
            n_samples = emg.shape[1]

            cue_closed = cycle.get("close_cue_idx") or cycle.get("close_start_idx", 0)
            cue_open   = cycle.get("open_cue_idx")  or cycle.get("open_start_idx", 0)

            if method == "tkeo":
                # TKEO: narrow window around cue so argmax cannot reach opposite transition
                closed_search_start = max(0, cue_closed - anticipatory_samples)
                closed_search_end   = min(cue_closed + post_cue_samples,
                                          (cycle.get("open_cue_idx") or
                                           cycle.get("open_start_idx", n_samples))
                                          - template_samples)
                closed_search_end   = max(closed_search_start + 1, closed_search_end)
                open_search_start   = max(0, cue_open - anticipatory_samples)
                open_search_end     = min(cue_open + post_cue_samples,
                                          n_samples - template_samples)
                open_search_end     = max(open_search_start + 1, open_search_end)
                closed_baseline_start = None  # not used by TKEO
                open_baseline_start   = None
            else:
                # Amplitude threshold: extend search backward to catch anticipatory movements.
                # Baseline is placed in the quiet rest BEFORE the search window.
                closed_search_start = max(0, cue_closed - anticipatory_samples)
                closed_search_end   = max(closed_search_start + 1, cue_open - template_samples)
                open_search_start   = max(0, cue_open - anticipatory_samples)
                open_search_end     = max(open_search_start + 1, n_samples - template_samples)
                closed_baseline_start = max(0, closed_search_start - baseline_samples)
                open_baseline_start   = max(0, open_search_start - baseline_samples)

            cycle_info = {
                "closed_channels_fired": [],
                "open_channels_fired": [],
                "closed_pos": None,
                "open_pos": None,
            }

            # CLOSED onset
            if method == "tkeo":
                result = detect_transition_onset(emg, closed_search_start, closed_search_end)
            else:
                result = detect_onset_per_channel(emg, closed_search_start, closed_search_end,
                                                  baseline_start=closed_baseline_start)
            if result["earliest_onset"] is not None:
                pos = place_template_at_onset(result["earliest_onset"], n_samples)
                pos = max(pos, closed_search_start)
                pos = min(pos, n_samples - template_samples)
                self.cycle_viewer._saved_closed_positions[idx] = pos
                cycle_info["closed_channels_fired"] = result["channels_fired"]
                cycle_info["closed_pos"] = pos
                n_closed_detected += 1
                fired = ",".join(str(ch+1) for ch in result["channels_fired"])
                print(f"  Cycle {idx+1} CLOSED: onset={result['earliest_onset']/config.FSAMP:.2f}s, "
                      f"channels=[{fired}]")
            else:
                print(f"  Cycle {idx+1} CLOSED: no onset detected (manual placement needed)")

            # OPEN onset
            if method == "tkeo":
                result = detect_transition_onset(emg, open_search_start, open_search_end)
            else:
                result = detect_onset_per_channel(emg, open_search_start, open_search_end,
                                                  baseline_start=open_baseline_start)
            if result["earliest_onset"] is not None:
                pos = place_template_at_onset(result["earliest_onset"], n_samples)
                pos = max(pos, open_search_start)
                pos = min(pos, n_samples - template_samples)
                self.cycle_viewer._saved_open_positions[idx] = pos
                cycle_info["open_channels_fired"] = result["channels_fired"]
                cycle_info["open_pos"] = pos
                n_open_detected += 1
                fired = ",".join(str(ch+1) for ch in result["channels_fired"])
                print(f"  Cycle {idx+1} OPEN:   onset={result['earliest_onset']/config.FSAMP:.2f}s, "
                      f"channels=[{fired}]")
            else:
                print(f"  Cycle {idx+1} OPEN:   no onset detected (manual placement needed)")

            onset_info.append(cycle_info)

        print(f"\n[ONSET] Detected: {n_closed_detected}/{n_total} CLOSED, "
              f"{n_open_detected}/{n_total} OPEN onsets")
        if n_closed_detected < n_total or n_open_detected < n_total:
            print(f"[ONSET] Cycles without detection will use default positions — adjust manually")

        # Store channel info for highlighting in review
        self.cycle_viewer._onset_info = onset_info
        self.cycle_viewer._update_display()

    def _on_templates_accepted(self, closed_template, open_template):
        """Called when templates are accepted from a cycle."""
        self._update_status()

    def _update_status(self):
        closed, opened = self.cycle_viewer.get_accepted_templates()
        self.status_label.setText(f"Ready to save: {len(closed)} CLOSED, {len(opened)} OPEN templates")

    def _save_templates(self):
        closed_templates, open_templates = self.cycle_viewer.get_accepted_templates()

        if not closed_templates and not open_templates:
            QMessageBox.warning(
                self, "No Templates",
                "No templates have been accepted. Please review cycles and accept templates before saving."
            )
            return

        # Get label from first recording
        label = self.recordings[0].get("label", "") if self.recordings else ""

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        # Infer mode from actual template data, not current config
        all_templates = closed_templates + open_templates
        is_differential = _infer_mode_from_templates(all_templates)
        mode_suffix = "sd" if is_differential else "mp"

        # Save as combined file (both OPEN and CLOSED together)
        templates_base_dir = "data/templates"
        os.makedirs(templates_base_dir, exist_ok=True)

        if label:
            filename = f"templates_{mode_suffix}_{timestamp}_{label}.pkl"
        else:
            filename = f"templates_{mode_suffix}_{timestamp}.pkl"

        save_path = os.path.join(templates_base_dir, filename)

        # Combined data structure
        combined_data = {
            "templates_open": open_templates,
            "templates_closed": closed_templates,
            "metadata_open": self.template_manager.template_metadata.get("open", []),
            "metadata_closed": self.template_manager.template_metadata.get("closed", []),
            "metadata": {
                "n_open": len(open_templates),
                "n_closed": len(closed_templates),
                "template_duration_s": self.template_manager.template_duration_s,
                "created_at": now.isoformat(),
                "label": label,
                "differential_mode": is_differential,
            }
        }

        with open(save_path, "wb") as f:
            pickle.dump(combined_data, f)

        print(f"[REVIEW] Saved combined templates to {save_path}")
        print(f"  OPEN: {len(open_templates)} templates")
        print(f"  CLOSED: {len(closed_templates)} templates")

        # Also update the template manager so templates are available for model creation
        self.template_manager.templates["open"] = open_templates
        self.template_manager.templates["closed"] = closed_templates

        self.saved = True

        msg = f"Templates saved successfully!\n\n"
        msg += f"CLOSED templates: {len(closed_templates)}\n"
        msg += f"OPEN templates: {len(open_templates)}\n\n"
        msg += f"Saved to:\n{save_path}"

        QMessageBox.information(self, "Templates Saved", msg)
        self.accept()


class TemplateReviewDialog(QDialog):
    """
    Dialog for visualizing extracted templates.

    Shows a side-by-side view of CLOSED and OPEN templates with:
    - List of templates for each class
    - Channel selector
    - Waveform plot of selected template
    """

    def __init__(self, templates_closed: List[np.ndarray], templates_open: List[np.ndarray], parent=None):
        super().__init__(parent)
        self.templates_closed = templates_closed
        self.templates_open = templates_open

        self.setWindowTitle("Template Review")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 850)
        self.setModal(False)  # Non-modal so user can continue working

        self._setup_ui()
        self._update_plots()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel(f"Extracted Templates: {len(self.templates_closed)} CLOSED, {len(self.templates_open)} OPEN")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Main splitter for CLOSED (left) and OPEN (right)
        splitter = QSplitter(Qt.Horizontal)

        # Compute global offset for stacked view from all templates
        all_templates = self.templates_closed + self.templates_open
        if all_templates:
            all_ranges = []
            for t in all_templates:
                for ch in range(t.shape[0]):
                    all_ranges.append(np.ptp(t[ch, :]))
            self._offset_step = np.median(all_ranges) * 1.5 if np.median(all_ranges) > 0 else 1.0
        else:
            self._offset_step = 1.0

        # CLOSED panel
        closed_panel = QWidget()
        closed_layout = QVBoxLayout(closed_panel)

        closed_header = QLabel("CLOSED Templates")
        closed_header.setStyleSheet("font-weight: bold; color: #FF5722;")
        closed_layout.addWidget(closed_header)

        # Template list
        self.closed_list = QListWidget()
        self.closed_list.setMaximumHeight(150)
        for i in range(len(self.templates_closed)):
            self.closed_list.addItem(f"Template {i+1}")
        if self.templates_closed:
            self.closed_list.setCurrentRow(0)
        self.closed_list.currentRowChanged.connect(self._on_closed_selection_changed)
        closed_layout.addWidget(self.closed_list)

        # Plot canvas
        self.closed_figure = Figure(figsize=(5, 6), dpi=100)
        self.closed_canvas = FigureCanvas(self.closed_figure)
        self.closed_ax = self.closed_figure.add_subplot(111)
        closed_layout.addWidget(self.closed_canvas)

        splitter.addWidget(closed_panel)

        # OPEN panel
        open_panel = QWidget()
        open_layout = QVBoxLayout(open_panel)

        open_header = QLabel("OPEN Templates")
        open_header.setStyleSheet("font-weight: bold; color: #2196F3;")
        open_layout.addWidget(open_header)

        # Template list
        self.open_list = QListWidget()
        self.open_list.setMaximumHeight(150)
        for i in range(len(self.templates_open)):
            self.open_list.addItem(f"Template {i+1}")
        if self.templates_open:
            self.open_list.setCurrentRow(0)
        self.open_list.currentRowChanged.connect(self._on_open_selection_changed)
        open_layout.addWidget(self.open_list)

        # Plot canvas
        self.open_figure = Figure(figsize=(5, 6), dpi=100)
        self.open_canvas = FigureCanvas(self.open_figure)
        self.open_ax = self.open_figure.add_subplot(111)
        open_layout.addWidget(self.open_canvas)

        splitter.addWidget(open_panel)

        layout.addWidget(splitter)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _on_closed_selection_changed(self, row: int):
        self._update_closed_plot()

    def _on_open_selection_changed(self, row: int):
        self._update_open_plot()

    def _update_plots(self):
        self._update_closed_plot()
        self._update_open_plot()

    def _plot_stacked_template(self, ax, figure, canvas, template, title, color):
        """Plot a template with all channels stacked vertically."""
        ax.clear()
        n_channels = template.shape[0]
        n_samples = template.shape[1]
        time_axis = np.arange(n_samples) / config.FSAMP

        yticks = []
        ytick_labels = []
        for ch in range(n_channels):
            offset = (n_channels - 1 - ch) * self._offset_step
            ax.plot(time_axis, template[ch, :] + offset, color=color, linewidth=0.8)
            yticks.append(offset)
            ytick_labels.append(f"CH{ch + 1}")

        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, fontsize=7)
        ax.set_xlabel('Time (s)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        figure.tight_layout()
        canvas.draw()

    def _update_closed_plot(self):
        self.closed_ax.clear()
        idx = self.closed_list.currentRow()
        if idx < 0 or idx >= len(self.templates_closed):
            self.closed_ax.text(0.5, 0.5, "No template selected", ha='center', va='center',
                               transform=self.closed_ax.transAxes)
            self.closed_canvas.draw()
            return

        template = self.templates_closed[idx]
        self._plot_stacked_template(
            self.closed_ax, self.closed_figure, self.closed_canvas,
            template, f'CLOSED Template {idx+1}', '#FF5722'
        )

    def _update_open_plot(self):
        self.open_ax.clear()
        idx = self.open_list.currentRow()
        if idx < 0 or idx >= len(self.templates_open):
            self.open_ax.text(0.5, 0.5, "No template selected", ha='center', va='center',
                             transform=self.open_ax.transAxes)
            self.open_canvas.draw()
            return

        template = self.templates_open[idx]
        self._plot_stacked_template(
            self.open_ax, self.open_figure, self.open_canvas,
            template, f'OPEN Template {idx+1}', '#2196F3'
        )


class TemplateStudyDialog(QDialog):
    """Dialog for analyzing template quality via DTW distance matrix and statistics."""

    def __init__(
        self,
        templates_closed: list,
        templates_open: list,
        default_feature: str = "rms",
        default_window_samples: int = 192,
        default_overlap_samples: int = 64,
        default_aggregation_idx: int = 0,
        default_dead_channels_text: str = "",
        metadata_closed: list = None,
        metadata_open: list = None,
        parent=None,
    ):
        super().__init__(parent)
        self.templates_closed = list(templates_closed)  # work on a copy
        self.templates_open = list(templates_open)
        self.metadata_closed = list(metadata_closed) if metadata_closed else []
        self.metadata_open = list(metadata_open) if metadata_open else []
        self.templates_modified = False
        # Cache for Tab 3 re-computation (populated after first _compute() call)
        self._cached_metrics = None
        self._cached_spatial_closed = None
        self._cached_spatial_open = None
        self.setWindowTitle("Template Study")
        # Make the dialog a proper resizable window with maximize/minimize buttons
        from PySide6.QtCore import Qt
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowTitleHint |
            Qt.WindowType.WindowSystemMenuHint |
            Qt.WindowType.WindowMinMaxButtonsHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.setSizeGripEnabled(True)
        # Start with a reasonable size but allow free resizing
        from PySide6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(1100, int(avail.width() * 0.85))
            h = min(850, int(avail.height() * 0.85))
            self.resize(w, h)
            # Center on screen
            self.move(avail.center().x() - w // 2, avail.center().y() - h // 2)
        else:
            self.resize(1000, 750)

        layout = QVBoxLayout(self)

        # ── Parameter section ──
        from mindmove.model.core.features.features_registry import FEATURES
        param_layout = QHBoxLayout()

        param_layout.addWidget(QLabel("Feature:"))
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(list(FEATURES.keys()))
        self.feature_combo.setCurrentText(default_feature)
        param_layout.addWidget(self.feature_combo)

        param_layout.addWidget(QLabel("Window (samp):"))
        self.window_spinbox = QSpinBox()
        self.window_spinbox.setRange(1, 2000)
        self.window_spinbox.setValue(default_window_samples)
        param_layout.addWidget(self.window_spinbox)

        param_layout.addWidget(QLabel("Overlap (samp):"))
        self.overlap_spinbox = QSpinBox()
        self.overlap_spinbox.setRange(0, 1999)
        self.overlap_spinbox.setValue(default_overlap_samples)
        param_layout.addWidget(self.overlap_spinbox)

        param_layout.addWidget(QLabel("Aggregation:"))
        self.agg_combo = QComboBox()
        self.agg_combo.addItems([
            "Average of 3 smallest",
            "Minimum distance",
            "Average of all",
        ])
        self.agg_combo.setCurrentIndex(default_aggregation_idx)
        param_layout.addWidget(self.agg_combo)

        param_layout.addWidget(QLabel("Dead Ch:"))
        self.dead_ch_input = QLineEdit()
        self.dead_ch_input.setPlaceholderText("e.g. 9,22")
        self.dead_ch_input.setText(default_dead_channels_text)
        self.dead_ch_input.setFixedWidth(80)
        param_layout.addWidget(self.dead_ch_input)

        self.compute_btn = QPushButton("Compute")
        self.compute_btn.clicked.connect(self._compute)
        param_layout.addWidget(self.compute_btn)

        self.compare_btn = QPushButton("Compare All Features")
        self.compare_btn.clicked.connect(self._compare_features)
        param_layout.addWidget(self.compare_btn)

        self.load_btn = QPushButton("Load from File")
        self.load_btn.clicked.connect(self._load_templates_from_file)
        param_layout.addWidget(self.load_btn)

        layout.addLayout(param_layout)

        # ── Results section: tab widget with DTW and Spatial pages ──
        from PySide6.QtWidgets import QTabWidget
        self.results_tabs = QTabWidget()

        # Tab 1: DTW Analysis
        tab1_widget = QWidget()
        tab1_layout = QVBoxLayout(tab1_widget)
        tab1_layout.setContentsMargins(0, 0, 0, 0)
        self.figure_dtw = Figure(figsize=(8, 6), dpi=100)
        self.canvas_dtw = FigureCanvas(self.figure_dtw)
        self.canvas_dtw.setMinimumHeight(200)
        tab1_layout.addWidget(self.canvas_dtw)
        self.results_tabs.addTab(tab1_widget, "DTW Analysis")

        # Tab 2: Spatial Analysis
        tab2_widget = QWidget()
        tab2_layout = QVBoxLayout(tab2_widget)
        tab2_layout.setContentsMargins(0, 0, 0, 0)
        self.figure_spatial = Figure(figsize=(8, 6), dpi=100)
        self.canvas_spatial = FigureCanvas(self.figure_spatial)
        self.canvas_spatial.setMinimumHeight(200)
        tab2_layout.addWidget(self.canvas_spatial)
        self.spatial_stats_label = QLabel("Run 'Compute' to see spatial analysis.")
        self.spatial_stats_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.spatial_stats_label.setWordWrap(True)
        tab2_layout.addWidget(self.spatial_stats_label)
        self.results_tabs.addTab(tab2_widget, "Spatial Analysis")

        # Tab 3: Coupled Correction
        tab3_widget = QWidget()
        tab3_layout = QVBoxLayout(tab3_widget)
        tab3_layout.setContentsMargins(4, 4, 4, 4)

        # Parameter controls row
        tab3_param_row = QHBoxLayout()
        tab3_param_row.addWidget(QLabel("Mode:"))
        self.coupled_mode_combo = QComboBox()
        self.coupled_mode_combo.addItem("Distance Scaling", "scaling")
        self.coupled_mode_combo.addItem("Gate", "gate")
        self.coupled_mode_combo.addItem("ReLU Scaling", "relu_scaling")
        self.coupled_mode_combo.addItem("ReLU Scaling (ext)", "relu_ext_scaling")
        tab3_param_row.addWidget(self.coupled_mode_combo)

        tab3_param_row.addWidget(QLabel("k:"))
        self.coupled_k_spinbox = QDoubleSpinBox()
        self.coupled_k_spinbox.setRange(0.5, 10.0)
        self.coupled_k_spinbox.setSingleStep(0.5)
        self.coupled_k_spinbox.setValue(3.0)
        self.coupled_k_spinbox.setDecimals(1)
        self.coupled_k_spinbox.setFixedWidth(65)
        tab3_param_row.addWidget(self.coupled_k_spinbox)

        tab3_param_row.addWidget(QLabel("Threshold:"))
        self.coupled_threshold_spinbox = QDoubleSpinBox()
        self.coupled_threshold_spinbox.setRange(0.0, 1.0)
        self.coupled_threshold_spinbox.setSingleStep(0.05)
        self.coupled_threshold_spinbox.setValue(0.50)
        self.coupled_threshold_spinbox.setDecimals(2)
        self.coupled_threshold_spinbox.setFixedWidth(65)
        tab3_param_row.addWidget(self.coupled_threshold_spinbox)

        tab3_param_row.addWidget(QLabel("Baseline:"))
        self.coupled_baseline_spinbox = QDoubleSpinBox()
        self.coupled_baseline_spinbox.setRange(0.01, 0.99)
        self.coupled_baseline_spinbox.setSingleStep(0.05)
        self.coupled_baseline_spinbox.setValue(0.30)
        self.coupled_baseline_spinbox.setDecimals(2)
        self.coupled_baseline_spinbox.setFixedWidth(65)
        tab3_param_row.addWidget(self.coupled_baseline_spinbox)

        self.coupled_compute_btn = QPushButton("Compute Coupled")
        self.coupled_compute_btn.clicked.connect(self._compute_coupled_tab_from_ui)
        tab3_param_row.addWidget(self.coupled_compute_btn)
        tab3_param_row.addStretch()
        tab3_layout.addLayout(tab3_param_row)

        self.figure_coupled = Figure(figsize=(8, 6), dpi=100)
        self.canvas_coupled = FigureCanvas(self.figure_coupled)
        self.canvas_coupled.setMinimumHeight(200)
        tab3_layout.addWidget(self.canvas_coupled)

        self.coupled_stats_label = QLabel("Run 'Compute' then switch to this tab, or click 'Compute Coupled'.")
        self.coupled_stats_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.coupled_stats_label.setWordWrap(True)
        tab3_layout.addWidget(self.coupled_stats_label)

        self.results_tabs.addTab(tab3_widget, "Coupled Correction")

        layout.addWidget(self.results_tabs)

        n_c = len(self.templates_closed)
        n_o = len(self.templates_open)
        if n_c > 0 or n_o > 0:
            initial_text = f"Loaded {n_c} CLOSED + {n_o} OPEN templates. Click 'Compute' to analyze."
        else:
            initial_text = "No templates loaded. Use 'Load from File' to load a template set, then click 'Compute'."
        self.stats_label = QLabel(initial_text)
        self.stats_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

        # ── Flagged templates section (populated after Compute) ──
        self.flagged_container = QWidget()
        self.flagged_layout = QVBoxLayout(self.flagged_container)
        self.flagged_layout.setContentsMargins(0, 0, 0, 0)

        flagged_scroll = QScrollArea()
        flagged_scroll.setWidget(self.flagged_container)
        flagged_scroll.setWidgetResizable(True)
        flagged_scroll.setMinimumHeight(60)
        flagged_scroll.setMaximumHeight(220)
        layout.addWidget(flagged_scroll)

        # ── Template count label ──
        self.count_label = QLabel(f"Templates: {n_c} CLOSED + {n_o} OPEN")
        self.count_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.count_label)

    def _parse_dead_channels(self) -> list:
        text = self.dead_ch_input.text().strip()
        if not text:
            return []
        dead = []
        for part in text.split(","):
            part = part.strip()
            if part.isdigit():
                ch = int(part)
                if 1 <= ch <= 32:
                    dead.append(ch - 1)
        return sorted(set(dead))

    def _load_templates_from_file(self):
        """Load templates from a .pkl file (combined template file or single-class files)."""
        import pickle
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        start_dir = "data/templates"
        if not os.path.exists(start_dir):
            start_dir = "data"

        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Templates File", start_dir, "Pickle files (*.pkl)"
        )
        if not filename:
            return

        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)

            templates_closed = []
            templates_open = []

            if isinstance(data, dict):
                # Combined format: templates_open + templates_closed
                if "templates_open" in data and "templates_closed" in data:
                    raw_open = data["templates_open"]
                    raw_closed = data["templates_closed"]
                    # Distinguish raw EMG (n_ch, n_samples) from feature matrices (n_windows, n_ch)
                    # Raw EMG: first dim is small (channels), second is large (samples)
                    # Feature: first dim is large (windows), second is small (channels)
                    if len(raw_open) > 0 and hasattr(raw_open[0], 'shape') and len(raw_open[0].shape) == 2:
                        if raw_open[0].shape[0] <= 64:  # likely raw EMG (n_ch, n_samples)
                            templates_open = raw_open
                        else:
                            QMessageBox.warning(
                                self, "Feature Templates",
                                "This file contains feature-extracted templates, not raw EMG.\n"
                                "Template Study needs raw EMG templates.",
                                QMessageBox.Ok,
                            )
                            return
                    else:
                        templates_open = raw_open
                    if len(raw_closed) > 0 and hasattr(raw_closed[0], 'shape') and len(raw_closed[0].shape) == 2:
                        if raw_closed[0].shape[0] <= 64:
                            templates_closed = raw_closed
                        else:
                            QMessageBox.warning(
                                self, "Feature Templates",
                                "This file contains feature-extracted templates, not raw EMG.\n"
                                "Template Study needs raw EMG templates.",
                                QMessageBox.Ok,
                            )
                            return
                    else:
                        templates_closed = raw_closed

                # Single-class format: templates + metadata
                elif "templates" in data and "metadata" in data:
                    class_label = data["metadata"].get("class_label", "").lower()
                    if "open" in class_label:
                        templates_open = data["templates"]
                    elif "closed" in class_label or "close" in class_label:
                        templates_closed = data["templates"]
                    else:
                        QMessageBox.warning(
                            self, "Unknown Class",
                            f"Could not determine class from metadata: '{class_label}'.\n"
                            "Expected 'open' or 'closed'.",
                            QMessageBox.Ok,
                        )
                        return
                else:
                    QMessageBox.warning(
                        self, "Invalid File",
                        "File format not recognized.\nExpected combined template file "
                        "(templates_open + templates_closed) or single-class file (templates + metadata).",
                        QMessageBox.Ok,
                    )
                    return
            else:
                QMessageBox.warning(
                    self, "Invalid File",
                    "File format not recognized (expected dict).",
                    QMessageBox.Ok,
                )
                return

            n_c = len(templates_closed)
            n_o = len(templates_open)
            if n_c == 0 and n_o == 0:
                QMessageBox.warning(self, "Empty", "No templates found in file.", QMessageBox.Ok)
                return

            # Infer channel count from templates
            sample = templates_closed[0] if templates_closed else templates_open[0]
            if hasattr(sample, 'shape') and len(sample.shape) == 2:
                n_ch = sample.shape[0]
                if n_ch != config.num_channels:
                    print(f"[TEMPLATE STUDY] Adjusting config.num_channels: {config.num_channels} -> {n_ch}")
                    config.num_channels = n_ch
                    config.active_channels = [i for i in range(n_ch) if i not in config.dead_channels]

            self.templates_closed = list(templates_closed)
            self.templates_open = list(templates_open)
            # Load provenance metadata if available in the file
            if isinstance(data, dict):
                if "templates_open" in data and "templates_closed" in data:
                    self.metadata_closed = list(data.get("metadata_closed", []))
                    self.metadata_open = list(data.get("metadata_open", []))
                elif "templates" in data and "metadata" in data:
                    class_label = data["metadata"].get("class_label", "").lower()
                    tmpl_meta = list(data.get("template_metadata", []))
                    if "open" in class_label:
                        self.metadata_open = tmpl_meta
                        self.metadata_closed = []
                    else:
                        self.metadata_closed = tmpl_meta
                        self.metadata_open = []
            self.templates_modified = True  # loaded new set
            basename = os.path.basename(filename)
            self.setWindowTitle(f"Template Study — {basename}")
            self.count_label.setText(f"Templates: {n_c} CLOSED + {n_o} OPEN")
            self.stats_label.setText(
                f"Loaded {n_c} CLOSED + {n_o} OPEN templates from:\n{basename}\n\n"
                f"Click 'Compute' to analyze."
            )
            print(f"[TEMPLATE STUDY] Loaded {n_c} CLOSED + {n_o} OPEN from {basename}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}", QMessageBox.Ok)
            import traceback
            traceback.print_exc()

    def _compute_metrics(self, feature_name=None, silent=False):
        """Core computation — returns (metrics, quality, params) or None on error."""
        from mindmove.model.template_study import (
            compute_template_metrics_with_aggregation,
            analyze_template_quality,
        )

        if feature_name is None:
            feature_name = self.feature_combo.currentText()
        window_samples = self.window_spinbox.value()
        overlap_samples = self.overlap_spinbox.value()
        increment_samples = max(1, window_samples - overlap_samples)

        agg_text = self.agg_combo.currentText()
        if "3 smallest" in agg_text:
            agg = "avg_3_smallest"
        elif "Minimum" in agg_text:
            agg = "minimum"
        else:
            agg = "average"

        dead = self._parse_dead_channels()
        n_ch = config.num_channels
        active_channels = [i for i in range(n_ch) if i not in dead]

        metrics = compute_template_metrics_with_aggregation(
            templates_closed=self.templates_closed,
            templates_open=self.templates_open,
            feature_name=feature_name,
            window_length=window_samples,
            window_increment=increment_samples,
            distance_aggregation=agg,
            active_channels=active_channels,
        )

        n_closed = len(self.templates_closed)
        n_open = len(self.templates_open)
        quality = analyze_template_quality(metrics, n_closed, n_open)

        params = {
            "feature_name": feature_name,
            "window_samples": window_samples,
            "overlap_samples": overlap_samples,
            "agg": agg,
            "dead": dead,
        }
        return metrics, quality, params

    def _compute(self):
        from mindmove.model.template_study import plot_distance_matrix
        from mindmove.model.core.algorithm import compute_spatial_profiles

        if not self.templates_closed and not self.templates_open:
            self.stats_label.setText("No templates loaded. Use 'Load from File' to load templates.")
            return

        self.compute_btn.setEnabled(False)
        self.compute_btn.setText("Computing...")
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            result = self._compute_metrics()
            if result is None:
                return
            metrics, quality, params = result

            # Compute spatial profiles (also stored for Tab 3 on-demand re-computation)
            spatial_closed = compute_spatial_profiles(self.templates_closed) if self.templates_closed else None
            spatial_open = compute_spatial_profiles(self.templates_open) if self.templates_open else None
            # Cache for Tab 3
            self._cached_metrics = metrics
            self._cached_spatial_closed = spatial_closed
            self._cached_spatial_open = spatial_open

            # Plot: DTW distance matrix (left) + spatial profiles (right) — Tab 1
            self.figure_dtw.clear()

            if spatial_closed is not None or spatial_open is not None:
                # Layout: DTW matrix (left) | two spatial subplots stacked (right)
                import matplotlib.gridspec as gridspec
                gs = gridspec.GridSpec(2, 2, figure=self.figure_dtw, width_ratios=[1.2, 1])
                ax_dtw = self.figure_dtw.add_subplot(gs[:, 0])
                ax_profile = self.figure_dtw.add_subplot(gs[0, 1])
                ax_weights = self.figure_dtw.add_subplot(gs[1, 1])

                n_ch = 0
                if spatial_closed is not None:
                    n_ch = len(spatial_closed["weights"])
                elif spatial_open is not None:
                    n_ch = len(spatial_open["weights"])

                channels = np.arange(n_ch)
                bar_width = 0.35
                xtick_labels = [str(c + 1) for c in channels]

                # --- Top: mean spatial pattern ---
                # Bars = max-normalized (interpretable: dominant channel = 1.0, with std)
                # Line = L2-normalized (actual ref_profile used in computation, rescaled to [0,1])
                if spatial_closed is not None:
                    ax_profile.bar(channels - bar_width / 2, spatial_closed["ref_profile_maxnorm"],
                                   bar_width, label="CLOSED (max-norm)", color="#d44", alpha=0.6,
                                   yerr=spatial_closed["per_template_rms_maxnorm"].std(axis=0),
                                   error_kw=dict(elinewidth=1.0, capsize=2, ecolor='#a00'))
                    l2_closed = spatial_closed["ref_profile"]
                    l2_closed_scaled = l2_closed / (l2_closed.max() + 1e-10)
                    ax_profile.plot(channels - bar_width / 2, l2_closed_scaled,
                                    color='#900', linewidth=1.5, marker='o', markersize=3,
                                    label="CLOSED (L2-norm, scaled)", zorder=5)
                if spatial_open is not None:
                    ax_profile.bar(channels + bar_width / 2, spatial_open["ref_profile_maxnorm"],
                                   bar_width, label="OPEN (max-norm)", color="#48b", alpha=0.6,
                                   yerr=spatial_open["per_template_rms_maxnorm"].std(axis=0),
                                   error_kw=dict(elinewidth=1.0, capsize=2, ecolor='#247'))
                    l2_open = spatial_open["ref_profile"]
                    l2_open_scaled = l2_open / (l2_open.max() + 1e-10)
                    ax_profile.plot(channels + bar_width / 2, l2_open_scaled,
                                    color='#136', linewidth=1.5, marker='o', markersize=3,
                                    label="OPEN (L2-norm, scaled)", zorder=5)
                ax_profile.set_ylabel("Relative RMS")
                ax_profile.set_title("Mean Spatial Pattern (bars=max-norm, line=L2-norm)")
                ax_profile.set_xticks(channels)
                ax_profile.set_xticklabels(xtick_labels, fontsize=7)
                ax_profile.set_ylim(0, 1.15)
                ax_profile.axhline(1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
                ax_profile.legend(fontsize=7)
                ax_profile.grid(axis='y', alpha=0.3)

                # --- Bottom: consistency weights ---
                if spatial_closed is not None:
                    ax_weights.bar(channels - bar_width / 2, spatial_closed["weights"],
                                   bar_width, label="CLOSED", color="#d44", alpha=0.8)
                if spatial_open is not None:
                    ax_weights.bar(channels + bar_width / 2, spatial_open["weights"],
                                   bar_width, label="OPEN", color="#48b", alpha=0.8)
                ax_weights.set_xlabel("Channel")
                ax_weights.set_ylabel("Weight")
                ax_weights.set_title("Consistency Weights (voting power)")
                ax_weights.set_xticks(channels)
                ax_weights.set_xticklabels(xtick_labels, fontsize=7)
                ax_weights.legend(fontsize=8)
                ax_weights.grid(axis='y', alpha=0.3)
            else:
                ax_dtw = self.figure_dtw.add_subplot(111)

            plot_distance_matrix(metrics, ax=ax_dtw)
            self.figure_dtw.tight_layout()
            self.canvas_dtw.draw()

            # ── Tab 2: Spatial Analysis ──
            self._compute_spatial_tab(metrics, spatial_closed, spatial_open)

            # ── Tab 3: Coupled Correction (auto-compute with current params) ──
            self._compute_coupled_tab(metrics, spatial_closed, spatial_open)

            # Build stats text
            ic = metrics["intra_closed"]
            io = metrics["intra_open"]
            ec = metrics["inter_closed_to_open"]
            eo = metrics["inter_open_to_closed"]

            ic_upper = ic["mean"] + ic["std"]
            io_upper = io["mean"] + io["std"]
            ec_lower = ec["mean"] - ec["std"]
            eo_lower = eo["mean"] - eo["std"]

            gap_closed = ec_lower - ic_upper
            gap_open = eo_lower - io_upper
            ok_c = "OK" if gap_closed > 0 else "OVERLAP"
            ok_o = "OK" if gap_open > 0 else "OVERLAP"

            midgap_closed = (ic_upper + ec_lower) / 2
            midgap_open = (io_upper + eo_lower) / 2

            text = (
                f"Intra CLOSED:  mean={ic['mean']:.4f}  std={ic['std']:.4f}  ->  mean+std={ic_upper:.4f}\n"
                f"Intra OPEN:    mean={io['mean']:.4f}  std={io['std']:.4f}  ->  mean+std={io_upper:.4f}\n"
                f"Inter (C->O):  mean={ec['mean']:.4f}  std={ec['std']:.4f}  ->  mean-std={ec_lower:.4f}\n"
                f"Inter (O->C):  mean={eo['mean']:.4f}  std={eo['std']:.4f}  ->  mean-std={eo_lower:.4f}\n"
                f"\n"
                f"Gap CLOSED: {gap_closed:.4f} [{ok_c}]   Mid-gap threshold: {midgap_closed:.4f}\n"
                f"Gap OPEN:   {gap_open:.4f} [{ok_o}]   Mid-gap threshold: {midgap_open:.4f}\n"
                f"\n"
                f"{quality['summary']}\n"
                f"\n"
                f"Aggregation: {params['agg']}  |  Feature: {params['feature_name']}  |  "
                f"Window: {params['window_samples']} samp  Overlap: {params['overlap_samples']} samp  |  "
                f"Dead channels: {[ch+1 for ch in params['dead']] if params['dead'] else 'None'}"
            )
            self.stats_label.setText(text)

            # Populate flagged templates with remove buttons
            self._populate_flagged_templates(quality)

        except Exception as e:
            self.stats_label.setText(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.compute_btn.setEnabled(True)
            self.compute_btn.setText("Compute")

    def _compute_spatial_tab(self, metrics, spatial_closed, spatial_open):
        """Compute and render Tab 2: Spatial Analysis (similarity matrix + scatter plot)."""
        import matplotlib.gridspec as gridspec

        self.figure_spatial.clear()

        n_closed = len(self.templates_closed)
        n_open = len(self.templates_open)
        n_total = n_closed + n_open

        # Need at least 2 templates and spatial profiles to be meaningful
        if n_total < 2 or (spatial_closed is None and spatial_open is None):
            ax = self.figure_spatial.add_subplot(111)
            ax.text(0.5, 0.5, "Spatial analysis requires templates with spatial profiles.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.axis("off")
            self.canvas_spatial.draw()
            self.spatial_stats_label.setText("Spatial analysis not available.")
            return

        # Stack per_template_rms: CLOSED first, then OPEN (L2-normalized unit vectors)
        rms_parts = []
        if spatial_closed is not None:
            rms_parts.append(spatial_closed["per_template_rms"])   # (n_closed, n_ch)
        if spatial_open is not None:
            rms_parts.append(spatial_open["per_template_rms"])     # (n_open, n_ch)
        rms_all = np.vstack(rms_parts)                             # (n_total, n_ch)

        # Pairwise cosine similarity matrix (unit vectors → dot product = cos sim)
        sim_matrix = np.clip(rms_all @ rms_all.T, 0.0, 1.0)       # (n_total, n_total)

        # Per-template labels with provenance (CLOSED first, then OPEN)
        def _meta_label(prefix, i, meta_list):
            # Use stored original id if available so labels are stable after removals
            if i < len(meta_list):
                m = meta_list[i]
                orig_id = m.get("id", i + 1)
                base = f"{prefix}{orig_id}"
                rec = str(m.get("recording", ""))
                cyc = m.get("cycle", "")
                # Abbreviate long recording names (keep last 8 chars)
                if len(rec) > 8:
                    rec = rec[-8:]
                if rec and cyc != "":
                    return f"{base}\n{rec}#{cyc}"
                elif rec:
                    return f"{base}\n{rec}"
                elif cyc != "":
                    return f"{base}\n#{cyc}"
                return base
            return f"{prefix}{i+1}"

        labels_sim = (
            [_meta_label("C", i, self.metadata_closed) for i in range(n_closed)] +
            [_meta_label("O", i, self.metadata_open) for i in range(n_open)]
        )

        gs2 = gridspec.GridSpec(1, 2, figure=self.figure_spatial, width_ratios=[1, 1.1])
        ax_sim = self.figure_spatial.add_subplot(gs2[0])
        ax_scatter = self.figure_spatial.add_subplot(gs2[1])

        # ── Similarity heatmap (mirrors DTW matrix layout) ──
        im = ax_sim.imshow(sim_matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        self.figure_spatial.colorbar(im, ax=ax_sim, label="Cosine Similarity", fraction=0.046)

        # Class boundary line between CLOSED and OPEN blocks
        if n_closed > 0 and n_open > 0:
            b = n_closed - 0.5
            ax_sim.axhline(b, color="white", linewidth=1.5, linestyle="--", alpha=0.8)
            ax_sim.axvline(b, color="white", linewidth=1.5, linestyle="--", alpha=0.8)

        # Tick labels — show all if few templates, else subsample
        step = max(1, n_total // 20)
        tick_positions = list(range(0, n_total, step))
        ax_sim.set_xticks(tick_positions)
        ax_sim.set_xticklabels([labels_sim[i] for i in tick_positions], fontsize=7, rotation=45)
        ax_sim.set_yticks(tick_positions)
        ax_sim.set_yticklabels([labels_sim[i] for i in tick_positions], fontsize=7)
        ax_sim.set_title("Spatial Similarity Matrix")
        ax_sim.set_xlabel("Template")
        ax_sim.set_ylabel("Template")

        # ── Scatter plot: DTW distance vs spatial similarity ──
        dist_matrix = metrics["distance_matrix"]   # (n_total, n_total), CLOSED first

        rows, cols = np.triu_indices(n_total, k=1)
        d_intra_c, s_intra_c = [], []
        d_intra_o, s_intra_o = [], []
        d_cross,   s_cross   = [], []

        for r, c in zip(rows, cols):
            d = dist_matrix[r, c]
            s = sim_matrix[r, c]
            r_closed = r < n_closed
            c_closed = c < n_closed
            if r_closed and c_closed:
                d_intra_c.append(d); s_intra_c.append(s)
            elif not r_closed and not c_closed:
                d_intra_o.append(d); s_intra_o.append(s)
            else:
                d_cross.append(d); s_cross.append(s)

        handles = []
        if d_intra_c:
            sc = ax_scatter.scatter(d_intra_c, s_intra_c, c="#d44", alpha=0.65, s=18, label="Intra-CLOSED")
            handles.append(sc)
        if d_intra_o:
            so = ax_scatter.scatter(d_intra_o, s_intra_o, c="#48b", alpha=0.65, s=18, label="Intra-OPEN")
            handles.append(so)
        if d_cross:
            sx = ax_scatter.scatter(d_cross, s_cross, c="#4a4", alpha=0.5, s=14, label="Cross-class", marker="^")
            handles.append(sx)

        ax_scatter.set_xlabel("DTW Distance")
        ax_scatter.set_ylabel("Cosine Similarity")
        ax_scatter.set_title("DTW Distance vs Spatial Similarity")
        if handles:
            ax_scatter.legend(fontsize=8, handles=handles)
        ax_scatter.grid(alpha=0.3)

        self.figure_spatial.tight_layout()
        self.canvas_spatial.draw()

        # Stats text
        all_intra_sim = s_intra_c + s_intra_o
        mean_intra_sim = np.mean(all_intra_sim) if all_intra_sim else float("nan")
        mean_cross_sim = np.mean(s_cross) if s_cross else float("nan")
        sim_gap = mean_cross_sim - mean_intra_sim if all_intra_sim and s_cross else float("nan")

        lines = ["Spatial Similarity (cosine, L2-normalized RMS vectors)"]
        if d_intra_c:
            lines.append(f"  Intra-CLOSED:  n={len(d_intra_c):3d}  "
                         f"mean_sim={np.mean(s_intra_c):.4f}  mean_dtw={np.mean(d_intra_c):.4f}")
        if d_intra_o:
            lines.append(f"  Intra-OPEN:    n={len(d_intra_o):3d}  "
                         f"mean_sim={np.mean(s_intra_o):.4f}  mean_dtw={np.mean(d_intra_o):.4f}")
        if d_cross:
            lines.append(f"  Cross-class:   n={len(d_cross):3d}  "
                         f"mean_sim={np.mean(s_cross):.4f}  mean_dtw={np.mean(d_cross):.4f}")
        if not np.isnan(sim_gap):
            direction = ">" if sim_gap > 0 else "<"
            lines.append(f"\n  Cross-class sim is {direction} intra sim by {abs(sim_gap):.4f}  "
                         f"({'good: classes are spatially distinct' if sim_gap < 0 else 'warning: classes overlap spatially'})")
        self.spatial_stats_label.setText("\n".join(lines))

    def _compute_coupled_tab_from_ui(self):
        """Called by 'Compute Coupled' button — uses cached metrics/spatial data."""
        metrics = getattr(self, '_cached_metrics', None)
        spatial_closed = getattr(self, '_cached_spatial_closed', None)
        spatial_open = getattr(self, '_cached_spatial_open', None)
        if metrics is None:
            self.coupled_stats_label.setText("Run 'Compute' first to compute DTW metrics.")
            return
        self._compute_coupled_tab(metrics, spatial_closed, spatial_open)

    def _compute_coupled_tab(self, metrics, spatial_closed, spatial_open):
        """Compute and render Tab 3: Coupled Correction analysis."""
        from mindmove.model.core.algorithm import aggregate_distances_with_per_template_spatial
        from mindmove.model.template_study import analyze_template_quality
        import matplotlib.gridspec as gridspec

        self.figure_coupled.clear()

        n_closed = len(self.templates_closed)
        n_open = len(self.templates_open)
        n_total = n_closed + n_open

        if n_total < 2 or (spatial_closed is None and spatial_open is None):
            ax = self.figure_coupled.add_subplot(111)
            ax.text(0.5, 0.5, "Coupled correction requires spatial profiles (per_template_rms).",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.axis("off")
            self.canvas_coupled.draw()
            self.coupled_stats_label.setText("Spatial profiles not available.")
            return

        # Check per_template_rms availability
        has_ptpl_c = spatial_closed is not None and spatial_closed.get("per_template_rms") is not None
        has_ptpl_o = spatial_open is not None and spatial_open.get("per_template_rms") is not None
        if not has_ptpl_c and not has_ptpl_o:
            ax = self.figure_coupled.add_subplot(111)
            ax.text(0.5, 0.5, "per_template_rms not available in spatial profiles.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.axis("off")
            self.canvas_coupled.draw()
            self.coupled_stats_label.setText("per_template_rms not available.")
            return

        spatial_mode = self.coupled_mode_combo.currentData()
        k = self.coupled_k_spinbox.value()
        threshold = self.coupled_threshold_spinbox.value()
        baseline = self.coupled_baseline_spinbox.value()

        # Stack per_template_rms: CLOSED first, then OPEN
        rms_parts = []
        if has_ptpl_c:
            rms_parts.append(spatial_closed["per_template_rms"])
        else:
            # Fallback: zeros for closed templates if missing
            if n_closed > 0 and has_ptpl_o:
                n_ch = spatial_open["per_template_rms"].shape[1]
                rms_parts.append(np.zeros((n_closed, n_ch)))
        if has_ptpl_o:
            rms_parts.append(spatial_open["per_template_rms"])
        else:
            if n_open > 0 and has_ptpl_c:
                n_ch = spatial_closed["per_template_rms"].shape[1]
                rms_parts.append(np.zeros((n_open, n_ch)))
        rms_all = np.vstack(rms_parts)  # (n_total, n_ch)

        # Pairwise cosine similarities (L2-unit vectors)
        sim_matrix = np.clip(rms_all @ rms_all.T, 0.0, 1.0)  # (n_total, n_total)

        # Apply per-template coupled correction to the raw distance matrix
        dist_matrix_raw = metrics["distance_matrix"].copy()  # (n_total, n_total)
        dist_matrix_corr = np.zeros_like(dist_matrix_raw)

        for i in range(n_total):
            # For each template i as the "live" signal, correct distances to all others
            live_rms_i = rms_all[i]  # unit vector
            raw_row = dist_matrix_raw[i].copy()
            # Per-template similarities: sim_j = dot(live_rms_i, rms_j) = sim_matrix[i]
            sims_row = sim_matrix[i]
            corr_row = np.empty(n_total)
            for j in range(n_total):
                D_j = raw_row[j]
                s_j = float(sims_row[j])
                if i == j:
                    corr_row[j] = 0.0
                    continue
                if spatial_mode == "gate":
                    corr_row[j] = np.inf if s_j < threshold else D_j
                elif spatial_mode in ("scaling",):
                    corr_row[j] = D_j / max(s_j, 0.1) ** k
                elif spatial_mode == "relu_scaling":
                    t = max(threshold, 1e-6)
                    k_ = max(k, 0.1)
                    f = 1.0 if s_j >= t else baseline ** ((1.0 - s_j / t) ** (1.0 / k_))
                    corr_row[j] = D_j / max(f, 0.01)
                elif spatial_mode == "relu_ext_scaling":
                    t = max(threshold, 1e-6)
                    k_ = max(k, 0.1)
                    if s_j >= t:
                        rt = (s_j - t) / max(1.0 - t, 1e-6)
                        f_ext = (1.0 / baseline) ** (rt ** (1.0 / k_))
                    else:
                        f_ext = baseline ** ((1.0 - s_j / t) ** (1.0 / k_))
                    corr_row[j] = D_j / max(f_ext, 0.01)
                else:
                    corr_row[j] = D_j
            dist_matrix_corr[i] = corr_row

        # Replace inf with large finite value for display only
        display_corr = np.where(np.isinf(dist_matrix_corr), np.nanmax(dist_matrix_raw) * 2, dist_matrix_corr)

        # Re-symmetrize (average i→j and j→i)
        display_corr = (display_corr + display_corr.T) / 2

        # Compute corrected metrics from upper triangle (symmetrized)
        rows_ut, cols_ut = np.triu_indices(n_total, k=1)
        d_ic_raw, d_ic_corr = [], []
        d_io_raw, d_io_corr = [], []
        d_cross_raw, d_cross_corr = [], []
        s_ic, s_io, s_cross = [], [], []

        for r, c in zip(rows_ut, cols_ut):
            d_raw = dist_matrix_raw[r, c]
            d_corr_val = display_corr[r, c]
            s = sim_matrix[r, c]
            r_closed = r < n_closed
            c_closed = c < n_closed
            if r_closed and c_closed:
                d_ic_raw.append(d_raw); d_ic_corr.append(d_corr_val); s_ic.append(s)
            elif not r_closed and not c_closed:
                d_io_raw.append(d_raw); d_io_corr.append(d_corr_val); s_io.append(s)
            else:
                d_cross_raw.append(d_raw); d_cross_corr.append(d_corr_val); s_cross.append(s)

        # Grade computation (same metric as Tab 1: separation gap / intra std)
        def _grade(d_intra_c, d_intra_o, d_cross_c, d_cross_o):
            if not d_intra_c and not d_intra_o:
                return 0.0
            ic_m = np.mean(d_intra_c) if d_intra_c else 0.0
            ic_s = np.std(d_intra_c) if d_intra_c else 0.0
            io_m = np.mean(d_intra_o) if d_intra_o else 0.0
            io_s = np.std(d_intra_o) if d_intra_o else 0.0
            ec_m = np.mean(d_cross_c) if d_cross_c else float('inf')
            eo_m = np.mean(d_cross_o) if d_cross_o else float('inf')
            gap_c = ec_m - (ic_m + ic_s)
            gap_o = eo_m - (io_m + io_s)
            gap = (gap_c + gap_o) / 2
            denom = (ic_s + io_s) / 2 + 1e-8
            score = min(10.0, max(0.0, 5.0 + 5.0 * gap / denom))
            return score

        grade_raw = _grade(d_ic_raw, d_io_raw, d_cross_raw, d_cross_raw)
        grade_corr = _grade(d_ic_corr, d_io_corr, d_cross_corr, d_cross_corr)

        # ── Plots ──
        gs3 = gridspec.GridSpec(1, 2, figure=self.figure_coupled, width_ratios=[1, 1.1])
        ax_mat = self.figure_coupled.add_subplot(gs3[0])
        ax_sc = self.figure_coupled.add_subplot(gs3[1])

        # Left: corrected distance matrix heatmap
        im = ax_mat.imshow(display_corr, cmap="viridis", aspect="auto")
        self.figure_coupled.colorbar(im, ax=ax_mat, label="Corrected Distance", fraction=0.046)
        if n_closed > 0 and n_open > 0:
            b = n_closed - 0.5
            ax_mat.axhline(b, color="white", linewidth=1.5, linestyle="--", alpha=0.8)
            ax_mat.axvline(b, color="white", linewidth=1.5, linestyle="--", alpha=0.8)
        step = max(1, n_total // 20)
        ticks = list(range(0, n_total, step))
        tick_labels = (
            [f"C{i+1}" for i in range(n_closed)] +
            [f"O{i+1}" for i in range(n_open)]
        )
        ax_mat.set_xticks(ticks)
        ax_mat.set_xticklabels([tick_labels[i] for i in ticks], fontsize=7, rotation=45)
        ax_mat.set_yticks(ticks)
        ax_mat.set_yticklabels([tick_labels[i] for i in ticks], fontsize=7)
        ax_mat.set_title(f"Corrected Distance Matrix\n({spatial_mode}, k={k:.1f}, t={threshold:.2f})")

        # Right: DTW vs Spatial scatter (raw distances, color by class)
        if d_ic_raw:
            ax_sc.scatter(d_ic_raw, s_ic, c="#d44", alpha=0.65, s=18, label="Intra-CLOSED")
        if d_io_raw:
            ax_sc.scatter(d_io_raw, s_io, c="#48b", alpha=0.65, s=18, label="Intra-OPEN")
        if d_cross_raw:
            ax_sc.scatter(d_cross_raw, s_cross, c="#4a4", alpha=0.5, s=14, label="Cross-class", marker="^")
        ax_sc.set_xlabel("DTW Distance (raw)")
        ax_sc.set_ylabel("Cosine Similarity")
        ax_sc.set_title("Raw DTW vs Per-Template Spatial Similarity")
        ax_sc.legend(fontsize=8)
        ax_sc.grid(alpha=0.3)

        self.figure_coupled.tight_layout()
        self.canvas_coupled.draw()

        # ── Stats text ──
        def _fmt(vals, label):
            if not vals:
                return f"  {label}: n=0  (no pairs)"
            return (f"  {label}: n={len(vals):3d}  "
                    f"mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

        def _gap(intra_list, cross_list):
            if not intra_list or not cross_list:
                return float("nan")
            return np.mean(cross_list) - (np.mean(intra_list) + np.std(intra_list))

        gap_c_raw = _gap(d_ic_raw, d_cross_raw)
        gap_o_raw = _gap(d_io_raw, d_cross_raw)
        gap_c_corr = _gap(d_ic_corr, d_cross_corr)
        gap_o_corr = _gap(d_io_corr, d_cross_corr)

        lines_c = [
            "Coupled Correction Analysis",
            f"  Mode: {spatial_mode}  k={k:.1f}  threshold={threshold:.2f}  baseline={baseline:.2f}",
            "",
            "Raw distances:",
            _fmt(d_ic_raw, "Intra-CLOSED"),
            _fmt(d_io_raw, "Intra-OPEN  "),
            _fmt(d_cross_raw, "Cross-class "),
            "",
            "Corrected distances:",
            _fmt(d_ic_corr, "Intra-CLOSED"),
            _fmt(d_io_corr, "Intra-OPEN  "),
            _fmt(d_cross_corr, "Cross-class "),
            "",
            f"  Separation gap (raw):  CLOSED={gap_c_raw:+.4f}  OPEN={gap_o_raw:+.4f}",
            f"  Separation gap (corr): CLOSED={gap_c_corr:+.4f}  OPEN={gap_o_corr:+.4f}",
            f"  Gap improvement:       CLOSED Δ={gap_c_corr-gap_c_raw:+.4f}  OPEN Δ={gap_o_corr-gap_o_raw:+.4f}",
            "",
            f"  Grade (raw):  {grade_raw:.1f} / 10",
            f"  Grade (corr): {grade_corr:.1f} / 10",
        ]
        self.coupled_stats_label.setText("\n".join(lines_c))

    def _populate_flagged_templates(self, quality):
        """Build the flagged-templates section with Remove buttons."""
        # Clear previous content
        while self.flagged_layout.count():
            child = self.flagged_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        flagged = []
        for t in quality.get("closed_analysis", []):
            if t["n_flags"] > 0:
                flagged.append(("closed", t))
        for t in quality.get("open_analysis", []):
            if t["n_flags"] > 0:
                flagged.append(("open", t))

        if not flagged:
            lbl = QLabel("No flagged templates.")
            lbl.setStyleSheet("color: green; font-style: italic;")
            self.flagged_layout.addWidget(lbl)
            return

        header = QLabel(f"{len(flagged)} flagged template(s) — remove outliers to improve quality:")
        header.setStyleSheet("font-weight: bold;")
        self.flagged_layout.addWidget(header)

        for class_name, t in flagged:
            row = QHBoxLayout()
            flag_str = ", ".join(t["flags"])
            consensus = " [OUTLIER]" if t["is_outlier"] else ""
            idx = t["index"]
            # Build stable label and provenance string from metadata
            meta_list = self.metadata_closed if class_name == "closed" else self.metadata_open
            prov = ""
            stable_label = t["label"]  # fallback: C{idx+1} / O{idx+1} from template_study
            if idx < len(meta_list):
                m = meta_list[idx]
                orig_id = m.get("id", idx + 1)
                prefix = "C" if class_name == "closed" else "O"
                stable_label = f"{prefix}{orig_id}"
                rec = m.get("recording", "")
                cyc = m.get("cycle", "")
                parts = []
                if rec:
                    parts.append(rec)
                if cyc != "":
                    parts.append(f"#{cyc}")
                if parts:
                    prov = f"  ({', '.join(parts)})"
            info = QLabel(
                f"{stable_label}{prov}  intra={t['intra']:.4f}  inter={t['inter']:.4f}  "
                f"margin={t['margin']:.4f}  sil={t['silhouette']:.3f}  "
                f"[{flag_str}]{consensus}"
            )
            info.setStyleSheet(
                "font-family: monospace; font-size: 11px;"
                + (" color: red;" if t["is_outlier"] else " color: #cc6600;")
            )
            row.addWidget(info, stretch=1)

            remove_btn = QPushButton("Remove")
            remove_btn.setFixedWidth(70)
            remove_btn.setStyleSheet("background-color: #ff6666;")
            idx = t["index"]
            remove_btn.clicked.connect(
                lambda checked=False, cn=class_name, i=idx: self._remove_template(cn, i)
            )
            row.addWidget(remove_btn)

            container = QWidget()
            container.setLayout(row)
            self.flagged_layout.addWidget(container)

    def _remove_template(self, class_name, index):
        """Remove a template by class and index, then re-compute."""
        target = self.templates_closed if class_name == "closed" else self.templates_open
        meta_target = self.metadata_closed if class_name == "closed" else self.metadata_open
        if 0 <= index < len(target):
            # Build display label using stored original ID if available
            m = meta_target[index] if index < len(meta_target) else {}
            orig_id = m.get("id", index + 1)
            removed_label = f"{'C' if class_name == 'closed' else 'O'}{orig_id}"
            del target[index]
            if index < len(meta_target):
                del meta_target[index]
            self.templates_modified = True
            n_c = len(self.templates_closed)
            n_o = len(self.templates_open)
            self.count_label.setText(f"Templates: {n_c} CLOSED + {n_o} OPEN")
            print(f"[TEMPLATE STUDY] Removed {removed_label} — now {n_c} CLOSED + {n_o} OPEN")
            # Re-compute automatically
            self._compute()

    def _compare_features(self):
        """Run analysis across all features and display a comparison table."""
        if not self.templates_closed and not self.templates_open:
            self.stats_label.setText("No templates loaded. Use 'Load from File' to load templates.")
            return

        from mindmove.model.core.features.features_registry import FEATURES
        from PySide6.QtWidgets import QApplication

        self.compare_btn.setEnabled(False)
        self.compare_btn.setText("Comparing...")
        QApplication.processEvents()

        try:
            results = []
            for feature_name in FEATURES:
                try:
                    result = self._compute_metrics(feature_name=feature_name)
                    if result is None:
                        continue
                    metrics, quality, params = result
                    grade = quality["grade"]

                    ic = metrics["intra_closed"]
                    io = metrics["intra_open"]
                    ec = metrics["inter_closed_to_open"]
                    eo = metrics["inter_open_to_closed"]

                    gap_c = (ec["mean"] - ec["std"]) - (ic["mean"] + ic["std"])
                    gap_o = (eo["mean"] - eo["std"]) - (io["mean"] + io["std"])

                    results.append({
                        "feature": feature_name,
                        "total": grade["total"],
                        "sep": grade["separation"],
                        "con": grade["consistency"],
                        "rob": grade["robustness"],
                        "gap_c": gap_c,
                        "gap_o": gap_o,
                        "outliers": grade["details"]["n_outliers"],
                        "crossers": grade["details"]["n_crossers"],
                    })
                except Exception as e:
                    print(f"[COMPARE] {feature_name}: Error — {e}")

            if not results:
                self.stats_label.setText("No features could be computed.")
                return

            # Sort by total grade descending
            results.sort(key=lambda r: r["total"], reverse=True)

            # Build comparison table
            header = (
                f"{'Feature':<20} {'Grade':>6} {'Sep':>5} {'Con':>5} {'Rob':>5} "
                f"{'GapC':>8} {'GapO':>8} {'Out':>4} {'Cross':>5}\n"
                f"{'-'*20} {'-'*6} {'-'*5} {'-'*5} {'-'*5} "
                f"{'-'*8} {'-'*8} {'-'*4} {'-'*5}"
            )
            rows = []
            for r in results:
                rows.append(
                    f"{r['feature']:<20} {r['total']:>6.1f} {r['sep']:>5.1f} {r['con']:>5.1f} {r['rob']:>5.1f} "
                    f"{r['gap_c']:>8.4f} {r['gap_o']:>8.4f} {r['outliers']:>4d} {r['crossers']:>5d}"
                )

            best = results[0]
            text = (
                f"FEATURE COMPARISON  ({len(results)} features tested)\n"
                f"Settings: Aggregation={self.agg_combo.currentText()}, "
                f"Window={self.window_spinbox.value()} samp, "
                f"Overlap={self.overlap_spinbox.value()} samp\n\n"
                f"{header}\n" + "\n".join(rows) + "\n\n"
                f"Best feature: {best['feature']} (Grade: {best['total']:.1f}/30)"
            )
            self.stats_label.setText(text)

        except Exception as e:
            self.stats_label.setText(f"Error during comparison: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.compare_btn.setEnabled(True)
            self.compare_btn.setText("Compare All Features")


class TrainingProtocol(QObject):
    def __init__(self, parent: MindMove | None = ...) -> None:
        super().__init__(parent)

        self.main_window = parent

        # Initialize Protocol UI
        self._setup_protocol_ui()

        # Initialize Protocol
        self.selected_recordings: dict[str, dict] = None
        self.selected_dataset_filepath: dict[str, np.ndarray] = None

        # Threads
        self.create_dataset_thread = None
        self.train_model_thread = None
        self.extract_activations_thread = None

        # Model Interface
        self.model_interface: MindMoveInterface = MindMoveInterface(
            parent=self.main_window
        )

        # Template Manager
        self.template_manager: TemplateManager = TemplateManager()
        self.selected_extraction_recordings: List[str] = []
        self.legacy_emg_folder: Optional[str] = None
        self.legacy_gt_folder: Optional[str] = None
        self._guided_recordings_for_review: List[dict] = []  # Store recordings for template review
        self._onset_method: str = "amplitude"  # "amplitude" or "tkeo"

        # File management:
        self.recordings_dir_path: str = "data/recordings/"
        self.models_dir_path: str = "data/models/"
        self.datasets_dir_path: str = "data/datasets/"
        self.legacy_data_path: str = "data/legacy/"

        # Initialize Template Extraction UI
        self._setup_template_extraction_ui()

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

    def select_recordings(self) -> None:
        self.training_create_dataset_push_button.setEnabled(False)
        if not os.path.exists(self.recordings_dir_path):
            os.makedirs(self.recordings_dir_path)

        # Open dialog to select recordings
        dialog = QFileDialog(self.main_window)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilter("Pickle files (*.pkl)")
        dialog.setDirectory(self.recordings_dir_path)

        filenames, _ = dialog.getOpenFileNames()
        self.selected_recordings = {}
        self.training_create_dataset_selected_recordings_list_widget.clear()

        for file in filenames:
            with open(file, "rb") as f:
                recording = pickle.load(f)
                if not recording:
                    continue
                if type(recording) is not dict:
                    continue

                keys = recording.keys()

                # Check for MindMove virtual hand format
                mindmove_vh_keys = ["emg", "kinematics", "timings_emg", "label", "task"]
                # Check for MindMove keyboard format
                mindmove_kb_keys = ["emg", "gt", "timings_emg", "label", "task"]

                is_valid = (
                    all(key in keys for key in mindmove_vh_keys) or
                    all(key in keys for key in mindmove_kb_keys)
                )

                if not is_valid:
                    print(f" {f} is an invalid recording!")
                    print(f"  Keys found: {list(keys)}")
                    continue

                selected_recordings_key = (
                    recording["label"].capitalize()
                    + " "
                    + recording["task"].capitalize()
                )
                if selected_recordings_key in self.selected_recordings.keys():
                    print(
                        f" {f} has the same label and task as another recording! Skipping..."
                    )
                    continue

                self.selected_recordings[selected_recordings_key] = recording

        for key in self.selected_recordings.keys():
            self.training_create_dataset_selected_recordings_list_widget.addItem(key)

        if len(self.selected_recordings) == 0:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "No valid recordings selected!",
                QMessageBox.Ok,
            )
            return

        self.training_create_dataset_push_button.setEnabled(True)

    def _create_dataset(self) -> None:
        if not self.selected_recordings:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "No recordings selected!",
                QMessageBox.Ok,
            )
            return
        self.training_create_dataset_push_button.setEnabled(False)
        self.training_create_datasets_select_recordings_push_button.setEnabled(False)

        self.create_dataset_thread = PyQtThread(
            target=self._create_dataset_thread, parent=self.main_window
        )
        self.create_dataset_thread.has_finished_signal.connect(
            self.__create_dataset_thread_finished
        )
        self.create_dataset_thread.start()

    def __create_dataset_thread_finished(self) -> None:
        self.training_create_dataset_selected_recordings_list_widget.clear()
        self.train_create_dataset_progress_bar.setValue(0)
        self.training_create_dataset_label_line_edit.setText("")
        self.training_create_datasets_select_recordings_push_button.setEnabled(True)
        self.selected_recordings = None

    def _create_dataset_thread(self) -> None:
        df = {}
        for k, v in self.selected_recordings.items():
            task_name = v["task"]

            # Get EMG data
            emg = v["emg"]

            # Get kinematics/GT - handle both formats
            if "kinematics" in v:
                kinematics = v["kinematics"]
            elif "gt" in v:
                # Keyboard format: gt is binary at EMG sample rate
                gt = v["gt"]
                kinematics = gt.reshape(1, -1) if gt.ndim == 1 else gt
            else:
                print(f"Warning: No kinematics or gt found in {k}, skipping")
                continue

            if task_name in df.keys():
                df[task_name]["emg"] = np.concatenate([df[task_name]["emg"], emg])
                df[task_name]["kinematics"] = np.concatenate(
                    [df[task_name]["kinematics"], kinematics], axis=-1
                )
            else:
                df[task_name] = {}
                df[task_name]["emg"] = emg
                df[task_name]["kinematics"] = kinematics

        for k, v in df.items():
            print(k, v["emg"].shape, v["kinematics"].shape)

        label = self.training_create_dataset_label_line_edit.text()
        if not label:
            label = "default"

        now = datetime.now()
        formatted_now = now.strftime("%Y%m%d_%H%M%S%f")

        # TODO: Create dataset code in the model interface and dataset class
        dataset_dict = self.model_interface.create_dataset(df)

        file_name = f"MindMove_Dataset_{formatted_now}_{label.lower()}.pkl"

        if not os.path.exists(self.datasets_dir_path):
            os.makedirs(self.datasets_dir_path)

        with open(os.path.join(self.datasets_dir_path, file_name), "wb") as f:
            pickle.dump(dataset_dict, f)

    def _select_dataset(self) -> None:
        if not os.path.exists(self.datasets_dir_path):
            os.makedirs(self.datasets_dir_path)

        # Open dialog to select dataset
        dialog = QFileDialog(self.main_window)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("Pickle files (*.pkl)")
        dialog.setDirectory(self.datasets_dir_path)

        filename, _ = dialog.getOpenFileName()

        if not filename:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "No dataset selected!",
                QMessageBox.Ok,
            )
            self.training_selected_dataset_label.setText("No dataset selected!")
            return

        self.selected_dataset_filepath = filename
        self.training_selected_dataset_label.setText(
            self.selected_dataset_filepath.split("_")[-1].split(".")[0]
        )

        self.train_model_push_button.setEnabled(True)

    def _train_model(self) -> None:
        if not self.selected_dataset_filepath:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "No dataset selected!",
                QMessageBox.Ok,
            )
            return

        self.train_model_push_button.setEnabled(False)
        self.training_select_dataset_push_button.setEnabled(False)

        self.train_model_thread = PyQtThread(
            target=self._train_model_thread, parent=self.main_window
        )
        self.train_model_thread.has_finished_signal.connect(self._train_model_finished)
        self.train_model_thread.start()

    def _train_model_thread(self) -> None:
        label = self.training_model_label_line_edit.text()
        if not label:
            label = "default"

        assert self.selected_dataset_filepath is not None

        with open(self.selected_dataset_filepath, "rb") as file:
            dataset = pickle.load(file)

        # Validate dataset format
        if not isinstance(dataset, dict):
            raise TypeError(
                f"Invalid dataset format: expected a dictionary but got {type(dataset).__name__}. "
                "Make sure you selected a dataset file (not a template file). "
                "Dataset files are created via 'Create Dataset' and saved in data/datasets/."
            )

        if "training" not in dataset or "testing" not in dataset:
            raise KeyError(
                f"Invalid dataset structure: missing 'training' or 'testing' keys. "
                f"Found keys: {list(dataset.keys())}. "
                "Make sure you selected a valid dataset file created via 'Create Dataset'."
            )

        # TODO: Train model code in model interface and model class
        self.model_interface.train_model(dataset)

        # Save model
        now = datetime.now()
        formatted_now = now.strftime("%Y%m%d_%H%M%S%f")

        file_name = f"MindMove_Model_{formatted_now}_{label.lower()}.pkl"

        if not os.path.exists(self.models_dir_path):
            os.makedirs(self.models_dir_path)

        model_filepath = os.path.join(self.models_dir_path, file_name)

        # TODO: Save your model in the model interface and model class
        self.model_interface.save_model(model_filepath)

    def _train_model_finished(self) -> None:
        self.training_progress_bar.setValue(0)
        self.training_selected_dataset_label.setText("No dataset selected!")
        self.training_select_dataset_push_button.setEnabled(True)
        self.selected_dataset_filepath = None
        self.training_model_label_line_edit.setText("")

    def _load_existing_model(self) -> None:
        # Open dialog to select model
        dialog = QFileDialog(self.main_window)
        dialog.setFileMode(QFileDialog.ExistingFile)
        # TODO: Add model file filter
        dialog.setNameFilter("Torch Model (*.pt)")
        dialog.setDirectory(self.models_dir_path)

        filename, _ = dialog.getOpenFileName()

        if not filename:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "No model selected!",
                QMessageBox.Ok,
            )
            return

        # TODO: Load model in model interface and model class
        model = self.model_interface.load_model(filename)

        self.training_load_existing_model_label.setText(f"{filename}")

        return model

    def _toggle_train_model_stacked_widget(self, toggled: bool) -> None:
        if toggled:
            self.training_train_model_stacked_widget.setCurrentIndex(0)
        else:
            self.training_train_model_stacked_widget.setCurrentIndex(1)

    def _setup_protocol_ui(self) -> None:
        # Hide Create Datasets section (not needed for DTW workflow)
        self.training_create_dataset_group_box = (
            self.main_window.ui.trainingCreateDatasetGroupBox
        )
        self.training_create_dataset_group_box.setVisible(False)

        # Train Model - Repurpose the existing group box
        self.training_train_model_group_box = (
            self.main_window.ui.trainingTrainModelGroupBox
        )
        self.training_train_model_group_box.setTitle("Create Model (DTW)")

        # Hide old widgets we don't need
        self.main_window.ui.trainingTrainNewModelRadioButton.setVisible(False)
        self.main_window.ui.trainingTrainExistingModelRadioButton.setVisible(False)
        self.main_window.ui.trainingTrainModelStackedWidget.setVisible(False)
        self.main_window.ui.trainingSelectDatasetPushButton.setVisible(False)
        self.main_window.ui.trainingSelectedDatasetLabel.setVisible(False)
        self.main_window.ui.trainingTrainModelPushButton.setVisible(False)
        self.main_window.ui.trainingProgressBar.setVisible(False)

        # Keep the model label line edit but repurpose it
        self.training_model_label_line_edit = (
            self.main_window.ui.trainingModelLabelLineEdit
        )
        self.main_window.ui.label_8.setText("Model Name:")

        # Get the layout and add new widgets
        layout = self.training_train_model_group_box.layout()

        # Clear existing layout items positioning and add our new widgets
        # Row 0: Select Templates (combined file with both OPEN and CLOSED)
        self.select_templates_btn = QPushButton("Select Templates")
        self.select_templates_btn.clicked.connect(self._select_combined_template_file)
        self.templates_label = QLabel("No templates selected")
        layout.addWidget(self.select_templates_btn, 0, 0, 1, 1)
        layout.addWidget(self.templates_label, 0, 1, 1, 2)

        # Keep legacy paths for backwards compatibility but hidden
        self.selected_open_templates_path: Optional[str] = None
        self.selected_closed_templates_path: Optional[str] = None
        self.selected_combined_templates_path: Optional[str] = None

        # Row 1: Window/Overlap presets
        self.window_preset_label = QLabel("Window/Overlap:")
        self.window_preset_combo = QComboBox()
        self.window_preset_combo.addItems([
            "96/32 ms (Default)",
            "150/50 ms (Eddy's)",
            "200/100 ms",
            "Custom"
        ])
        self.window_preset_combo.currentIndexChanged.connect(self._on_window_preset_changed)
        layout.addWidget(self.window_preset_label, 1, 0, 1, 1)
        layout.addWidget(self.window_preset_combo, 1, 1, 1, 2)

        # Row 2: Custom window/overlap inputs (hidden by default)
        self.custom_window_label = QLabel("Window (ms):")
        self.custom_window_spinbox = QSpinBox()
        self.custom_window_spinbox.setRange(1, 500)
        self.custom_window_spinbox.setValue(96)
        self.custom_overlap_label = QLabel("Overlap (ms):")
        self.custom_overlap_spinbox = QSpinBox()
        self.custom_overlap_spinbox.setRange(0, 499)
        self.custom_overlap_spinbox.setValue(32)
        layout.addWidget(self.custom_window_label, 2, 0, 1, 1)
        layout.addWidget(self.custom_window_spinbox, 2, 1, 1, 1)
        layout.addWidget(self.custom_overlap_label, 2, 2, 1, 1)
        layout.addWidget(self.custom_overlap_spinbox, 2, 3, 1, 1)
        # Hide custom inputs by default
        self.custom_window_label.setVisible(False)
        self.custom_window_spinbox.setVisible(False)
        self.custom_overlap_label.setVisible(False)
        self.custom_overlap_spinbox.setVisible(False)

        # Row 3: Feature selection
        self.feature_label = QLabel("Feature:")
        self.feature_combo = QComboBox()
        # Add all features from registry
        from mindmove.model.core.features.features_registry import FEATURES
        self.feature_combo.addItems(list(FEATURES.keys()))
        self.feature_combo.setCurrentText("rms")  # Default to RMS
        layout.addWidget(self.feature_label, 3, 0, 1, 1)
        layout.addWidget(self.feature_combo, 3, 1, 1, 2)

        # Row 4: DTW algorithm selection
        self.dtw_algorithm_label = QLabel("DTW Algorithm:")
        self.dtw_algorithm_combo = QComboBox()
        self.dtw_algorithm_combo.addItems([
            "Numba (Cosine) - Recommended",
            "tslearn (Euclidean)",
            "dtaidistance (Euclidean)",
            "Pure Python (Cosine)"
        ])
        layout.addWidget(self.dtw_algorithm_label, 4, 0, 1, 1)
        layout.addWidget(self.dtw_algorithm_combo, 4, 1, 1, 2)

        # Row 5: Dead channels input (1-indexed for user)
        self.dead_channels_label = QLabel("Dead Channels:")
        self.dead_channels_input = QLineEdit()
        self.dead_channels_input.setPlaceholderText("e.g., 9, 22, 25 (1-indexed)")
        self.dead_channels_input.setToolTip("Enter channel numbers (1-32) separated by commas. These channels will be excluded from DTW computation.")
        layout.addWidget(self.dead_channels_label, 5, 0, 1, 1)
        layout.addWidget(self.dead_channels_input, 5, 1, 1, 2)

        # Row 6: Channel mode selection (global vs per-class)
        self.channel_mode_label = QLabel("Channel Mode:")
        self.channel_mode_combo = QComboBox()
        self.channel_mode_combo.addItems([
            "Global (all channels)",
            "Per-class (spatial separation)"
        ])
        self.channel_mode_combo.currentIndexChanged.connect(self._on_channel_mode_changed)
        layout.addWidget(self.channel_mode_label, 6, 0, 1, 1)
        layout.addWidget(self.channel_mode_combo, 6, 1, 1, 2)

        # Row 7: Per-class channel fields (hidden by default)
        self.closed_channels_label = QLabel("CLOSED Ch:")
        self.closed_channels_input = QLineEdit()
        self.closed_channels_input.setPlaceholderText("e.g., 5,6,8,9,10,11 (1-indexed)")
        self.closed_channels_input.setToolTip("Channels assigned to CLOSED class (1-indexed)")
        self.open_channels_label = QLabel("OPEN Ch:")
        self.open_channels_input = QLineEdit()
        self.open_channels_input.setPlaceholderText("e.g., 1,12,13 (1-indexed)")
        self.open_channels_input.setToolTip("Channels assigned to OPEN class (1-indexed)")
        layout.addWidget(self.closed_channels_label, 7, 0, 1, 1)
        layout.addWidget(self.closed_channels_input, 7, 1, 1, 2)
        layout.addWidget(self.open_channels_label, 7, 2, 1, 1)
        layout.addWidget(self.open_channels_input, 7, 3, 1, 1)
        # Hide per-class fields by default
        self.closed_channels_label.setVisible(False)
        self.closed_channels_input.setVisible(False)
        self.open_channels_label.setVisible(False)
        self.open_channels_input.setVisible(False)

        # Row 8: Auto-detect button for per-class channels (hidden by default)
        self.auto_detect_channels_btn = QPushButton("Auto-Detect Channels from Onset")
        self.auto_detect_channels_btn.setToolTip("Run per-class channel assignment using onset detection data")
        self.auto_detect_channels_btn.clicked.connect(self._auto_detect_per_class_channels)
        self.auto_detect_channels_btn.setEnabled(False)
        layout.addWidget(self.auto_detect_channels_btn, 8, 0, 1, 3)
        self.auto_detect_channels_btn.setVisible(False)

        # Row 9: Distance aggregation method
        self.distance_agg_label = QLabel("Distance Aggregation:")
        self.distance_agg_combo = QComboBox()
        self.distance_agg_combo.addItems([
            "Average of 3 smallest (Recommended)",
            "Minimum distance",
            "Average of all"
        ])
        self.distance_agg_combo.setToolTip("How to compute final distance from multiple templates")
        layout.addWidget(self.distance_agg_label, 9, 0, 1, 1)
        layout.addWidget(self.distance_agg_combo, 9, 1, 1, 2)

        # Row 10: Post-prediction smoothing
        self.smoothing_label = QLabel("State Smoothing:")
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItems([
            "Majority Vote (5 samples)",
            "5 Consecutive",
            "None"
        ])
        self.smoothing_combo.setCurrentIndex(1)  # Default: 5 Consecutive
        self.smoothing_combo.setToolTip("Method to smooth state transitions")
        layout.addWidget(self.smoothing_label, 10, 0, 1, 1)
        layout.addWidget(self.smoothing_combo, 10, 1, 1, 2)

        # Row 11: Decision model selector
        self.decision_model_label = QLabel("Decision Model:")
        self.decision_model_combo = QComboBox()
        self.decision_model_combo.addItems(["None", "CatBoost", "Neural Network", "Both"])
        self.decision_model_combo.setCurrentIndex(1)  # Default: CatBoost
        self.decision_model_combo.setToolTip(
            "Train a ML decision model alongside the DTW classifier.\n"
            "CatBoost: Gradient boosting (deterministic, no GPU needed)\n"
            "Neural Network: Small NN (requires PyTorch)\n"
            "None: Use only threshold-based decisions"
        )
        layout.addWidget(self.decision_model_label, 11, 0, 1, 1)
        layout.addWidget(self.decision_model_combo, 11, 1, 1, 2)

        # Row 12: Model name (reuse existing widget, just reposition)
        layout.addWidget(self.main_window.ui.label_8, 12, 0, 1, 1)
        layout.addWidget(self.training_model_label_line_edit, 12, 1, 1, 2)

        # Row 13: Create Model button
        self.create_model_btn = QPushButton("Create Model")
        self.create_model_btn.clicked.connect(self._create_dtw_model)
        self.create_model_btn.setEnabled(False)
        layout.addWidget(self.create_model_btn, 13, 0, 1, 3)

        # Row 14: Progress bar
        self.model_creation_progress_bar = self.main_window.ui.trainingProgressBar
        self.model_creation_progress_bar.setVisible(True)
        self.model_creation_progress_bar.setValue(0)
        layout.addWidget(self.model_creation_progress_bar, 14, 0, 1, 3)

    def _setup_template_extraction_ui(self) -> None:
        """Setup UI connections for template extraction group box."""
        # Initialize state variables first (before any signal connections)
        self._is_guided_mode: bool = False
        self._manual_selection_mode: bool = False
        self._manual_templates: List[np.ndarray] = []

        # Template Extraction Group Box
        self.template_extraction_group_box = (
            self.main_window.ui.trainingTemplateExtractionGroupBox
        )

        # Get grid layout for reordering
        grid_layout = self.template_extraction_group_box.layout()

        # Get references to labels and combos
        class_label = self.main_window.ui.label_9  # "Class:" label
        data_format_label = self.main_window.ui.label_12  # "Data Format:" label
        self.template_class_combo = self.main_window.ui.trainingTemplateClassComboBox
        self.data_format_combo = self.main_window.ui.trainingDataFormatComboBox

        # Remove from current positions
        grid_layout.removeWidget(class_label)
        grid_layout.removeWidget(self.template_class_combo)
        grid_layout.removeWidget(data_format_label)
        grid_layout.removeWidget(self.data_format_combo)

        # Re-add in swapped positions: Data Format at row 0, Class at row 1
        grid_layout.addWidget(data_format_label, 0, 0, 1, 1)
        grid_layout.addWidget(self.data_format_combo, 0, 1, 1, 1)
        grid_layout.addWidget(class_label, 1, 0, 1, 1)
        grid_layout.addWidget(self.template_class_combo, 1, 1, 1, 1)

        # Add "Both" option to Class combo (now: Open=0, Closed=1, Both=2)
        self.template_class_combo.addItem("Both")

        # Data format combo - add Guided Recording option
        # Existing: 0=Auto-detect (single files), 1=Legacy (EMG + GT folders)
        self.data_format_combo.addItem("Guided Recording (Bidirectional)")  # index 2
        self.data_format_combo.currentIndexChanged.connect(self._on_data_format_changed)
        # Rename "Auto-detect (single files)" to "Auto-Detect (myogestic)"
        self.data_format_combo.setItemText(0, "Auto-Detect (myogestic)")
        # Default to Guided Recording format - set later after all widgets created
        # (using blockSignals to avoid triggering _on_data_format_changed during init)

        # Template type combo - store original auto-detect options
        self.template_type_combo = self.main_window.ui.trainingTemplateTypeComboBox
        # UI file has: "Hold Only" at 0, "Onset + Hold" at 1
        # We'll manage options dynamically based on data format

        # Define template type options for each mode
        self._autodetect_template_types = [
            "Hold Only (skip 0.5s)",      # index 0
            "Onset + Hold (start -0.2s)", # index 1
            "Onset (GT=1 start)",         # index 2
            "Manual Selection",           # index 3
        ]
        self._guided_template_types = [
            "Manual Selection",          # index 0 - opens review dialog with two windows
            "After Audio Cue",           # index 1 - template starts at audio cue
            "After Reaction Time",       # index 2 - template starts after reaction time
            "Onset (Amplitude)",         # index 3 - auto-detect via mean+k·std, then review
            "Onset (TKEO)",              # index 4 - auto-detect via |d(TKEO)/dt|, then review
        ]

        # Add extra options for auto-detect mode (UI file has only 2)
        self.template_type_combo.addItem("Onset (GT=1 start)")  # index 2
        self.template_type_combo.addItem("Manual Selection")     # index 3
        self.template_type_combo.currentIndexChanged.connect(self._on_template_type_changed)

        # Recording selection (at original row 3)
        self.select_recordings_for_extraction_btn = (
            self.main_window.ui.trainingSelectRecordingsForExtractionPushButton
        )
        self.select_recordings_for_extraction_btn.clicked.connect(
            self._select_recordings_for_extraction
        )
        self.selected_recordings_for_extraction_label = (
            self.main_window.ui.trainingSelectedRecordingsForExtractionLabel
        )

        # Template duration combo (add dynamically after activation list, before plot buttons)
        self.template_duration_label = QLabel("Template Duration:")
        self.template_duration_combo = QComboBox()
        self.template_duration_combo.addItems(["0.5 s", "1.0 s", "1.5 s", "2.0 s"])
        self.template_duration_combo.setCurrentIndex(1)  # Default to 1.0 s
        self.template_duration_combo.setToolTip("Duration of each template in seconds")
        self.template_duration_combo.currentIndexChanged.connect(self._on_template_duration_changed)

        # Move existing widgets to make room - remove from row 3 and add to row 4
        # Then put our new widget at row 3
        # First, take references to the widgets we need to move
        select_btn = self.main_window.ui.trainingSelectRecordingsForExtractionPushButton
        select_label = self.main_window.ui.trainingSelectedRecordingsForExtractionLabel
        extract_btn = self.main_window.ui.trainingExtractActivationsPushButton
        extract_label = self.main_window.ui.trainingActivationCountLabel

        # Remove them from current positions (they'll be re-added at new positions)
        grid_layout.removeWidget(select_btn)
        grid_layout.removeWidget(select_label)
        grid_layout.removeWidget(extract_btn)
        grid_layout.removeWidget(extract_label)

        # Add template duration at row 3
        grid_layout.addWidget(self.template_duration_label, 3, 0, 1, 1)
        grid_layout.addWidget(self.template_duration_combo, 3, 1, 1, 1)

        # Re-add the moved widgets at shifted positions
        grid_layout.addWidget(select_btn, 4, 0, 1, 1)
        grid_layout.addWidget(select_label, 4, 1, 1, 1)
        grid_layout.addWidget(extract_btn, 5, 0, 1, 1)
        grid_layout.addWidget(extract_label, 5, 1, 1, 1)

        # Also need to shift the Selection Mode row
        mode_label = self.main_window.ui.label_11
        mode_combo = self.main_window.ui.trainingSelectionModeComboBox
        grid_layout.removeWidget(mode_label)
        grid_layout.removeWidget(mode_combo)
        grid_layout.addWidget(mode_label, 6, 0, 1, 1)
        grid_layout.addWidget(mode_combo, 6, 1, 1, 1)

        # Shift activation list widget
        list_widget = self.main_window.ui.trainingActivationListWidget
        grid_layout.removeWidget(list_widget)
        grid_layout.addWidget(list_widget, 7, 0, 1, 2)

        # Shift plot and select buttons
        plot_btn = self.main_window.ui.trainingPlotSelectedPushButton
        select_templates_btn = self.main_window.ui.trainingSelectTemplatesPushButton
        grid_layout.removeWidget(plot_btn)
        grid_layout.removeWidget(select_templates_btn)
        grid_layout.addWidget(plot_btn, 8, 0, 1, 1)
        grid_layout.addWidget(select_templates_btn, 8, 1, 1, 1)

        # Shift template count label
        count_label = self.main_window.ui.trainingTemplateCountLabel
        grid_layout.removeWidget(count_label)
        grid_layout.addWidget(count_label, 9, 0, 1, 1)

        # Extract activations
        self.extract_activations_btn = (
            self.main_window.ui.trainingExtractActivationsPushButton
        )
        self.extract_activations_btn.clicked.connect(self._extract_activations)
        self.extract_activations_btn.setEnabled(False)
        self.activation_count_label = self.main_window.ui.trainingActivationCountLabel

        # Selection mode combo
        self.selection_mode_combo = self.main_window.ui.trainingSelectionModeComboBox
        self.selection_mode_combo.currentIndexChanged.connect(self._on_selection_mode_changed)

        # Activation list widget
        self.activation_list_widget = self.main_window.ui.trainingActivationListWidget
        self.activation_list_widget.itemSelectionChanged.connect(self._on_activation_selection_changed)

        # Plot selected button
        self.plot_selected_btn = self.main_window.ui.trainingPlotSelectedPushButton
        self.plot_selected_btn.clicked.connect(self._plot_selected_activations)
        self.plot_selected_btn.setEnabled(False)

        # Channel selector for plotting (add dynamically) - 1-indexed for user
        self.plot_channel_label = QLabel("Ch:")
        self.plot_channel_spinbox = QSpinBox()
        self.plot_channel_spinbox.setRange(1, 32)
        self.plot_channel_spinbox.setValue(1)
        self.plot_channel_spinbox.setToolTip("Select which channel to plot (1-32)")
        self.plot_channel_spinbox.setFixedWidth(50)
        # Add next to plot button at row 8
        grid_layout.addWidget(self.plot_channel_label, 8, 0, 1, 1)
        grid_layout.addWidget(self.plot_channel_spinbox, 8, 0, 1, 1)
        # Reposition - put label and spinbox in a compact way
        # Actually let's put them after the plot button
        # Remove and re-add plot button to make room
        plot_btn = self.main_window.ui.trainingPlotSelectedPushButton
        grid_layout.removeWidget(plot_btn)
        # Create a horizontal layout for plot controls
        from PySide6.QtWidgets import QHBoxLayout, QWidget
        plot_controls_widget = QWidget()
        plot_controls_layout = QHBoxLayout(plot_controls_widget)
        plot_controls_layout.setContentsMargins(0, 0, 0, 0)
        plot_controls_layout.addWidget(plot_btn)
        plot_controls_layout.addWidget(self.plot_channel_label)
        plot_controls_layout.addWidget(self.plot_channel_spinbox)
        plot_controls_layout.addStretch()
        grid_layout.addWidget(plot_controls_widget, 8, 0, 1, 1)

        # Select templates button
        self.select_templates_btn = self.main_window.ui.trainingSelectTemplatesPushButton
        self.select_templates_btn.clicked.connect(self._select_templates)
        self.select_templates_btn.setEnabled(False)
        self.template_count_label = self.main_window.ui.trainingTemplateCountLabel

        # Template set name input (add dynamically since it's not in the UI file)
        # Create label and line edit for template set name
        self.template_set_name_label = QLabel("Template Set Name:")
        self.template_set_name_label.setToolTip("Optional name to distinguish this template set (e.g., 'subject1_session2')")
        self.template_set_name_line_edit = QLineEdit()
        self.template_set_name_line_edit.setPlaceholderText("(optional, e.g., subject1_session2)")

        # Add at row 9 col 1, and line edit at row 10
        grid_layout.addWidget(self.template_set_name_label, 9, 1, 1, 1)
        grid_layout.addWidget(self.template_set_name_line_edit, 10, 0, 1, 2)

        # Shift save and clear buttons
        save_btn = self.main_window.ui.trainingSaveTemplatesPushButton
        clear_btn = self.main_window.ui.trainingClearExtractionPushButton
        grid_layout.removeWidget(save_btn)
        grid_layout.removeWidget(clear_btn)
        grid_layout.addWidget(save_btn, 11, 0, 1, 1)
        grid_layout.addWidget(clear_btn, 11, 1, 1, 1)

        # Shift progress bar
        progress_bar = self.main_window.ui.trainingExtractionProgressBar
        grid_layout.removeWidget(progress_bar)
        grid_layout.addWidget(progress_bar, 12, 0, 1, 2)

        # Save templates button
        self.save_templates_btn = self.main_window.ui.trainingSaveTemplatesPushButton
        self.save_templates_btn.clicked.connect(self._save_templates)
        self.save_templates_btn.setEnabled(False)

        # Open Template Review button (create programmatically)
        self.open_template_review_btn = QPushButton("Open Template Review")
        self.open_template_review_btn.clicked.connect(self._open_template_review)
        self.open_template_review_btn.setEnabled(False)
        # Add after save button in the grid
        grid_layout.addWidget(self.open_template_review_btn, 13, 0, 1, 2)

        # Template Study button
        self.template_study_btn = QPushButton("Template Study")
        self.template_study_btn.clicked.connect(self._open_template_study)
        self.template_study_btn.setEnabled(True)
        self.template_study_btn.setToolTip("Analyze template quality: distance matrix heatmap and intra/inter-class statistics.\nCan load templates from file if none are extracted.")
        grid_layout.addWidget(self.template_study_btn, 14, 0, 1, 1)

        # Load Templates from File button
        self.load_templates_file_btn = QPushButton("Load from File")
        self.load_templates_file_btn.clicked.connect(self._import_templates_from_file)
        self.load_templates_file_btn.setToolTip("Load a saved template set (.pkl) into the extraction panel.\nEnables save under a new name and model creation.")
        grid_layout.addWidget(self.load_templates_file_btn, 14, 1, 1, 1)

        # Clear button
        self.clear_extraction_btn = self.main_window.ui.trainingClearExtractionPushButton
        self.clear_extraction_btn.clicked.connect(self._clear_extraction)

        # Progress bar
        self.extraction_progress_bar = self.main_window.ui.trainingExtractionProgressBar
        self.extraction_progress_bar.setValue(0)

        # Set default data format to Guided Recording (now that all widgets are created)
        self.data_format_combo.blockSignals(True)
        self.data_format_combo.setCurrentIndex(2)  # Guided Recording
        self.data_format_combo.blockSignals(False)
        # Manually trigger initial UI update
        self._on_data_format_changed(2)

    def _on_data_format_changed(self, index: int) -> None:
        """Handle data format combo box change."""
        # Index 0: Auto-detect (single files)
        # Index 1: Legacy (EMG + GT folders)
        # Index 2: Guided Recording (Bidirectional)

        self._is_guided_mode = (index == 2)

        if index == 1:  # Legacy
            self.select_recordings_for_extraction_btn.setText("Select EMG Folder")
        elif index == 2:  # Guided Recording
            self.select_recordings_for_extraction_btn.setText("Select Guided Recording")
        else:  # Auto-detect
            self.select_recordings_for_extraction_btn.setText("Select Recording(s)")

        # Update Class combo based on data format
        if self._is_guided_mode:
            # Guided Recording: Class fixed to "Both", disable combo
            self.template_class_combo.setCurrentIndex(2)  # "Both"
            self.template_class_combo.setEnabled(False)
        else:
            # Auto-detect / Legacy: Enable class selection
            self.template_class_combo.setEnabled(True)
            if self.template_class_combo.currentIndex() == 2:  # If was "Both"
                self.template_class_combo.setCurrentIndex(0)  # Reset to "Open"

        # Show/hide UI elements based on mode
        # In guided mode, extraction happens directly in the recording selection
        # so we hide the manual extraction workflow elements
        show_extraction_workflow = not self._is_guided_mode

        self.extract_activations_btn.setVisible(show_extraction_workflow)
        self.activation_count_label.setVisible(show_extraction_workflow)
        self.selection_mode_combo.setVisible(show_extraction_workflow)
        self.main_window.ui.label_11.setVisible(show_extraction_workflow)  # "Selection Mode:" label
        self.activation_list_widget.setVisible(show_extraction_workflow)
        self.plot_selected_btn.setVisible(show_extraction_workflow)
        self.select_templates_btn.setVisible(show_extraction_workflow)
        self.plot_channel_spinbox.setVisible(show_extraction_workflow)
        self.plot_channel_label.setVisible(show_extraction_workflow)

        # Update Template Type options based on data format
        self._update_template_type_options()

        # Clear previous selections
        self._clear_extraction()

    def _update_template_type_options(self) -> None:
        """Update template type combo options based on current data format."""
        # Block signals to prevent triggering _on_template_type_changed during update
        self.template_type_combo.blockSignals(True)

        # Clear current items
        self.template_type_combo.clear()

        if self._is_guided_mode:
            # Guided Recording mode options
            for option in self._guided_template_types:
                self.template_type_combo.addItem(option)
        else:
            # Auto-detect / Legacy mode options
            for option in self._autodetect_template_types:
                self.template_type_combo.addItem(option)

        self.template_type_combo.blockSignals(False)

        # Trigger the change handler to update internal state
        self._on_template_type_changed(0)

    def _on_template_type_changed(self, index: int) -> None:
        """Handle template type combo box change."""
        if self._is_guided_mode:
            # Guided Recording mode options:
            # Index 0: "Manual Selection"
            # Index 1: "After Audio Cue"
            # Index 2: "After Reaction Time"
            # Index 3: "Onset (Amplitude)" - mean + k·std threshold
            # Index 4: "Onset (TKEO)"      - |d(TKEO)/dt| peak

            self._manual_selection_mode = (index in (0, 3, 4))

            if index == 0:
                self.template_manager.template_type = "manual"
            elif index == 1:
                self.template_manager.template_type = "after_audio_cue"
            elif index == 2:
                self.template_manager.template_type = "after_reaction_time"
            elif index == 3:
                self.template_manager.template_type = "onset_detection"
                self._onset_method = "amplitude"
            elif index == 4:
                self.template_manager.template_type = "onset_detection"
                self._onset_method = "tkeo"
        else:
            # Auto-detect / Legacy mode options:
            # Index 0: "Hold Only (skip 0.5s)" - hold_only mode
            # Index 1: "Onset + Hold (start -0.2s)" - onset_hold mode
            # Index 2: "Onset (GT=1 start)" - onset mode (start exactly at GT=1)
            # Index 3: "Manual Selection" - manual interactive mode

            self._manual_selection_mode = (index == 3)

            if index == 0:
                self.template_manager.set_template_type(include_onset=False)
            elif index == 1:
                self.template_manager.set_template_type(include_onset=True)
            elif index == 2:
                # Set to "onset" mode - starts exactly at GT=1
                self.template_manager.template_type = "onset"
            elif index == 3:
                # Manual mode - will be handled in extraction
                self.template_manager.template_type = "manual"

    def _on_template_duration_changed(self, index: int) -> None:
        """Handle template duration combo box change."""
        durations = [0.5, 1.0, 1.5, 2.0]
        self.template_manager.template_duration_s = durations[index]
        print(f"Template duration set to {durations[index]} seconds")

    def _on_selection_mode_changed(self, index: int) -> None:
        """Handle selection mode combo box change."""
        # Enable/disable manual selection based on mode
        is_manual = index == 0  # "Manual Review" is at index 0
        self.activation_list_widget.setEnabled(is_manual)

    def _on_activation_selection_changed(self) -> None:
        """Handle activation list selection change."""
        selected_count = len(self.activation_list_widget.selectedItems())
        self.select_templates_btn.setEnabled(selected_count > 0)
        self.plot_selected_btn.setEnabled(selected_count > 0)

    def _plot_selected_activations(self) -> None:
        """Plot the selected activations for visual inspection."""
        from mindmove.model.core.plotting.template_plots import plot_activation_with_template_markers

        class_label = self.template_class_combo.currentText().lower()

        # Get selected indices
        selected_indices = [
            self.activation_list_widget.row(item)
            for item in self.activation_list_widget.selectedItems()
        ]

        if not selected_indices:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "No activations selected!",
                QMessageBox.Ok,
            )
            return

        # Get activations and settings
        activations = self.template_manager.all_activations[class_label]
        template_type = self.template_manager.template_type
        template_duration = self.template_manager.template_duration_s
        channel = self.plot_channel_spinbox.value() - 1  # Convert from 1-indexed UI to 0-indexed

        # Plot each selected activation (full segment with template markers)
        n_selected = len(selected_indices)
        print(f"Plotting {n_selected} selected activations (channel {channel})...")

        for i, idx in enumerate(selected_indices):
            if idx < len(activations):
                activation = activations[idx]

                # Plot FULL activation with template boundary markers
                plot_activation_with_template_markers(
                    activation,
                    title=f"Activation {idx + 1}",
                    template_type=template_type,
                    template_duration_s=template_duration,
                    channel=channel,
                    save_path=None,
                    show=True
                )

    def _select_recordings_for_extraction(self) -> None:
        """Open file/folder dialog to select recordings for template extraction."""
        format_index = self.data_format_combo.currentIndex()

        if format_index == 1:  # Legacy
            self._select_legacy_folders()
        elif format_index == 2:  # Guided Recording (Bidirectional)
            self._select_guided_recording()
        else:  # MindMove format
            self._select_recording_files()

    def _select_guided_recording(self) -> None:
        """Select one or more guided recording files and extract templates."""
        if not os.path.exists(self.recordings_dir_path):
            os.makedirs(self.recordings_dir_path)

        dialog = QFileDialog(self.main_window)
        dialog.setFileMode(QFileDialog.ExistingFiles)  # Allow multiple selection
        dialog.setNameFilter("Pickle files (*.pkl)")
        dialog.setDirectory(self.recordings_dir_path)

        filenames, _ = dialog.getOpenFileNames()

        if not filenames:
            return

        # Load and validate all recordings
        valid_recordings = []
        try:
            for filename in filenames:
                with open(filename, "rb") as f:
                    recording = pickle.load(f)

                # Check if it's a guided recording or has GT data
                gt_mode = recording.get("gt_mode", "")
                if gt_mode != "guided_animation":
                    # Also accept keyboard recordings with GT
                    if "emg" not in recording or "gt" not in recording:
                        print(f"[TRAINING] Skipping invalid recording: {os.path.basename(filename)}")
                        print(f"  gt_mode: {gt_mode}, Keys: {list(recording.keys())}")
                        continue

                print(f"\n[TRAINING] Loaded recording: {os.path.basename(filename)}")
                print(f"  gt_mode: {gt_mode}")
                print(f"  EMG shape: {recording.get('emg', np.array([])).shape}")
                if "cycles_completed" in recording:
                    print(f"  Cycles: {recording['cycles_completed']}")

                valid_recordings.append(recording)

            if not valid_recordings:
                QMessageBox.warning(
                    self.main_window,
                    "Invalid Recordings",
                    "None of the selected files contain valid guided recordings with GT data.",
                    QMessageBox.Ok,
                )
                return

            # Check template type to decide extraction method
            template_type = self.template_manager.template_type

            if template_type == "manual":
                # Manual Selection: Open the review dialog
                print(f"\n[TRAINING] Opening review dialog with {len(valid_recordings)} recording(s)")

                # Store recordings for later review
                self._guided_recordings_for_review = valid_recordings

                review_dialog = GuidedRecordingReviewDialog(
                    valid_recordings, self.template_manager, self.main_window
                )
                result = review_dialog.exec()

                if review_dialog.saved:
                    # Update UI to show templates were saved
                    self.selected_recordings_for_extraction_label.setText(
                        f"Templates saved from {len(valid_recordings)} recording(s)"
                    )
                    # Update template counts
                    n_closed = len(self.template_manager.templates.get("closed", []))
                    n_open = len(self.template_manager.templates.get("open", []))
                    self.template_count_label.setText(f"Saved: {n_closed} closed, {n_open} open")
                    self.save_templates_btn.setEnabled(True)
                    self.open_template_review_btn.setEnabled(True)
                    self.template_study_btn.setEnabled(True)

            elif template_type == "onset_detection":
                # Onset detection: auto-extract, store for optional review
                print(f"\n[TRAINING] Running onset detection on {len(valid_recordings)} recording(s)")
                self._guided_recordings_for_review = valid_recordings
                self._extract_templates_from_onset(valid_recordings)

            elif template_type in ("after_audio_cue", "after_reaction_time"):
                # Automatic extraction based on cue positions
                self._extract_templates_from_cues(valid_recordings, template_type)
                # Store recordings for later review
                self._guided_recordings_for_review = valid_recordings

        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"Failed to load guided recording(s): {e}",
                QMessageBox.Ok,
            )
            import traceback
            traceback.print_exc()

    def _extract_templates_from_cues(self, recordings: List[dict], template_type: str) -> None:
        """
        Automatically extract templates based on audio cue or reaction time positions.

        Args:
            recordings: List of guided recording dicts
            template_type: "after_audio_cue" or "after_reaction_time"
        """
        # Clear existing templates before new extraction (don't accumulate)
        self.template_manager.templates["closed"] = []
        self.template_manager.templates["open"] = []

        template_duration_s = self.template_manager.template_duration_s
        template_samples = int(template_duration_s * config.FSAMP)

        open_templates = []
        closed_templates = []
        total_cycles = 0

        for recording in recordings:
            # Extract complete cycles from this recording
            cycles = self.template_manager.extract_complete_cycles(recording)

            for cycle in cycles:
                emg = cycle['emg']
                n_samples = emg.shape[1]

                # Get cue/start indices
                close_cue_idx = cycle.get('close_cue_idx')
                open_cue_idx = cycle.get('open_cue_idx')
                close_start_idx = cycle.get('close_start_idx')  # GT transition (after reaction time)
                open_start_idx = cycle.get('open_start_idx')    # GT transition (after reaction time)

                # Determine extraction start points based on template type
                if template_type == "after_audio_cue":
                    closed_start = close_cue_idx
                    open_start = open_cue_idx
                    mode_label = "audio cue"
                else:  # after_reaction_time
                    closed_start = close_start_idx
                    open_start = open_start_idx
                    mode_label = "reaction time"

                # Extract CLOSED template (at close cue/start)
                if closed_start is not None:
                    end_idx = closed_start + template_samples
                    if end_idx <= n_samples:
                        template = emg[:, closed_start:end_idx]
                        closed_templates.append(template)
                        print(f"  Cycle {cycle['cycle_number']}: CLOSED template from {mode_label} "
                              f"({closed_start/config.FSAMP:.2f}s - {end_idx/config.FSAMP:.2f}s)")
                    else:
                        print(f"  Cycle {cycle['cycle_number']}: CLOSED template would exceed cycle bounds, skipping")
                else:
                    print(f"  Cycle {cycle['cycle_number']}: No {mode_label} index for CLOSED")

                # Extract OPEN template (at open cue/start)
                if open_start is not None:
                    end_idx = open_start + template_samples
                    if end_idx <= n_samples:
                        template = emg[:, open_start:end_idx]
                        open_templates.append(template)
                        print(f"  Cycle {cycle['cycle_number']}: OPEN template from {mode_label} "
                              f"({open_start/config.FSAMP:.2f}s - {end_idx/config.FSAMP:.2f}s)")
                    else:
                        print(f"  Cycle {cycle['cycle_number']}: OPEN template would exceed cycle bounds, skipping")
                else:
                    print(f"  Cycle {cycle['cycle_number']}: No {mode_label} index for OPEN")

                total_cycles += 1

        # Store extracted templates (replace, don't accumulate)
        self.template_manager.templates["closed"] = closed_templates
        self.template_manager.templates["open"] = open_templates

        # Update UI
        n_closed = len(closed_templates)
        n_open = len(open_templates)

        print(f"\n[TRAINING] Extracted {n_closed} CLOSED and {n_open} OPEN templates from {total_cycles} cycles")
        print(f"  Template type: {template_type}")
        print(f"  Template duration: {template_duration_s}s ({template_samples} samples)")

        self.selected_recordings_for_extraction_label.setText(
            f"Extracted from {len(recordings)} recording(s)"
        )
        self.template_count_label.setText(f"Extracted: {n_closed} closed, {n_open} open")

        # Enable save and review buttons
        if n_closed > 0 or n_open > 0:
            self.save_templates_btn.setEnabled(True)
            self.open_template_review_btn.setEnabled(True)
            self.template_study_btn.setEnabled(True)

    @staticmethod
    def _print_onset_diagnostics(result: dict, class_name: str, cycle_num: int) -> None:
        """Print per-channel diagnostics when onset detection fails."""
        env_data = result["envelope_data"]
        per_ch_env = env_data["per_channel_env"]
        env_time = env_data["env_time"]
        baselines = env_data["per_channel_baseline"]
        thresholds = env_data["per_channel_threshold"]
        search_start_s = env_data["search_start_s"]
        search_end_s = env_data["search_end_s"]

        from mindmove.model.template_study import ONSET_ENVELOPE_WINDOW_S
        env_step = int(ONSET_ENVELOPE_WINDOW_S * config.FSAMP) // 2
        search_start_env = max(0, int(search_start_s * config.FSAMP) // env_step)
        search_end_env = min(len(env_time), int(search_end_s * config.FSAMP) // env_step)

        n_ch = per_ch_env.shape[0]
        print(f"           Diagnostics for {class_name} (search: {search_start_s:.2f}-{search_end_s:.2f}s):")

        # Find channels with highest peak-to-threshold ratio
        ratios = []
        for ch in range(n_ch):
            search_seg = per_ch_env[ch, search_start_env:search_end_env]
            if len(search_seg) == 0:
                ratios.append((ch, 0, baselines[ch], thresholds[ch], 0))
                continue
            peak = float(np.max(search_seg))
            thr = thresholds[ch]
            ratio = peak / thr if thr > 0 else 0
            ratios.append((ch, peak, baselines[ch], thr, ratio))

        # Sort by ratio descending, show top 5 closest to firing
        ratios.sort(key=lambda x: x[4], reverse=True)
        for ch, peak, bl, thr, ratio in ratios[:5]:
            status = "CLOSE" if ratio > 0.7 else "far"
            print(f"           CH{ch+1}: baseline={bl:.4f}, threshold={thr:.4f}, "
                  f"peak={peak:.4f}, peak/thr={ratio:.2f} [{status}]")

    def _extract_templates_from_onset(self, recordings: List[dict]) -> None:
        """
        Extract templates using per-channel onset detection.
        Templates are placed at onset - 20% preparation, clamped to valid region.
        Stores onset info (channels fired) for use in review dialog.
        """
        from mindmove.model.template_study import (
            detect_onset_per_channel,
            detect_transition_onset,
            place_template_at_onset,
            detect_dead_channels,
            detect_artifact_channels,
            ONSET_ANTICIPATORY_S,
            ONSET_MAX_POST_CUE_S,
            ONSET_BASELINE_DURATION_S,
        )

        method = getattr(self, "_onset_method", "amplitude")
        print(f"\n[TRAINING] Onset detection method: {method}")

        self.template_manager.templates["closed"] = []
        self.template_manager.templates["open"] = []

        template_duration_s = self.template_manager.template_duration_s
        template_samples = int(template_duration_s * config.FSAMP)
        anticipatory_samples = int(ONSET_ANTICIPATORY_S * config.FSAMP)
        post_cue_samples = int(ONSET_MAX_POST_CUE_S * config.FSAMP)
        baseline_samples = int(ONSET_BASELINE_DURATION_S * config.FSAMP)

        closed_templates = []
        open_templates = []
        closed_meta = []
        open_meta = []
        total_cycles = 0

        # Store onset info for review dialog
        self._onset_info = []  # list of dicts per cycle

        for rec_idx, recording in enumerate(recordings):
            # pre_close_s=3.0 gives up to 3s lookback before the cue, covering
            # anticipatory movements where half of even a 6s hold is 3s
            cycles = self.template_manager.extract_complete_cycles(
                recording, pre_close_s=3.0, post_open_s=2.0
            )
            rec_label = recording.get('label', f'Recording {rec_idx+1}')
            for c in cycles:
                c['source_recording'] = rec_label

            for cycle in cycles:
                emg = cycle["emg"]
                n_samples = emg.shape[1]
                cn = cycle["cycle_number"]

                cue_closed = cycle.get("close_cue_idx") or cycle.get("close_start_idx", 0)
                cue_open = cycle.get("open_cue_idx") or cycle.get("open_start_idx", 0)

                if method == "tkeo":
                    # TKEO: narrow window around cue so argmax cannot reach opposite transition
                    closed_search_start = max(0, cue_closed - anticipatory_samples)
                    closed_search_end = min(cue_closed + post_cue_samples,
                                            cue_open - template_samples)
                    closed_search_end = max(closed_search_start + 1, closed_search_end)
                    open_search_start = max(0, cue_open - anticipatory_samples)
                    open_search_end = min(cue_open + post_cue_samples,
                                          n_samples - template_samples)
                    open_search_end = max(open_search_start + 1, open_search_end)
                    # Baselines used only for artifact detection
                    closed_baseline_start = max(0, closed_search_start - baseline_samples)
                    open_baseline_start = max(0, open_search_start - baseline_samples)
                else:
                    # Amplitude threshold: extend search backward to catch anticipatory movements.
                    # Baseline is placed in the quiet rest BEFORE the search window.
                    closed_search_start = max(0, cue_closed - anticipatory_samples)
                    closed_search_end = max(closed_search_start + 1, cue_open - template_samples)
                    open_search_start = max(0, cue_open - anticipatory_samples)
                    open_search_end = max(open_search_start + 1, n_samples - template_samples)
                    closed_baseline_start = max(0, closed_search_start - baseline_samples)
                    open_baseline_start = max(0, open_search_start - baseline_samples)

                cycle_info = {
                    "closed_channels_fired": [],
                    "open_channels_fired": [],
                    "closed_pos": None,
                    "open_pos": None,
                    "dead_channels": [],
                    "artifact_channels_closed": [],
                    "artifact_channels_open": [],
                }

                # Dead channel detection (whole cycle)
                cycle_info["dead_channels"] = detect_dead_channels(emg)
                if cycle_info["dead_channels"]:
                    print(f"  Cycle {cn} DEAD channels: {[ch+1 for ch in cycle_info['dead_channels']]}")

                # CLOSED onset
                if method == "tkeo":
                    result = detect_transition_onset(emg, closed_search_start, closed_search_end)
                else:
                    result = detect_onset_per_channel(emg, closed_search_start, closed_search_end,
                                                      baseline_start=closed_baseline_start)
                if result["earliest_onset"] is not None:
                    pos = place_template_at_onset(result["earliest_onset"], n_samples)
                    pos = max(pos, closed_search_start)
                    pos = min(pos, n_samples - template_samples)
                    end_idx = pos + template_samples
                    closed_templates.append(emg[:, pos:end_idx])
                    closed_meta.append({
                        "id": len(closed_templates),  # stable 1-based ID assigned at extraction
                        "recording": cycle.get("source_recording", f"Rec{rec_idx+1}"),
                        "cycle": cn,
                    })
                    cycle_info["closed_channels_fired"] = result["channels_fired"]
                    # Store as offset from GT close transition so it can be
                    # re-applied to the viewer cycle (different pre_close_s)
                    cycle_info["closed_pos"] = pos - cue_closed
                    fired = ",".join(str(ch+1) for ch in result["channels_fired"])
                    print(f"  Cycle {cn} CLOSED: onset={result['earliest_onset']/config.FSAMP:.2f}s, "
                          f"window={pos/config.FSAMP:.2f}s-{end_idx/config.FSAMP:.2f}s, ch=[{fired}]")

                    # Artifact detection for CLOSED template
                    art = detect_artifact_channels(
                        emg, pos, end_idx,
                        closed_baseline_start, closed_search_start,
                    )
                    cycle_info["artifact_channels_closed"] = art["artifact_channels"]
                    if art["artifact_channels"]:
                        art_str = ",".join(str(ch+1) for ch in art["artifact_channels"])
                        print(f"           CLOSED artifacts: CH[{art_str}]")
                else:
                    print(f"  Cycle {cn} CLOSED: no onset detected — skipped")

                # OPEN onset
                if method == "tkeo":
                    result = detect_transition_onset(emg, open_search_start, open_search_end)
                else:
                    result = detect_onset_per_channel(emg, open_search_start, open_search_end,
                                                      baseline_start=open_baseline_start)
                if result["earliest_onset"] is not None:
                    pos = place_template_at_onset(result["earliest_onset"], n_samples)
                    pos = max(pos, open_search_start)
                    pos = min(pos, n_samples - template_samples)
                    end_idx = pos + template_samples
                    open_templates.append(emg[:, pos:end_idx])
                    open_meta.append({
                        "id": len(open_templates),  # stable 1-based ID assigned at extraction
                        "recording": cycle.get("source_recording", f"Rec{rec_idx+1}"),
                        "cycle": cn,
                    })
                    cycle_info["open_channels_fired"] = result["channels_fired"]
                    # Store as offset from GT open transition (same convention as closed_pos)
                    cycle_info["open_pos"] = pos - cue_open
                    fired = ",".join(str(ch+1) for ch in result["channels_fired"])
                    print(f"  Cycle {cn} OPEN:   onset={result['earliest_onset']/config.FSAMP:.2f}s, "
                          f"window={pos/config.FSAMP:.2f}s-{end_idx/config.FSAMP:.2f}s, ch=[{fired}]")

                    # Artifact detection for OPEN template
                    art = detect_artifact_channels(
                        emg, pos, end_idx,
                        open_baseline_start, open_search_start,
                    )
                    cycle_info["artifact_channels_open"] = art["artifact_channels"]
                    if art["artifact_channels"]:
                        art_str = ",".join(str(ch+1) for ch in art["artifact_channels"])
                        print(f"           OPEN artifacts: CH[{art_str}]")
                else:
                    print(f"  Cycle {cn} OPEN:   no onset detected — skipped")

                self._onset_info.append(cycle_info)
                total_cycles += 1

        self.template_manager.templates["closed"] = closed_templates
        self.template_manager.templates["open"] = open_templates
        self.template_manager.template_metadata["closed"] = closed_meta
        self.template_manager.template_metadata["open"] = open_meta

        n_closed = len(closed_templates)
        n_open = len(open_templates)
        print(f"\n[ONSET] Extracted {n_closed} CLOSED and {n_open} OPEN templates from {total_cycles} cycles")

        self.selected_recordings_for_extraction_label.setText(
            f"Onset detection: {len(recordings)} recording(s)"
        )
        self.template_count_label.setText(f"Extracted: {n_closed} closed, {n_open} open")

        if n_closed > 0 or n_open > 0:
            self.save_templates_btn.setEnabled(True)
            self.open_template_review_btn.setEnabled(True)
            self.template_study_btn.setEnabled(True)

        # Auto-compute per-class channels from onset info
        if self._onset_info:
            from mindmove.model.template_study import compute_per_class_channels
            dead_channels = self._parse_dead_channels()
            result = compute_per_class_channels(self._onset_info, recordings, dead_channels)
            self._per_class_channels = result

            closed_str = ",".join(str(ch + 1) for ch in result["closed"])
            open_str = ",".join(str(ch + 1) for ch in result["open"])
            self.closed_channels_input.setText(closed_str)
            self.open_channels_input.setText(open_str)

            print(f"\n[PER-CLASS CHANNELS] Auto-computed:")
            print(f"  CLOSED ({len(result['closed'])}): [{closed_str}]")
            print(f"  OPEN ({len(result['open'])}): [{open_str}]")
            if result["unassigned"]:
                print(f"  Unassigned ({len(result['unassigned'])}): "
                      f"[{','.join(str(ch+1) for ch in result['unassigned'])}]")

            # Enable auto-detect button if in per-class mode
            self.auto_detect_channels_btn.setEnabled(True)

    def _select_recording_files(self) -> None:
        """Select recording files (auto-detect format). Supports .pkl and .mat files."""
        if not os.path.exists(self.recordings_dir_path):
            os.makedirs(self.recordings_dir_path)

        dialog = QFileDialog(self.main_window)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilter("Recording files (*.pkl *.mat);;Pickle files (*.pkl);;MAT files (*.mat)")
        dialog.setDirectory(self.recordings_dir_path)

        filenames, _ = dialog.getOpenFileNames()

        if not filenames:
            return

        # Validate selected recordings (accept MindMove, VHI pickle, and MAT formats)
        valid_recordings = []
        mode_validated = False  # Only show mode dialog once
        for filepath in filenames:
            try:
                recording = self._load_recording_file(filepath)
                if recording is None:
                    continue

                # Check for MindMove virtual hand format
                mindmove_vh_keys = ["emg", "kinematics"]
                # Check for MindMove keyboard format
                mindmove_kb_keys = ["emg", "gt"]
                # Check for VHI format
                vhi_keys = ["biosignal", "ground_truth"]

                is_valid = (all(key in recording for key in mindmove_vh_keys) or
                            all(key in recording for key in mindmove_kb_keys) or
                            all(key in recording for key in vhi_keys))

                if is_valid:
                    # Validate differential mode (once, from first valid recording)
                    if not mode_validated:
                        rec_mode = _detect_mode_from_data(recording)
                        if rec_mode is not None:
                            if not self._validate_differential_mode(rec_mode):
                                return  # User declined switch — abort entire selection
                        mode_validated = True
                    valid_recordings.append(filepath)
                else:
                    print(f"Invalid recording (unknown format): {filepath}")
                    print(f"  Keys found: {list(recording.keys())}")
            except Exception as e:
                print(f"Error loading recording {filepath}: {e}")
                import traceback
                traceback.print_exc()

        if not valid_recordings:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "No valid recordings selected!",
                QMessageBox.Ok,
            )
            return

        self.selected_extraction_recordings = valid_recordings
        self.selected_recordings_for_extraction_label.setText(
            f"{len(valid_recordings)} recording(s) selected"
        )
        self.extract_activations_btn.setEnabled(True)

    def _load_recording_file(self, filepath: str) -> Optional[dict]:
        """
        Load a recording file, supporting both .pkl and .mat formats.

        Args:
            filepath: Path to the recording file

        Returns:
            Recording dictionary or None if loading failed
        """
        if filepath.lower().endswith('.mat'):
            # Load MAT file
            try:
                import scipy.io as sio
                mat_data = sio.loadmat(filepath)

                # Convert to standard dictionary format
                recording = {}
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        val = mat_data[key]
                        # Handle MATLAB string arrays
                        if val.dtype.kind == 'U' or val.dtype.kind == 'S':
                            recording[key] = str(val.flat[0]) if val.size > 0 else ""
                        # Handle scalar values wrapped in arrays
                        elif val.size == 1:
                            recording[key] = val.flat[0]
                        else:
                            recording[key] = val

                print(f"Loaded MAT file: {filepath}")
                print(f"  Keys: {[k for k in recording.keys() if not k.startswith('_')]}")
                return recording

            except ImportError:
                print("scipy is required to load .mat files. Install with: pip install scipy")
                return None
            except Exception as e:
                print(f"Error loading MAT file {filepath}: {e}")
                return None

        else:
            # Load pickle file
            with open(filepath, "rb") as f:
                recording = pickle.load(f)
            return recording

    def _select_legacy_folders(self) -> None:
        """Select EMG and GT folders for legacy format."""
        if not os.path.exists(self.legacy_data_path):
            os.makedirs(self.legacy_data_path)

        # First, select EMG folder
        emg_folder = QFileDialog.getExistingDirectory(
            self.main_window,
            "Select EMG Folder",
            self.legacy_data_path,
            QFileDialog.ShowDirsOnly
        )

        if not emg_folder:
            return

        # Then, select GT folder
        gt_folder = QFileDialog.getExistingDirectory(
            self.main_window,
            "Select Ground Truth Folder",
            self.legacy_data_path,
            QFileDialog.ShowDirsOnly
        )

        if not gt_folder:
            return

        # Validate folders have .pkl files
        emg_files = [f for f in os.listdir(emg_folder) if f.endswith('.pkl')]
        gt_files = [f for f in os.listdir(gt_folder) if f.endswith('.pkl')]

        if not emg_files:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                f"No .pkl files found in EMG folder:\n{emg_folder}",
                QMessageBox.Ok,
            )
            return

        if not gt_files:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                f"No .pkl files found in GT folder:\n{gt_folder}",
                QMessageBox.Ok,
            )
            return

        self.legacy_emg_folder = emg_folder
        self.legacy_gt_folder = gt_folder
        self.selected_recordings_for_extraction_label.setText(
            f"EMG: {len(emg_files)} files, GT: {len(gt_files)} files"
        )
        self.extract_activations_btn.setEnabled(True)

    def _extract_activations(self) -> None:
        """Extract activations from selected recordings."""
        is_legacy = self.data_format_combo.currentIndex() == 1

        # Validate selection based on format
        if is_legacy:
            if not self.legacy_emg_folder or not self.legacy_gt_folder:
                QMessageBox.warning(
                    self.main_window,
                    "Warning",
                    "No folders selected! Please select EMG and GT folders.",
                    QMessageBox.Ok,
                )
                return
        else:
            if not self.selected_extraction_recordings:
                QMessageBox.warning(
                    self.main_window,
                    "Warning",
                    "No recordings selected!",
                    QMessageBox.Ok,
                )
                return

        # Disable UI during extraction
        self.extract_activations_btn.setEnabled(False)
        self.select_recordings_for_extraction_btn.setEnabled(False)

        # Get current class label
        class_label = self.template_class_combo.currentText().lower()

        # Clear previous activations for this class
        self.template_manager.clear_activations(class_label)

        # Start extraction thread
        self.extract_activations_thread = PyQtThread(
            target=lambda: self._extract_activations_thread(class_label, is_legacy),
            parent=self.main_window
        )
        self.extract_activations_thread.has_finished_signal.connect(
            self._extract_activations_finished
        )
        self.extract_activations_thread.start()

    def _extract_activations_thread(self, class_label: str, is_legacy: bool = False) -> None:
        """Thread function to extract activations from recordings."""
        # Check if we need to include pre-activation samples
        # Include them for onset_hold mode OR for manual mode (to show context)
        include_pre_activation = (
            self.template_manager.template_type == "onset_hold" or
            self._manual_selection_mode
        )

        if is_legacy:
            # Load legacy format (separate EMG + GT folders)
            try:
                print(f"Loading legacy format from:")
                print(f"  EMG folder: {self.legacy_emg_folder}")
                print(f"  GT folder: {self.legacy_gt_folder}")

                recordings = TemplateManager.load_legacy_format(
                    self.legacy_emg_folder,
                    self.legacy_gt_folder
                )

                print(f"Loaded {len(recordings)} recordings from legacy format")

                for i, recording in enumerate(recordings):
                    rec_name = recording.get("_legacy_filename", f"legacy_{i+1}")
                    # Extract activations from this recording
                    self.template_manager.extract_activations_from_recording(
                        recording,
                        class_label,
                        include_pre_activation=include_pre_activation,
                        recording_name=rec_name
                    )
                    progress = int((i + 1) / len(recordings) * 100)
                    print(f"Extraction progress: {progress}%")

            except Exception as e:
                print(f"Error loading legacy format: {e}")
                import traceback
                traceback.print_exc()

        else:
            # Load single-file formats (MindMove, VHI pickle, or MAT)
            total_recordings = len(self.selected_extraction_recordings)

            for i, filepath in enumerate(self.selected_extraction_recordings):
                try:
                    # Use the unified loader that handles both .pkl and .mat
                    recording = self._load_recording_file(filepath)
                    if recording is None:
                        print(f"Failed to load: {filepath}")
                        continue

                    rec_name = os.path.basename(filepath)
                    # Extract activations from this recording
                    self.template_manager.extract_activations_from_recording(
                        recording,
                        class_label,
                        include_pre_activation=include_pre_activation,
                        recording_name=rec_name
                    )

                    # Update progress
                    progress = int((i + 1) / total_recordings * 100)
                    print(f"Extraction progress: {progress}%")

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    import traceback
                    traceback.print_exc()

    def _extract_activations_finished(self) -> None:
        """Called when activation extraction is complete."""
        # Re-enable UI
        self.extract_activations_btn.setEnabled(True)
        self.select_recordings_for_extraction_btn.setEnabled(True)
        self.extraction_progress_bar.setValue(0)

        # Get current class label
        class_label = self.template_class_combo.currentText().lower()

        # Update activation count label
        activation_count = self.template_manager.get_activation_count(class_label)
        self.activation_count_label.setText(f"{activation_count} activations found")

        if activation_count == 0:
            return

        # Check if manual selection mode is enabled
        if self._manual_selection_mode:
            # Launch interactive manual selection for each activation
            self._start_manual_template_selection(class_label)
        else:
            # Populate activation list widget for automatic modes
            self._populate_activation_list(class_label)

            # Enable select templates button if we have activations
            if activation_count > 0:
                self.select_templates_btn.setEnabled(True)

    def _populate_activation_list(self, class_label: str) -> None:
        """Populate the activation list widget with extracted activations."""
        self.activation_list_widget.clear()

        durations = self.template_manager.get_activation_durations(class_label)
        metadata_list = self.template_manager.activation_metadata.get(class_label, [])

        for i, duration in enumerate(durations):
            if i < len(metadata_list):
                meta = metadata_list[i]
                rec_name = meta.get("recording_name", "")
                cycle_idx = meta.get("cycle_index", i + 1)
                total = meta.get("total_cycles_in_recording", 0)
                if rec_name:
                    # Shorten long filenames: keep last part
                    short_name = rec_name
                    if len(short_name) > 40:
                        short_name = "..." + short_name[-37:]
                    label = f"[{short_name}] Cycle {cycle_idx}/{total} — {duration:.2f}s"
                else:
                    label = f"Activation {i + 1}: {duration:.2f}s"
            else:
                label = f"Activation {i + 1}: {duration:.2f}s"
            item = QListWidgetItem(label)
            self.activation_list_widget.addItem(item)

    def _start_manual_template_selection(self, class_label: str) -> None:
        """
        Start interactive manual template selection for all activations.

        Opens a matplotlib plot for each activation where user can click
        to set the template start position.
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Cursor

        activations = self.template_manager.all_activations[class_label]
        template_duration_s = self.template_manager.template_duration_s
        template_samples = self.template_manager.template_nsamp
        channel = self.plot_channel_spinbox.value() - 1  # Convert to 0-indexed

        # Clear any previous manual templates
        self._manual_templates = []
        self.template_manager.templates[class_label] = []

        print(f"\n{'='*60}")
        print(f"MANUAL TEMPLATE SELECTION - {len(activations)} activations")
        print(f"Template duration: {template_duration_s}s ({template_samples} samples)")
        print(f"Click to set template start, close window to skip")
        print(f"{'='*60}\n")

        # Process each activation
        metadata_list = self.template_manager.activation_metadata.get(class_label, [])
        for i, activation in enumerate(activations):
            meta_str = ""
            if i < len(metadata_list):
                meta = metadata_list[i]
                rec_name = meta.get("recording_name", "")
                cycle_idx = meta.get("cycle_index", 0)
                if rec_name:
                    meta_str = f" [{rec_name} — Cycle {cycle_idx}]"
            print(f"\nProcessing activation {i + 1}/{len(activations)}{meta_str}...")

            # Get the GT signal for this activation (reconstruct from context)
            # For manual mode, we include manual_context_before_s before GT=1
            n_samples = activation.shape[1]
            pre_samples = int(self.template_manager.manual_context_before_s * config.FSAMP)

            # Build a simple GT overlay: 0 before GT=1, 1 after
            # The activation starts with pre_samples of GT=0, then GT=1
            gt_overlay = np.zeros(n_samples)
            if pre_samples < n_samples:
                gt_overlay[pre_samples:] = 1

            # Build activation title with metadata
            act_title = f"Activation {i + 1}/{len(activations)}"
            if i < len(metadata_list) and metadata_list[i].get("recording_name"):
                meta = metadata_list[i]
                act_title = (f"[{meta['recording_name']}] "
                            f"Cycle {meta['cycle_index']} — "
                            f"{i + 1}/{len(activations)}")

            # Call the interactive selection for this activation
            template = self._interactive_template_selection(
                activation,
                gt_overlay,
                channel,
                template_samples,
                activation_idx=i + 1,
                total_activations=len(activations),
                activation_title=act_title
            )

            if template is not None:
                self._manual_templates.append(template)
                self.template_manager.templates[class_label].append(template)
                print(f"  Template {len(self._manual_templates)} captured!")

        # Update UI after manual selection is complete
        self._manual_selection_finished(class_label)

    def _interactive_template_selection(
        self,
        activation: np.ndarray,
        gt_overlay: np.ndarray,
        channel: int,
        template_samples: int,
        activation_idx: int,
        total_activations: int,
        activation_title: str = ""
    ) -> Optional[np.ndarray]:
        """
        Show interactive plot for a single activation and let user click to select template start.

        Uses a Qt dialog with embedded matplotlib canvas to work properly with Qt event loop.

        Args:
            activation: EMG data (n_channels, n_samples)
            gt_overlay: Ground truth signal at same sample rate
            channel: Which channel to display (0-indexed)
            template_samples: Number of samples for template
            activation_idx: Current activation number (1-indexed for display)
            total_activations: Total number of activations
            activation_title: Optional descriptive title with recording name/cycle info

        Returns:
            Template array or None if skipped
        """
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
        from PySide6.QtCore import Qt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle

        n_samples = activation.shape[1]
        time_axis = np.arange(n_samples) / config.FSAMP

        # Find GT=1 start (rising edge)
        gt_start_idx = np.argmax(gt_overlay > 0.5)
        gt_start_time = gt_start_idx / config.FSAMP

        # Show full activation segment
        display_start_s = max(0, gt_start_time - 2.0)
        display_end_s = min(time_axis[-1], gt_start_time + 3.0)
        if display_end_s - display_start_s < 4.0:
            display_end_s = min(time_axis[-1], display_start_s + 5.0)

        # Get EMG signal and normalize
        emg_signal = activation[channel, :]
        emg_normalized = emg_signal / (np.max(np.abs(emg_signal)) + 1e-10)

        # Create dialog
        dialog = QDialog(self.main_window)
        title = activation_title or f"Activation {activation_idx}/{total_activations}"
        dialog.setWindowTitle(f'Manual Selection - {title}')
        dialog.setMinimumSize(1000, 600)
        dialog.setModal(True)

        # Store selection state
        selection_state = {'start_time': None, 'vline': None, 'rect': None, 'confirmed': False}

        # Create matplotlib figure
        fig = Figure(figsize=(12, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Plot EMG signal
        ax.plot(time_axis, emg_normalized, 'b-', linewidth=0.8, label=f'EMG Ch{channel + 1}')

        # Plot GT overlay
        ax.fill_between(time_axis, -1, 1, where=gt_overlay > 0.5,
                        alpha=0.2, color='green', label='GT=1 (Activation)')

        # Mark GT=1 start
        ax.axvline(x=gt_start_time, color='green', linestyle='--', linewidth=2,
                   label=f'GT=1 start ({gt_start_time:.2f}s)')

        # Set axis limits
        ax.set_xlim(display_start_s, display_end_s)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized EMG')
        ax.set_title(f'Activation {activation_idx}/{total_activations} - Click to set template start\n'
                     f'(Template duration: {template_samples/config.FSAMP:.2f}s)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        def on_click(event):
            """Handle mouse click to set template start."""
            if event.inaxes != ax:
                return
            if event.button != 1:
                return

            click_time = event.xdata
            if click_time is None:
                return

            click_idx = int(click_time * config.FSAMP)

            if click_idx + template_samples > n_samples:
                print(f"  Warning: Not enough samples after click position. Try clicking earlier.")
                return

            # Remove previous markers
            if selection_state['vline'] is not None:
                selection_state['vline'].remove()
            if selection_state['rect'] is not None:
                selection_state['rect'].remove()

            # Draw new markers
            selection_state['vline'] = ax.axvline(x=click_time, color='red', linestyle='-',
                                                   linewidth=2)
            template_end_time = click_time + template_samples / config.FSAMP
            selection_state['rect'] = ax.add_patch(
                Rectangle((click_time, -1.2), template_samples / config.FSAMP, 2.4,
                          facecolor='red', alpha=0.2, edgecolor='red', linewidth=2)
            )

            selection_state['start_time'] = click_time
            ax.set_title(f'Activation {activation_idx}/{total_activations} - Template: {click_time:.2f}s to {template_end_time:.2f}s\n'
                         f'Click "Confirm" or click again to adjust')
            canvas.draw()
            confirm_btn.setEnabled(True)
            status_label.setText(f"Selected: {click_time:.2f}s to {template_end_time:.2f}s")

        canvas.mpl_connect('button_press_event', on_click)

        # Create layout
        layout = QVBoxLayout(dialog)
        layout.addWidget(canvas)

        # Status label
        status_label = QLabel("Click on the plot to set template start position")
        status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(status_label)

        # Buttons
        btn_layout = QHBoxLayout()

        confirm_btn = QPushButton("Confirm Selection")
        confirm_btn.setEnabled(False)
        confirm_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px 16px;")
        def on_confirm():
            selection_state['confirmed'] = True
            dialog.accept()
        confirm_btn.clicked.connect(on_confirm)

        skip_btn = QPushButton("Skip This Activation")
        skip_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px 16px;")
        skip_btn.clicked.connect(dialog.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(confirm_btn)
        btn_layout.addWidget(skip_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Show dialog (blocking)
        result = dialog.exec()

        # Extract template if confirmed
        if selection_state['confirmed'] and selection_state['start_time'] is not None:
            start_idx = int(selection_state['start_time'] * config.FSAMP)
            end_idx = start_idx + template_samples
            if end_idx <= n_samples:
                return activation[:, start_idx:end_idx]

        return None

    def _manual_selection_finished(self, class_label: str) -> None:
        """Called when manual selection is complete."""
        template_count = len(self._manual_templates)
        target_count = config.TARGET_TEMPLATES_PER_CLASS

        print(f"\n{'='*60}")
        print(f"MANUAL SELECTION COMPLETE")
        print(f"Templates captured: {template_count}")
        print(f"{'='*60}\n")

        # Update UI
        self.activation_count_label.setText(f"{template_count} templates manually selected")
        self.template_count_label.setText(f"{template_count}/{target_count} templates")

        # Populate list with selected templates
        self.activation_list_widget.clear()
        for i in range(template_count):
            template = self._manual_templates[i]
            duration = template.shape[1] / config.FSAMP
            item = QListWidgetItem(f"Template {i + 1}: {duration:.2f}s")
            self.activation_list_widget.addItem(item)

        # Enable save button if we have templates
        if template_count > 0:
            self.save_templates_btn.setEnabled(True)
            self.plot_selected_btn.setEnabled(True)

        QMessageBox.information(
            self.main_window,
            "Manual Selection Complete",
            f"Manually selected {template_count} templates for class '{class_label}'.\n\n"
            f"You can plot them to review, then click 'Save Templates' when ready.",
            QMessageBox.Ok,
        )

    def _select_templates(self) -> None:
        """Apply selection mode to choose templates."""
        class_label = self.template_class_combo.currentText().lower()
        selection_mode = self.selection_mode_combo.currentText()
        target_count = config.TARGET_TEMPLATES_PER_CLASS

        if "Manual" in selection_mode:
            # Get selected items from list widget
            selected_indices = [
                self.activation_list_widget.row(item)
                for item in self.activation_list_widget.selectedItems()
            ]
            if not selected_indices:
                QMessageBox.warning(
                    self.main_window,
                    "Warning",
                    "No activations selected! Please select activations from the list.",
                    QMessageBox.Ok,
                )
                return
            self.template_manager.select_templates_manual(selected_indices, class_label)

        elif "Auto" in selection_mode:
            # Auto-select longest activations
            self.template_manager.select_templates_auto(class_label, n=target_count)

        else:  # "First 20"
            # Select first n activations
            self.template_manager.select_templates_first_n(class_label, n=target_count)

        # Update template count label
        template_count = self.template_manager.get_template_count(class_label)
        self.template_count_label.setText(f"{template_count}/{target_count} templates")

        # Enable save button if we have templates
        if template_count > 0:
            self.save_templates_btn.setEnabled(True)

        QMessageBox.information(
            self.main_window,
            "Templates Selected",
            f"Selected {template_count} templates for class '{class_label}'.",
            QMessageBox.Ok,
        )

    def _save_templates(self) -> None:
        """Save selected raw templates to disk."""
        # Check if we're in guided mode with both classes
        is_guided = getattr(self, '_is_guided_mode', False)
        templates_closed = self.template_manager.templates.get("closed", [])
        templates_open = self.template_manager.templates.get("open", [])

        # For guided mode or when we have both classes, save combined
        if is_guided or (templates_closed and templates_open):
            self._save_templates_combined()
            return

        # Original single-class save logic
        class_label = self.template_class_combo.currentText().lower()
        template_count = self.template_manager.get_template_count(class_label)

        if template_count == 0:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "No templates to save! Please select templates first.",
                QMessageBox.Ok,
            )
            return

        # Get optional template set name
        template_set_name = self.template_set_name_line_edit.text().strip()
        if not template_set_name:
            template_set_name = None

        # Save raw templates only (feature extraction happens in Train Model section)
        try:
            save_path = self.template_manager.save_templates(
                class_label,
                template_set_name=template_set_name
            )

            duration = self.template_manager.template_duration_s
            QMessageBox.information(
                self.main_window,
                "Templates Saved",
                f"Saved {template_count} raw templates ({duration}s each) to:\n{save_path}\n\n"
                f"Note: Feature extraction will happen when creating the model.",
                QMessageBox.Ok,
            )

        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"Failed to save templates: {e}",
                QMessageBox.Ok,
            )

    def _save_templates_combined(self) -> None:
        """Save both OPEN and CLOSED templates in a single file (for guided mode)."""
        templates_closed = self.template_manager.templates.get("closed", [])
        templates_open = self.template_manager.templates.get("open", [])

        n_closed = len(templates_closed)
        n_open = len(templates_open)

        if n_closed == 0 and n_open == 0:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "No templates to save!",
                QMessageBox.Ok,
            )
            return

        # Get optional template set name
        template_set_name = self.template_set_name_line_edit.text().strip()
        if not template_set_name:
            template_set_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build save dict with both classes
        duration = self.template_manager.template_duration_s
        save_dict = {
            "templates_open": templates_open,
            "templates_closed": templates_closed,
            "metadata_open": self.template_manager.template_metadata.get("open", []),
            "metadata_closed": self.template_manager.template_metadata.get("closed", []),
            "metadata": {
                "template_duration_s": duration,
                "n_open": n_open,
                "n_closed": n_closed,
                "format": "combined",  # Mark as combined format
                "created": datetime.now().isoformat(),
            }
        }

        # Save to file
        templates_dir = "data/templates"
        if not os.path.exists(templates_dir):
            os.makedirs(templates_dir)

        filename = f"templates_combined_{template_set_name}.pkl"
        save_path = os.path.join(templates_dir, filename)

        try:
            with open(save_path, "wb") as f:
                pickle.dump(save_dict, f)

            print(f"[TRAINING] Saved combined templates to: {save_path}")
            print(f"  OPEN: {n_open} templates")
            print(f"  CLOSED: {n_closed} templates")

            QMessageBox.information(
                self.main_window,
                "Templates Saved",
                f"Saved combined template file:\n{save_path}\n\n"
                f"OPEN: {n_open} templates\n"
                f"CLOSED: {n_closed} templates\n"
                f"Duration: {duration}s each",
                QMessageBox.Ok,
            )

        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"Failed to save templates: {e}",
                QMessageBox.Ok,
            )

    def _show_template_review_dialog(self) -> None:
        """Show dialog to visualize extracted templates."""
        # Get templates from template manager
        templates_closed = self.template_manager.templates.get("closed", [])
        templates_open = self.template_manager.templates.get("open", [])

        if not templates_closed and not templates_open:
            print("[TRAINING] No templates to visualize")
            return

        # Show the review dialog (non-modal)
        dialog = TemplateReviewDialog(templates_closed, templates_open, self.main_window)
        dialog.show()

    def _import_templates_from_file(self) -> None:
        """Load a saved template set into the template_manager for re-saving or model creation."""
        templates_dir = "data/templates"
        if not os.path.exists(templates_dir):
            templates_dir = "data"

        filename, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Select Templates File",
            templates_dir,
            "Pickle files (*.pkl)"
        )
        if not filename:
            return

        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)

            templates_closed = []
            templates_open = []

            if isinstance(data, dict):
                # Combined format
                if "templates_open" in data and "templates_closed" in data:
                    raw_open = data["templates_open"]
                    raw_closed = data["templates_closed"]
                    # Validate they're raw EMG (n_ch, n_samples), not feature matrices
                    if len(raw_open) > 0 and hasattr(raw_open[0], 'shape') and len(raw_open[0].shape) == 2:
                        if raw_open[0].shape[0] > 64:
                            QMessageBox.warning(
                                self.main_window, "Feature Templates",
                                "This file contains feature-extracted templates.\n"
                                "Please select a file with raw EMG templates.",
                                QMessageBox.Ok)
                            return
                    templates_open = list(raw_open)
                    templates_closed = list(raw_closed)

                # Single-class format
                elif "templates" in data and "metadata" in data:
                    class_label = data["metadata"].get("class_label", "").lower()
                    if "open" in class_label:
                        templates_open = list(data["templates"])
                    elif "closed" in class_label or "close" in class_label:
                        templates_closed = list(data["templates"])
                    else:
                        QMessageBox.warning(
                            self.main_window, "Unknown Class",
                            f"Could not determine class from metadata: '{class_label}'.",
                            QMessageBox.Ok)
                        return
                else:
                    QMessageBox.warning(
                        self.main_window, "Invalid File",
                        "File format not recognized.",
                        QMessageBox.Ok)
                    return
            else:
                QMessageBox.warning(
                    self.main_window, "Invalid File",
                    "File format not recognized (expected dict).",
                    QMessageBox.Ok)
                return

            n_c = len(templates_closed)
            n_o = len(templates_open)
            if n_c == 0 and n_o == 0:
                QMessageBox.warning(self.main_window, "Empty",
                                    "No templates found in file.", QMessageBox.Ok)
                return

            # Validate differential mode
            sample = templates_closed[0] if templates_closed else templates_open[0]
            if hasattr(sample, 'shape') and len(sample.shape) == 2:
                tmpl_mode = sample.shape[0] <= 16
                if not self._validate_differential_mode(tmpl_mode):
                    return

            # Clear existing and load into template_manager
            self.template_manager.clear_all()
            self.template_manager.templates["closed"] = templates_closed
            self.template_manager.templates["open"] = templates_open
            # Load provenance metadata if available
            if "templates_open" in data and "templates_closed" in data:
                self.template_manager.template_metadata["closed"] = list(data.get("metadata_closed", []))
                self.template_manager.template_metadata["open"] = list(data.get("metadata_open", []))
            elif "templates" in data and "metadata" in data:
                class_label_key = data["metadata"].get("class_label", "").lower()
                tmpl_meta = list(data.get("template_metadata", []))
                if "open" in class_label_key:
                    self.template_manager.template_metadata["open"] = tmpl_meta
                else:
                    self.template_manager.template_metadata["closed"] = tmpl_meta

            # Infer template duration from data
            if sample.shape[1] > 0:
                duration_s = sample.shape[1] / config.FSAMP
                self.template_manager.template_duration_s = duration_s

            # Update extraction section UI
            basename = os.path.basename(filename)
            self.template_count_label.setText(f"{n_c} closed, {n_o} open (from file)")
            self.save_templates_btn.setEnabled(True)
            self.template_study_btn.setEnabled(True)

            # Pre-fill template set name from filename
            name_stem = os.path.splitext(basename)[0]
            for prefix in ["raw_templates_", "templates_"]:
                if name_stem.startswith(prefix):
                    name_stem = name_stem[len(prefix):]
            self.template_set_name_line_edit.setText(name_stem)

            # Also set as selected templates for Create Model section
            self.selected_combined_templates_path = filename
            self.selected_open_templates_path = filename
            self.selected_closed_templates_path = filename
            duration_s = sample.shape[1] / config.FSAMP if sample.shape[1] > 0 else 0
            self.templates_label.setText(
                f"{n_o} OPEN + {n_c} CLOSED ({duration_s:.1f}s) - {basename}"
            )
            self._update_create_model_button_state()

            # No recordings for review, but templates are in the manager
            self._guided_recordings_for_review = []
            self._onset_info = None

            print(f"\n[TRAINING] Loaded templates from file: {basename}")
            print(f"  {n_c} CLOSED + {n_o} OPEN templates")
            if sample.shape[1] > 0:
                print(f"  Template shape: {sample.shape} ({duration_s:.1f}s)")

        except Exception as e:
            QMessageBox.critical(
                self.main_window, "Error",
                f"Failed to load templates:\n{e}", QMessageBox.Ok)
            import traceback
            traceback.print_exc()

    def _open_template_review(self) -> None:
        """Open the template review dialog showing full cycles for manual adjustment.

        This opens the same GuidedRecordingReviewDialog used for manual selection,
        allowing the user to review and adjust template positions visually.
        If onset detection was used, positions and channel info are pre-loaded.
        """
        if not self._guided_recordings_for_review:
            QMessageBox.warning(
                self.main_window,
                "No Recordings",
                "No recordings available for review.\n"
                "Please select and extract templates from guided recordings first.",
                QMessageBox.Ok,
            )
            return

        # Check if we have onset info from a previous onset detection run
        has_onset = hasattr(self, '_onset_info') and self._onset_info
        print(f"\n[TRAINING] Opening template review with {len(self._guided_recordings_for_review)} recording(s)"
              f"{' (onset detection positions)' if has_onset else ''}")

        review_dialog = GuidedRecordingReviewDialog(
            self._guided_recordings_for_review, self.template_manager, self.main_window,
            onset_detection=has_onset,
            onset_info=self._onset_info if has_onset else None,
            onset_method=getattr(self, "_onset_method", "amplitude"),
        )
        result = review_dialog.exec()

        if review_dialog.saved:
            n_closed = len(self.template_manager.templates.get("closed", []))
            n_open = len(self.template_manager.templates.get("open", []))
            self.template_count_label.setText(f"Updated: {n_closed} closed, {n_open} open")
            print(f"[TRAINING] Templates updated: {n_closed} closed, {n_open} open")

    def _open_template_study(self) -> None:
        """Open the Template Study dialog to analyze template quality."""
        templates_closed = self.template_manager.templates.get("closed", [])
        templates_open = self.template_manager.templates.get("open", [])

        # Read current defaults from the Create Model section
        window_samples, overlap_samples = self._get_window_overlap_samples()
        feature = self.feature_combo.currentText()
        agg_idx = self.distance_agg_combo.currentIndex()
        dead_text = self.dead_channels_input.text().strip()

        dialog = TemplateStudyDialog(
            templates_closed=templates_closed,
            templates_open=templates_open,
            default_feature=feature,
            default_window_samples=window_samples,
            default_overlap_samples=overlap_samples,
            default_aggregation_idx=agg_idx,
            default_dead_channels_text=dead_text,
            metadata_closed=self.template_manager.template_metadata.get("closed", []),
            metadata_open=self.template_manager.template_metadata.get("open", []),
            parent=self.main_window,
        )
        dialog.finished.connect(lambda: self._on_template_study_closed(dialog))
        dialog.show()

    def _on_template_study_closed(self, dialog) -> None:
        """Sync templates back from the Template Study dialog if modified."""
        if not dialog.templates_modified:
            return
        self.template_manager.templates["closed"] = dialog.templates_closed
        self.template_manager.templates["open"] = dialog.templates_open
        self.template_manager.template_metadata["closed"] = dialog.metadata_closed
        self.template_manager.template_metadata["open"] = dialog.metadata_open
        n_c = len(dialog.templates_closed)
        n_o = len(dialog.templates_open)
        self.template_count_label.setText(f"Updated: {n_c} closed, {n_o} open")
        print(f"[TRAINING] Templates updated from Template Study: {n_c} CLOSED + {n_o} OPEN")

    def _clear_extraction(self) -> None:
        """Clear all extracted activations and templates."""
        class_label = self.template_class_combo.currentText().lower()

        # Clear template manager data
        self.template_manager.clear_all(class_label)

        # Clear UI
        self.selected_extraction_recordings = []
        self._guided_recordings_for_review = []
        self.legacy_emg_folder = None
        self.legacy_gt_folder = None
        self.selected_recordings_for_extraction_label.setText("No recordings selected")
        self.activation_count_label.setText("0 activations found")
        self.template_count_label.setText("0/20 templates")
        self.activation_list_widget.clear()
        self.extraction_progress_bar.setValue(0)
        self.template_set_name_line_edit.clear()

        # Disable buttons
        self.extract_activations_btn.setEnabled(False)
        self.select_templates_btn.setEnabled(False)
        self.plot_selected_btn.setEnabled(False)
        self.save_templates_btn.setEnabled(False)
        self.open_template_review_btn.setEnabled(False)
        # Template Study stays enabled (can load from file)

    # ==================== DTW Model Creation Methods ====================

    def _select_combined_template_file(self) -> None:
        """Select a combined template file containing both OPEN and CLOSED templates."""
        templates_dir = "data/templates"
        if not os.path.exists(templates_dir):
            templates_dir = "data"

        filename, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Select Templates File",
            templates_dir,
            "Pickle files (*.pkl)"
        )

        if not filename:
            return

        # Validate the template file
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)

            # Check for combined format (has both templates_open and templates_closed)
            if isinstance(data, dict) and "templates_open" in data and "templates_closed" in data:
                n_open = len(data["templates_open"])
                n_closed = len(data["templates_closed"])
                duration = data.get("metadata", {}).get("template_duration_s", "?")
                label_text = f"{n_open} OPEN + {n_closed} CLOSED ({duration}s)"

                # Validate differential mode
                tmpl_mode = _detect_mode_from_data(data)
                # If metadata doesn't have it, infer from raw template shape (n_channels, n_samples)
                if tmpl_mode is None and n_open > 0:
                    t = data["templates_open"][0]
                    if hasattr(t, 'shape') and len(t.shape) == 2:
                        tmpl_mode = t.shape[0] <= 16
                if tmpl_mode is not None:
                    if not self._validate_differential_mode(tmpl_mode):
                        return  # User declined switch

                self.selected_combined_templates_path = filename
                self.selected_open_templates_path = filename  # For compatibility
                self.selected_closed_templates_path = filename  # For compatibility
                self.templates_label.setText(label_text)
                self._update_create_model_button_state()
                return

            # Check for single-class format (dict with templates and metadata)
            if isinstance(data, dict) and "templates" in data and "metadata" in data:
                n_templates = len(data["templates"])
                duration = data["metadata"].get("template_duration_s", "?")
                QMessageBox.warning(
                    self.main_window,
                    "Single-Class File",
                    f"This file contains only {n_templates} templates of one class.\n\n"
                    f"Please select a combined template file (containing both OPEN and CLOSED).",
                    QMessageBox.Ok,
                )
                return

            # Check for legacy format (list of templates)
            if isinstance(data, list):
                QMessageBox.warning(
                    self.main_window,
                    "Legacy Format",
                    f"This is a legacy format file with {len(data)} templates.\n\n"
                    f"Please select a combined template file (containing both OPEN and CLOSED).",
                    QMessageBox.Ok,
                )
                return

            QMessageBox.warning(
                self.main_window,
                "Invalid File",
                "Selected file is not a valid template file.",
                QMessageBox.Ok,
            )

        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"Failed to load template file: {e}",
                QMessageBox.Ok,
            )

    def _select_template_file(self, class_label: str) -> None:
        """Select a template file for the given class."""
        templates_dir = "data"
        if not os.path.exists(templates_dir):
            os.makedirs(templates_dir)

        # Use static method with explicit title parameter
        filename, _ = QFileDialog.getOpenFileName(
            self.main_window,
            f"Select {class_label.capitalize()} Templates",
            templates_dir,
            "Pickle files (*.pkl)"
        )

        if not filename:
            return

        # Validate the template file
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)

            # Check if it's the new format (dict with templates and metadata)
            if isinstance(data, dict) and "templates" in data and "metadata" in data:
                n_templates = len(data["templates"])
                duration = data["metadata"].get("template_duration_s", "?")
                label_text = f"{n_templates} templates ({duration}s)"

                # Validate differential mode from metadata or template shape
                tmpl_mode = _detect_mode_from_data(data)
                if tmpl_mode is None and n_templates > 0:
                    t = data["templates"][0]
                    if hasattr(t, 'shape') and len(t.shape) == 2:
                        tmpl_mode = t.shape[0] <= 16
                if tmpl_mode is not None:
                    if not self._validate_differential_mode(tmpl_mode):
                        return
            # Check if it's the old format (list of templates)
            elif isinstance(data, list):
                n_templates = len(data)
                label_text = f"{n_templates} templates (legacy format)"

                # Infer mode from template shape
                if n_templates > 0 and hasattr(data[0], 'shape') and len(data[0].shape) == 2:
                    tmpl_mode = data[0].shape[0] <= 16
                    if not self._validate_differential_mode(tmpl_mode):
                        return
            else:
                QMessageBox.warning(
                    self.main_window,
                    "Invalid File",
                    "Selected file is not a valid template file.",
                    QMessageBox.Ok,
                )
                return

            if class_label == "open":
                self.selected_open_templates_path = filename
                self.open_templates_label.setText(label_text)
            else:
                self.selected_closed_templates_path = filename
                self.closed_templates_label.setText(label_text)

            self._update_create_model_button_state()

        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"Failed to load template file: {e}",
                QMessageBox.Ok,
            )

    def _on_channel_mode_changed(self, index: int) -> None:
        """Handle channel mode combo change (0=Global, 1=Per-class)."""
        is_per_class = index == 1
        self.closed_channels_label.setVisible(is_per_class)
        self.closed_channels_input.setVisible(is_per_class)
        self.open_channels_label.setVisible(is_per_class)
        self.open_channels_input.setVisible(is_per_class)
        self.auto_detect_channels_btn.setVisible(is_per_class)
        # Enable auto-detect button if onset info is available
        if is_per_class:
            has_onset = hasattr(self, '_onset_info') and self._onset_info
            self.auto_detect_channels_btn.setEnabled(has_onset)

    def _auto_detect_per_class_channels(self) -> None:
        """Auto-detect per-class channels from onset detection info."""
        from mindmove.model.template_study import compute_per_class_channels

        if not hasattr(self, '_onset_info') or not self._onset_info:
            QMessageBox.warning(
                self.main_window,
                "No Onset Data",
                "Run template extraction with onset detection first.",
                QMessageBox.Ok,
            )
            return

        # Gather recordings
        recordings = self._get_loaded_recordings()
        dead_channels = self._parse_dead_channels()

        result = compute_per_class_channels(self._onset_info, recordings, dead_channels)
        self._per_class_channels = result

        # Populate fields (1-indexed for display)
        closed_str = ",".join(str(ch + 1) for ch in result["closed"])
        open_str = ",".join(str(ch + 1) for ch in result["open"])
        self.closed_channels_input.setText(closed_str)
        self.open_channels_input.setText(open_str)

        # Print summary
        print(f"\n[PER-CLASS CHANNELS] Auto-detected from {len(self._onset_info)} cycles:")
        print(f"  CLOSED channels (1-indexed): [{closed_str}]")
        print(f"  OPEN channels (1-indexed): [{open_str}]")
        if result["unassigned"]:
            unassigned_str = ",".join(str(ch + 1) for ch in result["unassigned"])
            print(f"  Unassigned: [{unassigned_str}]")
        if result["details"]:
            print(f"  Conflict resolution details:")
            for ch, info in sorted(result["details"].items()):
                if "ratio_closed_over_open" in info:
                    assigned = "CLOSED" if ch in result["closed"] else "OPEN"
                    print(f"    CH{ch+1}: fired {info['closed_fire_count']}x CLOSED, "
                          f"{info['open_fire_count']}x OPEN, "
                          f"RMS ratio(C/O)={info['ratio_closed_over_open']:.2f} → {assigned}")

    def _get_loaded_recordings(self) -> list:
        """Get the list of loaded recording dicts (used by per-class channel detection)."""
        if hasattr(self, '_guided_recordings_for_review') and self._guided_recordings_for_review:
            return self._guided_recordings_for_review
        return []

    def _parse_per_class_channels(self, text: str) -> list:
        """Parse channel list from text field (1-indexed input to 0-indexed)."""
        if not text.strip():
            return []
        channels = []
        for part in text.split(","):
            part = part.strip()
            if part.isdigit():
                ch = int(part)
                if 1 <= ch <= 32:
                    channels.append(ch - 1)
        return sorted(set(channels))

    def _on_window_preset_changed(self, index: int) -> None:
        """Handle window/overlap preset change."""
        presets = [
            (96, 32),   # Default
            (150, 50),  # Eddy's
            (200, 100), # Larger
            None        # Custom
        ]

        is_custom = index == 3
        self.custom_window_label.setVisible(is_custom)
        self.custom_window_spinbox.setVisible(is_custom)
        self.custom_overlap_label.setVisible(is_custom)
        self.custom_overlap_spinbox.setVisible(is_custom)

        if not is_custom and presets[index]:
            window_ms, overlap_ms = presets[index]
            self.custom_window_spinbox.setValue(window_ms)
            self.custom_overlap_spinbox.setValue(overlap_ms)

    def _update_create_model_button_state(self) -> None:
        """Enable Create Model button only when templates are selected."""
        # Check for combined template file (new format)
        has_combined = getattr(self, 'selected_combined_templates_path', None) is not None
        # Check for separate open/closed files (legacy support)
        has_open = self.selected_open_templates_path is not None
        has_closed = self.selected_closed_templates_path is not None

        self.create_model_btn.setEnabled(has_combined or (has_open and has_closed))

    def _get_window_overlap_samples(self) -> tuple:
        """Get window and overlap in samples based on current settings."""
        window_ms = self.custom_window_spinbox.value()
        overlap_ms = self.custom_overlap_spinbox.value()

        window_samples = int(window_ms / 1000 * config.FSAMP)
        overlap_samples = int(overlap_ms / 1000 * config.FSAMP)

        return window_samples, overlap_samples

    def _create_dtw_model(self) -> None:
        """Create a DTW model from selected templates."""
        if not self.selected_open_templates_path or not self.selected_closed_templates_path:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "Please select both open and closed template files!",
                QMessageBox.Ok,
            )
            return

        self.create_model_btn.setEnabled(False)
        self.model_creation_progress_bar.setValue(0)

        # Store decision model choice (read from UI before thread starts)
        self._decision_model_choice = self.decision_model_combo.currentText()

        # Start model creation in a thread
        self.create_model_thread = PyQtThread(
            target=self._create_dtw_model_thread,
            parent=self.main_window
        )
        self.create_model_thread.has_finished_signal.connect(self._create_dtw_model_finished)
        self.create_model_thread.start()

    def _parse_dead_channels(self) -> List[int]:
        """Parse dead channels from input field (1-indexed user input to 0-indexed)."""
        text = self.dead_channels_input.text().strip()
        if not text:
            return []

        dead_channels = []
        for part in text.split(","):
            part = part.strip()
            if part.isdigit():
                ch = int(part)
                if 1 <= ch <= 32:
                    dead_channels.append(ch - 1)  # Convert to 0-indexed
                else:
                    print(f"Warning: Channel {ch} out of range (1-32), ignoring")
        return sorted(set(dead_channels))

    def _print_template_statistics(self, stats: dict, class_label: str, n_templates: int) -> None:
        """Print compact template statistics summary.

        Args:
            stats: Dictionary from compute_per_template_statistics()
            class_label: Label for the class (e.g., "OPEN", "CLOSED")
            n_templates: Number of templates
        """
        print(f"\n  === {class_label} Templates ({n_templates}) ===")
        print(f"  Overall: mean={stats['overall_mean']:.4f}, std={stats['overall_std']:.4f}")

        q = stats['quartiles']
        print(f"  Distribution: min={q[0]:.4f}, Q1={q[1]:.4f}, median={q[2]:.4f}, Q3={q[3]:.4f}, max={q[4]:.4f}")
        print(f"  Consistency score: {stats['consistency_score']:.2f} (std/mean, lower=better)")

        # Worst pairs (most dissimilar)
        if stats['worst_pairs']:
            print(f"\n  Worst pairs (most dissimilar):")
            for i, j, dist in stats['worst_pairs']:
                print(f"    Templates {i+1}-{j+1}: {dist:.4f}")

        # Potential outliers
        if stats['outliers']:
            print(f"\n  Potential outliers (consider removing):")
            for idx, avg, sigma in stats['outliers']:
                print(f"    Template {idx}: avg={avg:.4f} ({sigma:.1f} sigma above mean)")
        else:
            print(f"\n  No outliers detected (all templates consistent)")

        # Best templates (most consistent)
        if stats['best_indices']:
            print(f"\n  Best templates (most consistent):")
            for idx in stats['best_indices'][:3]:
                avg = stats['per_template_avg'][idx - 1]  # Convert back to 0-indexed
                print(f"    Template {idx}: avg={avg:.4f}")

    def _create_dtw_model_thread(self) -> None:
        """Thread function to create DTW model."""
        from mindmove.model.core.features.features_registry import FEATURES
        from mindmove.model.core.windowing import sliding_window
        from mindmove.model.core.algorithm import (
            compute_threshold,
            compute_threshold_with_aggregation,
            compute_per_template_statistics,
            compute_cross_class_distances,
            compute_cross_class_distances_with_aggregation,
            compute_threshold_presets,
            compute_spatial_profiles
        )
        from mindmove.model.template_study import (
            compute_template_metrics_with_aggregation,
            analyze_template_quality,
        )

        # Get parameters
        window_samples, overlap_samples = self._get_window_overlap_samples()
        feature_name = self.feature_combo.currentText()
        dtw_algorithm = self.dtw_algorithm_combo.currentText()
        model_name = self.training_model_label_line_edit.text().strip() or "default"

        # Get dead channels (0-indexed internally)
        dead_channels = self._parse_dead_channels()
        # Display as 1-indexed for user
        dead_channels_display = [ch + 1 for ch in dead_channels]

        # Get distance aggregation method
        distance_agg_text = self.distance_agg_combo.currentText()
        if "3 smallest" in distance_agg_text:
            distance_aggregation = "avg_3_smallest"
        elif "Minimum" in distance_agg_text:
            distance_aggregation = "minimum"
        else:
            distance_aggregation = "average"

        # Get smoothing method
        smoothing_text = self.smoothing_combo.currentText()
        if "Majority" in smoothing_text:
            smoothing_method = "MAJORITY VOTE"
        elif "Consecutive" in smoothing_text:
            smoothing_method = "5 CONSECUTIVE"
        else:
            smoothing_method = "NONE"

        # Get per-class channel configuration
        is_per_class = self.channel_mode_combo.currentIndex() == 1
        active_channels_closed = None
        active_channels_open = None
        if is_per_class:
            active_channels_closed = self._parse_per_class_channels(self.closed_channels_input.text())
            active_channels_open = self._parse_per_class_channels(self.open_channels_input.text())
            if not active_channels_closed or not active_channels_open:
                print("[WARNING] Per-class mode selected but channels are empty — falling back to global")
                is_per_class = False

        print(f"\n{'='*60}")
        print("Creating DTW Model")
        print(f"{'='*60}")
        print(f"Window: {window_samples} samples ({window_samples/config.FSAMP*1000:.0f}ms)")
        print(f"Overlap: {overlap_samples} samples ({overlap_samples/config.FSAMP*1000:.0f}ms)")
        print(f"Feature: {feature_name}")
        print(f"DTW Algorithm: {dtw_algorithm}")
        print(f"Dead channels (1-indexed): {dead_channels_display if dead_channels_display else 'None'}")
        print(f"Distance aggregation: {distance_aggregation}")
        print(f"Smoothing method: {smoothing_method}")
        if is_per_class:
            print(f"Channel mode: Per-class (spatial separation)")
            print(f"  CLOSED channels (0-indexed): {active_channels_closed}")
            print(f"  OPEN channels (0-indexed): {active_channels_open}")
        else:
            print(f"Channel mode: Global")
        print(f"Model name: {model_name}")

        # Load templates
        print("\nLoading templates...")

        # Check if using combined template file
        combined_path = getattr(self, 'selected_combined_templates_path', None)
        if combined_path and self.selected_open_templates_path == self.selected_closed_templates_path:
            # Load from combined file
            open_templates_raw, closed_templates_raw = self._load_combined_templates(combined_path)
        else:
            # Load from separate files (legacy)
            open_templates_raw = self._load_templates_from_file(self.selected_open_templates_path)
            closed_templates_raw = self._load_templates_from_file(self.selected_closed_templates_path)

        print(f"  Open templates: {len(open_templates_raw)}")
        print(f"  Closed templates: {len(closed_templates_raw)}")

        # Compute spatial profiles from raw templates (before feature extraction)
        print("\nComputing spatial profiles...")
        spatial_profile_open = compute_spatial_profiles(open_templates_raw, class_label="open")
        spatial_profile_closed = compute_spatial_profiles(closed_templates_raw, class_label="closed")

        # Auto-compute spatial sharpness exponent
        from mindmove.model.core.algorithm import compute_spatial_sharpness
        spatial_sharpness_k = compute_spatial_sharpness(
            open_templates_raw, closed_templates_raw,
            spatial_profile_open, spatial_profile_closed,
        )

        # Extract features from templates
        print("\nExtracting features...")
        feature_fn = FEATURES[feature_name]["function"]
        increment_samples = max(1, window_samples - overlap_samples)
        print(f"  Window shift (increment): {increment_samples} samples")

        open_templates_features = []
        for template in open_templates_raw:
            windowed = sliding_window(template, window_samples, increment_samples)
            features = feature_fn(windowed)
            open_templates_features.append(features)

        closed_templates_features = []
        for template in closed_templates_raw:
            windowed = sliding_window(template, window_samples, increment_samples)
            features = feature_fn(windowed)
            closed_templates_features.append(features)

        print(f"  Open features shape: {open_templates_features[0].shape if open_templates_features else 'N/A'}")
        print(f"  Closed features shape: {closed_templates_features[0].shape if closed_templates_features else 'N/A'}")

        # Compute thresholds
        print("\nComputing thresholds...")

        # Set DTW algorithm in config temporarily
        original_numba = config.USE_NUMBA_DTW
        original_tslearn = config.USE_TSLEARN_DTW

        if "Numba" in dtw_algorithm:
            config.USE_NUMBA_DTW = True
            config.USE_TSLEARN_DTW = False
        elif "tslearn" in dtw_algorithm:
            config.USE_NUMBA_DTW = False
            config.USE_TSLEARN_DTW = True
        elif "dtaidistance" in dtw_algorithm:
            # Will implement dtaidistance support
            config.USE_NUMBA_DTW = False
            config.USE_TSLEARN_DTW = False
        else:  # Pure Python
            config.USE_NUMBA_DTW = False
            config.USE_TSLEARN_DTW = False

        # Compute intra-class distances WITH aggregation (matches Template Study)
        # For each template, distances to all others are aggregated (e.g. avg_3_smallest)
        # before computing mean/std → consistent with online prediction behavior
        mean_open, std_open, threshold_open = compute_threshold_with_aggregation(
            open_templates_features,
            active_channels=active_channels_open if is_per_class else None,
            distance_aggregation=distance_aggregation,
        )
        mean_closed, std_closed, threshold_closed = compute_threshold_with_aggregation(
            closed_templates_features,
            active_channels=active_channels_closed if is_per_class else None,
            distance_aggregation=distance_aggregation,
        )

        # Restore config
        config.USE_NUMBA_DTW = original_numba
        config.USE_TSLEARN_DTW = original_tslearn

        print(f"  Open intra-class: mean={mean_open:.4f}, std={std_open:.4f} (mean+std={mean_open+std_open:.4f})")
        print(f"  Closed intra-class: mean={mean_closed:.4f}, std={std_closed:.4f} (mean+std={mean_closed+std_closed:.4f})")
        print(f"  (aggregation: {distance_aggregation})")

        # Compute cross-class distances with same aggregation (directional)
        # open→closed and closed→open are computed separately (like Template Study)
        print("\nComputing cross-class distances for threshold presets...")
        cross_active_channels = None
        if is_per_class:
            cross_active_channels = sorted(set(active_channels_open) | set(active_channels_closed))
        cross_result = compute_cross_class_distances_with_aggregation(
            open_templates_features,
            closed_templates_features,
            active_channels=cross_active_channels,
            distance_aggregation=distance_aggregation,
        )
        # Directional: open→closed (used for OPEN threshold), closed→open (used for CLOSED threshold)
        # Template Study convention: inter_closed_to_open = each CLOSED template vs all OPEN
        # In our call: A=open, B=closed → a_to_b = open→closed, b_to_a = closed→open
        cross_open_to_closed = cross_result["a_to_b"]    # each OPEN template vs all CLOSED
        cross_closed_to_open = cross_result["b_to_a"]    # each CLOSED template vs all OPEN

        print(f"  OPEN->CLOSED:  mean={cross_open_to_closed['mean']:.4f}, std={cross_open_to_closed['std']:.4f}")
        print(f"  CLOSED->OPEN:  mean={cross_closed_to_open['mean']:.4f}, std={cross_closed_to_open['std']:.4f}")

        # ── Template Study metrics (single distance matrix, same as Template Study dialog) ──
        print(f"\n{'-'*60}")
        print("Template Study Metrics (unified distance matrix)")
        print(f"{'-'*60}")
        ts_metrics = compute_template_metrics_with_aggregation(
            templates_closed=closed_templates_raw,
            templates_open=open_templates_raw,
            feature_name=feature_name,
            window_length=window_samples,
            window_increment=window_samples - overlap_samples,
            distance_aggregation=distance_aggregation,
            active_channels=cross_active_channels if is_per_class else None,
        )
        ts_ic = ts_metrics["intra_closed"]
        ts_io = ts_metrics["intra_open"]
        ts_ec = ts_metrics["inter_closed_to_open"]
        ts_eo = ts_metrics["inter_open_to_closed"]

        ts_ic_upper = ts_ic["mean"] + ts_ic["std"]
        ts_io_upper = ts_io["mean"] + ts_io["std"]
        ts_ec_lower = ts_ec["mean"] - ts_ec["std"]
        ts_eo_lower = ts_eo["mean"] - ts_eo["std"]

        ts_gap_closed = ts_ec_lower - ts_ic_upper
        ts_gap_open = ts_eo_lower - ts_io_upper
        ts_ok_c = "OK" if ts_gap_closed > 0 else "OVERLAP"
        ts_ok_o = "OK" if ts_gap_open > 0 else "OVERLAP"

        ts_midgap_closed = (ts_ic_upper + ts_ec_lower) / 2
        ts_midgap_open = (ts_io_upper + ts_eo_lower) / 2

        print(f"  Intra CLOSED:  mean={ts_ic['mean']:.4f}  std={ts_ic['std']:.4f}  ->  mean+std={ts_ic_upper:.4f}")
        print(f"  Intra OPEN:    mean={ts_io['mean']:.4f}  std={ts_io['std']:.4f}  ->  mean+std={ts_io_upper:.4f}")
        print(f"  Inter (C->O):  mean={ts_ec['mean']:.4f}  std={ts_ec['std']:.4f}  ->  mean-std={ts_ec_lower:.4f}")
        print(f"  Inter (O->C):  mean={ts_eo['mean']:.4f}  std={ts_eo['std']:.4f}  ->  mean-std={ts_eo_lower:.4f}")
        print(f"")
        print(f"  Gap CLOSED: {ts_gap_closed:.4f} [{ts_ok_c}]   Mid-gap threshold: {ts_midgap_closed:.4f}")
        print(f"  Gap OPEN:   {ts_gap_open:.4f} [{ts_ok_o}]   Mid-gap threshold: {ts_midgap_open:.4f}")

        # Quality analysis
        n_closed_tpl = len(closed_templates_raw)
        n_open_tpl = len(open_templates_raw)
        ts_quality = analyze_template_quality(ts_metrics, n_closed_tpl, n_open_tpl)
        print(f"\n  {ts_quality['summary']}")
        print(f"{'-'*60}")

        # For backward compatibility, store combined cross-class stats too
        mean_cross = cross_result["combined_mean"]
        std_cross = cross_result["combined_std"]

        # Default threshold: mid-gap using DIRECTIONAL cross-class distances
        # OPEN threshold uses inter_open_to_closed (each OPEN vs all CLOSED)
        # CLOSED threshold uses inter_closed_to_open (each CLOSED vs all OPEN)
        mean_cross_open = cross_open_to_closed["mean"]
        std_cross_open = cross_open_to_closed["std"]
        mean_cross_closed = cross_closed_to_open["mean"]
        std_cross_closed = cross_closed_to_open["std"]

        threshold_open = ((mean_open + std_open) + (mean_cross_open - std_cross_open)) / 2
        threshold_closed = ((mean_closed + std_closed) + (mean_cross_closed - std_cross_closed)) / 2
        print(f"\n  Model creation thresholds (separate distance matrices):")
        print(f"    Intra OPEN:   mean={mean_open:.4f}  std={std_open:.4f}  ->  mean+std={mean_open+std_open:.4f}")
        print(f"    Intra CLOSED: mean={mean_closed:.4f}  std={std_closed:.4f}  ->  mean+std={mean_closed+std_closed:.4f}")
        print(f"    Cross OPEN->CLOSED: mean={mean_cross_open:.4f}  std={std_cross_open:.4f}  ->  mean-std={mean_cross_open-std_cross_open:.4f}")
        print(f"    Cross CLOSED->OPEN: mean={mean_cross_closed:.4f}  std={std_cross_closed:.4f}  ->  mean-std={mean_cross_closed-std_cross_closed:.4f}")
        print(f"\n  Default thresholds (mid-gap, directional):")
        print(f"    OPEN:   {threshold_open:.4f}  (gap: [{mean_open+std_open:.4f}, {mean_cross_open-std_cross_open:.4f}])")
        print(f"    CLOSED: {threshold_closed:.4f}  (gap: [{mean_closed+std_closed:.4f}, {mean_cross_closed-std_cross_closed:.4f}])")
        print(f"\n  Template Study mid-gap (for comparison):")
        print(f"    OPEN:   {ts_midgap_open:.4f}")
        print(f"    CLOSED: {ts_midgap_closed:.4f}")
        if abs(threshold_open - ts_midgap_open) > 0.0001 or abs(threshold_closed - ts_midgap_closed) > 0.0001:
            print(f"  *** MISMATCH DETECTED ***")
            print(f"    OPEN diff:   {threshold_open - ts_midgap_open:+.4f}")
            print(f"    CLOSED diff: {threshold_closed - ts_midgap_closed:+.4f}")

        # Compute threshold presets using directional cross-class stats
        print("\nComputing threshold presets for OPEN...")
        open_presets = compute_threshold_presets(mean_open, std_open, mean_cross_open, std_cross_open)
        for key, preset in open_presets.items():
            print(f"  {preset['name']}: threshold={preset['threshold']:.4f} (s={preset['s']:.2f})")

        # Compute threshold presets for CLOSED class
        print("\nComputing threshold presets for CLOSED...")
        closed_presets = compute_threshold_presets(mean_closed, std_closed, mean_cross_closed, std_cross_closed)
        for key, preset in closed_presets.items():
            print(f"  {preset['name']}: threshold={preset['threshold']:.4f} (s={preset['s']:.2f})")

        # Build combined presets dictionary for model storage
        threshold_presets = {}
        for key in open_presets.keys():
            threshold_presets[key] = {
                "s_open": open_presets[key]["s"],
                "s_closed": closed_presets[key]["s"],
                "threshold_open": open_presets[key]["threshold"],
                "threshold_closed": closed_presets[key]["threshold"],
                "name": open_presets[key]["name"],
                "description": open_presets[key]["description"]
            }

        # Compute per-template statistics
        print("\nAnalyzing template quality...")

        # Open templates statistics
        open_stats = compute_per_template_statistics(open_templates_features, n_worst=3)
        self._print_template_statistics(open_stats, "OPEN", len(open_templates_features))

        # Closed templates statistics
        closed_stats = compute_per_template_statistics(closed_templates_features, n_worst=3)
        self._print_template_statistics(closed_stats, "CLOSED", len(closed_templates_features))

        # ── Train decision models from templates ──
        # Uses template-vs-template distances with perfect labels (no recording needed).
        decision_nn_weights = None
        decision_catboost_model = None
        decision_choice = getattr(self, '_decision_model_choice', 'None')
        train_catboost = decision_choice in ("CatBoost", "Both")
        train_nn = decision_choice in ("Neural Network", "Both")

        if train_catboost or train_nn:
            # Spatial refs for decision model features
            spatial_ref_open_dict = None
            spatial_ref_closed_dict = None
            if spatial_profile_open and spatial_profile_closed:
                spatial_ref_open_dict = {
                    "ref_profile": spatial_profile_open["ref_profile"],
                    "weights": spatial_profile_open["weights"],
                }
                spatial_ref_closed_dict = {
                    "ref_profile": spatial_profile_closed["ref_profile"],
                    "weights": spatial_profile_closed["weights"],
                }

        if train_catboost:
            print(f"\n{'='*60}")
            print("Training CatBoost decision model (template-based)...")
            print(f"{'='*60}")
            try:
                from mindmove.model.core.decision_network import train_transition_catboost_from_templates
                decision_catboost_model = train_transition_catboost_from_templates(
                    templates_open_features=open_templates_features,
                    templates_closed_features=closed_templates_features,
                    distance_aggregation=distance_aggregation,
                    templates_open_raw=open_templates_raw,
                    templates_closed_raw=closed_templates_raw,
                    spatial_ref_open=spatial_ref_open_dict,
                    spatial_ref_closed=spatial_ref_closed_dict,
                    verbose=True,
                )
            except Exception as e:
                print(f"[WARNING] CatBoost training failed: {e}")
                import traceback
                traceback.print_exc()

        if train_nn:
            print(f"\n{'='*60}")
            print("Training Neural Network decision model (template-based)...")
            print(f"{'='*60}")
            try:
                from mindmove.model.core.decision_network import train_decision_network_from_templates
                decision_nn_weights = train_decision_network_from_templates(
                    templates_open_features=open_templates_features,
                    templates_closed_features=closed_templates_features,
                    distance_aggregation=distance_aggregation,
                    templates_open_raw=open_templates_raw,
                    templates_closed_raw=closed_templates_raw,
                    spatial_ref_open=spatial_ref_open_dict,
                    spatial_ref_closed=spatial_ref_closed_dict,
                    verbose=True,
                )
            except ImportError:
                print("[INFO] PyTorch not installed — skipping NN training")
            except Exception as e:
                print(f"[WARNING] NN training failed: {e}")
                import traceback
                traceback.print_exc()

        if not train_catboost and not train_nn:
            print("\n[INFO] No decision model selected — using threshold-based decisions only")

        # Save model
        print("\nSaving model...")
        now = datetime.now()
        formatted_now = now.strftime("%Y%m%d_%H%M%S")

        # Infer mode from actual template data, not current config
        # Feature templates are (n_windows, n_channels) — check axis 1
        is_differential = _infer_mode_from_templates(open_templates_raw)
        mode_suffix = "sd" if is_differential else "mp"

        model_data = {
            "open_templates": open_templates_features,
            "closed_templates": closed_templates_features,
            "threshold_base_open": threshold_open,
            "threshold_base_closed": threshold_closed,
            "mean_open": mean_open,
            "std_open": std_open,
            "mean_closed": mean_closed,
            "std_closed": std_closed,
            "feature_name": feature_name,
            # New: dead channels (0-indexed for internal use)
            "dead_channels": dead_channels,
            # New: distance aggregation method
            "distance_aggregation": distance_aggregation,
            # New: post-prediction smoothing method
            "smoothing_method": smoothing_method,
            # New: differential mode flag
            "differential_mode": is_differential,
            # Per-class active channels (None means global mode)
            "active_channels_open": active_channels_open if is_per_class else None,
            "active_channels_closed": active_channels_closed if is_per_class else None,
            # Cross-class statistics (combined, for backward compatibility)
            "mean_cross": mean_cross,
            "std_cross": std_cross,
            # Directional cross-class statistics (matching Template Study)
            "mean_cross_open": mean_cross_open,
            "std_cross_open": std_cross_open,
            "mean_cross_closed": mean_cross_closed,
            "std_cross_closed": std_cross_closed,
            # Threshold presets (computed from intra-class and cross-class statistics)
            "threshold_presets": threshold_presets,
            # Spatial profiles for consistency-weighted spatial correction
            "spatial_profiles": {
                "open": {
                    "ref_profile": spatial_profile_open["ref_profile"],
                    "weights": spatial_profile_open["weights"],
                    "consistency": spatial_profile_open["consistency"],
                    "per_template_rms": spatial_profile_open["per_template_rms"],
                } if spatial_profile_open else None,
                "closed": {
                    "ref_profile": spatial_profile_closed["ref_profile"],
                    "weights": spatial_profile_closed["weights"],
                    "consistency": spatial_profile_closed["consistency"],
                    "per_template_rms": spatial_profile_closed["per_template_rms"],
                } if spatial_profile_closed else None,
                "threshold": 0.5,  # Default spatial similarity threshold
                "sharpness": spatial_sharpness_k,  # Auto-computed exponent for scaling/contrast
            },
            "parameters": {
                "window_samples": window_samples,
                "overlap_samples": overlap_samples,
                "increment_samples": increment_samples,
                "window_ms": window_samples / config.FSAMP * 1000,
                "overlap_ms": overlap_samples / config.FSAMP * 1000,
                "dtw_algorithm": dtw_algorithm,
                "fsamp": config.FSAMP,
                "num_channels": config.num_channels,
            },
            # Decision models (trained on guided recording)
            "decision_nn_weights": decision_nn_weights,
            "decision_catboost_model": decision_catboost_model,
            "metadata": {
                "created_at": now.isoformat(),
                "model_name": model_name,
                "n_open_templates": len(open_templates_features),
                "n_closed_templates": len(closed_templates_features),
                "dead_channels_display": dead_channels_display,  # 1-indexed for display
                "differential_mode": is_differential,
            }
        }

        models_dir = "data/models"
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"MindMove_Model_{mode_suffix}_{formatted_now}_{model_name}.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"\nModel saved to: {model_path}")
        print(f"  Saved thresholds (mid-gap):")
        print(f"    OPEN:   {threshold_open:.4f}")
        print(f"    CLOSED: {threshold_closed:.4f}")
        print(f"{'='*60}\n")

        # Store for UI update
        self._created_model_path = model_path

    def _load_templates_from_file(self, filepath: str) -> List[np.ndarray]:
        """Load templates from a file, handling both old and new formats."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # New format: dict with templates and metadata
        if isinstance(data, dict) and "templates" in data:
            return data["templates"]
        # Old format: list of templates
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unknown template format in {filepath}")

    def _load_combined_templates(self, filepath: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load combined template file containing both OPEN and CLOSED templates."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Combined template file must be a dict: {filepath}")

        if "templates_open" not in data or "templates_closed" not in data:
            raise ValueError(f"Combined template file must have 'templates_open' and 'templates_closed': {filepath}")

        return data["templates_open"], data["templates_closed"]

    def _create_dtw_model_finished(self) -> None:
        """Called when model creation is complete."""
        self.model_creation_progress_bar.setValue(100)
        self._update_create_model_button_state()

        if hasattr(self, '_created_model_path'):
            QMessageBox.information(
                self.main_window,
                "Model Created",
                f"DTW model created successfully!\n\nSaved to:\n{self._created_model_path}",
                QMessageBox.Ok,
            )
            delattr(self, '_created_model_path')
