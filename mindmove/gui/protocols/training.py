from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Dict, Tuple

from PySide6.QtCore import QObject, Signal, QThread, Qt
from PySide6.QtWidgets import (
    QFileDialog, QMessageBox, QListWidgetItem, QLabel, QLineEdit,
    QComboBox, QPushButton, QSpinBox, QDoubleSpinBox, QGroupBox,
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QWidget, QListWidget
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
        self.current_channel: int = 0  # 0-indexed

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

        # Channel selector
        header_layout.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.setFixedWidth(60)
        for i in range(1, 33):
            self.channel_combo.addItem(str(i))
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        self.channel_combo.setToolTip("Use Up/Down arrows to switch channels, Left/Right for cycle navigation")
        header_layout.addWidget(self.channel_combo)

        # Keyboard hint
        hint = QLabel("(↑↓ ch, ←→ nav)")
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

        # Matplotlib figure - wider for better signal visibility
        self.figure = Figure(figsize=(14, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(400)
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

        # Update channel combo for actual channel count
        n_channels = cycles[0]['emg'].shape[0] if cycles else 32
        self.channel_combo.clear()
        for i in range(1, n_channels + 1):
            self.channel_combo.addItem(str(i))

        self._update_display()

    def _on_channel_changed(self, index: int):
        if index < 0:
            return  # Ignore invalid index (happens when combo is cleared)
        # Save current window positions before changing channel
        self._save_window_positions()
        self.current_channel = index
        self._update_display()

    def keyPressEvent(self, event):
        """Handle arrow key presses for fast channel switching."""
        n_channels = self.channel_combo.count()
        if event.key() == Qt.Key_Up:
            new_idx = max(0, self.channel_combo.currentIndex() - 1)
            self.channel_combo.setCurrentIndex(new_idx)
            event.accept()
        elif event.key() == Qt.Key_Down:
            new_idx = min(n_channels - 1, self.channel_combo.currentIndex() + 1)
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

        n_samples = emg.shape[1]
        time_axis = np.arange(n_samples) / config.FSAMP

        # Get selected channel's EMG
        channel = max(0, min(self.current_channel, emg.shape[0] - 1))
        emg_signal = emg[channel, :]
        max_val = np.max(np.abs(emg_signal)) if np.max(np.abs(emg_signal)) > 0 else 1

        # Plot GT as overlapping line scaled to EMG amplitude
        gt_scaled = gt * max_val * 0.95
        self.ax.plot(time_axis, gt_scaled, 'r-', linewidth=1.5, alpha=0.7, label='GT')

        # Mark audio cue times with blue dashed lines (if available)
        if close_cue_idx is not None:
            close_cue_time = close_cue_idx / config.FSAMP
            self.ax.axvline(close_cue_time, color='blue', linestyle='--', linewidth=1.5,
                           alpha=0.7, label='Close cue')

        if open_cue_idx is not None:
            open_cue_time = open_cue_idx / config.FSAMP
            self.ax.axvline(open_cue_time, color='blue', linestyle='--', linewidth=1.5,
                           alpha=0.7, label='Open cue')

        # Plot EMG signal
        self.ax.plot(time_axis, emg_signal, 'k-', linewidth=0.5, alpha=0.9, label=f'EMG Ch{channel + 1}')

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
                # Fallback based on protocol mode:
                # Standard: CLOSED is first transition (first quarter)
                # Inverted: CLOSED is second transition (third quarter)
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
                # Fallback based on protocol mode:
                # Standard: OPEN is second transition (third quarter)
                # Inverted: OPEN is first transition (first quarter)
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
        self.ax.set_ylabel(f'Channel {channel + 1} (µV)')
        status = " [ACCEPTED]" if (has_closed and has_open) else (" [CLOSED]" if has_closed else (" [OPEN]" if has_open else ""))
        mode_indicator = " [INV]" if protocol_mode == "inverted" else ""
        # Include recording name in title if available
        rec_name = f'{self.recording_name} - ' if hasattr(self, 'recording_name') and self.recording_name else ''
        self.ax.set_title(f'{rec_name}Cycle {self.current_cycle_idx + 1}{mode_indicator}{status}: Drag windows to select templates')
        self.ax.set_xlim(0, n_samples / config.FSAMP)

        # Set explicit y-axis limits based on EMG amplitude
        y_margin = max_val * 1.3
        self.ax.set_ylim(-y_margin, y_margin)

        # Legend outside the plot area to avoid overlap
        self.ax.legend(loc='upper left', fontsize=7, framealpha=0.8)
        self.ax.grid(True, alpha=0.3)

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

    def __init__(self, recordings: List[dict], template_manager: TemplateManager, parent=None):
        super().__init__(parent)
        if isinstance(recordings, dict):
            self.recordings = [recordings]
        else:
            self.recordings = recordings
        self.template_manager = template_manager
        self.saved = False

        n_recordings = len(self.recordings)
        self.setWindowTitle(f"Review & Extract Templates ({n_recordings} recording{'s' if n_recordings > 1 else ''})")
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
            all_cycles.extend(cycles)

        # Get recording name from first recording
        recording_name = self.recordings[0].get('label', 'Recording') if self.recordings else None

        self.cycle_viewer.set_cycles(all_cycles, recording_name=recording_name)
        print(f"[REVIEW] Total: {len(all_cycles)} complete cycles")
        self._update_status()

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
        mode_suffix = "sd" if config.ENABLE_DIFFERENTIAL_MODE else "mp"

        if label:
            folder_name = f"templates_{mode_suffix}_{timestamp}_{label}"
        else:
            folder_name = f"templates_{mode_suffix}_{timestamp}"

        templates_base_dir = "data/templates"
        folder_path = os.path.join(templates_base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        saved_files = []

        # Save CLOSED templates
        if closed_templates:
            closed_data = {
                "templates": closed_templates,
                "metadata": {
                    "class_label": "closed",
                    "n_templates": len(closed_templates),
                    "template_duration_s": self.template_manager.template_duration_s,
                    "created_at": now.isoformat(),
                    "label": label,
                    "differential_mode": config.ENABLE_DIFFERENTIAL_MODE,
                }
            }
            closed_path = os.path.join(folder_path, "templates_closed.pkl")
            with open(closed_path, "wb") as f:
                pickle.dump(closed_data, f)
            saved_files.append(closed_path)
            print(f"[REVIEW] Saved {len(closed_templates)} CLOSED templates to {closed_path}")

        # Save OPEN templates
        if open_templates:
            open_data = {
                "templates": open_templates,
                "metadata": {
                    "class_label": "open",
                    "n_templates": len(open_templates),
                    "template_duration_s": self.template_manager.template_duration_s,
                    "created_at": now.isoformat(),
                    "label": label,
                    "differential_mode": config.ENABLE_DIFFERENTIAL_MODE,
                }
            }
            open_path = os.path.join(folder_path, "templates_open.pkl")
            with open(open_path, "wb") as f:
                pickle.dump(open_data, f)
            saved_files.append(open_path)
            print(f"[REVIEW] Saved {len(open_templates)} OPEN templates to {open_path}")

        self.saved = True

        msg = f"Templates saved successfully!\n\n"
        msg += f"CLOSED templates: {len(closed_templates)}\n"
        msg += f"OPEN templates: {len(open_templates)}\n\n"
        msg += f"Saved to folder:\n{folder_path}"

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
        self.setMinimumSize(1000, 600)
        self.resize(1200, 700)
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

        # Channel selector
        ch_layout = QHBoxLayout()
        ch_layout.addWidget(QLabel("Channel:"))
        self.closed_channel_combo = QComboBox()
        n_ch = self.templates_closed[0].shape[0] if self.templates_closed else 32
        for i in range(1, n_ch + 1):
            self.closed_channel_combo.addItem(str(i))
        self.closed_channel_combo.currentIndexChanged.connect(self._update_closed_plot)
        ch_layout.addWidget(self.closed_channel_combo)
        ch_layout.addStretch()
        closed_layout.addLayout(ch_layout)

        # Plot canvas
        self.closed_figure = Figure(figsize=(5, 4), dpi=100)
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

        # Channel selector
        ch_layout2 = QHBoxLayout()
        ch_layout2.addWidget(QLabel("Channel:"))
        self.open_channel_combo = QComboBox()
        n_ch_open = self.templates_open[0].shape[0] if self.templates_open else 32
        for i in range(1, n_ch_open + 1):
            self.open_channel_combo.addItem(str(i))
        self.open_channel_combo.currentIndexChanged.connect(self._update_open_plot)
        ch_layout2.addWidget(self.open_channel_combo)
        ch_layout2.addStretch()
        open_layout.addLayout(ch_layout2)

        # Plot canvas
        self.open_figure = Figure(figsize=(5, 4), dpi=100)
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

    def _update_closed_plot(self):
        self.closed_ax.clear()
        idx = self.closed_list.currentRow()
        if idx < 0 or idx >= len(self.templates_closed):
            self.closed_ax.text(0.5, 0.5, "No template selected", ha='center', va='center',
                               transform=self.closed_ax.transAxes)
            self.closed_canvas.draw()
            return

        template = self.templates_closed[idx]
        channel = self.closed_channel_combo.currentIndex()
        if channel >= template.shape[0]:
            channel = 0

        signal = template[channel, :]
        time_axis = np.arange(len(signal)) / config.FSAMP

        self.closed_ax.plot(time_axis, signal, 'r-', linewidth=0.8)
        self.closed_ax.set_xlabel('Time (s)')
        self.closed_ax.set_ylabel('Amplitude (µV)')
        self.closed_ax.set_title(f'CLOSED Template {idx+1}, Channel {channel+1}')
        self.closed_ax.grid(True, alpha=0.3)
        self.closed_figure.tight_layout()
        self.closed_canvas.draw()

    def _update_open_plot(self):
        self.open_ax.clear()
        idx = self.open_list.currentRow()
        if idx < 0 or idx >= len(self.templates_open):
            self.open_ax.text(0.5, 0.5, "No template selected", ha='center', va='center',
                             transform=self.open_ax.transAxes)
            self.open_canvas.draw()
            return

        template = self.templates_open[idx]
        channel = self.open_channel_combo.currentIndex()
        if channel >= template.shape[0]:
            channel = 0

        signal = template[channel, :]
        time_axis = np.arange(len(signal)) / config.FSAMP

        self.open_ax.plot(time_axis, signal, 'b-', linewidth=0.8)
        self.open_ax.set_xlabel('Time (s)')
        self.open_ax.set_ylabel('Amplitude (µV)')
        self.open_ax.set_title(f'OPEN Template {idx+1}, Channel {channel+1}')
        self.open_ax.grid(True, alpha=0.3)
        self.open_figure.tight_layout()
        self.open_canvas.draw()


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

        # File management:
        self.recordings_dir_path: str = "data/recordings/"
        self.models_dir_path: str = "data/models/"
        self.datasets_dir_path: str = "data/datasets/"
        self.legacy_data_path: str = "data/legacy/"

        # Initialize Template Extraction UI
        self._setup_template_extraction_ui()

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
        self.custom_window_spinbox.setRange(50, 500)
        self.custom_window_spinbox.setValue(96)
        self.custom_overlap_label = QLabel("Overlap (ms):")
        self.custom_overlap_spinbox = QSpinBox()
        self.custom_overlap_spinbox.setRange(10, 200)
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
        self.feature_combo.setCurrentText("wl")  # Default to waveform length
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

        # Row 6: Distance aggregation method
        self.distance_agg_label = QLabel("Distance Aggregation:")
        self.distance_agg_combo = QComboBox()
        self.distance_agg_combo.addItems([
            "Average of 3 smallest (Recommended)",
            "Minimum distance",
            "Average of all"
        ])
        self.distance_agg_combo.setToolTip("How to compute final distance from multiple templates")
        layout.addWidget(self.distance_agg_label, 6, 0, 1, 1)
        layout.addWidget(self.distance_agg_combo, 6, 1, 1, 2)

        # Row 7: Post-prediction smoothing
        self.smoothing_label = QLabel("State Smoothing:")
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItems([
            "Majority Vote (5 samples)",
            "5 Consecutive",
            "None"
        ])
        self.smoothing_combo.setToolTip("Method to smooth state transitions")
        layout.addWidget(self.smoothing_label, 7, 0, 1, 1)
        layout.addWidget(self.smoothing_combo, 7, 1, 1, 2)

        # Row 8: Model name (reuse existing widget, just reposition)
        layout.addWidget(self.main_window.ui.label_8, 8, 0, 1, 1)
        layout.addWidget(self.training_model_label_line_edit, 8, 1, 1, 2)

        # Row 9: Create Model button
        self.create_model_btn = QPushButton("Create Model")
        self.create_model_btn.clicked.connect(self._create_dtw_model)
        self.create_model_btn.setEnabled(False)
        layout.addWidget(self.create_model_btn, 9, 0, 1, 3)

        # Row 10: Progress bar
        self.model_creation_progress_bar = self.main_window.ui.trainingProgressBar
        self.model_creation_progress_bar.setVisible(True)
        self.model_creation_progress_bar.setValue(0)
        layout.addWidget(self.model_creation_progress_bar, 10, 0, 1, 3)

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
            "Manual Selection",     # index 0 - opens review dialog with two windows
            "After Audio Cue",      # index 1 - template starts at audio cue
            "After Reaction Time",  # index 2 - template starts after reaction time
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
            # Index 0: "Manual Selection" - opens review dialog
            # Index 1: "After Audio Cue" - template starts at audio cue
            # Index 2: "After Reaction Time" - template starts after reaction time

            self._manual_selection_mode = (index == 0)

            if index == 0:
                self.template_manager.template_type = "manual"
            elif index == 1:
                self.template_manager.template_type = "after_audio_cue"
            elif index == 2:
                self.template_manager.template_type = "after_reaction_time"
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

                if all(key in recording for key in mindmove_vh_keys):
                    valid_recordings.append(filepath)
                elif all(key in recording for key in mindmove_kb_keys):
                    valid_recordings.append(filepath)
                elif all(key in recording for key in vhi_keys):
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
                    # Extract activations from this recording
                    self.template_manager.extract_activations_from_recording(
                        recording,
                        class_label,
                        include_pre_activation=include_pre_activation
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

                    # Extract activations from this recording
                    self.template_manager.extract_activations_from_recording(
                        recording,
                        class_label,
                        include_pre_activation=include_pre_activation
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

        for i, duration in enumerate(durations):
            item = QListWidgetItem(f"Activation {i + 1}: {duration:.2f}s")
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
        for i, activation in enumerate(activations):
            print(f"\nProcessing activation {i + 1}/{len(activations)}...")

            # Get the GT signal for this activation (reconstruct from context)
            # For manual mode, we include manual_context_before_s before GT=1
            n_samples = activation.shape[1]
            pre_samples = int(self.template_manager.manual_context_before_s * config.FSAMP)

            # Build a simple GT overlay: 0 before GT=1, 1 after
            # The activation starts with pre_samples of GT=0, then GT=1
            gt_overlay = np.zeros(n_samples)
            if pre_samples < n_samples:
                gt_overlay[pre_samples:] = 1

            # Call the interactive selection for this activation
            template = self._interactive_template_selection(
                activation,
                gt_overlay,
                channel,
                template_samples,
                activation_idx=i + 1,
                total_activations=len(activations)
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
        total_activations: int
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
        dialog.setWindowTitle(f'Manual Selection - Activation {activation_idx}/{total_activations}')
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

    def _open_template_review(self) -> None:
        """Open the template review dialog showing full cycles for manual adjustment.

        This opens the same GuidedRecordingReviewDialog used for manual selection,
        allowing the user to review and adjust template positions visually.
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

        print(f"\n[TRAINING] Opening template review with {len(self._guided_recordings_for_review)} recording(s)")

        # Open the review dialog (same as manual selection)
        review_dialog = GuidedRecordingReviewDialog(
            self._guided_recordings_for_review, self.template_manager, self.main_window
        )
        result = review_dialog.exec()

        if review_dialog.saved:
            # Update template counts
            n_closed = len(self.template_manager.templates.get("closed", []))
            n_open = len(self.template_manager.templates.get("open", []))
            self.template_count_label.setText(f"Updated: {n_closed} closed, {n_open} open")
            print(f"[TRAINING] Templates updated: {n_closed} closed, {n_open} open")

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
            # Check if it's the old format (list of templates)
            elif isinstance(data, list):
                n_templates = len(data)
                label_text = f"{n_templates} templates (legacy format)"
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
                print(f"    Template {idx}: avg={avg:.4f} ({sigma:.1f}σ above mean)")
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
            compute_per_template_statistics,
            compute_cross_class_distances,
            compute_threshold_presets
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

        # Extract features from templates
        print("\nExtracting features...")
        feature_fn = FEATURES[feature_name]["function"]

        open_templates_features = []
        for template in open_templates_raw:
            windowed = sliding_window(template, window_samples, overlap_samples)
            features = feature_fn(windowed)
            open_templates_features.append(features)

        closed_templates_features = []
        for template in closed_templates_raw:
            windowed = sliding_window(template, window_samples, overlap_samples)
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

        # compute_threshold returns: (mean, std, threshold)
        mean_open, std_open, threshold_open = compute_threshold(open_templates_features)
        mean_closed, std_closed, threshold_closed = compute_threshold(closed_templates_features)

        # Restore config
        config.USE_NUMBA_DTW = original_numba
        config.USE_TSLEARN_DTW = original_tslearn

        print(f"  Open threshold: {threshold_open:.4f} (mean: {mean_open:.4f}, std: {std_open:.4f})")
        print(f"  Closed threshold: {threshold_closed:.4f} (mean: {mean_closed:.4f}, std: {std_closed:.4f})")

        # Compute cross-class distances for intelligent threshold presets
        print("\nComputing cross-class distances for threshold presets...")
        mean_cross, std_cross, cross_distances = compute_cross_class_distances(
            open_templates_features,
            closed_templates_features,
            active_channels=None  # Will use config.active_channels
        )
        print(f"  Cross-class distance: mean={mean_cross:.4f}, std={std_cross:.4f}")
        print(f"  Number of cross-class comparisons: {len(cross_distances)}")

        # Compute threshold presets for OPEN class
        print("\nComputing threshold presets for OPEN...")
        open_presets = compute_threshold_presets(mean_open, std_open, mean_cross, std_cross)
        for key, preset in open_presets.items():
            print(f"  {preset['name']}: threshold={preset['threshold']:.4f} (s={preset['s']:.2f})")

        # Compute threshold presets for CLOSED class
        print("\nComputing threshold presets for CLOSED...")
        closed_presets = compute_threshold_presets(mean_closed, std_closed, mean_cross, std_cross)
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

        # Save model
        print("\nSaving model...")
        now = datetime.now()
        formatted_now = now.strftime("%Y%m%d_%H%M%S")

        # Get mode suffix (_mp_ for monopolar, _sd_ for single differential)
        mode_suffix = "sd" if config.ENABLE_DIFFERENTIAL_MODE else "mp"

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
            "differential_mode": config.ENABLE_DIFFERENTIAL_MODE,
            # Cross-class statistics for intelligent threshold presets
            "mean_cross": mean_cross,
            "std_cross": std_cross,
            # Threshold presets (computed from intra-class and cross-class statistics)
            "threshold_presets": threshold_presets,
            "parameters": {
                "window_samples": window_samples,
                "overlap_samples": overlap_samples,
                "window_ms": window_samples / config.FSAMP * 1000,
                "overlap_ms": overlap_samples / config.FSAMP * 1000,
                "dtw_algorithm": dtw_algorithm,
                "fsamp": config.FSAMP,
                "num_channels": config.num_channels,
            },
            "metadata": {
                "created_at": now.isoformat(),
                "model_name": model_name,
                "n_open_templates": len(open_templates_features),
                "n_closed_templates": len(closed_templates_features),
                "dead_channels_display": dead_channels_display,  # 1-indexed for display
                "differential_mode": config.ENABLE_DIFFERENTIAL_MODE,
            }
        }

        models_dir = "data/models"
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"MindMove_Model_{mode_suffix}_{formatted_now}_{model_name}.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {model_path}")
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
