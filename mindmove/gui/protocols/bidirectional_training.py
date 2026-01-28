"""
Bidirectional Training Protocol for template extraction from guided recordings.

This protocol provides a specialized interface for reviewing guided recordings
and extracting both CLOSED and OPEN templates from complete open→close→open cycles.

Features:
- Load guided recordings from file
- View complete cycles with audio cue markers
- Draggable template windows for precise selection
- Linear GT visualization during transitions
- Save extracted templates
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Dict, Tuple
from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QFileDialog, QMessageBox,
    QListWidget, QListWidgetItem, QSplitter
)
import pickle
import os
import numpy as np
from datetime import datetime

# Matplotlib imports
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import SpanSelector

from mindmove.config import config
from mindmove.model.templates.template_manager import TemplateManager

if TYPE_CHECKING:
    from mindmove.gui.mindmove import MindMove


class DraggableTemplateWindow:
    """A draggable rectangle representing a 1-second template window."""

    def __init__(self, ax, start_time: float, duration: float, color: str, label: str):
        self.ax = ax
        self.start_time = start_time
        self.duration = duration
        self.color = color
        self.label = label

        # Create rectangle patch
        self.rect = Rectangle(
            (start_time, -1e6), duration, 2e6,
            alpha=0.3, facecolor=color, edgecolor=color, linewidth=2
        )
        ax.add_patch(self.rect)

        # Create start/end line markers
        self.start_line = ax.axvline(start_time, color=color, linewidth=2, linestyle='-')
        self.end_line = ax.axvline(start_time + duration, color=color, linewidth=2, linestyle='-')

        # Dragging state
        self.dragging = False
        self.drag_start_x = None

    def set_position(self, start_time: float):
        """Set the position of the template window."""
        self.start_time = start_time
        self.rect.set_x(start_time)
        self.start_line.set_xdata([start_time, start_time])
        self.end_line.set_xdata([start_time + self.duration, start_time + self.duration])

    def contains(self, x: float) -> bool:
        """Check if x coordinate is within the window."""
        return self.start_time <= x <= self.start_time + self.duration

    def get_time_range(self) -> Tuple[float, float]:
        """Return (start_time, end_time)."""
        return self.start_time, self.start_time + self.duration


class CycleViewerWidget(QWidget):
    """
    Widget for viewing and selecting templates from complete cycles.

    Shows one cycle at a time with:
    - 2s context before close cue
    - The complete cycle
    - 2s context after opening
    - Blue vertical lines at audio cue times
    - GT signal overlapped with EMG
    - Draggable template windows
    """

    templates_accepted = Signal(np.ndarray, np.ndarray)  # closed_template, open_template

    def __init__(self, parent=None):
        super().__init__(parent)

        self.recording: Optional[dict] = None
        self.cycles: List[dict] = []
        self.current_cycle_idx: int = 0
        self.current_channel: int = 0

        # Template windows
        self.closed_window: Optional[DraggableTemplateWindow] = None
        self.open_window: Optional[DraggableTemplateWindow] = None
        self.template_duration_s = 1.0

        # Accepted templates
        self.accepted_closed_templates: List[np.ndarray] = []
        self.accepted_open_templates: List[np.ndarray] = []

        # Mouse dragging
        self._dragging_window: Optional[DraggableTemplateWindow] = None
        self._drag_offset: float = 0

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header
        header_layout = QHBoxLayout()

        self.cycle_label = QLabel("Cycle 0 / 0")
        self.cycle_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(self.cycle_label)

        header_layout.addStretch()

        self.accepted_label = QLabel("Accepted: 0 CLOSED, 0 OPEN")
        self.accepted_label.setStyleSheet("color: green;")
        header_layout.addWidget(self.accepted_label)

        header_layout.addSpacing(20)

        header_layout.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.setFixedWidth(60)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        header_layout.addWidget(self.channel_combo)

        layout.addLayout(header_layout)

        # Instructions
        self.instruction_label = QLabel(
            "Drag the orange (CLOSED) and blue (OPEN) windows to select 1-second templates. "
            "Click 'Accept' when satisfied."
        )
        self.instruction_label.setStyleSheet(
            "color: #666; background-color: #f5f5f5; padding: 8px; border-radius: 4px;"
        )
        self.instruction_label.setWordWrap(True)
        layout.addWidget(self.instruction_label)

        # Matplotlib figure
        self.figure = Figure(figsize=(12, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(400)
        self.ax = self.figure.add_subplot(111)

        # Connect mouse events for dragging
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

        layout.addWidget(self.canvas)

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

        self.reset_button = QPushButton("Reset Windows")
        self.reset_button.clicked.connect(self._reset_windows)
        button_layout.addWidget(self.reset_button)

        self.accept_button = QPushButton("✓ Accept Templates")
        self.accept_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 16px; }"
        )
        self.accept_button.clicked.connect(self._accept_templates)
        button_layout.addWidget(self.accept_button)

        self.skip_button = QPushButton("Skip Cycle")
        self.skip_button.clicked.connect(self._go_next)
        button_layout.addWidget(self.skip_button)

        layout.addLayout(button_layout)

    def load_recording(self, recording: dict):
        """Load a guided recording and extract cycles."""
        self.recording = recording
        self.cycles = []
        self.current_cycle_idx = 0
        self.accepted_closed_templates = []
        self.accepted_open_templates = []

        # Extract cycles from recording
        self._extract_cycles_from_recording()

        # Update channel combo
        n_channels = recording['emg'].shape[0]
        self.channel_combo.clear()
        for i in range(1, n_channels + 1):
            self.channel_combo.addItem(str(i))

        self._update_display()

    def _extract_cycles_from_recording(self):
        """Extract cycle data from the recording."""
        if self.recording is None:
            return

        emg = self.recording['emg']
        gt = self.recording['gt']
        emg_timestamps = self.recording['timings_emg']
        cycle_boundaries = self.recording.get('cycles', [])
        animation_config = self.recording.get('animation_config', {})

        # Get timing parameters
        hold_open_s = animation_config.get('hold_open_s', 2.0)
        closing_s = animation_config.get('closing_s', 0.35)
        hold_closed_s = animation_config.get('hold_closed_s', 2.0)
        opening_s = animation_config.get('opening_s', 0.35)
        reaction_time_s = animation_config.get('reaction_time_s', 0.2)

        recording_start_time = emg_timestamps[0] if len(emg_timestamps) > 0 else 0

        for cycle_info in cycle_boundaries:
            cycle_start = cycle_info.get('start_time', 0)
            cycle_end = cycle_info.get('end_time', cycle_start + 10)
            close_cue_time = cycle_info.get('close_cue_time')
            open_cue_time = cycle_info.get('open_cue_time')

            if close_cue_time is None or open_cue_time is None:
                continue

            # Calculate view window: 2s before close cue to 2s after cycle ends
            view_start_time = close_cue_time - 2.0
            view_end_time = cycle_end + 2.0

            # Convert to sample indices
            view_start_idx = self._time_to_sample_idx(view_start_time, recording_start_time, emg.shape[1], emg_timestamps)
            view_end_idx = self._time_to_sample_idx(view_end_time, recording_start_time, emg.shape[1], emg_timestamps)

            # Clamp to valid range
            view_start_idx = max(0, view_start_idx)
            view_end_idx = min(emg.shape[1], view_end_idx)

            if view_end_idx <= view_start_idx:
                continue

            # Extract EMG and GT for this view
            cycle_emg = emg[:, view_start_idx:view_end_idx]
            cycle_gt = gt[view_start_idx:view_end_idx]

            # Calculate relative times within the view
            view_start_abs_time = view_start_time
            close_cue_relative = close_cue_time - view_start_abs_time
            open_cue_relative = open_cue_time - view_start_abs_time

            # Initial window positions (centered on expected template locations)
            # CLOSED template: during closing phase (after reaction time)
            closed_window_start = close_cue_relative + reaction_time_s
            # OPEN template: during opening phase (after reaction time)
            open_window_start = open_cue_relative + reaction_time_s

            self.cycles.append({
                'emg': cycle_emg,
                'gt': cycle_gt,
                'close_cue_relative': close_cue_relative,
                'open_cue_relative': open_cue_relative,
                'closed_window_start': closed_window_start,
                'open_window_start': open_window_start,
                'duration_s': cycle_emg.shape[1] / config.FSAMP,
                'cycle_number': cycle_info.get('cycle_number', 0),
                'timing_config': cycle_info.get('timing_config', animation_config),
            })

    def _time_to_sample_idx(self, abs_time: float, recording_start: float, n_samples: int, timestamps: np.ndarray) -> int:
        """Convert absolute time to sample index."""
        if len(timestamps) == 0:
            return 0
        recording_duration = timestamps[-1] - timestamps[0]
        if recording_duration <= 0:
            return 0
        relative_time = abs_time - recording_start
        return int((relative_time / recording_duration) * n_samples)

    def _on_channel_changed(self, index: int):
        self.current_channel = index
        self._update_display()

    def _reset_windows(self):
        """Reset template windows to default positions."""
        if self.current_cycle_idx < len(self.cycles):
            cycle = self.cycles[self.current_cycle_idx]
            if self.closed_window:
                self.closed_window.set_position(cycle['closed_window_start'])
            if self.open_window:
                self.open_window.set_position(cycle['open_window_start'])
            self.canvas.draw()

    def _update_display(self):
        """Update the plot."""
        n_cycles = len(self.cycles)

        # Update labels
        self.cycle_label.setText(f"Cycle {self.current_cycle_idx + 1} / {n_cycles}" if n_cycles > 0 else "No cycles")
        self.cycle_index_label.setText(f"{self.current_cycle_idx + 1} / {n_cycles}" if n_cycles > 0 else "0 / 0")
        self.prev_button.setEnabled(self.current_cycle_idx > 0)
        self.next_button.setEnabled(self.current_cycle_idx < n_cycles - 1)
        self.skip_button.setEnabled(self.current_cycle_idx < n_cycles - 1)

        self.accepted_label.setText(
            f"Accepted: {len(self.accepted_closed_templates)} CLOSED, "
            f"{len(self.accepted_open_templates)} OPEN"
        )

        # Clear and redraw
        self.ax.clear()
        self.closed_window = None
        self.open_window = None

        if n_cycles == 0 or self.current_cycle_idx >= n_cycles:
            self.ax.text(0.5, 0.5, "No cycles to display", ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return

        cycle = self.cycles[self.current_cycle_idx]
        emg = cycle['emg']
        gt = cycle['gt']
        close_cue_rel = cycle['close_cue_relative']
        open_cue_rel = cycle['open_cue_relative']

        n_samples = emg.shape[1]
        time_axis = np.arange(n_samples) / config.FSAMP

        # Get channel EMG
        channel = min(self.current_channel, emg.shape[0] - 1)
        emg_signal = emg[channel, :]

        # Normalize for display
        emg_min, emg_max = np.min(emg_signal), np.max(emg_signal)
        emg_range = emg_max - emg_min if emg_max > emg_min else 1

        # Plot GT as background (scaled to EMG range)
        gt_scaled = gt * emg_range + emg_min
        self.ax.fill_between(time_axis, emg_min, gt_scaled, alpha=0.2, color='orange', label='GT (0=open, 1=closed)')

        # Plot EMG
        self.ax.plot(time_axis, emg_signal, 'k-', linewidth=0.5, alpha=0.9, label=f'Channel {channel + 1}')

        # Plot audio cue markers
        self.ax.axvline(close_cue_rel, color='blue', linewidth=2, linestyle='--', label='Close cue')
        self.ax.axvline(open_cue_rel, color='blue', linewidth=2, linestyle='--', label='Open cue')

        # Create draggable template windows
        self.closed_window = DraggableTemplateWindow(
            self.ax, cycle['closed_window_start'], self.template_duration_s,
            '#FF5722', 'CLOSED'
        )
        self.open_window = DraggableTemplateWindow(
            self.ax, cycle['open_window_start'], self.template_duration_s,
            '#2196F3', 'OPEN'
        )

        # Labels and formatting
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel(f'EMG (µV)')
        self.ax.set_title(f'Cycle {cycle["cycle_number"]}: Drag windows to select templates')
        self.ax.set_xlim(0, time_axis[-1])
        self.ax.legend(loc='upper right', fontsize=8)
        self.ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def _on_mouse_press(self, event):
        """Handle mouse press for dragging."""
        if event.inaxes != self.ax or event.xdata is None:
            return

        x = event.xdata

        # Check if clicking on a window
        if self.closed_window and self.closed_window.contains(x):
            self._dragging_window = self.closed_window
            self._drag_offset = x - self.closed_window.start_time
        elif self.open_window and self.open_window.contains(x):
            self._dragging_window = self.open_window
            self._drag_offset = x - self.open_window.start_time

    def _on_mouse_release(self, event):
        """Handle mouse release."""
        self._dragging_window = None

    def _on_mouse_move(self, event):
        """Handle mouse move for dragging."""
        if self._dragging_window is None or event.xdata is None:
            return

        if self.current_cycle_idx >= len(self.cycles):
            return

        cycle = self.cycles[self.current_cycle_idx]
        max_time = cycle['duration_s'] - self.template_duration_s

        new_start = event.xdata - self._drag_offset
        new_start = max(0, min(new_start, max_time))

        self._dragging_window.set_position(new_start)
        self.canvas.draw()

    def _go_prev(self):
        if self.current_cycle_idx > 0:
            self.current_cycle_idx -= 1
            self._update_display()

    def _go_next(self):
        if self.current_cycle_idx < len(self.cycles) - 1:
            self.current_cycle_idx += 1
            self._update_display()

    def _accept_templates(self):
        """Accept the current template windows."""
        if self.current_cycle_idx >= len(self.cycles):
            return
        if self.closed_window is None or self.open_window is None:
            return

        cycle = self.cycles[self.current_cycle_idx]
        emg = cycle['emg']

        # Extract CLOSED template
        closed_start_s, closed_end_s = self.closed_window.get_time_range()
        closed_start_idx = int(closed_start_s * config.FSAMP)
        closed_end_idx = int(closed_end_s * config.FSAMP)
        closed_end_idx = min(closed_end_idx, emg.shape[1])

        if closed_end_idx > closed_start_idx:
            closed_template = emg[:, closed_start_idx:closed_end_idx]
            self.accepted_closed_templates.append(closed_template)

        # Extract OPEN template
        open_start_s, open_end_s = self.open_window.get_time_range()
        open_start_idx = int(open_start_s * config.FSAMP)
        open_end_idx = int(open_end_s * config.FSAMP)
        open_end_idx = min(open_end_idx, emg.shape[1])

        if open_end_idx > open_start_idx:
            open_template = emg[:, open_start_idx:open_end_idx]
            self.accepted_open_templates.append(open_template)

        # Emit signal
        if closed_end_idx > closed_start_idx and open_end_idx > open_start_idx:
            self.templates_accepted.emit(
                self.accepted_closed_templates[-1],
                self.accepted_open_templates[-1]
            )

        # Move to next cycle
        if self.current_cycle_idx < len(self.cycles) - 1:
            self.current_cycle_idx += 1
            self._update_display()
        else:
            self.accepted_label.setText(
                f"All cycles done! Accepted: {len(self.accepted_closed_templates)} CLOSED, "
                f"{len(self.accepted_open_templates)} OPEN"
            )
            self._update_display()

    def get_accepted_templates(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return (closed_templates, open_templates)."""
        return self.accepted_closed_templates, self.accepted_open_templates


class BidirectionalTrainingProtocol(QObject):
    """
    Protocol for extracting templates from guided recordings.

    Provides a cycle-based view for precise template selection with
    draggable windows and audio cue markers.
    """

    def __init__(self, parent: MindMove | None = None) -> None:
        super().__init__(parent)

        self.main_window = parent
        self.template_manager = TemplateManager()

        self.recordings_dir_path = "data/recordings/"
        self.templates_dir_path = "data/templates/"

        self.loaded_recordings: List[dict] = []

        self._setup_ui()

    def _setup_ui(self):
        """Create the UI for bidirectional training."""
        self.main_widget = QWidget()
        main_layout = QVBoxLayout(self.main_widget)

        # === Load Recordings Section ===
        load_group = QGroupBox("Load Guided Recordings")
        load_layout = QVBoxLayout(load_group)

        load_buttons = QHBoxLayout()

        self.load_button = QPushButton("Load Recording(s)")
        self.load_button.clicked.connect(self._load_recordings)
        load_buttons.addWidget(self.load_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._clear_recordings)
        load_buttons.addWidget(self.clear_button)

        load_buttons.addStretch()
        load_layout.addLayout(load_buttons)

        self.recordings_list = QListWidget()
        self.recordings_list.setMaximumHeight(100)
        load_layout.addWidget(self.recordings_list)

        self.load_status_label = QLabel("No recordings loaded")
        self.load_status_label.setStyleSheet("color: #666;")
        load_layout.addWidget(self.load_status_label)

        main_layout.addWidget(load_group)

        # === Cycle Viewer ===
        viewer_group = QGroupBox("Cycle Review & Template Selection")
        viewer_layout = QVBoxLayout(viewer_group)

        self.cycle_viewer = CycleViewerWidget()
        self.cycle_viewer.templates_accepted.connect(self._on_templates_accepted)
        viewer_layout.addWidget(self.cycle_viewer)

        main_layout.addWidget(viewer_group, stretch=1)

        # === Save Section ===
        save_group = QGroupBox("Save Templates")
        save_layout = QHBoxLayout(save_group)

        save_layout.addWidget(QLabel("Label:"))
        from PySide6.QtWidgets import QLineEdit
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("e.g., subject1_session1")
        self.label_edit.setMaximumWidth(200)
        save_layout.addWidget(self.label_edit)

        save_layout.addStretch()

        self.save_button = QPushButton("Save All Templates")
        self.save_button.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px 20px; }"
        )
        self.save_button.clicked.connect(self._save_templates)
        save_layout.addWidget(self.save_button)

        main_layout.addWidget(save_group)

    def _load_recordings(self):
        """Load guided recording files."""
        if not os.path.exists(self.recordings_dir_path):
            os.makedirs(self.recordings_dir_path)

        dialog = QFileDialog(self.main_widget)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilter("Guided Recordings (*GuidedRecording*.pkl)")
        dialog.setDirectory(self.recordings_dir_path)

        filenames, _ = dialog.getOpenFileNames()

        if not filenames:
            return

        for filepath in filenames:
            try:
                with open(filepath, 'rb') as f:
                    recording = pickle.load(f)

                # Validate it's a guided recording
                if recording.get('gt_mode') != 'guided_animation':
                    print(f"Skipping {filepath}: not a guided recording")
                    continue

                self.loaded_recordings.append(recording)

                # Add to list widget
                label = recording.get('label', 'unknown')
                cycles = recording.get('cycles_completed', 0)
                item = QListWidgetItem(f"{label} ({cycles} cycles)")
                self.recordings_list.addItem(item)

            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        if self.loaded_recordings:
            # Load first recording into viewer
            self.cycle_viewer.load_recording(self.loaded_recordings[0])
            self.load_status_label.setText(f"Loaded {len(self.loaded_recordings)} recording(s)")
            self.load_status_label.setStyleSheet("color: green;")
        else:
            self.load_status_label.setText("No valid recordings found")
            self.load_status_label.setStyleSheet("color: red;")

    def _clear_recordings(self):
        """Clear loaded recordings."""
        self.loaded_recordings = []
        self.recordings_list.clear()
        self.load_status_label.setText("No recordings loaded")
        self.load_status_label.setStyleSheet("color: #666;")

    def _on_templates_accepted(self, closed_template, open_template):
        """Called when templates are accepted from the viewer."""
        # Could add status update here
        pass

    def _save_templates(self):
        """Save all accepted templates."""
        closed_templates, open_templates = self.cycle_viewer.get_accepted_templates()

        if not closed_templates and not open_templates:
            QMessageBox.warning(
                self.main_widget,
                "No Templates",
                "No templates have been accepted. Please review cycles and select templates first."
            )
            return

        # Get label
        label = self.label_edit.text().strip()
        if not label:
            label = "bidirectional"

        # Create folder
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        mode_suffix = "sd" if config.ENABLE_DIFFERENTIAL_MODE else "mp"

        folder_name = f"templates_{mode_suffix}_{timestamp}_{label}"
        folder_path = os.path.join(self.templates_dir_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        saved_files = []

        # Save CLOSED templates
        if closed_templates:
            closed_data = {
                "templates": closed_templates,
                "metadata": {
                    "class_label": "closed",
                    "n_templates": len(closed_templates),
                    "template_duration_s": self.cycle_viewer.template_duration_s,
                    "created_at": now.isoformat(),
                    "label": label,
                    "differential_mode": config.ENABLE_DIFFERENTIAL_MODE,
                    "extraction_method": "bidirectional_training",
                }
            }
            closed_path = os.path.join(folder_path, "templates_closed.pkl")
            with open(closed_path, 'wb') as f:
                pickle.dump(closed_data, f)
            saved_files.append(closed_path)
            print(f"Saved {len(closed_templates)} CLOSED templates")

        # Save OPEN templates
        if open_templates:
            open_data = {
                "templates": open_templates,
                "metadata": {
                    "class_label": "open",
                    "n_templates": len(open_templates),
                    "template_duration_s": self.cycle_viewer.template_duration_s,
                    "created_at": now.isoformat(),
                    "label": label,
                    "differential_mode": config.ENABLE_DIFFERENTIAL_MODE,
                    "extraction_method": "bidirectional_training",
                }
            }
            open_path = os.path.join(folder_path, "templates_open.pkl")
            with open(open_path, 'wb') as f:
                pickle.dump(open_data, f)
            saved_files.append(open_path)
            print(f"Saved {len(open_templates)} OPEN templates")

        # Show confirmation
        msg = f"Templates saved successfully!\n\n"
        msg += f"CLOSED templates: {len(closed_templates)}\n"
        msg += f"OPEN templates: {len(open_templates)}\n\n"
        msg += f"Saved to:\n{folder_path}"

        QMessageBox.information(self.main_widget, "Templates Saved", msg)

    def get_widget(self) -> QWidget:
        """Return the main widget for this protocol."""
        return self.main_widget
