from __future__ import annotations
from typing import Dict, Union, TYPE_CHECKING
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QCheckBox
from mindmove.gui.protocol import Protocol
from mindmove.gui.ui_compiled.main_window import Ui_MindMove
from mindmove.config import config
import numpy as np
import time
from mindmove.gui.virtual_hand_interface import VirtualHandInterface

if TYPE_CHECKING:
    from mindmove.device_interfaces.gui.muovi_widget import MuoviWidget
    from mindmove.gui_custom_elements.vispy_plot_widget import VispyPlotWidget


class MindMove(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MindMove()
        self.ui.setupUi(self)

        # Set column stretch factors for 30/70 layout (controls / plot)
        self.ui.gridLayout.setColumnStretch(0, 3)  # Tab widget (controls) - 30%
        self.ui.gridLayout.setColumnStretch(1, 7)  # Plot widget - 70%

        # Tab Widget
        self.tab_widget = self.ui.mindMoveTabWidget
        self.tab_widget.setCurrentIndex(0)

        # Plot Setup
        self.plot: VispyPlotWidget = self.ui.vispyPlotWidget
        self.plot_enabled_check_box: QCheckBox = self.ui.vispyPlotEnabledCheckBox
        self.plot_enabled_check_box.setStyleSheet(
            "QCheckBox{background-color: rgba(255, 255, 255, 0.2); color: black;}"
        )
        self.plot_enabled_check_box.setChecked(True)

        self.display_time = 5

        # Device Setup
        self.device: MuoviWidget = self.ui.muoviWidget
        self.device.ready_read_signal.connect(self.update)
        self.device.differential_mode_changed.connect(self._on_differential_mode_changed)

        # Procotol Setup
        self.protocol: Protocol = Protocol(self)

        # Output Setup
        self.virtual_hand_interface: VirtualHandInterface = VirtualHandInterface(self)

        # Initialize
        self._prepare_plot()

    def update(self, data: np.ndarray):
        # EMG Data
        emg_data = self.device.extract_emg_data(data)
        if self.plot_enabled_check_box.isChecked():
            self.plot.set_plot_data(emg_data / 1000)

    def _prepare_plot(self):
        sampling_frequency = 2000
        lines = 16 if config.ENABLE_DIFFERENTIAL_MODE else 32
        if sampling_frequency and lines:
            self.plot.refresh_plot()
            self.plot.configure_lines_plot(
                self.display_time,
                fs=sampling_frequency,
                lines=lines,
            )

    def _on_differential_mode_changed(self, is_differential: bool) -> None:
        """Reconfigure plot when differential mode changes."""
        self._prepare_plot()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.device.closeEvent(event)
        self.virtual_hand_interface.closeEvent(event)
        return super().closeEvent(event)
