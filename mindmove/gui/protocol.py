""" """

from __future__ import annotations
from typing import TYPE_CHECKING, Union, Optional
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QRadioButton
from mindmove.gui.protocols.record import RecordProtocol
from mindmove.gui.protocols.training import TrainingProtocol
from mindmove.gui.protocols.online import OnlineProtocol
from mindmove.gui.protocols.guided_record import GuidedRecordProtocol

if TYPE_CHECKING:
    from mindmove.gui.mindmove import MindMove


class Protocol(QObject):
    def __init__(self, parent: MindMove | None = ...) -> None:
        super().__init__(parent)

        self.main_window = parent

        # Initialize Protocol UI
        self._setup_procotol_ui()

        # Initialize Protocol
        self.current_protocol: Optional[
            Union[RecordProtocol, TrainingProtocol, OnlineProtocol, GuidedRecordProtocol]
        ] = None

        self.available_protocols: list[
            Union[RecordProtocol, TrainingProtocol, OnlineProtocol, GuidedRecordProtocol]
        ] = [
            RecordProtocol(self.main_window),
            TrainingProtocol(self.main_window),
            OnlineProtocol(self.main_window),
            GuidedRecordProtocol(self.main_window),
        ]

        # Add custom protocol widgets to stacked widget
        self._add_custom_protocol_widgets()

    def _protocol_record_toggled(self, checked: bool) -> None:
        if checked:
            self.protocol_mode_stacked_widget.setCurrentIndex(0)
            self.current_protocol = self.available_protocols[0]

            print("Record Protocol toggled")

    def _protocol_training_toggled(self, checked: bool) -> None:
        if checked:
            self.protocol_mode_stacked_widget.setCurrentIndex(1)
            self.current_protocol = self.available_protocols[1]

            print("Training Protocol toggled")

    def _protocol_online_toggled(self, checked: bool) -> None:
        if checked:
            self.protocol_mode_stacked_widget.setCurrentIndex(2)
            self.current_protocol = self.available_protocols[2]

            print("Online Protocol toggled")

    def _protocol_guided_toggled(self, checked: bool) -> None:
        if checked:
            self.protocol_mode_stacked_widget.setCurrentIndex(3)
            self.current_protocol = self.available_protocols[3]

            print("Guided Record Protocol toggled")

    def _add_custom_protocol_widgets(self) -> None:
        """Add custom protocol widgets to the stacked widget."""
        # Add Guided Record widget (index 3)
        guided_protocol = self.available_protocols[3]
        guided_widget = guided_protocol.get_widget()
        self.protocol_mode_stacked_widget.addWidget(guided_widget)

    def _setup_procotol_ui(self):
        self.protocol_mode_stacked_widget = (
            self.main_window.ui.protocolModeStackedWidget
        )
        self.protocol_mode_stacked_widget.setCurrentIndex(0)
        self.protocol_record_radio_button = (
            self.main_window.ui.protocolRecordRadioButton
        )
        self.protocol_record_radio_button.setChecked(True)
        self.protocol_record_radio_button.toggled.connect(self._protocol_record_toggled)
        self.protocol_training_radio_button = (
            self.main_window.ui.protocolTrainingRadioButton
        )
        self.protocol_training_radio_button.toggled.connect(
            self._protocol_training_toggled
        )
        self.protocol_online_radio_button = (
            self.main_window.ui.protocolOnlineRadioButton
        )
        self.protocol_online_radio_button.toggled.connect(self._protocol_online_toggled)

        # Add Guided Record radio button programmatically
        self._add_guided_radio_button()

    def _add_guided_radio_button(self) -> None:
        """Add the Guided Record radio button to the protocol selection."""
        # Find the layout containing the other radio buttons
        parent_layout = self.protocol_online_radio_button.parent().layout()

        # Create Guided Record radio button
        self.protocol_guided_radio_button = QRadioButton("Guided Record")
        self.protocol_guided_radio_button.toggled.connect(self._protocol_guided_toggled)
        if parent_layout:
            parent_layout.addWidget(self.protocol_guided_radio_button)
