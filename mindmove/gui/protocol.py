""" """

from __future__ import annotations
from typing import TYPE_CHECKING, Union, Optional
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QRadioButton
from mindmove.gui.protocols.record import RecordProtocol
from mindmove.gui.protocols.training import TrainingProtocol
from mindmove.gui.protocols.online import OnlineProtocol

if TYPE_CHECKING:
    from mindmove.gui.mindmove import MindMove


class Protocol(QObject):
    def __init__(self, parent: MindMove | None = ...) -> None:
        super().__init__(parent)

        self.main_window = parent

        # Initialize Protocol UI
        self._setup_procotol_ui()

        # Initialize Protocol
        # Note: GuidedRecordProtocol is now embedded in RecordProtocol
        self.current_protocol: Optional[
            Union[RecordProtocol, TrainingProtocol, OnlineProtocol]
        ] = None

        self.available_protocols: list[
            Union[RecordProtocol, TrainingProtocol, OnlineProtocol]
        ] = [
            RecordProtocol(self.main_window),
            TrainingProtocol(self.main_window),
            OnlineProtocol(self.main_window),
        ]

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

        # Note: Guided Record is now embedded in Record Protocol as a mode selector
