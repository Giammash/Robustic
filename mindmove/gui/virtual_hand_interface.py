""" """

from __future__ import annotations
import re
from typing import TYPE_CHECKING, Union, Optional
from PySide6.QtCore import QObject, Signal, QByteArray
from PySide6.QtNetwork import QUdpSocket, QHostAddress
from PySide6.QtWidgets import QMessageBox, QLineEdit
from PySide6.QtGui import QCloseEvent
import numpy as np
import ast
import time

if TYPE_CHECKING:
    from mindmove.gui.mindmove import MindMove


class VirtualHandInterface(QObject):
    output_message_signal = Signal(QByteArray)
    input_message_signal = Signal(np.ndarray)

    def __init__(self, parent: MindMove | None = ...) -> None:
        super().__init__(parent)

        self.main_window = parent

        # Initialize Virtual Hand Interface UI
        self._setup_virtual_hand_interface_ui()

        # Initialize MindMove UDP Socket
        self._setup_virtual_hand_interface()

        is_streaming: bool = False

    def _read_message(self) -> None:
        while self.mind_move_udp_socket.hasPendingDatagrams():
            datagram, host, port = self.mind_move_udp_socket.readDatagram(
                self.mind_move_udp_socket.pendingDatagramSize()
            )

            if (
                not host.toString() == self.virtual_hand_interface_udp_ip
                and not port == self.virtual_hand_interface_udp_port
            ):
                continue
            self.input_message_signal.emit(
                np.array(ast.literal_eval(datagram.data().decode("utf-8")))
            )

    def _write_message(self, message: QByteArray) -> None:
        if self.virtual_hand_interface_toggle_streaming_push_button.isChecked():
            if (
                time.time() - self.last_message_time
                < self.time_difference_between_messages
            ):
                return
            self.last_message_time = time.time()
            output_bytes = self.mind_move_udp_socket.writeDatagram(
                message,
                QHostAddress(self.virtual_hand_interface_udp_ip),
                self.virtual_hand_interface_udp_port,
            )

            if output_bytes == -1:
                print("Error sending message")
                return

    def _configure_streaming(self) -> None:
        if self.virtual_hand_interface_configure_toggle_push_button.isChecked():
            self.virtual_hand_interface_udp_ip = (
                self.virtual_hand_interface_udp_ip_line_edit.text()
            )
            self.virtual_hand_interface_udp_port = int(
                self.virtual_hand_interface_udp_port_line_edit.text()
            )

            self.mind_move_udp_ip = self.mindmove_udp_ip_line_edit.text()
            self.mind_move_udp_port = int(self.mindmove_udp_port_line_edit.text())

            self.mindmove_udp_socket_group_box.setEnabled(False)
            self.virtual_hand_interface_group_box.setEnabled(False)

            self.virtual_hand_interface_configure_toggle_push_button.setText(
                "Change Configuration"
            )

            print("Virtual Hand Interface configured")

        else:
            self.mindmove_udp_socket_group_box.setEnabled(True)
            self.virtual_hand_interface_group_box.setEnabled(True)
            self.virtual_hand_interface_configure_toggle_push_button.setText(
                "Configure"
            )

    def _toggle_streaming(self) -> None:
        if self.virtual_hand_interface_toggle_streaming_push_button.isChecked():
            self.virtual_hand_interface_toggle_streaming_push_button.setText(
                "Stop Streaming"
            )
            self.virtual_hand_interface_configure_toggle_push_button.setEnabled(False)
            self.mind_move_udp_socket.bind(
                QHostAddress(self.mind_move_udp_ip), self.mind_move_udp_port
            )

            self.time_difference_between_messages = float(
                1
                / int(
                    self.mindmove_streaming_frequency_combo_box.currentText().split(
                        " "
                    )[0]
                )
            )
            self.is_streaming = True

        else:
            self.mind_move_udp_socket.close()
            self.virtual_hand_interface_toggle_streaming_push_button.setText(
                "Start Streaming"
            )
            self.virtual_hand_interface_configure_toggle_push_button.setEnabled(True)

            self.is_streaming = False

    def _check_and_validate_ip(self, ip_line_edit: QLineEdit, default: str):
        ip = ip_line_edit.text()
        if not self._check_for_valid_ip(ip):
            QMessageBox.warning(
                self, "Invalid IP", "The IP address you entered is not valid."
            )
            ip_line_edit.setText(default)

    def _check_and_validate_port(self, port_line_edit: QLineEdit, default: int):
        port = port_line_edit.text()
        if not self._check_for_correct_port(port):
            QMessageBox.warning(
                self, "Invalid Port", "The port you entered is not valid."
            )
            port_line_edit.setText(str(default))

    def _check_for_valid_ip(self, ip: str) -> bool:
        """
        Checks if the provided IP is valid.

        Args:
            ip (str): IP to be checked.

        Returns:
            bool: True if IP is valid. False if not.
        """
        ip_pattern = re.compile(
            r"^([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\."
            r"([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\."
            r"([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\."
            r"([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$"
        )

        return bool(ip_pattern.match(ip))

    def _check_for_correct_port(self, port: int) -> bool:
        """
        Checks if the provided port is valid.

        Args:
            port (str): Port to be checked.

        Returns:
            bool: True if port is valid. False if not.
        """
        try:
            port_num = int(port)
            return 0 <= port_num <= 65535
        except ValueError:
            return False

    def _setup_virtual_hand_interface(self):
        self.mind_move_udp_ip: str = None
        self.mind_move_udp_ip_default: str = self.mindmove_udp_ip_line_edit.text()
        self.mind_move_udp_port: int = None
        self.mind_move_udp_port_default: int = int(
            self.mindmove_udp_port_line_edit.text()
        )
        self.mindmove_udp_ip_line_edit.editingFinished.connect(
            lambda: self._check_and_validate_ip(
                self.mindmove_udp_ip_line_edit, self.mind_move_udp_ip_default
            )
        )
        self.mindmove_udp_port_line_edit.editingFinished.connect(
            lambda: self._check_and_validate_port(
                self.mindmove_udp_port_line_edit, self.mind_move_udp_port_default
            )
        )

        self.mind_move_udp_socket = QUdpSocket(self)
        self.mind_move_udp_socket.readyRead.connect(self._read_message)
        self.output_message_signal.connect(self._write_message)

        self.virtual_hand_interface_udp_ip: str = None
        self.virtual_hand_interface_udp_ip_default: str = (
            self.virtual_hand_interface_udp_ip_line_edit.text()
        )
        self.virtual_hand_interface_udp_port: int = None
        self.virtual_hand_interface_udp_port_default: int = int(
            self.virtual_hand_interface_udp_port_line_edit.text()
        )
        self.virtual_hand_interface_udp_ip_line_edit.editingFinished.connect(
            lambda: self._check_and_validate_ip(
                self.virtual_hand_interface_udp_ip_line_edit,
                self.virtual_hand_interface_udp_ip_default,
            )
        )
        self.virtual_hand_interface_udp_port_line_edit.editingFinished.connect(
            lambda: self._check_and_validate_port(
                self.virtual_hand_interface_udp_port_line_edit,
                self.virtual_hand_interface_udp_port_default,
            )
        )

        self.last_message_time = time.time()
        self.time_difference_between_messages: float = None

    def _setup_virtual_hand_interface_ui(self) -> None:
        self.mindmove_udp_socket_group_box = (
            self.main_window.ui.mindmoveUDPSocketGroupBox
        )
        self.mindmove_udp_ip_line_edit = self.main_window.ui.mindmoveUDPIPLineEdit
        self.mindmove_udp_port_line_edit = self.main_window.ui.mindmoveUDPPortLineEdit
        self.mindmove_streaming_frequency_combo_box = (
            self.main_window.ui.mindmoveStreamingFrequencyComboBox
        )

        self.virtual_hand_interface_group_box = (
            self.main_window.ui.virtualHandInterfaceGroupBox
        )
        self.virtual_hand_interface_udp_ip_line_edit = (
            self.main_window.ui.virtualHandInterfaceIPLineEdit
        )
        self.virtual_hand_interface_udp_port_line_edit = (
            self.main_window.ui.virtualHandInterfacePortLineEdit
        )

        self.virtual_hand_interface_configure_toggle_push_button = (
            self.main_window.ui.configureAppUdpSocketPushButton
        )
        self.virtual_hand_interface_configure_toggle_push_button.toggled.connect(
            self._configure_streaming
        )
        self.virtual_hand_interface_toggle_streaming_push_button = (
            self.main_window.ui.outputToggleStreamingPushButton
        )
        self.virtual_hand_interface_toggle_streaming_push_button.toggled.connect(
            self._toggle_streaming
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        return super().closeEvent(event)
