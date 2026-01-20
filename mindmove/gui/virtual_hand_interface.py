"""
Virtual Hand Interface for MindMove

This module handles UDP communication with the Unity Virtual Hand Interface application.
It can launch the Unity executable automatically or connect to an externally running instance.

Communication Protocol:
- MindMove sends predictions to port 1236 (VHI listens here)
- VHI sends predicted hand data back on port 1234
- Status handshake: "status" -> "active"
"""

from __future__ import annotations
import ast
import platform
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QByteArray, QObject, QProcess, QTimer, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtNetwork import QHostAddress, QUdpSocket
from PySide6.QtWidgets import QLineEdit, QMessageBox, QWidget, QSizePolicy

if TYPE_CHECKING:
    from mindmove.gui.mindmove import MindMove


# Stylesheets for connection status indicator
NOT_CONNECTED_STYLESHEET = "background-color: red; border-radius: 5px;"
CONNECTED_STYLESHEET = "background-color: green; border-radius: 5px;"

# Constants
STREAMING_FREQUENCY = 32  # Hz
TIME_BETWEEN_MESSAGES = 1 / STREAMING_FREQUENCY

SOCKET_IP = "127.0.0.1"
STATUS_REQUEST = "status"
STATUS_RESPONSE = "active"

# UDP Ports
MINDMOVE_UDP_PORT = 1233  # MindMove listens here for status responses
VHI_UDP_PORT = 1236  # VHI listens for predictions from MindMove
VHI_PREDICTION_UDP_PORT = 1234  # VHI sends predicted hand data here


class VirtualHandInterface(QObject):
    """
    Virtual Hand Interface for MindMove.

    Handles:
    - Launching the Unity Virtual Hand Interface application
    - UDP communication for sending predictions and receiving hand data
    - Status checking and connection management
    """

    output_message_signal = Signal(QByteArray)
    input_message_signal = Signal(np.ndarray)
    predicted_hand_signal = Signal(np.ndarray)

    def __init__(self, parent: MindMove | None = None) -> None:
        super().__init__(parent)

        self.main_window = parent
        self.is_streaming: bool = False
        self._is_connected: bool = False

        # Unity process
        self._unity_process = QProcess(self)
        self._unity_process.setProgram(str(self._get_unity_executable()))
        self._unity_process.finished.connect(self._on_unity_process_finished)

        # UDP Sockets
        self._streaming_udp_socket: QUdpSocket | None = None
        self._predicted_hand_udp_socket: QUdpSocket | None = None

        # Predicted hand recording buffer (for online protocol)
        self._predicted_hand_recording_buffer: list[tuple[float, np.ndarray]] = []

        # Initialize timers
        self._setup_timers()

        # Initialize UI
        self._setup_virtual_hand_interface_ui()

        # Initialize UDP settings
        self._setup_virtual_hand_interface()

        # Timing
        self._last_message_time = time.time()
        self.time_difference_between_messages: float = TIME_BETWEEN_MESSAGES

    @staticmethod
    def _get_unity_executable() -> Path:
        """Get the path to the Unity executable based on the platform."""
        # Search paths for the Unity executable
        base_dirs = [
            Path("virtual-hand-interface-main"),
            Path("dist"),
        ]

        # Add PyInstaller paths if frozen
        if hasattr(sys, "_MEIPASS"):
            base_dirs.insert(0, Path(sys._MEIPASS, "dist"))

        unity_executable_paths = {
            "Windows": "windows/Virtual Hand Interface.exe",
            "Darwin": "macOS/Virtual Hand Interface.app/Contents/MacOS/Virtual Hand Interface",
            "Linux": "linux/VirtualHandInterface.x86_64",
        }

        for base_dir in base_dirs:
            executable = base_dir / unity_executable_paths.get(platform.system(), "")
            if executable.exists():
                return executable

        # Return expected path even if not found (will fail gracefully later)
        return Path("virtual-hand-interface-main") / unity_executable_paths.get(platform.system(), "")

    def _setup_timers(self) -> None:
        """Setup timers for status checking."""
        # Timer to send status requests every 2 seconds
        self._status_request_timer = QTimer(self)
        self._status_request_timer.setInterval(2000)
        self._status_request_timer.timeout.connect(self._write_status_message)

        # Timeout timer for status response (1 second)
        self._status_timeout_timer = QTimer(self)
        self._status_timeout_timer.setSingleShot(True)
        self._status_timeout_timer.setInterval(1000)
        self._status_timeout_timer.timeout.connect(self._on_status_timeout)

    def _on_status_timeout(self) -> None:
        """Handle status response timeout - mark as disconnected."""
        self._is_connected = False
        self._update_status_indicator()

    def _update_status_indicator(self) -> None:
        """Update the visual status indicator."""
        if hasattr(self, '_status_widget') and self._status_widget:
            stylesheet = CONNECTED_STYLESHEET if self._is_connected else NOT_CONNECTED_STYLESHEET
            self._status_widget.setStyleSheet(stylesheet)

    def _on_unity_process_finished(self) -> None:
        """Handle Unity process termination."""
        self._is_connected = False
        self._update_status_indicator()

        # Reset UI state
        self.virtual_hand_interface_toggle_streaming_push_button.setChecked(False)
        self.virtual_hand_interface_toggle_streaming_push_button.setText("Start Streaming")
        self.virtual_hand_interface_configure_toggle_push_button.setEnabled(True)

        print("Virtual Hand Interface process ended")

    def start_unity_interface(self) -> bool:
        """Start the Unity Virtual Hand Interface application."""
        executable_path = self._get_unity_executable()

        if not executable_path.exists():
            QMessageBox.warning(
                self.main_window,
                "Unity Executable Not Found",
                f"Could not find Unity executable at:\n{executable_path}\n\n"
                "Please ensure the Virtual Hand Interface is installed correctly."
            )
            return False

        self._unity_process.setProgram(str(executable_path))
        self._unity_process.start()

        if not self._unity_process.waitForStarted(5000):
            QMessageBox.warning(
                self.main_window,
                "Failed to Start",
                "Failed to start the Virtual Hand Interface application."
            )
            return False

        print(f"Started Virtual Hand Interface: {executable_path}")
        return True

    def stop_unity_interface(self) -> None:
        """Stop the Unity Virtual Hand Interface application."""
        if self._unity_process.state() != QProcess.NotRunning:
            self._unity_process.kill()
            self._unity_process.waitForFinished(3000)

    def _read_message(self) -> None:
        """Read incoming UDP messages (status responses and data)."""
        while self._streaming_udp_socket and self._streaming_udp_socket.hasPendingDatagrams():
            datagram, host, port = self._streaming_udp_socket.readDatagram(
                self._streaming_udp_socket.pendingDatagramSize()
            )

            raw_data = datagram.data()

            try:
                data = raw_data.decode("utf-8")
                if not data:
                    continue

                # Check for status response
                if data == STATUS_RESPONSE:
                    self._is_connected = True
                    self._update_status_indicator()
                    self._status_timeout_timer.stop()
                    print(f"VHI connected! (status response received)")
                    continue

                # Parse as array data
                print(f"Received text data: {data[:100]}...")  # Debug
                self.input_message_signal.emit(np.array(ast.literal_eval(data)))
            except UnicodeDecodeError:
                # Binary data - print hex for debugging
                print(f"Received binary data ({len(raw_data)} bytes): {raw_data[:20].hex()}...")
            except (ValueError, SyntaxError) as e:
                print(f"Parse error: {e}")

    def _read_predicted_hand(self) -> None:
        """Read predicted hand data from VHI (kinematics/ground truth at 60Hz)."""
        while self._predicted_hand_udp_socket and self._predicted_hand_udp_socket.hasPendingDatagrams():
            datagram, _, _ = self._predicted_hand_udp_socket.readDatagram(
                self._predicted_hand_udp_socket.pendingDatagramSize()
            )

            raw_data = datagram.data()

            try:
                data = raw_data.decode("utf-8")
                if not data:
                    continue

                hand_data = np.array(ast.literal_eval(data))
                # Only print occasionally to avoid spam
                if not hasattr(self, '_hand_msg_count'):
                    self._hand_msg_count = 0
                self._hand_msg_count += 1
                if self._hand_msg_count % 60 == 1:  # Print every ~1 second
                    print(f"Kinematics data (port 1234): {hand_data}")
                self.predicted_hand_signal.emit(hand_data)
            except UnicodeDecodeError:
                print(f"Binary kinematics ({len(raw_data)} bytes): {raw_data[:20].hex()}...")
            except (ValueError, SyntaxError) as e:
                print(f"Kinematics parse error: {e}")

    def _write_message(self, message: QByteArray) -> None:
        """Send a prediction message to the Virtual Hand Interface."""
        if not self._is_connected:
            return

        if self.virtual_hand_interface_toggle_streaming_push_button.isChecked():
            # Rate limiting
            if time.time() - self._last_message_time < self.time_difference_between_messages:
                return

            self._last_message_time = time.time()

            output_bytes = self._streaming_udp_socket.writeDatagram(
                message,
                QHostAddress(self.virtual_hand_interface_udp_ip),
                self.virtual_hand_interface_udp_port,
            )

            if output_bytes == -1:
                print("Error sending message to Virtual Hand Interface")

    def _write_status_message(self) -> None:
        """Send a status check message to VHI."""
        if not self._streaming_udp_socket:
            return

        output_bytes = self._streaming_udp_socket.writeDatagram(
            STATUS_REQUEST.encode("utf-8"),
            QHostAddress(self.virtual_hand_interface_udp_ip),
            self.virtual_hand_interface_udp_port,
        )

        if output_bytes == -1:
            print("Error sending status message")
            return

        # Start timeout timer
        self._status_timeout_timer.start()

    def _configure_streaming(self) -> None:
        """Configure UDP streaming settings."""
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
        """Toggle UDP streaming on/off."""
        if self.virtual_hand_interface_toggle_streaming_push_button.isChecked():
            # Start streaming
            self.virtual_hand_interface_toggle_streaming_push_button.setText(
                "Stop Streaming"
            )
            self.virtual_hand_interface_configure_toggle_push_button.setEnabled(False)

            # Create and bind main UDP socket
            self._streaming_udp_socket = QUdpSocket(self)
            self._streaming_udp_socket.readyRead.connect(self._read_message)
            self._streaming_udp_socket.bind(
                QHostAddress(self.mind_move_udp_ip), self.mind_move_udp_port
            )

            # Create and bind predicted hand UDP socket
            self._predicted_hand_udp_socket = QUdpSocket(self)
            self._predicted_hand_udp_socket.readyRead.connect(self._read_predicted_hand)
            self._predicted_hand_udp_socket.bind(
                QHostAddress(SOCKET_IP), VHI_PREDICTION_UDP_PORT
            )

            # Calculate time between messages based on frequency setting
            freq_text = self.mindmove_streaming_frequency_combo_box.currentText()
            try:
                freq = int(freq_text.split(" ")[0])
                self.time_difference_between_messages = 1.0 / freq
            except (ValueError, IndexError):
                self.time_difference_between_messages = TIME_BETWEEN_MESSAGES

            self._last_message_time = time.time()
            self.is_streaming = True

            # Start status checking
            self._status_request_timer.start()

            # Try to start Unity if not already running
            if self._unity_process.state() == QProcess.NotRunning:
                self.start_unity_interface()

        else:
            # Stop streaming
            self._status_request_timer.stop()
            self._status_timeout_timer.stop()

            if self._streaming_udp_socket:
                self._streaming_udp_socket.close()
                self._streaming_udp_socket = None

            if self._predicted_hand_udp_socket:
                self._predicted_hand_udp_socket.close()
                self._predicted_hand_udp_socket = None

            self.virtual_hand_interface_toggle_streaming_push_button.setText(
                "Start Streaming"
            )
            self.virtual_hand_interface_configure_toggle_push_button.setEnabled(True)

            self.is_streaming = False
            self._is_connected = False
            self._update_status_indicator()

    def _check_and_validate_ip(self, ip_line_edit: QLineEdit, default: str) -> None:
        """Validate IP address input."""
        ip = ip_line_edit.text()
        if not self._check_for_valid_ip(ip):
            QMessageBox.warning(
                self.main_window, "Invalid IP", "The IP address you entered is not valid."
            )
            ip_line_edit.setText(default)

    def _check_and_validate_port(self, port_line_edit: QLineEdit, default: int) -> None:
        """Validate port input."""
        port = port_line_edit.text()
        if not self._check_for_correct_port(port):
            QMessageBox.warning(
                self.main_window, "Invalid Port", "The port you entered is not valid."
            )
            port_line_edit.setText(str(default))

    def _check_for_valid_ip(self, ip: str) -> bool:
        """Check if the provided IP is valid."""
        ip_pattern = re.compile(
            r"^([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\."
            r"([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\."
            r"([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\."
            r"([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$"
        )
        return bool(ip_pattern.match(ip))

    def _check_for_correct_port(self, port: str) -> bool:
        """Check if the provided port is valid."""
        try:
            port_num = int(port)
            return 0 <= port_num <= 65535
        except ValueError:
            return False

    def _setup_virtual_hand_interface(self) -> None:
        """Initialize UDP socket settings."""
        # MindMove UDP settings
        self.mind_move_udp_ip: str = self.mindmove_udp_ip_line_edit.text()
        self.mind_move_udp_ip_default: str = self.mindmove_udp_ip_line_edit.text()
        self.mind_move_udp_port: int = int(self.mindmove_udp_port_line_edit.text())
        self.mind_move_udp_port_default: int = int(self.mindmove_udp_port_line_edit.text())

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

        # Connect output signal
        self.output_message_signal.connect(self._write_message)

        # Virtual Hand Interface UDP settings
        self.virtual_hand_interface_udp_ip: str = self.virtual_hand_interface_udp_ip_line_edit.text()
        self.virtual_hand_interface_udp_ip_default: str = self.virtual_hand_interface_udp_ip_line_edit.text()
        self.virtual_hand_interface_udp_port: int = int(self.virtual_hand_interface_udp_port_line_edit.text())
        self.virtual_hand_interface_udp_port_default: int = int(self.virtual_hand_interface_udp_port_line_edit.text())

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

    def _setup_virtual_hand_interface_ui(self) -> None:
        """Setup UI element references."""
        # MindMove UDP group box
        self.mindmove_udp_socket_group_box = (
            self.main_window.ui.mindmoveUDPSocketGroupBox
        )
        self.mindmove_udp_ip_line_edit = self.main_window.ui.mindmoveUDPIPLineEdit
        self.mindmove_udp_port_line_edit = self.main_window.ui.mindmoveUDPPortLineEdit
        self.mindmove_streaming_frequency_combo_box = (
            self.main_window.ui.mindmoveStreamingFrequencyComboBox
        )

        # Virtual Hand Interface group box
        self.virtual_hand_interface_group_box = (
            self.main_window.ui.virtualHandInterfaceGroupBox
        )
        self.virtual_hand_interface_udp_ip_line_edit = (
            self.main_window.ui.virtualHandInterfaceIPLineEdit
        )
        self.virtual_hand_interface_udp_port_line_edit = (
            self.main_window.ui.virtualHandInterfacePortLineEdit
        )

        # Control buttons
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

        # Create status indicator widget
        self._status_widget = getattr(self.main_window.ui, 'virtualHandInterfaceStatusWidget', None)
        if not self._status_widget:
            # Create status widget programmatically if not in UI
            self._status_widget = QWidget()
            self._status_widget.setObjectName("virtualHandInterfaceStatusWidget")
            size_policy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self._status_widget.setSizePolicy(size_policy)
            self._status_widget.setMinimumSize(10, 10)
            self._status_widget.setMaximumSize(10, 10)
            # Add to the control buttons group box layout
            control_group_box = self.main_window.ui.groupBox_2
            if control_group_box and control_group_box.layout():
                control_group_box.layout().addWidget(self._status_widget, 0, 2)
        self._status_widget.setStyleSheet(NOT_CONNECTED_STYLESHEET)

    def get_custom_save_data(self) -> dict:
        """Get recorded predicted hand data for saving."""
        if not self._predicted_hand_recording_buffer:
            return {}

        return {
            "predicted_hand": np.vstack(
                [data for _, data in self._predicted_hand_recording_buffer],
            ).T,
            "predicted_hand_timings": np.array(
                [t for t, _ in self._predicted_hand_recording_buffer],
            ),
        }

    def clear_recording_buffer(self) -> None:
        """Clear the predicted hand recording buffer."""
        self._predicted_hand_recording_buffer = []

    def record_predicted_hand(self, data: np.ndarray, start_time: float) -> None:
        """Record predicted hand data with timestamp."""
        self._predicted_hand_recording_buffer.append(
            (time.time() - start_time, data)
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle close event - cleanup resources."""
        # Stop timers
        self._status_request_timer.stop()
        self._status_timeout_timer.stop()

        # Close sockets
        if self._streaming_udp_socket:
            self._streaming_udp_socket.close()
        if self._predicted_hand_udp_socket:
            self._predicted_hand_udp_socket.close()

        # Stop Unity process
        self.stop_unity_interface()
