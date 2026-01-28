from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Union
from PySide6.QtWidgets import QWidget, QGroupBox, QGridLayout, QPushButton, QLabel
from PySide6.QtCore import Signal
from mindmove.device_interfaces.muovi import Muovi
from mindmove.device_interfaces.dicts.muovi import *
from mindmove.device_interfaces.enums.muovi import *
from mindmove.device_interfaces.enums.device import LoggerLevel
from mindmove.device_interfaces.gui.ui_compiled.muovi_widget import Ui_MuoviForm
from mindmove.model.core.filtering import RealTimeEMGFilter
from mindmove.config import config
import numpy as np

if TYPE_CHECKING:
    pass


class MuoviWidget(QWidget):
    ready_read_signal = Signal(np.ndarray)
    device_connected_signal = Signal(bool)
    device_configured_signal = Signal(bool)
    differential_mode_changed = Signal(bool)  # Emits True for differential, False for monopolar

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_MuoviForm()
        self.ui.setupUi(self)

        self.parent_object = parent

        # Device Setup
        self.device = Muovi(is_muovi_plus=False)
        self.device.data_available_signal.connect(self.update)
        self.device_params: dict = {}
        self._initialize_device_params()

        # Push Buttons
        self.connect_button = self.ui.commandConnectionPushButton
        self.connect_button.clicked.connect(self.toggle_connection)
        self.device.connected_signal.connect(self.toggle_connected)

        self.configure_button = self.ui.commandConfigurationPushButton
        self.configure_button.clicked.connect(self.configure_device)
        self.configure_button.setEnabled(False)
        self.device.configured_signal.connect(self.toggle_configured)

        self.stream_button = self.ui.commandStreamPushButton
        self.stream_button.clicked.connect(self.toggle_streaming)
        self.stream_button.setEnabled(False)

        # Connection parameters
        self.connection_group_box = self.ui.connectionGroupBox
        self.connection_ip_address_label = self.ui.connectionIPAddressLabel
        self.connection_port_label = self.ui.connectionPortLabel
        self.connection_update_push_button = self.ui.connectionUpdatePushButton
        self.connection_update_push_button.clicked.connect(
            lambda: self.connection_ip_address_label.setText(
                self.device.get_server_wifi_ip_address()
            )
        )
        # Network parameters
        self.connection_ip_address_label.setText(
            self.device.get_server_wifi_ip_address()
        )
        self.connection_port_label.setText(
            str(
                MUOVI_NETWORK_CHARACTERISTICS_DICT[
                    MuoviNetworkCharacteristics.EXTERNAL_NETWORK
                ]["port"]
            )
        )

        # Input parameters
        self.input_parameters_group_box = self.ui.inputGroupBox
        self.input_working_mode_combo_box = self.ui.inputWorkingModeComboBox
        self.input_detection_mode_combo_box = self.ui.inputDetectionModeComboBox

        # Configuration parameters
        self.configuration_group_boxes: list[QGroupBox] = [
            self.input_parameters_group_box,
        ]

        # Signal Processing - Real-time filter
        self.rt_filter = RealTimeEMGFilter(n_channels=config.num_channels)
        self.filtering_enabled = config.ENABLE_FILTERING
        self.differential_mode_enabled = config.ENABLE_DIFFERENTIAL_MODE
        self._current_raw_data = None  # Cache for current packet
        self._current_filtered_emg = None  # Cache for filtered result
        self._setup_signal_processing_group()

    def toggle_connection(self):
        if not self.device.is_connected:
            self.connect_button.setEnabled(False)

        self.device.toggle_connection(
            (
                self.connection_ip_address_label.text(),
                int(self.connection_port_label.text()),
            )
        )

    def toggle_connected(self, is_connected: bool) -> None:
        self.connect_button.setEnabled(True)
        if is_connected:
            self.connect_button.setText("Disconnect")
            self.connect_button.setChecked(True)
            self.configure_button.setEnabled(True)
            self.device.log_info("Connected")
            self.connection_group_box.setEnabled(False)
        else:
            self.connect_button.setText("Connect")
            self.connect_button.setChecked(False)
            self.configure_button.setEnabled(False)
            self.stream_button.setEnabled(False)
            self.device.log_info("Disconnected")
            self.connection_group_box.setEnabled(True)

        self.device_connected_signal.emit(is_connected)

    def configure_device(self) -> None:
        self.device_params["working_mode"] = MuoviWorkingMode(
            self.input_working_mode_combo_box.currentIndex()
        )
        self.device_params["detection_mode"] = MuoviDetectionMode(
            self.input_detection_mode_combo_box.currentIndex()
        )

        self.device.configure_device(self.device_params)

    def toggle_configured(self, is_configured: bool) -> None:
        if is_configured:
            self.stream_button.setEnabled(True)
            self.device.log_info("Configured")
        else:
            self.device.reset_configuration()

        self.device_configured_signal.emit(is_configured)

    def _toggle_configuration_group_boxes(self) -> None:
        for group_box in self.configuration_group_boxes:
            group_box.setEnabled(not group_box.isEnabled())

    def toggle_streaming(self) -> None:
        self.device.toggle_streaming()
        if self.device.is_streaming:
            # Reset filter state for new stream to avoid transients
            self.rt_filter.reset()
            self.stream_button.setText("Stop Streaming")
            self.stream_button.setChecked(True)
            self.configure_button.setEnabled(False)
            self.device.log_info("Streaming")
            self._toggle_configuration_group_boxes()
        else:
            self.stream_button.setText("Stream")
            self.stream_button.setChecked(False)
            self.configure_button.setEnabled(True)
            self._toggle_configuration_group_boxes()
            self.device.log_info("Stopped Streaming")

    def update(self, data: np.ndarray) -> None:
        # Store raw data for extract_emg_data to process
        # Filtering happens in extract_emg_data, but we cache the result
        # to avoid filtering the same packet multiple times
        self._current_raw_data = data
        self._current_filtered_emg = None  # Reset cache for new packet
        self.ready_read_signal.emit(data)

    def extract_emg_data(
        self, data: np.ndarray, milli_volts: bool = False
    ) -> np.ndarray:
        """
        Extracts the EMG Signals from the transmitted data.

        When filtering is enabled, applies real-time bandpass and notch
        filtering to the extracted EMG data.

        IMPORTANT: Uses caching to ensure the filter is only called ONCE
        per packet, even if multiple consumers call this method. This is
        critical because the IIR filter has internal state that must not
        be advanced multiple times for the same data.

        Args:
            data (np.ndarray):
                Raw data that got transmitted.

            milli_volts (bool, optional):
                If True, the EMG data is converted to milli volts.
                Defaults to False.

        Returns:
            np.ndarray:
                Extracted (and optionally filtered) EMG channels.
        """
        # Check if this is the same packet we already processed
        # (multiple consumers may call extract_emg_data for the same packet)
        is_same_packet = (
            hasattr(self, '_current_raw_data') and
            self._current_raw_data is data
        )

        if is_same_packet and hasattr(self, '_current_filtered_emg') and self._current_filtered_emg is not None:
            # Return cached result to avoid filtering the same packet twice
            return self._current_filtered_emg

        # Extract EMG from raw data
        emg_data = self.device.extract_emg_data(data, milli_volts)

        # Apply real-time filtering if enabled
        if self.filtering_enabled and emg_data.size > 0:
            emg_data = self.rt_filter.filter(emg_data)

        # Apply differential transform if enabled
        if self.differential_mode_enabled and emg_data.size > 0:
            emg_data = self.apply_differential_transform(emg_data)

        # Cache the result for this packet
        if is_same_packet:
            self._current_filtered_emg = emg_data

        return emg_data

    def apply_differential_transform(self, emg_data: np.ndarray) -> np.ndarray:
        """
        Compute single differential from 32 monopolar channels.

        The Muovi bracelet has 2 rows of 16 electrodes:
        - Row 1: channels 0-15 (1-16 in 1-indexed)
        - Row 2: channels 16-31 (17-32 in 1-indexed)

        Single differential is computed as the longitudinal difference:
        diff[i] = emg[i + 16] - emg[i] for i in 0..15

        Args:
            emg_data: (32, N) monopolar EMG data

        Returns:
            (16, N) single differential EMG data
        """
        if emg_data.shape[0] != 32:
            # If already 16 channels or different shape, return as-is
            return emg_data

        # Row 1: channels 0-15, Row 2: channels 16-31
        row1 = emg_data[:16, :]   # Channels 1-16
        row2 = emg_data[16:, :]   # Channels 17-32
        return row2 - row1  # Longitudinal difference

    def extract_aux_data(self, data: np.ndarray, index: int = 0) -> np.ndarray:
        """
        Extract a defined AUX channel from the transmitted data.

        Args:
            data (np.ndarray):
                Raw data that got transmitted.
            index (int, optional): Index of the AUX channel to be extracted.
                Defaults to 0.

        Returns:
            np.ndarray:
                Extracted AUX channel data.
        """
        return self.device.extract_aux_data(data, index)

    def get_device_information(self) -> Dict[str, Enum | int | float | str]:
        """
        Gets the current configuration of the device.

        Returns:
            Dict[str, Enum | int | float | str]:
                Dictionary that holds information about the
                current device configuration and status.
        """

        return self.device.get_device_information()

    def force_disconnect(self) -> None:
        self.device.force_disconnect()

    def _initialize_device_params(self) -> None:
        self.device_params = {
            "working_mode": MuoviWorkingMode.EMG,
            "detection_mode": MuoviDetectionMode.MONOPOLAR_GAIN_8,
            "streaming_mode": MuoviStream.STOP,
        }

    def _setup_signal_processing_group(self) -> None:
        """Create Signal Processing group box with filter and differential mode toggles."""
        # Create group box
        self.signal_processing_group_box = QGroupBox("Signal Processing")
        self.signal_processing_group_box.setObjectName("signalProcessingGroupBox")

        # Create layout for the group box
        layout = QGridLayout(self.signal_processing_group_box)

        # Create filter toggle button
        self.filter_toggle_button = QPushButton("Filter: OFF")
        self.filter_toggle_button.setCheckable(True)
        self.filter_toggle_button.setChecked(self.filtering_enabled)
        self.filter_toggle_button.toggled.connect(self._on_filter_toggled)

        # Create filter description label
        self.filter_description_label = QLabel(self.rt_filter.get_filter_description())

        # Add filter widgets to layout (row 0)
        layout.addWidget(self.filter_toggle_button, 0, 0)
        layout.addWidget(self.filter_description_label, 0, 1)

        # Create differential mode toggle button
        self.differential_mode_button = QPushButton("Mode: Monopolar")
        self.differential_mode_button.setCheckable(True)
        self.differential_mode_button.setChecked(self.differential_mode_enabled)
        self.differential_mode_button.toggled.connect(self._on_differential_mode_toggled)

        # Create differential mode description label
        self.differential_mode_label = QLabel("32 channels")

        # Add differential mode widgets to layout (row 1)
        layout.addWidget(self.differential_mode_button, 1, 0)
        layout.addWidget(self.differential_mode_label, 1, 1)

        # Update button appearances based on initial state
        self._update_filter_button_style()
        self._update_differential_button_style()

        # Add group box to main layout (after Commands group box, row 4)
        self.ui.gridLayout.addWidget(self.signal_processing_group_box, 4, 0, 1, 2)

        # Move spacer to row 5
        self.ui.gridLayout.removeItem(self.ui.verticalSpacer)
        self.ui.gridLayout.addItem(self.ui.verticalSpacer, 5, 0, 1, 1)

    def _on_filter_toggled(self, checked: bool) -> None:
        """Handle filter toggle button state change."""
        self.filtering_enabled = checked
        config.ENABLE_FILTERING = checked  # Update global config for consistency
        self._update_filter_button_style()

        if checked:
            self.rt_filter.reset()  # Fresh start when enabling
            self.device.log_info("Filter enabled: " + self.rt_filter.get_filter_description())
        else:
            self.device.log_info("Filter disabled: Raw EMG signal")

    def _update_filter_button_style(self) -> None:
        """Update filter button appearance based on state."""
        if self.filtering_enabled:
            self.filter_toggle_button.setText("Filter: ON")
            self.filter_toggle_button.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
                "QPushButton:checked { background-color: #4CAF50; }"
            )
        else:
            self.filter_toggle_button.setText("Filter: OFF")
            self.filter_toggle_button.setStyleSheet("")

    def _on_differential_mode_toggled(self, checked: bool) -> None:
        """Handle differential mode toggle button state change."""
        self.differential_mode_enabled = checked
        config.ENABLE_DIFFERENTIAL_MODE = checked  # Update global config for consistency
        self._update_differential_button_style()

        # Reset filter state when changing modes to avoid transients
        self.rt_filter.reset()
        # Clear cached data
        self._current_filtered_emg = None

        if checked:
            self.device.log_info("Mode: Single Differential (16 channels)")
        else:
            self.device.log_info("Mode: Monopolar (32 channels)")

        # Notify listeners (e.g., plot widget) that mode changed
        self.differential_mode_changed.emit(checked)

    def _update_differential_button_style(self) -> None:
        """Update differential mode button appearance based on state."""
        if self.differential_mode_enabled:
            self.differential_mode_button.setText("Mode: Differential")
            self.differential_mode_label.setText("16 channels (row2 - row1)")
            self.differential_mode_button.setStyleSheet(
                "QPushButton { background-color: #2196F3; color: white; font-weight: bold; }"
                "QPushButton:checked { background-color: #2196F3; }"
            )
        else:
            self.differential_mode_button.setText("Mode: Monopolar")
            self.differential_mode_label.setText("32 channels")
            self.differential_mode_button.setStyleSheet("")

    def get_current_channel_count(self) -> int:
        """Return the current number of output channels based on mode."""
        return 16 if self.differential_mode_enabled else 32

    def get_mode_suffix(self) -> str:
        """Return the mode suffix for filenames (_mp_ or _sd_)."""
        return "_sd_" if self.differential_mode_enabled else "_mp_"
