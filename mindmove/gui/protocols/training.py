from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from PySide6.QtCore import QObject, Signal, QThread
from PySide6.QtWidgets import (
    QFileDialog, QMessageBox, QListWidgetItem, QLabel, QLineEdit,
    QComboBox, QPushButton, QSpinBox, QDoubleSpinBox, QGroupBox
)
import pickle
from datetime import datetime
import numpy as np
import os

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
        # Row 0: Select Open Templates
        self.select_open_templates_btn = QPushButton("Select Open Templates")
        self.select_open_templates_btn.clicked.connect(lambda: self._select_template_file("open"))
        self.open_templates_label = QLabel("No templates selected")
        layout.addWidget(self.select_open_templates_btn, 0, 0, 1, 1)
        layout.addWidget(self.open_templates_label, 0, 1, 1, 2)

        # Row 1: Select Closed Templates
        self.select_closed_templates_btn = QPushButton("Select Closed Templates")
        self.select_closed_templates_btn.clicked.connect(lambda: self._select_template_file("closed"))
        self.closed_templates_label = QLabel("No templates selected")
        layout.addWidget(self.select_closed_templates_btn, 1, 0, 1, 1)
        layout.addWidget(self.closed_templates_label, 1, 1, 1, 2)

        # Row 2: Window/Overlap presets
        self.window_preset_label = QLabel("Window/Overlap:")
        self.window_preset_combo = QComboBox()
        self.window_preset_combo.addItems([
            "96/32 ms (Default)",
            "150/50 ms (Eddy's)",
            "200/100 ms",
            "Custom"
        ])
        self.window_preset_combo.currentIndexChanged.connect(self._on_window_preset_changed)
        layout.addWidget(self.window_preset_label, 2, 0, 1, 1)
        layout.addWidget(self.window_preset_combo, 2, 1, 1, 2)

        # Row 3: Custom window/overlap inputs (hidden by default)
        self.custom_window_label = QLabel("Window (ms):")
        self.custom_window_spinbox = QSpinBox()
        self.custom_window_spinbox.setRange(50, 500)
        self.custom_window_spinbox.setValue(96)
        self.custom_overlap_label = QLabel("Overlap (ms):")
        self.custom_overlap_spinbox = QSpinBox()
        self.custom_overlap_spinbox.setRange(10, 200)
        self.custom_overlap_spinbox.setValue(32)
        layout.addWidget(self.custom_window_label, 3, 0, 1, 1)
        layout.addWidget(self.custom_window_spinbox, 3, 1, 1, 1)
        layout.addWidget(self.custom_overlap_label, 3, 2, 1, 1)
        layout.addWidget(self.custom_overlap_spinbox, 3, 3, 1, 1)
        # Hide custom inputs by default
        self.custom_window_label.setVisible(False)
        self.custom_window_spinbox.setVisible(False)
        self.custom_overlap_label.setVisible(False)
        self.custom_overlap_spinbox.setVisible(False)

        # Row 4: Feature selection
        self.feature_label = QLabel("Feature:")
        self.feature_combo = QComboBox()
        # Add all features from registry
        from mindmove.model.core.features.features_registry import FEATURES
        self.feature_combo.addItems(list(FEATURES.keys()))
        self.feature_combo.setCurrentText("wl")  # Default to waveform length
        layout.addWidget(self.feature_label, 4, 0, 1, 1)
        layout.addWidget(self.feature_combo, 4, 1, 1, 2)

        # Row 5: DTW algorithm selection
        self.dtw_algorithm_label = QLabel("DTW Algorithm:")
        self.dtw_algorithm_combo = QComboBox()
        self.dtw_algorithm_combo.addItems([
            "Numba (Cosine) - Recommended",
            "tslearn (Euclidean)",
            "dtaidistance (Euclidean)",
            "Pure Python (Cosine)"
        ])
        layout.addWidget(self.dtw_algorithm_label, 5, 0, 1, 1)
        layout.addWidget(self.dtw_algorithm_combo, 5, 1, 1, 2)

        # Row 6: Dead channels input (1-indexed for user)
        self.dead_channels_label = QLabel("Dead Channels:")
        self.dead_channels_input = QLineEdit()
        self.dead_channels_input.setPlaceholderText("e.g., 9, 22, 25 (1-indexed)")
        self.dead_channels_input.setToolTip("Enter channel numbers (1-32) separated by commas. These channels will be excluded from DTW computation.")
        layout.addWidget(self.dead_channels_label, 6, 0, 1, 1)
        layout.addWidget(self.dead_channels_input, 6, 1, 1, 2)

        # Row 7: Distance aggregation method
        self.distance_agg_label = QLabel("Distance Aggregation:")
        self.distance_agg_combo = QComboBox()
        self.distance_agg_combo.addItems([
            "Average of 3 smallest (Recommended)",
            "Minimum distance",
            "Average of all"
        ])
        self.distance_agg_combo.setToolTip("How to compute final distance from multiple templates")
        layout.addWidget(self.distance_agg_label, 7, 0, 1, 1)
        layout.addWidget(self.distance_agg_combo, 7, 1, 1, 2)

        # Row 8: Post-prediction smoothing
        self.smoothing_label = QLabel("State Smoothing:")
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItems([
            "Majority Vote (5 samples)",
            "5 Consecutive",
            "None"
        ])
        self.smoothing_combo.setToolTip("Method to smooth state transitions")
        layout.addWidget(self.smoothing_label, 8, 0, 1, 1)
        layout.addWidget(self.smoothing_combo, 8, 1, 1, 2)

        # Row 9: Model name (reuse existing widget, just reposition)
        layout.addWidget(self.main_window.ui.label_8, 9, 0, 1, 1)
        layout.addWidget(self.training_model_label_line_edit, 9, 1, 1, 2)

        # Row 10: Create Model button
        self.create_model_btn = QPushButton("Create Model")
        self.create_model_btn.clicked.connect(self._create_dtw_model)
        self.create_model_btn.setEnabled(False)
        layout.addWidget(self.create_model_btn, 10, 0, 1, 3)

        # Row 11: Progress bar
        self.model_creation_progress_bar = self.main_window.ui.trainingProgressBar
        self.model_creation_progress_bar.setVisible(True)
        self.model_creation_progress_bar.setValue(0)
        layout.addWidget(self.model_creation_progress_bar, 11, 0, 1, 3)

        # Store selected template paths
        self.selected_open_templates_path: Optional[str] = None
        self.selected_closed_templates_path: Optional[str] = None

    def _setup_template_extraction_ui(self) -> None:
        """Setup UI connections for template extraction group box."""
        # Template Extraction Group Box
        self.template_extraction_group_box = (
            self.main_window.ui.trainingTemplateExtractionGroupBox
        )

        # Class selection combo
        self.template_class_combo = self.main_window.ui.trainingTemplateClassComboBox

        # Data format combo
        self.data_format_combo = self.main_window.ui.trainingDataFormatComboBox
        self.data_format_combo.currentIndexChanged.connect(self._on_data_format_changed)

        # Template type combo - add new options programmatically
        self.template_type_combo = self.main_window.ui.trainingTemplateTypeComboBox
        # Add new options (UI file has "Hold Only" at 0, "Onset + Hold" at 1)
        self.template_type_combo.addItem("Onset (GT=1 start)")  # index 2
        self.template_type_combo.addItem("Manual Selection")     # index 3
        self.template_type_combo.currentIndexChanged.connect(self._on_template_type_changed)

        # Track if we're in manual selection mode
        self._manual_selection_mode: bool = False
        self._manual_templates: List[np.ndarray] = []

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
        # We'll insert it at row 6.5 visually by using row 11 and letting Qt handle it
        grid_layout = self.template_extraction_group_box.layout()
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

        # Clear button
        self.clear_extraction_btn = self.main_window.ui.trainingClearExtractionPushButton
        self.clear_extraction_btn.clicked.connect(self._clear_extraction)

        # Progress bar
        self.extraction_progress_bar = self.main_window.ui.trainingExtractionProgressBar
        self.extraction_progress_bar.setValue(0)

    def _on_data_format_changed(self, index: int) -> None:
        """Handle data format combo box change."""
        is_legacy = index == 1  # "Legacy (EMG + GT folders)" is at index 1
        if is_legacy:
            self.select_recordings_for_extraction_btn.setText("Select EMG Folder")
        else:
            self.select_recordings_for_extraction_btn.setText("Select Recording(s)")

        # Clear previous selections
        self._clear_extraction()

    def _on_template_type_changed(self, index: int) -> None:
        """Handle template type combo box change."""
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
        is_legacy = self.data_format_combo.currentIndex() == 1

        if is_legacy:
            self._select_legacy_folders()
        else:
            self._select_recording_files()

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

    def _clear_extraction(self) -> None:
        """Clear all extracted activations and templates."""
        class_label = self.template_class_combo.currentText().lower()

        # Clear template manager data
        self.template_manager.clear_all(class_label)

        # Clear UI
        self.selected_extraction_recordings = []
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

    # ==================== DTW Model Creation Methods ====================

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
        """Enable Create Model button only when both template sets are selected."""
        has_open = self.selected_open_templates_path is not None
        has_closed = self.selected_closed_templates_path is not None
        self.create_model_btn.setEnabled(has_open and has_closed)

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

    def _create_dtw_model_thread(self) -> None:
        """Thread function to create DTW model."""
        from mindmove.model.core.features.features_registry import FEATURES
        from mindmove.model.core.windowing import sliding_window
        from mindmove.model.core.algorithm import compute_threshold, compute_per_template_statistics

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

        # Compute per-template statistics
        print("\nAnalyzing template quality...")

        # Open templates statistics
        print("\n  --- OPEN Templates ---")
        open_stats = compute_per_template_statistics(open_templates_features, n_worst=3)
        print(f"  Per-template avg distances:")
        for i, avg_dist in enumerate(open_stats['per_template_avg']):
            marker = " ***" if (i + 1) in open_stats['worst_indices'] else ""
            print(f"    Template {i + 1}: avg={avg_dist:.4f}, max={open_stats['per_template_max'][i]:.4f}, min={open_stats['per_template_min'][i]:.4f}{marker}")
        print(f"  WORST templates (highest avg distance): {open_stats['worst_indices']}")
        print(f"  BEST templates (lowest avg distance): {open_stats['best_indices']}")

        # Closed templates statistics
        print("\n  --- CLOSED Templates ---")
        closed_stats = compute_per_template_statistics(closed_templates_features, n_worst=3)
        print(f"  Per-template avg distances:")
        for i, avg_dist in enumerate(closed_stats['per_template_avg']):
            marker = " ***" if (i + 1) in closed_stats['worst_indices'] else ""
            print(f"    Template {i + 1}: avg={avg_dist:.4f}, max={closed_stats['per_template_max'][i]:.4f}, min={closed_stats['per_template_min'][i]:.4f}{marker}")
        print(f"  WORST templates (highest avg distance): {closed_stats['worst_indices']}")
        print(f"  BEST templates (lowest avg distance): {closed_stats['best_indices']}")

        print("\n  Note: Templates marked *** have highest avg distance to others")
        print("  Consider removing these if model performance is poor.")

        # Save model
        print("\nSaving model...")
        now = datetime.now()
        formatted_now = now.strftime("%Y%m%d_%H%M%S")

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
            }
        }

        models_dir = "data/models"
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"MindMove_Model_{formatted_now}_{model_name}.pkl")

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
