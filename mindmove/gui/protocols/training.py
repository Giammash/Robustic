from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal, QThread
from PySide6.QtWidgets import QFileDialog, QMessageBox
import pickle
from datetime import datetime
import numpy as np
import os

# MindMove imports
from mindmove.model.interface import MindMoveInterface

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

        # Model Interface
        self.model_interface: MindMoveInterface = MindMoveInterface(
            parent=self.main_window
        )

        # File management:
        self.recordings_dir_path: str = "data/recordings/"
        self.models_dir_path: str = "data/models/"
        self.datasets_dir_path: str = "data/datasets/"

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

                required_keys = [
                    "emg",
                    "kinematics",
                    "timings_kinematics",
                    "timings_emg",
                    "label",
                    "task",
                ]

                if not all(key in keys for key in required_keys):
                    print(f" {f} is an invalid recording!")
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
            if task_name in df.keys():
                df[task_name]["emg"] = np.concatenate([df[task_name]["emg"], v["emg"]])
                df[task_name]["kinematics"] = np.concatenate(
                    [df[task_name]["kinematics"], v["kinematics"]], axis=-1
                )
            else:
                df[task_name] = {}
                df[task_name]["emg"] = v["emg"]
                df[task_name]["kinematics"] = v["kinematics"]

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
        # Create Datasets
        self.training_create_dataset_group_box = (
            self.main_window.ui.trainingCreateDatasetGroupBox
        )
        self.training_create_datasets_select_recordings_push_button = (
            self.main_window.ui.trainingCreateDatasetsSelectRecordingsPushButton
        )
        self.training_create_datasets_select_recordings_push_button.clicked.connect(
            self.select_recordings
        )
        self.training_create_dataset_selected_recordings_list_widget = (
            self.main_window.ui.trainingCreateDatasetSelectedRecordingsListWidget
        )
        self.training_create_dataset_selected_recordings_list_widget.clear()

        self.training_create_dataset_label_line_edit = (
            self.main_window.ui.trainingCreateDatasetLabelLineEdit
        )
        self.training_create_dataset_push_button = (
            self.main_window.ui.trainingCreateDatasetPushButton
        )
        self.training_create_dataset_push_button.clicked.connect(self._create_dataset)
        self.training_create_dataset_push_button.setEnabled(False)
        self.train_create_dataset_progress_bar = (
            self.main_window.ui.trainingCreateDatasetProgressBar
        )
        self.train_create_dataset_progress_bar.setValue(0)

        # Train Model
        self.training_train_model_group_box = (
            self.main_window.ui.trainingTrainModelGroupBox
        )
        self.training_select_dataset_push_button = (
            self.main_window.ui.trainingSelectDatasetPushButton
        )
        self.training_select_dataset_push_button.clicked.connect(self._select_dataset)
        self.training_selected_dataset_label = (
            self.main_window.ui.trainingSelectedDatasetLabel
        )
        self.training_selected_dataset_label.setText("No dataset selected!")
        self.training_model_label_line_edit = (
            self.main_window.ui.trainingModelLabelLineEdit
        )
        self.training_train_new_model_radio_button = (
            self.main_window.ui.trainingTrainNewModelRadioButton
        )
        self.training_train_existing_model_radio_button = (
            self.main_window.ui.trainingTrainExistingModelRadioButton
        )
        self.training_load_existing_model_push_button = (
            self.main_window.ui.trainingLoadExistingModelPushButton
        )
        self.training_load_existing_model_label = (
            self.main_window.ui.trainingLoadExistingModelLabel
        )
        self.training_load_existing_model_push_button.clicked.connect(
            self._load_existing_model
        )

        self.training_train_new_model_radio_button.setChecked(True)
        self.training_train_new_model_radio_button.toggled.connect(
            self._toggle_train_model_stacked_widget
        )

        self.train_model_push_button = self.main_window.ui.trainingTrainModelPushButton
        self.train_model_push_button.clicked.connect(self._train_model)
        self.train_model_push_button.setEnabled(False)
        self.training_train_model_stacked_widget = (
            self.main_window.ui.trainingTrainModelStackedWidget
        )
        self.training_train_model_stacked_widget.setCurrentIndex(0)
        self.training_progress_bar = self.main_window.ui.trainingProgressBar
        self.training_progress_bar.setValue(0)
