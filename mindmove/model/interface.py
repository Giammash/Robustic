from __future__ import annotations
from typing import TYPE_CHECKING, Any, Tuple, Optional, Dict, Union

if TYPE_CHECKING:
    pass

from PySide6.QtCore import QObject

# MindMove imports
from mindmove.model.core.model import Model
from mindmove.model.core.dataset import Dataset


class MindMoveInterface(QObject):
    def __init__(self, parent: QObject | None = ...) -> None:
        super().__init__(parent)

        self.model: Model = None
        self.dataset: Dataset = None

        self.model_is_loaded: bool = False

    def create_dataset(self, dataset: Dict) -> Dict[str, Dict[str, Any]]:
        self.dataset = Dataset(dataset)

        return self.dataset.create_dataset()

    def train_model(self, dataset: Dict[str, Dict[str, Any]]) -> None:
        training_data = dataset["training"]
        testing_data = dataset["testing"]

        self.model = Model()
        self.model.fit(training_data, testing_data)

    def predict(self, x: Any) -> Any:
        if not self.model_is_loaded:
            raise Exception("Model is not loaded!")

        # TODO: Implement the prediction
        prediction = self.model.predict(x)
        return prediction

    def save_model(self, model_path: str) -> None:
        self.model.save(model_path)

    def load_model(self, model_path: str) -> None:
        self.model = Model()
        self.model.load(model_path)

        # Set the model as loaded
        self.model_is_loaded = True
