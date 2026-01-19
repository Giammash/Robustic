# from __future__ import annotations
# from typing import TYPE_CHECKING, Any, Tuple, Optional, Dict, Union

# if TYPE_CHECKING:
#     pass

# from PySide6.QtCore import QObject

# # MindMove imports
# from mindmove.model.core.model import Model
# from mindmove.model.core.dataset import Dataset


# class MindMoveInterface(QObject):
#     def __init__(self, parent: QObject | None = ...) -> None:
#         super().__init__(parent)

#         #### diagnostic
        
#         self.model: Model = None
#         self.dataset: Dataset = None

#         self.model_is_loaded: bool = False

#     def create_dataset(self, dataset: Dict) -> Dict[str, Dict[str, Any]]:
#         self.dataset = Dataset(dataset)

#         return self.dataset.create_dataset()

#     def train_model(self, dataset: Dict[str, Dict[str, Any]]) -> None:
#         training_data = dataset["training"]
#         testing_data = dataset["testing"]

#         self.model = Model()
#         self.model.fit(training_data, testing_data)

#     def predict(self, x: Any) -> Any:
#         if not self.model_is_loaded:
#             raise Exception("Model is not loaded!")

#         # TODO: Implement the prediction
#         prediction = self.model.predict(x)
#         return prediction

#     def save_model(self, model_path: str) -> None:
#         self.model.save(model_path)

#     def load_model(self, model_path: str) -> None:
#         self.model = Model()
#         self.model.load(model_path)

#         # Set the model as loaded
#         self.model_is_loaded = True

"""
Modified interface.py to support diagnostic model.

Add this to your mindmove/model/interface.py or use it as a replacement.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    pass

from PySide6.QtCore import QObject

# MindMove imports
from mindmove.model.core.model import Model
from mindmove.model.core.dataset import Dataset

# Import diagnostic model
try:
    from mindmove.model.core.diagnostic import DiagnosticModel
    DIAGNOSTIC_AVAILABLE = True
except ImportError:
    DIAGNOSTIC_AVAILABLE = False
    print("Warning: DiagnosticModel not available")


class MindMoveInterface(QObject):
    def __init__(self, parent: QObject | None = ..., use_diagnostic: bool = False) -> None:
        """
        Initialize the MindMove Interface.
        
        Args:
            parent: Parent QObject
            use_diagnostic: If True, use lightweight DiagnosticModel instead of full Model
        """
        super().__init__(parent)

        self.use_diagnostic = use_diagnostic
        self.model: Model | DiagnosticModel = None
        self.dataset: Dataset = None
        self.model_is_loaded: bool = False

    def create_dataset(self, dataset: Dict) -> Dict[str, Dict[str, Any]]:
        self.dataset = Dataset(dataset)
        return self.dataset.create_dataset()

    def train_model(self, dataset: Dict[str, Dict[str, Any]]) -> None:
        if self.use_diagnostic:
            print("Training not available in diagnostic mode")
            return
            
        training_data = dataset["training"]
        testing_data = dataset["testing"]

        self.model = Model()
        self.model.fit(training_data, testing_data)

    def predict(self, x: Any) -> Any:
        if not self.model_is_loaded:
            raise Exception("Model is not loaded!")

        prediction = self.model.predict(x)
        return prediction

    def save_model(self, model_path: str) -> None:
        if self.use_diagnostic:
            print("Save not available in diagnostic mode")
            return
            
        self.model.save(model_path)

    def load_model(self, model_path: str) -> None:
        """
        Load model. If use_diagnostic is True, creates DiagnosticModel instead.
        """
        if self.use_diagnostic:
            if not DIAGNOSTIC_AVAILABLE:
                raise ImportError("DiagnosticModel not available")
            
            print("\n" + "="*70)
            print("LOADING DIAGNOSTIC MODEL (lightweight)")
            print("="*70)
            self.model = DiagnosticModel()
            self.model_is_loaded = True
            print("Diagnostic model ready - no file loaded")
            print("="*70 + "\n")
        else:
            # Normal model loading
            self.model = Model()
            self.model.load(model_path)
            self.model_is_loaded = True
    
    def get_latest_features(self):
        """Get latest features from diagnostic model."""
        if self.use_diagnostic and hasattr(self.model, 'get_latest_features'):
            return self.model.get_latest_features()
        return None
    
    def get_feature_history(self):
        """Get feature history from diagnostic model."""
        if self.use_diagnostic and hasattr(self.model, 'get_feature_history'):
            return self.model.get_feature_history()
        return None
    
    def print_summary(self):
        """Print timing summary from diagnostic model."""
        if self.use_diagnostic and hasattr(self.model, 'print_summary'):
            self.model.print_summary()

    def update_threshold_open(self, s_open: float) -> None:
        """Update OPEN threshold with new s_open value."""
        if self.model and hasattr(self.model, 'update_threshold_open'):
            self.model.update_threshold_open(s_open)

    def update_threshold_closed(self, s_closed: float) -> None:
        """Update CLOSED threshold with new s_closed value."""
        if self.model and hasattr(self.model, 'update_threshold_closed'):
            self.model.update_threshold_closed(s_closed)

    def update_thresholds(self, s_open: float = None, s_closed: float = None) -> None:
        """Update thresholds with new s values (standard deviation multipliers)."""
        if self.model and hasattr(self.model, 'update_thresholds'):
            self.model.update_thresholds(s_open, s_closed)

    def reset_history(self) -> None:
        """Reset history buffers for new acquisition session."""
        if self.model and hasattr(self.model, 'reset_history'):
            self.model.reset_history()

    def get_distance_history(self):
        """Get distance history for plotting."""
        if self.model and hasattr(self.model, 'get_distance_history'):
            return self.model.get_distance_history()
        return None

    def get_current_thresholds(self):
        """Get current threshold values."""
        if self.model:
            return {
                "threshold_open": getattr(self.model, 'THRESHOLD_OPEN', None),
                "threshold_closed": getattr(self.model, 'THRESHOLD_CLOSED', None),
                "s_open": getattr(self.model, 's_open', 1.0),
                "s_closed": getattr(self.model, 's_closed', 1.0),
                "mean_open": getattr(self.model, 'mean_open', None),
                "std_open": getattr(self.model, 'std_open', None),
                "mean_closed": getattr(self.model, 'mean_closed', None),
                "std_closed": getattr(self.model, 'std_closed', None),
            }
        return None