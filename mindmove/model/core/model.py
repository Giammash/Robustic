from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Dict
import numpy as np

from mindmove.config import config

# IMPORT YOUR SIGNAL PROCESSING / FEATURE / DTW FUNCTIONS HERE
# Example (you MUST adapt these paths to your project):

from mindmove.model.core.features.features_registry import FEATURES
from mindmove.model.core.features import *
from mindmove.model.core.windowing import sliding_window
from mindmove.model.core.filtering import apply_rtfiltering
from mindmove.model.core.algorithm import compute_distance_from_training_set_online

if TYPE_CHECKING:
    pass

# Import PyTorch -> Install PyTorch if you need
# poetry run pip install torch --index-url https://download.pytorch.org/whl/cu121 --upgrade
# if CUDA is available, otherwise use the CPU version


# TODO: Implement the Model class
class Model:
    def __init__(self) -> None:
        """ """
        ######### Begin of Modified Code #########

        # Initialization of variables for online protocol
        self.FSAMP = config.FSAMP  # Sampling frequency of Muovi EMG
        self.buffer_length_s = config.template_duration  # Length of the buffer sliding window in seconds
        self.buffer_length = self.FSAMP * self.buffer_length_s  # Length of the buffer in samples
        self.increment_dtw_s = config.increment_dtw  # Overlap of the buffer sliding window in seconds
        self.increment_dtw = int(self.FSAMP * self.increment_s)  # Overlap of the buffer in samples
        self.window_length = config.window_length # Length of the window for feature extraction in samples
        self.increment = config.increment # Overlap for feature extraction in samples
        self.num_channels = config.num_channels # Number of EMG channels
        self.dead_channels = config.dead_channels  # 1-indexed list of dead channels to be removed
        self.active_channels = config.active_channels  # List of active channels after removing dead channels

        self.emg_rt_buffer = np.zeros((self.num_channels, self.buffer_length))
        # so every update has to do:
        # self.emg_rt_buffer = np.roll(self.emg_rt_buffer, -new_samples, axis=1)
        # self.emg_rt_buffer[:, -new_samples:] = emg_data

        # templates and thresholds
        self.templates_open = None
        self.templates_closed = None
        self.THRESHOLD_OPEN = None
        self.THRESHOLD_CLOSED = None

        # state machine
        self.current_state = "OPEN" # or "CLOSED"

        # feature choice
        self.feature_name = "wl"

        # control the majority more or 5 consecutive predictions to switch state
        self.consecutive_required = 5
        self.window_majority_length = 5
        self.last_predictions = []


    def _update_buffer(self, new_samples: np.ndarray):
        # new_samples shape: (n_channels, n_new_samples)
        n_new_samples = new_samples.shape[1]
        if n_new_samples > self.buffer_length:
            # keep only the òast buffer_length samples from new_samples
            self.emg_rt_buffer = new_samples[:, -self.buffer_length :].copy()
        else:
            self.emg_rt_buffer = np.roll(self.emg_rt_buffer, -n_new_samples, axis=1)
            self.emg_rt_buffer[:, -n_new_samples :] = new_samples

    # TODO: Implement the fit method
    def fit(self, training_data: Dict[str, Any], testing_data: Dict[str, Any]) -> None:
        """
        Fit the model to the training data.
        """
        pass

    # TODO: Implement the save method
    def save(self, model_path: str) -> None:
        """
        Save the model to the model_path.
        """
        save_dict = {
            "templates_open": self.templates_open,
            "templates_closed": self.templates_closed,
            "THRESHOLD_OPEN": self.THRESHOLD_OPEN,
            "THRESHOLD_CLOSED": self.THRESHOLD_CLOSED,
            "feature_name": self.feature_name,
        }

        import pickle
        with open(model_path, "wb") as f:
            pickle.dump(save_dict, f)
        # pass

    # TODO: Implement the load method
    def load(self, model_path: str) -> None:
        """
        Load the model from the model_path.
        """
        import pickle
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.templates_open = data["templates_open"]
        self.templates_closed = data["templates_closed"]
        self.THRESHOLD_OPEN = data["THRESHOLD_OPEN"]
        self.THRESHOLD_CLOSED = data["THRESHOLD_CLOSED"]
        self.feature_name = data["feature_name"]
        # pass

    # TODO: Implement the predict method
    def predict(self, x: Any) -> List[float]:
        """
        Predict the output for the input x.
        """

        # Update real-time buffer
        self._update_buffer(x)

        # apply filtering
        
        emg_buffer = apply_rtfiltering(self.emg_rt_buffer) if config.ENABLE_FILTERING else self.emg_rt_buffer.copy()
        # print("EMG BUFFER SHAPE:", emg_buffer.shape)

        # extract features
        windowed_emg_buffer = sliding_window(emg_buffer, window_length=self.window_length, overlap=self.overlap)

        feature_info = FEATURES[self.feature_name]
        feature_fn = feature_info["function"]
        features_emg_buffer = feature_fn(windowed_emg_buffer)

        # DTW distances 
        if self.current_state == "OPEN":
            D_closed = compute_distance_from_training_set_online(features_emg_buffer, self.templates_closed, self.feature_name)
            if D_closed < self.THRESHOLD_CLOSED:
                triggered_state = "CLOSED"
            else:
                triggered_state = "OPEN"

        elif self.current_state == "CLOSED":
            D_open = compute_distance_from_training_set_online(features_emg_buffer, self.templates_open, self.feature_name)
            if D_open < self.THRESHOLD_OPEN:
                triggered_state = "OPEN"
            else:
                triggered_state = "CLOSED"
        


        # D_open = compute_distance_from_training_set_online(features_emg_buffer, templates_open, feature_name)
        # D_closed = compute_distance_from_training_set_online(features_emg_buffer, templates_closed, feature_name)

        # State machine logic
        if config.POST_PREDICTION_SMOOTHING != "NONE":
            self.last_predictions.append(triggered_state)
            if len(self.last_predictions) > self.window_majority_length:
                self.last_predictions.pop(0)
            
                if config.POST_PREDICTION_SMOOTHING == "MAJORITY VOTE":
                    if self.last_predictions.count("CLOSED") > self.last_predictions.count("OPEN"):
                        self.current_state = "CLOSED"
                        self.last_predictions = []

                    else:
                        self.current_state = "OPEN"
                        self.last_predictions = []

                elif config.POST_PREDICTION_SMOOTHING == "5 CONSECUTIVE":
                    if len(self.last_predictions) == self.consecutive_required and all(p == triggered_state for p in self.last_predictions):
                        self.current_state = triggered_state
                        self.last_predictions = []
        else:
            # no smoothing
            self.current_state = triggered_state
                    
        # return self.current_state
        return 1.0 if self.current_state == "CLOSED" else 0.0

        # pass
