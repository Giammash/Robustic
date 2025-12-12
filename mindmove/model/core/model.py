from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Dict
import numpy as np
import time

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
        """Initialize the DTW-based hand state classifier model."""
        ######### Begin of Modified Code #########

        # Initialization of variables for online protocol

        # sampling configuration
        self.FSAMP = config.FSAMP  # Sampling frequency of Muovi EMG
        self.num_channels = config.num_channels # Number of EMG channels
        self.dead_channels = config.dead_channels  # 1-indexed list of dead channels to be removed
        self.active_channels = config.active_channels  # List of active channels after removing dead channels

        # buffer configuration (1 second buffer with sliding window)
        self.buffer_length_s = config.template_duration  # Length of the buffer sliding window in seconds
        self.buffer_length = self.FSAMP * self.buffer_length_s  # Length of the buffer in samples
        
        # DTW computation timing 
        self.increment_dtw_s = config.increment_dtw  # Overlap of the buffer sliding window in seconds
        self.increment_dtw = int(self.FSAMP * self.increment_dtw_s)  # Overlap of the buffer in samples
        
        # feature extraction configuration
        self.window_length = config.window_length # Length of the window for feature extraction in samples
        self.increment = config.increment # Overlap for feature extraction in samples
        
        # initialize real-time buffer
        self.emg_rt_buffer = np.zeros((self.num_channels, self.buffer_length))
        # so every update has to do:
        # self.emg_rt_buffer = np.roll(self.emg_rt_buffer, -new_samples, axis=1)
        # self.emg_rt_buffer[:, -new_samples:] = emg_data
        
        # track samples from last DTW computation
        self.samples_since_last_dtw = 0

        # templates and thresholds
        self.templates_open = None
        self.templates_closed = None
        self.THRESHOLD_OPEN = None
        self.THRESHOLD_CLOSED = None

        # state machine
        # self.current_state = "OPEN" # or "CLOSED"
        # self.current_state = "CLOSED"

        # feature choice
        self.feature_name = "wl"

        # control the majority more or 5 consecutive predictions to switch state
        self.consecutive_required = config.SMOOTHING_WINDOW
        self.window_majority_length = config.SMOOTHING_WINDOW
        self.last_predictions = []

        # refractory period variables
        self.refractory_period_s = 0.75  # seconds
        # self.time_since_last_switch = 0.0 # seconds

        # time for debugging
        self.last_dtw_time = time.time()
        self.dtw_count = 0

        self.dtw_distances = []
        self.dtw_times = []



    def _update_buffer(self, new_samples: np.ndarray) -> bool:
        """
        Update the real-time buffer with new incoming samples.
        Args:
            new_samples (np.ndarray): New incoming EMG samples of shape (n_channels, new_samples).
        Returns:
            bool: True if enough samples are available for DTW computation, False otherwise.
        """
        # new_samples shape: (n_channels, n_new_samples)
        n_new_samples = new_samples.shape[1]
        # print(n_new_samples.shape)
        # print(f"number of new samples: {n_new_samples}")
        # print(f"buffer length: {self.buffer_length}")

        if n_new_samples >= self.buffer_length:
            # keep only the last buffer_length samples from new_samples
            # if get more than the buffer length, just keep the last buffer_length samples
            self.emg_rt_buffer = new_samples[:, -self.buffer_length :].copy()
            self.samples_since_last_dtw = self.buffer_length

        else:
            self.emg_rt_buffer = np.roll(self.emg_rt_buffer, -n_new_samples, axis=1)
            self.emg_rt_buffer[:, -n_new_samples :] = new_samples
            self.samples_since_last_dtw += n_new_samples
        
        # check id should compute DTW
        should_compute = self.samples_since_last_dtw >= self.increment_dtw

        if should_compute:
            self.samples_since_last_dtw = 0 # reset counter
        
        # print(f"should compute? {should_compute}")
        
        return should_compute
    

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
        if self.templates_open is None or self.templates_closed is None:
            raise ValueError("Model templates are not trained yet.")
        
        save_dict = {
            "templates_open": self.templates_open,
            "templates_closed": self.templates_closed,
            "THRESHOLD_OPEN": self.THRESHOLD_OPEN,
            "THRESHOLD_CLOSED": self.THRESHOLD_CLOSED,
            "feature_name": self.feature_name,
        }

        import pickle
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(save_dict, f)
        # pass
        print(f"Model saved to: {model_path}")

    # TODO: Implement the load method
    def load(self, model_path: str) -> None:
        """
        Load the model from the model_path.
        """
        import pickle

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.templates_open = data["open_templates"]
        self.templates_closed = data["closed_templates"]
        self.THRESHOLD_OPEN = data["threshold_base_open"] *2
        self.mean_open = data["mean_open"]
        self.std_open = data["std_open"]
        self.THRESHOLD_CLOSED = data["threshold_base_closed"] *2
        self.mean_closed = data["mean_closed"]
        self.std_closed = data["std_closed"]
        self.feature_name = data["feature_name"]
        
        self.current_state = "CLOSED"

        # pass
        print(f"Model loaded from: {model_path}")
        print(f"  - OPEN templates: {len(self.templates_open)}")
        print(f"  - CLOSED templates: {len(self.templates_closed)}")
        print(f"  - OPEN threshold: {self.THRESHOLD_OPEN:.4f}")
        print(f"  - CLOSED threshold: {self.THRESHOLD_CLOSED:.4f}")
        print(f"  - Feature: {self.feature_name}")

    # TODO: Implement the predict method
    def predict(self, x: Any) -> List[float]:
        """
        Predict the hand state from new EMG data

        Args:
            x (Any): New EMG data for prediction.
        
        Returns:
            List[float]: Predicted hand state (0.0 for OPEN, 1.0 for CLOSED).
        
        """
        if self.templates_open is None or self.templates_closed is None:
            raise ValueError("Model not loaded! Call load() first.")
        
        # Update real-time buffer
        # print(f"x shape: {x.shape}")
        should_compute_dtw = self._update_buffer(x)

        if not should_compute_dtw:
            # Not enough new samples to compute DTW yet
            return 1.0 if self.current_state == "CLOSED" else 0.0
        
        # --- DTW computation (every increment_dtw samples) ---
        dtw_start_time = time.perf_counter()
        current_time = time.time()
        print(f"Time since last DTW: {(current_time - self.last_dtw_time) * 1000} ms")
        time_since_last =  current_time - self.last_dtw_time
        
        # apply filtering
        emg_buffer = apply_rtfiltering(self.emg_rt_buffer) if config.ENABLE_FILTERING else self.emg_rt_buffer.copy()
        # print("EMG BUFFER SHAPE:", emg_buffer.shape)

        # extract features from the template_nsamp buffer
        windowed_emg_buffer = sliding_window(emg_buffer, self.window_length, self.increment)

        feature_info = FEATURES[self.feature_name]
        feature_fn = feature_info["function"]
        features_emg_buffer = feature_fn(windowed_emg_buffer)

        # DTW distances 
        if self.current_state == "OPEN":
            # Check if should switch to CLOSED
            D_closed = compute_distance_from_training_set_online(features_emg_buffer, self.templates_closed)
            if D_closed < self.THRESHOLD_CLOSED:
                triggered_state = "CLOSED"
            else:
                triggered_state = "OPEN"
            print(f"[DTW] State: OPEN | D_closed: {D_closed:.4f} | Threshold: {self.THRESHOLD_CLOSED:.4f} | "
              f"Trigger: {triggered_state} | Δt: {time_since_last:.1f}ms")
            
            self.dtw_distances.append(("closed", D_closed, self.THRESHOLD_CLOSED))


        elif self.current_state == "CLOSED":
            # Check if should switch to OPEN
            D_open = compute_distance_from_training_set_online(features_emg_buffer, self.templates_open)
            if D_open < self.THRESHOLD_OPEN:
                triggered_state = "OPEN"
            else:
                triggered_state = "CLOSED"

            print(f"[DTW] State: CLOSED | D_open: {D_open:.4f} | Threshold: {self.THRESHOLD_OPEN:.4f} | "
              f"Trigger: {triggered_state} | Δt: {time_since_last:.1f}ms")

            self.dtw_distances.append(("open", D_open, self.THRESHOLD_OPEN))
        
        # D_open = compute_distance_from_training_set_online(features_emg_buffer, templates_open, feature_name)
        # D_closed = compute_distance_from_training_set_online(features_emg_buffer, templates_closed, feature_name)

        # State machine logic

        if len(self.dtw_distances) < 100:
            self.dtw_distances.pop(0)

        previous_state = self.current_state
        
        if config.POST_PREDICTION_SMOOTHING != "NONE":
            self.last_predictions.append(triggered_state)

            if len(self.last_predictions) > self.window_majority_length:
                self.last_predictions.pop(0)
            
                if config.POST_PREDICTION_SMOOTHING == "MAJORITY VOTE":
                    if self.last_predictions.count("CLOSED") > self.last_predictions.count("OPEN"):
                        self.current_state = "CLOSED"
                    else:
                        self.current_state = "OPEN"
                    
                    # clear the buffer to start new majority vote
                    self.last_predictions = []

                elif config.POST_PREDICTION_SMOOTHING == "5 CONSECUTIVE":
                    if len(self.last_predictions) == self.consecutive_required and all(p == triggered_state for p in self.last_predictions):
                        self.current_state = triggered_state
                        self.last_predictions = []

        else: 
            # no smoothing
            self.current_state = triggered_state

        # Debug timing
        current_time = time.time()
        time_diff = current_time - self.last_dtw_time 
        self.last_dtw_time = current_time

        if time_diff > 0:
            print(f"DTW computed: {self.current_state} (Δt={time_diff*1000:.1f}ms)")
        
        # Print transition
        if previous_state != self.current_state:
            print(f"\n{'='*60}")
            print(f"🔄 STATE TRANSITION: {previous_state} → {self.current_state}")
            print(f"{'='*60}\n")
        
        # update timing
        dtw_end_time = time.perf_counter()
        dtw_computation_time = (dtw_end_time - dtw_start_time) * 1000
        self.dtw_times.append(dtw_computation_time)
        if len(self.dtw_times) < 100:
            self.dtw_times.pop(0)

        self.last_dtw_time = current_time
        self.dtw_count += 1

        # everz 20 DTW computations print statistic

        if self.dtw_count % 20 == 0:
            avg_time = np.mean(self.dtw_times)
            print(f"\n stats (last {len(self.dtw_times)} DTW): "
            f"AVG computation={avg_time}ms"
            f"avg dt={np.mean([time_since_last])}ms")
                    
        # return self.current_state
        return 1.0 if self.current_state == "CLOSED" else 0.0

        # pass
