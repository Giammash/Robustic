from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Dict
import numpy as np
import time
from datetime import datetime

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
        self.num_channels = config.num_channels  # Number of EMG channels
        # dead_channels stored as 0-indexed (index 0-31 for 32 channels)
        self.dead_channels = config.dead_channels
        self.active_channels = config.active_channels  # List of active channel indices (0-indexed)

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

        # state machine - initialize to CLOSED (hand starts closed)
        self.current_state = "CLOSED"

        # feature choice
        self.feature_name = "wl"

        # distance aggregation method (can be overridden by model settings)
        # Options: "average", "minimum", "avg_3_smallest"
        self.distance_aggregation = "average"

        # smoothing method (can be overridden by model settings)
        # Options: "MAJORITY VOTE", "5 CONSECUTIVE", "NONE"
        self.smoothing_method = config.POST_PREDICTION_SMOOTHING

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

        # History for post-acquisition plotting
        # Each entry: (timestamp, D_open or None, D_closed or None, current_state)
        self.distance_history = []
        self.state_transitions = []  # (timestamp, from_state, to_state)

        # Threshold tuning parameters (separate s values for each class)
        self.s_open = 1.0  # Default s=1 for OPEN threshold
        self.s_closed = 1.0  # Default s=1 for CLOSED threshold

        # History for post-acquisition plotting
        # Each entry: (timestamp, D_open, D_closed, current_state, triggered_state)
        self.distance_history = []
        self.emg_history = []  # Store EMG snapshots for plotting
        self.state_history = []  # Store (timestamp, state) for transitions

        # Threshold tuning parameter (s value)
        self.threshold_s = 1.0  # Default s=1 (mean + 1*std)

        # Last computed distance and threshold (for cleaner terminal output)
        self._last_distance = 0.0
        self._last_threshold = 0.0



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
        # Thresholds are now correctly computed as mean + s*std (s=1 by default)
        # Use a configurable multiplier for tuning if needed
        threshold_multiplier = data.get("threshold_multiplier", 1.0)
        self.THRESHOLD_OPEN = data["threshold_base_open"] * threshold_multiplier
        self.mean_open = data["mean_open"]
        self.std_open = data["std_open"]
        self.THRESHOLD_CLOSED = data["threshold_base_closed"] * threshold_multiplier
        self.mean_closed = data["mean_closed"]
        self.std_closed = data["std_closed"]
        self.feature_name = data["feature_name"]

        # Load new model settings (with defaults for backwards compatibility)
        # dead_channels stored as 0-indexed
        self.dead_channels = data.get("dead_channels", config.dead_channels)
        # Compute active channels from dead channels
        self.active_channels = [i for i in range(self.num_channels) if i not in self.dead_channels]

        # Distance aggregation method: "average", "minimum", "avg_3_smallest"
        self.distance_aggregation = data.get("distance_aggregation", "average")

        # Smoothing method: "MAJORITY VOTE", "5 CONSECUTIVE", "NONE"
        smoothing_method = data.get("smoothing_method", config.POST_PREDICTION_SMOOTHING)
        self.smoothing_method = smoothing_method

        # Load training parameters for feature extraction consistency
        params = data.get("parameters", {})
        if params:
            self.window_length = params.get("window_samples", config.window_length)
            self.increment = params.get("overlap_samples", config.increment)
            print(f"  - Window/overlap (from model): {self.window_length}/{self.increment} samples")
        else:
            # Fallback to config for backwards compatibility
            self.window_length = config.window_length
            self.increment = config.increment
            print(f"  - Window/overlap (from config): {self.window_length}/{self.increment} samples")

        self.current_state = "CLOSED"

        # pass
        print(f"Model loaded from: {model_path}")
        print(f"  - OPEN templates: {len(self.templates_open)}")
        print(f"  - CLOSED templates: {len(self.templates_closed)}")
        print(f"  - OPEN threshold: {self.THRESHOLD_OPEN:.4f}")
        print(f"  - CLOSED threshold: {self.THRESHOLD_CLOSED:.4f}")
        print(f"  - Feature: {self.feature_name}")
        print(f"  - Dead channels (0-indexed): {self.dead_channels}")
        print(f"  - Active channels: {len(self.active_channels)}")
        print(f"  - Distance aggregation: {self.distance_aggregation}")
        print(f"  - Smoothing method: {self.smoothing_method}")

    def update_threshold_open(self, s_open: float) -> None:
        """
        Update OPEN threshold based on s_open value.

        Args:
            s_open: Standard deviations above mean for OPEN threshold.
        """
        if not hasattr(self, 'mean_open') or not hasattr(self, 'std_open'):
            print("Warning: Model statistics not loaded, cannot update threshold")
            return

        self.s_open = s_open
        self.THRESHOLD_OPEN = self.mean_open + s_open * self.std_open

        print(f"[THRESHOLD] OPEN: s={s_open:.2f} → threshold={self.THRESHOLD_OPEN:.4f}")

    def update_threshold_closed(self, s_closed: float) -> None:
        """
        Update CLOSED threshold based on s_closed value.

        Args:
            s_closed: Standard deviations above mean for CLOSED threshold.
        """
        if not hasattr(self, 'mean_closed') or not hasattr(self, 'std_closed'):
            print("Warning: Model statistics not loaded, cannot update threshold")
            return

        self.s_closed = s_closed
        self.THRESHOLD_CLOSED = self.mean_closed + s_closed * self.std_closed

        print(f"[THRESHOLD] CLOSED: s={s_closed:.2f} → threshold={self.THRESHOLD_CLOSED:.4f}")

    def set_threshold_open_direct(self, threshold: float) -> None:
        """
        Set OPEN threshold directly and compute corresponding s value.

        Args:
            threshold: Direct threshold value.
        """
        if not hasattr(self, 'mean_open') or not hasattr(self, 'std_open'):
            print("Warning: Model statistics not loaded, cannot update threshold")
            return

        self.THRESHOLD_OPEN = max(0, threshold)  # Clamp to non-negative
        # Compute s from threshold: threshold = mean + s*std => s = (threshold - mean) / std
        if self.std_open > 0:
            self.s_open = (self.THRESHOLD_OPEN - self.mean_open) / self.std_open
        else:
            self.s_open = 0

        print(f"[THRESHOLD] OPEN: threshold={self.THRESHOLD_OPEN:.4f} (s={self.s_open:.2f})")

    def set_threshold_closed_direct(self, threshold: float) -> None:
        """
        Set CLOSED threshold directly and compute corresponding s value.

        Args:
            threshold: Direct threshold value.
        """
        if not hasattr(self, 'mean_closed') or not hasattr(self, 'std_closed'):
            print("Warning: Model statistics not loaded, cannot update threshold")
            return

        self.THRESHOLD_CLOSED = max(0, threshold)  # Clamp to non-negative
        # Compute s from threshold: threshold = mean + s*std => s = (threshold - mean) / std
        if self.std_closed > 0:
            self.s_closed = (self.THRESHOLD_CLOSED - self.mean_closed) / self.std_closed
        else:
            self.s_closed = 0

        print(f"[THRESHOLD] CLOSED: threshold={self.THRESHOLD_CLOSED:.4f} (s={self.s_closed:.2f})")

    def update_thresholds(self, s_open: float = None, s_closed: float = None) -> None:
        """
        Update both thresholds. If only one is provided, only that one is updated.

        Args:
            s_open: Standard deviations for OPEN threshold (None = keep current)
            s_closed: Standard deviations for CLOSED threshold (None = keep current)
        """
        if s_open is not None:
            self.update_threshold_open(s_open)
        if s_closed is not None:
            self.update_threshold_closed(s_closed)

    def reset_history(self) -> None:
        """Reset all history buffers for a new acquisition session."""
        self.distance_history = []
        self.state_transitions = []
        self.dtw_distances = []
        self.dtw_times = []
        self.last_predictions = []
        self.dtw_count = 0
        self.last_dtw_time = time.time()
        print("[MODEL] History reset for new session")

    def get_last_result(self) -> Dict[str, Any]:
        """
        Get the last prediction result with extended info for Unity.

        Returns:
            Dictionary with state, distance, threshold, and state_name.
        """
        if hasattr(self, '_last_result'):
            return self._last_result
        return {
            'state': 1.0 if self.current_state == "CLOSED" else 0.0,
            'distance': 0.0,
            'threshold': 0.0,
            'state_name': self.current_state
        }

    def get_distance_history(self) -> Dict[str, Any]:
        """
        Get the distance history for plotting.

        Returns:
            Dictionary with timestamps, distances (with None for gaps), states, and thresholds.
        """
        if not self.distance_history:
            return None

        timestamps = [h[0] for h in self.distance_history]
        D_open = [h[1] for h in self.distance_history]  # None when state was OPEN
        D_closed = [h[2] for h in self.distance_history]  # None when state was CLOSED
        states = [h[3] for h in self.distance_history]

        # Normalize timestamps to start from 0
        t0 = timestamps[0] if timestamps else 0
        timestamps = [t - t0 for t in timestamps]

        return {
            "timestamps": timestamps,
            "D_open": D_open,
            "D_closed": D_closed,
            "states": states,
            "threshold_open": self.THRESHOLD_OPEN,
            "threshold_closed": self.THRESHOLD_CLOSED,
            "mean_open": getattr(self, 'mean_open', None),
            "std_open": getattr(self, 'std_open', None),
            "mean_closed": getattr(self, 'mean_closed', None),
            "std_closed": getattr(self, 'std_closed', None),
            "s_open": self.s_open,
            "s_closed": self.s_closed,
            "state_transitions": self.state_transitions,
        }

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
        computation_start = time.perf_counter()
        current_time = time.time()
        time_since_last_ms = (current_time - self.last_dtw_time) * 1000

        # NOTE: Filtering now happens at source level in MuoviWidget.extract_emg_data()
        # The buffer already contains filtered data when config.ENABLE_FILTERING is True
        emg_buffer = self.emg_rt_buffer.copy()

        # --- Feature extraction timing ---
        feature_start = time.perf_counter()
        windowed_emg_buffer = sliding_window(emg_buffer, self.window_length, self.increment)

        feature_info = FEATURES[self.feature_name]
        feature_fn = feature_info["function"]
        features_emg_buffer = feature_fn(windowed_emg_buffer)
        feature_time_ms = (time.perf_counter() - feature_start) * 1000

        # --- DTW computation timing ---
        dtw_start = time.perf_counter()

        # DTW distances (use model's active_channels and distance_aggregation)
        # Only compute distance to OPPOSITE state templates (efficient)
        timestamp = time.time()

        if self.current_state == "OPEN":
            # Check if should switch to CLOSED
            D_closed = compute_distance_from_training_set_online(
                features_emg_buffer,
                self.templates_closed,
                active_channels=self.active_channels,
                distance_aggregation=self.distance_aggregation
            )
            if D_closed < self.THRESHOLD_CLOSED:
                triggered_state = "CLOSED"
            else:
                triggered_state = "OPEN"

            self.dtw_distances.append(("closed", D_closed, self.THRESHOLD_CLOSED))
            # Store in history: (timestamp, D_open=None, D_closed, state)
            self.distance_history.append((timestamp, None, D_closed, self.current_state))
            self._last_distance = D_closed
            self._last_threshold = self.THRESHOLD_CLOSED

        elif self.current_state == "CLOSED":
            # Check if should switch to OPEN
            D_open = compute_distance_from_training_set_online(
                features_emg_buffer,
                self.templates_open,
                active_channels=self.active_channels,
                distance_aggregation=self.distance_aggregation
            )
            if D_open < self.THRESHOLD_OPEN:
                triggered_state = "OPEN"
            else:
                triggered_state = "CLOSED"

            self.dtw_distances.append(("open", D_open, self.THRESHOLD_OPEN))
            # Store in history: (timestamp, D_open, D_closed=None, state)
            self.distance_history.append((timestamp, D_open, None, self.current_state))
            self._last_distance = D_open
            self._last_threshold = self.THRESHOLD_OPEN

        dtw_time_ms = (time.perf_counter() - dtw_start) * 1000

        # State machine logic

        # Keep dtw_distances buffer bounded (for debugging/logging)
        if len(self.dtw_distances) >= 100:
            self.dtw_distances.pop(0)

        previous_state = self.current_state

        # Use model's smoothing method (defaults to config if not set)
        smoothing = getattr(self, 'smoothing_method', config.POST_PREDICTION_SMOOTHING)

        if smoothing == "NONE":
            # No smoothing - directly use triggered state
            self.current_state = triggered_state

        elif smoothing == "MAJORITY VOTE":
            # Sliding window majority vote
            self.last_predictions.append(triggered_state)

            # Keep window at fixed size (sliding window)
            if len(self.last_predictions) > self.window_majority_length:
                self.last_predictions.pop(0)

            # Once we have enough predictions, use majority vote
            if len(self.last_predictions) >= self.window_majority_length:
                closed_count = self.last_predictions.count("CLOSED")
                open_count = self.last_predictions.count("OPEN")
                if closed_count > open_count:
                    self.current_state = "CLOSED"
                elif open_count > closed_count:
                    self.current_state = "OPEN"
                # If tied, keep current state (no change)

        elif smoothing == "5 CONSECUTIVE":
            # Require N consecutive identical predictions to switch state
            self.last_predictions.append(triggered_state)

            # Keep window at fixed size
            if len(self.last_predictions) > self.consecutive_required:
                self.last_predictions.pop(0)

            # Check if all predictions in window are the same AND different from current
            if len(self.last_predictions) >= self.consecutive_required:
                if all(p == self.last_predictions[0] for p in self.last_predictions):
                    new_state = self.last_predictions[0]
                    if new_state != self.current_state:
                        self.current_state = new_state
                        # Clear buffer after state transition to require fresh consecutive predictions
                        self.last_predictions = []

        # Update timing
        current_time = time.time()
        self.last_dtw_time = current_time

        # Print transition with clear, compact message
        if previous_state != self.current_state:
            transition_time = time.time()
            # Store state transition for plotting
            self.state_transitions.append((transition_time, previous_state, self.current_state))

            time_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            if self.current_state == "OPEN":
                print(f"\n{'='*50}")
                print(f"  >>> HAND OPENED <<<  [{time_str}]")
                print(f"  Distance: {self._last_distance:.4f} < Threshold: {self._last_threshold:.4f}")
                print(f"{'='*50}\n")
            else:
                print(f"\n{'='*50}")
                print(f"  >>> HAND CLOSED <<<  [{time_str}]")
                print(f"  Distance: {self._last_distance:.4f} < Threshold: {self._last_threshold:.4f}")
                print(f"{'='*50}\n")

        # Update timing stats
        total_time_ms = (time.perf_counter() - computation_start) * 1000

        # Store timing info for statistics
        if not hasattr(self, 'timing_history'):
            self.timing_history = {
                'interval': [],
                'feature': [],
                'dtw': [],
                'total': []
            }

        self.timing_history['interval'].append(time_since_last_ms)
        self.timing_history['feature'].append(feature_time_ms)
        self.timing_history['dtw'].append(dtw_time_ms)
        self.timing_history['total'].append(total_time_ms)

        # Keep timing history bounded
        max_history = 100
        for key in self.timing_history:
            if len(self.timing_history[key]) > max_history:
                self.timing_history[key].pop(0)

        self.dtw_times.append(total_time_ms)
        if len(self.dtw_times) >= 100:
            self.dtw_times.pop(0)

        self.dtw_count += 1

        # Print timing every computation (for real-time monitoring)
        # Format: [count] Interval: Xms | Feature: Xms | DTW: Xms | Total: Xms | State | D vs T
        interval_status = "OK" if 45 <= time_since_last_ms <= 55 else "DRIFT"
        print(f"[{self.dtw_count:4d}] Δt:{time_since_last_ms:5.1f}ms ({interval_status}) | "
              f"Feat:{feature_time_ms:4.1f}ms | DTW:{dtw_time_ms:5.1f}ms | "
              f"Total:{total_time_ms:5.1f}ms | {self.current_state:6s} | "
              f"D={self._last_distance:.4f} T={self._last_threshold:.4f}")

        # Print summary statistics every 100 computations (every ~5 seconds)
        if self.dtw_count % 100 == 0:
            avg_interval = np.mean(self.timing_history['interval'])
            avg_feature = np.mean(self.timing_history['feature'])
            avg_dtw = np.mean(self.timing_history['dtw'])
            avg_total = np.mean(self.timing_history['total'])

            print(f"\n{'='*70}")
            print(f"  TIMING SUMMARY (last 100 computations)")
            print(f"  Interval: {avg_interval:.1f}ms avg (target: 50ms)")
            print(f"  Feature extraction: {avg_feature:.2f}ms avg")
            print(f"  DTW computation: {avg_dtw:.2f}ms avg")
            print(f"  Total computation: {avg_total:.2f}ms avg")
            print(f"  Headroom: {50 - avg_total:.1f}ms (time available before next update)")
            print(f"{'='*70}\n")

        # Prepare result with extended info for Unity
        state_value = 1.0 if self.current_state == "CLOSED" else 0.0
        self._last_result = {
            'state': state_value,
            'distance': self._last_distance,
            'threshold': self._last_threshold,
            'state_name': self.current_state
        }

        return state_value
