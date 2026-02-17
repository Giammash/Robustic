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
from mindmove.model.core.algorithm import compute_distance_from_training_set_online, compute_spatial_similarity

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
        # Per-class channels (default: same as global)
        self.active_channels_open = config.active_channels
        self.active_channels_closed = config.active_channels

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

        # Refractory period variables
        self.refractory_period_s = 1.0  # seconds (default)
        self.last_state_change_time = 0.0  # timestamp of last state change
        self.in_refractory = False  # True if currently in refractory period

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

        # Cross-class statistics and threshold presets (loaded from model)
        self.mean_cross = None
        self.std_cross = None
        self.threshold_presets = None  # dict of preset_key -> {s_open, s_closed, threshold_open, threshold_closed, name, description}

        # Spatial correction (consistency-weighted spatial match)
        self.use_spatial_correction = False  # Toggled from UI
        self.spatial_threshold = 0.5  # Default threshold for spatial similarity
        self.spatial_ref_open = None  # dict: ref_profile, weights, consistency
        self.spatial_ref_closed = None  # dict: ref_profile, weights, consistency
        self.spatial_similarity_history = []  # (timestamp, sim_open, sim_closed)



    def _update_buffer(self, new_samples: np.ndarray) -> bool:
        """
        Update the real-time buffer with new incoming samples.
        Args:
            new_samples (np.ndarray): New incoming EMG samples of shape (n_channels, new_samples).
        Returns:
            bool: True if enough samples are available for DTW computation, False otherwise.
        """
        # new_samples shape: (n_channels, n_new_samples)
        incoming_channels = new_samples.shape[0]
        n_new_samples = new_samples.shape[1]

        # Check if buffer needs to be resized for different channel count
        # This handles switching between monopolar (32ch) and SD (16ch) modes
        if self.emg_rt_buffer.shape[0] != incoming_channels:
            print(f"[MODEL] Buffer channel mismatch: buffer={self.emg_rt_buffer.shape[0]}, "
                  f"incoming={incoming_channels}. Reinitializing buffer.")
            self.emg_rt_buffer = np.zeros((incoming_channels, self.buffer_length))
            self.num_channels = incoming_channels
            # Recompute active_channels for new channel count
            # Filter dead_channels to only include valid indices for current channel count
            valid_dead = [ch for ch in self.dead_channels if ch < incoming_channels]
            self.active_channels = [i for i in range(incoming_channels) if i not in valid_dead]
            self.samples_since_last_dtw = 0

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
        # Thresholds loaded directly (may be mid-gap or mean+s*std depending on model)
        threshold_multiplier = data.get("threshold_multiplier", 1.0)
        self.THRESHOLD_OPEN = data["threshold_base_open"] * threshold_multiplier
        self.mean_open = data["mean_open"]
        self.std_open = data["std_open"]
        self.THRESHOLD_CLOSED = data["threshold_base_closed"] * threshold_multiplier
        self.mean_closed = data["mean_closed"]
        self.std_closed = data["std_closed"]
        # Derive s values from loaded thresholds for slider consistency
        self.s_open = ((self.THRESHOLD_OPEN - self.mean_open) / self.std_open
                       if self.std_open > 0 else 1.0)
        self.s_closed = ((self.THRESHOLD_CLOSED - self.mean_closed) / self.std_closed
                         if self.std_closed > 0 else 1.0)
        self.feature_name = data["feature_name"]

        # Load new model settings (with defaults for backwards compatibility)
        # dead_channels stored as 0-indexed
        self.dead_channels = data.get("dead_channels", config.dead_channels)

        # Detect channel count from templates (more reliable than config)
        # Templates are stored as feature matrices: (n_windows, n_channels)
        if self.templates_open and len(self.templates_open) > 0:
            template_channels = self.templates_open[0].shape[1]  # n_channels from first template
            self.num_channels = template_channels
            print(f"  - Detected channel count from templates: {template_channels}")
        else:
            # Fallback to config if no templates
            self.num_channels = config.num_channels

        # Compute active channels from dead channels, ensuring they don't exceed num_channels
        valid_dead = [ch for ch in self.dead_channels if ch < self.num_channels]
        self.active_channels = [i for i in range(self.num_channels) if i not in valid_dead]

        # Per-class active channels (spatial separation)
        self.active_channels_open = data.get("active_channels_open", None)
        self.active_channels_closed = data.get("active_channels_closed", None)
        if self.active_channels_open is None or self.active_channels_closed is None:
            # Backward compat: use global active_channels for both
            self.active_channels_open = self.active_channels
            self.active_channels_closed = self.active_channels
            print(f"  - Channel mode: Global ({len(self.active_channels)} channels)")
        else:
            print(f"  - Channel mode: Per-class")
            print(f"    CLOSED channels (0-indexed): {self.active_channels_closed}")
            print(f"    OPEN channels (0-indexed): {self.active_channels_open}")

        # Reinitialize buffer with correct channel count
        self.emg_rt_buffer = np.zeros((self.num_channels, self.buffer_length))

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

        # Load cross-class statistics and threshold presets (for intelligent threshold tuning)
        self.mean_cross = data.get("mean_cross", None)
        self.std_cross = data.get("std_cross", None)
        self.threshold_presets = data.get("threshold_presets", None)

        # Spatial correction profiles (consistency-weighted spatial match)
        spatial_data = data.get("spatial_profiles", None)
        if spatial_data is not None:
            self.spatial_ref_open = spatial_data.get("open", None)
            self.spatial_ref_closed = spatial_data.get("closed", None)
            self.spatial_threshold = spatial_data.get("threshold", 0.5)
            print(f"  - Spatial correction: available (threshold={self.spatial_threshold:.2f})")
        else:
            self.spatial_ref_open = None
            self.spatial_ref_closed = None
            print(f"  - Spatial correction: not available (legacy model)")

        # Differential mode: load from model or infer from channel count
        self.differential_mode = data.get("differential_mode", None)
        if self.differential_mode is None:
            # Infer from detected channel count (templates already set self.num_channels above)
            self.differential_mode = self.num_channels <= 16
            print(f"  - Differential mode (inferred from {self.num_channels} channels): "
                  f"{'SD' if self.differential_mode else 'MP'}")
        else:
            print(f"  - Differential mode (from model): "
                  f"{'SD' if self.differential_mode else 'MP'}")

        # pass
        print(f"Model loaded from: {model_path}")
        print(f"  - OPEN templates: {len(self.templates_open)}")
        print(f"  - CLOSED templates: {len(self.templates_closed)}")
        print(f"  - OPEN threshold (mid-gap): {self.THRESHOLD_OPEN:.4f}  (s={self.s_open:.2f})")
        print(f"  - CLOSED threshold (mid-gap): {self.THRESHOLD_CLOSED:.4f}  (s={self.s_closed:.2f})")
        print(f"  - Feature: {self.feature_name}")
        print(f"  - Dead channels (0-indexed): {self.dead_channels}")
        print(f"  - Active channels: {len(self.active_channels)}")
        print(f"  - Distance aggregation: {self.distance_aggregation}")
        print(f"  - Smoothing method: {self.smoothing_method}")

        # Print cross-class statistics if available
        if self.mean_cross is not None:
            print(f"  - Cross-class distance: mean={self.mean_cross:.4f}, std={self.std_cross:.4f}")
        if self.threshold_presets is not None:
            print(f"  - Threshold presets available: {list(self.threshold_presets.keys())}")

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

    def apply_threshold_preset(self, preset_key: str) -> bool:
        """
        Apply a threshold preset by key.

        Presets are computed during training based on intra-class and cross-class
        distance statistics. Available presets:
        - "current": Standard method (mean + 1*std within class)
        - "cross_class": Midpoint between intra-class and cross-class means
        - "conservative": Strict threshold to prevent unwanted state changes
        - "safety_margin": 50% between intra-class mean and cross-class mean

        Args:
            preset_key: The key of the preset to apply (e.g., "conservative").

        Returns:
            True if the preset was applied successfully, False otherwise.
        """
        if self.threshold_presets is None:
            print(f"[THRESHOLD] No presets available (legacy model)")
            return False

        if preset_key not in self.threshold_presets:
            print(f"[THRESHOLD] Unknown preset: {preset_key}")
            print(f"[THRESHOLD] Available presets: {list(self.threshold_presets.keys())}")
            return False

        preset = self.threshold_presets[preset_key]
        s_open = preset["s_open"]
        s_closed = preset["s_closed"]
        preset_name = preset.get("name", preset_key)

        # Apply using s values to ensure consistency with slider UI
        self.update_threshold_open(s_open)
        self.update_threshold_closed(s_closed)

        print(f"\n{'='*50}")
        print(f"  Applied preset: {preset_name}")
        print(f"  OPEN:   s={s_open:.2f} → threshold={self.THRESHOLD_OPEN:.4f}")
        print(f"  CLOSED: s={s_closed:.2f} → threshold={self.THRESHOLD_CLOSED:.4f}")
        print(f"{'='*50}\n")

        return True

    def get_available_presets(self) -> dict:
        """
        Return available threshold presets.

        Returns:
            Dictionary of preset_key -> preset_info, or None if no presets available.
            Each preset_info contains: s_open, s_closed, threshold_open, threshold_closed,
            name, and description.
        """
        return self.threshold_presets

    def has_threshold_presets(self) -> bool:
        """Check if this model has threshold presets available."""
        return self.threshold_presets is not None and len(self.threshold_presets) > 0

    def reset_history(self) -> None:
        """Reset all history buffers for a new acquisition session."""
        self.distance_history = []
        self.state_transitions = []
        self.dtw_distances = []
        self.dtw_times = []
        self.last_predictions = []
        self.dtw_count = 0
        self.last_dtw_time = time.time()
        self.last_state_change_time = 0.0  # Reset refractory timer
        self.in_refractory = False
        self.spatial_similarity_history = []
        print("[MODEL] History reset for new session")

    def set_refractory_period(self, period_s: float) -> None:
        """Set the refractory period in seconds."""
        self.refractory_period_s = max(0.0, period_s)
        print(f"[MODEL] Refractory period set to {self.refractory_period_s:.2f}s")

    def get_refractory_period(self) -> float:
        """Get the current refractory period in seconds."""
        return self.refractory_period_s

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

        # Per-step thresholds (backwards compatible: old entries may have only 4 elements)
        thresholds_open_over_time = [h[4] if len(h) > 4 else self.THRESHOLD_OPEN for h in self.distance_history]
        thresholds_closed_over_time = [h[5] if len(h) > 5 else self.THRESHOLD_CLOSED for h in self.distance_history]

        # Normalize timestamps to start from 0
        t0 = timestamps[0] if timestamps else 0
        timestamps = [t - t0 for t in timestamps]

        # Build spatial similarity arrays (aligned with distance_history timestamps)
        spatial_sim_open = None
        spatial_sim_closed = None
        if self.spatial_similarity_history:
            spatial_sim_open = [h[1] for h in self.spatial_similarity_history]
            spatial_sim_closed = [h[2] for h in self.spatial_similarity_history]

        return {
            "timestamps": timestamps,
            "D_open": D_open,
            "D_closed": D_closed,
            "states": states,
            "threshold_open": self.THRESHOLD_OPEN,
            "threshold_closed": self.THRESHOLD_CLOSED,
            "thresholds_open_over_time": thresholds_open_over_time,
            "thresholds_closed_over_time": thresholds_closed_over_time,
            "mean_open": getattr(self, 'mean_open', None),
            "std_open": getattr(self, 'std_open', None),
            "mean_closed": getattr(self, 'mean_closed', None),
            "std_closed": getattr(self, 'std_closed', None),
            "s_open": self.s_open,
            "s_closed": self.s_closed,
            "state_transitions": self.state_transitions,
            "spatial_sim_open": spatial_sim_open,
            "spatial_sim_closed": spatial_sim_closed,
            "spatial_threshold": self.spatial_threshold if self.use_spatial_correction else None,
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
            D_closed, all_distances_closed = compute_distance_from_training_set_online(
                features_emg_buffer,
                self.templates_closed,
                active_channels=self.active_channels_closed,
                distance_aggregation=self.distance_aggregation,
                return_all_distances=True
            )
            if D_closed < self.THRESHOLD_CLOSED:
                triggered_state = "CLOSED"
            else:
                triggered_state = "OPEN"

            self.dtw_distances.append(("closed", D_closed, self.THRESHOLD_CLOSED))
            # Store in history: (timestamp, D_open=None, D_closed, state, threshold_open, threshold_closed)
            self.distance_history.append((timestamp, None, D_closed, self.current_state, self.THRESHOLD_OPEN, self.THRESHOLD_CLOSED))
            self._last_distance = D_closed
            self._last_threshold = self.THRESHOLD_CLOSED
            self._last_all_distances = all_distances_closed
            self._last_template_class = "CLOSED"

        elif self.current_state == "CLOSED":
            # Check if should switch to OPEN
            D_open, all_distances_open = compute_distance_from_training_set_online(
                features_emg_buffer,
                self.templates_open,
                active_channels=self.active_channels_open,
                distance_aggregation=self.distance_aggregation,
                return_all_distances=True
            )
            if D_open < self.THRESHOLD_OPEN:
                triggered_state = "OPEN"
            else:
                triggered_state = "CLOSED"

            self.dtw_distances.append(("open", D_open, self.THRESHOLD_OPEN))
            # Store in history: (timestamp, D_open, D_closed=None, state, threshold_open, threshold_closed)
            self.distance_history.append((timestamp, D_open, None, self.current_state, self.THRESHOLD_OPEN, self.THRESHOLD_CLOSED))
            self._last_distance = D_open
            self._last_threshold = self.THRESHOLD_OPEN
            self._last_all_distances = all_distances_open
            self._last_template_class = "OPEN"

        dtw_time_ms = (time.perf_counter() - dtw_start) * 1000

        # --- Spatial similarity computation (every tick, independent of DTW decision) ---
        sim_open = None
        sim_closed = None
        spatial_blocked = False

        if self.use_spatial_correction and self.spatial_ref_open is not None and self.spatial_ref_closed is not None:
            if self.current_state == "OPEN":
                # Checking transition to CLOSED — compute similarity to CLOSED reference
                sim_closed = compute_spatial_similarity(
                    emg_buffer, self.spatial_ref_closed["ref_profile"], self.spatial_ref_closed["weights"]
                )
                if triggered_state == "CLOSED" and sim_closed < self.spatial_threshold:
                    spatial_blocked = True
                    triggered_state = "OPEN"  # Block transition
            elif self.current_state == "CLOSED":
                # Checking transition to OPEN — compute similarity to OPEN reference
                sim_open = compute_spatial_similarity(
                    emg_buffer, self.spatial_ref_open["ref_profile"], self.spatial_ref_open["weights"]
                )
                if triggered_state == "OPEN" and sim_open < self.spatial_threshold:
                    spatial_blocked = True
                    triggered_state = "CLOSED"  # Block transition

            self.spatial_similarity_history.append((timestamp, sim_open, sim_closed))
            # Keep bounded
            if len(self.spatial_similarity_history) > 10000:
                self.spatial_similarity_history = self.spatial_similarity_history[-5000:]

        # State machine logic

        # Keep dtw_distances buffer bounded (for debugging/logging)
        if len(self.dtw_distances) >= 100:
            self.dtw_distances.pop(0)

        previous_state = self.current_state

        # --- Refractory Period Check ---
        # Check if we're still in refractory period after last state change
        time_since_last_change = current_time - self.last_state_change_time
        was_in_refractory = self.in_refractory

        if self.in_refractory:
            if time_since_last_change >= self.refractory_period_s:
                # Refractory period ended - reset smoothing buffer
                self.in_refractory = False
                self.last_predictions = []  # Start fresh for smoothing
                # print(f"[REFRACTORY] Ended after {time_since_last_change:.2f}s - smoothing buffer reset")
            else:
                # Still in refractory - skip smoothing, keep current state
                # Don't accumulate predictions during refractory
                pass

        # Only apply smoothing logic if NOT in refractory period
        if not self.in_refractory:
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

            # --- Start Refractory Period ---
            self.last_state_change_time = transition_time
            self.in_refractory = True
            self.last_predictions = []  # Clear smoothing buffer

            time_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            if self.current_state == "OPEN":
                print(f"\n{'='*50}")
                print(f"  >>> HAND OPENED <<<  [{time_str}]")
                print(f"  Distance: {self._last_distance:.4f} < Threshold: {self._last_threshold:.4f}")
                if self.use_spatial_correction and sim_open is not None:
                    print(f"  Spatial sim (OPEN): {sim_open:.4f} >= {self.spatial_threshold:.4f}")
                print(f"  Refractory: {self.refractory_period_s:.1f}s")
            else:
                print(f"\n{'='*50}")
                print(f"  >>> HAND CLOSED <<<  [{time_str}]")
                print(f"  Distance: {self._last_distance:.4f} < Threshold: {self._last_threshold:.4f}")
                if self.use_spatial_correction and sim_closed is not None:
                    print(f"  Spatial sim (CLOSED): {sim_closed:.4f} >= {self.spatial_threshold:.4f}")
                print(f"  Refractory: {self.refractory_period_s:.1f}s")

            # Print top 10 closest templates
            if hasattr(self, '_last_all_distances') and self._last_all_distances is not None:
                sorted_idx = np.argsort(self._last_all_distances)
                n_show = min(10, len(sorted_idx))
                print(f"  Top {n_show} closest {self._last_template_class} templates:")
                for rank, idx in enumerate(sorted_idx[:n_show], 1):
                    print(f"    {rank:2d}. Template {idx+1:2d}: {self._last_all_distances[idx]:.4f}")
            print(f"{'='*50}\n")

        # Print when spatial correction blocks a transition
        if spatial_blocked:
            target_class = "CLOSED" if self.current_state == "OPEN" else "OPEN"
            sim_val = sim_closed if target_class == "CLOSED" else sim_open
            print(f"[SPATIAL] Blocked {target_class} transition: "
                  f"sim={sim_val:.4f} < {self.spatial_threshold:.4f}")

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

        # Print status every 40 computations (every ~2 seconds)
        if self.dtw_count % 40 == 0:
            elapsed_s = self.dtw_count * 0.05
            spatial_str = ""
            if self.use_spatial_correction:
                sim_val = sim_closed if sim_closed is not None else sim_open
                if sim_val is not None:
                    spatial_str = f" | S={sim_val:.4f}"
            print(f"[{elapsed_s:5.1f}s] {self.current_state:6s} | D={self._last_distance:.4f} T={self._last_threshold:.4f}{spatial_str}")

        # Print timing summary every 100 computations (every ~5 seconds)
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
