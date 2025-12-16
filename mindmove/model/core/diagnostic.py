"""
Lightweight diagnostic model for testing real-time performance
without expensive DTW computations.
"""

from __future__ import annotations
from typing import Any, List
import numpy as np
import time

from mindmove.config import config
from mindmove.model.core.features.features_registry import FEATURES
from mindmove.model.core.windowing import sliding_window
from mindmove.model.core.filtering import apply_rtfiltering


class DiagnosticModel:

    def __init__(self) -> None:

        # sampling config
        self.FSAMP = config.FSAMP
        self.num_channels = config.num_channels
        self.dead_channels = config.dead_channels
        self.active_channels = config.active_channels

        # buffer config
        self.buffer_length_s = config.template_duration
        self.buffer_length = self.FSAMP * self.buffer_length_s

        # update time (how often should processing occur)
        self.increment_dtw_s = config.increment_dtw
        self.increment_dtw = int(self.FSAMP * self.increment_dtw_s)

        # feature config
        self.window_length = config.window_length
        self.increment = config.increment

        # initialize buffer
        self.emg_rt_buffer = np.zeros((self.num_channels, self.buffer_length))

        # track samples
        self.samples_since_last_computation = 0

        # feature choice
        self.feature_name = "rms" 

        # timing tracking
        self.last_computation_time = time.time()
        self.computation_count = 0
        self.computation_times = []
        self.time_between_computations = []

        # feature storage
        self.latest_features = None
        self.feature_history = []
        self.max_history = 20  # Keep last 20 feature vectors

        print("="*60)
        print("DIAGNOSTIC MODEL INITIALIZED")
        print("="*60)
        print(f"Buffer length: {self.buffer_length_s}s ({self.buffer_length} samples)")
        print(f"Update interval: {self.increment_dtw_s}s ({self.increment_dtw} samples)")
        print(f"Feature window: {self.window_length} samples")
        print(f"Feature increment: {self.increment} samples")
        print(f"Active channels: {len(self.active_channels)}/{self.num_channels}")
        print(f"Feature: {self.feature_name}")
        print("="*60)

    def _update_buffer(self, new_samples: np.ndarray) -> bool:
        """
        Update the real-time buffer with new incoming samples.
        Args:
            new_samples (np.ndarray): New incoming EMG samples of shape (n_channels, new_samples).
        Returns:
            bool: True if enough samples are available for DTW computation, False otherwise.
        """
        n_new_samples = new_samples.shape[1]

        # Shift buffer to the left and append new samples
        if n_new_samples >= self.buffer_length:
            self.emg_rt_buffer = new_samples[:, -self.buffer_length:].copy()
            self.samples_since_last_dtw = self.buffer_length
        else:
            # roll buffer and add new samples
            self.emg_rt_buffer = np.roll(self.emg_rt_buffer, -n_new_samples, axis=1)
            self.emg_rt_buffer[:, -n_new_samples :] = new_samples
            self.samples_since_last_dtw += n_new_samples

        should_compute = self.samples_since_last_dtw >= self.increment_dtw

        if should_compute:
            self.samples_since_last_dtw = 0 # ma non salta dei campioni ??????
        
        return should_compute
    
    def predict(self, emg_data: np.ndarray) -> List[Any]:
        """
        Process new EMG data and extract features for visualization.
        Args:
            x: New EMG data of shape (n_channels, n_samples)
        Returns:
            List[float]: Dummy prediction (always 0.0)
        """

        # update buffer
        should_compute = self._update_buffer(emg_data)

        if not should_compute:
            return [0.0]  # No new computation needed
        
        # timing tracking
        computation_start = time.perf_counter()
        current_time = time.time()
        
        # time since last computation
        time_since_last = current_time - self.last_computation_time
        self.time_between_computations.append(time_since_last*1000)  # in ms

        # Apply real-time filtering
        filter_start = time.perf_counter()
        if config.ENABLE_FILTERING:
            emg_buffer = apply_rtfiltering(self.emg_rt_buffer)
        else:
            emg_buffer = self.emg_rt_buffer.copy()
        filter_time = (time.perf_counter() - filter_start) * 1000

        # extract features using sliding window
        

        window_start = time.perf_counter()
        windowed_emg_buffer = sliding_window(emg_buffer, self.window_length, self.increment)
        window_time = (time.perf_counter() - window_start) * 1000

        
        feature_start = time.perf_counter()
        feature_info = FEATURES[self.feature_name]
        feature_fn = feature_info["function"]
        features = feature_fn(windowed_emg_buffer)
        feature_time = (time.perf_counter() - feature_start) * 1000

        # store features
        self.latest_features = features
        self.feature_history.append(features.copy())
        if len(self.feature_history) > self.max_history:
            self.feature_history.pop(0)

        # end timing
        total_computation_time = (time.perf_counter() - computation_start) * 1000
        self.computation_times.append(total_computation_time)

        # keep only last measurements
        if len(self.computation_times) > 100:
            self.computation_times.pop(0)
        if len(self.time_between_computations) > 100:
            self.time_between_computations.pop(0)
        
        # logging
        self.computation_count += 1

        # printing
        # Print every computation
        print(f"\n{'='*70}")
        print(f"COMPUTATION #{self.computation_count}")
        print(f"{'='*70}")
        print(f"⏱️  Time since last: {time_since_last*1000:.1f} ms")
        print(f"⚡ Filtering:       {filter_time:.2f} ms")
        print(f"🪟 Windowing:       {window_time:.2f} ms")
        print(f"📊 Features:        {feature_time:.2f} ms")
        print(f"🔧 Total:           {total_computation_time:.2f} ms")
        print(f"-"*70)
        print(f"📐 Feature shape:   {features.shape}")
        print(f"   (windows × channels)")
        print(f"-"*70)
        
        # Print feature statistics for active channels only
        active_features = features[:, self.active_channels]
        print(f"📈 Feature Stats (active channels only):")
        print(f"   Mean:  {np.mean(active_features):.4f}")
        print(f"   Std:   {np.std(active_features):.4f}")
        print(f"   Min:   {np.min(active_features):.4f}")
        print(f"   Max:   {np.max(active_features):.4f}")
        
        # Print per-channel feature means (first 5 active channels)
        print(f"-"*70)
        print(f"🎯 Feature means (first 5 active channels):")
        for i, ch in enumerate(self.active_channels[:5]):
            ch_mean = np.mean(features[:, ch])
            print(f"   Ch {ch:2d}: {ch_mean:.4f}")
        
        # Print statistics every 20 computations
        if self.computation_count % 20 == 0:
            avg_total = np.mean(self.computation_times)
            avg_between = np.mean(self.time_between_computations)
            
            print(f"\n{'#'*70}")
            print(f"📊 STATISTICS (last {len(self.computation_times)} computations)")
            print(f"{'#'*70}")
            print(f"⚡ Avg computation time: {avg_total:.2f} ms")
            print(f"⏱️  Avg time between:     {avg_between:.1f} ms")
            print(f"🎯 Expected interval:     {self.increment_dtw_s*1000:.1f} ms")
            print(f"{'#'*70}\n")
        
        self.last_computation_time = current_time
        
        return [0.0]  # Dummy prediction
    
    def get_latest_features(self) -> np.ndarray:
        """Get the most recently computed features."""
        return self.latest_features
    
    def get_feature_history(self) -> List[np.ndarray]:
        """Get history of feature vectors."""
        return self.feature_history
    
    def print_summary(self):
        """Print a summary of all timing statistics."""
        if len(self.computation_times) == 0:
            print("No computations performed yet.")
            return
        
        print(f"\n{'='*70}")
        print(f"DIAGNOSTIC MODEL SUMMARY")
        print(f"{'='*70}")
        print(f"Total computations: {self.computation_count}")
        print(f"-"*70)
        print(f"Computation Time Statistics:")
        print(f"  Mean:   {np.mean(self.computation_times):.2f} ms")
        print(f"  Median: {np.median(self.computation_times):.2f} ms")
        print(f"  Std:    {np.std(self.computation_times):.2f} ms")
        print(f"  Min:    {np.min(self.computation_times):.2f} ms")
        print(f"  Max:    {np.max(self.computation_times):.2f} ms")
        print(f"-"*70)
        print(f"Time Between Computations:")
        print(f"  Mean:   {np.mean(self.time_between_computations):.1f} ms")
        print(f"  Median: {np.median(self.time_between_computations):.1f} ms")
        print(f"  Std:    {np.std(self.time_between_computations):.1f} ms")
        print(f"  Min:    {np.min(self.time_between_computations):.1f} ms")
        print(f"  Max:    {np.max(self.time_between_computations):.1f} ms")
        print(f"-"*70)
        print(f"Expected interval: {self.increment_dtw_s*1000:.1f} ms")
        print(f"Actual avg:        {np.mean(self.time_between_computations):.1f} ms")
        print(f"Deviation:         {np.mean(self.time_between_computations) - self.increment_dtw_s*1000:.1f} ms")
        print(f"{'='*70}\n")