"""
Template Manager for MindMove.

Handles extraction, selection, and management of EMG templates from recordings.
Supports both "hold only" and "onset + hold" template types.

Supports four recording formats:
1. MindMove Virtual Hand format: {emg, kinematics, gt_mode: "virtual_hand", ...}
2. MindMove Keyboard format: {emg, gt, gt_mode: "keyboard", ...}
3. VHI format: {biosignal, ground_truth, biosignal_timings, ground_truth_timings, ...}
4. Legacy format: Separate EMG files + GT files with timestamp synchronization
"""

import os
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Literal, Tuple

from mindmove.config import config
from mindmove.model.core.windowing import sliding_window
from mindmove.model.core.features.features_registry import FEATURES


class TemplateManager:
    """Manages template extraction from recordings."""

    # Recording format constants
    FORMAT_MINDMOVE_VH = "mindmove_vh"  # emg, kinematics keys (virtual hand)
    FORMAT_MINDMOVE_KB = "mindmove_kb"  # emg, gt keys (keyboard)
    FORMAT_VHI = "vhi"  # biosignal, ground_truth keys
    FORMAT_LEGACY = "legacy"  # Separate EMG + GT files
    # Backwards compatibility alias
    FORMAT_MINDMOVE = "mindmove_vh"

    # Template duration options (in seconds)
    DURATION_OPTIONS = [0.5, 1.0, 1.5, 2.0]

    def __init__(self):
        self.templates: Dict[str, List[np.ndarray]] = {"open": [], "closed": []}
        self.all_activations: Dict[str, List[np.ndarray]] = {"open": [], "closed": []}
        self.activation_metadata: Dict[str, List[dict]] = {"open": [], "closed": []}
        self.template_type: Literal["hold_only", "onset_hold", "onset", "manual"] = "hold_only"
        self.selection_mode: Literal["manual", "auto", "first_n"] = "manual"

        # Configurable template duration (default 1.0 second)
        self._template_duration_s: float = 1.0

        # Context window for manual mode (seconds before GT=1 to include)
        self.manual_context_before_s: float = 2.0

        # Paths
        self.templates_base_path = "data"

    @property
    def template_duration_s(self) -> float:
        """Get template duration in seconds."""
        return self._template_duration_s

    @template_duration_s.setter
    def template_duration_s(self, value: float) -> None:
        """Set template duration in seconds."""
        self._template_duration_s = value

    @property
    def template_nsamp(self) -> int:
        """Get template duration in samples."""
        return int(self._template_duration_s * config.FSAMP)

    @staticmethod
    def detect_recording_format(recording: dict) -> str:
        """
        Auto-detect the format of a recording dictionary.

        Returns:
            One of: FORMAT_MINDMOVE_VH, FORMAT_MINDMOVE_KB, FORMAT_VHI
        """
        # Check gt_mode field if present
        gt_mode = recording.get("gt_mode", None)

        if "emg" in recording:
            if gt_mode == "keyboard" or "gt" in recording:
                # Keyboard mode: emg + gt (binary)
                return TemplateManager.FORMAT_MINDMOVE_KB
            elif "kinematics" in recording:
                # Virtual hand mode: emg + kinematics
                return TemplateManager.FORMAT_MINDMOVE_VH
        elif "biosignal" in recording and "ground_truth" in recording:
            return TemplateManager.FORMAT_VHI

        raise ValueError(f"Unknown recording format. Keys: {list(recording.keys())}")

    @staticmethod
    def normalize_recording(recording: dict) -> dict:
        """
        Convert any recording format to the standard MindMove format.

        Args:
            recording: Recording dict in any supported format

        Returns:
            Normalized dict with keys: emg, kinematics, label, task
            Note: For keyboard format, kinematics will be the binary GT reshaped to (1, n_samples)
        """
        fmt = TemplateManager.detect_recording_format(recording)

        if fmt == TemplateManager.FORMAT_MINDMOVE_VH:
            # Virtual hand format - already in correct format
            return recording

        elif fmt == TemplateManager.FORMAT_MINDMOVE_KB:
            # Keyboard format - gt is already at EMG sample rate
            gt = recording["gt"]
            # Reshape to (1, n_samples) to match expected kinematics shape
            kinematics = gt.reshape(1, -1) if gt.ndim == 1 else gt

            # Preserve the original gt_mode (could be "keyboard" or "guided_animation")
            original_gt_mode = recording.get("gt_mode", "keyboard")

            return {
                "emg": recording["emg"],
                "kinematics": kinematics,  # GT as "kinematics"
                "timings_emg": recording.get("timings_emg"),
                "timings_kinematics": recording.get("timings_emg"),  # Same as EMG
                "label": recording.get("label", "default"),
                "task": recording.get("task", "unknown"),
                "gt_mode": original_gt_mode,  # Preserve original mode
            }

        elif fmt == TemplateManager.FORMAT_VHI:
            # Convert VHI format
            # biosignal shape: (n_channels, samples_per_frame, n_frames)
            biosignal = recording["biosignal"]

            # Reshape: concatenate frames
            # From (38, 18, 3334) to (38, 18*3334) then take first 32 channels
            emg = np.concatenate(biosignal.T, axis=0).T
            emg = emg[:config.num_channels, :]

            return {
                "emg": emg,
                "kinematics": recording["ground_truth"],
                "timings_emg": recording.get("biosignal_timings"),
                "timings_kinematics": recording.get("ground_truth_timings"),
                "label": recording.get("recording_label", "default"),
                "task": recording.get("task", "unknown"),
            }

        else:
            raise ValueError(f"Unsupported format: {fmt}")

    @staticmethod
    def load_legacy_format(
        emg_folder: str,
        gt_folder: str
    ) -> List[dict]:
        """
        Load recordings from legacy format (separate EMG + GT files).

        The EMG and GT files are synchronized using timestamps in filenames.
        Supports multiple filename formats:
        - VHI_Recording_YYYYMMDD_HHMMSSfff_*.pkl
        - raw_xxx_YYYYMMDD_HHMMSSfff_*.pkl
        GT files contain: {start_datetime, key_events}

        Args:
            emg_folder: Path to folder containing EMG .pkl files
            gt_folder: Path to folder containing GT .pkl files

        Returns:
            List of normalized recording dicts
        """
        emg_files = sorted(f for f in os.listdir(emg_folder) if f.endswith('.pkl'))
        gt_files = sorted(f for f in os.listdir(gt_folder) if f.endswith('.pkl'))

        if len(emg_files) != len(gt_files):
            print(f"Warning: EMG files ({len(emg_files)}) != GT files ({len(gt_files)})")

        recordings = []

        for emg_filename, gt_filename in zip(emg_files, gt_files):
            try:
                # Parse EMG start time from filename
                base = os.path.basename(emg_filename)
                parts = base.split("_")

                # Handle different filename formats
                if base.startswith("VHI_Recording_"):
                    # VHI_Recording_YYYYMMDD_HHMMSSfff_task_label.pkl
                    date_str = parts[2]  # YYYYMMDD
                    time_str = parts[3]  # HHMMSSfff
                else:
                    # raw_xxx_YYYYMMDD_HHMMSSfff_xxx.pkl or similar
                    date_str = parts[2]  # YYYYMMDD
                    time_str = parts[3]  # HHMMSSfff

                emg_start = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S%f")

                # Load EMG
                emg_path = os.path.join(emg_folder, emg_filename)
                with open(emg_path, 'rb') as f:
                    emg_recording = pickle.load(f)

                # Pre-process EMG
                emg = emg_recording["biosignal"]
                emg = np.concatenate(emg.T, axis=0).T
                emg = emg[:config.num_channels, :]

                # Load GT
                gt_path = os.path.join(gt_folder, gt_filename)
                with open(gt_path, 'rb') as f:
                    gt = pickle.load(f)

                # Compute time offset and adjust GT events
                offset = (emg_start - gt['start_datetime']).total_seconds()
                adjusted_events = [(t - offset, state) for t, state in gt['key_events']]

                print(f"Loaded {emg_filename}")
                print(f"  EMG start: {emg_start}, GT start: {gt['start_datetime']}")
                print(f"  Offset: {offset:.3f}s, EMG shape: {emg.shape}")
                print(f"  Key events: {len(gt['key_events'])}")

                # Convert key_events to binary kinematics at EMG sample rate
                n_samples = emg.shape[1]
                binary_gt = TemplateManager._key_events_to_binary(adjusted_events, n_samples)

                # Create normalized recording dict
                recordings.append({
                    "emg": emg,
                    "kinematics": binary_gt.reshape(1, -1),  # (1, n_samples)
                    "timings_emg": None,
                    "timings_kinematics": None,
                    "label": "default",
                    "task": "legacy",
                    "_legacy_gt": gt,  # Keep original GT for reference
                    "_legacy_filename": emg_filename,
                    "_offset": offset,
                    "_adjusted_events": adjusted_events,
                })

            except Exception as e:
                print(f"Error loading {emg_filename}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return recordings

    @staticmethod
    def _key_events_to_binary(key_events: List[Tuple[float, int]], n_samples: int) -> np.ndarray:
        """
        Convert key_events list to binary vector at EMG sample rate.

        Args:
            key_events: List of (time_seconds, state) tuples
            n_samples: Number of EMG samples

        Returns:
            Binary numpy array of shape (n_samples,)
        """
        binary = np.zeros(n_samples, dtype=int)
        event_idx = 0
        current_state = 0

        for i in range(n_samples):
            t = i / config.FSAMP
            while event_idx < len(key_events) and key_events[event_idx][0] <= t:
                current_state = key_events[event_idx][1]
                event_idx += 1
            binary[i] = current_state

        return binary

    def set_template_type(self, include_onset: bool) -> None:
        """
        Set template cutting mode.

        Args:
            include_onset: If True, use "onset_hold" mode (start 0.2s before GT=1).
                          If False, use "hold_only" mode (skip 0.5s after GT=1).
        """
        self.template_type = "onset_hold" if include_onset else "hold_only"

    def extract_activations_from_recording(
        self,
        recording: dict,
        class_label: str,
        include_pre_activation: bool = False,
        recording_name: str = ""
    ) -> List[np.ndarray]:
        """
        Extract all activation segments from a recording.

        Works with recordings of any length - extracts all activations found.
        Automatically detects and normalizes recording format.

        Args:
            recording: Recording dict in any supported format
            class_label: "open" or "closed"
            include_pre_activation: If True, include ONSET_OFFSET_S before activation start
            recording_name: Name of the source recording file (for metadata tracking)

        Returns:
            List of EMG segments (each is n_channels x n_samples)
        """
        # Normalize recording to standard format
        normalized = self.normalize_recording(recording)

        emg = normalized["emg"]
        kinematics = normalized["kinematics"]
        gt_mode = normalized.get("gt_mode", None)

        # Determine how to process ground truth
        if gt_mode == "keyboard" or kinematics.shape[0] == 1:
            # Keyboard format or legacy format - kinematics is already binary at EMG rate
            gt_binary = kinematics.flatten().astype(int)
        else:
            # Virtual hand: Multi-dimensional kinematics - convert to binary
            gt_binary = self._convert_kinematics_to_binary(kinematics, emg.shape[1])

        # Extract activation segments
        segments = self._extract_activation_segments(
            emg,
            gt_binary,
            min_duration_s=config.MIN_ACTIVATION_DURATION_S,
            include_pre_activation=include_pre_activation
        )

        # Add to all_activations with metadata
        self.all_activations[class_label].extend(segments)
        for cycle_idx in range(len(segments)):
            self.activation_metadata[class_label].append({
                "recording_name": recording_name,
                "cycle_index": cycle_idx + 1,
                "total_cycles_in_recording": len(segments),
            })

        return segments

    def _convert_kinematics_to_binary(
        self,
        kinematics: np.ndarray,
        n_samples_emg: int
    ) -> np.ndarray:
        """
        Convert virtual hand kinematics to binary activation signal.

        Args:
            kinematics: (n_dims, n_samples) kinematic data from virtual hand
            n_samples_emg: Number of EMG samples to match

        Returns:
            Binary vector at EMG sampling rate (0=open, 1=closed)
        """
        # Use mean of joint values as activation measure
        # Higher values = more flexion = closed hand
        mean_activation = np.mean(kinematics, axis=0)

        # Threshold at midpoint between min and max
        threshold = (np.max(mean_activation) + np.min(mean_activation)) / 2
        binary_kin = (mean_activation > threshold).astype(int)

        # Upsample to EMG rate (typically 60Hz -> 2000Hz)
        from scipy.ndimage import zoom
        zoom_factor = n_samples_emg / len(binary_kin)
        binary_emg = zoom(binary_kin.astype(float), zoom_factor, order=0)
        binary_emg = (binary_emg > 0.5).astype(int)

        return binary_emg[:n_samples_emg]

    def _extract_activation_segments(
        self,
        emg: np.ndarray,
        gt_binary: np.ndarray,
        min_duration_s: float = 1.5,
        include_pre_activation: bool = False
    ) -> List[np.ndarray]:
        """
        Extract EMG segments where GT=1.

        Args:
            emg: (n_channels, n_samples)
            gt_binary: Binary ground truth at EMG sample rate
            min_duration_s: Minimum activation duration to keep
            include_pre_activation: If True, include pre-activation samples
                                   (uses manual_context_before_s for manual mode,
                                    ONSET_OFFSET_S otherwise)

        Returns:
            List of EMG segments
        """
        # Find rising/falling edges
        diffs = np.diff(gt_binary, prepend=gt_binary[0])
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        # Handle edge cases
        if len(ends) == 0 or (len(starts) > 0 and len(ends) > 0 and ends[0] < starts[0]):
            # GT starts with 1
            starts = np.insert(starts, 0, 0)
        if gt_binary[-1] == 1:
            ends = np.append(ends, len(gt_binary))

        # Ensure same length
        n_segments = min(len(starts), len(ends))
        starts = starts[:n_segments]
        ends = ends[:n_segments]

        # Extract segments
        segments = []
        min_samples = int(min_duration_s * config.FSAMP)

        # Determine pre-activation samples based on mode
        if include_pre_activation:
            if self.template_type == "manual":
                # Use larger context for manual mode
                pre_samples = int(self.manual_context_before_s * config.FSAMP)
            else:
                pre_samples = int(config.ONSET_OFFSET_S * config.FSAMP)
        else:
            pre_samples = 0

        for start, end in zip(starts, ends):
            duration = end - start
            if duration >= min_samples:
                # Include pre-activation samples if requested
                actual_start = max(0, start - pre_samples)
                segments.append(emg[:, actual_start:end])

        return segments

    def cut_template_from_activation(
        self,
        activation: np.ndarray,
        pre_activation_samples: int = 0,
        manual_start_idx: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Cut a template of configured duration from an activation segment.

        Args:
            activation: Raw activation segment (n_channels, n_samples)
            pre_activation_samples: Number of samples before GT=1 included in activation
            manual_start_idx: If provided, use this as the start index (for manual mode)

        Returns:
            Template (n_channels, template_nsamp) or None if too short
        """
        template_samples = self.template_nsamp

        if manual_start_idx is not None:
            # Manual mode: use the provided start index
            start_idx = manual_start_idx
        elif self.template_type == "hold_only":
            # Skip 0.5s after GT=1, take template
            # If pre_activation_samples > 0, we need to account for that
            start_idx = pre_activation_samples + int(config.HOLD_SKIP_S * config.FSAMP)
        elif self.template_type == "onset":
            # Start exactly at GT=1 (accounting for any pre-activation samples)
            start_idx = pre_activation_samples
        else:  # onset_hold
            # Start from the beginning (which includes 0.2s before GT=1)
            start_idx = 0

        end_idx = start_idx + template_samples

        # Check if we have enough samples
        if end_idx > activation.shape[1]:
            return None

        return activation[:, start_idx:end_idx]

    def process_activations_to_templates(self, class_label: str) -> int:
        """
        Process all activations to extract templates based on current template_type.

        Args:
            class_label: "open" or "closed"

        Returns:
            Number of valid templates extracted
        """
        templates = []
        pre_samples = int(config.ONSET_OFFSET_S * config.FSAMP)

        for activation in self.all_activations[class_label]:
            template = self.cut_template_from_activation(activation, pre_samples)
            if template is not None:
                templates.append(template)

        self.templates[class_label] = templates
        return len(templates)

    def select_templates_manual(self, indices: List[int], class_label: str) -> None:
        """
        Manually select which activations to use as templates.

        Args:
            indices: List of indices into all_activations to select
            class_label: "open" or "closed"
        """
        pre_samples = int(config.ONSET_OFFSET_S * config.FSAMP)
        selected = []

        for i in indices:
            if i < len(self.all_activations[class_label]):
                template = self.cut_template_from_activation(
                    self.all_activations[class_label][i],
                    pre_samples
                )
                if template is not None:
                    selected.append(template)

        self.templates[class_label] = selected

    def select_templates_auto(self, class_label: str, n: int = 20) -> None:
        """
        Auto-select n longest activations as templates.

        Args:
            class_label: "open" or "closed"
            n: Number of templates to select (default 20)
        """
        # Sort by duration (longest first)
        indexed_activations = [
            (i, act) for i, act in enumerate(self.all_activations[class_label])
        ]
        sorted_activations = sorted(
            indexed_activations,
            key=lambda x: x[1].shape[1],
            reverse=True
        )

        # Take top n
        indices = [i for i, _ in sorted_activations[:n]]
        self.select_templates_manual(indices, class_label)

    def select_templates_first_n(self, class_label: str, n: int = 20) -> None:
        """
        Simply take first n valid activations as templates.

        Args:
            class_label: "open" or "closed"
            n: Number of templates to select (default 20)
        """
        indices = list(range(min(n, len(self.all_activations[class_label]))))
        self.select_templates_manual(indices, class_label)

    def save_templates(
        self,
        class_label: str,
        template_set_name: Optional[str] = None
    ) -> str:
        """
        Save raw templates to disk with metadata.

        Templates are saved as raw EMG data. Feature extraction happens
        during model creation in the Train Model section.

        Args:
            class_label: "open" or "closed"
            template_set_name: Optional name for the template set (e.g., "subject1_session2")
                              If provided, templates are saved to templates_{class_label}_{template_set_name}/

        Returns:
            Path where templates were saved
        """
        # Infer mode from actual template data, not current config
        templates_list = self.templates[class_label]
        is_differential = (
            templates_list[0].shape[0] <= 16
            if templates_list and hasattr(templates_list[0], 'shape')
            else config.ENABLE_DIFFERENTIAL_MODE
        )
        mode_suffix = "sd" if is_differential else "mp"

        if template_set_name:
            folder_name = f"templates_{mode_suffix}_{class_label}_{template_set_name}"
        else:
            folder_name = f"templates_{mode_suffix}_{class_label}"
        base_folder = os.path.join(self.templates_base_path, folder_name)
        os.makedirs(base_folder, exist_ok=True)

        # Save raw templates with metadata
        raw_filepath = os.path.join(base_folder, f"raw_templates_{class_label}.pkl")

        # Include metadata for reproducibility
        save_data = {
            "templates": self.templates[class_label],
            "metadata": {
                "class_label": class_label,
                "template_set_name": template_set_name,
                "template_duration_s": self._template_duration_s,
                "template_type": self.template_type,
                "n_templates": len(self.templates[class_label]),
                "n_channels": self.templates[class_label][0].shape[0] if self.templates[class_label] else 0,
                "n_samples": self.templates[class_label][0].shape[1] if self.templates[class_label] else 0,
                "fsamp": config.FSAMP,
                "created_at": datetime.now().isoformat(),
                "differential_mode": is_differential,
            }
        }

        with open(raw_filepath, "wb") as f:
            pickle.dump(save_data, f)

        print(f"Saved {len(self.templates[class_label])} raw templates to {raw_filepath}")
        print(f"  Template duration: {self._template_duration_s}s")
        print(f"  Template type: {self.template_type}")
        print(f"  Differential mode: {is_differential}")

        return base_folder

    def get_activation_count(self, class_label: str) -> int:
        """Get number of extracted activations for a class."""
        return len(self.all_activations[class_label])

    def get_template_count(self, class_label: str) -> int:
        """Get number of selected templates for a class."""
        return len(self.templates[class_label])

    def get_activation_durations(self, class_label: str) -> List[float]:
        """Get durations of all activations in seconds."""
        return [
            act.shape[1] / config.FSAMP
            for act in self.all_activations[class_label]
        ]

    def clear_activations(self, class_label: Optional[str] = None) -> None:
        """Clear extracted activations and their metadata."""
        if class_label:
            self.all_activations[class_label] = []
            self.activation_metadata[class_label] = []
        else:
            self.all_activations = {"open": [], "closed": []}
            self.activation_metadata = {"open": [], "closed": []}

    def clear_templates(self, class_label: Optional[str] = None) -> None:
        """Clear selected templates."""
        if class_label:
            self.templates[class_label] = []
        else:
            self.templates = {"open": [], "closed": []}

    def clear_all(self, class_label: Optional[str] = None) -> None:
        """Clear both activations and templates."""
        self.clear_activations(class_label)
        self.clear_templates(class_label)

    def extract_activations_bidirectional(
        self,
        recording: dict,
        min_duration_s: float = 1.5,
        pre_context_s: float = 0.5,
        max_duration_s: float = 4.0,
        return_gt: bool = False,
        recording_name: str = ""
    ) -> Dict[str, List[np.ndarray]]:
        """
        Extract both OPEN and CLOSED activations from the same recording.

        Designed for guided recording protocol where one recording contains
        multiple open→close→open cycles. Templates are classified based on
        GT transitions:
        - Rising edge (0→1): CLOSED activation (closing movement)
        - Falling edge (1→0): OPEN activation (opening movement)

        For inverted protocol recordings (close→open→close), the mapping is swapped:
        - First transition (falling 1→0): OPEN activation
        - Second transition (rising 0→1): CLOSED activation

        Args:
            recording: Recording dict in any supported format
            min_duration_s: Minimum activation duration to keep
            pre_context_s: Seconds of context to include before the transition
            max_duration_s: Maximum duration for each activation segment
            return_gt: If True, also return GT signal for each activation

        Returns:
            Dict with "open" and "closed" keys, each containing list of
            EMG segments (n_channels, n_samples).
            If return_gt=True, also includes "open_gt" and "closed_gt" keys
            with corresponding GT signals.
        """
        # Normalize recording to standard format
        normalized = self.normalize_recording(recording)

        emg = normalized["emg"]
        kinematics = normalized["kinematics"]
        gt_mode = normalized.get("gt_mode", None)

        # Detect protocol mode (standard or inverted)
        protocol_mode = recording.get("animation_config", {}).get("protocol_mode", "standard")

        # Determine how to process ground truth
        if gt_mode in ["keyboard", "guided_animation"] or kinematics.shape[0] == 1:
            # Keyboard/guided format - kinematics is already binary at EMG rate
            gt_binary = kinematics.flatten().astype(int)
        else:
            # Virtual hand: Multi-dimensional kinematics - convert to binary
            gt_binary = self._convert_kinematics_to_binary(kinematics, emg.shape[1])

        # Find all transitions
        gt_diff = np.diff(gt_binary, prepend=gt_binary[0])
        rising_edges = np.where(gt_diff == 1)[0]   # 0→1 = CLOSING starts (standard) or CLOSING (inverted)
        falling_edges = np.where(gt_diff == -1)[0]  # 1→0 = OPENING starts (standard) or OPENING (inverted)

        min_samples = int(min_duration_s * config.FSAMP)
        pre_context_samples = int(pre_context_s * config.FSAMP)
        max_samples = int(max_duration_s * config.FSAMP)

        closed_activations = []
        closed_gt = []
        open_activations = []
        open_gt = []

        # CLOSED templates: GT=1 segments after rising edge
        # Include pre-context to show the 0→1 transition
        for rise_idx in rising_edges:
            # Find next falling edge
            next_falls = falling_edges[falling_edges > rise_idx]
            if len(next_falls) > 0:
                end_idx = next_falls[0]
            else:
                end_idx = len(gt_binary)

            # Include pre-context and limit max duration
            start_idx = max(0, rise_idx - pre_context_samples)
            end_idx = min(end_idx, start_idx + max_samples)

            # Extract segment
            segment = emg[:, start_idx:end_idx]
            if segment.shape[1] >= min_samples:
                closed_activations.append(segment)
                if return_gt:
                    closed_gt.append(gt_binary[start_idx:end_idx])

        # OPEN templates: GT=0 segments after falling edge
        # Include pre-context to show the 1→0 transition
        for fall_idx in falling_edges:
            # Find next rising edge (or end of recording)
            next_rises = rising_edges[rising_edges > fall_idx]
            if len(next_rises) > 0:
                end_idx = next_rises[0]
            else:
                end_idx = len(gt_binary)

            # Include pre-context and limit max duration
            start_idx = max(0, fall_idx - pre_context_samples)
            end_idx = min(end_idx, start_idx + max_samples)

            # Extract segment
            segment = emg[:, start_idx:end_idx]
            if segment.shape[1] >= min_samples:
                open_activations.append(segment)
                if return_gt:
                    open_gt.append(gt_binary[start_idx:end_idx])

        # Store in all_activations with metadata
        self.all_activations["closed"].extend(closed_activations)
        for cycle_idx in range(len(closed_activations)):
            self.activation_metadata["closed"].append({
                "recording_name": recording_name,
                "cycle_index": cycle_idx + 1,
                "total_cycles_in_recording": len(closed_activations),
            })
        self.all_activations["open"].extend(open_activations)
        for cycle_idx in range(len(open_activations)):
            self.activation_metadata["open"].append({
                "recording_name": recording_name,
                "cycle_index": cycle_idx + 1,
                "total_cycles_in_recording": len(open_activations),
            })

        print(f"[BIDIRECTIONAL] Extracted {len(closed_activations)} CLOSED, {len(open_activations)} OPEN activations")
        print(f"  Protocol mode: {protocol_mode}")
        print(f"  Pre-context: {pre_context_s}s, Max duration: {max_duration_s}s")

        result = {"open": open_activations, "closed": closed_activations}
        if return_gt:
            result["open_gt"] = open_gt
            result["closed_gt"] = closed_gt

        return result

    def extract_complete_cycles(
        self,
        recording: dict,
        pre_close_s: float = 1.0,
        post_open_s: float = 1.0
    ) -> List[Dict]:
        """
        Extract complete open→close→open cycles from a guided recording.

        Each cycle contains:
        - Last `pre_close_s` seconds of HOLD OPEN (before closing starts)
        - The CLOSING transition
        - The entire HOLD CLOSED period
        - The OPENING transition
        - First `post_open_s` seconds after opening completes

        This allows viewing the complete cycle context for template selection.

        Args:
            recording: Recording dict in any supported format
            pre_close_s: Seconds before closing transition to include
            post_open_s: Seconds after opening transition to include

        Returns:
            List of cycle dicts, each containing:
            - 'emg': EMG data for the cycle (n_channels, n_samples)
            - 'gt': GT signal for the cycle (float 0.0-1.0 for linear transitions)
            - 'close_start_idx': Sample index where GT transitions 0→1 (relative to cycle start)
            - 'open_start_idx': Sample index where GT transitions 1→0 (relative to cycle start)
            - 'close_cue_idx': Sample index of close audio cue (if available)
            - 'open_cue_idx': Sample index of open audio cue (if available)
            - 'cycle_number': 1-indexed cycle number
        """
        # Normalize recording to standard format
        normalized = self.normalize_recording(recording)

        emg = normalized["emg"]
        kinematics = normalized["kinematics"]
        gt_mode = normalized.get("gt_mode", None)

        # Get the GT signal (may be float 0.0-1.0 for guided recordings with linear transitions)
        if gt_mode in ["keyboard", "guided_animation"] or kinematics.shape[0] == 1:
            gt_signal = kinematics.flatten()
            # For cycle detection, binarize (threshold at 0.5)
            gt_binary = (gt_signal > 0.5).astype(int)
        else:
            gt_binary = self._convert_kinematics_to_binary(kinematics, emg.shape[1])
            gt_signal = gt_binary.astype(float)

        # Detect protocol mode (standard or inverted)
        protocol_mode = recording.get("animation_config", {}).get("protocol_mode", "standard")

        # Get audio cue times from guided recording metadata (if available)
        cycle_boundaries = recording.get('cycles', [])
        timings_emg = recording.get('timings_emg')
        recording_start_time = timings_emg[0] if timings_emg is not None and len(timings_emg) > 0 else None

        # Pre-build flat lists of ALL cue sample indices from all user cycles.
        # This handles multi-rep recordings where cycle_boundaries has one entry per
        # user-initiated cycle but extract_complete_cycles produces one entry per
        # repetition — index-based mapping would fail for rep 2+ of each cycle.
        all_close_cue_samples = []  # absolute sample indices
        all_open_cue_samples = []
        if recording_start_time is not None:
            for cb in cycle_boundaries:
                cct = cb.get('close_cue_time')
                oct_val = cb.get('open_cue_time')
                if cct is not None:
                    all_close_cue_samples.append(int((cct - recording_start_time) * config.FSAMP))
                if oct_val is not None:
                    all_open_cue_samples.append(int((oct_val - recording_start_time) * config.FSAMP))

        # Find all transitions from the binarized GT
        gt_diff = np.diff(gt_binary, prepend=gt_binary[0])
        rising_edges = np.where(gt_diff == 1)[0]   # 0→1 = CLOSING starts
        falling_edges = np.where(gt_diff == -1)[0]  # 1→0 = OPENING starts


        pre_close_samples = int(pre_close_s * config.FSAMP)
        post_open_samples = int(post_open_s * config.FSAMP)

        cycles = []

        if protocol_mode == "inverted":
            # Inverted: cycle goes 1→0→1 (closed→open→closed)
            # First transition is falling (opening), second is rising (closing)
            # For each falling edge (opening start), find the next rising edge (closing start)
            for cycle_num, fall_idx in enumerate(falling_edges, start=1):
                # Find the next rising edge after this falling edge
                next_rises = rising_edges[rising_edges > fall_idx]
                if len(next_rises) == 0:
                    # No rising edge found - incomplete cycle, skip
                    continue
                rise_idx = next_rises[0]

                # Calculate cycle boundaries
                # Start: pre_open_s before the falling edge (use pre_close_s for consistency)
                cycle_start = max(0, fall_idx - pre_close_samples)
                # End: post_close_s after the rising edge, but never past the next
                # falling edge — same multi-rep bleed protection as standard mode.
                next_falls_after_rise = falling_edges[falling_edges > rise_idx]
                if len(next_falls_after_rise) > 0:
                    safe_end = next_falls_after_rise[0]
                else:
                    safe_end = len(gt_binary)
                cycle_end = min(safe_end, rise_idx + post_open_samples)

                # Extract EMG and GT for this cycle
                cycle_emg = emg[:, cycle_start:cycle_end]
                cycle_gt = gt_signal[cycle_start:cycle_end]

                # Calculate relative indices within the cycle
                # In inverted mode: first transition is opening, second is closing
                open_start_relative = fall_idx - cycle_start   # Where GT goes 1→0 (first transition)
                close_start_relative = rise_idx - cycle_start  # Where GT goes 0→1 (second transition)

                # Hold durations for onset detection lookback
                hold_open_samples = rise_idx - fall_idx  # GT=0 segment (open contraction)
                # Hold closed = time since previous 0→1 edge (or 0 if first cycle)
                prev_rises = rising_edges[rising_edges < fall_idx]
                hold_closed_samples = int(fall_idx - prev_rises[-1]) if len(prev_rises) > 0 else 0

                # Find audio cue indices by nearest-match (same logic as standard mode)
                cue_tolerance = int(1.0 * config.FSAMP)
                close_cue_idx = None
                open_cue_idx = None

                for abs_cue in all_close_cue_samples:
                    if abs(abs_cue - rise_idx) <= cue_tolerance:
                        rel = abs_cue - cycle_start
                        if 0 <= rel < cycle_emg.shape[1]:
                            close_cue_idx = rel
                        break

                for abs_cue in all_open_cue_samples:
                    if abs(abs_cue - fall_idx) <= cue_tolerance:
                        rel = abs_cue - cycle_start
                        if 0 <= rel < cycle_emg.shape[1]:
                            open_cue_idx = rel
                        break

                cycles.append({
                    'emg': cycle_emg,
                    'gt': cycle_gt,
                    'close_start_idx': close_start_relative,  # Where GT goes 0→1 (second in inverted)
                    'open_start_idx': open_start_relative,    # Where GT goes 1→0 (first in inverted)
                    'close_cue_idx': close_cue_idx,           # Audio cue time (if available)
                    'open_cue_idx': open_cue_idx,             # Audio cue time (if available)
                    'hold_closed_samples': hold_closed_samples,  # Duration of preceding CLOSED hold
                    'hold_open_samples': hold_open_samples,      # Duration of OPEN hold (GT=0)
                    'cycle_number': cycle_num,
                    'duration_s': cycle_emg.shape[1] / config.FSAMP,
                    'protocol_mode': protocol_mode,
                })
        else:
            # Standard: cycle goes 0→1→0 (open→closed→open)
            # First transition is rising (closing), second is falling (opening)
            # For each rising edge (closing start), find the corresponding falling edge (opening start)
            for cycle_num, rise_idx in enumerate(rising_edges, start=1):
                # Find the next falling edge after this rising edge
                next_falls = falling_edges[falling_edges > rise_idx]
                if len(next_falls) == 0:
                    # No falling edge found - incomplete cycle, skip
                    continue
                fall_idx = next_falls[0]

                # Calculate cycle boundaries
                # Start: pre_close_s before the rising edge
                cycle_start = max(0, rise_idx - pre_close_samples)
                # End: post_open_s after the falling edge, but never past the next
                # rising edge — with multi-rep recordings, a short HOLD_OPEN between
                # reps means the next CLOSING could start before post_open_s expires,
                # causing onset detection to pick up the wrong movement.
                next_rises_after_fall = rising_edges[rising_edges > fall_idx]
                if len(next_rises_after_fall) > 0:
                    safe_end = next_rises_after_fall[0]
                else:
                    safe_end = len(gt_binary)
                cycle_end = min(safe_end, fall_idx + post_open_samples)

                # Extract EMG and GT for this cycle
                cycle_emg = emg[:, cycle_start:cycle_end]
                cycle_gt = gt_signal[cycle_start:cycle_end]

                # Calculate relative indices within the cycle
                close_start_relative = rise_idx - cycle_start
                open_start_relative = fall_idx - cycle_start

                # Hold durations — used by onset detection to extend the search window
                # back into the preceding hold (anticipatory movements)
                hold_closed_samples = fall_idx - rise_idx  # GT=1 segment for this cycle
                # Hold open = time since previous GT 1→0 edge (or 0 if first cycle)
                prev_falls = falling_edges[falling_edges < rise_idx]
                hold_open_samples = int(rise_idx - prev_falls[-1]) if len(prev_falls) > 0 else 0

                # Find audio cue indices by matching to the nearest cue within ±1s
                # of each GT transition. This works for any number of reps per cycle.
                cue_tolerance = int(1.0 * config.FSAMP)
                close_cue_idx = None
                open_cue_idx = None

                for abs_cue in all_close_cue_samples:
                    if abs(abs_cue - rise_idx) <= cue_tolerance:
                        rel = abs_cue - cycle_start
                        if 0 <= rel < cycle_emg.shape[1]:
                            close_cue_idx = rel
                        break

                for abs_cue in all_open_cue_samples:
                    if abs(abs_cue - fall_idx) <= cue_tolerance:
                        rel = abs_cue - cycle_start
                        if 0 <= rel < cycle_emg.shape[1]:
                            open_cue_idx = rel
                        break

                cycles.append({
                    'emg': cycle_emg,
                    'gt': cycle_gt,
                    'close_start_idx': close_start_relative,  # Where GT goes 0→1
                    'open_start_idx': open_start_relative,    # Where GT goes 1→0
                    'close_cue_idx': close_cue_idx,           # Audio cue time (if available)
                    'open_cue_idx': open_cue_idx,             # Audio cue time (if available)
                    'hold_closed_samples': hold_closed_samples,  # Duration of CLOSED hold (GT=1)
                    'hold_open_samples': hold_open_samples,      # Duration of preceding OPEN hold
                    'cycle_number': cycle_num,
                    'duration_s': cycle_emg.shape[1] / config.FSAMP,
                    'protocol_mode': protocol_mode,
                })

        # Simple summary output
        if cycles:
            print(f"Extracted {len(cycles)} cycles ({cycles[0]['duration_s']:.1f}s each, {protocol_mode} mode)")

        return cycles

    def extract_from_guided_recording(
        self,
        recording_path: str,
        min_duration_s: float = 1.5
    ) -> Dict[str, int]:
        """
        Convenience method to extract templates from a guided recording file.

        Args:
            recording_path: Path to the guided recording .pkl file
            min_duration_s: Minimum activation duration

        Returns:
            Dict with counts: {"open": n_open, "closed": n_closed}
        """
        with open(recording_path, "rb") as f:
            recording = pickle.load(f)

        # Verify it's a guided recording
        if recording.get("gt_mode") != "guided_animation":
            print(f"Warning: Recording is not guided format (gt_mode: {recording.get('gt_mode')})")

        # Extract bidirectional
        activations = self.extract_activations_bidirectional(recording, min_duration_s)

        return {
            "open": len(activations["open"]),
            "closed": len(activations["closed"])
        }
