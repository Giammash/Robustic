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
            # Keyboard format - gt is already binary at EMG sample rate
            gt = recording["gt"]
            # Reshape to (1, n_samples) to match expected kinematics shape
            kinematics = gt.reshape(1, -1) if gt.ndim == 1 else gt

            return {
                "emg": recording["emg"],
                "kinematics": kinematics,  # Binary GT as "kinematics"
                "timings_emg": recording.get("timings_emg"),
                "timings_kinematics": recording.get("timings_emg"),  # Same as EMG
                "label": recording.get("label", "default"),
                "task": recording.get("task", "unknown"),
                "gt_mode": "keyboard",
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
        include_pre_activation: bool = False
    ) -> List[np.ndarray]:
        """
        Extract all activation segments from a recording.

        Works with recordings of any length - extracts all activations found.
        Automatically detects and normalizes recording format.

        Args:
            recording: Recording dict in any supported format
            class_label: "open" or "closed"
            include_pre_activation: If True, include ONSET_OFFSET_S before activation start

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

        # Add to all_activations
        self.all_activations[class_label].extend(segments)

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
        if template_set_name:
            folder_name = f"templates_{class_label}_{template_set_name}"
        else:
            folder_name = f"templates_{class_label}"
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
            }
        }

        with open(raw_filepath, "wb") as f:
            pickle.dump(save_data, f)

        print(f"Saved {len(self.templates[class_label])} raw templates to {raw_filepath}")
        print(f"  Template duration: {self._template_duration_s}s")
        print(f"  Template type: {self.template_type}")

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
        """Clear extracted activations."""
        if class_label:
            self.all_activations[class_label] = []
        else:
            self.all_activations = {"open": [], "closed": []}

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
        min_duration_s: float = 1.5
    ) -> Dict[str, List[np.ndarray]]:
        """
        Extract both OPEN and CLOSED activations from the same recording.

        Designed for guided recording protocol where one recording contains
        multiple open→close→open cycles. Templates are classified based on
        GT transitions:
        - Rising edge (0→1): CLOSED activation (closing movement)
        - Falling edge (1→0): OPEN activation (opening movement)

        Args:
            recording: Recording dict in any supported format
            min_duration_s: Minimum activation duration to keep

        Returns:
            Dict with "open" and "closed" keys, each containing list of
            EMG segments (n_channels, n_samples)
        """
        # Normalize recording to standard format
        normalized = self.normalize_recording(recording)

        emg = normalized["emg"]
        kinematics = normalized["kinematics"]
        gt_mode = normalized.get("gt_mode", None)

        # Determine how to process ground truth
        if gt_mode in ["keyboard", "guided_animation"] or kinematics.shape[0] == 1:
            # Keyboard/guided format - kinematics is already binary at EMG rate
            gt_binary = kinematics.flatten().astype(int)
        else:
            # Virtual hand: Multi-dimensional kinematics - convert to binary
            gt_binary = self._convert_kinematics_to_binary(kinematics, emg.shape[1])

        # Find all transitions
        gt_diff = np.diff(gt_binary, prepend=gt_binary[0])
        rising_edges = np.where(gt_diff == 1)[0]   # 0→1 = CLOSING starts
        falling_edges = np.where(gt_diff == -1)[0]  # 1→0 = OPENING starts

        min_samples = int(min_duration_s * config.FSAMP)

        closed_activations = []
        open_activations = []

        # CLOSED templates: GT=1 segments after rising edge
        # These represent the closing movement and hold-closed phase
        for rise_idx in rising_edges:
            # Find next falling edge
            next_falls = falling_edges[falling_edges > rise_idx]
            if len(next_falls) > 0:
                end_idx = next_falls[0]
            else:
                end_idx = len(gt_binary)

            # Extract segment
            segment = emg[:, rise_idx:end_idx]
            if segment.shape[1] >= min_samples:
                closed_activations.append(segment)

        # OPEN templates: GT=0 segments after falling edge
        # These represent the opening movement
        # Note: We look for segments between falling edge and next rising edge
        for fall_idx in falling_edges:
            # Find next rising edge (or end of recording)
            next_rises = rising_edges[rising_edges > fall_idx]
            if len(next_rises) > 0:
                end_idx = next_rises[0]
            else:
                # Use end of recording, but limit to reasonable duration
                # (we don't want the "waiting" period between cycles)
                # Use 2x the min_duration as max
                max_samples = int(2 * min_duration_s * config.FSAMP)
                end_idx = min(fall_idx + max_samples, len(gt_binary))

            # Extract segment
            segment = emg[:, fall_idx:end_idx]
            if segment.shape[1] >= min_samples:
                open_activations.append(segment)

        # Store in all_activations
        self.all_activations["closed"].extend(closed_activations)
        self.all_activations["open"].extend(open_activations)

        print(f"[BIDIRECTIONAL] Extracted {len(closed_activations)} CLOSED, {len(open_activations)} OPEN activations")

        return {"open": open_activations, "closed": closed_activations}

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
