"""
Template management module for MindMove.

Contains utilities for loading, extracting, and managing EMG templates.
"""

from mindmove.model.templates.template_manager import TemplateManager
from mindmove.model.templates.data_loading import (
    load_templates_recordings,
    load_ground_truth_recordings,
    load_emg_and_gt,
    convert_gt_to_binary,
    activation_extractor,
    extract_intervals,
    convert_kinematics_to_binary,
    extract_activation_segments_from_binary,
    load_mindmove_recording,
)

__all__ = [
    "TemplateManager",
    "load_templates_recordings",
    "load_ground_truth_recordings",
    "load_emg_and_gt",
    "convert_gt_to_binary",
    "activation_extractor",
    "extract_intervals",
    "convert_kinematics_to_binary",
    "extract_activation_segments_from_binary",
    "load_mindmove_recording",
]
