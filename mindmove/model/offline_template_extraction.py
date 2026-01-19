"""
Offline Template Extraction Script for MindMove.

This script extracts EMG templates from legacy format recordings (separate EMG + GT folders)
and generates plots for each template to verify onset+hold capture.

Usage:
    python -m mindmove.model.offline_template_extraction

Configure the paths below before running.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

from mindmove.config import config
from mindmove.model.templates.template_manager import TemplateManager
from mindmove.model.core.plotting.template_plots import (
    plot_all_templates,
    plot_activation_with_gt,
    plot_templates_grid,
)


# ============ CONFIGURATION ============

# Data paths - UPDATE THESE FOR YOUR DATA
DATA_BASE = "data/19.11.2025"

# Open class
OPEN_EMG_FOLDER = os.path.join(DATA_BASE, "templates/open")
OPEN_GT_FOLDER = os.path.join(DATA_BASE, "ground truths/open")

# Closed class
CLOSED_EMG_FOLDER = os.path.join(DATA_BASE, "templates/closed")
CLOSED_GT_FOLDER = os.path.join(DATA_BASE, "ground truths/closed")

# Template extraction settings
TEMPLATE_TYPE = "onset_hold"  # "onset_hold" or "hold_only"
SELECTION_MODE = "auto"  # "auto" (longest), "first_n", or "manual"
NUM_TEMPLATES = 20

# Output settings
OUTPUT_FOLDER = "data/extracted_templates"
SAVE_PLOTS = True
SHOW_PLOTS = False  # Set to True for interactive mode

# ============ END CONFIGURATION ============


def extract_templates_for_class(
    emg_folder: str,
    gt_folder: str,
    class_label: str,
    template_manager: TemplateManager
) -> int:
    """
    Extract templates from legacy format for one class.

    Returns:
        Number of templates extracted
    """
    print(f"\n{'='*60}")
    print(f"Processing {class_label.upper()} class")
    print(f"{'='*60}")
    print(f"EMG folder: {emg_folder}")
    print(f"GT folder: {gt_folder}")

    # Load recordings from legacy format
    print("\nLoading recordings...")
    recordings = TemplateManager.load_legacy_format(emg_folder, gt_folder)
    print(f"Loaded {len(recordings)} recordings")

    # Clear any previous activations
    template_manager.clear_all(class_label)

    # Set template type
    include_onset = TEMPLATE_TYPE == "onset_hold"
    template_manager.set_template_type(include_onset)
    print(f"\nTemplate type: {TEMPLATE_TYPE}")
    print(f"  - Include pre-activation: {include_onset}")
    if include_onset:
        print(f"  - Onset offset: {config.ONSET_OFFSET_S}s before GT=1")
    else:
        print(f"  - Hold skip: {config.HOLD_SKIP_S}s after GT=1")

    # Extract activations from all recordings
    print("\nExtracting activations...")
    for i, recording in enumerate(recordings):
        segments = template_manager.extract_activations_from_recording(
            recording,
            class_label,
            include_pre_activation=include_onset
        )
        print(f"  Recording {i+1}: {len(segments)} activations extracted")

        # Optional: plot the recording with GT overlay
        if SAVE_PLOTS:
            recording_output = os.path.join(OUTPUT_FOLDER, class_label, "recordings")
            os.makedirs(recording_output, exist_ok=True)

            emg = recording["emg"]
            gt = recording["kinematics"].flatten()
            filename = recording.get("_legacy_filename", f"recording_{i+1}")

            plot_activation_with_gt(
                emg, gt,
                title=f"Recording {i+1}: {filename}",
                save_path=os.path.join(recording_output, f"recording_{i+1:02d}.png"),
                show=False
            )

    # Get total activation count
    total_activations = template_manager.get_activation_count(class_label)
    print(f"\nTotal activations found: {total_activations}")

    # Get activation durations
    durations = template_manager.get_activation_durations(class_label)
    if durations:
        print(f"Activation durations: min={min(durations):.2f}s, max={max(durations):.2f}s, mean={sum(durations)/len(durations):.2f}s")

    # Select templates based on mode
    print(f"\nSelecting templates (mode: {SELECTION_MODE}, target: {NUM_TEMPLATES})...")
    if SELECTION_MODE == "auto":
        template_manager.select_templates_auto(class_label, n=NUM_TEMPLATES)
    elif SELECTION_MODE == "first_n":
        template_manager.select_templates_first_n(class_label, n=NUM_TEMPLATES)
    else:
        # Manual selection - just take all for now
        indices = list(range(min(NUM_TEMPLATES, total_activations)))
        template_manager.select_templates_manual(indices, class_label)

    # Get template count
    template_count = template_manager.get_template_count(class_label)
    print(f"Selected {template_count} templates")

    return template_count


def main():
    print("\n" + "="*60)
    print("MindMove Offline Template Extraction")
    print("="*60)

    # Create template manager
    template_manager = TemplateManager()
    template_manager.templates_base_path = OUTPUT_FOLDER

    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Process OPEN class
    if os.path.exists(OPEN_EMG_FOLDER) and os.path.exists(OPEN_GT_FOLDER):
        open_count = extract_templates_for_class(
            OPEN_EMG_FOLDER, OPEN_GT_FOLDER, "open", template_manager
        )

        # Plot templates
        if SAVE_PLOTS or SHOW_PLOTS:
            print("\nPlotting OPEN templates...")
            templates_open = template_manager.templates["open"]
            plot_output = os.path.join(OUTPUT_FOLDER, "open", "plots")

            plot_all_templates(
                templates_open,
                class_label="open",
                template_type=TEMPLATE_TYPE,
                output_folder=plot_output if SAVE_PLOTS else None,
                show_individual=False,
                show_grid=SHOW_PLOTS
            )

        # Save templates
        print("\nSaving OPEN templates...")
        template_manager.save_templates("open", save_raw=True, save_features=True)
    else:
        print(f"\nSkipping OPEN class - folders not found:")
        print(f"  EMG: {OPEN_EMG_FOLDER} (exists: {os.path.exists(OPEN_EMG_FOLDER)})")
        print(f"  GT: {OPEN_GT_FOLDER} (exists: {os.path.exists(OPEN_GT_FOLDER)})")

    # Process CLOSED class
    if os.path.exists(CLOSED_EMG_FOLDER) and os.path.exists(CLOSED_GT_FOLDER):
        closed_count = extract_templates_for_class(
            CLOSED_EMG_FOLDER, CLOSED_GT_FOLDER, "closed", template_manager
        )

        # Plot templates
        if SAVE_PLOTS or SHOW_PLOTS:
            print("\nPlotting CLOSED templates...")
            templates_closed = template_manager.templates["closed"]
            plot_output = os.path.join(OUTPUT_FOLDER, "closed", "plots")

            plot_all_templates(
                templates_closed,
                class_label="closed",
                template_type=TEMPLATE_TYPE,
                output_folder=plot_output if SAVE_PLOTS else None,
                show_individual=False,
                show_grid=SHOW_PLOTS
            )

        # Save templates
        print("\nSaving CLOSED templates...")
        template_manager.save_templates("closed", save_raw=True, save_features=True)
    else:
        print(f"\nSkipping CLOSED class - folders not found:")
        print(f"  EMG: {CLOSED_EMG_FOLDER} (exists: {os.path.exists(CLOSED_EMG_FOLDER)})")
        print(f"  GT: {CLOSED_GT_FOLDER} (exists: {os.path.exists(CLOSED_GT_FOLDER)})")

    print("\n" + "="*60)
    print("Template extraction complete!")
    print(f"Output saved to: {OUTPUT_FOLDER}")
    print("="*60)


if __name__ == "__main__":
    main()
