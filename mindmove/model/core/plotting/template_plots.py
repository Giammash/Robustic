"""
Template plotting utilities for MindMove.

Provides visualization of EMG templates showing onset and hold phases.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import os

from mindmove.config import config


def plot_template(
    template: np.ndarray,
    title: str = "EMG Template",
    template_type: str = "onset_hold",
    save_path: Optional[str] = None,
    show: bool = True,
    channels_to_plot: Optional[List[int]] = None
) -> plt.Figure:
    """
    Plot a single EMG template with onset/hold phase markers.

    Args:
        template: EMG data (n_channels, n_samples)
        title: Plot title
        template_type: "onset_hold" or "hold_only" - affects phase markers
        save_path: If provided, save figure to this path
        show: If True, display the figure
        channels_to_plot: List of channel indices to plot (default: all)

    Returns:
        matplotlib Figure object
    """
    n_channels, n_samples = template.shape
    duration_s = n_samples / config.FSAMP
    time_axis = np.linspace(0, duration_s, n_samples)

    if channels_to_plot is None:
        channels_to_plot = list(range(n_channels))

    n_plot_channels = len(channels_to_plot)

    # Create figure
    fig, axes = plt.subplots(
        n_plot_channels, 1,
        figsize=(12, max(6, n_plot_channels * 0.5)),
        sharex=True
    )

    if n_plot_channels == 1:
        axes = [axes]

    # Calculate mean EMG envelope for summary
    mean_emg = np.mean(np.abs(template), axis=0)

    # Plot each channel
    for i, ch_idx in enumerate(channels_to_plot):
        ax = axes[i]
        ax.plot(time_axis, template[ch_idx, :], 'b-', linewidth=0.5, alpha=0.7)
        ax.set_ylabel(f'Ch {ch_idx}', fontsize=8)
        ax.tick_params(axis='y', labelsize=6)

        # Add phase markers
        if template_type == "onset_hold":
            # Onset phase: 0 to 0.2s (before GT=1)
            ax.axvspan(0, config.ONSET_OFFSET_S, alpha=0.2, color='yellow', label='Onset')
            # Hold phase: 0.2s to end
            ax.axvspan(config.ONSET_OFFSET_S, duration_s, alpha=0.1, color='green', label='Hold')
            ax.axvline(x=config.ONSET_OFFSET_S, color='red', linestyle='--', linewidth=1, alpha=0.7)
        else:  # hold_only
            # Hold phase starts at 0.5s from original activation
            ax.axvspan(0, duration_s, alpha=0.1, color='green', label='Hold')

    # Set common properties
    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title(f'{title}\n(Type: {template_type}, Duration: {duration_s:.2f}s)')

    # Add legend to first subplot
    if template_type == "onset_hold":
        axes[0].legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved template plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_template_summary(
    template: np.ndarray,
    title: str = "EMG Activation",
    template_type: str = "onset_hold",
    save_path: Optional[str] = None,
    show: bool = True,
    channel: int = 0,
    template_duration_s: float = 1.0
) -> plt.Figure:
    """
    Plot raw EMG signal from a single channel with template boundary markers.

    Shows:
    - Raw EMG signal for the selected channel
    - RED vertical line: where GT=1 starts (activation start)
    - BLUE vertical lines: template boundaries
      - For onset_hold: starts 0.2s before activation, ends after template_duration
      - For hold_only: starts 0.5s after activation, ends after template_duration

    Args:
        template: EMG data (n_channels, n_samples) - this is the activation segment
        title: Plot title
        template_type: "onset_hold" or "hold_only"
        save_path: If provided, save figure to this path
        show: If True, display the figure
        channel: Which channel to plot (0-indexed, default 0)
        template_duration_s: Template duration in seconds (default 1.0)

    Returns:
        matplotlib Figure object
    """
    n_channels, n_samples = template.shape
    duration_s = n_samples / config.FSAMP
    time_axis = np.linspace(0, duration_s, n_samples)

    # Clamp channel to valid range
    channel = max(0, min(channel, n_channels - 1))

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))

    # Plot raw EMG for selected channel
    signal = template[channel, :]
    ax.plot(time_axis, signal, 'k-', linewidth=0.5, label=f'Channel {channel}')

    # Calculate marker positions based on template type
    if template_type == "onset_hold":
        # For onset_hold mode, the activation segment includes pre-activation samples
        # The activation (GT=1) starts at ONSET_OFFSET_S into the segment
        activation_start = config.ONSET_OFFSET_S
        template_start = 0  # Template starts at beginning (includes onset)
        template_end = template_duration_s
    else:  # hold_only
        # For hold_only mode, activation starts at the beginning
        # Template starts 0.5s after activation
        activation_start = 0
        template_start = config.HOLD_SKIP_S
        template_end = template_start + template_duration_s

    # Draw RED line at activation start (GT=1)
    ax.axvline(x=activation_start, color='red', linestyle='-', linewidth=2,
               label=f'GT=1 (activation start)')

    # Draw BLUE lines for template boundaries
    ax.axvline(x=template_start, color='blue', linestyle='--', linewidth=2,
               label=f'Template start')
    ax.axvline(x=template_end, color='blue', linestyle='--', linewidth=2,
               label=f'Template end ({template_duration_s}s)')

    # Shade the template region
    ax.axvspan(template_start, template_end, alpha=0.15, color='blue',
               label='Template region')

    # Labels and formatting
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('EMG Amplitude', fontsize=10)
    ax.set_title(f'{title}\nChannel {channel} | Duration: {duration_s:.2f}s | Template type: {template_type}',
                 fontsize=11)
    ax.set_xlim(0, duration_s)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved template plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_all_templates(
    templates: List[np.ndarray],
    class_label: str = "unknown",
    template_type: str = "onset_hold",
    output_folder: Optional[str] = None,
    show_individual: bool = False,
    show_grid: bool = True
) -> None:
    """
    Plot all templates in a collection.

    Args:
        templates: List of EMG templates (each is n_channels x n_samples)
        class_label: "open" or "closed"
        template_type: "onset_hold" or "hold_only"
        output_folder: If provided, save plots to this folder
        show_individual: If True, show each template individually
        show_grid: If True, show a grid overview of all templates
    """
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    n_templates = len(templates)
    print(f"\nPlotting {n_templates} templates for class '{class_label}'")

    # Plot individual templates
    if show_individual or output_folder:
        for i, template in enumerate(templates):
            title = f"Template {i+1}/{n_templates} - {class_label.capitalize()}"
            save_path = None
            if output_folder:
                save_path = os.path.join(output_folder, f"template_{class_label}_{i+1:02d}.png")

            plot_template_summary(
                template,
                title=title,
                template_type=template_type,
                save_path=save_path,
                show=show_individual
            )

    # Plot grid overview
    if show_grid and n_templates > 0:
        plot_templates_grid(
            templates,
            class_label=class_label,
            template_type=template_type,
            save_path=os.path.join(output_folder, f"templates_grid_{class_label}.png") if output_folder else None,
            show=True
        )


def plot_templates_grid(
    templates: List[np.ndarray],
    class_label: str = "unknown",
    template_type: str = "onset_hold",
    save_path: Optional[str] = None,
    show: bool = True,
    max_templates: int = 20,
    channels_to_show: int = 4
) -> plt.Figure:
    """
    Plot a grid overview of all templates showing raw EMG from representative channels.

    Args:
        templates: List of EMG templates
        class_label: "open" or "closed"
        template_type: "onset_hold" or "hold_only"
        save_path: If provided, save figure
        show: If True, display figure
        max_templates: Maximum number of templates to show
        channels_to_show: Number of representative channels to overlay

    Returns:
        matplotlib Figure object
    """
    n_templates = min(len(templates), max_templates)

    if n_templates == 0:
        print("No templates to plot")
        return None

    # Calculate grid dimensions
    n_cols = min(5, n_templates)
    n_rows = (n_templates + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))

    if n_templates == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Use a colormap for different channels
    colors = plt.cm.tab10(np.linspace(0, 1, channels_to_show))

    for i in range(n_templates):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        template = templates[i]
        n_channels, n_samples = template.shape
        duration_s = n_samples / config.FSAMP
        time_axis = np.linspace(0, duration_s, n_samples)

        # Select representative channels (evenly spaced across all channels)
        channel_indices = np.linspace(0, n_channels - 1, channels_to_show, dtype=int)

        # Plot raw EMG from selected channels (overlaid)
        for j, ch in enumerate(channel_indices):
            signal = template[ch, :]
            # Normalize for visibility
            if np.max(np.abs(signal)) > 0:
                signal_norm = signal / np.max(np.abs(signal))
            else:
                signal_norm = signal
            ax.plot(time_axis, signal_norm, color=colors[j], linewidth=0.5, alpha=0.7)

        ax.set_title(f'T{i+1}', fontsize=10)
        ax.set_ylim(-1.5, 1.5)

        # Add onset marker
        if template_type == "onset_hold":
            ax.axvline(x=config.ONSET_OFFSET_S, color='red', linestyle='--',
                      linewidth=1, alpha=0.7)

        ax.set_xlim(0, duration_s)
        ax.tick_params(axis='both', labelsize=7)

        if row == n_rows - 1:
            ax.set_xlabel('Time (s)', fontsize=8)
        if col == 0:
            ax.set_ylabel('Norm. EMG', fontsize=8)

    # Hide empty subplots
    for i in range(n_templates, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle(
        f'{class_label.capitalize()} Templates ({n_templates} total)\n'
        f'Red line = activation start (GT=1)',
        fontsize=12
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved templates grid to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_activation_with_template_markers(
    activation: np.ndarray,
    title: str = "Activation",
    template_type: str = "onset_hold",
    template_duration_s: float = 1.0,
    channel: int = 0,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a single channel of the full activation segment with template boundary markers.

    Shows:
    - Raw EMG signal for the selected channel
    - RED vertical line: where GT=1 starts (activation start)
    - BLUE vertical lines: template boundaries showing what would be cut

    For onset_hold:
    - Template starts 0.2s BEFORE activation (first blue line)
    - Template ends template_duration_s after first blue line (second blue line)

    For hold_only:
    - Template starts 0.5s AFTER activation (first blue line)
    - Template ends template_duration_s after first blue line (second blue line)

    Args:
        activation: Full activation segment (n_channels, n_samples)
                   For onset_hold, this includes pre_activation samples at the start
        title: Plot title
        template_type: "onset_hold" or "hold_only"
        template_duration_s: Template duration in seconds
        channel: Which channel to plot (0-indexed)
        save_path: If provided, save figure to this path
        show: If True, display the figure

    Returns:
        matplotlib Figure object
    """
    n_channels, n_samples = activation.shape
    duration_s = n_samples / config.FSAMP
    time_axis = np.linspace(0, duration_s, n_samples)

    # Clamp channel to valid range
    channel = max(0, min(channel, n_channels - 1))

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    # Plot raw EMG for selected channel
    signal = activation[channel, :]
    ax.plot(time_axis, signal, 'k-', linewidth=0.5)

    # Calculate marker positions
    if template_type == "onset_hold":
        # For onset_hold mode, the activation segment includes ONSET_OFFSET_S pre-activation samples
        # So GT=1 (activation start) is at ONSET_OFFSET_S into the segment
        pre_activation_time = config.ONSET_OFFSET_S
        activation_start = pre_activation_time  # RED line position

        # Template starts at beginning (includes 0.2s before GT=1)
        template_start = 0  # First BLUE line
        template_end = template_duration_s  # Second BLUE line
    else:  # hold_only
        # For hold_only mode, GT=1 is at the beginning of the segment
        activation_start = 0  # RED line position

        # Template starts 0.5s after activation
        template_start = config.HOLD_SKIP_S  # First BLUE line
        template_end = template_start + template_duration_s  # Second BLUE line

    # Draw RED line at activation start (GT=1)
    ax.axvline(x=activation_start, color='red', linestyle='-', linewidth=2,
               label=f'GT=1 (activation start) @ {activation_start:.2f}s')

    # Draw BLUE lines for template boundaries
    ax.axvline(x=template_start, color='blue', linestyle='--', linewidth=2,
               label=f'Template start @ {template_start:.2f}s')

    if template_end <= duration_s:
        ax.axvline(x=template_end, color='blue', linestyle='--', linewidth=2,
                   label=f'Template end @ {template_end:.2f}s')
        # Shade the template region
        ax.axvspan(template_start, template_end, alpha=0.15, color='blue')
    else:
        # Template extends beyond activation - mark with dashed line at edge
        ax.axvline(x=duration_s, color='blue', linestyle=':', linewidth=2,
                   label=f'Template end @ {template_end:.2f}s (beyond segment!)')
        ax.axvspan(template_start, duration_s, alpha=0.15, color='orange')
        print(f"WARNING: Template ({template_duration_s}s) extends beyond activation ({duration_s:.2f}s)")

    # Labels and formatting
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('EMG Amplitude', fontsize=11)
    ax.set_title(f'{title} | Channel {channel} | Activation: {duration_s:.2f}s | Template: {template_duration_s}s ({template_type})',
                 fontsize=11)
    ax.set_xlim(0, duration_s)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved activation plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_activation_with_gt(
    emg: np.ndarray,
    gt_binary: np.ndarray,
    title: str = "Activation with Ground Truth",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot EMG signal with ground truth overlay.

    Args:
        emg: EMG data (n_channels, n_samples)
        gt_binary: Binary ground truth (n_samples,)
        title: Plot title
        save_path: If provided, save figure
        show: If True, display figure

    Returns:
        matplotlib Figure object
    """
    n_channels, n_samples = emg.shape
    duration_s = n_samples / config.FSAMP
    time_axis = np.linspace(0, duration_s, n_samples)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Top: Mean EMG envelope
    mean_emg = np.mean(np.abs(emg), axis=0)
    ax1.plot(time_axis, mean_emg, 'b-', linewidth=0.8)
    ax1.set_ylabel('Mean |EMG|')
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    # Highlight activation regions
    in_activation = False
    start_idx = 0
    for i in range(len(gt_binary)):
        if gt_binary[i] == 1 and not in_activation:
            start_idx = i
            in_activation = True
        elif gt_binary[i] == 0 and in_activation:
            ax1.axvspan(time_axis[start_idx], time_axis[i-1],
                       alpha=0.3, color='green', label='Activation' if start_idx == 0 else '')
            in_activation = False

    if in_activation:
        ax1.axvspan(time_axis[start_idx], time_axis[-1], alpha=0.3, color='green')

    # Bottom: Ground truth
    ax2.plot(time_axis, gt_binary, 'r-', linewidth=1.5)
    ax2.set_ylabel('GT')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Open', 'Closed'])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
