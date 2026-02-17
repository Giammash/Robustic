"""
Template visualization script.

Plots all templates overlapped per channel to inspect similarity and variability.
Supports both raw EMG and feature-extracted views, and a "cycles" mode to visualize
complete recording cycles with all channels stacked.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mindmove.config import config
from mindmove.model.core.windowing import sliding_window
from mindmove.model.core.features.features_registry import FEATURES


# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to templates file (relative to project root) — used for "raw" and feature modes
# TEMPLATES_FILE = "data/recordings/patient S1/templates_sd_20260206_121311_guided_4cycles.pkl"
TEMPLATES_FILE = "data/templates/templates_mp_20260212_105426_guided_16cycles.pkl"

# Plot mode: "raw" for raw EMG, a feature name (e.g. "wl", "rms", "kurtosis"),
#            "cycles" to plot each cycle individually with all channels stacked,
#            "cycles_overlap" to overlap all cycle segments (OPEN/CLOSED) on one figure
PLOT_MODE = "wl"

# Plot style: "overlap" = all templates overlapped per channel (one figure per channel)
#             "stacked" = all channels stacked per template (one figure per template)
#             "stacked_overlap" = all templates overlapped on one figure, all channels stacked
PLOT_STYLE = "stacked_overlap"

# Recording files for "cycles" mode (relative to project root)
RECORDING_FILES = [
    # "data/recordings/recording_example.pkl",
]

# =============================================================================


def plot_templates_overlap(
    templates_open: list,
    templates_closed: list,
    title: str = "Templates Overlap",
    x_label: str = "Time (ms)",
    y_label: str = "Amplitude (µV)",
):
    """
    Plot all templates overlapped. One figure per channel, with OPEN and CLOSED
    as two subplots (top/bottom).

    Args:
        templates_open: List of templates. Raw: (n_channels, n_samples) or
                        feature-extracted: (n_windows, n_channels)
        templates_closed: Same format as templates_open
        title: Plot title prefix
        x_label: Label for x-axis
        y_label: Label for y-axis
    """
    # Detect format: raw is (n_channels, n_samples), features is (n_windows, n_channels)
    sample = templates_open[0]
    if sample.ndim == 2 and sample.shape[0] > sample.shape[1]:
        # Feature-extracted: (n_windows, n_channels) — channels on axis 1
        n_channels = sample.shape[1]
        get_channel = lambda t, ch: t[:, ch]
        n_points = sample.shape[0]
        x_axis = np.arange(n_points)
    else:
        # Raw EMG: (n_channels, n_samples) — channels on axis 0
        n_channels = sample.shape[0]
        get_channel = lambda t, ch: t[ch, :]
        n_points = sample.shape[1]
        x_axis = np.arange(n_points) / config.FSAMP * 1000  # ms

    cmap_open = plt.colormaps['tab20'].resampled(len(templates_open))
    cmap_closed = plt.colormaps['tab20'].resampled(len(templates_closed))

    for ch in range(n_channels):
        fig, (ax_open, ax_closed) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(f"{title} — CH{ch+1}", fontsize=14)

        for i, t in enumerate(templates_open):
            ax_open.plot(x_axis, get_channel(t, ch), color=cmap_open(i), alpha=0.7, linewidth=0.8, label=f"T{i+1}")
        ax_open.set_title(f"OPEN ({len(templates_open)} templates)")
        ax_open.set_ylabel(y_label)
        ax_open.grid(True, alpha=0.3)
        ax_open.legend(fontsize=7, ncol=4, loc='upper right')

        for i, t in enumerate(templates_closed):
            ax_closed.plot(x_axis, get_channel(t, ch), color=cmap_closed(i), alpha=0.7, linewidth=0.8, label=f"T{i+1}")
        ax_closed.set_title(f"CLOSED ({len(templates_closed)} templates)")
        ax_closed.set_ylabel(y_label)
        ax_closed.set_xlabel(x_label)
        ax_closed.grid(True, alpha=0.3)
        ax_closed.legend(fontsize=7, ncol=4, loc='upper right')

        fig.tight_layout()


def plot_templates_stacked(
    templates_open: list,
    templates_closed: list,
    title: str = "Template",
    x_label: str = "Time (ms)",
    y_label: str = "Amplitude (µV)",
):
    """
    Plot each template as a separate figure with all channels stacked vertically.
    Each figure has OPEN (left) and CLOSED (right) side by side.

    Args:
        templates_open: List of templates. Raw: (n_channels, n_samples) or
                        feature-extracted: (n_windows, n_channels)
        templates_closed: Same format as templates_open
        title: Plot title prefix
        x_label: Label for x-axis
        y_label: Label for y-axis
    """
    sample = templates_open[0]
    if sample.ndim == 2 and sample.shape[0] > sample.shape[1]:
        n_channels = sample.shape[1]
        get_channel = lambda t, ch: t[:, ch]
        n_points = sample.shape[0]
        x_axis = np.arange(n_points)
    else:
        n_channels = sample.shape[0]
        get_channel = lambda t, ch: t[ch, :]
        n_points = sample.shape[1]
        x_axis = np.arange(n_points) / config.FSAMP * 1000  # ms

    # Compute global offset: max range across all templates and channels
    all_ranges = []
    for t in templates_open + templates_closed:
        for ch in range(n_channels):
            sig = get_channel(t, ch)
            all_ranges.append(np.ptp(sig))
    offset_step = np.median(all_ranges) * 1.5

    n_templates = max(len(templates_open), len(templates_closed))

    for idx in range(n_templates):
        fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
        fig.suptitle(f"{title} — Template {idx+1}/{n_templates}", fontsize=14)

        for col, (templates, label) in enumerate([
            (templates_open, "OPEN"), (templates_closed, "CLOSED")
        ]):
            ax = axes[col]
            if idx < len(templates):
                t = templates[idx]
                yticks = []
                ytick_labels = []
                for ch in range(n_channels):
                    sig = get_channel(t, ch)
                    offset = (n_channels - 1 - ch) * offset_step
                    ax.plot(x_axis, sig + offset, color='C0', linewidth=0.8)
                    yticks.append(offset)
                    ytick_labels.append(f"CH{ch+1}")
                ax.set_yticks(yticks)
                ax.set_yticklabels(ytick_labels)
            ax.set_title(f"{label} T{idx+1}")
            ax.set_xlabel(x_label)
            ax.grid(True, alpha=0.3, axis='x')

        axes[0].set_ylabel(y_label)
        fig.tight_layout()


def plot_templates_stacked_overlap(
    templates_open: list,
    templates_closed: list,
    title: str = "Templates Stacked Overlap",
    x_label: str = "Time (ms)",
    y_label: str = "Amplitude (µV)",
):
    """
    Plot ALL templates overlapped on one figure with all channels stacked.
    One subplot for OPEN (left) and one for CLOSED (right).
    Each template gets a different color.

    Works with both raw (n_channels, n_samples) and feature (n_windows, n_channels) formats.
    """
    sample = templates_open[0] if templates_open else templates_closed[0]
    if sample.ndim == 2 and sample.shape[0] > sample.shape[1]:
        n_channels = sample.shape[1]
        get_channel = lambda t, ch: t[:, ch]
        n_points = sample.shape[0]
        x_axis = np.arange(n_points)
    else:
        n_channels = sample.shape[0]
        get_channel = lambda t, ch: t[ch, :]
        n_points = sample.shape[1]
        x_axis = np.arange(n_points) / config.FSAMP * 1000  # ms

    # Compute global offset from all templates
    all_ranges = []
    for t in templates_open + templates_closed:
        for ch in range(n_channels):
            all_ranges.append(np.ptp(get_channel(t, ch)))
    offset_step = np.median(all_ranges) * 1.5 if np.median(all_ranges) > 0 else 1.0

    for templates, label in [
        (templates_open, "OPEN"),
        (templates_closed, "CLOSED"),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        n_t = len(templates)

        # Pre-compute colors: each channel gets a unique hue,
        # each template is a different shade (lightness) within that hue
        from matplotlib.colors import hsv_to_rgb
        channel_colors = []  # channel_colors[ch][i] = color for template i on channel ch
        for ch in range(n_channels):
            hue = ch / n_channels
            shades = []
            for i in range(n_t):
                # Vary value from 0.35 (dark) to 0.9 (bright)
                val = 0.35 + 0.55 * i / max(n_t - 1, 1)
                shades.append(hsv_to_rgb([hue, 0.75, val]))
            channel_colors.append(shades)

        yticks = []
        ytick_labels = []
        for ch in range(n_channels):
            offset = (n_channels - 1 - ch) * offset_step
            yticks.append(offset)
            ytick_labels.append(f"CH{ch + 1}")

            for i, t in enumerate(templates):
                sig = get_channel(t, ch)
                ax.plot(x_axis, sig + offset, color=channel_colors[ch][i], linewidth=0.6,
                       alpha=0.7, label=f"T{i+1}" if ch == 0 else None)

        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, fontsize=7)
        ax.set_title(f"{title} — {label} ({n_t} templates)")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3, axis='x')
        if n_t <= 20:
            ax.legend(fontsize=6, ncol=2, loc='upper right', framealpha=0.7)
        fig.tight_layout()


def plot_cycles_stacked_overlap(recording_files: list, base_path: Path):
    """
    Load recording(s), extract cycles, then extract OPEN and CLOSED steady-state
    segments from each cycle and plot them overlapped with all channels stacked.

    One figure with two subplots: OPEN (left) and CLOSED (right).
    Each cycle segment gets a different color.
    """
    from mindmove.model.templates.template_manager import TemplateManager

    tm = TemplateManager()
    open_segments = []
    closed_segments = []

    for rec_file in recording_files:
        rec_path = base_path / rec_file
        print(f"\nLoading recording: {rec_path}")
        with open(rec_path, 'rb') as f:
            recording = pickle.load(f)

        emg = recording.get('emg', recording.get('data', np.array([])))
        if hasattr(emg, 'shape') and emg.ndim == 2 and emg.shape[0] <= 16:
            config.ENABLE_DIFFERENTIAL_MODE = True
            config.num_channels = 16

        cycles = tm.extract_complete_cycles(recording)
        print(f"  Extracted {len(cycles)} cycles")

        for cycle in cycles:
            emg = cycle['emg']
            gt = cycle['gt']
            n_samples = emg.shape[1]

            # Find steady-state regions from GT
            gt_binary = (gt > 0.5).astype(int)
            gt_diff = np.diff(gt_binary, prepend=gt_binary[0])
            transitions = np.where(gt_diff != 0)[0]
            region_starts = np.concatenate([[0], transitions])
            region_ends = np.concatenate([transitions, [n_samples]])

            for rs, re in zip(region_starts, region_ends):
                if re - rs < int(config.FSAMP * 0.3):
                    continue  # Skip short transitions
                region_val = gt_binary[rs + (re - rs) // 2]
                segment = emg[:, rs:re]
                if region_val == 1:
                    closed_segments.append(segment)
                else:
                    open_segments.append(segment)

    print(f"\n  OPEN segments: {len(open_segments)}")
    print(f"  CLOSED segments: {len(closed_segments)}")

    if not open_segments and not closed_segments:
        print("No segments found!")
        return

    # Find common channel count
    n_channels = open_segments[0].shape[0] if open_segments else closed_segments[0].shape[0]

    # Compute global offset from all segments
    all_ranges = []
    for seg in open_segments + closed_segments:
        for ch in range(n_channels):
            all_ranges.append(np.ptp(seg[ch, :]))
    offset_step = np.median(all_ranges) * 1.5 if np.median(all_ranges) > 0 else 1.0

    rec_names = ", ".join(Path(f).stem for f in recording_files)

    for segments, label in [
        (open_segments, "OPEN"), (closed_segments, "CLOSED"),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        n_seg = len(segments)

        # Each channel gets a unique hue, each segment a different shade
        from matplotlib.colors import hsv_to_rgb
        channel_colors = []
        for ch in range(n_channels):
            hue = ch / n_channels
            shades = []
            for i in range(n_seg):
                val = 0.35 + 0.55 * i / max(n_seg - 1, 1)
                shades.append(hsv_to_rgb([hue, 0.75, val]))
            channel_colors.append(shades)

        yticks = []
        ytick_labels = []
        for ch in range(n_channels):
            offset = (n_channels - 1 - ch) * offset_step
            yticks.append(offset)
            ytick_labels.append(f"CH{ch + 1}")

            for i, seg in enumerate(segments):
                sig = seg[ch, :]
                x_axis = np.arange(len(sig)) / config.FSAMP * 1000  # ms
                ax.plot(x_axis, sig + offset, color=channel_colors[ch][i], linewidth=0.6,
                       alpha=0.7, label=f"C{i+1}" if ch == 0 else None)

        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, fontsize=7)
        ax.set_title(f"Cycles Overlap — {label} ({n_seg} segments) — {rec_names}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True, alpha=0.3, axis='x')
        if n_seg <= 20:
            ax.legend(fontsize=6, ncol=2, loc='upper right', framealpha=0.7)
        fig.tight_layout()

    print(f"\nCycles overlap plotted: {len(open_segments)} OPEN, {len(closed_segments)} CLOSED")


def plot_cycles_stacked(recording_files: list, base_path: Path):
    """
    Load recording(s), extract cycles, and plot each cycle with all channels stacked.

    Each cycle gets one figure showing all EMG channels stacked vertically
    with the GT signal at the bottom.

    Args:
        recording_files: List of recording file paths (relative to base_path)
        base_path: Project root path
    """
    from mindmove.model.templates.template_manager import TemplateManager

    tm = TemplateManager()
    cycle_num = 0

    for rec_file in recording_files:
        rec_path = base_path / rec_file
        print(f"\nLoading recording: {rec_path}")
        with open(rec_path, 'rb') as f:
            recording = pickle.load(f)

        # Detect differential mode
        emg = recording.get('emg', recording.get('data', np.array([])))
        if hasattr(emg, 'shape') and emg.ndim == 2 and emg.shape[0] <= 16:
            config.ENABLE_DIFFERENTIAL_MODE = True
            config.num_channels = 16

        cycles = tm.extract_complete_cycles(recording)
        print(f"  Extracted {len(cycles)} cycles")

        for i, cycle in enumerate(cycles):
            cycle_num += 1
            emg = cycle['emg']
            gt = cycle['gt']
            n_channels = emg.shape[0]
            n_samples = emg.shape[1]
            time_axis = np.arange(n_samples) / config.FSAMP

            # Compute offset from median peak-to-peak
            all_ranges = [np.ptp(emg[ch, :]) for ch in range(n_channels)]
            offset_step = np.median(all_ranges) * 1.5 if np.median(all_ranges) > 0 else 1.0

            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            rec_name = Path(rec_file).stem

            yticks = []
            ytick_labels = []
            for ch in range(n_channels):
                offset = (n_channels - ch) * offset_step  # CH1 at top, slot 0 for GT
                ax.plot(time_axis, emg[ch, :] + offset, 'k-', linewidth=0.5, alpha=0.9)
                yticks.append(offset)
                ytick_labels.append(f"CH{ch + 1}")

            # GT as bottom trace
            gt_scaled = gt * offset_step * 0.8
            ax.plot(time_axis, gt_scaled, 'r-', linewidth=1.5, alpha=0.7, label='GT')
            yticks.append(0)
            ytick_labels.append("GT")

            # Add OPEN/CLOSED text labels on constant GT regions
            # GT=0 → OPEN, GT=1 → CLOSED (regardless of protocol mode)
            gt_binary = (gt > 0.5).astype(int)
            gt_diff = np.diff(gt_binary, prepend=gt_binary[0])
            transitions = np.where(gt_diff != 0)[0]
            region_starts = np.concatenate([[0], transitions])
            region_ends = np.concatenate([transitions, [n_samples]])
            gt_label_y = offset_step * 0.9
            for rs, re in zip(region_starts, region_ends):
                if re - rs < config.FSAMP * 0.3:
                    continue  # Skip very short transition regions
                mid_time = (rs + re) / 2 / config.FSAMP
                region_val = gt_binary[min(rs + (re - rs) // 2, n_samples - 1)]
                state_label = "CLOSED" if region_val == 1 else "OPEN"
                state_color = '#FF5722' if region_val == 1 else '#2196F3'
                ax.text(mid_time, gt_label_y, state_label,
                       ha='center', va='bottom', fontsize=8, fontweight='bold',
                       color=state_color, alpha=0.8)

            # Audio cue markers with distinct colors
            close_cue_idx = cycle.get('close_cue_idx')
            open_cue_idx = cycle.get('open_cue_idx')
            if close_cue_idx is not None:
                ax.axvline(close_cue_idx / config.FSAMP, color='#FF5722', linestyle='--',
                          linewidth=1.5, alpha=0.7, label='Close cue')
            if open_cue_idx is not None:
                ax.axvline(open_cue_idx / config.FSAMP, color='#2196F3', linestyle='--',
                          linewidth=1.5, alpha=0.7, label='Open cue')

            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_labels, fontsize=7)
            ax.set_xlabel('Time (s)')
            ax.set_title(f'{rec_name} — Cycle {cycle_num}')
            ax.set_xlim(0, n_samples / config.FSAMP)
            ax.set_ylim(-offset_step * 0.5, (n_channels + 1) * offset_step)
            ax.legend(loc='upper left', fontsize=7, framealpha=0.8)
            ax.grid(True, alpha=0.3, axis='x')
            fig.tight_layout()

    print(f"\nTotal cycles plotted: {cycle_num}")


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent.parent

    if PLOT_MODE in ("cycles", "cycles_overlap"):
        if not RECORDING_FILES:
            print("ERROR: RECORDING_FILES is empty. Set recording file paths for cycles mode.")
        elif PLOT_MODE == "cycles_overlap":
            plot_cycles_stacked_overlap(RECORDING_FILES, base_path)
        else:
            plot_cycles_stacked(RECORDING_FILES, base_path)
        plt.show()
        raise SystemExit(0)

    templates_path = base_path / TEMPLATES_FILE

    print(f"Loading templates: {templates_path}")
    with open(templates_path, 'rb') as f:
        data = pickle.load(f)

    raw_open = data['templates_open']
    raw_closed = data['templates_closed']
    print(f"  OPEN:   {len(raw_open)} templates, shape {raw_open[0].shape}")
    print(f"  CLOSED: {len(raw_closed)} templates, shape {raw_closed[0].shape}")

    is_diff = (
        data.get('differential_mode', False)
        or data.get('metadata', {}).get('differential_mode', False)
        or raw_open[0].shape[0] <= 16
    )
    if is_diff:
        config.ENABLE_DIFFERENTIAL_MODE = True
        config.num_channels = 16
        print(f"  Differential mode detected (16 channels)")

    if PLOT_STYLE == "stacked_overlap":
        plot_fn = plot_templates_stacked_overlap
    elif PLOT_STYLE == "stacked":
        plot_fn = plot_templates_stacked
    else:
        plot_fn = plot_templates_overlap

    if PLOT_MODE == "raw":
        print(f"\nPlotting raw EMG templates ({PLOT_STYLE} style)...")
        plot_fn(
            raw_open, raw_closed,
            title=f"{Path(TEMPLATES_FILE).stem} (raw)",
            x_label="Time (ms)",
            y_label="Amplitude (µV)",
        )
    else:
        feat_name = PLOT_MODE
        feat_fn = FEATURES[feat_name]["function"]
        feat_display = FEATURES[feat_name]["name"]
        feat_unit = FEATURES[feat_name]["unit"]

        print(f"\nExtracting feature: {feat_display} ({feat_name})")

        feat_open = []
        for t in raw_open:
            windowed = sliding_window(t, config.window_length, config.increment)
            feat_open.append(feat_fn(windowed))

        feat_closed = []
        for t in raw_closed:
            windowed = sliding_window(t, config.window_length, config.increment)
            feat_closed.append(feat_fn(windowed))

        print(f"  Feature template shape: {feat_open[0].shape}")

        plot_fn(
            feat_open, feat_closed,
            title=f"{Path(TEMPLATES_FILE).stem} — {feat_display}",
            x_label="Window index",
            y_label=f"{feat_display} ({feat_unit})",
        )

    plt.show()
