"""
Template Metrics — Evaluate quality of a saved templates set.

Loads a templates .pkl file, extracts features, computes DTW pairwise
distances, separability ratio, silhouette scores, and plots a distance matrix.

Usage:
    python -m mindmove.model.templates_metrics
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mindmove.config import config
from mindmove.model.core.windowing import sliding_window
from mindmove.model.core.features.features_registry import FEATURES
from mindmove.model.template_study import (
    compute_template_metrics,
    plot_distance_matrix,
    print_metrics,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Templates file (relative to project root)
TEMPLATES_FILE = "data/recordings/patient S1/templates_mp_20260212_105426_guided_16cycles.pkl"
# TEMPLATES_FILE = "data/recordings/patient S1/templates_sd_20260206_121311_guided_4cycles.pkl"

# Feature to extract from raw templates before computing DTW distances
FEATURE = "rms"

# =============================================================================


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent.parent
    templates_path = base_path / TEMPLATES_FILE

    print("=" * 60)
    print("TEMPLATE METRICS")
    print("=" * 60)
    print(f"  File:    {TEMPLATES_FILE}")
    print(f"  Feature: {FEATURE}")
    print("=" * 60)

    # Load templates
    with open(templates_path, 'rb') as f:
        data = pickle.load(f)

    raw_open = data['templates_open']
    raw_closed = data['templates_closed']
    print(f"\n  Raw templates: {len(raw_open)} OPEN, {len(raw_closed)} CLOSED")
    print(f"  Template shape: {raw_open[0].shape}")

    # Detect differential mode
    n_ch = raw_open[0].shape[0]
    if n_ch <= 16:
        config.ENABLE_DIFFERENTIAL_MODE = True
        config.num_channels = 16
    config.active_channels = list(range(config.num_channels))

    # Extract features
    feat_fn = FEATURES[FEATURE]["function"]
    window_length = config.window_length
    increment = config.increment

    feat_open = [feat_fn(sliding_window(t, window_length, increment)) for t in raw_open]
    feat_closed = [feat_fn(sliding_window(t, window_length, increment)) for t in raw_closed]

    print(f"  Feature shape: {feat_open[0].shape}")

    # Compute metrics
    print("\n  Computing pairwise DTW distances...")
    metrics = compute_template_metrics(feat_open, feat_closed)

    # Print results
    print_metrics(metrics)

    # Per-template silhouette breakdown
    n_closed = len(feat_closed)
    n_open = len(feat_open)
    sil = metrics['per_template_silhouette']

    print("\n  Per-template silhouette scores:")
    print(f"  {'Template':<12} {'Class':<8} {'Silhouette':>10}")
    print(f"  {'-'*32}")
    for i in range(n_closed):
        marker = " <-- low" if sil[i] < 0.3 else ""
        print(f"  C{i+1:<11} {'CLOSED':<8} {sil[i]:>10.3f}{marker}")
    for i in range(n_open):
        marker = " <-- low" if sil[n_closed + i] < 0.3 else ""
        print(f"  O{i+1:<11} {'OPEN':<8} {sil[n_closed + i]:>10.3f}{marker}")

    # Plot distance matrix
    plot_distance_matrix(metrics)

    # Save figure
    save_path = templates_path.parent / f"metrics_{templates_path.stem}_{FEATURE}.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved to: {save_path}")

    import sys
    if "--no-show" not in sys.argv:
        plt.show()
