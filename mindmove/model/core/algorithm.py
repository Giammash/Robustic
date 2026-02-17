import numpy as np
from numba import njit
from tslearn.metrics import dtw_path
from mindmove.config import config
from mindmove.model.core.features.features_registry import FEATURES
from mindmove.model.core.features import *
from mindmove.model.core.windowing import sliding_window

# Optional GPUDTW import (requires pycuda or pyopencl)
try:
    from GPUDTW import cuda_dtw, opencl_dtw
    GPUDTW_AVAILABLE = True
except ImportError:
    GPUDTW_AVAILABLE = False


@njit(cache=True)
def _dtw_cosine_numba(t1: np.ndarray, t2: np.ndarray, active_channels: np.ndarray) -> float:
    """Numba-optimized DTW with cosine distance.

    Args:
        t1: First template, shape (n_windows, nch)
        t2: Second template, shape (n_windows, nch)
        active_channels: Array of active channel indices

    Returns:
        DTW alignment cost
    """
    N = t1.shape[0]
    M = t2.shape[0]

    # Cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    cost_mat[0, 1:] = np.inf
    cost_mat[1:, 0] = np.inf

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # Cosine distance for active channels
            t1_vec = t1[i - 1, active_channels]
            t2_vec = t2[j - 1, active_channels]

            dot_product = 0.0
            norm1 = 0.0
            norm2 = 0.0
            for k in range(len(active_channels)):
                dot_product += t1_vec[k] * t2_vec[k]
                norm1 += t1_vec[k] * t1_vec[k]
                norm2 += t2_vec[k] * t2_vec[k]

            norm1 = np.sqrt(norm1)
            norm2 = np.sqrt(norm2)
            dist = 1.0 - dot_product / (norm1 * norm2 + 1e-8)

            # DTW recurrence
            penalty = min(cost_mat[i - 1, j - 1], cost_mat[i - 1, j], cost_mat[i, j - 1])
            cost_mat[i, j] = dist + penalty

    return cost_mat[N, M]




def dtw(t1, t2, active_channels=None):
    """compute the alignment cost between two templates
        Args:
        t1 (n_windows x nch): _description_
        t2 (n_windows x nch): _description_
        active_channels: list of active channel indices (0-indexed). If None, uses config.active_channels

        output:
        alignment_cost : float. value of the DTW distance between the two 1 second templates
    """
    if active_channels is None:
        active_channels = config.active_channels

    N, nch = t1.shape
    M, _ = t2.shape
    # print(f"t1 shape: {t1.shape}")
    # print(f"t2 shape: {t2.shape}")

    cost_mat = np.zeros((N+1,M+1))
    cost_mat[0, 1:] = np.inf
    cost_mat[1:, 0] = np.inf

    traceback_mat = np.zeros((N,M), dtype=int)
    for i in range(1, N+1):
        for j in range(1, M+1):

            ######## Euclidean distance
            #dist = np.linalg.norm(t1[[i-1], [active_channels]] - t2[[j-1], [active_channels]], axis=0) # to obtain a distance for every channel
            # dist = np.linalg.norm(t1[i-1, active_channels] - t2[j-1, active_channels], axis=0)

            ######## Cosine distance
            num = np.dot(t1[i-1, active_channels], t2[j-1, active_channels])
            den = (np.linalg.norm(t1[i-1, active_channels]) * np.linalg.norm(t2[j-1, active_channels]) + 1e-8)
            dist = 1 - num / den

            penalty = [
                cost_mat[i-1, j-1], # match
                cost_mat[i-1, j], # insertion
                cost_mat[i, j-1], # deletion
            ]
            i_penalty = np.argmin(penalty)
            cost_mat[i, j] = dist + penalty[i_penalty]
            traceback_mat [i-1, j-1] = i_penalty

    cost_mat = cost_mat[1:, 1:]
    alignment_cost = cost_mat[-1, -1]
    return alignment_cost #, (path[::-1], cost_mat)


def dtw_numba(t1, t2, active_channels=None):
    """Compute DTW distance using numba-optimized implementation.

    Uses the same cosine distance metric as the original dtw() function,
    but with JIT compilation for significant speedup.

    Args:
        t1 (np.ndarray): First template, shape (n_windows, nch)
        t2 (np.ndarray): Second template, shape (n_windows, nch)
        active_channels: list of active channel indices (0-indexed). If None, uses config.active_channels

    Returns:
        float: DTW distance between the two templates
    """
    if active_channels is None:
        active_channels = config.active_channels
    active_channels_arr = np.array(active_channels, dtype=np.int64)
    return _dtw_cosine_numba(
        t1.astype(np.float64),
        t2.astype(np.float64),
        active_channels_arr
    )


def dtw_tslearn(t1, t2, active_channels=None):
    """Compute DTW distance using tslearn's dtw_path (Euclidean distance).

    This uses the same approach as the paper's author.
    NOTE: Uses Euclidean distance, so requires a model trained with tslearn.
    Do not mix with models trained using cosine distance (numba/original).

    Args:
        t1 (np.ndarray): First template, shape (n_windows, nch)
        t2 (np.ndarray): Second template, shape (n_windows, nch)
        active_channels: list of active channel indices (0-indexed). If None, uses config.active_channels

    Returns:
        float: DTW distance between the two templates
    """
    if active_channels is None:
        active_channels = config.active_channels

    # Extract only active channels
    t1_active = t1[:, active_channels].astype(np.float64)
    t2_active = t2[:, active_channels].astype(np.float64)

    # Use tslearn's dtw_path with Euclidean distance (default)
    _, distance = dtw_path(t1_active, t2_active)

    return distance


def dtw_gpudtw(t1, t2, active_channels=None, use_cuda=True):
    """Compute DTW distance using GPU-accelerated GPUDTW library.

    GPUDTW works on 1D sequences, so for multivariate data we compute
    DTW per channel and sum the distances. This is done efficiently by
    batching all channels in a single GPU call.

    NOTE: Uses Euclidean distance. Models trained with cosine distance
    (numba/original) may not work well with this implementation.

    Args:
        t1 (np.ndarray): First template, shape (n_windows, nch)
        t2 (np.ndarray): Second template, shape (n_windows, nch)
        active_channels: list of active channel indices (0-indexed). If None, uses config.active_channels
        use_cuda (bool): If True, use CUDA. If False, use OpenCL.

    Returns:
        float: DTW distance between the two templates (sum over channels)
    """
    if not GPUDTW_AVAILABLE:
        raise ImportError(
            "GPUDTW is not available. Install it with: pip install GPUDTW\n"
            "Also requires pycuda (NVIDIA) or pyopencl (AMD/Intel)."
        )

    if active_channels is None:
        active_channels = config.active_channels

    # Extract only active channels and transpose to (n_channels, n_windows)
    t1_active = t1[:, active_channels].astype(np.float32).T  # (n_ch, n_win)
    t2_active = t2[:, active_channels].astype(np.float32).T  # (n_ch, n_win)

    # GPUDTW computes pairwise distances between all sequences in S and T
    # S shape: (n_sequences_S, sequence_length)
    # T shape: (n_sequences_T, sequence_length)
    # Output: (n_sequences_S, n_sequences_T)
    #
    # We pass channels as sequences, so we get DTW distance for each channel pair.
    # We want the diagonal (channel i of t1 vs channel i of t2).

    if use_cuda:
        distance_matrix = cuda_dtw(t1_active, t2_active)
    else:
        distance_matrix = opencl_dtw(t1_active, t2_active)

    # Sum the diagonal elements (channel-wise DTW distances)
    total_distance = np.trace(distance_matrix)

    return total_distance


def compute_dtw(t1, t2, active_channels=None):
    """Wrapper function to compute DTW using the configured implementation.

    Uses numba-optimized version by default for best performance with identical
    results to the original implementation. Options:
    - USE_GPUDTW=True: GPU-accelerated (requires CUDA/OpenCL)
    - USE_TSLEARN_DTW=True: tslearn library (Euclidean distance)
    - USE_NUMBA_DTW=True: numba JIT-compiled (fastest CPU, cosine distance)
    - All False: original pure Python implementation

    Args:
        t1 (np.ndarray): First template, shape (n_windows, nch)
        t2 (np.ndarray): Second template, shape (n_windows, nch)
        active_channels: list of active channel indices (0-indexed). If None, uses config.active_channels

    Returns:
        float: DTW distance between the two templates
    """
    if getattr(config, 'USE_GPUDTW', False):
        return dtw_gpudtw(t1, t2, active_channels=active_channels)
    elif getattr(config, 'USE_TSLEARN_DTW', False):
        return dtw_tslearn(t1, t2, active_channels=active_channels)
    elif getattr(config, 'USE_NUMBA_DTW', True):
        return dtw_numba(t1, t2, active_channels=active_channels)
    else:
        return dtw(t1, t2, active_channels=active_channels)


def compute_dtw_multivariate(feature_arrays_1, feature_arrays_2, active_channels=None):
    """Multivariate DTW: one shared warping path, per-feature cosine distances.

    Computes DTW using the sum of cosine distances across multiple features
    at each cell, then decomposes the total cost into per-feature contributions
    along the optimal path.

    Args:
        feature_arrays_1: list of np.ndarray, each (n_windows, n_ch) — one per feature
        feature_arrays_2: list of np.ndarray, each (n_windows, n_ch) — same features
        active_channels: list of active channel indices (0-indexed). If None, uses config.active_channels

    Returns:
        dict with:
            'total': float — total DTW distance (sum of all feature contributions)
            'per_feature': list of float — DTW cost contribution per feature along optimal path
    """
    if active_channels is None:
        active_channels = config.active_channels
    active_channels = np.array(active_channels)

    n_features = len(feature_arrays_1)
    N = feature_arrays_1[0].shape[0]
    M = feature_arrays_2[0].shape[0]

    # Build cost matrix: sum of cosine distances across features
    cost_mat = np.full((N + 1, M + 1), np.inf)
    cost_mat[0, 0] = 0.0

    # Per-feature distance at each cell (for decomposition along path)
    feat_dist = np.zeros((N, M, n_features))

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cell_cost = 0.0
            for f in range(n_features):
                v1 = feature_arrays_1[f][i - 1, active_channels]
                v2 = feature_arrays_2[f][j - 1, active_channels]
                dot = np.dot(v1, v2)
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                d = 1.0 - dot / (n1 * n2 + 1e-8)
                feat_dist[i - 1, j - 1, f] = d
                cell_cost += d

            penalty = min(cost_mat[i - 1, j - 1], cost_mat[i - 1, j], cost_mat[i, j - 1])
            cost_mat[i, j] = cell_cost + penalty

    total = cost_mat[N, M]

    # Traceback optimal path to decompose per-feature costs
    path = []
    i, j = N - 1, M - 1
    path.append((i, j))
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [
                (cost_mat[i, j], i - 1, j - 1),      # match
                (cost_mat[i, j + 1], i - 1, j),       # insertion
                (cost_mat[i + 1, j], i, j - 1),       # deletion
            ]
            _, i, j = min(candidates, key=lambda x: x[0])
        path.append((i, j))

    # Sum per-feature distances along path
    per_feature = np.zeros(n_features)
    for (pi, pj) in path:
        per_feature += feat_dist[pi, pj, :]

    return {
        'total': total,
        'per_feature': per_feature.tolist(),
    }


def compute_threshold(templates, s=1, active_channels=None, verbose=False):
    """
    Compute threshold from template inter-distances.

    Parameters
    ----------
    templates : list of np.ndarray
        List of emg templates already windowed and feature extracted (e.g. : rms)
        Each element has shape num_windows x nch
    s : float
        Number of standard deviations away from the mean distance between samples allowed
    active_channels : list or None
        List of active channel indices (0-indexed). If None, uses config.active_channels
    verbose : bool
        If True, print every pairwise distance (default: False)

    Returns
    -------
    mean_distance : float
    std_distance : float
    threshold : float
    """
    n_elements = len(templates)
    # num_windows, nch = templates[0].shape # assumes all the templates have the same shape

    D = 0 # D is the sum of distances of all unique template combinations
    N = ( n_elements * (n_elements - 1) ) / 2 # number of unique combinations
    distances = np.zeros(int(N))
    n = 0 # counter of unique combinations n (0:N-1)

    for i in range(len(templates)):
        for j in range(i+1, len(templates)):
            # (i, j) is a unique pair
            t1 = templates[i]
            t2 = templates[j]
            # alignment_cost, _ = dtw(t1,t2)
            alignment_cost = compute_dtw(t1, t2, active_channels=active_channels)
            if verbose:
                print(f"alignment cost between template {i+1} and {j+1}: {alignment_cost}")

            distances[n] = alignment_cost
            n += 1

    D = sum(distances)
    threshold = D/N + s*np.std(distances)

    return D/N, np.std(distances), threshold

def tune_thresholds(mean_distance, std_distance, s=1):
    """
    mean_distance : mean distance computed on the training set
    std_distance : std deviation of the distances computed on the training set
    s : number of standard deviations away from the mean distance allowed
    """
    threshold = mean_distance + s*std_distance
    return threshold


def compute_threshold_with_aggregation(
    templates,
    active_channels=None,
    distance_aggregation="avg_3_smallest",
    verbose=False,
):
    """
    Compute intra-class threshold using per-template distance aggregation.

    Matches the methodology used in Template Study: for each template, compute
    distances to all other same-class templates, aggregate (e.g. avg_3_smallest),
    then compute mean/std of those aggregated values. This produces statistics
    consistent with online prediction's distance aggregation.

    Parameters
    ----------
    templates : list of np.ndarray
        List of feature-extracted templates (n_windows x n_channels).
    active_channels : list or None
        Active channel indices (0-indexed). If None, uses config.active_channels.
    distance_aggregation : str
        "avg_3_smallest", "minimum", or "average".
    verbose : bool
        If True, print per-template aggregated distances.

    Returns
    -------
    mean_distance : float
    std_distance : float
    threshold : float (mean + 1*std, for compatibility)
    """
    n = len(templates)
    if n < 2:
        return 0.0, 0.0, 0.0

    # Build distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_dtw(templates[i], templates[j], active_channels=active_channels)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    def _aggregate(distances):
        if len(distances) == 0:
            return 0.0
        if distance_aggregation == "minimum":
            return np.min(distances)
        elif distance_aggregation == "avg_3_smallest":
            n_smallest = min(3, len(distances))
            return np.mean(np.sort(distances)[:n_smallest])
        else:  # "average"
            return np.mean(distances)

    # For each template, aggregate distances to all other same-class templates
    per_template = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        dists = np.array([dist_matrix[i, j] for j in others])
        agg = _aggregate(dists)
        per_template.append(agg)
        if verbose:
            print(f"  Template {i+1}: aggregated distance = {agg:.4f}")

    per_template = np.array(per_template)
    mean_distance = float(np.mean(per_template))
    std_distance = float(np.std(per_template))
    threshold = mean_distance + 1.0 * std_distance

    return mean_distance, std_distance, threshold


def compute_cross_class_distances_with_aggregation(
    templates_class_a,
    templates_class_b,
    active_channels=None,
    distance_aggregation="avg_3_smallest",
):
    """
    Compute cross-class distances using per-template aggregation.

    For each template in class A, compute distances to ALL templates in class B,
    aggregate (e.g. avg_3_smallest), producing one value per template.
    Returns both directional stats (A→B, B→A) separately, matching
    Template Study's inter_closed_to_open / inter_open_to_closed.

    Parameters
    ----------
    templates_class_a : list of np.ndarray
        Templates from first class (e.g., OPEN).
    templates_class_b : list of np.ndarray
        Templates from second class (e.g., CLOSED).
    active_channels : list or None
        Active channel indices. If None, uses config.active_channels.
    distance_aggregation : str
        "avg_3_smallest", "minimum", or "average".

    Returns
    -------
    dict with keys:
        "a_to_b": {"mean": float, "std": float, "per_template": np.ndarray}
        "b_to_a": {"mean": float, "std": float, "per_template": np.ndarray}
        "combined_mean": float (mean of all directional aggregated values)
        "combined_std": float
    """
    n_a = len(templates_class_a)
    n_b = len(templates_class_b)

    # Build cross-class distance matrix (n_a x n_b)
    cross_matrix = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            cross_matrix[i, j] = compute_dtw(
                templates_class_a[i], templates_class_b[j],
                active_channels=active_channels
            )

    def _aggregate(distances):
        if len(distances) == 0:
            return 0.0
        if distance_aggregation == "minimum":
            return np.min(distances)
        elif distance_aggregation == "avg_3_smallest":
            n_smallest = min(3, len(distances))
            return np.mean(np.sort(distances)[:n_smallest])
        else:  # "average"
            return np.mean(distances)

    # A→B: for each template in A, aggregate distances to all B templates
    per_template_a_to_b = []
    for i in range(n_a):
        dists = cross_matrix[i, :]
        per_template_a_to_b.append(_aggregate(dists))
    per_template_a_to_b = np.array(per_template_a_to_b)

    # B→A: for each template in B, aggregate distances to all A templates
    per_template_b_to_a = []
    for j in range(n_b):
        dists = cross_matrix[:, j]
        per_template_b_to_a.append(_aggregate(dists))
    per_template_b_to_a = np.array(per_template_b_to_a)

    all_aggregated = np.concatenate([per_template_a_to_b, per_template_b_to_a])

    return {
        "a_to_b": {
            "mean": float(np.mean(per_template_a_to_b)),
            "std": float(np.std(per_template_a_to_b)),
            "per_template": per_template_a_to_b,
        },
        "b_to_a": {
            "mean": float(np.mean(per_template_b_to_a)),
            "std": float(np.std(per_template_b_to_a)),
            "per_template": per_template_b_to_a,
        },
        "combined_mean": float(np.mean(all_aggregated)),
        "combined_std": float(np.std(all_aggregated)),
    }


def compute_cross_class_distances(templates_class_a, templates_class_b, active_channels=None):
    """
    Compute DTW distances between templates of two different classes.

    This is used for intelligent threshold preset computation. By computing
    the distances between OPEN and CLOSED templates, we can determine
    how separable the classes are and set thresholds accordingly.

    Parameters
    ----------
    templates_class_a : list of np.ndarray
        List of templates from first class (e.g., OPEN).
        Each element has shape (n_windows, n_channels).
    templates_class_b : list of np.ndarray
        List of templates from second class (e.g., CLOSED).
        Each element has shape (n_windows, n_channels).
    active_channels : list or None
        List of active channel indices (0-indexed). If None, uses config.active_channels.

    Returns
    -------
    mean_distance : float
        Mean of all cross-class distances.
    std_distance : float
        Standard deviation of all cross-class distances.
    distances : np.ndarray
        Array of all pairwise cross-class distances.
    """
    distances = []

    for t_a in templates_class_a:
        for t_b in templates_class_b:
            dist = compute_dtw(t_a, t_b, active_channels=active_channels)
            distances.append(dist)

    distances = np.array(distances)
    return np.mean(distances), np.std(distances), distances


def compute_threshold_presets(mean_intra, std_intra, mean_cross, std_cross):
    """
    Compute threshold presets based on intra-class and cross-class statistics.

    Returns 4 presets optimized for different use cases:
    1. Current (Intra-class): Standard method, s=1
    2. Cross-class Validation: Midpoint between intra and cross-class means
    3. Conservative (No False Opens): Strict threshold to prevent unwanted state changes
    4. Safety Margin (50%): Threshold at 50% between intra and cross-class means

    Parameters
    ----------
    mean_intra : float
        Mean distance within the class (from compute_threshold).
    std_intra : float
        Standard deviation of distances within the class.
    mean_cross : float
        Mean distance to templates of the opposite class.
    std_cross : float
        Standard deviation of cross-class distances.

    Returns
    -------
    dict
        Dictionary mapping preset names to their computed threshold values and
        corresponding s values (for UI display).

    Notes
    -----
    For GraspAgain project, the primary goal is minimizing false positives
    (unwanted hand openings during grasp). The "conservative" preset is
    designed specifically for this use case.
    """
    presets = {}

    # 1. Mid-Gap (Default): Midpoint between intra upper tail and inter lower tail
    # threshold = ((mean_intra + std_intra) + (mean_cross - std_cross)) / 2
    threshold_midgap = ((mean_intra + std_intra) + (mean_cross - std_cross)) / 2
    s_midgap = (threshold_midgap - mean_intra) / std_intra if std_intra > 0 else 1.0
    presets["mid_gap"] = {
        "threshold": threshold_midgap,
        "s": s_midgap,
        "name": "Mid-Gap (Default)",
        "description": "Midpoint between intra+std and inter-std (center of gap)"
    }

    # 2. Intra-class: Standard method, s=1
    # threshold = mean_intra + 1 * std_intra
    threshold_current = mean_intra + 1.0 * std_intra
    s_current = 1.0
    presets["current"] = {
        "threshold": threshold_current,
        "s": s_current,
        "name": "Intra-class (mean+std)",
        "description": "Standard method: mean + 1*std within class"
    }

    # 3. Cross-class Validation: Midpoint between intra and cross-class means
    # This gives equal weight to both within-class consistency and between-class separation
    threshold_cross = (mean_intra + mean_cross) / 2
    # Compute equivalent s value: threshold = mean_intra + s*std_intra
    # s = (threshold - mean_intra) / std_intra
    s_cross = (threshold_cross - mean_intra) / std_intra if std_intra > 0 else 1.0
    presets["cross_class"] = {
        "threshold": threshold_cross,
        "s": s_cross,
        "name": "Cross-class Midpoint",
        "description": "Midpoint between intra-class and cross-class means"
    }

    # 3. Conservative (No False Opens): Strict threshold
    # Set threshold close to the cross-class mean minus one standard deviation
    # This makes it harder to trigger a state change (requires very close match)
    threshold_conservative = mean_cross - 1.0 * std_cross
    # Ensure it's at least as strict as the current threshold
    threshold_conservative = max(threshold_conservative, threshold_current)
    s_conservative = (threshold_conservative - mean_intra) / std_intra if std_intra > 0 else 1.0
    presets["conservative"] = {
        "threshold": threshold_conservative,
        "s": s_conservative,
        "name": "Conservative (No False Triggers)",
        "description": "Strict threshold to prevent unwanted state changes"
    }

    # 4. Safety Margin (50%): Threshold at 50% between intra and cross-class means
    # More conservative than cross-class midpoint, but not as strict as conservative
    threshold_safety = mean_intra + 0.5 * (mean_cross - mean_intra)
    s_safety = (threshold_safety - mean_intra) / std_intra if std_intra > 0 else 1.0
    presets["safety_margin"] = {
        "threshold": threshold_safety,
        "s": s_safety,
        "name": "Safety Margin (50%)",
        "description": "50% between intra-class mean and cross-class mean"
    }

    return presets


def compute_per_template_statistics(templates, active_channels=None, n_worst=3):
    """
    Compute per-template distance statistics to identify outliers.

    For each template, computes its average distance to all other templates.
    Templates with high average distance may be poor quality or inconsistent.

    Parameters
    ----------
    templates : list of np.ndarray
        List of templates already windowed and feature extracted.
        Each element has shape (n_windows, n_channels).
    active_channels : list or None
        List of active channel indices (0-indexed). If None, uses config.active_channels.
    n_worst : int
        Number of worst/best pairs and templates to identify (default: 3).

    Returns
    -------
    dict with keys:
        'per_template_avg': list of float - Average distance for each template
        'per_template_max': list of float - Max distance for each template
        'per_template_min': list of float - Min distance for each template
        'worst_indices': list of int - Indices of worst templates (1-indexed for display)
        'best_indices': list of int - Indices of best templates (1-indexed for display)
        'overall_mean': float - Mean of all pairwise distances
        'overall_std': float - Std of all pairwise distances
        'distance_matrix': np.ndarray - Full distance matrix
        'worst_pairs': list of tuples - (i, j, distance) for most dissimilar pairs (0-indexed)
        'best_pairs': list of tuples - (i, j, distance) for most similar pairs (0-indexed)
        'quartiles': np.ndarray - [min, Q1, median, Q3, max] of all pairwise distances
        'consistency_score': float - Coefficient of variation (std/mean), lower is better
        'outliers': list of tuples - (template_idx, avg_dist, sigma) for outliers (1-indexed)
    """
    n_templates = len(templates)
    if n_templates < 2:
        return {
            'per_template_avg': [0.0] * n_templates,
            'per_template_max': [0.0] * n_templates,
            'per_template_min': [0.0] * n_templates,
            'worst_indices': [],
            'best_indices': [],
            'overall_mean': 0.0,
            'overall_std': 0.0,
            'distance_matrix': np.zeros((n_templates, n_templates)),
            'worst_pairs': [],
            'best_pairs': [],
            'quartiles': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            'consistency_score': 0.0,
            'outliers': [],
        }

    # Distance matrix (only upper triangle computed)
    distance_matrix = np.zeros((n_templates, n_templates))

    for i in range(n_templates):
        for j in range(i + 1, n_templates):
            dist = compute_dtw(templates[i], templates[j], active_channels=active_channels)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric

    # Compute per-template statistics
    per_template_avg = []
    per_template_max = []
    per_template_min = []

    for i in range(n_templates):
        # Get distances to all OTHER templates (exclude diagonal = 0)
        other_distances = [distance_matrix[i, j] for j in range(n_templates) if i != j]
        per_template_avg.append(np.mean(other_distances))
        per_template_max.append(np.max(other_distances))
        per_template_min.append(np.min(other_distances))

    # Extract upper triangle for overall statistics
    all_distances = distance_matrix[np.triu_indices(n_templates, k=1)]
    overall_mean = np.mean(all_distances)
    overall_std = np.std(all_distances)

    # Identify worst templates (highest average distance = most different from others)
    sorted_by_avg = np.argsort(per_template_avg)[::-1]  # Descending
    worst_indices = [int(idx) + 1 for idx in sorted_by_avg[:n_worst]]  # 1-indexed for display

    # Identify best templates (lowest average distance = most consistent)
    best_indices = [int(idx) + 1 for idx in sorted_by_avg[-n_worst:][::-1]]  # 1-indexed for display

    # Extract all pairwise distances with indices for worst/best pairs
    pairs = []
    for i in range(n_templates):
        for j in range(i + 1, n_templates):
            pairs.append((i, j, distance_matrix[i, j]))

    # Sort by distance
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    worst_pairs = pairs_sorted[:n_worst]  # Highest distances
    best_pairs = pairs_sorted[-n_worst:][::-1]  # Lowest distances (reversed for ascending order)

    # Quartiles [min, Q1, median, Q3, max]
    quartiles = np.percentile(all_distances, [0, 25, 50, 75, 100])

    # Consistency score (coefficient of variation: std/mean, lower is better)
    consistency_score = overall_std / overall_mean if overall_mean > 0 else 0.0

    # Outlier detection: templates with avg distance > mean + 1.5*std
    per_template_avg_arr = np.array(per_template_avg)
    per_template_mean = np.mean(per_template_avg_arr)
    per_template_std = np.std(per_template_avg_arr)
    outlier_threshold = per_template_mean + 1.5 * per_template_std

    outliers = []
    if per_template_std > 0:  # Avoid division by zero
        for i, avg in enumerate(per_template_avg):
            if avg > outlier_threshold:
                sigma = (avg - per_template_mean) / per_template_std
                outliers.append((i + 1, avg, sigma))  # 1-indexed for display

    return {
        'per_template_avg': per_template_avg,
        'per_template_max': per_template_max,
        'per_template_min': per_template_min,
        'worst_indices': worst_indices,
        'best_indices': best_indices,
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'distance_matrix': distance_matrix,
        'worst_pairs': worst_pairs,
        'best_pairs': best_pairs,
        'quartiles': quartiles,
        'consistency_score': consistency_score,
        'outliers': outliers,
    }


def compute_distance_from_training_set_offline(test_recording, templates, feature_name = 'wl' , window_length=1, increment = 0.050):
    """
    test_recording : offling emg recording (has to be extracted for data)
    templates : list of emg templates already windowed and feature extracted (e.g. : wl (waveform length))
    so the list has n_templates element and each element has shape num_windows x nch
    """
    nch, nsamp = test_recording.shape
    window_samples = window_length * config.FSAMP
    incr_samples = increment * config.FSAMP
    

    num_new_templates = int((nsamp - window_samples) // incr_samples + 1)
    distance = []
    D = np.zeros(num_new_templates)

    for i in range(num_new_templates):
        start = int(i * incr_samples)
        end = int(start + window_samples)

        current_template = test_recording[:,start:end]
        distances_from_templates = np.zeros(len(templates))
        ######## PAY ATTENTION THE WINDOWS AND OVERLAP CHOSEN FOR THE **NEW** RECORDING
        t_test = sliding_window(current_template, window_size=int(0.096*config.FSAMP), window_shift=int(0.032*config.FSAMP))
        
        feature_info = FEATURES[feature_name]
        feature_fn = feature_info["function"]
        # if feature_name == 'rms':
        #     t_test_feature = compute_rms(t_test)
        # elif feature_name == 'wl':
        #     t_test_feature = compute_waveform_length(t_test)
        
        t_test_feature = feature_fn(t_test)

        for j in range(len(templates)):
            tj = templates[j]
            # print("Test template shape:", t_test_rms.shape)
            # print("Training template shape:", tj.shape)
            # distances_from_templates[j], _ = dtw(t_test_rms, tj)
            distances_from_templates[j] = compute_dtw(t_test_feature, tj)

        distance.append(distances_from_templates)
        D[i] = np.mean(distances_from_templates)
        if i % 100 == 0: print(f"distance between 1 sec window{i}/{num_new_templates} and templates: {D[i]}")
    
    return D, distance


 
def compute_distance_from_training_set_online(
        features_buffer,
        templates,
        active_channels=None,
        distance_aggregation="average",
        return_all_distances=False,
        ):
    """
    Compute DTW distance between current online features and stored templates.

    Parameters
    ----------
    features_buffer : np.ndarray
        Shape: n_windows x n_channels (the sliding window features from the online buffer)
    templates : list of np.ndarray
        Each template is already windowed and feature extracted: n_windows x n_channels
    active_channels : list or None
        List of active channel indices (0-indexed). If None, uses config.active_channels
    distance_aggregation : str
        Method for aggregating distances across templates:
        - "average": Mean of all distances (default)
        - "minimum": Minimum distance to any template
        - "avg_3_smallest": Average of the 3 smallest distances (more robust)
    return_all_distances : bool
        If True, also return the array of all individual template distances.
        Default is False for backwards compatibility.

    Returns
    -------
    aggregated_distance : float
        The aggregated DTW distance based on the selected method.
    all_distances : np.ndarray (only if return_all_distances=True)
        Array of distances to each template.
    """
    nwin, nch = features_buffer.shape

    distances = np.zeros(len(templates))

    for i, template in enumerate(templates):
        distances[i] = compute_dtw(features_buffer, template, active_channels=active_channels)

    # Aggregate distances based on the selected method
    if distance_aggregation == "minimum":
        aggregated_distance = np.min(distances)
    elif distance_aggregation == "avg_3_smallest":
        # Take average of 3 smallest distances (or all if less than 3 templates)
        n_smallest = min(3, len(distances))
        smallest_distances = np.sort(distances)[:n_smallest]
        aggregated_distance = np.mean(smallest_distances)
    else:  # "average" or default
        aggregated_distance = np.mean(distances)

    if return_all_distances:
        return aggregated_distance, distances.copy()
    return aggregated_distance


def compute_calibration_distances(
    emg: np.ndarray,
    gt: np.ndarray,
    templates_open: list,
    templates_closed: list,
    feature_name: str = 'wl',
    window_length: int = None,
    increment: int = None,
    buffer_duration_s: float = 1.0,
    dtw_interval_s: float = 0.05,
    active_channels: list = None,
    distance_aggregation: str = "average",
    active_channels_open: list = None,
    active_channels_closed: list = None,
):
    """
    Compute continuous DTW distances over a recording for offline calibration.

    Simulates real-time processing: slides a 1-second buffer over the recording
    and computes DTW distances at regular intervals (like online protocol).

    Parameters
    ----------
    emg : np.ndarray
        EMG data, shape (n_channels, n_samples).
    gt : np.ndarray
        Ground truth signal, shape (n_samples,). Values 0=OPEN, 1=CLOSED.
    templates_open : list of np.ndarray
        Feature-extracted OPEN templates, each (n_windows, n_channels).
    templates_closed : list of np.ndarray
        Feature-extracted CLOSED templates, each (n_windows, n_channels).
    feature_name : str
        Feature to use (default 'wl').
    window_length : int
        Window length for feature extraction in samples. If None, uses config.
    increment : int
        Overlap for feature extraction in samples. If None, uses config.
    buffer_duration_s : float
        Duration of the sliding buffer in seconds (default 1.0).
    dtw_interval_s : float
        Interval between DTW computations in seconds (default 0.05 = 50ms).
    active_channels : list or None
        Active channel indices. If None, uses config.active_channels.
        Used for both classes unless per-class channels are specified.
    distance_aggregation : str
        Method for aggregating distances ("average", "minimum", "avg_3_smallest").
    active_channels_open : list or None
        Per-class active channels for OPEN distance. Overrides active_channels for OPEN.
    active_channels_closed : list or None
        Per-class active channels for CLOSED distance. Overrides active_channels for CLOSED.

    Returns
    -------
    dict with keys:
        "timestamps": np.ndarray - Time of each DTW computation (seconds).
        "D_open": np.ndarray - Distance to OPEN templates at each timestamp.
        "D_closed": np.ndarray - Distance to CLOSED templates at each timestamp.
        "gt_at_dtw": np.ndarray - GT value at each DTW timestamp.
    """
    from mindmove.model.core.windowing import sliding_window

    # Use config defaults if not specified
    if window_length is None:
        window_length = config.window_length
    if increment is None:
        increment = config.increment
    if active_channels is None:
        active_channels = config.active_channels

    # Get feature function
    feature_info = FEATURES[feature_name]
    feature_fn = feature_info["function"]

    # Calculate sample counts
    n_channels, n_samples = emg.shape
    buffer_samples = int(buffer_duration_s * config.FSAMP)
    dtw_interval_samples = int(dtw_interval_s * config.FSAMP)

    # Storage for results
    timestamps = []
    D_open_list = []
    D_closed_list = []
    gt_at_dtw_list = []

    # Slide buffer over recording
    # Start when we have at least one full buffer
    start_idx = buffer_samples
    end_idx = n_samples

    current_idx = start_idx
    while current_idx <= end_idx:
        # Extract 1-second buffer ending at current_idx
        buffer_start = current_idx - buffer_samples
        buffer_end = current_idx
        emg_buffer = emg[:, buffer_start:buffer_end]

        # Extract features
        windowed = sliding_window(emg_buffer, window_length, increment)
        features = feature_fn(windowed)

        # Compute DTW to both template sets (use per-class channels if specified)
        ch_open = active_channels_open if active_channels_open is not None else active_channels
        ch_closed = active_channels_closed if active_channels_closed is not None else active_channels

        D_open = compute_distance_from_training_set_online(
            features, templates_open,
            active_channels=ch_open,
            distance_aggregation=distance_aggregation
        )
        D_closed = compute_distance_from_training_set_online(
            features, templates_closed,
            active_channels=ch_closed,
            distance_aggregation=distance_aggregation
        )

        # Get GT at this timestamp (use the value at the end of the buffer)
        gt_idx = min(buffer_end - 1, len(gt) - 1)
        gt_value = gt[gt_idx] if gt_idx >= 0 else 0

        # Store results
        timestamps.append(buffer_end / config.FSAMP)
        D_open_list.append(D_open)
        D_closed_list.append(D_closed)
        gt_at_dtw_list.append(gt_value)

        # Advance by DTW interval
        current_idx += dtw_interval_samples

    return {
        "timestamps": np.array(timestamps),
        "D_open": np.array(D_open_list),
        "D_closed": np.array(D_closed_list),
        "gt_at_dtw": np.array(gt_at_dtw_list),
    }


def find_plateau_thresholds(
    D_open: np.ndarray,
    D_closed: np.ndarray,
    gt: np.ndarray,
    confidence_k: float = 1.0,
    plateau_percentile: float = 10.0,
):
    """
    Find plateau values during state transitions and compute thresholds.

    During a true transition (e.g., OPEN->CLOSED), the distance to the target
    state templates (D_closed) drops and reaches a stable minimum (plateau).
    The threshold should be set just above this plateau to reliably detect
    transitions while avoiding false triggers.

    Parameters
    ----------
    D_open : np.ndarray
        Distance to OPEN templates over time.
    D_closed : np.ndarray
        Distance to CLOSED templates over time.
    gt : np.ndarray
        Ground truth at each DTW timestamp (0=OPEN, 1=CLOSED).
    confidence_k : float
        Multiplier for standard deviation to add to plateau value (default 1.0).
        threshold = plateau_mean + k * plateau_std
    plateau_percentile : float
        Percentile of distances to use as plateau (default 10th percentile).
        Lower values = more aggressive (closer to minimum).

    Returns
    -------
    dict with keys:
        "open_plateau": float - Mean distance during OPENING transitions plateau.
        "closed_plateau": float - Mean distance during CLOSING transitions plateau.
        "threshold_open": float - Computed threshold for OPEN detection.
        "threshold_closed": float - Computed threshold for CLOSED detection.
        "open_std": float - Std of OPEN plateau distances.
        "closed_std": float - Std of CLOSED plateau distances.
        "open_distances_in_state": np.ndarray - D_open when GT=1 (hand closed).
        "closed_distances_in_state": np.ndarray - D_closed when GT=0 (hand open).
    """
    # Ensure gt is 1D
    if gt.ndim > 1:
        gt = gt.flatten()

    # For OPEN detection: we look at D_open when GT=1 (hand is closed, checking for open)
    # The plateau is when D_open is at its minimum during closed state
    # (i.e., when we're about to transition to open or during active opening)
    closed_mask = gt > 0.5  # GT=1 means hand is CLOSED
    open_mask = gt <= 0.5   # GT=0 means hand is OPEN

    # D_open when hand is CLOSED (checking if should open)
    D_open_when_closed = D_open[closed_mask] if np.any(closed_mask) else np.array([])

    # D_closed when hand is OPEN (checking if should close)
    D_closed_when_open = D_closed[open_mask] if np.any(open_mask) else np.array([])

    # Find plateau values using percentile (more robust than minimum)
    if len(D_open_when_closed) > 0:
        # Use low percentile as plateau estimate
        open_plateau = np.percentile(D_open_when_closed, plateau_percentile)
        # Compute std of values near the plateau (bottom 25%)
        threshold_for_std = np.percentile(D_open_when_closed, 25)
        near_plateau_open = D_open_when_closed[D_open_when_closed <= threshold_for_std]
        open_std = np.std(near_plateau_open) if len(near_plateau_open) > 1 else 0.01
    else:
        open_plateau = 0.1
        open_std = 0.01

    if len(D_closed_when_open) > 0:
        # Use low percentile as plateau estimate
        closed_plateau = np.percentile(D_closed_when_open, plateau_percentile)
        # Compute std of values near the plateau (bottom 25%)
        threshold_for_std = np.percentile(D_closed_when_open, 25)
        near_plateau_closed = D_closed_when_open[D_closed_when_open <= threshold_for_std]
        closed_std = np.std(near_plateau_closed) if len(near_plateau_closed) > 1 else 0.01
    else:
        closed_plateau = 0.1
        closed_std = 0.01

    # Compute thresholds: plateau + k * std
    threshold_open = open_plateau + confidence_k * open_std
    threshold_closed = closed_plateau + confidence_k * closed_std

    return {
        "open_plateau": open_plateau,
        "closed_plateau": closed_plateau,
        "threshold_open": threshold_open,
        "threshold_closed": threshold_closed,
        "open_std": open_std,
        "closed_std": closed_std,
        "open_distances_in_state": D_open_when_closed,
        "closed_distances_in_state": D_closed_when_open,
    }


def find_transition_based_thresholds(
    D_open: np.ndarray,
    D_closed: np.ndarray,
    gt: np.ndarray,
    timestamps: np.ndarray,
    window_s: float = 0.5,
    confidence_k: float = 1.0,
):
    """
    Find thresholds by focusing on GT transition neighborhoods.

    The idea: During a real transition (e.g., OPEN->CLOSED), look at a small
    window around the GT change and find the minimum distance to the target
    class templates. This minimum is where the classifier should trigger.

    Parameters
    ----------
    D_open : np.ndarray
        Distance to OPEN templates over time.
    D_closed : np.ndarray
        Distance to CLOSED templates over time.
    gt : np.ndarray
        Ground truth at each DTW timestamp (0=OPEN, 1=CLOSED).
    timestamps : np.ndarray
        Timestamps corresponding to each distance measurement.
    window_s : float
        Half-window size in seconds to search around each transition (default 0.5s).
    confidence_k : float
        Multiplier for standard deviation: threshold = min + k * std.

    Returns
    -------
    dict with threshold values and transition info.
    """
    # Ensure gt is 1D
    if gt.ndim > 1:
        gt = gt.flatten()

    # Find GT transitions
    gt_diff = np.diff(gt)
    closing_transitions = np.where(gt_diff > 0.5)[0]  # GT goes 0->1 (OPEN to CLOSED)
    opening_transitions = np.where(gt_diff < -0.5)[0]  # GT goes 1->0 (CLOSED to OPEN)

    print(f"\n[THRESHOLD] Found {len(closing_transitions)} closing and {len(opening_transitions)} opening transitions")

    # Collect minimum D_closed values around closing transitions (OPEN->CLOSED)
    # During closing, D_closed should drop - we want the minimum
    min_d_closed_at_closing = []
    for idx in closing_transitions:
        trans_time = timestamps[idx] if idx < len(timestamps) else timestamps[-1]
        # Find samples within window_s of this transition
        mask = np.abs(timestamps - trans_time) <= window_s
        if np.any(mask):
            d_closed_in_window = D_closed[mask]
            # Filter out None values
            d_closed_valid = [d for d in d_closed_in_window if d is not None]
            if d_closed_valid:
                min_d = np.min(d_closed_valid)
                min_d_closed_at_closing.append(min_d)
                print(f"  Closing transition at {trans_time:.2f}s: min D_closed = {min_d:.4f}")

    # Collect minimum D_open values around opening transitions (CLOSED->OPEN)
    # During opening, D_open should drop - we want the minimum
    min_d_open_at_opening = []
    for idx in opening_transitions:
        trans_time = timestamps[idx] if idx < len(timestamps) else timestamps[-1]
        # Find samples within window_s of this transition
        mask = np.abs(timestamps - trans_time) <= window_s
        if np.any(mask):
            d_open_in_window = D_open[mask]
            # Filter out None values
            d_open_valid = [d for d in d_open_in_window if d is not None]
            if d_open_valid:
                min_d = np.min(d_open_valid)
                min_d_open_at_opening.append(min_d)
                print(f"  Opening transition at {trans_time:.2f}s: min D_open = {min_d:.4f}")

    # Compute thresholds from transition minimums
    if min_d_closed_at_closing:
        closed_mins = np.array(min_d_closed_at_closing)
        closed_mean = np.mean(closed_mins)
        closed_std = np.std(closed_mins) if len(closed_mins) > 1 else closed_mean * 0.1
        threshold_closed = closed_mean + confidence_k * closed_std
        print(f"\n[THRESHOLD] D_closed at transitions: mean={closed_mean:.4f}, std={closed_std:.4f}")
    else:
        threshold_closed = 0.15
        closed_mean = 0.1
        closed_std = 0.02
        print("[THRESHOLD] No closing transitions found, using default")

    if min_d_open_at_opening:
        open_mins = np.array(min_d_open_at_opening)
        open_mean = np.mean(open_mins)
        open_std = np.std(open_mins) if len(open_mins) > 1 else open_mean * 0.1
        threshold_open = open_mean + confidence_k * open_std
        print(f"[THRESHOLD] D_open at transitions: mean={open_mean:.4f}, std={open_std:.4f}")
    else:
        threshold_open = 0.15
        open_mean = 0.1
        open_std = 0.02
        print("[THRESHOLD] No opening transitions found, using default")

    print(f"\n[THRESHOLD] Computed thresholds: OPEN={threshold_open:.4f}, CLOSED={threshold_closed:.4f}")

    return {
        "threshold_open": threshold_open,
        "threshold_closed": threshold_closed,
        "open_transition_mins": min_d_open_at_opening,
        "closed_transition_mins": min_d_closed_at_closing,
        "open_mean": open_mean,
        "open_std": open_std,
        "closed_mean": closed_mean,
        "closed_std": closed_std,
        "n_opening_transitions": len(opening_transitions),
        "n_closing_transitions": len(closing_transitions),
    }


def compute_spatial_profiles(
    templates_raw: list,
    class_label: str = "",
) -> dict:
    """
    Compute per-class spatial reference profile and consistency weights from raw templates.

    For each template, takes the second half (the active portion after onset) and
    computes RMS per channel. Then across all templates:
    - ref_profile: mean per-channel RMS (the average spatial shape)
    - consistency: mean/std per channel (channels that fire reliably get high weight)
    - weights: consistency normalized to sum to 1

    Args:
        templates_raw: List of raw EMG templates, each shape (n_channels, n_samples)
        class_label: Label for printing ("open" or "closed")

    Returns:
        Dict with:
            - "ref_profile": np.ndarray (n_channels,) — mean RMS per channel
            - "weights": np.ndarray (n_channels,) — normalized consistency weights
            - "consistency": np.ndarray (n_channels,) — raw consistency (mean/std)
            - "per_template_rms": np.ndarray (n_templates, n_channels)
    """
    if not templates_raw:
        return None

    n_ch = templates_raw[0].shape[0]
    n_templates = len(templates_raw)

    # Compute per-channel RMS from second half of each template
    per_template_rms = np.zeros((n_templates, n_ch))
    for i, template in enumerate(templates_raw):
        n_samples = template.shape[1]
        half = n_samples // 2
        second_half = template[:, half:]  # (n_ch, n_samples/2)
        per_template_rms[i] = np.sqrt(np.mean(second_half ** 2, axis=1))

    # Mean and std across templates for each channel
    mean_rms = np.mean(per_template_rms, axis=0)  # (n_ch,)
    std_rms = np.std(per_template_rms, axis=0)     # (n_ch,)

    # Consistency = mean / std (like SNR: high = reliable)
    epsilon = 1e-10
    consistency = mean_rms / (std_rms + epsilon)

    # Normalize weights to sum to 1
    weights = consistency / (np.sum(consistency) + epsilon)

    if class_label:
        print(f"\n[SPATIAL] {class_label.upper()} reference profile ({n_templates} templates):")
        # Show top 5 most consistent channels
        sorted_idx = np.argsort(consistency)[::-1]
        for rank, ch in enumerate(sorted_idx[:5], 1):
            print(f"  {rank}. CH{ch+1}: mean_rms={mean_rms[ch]:.4f}, "
                  f"consistency={consistency[ch]:.2f}, weight={weights[ch]:.4f}")

    return {
        "ref_profile": mean_rms,
        "weights": weights,
        "consistency": consistency,
        "per_template_rms": per_template_rms,
    }


def compute_spatial_similarity(
    emg_buffer: np.ndarray,
    ref_profile: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Compute weighted cosine similarity between current EMG spatial pattern
    and a class reference profile.

    Args:
        emg_buffer: Current EMG buffer, shape (n_channels, n_samples)
        ref_profile: Reference spatial profile (mean RMS per channel), shape (n_channels,)
        weights: Consistency weights per channel, shape (n_channels,)

    Returns:
        Weighted cosine similarity (0 to 1, higher = better match)
    """
    # Compute per-channel RMS of current buffer
    current_rms = np.sqrt(np.mean(emg_buffer ** 2, axis=1))  # (n_ch,)

    # Normalize both to unit vectors
    current_norm = np.linalg.norm(current_rms)
    ref_norm = np.linalg.norm(ref_profile)

    if current_norm < 1e-10 or ref_norm < 1e-10:
        return 0.0  # No signal or no reference

    current_unit = current_rms / current_norm
    ref_unit = ref_profile / ref_norm

    # Weighted cosine similarity
    # Weight each channel's contribution by its consistency
    weighted_dot = np.sum(weights * current_unit * ref_unit)

    # Normalize by sqrt of weighted norms to keep in [0, 1] range
    weighted_norm_current = np.sqrt(np.sum(weights * current_unit ** 2))
    weighted_norm_ref = np.sqrt(np.sum(weights * ref_unit ** 2))

    if weighted_norm_current < 1e-10 or weighted_norm_ref < 1e-10:
        return 0.0

    similarity = weighted_dot / (weighted_norm_current * weighted_norm_ref)

    return float(np.clip(similarity, 0.0, 1.0))
