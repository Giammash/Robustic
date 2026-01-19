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


def compute_threshold(templates, s=1, active_channels=None):
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
            print(f"alignment cost between template {i+1} and {j+1}: {alignment_cost}")

            distances[n] = alignment_cost
            n += 1

    D = sum(distances)
    print(f"std_dev = {np.std(distances)}")
    print(f"mean = {D/N}")
    print(f"s = {s}")
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

    Returns
    -------
    aggregated_distance : float
        The aggregated DTW distance based on the selected method.
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

    return aggregated_distance
