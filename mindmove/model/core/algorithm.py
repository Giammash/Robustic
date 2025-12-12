import numpy as np
from mindmove.config import config
from mindmove.model.core.features.features_registry import FEATURES
from mindmove.model.core.features import *
from mindmove.model.core.windowing import sliding_window




def dtw(t1,t2):
    """compute the alignment cost between two templates
        Args:
        t1 (n_windows x nch): _description_
        t2 (n_windows x nch): _description_

        output: 
        alignment_cost : float. value of the DTW distance between the two 1 second templates
    """
    N, nch = t1.shape
    M, _ = t2.shape

    cost_mat = np.zeros((N+1,M+1))
    cost_mat[0, 1:] = np.inf
    cost_mat[1:, 0] = np.inf

    traceback_mat = np.zeros((N,M), dtype=int)
    for i in range(1, N+1):
        for j in range(1, M+1):

            ######## Euclidean distance
            #dist = np.linalg.norm(t1[[i-1], [config.active_channels]] - t2[[j-1], [config.active_channels]], axis=0) # to obtain a distance for every channel
            # dist = np.linalg.norm(t1[i-1, config.active_channels] - t2[j-1, config.active_channels], axis=0) 

            ######## Cosine distance
            num = np.dot(t1[i-1, config.active_channels], t2[j-1, config.active_channels])
            den = (np.linalg.norm(t1[i-1, config.active_channels]) * np.linalg.norm(t2[j-1, config.active_channels]) + 1e-8)
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

    

def compute_threshold(templates, s = 1):
    """ 
    templates : list of emg templates already windowed and feature extracted (e.g. : rms)
    so the list has n_templates element and each element has shape num_windows x nch
    s : number of standard deviations away from the mean distance between samples allowed
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
            alignment_cost = dtw(t1,t2)
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
            distances_from_templates[j] = dtw(t_test_feature, tj)

        distance.append(distances_from_templates)
        D[i] = np.mean(distances_from_templates)
        if i % 100 == 0: print(f"distance between 1 sec window{i}/{num_new_templates} and templates: {D[i]}")
    
    return D, distance


 
def compute_distance_from_training_set_online(
        features_buffer, 
        templates, 
        # feature_name = 'wl', 
        # window_samples=config.template_nsamp, 
        # incr_samples = config.increment_dtw_samples,
        # feature_window_samples = config.window_length,
        # feature_window_increment = config.increment,
        ):
    """
    Compute DTW distance between current online features and stored templates.
    
    features_buffer : np.ndarray
        Shape: n_windows x n_channels (the sliding window features from the online buffer)
    templates : list of np.ndarray
        Each template is already windowed and feature extracted: n_windows x n_channels
    feature_name : str
        Name of the feature (for logging)
    
    Returns
    -------
    min_distance : float
        The minimum DTW distance across all templates.
    """
    nwin, nch = features_buffer.shape

    distances = np.zeros(len(templates))

    for i, template in enumerate(templates):
        
        distances[i] = dtw(features_buffer, template)
    
    # min_distance = np.min(distances)
    mean_distance = np.mean(distances)

    # best_template_index = np.argmin(distances)
    # best_distance = distances[best_template_index]

    # print(f"Window {i}: closest template = {best_template_index}, distance = {best_distance}")


    # return min_distance, mean_distance, distances, best_template_index, best_distance
    # return min_distance
    return mean_distance
