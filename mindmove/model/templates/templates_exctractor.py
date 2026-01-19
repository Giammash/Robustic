from mindmove.model.core.features import *
from mindmove.model.core.windowing import sliding_window
from config import config
from mindmove.model.core.features.features_registry import FEATURES 

def extract_templates(activations_list, idxs, feature_name, wait_05_sec=True):
    """
    Extracts the one second templates from the activations list and pre-process them dividing them in windows (parameters of the windows in config.py)
    """
    templates_list = [activations_list[i] for i in idxs]
    windowed_templates_list = []
    feature_templates_list = []
    raw_templates_list = []
    if feature_name == "raw":
        for t in range(len(templates_list)):
        # templates_open[t] = templates_open[t][:,:config.template_nsamp]
            if wait_05_sec:
                templates_list[t] = templates_list[t][:,int(config.FSAMP/2) : int(config.FSAMP/2) + config.template_duration*config.FSAMP]
            else:
                templates_list[t] = templates_list[t][:, :config.template_duration*config.FSAMP]

        return templates_list

    else:

        feature_info = FEATURES[feature_name]
        feature_fn = feature_info["function"]
        
        for t in range(len(templates_list)):
            # templates_open[t] = templates_open[t][:,:config.template_nsamp]
            if wait_05_sec:
                templates_list[t] = templates_list[t][:,int(config.FSAMP/2) : int(config.FSAMP/2) + config.template_duration*config.FSAMP]
            else:
                templates_list[t] = templates_list[t][:, :config.template_duration*config.FSAMP]

            wt = sliding_window(templates_list[t], window_size=config.window_length, window_shift=config.increment)
            # print(wt.shape)
            windowed_templates_list.append(wt)
            
            feature = feature_fn(wt)
            # print(rms.shape)
            feature_templates_list.append(feature)

    return feature_templates_list


