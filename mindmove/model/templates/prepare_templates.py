import os
import pickle
import numpy as np
from mindmove.config import config
from mindmove.model.core.algorithm import compute_threshold
from mindmove.model.core.windowing import sliding_window
from mindmove.model.core.features.features_registry import FEATURES

def load_templates_from_folder(folder_path, feature_name='wl', template_type='raw'):
    """
    Load all template files from a folder and extract features.
    
    Parameters
    ----------
    folder_path : str
        Path to folder containing .pkl template files (should be either the folder 'templates_open' or 'templates_closed').
    feature_name : str
        Name of feature to extract (default: 'wl')
    template_type='raw' : str
        Type of template stored in the files: 'raw' for raw EMG data, 'features' for pre-extracted features.
        
    Returns
    -------
    list of np.ndarray
        List of feature-extracted templates, each shape (n_windows, n_channels)
    """
    templates = []

    if template_type not in ['raw', 'features']:
        raise ValueError(f"Invalid template_type: {template_type}. Must be 'raw' or 'features'.")
    if template_type == 'features':
        folder_path = os.path.join(folder_path, 'features')
    elif template_type == 'raw':
        folder_path = os.path.join(folder_path, 'raw')

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Template folder not found: {folder_path}")
    
    # extract the pickle file contained in the desired folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'rb') as f:
                template_data = pickle.load(f)  # list of templates of shape: 
                                                # (n_channels, n_samples) if raw, (n_windows, n_channels) if features
            
            if template_type == 'raw':
                # extract features from raw data
                feature_extracted_templates = []
                for template in template_data:
                    # template shape: (n_channels, n_samples)
                    windows = sliding_window(
                        template,
                        window_samples=config.window_length,
                        incr_samples=config.increment,
                    )  # shape: (n_windows, n_channels, window_samples)
                    
                    n_windows, n_channels, _ = windows.shape
                    features = np.zeros((n_windows, n_channels))
                    
                    feature_info = FEATURES[feature_name]
                    feature_fn = feature_info["function"]
                    
                    for ch in range(n_channels):
                        for w in range(n_windows):
                            features[w, ch] = feature_fn(windows[w, ch, :])
                    
                    feature_extracted_templates.append(features)  # shape: (n_windows, n_channels)
                
                templates.extend(feature_extracted_templates)
            else:
                templates.extend(template_data)  # already feature-extracted templates

    return templates


def prepare_model_from_recordings(    
    open_folder=config.OPEN_TEMPLATES_FOLDER,
    closed_folder=config.CLOSED_TEMPLATES_FOLDER,
    feature_name='wl',
    template_type= 'features',
    threshold_std_multiplier_open=1,
    threshold_std_multiplier_closed=1,
):
    """
    Prepare a complete model from recorded template folders.
    
    Parameters
    ----------
    open_folder : str
        Path to folder with "open" hand templates
    closed_folder : str
        Path to folder with "closed" hand templates
    feature_name : str
        Feature to use for DTW comparison
    threshold_std_multiplier : float
        Number of std deviations for threshold (default: 1)

        Returns
    -------
    dict
        Dictionary with templates and MEAN and STD ready to compute thresholds
    """

    print("Loading OPEN templates...")

    open_templates = load_templates_from_folder(
        open_folder, feature_name=feature_name, template_type=template_type
    )
    
    print("Loading CLOSED templates...")

    closed_templates = load_templates_from_folder(
        closed_folder, feature_name=feature_name, template_type=template_type
    )

    # COMPUTE THRESHOLDS
    print("Computing thresholds...")
    mean_open, std_open, threshold_base_open = compute_threshold(
        open_templates,
        s=threshold_std_multiplier_open,
    )
    mean_closed, std_closed, threshold_base_closed = compute_threshold(
        closed_templates,
        s=threshold_std_multiplier_closed,
    )
    model_dict = {
        "open_templates": open_templates,
        "closed_templates": closed_templates,
        "mean_open": mean_open,
        "std_open": std_open,
        "threshold_base_open": threshold_base_open,
        "mean_closed": mean_closed,
        "std_closed": std_closed,
        "threshold_base_closed": threshold_base_closed,
        "feature_name": feature_name,
    }
    return model_dict

def save_model_to_file(model_dict, file_path):
    """
    Save the model dictionary to a pickle file.
    
    Parameters
    ----------
    model_dict : dict
        Model dictionary to save
    file_path : str
        Path to save the pickle file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as f:
        pickle.dump(model_dict, f)
    
    print(f"Model saved to {file_path}")

def load_model_from_file(file_path):
    """
    Load the model dictionary from a pickle file.
    
    Parameters
    ----------
    file_path : str
        Path to the pickle file
    
    Returns
    -------
    dict
        Loaded model dictionary
    """
    with open(file_path, 'rb') as f:
        model_dict = pickle.load(f)
    print(f"Model loaded from {file_path}")
    return model_dict


# Main script to create a model offline

if __name__ == "__main__":
    model_dict = prepare_model_from_recordings(
        open_folder=config.OPEN_TEMPLATES_FOLDER,
        closed_folder=config.CLOSED_TEMPLATES_FOLDER,
        feature_name='wl',
        template_type='features',
        threshold_std_multiplier_open=1,
        threshold_std_multiplier_closed=1,
    )

    output_model_path = "data/models/dtw_model_0.pkl"
    save_model_to_file(model_dict, output_model_path)

    print("Model ready to be loaded in the GUI for online predictions.")


    
    