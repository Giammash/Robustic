from mindmove.model.core.features.time_domain import *
from mindmove.model.core.features.frequency_domain import *
from mindmove.model.core.windowing import *

FEATURES = {

    "rms": {"function": compute_rms, "name": "RMS", "unit": "uV"},
    "mav": {"function": compute_mav, "name": "MAV", "unit": "uV"},
    "iemg": {"function": compute_iemg, "name": "Integrated EMG", "unit": "uV·samples"},
    "ssi": {"function": compute_ssi, "name": "Simple Square Integral", "unit": "uV²"},
    "log_detector": {"function": compute_log_detector, "name": "Log Detector", "unit": "uV"},
    
    # Time-domain: Variability
    "variance": {"function": compute_variance, "name": "Variance", "unit": "uV²"},
    "std": {"function": compute_std, "name": "Standard Deviation", "unit": "uV"},
    "mad": {"function": compute_mad, "name": "Median Absolute Deviation", "unit": "uV"},
    
    # Time-domain: Shape
    "skewness": {"function": compute_skewness, "name": "Skewness", "unit": "dimensionless"},
    "kurtosis": {"function": compute_kurtosis, "name": "Kurtosis", "unit": "dimensionless"},
    # # "v_order": {"function": compute_v_order, "name": "V-Order", "unit": "mV^v"},
    "temporal_moment": {"function": compute_temporal_moment, "name": "Temporal Moment", "unit": "uV^n"},
    
    # Time-domain: Frequency proxies
    "wl": {"function": compute_waveform_length, "name": "Waveform Length", "unit": "uV"},
    "zero_crossings": {"function": compute_zero_crossings, "name": "Zero Crossings", "unit": "count"},
    "ssc": {"function": compute_ssc, "name": "Slope Sign Change", "unit": "count"},
    "wamp": {"function": compute_wamp, "name": "Willison Amplitude", "unit": "count"},
    "myop": {"function": compute_myop, "name": "Myopulse %", "unit": "%"},
    "cardinality": {"function": compute_cardinality, "name": "Cardinality", "unit": "count"},

    # --- Frequency domain ---
   

    # --- Time-frequency domain ---

}