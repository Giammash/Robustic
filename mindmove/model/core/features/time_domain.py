import numpy as np
from scipy.stats import skew, kurtosis


# ========== TIME-DOMAIN FEATURES ==========

def compute_rms(windowed_signal):
    """Root Mean Square - overall signal energy."""
    return np.sqrt(np.mean(windowed_signal**2, axis=2))

def compute_waveform_length(windowed_signal):
    return np.sum(np.abs(np.diff(windowed_signal, axis=2)), axis=2)

def compute_wamp(windowed_signal, threshold=5):
    """
    Compute the Willison Amplitude for each window.

    The Willison Amplitude (WAMP) counts how many 
    times the absolute difference between consecutive 
    samples exceeds a given threshold.
    """
    diff = np.abs(np.diff(windowed_signal, axis=2))
    wamp = np.sum(diff >= threshold, axis=2)
    return wamp

def compute_myop(windowed_signal, threshold=10):
    """
    The Myopulse Percentage Rate (MYOP) quantifies
    the percentage of time (or samples) within a 
    window where the EMG signal's absolute value
    exceeds a certain threshold.
    
    the output is a percentage
    """
    active = np.abs(windowed_signal) >= threshold
    myop = 100 * np.mean(active, axis=2)
    return myop

def compute_mav(windowed_signal):
    """
    Compute MAV for each windowed signal.
    """
    mav = np.mean(np.abs(windowed_signal), axis=2)
    return mav

def compute_iemg(windowed_signal):
    """
    Integrated EMG - sum of absolute values.
    Related to muscle activation level.
    """
    return np.sum(np.abs(windowed_signal), axis=2)

def compute_ssi(windowed_signal):
    """
    Simple Square Integral - sum of squared values.
    Similar to energy, more sensitive to outliers.
    """
    return np.sum(windowed_signal**2, axis=2)

def compute_variance(windowed_signal):
    """Compute variance for each window."""
    variance = np.var(windowed_signal, axis=2)
    return variance

# Standard Deviation
def compute_std(windowed_signal):
    """Compute standard deviation for each window."""
    std = np.std(windowed_signal, axis=2)
    return std

def compute_log_detector(windowed_signal):
    """
    Log Detector - exponential of average log absolute value.
    Useful for pattern recognition.
    """
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    log_abs = np.log(np.abs(windowed_signal) + epsilon)
    return np.exp(np.mean(log_abs, axis=2))

def compute_ssc(windowed_signal, threshold=0):
    """
    Slope Sign Change - number of times slope changes sign.
    Indicates frequency content, similar to ZCR.
    """
    num_windows, num_channels, window_size = windowed_signal.shape
    ssc = np.zeros((num_windows, num_channels))
    
    for win_idx in range(num_windows):
        for ch_idx in range(num_channels):
            signal = windowed_signal[win_idx, ch_idx, :]
            # Count sign changes in first derivative
            diff1 = np.diff(signal)
            count = 0
            for i in range(len(diff1) - 1):
                if (diff1[i] * diff1[i+1] < 0) and (abs(diff1[i] - diff1[i+1]) >= threshold):
                    count += 1
            ssc[win_idx, ch_idx] = count
    
    return ssc

def compute_zero_crossings(windowed_signal):
    """
    Compute number of Zero Crossings for each window.
    ( more crossings -> higher frequency)
    """
    num_windows, num_channels, window_size = windowed_signal.shape
    zcr = np.zeros((num_windows, num_channels))

    for win_idx in range(num_windows):
        for ch_idx in range(num_channels):
            signal = windowed_signal[win_idx, ch_idx, :]
            # Count sign changes
            crossings = np.sum(np.diff(np.sign(signal)) != 0)
            zcr[win_idx, ch_idx] = crossings

    return zcr

def compute_v_order(windowed_signal, v=2):
    """
    V-order - generalized absolute moment.
    v=2 gives variance, v=3 gives thirs moment, etc.
    """
    return np.mean(np.abs(windowed_signal**v, axis=2))


def compute_temporal_moment(windowed_signal, order=3):
    """
    Temporal Moment - statistical moment of the signal.
    order=3: skewness-related, order=4: kurtosis-related
    """
    return np.mean(windowed_signal**order, axis=2)

def compute_skewness(windowed_signal):
    """
    Skewness - measure of asymmetry of distribution.
    Positive = right-skewed, Negative = left-skewed.
    """
    return skew(windowed_signal, axis=2)

def compute_kurtosis(windowed_signal):
    """
    Kurtosis - measure of "tailedness" of distribution.
    High = heavy tails (outliers), Low = light tails.
    """
    return kurtosis(windowed_signal, axis=2)

def compute_mad(windowed_signal):
    """
    Median Absolute Deviation - robust measure of variability.
    Less sensitive to outliers than standard deviation.
    """
    median = np.median(windowed_signal, axis=2, keepdims=True)
    mad = np.median(np.abs(windowed_signal - median), axis=2)
    return mad

def compute_cardinality(windowed_signal, threshold=10):
    """
    Cardinality - number of times signal exceeds threshold.
    Similar to MYOP but counts instead of percentage.
    """
    return np.sum(np.abs(windowed_signal) > threshold, axis=2)