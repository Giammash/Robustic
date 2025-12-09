import numpy as np
# import sys
# sys.path.append(r"C:\Users\alber\Desktop\TESI\Python\mindmove-framework-main\Real-Time-Filter")
# # from rtfilter.butterworth import Butterworth, FilterType
from rtfilter.butterworth import Butterworth, FilterType

from scipy.signal import welch, butter, filtfilt, cheby1, iirnotch
from mindmove.config import config




def apply_filtering(emg, bandwidth, notch = True):
    """
    Apply bandpass and notch filtering to EMG signal.
    emg : 32ch x nsamp recording
    """
    emg_filt = np.zeros_like(emg)

    nch, nsamp = emg.shape
    b, a = cheby1(6, 0.5, [bandwidth[0]/config.fNy, bandwidth[1]/config.fNy], btype='bandpass')
    for ch in range(nch):
        emg_filt[ch, :] = filtfilt(b, a, emg[ch, :])

    if notch:
        notch_freqs = [50, 100, 150, 200, 250, 300, 350, 400]
        for ch in range(nch):
            for f0 in notch_freqs:
                b, a = iirnotch(f0, 25, fs=config.FSAMP) #default Q = 25
                emg_filt[ch, :] = filtfilt(b, a, emg_filt[ch, :])

    return emg_filt

def apply_rtfiltering(emg, bandwidth=(20,500)):
    nch, nsamp = emg.shape
    # emg_filt = np.zeros_like(emg)

    # Initialize Butterworth filter
    filter_type = FilterType.Bandpass
    filter_params = {
        "order": 4,
        "lowcut": bandwidth[0],
        "highcut": bandwidth[1],
        "fs": config.FSAMP,
    }
    bandpass = Butterworth(nch, filter_type, filter_params)

    # Initialize Notch Filter
    filter_type = FilterType.Notch
    filter_params = {
        "center_freq": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700],
        "fs": config.FSAMP
    }
    notch = Butterworth(nch, filter_type, filter_params)

    # apply filters
    # if bandpass + notch
    # emg_filt_band = bandpass.filter(emg)
    # emg_filt = notch.filter(emg_filt_band)

    # if only notch 
    emg_filt = notch.filter(emg)


    return emg_filt



def extract_envelope(emg):
    """Extract EMG envelope using rectification and low-pass filter."""
    nch, nsamp = emg.shape
    emg_envelope = np.abs(emg)
    cutoff = 12
    order = 4
    b, a = butter(order, cutoff / (config.fNy), btype='low')

    for ch in range(nch):
        emg_envelope[ch,:] = filtfilt(b, a, emg_envelope[ch,:])
    return emg_envelope

def extract_envelope_rt(emg):
    """Extract EMG envelope using rectification and low-pass filter. In real time"""
    nch, nsamp = emg.shape
    emg_envelope = np.abs(emg)

    # Initialize Butterworth filter
    filter_type = FilterType.Lowpass
    filter_params = {
        "order": 4,
        "cutoff": 12,
        "fs": config.FSAMP,
    }
    lowpass = Butterworth(nch, filter_type, filter_params)

    emg_envelope_filt = lowpass.filter(emg_envelope)
    return emg_envelope_filt