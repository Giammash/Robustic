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


class RealTimeEMGFilter:
    """
    Real-time EMG filter with state preservation for streaming data.

    Uses rtfilter library to maintain filter states between consecutive
    data blocks, ensuring continuous filtering without edge artifacts.

    Filter chain: Bandpass (removes DC + high freq noise) -> Notch (removes powerline harmonics)
    """

    def __init__(self, n_channels: int = 32, fs: int = None,
                 lowcut: float = None, highcut: float = None,
                 notch_freq: float = None, notch_harmonics: int = None):
        """
        Initialize the real-time EMG filter.

        Args:
            n_channels: Number of EMG channels (default: 32)
            fs: Sampling frequency in Hz (default: from config)
            lowcut: High-pass cutoff frequency in Hz (default: from config)
            highcut: Low-pass cutoff frequency in Hz (default: from config)
            notch_freq: Powerline frequency for notch filter (default: from config)
            notch_harmonics: Number of harmonics to filter (default: from config)
        """
        # Use config values as defaults
        self.fs = fs if fs is not None else config.FSAMP
        self.lowcut = lowcut if lowcut is not None else config.FILTER_LOWCUT
        self.highcut = highcut if highcut is not None else config.FILTER_HIGHCUT
        self.notch_freq = notch_freq if notch_freq is not None else config.NOTCH_FREQ
        self.notch_harmonics = notch_harmonics if notch_harmonics is not None else config.NOTCH_HARMONICS
        self.n_channels = n_channels

        # Create filter instances
        self._init_filters()

        self._is_initialized = True

    def _init_filters(self):
        """Initialize bandpass and notch filters using rtfilter."""
        # Bandpass filter
        filter_params_bp = {
            "order": config.FILTER_ORDER,
            "lowcut": self.lowcut,
            "highcut": self.highcut,
            "fs": self.fs,
        }
        self.bandpass = Butterworth(self.n_channels, FilterType.Bandpass, filter_params_bp)

        # Notch filter for powerline harmonics
        # Generate list of frequencies: 50, 100, 150, ... up to Nyquist
        notch_freqs = []
        for i in range(1, self.notch_harmonics + 1):
            freq = self.notch_freq * i
            if freq < self.fs / 2:  # Must be below Nyquist
                notch_freqs.append(freq)

        filter_params_notch = {
            "center_freq": notch_freqs,
            "fs": self.fs,
        }
        self.notch = Butterworth(self.n_channels, FilterType.Notch, filter_params_notch)

    def reset(self):
        """
        Reset filter states. Call when starting a new streaming session
        to avoid transients from previous data.
        """
        # Reinitialize filters to reset internal states
        self._init_filters()
        self._is_warmed_up = False
        self._warmup_samples_needed = int(0.1 * self.fs)  # 100ms warmup
        self._warmup_samples_received = 0

    def is_warmed_up(self) -> bool:
        """Check if filter has processed enough samples to be past transient period."""
        return getattr(self, '_is_warmed_up', False)

    def get_warmup_progress(self) -> float:
        """Get warmup progress as percentage (0-100)."""
        if self.is_warmed_up():
            return 100.0
        needed = getattr(self, '_warmup_samples_needed', 200)
        received = getattr(self, '_warmup_samples_received', 0)
        return min(100.0, (received / needed) * 100)

    def filter(self, data: np.ndarray) -> np.ndarray:
        """
        Filter a block of EMG data, maintaining state between calls.

        The filter maintains internal state between calls, so consecutive
        blocks are filtered as if they were one continuous stream.

        Note on transients: The first ~50-100ms after reset() will have
        transient artifacts as the filter "charges up". Use is_warmed_up()
        to check if the filter has processed enough samples.

        Args:
            data: EMG data array of shape (n_channels, n_samples)

        Returns:
            Filtered EMG data of same shape as input
        """
        if data.size == 0:
            return data

        # Ensure correct shape
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_samples = data.shape[1]

        # Track warmup progress
        if not getattr(self, '_is_warmed_up', False):
            self._warmup_samples_received = getattr(self, '_warmup_samples_received', 0) + n_samples
            warmup_needed = getattr(self, '_warmup_samples_needed', int(0.1 * self.fs))
            if self._warmup_samples_received >= warmup_needed:
                self._is_warmed_up = True
                print(f"[FILTER] Warmup complete ({self._warmup_samples_received} samples, "
                      f"{self._warmup_samples_received/self.fs*1000:.0f}ms)")

        # Apply filter chain: bandpass -> notch
        filtered = self.bandpass.filter(data)
        filtered = self.notch.filter(filtered)

        return filtered

    def get_filter_description(self) -> str:
        """Return a human-readable description of filter settings."""
        return f"{self.lowcut}-{self.highcut} Hz + {self.notch_freq} Hz notch"