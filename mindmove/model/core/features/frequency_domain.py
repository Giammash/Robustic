import numpy as np
from mindmove.config import config


# ========== FREQUENCY-DOMAIN FEATURES ==========

def compute_mean_frequency(windowed_signal):
    """Mean frequency (spectral centroid) per window per channel.

    Args:
        windowed_signal: (n_windows, n_ch, window_length)

    Returns:
        (n_windows, n_ch) mean frequency in Hz
    """
    n_windows, n_ch, win_len = windowed_signal.shape
    freqs = np.fft.rfftfreq(win_len, d=1.0 / config.FSAMP)
    result = np.zeros((n_windows, n_ch))

    for w in range(n_windows):
        for ch in range(n_ch):
            psd = np.abs(np.fft.rfft(windowed_signal[w, ch, :])) ** 2
            total_power = np.sum(psd)
            if total_power > 0:
                result[w, ch] = np.sum(freqs * psd) / total_power
            else:
                result[w, ch] = 0.0
    return result


def compute_median_frequency(windowed_signal):
    """Median frequency (50% cumulative spectral power) per window per channel.

    Args:
        windowed_signal: (n_windows, n_ch, window_length)

    Returns:
        (n_windows, n_ch) median frequency in Hz
    """
    n_windows, n_ch, win_len = windowed_signal.shape
    freqs = np.fft.rfftfreq(win_len, d=1.0 / config.FSAMP)
    result = np.zeros((n_windows, n_ch))

    for w in range(n_windows):
        for ch in range(n_ch):
            psd = np.abs(np.fft.rfft(windowed_signal[w, ch, :])) ** 2
            total_power = np.sum(psd)
            if total_power > 0:
                cum_power = np.cumsum(psd)
                idx = np.searchsorted(cum_power, total_power / 2.0)
                idx = min(idx, len(freqs) - 1)
                result[w, ch] = freqs[idx]
            else:
                result[w, ch] = 0.0
    return result


def compute_spectral_entropy(windowed_signal):
    """Spectral entropy (normalized) per window per channel.

    Higher values indicate more uniform spectral distribution (noise-like),
    lower values indicate concentrated spectral energy (tonal).

    Args:
        windowed_signal: (n_windows, n_ch, window_length)

    Returns:
        (n_windows, n_ch) spectral entropy (0 to 1)
    """
    n_windows, n_ch, win_len = windowed_signal.shape
    result = np.zeros((n_windows, n_ch))
    n_bins = win_len // 2 + 1
    log_n = np.log2(n_bins) if n_bins > 1 else 1.0

    for w in range(n_windows):
        for ch in range(n_ch):
            psd = np.abs(np.fft.rfft(windowed_signal[w, ch, :])) ** 2
            total_power = np.sum(psd)
            if total_power > 0:
                p = psd / total_power
                p = p[p > 0]  # avoid log(0)
                result[w, ch] = -np.sum(p * np.log2(p)) / log_n
            else:
                result[w, ch] = 0.0
    return result
