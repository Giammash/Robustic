import numpy as np

### Windowing

def sliding_window(signal, window_size, window_shift):
    """
    Create overlapping windows from signal.
    window_size : number of samples
    window_shift : number of samples
    """
    single_channel = False
    if signal.ndim == 1:
        # signal = signal[np.newaxis, :]
        single_channel = True
    
    if single_channel:
        nsamp = len(signal)
        num_windows = (nsamp - window_size) // window_shift + 1
        windowed_signal = np.zeros((num_windows,window_size))
        for i in range(num_windows):
            start = i * window_shift
            end = start + window_size
            windowed_signal[i, :] = signal[start:end]

    else:
        nch, nsamp = signal.shape
        num_windows = (nsamp - window_size) // window_shift + 1
        windowed_signal = np.zeros((num_windows, nch, window_size)) 
        for i in range(num_windows):
            start = i * window_shift
            end = start + window_size
            windowed_signal[i, :, :] = signal[:, start:end]
            
    return windowed_signal
