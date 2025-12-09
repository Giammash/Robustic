import numpy as np

def spatial_filter(emg, dead_channels, interpolate=False):
    """
    spatial filter for the 32 electrodes EMG bracelet
    (17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)
    ( 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16)
    
    inputs:
    - emg: 32ch * nsamp matrix
    - dead_channels: list of the 1-indexed numbers of the channels not working or corrupted
    - interpolate: set True to substitute the corrupted signal with the interpolation of the twho adjacent signals
    
    outputs:
    - sd_long: spatial derivative longitudinal, top - bottom (16 x nsamp)
    - sd_circ: spatial derivative radial (2 row x 16 x nsamp)
    - eventually 
    """
    nch, nsamp = emg.shape
    dead_ch = [ch - 1 for ch in dead_channels] # zero-indexed
    M = np.zeros((2,16,nsamp))

    for ch in range(nch):
        if ch in dead_ch:
            if interpolate:
                continue #add interpolation
            else:
                continue # zeros in the dead channels
        else:
            if ch < 16:
                M[0, ch, :] = emg[ch, :] # bottom row
            else:
                M[1, ch-16, :] = emg[ch, :] # top row

    # spatial filters
    sd_long = M[1,:,:] - M[0,:,:] # Single Differential on longitudinal direction
    # [ (17- 1) (18- 2) (19- 3) ... (31-15) (32-16) ]

    sd_circ = M[:, 1:, :] - M[:, :-1, :] # Single Differential on adjacent columns ( 2 x 15 x nsamp )
    wrap = M[:, 0:1, :] - M[:, -1:, :] 
    sd_circ = np.concatenate([sd_circ, wrap], axis=1) # ( 2 x 16 x nsamp )   
    # [ (18-17) (19-18) (20-19) ... (32-31) (17-32) ]
    # [ ( 2- 1) ( 3- 2) ( 4- 3) ... (16-15) ( 1-16) ] 
    sd_circ_flat = sd_circ.reshape(32, nsamp) # ( 32 x nsamp) 
    # [ ( 2- 1) ( 3- 2) ( 4- 3) ... (16-15) ( 1-16) (18-17) (19-18) (20-19) ... (32-31) (17-32) ] 

    # removal of derivatives that are using dead channels
    if not interpolate:
        for dc in dead_ch:
            if dc < 16:
                sd_long[dc,:] = 0
                #sd_sirc ? sd_circ_flat?
            else:
                sd_long[dc-16,:] = 0
                #sd_sirc ? sd_circ_flat?
            sd_circ_flat[dc,:] = 0
            sd_circ_flat[(dc-1) % 32] = 0

    return sd_long, sd_circ_flat
            