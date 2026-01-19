import matplotlib.pyplot as plt
from config import config
from mindmove.model.core.filtering import apply_rtfiltering, extract_envelope_rt, apply_filtering, extract_envelope
from mindmove.model.templates.data_loading import convert_gt_to_binary
import numpy as np

def data_list_plotting(data_list, label="str", gt_list= None, sd_long=False, sd_circ=False):
    for nrec in range(len(data_list)):
        # nrec=3
        emg_data = data_list[nrec] # 32ch x nsamp recording
        #emg_data_filt = apply_rtfiltering(emg_data, bandwidth=(40,500))
        # emg_data_filt = apply_filtering(emg_data, bandwidth=(40,500), notch=True)

        #emg_data_env = extract_envelope_rt(emg_data_filt)
        #emg_data_env = extract_envelope(emg_data_filt)
        nch, nsamp = emg_data.shape
        time_axis = np.arange(nsamp) / config.FSAMP
    #for ch in range(nch):
        ch = 20
        plt.figure()
        plt.plot(time_axis, emg_data[ch,:], label='raw')
        #plt.plot(time_axis, emg_data_filt[ch,:], label='filt')
        # plt.plot(time_axis, emg_data_env[ch,:], label='envelope')
        #plt.axhline(np.mean(emg_data_env[ch,:]),label='mean')
        if gt_list is not None:
            plt.plot(time_axis, gt_list[nrec] * np.max(emg_data[ch, :]), label='Ground Truth', alpha=0.6)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (uV)")
        plt.grid()
        plt.legend()
        plt.title(f'hand {label}, recording {nrec+1}, channel {ch+1}')
    plt.show()


def data_plotting_sd(data_list, data_list_filt, label="str", sd_long=False, sd_circ=False):
    for nrec in range(len(data_list)):
        # nrec=3
        emg_data = data_list[nrec] # 32ch x nsamp recording

        emg_data_filt_post = apply_filtering(emg_data, bandwidth=(40,500), notch=True)
        emg_data_filt_pre = data_list_filt[nrec]

        #emg_data_env = extract_envelope(emg_data_filt_pre)
        emg_data_env = extract_envelope_rt(emg_data_filt_pre)

        nch, nsamp = emg_data.shape
        time_axis = np.arange(nsamp) / config.FSAMP
        # for ch in range(nch):
        ch = 0
        plt.figure()
        plt.plot(time_axis, emg_data[ch,:], label='raw')
        # plt.plot(time_axis, emg_data_filt_post[ch,:], label='filt post spatial filter')
        plt.plot(time_axis, emg_data_filt_pre[ch,:], label='filt pre spatial filter')

        plt.plot(time_axis, emg_data_env[ch,:], label='envelope')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (uV)")
        plt.grid()
        plt.legend()
        if sd_long:
            plt.title(f'hand {label}, recording {nrec+1}, channels {ch+17}-{ch+1}')
        if sd_circ:
            if ch < 16:
                # bottom row
                lower = ch + 1
                upper = (ch + 1) % 16 + 1
            else:
                # top row
                ch_top = ch - 16
                lower = 17 + ch_top
                upper = 17 + ((ch_top + 1) % 16)

            plt.title(f'hand {label}, recording {nrec+1}, channels {upper}-{lower}')
    plt.show()


def distance_plotting(test_emg, test_gt, D_open, D_closed, threshold_open, threshold_closed, acctivation_label, increment=0.05, ch_to_plot=0):
    # For D (sliding windows)
    num_windows = len(D_open)
    time_D = np.arange(num_windows) * increment   # center of 150 ms window = 0.075 s
    # For GT
    nsamp = test_emg.shape[1]
    time_gt = np.arange(nsamp) / config.FSAMP

    # Convert test_gt dict/list to a binary array like before
    gt_binary = convert_gt_to_binary(test_gt, nsamp)

    # -------------------
    # Plot
    # -------------------
    fig, axs = plt.subplots(3, 1, figsize=(14,8), sharex=True)

    # Top: ground truth
    axs[0].plot(time_gt[:20*config.FSAMP], gt_binary[:20*config.FSAMP]* np.max(test_emg[ch_to_plot, :20*config.FSAMP]), color='tab:purple', label='Ground Truth')
    axs[0].plot(time_gt[:20*config.FSAMP], test_emg[ch_to_plot,:20*config.FSAMP], color='tab:blue', label=f'CH {ch_to_plot+1}')
    axs[0].set_title(f"Raw EMG (GT=1 for {acctivation_label})")
    axs[0].set_xlabel("Time [s]")

    axs[0].set_ylabel("uV")
    axs[0].grid()
    axs[0].legend()

    # Bottom: D and threshold

    axs[1].set_title(f"Distance between CLOSED templates and {acctivation_label} recording")
    axs[1].plot(time_D, D_closed, color='tab:orange', label='Distance D')
    axs[1].axhline(threshold_closed, color='tab:red', linestyle='--', label='Threshold')
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("DTW distance")
    axs[1].grid()
    axs[1].legend()

    axs[2].set_title(f"Distance between OPEN templates and {acctivation_label} recording")
    axs[2].plot(time_D, D_open, color='tab:orange', label='Distance D')
    axs[2].axhline(threshold_open, color='tab:red', linestyle='--', label='Threshold')
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("DTW distance")
    axs[2].grid()
    axs[2].legend()
    plt.tight_layout()