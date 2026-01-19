import os
import numpy as np
import matplotlib.pyplot as plt
from config import config
from processing.features import *
from processing.features.features_registry import FEATURES 

def _extract_feature_distribution(signal_list, feature_fn, wsize, wshift):
    """Return 1D array with all feature values from all signals."""
    


    nch = signal_list[0].shape[0]

    channel_values = [[] for _ in range(nch)]

    for sig in signal_list:
        # shape: (n_windows, n_channels, wsize)
        W = sliding_window(sig, wsize, wshift)
        # print(W.shape)

        # feature shape: (n_windows, n_channels)
        F = feature_fn(W)
        # print(F.shape)

        for ch in range(nch):
            channel_values[ch].append(F[:, ch])

    # concatenate each channel into 1 vector
    channel_values = [np.concatenate(ch_list) if len(ch_list) > 0 else np.array([]) 
                    for ch_list in channel_values]
    
    return channel_values
    

def feature_histogram(data_lists_dict, feature_name, plot_per_channel = True,
                      window_size=config.window_length, window_shift=config.increment):
    """ 
    data_lists_dict: dict like:
        {
            "openings": activations_open,
            "closings": activations_closed,
            "rest": non_act_open + non_act_closed
        }
        feature_name: feature name that has to be in the feature registry and itìs connected to the function name and pretty name

        plot_per_channel:

    """
    class_names = list(data_lists_dict.keys())
    n_classes = len(class_names)

    # feature info 
    feature_info = FEATURES[feature_name]
    feature_fn = feature_info["function"]
    pretty_name = feature_info["name"]
    # extract distributions
    distributions = {}
    for cname, signals in data_lists_dict.items():
        distributions[cname] = _extract_feature_distribution(
            signals, feature_fn, window_size, window_shift
            )
    n_channels = len(distributions[class_names[0]])

    if plot_per_channel:
        for ch in range(n_channels):
            if ch+1 not in config.dead_channels:
                plt.figure(figsize=(10, 6))
                for cname in class_names:
                    vals = distributions[cname][ch]
                    if vals.size > 0:
                        plt.hist(vals, bins="auto", alpha=0.5, density=True, label=f'{cname} (n={len(vals)})')
                plt.title(f"{pretty_name} – Channel {ch+1}")
                plt.xlabel(f"{pretty_name} {feature_info["unit"]}")
                plt.ylabel("Density")
                plt.legend()
                plt.grid(True)
    
    else:
        plt.figure(figsize=(12,4))

        y_height = 1

        for cname_idx, cname in enumerate(class_names):
            
            # compute median per channel
            medians = [
                np.median(distributions[cname][ch])
                for ch in range(n_channels)
                if ch+1 not in config.dead_channels and distributions[cname][ch].size > 0
            ]
            channels = [
                ch+1 for ch in range(n_channels)
                if ch+1 not in config.dead_channels and distributions[cname][ch].size > 0
            ]

            for ch_idx, median in enumerate(medians):
                x_jitter = np.random.uniform(-0.1, 0.1)
                plt.vlines(
                    x=median+x_jitter,
                    ymin=0, ymax=y_height,
                    color=f"C{cname_idx}",
                    linewidth=2
                )
                plt.plot(
                    median+x_jitter, y_height, 'o', color=f"C{cname_idx}", label=None
                )
                plt.text(
                    median, y_height + 0.01, str(channels[ch_idx]),
                    ha='center', va='bottom', fontsize=9
            )
        plt.xlabel(f"{pretty_name} {feature_info['unit']}")
        # plt.ylabel()
        plt.title(f"Medians of {pretty_name} per channel")
        plt.yticks([])  # remove y-axis ticks
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.legend(class_names)

            # plt.hist(
            #     medians,
            #     bins="auto",
            #     alpha=0.5,
            #     density=False,
            #     label=f"{cname} (n_ch={medians.size})"
            # )
            # plt.title(f"Medians of {pretty_name} across channels")
            # plt.xlabel(f"{pretty_name} {feature_info['unit']}")
            # plt.ylabel("Density")
            # plt.legend()
            # plt.grid(True)

    return
