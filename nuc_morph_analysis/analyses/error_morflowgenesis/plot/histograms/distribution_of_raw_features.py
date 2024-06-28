import matplotlib.pyplot as plt
import numpy as np
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot

fs = 15


def length_of_tracks(df):
    """
    This function takes a dataframe as input. It computes the length of each track in the dataframe,
    and plots the distributions of their durations. Because this is a fixed control dataset containing 20 frames,
    we should expect to see a peak at 20 frames. Originally there are a bunch of shorter tracks.
    We will remove those and just look at ones that are present for the full duration of the pseudo-timelapse
    in order to get a better measurement for the error in our segmentations.

    Parameters
    ----------
    df: dataframe
        fixed control dataframe

    Returns
    -------
    figure
    """

    length_list = []
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    for track_id, dft in df.groupby("track_id"):
        x = dft.index_sequence.to_numpy()
        x = len(x)
        length_list.append(x)

    plt.hist(length_list, bins=50, width=1.0)
    plt.xlim(0, 21)
    plt.ylabel("number of tracks", fontsize=fs)
    plt.xlabel("length of tracks (frames)", fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.tight_layout()
    save_and_show_plot(f"error_morflowgenesis/figures/distribution_of_track_lengths")


def median_track_features(df, pixel_size, feature_type):
    """
    This function takes a dataframe, a pixel size, and a feature type as input. It computes the median value
    of the specified feature for each track in the dataframe, and plots a histogram of these median values.
    The feature type can be one of 'height', 'surface_area', or 'volume'.

    Parameters
    ----------
    df: dataframe
        fixed control dataframe
    pixel_size: float
        pixel size returned by load_data.get_dataset_pixel_size('fixed_control')
    feature_type: string
        one of 'height', 'surface_area', or 'volume'

    Returns
    -------
    figure
    """

    feature_list = []
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    for _, dft in df.groupby("track_id"):
        if feature_type == "height":
            x = dft.height.to_numpy() * pixel_size
            plt.xlabel("Median height of tracks (µm)", fontsize=fs)
        elif feature_type == "surface_area":
            x = dft.mesh_sa.to_numpy() * pixel_size * pixel_size
            plt.xlabel("Median surface area of tracks (µm²)", fontsize=fs)
        elif feature_type == "volume":
            x = dft.volume.to_numpy() * pixel_size * pixel_size * pixel_size
            plt.xlabel("Median volume of tracks (µm³)", fontsize=fs)
        else:
            print("Invalid feature type")
            return
        x = np.median(x)
        feature_list.append(x)

    plt.hist(feature_list)
    plt.ylabel("number of tracks", fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.tight_layout()
    save_and_show_plot(f"error_morflowgenesis/figures/median_track_{feature_type}")
