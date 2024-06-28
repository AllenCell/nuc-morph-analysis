import numpy as np
import pandas as pd
from scipy.signal import medfilt


# %%
def find_tp_outliers_by_volume(vol, thresh=0.10, pad_size=15, kernel=9):
    """
    This function is used to detect outliers in the volume of a track.
    These could be due to errors in the instance segmentation or tracking.

    The track is compared to a rolling median. At the beginning of the track,
    the volume array is padded with the first value. At the end of the track,
    the volume array is padded with the median of the last 3 frames. Preventing
    the real rapid growth phase from A-B from being called an outlier, while catching
    spikes and drops in volumes at mitosis at the end of the track.

    Parameters
    ----------
    vol : array-like
        Volume of the track.
    thresh : float
        Threshold for the outlier detection.
    pad_size : int
        Size of the padding.
    kernel : int
        Kernel size for the median filter.

    Returns
    -------
    outliers : array-like
        Locations of the outliers.
    """
    # pad the ends of the volume array
    vol_pad1 = np.pad(vol, (pad_size, 0), mode="edge")
    vol_pad2 = np.pad(vol_pad1, (0, pad_size), mode="median", stat_length=3)

    rolling_median_volume = medfilt(vol_pad2, kernel)[pad_size:-pad_size]
    difference = abs(vol / rolling_median_volume - 1)
    outliers = difference > thresh  # find locations of deviations above threshold
    outliers = find_breakdown_timepoint_outliers(vol, outliers)
    return outliers


def find_breakdown_timepoint_outliers(vol, outliers, thresh=0.05):
    # Convert the last 5 volume values to a pandas Series
    last_5_vol_series = pd.Series(vol[-5:])

    # Calculate the percent change from timepoint to timepoint
    percent_change = last_5_vol_series.pct_change()

    # Iterate over the last 5 percent changes
    for i in range(1, len(percent_change)):
        # Check if the absolute percent change is greater than threshold
        if abs(percent_change.iloc[i]) > thresh:
            # If it is, mark it as an outlier
            outliers[-5 + i] = True
    return outliers


def outlier_detection(df):
    """
    For each track in the dataframe that is longer than 5 frames, this function
    will check for volume outliers using the find_tp_outliers_by_volume function.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing all the tracks to be analyzed.

    Returns
    -------
    df : DataFrame
        The original DataFrame with a new column 'is_tp_outlier' added.
        This column contains a boolean value indicating whether a track is an outlier.
    """
    df = df.sort_values("index_sequence")
    outlier_dict = {}  # initialize dictionary to store outlier information
    for _, dft in df.groupby("track_id"):
        if len(dft) > 5:
            outliers = find_tp_outliers_by_volume(dft["volume"].values)
            outlier_dict.update(dict(zip(dft.index, outliers)))

    outlier_series = pd.Series(
        outlier_dict, name="is_tp_outlier"
    )  # Convert the dictionary to a Series
    df = df.join(outlier_series)  # Join the new Series to the original DataFrame
    df["is_tp_outlier"] = df["is_tp_outlier"].fillna(False)  # Fill any missing values with False
    return df
