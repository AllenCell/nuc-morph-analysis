from nuc_morph_analysis.lib.preprocessing import load_data, filter_data
from nuc_morph_analysis.analyses.height import calculate_features
import numpy as np


def get_colony_time_shift(df):
    """
    Get the colony time shift for the small, medium, and large baseline colonies. The time shift is calculated by finding the time delay
    between the mean height trajectories of the two colonies.

    Parameters
    ----------
    df: DataFrame
        The dataframe containing all datasets

    Returns
    -------
    time_lag_small_medium: int
        The time delay between the small and medium colonies in frames
    time_lag_medium_large: int
        The time delay between the medium and large colonies in frames
    """
    pixel_size = load_data.get_dataset_pixel_size("all_baseline")
    interval = load_data.get_dataset_time_interval_in_min("all_baseline")
    df_filtered = filter_data.all_timepoints_minimal_filtering(df)
    time_lag_small_medium = calculate_features.find_colony_time_alignment(
        df_filtered[df_filtered["colony"] == "small"],
        df_filtered[df_filtered["colony"] == "medium"],
        pixel_size,
        interval,
    )
    time_lag_medium_large = calculate_features.find_colony_time_alignment(
        df_filtered[df_filtered["colony"] == "medium"],
        df_filtered[df_filtered["colony"] == "large"],
        pixel_size,
        interval,
    )
    return time_lag_small_medium, time_lag_medium_large


def add_colony_time(df, time_lag_small_medium, time_lag_medium_large):
    """
    adds a column called: 'colony_time' to the dataset with the pre-calculated frame/time-shift depending on the colony name. 
    The colony time column contains time in frames.

    v2023 medium=+156, large=+156+140
    morflowgenesis v0.2.1 medium=+148, large=+148+140
    morflowgenesis v0.3.0 and bioRxiv medium=+123, large=+123+150
    
    Using height_percentile as height the shift is now:
    morflowgenesis v0.3.0 medium=+123, large=+123+148

    Parameters
    ------------
    df : DataFrame
        loaded dataframe for the manifest of interest
    data_name: string
        name of the manifest of interest

    Returns
    -----------
    df : DataFrame
        DataFrame with added column 'colony_time'
    """

    if (df.colony == "small").all():
        df["colony_time"] = df["index_sequence"]
    elif (df.colony == "medium").all():
        df["colony_time"] = df["index_sequence"] + time_lag_small_medium
    elif (df.colony == "large").all():
        df["colony_time"] = df["index_sequence"] + time_lag_small_medium + time_lag_medium_large
    else:
        print("colony_time cannot be generated due to alignment issues")

    return df


def add_colony_time_all_datasets(df):
    """
    adds a column called: 'colony_time' to the dataset and caclulates frame/time-shift depending on the colony name. The colony time column contains time in frames.

    Parameters
    ------------
    df: pandas.DataFrame
        Loaded dataframe for the manifest of interest. Must have column 'colony'

    Returns
    -----------
    DataFrame
        DataFrame with added column 'colony_time'
    """
    df["colony_time"] = np.nan
    time_lag_small_medium, time_lag_medium_large = get_colony_time_shift(df)

    for colony, df_dataset in df.groupby("colony"):
        df_dataset = add_colony_time(df_dataset, time_lag_small_medium, time_lag_medium_large)
        df.loc[df.colony == colony, "colony_time"] = df_dataset["colony_time"]

    return df
