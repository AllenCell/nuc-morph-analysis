from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.analyses.lineage.get_features.family_functions import get_sibling_pairs
import numpy as np
import pandas as pd


def calculate_distance(row, pixel_size, time_at_1, time_at_2):
    """
    How far apart are any two tracks are at birth (ie. sister pairs) or
    distance at breakdown and formation (ie mother daughter pairs) in microns

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe
    pixel_size: Float
        The pixel size for the dataset
    time_at_1: Str
        "A" or "C"
    time_at_2: Str
        "A" or "C"

    Returns
    -------
    distance: Int
        distance in microns
    """
    dx = row[f"tid1_location_x_at_{time_at_1}"] - row[f"tid2_location_x_at_{time_at_2}"]
    dy = row[f"tid1_location_y_at_{time_at_1}"] - row[f"tid2_location_y_at_{time_at_2}"]
    return np.sqrt(dx**2 + dy**2) * pixel_size


def get_distance(df, time_at_1, time_at_2):
    """
    Calculate the distance between two tracks at two different time points

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe
    time_at_1: Str
        "A" or "C"
    time_at_2: Str
        "A" or "C"

    Returns
    -------
    df: Dataframe
        The dataframe with the distance column added
    """
    pixel_size = load_data.get_dataset_pixel_size("all_baseline")
    df["distance"] = df.apply(
        lambda row: calculate_distance(row, pixel_size, time_at_1, time_at_2), axis=1
    )
    return df


def add_difference_between(df):
    feature_list = ["time_at_B", "volume_at_B", "volume_at_C", "SA_at_B"]
    for feature in feature_list:
        df[f"difference_{feature}"] = abs(df[f"tid1_{feature}"] - df[f"tid2_{feature}"])
    return df


def sum_between(
    df, feature_list=["volume_at_A", "volume_at_B", "volume_at_C", "SA_at_B", "SA_at_C"]
):
    for feature in feature_list:
        df[f"sum_{feature}"] = df[f"tid1_{feature}"] + df[f"tid2_{feature}"]
    return df


def difference_half_vol_at_C_and_B(df):
    df["half_volume_at_C"] = df["tid2_volume_at_C"] / 2
    df["difference_half_vol_at_C_and_B"] = abs(df["half_volume_at_C"] - df["tid1_volume_at_B"])
    return df


def get_symmetric_and_asymmetric_sister_pairs(df):
    """
    Get a list of sister pairs that come from a division event that was symmetric and asymmetric.
    To capture the most similar aka symmetric division events, the difference in volume at B is set to less than 30 µm^3.
    To capture the most dissimilar aka asymmetric division events, the difference in volume at B is set to greater than 50 µm^3.
    The sister pairs found between these two thresholds are considered similar.

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe with single track features

    Returns
    -------
    df_pairs: Dataframe
        The dataframe with the sister pairs that come from symmetric and asymmetric division events
    """
    # Initialize an empty DataFrame
    df_pairs = pd.DataFrame(
        columns=[
            "tid1",
            "tid2",
            "pid",
            "small_sister",
            "symmetric",
            "similar",
            "asymmetric",
            "difference_at_B",
            "time_at_B_min",
        ]
    )

    sister_pairs = get_sibling_pairs(df)
    for tid1, tid2 in sister_pairs:
        dft1 = df[df.track_id == tid1]
        dft2 = df[df.track_id == tid2]
        volume_at_B_difference = abs(dft1.volume_at_B.values[0] - dft2.volume_at_B.values[0])
        time_at_B_difference = abs(dft1.time_at_B.values[0] * 60 - dft2.time_at_B.values[0] * 60)
        if (
            time_at_B_difference < 30
        ):  # if B varies largely, they might not be asymmetric, it's more likely that B is off.
            small_sister = tid1 if dft1.volume_at_B.values[0] < dft2.volume_at_B.values[0] else tid2
            symmetric = volume_at_B_difference <= 30
            similar = 30 < volume_at_B_difference <= 50
            asymmetric = volume_at_B_difference > 50
            pid = dft1.parent_id.values[0]
            time_at_B_min = dft1.time_at_B.values[0] * 60
            new_row = pd.DataFrame(
                {
                    "tid1": [tid1],
                    "tid2": [tid2],
                    "pid": [pid],
                    "small_sister": [small_sister],
                    "symmetric": [symmetric],
                    "similar": [similar],
                    "asymmetric": [asymmetric],
                    "difference_at_B": [volume_at_B_difference],
                    "time_at_B_min": [time_at_B_min],
                }
            )
            df_pairs = pd.concat([df_pairs, new_row], ignore_index=True)

    # Cast the boolean columns to bool dtype
    bool_cols = ["symmetric", "similar", "asymmetric"]
    df_pairs[bool_cols] = df_pairs[bool_cols].astype(bool)

    return df_pairs


def get_single_feature_for_sym_or_asym_tracks(df, df_pairs, condition):
    """
    Subset the sister pairs by asymmetric and symmetric dividers and indicate which is the smaller sister

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe with single track features
    df_pairs: Dataframe
        Dataframe resulting from get_symmetric_and_asymmetric_sister_pairs
    condition: Str
        "asymmetric" or "symmetric"
    Returns
    -------
    df: Dataframe
        The dataframe with the sister tracks that come from symmetric or asymmetric division events
    """
    sisters = df_pairs[df_pairs[condition]][["tid1", "tid2"]].values.ravel()
    return df[df["track_id"].isin(sisters)].assign(
        small_sister=lambda df: df["track_id"].isin(df_pairs.small_sister.values)
    )
