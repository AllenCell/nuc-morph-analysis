# %%
import os
from pathlib import Path
import pandas as pd
from nuc_morph_analysis.lib.preprocessing import (
    load_data,
    filter_data,
    add_times,
    is_tp_outlier,
    add_features,
    add_neighborhood_avg_features,
    compute_change_over_time,
)
from nuc_morph_analysis.analyses.volume import add_growth_features
from nuc_morph_analysis.analyses.colony_context.colony_context_analysis import (
    add_colony_position_columns,
    add_fov_touch_timepoint_for_colonies,
)
from nuc_morph_analysis.analyses.height.add_colony_time import add_colony_time_all_datasets
from nuc_morph_analysis.lib.preprocessing import labeling_neighbors_helper

def load_dataset_with_features(
    dataset="all_baseline",
    remove_growth_outliers=True,
    load_local=False,
    save_local=False,
    num_workers=32,
):
    """
    Loads one dataset or all datasets in a uniform way and calculates various features for each nucleus at different timepoints.

    Parameters
    ----------
    dataset: string
        Name of dataset (or "all_baseline" to combine all datasets) to load
    remove_growth_outliers: bool
        Flag to remove tracks that are growth feature outliers
    load_local: bool
        Flag to load a local dataset with features if it exists
    save_local: bool
        Flag to save the dataset with features locally
    num_workers: int
        Number of workers to use for parallel processing

    Returns:
    --------
    df_master: Dataframe
        The dataframe with all datasets and calculated features.
    """
    if load_local:
        try:
            # Try to load the dataset from a local disk cache. This is risky: you have to make sure
            # you are deleting the cache manually when it is out of date, but it can save time.
            if dataset in ["all_baseline", "all_feeding_control", "all_drug_perturbation"]:
                df_master = load_local_dataset(dataset, title="with_features")
            else:
                experiment_group = load_data.get_dataset_experiment_group_by_name(dataset)
                df_master = load_local_dataset(f"all_{experiment_group}", title="with_features")
                df_master = df_master.loc[df_master.colony == dataset]
        except FileNotFoundError:
            load_local = False

    if not load_local:
        # Build the dataset the regular way
        if dataset in ["all_baseline", "all_feeding_control", "all_drug_perturbation"]:
            df = load_data.load_all_datasets(dataset)
        else:
            df = load_data.load_dataset(dataset)

        pix_size = load_data.get_dataset_pixel_size(dataset)
        thresh = load_data.get_length_threshold_in_tps(dataset)
        interval = load_data.get_dataset_time_interval_in_min(dataset)

        n_tracks_prefilter = df["track_id"].nunique()
        print(f"{n_tracks_prefilter} tracks before any filtering")
        if 'level_0' in df.columns:
            print('WARNING: level_0 column found in df, dropping')
        df_all = process_all_tracks(df, dataset, remove_growth_outliers, num_workers)
        df_all_filtered = filter_data.all_timepoints_minimal_filtering(df_all)
        n_tracks = df_all_filtered["track_id"].nunique()
        print(f"{n_tracks} tracks after all outliers removed")
        df_full = process_full_tracks(df_all, thresh, pix_size, interval)
        df_full_filtered = filter_data.all_timepoints_full_tracks(df_full)
        n_full_tracks = df_full_filtered["track_id"].nunique()
        print(f"{n_full_tracks} full tracks after all outliers removed")
        df_master = merge_datasets(df_all, df_full)

        df_master = filter_data.add_analyzed_dataset_columns(df_master, dataset)
        df_master = remove_columns(df_master)
        assert df_master.index.name == "CellId"

        # check that no rows have been dropped
        if df_master.shape[0] != df.shape[0]:
            raise Exception(
                f"The loaded manifest has {df.shape[0]} rows and your \
                final manifest has {df_master.shape[0]} rows.\
                Please revise code to leave manifest rows unchanged."
            )

    if load_local & save_local:
        print("WARNING!: Loading and saving local dataset with features.")
        print("this is a redundant operation and may not be necessary")
    if save_local:
        write_local(df_master, dataset, "with_features")
    return df_master


def load_local_dataset(dataset, title):
    filename = name_local_file(dataset, title)
    if os.path.exists(filename):
        print("WARNING!: Loading local dataset with features.")
        print("!!!This saves time but may not be the most recent version!!!")
        df_master = pd.read_parquet(filename)
    else:
        raise FileNotFoundError(
            f"Local dataset with features not found: {filename}\nPlease set load_local=False to generate the file."
        )
    return df_master


def name_local_file(dataset, title, destdir=None, format="parquet"):
    """
    Parameters
    ----------
    dataset: str
    title: str
    destdir: str or Path, optional
        Absolute path to write to. Defaults to nuc-morph-analysis/data
    format: str, optional
        "parquet" or "csv"
    """
    if destdir is None:
        destdir = Path(__file__).parent.parent.parent.parent / "data"
    os.makedirs(destdir, exist_ok=True)
    filename = f"{destdir}/{dataset}_{title}.{format}"
    return filename


def write_local(df, dataset, title, destdir=None, format="parquet"):
    """
    Parameters
    ----------
    df: pandas.DataFrame
    destdir: str or Path, optional
        Absolute path to write to. Defaults to nuc-morph-analysis/data
    format: str, optional
        "parquet" or "csv"
    """
    filename = name_local_file(dataset, title, destdir, format)
    if format == "parquet":
        df.to_parquet(filename, index=True)
    elif format == "csv":
        df.to_csv(filename, index=True)
    else:
        raise ValueError(f"Unknown format: {format}")


def process_all_tracks(df, dataset, remove_growth_outliers, num_workers):
    """
    Add features that can be applied to all tracks.
    Featurs include: colony time, aspect ratio, family id, and colony position.

    Paramaters:
    -----------
    df: Dataframe
        The input dataframe.
    remove_growth_outliers: bool
        Flag to remove tracks that are growth feature outliers.
    num_workers: int
        Number of workers to use for parallel processing.

    Returns:
    --------
    df_all: Dataframe
        The dataframe with all datasets and calculated features
    dataset: Str
        Name of dataset
    remove_growth_outliers: bool
        Flag to remove tracks that are growth feature outliers
    """
    # add outlier flags
    assert df.index.name == "CellId"
    df = filter_data.flag_missed_apoptotic(df)
    df = is_tp_outlier.outlier_detection(df)
    df = add_features.add_division_entry_and_exit_annotations(df)
    df = filter_data.add_and_pool_outlier_flags(df, remove_growth_outliers=remove_growth_outliers)

    # add labels for neighbors of mitotic and dying cells
    df = labeling_neighbors_helper.label_nuclei_that_neighbor_current_mitotic_event(df)
    df = labeling_neighbors_helper.label_nuclei_that_neighbor_current_death_event(df)

    # add features
    df = add_features.add_aspect_ratio(df)
    df = add_features.add_family_id_all_datasets(df)
    df = add_features.add_SA_vol_ratio(df)
    df = add_colony_position_columns(df)
    df = add_fov_touch_timepoint_for_colonies(df)
    df = add_features.add_non_interphase_size_shape_flag(df)
    df = add_change_over_time(df)
    df = add_neighborhood_avg_features.run_script(df, num_workers=num_workers)

    if dataset == "all_baseline":
        df = add_colony_time_all_datasets(df)

    assert df.index.name == "CellId"
    return df


def process_full_tracks(df_all, thresh, pix_size, interval):
    """
    Add features that can be applied to full tracks.
    Features include: transition point, normalized time, synchronized time, duration, growth rate, and volume change.
    Features at formation, transition and breakdown include: volume, location, time, and colony time.
    Flags for full tracks and curated outliers based on the feature measurements.

    Parameters
    ----------
    df_all : pandas.DataFrame
        The input dataframe.
    thresh : float
        The threshold for filtering full tracks.
    pix_size : float
        The size of a pixel in the image.
    interval : float
        The time interval between frames in minutes.

    Returns
    -------
    df_full : pandas.DataFrame
        The dataframe with the calculated features.
    """
    assert df_all.index.name == "CellId"
    df_full = filter_data.get_dataframe_of_full_tracks(
        df_all, thresh
    )  # edges and outliers are filtered here
    df_full = filter_data.filter_apoptotic(df_full)
    df_full = add_times.add_transition_point(df_full, pix_size)
    df_full = add_times.add_synchronized_time(df_full, "Ff")

    for frame in ["Ff", "frame_transition", "Fb"]:
        df_full = add_features.add_volume_at(df_full, pix_size, frame)
        df_full = add_features.add_location_at(df_full, frame, "x")
        df_full = add_features.add_location_at(df_full, frame, "y")
        df_full = add_features.add_time_at(df_full, frame, interval)
        df_full = add_features.add_colony_time_at(df_full, frame, interval)

    df_full = add_features.add_duration_in_frames(df_full, "Ff", "frame_transition")
    df_full = add_features.add_duration_in_frames(df_full, "frame_transition", "Fb")
    df_full = add_features.add_duration_in_frames(df_full, "Ff", "Fb")
    df_full = add_features.add_feature_change_BC(df_full, "volume", "volume", pix_size**3)
    df_full = add_features.add_feature_change_BC(df_full, "SA", "mesh_sa", pix_size**2)
    df_full = add_features.add_fold_change_track_fromB(df_full, "volume", "volume", pix_size**3)
    df_full = add_features.add_fold_change_track_fromB(df_full, "SA", "mesh_sa", pix_size**2)
    df_full = add_growth_features.add_early_growth_rate(df_full, interval)
    df_full = add_growth_features.add_late_growth_rate_by_endpoints(df_full)
    df_full = add_growth_features.fit_tracks_to_time_powerlaw(df_full, "volume", interval)

    # Add flag for use after merging back to main manifest
    df_full = add_features.add_full_track_flag(df_full)
    assert df_full.index.name == "CellId"
    return df_full


def merge_datasets(df_all, df_full):
    """
    Merges the full tracks dataframe with the all timepoints dataframe.

    Parameters
    ----------
    df_all : pandas.DataFrame
        The all timepoints dataframe.
    df_full : pandas.DataFrame
        The full tracks dataframe.

    Returns
    -------
    df_master : pandas.DataFrame
        The merged dataframe.
    """
    assert df_all.index.name == "CellId"
    df_all = df_all.drop(["Ff", "Fb"], axis=1)  # these cols are updated in df_full
    df_full = df_full.drop(df_all.columns.tolist(), axis=1)
    df_all.reset_index(inplace=True)
    df_full.reset_index(inplace=True)
    df_master = df_all.merge(df_full, on="CellId", how="outer")
    df_master["is_full_track"].fillna(False, inplace=True)
    df_master.set_index("CellId", inplace=True) 
    assert df_master.index.name == "CellId"
    return df_master


COLUMNS_TO_DROP = [
    "scale_micron",
    "max_distance_from_centroid",
    "colony_non_circularity",
    "colony_non_circularity_scaled",
    "max_colony_depth",
    "dxdt_48_volume_per_V",
    "neighbor_avg_dxdt_48_volume_per_V_90um",
    "neighbor_avg_dxdt_48_volume_per_V_whole_colony",
    "dataset",
    "height_percentile",
    "raw_full_zstack_path",
    "seg_full_zstack_path",
]


def remove_columns(df, column_list=COLUMNS_TO_DROP):
    """
    Removes columns from the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    column_list : list
        The list of columns to remove.

    Returns
    -------
    df : pandas.DataFrame
        The dataframe with the columns removed.
    """
    df = df.drop(columns=column_list)
    return df


def add_change_over_time(df):
    """
    Adds new columns to the dataframe with the local rate of change for a given feature for a nucleus

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.

    Returns
    -------
    df : pandas.DataFrame
        The dataframe with the new column added.
    """
    dfm = df.copy()
    for bin_interval in compute_change_over_time.BIN_INTERVAL_LIST:
        dfm = compute_change_over_time.run_script(dfm, bin_interval=bin_interval)

    # now check that all columns in df have the same dtype as columns in dfm
    for col in df.columns:
        if dfm[col].dtype != df[col].dtype:
            print(f"column {col} has dtype {dfm[col].dtype} in dfm and {df[col].dtype} in df")

    if dfm.shape[0] != df.shape[0]:
        raise Exception(
            f"The loaded manifest has {df.shape[0]} rows and your \
            final manifest has {dfm.shape[0]} rows.\
            Please revise code to leave manifest rows unchanged."
        )
    return dfm


# %%
if __name__ == "__main__":
    for dataset in ["all_baseline", "all_feeding_control", "all_drug_perturbation"]:
        df = load_dataset_with_features(dataset, save_local=True, num_workers=32)
