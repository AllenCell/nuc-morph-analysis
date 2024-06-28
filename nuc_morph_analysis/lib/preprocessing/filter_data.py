import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.preprocessing import add_times

# id: reason for marking as outlier v0.3.0
OUTLIER_BY_ID_DICT = {
    72657: "merge error",
    96713: "debris",
    99328: "B incorrect",
    910048: "B incorrect",
    76358: "Did not divide, neighborhood apoptotic debris",
    77209: "missing A-B",
    910991: "segmentation error",
    88358: "tracking error",
    7274: "missing A-B",
    71679: "tracking error",
    72914: "on edge around B",
    73744: "on edge around B",
    81809: "missing A-B",
    92862: "missing A-B",
    94799: "debris at beginning",
    911116: "majority of track on edge",
    911818: "on edge around B",
    97368: "movie ends",
    9484: "movie ends",
    9555: "tracking error",
    912418: "tracking error",
}

APOPTOTIC_BY_ID = [76846, 7318, 98653, 96973, 99660, 911044]

# Outlier tracks where we know the daughters die
DAUGHTERS_APOPTOSIS_BY_ID = [
    71119,
    73145,
    73897,
    75366,
    75525,
    77605,
    8897,
    81397,
    81892,
    85246,
    9805,
    91875,
    91907,
    93267,
    94100,
    95072,
    95247,
    96301,
    910929,
    911110,
    911796,
]
# Long tracks (duration BC longer than 22 hours) where we dont know their fate
LONG_OUTLIERS_BY_ID = [72089, 72096, 83102, 86508, 9112, 93734, 94722, 98190, 98332, 99670, 911092]

# Volume at C and fold change outliers by id that divide okay but have similar growth
GROWTH_FEATURE_OUTLIER_BY_ID = [74913, 71937, 75201, 71044, 71375, 76885, 71675, 8348, 99051, 76695]

# Daughters of succeful outlier dividers
OUTLIER_DAUGHTERS_BY_ID = [71406, 7341, 71587, 77695, 7669, 7345, 73840, 7664, 7672]


def filter_edge(df):
    """
    This function removes rows from the dataset containing the
    data representing a single frame of a single nucleus which gets
    cut off at the edge of the image.

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe containing a "fov_edge" column
        containing True or False to indicate of a nucleus gets
        cut off at the image edge in this frame

    Returns
    -------
    Dataframe
        Returns the input dataframe, with edge nuclei removed
    """

    return df.loc[~df.fov_edge].copy()


def filter_tp_outliers(df):
    """
    This function removes rows from the dataset containing the
    data representing a single frame of a single nucleus which
    has been marked as an outlier based on the volume relative
    to the frames before and after it

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe containing a "is_tp_outlier" column
        containing True or False to indicate of a nucleus has
        been identified as an outlier

    Returns
    -------
    Dataframe
        Returns the input dataframe, with outlier nuclei removed
    """

    return df.loc[~df.is_tp_outlier].copy()


def filter_outliers_by_ID(df):
    """
    This function removes rows from the dataset containing the
    data representing a nucleus at all its timepoints which
    has been marked as an outlier by ID. These are hand annotated
    outliers found in the dictionary OUTLIER_BY_ID_DICT.

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe containing a "is_outlier_curated_by_id" column
        containing True or False to indicate of a nucleus has
        been identified as an outlier by ID

    Returns
    -------
    Dataframe
        Returns the input dataframe, with outlier nuclei removed
    """
    return df.loc[~df.is_outlier_curated_by_id].copy()


def filter_apoptotic(df):
    """
    Filters out all nuclei annotated as underdoing apoptosis.

    Parameters
    ----------
    df: DataFrame
        Dataset containing annotated 'termination' column, with
        value 2 indicating an apoptosis event

    Returns
    -------
    df: DataFrame
        Dataset with apoptotic nuclei remove
    """
    n_outliers = df.loc[df["termination"] == 2, "track_id"].nunique()
    return df.loc[df.termination != 2].copy()


def filter_all_outliers(df):
    """
    This function removes rows from the dataset containing the
    data representing a single frame of a single nucleus which
    has been marked as an outlier based on the volume relative
    to the frames before and after it

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe containing a "is_toutlier" column
        containing True or False to indicate of a nucleus has
        been identified as an outlier

    Returns
    -------
    Dataframe
        Returns the input dataframe, with outlier nuclei removed
    """

    df = df.loc[~df.is_outlier].copy()

    return df


def filter_data(
    df,
    frame_formation_col="predicted_formation",
    frame_breakdown_col="predicted_breakdown",
    filter_edge_flag=True,
    filter_tp_outliers_flag=True,
    filter_out_short_flag=True,
    length_threshold=120,
    filter_to_full_cell_cycle_flag=True,
    make_plots=False,
    figdir=None,
    pix_size=None,
):
    """
    This is the master function for running all desired data filtering.
    Flags are provided so that only the desired set of filters are applied.
    Default flags are set to those most commonly used across all nuc morph
    data analysis to help us maintain a consistent filtering across all
    analyses.

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe

    frame_formation_col: string
        Name of column with formation frames

    frame_breakdown_col: string
        Name of column with breakdown frames

    filter_edge_flag: Boolean
        Flag to remove edge nuclei

    filter_tp_outliers_flag: Boolean
        Flag to remove outlier nuclei

    filter_out_short_flag: Boolean
        Flag to remove nuclei with tracks under some threshold length
        given in number of frames. The default threshold length is 76 frames.

    filter_to_full_cell_cycle_flag: Boolean
        Filters to remove nuclei which do not contain an entire
        cell cycle from formation to breakdown

    make_plots: Boolean
        Flag to generate and save plots

    figdir: Path
        Path to directory to store figures

    pix_size: float
        size of pixels in microns for this dataset

    Returns
    -------
    all_tracks: Dataframe
        Returns the input dataframe, with the formation and breakdown
        frame values shifted as necessary such that every formation and
        breakdown frame has an associated segmentation.
    """

    if filter_edge_flag:
        df = filter_edge(df)
    if filter_tp_outliers_flag:
        df = filter_tp_outliers(df)
    if filter_out_short_flag:
        df = filter_out_short(df, length_threshold)
    if filter_to_full_cell_cycle_flag:
        df = get_dataframe_of_full_tracks(
            df, col_formation=frame_formation_col, col_breakdown=frame_breakdown_col
        )

    # Save plots of all volume trajectories after all filtering is complete
    if make_plots:
        plot_tracks(df, pix_size, figdir)

    ntracks = df["track_id"].nunique()
    print(f"Dataset filtered. {ntracks} tracks in filtered dataset.")
    return df


def _deprecated_filter_to_single_tp_predictions(
    df_singletp,
    require_both_tp=False,
    col_formation="predicted_formation",
    col_breakdown="predicted_breakdown",
):
    """
    Find the subset of tracks for which we have a single formation AND
    breakdown prediction. Remember that "[-1]" is a prediction for
    regular timepoint. i.e. neither formation or breadkdown.

    Parameters
    ----------
    df_singletp: Pandas dataframe
        Input manifest which we will prune to contain only one formation
        and one breakdown timepoint

    require_both_tp: bool
        Flag for whether to require predictions existing for both formation
        and breakdown (True) or not (ie a prediction existing for only formation
        or breakdown, False)

    col_formation: str
        Name of column to be used as formation. Can also be curated_formation
        in case that is available

    col_breakdown: str
        Name of column to be used as breakdown. Can also be curated_breakdown
        in case that is available

    Returns
    -------
    df_singletp: Dataframe
        Dataframe with only single values for predicted formation and
        breakdown timepoints. If require_both_tp was true, only tracks
        with an existing but single-valued prediction for both formation
        and breakdown are included. If it was false, some entries will
        contain a single predicted value for one timepoint and a nan
        for the other.
    """

    cols = ["Ff", "Fb"]
    df_singletp[cols[0]] = df_singletp[col_formation]
    df_singletp[cols[1]] = df_singletp[col_breakdown]

    # if tracks require both a formation and breakdown, set to drop rows with any tp missing
    # if tracks require only a formation OR a breakdown, set to drop rows with all tp missing
    dropnans = "all_baseline"
    if require_both_tp:
        dropnans = "any"

    # Function to ignore tracks with multiple predicted formation and breakdown
    list_to_nan = lambda x: np.nan if "," in x else eval(x)[0]

    for col in cols:
        # Replace nans with "[-1]"s with nans for processing of strings with lambda function
        df_singletp.replace({col: {np.nan: "[-1]"}}, inplace=True)

        # Ignore tracks with nonnan values that have multple predicted timepoints for formation or breakdwon
        df_singletp[col] = df_singletp[col].apply(list_to_nan)

        # Replace all -1's with nans for easier dropping of nonexisting values
        df_singletp.replace({col: {-1: np.nan}}, inplace=True)

    # drop nans based on whether user selected to drop tracks missing one or both timepoint values
    df_singletp = df_singletp.dropna(subset=cols, how=dropnans).copy()

    return df_singletp


def filter_to_single_tp_predictions(
    df,
    require_both_tp=False,
    col_formation="predicted_formation",
    col_breakdown="predicted_breakdown",
):
    """
    Get just the rows with a formation AND/OR breakdown prediction.
    Updated for simplified formation and breakdown values.

    Parameters
    ----------
    df: pandas.DataFrame
        Input manifest to filter.

    require_both_tp: bool
        Flag for whether to require predictions existing for both formation
        and breakdown (True) or not (ie a prediction existing for only formation
        or breakdown, False)

    col_formation: str
        Name of column to be used as formation.

    col_breakdown: str
        Name of column to be used as breakdown.

    Returns
    -------
    pandas.DataFrame
        Subset of df containing rows with a formation AND/OR breakdown prediction.
        Adds the columns "Ff" and "Fb" identical to col_formation and col_breakdown except that the
        "no prediction" value is NaN instead of -1.
    """
    result = df.copy()

    # Add Ff and Fb columns
    new_cols = ["Ff", "Fb"]
    result[new_cols] = result[[col_formation, col_breakdown]].replace(-1, np.nan)

    # if tracks require only a formation OR a breakdown, drop rows with both missing
    dropnans = "all_baseline"
    if require_both_tp:
        # if tracks require both a formation AND breakdown, drop rows with either missing
        dropnans = "any"

    # drop nans based on whether user selected to drop tracks missing one or both timepoint values
    return result.dropna(subset=new_cols, how=dropnans).copy()


def select_predicted_timepoints(df, require_both_tp=False, use_tp_before_pred_breakdown=True):
    """
    Update selection of single formation and breakdown timepoints by
    using the timepoint one frame before breakdown and removing tracks for
    which the difference between first available frame and predicted frame
    is greater than two

    Parameters
    ----------
    df: Pandas dataframe
        NucMorph dataframe as generated by the func load_data.load_dataset
        function

    require_both_tp: bool
        Flag for whether to require predictions existing for both formation
        and breakdown (True) or not (ie a prediction existing for only formation
        or breakdown, False)

    use_tp_before_pred_breakdown: bool
        Whether or not to force last tp to be one tp before the pred breakdown

    Returns
    -------
    df: Dataframe
        Dataframe with updated selection of predicted timepoints
    """

    cols = ["Ff", "Fb"]
    if use_tp_before_pred_breakdown:
        df[cols[1]] -= 1
    # Sometimes tp of pred formation or breakdown are not present
    # in the dataframe. In these cases we want to find the nearest
    # available tp.
    for _, df_track in df.groupby("track_id"):
        for col in cols:
            absdif = np.abs(df_track.index_sequence.values - df_track[col].min())
            delta = np.argmin(absdif)
            # Remove tracks if difference between prediction and first
            # available frame larger than 2 frames.
            frame = df_track.at[df_track.index[delta], "index_sequence"]
            if absdif.min() > 2 or np.isnan(absdif.min()):
                frame = np.nan
            df.loc[df_track.index, col] = frame

    dropnans = "all_baseline"
    if require_both_tp:
        dropnans = "any"
    df = df.dropna(subset=cols, how=dropnans).copy()
    return df


def get_dataframe_of_full_tracks(
    df,
    min_length_in_tps=120,
    apply_filters=True,
    col_formation="predicted_formation",
    col_breakdown="predicted_breakdown",
    require_both_tp=True,
    use_tp_before_pred_breakdown=True,
    quiet=True,
    use_old_formation_breakdown=False,
):
    """
    Find the subset of tracks for which we have a single formation AND
    breakdown prediction. Remember that "[-1]" is a prediction for
    regular timepoint. i.e. neither formation or breadkdown.

    Parameters
    ----------
    df: Pandas dataframe
        NucMorph dataframe as generated by the func load_data.load_dataset
        function

    min_length_in_tps: int
        Minimum number of frames between formation and breakdown for a given
        track

    col_formation: str
        Name of column to be used as formation. Can also be curated_formation
        in case that is available

    col_breakdown: str
        Name of column to be used as breakdown. Can also be curated_breakdown
        in case that is available

    require_both_tp: bool
        Flag for whether to require predictions existing for both formation
        and breakdown (True) or not (ie a prediction existing for only formation
        or breakdown, False)

    use_tp_before_pred_breakdown: bool
        Whether or not to force last tp to be one tp before the pred breakdown

    quiet: bool
        Suppress print statements

    use_old_formation_breakdown: bool
        Legacy flag to expect formation and breakdown in the string format with multiple predicted
        timepoints. Should usually be False.

    Returns
    -------
    df_full_tracks: Dataframe
        Dataframe with full tracks only
    """

    cols = ["Ff", "Fb"]
    df[cols[0]] = df[col_formation]
    df[cols[1]] = df[col_breakdown]

    # Exclude edge and outliers
    if apply_filters:
        if not quiet:
            print(f"{df.shape[0]} single-timepoint nuclei before filtering edge-cells and outliers")
        df_filt = filter_edge(df)
        df_filt = filter_tp_outliers(df_filt)
        df_filt = filter_outliers_by_ID(df_filt)
        df_filt = filter_apoptotic(df_filt)
        if not quiet:
            print(
                f"{df_filt.shape[0]} single-timepoint nuclei after filtering edge-cells and outliers"
            )
    else:
        df_filt = df.copy()

    # Select single predicted formation and breakdown timepoints
    if use_old_formation_breakdown:
        df_tp = _deprecated_filter_to_single_tp_predictions(
            df_filt, require_both_tp, col_formation, col_breakdown
        )
    else:
        df_tp = filter_to_single_tp_predictions(
            df_filt, require_both_tp, col_formation, col_breakdown
        )
    df_tp = select_predicted_timepoints(df_tp, require_both_tp, use_tp_before_pred_breakdown)

    # Add normalized time and filter to long tracks
    df_full_tracks = add_times.add_normalized_time_for_full_tracks(df_tp)
    df_full_tracks = filter_out_short(df_full_tracks, min_length_in_tps)
    return df_full_tracks


def filter_out_short(df, length_threshold=120):
    """
    This function removes rows from the dataset containing the
    data for nuclear tracks under a certain lentgh threshold.

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe

    length_threshold: int
        The minimum length (in number of frames) of nucleus volume
        track to keep in the dataset.

    Returns
    -------
    df: Dataframe
        Returns the input dataframe, with tracks shorter than the length
        threshold value removed
    """

    df_track_length = pd.DataFrame(df[["track_id"]].groupby(["track_id"]).size())
    df_track_length.columns = ["length"]
    df_long_tracks = df_track_length.loc[df_track_length.length > length_threshold]
    df_filtered = df.loc[df.track_id.isin(df_long_tracks.index)]
    ntracks = df_filtered["track_id"].nunique()
    return df_filtered


def create_trajectory_filestructure(figdir):
    """
    This function creates a series of directories and subdirectories
    for figures generated in this analysis to be save to, in a
    pre-determined organized way. It first checks to see if each
    subdirectory exists, then creates it if it does not.

    Parameters
    ----------
    figdir: Path
        Path to figure directory for this analysis

    Returns
    -------
    dir: Path
        Path to the base directory to save figures to
    """

    # Cycle through layers of nested subdirectories and create all
    for nested_dir in ["trajectories", "filtered_tracks"]:
        new_dir = figdir / nested_dir
        new_dir.mkdir(parents=True, exist_ok=True)
        figdir = new_dir

    return new_dir


def plot_tracks(df, pix_size, figdir=None):
    """
    Plots volume over time for all nuclear trajectories after filtration.

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe

    pix_size: float
        size of pixels in microns for this dataset

    figdir: Path
        Path to directory to store figures
    """

    if figdir is not None:
        trackdir = create_trajectory_filestructure(figdir)
    else:
        trackdir = None

    for track, df_track in df.groupby("track_id"):
        df_track = df_track.sort_values("index_sequence")
        x = df_track.index_sequence.values
        y = df_track.volume.values * (pix_size**3)
        plt.clf()
        plt.plot(x, y)
        plt.title(track)
        plt.xlabel("Frames")
        plt.ylabel(r"Volume ($\mu m^3$)")
        if figdir is None:
            plt.savefig(f"{trackdir}/{track}.png")
        plt.show()


def flag_outlier_by_short_track(df, threshold=5):
    """
    Flags all nuclei with tracks under a given threshold length
    This threshold is intentionally very low, intended to remove
    segmentation errors such as merged nuclei

    Parameters
    ----------
    df: DataFrame
        Dataset

    Returns
    -------
    df: DataFrame
        Dataset with 'is_outlier_by_short_track' column added
    outlier_col: string
        Name of new outlier column 'is_outlier_by_short_track'
    """

    outlier_col = "is_outlier_by_short_track"
    df[outlier_col] = False

    df["track_length"] = np.nan
    for track_id, df_track in df.groupby("track_id"):
        track_length = df_track.shape[0]
        df.loc[df["track_id"] == track_id, "track_length"] = track_length
    df[outlier_col] = df["track_length"] < threshold
    n_outliers = df.loc[df[outlier_col], "track_id"].nunique()
    return df, outlier_col


def flag_growth_feature_outliers(df, long_outliers=True, growth_outliers=True):
    """
    Flags all nuclei with tracks that were flagged as outliers
    based on growth feature values. These include long tracks where
    the daughters die upon division, and feature outliers that grow
    like these tracks with long durations, with large volumes at C
    and large fold change that lie outside the normal distribution.

    Parameters
    ----------
    df: DataFrame
        Dataset

    Returns
    -------
    df: DataFrame
        Dataset with 'is_growth_outlier' column added
    outlier_col: string
        Name of new outlier column 'is_growth_outlier'
    """
    POOL_OUTLIERS = DAUGHTERS_APOPTOSIS_BY_ID.copy()
    if long_outliers:
        POOL_OUTLIERS += LONG_OUTLIERS_BY_ID
    if growth_outliers:
        POOL_OUTLIERS += GROWTH_FEATURE_OUTLIER_BY_ID
        POOL_OUTLIERS += OUTLIER_DAUGHTERS_BY_ID

    outlier_col = "is_growth_outlier"
    df[outlier_col] = False
    df[outlier_col] = df["track_id"].isin(POOL_OUTLIERS)
    n_outlier = df.loc[df[outlier_col], "track_id"].nunique()
    return df, outlier_col


def flag_missed_apoptotic(df):
    """
    Flag nuclei that were missed as apoptotic in the dataset

    Parameters
    ----------
    df: DataFrame
        Dataset containing annotated 'termination' column, with
        value 2 indicating an apoptosis event

    Returns
    -------
    df: DataFrame
        Dataset with all apoptotic nuclei flagged
    """
    df["termination"] = np.where(df["track_id"].isin(APOPTOTIC_BY_ID), 2, df["termination"])
    return df


def flag_outlier_curated_by_ID(df):
    """
    Flags all nuclei designated by ID as curated outliers. These
    were hand annotated and not generated by a programmatic filter
    and are each paired with a very brief explanation of why they
    were flagged

    Parameters
    ----------
    df: DataFrame
        Dataset

    Returns
    -------
    df: DataFrame
        Dataset with 'is_outlier_curated_by_id' column added
    outlier_col: string
        Name of new outlier column 'is_outlier_curated_by_id'
    """

    outlier_col = "is_outlier_curated_by_id"
    df[outlier_col] = False
    for key in OUTLIER_BY_ID_DICT:
        df.loc[df["track_id"] == key, outlier_col] = True
    n_outliers = df.loc[df[outlier_col], "track_id"].nunique()
    return df, outlier_col


def flag_outlier_tracks(df, remove_growth_outliers):
    """
    Flags all nuclear tracks designated as outliers, either because
    they were annotated to undergo apoptosis, annotated as having
    a track issue, or have a very short track which can be an indicator
    of erroneous segmentations

    Parameters
    ----------
    df: DataFrame
        Dataset
    remove_growth_outliers: bool
        Flag to remove tracks that are growth feature outliers

    Returns
    -------
    df: DataFrame
        Dataset with 'is_outlier_track' column added
        which has a value of true if any one or more of the
        individual outlier types is flagged
    """

    df1, col1 = flag_outlier_by_short_track(df)
    df2, col2 = flag_outlier_curated_by_ID(df1)
    df3, col3 = flag_growth_feature_outliers(df2)
    outlier_cols = [col1, col2]
    df_filtered = df2

    if remove_growth_outliers:
        outlier_cols.append(col3)
        df_filtered = df3

    df_filtered["is_outlier_track"] = df_filtered[outlier_cols].any(axis="columns")
    return df_filtered


def add_and_pool_outlier_flags(df, remove_growth_outliers):
    """
    Flags as outlier:
    - all single timepoint nuclear datapoints flagged as outliers
    by automated detection
    - all timpepoints that occur before formation or after breakdown timepoints
    - any nuclear tracks designated as outliers either because they
    were annotated to undergo apoptosis, annotated as having a track
    issue, or have a very short track which can be an indicator
    of erroneous segmentations

    Parameters
    ----------
    df: DataFrame
        Dataset
    remove_growth_outliers: bool
        Flag to remove tracks that are growth feature outliers

    Returns
    -------
    df: DataFrame
        Dataset with 'is_outlier' column added
        which has a value of true if any one or more of the
        individual outlier types for whole tracks or single
        timepoints is flagged
    """
    # add all track-based outlier flags
    df = flag_outlier_tracks(df, remove_growth_outliers)

    # make one flag for whether any nucleus at any timepoint is any kind of outlier
    df["is_outlier"] = df[
        ["is_tp_outlier", "is_after_breakdown_before_formation_outlier", "is_outlier_track"]
    ].any(axis="columns")
    return df


def filter_out_cells_entering_or_exiting_mitosis(df, quiet=True):
    if not quiet:
        print(
            f"filtering out {np.sum(df['entering_or_exiting_division']==True)} timepoints close to division events"
        )
    df = df[df["entering_or_exiting_division"] == False]
    return df


def filter_out_before_formation_and_after_breakdown_outliers(df, quiet=True):
    if not quiet:
        print(
            f"filtering out {np.sum(df['is_after_breakdown_before_formation_outlier']==True)} timepoints before formation and after breakdown"
        )
    df = df[df["is_after_breakdown_before_formation_outlier"] == False]
    return df


def filter_out_non_interphase_size_shape_flag(df):
    """
    This function removes rows in the dataset FLAGGED as `non_interphase_size_shape` because the nuclear volume, surface area and/or SA/V ratio are outside the range of values expected for interphase nuclei.
    This flag is added by the function `add_features.add_non_interphase_size_shape_flag`
    The "normal interphase nuclei" used to define the expected values are the `is_full_track`==True and `exiting_mitosis`==True nuclei. That is nuclei that are in interphase (well after exiting mitosis).
    The thresholds are explored and determined in `explore_outlier_filters.py`.
    After applying filter based on this flag, we expect to have only interphase nuclei remaining in the dataset
    This flag is critical for analyzing the growth rate of neighboring interphase cells, since we need to filter out nuclei that still exiting mitosis,
    (rapid growth of nuclei during mitosis exit will confound results), and we want to filter out "nuclei" that are debris (very small, high SA/V objects)

    Very important: this should NOT flag ANY nuclei from the full tracks (that are after mitosis exit)
    If they do, the filters are too strict and should be adjusted.
    The analysis code will check if this has occured or not.

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe
        REQUIRES THE COLUMN `exiting_mitosis`

    Returns
    -------
    df: Dataframe
        Returns the input dataframe, with non-interphase nuclei removed

    """
    # these thresholds are determined by finding the minimum and maximum values for the given features from the full_tracks (that are after mitosis exit)
    # to be safe, make the threshold values 10% lower or 10% higher so that the filters are not likely to remove true interphase nuclei
    # {"column": min_thresh, max_thresh}

    # filter out the non-interphase nuclei
    df = df[df["non_interphase_size_shape"] == False]
    return df


def add_analyzed_dataset_columns(df, dataset):
    """
    Dataset with all the timpoints with three new columns to easily identify subsets
    of the data that were analyzed in the manuscript.

    0. exploratory_dataset: all timepoints with minimal exploratory filtering
       which includes apoptic nuclei for exploration in Timelapse Feature Explorer
    1. baseline_colonies_dataset: all timepoints with minimal filtering
    2. full_interphase_dataset: all timepoints with full tracks excluding growth outliers
    3. lineage_annotated_dataset: full tracks with lineage annotated colonies

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe

    Returns
    -------
    df: Dataframe
        Returns the input dataframe with three new columns
    """
    if dataset == "all_baseline":

        df_all_tp_exploratory_tfe = all_timepoints_tfe_exploratory_filtering(df)
        df["exploratory_dataset"] = df.index.isin(df_all_tp_exploratory_tfe.index)

        df_all_tp = all_timepoints_minimal_filtering(df)
        df["baseline_colonies_dataset"] = df.index.isin(df_all_tp.index)

        df_full = all_timepoints_full_tracks(df)
        df_full = df_full[df_full["is_growth_outlier"] == False]
        df["full_interphase_dataset"] = df.index.isin(df_full.index)

        df_sub = df_full[df_full["colony"].isin(["small", "medium"])]
        df["lineage_annotated_dataset"] = df.index.isin(df_sub.index)

    return df


def all_timepoints_tfe_exploratory_filtering(df):
    """
    This function applies minimal exploratory filtering to the dataframe by filtering out edge nuclei and outliers.

    Parameters:
    -----------
    df: Dataframe
        Dataframe with minimal set of filters (edge, apoptotic, and outliers).

    Returns:
    --------
    df: Dataframe
    """
    df = filter_edge(df)
    df = filter_out_before_formation_and_after_breakdown_outliers(df)
    return df


def all_timepoints_minimal_filtering(df):
    """
    This function applies minimal filtering to the dataframe by filtering out edge nuclei, apoptotic nuclei and outliers.

    Parameters:
    -----------
    df: Dataframe
        Dataframe with minimal set of filters (edge, apoptotic, and outliers).

    Returns:
    --------
    df: Dataframe
    """
    df = filter_edge(df)
    df = filter_apoptotic(df)
    df = filter_all_outliers(
        df
    )  # is_outlier_track and is_outlier=['is_tp_outlier', 'is_after_breakdown_before_formation_outlier', 'is_outlier_track']
    return df


def all_timepoints_full_tracks(df):
    """
    This function filters the dataframe to only include full tracks that are not curated outliers.

    Parameters:
    -----------
    df: Dataframe
        The master dataframe output from load_all_datasets_with_features.

    Returns:
    --------
    df: Dataframe
    """
    df = all_timepoints_minimal_filtering(df)
    df = df.loc[df["is_full_track"] == True]
    return df


def track_level_features(df):
    """
    This function filters the dataframe to only include full tracks that are not curated outliers.
    It returns a dataframe with a single value per track (such as duration, growth rate, etc) and
    that can be used for plotting and analyzing those track features such that each track is only
    represented once.

    Parameters:
    -----------
    df: Dataframe
        The master dataframe output from load_all_datasets_with_features.

    Returns:
    --------
    df: Dataframe
    """
    print(
        "If dataframe of full tracks is used as input to function calculating "
        "track level features, filters are already applied and no change in "
        "dataset size is expected."
    )
    df = all_timepoints_full_tracks(df)
    # keep columns with only one value per track
    df_grouped = df.groupby("track_id").nunique()
    columns_to_keep = df_grouped.columns[df_grouped.le(1).all()].tolist()
    columns_to_keep.insert(0, "track_id")
    df_subset_columns = df[columns_to_keep]
    # keep single value per track
    df = df_subset_columns.groupby("track_id").first().reset_index()
    return df
