from nuc_morph_analysis.analyses.lineage.get_features import lineage_trees
import numpy as np

FRAME_COL = {"Ff": "A", "frame_transition": "B", "Fb": "C"}
NON_INTERPHASE_FILTER_THRESHOLDS = {
    "volume": (276760, None),
    "mesh_sa": (25060, None),
    "SA_vol_ratio": (None, 0.11),
}


def add_division_entry_and_exit_annotations(df, breakdown_threshold=3, formation_threshold=24):
    """
    Classify each timepoint as entering or exiting mitosis or not based proximity to the predicted_breakdown and predicted_formation column times

    Parameters
    ----------
    df : pd.DataFrame
        dataframe that minimally has the following columns:
        ['index_sequence','predicted_breakdown','predicted_formation']
    breakdown_threshold : int, optional
        The number of timepoints before and after the predicted_breakdown timepoint to classify as a breakdown event. The default is 3 (15 min).
    formation_threshold : int, optional
        The number of timepoints before and after the predicted_formation timepoint to classify as a formation event. The default is 24 (2 hrs).
    """
    # make a copy to avoid editing the original dataframe columns when setting -1 values to nans
    dfc = df.copy()

    # set predicted_breakdown = nan if it is -1
    dfc.loc[dfc["predicted_breakdown"] == -1, "predicted_breakdown"] = np.nan

    # define a cell as entering mitosis if it occurs before the 'predicted_breakdown' frame and within the breakdown threshold
    time_until_breakdown = dfc["predicted_breakdown"] - dfc["index_sequence"].astype(int)
    dfc["entering_mitosis"] = (time_until_breakdown <= breakdown_threshold) & (
        time_until_breakdown >= 0
    )
    # set 'entering_mitosis' = 0 if predicted_breakdown is nan
    dfc.loc[dfc["predicted_breakdown"].isna(), "entering_mitosis"] = False

    # define an additional column that specifies whether the timepoint for that nucleus occurs after its 'predicted_breakdown' frame
    # this can happen if a nucleus is tracked through mitosis (without receiving a new track_id), but it should not happen
    dfc["after_breakdown_outlier"] = time_until_breakdown < 0
    dfc.loc[dfc["predicted_breakdown"].isna(), "after_breakdown_outlier"] = False

    # set predicted_formation = nan if it is -1
    dfc.loc[dfc["predicted_formation"] == -1, "predicted_formation"] = np.nan

    # collect formation events
    # define a cell as exiting mitosis if it occurs after the 'predicted_formation' frame and within the formation threshold
    time_since_formation = dfc["index_sequence"].astype(int) - dfc["predicted_formation"]
    dfc["exiting_mitosis"] = (time_since_formation <= formation_threshold) & (
        time_since_formation >= 0
    )
    # set exiting_mitosis = nan if predicted_formation is nan
    dfc.loc[dfc["predicted_formation"].isna(), "exiting_mitosis"] = False

    # define an additional column that specifies whether the timepoint for that nucleus occurs BEFORE its 'predicted_formation' frame
    # a few frames (1 or 2) could be segmented before a nucleus' "predicted_formation" if the segmentation model predicted segmentations at those timepoints.
    # These segmentations should not be trusted since there is not a fully formed lamin shell
    # It may also happen that nucleus was erroneously tracked through mitosis (without recieving a new track_id), but this should not happen and we would want to remove that with this flag
    dfc["before_formation_outlier"] = time_since_formation < 0
    dfc.loc[dfc["predicted_formation"].isna(), "before_formation_outlier"] = False

    # now add the columns to the original dataframe
    df["entering_mitosis"] = dfc["entering_mitosis"].astype(bool)
    df["exiting_mitosis"] = dfc["exiting_mitosis"].astype(bool)
    df["after_breakdown_outlier"] = dfc["after_breakdown_outlier"].astype(bool)
    df["before_formation_outlier"] = dfc["before_formation_outlier"].astype(bool)
    df["entering_or_exiting_division"] = df["entering_mitosis"].astype(bool) | df[
        "exiting_mitosis"
    ].astype(bool)
    df["is_after_breakdown_before_formation_outlier"] = df["after_breakdown_outlier"].astype(
        bool
    ) | df["before_formation_outlier"].astype(bool)
    return df


def add_aspect_ratio(df):
    """
    Adds aspect ratio columns to the dataframe.
    The x aspect is the longest axis in the xy plane.

    Paramaters:
    ----------
    df: DataFrame
        The dataframe

    Returns:
    df: Dataframe
        The dataframe with the added aspect ratio columns.
    """
    df["xz_aspect"] = df["length"] / df["height"]  # x/z
    df["xy_aspect"] = df["length"] / df["width"]  # x/y
    df["zy_aspect"] = df["height"] / df["width"]  # z/y
    return df


def add_SA_vol_ratio(df):
    """
    Adds surface area to volume ratio column to the dataframe.

    Paramaters:
    ----------
    df: DataFrame
        The dataframe

    Returns:
    df: Dataframe
        The dataframe with the added surface area to volume ratio column.
    """
    df["SA_vol_ratio"] = df["mesh_sa"] / df["volume"]
    return df


def add_feature_at(df, frame_column, feature, feature_column, multiplier=1):
    """
    Adds an feature at a frame of interest to the dataframe.

    Parameters:
    df: DataFrame
        The dataframe
    frame_column: str
        The name of the column that contains the frame at which to calculate the feature.
    feature: str
        The name of the feature to add.
    feature_column: str
        The name of the column that contains the feature.
    multiplier: float, optional
        A multiplier to apply to the feature. Default is 1.

    Returns:
    df: DataFrame
        The dataframe with the added feature column.
    """
    for tid, df_track in df.groupby("track_id"):
        frame = df_track.iloc[0][frame_column]
        value = df_track.loc[df_track["index_sequence"] == frame, feature_column].iloc[0]
        df.loc[df.track_id == tid, f"{feature}_at_{FRAME_COL.get(frame_column)}"] = (
            value * multiplier
        )
    return df

def add_mean_feature_over_trajectory(df, feature_list, multiplier_list):
    """
    Add the mean of a given feature over the growth trajectory 
    from transition to frame breakdown. 
    
    Parameters
    ----------
    df : DataFrame
        The dataframe
    feature_list : list
        List of column names
    multiplier_list : list
        List of scale to multiply the mean by
        
    Returns
    -------
    df : DataFrame
        The dataframe with the added mean feature columns
    """
    for feature, multiplier in zip(feature_list, multiplier_list):
        for tid, dft in df.groupby("track_id"):
            start = dft.frame_transition.values[0]
            stop = dft.Fb.values[0]
            df_mean = dft[(dft['index_sequence'] >= start) & (dft['index_sequence'] <= stop)]
            mean = df_mean[feature].mean() * multiplier
            df.loc[df.track_id == tid, f"mean_{feature}"] = mean
    return df


def get_early_transient_gr_of_whole_colony(df, scale, time_shift=25):
    """
    Get the transient growth rate of the colony 2 hours into the growth trajectory. 
    
    This time shift of two hours into the growth trajectory is necessary because the metric
    is calculated as the average of a 4 hour rolling window. The middle of a four hour window
    does not occur until two hours into the timelapse. To calculate this feature equivalently 
    for each trajectory, two hours was used for all tracks to get a metric for the transient 
    growth rate of the colony early in the growth trajectory. 
    
    Parameters
    ----------
    df : DataFrame
        The dataframe
    time_shift : int
        The time shift in frames to calculate the transient growth rate in frames
        
    Returns
    -------
    df : DataFrame
        The dataframe with the added transient growth rate feature columns
    """
    for tid, dft in df.groupby("track_id"):
        t_calculate = dft.index_sequence.min() + time_shift
        transient_gr_whole_colony = df.loc[df.index_sequence == t_calculate, "neighbor_avg_dxdt_48_volume_whole_colony"].values[0]
        df.loc[df.track_id == tid, "early_transient_gr_whole_colony"] = transient_gr_whole_colony * scale

    return df
    

def add_std_feature_over_trajectory(df, feature_list, multiplier_list):
    """
    Add the standard deviation of a given feature over the growth trajectory 
    from transition to frame breakdown. 
    
    Parameters
    ----------
    df : DataFrame
        The dataframe
    feature_list : list
        List of column names
    multiplier_list : list
        List of scale to multiply the std by
        
    Returns
    -------
    df : DataFrame
        The dataframe with the added standard deviation feature columns
    """
    for feature, multiplier in zip(feature_list, multiplier_list):
        for tid, dft in df.groupby("track_id"):
            start = dft.frame_transition.values[0]
            stop = dft.Fb.values[0]
            df_std = dft[(dft['index_sequence'] >= start) & (dft['index_sequence'] <= stop)]
            std = df_std[feature].std() * multiplier
            df.loc[df.track_id == tid, f"std_{feature}"] = std
    return df


def add_volume_at(df, pixel_size, frame_column):
    """
    Calculate the volume in units of microns cubed at specific timepoints.

    Parameters
    ----------
    df : DataFrame
        The dataframe
    pixel_size : float
        The size of a pixel in microns.
    frame_column : str
        "Ff", "frame_transition", or "Fb"

    Returns
    -------
    df: DataFrame
        The calculated volume in units of microns cube at specified frame in new column.
    """
    return add_feature_at(df, frame_column, "volume", "volume", pixel_size**3)


def add_location_at(df, frame_column, coordinate):
    """
    Get the centroid location at specific timepoints.

    Parameters
    ----------
    df : DataFrame
        The dataframe
    frame_column : str
        "Ff", "frame_transition", or "Fb"
    coordinate : str
        "x" or "y"

    Returns
    -------
    df: DataFrame
        The centroid location in units of pixels at specified frame in new column.
    """
    return add_feature_at(df, frame_column, f"location_{coordinate}", f"centroid_{coordinate}")


def add_time_at(df, frame_column, interval):
    """
    Get time in units of hours at specific timepoints.

    Parameters
    ----------
    df : DataFrame
        The dataframe
    frame_column : str
        "Ff", "frame_transition", or "Fb"
    interval : int
        Time interval between frames in minutes

    Returns
    -------
    df: DataFrame
        The time in units of hours at specified frame in new column.
    """
    return add_feature_at(df, frame_column, "time", "index_sequence", interval / 60)


def add_colony_time_at(df, frame_column, interval):
    """
    Get colony time in units of hours at specific timepoints.

    Parameters
    ----------
    df : DataFrame
        The dataframe
    frame_column : str
        "Ff", "frame_transition", or "Fb"
    interval : int
        Time interval between frames in minutes

    Returns
    -------
    df: DataFrame
        The colony time in units of hours at specified frame in new column.
    """
    return add_feature_at(df, frame_column, "colony_time", "colony_time", interval / 60)


def add_duration_in_frames(df, start_column, end_column):
    """
    Calculate duration in units of frames

    Paramaters:
    ----------
    df: DataFrame
        The dataframe
    start_col: str
        The name of the column that contains the start frame.
    end_col: str
        The name of the column that contains the end frame.

    Returns:
    df: Dataframe
        The dataframe with the added duration column.
    """
    df = df.copy()
    df[f"duration_{FRAME_COL.get(start_column)}{FRAME_COL.get(end_column)}"] = (
        df[end_column] - df[start_column]
    )
    return df


def add_fold_change_track_fromB(df, feature_name, feature_column, multiplier=1):
    """
    Calculate fold change of a feature at all times relative to its value at B

    Parameters
    ----------
    df : DataFrame
        The dataframe
    feature_name: string
        Clear shorthand name for feature
    feature_column: string
        Column name for feature
    multiplier: float
        Multiplier to convert feature to real units

    Returns
    -------
    df: DataFrame
        The feature fold change relative to B at each timepoint is added as a column
    """
    if f"{feature_name}_at_B" not in df.columns:
        df = add_feature_at(df, "frame_transition", feature_name, feature_column, multiplier)
    df[f"{feature_name}_fold_change_fromB"] = (
        df[f"{feature_column}"] * multiplier / df[f"{feature_name}_at_B"]
    )
    return df


def add_feature_change_BC(df, feature_name, feature_column, multiplier=1):
    """
    Calculate the absolute and fold change from B-C for a feature

    Paramaters:
    ----------
    df: DataFrame
        The dataframe
    feature_name: string
        Clear shorthand name for feature
    feature_column: string
        Column name for feature
    multiplier: float
        Multiplier to convert feature to real units

    Returns:
    --------
    df: Dataframe
        The dataframe with the added absolute change and fold change columns.
    """
    if f"{feature_name}_at_B" not in df.columns:
        df = add_feature_at(df, "frame_transition", feature_name, feature_column, multiplier)
    if f"{feature_name}_at_C" not in df.columns:
        df = add_feature_at(df, "Fb", feature_name, feature_column, multiplier)
    df[f"delta_{feature_name}_BC"] = df[f"{feature_name}_at_C"] - df[f"{feature_name}_at_B"]
    df[f"{feature_name}_fold_change_BC"] = df[f"{feature_name}_at_C"] / df[f"{feature_name}_at_B"]
    return df


def add_full_track_flag(df):
    df["is_full_track"] = True
    return df


def add_family_id_all_datasets(df):
    """
    Adds a column called "family_id" that contains a unique id for each lineage tree manually annotated.

    Parameters
    ----------
    df: DataFrame
        loaded dataframe with the manifest of interest

    Returns
    --------
    df: DataFrame
        DataFrame with added column - "family_id"
    """

    df["family_id"] = np.nan

    df_with_lineage_annotations = df[df["colony"].isin(["small", "medium"])]

    root_list = lineage_trees.get_roots(
        df_with_lineage_annotations
    )  # get the first nucleus of each lineage tree
    for count, root in enumerate(root_list):
        dec_list = lineage_trees.get_ids_in_same_lineage_as(df_with_lineage_annotations, root)
        df_with_lineage_annotations.loc[
            df_with_lineage_annotations["track_id"].isin(dec_list), "family_id"
        ] = count
    df.loc[df_with_lineage_annotations.index, "family_id"] = df_with_lineage_annotations[
        "family_id"
    ]

    return df


def add_non_interphase_size_shape_flag(df):
    """
    This function flags rows in the dataset where the nuclear volume, surface area and/or SA/V ratio are outside the range of values expected for interphase nuclei.
    The "normal interphase nuclei" used to define the expected values are the `is_full_track`==True and `exiting_mitosis`==False nuclei. That is nuclei that are in interphase (well after exiting mitosis).
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

    Returns
    -------
    df: Dataframe
        Returns the input dataframe, with non-interphase nuclei FLAGGED

    """
    # these thresholds are determined by finding the minimum and maximum values for the given features from the full_tracks (that are after mitosis exit)
    # to be safe, make the threshold values 10% lower or 10% higher so that the filters are not likely to remove true interphase nuclei
    # {"column": min_thresh, max_thresh}

    # now add the flags
    for filter_key in NON_INTERPHASE_FILTER_THRESHOLDS.keys():
        min_thresh, max_thresh = NON_INTERPHASE_FILTER_THRESHOLDS[filter_key]

        # intialize the flag column
        df[f"non_interphase_{filter_key}"] = False
        if min_thresh is not None:
            df.loc[df[filter_key] < min_thresh, f"non_interphase_{filter_key}"] = True
        if max_thresh is not None:
            df.loc[df[filter_key] > max_thresh, f"non_interphase_{filter_key}"] = True

    # combine the flags using OR logic
    df["non_interphase_size_shape"] = df[
        [f"non_interphase_{filter_key}" for filter_key in NON_INTERPHASE_FILTER_THRESHOLDS.keys()]
    ].any(axis=1)
    return df

def get_sister(df, pid, current_tid):
    """
    Gets the track_id of the sibling

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe
    track_id: int
        The track_id of the cell

    Returns
    -------
    sister_id: List
        List containing the track_id of the sibling cell
    """
    df_sisters = df.loc[df.parent_id == pid]
    tids = df_sisters.track_id.unique()
    sister_id = [tid for tid in tids if tid != current_tid]
    return sister_id

def add_lineage_features(df, feature_list):
    """
    If the full track has a full track sister or mother, add the given relative's feature as a single track feature column in the dataframe. 
    
    Paramaters
    ----------
    df: DataFrame
        The dataframe
    feature_list: list
        List of column names
        
    Returns
    -------
    df: DataFrame
        The dataframe with new columns (ie mothers_vol_at_B, sisters_duration)
    """
    
    for feature in feature_list:
        df[f"mothers_{feature}"] = np.nan
        df[f"sisters_{feature}"] = np.nan

    df_lineage = df[df['colony'].isin(['small', 'medium'])]

    for tid, dft in df_lineage.groupby("track_id"):
        parent_id = dft.parent_id.values[0]
        if parent_id != -1 and parent_id in df_lineage.track_id.unique():
            for feature in feature_list:
                df.loc[df.track_id == tid, f"mothers_{feature}"] = df_lineage.loc[df_lineage.track_id == parent_id, feature].values[0]
        if parent_id != -1:        
            sister_id = get_sister(df_lineage, parent_id, tid)
            if len(sister_id) > 0:
                for feature in feature_list:
                    df.loc[df.track_id == tid, f"sisters_{feature}"] = df_lineage.loc[df_lineage.track_id == sister_id[0], feature].values[0]

    return df

def sum_events_along_full_track(df0, feature_list, index_columns=['track_id','index_sequence','Fb','frame_transition']):
    """
    this function sums the number of events along the full track of each nucleus. An example event is having a mitotic neighbor.
    the events are defined by the feature_list

    Parameters
    ----------
    df0 : pd.DataFrame
        dataframe that minimally has the following columns:
        [index_columns] + [feature_list]
    feature_list : list
        list of features to sum along the full track
        can be boolean or numerical
    index_columns : list, optional
        list of columns to keep in the final dataframe for subsetting to full tracks
        The default is ['track_id','index_sequence','Fb','frame_transition'].

    Returns
    -------
    df2 : pd.DataFrame
        dataframe with the sum of events along the full track
    """

    df = df0.dropna(subset=['frame_transition','Fb']) # extra NaN values in these columns can cause problems at log1 log2 steps
    log1 = df['index_sequence'] >= df['frame_transition'] # restrict track to start from frame_transition
    log2 = df['index_sequence'] <= df['Fb'] # restrict track to end at Fb (breakdown)
    log = log1 & log2
    df_mean = df.loc[log,index_columns+feature_list] # only keep relevant columns

    # group by track_id and calculate mean
    dfg = df_mean.groupby('track_id')[feature_list].sum()
    dfg.rename(columns={col: f'sum_{col}' for col in feature_list}, inplace=True)
    
    # merge the results
    df2 = df0.reset_index().merge(dfg, on='track_id', how='left', suffixes=('','_dup2'),).set_index('CellId')
    return df2


def sum_mitotic_events_along_full_track(df0, feature_list=[]):
    """
    this function calls sum_events_along_full_track with the default feature_list of mitotic events

    Parameters
    ----------
    df0 : pd.DataFrame
        dataframe that minimally has the following columns:
        ['track_id','index_sequence','Fb','frame_transition'] + mitotic_event_features
    mitotic_event_features : list, optional
        list of features to sum along the full track
        can be boolean or numerical
        The default is selected if list = [].
    
    Returns
    -------
    df2 : pd.DataFrame
        dataframe with the sum of events along the full track
    """

    mitotic_event_features = [
        'number_of_frame_of_breakdown_neighbors',
        'number_of_frame_of_formation_neighbors',
        'has_mitotic_neighbor_breakdown',
        'has_mitotic_neighbor_formation',
        'has_mitotic_neighbor_breakdown_forward_dilated',
        'has_mitotic_neighbor_formation_backward_dilated',
        'has_mitotic_neighbor',
        'has_mitotic_neighbor_dilated',
        'has_dying_neighbor',
        'has_dying_neighbor_forward_dilated',
        'number_of_frame_of_death_neighbors'
    ]

    if len(feature_list) == 0:
        feature_list = mitotic_event_features

    return sum_events_along_full_track(df0, feature_list)

