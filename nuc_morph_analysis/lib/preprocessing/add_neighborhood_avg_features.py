# %%
import pandas as pd
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count
from nuc_morph_analysis.lib.preprocessing.load_data import get_dataset_pixel_size
from nuc_morph_analysis.lib.preprocessing import filter_data
from nuc_morph_analysis.lib.preprocessing.filter_data import all_timepoints_minimal_filtering

LOCAL_RADIUS_LIST = [90, -1]
LOCAL_RADIUS_STR_LIST = ["90um", "whole_colony"]
NEIGHBOR_FEATURE_LIST = ["volume"]
NEIGHBOR_PREFIX = "neighbor_avg_"


def run_in_parallel(args):
    return get_neighbor_avg_at_t(*args)


def get_neighbor_avg_at_t(dft, local_radius_list, columns, min_neighbor_thresh=5):
    """
    Compute the average value of the columns for each CellId at a given timepoint within a given colony
    by finding its neighbors within a given radius=local_radius

    Parameters
    ----------
    dft : pd.DataFrame
        dataframe that minimally has the following columns:
        ['index_sequence','track_id','centroid_x','centroid_y']
        with index = "CellId"
    local_radius_list : list
        list of integers that represent the radius in microns to use for finding neighbors
        -1 signifies the whole colony
    columns : list
        list of column names to compute the average value for
    min_neighbor_thresh : int
        minimum number of neighbors required to compute the average value

    Returns
    -------
    dft : pd.DataFrame
        dataframe with the average value of the columns for each CellId at a given timepoint within a given colony
    """

    # now get the centroid values for all CellIds at that index sequence
    centroid_xy = dft[["centroid_x", "centroid_y"]].values
    # now compute pair-wise distances between all centroid_xy values
    dists = np.linalg.norm(centroid_xy[:, None] - centroid_xy[None], axis=2)

    # initialize the new columns to be added to the dataframe
    # to avoid the PerformanceWarning: DataFrame is highly fragmented occuring in the last line of this function
    data = {}
    for li, local_radius in enumerate(local_radius_list):
        local_radius_str = LOCAL_RADIUS_STR_LIST[li]
        for column in columns:
            data[f"{NEIGHBOR_PREFIX}{column}_{local_radius_str}"] = np.nan

    dftnew = pd.DataFrame(data, index=dft.index)

    for li, local_radius in enumerate(local_radius_list):
        local_radius_str = LOCAL_RADIUS_STR_LIST[li]
        # now for each CellId find the neighbors within a circle with radius = radius
        pix_size = get_dataset_pixel_size(dft.colony.values[0])
        radius = local_radius / pix_size  # pixels
        if local_radius < 0:
            neighbors = np.ones(dists.shape, dtype=bool)
        else:
            neighbors = dists < radius

        # don't allow "self" to be a neighbor
        np.fill_diagonal(neighbors, False)

        for column in columns:
            col_vals1d = dft[column].values
            col_vals2d = np.tile(col_vals1d, (col_vals1d.size, 1)).astype(float)
            # now to make the diagonal values nan so that the cell of interest is not included in the average, but only the neighbors
            col_vals2d[~neighbors] = np.nan

            # some rows could be all nan
            # so we need to track the indices of the rows that are not all nan
            # and only use those indices to update the dataframe
            # this is important for avoiding an annoyint RuntimeWarning: Mean of empty slice
            non_nan_neighbors_per_row = np.sum(~np.isnan(col_vals2d), axis=1)

            non_nan_rows = non_nan_neighbors_per_row >= min_neighbor_thresh
            col_vals2d = col_vals2d[non_nan_rows]

            # now compute the average of the column values for each row (average of the neighbors)
            # Compute the average value of the column for each neighborhood
            col_vals_avg = np.nanmean(col_vals2d, axis=1)

            # Add the average values to the dataframe
            # dft.loc[non_nan_rows,f'{prefix}{column}_{local_radius_str}'] = col_vals_avg
            dftnew.loc[
                dft.index.values[non_nan_rows], f"{NEIGHBOR_PREFIX}{column}_{local_radius_str}"
            ] = col_vals_avg
    dft = dft.join(dftnew)
    return dft


def run_script(
    df,
    num_workers=1,
    feature_list=NEIGHBOR_FEATURE_LIST,
    local_radius_list=LOCAL_RADIUS_LIST,
    exclude_outliers=True,
):
    """
    Determine average values of features within neighborhoods of defined radius around each cell

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with index=CellId, and the following columns minimally:
        ['index_sequence','track_id','centroid_x','centroid_y'] + feature_list + [x for x in df.columns if 'dxdt' in x]
    num_workers : int, optional
        number of workers to use for parallel processing. The default is 1.
    feature_list : list, optional
        list of column names that represent the features to proces. The default is NEIGHBOR_FEATURE_LIST
    local_radius_list : list, optional
        list of integers that represent the radius in microns to use for finding neighbors. The default is LOCAL_RADIUS_LIST.
        -1 signifies the whole colony
    exclude_outliers : bool, optional
        whether to exclude outliers. The default is False.

    Returns
    -------
    dforig : pd.DataFrame
        original dataframe with the newly computed columns added
        index of dataframe = CellId
    """

    dforig = df.copy()
    original_columns = dforig.columns.tolist()
    if exclude_outliers:
        # to ensure that outlier data points are not used for the neighborhood avgs, filter out time point outliers here
        df = all_timepoints_minimal_filtering(df)

        # also be sure to filter out non-interphase cells from neighborhood
        df = filter_data.filter_out_non_interphase_size_shape_flag(df)
        df = filter_data.filter_out_cells_entering_or_exiting_mitosis(df)

    for colony in df.colony.unique():
        dfi = df[df["colony"] == colony]
        pass_cols = ["index_sequence", "colony", "track_id", "centroid_x", "centroid_y"]

        columns = feature_list + [x for x in dfi.columns if "dxdt" in x]

        # first find the unique index_sequence values
        index_sequences = dfi["index_sequence"].unique()
        dfin = dfi[
            pass_cols + columns
        ]  # reduce the size of the dataframe being passed in by only including necessary columns

        # run in parallel
        args_list = [
            (dfin[dfin["index_sequence"] == t], local_radius_list, columns) for t in index_sequences
        ]
        num_workers = np.min([num_workers, cpu_count()])
        if num_workers == 1:
            results = list(map(run_in_parallel, args_list))
        else:
            results = list(Pool(num_workers).imap_unordered(run_in_parallel, args_list))
        dft = pd.concat(results)
        new_cols = [x for x in dft.columns if x not in original_columns]
        dforig.loc[dft.index, new_cols] = dft[new_cols]
    return dforig
