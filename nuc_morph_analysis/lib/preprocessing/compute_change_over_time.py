# %%
import numpy as np
from nuc_morph_analysis.lib.preprocessing import filter_data
from nuc_morph_analysis.lib.preprocessing.filter_data import all_timepoints_minimal_filtering

BIN_INTERVAL_LIST = [48]
DXDT_FEATURE_LIST = ["volume"]
DXDT_PREFIX = "dxdt_"


def get_change_over_time_array(dfd, time_cols, bin_interval):
    """
    Compute a rolling window-based change_over_time for all tracks.
    The change_over_time at T=t is computed as the difference between the values at t+bin_interval/2 divided and t-bin_interval/2 divided by bin_interval.

        Parameters
    ----------
    dfd : pd.DataFrame
        dataframe with columns ['track_id','index_sequence',time_cols]
    time_cols : list
        list of columns to compute growth on
    bin_interval : int
        number of frames to compute growth over
    Returns
    -------
    pd.DataFrame
        dataframe with columns ['track_id','index_sequence','CellId'] + ['dxdt_{bin_interval}_volume_per_V'] + ['dxdt_{bin_interval}_{feature}' for feature in time_cols]
    """
    prefix = f"{DXDT_PREFIX}{bin_interval}_"
    # create a dataframe where the index is index_sequence and the columns are track_id and the values are on of the columns from time_cols
    dfp = dfd.pivot(index="index_sequence", columns="track_id", values=time_cols)
    # ensure that all timpepoints are present in dfp index_sequence
    if dfp.index.values.tolist() != list(range(dfp.index.values.min(), dfp.index.values.max() + 1)):
        # if not, fill in missing timepoints with np.nan
        dfp = dfp.reindex(index=range(dfp.index.values.min(), dfp.index.values.max() + 1))

    # now we want to compute the difference
    # the difference is the value at timepoint t+bin_interval - the value at timepoint t
    # because we want the difference centered at timepoint t, we will shift the difference by bin_interval//2
    diff = dfp.diff(axis=0, periods=bin_interval).shift(-1 * bin_interval // 2)
    # now normalize the changes by the bin_interval
    diff = diff / float(bin_interval)

    #  now in addition to the volume level, add volume_per_V as a level to diff
    diff = diff.join(diff[["volume"]].div(dfp["volume"]).rename(columns={"volume": "volume_per_V"}))
    # now transform diff back into the form of dfd
    dfm = diff.stack().reset_index()
    dfm = dfm.rename(columns={x: f"{prefix}{x}" for x in time_cols})
    dfm = dfm.rename(columns={"volume_per_V": f"{prefix}volume_per_V"})

    # now drop rows with nan values
    dfm = dfm.dropna(axis=0)

    # now recover the CellId values
    dfmi = dfm.set_index(["index_sequence", "track_id"])
    dfdi = dfd.set_index(["index_sequence", "track_id"])
    dfmi.loc[dfmi.index.values, "CellId"] = dfdi.loc[dfmi.index.values, "CellId"]
    return dfmi.reset_index().set_index("CellId")


# %%
def run_script(df=None, bin_interval_list=BIN_INTERVAL_LIST, dxdt_feature_list=DXDT_FEATURE_LIST, exclude_outliers=True):
    """
    run the compute_change_over_time workflow for a given bin_interval

    Parameters
    ----------
    df : pd.DataFrame
        dataframe on which to compute change_over_time
        with columns ['colony','track_id','index_sequence','label_img']+time_cols
    bin_interval_list : list
        list of integers that represents the number of frames to compute growth over
    exclude_outliers : bool
        if True, exclude outlier time points from the growth rate calculation
    dxdt_feature_list : list
        list of features to compute growth rate on, default is DXDT_FEATURE_LIST

    Returns
    -------
    pd.DataFrame
        dataframe with change_over_time values for each track at each time point
    """

    dforig = df.copy()
    if exclude_outliers:
        # to ensure that outlier datapoints are used for the growth rate calculation, filter out time point outliers here
        df = all_timepoints_minimal_filtering(df)
        df = filter_data.filter_out_cells_entering_or_exiting_mitosis(df)
        df = filter_data.filter_out_non_interphase_size_shape_flag(df)

    # only keep the necessary columns; remove shcoeffs columns to dataframe is no so large
    # CellId becomes a column too after reset_index
    dfd = df[
        ["colony", "track_id", "index_sequence", "label_img"] + dxdt_feature_list
    ].reset_index()

    # convert all time_cols to float32
    dfd[dxdt_feature_list] = dfd[dxdt_feature_list].astype(np.float32)

    # returns dfo with index=CellId
    for bin_interval in bin_interval_list:
        dfo = get_change_over_time_array(dfd, dxdt_feature_list, bin_interval)
        new_columns = [x for x in dfo.columns.tolist() if x not in dforig.columns.tolist()]
        # add new columns to original dataframe
        dforig.loc[dfo.index.values, new_columns] = dfo.loc[dfo.index.values, new_columns]
    return dforig
