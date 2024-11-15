# %%
import numpy as np
from nuc_morph_analysis.lib.preprocessing import filter_data
from nuc_morph_analysis.lib.preprocessing.filter_data import all_timepoints_minimal_filtering

BIN_INTERVAL_LIST = [48]
DXDT_FEATURE_LIST = ["volume"]
DXDT_PREFIX = "dxdt_"


def compute_change_over_time_on_dataframe(dfpi,bin_interval,time_cols,prefix,time_location='center'):
    # now we want to compute the difference
    # the difference is the value at timepoint t+bin_interval - the value at timepoint t
    
    if time_location=='center':
        # because we want the difference centered at timepoint t, we will shift the difference by bin_interval//2
        diff = dfpi.diff(axis=0, periods=bin_interval).shift(-1 * bin_interval // 2)
        suffix=''
    elif time_location=='end':
        diff = dfpi.diff(axis=0, periods=bin_interval)
        suffix='_end'

    # now normalize the changes by the bin_interval
    diff = diff / float(bin_interval)
    # now transform diff back into the form of dfd
    dfm = diff.stack().reset_index()
    dfm = dfm.rename(columns={x: f"{prefix}{x}{suffix}" for x in time_cols})
    return dfm

def get_change_over_time_array(dfd, time_cols, bin_interval, time_location='center'):
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

    dfm = compute_change_over_time_on_dataframe(dfp, bin_interval, time_cols, prefix, time_location)
    # now drop rows with nan values
    # dfm = dfm.dropna(axis=0,how='all')
    
    # now recover the CellId values
    dfmi = dfm.set_index(["index_sequence", "track_id"])
    dfdi = dfd.set_index(["index_sequence", "track_id"])

    # find all dfmi index values NOT in dfdi index values
    # this is the set of index values that are not in the original dataframe
    # they are added during the pivot operation
    # we will drop these rows
    not_in_dfdi = dfmi.index.difference(dfdi.index)
    # print(f"dropping {len(not_in_dfdi)} rows")
    dfmi.drop(not_in_dfdi, inplace=True)

    dfmi.loc[dfmi.index.values, "CellId"] = dfdi.loc[dfmi.index.values, "CellId"]
    return dfmi.reset_index().set_index("CellId")


# %%
def run_script(df=None, dxdt_feature_list = None, bin_interval_list=None, exclude_outliers=True, time_location='center'):
    """
    run the compute_change_over_time workflow for a given bin_interval

    Parameters
    ----------
    df : pd.DataFrame
        dataframe on which to compute change_over_time
        with columns ['colony','track_id','index_sequence','label_img']+time_cols
    dxdt_feature_list : list
        list of columns to compute growth on
    bin_interval_list : list
        list of integers, which represents the number of frames to compute growth over
    exclude_outliers : bool
        if True, exclude outlier time points from the growth rate calculation
    time_location : str
        'center' or 'end', determines where the change over time value is returned in the bin_interval
        default is 'center' (e.g. for bin_interval=48, the change over time value is returned at timepoint 24)
        when 'end', the change over time value is returned at timepoint 0

    Returns
    -------
    pd.DataFrame
        dataframe with change_over_time values for each track at each time point
    """
    if dxdt_feature_list is None:
        dxdt_feature_list = DXDT_FEATURE_LIST
    if bin_interval_list is None:
        bin_interval_list = BIN_INTERVAL_LIST

    assert df.index.name == "CellId"
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
        dfo = get_change_over_time_array(dfd, dxdt_feature_list, bin_interval, time_location)
        new_columns = [x for x in dfo.columns.tolist() if x not in dforig.columns.tolist()]
        # add new columns to original dataframe
        dforig.loc[dfo.index.values, new_columns] = dfo.loc[dfo.index.values, new_columns]
    assert dforig.index.name == "CellId"
    return dforig


def add_dvdt_over_V(df,columns=None,volume_col = 'volume'):
    """
    adds dvdt over V for all timepoints if dxdt_{time}_volume columns exist

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with columns  columns + ['volume']
    columns : list
        list of columns to compute change normalized by volume
    volume_col : str
        name of the volume column, default is 'volume'

    Returns
    -------
    df : pd.DataFrame
        dataframe with columns ['{column}_per_V'] added
    """
    if columns is None:
        columns = [f"{DXDT_PREFIX}{bin_interval}_{feature}" for bin_interval in BIN_INTERVAL_LIST for feature in DXDT_FEATURE_LIST]

    for col in columns:
        df[f"{col}_per_V"] = df[col] / df[volume_col]
    return df
