import numpy as np
import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, peak_widths
import numpy as np
# find peaks on volume_sub_fit_volume
def find_drops_relative_to_fit(y,prominence=(40,None), width=(2,24),height=(20,None),threshold=(None,None),rel_height=0.5,wlen=25):
    """
    TODO: interpolate the peaks to get a more accurate estimate of the peak width
    """
    peaks, props = find_peaks(y, prominence=prominence, width=width,height=height,threshold=threshold,rel_height=rel_height,wlen=wlen)
    results_full = peak_widths(y, peaks, rel_height=rel_height)
    return peaks, props, results_full

def remove_peaks(y,props):
    nanmask = np.ones(y.shape,dtype='float32')
    if props['left_bases'].size == 0:
        return y,nanmask
    peak_list = [np.arange(x,y+1,1,dtype='uint16') for x,y in zip(props['left_bases'],props['right_bases'])]
    peak_list = np.concatenate(peak_list)
    y_filt = y.copy()
    y_filt[peak_list] = np.nan
    nanmask[peak_list] = np.nan
    return y_filt, nanmask


def find_and_remove(y):
    peaks, props, results_full = find_drops_relative_to_fit(y)
    y_filt,_ = remove_peaks(y,props)
    return y_filt

def find_and_remove_from_pivot(y,return_mask=False):
    peaks, props, results_full = find_drops_relative_to_fit(y)
    _,y_mask = remove_peaks(y,props)

    peak_centers_bool_array = np.zeros(y.shape,dtype='bool')
    peak_centers_bool_array[peaks] = True
    if return_mask:
        return y_mask
    else:
        return peak_centers_bool_array

def get_fit_volume_minus_smoothed_y(volume,fit_volume, return_all=False):
    smooth_volume = savgol_filter(volume, window_length=12, polyorder=2, mode='constant', cval=np.nan)
    y = smooth_volume - fit_volume
    if return_all:
        return y, fit_volume, smooth_volume
    else:
        return y

def get_fit_volume_minus_smoothed_y_from_df(dftrack, return_all=False):
    yscale, ylabel, yunit, _ = get_plot_labels_for_metric("volume")
    volume = dftrack['volume'].values * yscale
    fit_volume = dftrack['fit_volume'].values
    if return_all:
        y,_,smooth_volume = get_fit_volume_minus_smoothed_y(volume,fit_volume, return_all=return_all)
    else:
        y = get_fit_volume_minus_smoothed_y(volume,fit_volume, return_all=return_all)
    
    if return_all:
        return y, volume, fit_volume, smooth_volume
    else:
        return y

def plot_features_and_peaks(dftrack,volume,smooth_volume,fit_volume,peaks,props,ax):
    x = dftrack['index_sequence'].values
    track_id = dftrack['track_id'].values[0]
    ax.plot(x,volume)
    ax.plot(x,smooth_volume,'g',zorder=-200,linewidth=2,)
    ax.plot(x,fit_volume,'r--')
    y = smooth_volume - fit_volume
    peaks, props, results_full = find_drops_relative_to_fit(-y)
    ax.plot(x[peaks],fit_volume[peaks],'r+')
    y_filt,_ = remove_peaks(volume,props)
    ax.plot(x,y_filt,'k--')
    # plot the widths
    for left,right in zip(props['left_bases'],props['right_bases']):
        yh = np.max(fit_volume[left:right])
        ax.plot([x[left],x[right]],[yh,yh],'g-')
    plt.title(f"track {track_id}, prom={props['prominences']}")
    text = '\n'.join([f"{k}: {v}" for k,v in props.items()])
    ax.text(0.00,0.99,
               f"{text}",
               ha='left',va='top',transform=ax.transAxes)
    
    return y_filt


#%%
def filter_out_volume_dips(dfd, volume_cols,):
    """
    Remove the volume dips from the volume data

        Parameters
    ----------
    dfd : pd.DataFrame
        dataframe with columns ['track_id','index_sequence'] + volume_cols
    volume_cols : list
        list of columns to needed to find and filter out volume dips

    Returns
    -------
    pd.DataFrame
        dataframe with columns ['track_id','index_sequence','CellId'] + ['smooth_volume_dips_removed_um','volume_dips_removed_um','volume_drop_mask','volume_drop_centers']
    """
    # create a dataframe where the index is index_sequence and the columns are track_id and the values are on of the columns from time_cols
    dfp = dfd.pivot(index="index_sequence", columns="track_id", values=['volume','fit_volume'])

    # ensure that all timpepoints are present in dfp index_sequence
    if dfp.index.values.tolist() != list(range(dfp.index.values.min(), dfp.index.values.max() + 1)):
        # if not, fill in missing timepoints with np.nan
        dfp = dfp.reindex(index=range(dfp.index.values.min(), dfp.index.values.max() + 1))
    # interpolate the missing internal nan values while keeping the beginning and ending stretch of nans
    # (this is important so the savgol filter will not have gaps)
    dfp = dfp.interpolate(method='linear', axis=0, limit_area='inside') 

    # now apply savitzky golay filter to each column
    # dfp_sg = dfp.apply(lambda x: savgol_filter(x, window_length=12, polyorder=2, mode='constant', cval=np.nan), axis=0)
    yscale, _, _, _ = get_plot_labels_for_metric("volume")
    dfp_vol_sg = dfp['volume'].apply(lambda x: savgol_filter(x*yscale, window_length=12, polyorder=2, mode='constant', cval=np.nan), axis=0)
    dfp_fit_vol = dfp['fit_volume']
    dfp_vol_sg_sub_fit = dfp_vol_sg - dfp_fit_vol

    # now find peaks and remove them in the volume_sub_fit_volume
    # using find_and_remove_from_pivot
    dfp_mask = dfp_vol_sg_sub_fit.apply(lambda x: find_and_remove_from_pivot(-x,return_mask=True),axis=0)
    dfp_centers = dfp_vol_sg_sub_fit.apply(lambda x: find_and_remove_from_pivot(-x,return_mask=False),axis=0)

    dfp_vol1 = dfp_vol_sg * dfp_mask.values
    dfp_vol2 = dfp['volume'] * dfp_mask.values
    dfp_vol2 = dfp_vol2 * yscale # convert to um^3
    
    # fill in the gaps
    dfp_vol1 = dfp_vol1.interpolate(method='linear', axis=0, limit_area='inside') 
    dfp_vol2 = dfp_vol2.interpolate(method='linear', axis=0, limit_area='inside') 


    dfm1 = dfp_vol1.stack().reset_index()
    dfm1.rename(columns={0:'smooth_volume_dips_removed_um'},inplace=True)
    
    dfm2 = dfp_vol2.stack().reset_index()
    dfm2.rename(columns={0:'volume_dips_removed_um'},inplace=True)
    dfm = dfm1.merge(dfm2,on=['index_sequence','track_id'],how='outer')

    dfm3 = dfp_mask.isna().stack().reset_index()
    dfm3.rename(columns={0:'volume_drop_mask'},inplace=True)
    dfm = dfm.merge(dfm3,on=['index_sequence','track_id'],how='outer')


    dfm4 = dfp_centers.stack().reset_index()
    dfm4.rename(columns={0:'volume_drop_centers'},inplace=True)
    dfm = dfm.merge(dfm4,on=['index_sequence','track_id'],how='outer')

    # now recover the CellId values
    dfmi = dfm.set_index(["index_sequence", "track_id"])
    dfdi = dfd.set_index(["index_sequence", "track_id"], drop=False)

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
def run_script(df=None,volume_cols=['volume','fit_volume']):
    """
    run the workflow

    Parameters
    ----------
    df : pd.DataFrame
        dataframe on which to compute change_over_time
        with columns ['colony','track_id','index_sequence','label_img']+time_colsion

    Returns
    -------
    pd.DataFrame
        dataframe with volume dips filtered out values for each track at each time point
        new columns are ['smooth_volume_dips_removed_um','volume_dips_removed_um']
        both are in units of um^3
    """
    
    assert df.index.name == "CellId"
    dforig = df.copy()

    # only keep the necessary columns; remove shcoeffs columns to dataframe is no so large
    # CellId becomes a column too after reset_index
    dfd = df[
        ["colony", "track_id", "index_sequence", "label_img"] + volume_cols
    ].reset_index()

    # convert all time_cols to float32
    dfd[volume_cols] = dfd[volume_cols].astype(np.float32)

    # returns dfo with index=CellId
    dfo = filter_out_volume_dips(dfd, volume_cols)
    new_columns = [x for x in dfo.columns.tolist() if x not in dforig.columns.tolist()]
    # add new columns to original dataframe
    dforig.loc[dfo.index.values, new_columns] = dfo.loc[dfo.index.values, new_columns]
    assert dforig.index.name == "CellId"
    return dforig
