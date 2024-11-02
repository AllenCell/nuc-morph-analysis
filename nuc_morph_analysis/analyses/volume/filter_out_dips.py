import numpy as np
import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, peak_widths
import numpy as np
import pandas as pd
# find peaks on volume_sub_fit_volume
# def find_drops_relative_to_fit(y,prominence=(40,None), width=(2,24),height=(20,None),threshold=(None,None),rel_height=0.5,wlen=25):
def find_drops_relative_to_fit(y,
                               prominence=(20,None),
                                 width=(2,24),
                                 height=(20,None),
                                 threshold=(None,None),
                                 rel_height=1,
                                 wlen=25):
    """
    TODO: interpolate the peaks to get a more accurate estimate of the peak width
    """
    peaks, props = find_peaks(y, prominence=prominence, width=width,height=height,threshold=threshold,rel_height=rel_height,wlen=wlen)
    results_full = peak_widths(y, peaks, rel_height=rel_height)
    return peaks, props, results_full

def dataframeify_peaks(y,track_id):
    peaks, props, results_full = find_drops_relative_to_fit(y)
    feats = {}
    dflist = []
    feats.update({'track_id':track_id})
    if len(peaks)==0:
        feats.update({'peak':np.nan})
        for pi, peak in enumerate(peaks):
            feats.update({f'peak':peak})
            feats.update({f'peak_num':pi})
            for k,v in props.items():
                feats.update({k:v[pi]})
            dflist.append(pd.DataFrame(data=feats.values(),index=feats.keys()).T)
    return pd.concat(dflist)


            
def remove_peaks(y,props):
    boolmask = np.zeros(y.shape,dtype='bool')
    if props['left_bases'].size == 0:
        return y,boolmask
    peak_list = [np.arange(x,y+1,1,dtype='uint16') for x,y in zip(props['left_bases'],props['right_bases'])]
    peak_list = np.concatenate(peak_list)
    y_filt = y.copy()
    y_filt[peak_list] = True
    boolmask[peak_list] = True
    return y_filt, boolmask


def find_and_remove(y):
    peaks, props, results_full = find_drops_relative_to_fit(y)
    y_filt,_ = remove_peaks(y,props)
    return y_filt

def find_and_remove_peaks_combined(vol_det_array, vol_array, index_sequence_vec, track_id_vec, peak_str='drops'):

    # apply find_drops_relative_to_fit(y) to each column of vol_det_array
    peaks_array, props_array, results_full_array = [], [], []
    min_index = index_sequence_vec.min()
    out = [find_and_remove_from_pivot_2d(vol_det_array[:,col],vol_array[:,col],min_index=min_index,peak_str=peak_str) for col in range(vol_det_array.shape[1])]
    # flatten the output
    keys = [f'volume_{peak_str}_mask',f'volume_{peak_str}_centers',f'volume_{peak_str}_has_peak',f'volume_{peak_str}_prom',f'volume_{peak_str}_left_bases',f'volume_{peak_str}_right_bases',f'volume_{peak_str}_magnitude', f'volume_{peak_str}_max_magnitude']

    dfout_list = [pd.DataFrame(x,columns = index_sequence_vec, index=keys).T for x in out]
    dfout_list = [x.reset_index().rename(columns={'index':'index_sequence'}).set_index('index_sequence') for x in dfout_list]
    # keys = ['volume_drops_mask','volume_drops_centers','volume_drops_has_peak','volume_drops_prom','volume_drops_left_base','volume_drops_right_base','volume_drops_y2_magnitude']
    dfout = pd.concat(dfout_list,axis=0,keys=track_id_vec, names=['track_id']).reset_index()
    
    return dfout
    

def find_and_remove_from_pivot_2d(y,y2,select_return='all',min_index=0,peak_str='drops'):
    peaks, props, results_full = find_drops_relative_to_fit(y)
    _,y_mask = remove_peaks(y,props)

    peak_centers_bool_array = np.zeros(y.shape,dtype='bool')
    peak_centers_bool_array[peaks] = True

    has_peak_bool_array = np.zeros(y.shape,dtype='bool') if len(peaks)==0 else np.ones(y.shape,dtype='bool')
    
    # initialize the peak_magnitude_array as nans
    prom_array = np.zeros(y.shape,dtype='float32') * np.nan
    left_base_array = np.zeros(y.shape,dtype='float32') * np.nan
    right_base_array = np.zeros(y.shape,dtype='float32') * np.nan
    y2_magnitude_array = np.zeros(y.shape,dtype='float32') * np.nan
    if len(peaks)>0:
        for pi, peak in enumerate(peaks):
            prom_array[peak] = props['prominences'][pi]
            left_base_array[peak] = props['left_bases'][pi].copy() + min_index
            right_base_array[peak] = props['right_bases'][pi] + min_index

            left_base = int(props['left_bases'][pi].copy())
            y2_magnitude_array[peak] = y2[left_base] - y2[peak]
    
    func = np.nanmin if peak_str=='drops' else np.nanmax
    max_val = func(y2_magnitude_array) if len(peaks)>0 else np.nan
    max_peak_val_array = np.ones(y.shape,dtype='float32') * max_val
    if select_return=='all':
        return y_mask, peak_centers_bool_array, has_peak_bool_array, prom_array, left_base_array, right_base_array, y2_magnitude_array, max_peak_val_array


def find_and_remove_from_pivot(y,select_return='mask',min_index=0,y2=None):
    peaks, props, results_full = find_drops_relative_to_fit(y)
    _,y_mask = remove_peaks(y,props)

    peak_centers_bool_array = np.zeros(y.shape,dtype='bool')
    peak_centers_bool_array[peaks] = True

    has_peak_bool_array = np.zeros(y.shape,dtype='bool') if len(peaks)==0 else np.ones(y.shape,dtype='bool')
    
    # initialize the peak_magnitude_array as nans
    prom_array = np.zeros(y.shape,dtype='float32') * np.nan
    left_base_array = np.zeros(y.shape,dtype='float32') * np.nan
    right_base_array = np.zeros(y.shape,dtype='float32') * np.nan
    y2_magnitude_array = np.zeros(y.shape,dtype='float32') * np.nan
    peak_center_val_array = np.zeros(y.shape,dtype='float32') * np.nan
    if len(peaks)>0:
        for pi, peak in enumerate(peaks):
            prom_array[peak] = props['prominences'][pi]
            left_base_array[peak] = props['left_bases'][pi] + min_index
            right_base_array[peak] = props['right_bases'][pi] + min_index
            peak_center_val_array[peak] = peak + min_index

        if y2 is not None:
            for pi, peak in enumerate(peaks):
                left_base = int(left_base_array[peak])
                y2_magnitude_array[peak] = y2[left_base] - y2[peak]


    if select_return=='mask':
        return y_mask
    elif select_return=='center':
        return peak_centers_bool_array
    elif select_return=='center_val':
        return peak_center_val_array
    elif select_return=='magnitude':
        return prom_array
    elif select_return=='left_base':
        return left_base_array
    elif select_return=='right_base':
        return right_base_array
    elif select_return=='all':
        return y_mask, peak_centers_bool_array, has_peak_bool_array, prom_array, left_base_array, right_base_array

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

def get_fit_volume_minus_smoothed_y_from_df_dydx(dftrack, return_all=False):
    yscale, ylabel, yunit, _ = get_plot_labels_for_metric("volume")
    volume = dftrack['volume'].values * yscale
    fit_volume = dftrack['fit_volume'].values
    # determine first derivative
    if return_all:
        y,_,smooth_volume = get_fit_volume_minus_smoothed_y(volume,fit_volume, return_all=return_all)
        dydx = np.gradient(smooth_volume)
    
    else:
        y = get_fit_volume_minus_smoothed_y(volume,fit_volume, return_all=return_all)
    
    if return_all:
        return y, volume, fit_volume, smooth_volume,dydx
    else:
        return y

def plot_features_and_peaks(dftrack,volume,smooth_volume,fit_volume,peaks,props,results_full,ax):
    x = dftrack['index_sequence'].values
    track_id = dftrack['track_id'].values[0]
    ax.plot(x,volume)
    ax.plot(x,smooth_volume,'g',zorder=-200,linewidth=2,)
    ax.plot(x,fit_volume,'r--')
    y = smooth_volume - fit_volume
    # peaks, props, results_full = find_drops_relative_to_fit(-y)
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
def filter_out_volume_drops(dfd, volume_cols,find_drops=True,return_intermediates=False):
    """
    Remove the volume dips from the volume data

        Parameters
    ----------
    dfd : pd.DataFrame
        dataframe with columns ['track_id','index_sequence'] + volume_cols
    volume_cols : list
        list of columns to needed to find and filter out volume dips
    find_drops : bool
        if True, find and remove the volume dips (input to peak finder is inverse of detrended+smoothed volume)

    Returns
    -------
    pd.DataFrame
        dataframe with columns ['track_id','index_sequence','CellId'] + ['smooth_volume_drops_removed_um','volume_drops_removed_um','volume_drops_mask','volume_drops_centers','volume_drops_magnitude']
    """

    # initialize dictionary to store dataframes for each step
    dfdict = {}

    # create a dataframe where the index is index_sequence and the columns are track_id and the values are on of the columns from time_cols
    dfp = dfd.pivot(index="index_sequence", columns="track_id", values=['volume','fit_volume'])

    # ensure that all timpepoints are present in dfp index_sequence
    if dfp.index.values.tolist() != list(range(dfp.index.values.min(), dfp.index.values.max() + 1)):
        # if not, fill in missing timepoints with np.nan
        dfp = dfp.reindex(index=range(dfp.index.values.min(), dfp.index.values.max() + 1))
    # interpolate the missing internal nan values while keeping the beginning and ending stretch of nans
    # (this is important so the savgol filter will not have gaps)
    dfp = dfp.interpolate(method='linear', axis=0, limit_area='inside') 
    dfdict.update({'volume_interpolated':dfp['volume']})
    dfdict.update({'fit_volume_interpolated':dfp['fit_volume']})

    # now apply savitzky golay filter to each column
    # dfp_sg = dfp.apply(lambda x: savgol_filter(x, window_length=12, polyorder=2, mode='constant', cval=np.nan), axis=0)
    yscale, _, _, _ = get_plot_labels_for_metric("volume")

    #step 1: apply savgol filter to each column
    dfp_vol_sg = dfp['volume'].apply(lambda x: savgol_filter(x*yscale, window_length=12, polyorder=2, mode='constant', cval=np.nan), axis=0)
    dfdict.update({'volume_sg':dfp_vol_sg})

    # step 2: detrend the data by subtracting the power law fit volume
    dfp_fit_vol = dfp['fit_volume']
    dfp_vol_sg_sub_fit = dfp_vol_sg - dfp_fit_vol
    dfdict.update({'volume_sg_sub_fit':dfp_vol_sg_sub_fit})


    # step 3: invert the detrended data to find the peaks (if find_drops is True)
    if find_drops:
        dfp_vol_sg_sub_fit = dfp_vol_sg_sub_fit * -1
        peak_str = "drops"
    else:
        peak_str = "jumps"
    dfdict.update({f'volume_sg_sub_fit_{peak_str}':dfp_vol_sg_sub_fit})


    # step 4: perform peak finding and get peak features
    
    #ALT approach. the only step that is difficult is to apply the mask and interpolate the values
    out = find_and_remove_peaks_combined(dfp_vol_sg_sub_fit.values, dfp['volume'].values, dfp_vol_sg_sub_fit.index.values, dfp_vol_sg_sub_fit.columns.values, peak_str=peak_str)
    cols = [x for x in out.columns if x not in ['index_sequence','track_id']]
    dfpeaks = out.pivot(index="index_sequence", columns="track_id", values=cols)
    
    # # now find peaks and remove them in the volume_sub_fit_volume
    # # using find_and_remove_from_pivot
    # dfp_mask = dfp_vol_sg_sub_fit.apply(lambda x: find_and_remove_from_pivot(x,select_return='mask'),axis=0)
    # dfp_centers = dfp_vol_sg_sub_fit.apply(lambda x: find_and_remove_from_pivot(x,select_return='center'),axis=0)
    # dfp_magnitude = dfp_vol_sg_sub_fit.apply(lambda x: find_and_remove_from_pivot(x,select_return='magnitude'),axis=0)
    
    # min_index_value = dfp_vol_sg_sub_fit.index.min()

    # dfp_left_base = dfp_vol_sg_sub_fit.apply(lambda x: find_and_remove_from_pivot(x,select_return='left_base',min_index=min_index_value),axis=0) 
    # dfp_right_base = dfp_vol_sg_sub_fit.apply(lambda x: find_and_remove_from_pivot(x,select_return='right_base',min_index=min_index_value),axis=0) 
    # dfp_center_vals = dfp_vol_sg_sub_fit.apply(lambda x: find_and_remove_from_pivot(x,select_return='center_val',min_index=min_index_value),axis=0)
    for col in cols:
        dfdict.update({f'{col}':dfpeaks[col]})
    # dfdict.update({f'volume_{peak_str}_mask':dfp_mask})
    # dfdict.update({f'volume_{peak_str}_centers':dfp_centers})
    # dfdict.update({f'volume_{peak_str}_props_magnitude':dfp_magnitude})
    # dfdict.update({f'volume_{peak_str}_left_bases':dfp_left_base})
    # dfdict.update({f'volume_{peak_str}_right_bases':dfp_right_base})
    # dfdict.update({f'volume_{peak_str}_centers_vals':dfp_center_vals})

    # compute the volume difference from center_vals and left_base

    # step 5: remove the peaks from the volume data
    mask_col = f'volume_{peak_str}_mask'
    dfp_mask = dfdict[mask_col].astype('bool')
    nanmask = np.ones(dfp_mask.shape, dtype='float')
    nanmask[dfp_mask.values] = np.nan
    dfp_vol1_unfilled = dfp_vol_sg * nanmask
    dfp_vol2_unfilled = dfp['volume'] * nanmask
    dfp_vol2_unfilled = dfp_vol2_unfilled * yscale # convert to um^3
    dfdict.update({f'smooth_volume_{peak_str}_removed_um_unfilled':dfp_vol1_unfilled})
    dfdict.update({f'volume_{peak_str}_removed_um_unfilled':dfp_vol2_unfilled})

    # fill in the gaps
    dfp_vol1 = dfp_vol1_unfilled.interpolate(method='linear', axis=0, limit_area='inside') 
    dfp_vol2 = dfp_vol2_unfilled.interpolate(method='linear', axis=0, limit_area='inside') 
    dfdict.update({f'smooth_volume_{peak_str}_removed_um':dfp_vol1})
    dfdict.update({f'volume_{peak_str}_removed_um':dfp_vol2})

    ## has_volume_drop column
    # dfp_has_drop = dfp_mask.copy()
    # dfp_has_drop[dfp_has_drop.values==0] = np.nan
    # dfp_has_drop = dfp_has_drop.interpolate(method='linear', axis=0)
    # dfdict.update({f'volume_{peak_str}_mask':dfp_has_drop})

    # step 6: collect the results (and specific peak features)
    key_list = list(dfdict.keys())
    if not return_intermediates:
        # remove the following keys ['volume','fit_volume','volume_sg_sub_fit_{peak_str}']
        key_list = [x for x in key_list if x not in ['volume','fit_volume','volume_sg_sub_fit_{peak_str}']]

    for i,key in enumerate(key_list):
        df = dfdict[key]
        if i == 0:
            dfm = df.stack().reset_index()
            dfm.rename(columns={0:key},inplace=True)
        else:
            dfm1 = df.stack().reset_index()
            dfm1.rename(columns={0:key},inplace=True)
            dfm = dfm.merge(dfm1,on=['index_sequence','track_id'],how='outer')
    
    # set the datatypes (because they are lost in the pivot operation)
    
    # set data types
    for col in dfm.columns:
        if col in [f'volume_{peak_str}_mask',f'volume_{peak_str}_centers',f'volume_{peak_str}_has_peak']:
            dfm[col] = dfm[col].astype('bool')
        elif col in ['track_id','index_sequence']:
            dfm[col] = dfm[col].astype('int')
        else:
            dfm[col] = dfm[col].astype('float')
        
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

    if return_intermediates:
        return dfmi.reset_index().set_index("CellId")
    else:
        return dfmi.reset_index().set_index("CellId")


# %%
def run_script(df=None,volume_cols=['volume','fit_volume'],return_intermediates=False):
    """
    run the workflow

    Parameters
    ----------
    df : pd.DataFrame
        dataframe on which to compute change_over_time
        with columns ['colony','track_id','index_sequence','label_img']+time_colsion
    volume_cols : list
        list of columns to needed to find and filter out volume dips
    return_intermediates : bool
        if True, return the intermediate dataframes for validation/visualization

    Returns
    -------
    pd.DataFrame
        dataframe with volume dips filtered out values for each track at each time point
        new columns are ['smooth_volume_drops_removed_um','volume_drops_removed_um']
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

    for find_drops in [True,False]: # find drops and jumps
        # returns dfo with index=CellId
        dfo = filter_out_volume_drops(dfd, volume_cols,find_drops=find_drops,return_intermediates=return_intermediates)
        new_columns = [x for x in dfo.columns.tolist() if x not in dforig.columns.tolist()]
        # add new columns to original dataframe
        dforig.loc[dfo.index.values, new_columns] = dfo.loc[dfo.index.values, new_columns]
    assert dforig.index.name == "CellId"
    return dforig
