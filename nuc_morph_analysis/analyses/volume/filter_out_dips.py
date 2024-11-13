import numpy as np
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import numpy as np
import pandas as pd

# def custom_peak_finder(dftrack, window_size=12):

#     # take derivatives
#     y = dftrack['volume_sg'] #smoothed volume
#     x = y.index.values
#     dy1 = y.diff(1) # take first derivative
#     dy_sg = pd.Series(index = dy1.index, data=savgol_filter(dy1.values, window_length=24, polyorder=2, mode='interp')) # smooth first derivative
#     dy2 = dy_sg.diff(1) # take second derivative


#     # now find peaks
#     # first find negative peaks in dy1
#     peaks, properties = find_peaks(-dy1.values, prominence=(20,None))
#     peak_feats={}
#     for peak in peaks:
#         peak_trough = np.argmin(np.abs(dy1[peak : peak + window_size])) + peak
#         peak_left = np.argmin(np.abs(dy1[peak - window_size : peak])) + peak - window_size
#         peak_right = np.argmin(np.abs(dy2[peak_trough : peak_trough + window_size])) + peak_trough

#         peak_feats[peak] = {'trough':peak_trough,
#                             'left':peak_left,
#                             'right':peak_right}
#     # now plot
#     fig,axlist = plt.subplots(3,1,figsize=(2*2,3*2),sharex=True)
#     axlist[0].plot(x,y.values)
#     axlist[1].plot(x,dy1.values)
#     axlist[1].plot(x,dy_sg)
#     axlist[2].plot(x,dy2.values)
    
#     for peak in peaks:
#         axlist[0].plot(x[peak_feats[peak]['trough']],y.values[peak_feats[peak]['trough']],'bo')
#         axlist[0].plot(x[peak_feats[peak]['left']],y.values[peak_feats[peak]['left']],'go')
#         axlist[0].plot(x[peak_feats[peak]['right']],y.values[peak_feats[peak]['right']],'ro')
#     return 


def find_dips_relative_to_fit(y,
                               prominence=(20,None),
                                 width=(2,24),
                                 height=(None,None),
                                 threshold=(None,None),
                                 rel_height=1,
                                 wlen=25):
    """
    find the dips in a trajectory
    the default parameters assume nuclear volume data smoothed AND relative to the power law fit volume (or smoothed only)

    Parameters
    ----------
    y : np.array
        volume data
    parameters for the find peaks function in scipy

    Returns
    -------
    tuple
        peaks, props

    """
    peaks, props = find_peaks(y, prominence=prominence, width=width,height=height,threshold=threshold,rel_height=rel_height,wlen=wlen)
    return peaks, props

def find_and_remove_from_pivot(y,y2,min_index=0,peak_str='dips'):
    """
    find peaks in y (volume data, smoothed and detrended OR smoothed only), 
    collect features, and determine mask to use in order to remove the peaks from the volume data

    Parameters
    ----------
    y : np.array
        volume data (smoothed volume, detrended or not)
    y2 : np.array
        raw volume data used to compute the peak magnitude
    min_index : int
        minimum index value for the volume data
        this is needed to ensure the index values are correct in the final dataframe
    peak_str : str
        string to use for the peak type ('dips' or 'jumps')

    Returns
    -------
    dict
        dictionary with keys ['volume_{}_peak_region_mask',
        'volume_{}_peak_center_mask',
        'volume_{}_has_peak',
        'volume_{}_prom',
        'volume_{}_left_base',
        'volume_{}_right_base',
        'volume_{}_magnitude',
        'volume_{}_max_magnitude',
        'volume_{}_max_prominence']
        
        where {} is the peak_str
    """

    peaks, props = find_dips_relative_to_fit(y)
    has_peak_bool_array = np.zeros(y.shape,dtype='bool') if len(peaks)==0 else np.ones(y.shape,dtype='bool')
    
    # initialize the peak_magnitude_array as nans
    peak_center_mask = np.zeros(y.shape,dtype='bool')
    peak_region_mask = np.zeros(y.shape,dtype='bool')
    left_base_array = np.zeros(y.shape,dtype='float32') * np.nan
    right_base_array = np.zeros(y.shape,dtype='float32') * np.nan
    prominence_array = np.zeros(y.shape,dtype='float32') * np.nan
    left_magnitude_array = np.zeros(y.shape,dtype='float32') * np.nan
    left_magnitude_mask = np.zeros(y.shape,dtype='float32') * np.nan
    right_magnitude_array = np.zeros(y.shape,dtype='float32') * np.nan
    right_magnitude_mask = np.zeros(y.shape,dtype='float32') * np.nan

    right_magnitude_array = np.zeros(y.shape,dtype='float32') * np.nan
    right_magnitude_mask = np.zeros(y.shape,dtype='float32') * np.nan

    peak_id_array = np.zeros(y.shape,dtype='float32') * np.nan
    peak_id_mask = np.zeros(y.shape,dtype='float32') * np.nan
    if len(peaks)>0:
        for pi, peak in enumerate(peaks):

            left_base = props['left_bases'][pi].copy()
            right_base = props['right_bases'][pi].copy()
            left_volume_change = y2[peak] - y2[left_base] # changed so that dips will be negative
            right_volume_change = y2[peak] - y2[right_base]

            peak_center_mask[peak] = True

            peak_indices = np.arange(left_base,right_base+1,1,dtype='uint16')
            peak_region_mask[peak_indices] = True

            left_magnitude_array[peak] = left_volume_change  
            left_magnitude_mask[peak_indices] = left_volume_change

            right_magnitude_array[peak] = right_volume_change
            right_magnitude_mask[peak_indices] = right_volume_change

            peak_id_array[peak] = pi
            peak_id_mask[peak_indices] = pi

            prominence_array[peak] = props['prominences'][pi]
            left_base_array[peak] = props['left_bases'][pi].copy() + min_index
            right_base_array[peak] = props['right_bases'][pi] + min_index
    
    # take minimum of peak_magnitude if peak_str is dips ()
    func = np.nanmin if peak_str=='dips' else np.nanmax
    max_val = func(left_magnitude_array) if len(peaks)>0 else np.nan

    max_peak_val_array = np.ones(y.shape,dtype='float32') * max_val
    max_prominence_array = np.ones(y.shape,dtype='float32') * np.nanmax(prominence_array)
    
    return {
        f'volume_{peak_str}_peak_region_mask':peak_region_mask, # boolean array, true at all points within peak region(s)
        f'volume_{peak_str}_peak_center_mask':peak_center_mask, # boolean array, true at all peak centers
        f'volume_{peak_str}_has_peak':has_peak_bool_array, #boolean array, true at all points if there is a peak
        f'volume_{peak_str}_prom':prominence_array, # prominence of each peak
        f'volume_{peak_str}_left_bases':left_base_array, # left base of each peak region
        f'volume_{peak_str}_right_bases':right_base_array, # right base of each peak region
        f'volume_{peak_str}_left_magnitude':left_magnitude_array, # magnitude value at each peak center (left_base - peak)
        f'volume_{peak_str}_left_magnitude_mask':left_magnitude_mask, # magnitude values at all points within peak region(s) (left_base - peak)
        f'volume_{peak_str}_right_magnitude':right_magnitude_array, # magnitude value at each peak center (right_base - peak)
        f'volume_{peak_str}_right_magnitude_mask':right_magnitude_mask, # magnitude values at all points within peak region(s) (right_base - peak)
        f'volume_{peak_str}_max_magnitude':max_peak_val_array, # maximum magnitude value (at all points in array)
        f'volume_{peak_str}_max_prominence':max_prominence_array, # maximum prominence value (at all points in array)
    }

def find_and_remove_peaks_combined(vol_det_array, vol_array, index_sequence_vec, track_id_vec, peak_str='dips'):
    """
    run the peak finding algorithm for each track_id (column) in detrended volume data (vol_det_array) and extract features from peaks (using both vol_det_array and vol_array)
    the collect the outputs and return a dataframe

    Parameters
    ----------
    vol_det_array : np.array
        detrended volume data (smoothed volume - fit volume), but can be any data that has peaks
    vol_array : np.array
        volume data (smoothed volume), used to compute the peak magnitude
    index_sequence_vec : np.array
        index sequence values for the volume data
    track_id_vec : np.array
        track_id values for the volume data (columns)
    peak_str : str
        string to use for the peak type ('dips' or 'jumps')

    Returns
    -------
    pd.DataFrame
        dataframe with columns ['track_id','index_sequence'] + 
        ['volume_{peak_str}_peak_region_mask',
        'volume_{peak_str}_peak_center_mask',
        'volume_{peak_str}_has_peak',
        'volume_{peak_str}_prom',
        'volume_{peak_str}_left_base',
        'volume_{peak_str}_right_base',
        'volume_{peak_str}_left_magnitude',
        'volume_{peak_str}_max_magnitude',
        'volume_{peak_str}_max_prominence',
        'volume_{peak_str}_left_magnitude_mask']

    """

    # apply find_dips_relative_to_fit(y) to each column of vol_det_array
    min_index = index_sequence_vec.min()

    # iterate through each track_id
    out = [find_and_remove_from_pivot(vol_det_array[:,col], vol_array[:,col], min_index=min_index, peak_str=peak_str) for col in range(vol_det_array.shape[1])]
    
    dfout_list = [pd.DataFrame(x.values(),columns = index_sequence_vec, index=x.keys()).T for x in out]
    dfout_list = [x.reset_index().rename(columns={'index':'index_sequence'}).set_index('index_sequence') for x in dfout_list]
    # keys = ['volume_dips_peak_region_mask','volume_dips_centers','volume_dips_has_peak','volume_dips_prom','volume_dips_left_base','volume_dips_right_base','volume_dips_y2_magnitude']
    dfout = pd.concat(dfout_list,axis=0,keys=track_id_vec, names=['track_id']).reset_index()
    
    return dfout

def filter_out_volume_dips(dfd, volume_cols, find_dips=True, use_detrended=True, return_intermediates=False, prefix=''):
    """
    Remove the volume dips from the volume data

        Parameters
    ----------
    dfd : pd.DataFrame
        dataframe with columns ['track_id','index_sequence'] + volume_cols
    volume_cols : list
        list of columns to needed to find and filter out volume dips
        default is ['volume','fit_volume'], fit_volume is used to detrend the volume data
    find_dips : bool
        if True, find and remove the volume dips (input to peak finder is inverse of detrended+smoothed volume)
    use_detrended : bool
        if True, use the detrended volume data (subtracted by the power law fit volume)
    return_intermediates : bool
        if True, return the intermediate dataframes for validation/visualization
    prefix : str
        string to add to the beginning of the new columns

    Returns
    -------
    pd.DataFrame
        dataframe with columns ['track_id','index_sequence','CellId'] +
        [
            f"input_{peak_str}", # input to peak finder
            f"volume_sg", # smoothed volume data
            f"volume_interpolated", # interpolated volume data
            f"fit_volume_interpolated", # interpolated fit_volume data
            f"volume_sg_sub_fit", # detrended volume data

            f"volume_{peak_str}_peak_region_mask",
            f"volume_{peak_str}_peak_center_mask",
            f"volume_{peak_str}_has_peak",
            f"volume_{peak_str}_prom",
            f"volume_{peak_str}_left_base",
            f"volume_{peak_str}_right_base",
            f"volume_{peak_str}_left_magnitude",
            f"volume_{peak_str}_max_magnitude",
            f"volume_{peak_str}_max_prominence",
            f"volume_{peak_str}_left_magnitude_mask",

            f"smooth_volume_{peak_str}_removed_um_unfilled", # dips or jumps removed and nans filled in
            f"volume_{peak_str}_removed_um_unfilled", # dips or jumps removed and nans filled in
            f"smooth_volume_{peak_str}_removed_um", # dips or jumps removed and linearly interpolated to fill in
            f"volume_{peak_str}_removed_um", # dips or jumps removed and linearly interpolated to fill in
        ]

        where {} is the peak_str ('dips' or 'jumps')

                
    """

    # initialize dictionary to store dataframes for each step
    dfdict = {}
    volume_col = volume_cols[0]
    fit_volume_col = volume_cols[1]

    # create a dataframe where the index is index_sequence and the columns are track_id and the values are on of the columns from time_cols
    dfp = dfd.pivot(index="index_sequence", columns="track_id", values=volume_cols)

    # ensure that all timpepoints are present in dfp index_sequence
    if dfp.index.values.tolist() != list(range(dfp.index.values.min(), dfp.index.values.max() + 1)):
        # if not, fill in missing timepoints with np.nan
        dfp = dfp.reindex(index=range(dfp.index.values.min(), dfp.index.values.max() + 1))
    # interpolate the missing internal nan values while keeping the beginning and ending stretch of nans
    # (this is important so the savgol filter will not have gaps)
    dfp = dfp.interpolate(method='linear', axis=0, limit_area='inside') 
    
    # add the interpolated volume data to the dictionary
    dfdict.update({'volume_interpolated':dfp[volume_col]}) # add interpolated volume data to dictionary
    dfdict.update({'fit_volume_interpolated':dfp[fit_volume_col]}) # add interpolated fit_volume data to dictionary

    # now apply savitzky golay filter to each column
    yscale, _, _, _ = get_plot_labels_for_metric(volume_col)

    #step 1: apply savgol filter to each column
    dfp_vol_sg = dfp[volume_col].apply(lambda x: savgol_filter(x*yscale, window_length=12, polyorder=2, mode='constant',cval=np.nan), axis=0)
    dfdict.update({'volume_sg':dfp_vol_sg}) # add smoothed volume data to dictionary

    # step 2: detrend the data by subtracting the power law fit volume
    dfp_fit_vol = dfp[fit_volume_col]
    dfp_vol_sg_sub_fit = dfp_vol_sg - dfp_fit_vol
    dfdict.update({'volume_sg_sub_fit':dfp_vol_sg_sub_fit}) # add detrended volume data to dictionary
 
    # step 3: invert the input data (smoothed volume or detrended data) to find the peaks (if find_dips is True)
    drop_scale = -1 
    if find_dips:
        drop_scale = -1
        peak_str = "dips"
    else:
        peak_str = "jumps"
        drop_scale = 1

    # use smoothed+detrended data or just smoothed data
    input = dfp_vol_sg_sub_fit * drop_scale if use_detrended else dfp_vol_sg * drop_scale
    dfdict.update({f'input_{peak_str}':input})

    # step 4: perform peak finding and get peak features
    out = find_and_remove_peaks_combined(input.values, dfp[volume_col].values * yscale, input.index.values, input.columns.values, peak_str=peak_str)
    
    # combine the output into a pivot table
    cols = [x for x in out.columns if x not in ['index_sequence','track_id']]
    dfpeaks = out.pivot(index="index_sequence", columns="track_id", values=cols)
    
    # unpack the dataframes containing the peak features into the dictionary
    for col in cols:
        dfdict.update({f'{col}':dfpeaks[col]})
 
    ############################################################
    # step 5: remove the peaks from the volume data
    mask_col = f'volume_{peak_str}_peak_region_mask'
    dfp_mask = dfdict[mask_col].astype('bool')
    nanmask = np.ones(dfp_mask.shape, dtype='float')

    # determine which peak regions to remove based on threshold
    drop_thresh = -20 if find_dips else 20
    if drop_thresh is not None:
        col = f'volume_{peak_str}_left_magnitude_mask' if find_dips else f'volume_{peak_str}_right_magnitude_mask'
        dfp_mag = dfdict[col].values
        thresh_bool = dfp_mag<drop_thresh if find_dips else dfp_mag>drop_thresh
        # if the value crosses the threshold then it is True and we want it to be set to NaN, if False then we want to keep the value
        # the nan values will later be interpolated
    else: 
        thresh_bool = np.ones(dfp_mask.shape, dtype='bool')
    mask_values = dfp_mask.values & thresh_bool
    nanmask[mask_values] = np.nan

    # explicitly add the peak regions to remove
    dfdict.update({f"volume_{peak_str}_peak_regions_to_remove": pd.DataFrame(mask_values, index=dfp_mask.index, columns=dfp_mask.columns)})

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
    

    # set the data types (because they are lost in the pivot operation)
    for col in dfm.columns:
        if col in [f'volume_{peak_str}_peak_region_mask',f'volume_{peak_str}_peak_center_mask',f'volume_{peak_str}_has_peak']:
            dfm[col] = dfm[col].astype('bool')
        elif col in ['track_id','index_sequence']:
            dfm[col] = dfm[col].astype('int')
        else:
            dfm[col] = dfm[col].astype('float')

    if prefix != '':
        # rename columns with prefix
        rename_dict = {f"{x}":f"{prefix}{x}" for x in dfm.columns.tolist() if x not in ['index_sequence','track_id']}
        dfm.rename(columns=rename_dict,inplace=True)
        
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

    dfmi.loc[:,f"{peak_str}_threshold"] = drop_thresh
    if return_intermediates:
        return dfmi.reset_index().set_index("CellId")
    else:
        return dfmi.reset_index().set_index("CellId")


# %%
def run_script(df=None,volume_cols=['volume','fit_volume'],use_detrended=True,return_intermediates=False,prefix=''):
    """
    run the workflow

    Parameters
    ----------
    df : pd.DataFrame
        dataframe on which to compute change_over_time
        with columns ['colony','track_id','index_sequence','label_img']+time_colsion
    volume_cols : list
        list of columns to needed to find and filter out volume dips
    use_detrended : bool
        if True, use the detrended volume data (subtracted by the power law fit volume)
    return_intermediates : bool
        if True, return the intermediate dataframes for validation/visualization
    prefix : str
        string to add to the beginning of the new columns

    Returns
    -------
    pd.DataFrame
        dataframe with volume dips filtered out values for each track at each time point
        new columns are 
        
        [
            f"input_{peak_str}", # input to peak finder
            f"volume_sg", # smoothed volume data
            f"volume_interpolated", # interpolated volume data
            f"fit_volume_interpolated", # interpolated fit_volume data
            f"volume_sg_sub_fit", # detrended volume data

            f"volume_{peak_str}_peak_region_mask",
            f"volume_{peak_str}_peak_center_mask",
            f"volume_{peak_str}_has_peak",
            f"volume_{peak_str}_prom",
            f"volume_{peak_str}_left_base",
            f"volume_{peak_str}_right_base",
            f"volume_{peak_str}_left_magnitude",
            f"volume_{peak_str}_max_magnitude",
            f"volume_{peak_str}_max_prominence",
            f"volume_{peak_str}_left_magnitude_mask",
            

            f"smooth_volume_{peak_str}_removed_um_unfilled", # dips or jumps removed and nans filled in
            f"volume_{peak_str}_removed_um_unfilled", # dips or jumps removed and nans filled in
            f"smooth_volume_{peak_str}_removed_um", # dips or jumps removed and linearly interpolated to fill in
            f"volume_{peak_str}_removed_um", # dips or jumps removed and linearly interpolated to fill in
        ]
        
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

    for find_dips in [True,False]: # find dips and jumps
        # returns dfo with index=CellId
        dfo = filter_out_volume_dips(dfd, volume_cols,find_dips,use_detrended,return_intermediates,prefix)
        new_columns = [x for x in dfo.columns.tolist() if x not in dforig.columns.tolist()]
        
        # add new columns to original dataframe
        dforig.loc[dfo.index.values, new_columns] = dfo.loc[dfo.index.values, new_columns]
    assert dforig.index.name == "CellId"
    return dforig
