import numpy as np
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import numpy as np
import pandas as pd

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

def find_and_remove_from_pivot(y,y2,min_index=0,peak_str='dips',volume_change_threshold=0):
    """
    find peaks in y (volume data, smoothed and detrended OR smoothed only), 
    collect features (from y and/or y2) and return a dictionary with the features

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
        dictionary with keys [
            f'volume_{peak_str}_peak_mask_at_region', # boolean array, true at all points within peak region(s)
            f'volume_{peak_str}_peak_mask_at_center', # boolean array, true at all peak centers
            f'volume_{peak_str}_has_peak', # boolean array, true at all points if there is a peak
            f'volume_{peak_str}_volume_change_at_center', # magnitude value at each peak center (left_base - peak)
            f'volume_{peak_str}_volume_change_at_region', # magnitude values at all points within peak region(s) (left_base - peak)
            f'volume_{peak_str}_width_at_center', # width of the peak at the peak center index
            f'volume_{peak_str}_width_at_region', # width of the peak at all indices in the peak region
            f'volume_{peak_str}_max_volume_change', # maximum magnitude value (at all points in array)
            f'volume_{peak_str}_peak_id_at_center', # peak id at the peak center index
            f'volume_{peak_str}_peak_id_at_region', # peak id at all indices in the peak region
            f'volume_{peak_str}_total_number', # total number of peaks
            ]
        
        where {} is the peak_str
    """

    peaks, props = find_dips_relative_to_fit(y) # run scipy peak finder
    
    # if peaks present, set entire array to True
    has_peak_bool_array = np.zeros(y.shape,dtype='bool') if len(peaks)==0 else np.ones(y.shape,dtype='bool')
    
    # initialize new feature arrays with False or NaN
    peak_mask_at_center = np.zeros(y.shape,dtype='bool') # boolean array, true at all peak centers
    peak_mask_at_region = np.zeros(y.shape,dtype='bool') # boolean array, true at all points within peak region(s)
    volume_change_at_center = np.zeros(y.shape,dtype='float32') * np.nan # the change in volume (V(left base) - V(peak center)) value placed in an array at the peak center index
    volume_change_at_region = np.zeros(y.shape,dtype='float32') * np.nan # the change in volume (V(left base) - V(peak center)) value placed in an array at all indices in the peak region
    peak_widths_at_center = np.zeros(y.shape,dtype='float32') * np.nan # the width of the peak at the peak center index
    peak_widths_at_region = np.zeros(y.shape,dtype='float32') * np.nan # the width of the peak at all indices in the peak region
    peak_id_at_center = np.zeros(y.shape,dtype='float32') * np.nan # the peak id at the peak center index
    peak_id_at_region = np.zeros(y.shape,dtype='float32') * np.nan # the peak id at all indices in the peak region
    if len(peaks)>0:
        for pi, peak in enumerate(peaks):

            left_base = props['left_bases'][pi].copy()
            right_base = props['right_bases'][pi].copy()
            left_volume_change = y2[peak] - y2[left_base] 
            peak_indices = np.arange(left_base,right_base+1,1,dtype='uint16')
            peak_width = right_base - left_base

            peak_above_threshold = left_volume_change < volume_change_threshold if peak_str=='dips' else left_volume_change > volume_change_threshold
            if not peak_above_threshold:
                continue
            peak_mask_at_center[peak] = True
            peak_mask_at_region[peak_indices] = True

            volume_change_at_center[peak] = left_volume_change  
            volume_change_at_region[peak_indices] = left_volume_change

            peak_widths_at_center[peak] = peak_width
            peak_widths_at_region[peak_indices] = peak_width

            peak_id_at_center[peak] = pi
            peak_id_at_region[peak_indices] = pi

    # take minimum of peak_magnitude if peak_str is dips ()
    func = np.nanmin if peak_str=='dips' else np.nanmax
    max_val = func(volume_change_at_center) if len(peaks)>0 else np.nan
    max_volume_change = np.ones(y.shape,dtype='float32') * max_val

    number_of_peaks = np.ones(y.shape,dtype='float32') * len(peaks)
    
    return {
        f'volume_{peak_str}_peak_mask_at_region':peak_mask_at_region, # boolean array, true at all points within peak region(s)
        f'volume_{peak_str}_peak_mask_at_center':peak_mask_at_center, # boolean array, true at all peak centers
        f'volume_{peak_str}_has_peak':has_peak_bool_array, #boolean array, true at all points if there is a peak
        f'volume_{peak_str}_volume_change_at_center':volume_change_at_center, # magnitude value at each peak center (left_base - peak)
        f'volume_{peak_str}_volume_change_at_region':volume_change_at_region, # magnitude values at all points within peak region(s) (left_base - peak)
        f'volume_{peak_str}_width_at_center':peak_widths_at_center, # width of the peak at the peak center index
        f'volume_{peak_str}_width_at_region':peak_widths_at_region, # width of the peak at all indices in the peak region
        f'volume_{peak_str}_max_volume_change':max_volume_change, # maximum magnitude value (at all points in array)
        f'volume_{peak_str}_peak_id_at_center':peak_id_at_center, # peak id at the peak center index
        f'volume_{peak_str}_peak_id_at_region':peak_id_at_region, # peak id at all indices in the peak region
        f'volume_{peak_str}_total_number':number_of_peaks, # total number of peaks
    }

def find_peaks_and_collect_features(vol_det_array, vol_array, index_sequence_vec, track_id_vec, peak_str='dips'):
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
        columns defined in find_and_remove_from_pivot

    """

    # iterate through each track_id (will be slow if not using full tracks) and find peaks
    out = [find_and_remove_from_pivot(vol_det_array[:,track], vol_array[:,track], min_index=index_sequence_vec.min(), peak_str=peak_str) for track in range(vol_det_array.shape[1])]
    
    # convert the output to a dataframe
    dfout_list = [pd.DataFrame(x.values(),columns = index_sequence_vec, index=x.keys()).T for x in out]
    dfout_list = [x.reset_index().rename(columns={'index':'index_sequence'}).set_index('index_sequence') for x in dfout_list]
    # keys = ['volume_dips_peak_mask_at_region','volume_dips_centers','volume_dips_has_peak','volume_dips_prom','volume_dips_left_base','volume_dips_right_base','volume_dips_y2_magnitude']
    dfout = pd.concat(dfout_list,axis=0,keys=track_id_vec, names=['track_id']).reset_index()
    
    return dfout

def filter_out_volume_dips(dfd, volume_cols, find_dips=True, use_detrended=True, return_intermediates=False):
    """
    Remove the volume dips from the volume data

    STEPS:
    - inputs are volume trajectory and power law fit (used for detrending)
    - both inputs are interpolated so there is a value at each point in time (necessary for smoothing)
    - volume trajectory is smoothed using savgol filter 
    - smoothed volume trajectory is detrended by subtracting the power law fit
    - search for volume dips by putting inverse of detrended smoothed volume trajectory into scipy.signal.find_peaks
    - peak regions are removed from the volume trajectories by filling with nans. 

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

    Returns
    -------
    pd.DataFrame
        dataframe with columns ['track_id','index_sequence','CellId'] +
        [
            # intermediate columns (only kept if return_intermediates is True)
                f"input_{peak_str}", # input to peak finder
                f"volume_smooth", # smoothed volume data
                f"volume_interpolated", # interpolated volume data
                f"fit_volume_interpolated", # interpolated fit_volume data
                f"volume_smooth_detrended", # detrended volume data

            # values from find_peaks_and_collect_features
                f'volume_{peak_str}_peak_mask_at_region', # boolean array, true at all points within peak region(s)
                f'volume_{peak_str}_peak_mask_at_center', # boolean array, true at all peak centers
                f'volume_{peak_str}_has_peak', # boolean array, true at all points if there is a peak
                f'volume_{peak_str}_volume_change_at_center', # magnitude value at each peak center (left_base - peak)
                f'volume_{peak_str}_volume_change_at_region', # magnitude values at all points within peak region(s) (left_base - peak)
                f'volume_{peak_str}_width_at_center', # width of the peak at the peak center index
                f'volume_{peak_str}_width_at_region', # width of the peak at all indices in the peak region
                f'volume_{peak_str}_max_volume_change', # maximum magnitude value (at all points in array)
                f'volume_{peak_str}_peak_id_at_center', # peak id at the peak center index
                f'volume_{peak_str}_peak_id_at_region', # peak id at all indices in the peak region
                f'volume_{peak_str}_total_number', # total number of peaks


            # values after peak regions are removed
                f"volume_{peak_str}_removed_um_unfilled", # dips or jumps removed (refilled with nans)
        ]

        where {} is the peak_str ('dips' or 'jumps')

                
    """

    # initialize dictionary to store dataframes for each step
    dfdict = {}
    volume_col = volume_cols[0]
    fit_volume_col = volume_cols[1]

    # create a pivot dataframe where the index is index_sequence and the columns are track_id and the values are on of the columns from volume_cols
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
    dfdict.update({'fit_volume_interpolated':dfp[fit_volume_col]}) # add interpolated fit_volume data to dictionary (used to detrend the volume data)

    # now apply savitzky golay filter to each column
    # rescale the y values to um^3 (from pixels^3)
    yscale, _, _, _ = get_plot_labels_for_metric(volume_col)

    #step 1: apply savgol filter to each column
    dfp_vol_smooth = dfp[volume_col].apply(lambda x: savgol_filter(x*yscale, window_length=12, polyorder=2, mode='constant',cval=np.nan), axis=0)
    dfdict.update({'volume_smooth':dfp_vol_smooth}) # add smoothed volume data to dictionary

    # step 2: detrend the data by subtracting the power law fit volume
    dfp_vol_fit = dfp[fit_volume_col]
    dfp_vol_smooth_detrended = dfp_vol_smooth - dfp_vol_fit
    dfdict.update({'volume_smooth_detrended':dfp_vol_smooth_detrended}) # add detrended volume data to dictionary
 
    # step 3: invert the input data (smoothed volume or detrended data) to find the peaks (if find_dips is True)
    drop_scale = -1 
    if find_dips:
        drop_scale = -1
        peak_str = "dips"
    else:
        peak_str = "jumps"
        drop_scale = 1

    # use smoothed+detrended data or just smoothed data
    input = dfp_vol_smooth_detrended * drop_scale if use_detrended else dfp_vol_smooth * drop_scale
    dfdict.update({f'input_{peak_str}':input})

    # step 4: perform peak finding and get peak features
    out = find_peaks_and_collect_features(input.values, dfp[volume_col].values * yscale, input.index.values, input.columns.values, peak_str=peak_str)
    
    # combine the output into a pivot table
    cols = [x for x in out.columns if x not in ['index_sequence','track_id']]
    dfpeaks = out.pivot(index="index_sequence", columns="track_id", values=cols)
    
    # unpack the dataframes containing the peak features into the dictionary
    for col in cols:
        dfdict.update({f'{col}':dfpeaks[col]})
 
    ############################################################
    # step 5: remove the peaks from the volume data
    dfp_mask = dfdict[f'volume_{peak_str}_peak_mask_at_region'].astype('bool')
    nanmask = np.ones(dfp_mask.shape, dtype='float')
    mask_values = dfp_mask.values
    nanmask[mask_values] = np.nan

    dfp_vol2_unfilled = dfp['volume'] * nanmask
    dfp_vol2_unfilled = dfp_vol2_unfilled * yscale # convert to um^3
    dfdict.update({f'volume_{peak_str}_removed_um_unfilled':dfp_vol2_unfilled})

    # step 6: collect the results (and specific peak features)
    key_list = list(dfdict.keys())
    if not return_intermediates:
        intermediate_list = [
                             'volume_interpolated',
                             'fit_volume_interpolated',
                             'volume_smooth',
                             'volume_smooth_detrended',
                             f'input_{peak_str}',
                             ]
        key_list = [x for x in key_list if x not in intermediate_list]

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
        if col in [f'volume_{peak_str}_peak_mask_at_region',f'volume_{peak_str}_peak_mask_at_center',f'volume_{peak_str}_has_peak']:
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
def run_script(df=None,volume_cols=['volume','fit_volume'],use_detrended=True,return_intermediates=False):
    """
    run the workflow to find, characterize, and remove volume dips from the volume data

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
            # intermediate columns (only kept if return_intermediates is True)
                f"input_{peak_str}", # input to peak finder
                f"volume_smooth", # smoothed volume data
                f"volume_interpolated", # interpolated volume data
                f"fit_volume_interpolated", # interpolated fit_volume data
                f"volume_smooth_detrended", # detrended volume data

            # values from find_peaks_and_collect_features
                f'volume_{peak_str}_peak_mask_at_region', # boolean array, true at all points within peak region(s)
                f'volume_{peak_str}_peak_mask_at_center', # boolean array, true at all peak centers
                f'volume_{peak_str}_has_peak', # boolean array, true at all points if there is a peak
                f'volume_{peak_str}_volume_change_at_center', # magnitude value at each peak center (left_base - peak)
                f'volume_{peak_str}_volume_change_at_region', # magnitude values at all points within peak region(s) (left_base - peak)
                f'volume_{peak_str}_width_at_center', # width of the peak at the peak center index
                f'volume_{peak_str}_width_at_region', # width of the peak at all indices in the peak region
                f'volume_{peak_str}_max_volume_change', # maximum magnitude value (at all points in array)
                f'volume_{peak_str}_peak_id_at_center', # peak id at the peak center index
                f'volume_{peak_str}_peak_id_at_region', # peak id at all indices in the peak region
                f'volume_{peak_str}_total_number', # total number of peaks

            # values after peak regions are removed
                f"volume_{peak_str}_removed_um_unfilled", # dips or jumps removed (refilled with nans)
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

    # returns dfo with index=CellId
    dfo = filter_out_volume_dips(dfd, volume_cols,True,use_detrended,return_intermediates)
    new_columns = [x for x in dfo.columns.tolist() if x not in dforig.columns.tolist()]
    
    # add new columns to original dataframe
    dforig.loc[dfo.index.values, new_columns] = dfo.loc[dfo.index.values, new_columns]
    assert dforig.index.name == "CellId"
    return dforig
