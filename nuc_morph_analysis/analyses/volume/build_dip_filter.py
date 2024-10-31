# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.analyses.volume.plot_help import adjust_axis_positions, plot_track_with_fit_line,update_plotting_params, plot_track_with_volume_dip
from nuc_morph_analysis.lib.preprocessing import filter_data, compute_change_over_time
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot 
import os
from nuc_morph_analysis.analyses.dataset_images_for_figures import figure_helper
from matplotlib.collections import PatchCollection  # type: ignore
import pandas as pd
from nuc_morph_analysis.lib.visualization.example_tracks import EXAMPLE_TRACKS
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from scipy.signal import savgol_filter
import numpy as np
#%%
#%% update plotting parameters
fs, fw, fh = update_plotting_params()

#%%
# load the data
df0 = load_dataset_with_features('all_baseline',load_local=True, remove_growth_outliers=False)
df0 = filter_data.all_timepoints_minimal_filtering(df0)
# df0 = filter_data.all_timepoints_full_tracks(df0)
df0 = compute_change_over_time.run_script(df0, dxdt_feature_list=['volume'], bin_interval_list=[5])
df0 = compute_change_over_time.run_script(df0, dxdt_feature_list=['dxdt_5_volume_start'], bin_interval_list=[5])
#%%
yscale, ylabel, yunit, _ = get_plot_labels_for_metric("volume")
df0['volume_drop'] = df0['dxdt_5_volume_start'] * yscale * 5
df0['dx_volume_drop'] = df0['dxdt_5_dxdt_5_volume_start_start'] * yscale * (5*5)
df0['volume_sub_fit_volume'] = df0['volume']*yscale - df0['fit_volume']
#%%
df_full = filter_data.all_timepoints_full_tracks(df0)
#%%
save_dir = Path(__file__).parent / 'figures' / 'volume_dip_figures'

#%%
# find peaks on volume_sub_fit_volume
from scipy.signal import find_peaks, peak_widths
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

def find_and_remove_from_pivot(y):
    peaks, props, results_full = find_drops_relative_to_fit(y)
    _,y_mask = remove_peaks(y,props)
    return y_mask

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
    ax.plot(x,volume)
    ax.plot(x,smooth_volume,'g',zorder=-200,linewidth=2,)
    ax.plot(x,fit_volume,'r--')
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

track_list = [86246,85345,77291,86570,83675,84663,86212]

for track_id in track_list:
    dftrack = df0[df0["track_id"] == track_id]
    x = dftrack['index_sequence'].values

    y, volume, fit_volume, smooth_volume = get_fit_volume_minus_smoothed_y_from_df(dftrack, return_all=True)

    peaks, props, results_full = find_drops_relative_to_fit(-y)

    fig,ax = plt.subplots(1,1,figsize=(8,4), sharex=True)
    plot_features_and_peaks(dftrack,volume,smooth_volume,fit_volume,peaks,props,ax)
    plt.show()


#%%
dfd = df0[df0["colony"] == 'small']
ided_tracks = []
for track_id in dfd.track_id.unique():
    # y = dfd[dfd.track_id == track_id]['volume_sub_fit_volume'].values
    dftrack = df0[df0["track_id"] == track_id]
    x = dftrack['index_sequence'].values
    
    y, volume, fit_volume, smooth_volume = get_fit_volume_minus_smoothed_y_from_df(dftrack, return_all=True)
    peaks, props, results_full = find_drops_relative_to_fit(-y)
    out = find_and_remove_from_pivot(-y)
    if np.isnan(out).any():
        ided_tracks.append(track_id)

        fig,ax = plt.subplots(1,1,figsize=(8,4), sharex=True)
        plot_features_and_peaks(dftrack,volume,smooth_volume,fit_volume,peaks,props,ax)
        plt.show()
print(len(ided_tracks))
#%%


#%%
return_smoothed_volume=True
dfd = df_full.copy()
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
yscale, ylabel, yunit, _ = get_plot_labels_for_metric("volume")
dfp_vol_sg = dfp['volume'].apply(lambda x: savgol_filter(x*yscale, window_length=12, polyorder=2, mode='constant', cval=np.nan), axis=0)
dfp_fit_vol = dfp['fit_volume']
dfp_vol_sg_sub_fit = dfp_vol_sg - dfp_fit_vol

# now find peaks and remove them in the volume_sub_fit_volume
# using find_and_remove_from_pivot
dfp_mask = dfp_vol_sg_sub_fit.apply(lambda x: find_and_remove_from_pivot(-x),axis=0)
if return_smoothed_volume:
    dfp_vol = dfp_vol_sg * dfp_mask.values
else:
    dfp_vol = dfp['volume'] * dfp_mask.values

dfm = dfp_vol.stack().reset_index()
dfm.rename(columns={0:'smooth_volume'},inplace=True)


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

#%%
#%%
# try rerunning peak finding on the new dfp_vol_sg dataframe
dfd = df0[df0["colony"] == 'small']
ided_tracks = []
for track_id in track_list:
    # y = dfd[dfd.track_id == track_id]['volume_sub_fit_volume'].values

    series_1 = pd.DataFrame(dfp.loc[:,'volume'][track_id]).rename(columns = {track_id:'volume'})
    series_2 = pd.DataFrame(dfp.loc[:,'fit_volume'][track_id]).rename(columns={track_id:'fit_volume'})
    dftrack = pd.concat([series_1,series_2],axis=1).reset_index()
    x = dftrack['index_sequence'].values

    
    y, volume, fit_volume, smooth_volume = get_fit_volume_minus_smoothed_y_from_df(dftrack, return_all=True)
    peaks, props, results_full = find_drops_relative_to_fit(-y)
    out = find_and_remove_from_pivot(-y)
    if np.any(out):
        ided_tracks.append(track_id)

        fig,ax = plt.subplots(1,1,figsize=(8,4), sharex=True)
        plot_features_and_peaks(dftrack,volume,smooth_volume,fit_volume,peaks,props,ax)
        plt.show()
print(len(ided_tracks))



