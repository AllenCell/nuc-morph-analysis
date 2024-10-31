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
from nuc_morph_analysis.analyses.volume.filter_out_dips import find_drops_relative_to_fit, find_and_remove_from_pivot, get_fit_volume_minus_smoothed_y_from_df, plot_features_and_peaks
import numpy as np
#%%
#%% update plotting parameters
fs, fw, fh = update_plotting_params()

#%%
# load the data
df0 = load_dataset_with_features('all_baseline',load_local=True, remove_growth_outliers=False)
df0 = filter_data.all_timepoints_minimal_filtering(df0)
df_full = filter_data.all_timepoints_full_tracks(df0)

#%%
save_dir = Path(__file__).parent / 'figures' / 'volume_dip_filter_assessment'

#%%

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
from nuc_morph_analysis.analyses.volume import filter_out_dips
    
track_list = [86246,85345,77291,86570,83675,84663,86212]
dftracks = df_full[df_full['track_id'].isin(track_list)]

dftracks = filter_out_dips.run_script(dftracks)

dftracks = compute_change_over_time.run_script(dftracks, dxdt_feature_list=['smooth_volume_dips_removed_um','volume_dips_removed_um'], bin_interval_list=[48])
dftracks = compute_change_over_time.add_dvdt_over_V(dftracks,['dxdt_48_smooth_volume_dips_removed_um'],volume_col='smooth_volume_dips_removed_um')
dftracks = compute_change_over_time.add_dvdt_over_V(dftracks,['dxdt_48_volume_dips_removed_um'],volume_col='volume_dips_removed_um')

for track in track_list:
    dftrack = dftracks[dftracks['track_id'] == track]
    x = dftrack['index_sequence'].values
    yscale, ylabel, yunit, _ = get_plot_labels_for_metric("volume")
    y1 = dftrack['volume'].values * yscale
    y2 = dftrack['smooth_volume_dips_removed_um']
    plt.plot(x,y1,'tab:orange',marker='o',markersize=2)
    plt.plot(x,y2,'tab:blue',marker='o',markersize=2)
    
    dfsub = dftrack[dftrack['volume_drop_mask']>0]
    vmax = dfsub['volume'].max() * yscale
    x3 = dfsub['index_sequence'].values
    y3 = np.ones(x3.shape) * vmax
    plt.plot(x3,y3,'tab:green',marker='*',markersize=2)

    dfc = dftrack[dftrack['volume_drop_centers']>0]
    xcenter = dfc['index_sequence'].values
    ycenter = np.ones(xcenter.shape) * vmax
    plt.plot(xcenter,ycenter,'tab:red',marker='x',markersize=5)

    plt.show()