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

track_list = [7875,86246,85345,77291,86570,83675,84663,86212]
#%%
from nuc_morph_analysis.analyses.volume import filter_out_dips
find_drops=True
dftracks = df_full[df_full['track_id'].isin(track_list)]
columns_to_remove = dftracks.columns[dftracks.columns.str.contains('drop|jump')]
dftracks = dftracks.drop(columns=columns_to_remove)


dftracks_out = filter_out_dips.run_script(dftracks,return_intermediates=False)
    

   #%%

def add_peaks_to_plot(curr_ax,dftrack,ycol,peak_str = 'jumps',yscale=1):
    xscale, xlabel, xunit, _ = get_plot_labels_for_metric("index_sequence")
    # yscale, ylabel, yunit, _ = get_plot_labels_for_metric("volume")

    peaks = dftrack[f'volume_{peak_str}_centers'] # boolean array
    peaks = peaks[peaks>0].index
    for peak in peaks:
        xpeak = peak * xscale
        ypeak = dftrack.loc[peak,ycol] * yscale
        curr_ax.plot(xpeak,ypeak,'tab:red',marker='x',markersize=5)
        # ymax = ymin + dftrack.loc[peak,f"volume_{peak_str}_magnitude"]
        
        left_base = int(dftrack.loc[peak,f"volume_{peak_str}_left_bases"])
        vol_at_left_base = dftrack.loc[left_base,ycol] * yscale
        
        curr_ax.scatter(left_base * xscale,vol_at_left_base,color='c',marker='.',s=10,label='left base')
        
        # ymin = np.min([vol_at_left_base,ypeak])
        # ymax = np.max([vol_at_left_base,ypeak])

        # right_base = int(dftrack.loc[peak,f"volume_{peak_str}_right_bases"])
        # vol_at_right_base = dftrack.loc[right_base,ycol] * yscale

        # center = int(dftrack.loc[peak,f"volume_{peak_str}_centers_vals"])

        # curr_ax.scatter(right_base * xscale,vol_at_right_base,color='y',marker='.',s=10)
        # ymin = np.min([vol_at_left_base,ypeak])
        # ymax = np.max([vol_at_left_base,ypeak])

        # curr_ax.plot([xpeak,xpeak],[ymin,ymax],color='tab:red',linestyle='--')


        props_magnitude = dftrack.loc[peak,f"volume_{peak_str}_magnitude"] * (0.108**3)
        print(f"peak {peak} magnitude {props_magnitude}")
        ymin = ypeak
        ymax = ypeak + props_magnitude
        curr_ax.plot([xpeak,xpeak],[ymin,ymax],color='r',linewidth=2,zorder=-200,linestyle='-',label='props magnitude')
    return dftrack

def plot_peak_detection_validation(dftrack,peak_str = 'drops'):
    # axis 0 is raw volume 
    # axis 1 is interpolated raw_volume, sg_smoothed volume and power law fit volume (steps 0 and step1)
    # axis 2 is smoothed volume detrended by power law fit (and detected peaks)
    #     and it would be cool to draw the peak magnitude as vertical line on the plot
    # axis 3 is the raw volume with the peak annotated (and with the peak magnitude as vertical line)
    # axis 4 is the raw volume with the peak removed
    track_id = dftrack['track_id'].values[0]
    dftrack.set_index('index_sequence',inplace=True)
    ncols = 5
    nrows = 1
    fig,ax = plt.subplots(nrows,ncols,figsize=(ncols*2,nrows*2),sharex=True)
    ax = np.asarray([ax]) if type(ax) != np.ndarray else ax
    assert type(ax) == np.ndarray

    # ax0
    curr_ax = ax[0]
    xscale, xlabel, xunit, _ = get_plot_labels_for_metric("index_sequence")
    yscale, ylabel, yunit, _ = get_plot_labels_for_metric("volume")
    x = dftrack.index.values * xscale
    y = dftrack['volume'].values * yscale
    curr_ax.plot(x,y,'tab:grey',marker='.',markersize=2,linewidth=2)

    y = dftrack['volume_sg'].values 
    curr_ax.plot(x,y,'k',linewidth=1, label='smoothed and interpolated volume')
    curr_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))

    # ax1
    curr_ax = ax[1]
    colors = ['k','tab:green']
    for yi,ycol in enumerate(['volume_sg','fit_volume_interpolated',]):
        yscale, _, _, _ = get_plot_labels_for_metric("volume")
        if ycol in ['fit_volume_interpolated','volume_sg']:
            yscale = 1
        y = dftrack[ycol].values * yscale
        linewidth = 2
        curr_ax.plot(x,y,label=ycol,color=colors[yi],linewidth=linewidth)
    curr_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))

    yscale, _, _, _ = get_plot_labels_for_metric("volume")

    # ax2
    curr_ax = ax[2]
    ycol = f'volume_sg_sub_fit_{peak_str}'
    y = dftrack[ycol].values
    curr_ax.plot(x,y,color='k',linewidth=2,label='smoothed vol - fit',zorder=-100)
    add_peaks_to_plot(curr_ax,dftrack,ycol,peak_str = peak_str)
    curr_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))


    # ax3
    curr_ax = ax[3]
    ycol = f'volume_interpolated'
    y = dftrack[ycol].values * yscale
    curr_ax.plot(x,y,color='k',linewidth=2,label='raw vol interp.',zorder=-300)
    mask = dftrack[f'volume_{peak_str}_mask']
    xpeak = x[mask]
    ypeak = y[mask]
    curr_ax.scatter(xpeak,ypeak,color='y',marker='.',s=2,label='masked peaks')
    add_peaks_to_plot(curr_ax,dftrack,ycol,peak_str = peak_str,yscale=yscale)
    curr_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))

    # ax4
    curr_ax = ax[4]
    ycol = f'volume_{peak_str}_removed_um'
    y = dftrack[ycol].values * yscale
    mask = dftrack[f'volume_{peak_str}_mask']
    xpeak = x[mask]
    ypeak = y[mask]
    curr_ax.plot(x,y,color='k',linewidth=2,label='volume with peaks removed',zorder=-200)
    curr_ax.scatter(xpeak,ypeak,color='y',marker='.',s=2,label='masked peaks')
    
    curr_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))



    plt.suptitle(f"track {track_id}")
    plt.show()

for track in track_list:
    plot_peak_detection_validation(dftracks_out[dftracks_out['track_id'] == track],peak_str = 'jumps')
#%%


