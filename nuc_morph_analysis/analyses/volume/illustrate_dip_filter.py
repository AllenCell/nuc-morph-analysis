# %% SuppFigS10 E
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.analyses.volume.plot_help import update_plotting_params, adjust_axis_positions
from nuc_morph_analysis.lib.preprocessing import filter_data
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
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

track_list = [7875,72414,86246,85345,77291,86570,83675,84663,86212]
#%%
from nuc_morph_analysis.analyses.volume import filter_out_dips
find_dips=True
dftracks = df_full[df_full['track_id'].isin(track_list)]
columns_to_remove = dftracks.columns[dftracks.columns.str.contains('drop|jump')]
dftracks = dftracks.drop(columns=columns_to_remove)


dftracks_out = filter_out_dips.run_script(dftracks,
                                          return_intermediates=False,
                                          use_detrended=True,)
    

   #%%

def add_peaks_to_plot(curr_ax,dftrack,ycol,peak_str = 'jumps',yscale=1):
    xscale, xlabel, xunit, _ = get_plot_labels_for_metric("index_sequence")
    # yscale, ylabel, yunit, _ = get_plot_labels_for_metric("volume")

    xcol = 'transition_time'
    peaks = dftrack[f'volume_{peak_str}_peak_center_mask'] # boolean array
    peaks = peaks[peaks>0].index
    for peak in peaks:

        if peak not in dftrack.index:
            print(f"peak {peak} not in index")
            return dftrack
        xpeak = dftrack.loc[peak,xcol] * xscale
        ypeak = dftrack.loc[peak,ycol] * yscale

        thresh_col = f"{peak_str}_threshold"
        thresh = dftrack.loc[peak,thresh_col] 
        magnitude = dftrack.loc[peak,f"volume_{peak_str}_left_magnitude"]
        print(magnitude,thresh,peak_str)

        magnitude_bool = magnitude < thresh if peak_str == 'dips' else magnitude > thresh
        if not magnitude_bool:
            return dftrack
        # curr_ax.plot(xpeak,ypeak,'tab:red',marker='x',markersize=5)
        # ymax = ymin + dftrack.loc[peak,f"volume_{peak_str}_left_magnitude"]
        
        left_base_idx = int(dftrack.loc[peak,f"volume_{peak_str}_left_bases"])
        right_base_idx = int(dftrack.loc[peak,f"volume_{peak_str}_right_bases"])
        left_base = dftrack.loc[left_base_idx,xcol]
        right_base = dftrack.loc[right_base_idx,xcol]
        vol_at_left_base = dftrack.loc[left_base_idx,ycol] * yscale
        vol_at_right_base = dftrack.loc[right_base_idx,ycol] * yscale
        
        # curr_ax.scatter(left_base * xscale,vol_at_left_base,color='m',marker='o',s=10,label='left base')
        # curr_ax.scatter(right_base * xscale,vol_at_right_base,color='m',marker='o',s=10,label='left base')
        
        vols = dftrack.loc[[left_base_idx,peak,right_base_idx],ycol].values * yscale
        curr_ax.vlines(x=left_base * xscale, ymin=np.min(vols), ymax=np.max(vols),
                        color='m',linestyle='--')
        curr_ax.vlines(x=right_base * xscale, ymin = np.min(vols), ymax = np.max(vols),
        color='m',linestyle='--')

        mask = np.arange(left_base_idx,right_base_idx+1,dtype=int)
        mask = dftrack.index.isin(mask)
        xpeak_mask = dftrack.loc[mask,xcol] * xscale
        ypeak_mask = dftrack.loc[mask,ycol] * yscale

        curr_ax.plot(xpeak_mask,ypeak_mask,color='m',label='peak region')

    return dftrack

def plot_peak_detection_validation(dftrack,peak_str = 'dips'):
    # axis 0 is interpolated raw_volume, sg_smoothed volume and power law fit volume (steps 0 and step1)
    # axis 1 is smoothed volume detrended by power law fit (and detected peaks)
    #     and it would be cool to draw the peak magnitude as vertical line on the plot
    # axis 2 is the raw volume with the peak annotated (and with the peak magnitude as vertical line)
    # axis 3 is the raw volume with the peak removed
    track_id = dftrack['track_id'].values[0]
    transition = dftrack['frame_transition']
    dftrack['transition_time'] = dftrack['index_sequence'].copy() - transition
    dftrack.set_index('index_sequence',inplace=True)
    ncols = 3
    nrows = 1
    fig,ax = plt.subplots(nrows,ncols,figsize=(6.5,8))
    ax = np.asarray([ax]) if type(ax) != np.ndarray else ax
    assert type(ax) == np.ndarray

    # ax0
    # plot raw volume on ax0 and ax2
    for axi in [0,2]:
        curr_ax = ax[axi]
        xscale, xlabel, xunit, _ = get_plot_labels_for_metric("index_sequence")
        yscale, ylabel, yunit, _ = get_plot_labels_for_metric("volume")
        x = dftrack.transition_time.values * xscale
        y = dftrack['volume'].values * yscale
        curr_ax.plot(x,y,'tab:grey',linewidth=1,label='raw volume')
    curr_ax = ax[0]
    y = dftrack['volume_sg'].values 
    curr_ax.plot(x,y,'g',linewidth=0.7, label='smoothed')

    ycol = 'fit_volume_interpolated'
    yscale = 1
    y = dftrack[ycol].values * yscale
    curr_ax.plot(x,y,label='power law fit',color='m',linewidth=0.7,linestyle='-')
    curr_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3))
    curr_ax.set_ylabel(f"{ylabel} {yunit}")
    curr_ax.set_title('step1:\nsmooth volume trajectory\nand detrend with power law fit')
    curr_ax.set_yticks([400,600,800,1000,1200])
    curr_ax.set_ylim(400,1200)

    # ax1
    curr_ax = ax[1]
    ycol = f'volume_sg_sub_fit'
    y = dftrack[ycol].values
    curr_ax.plot(x,y,color='g',linewidth=0.7,label='smoothed & detrended',zorder=-100)
    # add_peaks_to_plot(curr_ax,dftrack,ycol,peak_str = peak_str)
    curr_ax.set_ylabel(f"Detrended volume {yunit}")
    curr_ax.set_title('step2:\nfind (inverse) peaks in\ndetrended volume trajectory')
    peaks = dftrack[f'volume_{peak_str}_peak_center_mask'] # boolean array
    peaks = peaks[peaks>0].index
    xpeak  = dftrack.loc[peaks,'transition_time'] * xscale
    ypeak = dftrack.loc[peaks,ycol]
    curr_ax.scatter(xpeak,ypeak,color='m',marker='o',s=2,label='detected peaks')
    curr_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
    curr_ax.set_yticks([-100,-50,0,50,100])
    curr_ax.set_ylim(-150,150)

    # ax3
    curr_ax = ax[2]
    ycol = f'volume_interpolated'
    ycol = f"volume_{peak_str}_removed_um_unfilled"
    yscale, ylabel, yunit, _ = get_plot_labels_for_metric("volume")
    yscale=1
    y = dftrack[ycol].values * yscale
    curr_ax.plot(x,y,'k',linewidth=1,label='volume with peaks removed',zorder=300)

    add_peaks_to_plot(ax[2],dftrack,ycol, peak_str = peak_str,yscale=yscale)
    curr_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3))
    curr_ax.set_ylabel(f"{ylabel} {yunit}")
    curr_ax.set_title('step3:\nannotate peak regions\n in raw volume trajectory')
    curr_ax.set_yticks([400,600,800,1000,1200])
    curr_ax.set_ylim(400,1200)

    # # ax4
    # curr_ax = ax[3]
    # ycol = f'volume_{peak_str}_removed_um'
    # y = dftrack[ycol].values * yscale
    # mask = dftrack[f'volume_{peak_str}_peak_region_mask']
    # xpeak = x[mask]
    # ypeak = y[mask]
    # curr_ax.plot(x,y,color='tab:grey',linewidth=2,label='volume with peaks removed',zorder=-200)

    # curr_ax.scatter(xpeak,ypeak,color='m',marker='.',s=2,label='masked peaks')
    
    # curr_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3))

    # set xlimit of all axes
    xlimmax = np.max([axx.get_xlim()[1] for axx in ax])
    for curr_ax in ax:
        curr_ax.set_xticks(np.arange(0,20,4))
        curr_ax.set_xlim(-2,xlimmax)

    # plt.suptitle(f"track {track_id}")
    savepath = save_dir / f"track_{track_id}_volume_{peak_str}_filtering"
    # fig,ax = adjust_axis_positions(fig,ax,curr_pos=None,width=2,height=2,space=0.6,keep_labels=True)
    fig,ax = adjust_axis_positions(fig,ax,curr_pos=None,width=0.6,height=0.6,space=0.6,keep_labels=True)

    for ext in ['.png','.pdf']:
        save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)
    plt.show()

for track in track_list:
    plot_peak_detection_validation(dftracks_out[dftracks_out['track_id'] == track],peak_str = 'dips')
    

# #%%
# from scipy.signal import savgol_filter
# import pandas as pd
# from scipy.signal import find_peaks
# dftrack = dftracks_out[dftracks_out['track_id'] == 7875]
# dftrack.set_index('index_sequence',inplace=True)

# vol = dftrack['volume_sg']
# xscale, xlabel, xunit, _ = get_plot_labels_for_metric("index_sequence")
# x = vol.index.values * xscale
# vold1 = vol.diff(1)
# # vold1_sg = savgol_filter(vold1, 24,2)
# vold1_sg = savgol_filter(vold1, window_length=24, polyorder=2, mode='interp')

# vold2 = pd.Series(index = vold1.index,data=vold1_sg).diff(1)
# fig,axlist = plt.subplots(3,1,figsize=(2*2,3*2),sharex=True)
# axlist[0].plot(x,vol.values)
# add_peaks_to_plot(axlist[0],dftrack,'volume_sg',peak_str = 'dips')
# axlist[1].plot(x,vold1.values)
# axlist[1].plot(x,vold1_sg)
# axlist[2].plot(x,vold2.values)

# #%%
# def peak_finder(dftrack, window_size=12):

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

# for track in track_list:
#     dftrack = dftracks_out[dftracks_out['track_id'] == track]
#     dftrack.set_index('index_sequence',inplace=True)
#     peak_finder(dftrack, window_size=12)


#%%


