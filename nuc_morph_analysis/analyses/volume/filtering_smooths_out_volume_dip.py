# %% SuppFig S10 E and D
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from nuc_morph_analysis.analyses.volume.plot_help import (
    adjust_axis_positions, plot_track_with_fit_line,update_plotting_params, plot_track_with_volume_dip
    )
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.lib.preprocessing import filter_data,compute_change_over_time
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot 
#%%
# load the data
remove_growth_outliers = False
df = load_dataset_with_features('all_baseline',load_local=True, remove_growth_outliers=False)
df = filter_data.all_timepoints_minimal_filtering(df) # apply minimal filterting
df_full = filter_data.all_timepoints_full_tracks(df) # filter to only full tracks
#%% update plotting parameters
fs,fw,fh = update_plotting_params()

#%%

# df_full = compute_change_over_time.run_script(df_full, dxdt_feature_list=['nondt_volume_dips_removed_um',
#                                                                             'nondt_volume_dips_removed_um_unfilled'], bin_interval_list=[48])
# df_full = compute_change_over_time.add_dvdt_over_V(df_full,['dxdt_48_nondt_volume_dips_removed_um'],volume_col='nondt_volume_dips_removed_um')
#%%
#%%
# make figure showing that the fit volume smooths out volume dips
figdir = Path(__file__).parent / 'figures' / 'fig5_fit_volume_smooths_out_volume_dips'


TRACK_ID_LIST = [71532,73610,82349,86570,77291,75411]
if remove_growth_outliers==False:
    TRACK_ID_LIST = [71044] + TRACK_ID_LIST
for track_id in TRACK_ID_LIST:
    df_track = df_full[df_full.track_id == track_id]
    fig,axlist = plt.subplots(2,1,figsize=(fw,fh),sharey=False)
    axlist = np.asarray([axlist]) if type(axlist) != np.ndarray else axlist



    _ = plot_track_with_fit_line(df_track,
                                  'index_sequence',
                                  'volume',
                                  'volume_dips_removed_um_unfilled',
                                    # 'nondt_volume_dips_removed_um',
                                  axlist[0])
    
    _ = plot_track_with_fit_line(df_track,
                                  'index_sequence',
                                  'dxdt_48_volume',
                                    'dxdt_48_volume_dips_removed_um_unfilled',
                                    # 'dxdt_48_nondt_volume_dips_removed_um',

                                  axlist[1])

    xlimmax = np.max([ax.get_xlim()[1] for ax in axlist])
    for ax in axlist:
        
        ax.set_xticks(np.arange(0,20,4))
        ax.set_xlim(-2,xlimmax)
    
    fig,axlist = adjust_axis_positions(fig,axlist,curr_pos=None,width=0.6,height=0.6,space=0.2,horizontal=False)
    axlist[0].text(0.05,0.99,f"track {track_id}",transform=axlist[0].transAxes,
            ha = 'left',va='top',fontsize=fs)
    axlist[0].legend(loc='lower left',bbox_to_anchor=(1.05,0.0),
                     fontsize=fs,frameon=False,
                     markerscale=1,handlelength=1,
                     labelspacing=0,
                     )
    
    # now save
    savename = f"volume_fit_volume_fit_track{track_id}"
    for ext in ['.png','.pdf']:
        save_path = str(figdir / savename)
        save_and_show_plot(str(save_path),ext,fig,transparent=False,keep_open=True)
    plt.show()

#%%
#
savedir = Path(__file__).parent / 'figures' / 'volume_dip_track'
fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(6.5,8))
axlist = np.asarray([ax]) if type(ax) != np.ndarray else ax
assert type(ax) == np.ndarray # for mypy

# add_time_point_lines=False,timepoint=None
main_track_list = [(86570, 263)] #[(86570, 263),(75725, 239), (71532,131)]
for main_track_id, timepoint in main_track_list:
    ax = axlist[0]
    ax = plot_track_with_volume_dip(ax,df_full,main_track_id,add_time_point_lines=True,timepoint=timepoint)
    ax = axlist[1]
    ax = plot_track_with_volume_dip(ax,df_full,main_track_id,xcol='index_sequence',ycol='dxdt_48_volume')
    fig,axlist = adjust_axis_positions(fig,axlist,curr_pos=None,width=0.6,height=0.6,space=0.2,horizontal=False)
    for ext in ['.png','.pdf']:
        savepath = savedir / f"track_{main_track_id}_volume_dip{ext}"
        save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)
    plt.show()