# %%
# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.analyses.volume import plot_help

from nuc_morph_analysis.analyses.volume.plot_help import (
    plot_neighbors_volume_over_time, plot_tracks_aligned_at_volume_drop_onset,update_plotting_params,
    adjust_axis_positions, plot_track_with_fit_line, plot_track_with_volume_dip,
    plot_dip_detection_validation, plot_dxdt_over_time_by_cell_cycle
)
from nuc_morph_analysis.lib.preprocessing import filter_data, compute_change_over_time, add_times
from nuc_morph_analysis.lib.visualization.matplotlib_to_axlist import type_axlist
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot 
from nuc_morph_analysis.analyses.volume import filter_out_dips
from nuc_morph_analysis.analyses.neighbor_of_X.misc_neighbor_helper_functions import get_a_cells_neighbors_as_track_id_list

from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization import plotting_tools
from nuc_morph_analysis.analyses.volume_variation import plot_features

from nuc_morph_analysis.lib.visualization.example_tracks import EXAMPLE_TRACKS

#%%
# now load data with growth outliers
df_outliers = load_dataset_with_features('all_baseline', remove_growth_outliers=False)
# and without outliers
df = load_dataset_with_features('all_baseline', remove_growth_outliers=True)
#%%
# apply minimal filtering
df_outliers = filter_data.all_timepoints_full_tracks(df_outliers)
df = filter_data.all_timepoints_full_tracks(df)
#%%
# update plotting parameters
fs,fw,fh = update_plotting_params()
save_dir = Path(__file__).parent / 'figures' / 'volume_dip_figures'
# %%
# S10 panel A, left
# choose one track and its neighbors (at a given time) to plot over time
MAIN_TRACK_ID = EXAMPLE_TRACKS['volume_dip_example']
TIMEPOINT = 239
track_id_list = get_a_cells_neighbors_as_track_id_list(df_outliers,MAIN_TRACK_ID,TIMEPOINT)
fig,_ = plot_neighbors_volume_over_time(df_outliers,track_id_list)

save_name = f"S10_A_left-immediate_neighbors_of_main_track_{MAIN_TRACK_ID}"
save_path = str(save_dir / save_name)
for ext in ['.png','.pdf']:
    save_and_show_plot(save_path,ext,fig,transparent=False,keep_open=True)
plt.show()

#%%
# S10 panel A, middle
fig,_ = plot_tracks_aligned_at_volume_drop_onset(df_outliers,track_id_list,MAIN_TRACK_ID,TIMEPOINT)

save_name = f"S10_A_middle-dip_shape_{MAIN_TRACK_ID}"
save_path = str(save_dir / save_name)
for ext in ['.png','.pdf']:
    save_and_show_plot(save_path,ext,fig,transparent=False,keep_open=True)
plt.show()

# %%
# S10 panel B, illustrate the effect of volume dip on transient growth rate
    
df_full = filter_data.all_timepoints_full_tracks(df) # filter to only full tracks

fig,axlist_untyped = plt.subplots(nrows=2,ncols=1,figsize=(6.5,8))
axlist = type_axlist(axlist_untyped)

# add_time_point_lines=False,timepoint=None
volume_dip_example_track = 86570

main_track_list = [(volume_dip_example_track, 263)]
for main_track_id, timepoint in main_track_list:
    ax = axlist[0]
    ax = plot_track_with_volume_dip(ax,df_full,main_track_id,add_time_point_lines=True,timepoint=timepoint)
    ax = axlist[1]
    ax = plot_track_with_volume_dip(ax,df_full,main_track_id,xcol='index_sequence',ycol='dxdt_48_volume')
    fig,axlist_untyped = adjust_axis_positions(fig,axlist,curr_pos=None,width=0.6,height=0.6,space=0.2,horizontal=False)
    for ext in ['.png','.pdf']:
        savepath = save_dir / f"S10B_track_{main_track_id}_volume_dip{ext}"
        save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)
    plt.show()

#%% S10panelC steps 1 and 2
dftracks = df_full[df_full['track_id'].isin([volume_dip_example_track])]
columns_to_remove = dftracks.columns[dftracks.columns.str.contains('dip|jump')]
dftracks = dftracks.drop(columns=columns_to_remove)

# reprocess so that intermediate arrays are returned
dftracks_out = filter_out_dips.run_script(dftracks,
                                          return_intermediates=True,
                                          use_detrended=True,)

fig,ax = plot_dip_detection_validation(dftracks_out[dftracks_out['track_id'] == volume_dip_example_track],peak_str = 'dips')
fig,ax = adjust_axis_positions(fig,ax,curr_pos=None,width=1.2,height=1.2,space=0.6,keep_labels=True)
for ext in ['.png','.pdf']:
    savepath = save_dir / f"S10C_left_middle-track_{volume_dip_example_track}_dip_detection_validation{ext}"
    save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)
#%%
# S10 panel C step3
df_track = df_full[df_full.track_id == volume_dip_example_track]
fig,axlist_untyped = plt.subplots(2,1,figsize=(fw,fh),sharey=False)
axlist = type_axlist(axlist_untyped)

_ = plot_track_with_fit_line(df_track,axlist[0],
                                ycol1='volume',
                                ycol2='volume_dips_removed_um_unfilled',
                                )

_ = plot_track_with_fit_line(df_track, ax = axlist[1],
                                ycol1='dxdt_48_volume',
                                ycol2 = 'dxdt_48_volume_dips_removed_um_unfilled',
                                )

xlimmax = np.max([ax.get_xlim()[1] for ax in axlist])
for ax in axlist:
    
    ax.set_xticks(np.arange(0,20,4))
    ax.set_xlim(-2,xlimmax)

fig,axlist_untyped = adjust_axis_positions(fig,axlist,curr_pos=None,width=0.6,height=0.6,space=0.2,horizontal=False)
axlist = type_axlist(axlist_untyped)
axlist[0].text(0.05,0.99,f"track {volume_dip_example_track}",transform=axlist[0].transAxes,
        ha = 'left',va='top',fontsize=fs)
axlist[0].legend(loc='lower left',bbox_to_anchor=(1.05,0.0),
                    fontsize=fs,frameon=False,
                    markerscale=1,handlelength=1,
                    labelspacing=0,
                    )

# now save
savename = f"S10C_right-volume_fit_volume_fit_track{volume_dip_example_track}"
for ext in ['.png','.pdf']:
    save_path = str(save_dir / savename)
    save_and_show_plot(str(save_path),ext,fig,transparent=False,keep_open=True)
plt.show()

# %%
# S10 panel D
magnitude_col = 'volume_dips_volume_change_at_center'
ycol = 'volume_dips_peak_mask_at_center' 
colony_list = ['small','medium','large']

for threshold in [-50,0]:
    fig,axlist_untyped = plt.subplots(1,1,figsize=(fw,fh))
    axlist = type_axlist(axlist_untyped)
    ax = axlist[0]
    for ci,colony in enumerate(colony_list):
        dfcolony = df[df['colony'] == colony]
        dfthresh = dfcolony[dfcolony[ycol]==True]
        dfthresh = dfthresh[dfthresh[magnitude_col] < threshold]
        dfcount = dfthresh.groupby('index_sequence').count()
        df_all = dfcolony[['track_id','index_sequence']].groupby('index_sequence').count()
        df_dips = dfcount['track_id']
        df_all['number_of_nuclei'] = df_all['track_id']
        df_all['number_of_dips'] = 0
        df_all.loc[df_dips.index,'number_of_dips'] = df_dips.values

        # now plot number of dips over time
        xscale,xlabel,xunit,_ = get_plot_labels_for_metric('index_sequence')
        x = df_all.index * xscale
        yd = df_all['number_of_dips']
        yn = df_all['number_of_nuclei']
        y = yd / yn *100

        zorderval = 1 if threshold !=0 else -1 # to ensure large colony is in front when it has fewer peaks
        ax.plot(x,y,label=colony,color=plotting_tools.COLONY_COLORS[colony],zorder=ci*1000*zorderval)
    ax.set_xlabel(f"{xlabel} {xunit}")
    ax.set_ylabel(f'% of nuclei')
    # ax.set_title('Number of dips over time')
    if threshold !=0 :
        text_str = f"dips < {threshold} μm\u00B3"
    else:
        text_str = f"all dips"
    ax.text(0.05,0.99,text_str,transform=ax.transAxes,
            ha = 'left',va='top',fontsize=fs)

    fig,axlist_untyped = adjust_axis_positions(fig,axlist,curr_pos=None,width=0.9,height=0.5,space=0.075)
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                fontsize=fs,frameon=False,
                markerscale=1,handlelength=1,
                labelspacing=0,
                )
    if threshold != -100:
        curr_ylim = ax.get_ylim()

        ax.set_yticks(np.arange(0,30,5))
        ax.set_ylim(0,curr_ylim[1]*1.2)
    else:
        ax.set_yticks(np.arange(0,30,10))
        ax.set_ylim(0,28)
    ax.set_xticks(np.arange(0,60,12))
    ax.set_xlim(0,48)
    ax.set_xlabel('Movie time (hr)')


    savename = save_dir / f"S10_D-volume_dips_over_time_all_colonies-{ycol}-{threshold}"
    for ext in ['.png','.pdf']:
        savepath = str(save_dir / f"{savename}")
        save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)


#%%
# S10 panel #
# now make plot with cell cycle bins overlayed on medium colony only (panel F right)
df_full = add_times.digitize_time_column(df_full,0,1,step_size=0.02,time_col='normalized_time',new_col='dig_time')

ycol = 'dxdt_48_volume_dips_removed_um_unfilled'
for colony in colony_list:
    dfc = df_full[df_full['colony']==colony]
    fig,ax = plot_dxdt_over_time_by_cell_cycle(dfc,ycol)

    fig,ax = adjust_axis_positions(fig,ax,curr_pos=None,width=0.9,height=0.6,space=0.075)

    plt.suptitle(f"{ycol}")
    for ext in ['.png','.pdf']:
        savepath = save_dir / f"S10_E-cell_cycle_bins_for_only_{colony}_{ycol}_{ext}"
        save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)
    plt.show()

#%%
# S10 panel F and G
colony='all_baseline'

dfc = df_full[df_full["colony"] == colony] if colony != "all_baseline" else df_full
for local_radius_str in ["90um", "whole_colony"]:
    for pngflag in [True, False]:
        plot_features.scatter_plot(
            dfc,
            "all_baseline",
            f"neighbor_avg_dxdt_48_volume_dips_removed_um_unfilled_{local_radius_str}",
            "dxdt_48_volume_dips_removed_um_unfilled",
            color_map="#808080",
            figdir=save_dir,
            fitting=False,
            n_resamples=2,
            require_square=False,
            opacity=0.1,
            markersize=10,
            titleheader="full_tracks for all timepoints",
            dpi=150,
            file_extension=".pdf",
            transparent=True,
            add_unity_line=True,
            remove_all_points_in_pdf=pngflag,
        )



#%%

