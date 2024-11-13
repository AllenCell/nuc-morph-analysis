# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from nuc_morph_analysis.analyses.volume.plot_help import adjust_axis_positions, plot_dfg, CYCLE_COLOR_DICT, update_plotting_params
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.lib.preprocessing import filter_data, add_times, compute_change_over_time
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot 
from nuc_morph_analysis.lib.visualization import plotting_tools
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.analyses.volume import filter_out_dips

#%%
# load the data
df = load_dataset_with_features('all_baseline',load_local=True, remove_growth_outliers=True)

#%%
#TEMP: add features
# df.drop(columns=['dxdt_48_volume','dxdt_48_fit_volume'],inplace=True)
# df.drop(columns=['dxdt_48_volume_per_V','dxdt_48_fit_volume_per_V'],inplace=True)
# df = compute_change_over_time.run_script(df, dxdt_feature_list=['volume'], bin_interval_list=[48])
# df = compute_change_over_time.add_dvdt_over_V(df,['dxdt_48_volume'],volume_col='volume')

df = filter_data.all_timepoints_minimal_filtering(df) # apply minimal filterting
#%%
df_full = filter_data.all_timepoints_full_tracks(df) # filter to only full tracks
# add digitzed normalized time column for cell cycle subest filtering
df_full = add_times.digitize_time_column(df_full,0,1,step_size=0.02,time_col='normalized_time',new_col='dig_time')

# drop_columns = [x for x in df_full.columns if 'drop' in x]
# jump_columns = [x for x in df_full.columns if 'jump' in x]
# df_full.drop(columns=drop_columns,inplace=True)
# df_full.drop(columns=jump_columns,inplace=True)
# df_full = filter_out_dips.run_script(df_full)
# #%%
df_full = compute_change_over_time.run_script(df_full,
                                               dxdt_feature_list=['nondt_volume_dips_removed_um',
                                                                  'nondt_volume_dips_removed_um_unfilled'],
                                                 bin_interval_list=[48])

#%% update plotting parameters
# set global font sizes to be 8
fs,fw,fh = update_plotting_params()

#%% make plot of medium colony only for panel F (left)
colony_list = ['small','medium','large']

nrows = 1
ncols = len(colony_list)
figdir = Path(__file__).parent / 'figures' / 'fit_fig5_all_nuclei'

ycol = 'dxdt_48_fit_volume_per_V'
xcol1 = 'index_sequence'
plot_type = 'mean'
ycol_list = [
    # 'dxdt_48_fit_volume_per_V',
    # 'dxdt_48_volume_per_V',
    # 'dxdt_48_smooth_volume_dips_removed_um_per_V',
    # 'dxdt_48_volume_dips_removed_um_per_V',
    # 'dxdt_48_smooth_volume_dips_removed_um',
    'dxdt_48_volume',

    # 'dxdt_48_volume_dips_removed_um',
    'dxdt_48_volume_dips_removed_um_unfilled',
    # 'dxdt_48_nondt_volume_dips_removed_um',
    'dxdt_48_nondt_volume_dips_removed_um_unfilled',

    # 'dxdt_48_fit_volume',
    ]
for ycol in ycol_list:
    fig,ax = plt.subplots(nrows,ncols,figsize=(fw,fh))
    ax = np.asarray([ax]) if type(ax) != np.ndarray else ax # for mypy
    assert type(ax) == np.ndarray # for mypy

    for ci,colony in enumerate(colony_list):
        dfc = df[df['colony']==colony]
        curr_ax = ax[ci]
        curr_ax = plot_dfg(dfc,xcol1,ycol,"",curr_ax,plot_type=plot_type,colorby='colony')

    fig,ax = adjust_axis_positions(fig,ax,curr_pos=None,width=1,height=0.7,space=0.075)
    plt.suptitle(f"{ycol} vs {xcol1}")
    for ext in ['.png','.pdf']:
        savepath = figdir / f"ALL_{ycol}_{xcol1}_{plot_type}{ext}"
        save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)
    plt.show()


#%%
colony_list = ['small','medium','large']

nrows = 1
ncols = len(colony_list)
figdir = Path(__file__).parent / 'figures' / 'fit_fig5_fulltrack_nuclei'

ycol = 'dxdt_48_fit_volume_per_V'
xcol1 = 'index_sequence'
plot_type = 'mean'
ycol_list = [
    # 'dxdt_48_fit_volume_per_V',
    # 'dxdt_48_volume_per_V',
    # 'dxdt_48_smooth_volume_dips_removed_um_per_V',
    # 'dxdt_48_volume_dips_removed_um_per_V',
    # 'dxdt_48_smooth_volume_dips_removed_um',
    'dxdt_48_volume',

    # 'dxdt_48_volume_dips_removed_um',
    'dxdt_48_volume_dips_removed_um_unfilled',
    # 'dxdt_48_nondt_volume_dips_removed_um',
    'dxdt_48_nondt_volume_dips_removed_um_unfilled',

    # 'dxdt_48_fit_volume',
    ]
for ycol in ycol_list:
    fig,ax = plt.subplots(nrows,ncols,figsize=(fw,fh))
    ax = np.asarray([ax]) if type(ax) != np.ndarray else ax # for mypy
    assert type(ax) == np.ndarray # for mypy

    for ci,colony in enumerate(colony_list):
        dfc = df_full[df_full['colony']==colony]
        curr_ax = ax[ci]
        curr_ax = plot_dfg(dfc,xcol1,ycol,"",curr_ax,plot_type=plot_type,colorby='colony')

    fig,ax = adjust_axis_positions(fig,ax,curr_pos=None,width=1,height=0.7,space=0.075)
    plt.suptitle(f"{ycol} vs {xcol1}")
    for ext in ['.png','.pdf']:
        savepath = figdir / f"ALL_{ycol}_{xcol1}_{plot_type}{ext}"
        save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)
    plt.show()


#%% now make plot with cell cycle bins overlayed on medium colony only (panel F right)
figdir = Path(__file__).parent / 'figures' / 'fit_fig5_fulltracks_cell_cycle_bins_by_colony'
figdir.mkdir(exist_ok=True,parents=True)

cell_cycle_width = 0.2
cell_cycle_centers = [0.3,0.5,0.7]
cell_cycle_bins = [(cc-cell_cycle_width/2,cc+cell_cycle_width/2) for cc in cell_cycle_centers]

plot_type = 'mean'
colony_list = ['small','medium','large']
# plot the mean of the value (ycol) binned by xcol for given cell cycle windows
xcol1 = 'index_sequence'
for ycol in ycol_list:
    for colony in colony_list:

        nrows = 1
        ncols = len(cell_cycle_bins)

        fig,ax = plt.subplots(nrows,ncols,figsize=(fw,fh))
        assert type(ax) == np.ndarray # for mypy

        dfc = df_full[df_full['colony']==colony]

        for ri,cell_cycle_bin in enumerate(cell_cycle_bins):
            curr_ax = ax[ri]
            curr_ax = plot_dfg(dfc,xcol1,ycol,"all",curr_ax,plot_type=plot_type,colorby='colony')
            dfcc = dfc[(dfc['dig_time'] >= cell_cycle_bin[0]) & (dfc['dig_time'] <= cell_cycle_bin[1])]
            
            # top column
            dfcc['cell_cycle'] = ri
            labelstr = f"{cell_cycle_bin[0]:.1f}-{cell_cycle_bin[1]:.1f}"
            curr_ax = plot_dfg(dfcc,xcol1,ycol,"cell cycle subset",curr_ax,plot_type=plot_type,colorby='k')
            curr_ax.set_title(labelstr,color='k')
        curr_ax.legend(loc='center left',bbox_to_anchor=(1.05,0.5),title='Cell cycle window')

        fig,ax = adjust_axis_positions(fig,ax,curr_pos=None,width=0.9,height=0.6,space=0.075)
        
        plt.suptitle(f"{ycol}")
        # savepath = figdir / f"cell_cycle_bins_{ycol}_{xcol1}_{plot_type}.png"
        for ext in ['.png','.pdf']:
            savepath = figdir / f"cell_cycle_bins_for_only_{colony}_{ycol}_{xcol1}_{plot_type}{ext}"
            save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)
        plt.show()


#%%
# # try a version that overalys all cell cycle bins together (panel F alt)
# figdir = Path(__file__).parent / 'figures' / 'fit_fig5_fulltracks_cell_cycle_bins_ONLY'
# figdir.mkdir(exist_ok=True,parents=True)

# cell_cycle_width = 0.2
# cell_cycle_centers = [0.3,0.5,0.7]
# cell_cycle_bins = [(cc-cell_cycle_width/2,cc+cell_cycle_width/2) for cc in cell_cycle_centers]

# # plot the mean of the value (ycol) binned by xcol for given cell cycle windows
# xcol1 = 'index_sequence'
# ycol = 'dxdt_48_fit_volume'
# plot_type = 'mean'
# for colony in colony_list:

#     nrows = 1
#     ncols = 1

#     fig,ax = plt.subplots(nrows,ncols,figsize=(fw,fh))
#     if type(ax) != np.ndarray:
#         ax = np.asarray([ax])
#     assert type(ax) == np.ndarray # for mypy

#     dfc = df_full[df_full['colony']==colony]

#     for ri,cell_cycle_bin in enumerate(cell_cycle_bins):
#         curr_ax = ax[0]
#         dfcc = dfc[(dfc['dig_time'] >= cell_cycle_bin[0]) & (dfc['dig_time'] <= cell_cycle_bin[1])]
        
#         # top column
#         dfcc['cell_cycle'] = ri
#         labelstr = f"{cell_cycle_bin[0]:.2f}-{cell_cycle_bin[1]:.2f}"
#         curr_ax = plot_dfg(dfcc,xcol1,ycol,labelstr,curr_ax,plot_type=plot_type,colorby=CYCLE_COLOR_DICT[ri])
#     curr_ax.legend(loc='center left',bbox_to_anchor=(1.05,0.5),title='Cell cycle window')

#     fig,ax = adjust_axis_positions(fig,ax,curr_pos=None,width=0.9,height=0.6,space=0.075)
    
#     plt.suptitle(f"{colony}")
#     # savepath = figdir / f"cell_cycle_bins_{ycol}_{xcol1}_{plot_type}.png"
#     for ext in ['.png','.pdf']:
#         savepath = figdir / f"cell_cycle_bins_for_only_{colony}_{ycol}_{xcol1}_{plot_type}{ext}"
#         save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)
#     plt.show()

#%%
# create supplemental plot showing how dxdt_48_volume changes over dig_time
figdir = Path(__file__).parent / 'figures' / 'fit_fig5_dig_time_supp'
figdir.mkdir(exist_ok=True,parents=True)

cell_cycle_width = 0.2
cell_cycle_centers = [0.3,0.5,0.7]
cell_cycle_bins = [(cc-cell_cycle_width/2,cc+cell_cycle_width/2) for cc in cell_cycle_centers]
cell_cycle_bins =  cell_cycle_bins
# plot the mean of the value (ycol) binned by xcol for given cell cycle windows
colony='medium'

xcol1 = 'dig_time'
ycol = 'dxdt_48_volume'
plot_type = 'mean'

nrows = 1
ncols = 1
for ycol in ycol_list:
    for colony in ['small','medium','large']:
        fig,ax = plt.subplots(nrows,ncols,figsize=(fw,fh))
        if type(ax) != np.ndarray:
            ax = np.asarray([ax])
        assert type(ax) == np.ndarray # for mypy

        dfc = df_full[df_full['colony']==colony]
        # # subset the dataframe to only include timepoints where all cell cycle bins are present
        # log1 = dfc.loc[dfc['dig_time']>0.4,'index_sequence'].values
        # log2 = dfc.loc[dfc['dig_time']<0.6,'index_sequence'].values
        # dfc = dfc.loc[dfc['index_sequence'].isin(np.intersect1d(log1,log2))]


        curr_ax = ax[0]
        dfcc = dfc[(dfc['dig_time'] >= cell_cycle_bins[0][0]) & (dfc['dig_time'] <= cell_cycle_bins[-1][1])]
        
        # top column
        labelstr = f"{cell_cycle_bin[0]:.2f} to {cell_cycle_bin[1]:.2f}"
        curr_ax = plot_dfg(dfcc,xcol1,ycol,colony,curr_ax,plot_type=plot_type,colorby='colony')
        # curr_ax.legend(loc='center left',bbox_to_anchor=(1.05,0.5),title='Cell cycle window')
        curr_ax.legend(loc='lower left',bbox_to_anchor=(0,-0.1),
                            fontsize=fs,frameon=False,
                            markerscale=1,handlelength=1,
                            labelspacing=0,
                            )

        fig,ax = adjust_axis_positions(fig,ax,curr_pos=None,width=0.9,height=0.6,space=0.075)
        
        plt.suptitle(f"{colony}-{ycol}")
        # savepath = figdir / f"cell_cycle_bins_{ycol}_{xcol1}_{plot_type}.png"
        for ext in ['.png','.pdf']:
            savepath = figdir / f"cell_cycle_bins_for_only_{colony}_{ycol}_{xcol1}_{plot_type}{ext}"
            save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)
        plt.show()

#%%
# overlay with aligned colony time
colony_list = ['small','medium','large']

nrows = 1
ncols = 1
figdir = Path(__file__).parent / 'figures' / 'fit_fig5_colony_time'

xcol1 = 'colony_time'
plot_type = 'mean'
ycol_list = [
    # 'dxdt_48_fit_volume_per_V',
    # 'dxdt_48_volume_per_V',
    # 'dxdt_48_smooth_volume_dips_removed_um_per_V',
    # 'dxdt_48_volume_dips_removed_um_per_V',
    # 'dxdt_48_smooth_volume_dips_removed_um',
    'dxdt_48_volume_dips_removed_um',
    'dxdt_48_volume',
    # 'dxdt_48_fit_volume',
    ]
for ycol in ycol_list:
    fig,ax = plt.subplots(nrows,ncols,figsize=(fw,fh))
    ax = np.asarray([ax]) if type(ax) != np.ndarray else ax # for mypy
    assert type(ax) == np.ndarray # for mypy

    for _,colony in enumerate(colony_list):
        dfc = df_full[df_full['colony']==colony]
        curr_ax = ax[0]
        curr_ax = plot_dfg(dfc,xcol1,ycol,"",curr_ax,plot_type=plot_type,colorby='colony')

    fig,ax = adjust_axis_positions(fig,ax,curr_pos=None,width=1,height=0.7,space=0.075)
    plt.suptitle(f"{ycol} vs {xcol1}")
    for ext in ['.png','.pdf']:
        savepath = figdir / f"ALL_{ycol}_{xcol1}_{plot_type}{ext}"
        save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)
    plt.show()
