# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.analyses.volume.plot_help import adjust_axis_positions
from nuc_morph_analysis.lib.preprocessing import filter_data, compute_change_over_time
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot 
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization import plotting_tools
from nuc_morph_analysis.analyses.volume.plot_help import update_plotting_params
#%%
#%% update plotting parameters
fs, fw, fh = update_plotting_params()

#%%
# load the data
df0 = load_dataset_with_features('all_baseline',load_local=True, remove_growth_outliers=False)
df0 = filter_data.all_timepoints_minimal_filtering(df0)
# df0 = filter_data.all_timepoints_full_tracks(df0)
#%% TEMP: compute feature
df0 = compute_change_over_time.run_script(df0, dxdt_feature_list=['volume'], bin_interval_list=[5])
df0 = compute_change_over_time.run_script(df0, dxdt_feature_list=['dxdt_5_volume_start'], bin_interval_list=[5])
df0 = compute_change_over_time.run_script(df0, dxdt_feature_list=['smooth_volume_dips_removed_um'], bin_interval_list=[5])
df0 = compute_change_over_time.run_script(df0, dxdt_feature_list=['dxdt_5_smooth_volume_dips_removed_um_start'], bin_interval_list=[5])
yscale, ylabel, yunit, _ = get_plot_labels_for_metric("volume")
df0['volume_drop'] = df0['dxdt_5_volume_start'] * yscale * 5
df0['volume_drop_smooth'] = df0['dxdt_5_smooth_volume_dips_removed_um_start'] * yscale * 5

#%%
df_full = filter_data.all_timepoints_full_tracks(df0)
#%%
save_dir = Path(__file__).parent / 'figures' / 'volume_dip_figures'
ycol = 'volume_drop'
threshold = -100
colony_list = ['small','medium','large']
fig,axlist = plt.subplots(1,1,figsize=(fw,fh))
axlist = np.asarray([axlist]) if type(axlist) != np.ndarray else axlist # for mypy
assert type(axlist) == np.ndarray # for mypy
ax = axlist[0]
for colony in colony_list:
    dfcolony = df0[df0['colony'] == colony]
    dfthresh = dfcolony[dfcolony[ycol] < threshold]

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
    ax.plot(x,y,label=colony,color=plotting_tools.COLONY_COLORS[colony])
ax.set_xlabel(f"{xlabel} {xunit}")
ax.set_ylabel(f'% of nuclei with volume\ndrop < {threshold} um^3')
ax.set_title('Number of dips over time')
fig,axlist = adjust_axis_positions(fig,axlist,curr_pos=None,width=0.9,height=0.9,space=0.075)
ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
            fontsize=fs,frameon=False,
            markerscale=1,handlelength=1,
            labelspacing=0,
            )
if threshold == -50:
    ax.set_yticks(np.arange(0,110,10))
    ax.set_ylim(0,30)
elif threshold == -100:
    ax.set_yticks(np.arange(0,8,2))
    ax.set_ylim(0,6)
ax.set_xticks(np.arange(0,60,12))
ax.set_xlim(0,48)
ax.set_xlabel('Movie time (hr)')


savename = save_dir / f"volume_dips_over_time_all_colonies-{ycol}"
for ext in ['.png','.pdf']:
    savepath = str(save_dir / f"{savename}")
    save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)