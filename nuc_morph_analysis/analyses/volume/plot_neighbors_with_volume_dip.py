#%% SuppFigS10 panel A left and middle (right is done in TFE)
import matplotlib.pyplot as plt
from pathlib import Path
from nuc_morph_analysis.analyses.neighbor_of_X.misc_neighbor_helper_functions import (
    find_immediate_neighbors,
    compute_distances_between_nuclei,
    get_ordered_list_of_nuclei_by_distance_to_track,
    get_a_cells_neighbors_as_track_id_list
)

from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.analyses.volume.plot_help import (
    plot_neighbors_volume_over_time, plot_tracks_aligned_at_volume_drop_onset,update_plotting_params
)
from nuc_morph_analysis.lib.preprocessing import filter_data, compute_change_over_time
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot 
from nuc_morph_analysis.analyses.volume import plot_help

#%%
# load data
df0 = load_dataset_with_features('all_baseline', remove_growth_outliers=False)

#%% TEMP: add feature
df0 = filter_data.all_timepoints_full_tracks(df0)
#%%
# update plotting parameters
fs,fw,fh = update_plotting_params()
save_dir = Path(__file__).parent / 'figures' / 'volume_dip_figures'

#%%
# identify the tracks
MAIN_TRACK_ID = 75725
TIMEPOINT = 239

track_id_list = get_a_cells_neighbors_as_track_id_list(df0,MAIN_TRACK_ID,TIMEPOINT)

#%%
#% plot main track in black (volume vs index_sequence)

fig,axlist = plot_neighbors_volume_over_time(dfcolony,track_id_list)

save_name = f"immediate_neighbors_of_main_track_{MAIN_TRACK_ID}"
save_path = str(save_dir / save_name)
for ext in ['.png','.pdf']:
    save_and_show_plot(save_path,ext,fig,transparent=False,keep_open=True)
plt.show()

#%%
# plot 2: neighbors volume over time (aligned at dip as t=0)
# t=0 is dip minimum
# y=0 is volume at t=-5

fig,axlist = plot_tracks_aligned_at_volume_drop_onset(dfcolony,track_id_list,MAIN_TRACK_ID,TIMEPOINT)

save_name = f"dip_shape_{MAIN_TRACK_ID}"
save_path = str(save_dir / save_name)
for ext in ['.png','.pdf']:
    save_and_show_plot(save_path,ext,fig,transparent=False,keep_open=True)
plt.show()
