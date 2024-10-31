#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.analyses.volume.plot_help import (
    plot_neighbors_volume_over_time, plot_tracks_aligned_at_volume_drop_onset,update_plotting_params
)
from nuc_morph_analysis.lib.preprocessing import filter_data, compute_change_over_time
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot 
from nuc_morph_analysis.analyses.volume import plot_help

#%% 
# define helper functions
def find_immediate_neighbors(dfsearch,main_track_id,timepoint):
    colony = dfsearch[dfsearch["track_id"] == main_track_id]["colony"].values[0]
    dfcolony=dfsearch[dfsearch['colony']==colony]

    # get tracks by looking for neighbors
    dftrack = dfcolony[(dfcolony["track_id"] == main_track_id) & (dfcolony["index_sequence"] == timepoint)]
    cell_ids = dftrack["neighbors"].apply(lambda x:eval(x)).values[0]

    dftime = dfcolony[dfcolony["index_sequence"] == timepoint]
    immediate_neighbor_track_ids = dftime.loc[dftime.index.isin(cell_ids), "track_id"].values
    return immediate_neighbor_track_ids
    
def compute_distances_between_nuclei(dftime,main_track_id):
    # alternative workflow to get tracks by measuring distance to neighbors
    from scipy.spatial import distance_matrix
    dftrack = dftime[(dftime["track_id"] == main_track_id)]
    centroids = dftrack[["centroid_x", "centroid_y"]].values
    centroids_time = dftime[["centroid_x", "centroid_y"]].values
    dist = distance_matrix(centroids, centroids_time)
    return dist

def get_ordered_list_of_nuclei_by_distance_to_track(dist,dftime):
    # now determine the ordered list of neighbors
    sorted_index = np.argsort(dist, axis=1).reshape(-1,)
    sorted_dist = np.sort(dist, axis=1).reshape(-1,)
    sorted_cell_ids = dftime.index.values[sorted_index]
    sorted_track_ids = dftime['track_id'].values[sorted_index]
    return sorted_dist,sorted_cell_ids,sorted_track_ids,sorted_index

#%%
# load data
df0 = load_dataset_with_features('all_baseline', remove_growth_outliers=False)

#%% TEMP: add feature
df0 = filter_data.all_timepoints_full_tracks(df0)
df0 = compute_change_over_time.run_script(df0, dxdt_feature_list=['volume'], bin_interval_list=[5])
#%%
# update plotting parameters
fs,fw,fh = update_plotting_params()
save_dir = Path(__file__).parent / 'figures' / 'volume_dip_figures'

#%%
# identify the tracks
MAIN_TRACK_ID = 75725
TIMEPOINT = 239

dftrack = df0[df0["track_id"] == MAIN_TRACK_ID]
colony = dftrack["colony"].values[0]
dfcolony = df0[df0["colony"] == colony]
dftime = dfcolony[dfcolony["index_sequence"] == TIMEPOINT]
df_after_transition_only = dfcolony[dfcolony['index_sequence'] > dfcolony['frame_transition']] # only include after growth

# find all neighbors that are immediate neighbors (and have passed the transition point)
immediate_neighbor_track_ids = find_immediate_neighbors(df_after_transition_only,MAIN_TRACK_ID,TIMEPOINT)

# compute distances between nuclei and sort them by distance to main track
dist = compute_distances_between_nuclei(dftime,MAIN_TRACK_ID)
sorted_dist,sorted_cell_ids,sorted_track_ids,sorted_index = get_ordered_list_of_nuclei_by_distance_to_track(dist,dftime)

# combine the main track with the immediate neighbors into a list
track_id_list = list(immediate_neighbor_track_ids) + [MAIN_TRACK_ID]

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
