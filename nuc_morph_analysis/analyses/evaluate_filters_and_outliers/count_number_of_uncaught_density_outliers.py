# this file can be deleted
# code to determine how many cells additional cells are filtered if we remove the colony depth filter
#%%
from nuc_morph_analysis.lib.preprocessing import (
    load_data, add_features, filter_data, is_tp_outlier, labeling_neighbors_helper
)
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import (
    add_neighborhood_avg_features, add_neighborhood_avg_features_lrm,
    add_colony_position_columns, add_fov_touch_timepoint_for_colonies, add_change_over_time)

import numpy as np
remove_growth_outliers=True
dataset = 'all_baseline'
df = load_data.load_all_datasets(dataset)
df = add_features.add_perimeter_ratio(df)
pix_size = load_data.get_dataset_pixel_size(dataset)
thresh = load_data.get_length_threshold_in_tps(dataset)
interval = load_data.get_dataset_time_interval_in_min(dataset)

n_tracks_prefilter = df["track_id"].nunique()
print(f"{n_tracks_prefilter} tracks before any filtering")
if 'level_0' in df.columns:
    print('WARNING: level_0 column found in df, dropping')
df = filter_data.flag_missed_apoptotic(df)
df = is_tp_outlier.outlier_detection(df)
df = add_features.add_division_entry_and_exit_annotations(df)
df = filter_data.add_and_pool_outlier_flags(df, remove_growth_outliers=remove_growth_outliers)

# add labels for neighbors of mitotic and dying cells
df = labeling_neighbors_helper.label_nuclei_that_neighbor_current_mitotic_event(df)
df = labeling_neighbors_helper.label_nuclei_that_neighbor_current_death_event(df)

# add features
df = add_features.add_aspect_ratio(df)
df = add_features.add_family_id_all_datasets(df)
df = add_features.add_SA_vol_ratio(df)
df = add_colony_position_columns(df)
df = add_fov_touch_timepoint_for_colonies(df)
df = add_features.add_non_interphase_size_shape_flag(df)
df = add_change_over_time(df)
df = add_neighborhood_avg_features.run_script(df, num_workers=1)
df = add_neighborhood_avg_features_lrm.run_script(df, num_workers=1, 
                                            feature_list=["volume", "height", "xy_aspect", "mesh_sa", "2d_area_nuc_cell_ratio"],
                                            exclude_outliers=True)

df = add_features.add_perimeter_ratio(df)

#%%
df_all = df.copy()
df_all = filter_data.all_timepoints_minimal_filtering(df_all)
df_all = filter_data.remove_expected_pseudo_cell_artifacts(df_all,verbose=True)
log1 = df_all['2d_perimeter_nuc_cell_ratio'] < 0.4
log2 = df_all['2d_perimeter_pseudo_cell'] > 500
log3 = df_all['2d_area_nuc_cell_ratio'] < 0.2
log4 = df_all['colony_depth'] <= 3

log123 = (log1 | log2 | log3)
log1234 = (log1 | log2 | log3) & log4

# now count how many cells are filtered out by each filter and combinations
log12 = log1 | log2
log13 = log1 | log3
log23 = log2 | log3

loglist = [log1, log2, log3, log4, log12, log13, log23, log123, log1234]
namelist = ["log1", "log2", "log3", "log4", "log12", "log13", "log23", "log123", "log1234"]
for log,logname in zip(loglist,namelist):
    print(f"{logname}: filtered out {np.sum(log)} cells ({100*np.sum(log)/len(log):.1f})%")
    for colony in df_all.colony.unique():
        log_colony = log & (df_all.colony == colony)
        print(f"    {colony}: {np.sum(log_colony)} cells ({100*np.sum(log_colony)/len(log_colony):.1f}%)")

    # percentage_of_cells_filtered = 100*np.sum(compiled_log)/len(compiled_log)
    # print(f"filtered out {percentage_of_cells_filtered:.1f}% of cells due to artifact pseudo cell filters")

#%%
difflog123log1234 = log123 & ~log1234 
print(f"extra cells if we cut colony depth:")
print(f"     {np.sum(difflog123log1234)} cells ({100*np.sum(difflog123log1234)/len(difflog123log1234):.1f}%)")
# find unique track ids and timepoints
dflog = df_all[difflog123log1234]
dfg = dflog.groupby(['track_id','index_sequence']).agg('count')
dfg
# printout
# extra cells if we cut colony depth:
#      2 cells (0.0%)
# level_0	index	roi	scale_micron	centroid_z	centroid_y	centroid_x	label_img	seg_full_zstack_path	raw_full_zstack_path	...	neighbor_avg_lrm_xy_aspect_whole_colony	neighbor_avg_lrm_mesh_sa_whole_colony	neighbor_avg_lrm_2d_area_nuc_cell_ratio_whole_colony	neighbor_avg_lrm_dxdt_48_volume_whole_colony	neighbor_avg_lrm_dxdt_48_volume_per_V_whole_colony	neighbor_avg_lrm_neighbor_avg_dxdt_48_volume_90um_whole_colony	neighbor_avg_lrm_neighbor_avg_dxdt_48_volume_per_V_90um_whole_colony	neighbor_avg_lrm_neighbor_avg_dxdt_48_volume_whole_colony_whole_colony	neighbor_avg_lrm_neighbor_avg_dxdt_48_volume_per_V_whole_colony_whole_colony	bad_pseudo_cells_segmentation
# track_id	index_sequence																					
# 95619	411	1	1	1	1	1	1	1	1	1	1	...	1	1	1	1	1	1	1	1	1	1
# 96974	550	1	1	1	1	1	1	1	1	1	1	...	1	1	1	0	0	0	0	0	0	1
# 2 rows × 744 columns
#%%
difflog123log1234 = log123 & ~log13
print(f"extra cells if we cut colony depth:")
print(f"     {np.sum(difflog123log1234)} cells ({100*np.sum(difflog123log1234)/len(difflog123log1234):.1f}%)")
# find unique track ids and timepoints
dflog = df_all[difflog123log1234]
dfg = dflog.groupby(['track_id','index_sequence']).agg('count')
dfg