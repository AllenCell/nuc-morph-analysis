# %%
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, filter_data
from datetime import datetime
from pathlib import Path


# %%
def check_columns(df1, df2):
    """
    Check if the columns in two dataframes are the same.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe
    df2 : pd.DataFrame
        Second dataframe
    """
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    if cols1 != cols2:
        raise ValueError(
            f"Columns in dataframes are not the same: {cols1.symmetric_difference(cols2)}"
        )


def save_dataset_for_quilt(df, dataset_name, destdir=None):
    """
    Save a dataset to a csv file in the data directory.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save
    dataset_name : str
        Name of the dataset
    destdir : str
        Destination

    Returns
    -------
    None
    """
    date = datetime.today().strftime("%Y-%m-%d")
    if destdir is None:
        destdir = Path(__file__).parent.parent.parent.parent
        datadir = destdir / "data"
        datadir.mkdir(exist_ok=True, parents=True)
    df.to_csv(f"{destdir}/data/{dataset_name}_{date}.csv", index=False)
    print(f"file saved to {destdir}/data/")


# %% Load baseline colonies
df_all_baseline = global_dataset_filtering.load_dataset_with_features()
# print(*[col for col in df_all_baseline.columns if "NUC_sh" not in col], sep="\n")
#%% old and new col lists
previous_cols = [
    "CellId",
    "label_img",
    "track_id",
    "colony",
    "index_sequence",
    "roi",
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "volume",
    "height",
    "mesh_vol",
    "mesh_sa",
    "SA_vol_ratio",
    "transform_params",
    "NUC_",
    "length",
    "width",
    "xz_aspect",
    "xy_aspect",
    "zy_aspect",
    "fov_edge",
    "predicted_formation",
    "predicted_breakdown",
    "Ff",
    "Fb",
    "after_breakdown_outlier",
    "before_formation_outlier",
    "is_after_breakdown_before_formation_outlier",
    "termination",
    "entering_mitosis",
    "exiting_mitosis",
    "entering_or_exiting_division",
    "neighbors",
    "neigh_distance",
    "density",
    "is_tp_outlier",
    "track_length",
    "is_outlier_by_short_track",
    "is_outlier_curated_by_id",
    "is_growth_outlier",
    "is_outlier_track",
    "is_outlier",
    "parent_id",
    "family_id",
    "distance_from_centroid",
    "colony_depth",
    "normalized_colony_depth",
    "normalized_distance_from_centroid",
    "colony_edge_in_fov",
    "colony_time",
    "non_interphase_volume",
    "non_interphase_mesh_sa",
    "non_interphase_SA_vol_ratio",
    "non_interphase_size_shape",
    "dxdt_48_volume",
    "neighbor_avg_volume_90um",
    "neighbor_avg_dxdt_48_volume_90um",
    "neighbor_avg_volume_whole_colony",
    "neighbor_avg_dxdt_48_volume_whole_colony",
    "normalized_time",
    "frame_transition",
    "sync_time_Ff",
    "volume_at_A",
    "location_x_at_A",
    "location_y_at_A",
    "time_at_A",
    "colony_time_at_A",
    "volume_at_B",
    "location_x_at_B",
    "location_y_at_B",
    "time_at_B",
    "colony_time_at_B",
    "volume_at_C",
    "location_x_at_C",
    "location_y_at_C",
    "time_at_C",
    "colony_time_at_C",
    "duration_AB",
    "duration_BC",
    "duration_AC",
    "delta_volume_BC",
    "volume_fold_change_BC",
    "SA_at_B",
    "SA_at_C",
    "delta_SA_BC",
    "SA_fold_change_BC",
    "volume_fold_change_fromB",
    "SA_fold_change_fromB",
    "growth_rate_AB",
    "late_growth_rate_by_endpoints",
    "tscale_linearityfit_volume",
    "atB_linearityfit_volume",
    "rate_linearityfit_volume",
    "RMSE_linearityfit_volume",
    "is_full_track",
    "exploratory_dataset",
    "baseline_colonies_dataset",
    "full_interphase_dataset",
    "lineage_annotated_dataset"
]
columns_list = [col for col in df_all_baseline.columns if "NUC_sh" not in col]

#%% Get differences
previous_not_in_current = [col for col in previous_cols if col not in columns_list]
current_not_in_previous = [col for col in columns_list if col not in previous_cols]
print("Columns in previous but not in current:")
print(previous_not_in_current)
print("\nColumns in current but not in previous:")
print(current_not_in_previous)

#%% 139 NEW COLUMNS!!
new_cols = current_not_in_previous
keep_list = [
    #needed to calc linear reg model feats
    "has_mitotic_neighbor",
    "has_dying_neighbor",
    "sum_has_dying_neighbor",
    "sum_has_mitotic_neighbor",
    "neighbor_avg_lrm_volume_90um",
    "neighbor_avg_lrm_height_90um",
    "neighbor_avg_lrm_xy_aspect_90um",
    "neighbor_avg_lrm_mesh_sa_90um",
    "neighbor_avg_lrm_2d_area_nuc_cell_ratio_90um",
    "neighbor_avg_lrm_dxdt_48_volume_90um",        
    
    #used in lrm
    "sisters_volume_at_B",
    "sisters_duration_BC",
    "sisters_delta_volume_BC",
    "height_at_B",
    "xy_aspect_at_B",
    "SA_vol_ratio_at_B",
    "neighbor_avg_lrm_volume_90um_at_B",
    "neighbor_avg_lrm_height_90um_at_B",
    "neighbor_avg_lrm_xy_aspect_90um_at_B",
    "neighbor_avg_lrm_mesh_sa_90um_at_B",
    "neighbor_avg_lrm_2d_area_nuc_cell_ratio_90um_at_B",
    "early_neighbor_avg_dxdt_48_volume_90um",
    "mean_neighbor_avg_dxdt_48_volume_90um",
    "mean_neighbor_avg_lrm_volume_90um",
    "mean_neighbor_avg_lrm_height_90um",
    "mean_neighbor_avg_lrm_xy_aspect_90um",
    "mean_neighbor_avg_lrm_mesh_sa_90um",
    "mean_neighbor_avg_lrm_2d_area_nuc_cell_ratio_90um", 
    "normalized_sum_has_mitotic_neighbor",
    "normalized_sum_has_dying_neighbor",]

new_cols = [col for col in new_cols if col not in keep_list]

print(len(keep_list))    
print(len(new_cols))
#%%
for col in new_cols:
    print(col)

#%% Suggest we remove      
drop_list = [
    'level_0', # there is code in global datasets filtering and lebeling neighbors helper that is checking for this and there is now warning printed??
    'index', # cell id should be the index column.. why is this here?
    
    # Loads of 2d columns are unused... work with chris to identify which ones we actually need. 
    ]


# %% Filter baseline colonies
df_baseline = filter_data.all_timepoints_minimal_filtering(df_all_baseline)
df_full_baseline = filter_data.all_timepoints_full_tracks(df_baseline)
df_lineage = df_full_baseline[df_full_baseline["colony"].isin(["small", "medium"])]

# %% ignore baseline analysis dataset columns in comparison
df_all_baseline = global_dataset_filtering.remove_columns(
    df_all_baseline,
    [
        "exploratory_dataset",
        "baseline_colonies_dataset",
        "full_interphase_dataset",
        "lineage_annotated_dataset",
    ],
)

# %% Load feeding control
df_all_feeding_control = global_dataset_filtering.load_dataset_with_features("all_feeding_control")
df_all_feeding_control = global_dataset_filtering.remove_columns(
    df_all_feeding_control, ["track_match_issue", "track_matched"]
)
check_columns(df_all_baseline, df_all_feeding_control)
df_full_feeding_control = filter_data.all_timepoints_full_tracks(df_all_feeding_control)

# %% Load inhibitor dataset
df_all_inhibitor = global_dataset_filtering.load_dataset_with_features("all_drug_perturbation")
check_columns(df_all_baseline, df_all_inhibitor)
df_all_inhibitor = filter_data.all_timepoints_minimal_filtering(df_all_inhibitor)

aphidicolin_scenes = [
    "drug_perturbation_1_scene0",
    "drug_perturbation_1_scene2",
    "drug_perturbation_1_scene4",
    "drug_perturbation_4_scene2",
    "drug_perturbation_4_scene4",
]
df_aphidicolin = df_all_inhibitor[df_all_inhibitor["colony"].isin(aphidicolin_scenes)]

importazole_scenes = ["drug_perturbation_2_scene0", "drug_perturbation_2_scene6"]
df_importazole = df_all_inhibitor[df_all_inhibitor["colony"].isin(importazole_scenes)]

# %%
save_dataset_for_quilt(df_all_baseline, "baseline_colonies_unfiltered_feature_dataset")
save_dataset_for_quilt(df_baseline, "baseline_colonies_analysis_dataset")
save_dataset_for_quilt(df_full_baseline, "full-interphase_dataset")
save_dataset_for_quilt(df_lineage, "lineage-annotated_analysis_dataset")
save_dataset_for_quilt(df_full_feeding_control, "feeding_control_analysis_dataset")
save_dataset_for_quilt(df_aphidicolin, "dna_replication_inhibitor_analysis_dataset")
save_dataset_for_quilt(df_importazole, "nuclear_import_inhibitor_analysis_dataset")

