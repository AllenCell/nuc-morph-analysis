# %%
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, load_data, filter_data
from nuc_morph_analysis.analyses.cell_health import cell_health_plots, cell_health_functions
import pandas as pd
import matplotlib.pyplot as plt
from nuc_morph_analysis.analyses.volume_variation import plot_features

interval = load_data.get_dataset_time_interval_in_min("all_baseline")
pixel_size = load_data.get_dataset_pixel_size("all_baseline")
# %%
df = global_dataset_filtering.load_dataset_with_features()
df_full = filter_data.all_timepoints_full_tracks(df)

#%%
dff = cell_health_functions.subset_main_dataset(df, df_full)
#%%
df_all_filtered = filter_data.all_timepoints_minimal_filtering(dff)
df_full_filtered = filter_data.all_timepoints_full_tracks(dff)
df_full_filtered = df_full_filtered.sort_values('index_sequence')
df_track_level_features_filtered = filter_data.track_level_features(dff)
df_track_level_features_filtered = plot_features.convert_duration_to_hours(df_track_level_features_filtered, interval)

#%%
figdir = 'cell_health/figures/'
figdir2 = 'cell_health/figures/first_40_hours/'

# %% 
cell_health_plots.plot_event_histogram(df, 'cell_death', figdir)
cell_health_plots.plot_event_histogram(df, 'cell_division', figdir)
    
# %%
vast_path = '/allen/aics/assay-dev/users/Chantelle/manual_analysis/NucMorph/hipsc/feeding_control/'
df1 = pd.read_csv(f'{vast_path}cell_death_scene_1.csv')
df4 = pd.read_csv(f'{vast_path}cell_death_scene_4.csv')
df7 = pd.read_csv(f'{vast_path}cell_death_scene_7.csv')
cell_health_plots.cell_death_feeding_controls([df1, df4, df7], 
                                              ['feeding_control_baseline', 
                                               'feeding_control_refeed', 
                                               'feeding_control_starved'], interval)


# %% Figure 2 height over time and colony time
from nuc_morph_analysis.analyses.height import plot
plot.height_colony_time_alignment(df_all_filtered, pixel_size, interval, time_axis="colony_time", show_legend=True, figdir=figdir2)

# %% 
import nuc_morph_analysis.analyses.volume.synchronized_tracks as plot_tracks
for time in ["sync_time_Ff", "normalized_time"]:
    print(f'number of full tracks under 40 hours: {df_full_filtered.track_id.nunique()}')
    plot_tracks.plot_all_tracks_synchronized(df_full_filtered, figdir2, "volume", time)

for feature in ["volume", "volume_fold_change_fromB"]:
    plot_tracks.plot_mean_track_with_ci(df_full_filtered, figdir2, feature, by_colony_flag=False)
    plot_tracks.plot_mean_track_with_ci(df_full_filtered, figdir2, feature, by_colony_flag=True)

plot_tracks.bc_fold_change_distribution(df_track_level_features_filtered, figdir2, "volume_fold_change_BC", by_colony_flag=True)
# %%
from nuc_morph_analysis.analyses.volume_variation import plot_features
plot_features.plot_traj(df_full_filtered, "volume", figdir2, interval, colony="all_baseline", 
                        color_map="colony_gradient", opacity=0.35, highlight_samples=True)


# %%
for feature in ["volume_fold_change_BC", "colony_time_at_B"]:
    plt.close()
    plot_features.scatter_plot(
        df_track_level_features_filtered,
        "all_baseline",
        feature,
        f"tscale_linearityfit_volume",
        color_map="#808080",
        figdir=figdir2,
    )

# %% Calculate average growth rate over beginning and end subsets of tracks
import numpy as np
from nuc_morph_analysis.analyses.volume_variation import plot_features
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, load_data, filter_data
from scipy import stats as spstats

feature_list = [
    "volume_fold_change_BC",
    "delta_volume_BC",
    "late_growth_rate_by_endpoints",
    "duration_BC_hr",
]
for feature in feature_list:
    plot_features.scatter_plot(
        df_track_level_features_filtered,
        "all_baseline",
        "volume_at_B",
        feature,
        color_map="colony_time_at_B",
        color_column="colony_time_at_B",
        colorbar=False,
        figdir=figdir2,
    )
# %%
