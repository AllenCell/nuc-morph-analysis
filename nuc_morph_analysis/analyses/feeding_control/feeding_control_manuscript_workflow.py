# %% Load data
from nuc_morph_analysis.lib.preprocessing import load_data, global_dataset_filtering, filter_data
from nuc_morph_analysis.analyses.volume_variation import plot_features

pixel_size = load_data.get_dataset_pixel_size("all_feeding_control")
interval = load_data.get_dataset_time_interval_in_min("all_feeding_control")

# %% Load feeding control datasets
df_feeding = global_dataset_filtering.load_dataset_with_features("all_feeding_control")
# %%
df_full_feeding = filter_data.all_timepoints_full_tracks(df_feeding)
track_level_feature_df_feeding = filter_data.track_level_features(df_feeding)

# %%
figdir = "feeding_control/figures"

# %% Supplemental figure: Feeding control
features = ["late_growth_rate_by_endpoints"]
for feature in features:
    for colony, dataframe in track_level_feature_df_feeding.groupby("colony"):
        dataframe = plot_features.convert_duration_to_hours(dataframe, interval)
        print(colony)
        plot_features.scatter_plot(
            dataframe,
            colony,
            "time_at_B",
            feature,
            color_map="colony",
            figdir="feeding_control/figures/",
            fitting=False,
            add_known_timepoints=False,
            plot_rolling_window_mean_flag=True,
            feeding_control=True,
        )
# %%
