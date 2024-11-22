# %%
import numpy as np
from nuc_morph_analysis.analyses.volume_variation import plot_features
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, load_data, filter_data
from scipy import stats as spstats


# %%
# load data
df = global_dataset_filtering.load_dataset_with_features()
df_full = filter_data.all_timepoints_full_tracks(df)
track_level_feature_df = filter_data.track_level_features(df_full)
interval = load_data.get_dataset_time_interval_in_min("all_baseline")
track_level_feature_df = plot_features.convert_duration_to_hours(track_level_feature_df, interval)

# %% Set figure save directory and feature list to plot
figdir = "volume_variation/figures/compensation/"

# %% MAIN FIGURE PANELS

# %%
# Make scatter plots for each feature
feature_list = [
    "volume_fold_change_BC",
    "delta_volume_BC",
    "late_growth_rate_by_endpoints",
    "duration_BC_hr",
]
for feature in feature_list:

    # add two-fold refernce line for volume fold change
    value_ref = 2.0 if feature == "volume_fold_change_BC" else np.nan

    # plot vs volume at B in gray for all colonies pooled
    plot_features.scatter_plot(
        track_level_feature_df,
        "all_baseline",
        "volume_at_B",
        feature,
        color_map="#808080",
        figdir=figdir,
        value_ref_line=value_ref,
    )

    # plot vs colony time in colony color
    # make with rolling window mean line per colony
    plot_features.scatter_plot(
        track_level_feature_df,
        "all_baseline",
        "colony_time_at_B",
        feature,
        color_map="colony",
        figdir=figdir,
        value_ref_line=value_ref,
        plot_rolling_window_mean_flag=True,
        line_per_colony_flag=True,
    )

    # plot vs volume at B in colony time at B color
    # make with and without colorbar for "all"
    colorbar_flag_list = [True, False]
    for colorbar_flag in colorbar_flag_list:
        plot_features.scatter_plot(
            track_level_feature_df,
            "all_baseline",
            "volume_at_B",
            feature,
            color_map="colony_time_at_B",
            color_column="colony_time_at_B",
            colorbar=colorbar_flag,
            figdir=figdir,
            value_ref_line=value_ref,
        )

# %%
# Add volume at B plot vs colony time in colony color

# make with rolling window mean line per colony
plot_features.scatter_plot(
    track_level_feature_df,
    "all_baseline",
    "colony_time_at_B",
    "volume_at_B",
    color_map="colony",
    figdir=figdir,
    value_ref_line=value_ref,
    plot_rolling_window_mean_flag=True,
    line_per_colony_flag=True,
)

# %%
# Get all correlations to add to above figures
feature_base = "colony_time_at_B"
for feature in [
    "volume_fold_change_BC",
    "late_growth_rate_by_endpoints",
    "duration_BC_hr",
    "volume_at_B",
    "volume_at_C",
    "delta_volume_BC",
]:
    print(f"Correlation between {feature_base} and {feature}")
    for colony in ["all_baseline", "small", "medium", "large"]:
        if colony == "all_baseline":
            df_d = track_level_feature_df
        else:
            df_d = track_level_feature_df[track_level_feature_df["colony"] == colony]
        pearson, p_pvalue = spstats.pearsonr(df_d[feature_base], df_d[feature])
        print(f"{colony}: Pearson: {pearson:.2f}, p-value: {p_pvalue}")

feature_base = "volume_at_B"
for feature in [
    "volume_fold_change_BC",
    "late_growth_rate_by_endpoints",
    "duration_BC_hr",
    "volume_at_C",
    "delta_volume_BC",
]:
    print(f"Correlation between {feature_base} and {feature}")
    for colony in ["all_baseline", "small", "medium", "large"]:
        if colony == "all_baseline":
            df_d = track_level_feature_df
        else:
            df_d = track_level_feature_df[track_level_feature_df["colony"] == colony]
        pearson, p_pvalue = spstats.pearsonr(df_d[feature_base], df_d[feature])
    print(f"{colony}: Pearson: {pearson:.2f}, p-value: {p_pvalue}")


# SUPPLEMENTAL FIGURE PANELS

# %%
# plot fold change vs features in gray for all colonies pooled
feature_list = [
    "late_growth_rate_by_endpoints",
    "duration_BC_hr",
    "delta_volume_BC",
]
for feature in feature_list:
    plot_features.scatter_plot(
        track_level_feature_df,
        "all_baseline",
        "volume_fold_change_BC",
        feature,
        color_map="#808080",
        figdir=figdir,
        value_ref_line=2.0,
    )

# %%
# plot growth rate vs volume at B in colony time at B color
# make with and without colorbar for "all" otherwise include colorbar
for colony in ["small", "medium", "large"]:
    for colorbar_flag in [True, False]:
        if feature != "volume_at_B":
            plot_features.scatter_plot(
                track_level_feature_df,
                colony,
                "volume_at_B",
                "late_growth_rate_by_endpoints",
                color_map="colony_time_at_B",
                color_column="colony_time_at_B",
                colorbar=colorbar_flag,
                figdir=figdir,
            )

# %%
# plot added volume vs volume at B in colony time at B color
# make with and without colorbar for "all" otherwise include colorbar
for feature in ["volume_fold_change_BC", "volume_at_C"]:
    for colorbar_flag in [True, False]:
        plot_features.scatter_plot(
            track_level_feature_df,
            "all_baseline",
            "volume_at_B",
            feature,
            color_map="colony_time_at_B",
            color_column="colony_time_at_B",
            colorbar=colorbar_flag,
            figdir=figdir,
        )
