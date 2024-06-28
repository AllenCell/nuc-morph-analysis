# %%
from nuc_morph_analysis.analyses.volume_variation import plot_features
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, load_data, filter_data
from nuc_morph_analysis.analyses.lineage.get_features.between_pairs import (
    get_symmetric_and_asymmetric_sister_pairs,
    get_single_feature_for_sym_or_asym_tracks,
)
from nuc_morph_analysis.analyses.lineage.plot import sister_symmetry
from nuc_morph_analysis.analyses.lineage.get_features import lineage_trees, lineage_pairs
from nuc_morph_analysis.analyses.lineage.plot import single_generation

# %% Load Data
df = global_dataset_filtering.load_dataset_with_features()
df_with_lineage_annotations = df[df["colony"].isin(["small", "medium"])]
# %%
df_full = filter_data.all_timepoints_full_tracks(df_with_lineage_annotations)
track_level_feature_df = filter_data.track_level_features(df_with_lineage_annotations)
interval = load_data.get_dataset_time_interval_in_min("all_baseline")
pixel_size = load_data.get_dataset_pixel_size("all_baseline")
# %%
track_level_feature_df = plot_features.convert_duration_to_hours(track_level_feature_df, interval)

# %% Get all releated pairs
family_trees = lineage_trees.get_trees(df_with_lineage_annotations)
related_pairs = lineage_pairs.get_related_pairs_dataset(track_level_feature_df, family_trees)
df_pairs = get_symmetric_and_asymmetric_sister_pairs(track_level_feature_df)
# %%
figdir = "lineage/figures/supp/lineage_asymmetric_division_workflow/"
# %%
sister_symmetry.visualize_volume_difference_for_division_symmetry_thresholds(df_pairs, figdir)
# %%
sister_symmetry.timing_of_asymmetric_and_symmetric_divisions(df_pairs, figdir)
sister_symmetry.plot_mother_volume_distribution(track_level_feature_df, df_pairs, figdir)
# %% Visualize symmetric and asymmetric trajectories
sister_symmetry.plot_sister_trajectories(
    df_full,
    df_pairs[df_pairs.symmetric],
    interval,
    pixel_size,
    figdir=f"{figdir}symmetric_sisters/",
)
sister_symmetry.plot_sister_trajectories(
    df_full,
    df_pairs[df_pairs.asymmetric],
    interval,
    pixel_size,
    figdir=f"{figdir}asymmetric_sisters/",
)

# %% Get track level features for symmetric and asymmetric dividers
df_same = get_single_feature_for_sym_or_asym_tracks(track_level_feature_df, df_pairs, "symmetric")
df_diff = get_single_feature_for_sym_or_asym_tracks(track_level_feature_df, df_pairs, "asymmetric")

# %% Density distribution plots for symmetric vs asymmetric
features = [
    "duration_BC_hr",
    "delta_volume_BC",
    "volume_fold_change_BC",
    "late_growth_rate_by_endpoints",
    "volume_at_B",
    "volume_at_C",
]
for feature in features:
    sister_symmetry.feature_density(track_level_feature_df, df_same, df_diff, feature, figdir)

# %% Key feature plots colored by symmetric vs asymmetric
features = [
    "duration_BC_hr",
    "delta_volume_BC",
    "volume_fold_change_BC",
    "volume_at_C",
    "late_growth_rate_by_endpoints",
]
for feature in features:
    plot_features.scatter_plot(
        df_same, "all_baseline", "volume_at_B", feature, color_map="#7600bf", figdir=figdir
    )
    plot_features.scatter_plot(
        df_diff, "all_baseline", "volume_at_B", feature, color_map="tab:green", figdir=figdir
    )

# %% Color by big and little sister
for feature in features:
    plot_features.scatter_plot(
        df_same, "all_baseline", "volume_at_B", feature, color_map="small_sister", figdir=figdir
    )
    plot_features.scatter_plot(
        df_diff, "all_baseline", "volume_at_B", feature, color_map="small_sister", figdir=figdir
    )
plot_features.scatter_plot(
    df_same,
    "all_baseline",
    "duration_BC_hr",
    "volume_fold_change_BC",
    color_map="small_sister",
    figdir=figdir,
)
plot_features.scatter_plot(
    df_diff,
    "all_baseline",
    "duration_BC_hr",
    "volume_fold_change_BC",
    color_map="small_sister",
    figdir=figdir,
)


# %% Compare the sum of sisters volume at B to the sum of their fold change
single_generation.scatter_avg_sister_pairs(
    related_pairs, "volume_at_B", "volume_fold_change_BC", figdir
)
single_generation.sum_feature_density(track_level_feature_df, related_pairs, "volume_at_B", figdir)
single_generation.sum_feature_density(
    track_level_feature_df, related_pairs, "volume_fold_change_BC", figdir
)
# %%
single_generation.difference_in_volume_at_B_vs_feature(
    related_pairs, "volume_fold_change_BC", figdir
)
single_generation.difference_in_volume_at_B_vs_feature(related_pairs, "duration_BC_hr", figdir)
# %%
