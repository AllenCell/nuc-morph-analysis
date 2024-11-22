# %%
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, load_data, filter_data
from nuc_morph_analysis.analyses.lineage.get_features import (
    lineage_trees,
    lineage_pairs,
    convert_units,
)
from nuc_morph_analysis.analyses.lineage.plot import combined_correlation_values, single_generation
from nuc_morph_analysis.analyses.lineage.get_features.between_pairs import (
    get_symmetric_and_asymmetric_sister_pairs,
    get_single_feature_for_sym_or_asym_tracks,
)
from nuc_morph_analysis.analyses.lineage.plot import (
    single_generation,
    mother_and_daughter_pairs,
    sister_symmetry,
    lineage_schematic,
)
from nuc_morph_analysis.lib.visualization.example_tracks import EXAMPLE_TRACKS
from nuc_morph_analysis.analyses.lineage.plot import size_distributions
from nuc_morph_analysis.analyses.volume_variation import plot_features

# %% load data
df = global_dataset_filtering.load_dataset_with_features()
track_level_feature_df_all = filter_data.track_level_features(df)
df_with_lineage_annotations = df[df["colony"].isin(["small", "medium"])]
df_full = filter_data.all_timepoints_full_tracks(df_with_lineage_annotations)
track_level_feature_df = filter_data.track_level_features(df_with_lineage_annotations)
interval = load_data.get_dataset_time_interval_in_min("all_baseline")
pixel_size = load_data.get_dataset_pixel_size("all_baseline")

### MAIN FIGURES ###

figdir = "lineage/figures/main/"
# %%
size_distributions.plot_size_distributions(
    track_level_feature_df_all, feature_list=["volume_at_B", "volume_at_C"], figdir=figdir
)

# %% Lineage tree visual
lineage_schematic.generate_lineage_schematic(
    df[df["colony"] == "medium"], EXAMPLE_TRACKS["lineage_schematic"], [0, 36, 240, 418], figdir
)

# %% Example mother daughter trajectory and compare mothers volume at C to sum of the daughters volume at B
df_mother_and_daughter_pairs = mother_and_daughter_pairs.get_mother_and_daughter_pairs(
    df_full, pixel_size
)
mother_and_daughter_pairs.scatterplot_sum_sisters_volume_at_C_vs_mother_vol_at_C(
    df_mother_and_daughter_pairs, figdir, True
)

# %% Get lineage trees and related/unrelated pairs
family_trees = lineage_trees.get_trees(df_with_lineage_annotations)
related_pairs = lineage_pairs.get_related_pairs_dataset(track_level_feature_df, family_trees)
unrelated_sister_control_pairs = lineage_pairs.get_sister_control_dataframe(track_level_feature_df)
unrelated_md_control_pairs = lineage_pairs.get_mother_daughter_control_dataframe(
    track_level_feature_df
)

# %% Correlation examples
single_generation.plot_corr(
    related_pairs, relationship_axis="width", feature="volume_at_B", figdir=figdir
)
single_generation.plot_corr(
    unrelated_sister_control_pairs,
    relationship_axis="width",
    feature="volume_at_B",
    control=True,
    col_threshold=[("distance", 60)],
    figdir="lineage/figures/supp/",
)

feature = "duration_BC_hr"
single_generation.plot_corr(
    related_pairs, relationship_axis="depth", feature=feature, figdir=figdir
)
single_generation.plot_corr(
    related_pairs, relationship_axis="width", feature=feature, figdir=figdir
)
single_generation.plot_corr(
    unrelated_sister_control_pairs,
    relationship_axis="width",
    feature=feature,
    control=True,
    col_threshold=[("distance", 60), ("difference_volume_at_B", 80)],
    figdir=figdir,
)
single_generation.plot_corr(
    unrelated_md_control_pairs,
    relationship_axis="depth",
    feature=feature,
    control=True,
    col_threshold=[("distance", 60), ("difference_half_vol_at_C_and_B", 60)],
    figdir=figdir,
)

# %% Summary correlation plot
feature_list = [
    "duration_BC_hr",
    "volume_at_B",
    "volume_at_C",
    "delta_volume_BC",
    "late_growth_rate_by_endpoints",
    "volume_fold_change_BC",
    "tscale_linearityfit_volume",
]
combined_correlation_values.plot(
    related_pairs,
    unrelated_sister_control_pairs,
    unrelated_md_control_pairs,
    feature_list=feature_list,
    sister_col_threshold=[("distance", 60), ("difference_volume_at_B", 80)],
    mother_daughter_col_threshold=[("distance", 60), ("difference_half_vol_at_C_and_B", 60)],
    figdir=figdir,
    legend=False,
)

# %% Density distribution plots for symmetric vs asymmetric dividers
track_level_feature_df = convert_units.convert_duration_to_hours(track_level_feature_df, interval)
df_asym_and_sym_pairs = get_symmetric_and_asymmetric_sister_pairs(track_level_feature_df)
df_symmetric = get_single_feature_for_sym_or_asym_tracks(
    track_level_feature_df, df_asym_and_sym_pairs, "symmetric"
)
df_asymmetric = get_single_feature_for_sym_or_asym_tracks(
    track_level_feature_df, df_asym_and_sym_pairs, "asymmetric"
)

sister_symmetry.feature_density(
    track_level_feature_df, df_symmetric, df_asymmetric, "volume_at_B", figdir=figdir
)

### SUPPLEMENT ###
# %%
figdir = "lineage/figures/supp1/"

mother_and_daughter_pairs.histogram_percent_of_volume(df_mother_and_daughter_pairs, figdir)
sister_symmetry.visualize_volume_difference_for_division_symmetry_thresholds(
    df_asym_and_sym_pairs, figdir
)

# %% Key feature plots colored by symmetric vs asymmetric
for feature in ["volume_fold_change_BC", "duration_BC_hr"]:
    sister_symmetry.feature_density(
        track_level_feature_df, df_symmetric, df_asymmetric, feature, figdir
    )
    plot_features.scatter_plot(
        df_symmetric, "all_baseline", "volume_at_B", feature, color_map="#7600bf", figdir=figdir
    )
    plot_features.scatter_plot(
        df_asymmetric, "all_baseline", "volume_at_B", feature, color_map="tab:green", figdir=figdir
    )

# %%
figdir = "lineage/figures/supp2/"

# %% Histograms
single_generation.plot_distance_between_sister_pairs(
    related_pairs, figdir="lineage/figures/supp/lineage_feature_correlations_worklfow/"
)
single_generation.plot_volume_difference_between_sister_pairs(
    related_pairs, figdir="lineage/figures/supp/lineage_feature_correlations_worklfow/"
)
single_generation.plot_transition_time_difference_between_sister_pairs(
    related_pairs, figdir="lineage/figures/supp/lineage_feature_correlations_worklfow/"
)

##% Correlation plots
feature_list = [
    "volume_at_B",
    "volume_at_C",
    "delta_volume_BC",
    "late_growth_rate_by_endpoints",
    "volume_fold_change_BC",
    "tscale_linearityfit_volume",
]
for feature in feature_list:
    single_generation.plot_corr(
        related_pairs, relationship_axis="depth", feature=feature, figdir=figdir
    )
    single_generation.plot_corr(
        related_pairs, relationship_axis="width", feature=feature, figdir=figdir
    )
    single_generation.plot_corr(
        unrelated_sister_control_pairs,
        relationship_axis="width",
        feature=feature,
        control=True,
        col_threshold=[("distance", 60), ("difference_volume_at_B", 80)],
        figdir=figdir,
    )
    single_generation.plot_corr(
        unrelated_md_control_pairs,
        relationship_axis="depth",
        feature=feature,
        control=True,
        col_threshold=[("distance", 60), ("difference_half_vol_at_C_and_B", 60)],
        figdir=figdir,
    )

# %%
