# %%
from nuc_morph_analysis.analyses.lineage.plot import mother_and_daughter_pairs
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, load_data, filter_data
from nuc_morph_analysis.lib.visualization.example_tracks import EXAMPLE_TRACKS
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "Arial"

# %% load data
df = global_dataset_filtering.load_dataset_with_features()
df_with_lineage_annotations = df[df["colony"].isin(["small", "medium"])]
df_full = filter_data.all_timepoints_full_tracks(df_with_lineage_annotations)
track_level_feature_df = filter_data.track_level_features(df_with_lineage_annotations)
interval = load_data.get_dataset_time_interval_in_min("all_baseline")
pixel_size = load_data.get_dataset_pixel_size("all_baseline")

df_mother_and_daughter_pairs = mother_and_daughter_pairs.get_mother_and_daughter_pairs(
    df_full, pixel_size
)
# %%
figdir = "lineage/figures/supp/lineage_timing_at_B_workflow/"

# %% Example mother daughter tracjectories
PID = EXAMPLE_TRACKS["lineage_mother_volume_at_C"]
df_mother = df_mother_and_daughter_pairs[df_mother_and_daughter_pairs["pid"] == PID]
mother_and_daughter_pairs.plot_mother_and_daughter_trajectories(
    df_full, df_mother, interval, figdir
)

# %% Crossing vs transition scatter
mother_and_daughter_pairs.scatterplot_sum_sisters_volume_at_C_vs_mother_vol_at_C(
    df_mother_and_daughter_pairs, figdir, True
)
mother_and_daughter_pairs.scatterplot_crossing_vs_transition(
    df_mother_and_daughter_pairs, interval, figdir
)

# %% Crossing vs transition histograms
mother_and_daughter_pairs.histogram_difference_in_timing(
    df_mother_and_daughter_pairs, interval, figdir
)
mother_and_daughter_pairs.histogram_difference_in_volume(df_mother_and_daughter_pairs, figdir)
mother_and_daughter_pairs.histogram_percent_of_volume(df_mother_and_daughter_pairs, figdir)
mother_and_daughter_pairs.two_feature_density(
    df_mother_and_daughter_pairs, "mothers_volume_at_C", "sum_sisters_volume_at_B", figdir
)

# %% spike at C
mother_and_daughter_pairs.plot_volume_at_C_variation(df_full, 3, interval, figdir)
mother_and_daughter_pairs.plot_volume_at_C_variation(df_full, 2, interval, figdir)

# %% mother daughter trajectories
mother_and_daughter_pairs.plot_mother_and_daughter_trajectories(
    df_full, df_mother_and_daughter_pairs, interval, f"{figdir}/trajectories/", add_track_info=True
)

# %%
mother_and_daughter_pairs.histogram_difference_half_mothers_volume_at_C_and_daughter_at_B(
    df_mother_and_daughter_pairs, figdir
)
mother_and_daughter_pairs.difference_in_sister_transition(
    df_mother_and_daughter_pairs, interval, figdir
)
mother_and_daughter_pairs.difference_in_sister_transition(
    df_mother_and_daughter_pairs, interval, figdir, xlim=50
)
mother_and_daughter_pairs.difference_in_sister_transition_scatter(
    df_mother_and_daughter_pairs, interval, figdir
)
mother_and_daughter_pairs.diff_vol_diff_timing(df_mother_and_daughter_pairs, interval, figdir)
# %%
