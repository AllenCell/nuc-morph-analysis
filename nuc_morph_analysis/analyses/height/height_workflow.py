# %%
from nuc_morph_analysis.analyses.colony_context.colony_context_analysis import plot_radial_profile
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.lib.preprocessing import load_data, filter_data, global_dataset_filtering
from nuc_morph_analysis.lib.preprocessing.load_data import get_dataset_pixel_size
from nuc_morph_analysis.analyses.height import plot
from nuc_morph_analysis.analyses.height.plot_crowding import plot_density_schematic
from nuc_morph_analysis.analyses.height.toymodel import toymodel
from nuc_morph_analysis.analyses.height.centroid import (
    get_centroid,
    get_neighbor_centroids,
)
from nuc_morph_analysis.lib.preprocessing.load_data import get_dataset_pixel_size
from pathlib import Path
import matplotlib.pyplot as plt

# %% Create radial height profile plot

# set figure directory
figdir = Path(__file__).parent / "figures"
figdir.mkdir(exist_ok=True)

# load data
df_all = load_dataset_with_features("all_baseline", remove_growth_outliers=False)

# plot radial profile plot
plot_radial_profile(
    df_all,
    col_to_plot="height",
    distance_col="normalized_colony_depth",
    align_to_colony_time=True,
    save_dir=figdir,
    correlation_metric="spearman",
    filter_edge=True,
    bootstrap_count=100,
    save_format="pdf",
    weight_by_r2=False,
    save_intermediate_correlations_colony="medium",
    make_height_profile_lineplot=False,
)
plt.close("all")

# %% Create colony time alignment height plot

# load data
df_all = filter_data.all_timepoints_minimal_filtering(df_all)
track_level_feature_df = filter_data.track_level_features(df_all)
interval = load_data.get_dataset_time_interval_in_min("all_baseline")
pixel_size = load_data.get_dataset_pixel_size("all_baseline")

# Plot height over aligned colony time
plot.height_colony_time_alignment(
    df_all, pixel_size, interval, time_axis="colony_time", show_legend=True
)


# %% Plot density and schematic

# set colony and timeppoint for schematic
df = df_all[df_all.colony == "medium"].reset_index()
df_timepoint = df[df.index_sequence == 0]

# get centroid of nucleus closest to colony centroid
frame_centroids, track_centroid, df_nuc = get_centroid(df_timepoint)
# get centroids of neighbors
neighbor_centroids = get_neighbor_centroids(df_nuc, df_timepoint)

# plot density schematic
pix_size = get_dataset_pixel_size("medium")
plot_density_schematic(
    df_timepoint, track_centroid, neighbor_centroids, frame_centroids, pix_size, figdir
)

# plot colony-averaged density over aligned colony time
for use_old_density in [True, False]:
    plot.density_colony_time_alignment(df_all, pixel_size, interval, time_axis="colony_time",
                                   use_old_density=use_old_density)

# %%
# Run and plot toy model
toymodel()
