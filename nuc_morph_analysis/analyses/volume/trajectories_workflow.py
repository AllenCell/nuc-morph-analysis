# %%
from pathlib import Path
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, filter_data
import nuc_morph_analysis.analyses.volume.synchronized_tracks as plot_tracks
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.example_tracks import EXAMPLE_TRACKS
import matplotlib.pyplot as plt
import numpy as np

# %%
# set up plot parameters and figure saving directory
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 18
plt.rcParams["figure.figsize"] = [7, 5]
figdir = "volume/figures/trajectories"

# %% load data
df = global_dataset_filtering.load_dataset_with_features()
df_full = filter_data.all_timepoints_full_tracks(df)
df_track_level_features = filter_data.track_level_features(df_full)

# %%
# Plot all tracks all datasets synchronized in real or normalized time
for time in ["sync_time_Ff", "normalized_time"]:
    plot_tracks.plot_all_tracks_synchronized(df_full, figdir, "volume", time)

# %% MAIN FIGURE PANELS

# %%
# Plot single sample track from same nucleus as in formation/breakdown figure

yscale, ylabel, yunits, _ = get_plot_labels_for_metric("volume")
xscale, xlabel, xunits, _ = get_plot_labels_for_metric("index_sequence")

sample_track_id = EXAMPLE_TRACKS["transition_point_workflow"]
df_sample = df.loc[df["track_id"] == sample_track_id]
df_sample = df_sample.sort_values("index_sequence")
plt.plot(df_sample["index_sequence"] * xscale, df_sample["volume"].values * yscale, color="k")
plt.xlabel(f"{xlabel} {xunits}")
plt.ylabel(f"{ylabel} {yunits}")
plt.vlines(df_sample["Ff"].values[0] * xscale, 300, 1200, color="k", linestyle=":")
plt.vlines(df_sample["frame_transition"].values[0] * xscale, 300, 1200, color="k", linestyle=":")
plt.vlines(df_sample["Fb"].values[0] * xscale, 300, 1200, color="k", linestyle=":")
plt.xlim(0, 13)
plt.ylim(300, 1200)
plt.tight_layout()
save_and_show_plot(f"{figdir}/sample_track_{sample_track_id}")

# %%
# Plot mean fold change over normalized time track
plot_tracks.plot_mean_track_with_ci(
    df_full, figdir, "volume_fold_change_fromB", by_colony_flag=False
)

# %%
# Histograms of fold changes
scale, label, units, _ = get_plot_labels_for_metric("volume_fold_change_BC")
cv = np.round(
    np.nanstd(df_track_level_features["volume_fold_change_BC"])
    / np.nanmean(df_track_level_features[f"volume_fold_change_BC"]),
    3,
)
plt.hist(
    df_track_level_features["volume_fold_change_BC"], bins=20, color="#B19CD8", label=f"CV: {cv}"
)
plt.axvline(2, color="k", linestyle=":")

plt.legend()
plt.tight_layout()
plt.xlabel(f"{label} {units}")
plt.ylabel("Counts")
save_and_show_plot(f"{figdir}/hist_volume_fold_change_BC")

# %% SUPPLEMENTAL FIGURE PANELS

# %%
# Plot mean of all volume and fold-change tracks with 5th to 95th percentile shaded
for feature in ["volume", "volume_fold_change_fromB"]:
    plot_tracks.plot_mean_track_with_ci(df_full, figdir, feature, by_colony_flag=True)
    plot_tracks.plot_mean_track_with_ci(df_full, figdir, feature, by_colony_flag=True)

# %%
# Plot distribution of B to C fold changes for volume and surface area
plot_tracks.bc_fold_change_distribution(
    df_track_level_features, figdir, "volume_fold_change_BC", by_colony_flag=True
)
