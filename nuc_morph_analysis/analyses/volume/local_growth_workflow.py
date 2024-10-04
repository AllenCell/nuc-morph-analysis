# %%
from pathlib import Path
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, load_data, filter_data
from nuc_morph_analysis.analyses.volume_variation import plot_features
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.analyses.volume.add_growth_features import plot_fit_parameter_distribution
from nuc_morph_analysis.lib.preprocessing.curve_fitting import powerfunc
from nuc_morph_analysis.lib.preprocessing.add_times import (
    add_binned_normalized_time_for_full_tracks,
)
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.example_tracks import EXAMPLE_TRACKS

import matplotlib
import matplotlib.pyplot as plt

# %%
# set up plot parameters and figure saving directory
matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({"font.size": 20})
plt.rcParams["figure.figsize"] = [5, 4]
figdir = f"volume/figures/local_growth/"

# %%
# load data
df = global_dataset_filtering.load_dataset_with_features()
df_full = filter_data.all_timepoints_full_tracks(df)
df_track_level_features = filter_data.track_level_features(df_full)
interval = load_data.get_dataset_time_interval_in_min("all_baseline")


# %% Plot volume trajectories for all tracks with examples of varying track shapes highlighted
plot_features.plot_traj(
    df_full,
    "volume",
    figdir,
    interval,
    colony="all_baseline",
    color_map="colony_gradient",
    opacity=0.35,
    highlight_samples=True,
)

# %% Plot sample tracks with extreme alphas
track_ids = [EXAMPLE_TRACKS["alpha_high_example"], EXAMPLE_TRACKS["alpha_low_example"]]
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
for track_id, color in zip(track_ids, ["#E7298A", "#E6AB02"]):

    # get track data
    df_track = df_full[df_full["track_id"] == track_id]
    transition = df_track["frame_transition"].min()
    fb = df_track["Fb"].values.min()
    df_track = df_track.sort_values("index_sequence")
    alpha = df_track["tscale_linearityfit_volume"].values[0]
    vB = df_track["atB_linearityfit_volume"].values[0]
    rate = df_track["rate_linearityfit_volume"].values[0]

    # plot full track
    yscale, ylabel, yunits, _ = get_plot_labels_for_metric("volume")
    x = df_track["index_sequence"].values * interval / 60
    x -= transition * interval / 60
    y = df_track["volume"].values * yscale
    ax.plot(x, y, label=f"Track {track_id}", c=color)

    # plot fitted growth during late growth phase
    df_track_trim = df_track[
        (df_track.index_sequence > transition) & (df_track.index_sequence <= fb)
    ]
    x_trim = df_track_trim["index_sequence"].values * interval / 60
    x_trim -= x_trim[0]
    y_trim = df_track_trim["volume"].values * yscale
    z = powerfunc(x_trim, rate, vB, alpha)
    plt.plot(x_trim, z, label=f"Alpha {alpha:.2f}", c=color, linestyle="--")

plt.legend(prop={"size": 12})
plt.ylim(200, 1050)
plt.xlim(-2, 15)
plt.ylabel(f"{ylabel} {yunits}")
plt.xlabel("Synchonrized nuclear growth time (hr)")
plt.tight_layout()
save_and_show_plot(f"{figdir}/sample_extreme_tracks_{track_ids[0]}_{track_ids[1]}", quiet=True)
plt.close()


# %% Plot distribution of tscale fitted values and root mean squared error
# both pooled for all baseline colonies and separated by colony
for feature in ["tscale_linearityfit_volume", "RMSE_linearityfit_volume"]:
    for by_colony_flag in [True, False]:
        if "RMSE" in feature and not by_colony_flag:
            allmodels_flag=True
        else:
            allmodels_flag=False
        plot_fit_parameter_distribution(
            df_track_level_features,
            figdir,
            feature,
            by_colony_flag=by_colony_flag,
            density_flag=True,
            allmodels_flag=allmodels_flag,
        )

# %% plot relationships of alpha to fold change and colony time for all data pooled
for feature in ["volume_fold_change_BC", "colony_time_at_B"]:
    plt.close()
    plot_features.scatter_plot(
        df_track_level_features,
        "all_baseline",
        feature,
        f"tscale_linearityfit_volume",
        color_map="#808080",
        figdir=figdir,
    )

# %% plot relationship of alpha to time within each movie
for colony in ["small", "medium", "large"]:
    plt.close()
    plot_features.scatter_plot(
        df_track_level_features,
        colony,
        "time_at_B",
        f"tscale_linearityfit_volume",
        color_map="colony",
        figdir=figdir,
    )

# %% Draw sample track with varying concavity for rolling window panel
plt.rcParams["figure.figsize"] = [9, 5]
track_id = EXAMPLE_TRACKS["alpha_high_example"]
df_track = df_full[df_full["track_id"] == track_id]
transition = df_track["frame_transition"].min()
fb = df_track["Fb"].values.min()
df_track = df_track.sort_values("index_sequence")
# get trimmed track times and volumes
yscale, ylabel, yunits, _ = get_plot_labels_for_metric("volume")
x = df_track["index_sequence"].values * interval / 60
x -= transition * interval / 60
y = df_track["volume"].values * yscale
plt.plot(x, y, label=f"Track {track_id}", c="#E7298A")
plt.legend()
plt.ylim(500, 1050)
plt.xlim(0, 14)
plt.ylabel(f"{ylabel} {yunits}")
plt.xlabel("Nuclear growth time (hr)")
plt.vlines(0, 500, 1050, color="k", linestyle=":")
plt.tight_layout()
save_and_show_plot(f"{figdir}/sample_slope_track_{track_id}")
plt.close()

# %% Calculate average growth rate over beginning and end subsets of tracks

# only choose values from defined regions of normalized interphase time
t1 = 0.3
t2 = 0.75
dfin = add_binned_normalized_time_for_full_tracks(df_full)
df_t1 = dfin[(dfin["digitized_normalized_time"] < t1)].groupby("track_id").agg("mean")
df_t2 = dfin[(dfin["digitized_normalized_time"] > t2)].groupby("track_id").agg("mean")

# now require that the track_id is in both dfq1 and dfq2
df_t1 = df_t1[df_t1.index.isin(df_t2.index)]
df_t2 = df_t2[df_t2.index.isin(df_t1.index)]
dfint = dfin.groupby("track_id").agg("first")

# now merge back to dfm
column = "dxdt_48_volume"
dfint.loc[df_t1.index.values, f"{column}_at_t1"] = df_t1.loc[df_t1.index.values, column]
dfint.loc[df_t2.index.values, f"{column}_at_t2"] = df_t2.loc[df_t2.index.values, column]
dfint["dxdt_t2-dxdt_t1"] = dfint[f"{column}_at_t2"] - dfint[f"{column}_at_t1"]

# %% scatter plot of alpha vs difference in late and early avg local growth rates for all individual trajectories
plot_features.scatter_plot(
    dfint,
    "all_baseline",
    "dxdt_t2-dxdt_t1",
    "tscale_linearityfit_volume",
    color_map="#808080",
    figdir=figdir,
    fitting=False,
    n_resamples=500,
    dpi=150,
    file_extension=".pdf",
    transparent=True,
)

# %% scatter plots of two samples tracks' transient growth rate from each time window relative to the average transient growth rate
# from the whole colony of a 90um neighborhood from that same time window
for local_radius_str in ["90um", "whole_colony"]:
    for tid in [EXAMPLE_TRACKS["alpha_low_example"], EXAMPLE_TRACKS["alpha_high_example"]]:
        df_track = dfin[dfin["track_id"] == tid]
        plot_features.scatter_plot(
            df_track,
            "all_baseline",
            f"neighbor_avg_dxdt_48_volume_{local_radius_str}",
            "dxdt_48_volume",
            color_map="colony",
            figdir=figdir,
            fitting=False,
            n_resamples=500,
            require_square=False,
            opacity=1,
            markersize=25,
            titleheader=f"track={tid}",
            dpi=150,
            file_extension=".pdf",
            transparent=True,
            colorby_time=True,
            suffix=f"track={tid}",
        )
# %% scatter plots of transient growth rate from each time window for all full interphase trajectories
# relative to the average transient growth rate from the whole colony of a 90um neighborhood from that same time window
for local_radius_str in ["90um", "whole_colony"]:
    for pngflag in [True, False]:
        plot_features.scatter_plot(
            dfin,
            "all_baseline",
            f"neighbor_avg_dxdt_48_volume_{local_radius_str}",
            "dxdt_48_volume",
            color_map="#808080",
            figdir=figdir,
            fitting=False,
            n_resamples=2,
            require_square=False,
            opacity=0.1,
            markersize=10,
            titleheader="full_tracks for all timepoints",
            dpi=150,
            file_extension=".pdf",
            transparent=True,
            add_unity_line=True,
            remove_all_points_in_pdf=pngflag,
        )
