# %%
import pandas as pd
from nuc_morph_analysis.analyses.volume_variation import plot_features
from nuc_morph_analysis.lib.preprocessing import filter_data, load_data, global_dataset_filtering
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.preprocessing.filter_data import (
    DAUGHTERS_APOPTOSIS_BY_ID,
    LONG_OUTLIERS_BY_ID,
    GROWTH_FEATURE_OUTLIER_BY_ID,
    OUTLIER_DAUGHTERS_BY_ID,
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# %% load data with outliers
df = global_dataset_filtering.load_dataset_with_features(remove_growth_outliers=False)
df_full = filter_data.all_timepoints_full_tracks(df)
interval = load_data.get_dataset_time_interval_in_min("all_baseline")

figdir = "evaluate_filters_and_outliers/figures/growth_outliers"

# %% Get long apoptotics
df_full_and_apoptosis = df_full.copy()
long_apoptosis = []
for track_id, dft in df.groupby("track_id"):
    if (dft.index_sequence.max() - dft.index_sequence.min()) > (19.5 / interval * 60):
        if dft.termination.unique() == 2 and track_id != 911895:
            long_apoptosis.append(track_id)
            dft.loc[:, "is_growth_outlier"] = "apoptosis"
            dft = dft[dft["fov_edge"] == False]
            dft = dft[dft["is_outlier"] == False]
            df_full_and_apoptosis = pd.concat([df_full_and_apoptosis, dft], ignore_index=True)


# %% Change color of outlier daughters:
df_full_and_apoptosis.loc[
    df_full_and_apoptosis["track_id"].isin(OUTLIER_DAUGHTERS_BY_ID), "is_growth_outlier"
] = "daughter"

# %%
track_level_feature_df = filter_data.track_level_features(df_full_and_apoptosis)
track_level_feature_df = plot_features.convert_duration_to_hours(track_level_feature_df, interval)

# %% Plot outlier stats
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
growth_outliers = (
    len(OUTLIER_DAUGHTERS_BY_ID)
    + len(DAUGHTERS_APOPTOSIS_BY_ID)
    + len(LONG_OUTLIERS_BY_ID)
    + len(GROWTH_FEATURE_OUTLIER_BY_ID)
)
# title = plt.title(f"Full interphase tracks with growth outliers: {len(track_level_feature_df)}\nFull interphase tracks without growth outliers: {len(track_level_feature_df)-growth_outliers}",
#           fontsize=12)
# title.set_position([.2,1])
labels = [
    "Second generation outlier",
    "Observed successful division",
    "Daughters fate unknown",
    "Daughters undergo apoptosis",
    "End in apoptosis",
]
values = [
    len(OUTLIER_DAUGHTERS_BY_ID),
    len(GROWTH_FEATURE_OUTLIER_BY_ID),
    len(LONG_OUTLIERS_BY_ID),
    len(DAUGHTERS_APOPTOSIS_BY_ID),
    len(LONG_OUTLIERS_BY_ID),
]
colors = [
    "#03619a",
    "#f19b07",
    "#f19b07",
    "#f19b07",
    "#ce534c",
]
plt.barh(labels, values, color=colors, alpha=0.9)
for i, v in enumerate(values):
    plt.text(v + 0.5, i, str(v), color="black", va="center")
plt.xlim(0, 24)
colors = ["#ce534c", "#f19b07", "#03619a"]
text = ["Partial trajectories", "Full trajectories", "Full trajectories"]
legend_patches = [
    mpatches.Patch(color=colors[i], label="{:s}".format(text[i])) for i in range(len(text))
]

plt.legend(handles=legend_patches, loc="lower right", fontsize=8)
plt.tight_layout()
save_and_show_plot(f"{figdir}/growth_outlier_stats")


# %% Figures in our results section with and without outliers
for feature in [
    "duration_BC_hr",
    "volume_fold_change_BC",
    "volume_at_C",
    "delta_volume_BC",
    "late_growth_rate_by_endpoints",
]:
    plot_features.scatter_plot(
        track_level_feature_df,
        "all_baseline",
        "volume_at_B",
        feature,
        color_map="growth_outliers",
        figdir=figdir,
        opacity=0.6,
        growth_outliers=True,
    )
plot_features.scatter_plot(
    track_level_feature_df,
    "all_baseline",
    "duration_BC",
    "volume_fold_change_BC",
    color_map="growth_outliers",
    figdir=figdir,
    opacity=0.6,
    growth_outliers=True,
)

# %% Visualize tracks and growth outliers
plot_features.plot_traj(
    df_full_and_apoptosis,
    "volume",
    figdir,
    interval,
    color_map="growth_outliers",
    opacity=0.35,
    growth_outliers=True,
)

# %%
