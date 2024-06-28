# %%
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, load_data, filter_data
from nuc_morph_analysis.lib.preprocessing.add_times import get_trajectory_inflection_frame
from nuc_morph_analysis.lib.visualization.example_tracks import EXAMPLE_TRACKS
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
import matplotlib.pyplot as plt
import numpy as np

# %% Load data
df = global_dataset_filtering.load_dataset_with_features()
df_full = filter_data.all_timepoints_full_tracks(df)
interval = load_data.get_dataset_time_interval_in_min("all_baseline")
pixel_size = load_data.get_dataset_pixel_size("all_baseline")
# %%
figdir = "volume/figures"

# %% Get example track
track_id = EXAMPLE_TRACKS["transition_point_supplement"]
df_track = df_full.loc[df_full.track_id == track_id]

# %% Calculate transition point figure
xo = df_track["Ff"].values[0]
x = df_track.index_sequence.values - xo
y = df_track.volume.values * (pixel_size**3)
fig, ax = get_trajectory_inflection_frame(x, y, track_id, display=False, example_track_figure=True)
save_and_show_plot(
    f"{figdir}/supplement_track_{track_id}_transition_point_calculation", bbox_inches="tight"
)

# %% Resulting track with transition point
x = df_track.index_sequence.values * interval
y = df_track.volume.values * (pixel_size**3)
transition = (df_track.frame_transition.values[0] * interval) - x.min()
plt.plot(
    x - x.min(), y, "-o", label=f"Track {track_id}", color="tab:orange", markersize=4, alpha=0.5
)
plt.fill_betweenx(
    y=[y.min() - 100, y.max() + 100],
    x1=0,
    x2=200,
    color="grey",
    edgecolor="none",
    alpha=0.2,
    label="Timepoints used to calculate transition",
)
plt.axvline(
    transition, label=f"Resulting time at transition = {transition} min", color="black", alpha=0.9
)
plt.ylim(y.min() - 25, y.max() + 25)
plt.xlabel("Time (min)")
plt.ylabel("Volume (µm³)")
plt.legend(frameon=False)
save_and_show_plot(
    f"{figdir}/supplement_track_{track_id}_transition_point_result", bbox_inches="tight"
)
# %%
