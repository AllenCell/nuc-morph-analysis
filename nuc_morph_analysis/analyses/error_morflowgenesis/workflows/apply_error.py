# %% Import functions
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
from nuc_morph_analysis.analyses.error_morflowgenesis.calculate_features import (
    get_precalculated_error,
)
from nuc_morph_analysis.analyses.error_morflowgenesis.plot.trajectories import baseline_colony_track
from nuc_morph_analysis.lib.preprocessing import filter_data

# %% [markdown]
#### Example of how to use the precalculated error values
# - The 99th percentile values can be easily pulled using the get_precalculated_error_value function
# - The error values can then be applied to the baseline colony tracks (volume trajectories are show here)
# - Absolute values and percent values are both shown

# %% Load baseline dataset
df = global_dataset_filtering.load_dataset_with_features()
df_full = filter_data.all_timepoints_full_tracks(df)

# %% Get absolute error value and plot it on the baseline colony volume tracks
error_value = get_precalculated_error.get_precalculated_error_value("absolute", "volume")
baseline_colony_track.full_track_with_error_bounds(
    df_full, "medium", error_value, error_type="absolute", n=5, feature="volume"
)

# %% Get percent error value and plot it on the baseline colony volume tracks
error_value = get_precalculated_error.get_precalculated_error_value("percent", "volume")
baseline_colony_track.full_track_with_error_bounds(
    df_full, "medium", error_value, error_type="percent", n=5, feature="volume"
)
# %%
