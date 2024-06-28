# %% Import functions
from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.analyses.error_morflowgenesis.plot.histograms import (
    distribution_of_error,
    distribution_of_raw_features,
)
from nuc_morph_analysis.analyses.error_morflowgenesis.calculate_features import (
    error,
    shape_modes_for_fixed_dataset,
)
from nuc_morph_analysis.analyses.error_morflowgenesis.plot.trajectories import raw_features
from nuc_morph_analysis.analyses.error_morflowgenesis.plot.scatter import check_size_dependence


# %% [markdown]
#### Goal: Quantify the error in our measured features due to segmentation.

#### Introduction to the dataset
# - The dataset used in this analysis is the fixed control dataset.
# - The cells are LMB1 and the colony is about the size of the baseline medium colony.
# - The cells are fixed with paraformaldehyde, such that nothing about the nuclei are changing.
# - The only differences between the 20 images taken are the background noise and a 5µm translation in x,y
# *(average distance nuclei move in a 5 minute interval)
# - Fluctuations in the nuclear volume, height, surface area and shapes can be attributed to error in our segmentations.

# %% Load and filter dataset
pixel_size = load_data.get_dataset_pixel_size("fixed_control")
interval = load_data.get_dataset_time_interval_in_min("fixed_control")
# %% Add shape modes to fixed dataset
df_all_filtered = shape_modes_for_fixed_dataset.get_all_dataframe_for_shape_space()
df_fix_filtered = (
    shape_modes_for_fixed_dataset.get_fixed_dataframe_for_shape_space()
)  # no tid_outliers specific to v.0.2.3 yet identified
df = shape_modes_for_fixed_dataset.get_dataframe_with_shape_modes(df_all_filtered, df_fix_filtered)

# %% Visualize distribition of raw features for the fixed control dataset
distribution_of_raw_features.length_of_tracks(df)
distribution_of_raw_features.median_track_features(df, pixel_size, feature_type="volume")
distribution_of_raw_features.median_track_features(df, pixel_size, feature_type="height")
distribution_of_raw_features.median_track_features(df, pixel_size, feature_type="surface_area")

# %% Visualize the raw features over time for the fixed control dataset
raw_features.feature_over_time(df, pixel_size, interval, feature_type="volume")
raw_features.feature_over_time(df, pixel_size, interval, feature_type="height")
raw_features.feature_over_time(df, pixel_size, interval, feature_type="surface_area")
raw_features.shape_modes_over_time(df, interval)

# %% [markdown]
#### Calculate the error
# - The median value of a track is assumed to be the true value
# - The absolute error and percent error are both calculated for every datapoint in all the tracks that persist for 20 frames.
# - The population variation calculates the error relative to the range of possible values for a given feature.
# - Measured features include volume, height, surface area, and shape modes

# %% Calculate and add error values as new columns
df = error.add_error_to_df(df, pixel_size, error_type="absolute")
df = error.add_error_to_df(df, pixel_size, error_type="percent")
df = error.add_error_to_df(df, pixel_size, error_type="pop_var")

# %% Visualize distribition of error for volume
distribution_of_error.plot_error_distrubution_table(
    df, error_type="absolute", feature_list=["volume"]
)
distribution_of_error.plot_error_distrubution_table(
    df, error_type="percent", feature_list=["volume"]
)

distribution_of_error.plot_error_distrubution_table(
    df, error_type="absolute", feature_list=["height"]
)
distribution_of_error.plot_error_distrubution_table(
    df, error_type="percent", feature_list=["height"]
)

distribution_of_error.plot_error_distrubution_table(
    df, error_type="absolute", feature_list=["surface_area"]
)
distribution_of_error.plot_error_distrubution_table(
    df, error_type="percent", feature_list=["surface_area"]
)

distribution_of_error.plot_error_distrubution_table(
    df, error_type="absolute", feature_list=["shape_modes"]
)
distribution_of_error.plot_error_distrubution_table(
    df, error_type="percent", feature_list=["shape_modes"]
)

# %% Check if the error is dependent on the size of the nucleus to determine if absolute or percent is more appropriate
check_size_dependence.check_measurement_vs_error(df, pixel_size, "absolute", "volume")
check_size_dependence.check_measurement_vs_error(df, pixel_size, "percent", "volume")
check_size_dependence.check_measurement_vs_error(df, pixel_size, "absolute", "height")
check_size_dependence.check_measurement_vs_error(df, pixel_size, "percent", "height")
check_size_dependence.check_measurement_vs_error(df, pixel_size, "absolute", "surface_area")
check_size_dependence.check_measurement_vs_error(df, pixel_size, "percent", "surface_area")

# %% [markdown]
#### Generate error metric
# - The 99th percentile is used as the error value for each measured feature
# - The error values are hard coded in nuc_morph_analysis.analyses.error.calculate_features.get_precalculated_error
# - These can be used to apply error bounds on the baseline colony

# %%
error.print_error_values(df, percentile="99%")

# %% [markdown]
# See apply_error workflow for an example of how to use these error values in your analysis.
# %%
