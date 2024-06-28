# %%
from nuc_morph_analysis.lib.preprocessing import load_data, filter_data, is_tp_outlier
from nuc_morph_analysis.analyses.error_morflowgenesis.plot.histograms import distribution_of_error
from nuc_morph_analysis.analyses.error_morflowgenesis.calculate_features import error
from nuc_morph_analysis.analyses.error_morflowgenesis.plot.trajectories import raw_features
from nuc_morph_analysis.analyses.error_morflowgenesis.plot.scatter import check_size_dependence

# %% Load and filter dataset
df_fix = load_data.load_morflowgenesis_dataframe("fixed_control")
df_fix["is_tp_outlier"] = is_tp_outlier.outlier_detection(df_fix).is_tp_outlier
df_fix_filtered = filter_data.filter_out_short(df_fix, length_threshold=19)
df = df_fix_filtered[~df_fix_filtered.is_tp_outlier & ~df_fix_filtered.fov_edge]

pixel_size = load_data.get_dataset_pixel_size("fixed_control")
interval = load_data.get_dataset_time_interval_in_min("fixed_control")

# %% Visualize the raw features over time for the fixed control dataset
raw_features.feature_over_time(df, pixel_size, interval, feature_type="volume")
raw_features.feature_over_time(df, pixel_size, interval, feature_type="height")
raw_features.feature_over_time(df, pixel_size, interval, feature_type="surface_area")

# %% Calculate and add error values as new columns
df = error.add_error_to_df(
    df, pixel_size, error_type="absolute", features=["volume", "surface_area", "height"]
)
df = error.add_error_to_df(
    df, pixel_size, error_type="percent", features=["volume", "surface_area", "height"]
)
df = error.add_error_to_df(
    df, pixel_size, error_type="pop_var", features=["volume", "surface_area", "height"]
)

# %% Check if the error is dependent on the size of the nucleus to determine if absolute or percent is more appropriate
check_size_dependence.check_measurement_vs_error(df, pixel_size, "absolute", "volume")
check_size_dependence.check_measurement_vs_error(df, pixel_size, "absolute", "height")
check_size_dependence.check_measurement_vs_error(df, pixel_size, "percent", "surface_area")

# %% Visualize distribition of error for volume
distribution_of_error.plot_distribution(
    df, "absolute", feature_list=["volume", "surface_area", "height"]
)
distribution_of_error.plot_distribution(
    df, "percent", feature_list=["volume", "surface_area", "height"]
)

# %%
