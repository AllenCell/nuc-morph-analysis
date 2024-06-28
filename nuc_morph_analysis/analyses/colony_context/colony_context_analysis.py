import numpy as np
import matplotlib.pyplot as plt

from nuc_morph_analysis.lib.preprocessing.load_data import (
    get_dataset_pixel_size,
    get_dataset_time_interval_in_min,
)
from nuc_morph_analysis.lib.visualization.plotting_tools import (
    create_scatter_plot,
    plot_time_series_by_frame,
    get_plot_labels_for_metric,
    plot_weighted_scatterplot_by_frame,
)
from nuc_morph_analysis.lib.visualization.reference_points import (
    COLONY_COLORS,
    FOV_EXIT_T_INDEX,
    FOV_TOUCH_T_INDEX,
)
from nuc_morph_analysis.utilities.analysis_utilities import (
    get_correlation_values,
)


# make plot text editable in Illustrator
import matplotlib
from matplotlib.patches import Rectangle

matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"


def is_categorical(var):
    return var in [
        "colony_depth",
        "colony_position_group_depth",
        "colony_position_group_distance",
        "colony_position_group",
    ]


def add_fov_touch_timepoint_for_colonies(df):
    """
    Adds a column to dataframe indicating the timepoint when the colony edge
    first touches the frame boundary

    Parameters
    ----------
    df: dataframe
        dataframe to add fov touch status

    Returns
    ----------
    Dataframe with added column "colony_edge_in_fov"
    """
    df["colony_edge_in_fov"] = np.nan
    for dataset, df_dataset in df.groupby("colony"):
        if dataset in FOV_TOUCH_T_INDEX:
            fov_touch_t_index = FOV_TOUCH_T_INDEX[dataset]
            fov_exit_t_index = FOV_EXIT_T_INDEX[dataset]
            df_dataset.loc[
                df_dataset["index_sequence"] < fov_touch_t_index, "colony_edge_in_fov"
            ] = "full"
            df_dataset.loc[
                (df_dataset["index_sequence"] >= fov_touch_t_index)
                & (df_dataset["index_sequence"] < fov_exit_t_index),
                "colony_edge_in_fov",
            ] = "partial"
            df_dataset.loc[
                df_dataset["index_sequence"] >= fov_exit_t_index, "colony_edge_in_fov"
            ] = "none"
            df.loc[df_dataset.index, "colony_edge_in_fov"] = df_dataset["colony_edge_in_fov"]

    return df


def get_list_of_dims():
    """
    Returns a list of all dimensions used for analysis
    """
    return ["x", "y", "z"]


def add_colony_position_columns(
    df_original,
    time_col="index_sequence",
    pix_size=get_dataset_pixel_size("all_baseline"),
    num_angle_bins=128,
):
    """
    Adds various columns related to colony position analysis to dataframe

    Added columns:
    - colony_centroid_{x,y,z}: centroid of colony
    - max_distance_from_centroid: maximum distance from centroid
    - distance_from_centroid: distance from centroid
    - normalized_distance_from_centroid: distance from centroid normalized by max distance
    - colony_non_circularity: circularity metric for colony, calculated as the standard
        deviation of the distance from the centroid in each angle bin
    - colony_non_circularity_scaled: circularity metric scaled to 0-1

    If colony depth is available, adds normalized colony depth column
    - colony_depth: 1 corresponds to colony edge and increases by 1 for each layer inwards
    - normalized_colony_depth: 0 for the innermost layer and 1 for the outermost layer

    Parameters
    ----------
    df_original: dataframe
        dataframe to add colony position columns to
    time_col: str
        Name of the column indicating time
    pix_size: float
        size of pixel in um
    num_radial_bins: int
        Number of bins to subdivide for circularity metric

    Returns
    ----------
    Dataframe with added colony position columns
    """
    centroid_cols = [f"centroid_{dim}" for dim in get_list_of_dims()]
    df = df_original.dropna(subset=centroid_cols)
    angle_bins = np.linspace(-np.pi, np.pi, num_angle_bins)

    for dim in get_list_of_dims():
        df[f"colony_centroid_{dim}"] = np.nan

    cols_to_add = [
        "distance_from_centroid",
        "max_distance_from_centroid",
        "normalized_distance_from_centroid",
        "colony_non_circularity",
        "colony_non_circularity_scaled",
    ]

    calculate_depth_columns = False
    if "colony_depth" in df.columns and df["colony_depth"].notna().any():
        df = df.dropna(subset=["colony_depth"])
        calculate_depth_columns = True
        cols_to_add += ["normalized_colony_depth", "max_colony_depth"]

    for col in cols_to_add:
        df[col] = np.nan

    for dataset, df_dataset in df.groupby("colony"):

        for _, df_frame in df_dataset.groupby(time_col):
            centroid_pos = df_frame[centroid_cols].values * pix_size
            colony_center_pos = np.mean(centroid_pos, axis=0)
            r_pos = np.linalg.norm(centroid_pos - colony_center_pos, axis=1)
            max_distance_from_centroid = np.nanmax(r_pos)
            norm_r_pos = r_pos / max_distance_from_centroid

            for dc, dim in enumerate(get_list_of_dims()):
                df.loc[df_frame.index, f"colony_centroid_{dim}"] = colony_center_pos[dc]

            df.loc[df_frame.index, "distance_from_centroid"] = r_pos
            df.loc[df_frame.index, "max_distance_from_centroid"] = max_distance_from_centroid
            df.loc[df_frame.index, "normalized_distance_from_centroid"] = norm_r_pos

            # calculate circularity
            angles = np.arctan2(
                centroid_pos[:, 1] - colony_center_pos[1],
                centroid_pos[:, 0] - colony_center_pos[0],
            )
            angle_inds = np.digitize(angles, angle_bins)
            r_max_vals = np.zeros(num_angle_bins)
            for ac in range(num_angle_bins):
                indices_in_bin = angle_inds == ac
                if np.any(indices_in_bin):
                    r_max_vals[ac] = np.max(r_pos[indices_in_bin])
            r_max_dev = np.std(r_max_vals)
            df.loc[df_frame.index, "colony_non_circularity"] = r_max_dev
            df.loc[df_frame.index, "colony_non_circularity_scaled"] = r_max_dev / np.mean(
                r_max_vals
            )

            # add colony depth columns
            if calculate_depth_columns:
                max_colony_depth = df_frame["colony_depth"].max()
                min_colony_depth = df_frame["colony_depth"].min()
                df.loc[df_frame.index, "max_colony_depth"] = max_colony_depth
                df.loc[df_frame.index, "normalized_colony_depth"] = 1 - (
                    df_frame["colony_depth"] - min_colony_depth
                ) / (
                    max_colony_depth - min_colony_depth
                )  # opposite direction to colony_depth
    df_original[cols_to_add] = df[cols_to_add]
    return df_original


def plot_radial_profile(
    df,
    col_to_plot="height",
    distance_col="normalized_distance_from_centroid",
    save_dir=None,
    smooth_weight=0,
    align_to_colony_time=False,
    save_intermediate_correlations=[0, 25, 50, 75, 100],
    save_intermediate_correlations_colony="all",
    correlation_metric="pearson",
    plot_scale=25,
    additional_col=None,
    save_format="pdf",
    bootstrap_count=100,
    filter_edge=True,
    weight_by_r2=False,
    make_height_profile_lineplot=True,
    make_height_profile_scatterplot=True,
):
    """
    Plots the radial profile for a column vs time

    Parameters
    ----------
    df: dataframe
        dataframe containing colony position columns
    col_to_plot: string
        name of column to plot
    distance_col: string
        name of column containing colony position information
    save_dir: path
        path at which images are saved
    smooth_weight: float
        weight for smoothing the data
    align_to_colony_time: bool
        whether to align the data to the colony time
    save_intermediate_correlations: list of ints
        list of timepoints at which to save the correlation scatter plot
    save_intermediate_correlations_colony: string
        name of colony to save intermediate correlation scatter plots for
        if "all" make for all colonies
    weight_by_r2: bool
        Whether to weight the scatter plot dot size by the r2 value
    make_height_profile_lineplot: bool
        Whether to make the line plot of the radial profile
    make_height_profile_scatterplot: bool
        Whether to make the scatter plot of the radial profile

    Returns
    -------
    fig, ax: matplotlib figure and axis objects
    """
    radial_save_dir = None
    corr_save_dir = None
    if save_dir is not None:
        radial_save_dir = save_dir / "radial_profile_plots"
        radial_save_dir.mkdir(parents=True, exist_ok=True)

        if save_intermediate_correlations is not None:
            corr_save_dir = radial_save_dir / "correlation_scatter"
            corr_save_dir.mkdir(parents=True, exist_ok=True)
    if filter_edge:
        df = df[~df["fov_edge"]].copy()
    df[f"radial_profile_corr_{col_to_plot}"] = np.nan
    df[f"slope_{col_to_plot}"] = np.nan
    if bootstrap_count > 0:
        df[f"radial_profile_corr_{col_to_plot}_ci_low"] = np.nan
        df[f"radial_profile_corr_{col_to_plot}_ci_high"] = np.nan
    df = df.dropna(subset=[distance_col, col_to_plot])
    scale_factor, label, unit, _ = get_plot_labels_for_metric(col_to_plot)
    time_interval = get_dataset_time_interval_in_min("all_baseline")

    for dataset, df_dataset in df.groupby("colony"):
        print(f"Plotting radial profile for {col_to_plot}, dataset: {dataset}")
        frames_to_snap = []
        if save_intermediate_correlations is not None:
            frames_to_snap = np.round(
                np.percentile(
                    range(df_dataset["index_sequence"].nunique()),
                    save_intermediate_correlations,
                )
            )
        df_dataset = df_dataset.sort_values("index_sequence")

        frame_count = 0
        for frame_num, (_, df_frame) in enumerate(df_dataset.groupby("index_sequence")):
            x_values = df_frame[distance_col].values
            y_values = df_frame[col_to_plot].values * scale_factor
            std_x = np.std(x_values)
            std_y = np.std(y_values)
            if (
                (corr_save_dir is not None)
                and (frame_num in frames_to_snap)
                and (
                    (save_intermediate_correlations_colony == "all")
                    or (dataset == save_intermediate_correlations_colony)
                )
            ):
                current_colony_time = df_frame["colony_time"].iloc[0] * time_interval / 60
                create_scatter_plot(
                    x_values,
                    y_values,
                    "Normalized Radial Position in the Colony\n"
                    f"at Colony Time {current_colony_time:0.2g}hr",
                    f"{label} {unit}",
                    corr_save_dir,
                    correlation_metric=correlation_metric,
                    save_format=save_format,
                    plot_linear_fit=True,
                    y_lim=[2, 13],
                    scatter_args={
                        "s": 20,
                        "c": COLONY_COLORS[df_dataset["colony"].values[0]],
                        "edgecolors": "none",
                    },
                    use_slope=True,
                )
                frame_count += 1
            corr_val, _, err_val, ci = get_correlation_values(
                x_values,
                y_values,
                correlation_metric=correlation_metric,
                bootstrap_count=bootstrap_count,
            )
            df.loc[df_frame.index, f"radial_profile_corr_{col_to_plot}"] = corr_val
            df.loc[df_frame.index, f"slope_{col_to_plot}"] = corr_val * std_y / std_x
            if ci is not None:
                df.loc[df_frame.index, f"radial_profile_corr_{col_to_plot}_ci_low"] = ci[0]
                df.loc[df_frame.index, f"radial_profile_corr_{col_to_plot}_ci_high"] = ci[1]

    if make_height_profile_lineplot:
        plot_time_series_by_frame(
            df,
            col_to_plot=f"radial_profile_corr_{col_to_plot}",
            err_cols=(
                [
                    f"radial_profile_corr_{col_to_plot}_ci_low",
                    f"radial_profile_corr_{col_to_plot}_ci_high",
                ]
                if bootstrap_count > 0
                else None
            ),
            smooth_frames=smooth_weight,
            align_to_colony_time=align_to_colony_time,
            save_dir=radial_save_dir,
            ylabel=f"Radial correlation for {label}",
            additional_col_to_plot=additional_col,
            save_format=save_format,
        )

    if make_height_profile_scatterplot:
        if weight_by_r2:
            weights_col = f"radial_profile_corr_{col_to_plot}"
        else:
            weights_col = None
        plot_weighted_scatterplot_by_frame(
            df,
            col_to_plot=f"slope_{col_to_plot}",
            align_to_colony_time=align_to_colony_time,
            save_dir=radial_save_dir,
            ylabel="Slope of nuclear height vs normalized \n distance from colony center",
            weights_col=weights_col,
            plot_scale=plot_scale,
            additional_col=additional_col,
            save_format=save_format,
        )
    plt.close()
