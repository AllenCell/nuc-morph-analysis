import random
import colour
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats as spstats
import seaborn as sb
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.example_tracks import EXAMPLE_TRACKS
from nuc_morph_analysis.lib.visualization.reference_points import (
    COLONY_COLORS,
    OBSERVED_TOUCH_COLONY,
    OBSERVED_MERGE_COLONY,
    FOV_TOUCH_T_INDEX,
    FOV_EXIT_T_INDEX,
)
from nuc_morph_analysis.lib.preprocessing import filter_data, load_data
import re

matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"

TIME_BIN_COLORS = {1.0: "#E7298A", 2.0: "#66A61E", 3.0: "#E6AB02"}
COLONY_COLORS_GRADIENT = {
    "small": ["#76c4ad", "#126e53"],
    "medium": ["#ecaf80", "#ad4c01"],
    "large": ["#aca9d1", "#5d598f"],
}
GROWTH_OUTLIERS = {False: "#cdcdcd", True: "#f19b07", "apoptosis": "#ce534c", "daughter": "#03619a"}
MAP_ZORDER = {"#cdcdcd": 0, "#f19b07": 10000, "#03619a": 10000, "#ce534c": 10000}
MAP_OPACITY = {"#cdcdcd": 0.5, "#f19b07": 0.85, "#03619a": 0.85, "#ce534c": 0.85}
SMALL_SISTER = {False: "tab:orange", True: "tab:purple"}


def get_color(df, color_map):
    """
    Parameters
    ----------
    df: DataFrame
        DataFrame containing dataset column and time bin column

    color_map: str
        Either "time_bin" or "colony" or a hex code for a single color

    Returns
    -------
    colors for each row in df based on mapping or single color specified
    """
    if color_map == "time_bin":
        time_bins = df["colony_time_bin_at_B"]
        colors = time_bins.map(TIME_BIN_COLORS)
    elif color_map == "colony":
        colonies = df["colony"]
        colors = colonies.map(COLONY_COLORS)
    elif color_map == "growth_outliers":
        growth_outliers = df["is_growth_outlier"]
        colors = growth_outliers.map(GROWTH_OUTLIERS)
    elif color_map == "small_sister":
        small_sister = df["small_sister"]
        colors = small_sister.map(SMALL_SISTER)
    else:
        colors = color_map
    return colors


def theilslopes_fitting(x, y, xplot=None):
    """
    Fits a line data to a line

    Parameters
    ----------
    x: list
        x values
    y: list
        y values
    xplot: list
        set of x values to plot the fitted line on

    Returns
    -------
    (x, fitted_y): tuple
        x values and y values of the fitted line
    label: str
        label for the fitted line
    fitted_line.slope: float
        slope of the fitted line
    """

    x = np.array(x)
    y = np.array(y)
    fitted_line = spstats.theilslopes(y, x, 0.95)

    if xplot is None:
        xplot = x

    fitted_y = fitted_line.intercept + fitted_line.slope * xplot
    label = f"n={len(x)}\nSlope={fitted_line.slope:.2f}"
    return (xplot, fitted_y), label, fitted_line.slope


def confidence_interval(df, n_resamples, column_1, column_2, corr=True):
    """
    Bootstrap calculate the confidence interval for a correlation between two columns in a pandas dataframe

    Parameters
    ----------
    df: Dataframe
        The subset dataframe containing the columns you want to compare

    n_resamples: Int
        Number of times to resample the data

    column_1: String
        Dataframe column name to compare (ie. 'Late_growth_rate_fitted')

    column_2: String
        Dataframe column name to compare (ie. 'Volume_at_B')

    Returns
    -------
    mean: Float
        Mean of the resampled correlations

    std_dev: Float
        Standard deviation of the resampled correlations

    re_sampled_corr: List
        List containing all the correlation values calculated for all the samples for easy visualization in a histogram.

    percent: List
        List containing the 5 and 95 percentile of the resampled correlations
    """
    df = df.reset_index()
    original_data = list(range(0, len(df)))
    re_sampled = []
    rng = np.random.default_rng(seed=42)
    for _ in range(n_resamples):
        rints = rng.integers(low=0, high=len(original_data), size=len(original_data))
        df_resampled = df.iloc[rints]
        if corr is True:
            corr_sample = df_resampled[column_1].corr(df_resampled[column_2], method="spearman")
            re_sampled.append(corr_sample)
        else:
            _, _, slope_sample = theilslopes_fitting(df_resampled[column_1], df_resampled[column_2])
            re_sampled.append(slope_sample)
    mean = np.mean(re_sampled)
    std_dev = np.std(re_sampled)
    percent = np.percentile(re_sampled, [5, 95])
    return mean, std_dev, re_sampled, percent


def scatter_plot(
    df,
    colony,
    column_1,
    column_2,
    color_map,
    figdir,
    opacity=0.4,
    fitting=False,
    color_column=None,
    add_known_timepoints=False,
    scaling_ref_line=np.nan,
    value_ref_line=np.nan,
    n_resamples=500,
    require_square=False,
    file_extension=".pdf",
    markersize=25,
    transparent=True,
    titleheader=None,
    dpi=300,
    colorby_time=False,
    suffix=None,
    plot_sample_tracks=False,
    plot_rolling_window_mean_flag=False,
    rolling_window_size_hr=10,
    colorbar=True,
    line_per_colony_flag=False,
    add_unity_line=False,
    remove_all_points_in_pdf=False,
    growth_outliers=False,
    feeding_control=False,
):
    """
    Scatter plot of two features in a dataframe

    df: DataFrame
        Dataframe containing a single metric per track

    colony: str
        Colony to plot

    column_1: str
        Feature to plot on x-axis

    column_2: str
        Feature to plot on y-axis

    color_map: str
        Either "time_bin" or "colony" or a hex code for a single color

    figdir: str
        Directory to save figure

    opacity: float
        Opacity of points

    fitting: bool
        Whether to fit a line to the data

    color_column: str
        Column name to color the points by (ie 'avg_colony_depth')

    add_known_timepoints: bool
        Whether to add known timepoints to the plot
        Must have the x axis be a time in hours column (ie. Time at C)
        Must be for a single colony.

    scaling_ref_line: float
        Scaling of y to x axis for visual reference line

    value_ref_line: float
        Value of y for visual reference line

    n_resamples: int
        Number of times to resample the data for confidence interval calculation

    require_square: bool
        Whether to force the xlimits and ylimits to be equal

    file_extension: str
        File extension to save the plot as
        either '.pdf' or '.png'

    markersize: int
        Size of the markers in the scatter plot

    transparent: bool
        Whether to save the plot as transparent

    titleheader: str
        Additiona (optional) header for the plot title

    dpi: int
        Resolution of the plot for saving

    colorby_time: bool
        Whether to color the points by time

    suffix: str
        Additional suffix to add to the file name

    plot_sample_tracks: bool
        Flag to mark where sample tracks are taken from in
        deltaV BC vs volB plot for paper figure

    plot_rolling_window_mean_flag: bool
        Whether to plot the rolling window mean of the data over colony time

    rolling_window_size_hr: int
        Size of the rolling window in hours

    line_per_colony_flag: bool
        Whether to plot a separate fitting or rolling mean line for each colony

    add_unity_line: bool
        Whether to add a unity line to the plot

    remove_all_points_in_pdf: bool
        Whether to remove all points in the pdf plot so it can open in illustator

    growth_outliers: bool
        Adjust default limits to accomidate growth outliers

    feeding_control: bool
        Adjust default limits to accomidate feeding control data

    Returns
    -------
    plot

    mean slope: float
        optional, mean slope of the fitted line
    confidence interval: list
        optional, confidence interval of the fitted line
    """
    if colony == "all_baseline":
        df_d = df
    else:
        df_d = df[df.colony == colony]

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=dpi)

    nas = np.logical_or(np.isnan(df_d[column_1]), np.isnan(df_d[column_2]))
    df_pearson = df_d[~nas]

    pearson, p_pvalue = spstats.pearsonr(df_pearson[column_1], df_pearson[column_2])

    corr = pearson
    mean_corr, _, _, percent = confidence_interval(
        df=df_d, n_resamples=n_resamples, column_1=column_1, column_2=column_2
    )
    nas = np.logical_or(np.isnan(df_d[column_1]), np.isnan(df_d[column_2]))
    df_pearson = df_d[~nas]
    pearson, p_pvalue = spstats.pearsonr(df_pearson[column_1], df_pearson[column_2])

    xscale, xlabel, xunits, xlim = get_plot_labels_for_metric(column_1)
    yscale, ylabel, yunits, ylim = get_plot_labels_for_metric(column_2)

    if colorby_time == True:
        cscale, clabel, cunits, clim = get_plot_labels_for_metric("index_sequence")
        # cis = df_d['normalized_time'].values
        cis = df_d["index_sequence"].values * cscale
        cis = cis - cis.min()
        norm = matplotlib.colors.Normalize(vmin=cis.min(), vmax=cis.max())
        ax.scatter(
            df_d[column_1] * xscale,
            df_d[column_2] * yscale,
            c=cis,
            alpha=opacity,
            s=markersize,
            edgecolors="none",
            cmap="viridis",
        )
        if colorbar:
            plt.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"),
                ax=ax,
                label=f"Time Relative to B {cunits}",
            )

    else:
        if color_column == None:
            ax.scatter(
                df_d[column_1] * xscale,
                df_d[column_2] * yscale,
                c=get_color(df_d, color_map),
                alpha=opacity,
                s=markersize,
                edgecolors="none",
            )
        else:
            if color_column == "colony_time_at_B":
                vmin = df["colony_time_at_B"].min()
                vmax = df["colony_time_at_B"].max()
            else:
                vmin = None
                vmax = None
            ax.scatter(
                df_d[column_1] * xscale,
                df_d[column_2] * yscale,
                c=df_d[color_column],
                cmap="cool",
                vmin=vmin,
                vmax=vmax,
                alpha=opacity,
                s=markersize,
                edgecolors="none",
            )

    if p_pvalue < 0.001:
        pstring = "P<.001"
    else:
        pstring = str(np.round(p_pvalue, 3))
        pstring = f"P={pstring[1:]}"
    plt.title(f"R={pearson:.2f}, {pstring}")

    fitting_title = ""
    if fitting:
        if xlim is not None:
            xplot = np.linspace(xlim[0], xlim[1], 100)
        else:
            xplot = df_d[column_1]

        if line_per_colony_flag:
            for colony, df_colony in df_d.groupby("colony"):
                linecolor = COLONY_COLORS[df_colony["colony"].iloc[0]]
                (x_fit, y_fit), label, _ = theilslopes_fitting(
                    df_colony[column_1], df_colony[column_2], xplot
                )
                mean_slope, _, _, percent_slope = confidence_interval(
                    df=df_colony,
                    n_resamples=n_resamples,
                    column_1=column_1,
                    column_2=column_2,
                    corr=False,
                )
                label = f"m={mean_slope:.2f} ({percent_slope[0]:.2f}, {percent_slope[1]:.2f})"
                ax.plot(x_fit, y_fit, label=label, color=linecolor, linewidth=3, alpha=0.9)
        else:
            if colony == "all_baseline":
                linecolor = "k"
            else:
                linecolor = COLONY_COLORS[df_d["colony"].iloc[0]]
            (x_fit, y_fit), label, _ = theilslopes_fitting(df_d[column_1], df_d[column_2], xplot)
            mean_slope, _, _, percent_slope = confidence_interval(
                df=df_d, n_resamples=n_resamples, column_1=column_1, column_2=column_2, corr=False
            )
            label = f"m={mean_slope:.2f} ({percent_slope[0]:.2f}, {percent_slope[1]:.2f})"
            ax.plot(x_fit, y_fit, label=label, color=linecolor, linewidth=3, alpha=0.9)

    if plot_rolling_window_mean_flag:
        if line_per_colony_flag:
            for colony, df_colony in df_d.groupby("colony"):
                linecolor = COLONY_COLORS[df_colony["colony"].iloc[0]]
                feature_avg, times = rolling_window_time_mean(
                    rolling_window_size_hr, df_colony, column_2, column_1
                )
                times = [t * xscale for t in times]
                feature_avg = [f * yscale for f in feature_avg]
                ax.plot(
                    times,
                    feature_avg,
                    label=f"Mean over {rolling_window_size_hr} hr rolling window",
                    color=linecolor,
                    linewidth=3,
                )
        else:
            if colony == "all_baseline":
                linecolor = "k"
            else:
                linecolor = COLONY_COLORS[df_d["colony"].iloc[0]]
            feature_avg, times = rolling_window_time_mean(
                rolling_window_size_hr, df_d, column_2, column_1
            )
            times = [t * xscale for t in times]
            feature_avg = [f * yscale for f in feature_avg]
            ax.plot(
                times,
                feature_avg,
                label=f"Mean over {rolling_window_size_hr} hr rolling window",
                color=linecolor,
                linewidth=3,
            )

    if titleheader == None:
        headerstr = ""
    else:
        headerstr = f"{titleheader}\n"

    if p_pvalue < 0.001:
        pstring = "P<.001"
    else:
        pstring = str(np.round(p_pvalue, 3))
        pstring = f"P={pstring[1:]}"
    plt.title(f"R={pearson:.2f}, {pstring}")
    ax.set_title(f"{headerstr}N = {len(df_d)}\nr={corr:.2f}, {pstring}{fitting_title}")
    if color_column != None:
        _, label, units, _ = get_plot_labels_for_metric(color_column)
        if color_map == "colony_time_at_B":
            colors = df[color_column]
        else:
            colors = df_d[color_column]
        norm = matplotlib.colors.Normalize(vmin=colors.min(), vmax=colors.max())
        if colorbar:
            fig.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap="cool"),
                ax=ax,
                orientation="vertical",
                label=f"{label} {units}",
            )
    if add_known_timepoints is True:
        colony = df_d["colony"].unique()[0]
        interval = load_data.get_dataset_time_interval_in_min("all_baseline")
        if OBSERVED_TOUCH_COLONY[colony] is not None:
            ax.axvline(
                x=OBSERVED_TOUCH_COLONY[colony] * interval / 60,
                color="k",
                linestyle="--",
                linewidth=1.2,
                label="1st Touch Event",
                alpha=0.5,
                zorder=0,
            )
            ax.axvline(
                x=OBSERVED_MERGE_COLONY[colony] * interval / 60,
                color="k",
                linestyle="--",
                linewidth=1.2,
                label="1st Merge Event",
                alpha=0.5,
                zorder=0,
            )
        if FOV_TOUCH_T_INDEX[colony] is not None:
            ax.axvline(
                x=FOV_TOUCH_T_INDEX[colony] * interval / 60,
                color="r",
                linestyle="--",
                linewidth=1.2,
                label="FOV Touch",
                alpha=0.5,
                zorder=0,
            )
        if FOV_EXIT_T_INDEX[colony] is not None:
            ax.axvline(
                x=FOV_EXIT_T_INDEX[colony] * interval / 60,
                color="b",
                linestyle="--",
                linewidth=1.2,
                label="FOV Exit",
                alpha=0.5,
                zorder=0,
            )
        plt.legend(loc="upper right", fontsize=8)

    if plot_sample_tracks:
        df_sample = df_d[
            df_d["track_id"].isin(
                [EXAMPLE_TRACKS["delta_v_BC_high"], EXAMPLE_TRACKS["delta_v_BC_low"]]
            )
        ]
        ax.scatter(
            df_sample[column_1] * xscale,
            df_sample[column_2] * yscale,
            c="k",
            alpha=1,
            s=markersize,
            edgecolors="none",
        )

    if growth_outliers is False:
        if ylim is not None:
            ax.set_ylim(ylim)

        if xlim is not None:
            ax.set_xlim(xlim)
            if ~np.isnan(scaling_ref_line):
                x = np.linspace(xlim[0], xlim[1], 100)
                y = scaling_ref_line * x
                ax.plot(x, y, linestyle="--", color="k")
            if ~np.isnan(value_ref_line):
                ax.axhline(y=value_ref_line, color="k", linestyle="--")

    if column_1 == "volume_at_B":
        if growth_outliers is False:
            ax.set_xticks([300, 400, 500, 600, 700])
        else:
            ax.set_xticks([300, 400, 500, 600, 700, 800])

    if feeding_control:
        ax.set_yticks([20, 30, 40, 50, 60, 70])
        ax.set_ylim(15, 70)

    ax.set_xlabel(f"{xlabel} {xunits}")
    ax.set_ylabel(f"{ylabel} {yunits}")

    if require_square:
        print("setting axis limits equal")
        limits = ax.axis("tight")
        print(limits)
        minlim, maxlim = min(limits[0], limits[2]), max(limits[1], limits[3])
        ax.axis([minlim, maxlim, minlim, maxlim])
        ax.axline(
            (minlim, minlim),
            (maxlim, maxlim),
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.5,
        )

    if add_unity_line:
        ax.axline((0, 0), (1, 1), color="black", linestyle="--", linewidth=1.2, alpha=0.5)

    color_name = color_map if color_column is None else color_column
    colors_used = "_".join(get_color(df_d, color_map).unique()) if color_map == "time_bin" else None
    file_name = f"features_{column_1}_{column_2}_dataset_{colony}"
    if color_name is not None:
        file_name = f"colorby_{color_name}_" + file_name
    if colors_used is not None:
        file_name += f"_timebincolor_{colors_used}"
    if add_known_timepoints is True:
        file_name += "_known_timepoints"
    if ~np.isnan(scaling_ref_line):
        file_name += f"_scaleref{scaling_ref_line}"
    if ~np.isnan(value_ref_line):
        file_name += f"_valueref{value_ref_line}"
    if fitting:
        file_name += "_fitted"
    if plot_rolling_window_mean_flag:
        file_name += "_rolling_window_mean"
    if line_per_colony_flag and (fitting or plot_rolling_window_mean_flag):
        file_name += "_line_per_colony"
    if suffix is not None:
        file_name += f"{suffix}"
    if plot_sample_tracks:
        file_name += "_mark_sample_tracks"
    if color_column is not None and colorbar:
        file_name += "_colorbar"

    save_and_show_plot(
        f'{figdir}/{re.sub("[#/()]", "", file_name)}',
        file_extension=file_extension,
        bbox_inches="tight",
        transparent=transparent,
        remove_all_points_in_pdf=remove_all_points_in_pdf,
    )

    plt.close()  # close plot after saving to avoid opening too many figure windows.
    if fitting:
        return mean_slope, percent_slope


def plot_traj(
    df,
    feature_col,
    figdir,
    time_interval_minutes,
    colony="all_baseline",
    colony_time=True,
    color_map="Thistle",
    opacity=0.5,
    legend=True,
    extra_label="",
    highlight_samples=False,
    growth_outliers=False,
):
    """
    This function plots all the volume trajectories in the loaded dataset.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing volume metric for all the tracked nuclear trajectories

    feature_col: string
        Name of column with feature tracks to plot: likely a volume or surface area tracks

    figdir: path, string
        Directory to save the trajectory plot

    pixel_size: float
        width of pixel in microns

    time_interval_minutes: int
        time interval in minutes between each image acquisition

    dataset: string
        name of the dataset of interest (eg. 'medium'). Default value="all_baseline" for all datasets
        Set colony_time=False if trajectories are to be plotted vs Real time

    colony_time: Boolean
        Default:True
        Set colony time as x-axis for the trajectories otherwise use Real time

    color_map: str
        Either "time_bin" or "colony" to use preset color maps
        A hex code or string for a single color

    opacity: float
        value can be between 0 to 1 for the transparency of the plot

    extra_label: str
        addition labeling for disambiguity subsetted data

    highlight_samples: bool
        Flag for whether to highlight example tracks with high, medium and low alphas

    growth_outliers: bool
        Adjust y limit to show growth outliers

    Returns
    -------
    fig, ax: matplotlib objects
        figure and axis objects containing the trajectory plot
    """

    yscale, ylabel, yunits, ylim = get_plot_labels_for_metric(feature_col)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    if colony == "all_baseline":
        df_d = df
    else:
        df_d = df[df.colony == colony]

    df_d["time"] = df_d["index_sequence"]
    if colony_time is True:
        df_d["time"] = df_d["colony_time"]

    # Set the seed to make random colors from get_random_color_between reproducible
    # We looked at the output and this seed looks nice
    random.seed(1235)
    for _, df_track in df_d.groupby("track_id"):
        df_track = df_track.sort_values("time")

        zorder = None

        if color_map == "colony":
            color = COLONY_COLORS[df_track["colony"].iloc[0]]
        elif color_map == "growth_outliers":
            color = GROWTH_OUTLIERS[df_track["is_growth_outlier"].iloc[0]]
            zorder = MAP_ZORDER[color]
            opacity = MAP_OPACITY[color]
        elif color_map == "colony_gradient":
            colors = COLONY_COLORS_GRADIENT[df_track["colony"].iloc[0]]
            color = get_random_color_between(colors[0], colors[1])
        else:
            color = color_map

        plt.plot(
            df_track.time * time_interval_minutes / 60,
            df_track[feature_col] * (yscale),
            color=color,
            alpha=opacity,
            zorder=zorder,
        )

    if highlight_samples:
        example_tracks = (
            EXAMPLE_TRACKS["trajectory_shape_high"]
            + EXAMPLE_TRACKS["trajectory_shape_mid"]
            + EXAMPLE_TRACKS["trajectory_shape_low"]
        )
        for track_id in example_tracks:
            df_track = df_d[df_d["track_id"] == track_id]
            plt.plot(
                df_track.time * time_interval_minutes / 60,
                df_track[feature_col] * (yscale),
                color="black",
                linewidth=1.5,
            )

    ax.set_xlim(-1, 70)
    ax.set_ylim(ylim)
    if growth_outliers:
        ax.set_ylim(200, 2000)
    ax.set_ylabel(f"{ylabel} {yunits}")
    if colony_time is True:
        ax.set_xlabel("Colony Time (hr)")
    else:
        ax.set_xlabel("Real Time (hr)")

    if legend is True:
        add_custom_legend(color_map)
    save_and_show_plot(
        f"{figdir}/{feature_col}_trajectory_{colony}_{color_map}_colonytime_{colony_time}{extra_label}",
        file_extension=".pdf",
        bbox_inches="tight",
    )
    return fig, ax


def add_custom_legend(color_map):
    handles, labels = plt.gca().get_legend_handles_labels()
    if color_map == "colony" or color_map == "colony_gradient":
        small_line = matplotlib.lines.Line2D(
            [0], [0], color=COLONY_COLORS["small"], label="Small", linewidth=2
        )
        medium_line = matplotlib.lines.Line2D(
            [0], [0], color=COLONY_COLORS["medium"], label="Medium", linewidth=2
        )
        large_line = matplotlib.lines.Line2D(
            [0], [0], color=COLONY_COLORS["large"], label="Large", linewidth=2
        )
        handles.extend([small_line, medium_line, large_line])
        location = "upper right"
    elif color_map == "growth_outliers":
        line1 = matplotlib.lines.Line2D(
            [0],
            [0],
            color=GROWTH_OUTLIERS[False],
            label="Baseline full interphase trajectory",
            linewidth=2,
        )
        line2 = matplotlib.lines.Line2D(
            [0],
            [0],
            color=GROWTH_OUTLIERS["apoptosis"],
            label="Partial trajectory ending in apoptosis",
            linewidth=2,
        )
        line3 = matplotlib.lines.Line2D(
            [0],
            [0],
            color=GROWTH_OUTLIERS[True],
            label="Full interphase growth outlier",
            linewidth=2,
        )
        line4 = matplotlib.lines.Line2D(
            [0],
            [0],
            color=GROWTH_OUTLIERS["daughter"],
            label="2nd generation growth outlier",
            linewidth=2,
        )
        handles.extend([line1, line2, line3, line4])
        location = "upper left"

    plt.legend(handles=handles, loc=location)


def get_random_color_between(color1, color2):
    """
    Given two colors, pick a random color along the gradient between them

    Parameters
    ----------
    color1: str
        hex code for color 1
    color2: str
        hex code for color 2

    Returns
    -------
    str
        hex code for a color between color1 and color2
    """
    color1_tuple = colour.notation.HEX_to_RGB(color1)
    color2_tuple = colour.notation.HEX_to_RGB(color2)
    r = random.random()
    return colour.notation.RGB_to_HEX(r * color1_tuple + (1 - r) * color2_tuple)


def convert_duration_to_hours(single_feature_per_track_df, interval):
    """
    Convert duration_BC from frames to hours

    Parameters
    ----------
    single_feature_per_track_df: DataFrame
        DataFrame containing a single metric per track

    interval: float
        time interval in minutes

    Returns
    -------
    single_feature_per_track_df: DataFrame
        DataFrame with duration_BC converted to hours
    """
    single_feature_per_track_df["duration_BC_hr"] = (
        single_feature_per_track_df["duration_BC"] * interval / 60
    )
    return single_feature_per_track_df


def rolling_window_time_mean(rolling_window, single_feature_per_track_df, feature, time):
    """
    Plot the slope of a feature vs volume at B over a rolling window of time

    Parameters
    ----------
    rolling_window: Int
        Size of the rolling window in hours

    single_feature_per_track_df: DataFrame
        DataFrame containing a single feature per track

    figdir: str
        Directory to save figure

    feature: str
        Feature to plot against volume at B

    time: str
        Time column to roll through

    Returns
    -------
    Mean values and associate rolling window time points (given as the window midpoint)
    """

    colony_time = np.arange(
        single_feature_per_track_df[time].min(), single_feature_per_track_df[time].max(), 1
    )
    windows = np.lib.stride_tricks.sliding_window_view(colony_time, rolling_window)

    avg_list = []
    time_list = []

    for i, window in enumerate(windows):
        df_sub = single_feature_per_track_df[
            (single_feature_per_track_df[time] > window.min())
            & (single_feature_per_track_df[time] < window.max())
        ]
        avg_list.append(np.nanmean(df_sub[feature]))
        time_list.append(window.mean())

    return avg_list, time_list
