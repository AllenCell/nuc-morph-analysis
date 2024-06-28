import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from nuc_morph_analysis.analyses.height.calculate_features import calculate_mean_height
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.reference_points import COLONY_COLORS, COLONY_LABELS

matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"


def height_colony_time_alignment(
    df,
    pixel_size,
    interval,
    time_axis="real_time",
    show_legend=False,
    error="percentile",
    figdir="height/figures",
):
    """
    Plot the mean nuclear height across the colony over time for each colony. This is done in real time and colony time.

    Parameters
    ----------
    df: Dataframe
        Dataframe containing all colonys

    pixel_size: float
        Pixel size in microns.

    interval: float
        Time interval in minutes

    time_axis: str
        "real_time" or "colony_time"

    show_legend: bool
        Whether to show the legend or not

    error: str
        "std" or percentile

    Returns
    -------
    Plot of mean nuclear height across the colony over time for each colony.
    """
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    for colony, df_colony in df.groupby("colony"):
        df_colony = df_colony.sort_values("index_sequence")
        mean_height, std_height = calculate_mean_height(df_colony, pixel_size)
        color = COLONY_COLORS[colony]

        if time_axis == "real_time":
            time = df_colony.index_sequence.unique() * interval / 60
            x_label = "Real Time (hr)"
        if time_axis == "colony_time":
            time = df_colony.colony_time.unique() * interval / 60
            x_label = "Aligned Colony Time (hr)"
        if error == "std":
            upper = np.array(mean_height) + np.array(std_height)
            lower = np.array(mean_height) - np.array(std_height)
        if error == "percentile":
            grouper = df_colony[["index_sequence"] + ["height"]].groupby("index_sequence")["height"]
            lower = grouper.quantile(0.05) * pixel_size
            upper = grouper.quantile(0.95) * pixel_size

        ax.fill_between(
            time,
            lower,
            upper,
            alpha=0.12,
            color=color,
            zorder=0,
            edgecolor="none",
            label=COLONY_LABELS[colony],
        )
        ax.plot(
            time, mean_height, linewidth=1.2, color=color, label=COLONY_LABELS[colony], zorder=20
        )

    ax.set_ylim(3.5, 11.75)
    ax.set_ylabel("Average Nuclear Height \n Across Colony (µm)")
    ax.set_xlabel(x_label)
    if show_legend is True:
        ax.legend(loc="upper right", handletextpad=0.7, frameon=False)
    plt.tight_layout()
    save_and_show_plot(
        f"{figdir}/avg_height_colony_{time_axis}_alignment",
        file_extension=".pdf",
        dpi=300,
        transparent=True,
    )


def calculate_mean_density(df, scale):
    """
    Calculate the mean height for a given index_sequence (i.e. timepoint) and the standard deviation of the mean.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    pixel_size : float
        Pixel size in microns.

    Returns
    -------
    mean_height : list
        List of mean heights for each index_sequence.
    standard_dev_height : list
        List of standard deviations of the mean heights for each index_sequence.
    """
    mean = []
    standard_dev = []
    for _, df_frame in df.groupby("index_sequence"):
        density = df_frame.density * scale
        mean.append(np.mean(density))
        standard_dev.append(np.std(density))
    return mean, standard_dev


def density_colony_time_alignment(
    df,
    pixel_size,
    interval,
    time_axis="real_time",
    show_legend=False,
    error="percentile",
    figdir="height/figures",
):
    """
    Plot the mean nuclear height across the colony over time for each colony. This is done in real time and colony time.

    Parameters
    ----------
    df: Dataframe
        Dataframe containing all colonys

    pixel_size: float
        Pixel size in microns.

    interval: float
        Time interval in minutes

    time_axis: str
        "real_time" or "colony_time"

    show_legend: bool
        Whether to show the legend or not

    error: str
        "std" or percentile

    Returns
    -------
    Plot of mean nuclear height across the colony over time for each colony.
    """
    scale, label, units, _ = get_plot_labels_for_metric("density")
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    for colony, df_colony in df.groupby("colony"):
        df_colony = df_colony.sort_values("index_sequence")
        mean_density, std_density = calculate_mean_density(df_colony, scale)
        color = COLONY_COLORS[colony]

        if time_axis == "real_time":
            time = df_colony.index_sequence.unique() * interval / 60
            x_label = "Real Time (hr)"
        if time_axis == "colony_time":
            time = df_colony.colony_time.unique() * interval / 60
            x_label = "Aligned Colony Time (hr)"
        if error == "std":
            upper = np.array(mean_density) + np.array(std_density)
            lower = np.array(mean_density) - np.array(std_density)
        if error == "percentile":
            grouper = df_colony[["index_sequence"] + ["density"]].groupby("index_sequence")[
                "density"
            ]
            lower = grouper.quantile(0.05)
            upper = grouper.quantile(0.95)

        ax.fill_between(
            time,
            lower * scale,
            upper * scale,
            alpha=0.12,
            color=color,
            zorder=0,
            edgecolor="none",
            label=COLONY_LABELS[colony],
        )
        ax.plot(
            time, mean_density, linewidth=1.2, color=color, label=COLONY_LABELS[colony], zorder=20
        )

    ax.set_ylim(0.0005, 0.0065)
    ax.set_ylabel(f"Average Density \n Across Colony {units}")
    ax.set_xlabel(x_label)
    if show_legend is True:
        ax.legend(loc="upper right", handletextpad=0.7, frameon=False)
    plt.tight_layout()
    save_and_show_plot(
        f"{figdir}/avg_density_colony_{time_axis}_alignment",
        file_extension=".pdf",
        dpi=300,
        transparent=True,
    )
