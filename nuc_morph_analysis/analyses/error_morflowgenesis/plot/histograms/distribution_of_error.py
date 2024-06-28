import matplotlib.pyplot as plt
import numpy as np
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot


fs = 20


def plot_distribution(
    df, error_type, feature_list=["volume", "height", "surface_area", "shape_modes"]
):
    """
    Plot the distribution of error.

    Parameters
    -----
    df: Dataframe
        The fixed control dataframe.
    error_type: string
        'absolute' or 'percent'
    feature_list: list of strings
        The features to plot the distribution of error for.
        The options are 'volume', 'surface_area', 'height', and 'shape_modes'.

    Returns
    -----
    figure
    """

    for feature in feature_list:
        if feature == "volume":
            column_names = [f"{error_type}_error_volume"]
            x_labels = ["Volume (µm³)"]
        if feature == "height":
            column_names = [f"{error_type}_error_height"]
            x_labels = ["Height (µm)"]
        if feature == "surface_area":
            column_names = [f"{error_type}_error_surface_area"]
            x_labels = ["Surface Area (µm²)"]
        if feature == "shape_modes":
            column_names = [
                f"{error_type}_error_NUC_PC1",
                f"{error_type}_error_NUC_PC2",
                f"{error_type}_error_NUC_PC3",
                f"{error_type}_error_NUC_PC4",
                f"{error_type}_error_NUC_PC5",
                f"{error_type}_error_NUC_PC6",
                f"{error_type}_error_NUC_PC7",
                f"{error_type}_error_NUC_PC8",
            ]
            x_labels = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"]

        for column_name, x_label in zip(column_names, x_labels):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            values = df[column_name].to_numpy()
            plt.hist(values, bins=50)
            plt.xlabel(f"{error_type.capitalize()} error for {x_label}", fontsize=fs)
            plt.ylabel("Number of timepoints", fontsize=fs)
            plt.xticks(fontsize=fs)
            plt.yticks(fontsize=fs)
            save_and_show_plot(f"error_morflowgenesis/figures/plot_distribution_{column_name}")

            stats = df[column_name].describe([0.95])
            print(stats)


def plot_error_distrubution_table(
    df, error_type, feature_list=["volume", "mesh_volume", "height", "surface_area", "shape_modes"]
):
    """
    Plot the distrubution of error as a histogram and as a box plot.
    Include a table with the statistics showing the median, 5th, 25th, 75th, 95th, and 99th percentiles.

    Parameters
    -----
    df: Dataframe
        The fixed control dataframe with error columns added
    error_type: string
        'absolute' or 'percent'
    feature_list: list of strings
        The features to plot the distribution of error for. Options: 'volume','height', 'surface_area', 'shape_modes'

    Returns
    -----
    figure
    """
    for feature in feature_list:
        if feature == "mesh_volume":
            column_names = [f"{error_type}_error_volume"]
            x_labels = ["Mesh Volume (µm³)"]
        if feature == "volume":
            column_names = [f"{error_type}_error_volume"]
            x_labels = ["Volume (µm³)"]
        if feature == "height":
            column_names = [f"{error_type}_error_height"]
            x_labels = ["Height (µm)"]
        if feature == "surface_area":
            column_names = [f"{error_type}_error_surface_area"]
            x_labels = ["Surface Area (µm²)"]
        if feature == "shape_modes":
            column_names = [
                f"{error_type}_error_NUC_PC1",
                f"{error_type}_error_NUC_PC2",
                f"{error_type}_error_NUC_PC3",
                f"{error_type}_error_NUC_PC4",
                f"{error_type}_error_NUC_PC5",
                f"{error_type}_error_NUC_PC6",
                f"{error_type}_error_NUC_PC7",
                f"{error_type}_error_NUC_PC8",
            ]
            x_labels = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"]

        for column_name, x_label in zip(column_names, x_labels):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            values = df[column_name].to_numpy()

            ax1.hist(values, bins=50)
            ax1.set_xlabel(
                f'{error_type.capitalize()} error for {x_label}\n Median: {"%.2f" % np.median(values)}',
                fontsize=fs,
            )
            ax1.set_ylabel("Number of timepoints", fontsize=fs)
            ax1.xaxis.set_tick_params(labelsize=fs)
            ax1.yaxis.set_tick_params(labelsize=fs)

            stats = df[column_name].describe([0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            stats = stats.to_frame()
            stats = stats.round(2)
            bbox = [0, 0, 1, 1]
            ax2.axis("off")
            mpl_table = ax2.table(
                cellText=stats.values, rowLabels=stats.index, bbox=bbox, colLabels=stats.columns
            )
            mpl_table.auto_set_font_size(False)
            mpl_table.set_fontsize(12)
            plt.tight_layout()
            save_and_show_plot(f"error_morflowgenesis/figures/plot_table_{column_name}")
