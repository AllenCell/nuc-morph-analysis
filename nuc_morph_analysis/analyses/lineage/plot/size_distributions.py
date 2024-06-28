import matplotlib.pyplot as plt
import numpy as np
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot


def plot_size_distributions(track_level_feature_df, feature_list, figdir):
    """
    Plot the size distributions for nuclei at timepoints B and C. Can also be generalized to plot the distribution of any feature.

    Parameters
    ----------
    track_level_feature_df: DataFrame
        DataFrame containing the track level features
    feature_list: List
        List of features for which the size distributions need to be plotted
    figdir: str
        Directory where the figures will be saved
    """
    for feature in feature_list:
        scale, label, units, lims = get_plot_labels_for_metric(feature)
        std = np.nanstd(track_level_feature_df[feature] * scale)
        mean = np.nanmean(track_level_feature_df[feature] * scale)
        cv = std / mean
        n = len(track_level_feature_df[feature])
        plt.hist(
            track_level_feature_df[feature] * scale,
            bins=20,
            color="#808080",
            label=f"Mean={mean:.2f}\nStdev={std:.2f}\nCV: {cv:.2f}\nN={n}",
        )
        plt.xlabel(f"{label} {units}")
        plt.ylabel("Counts")
        plt.legend()
        save_and_show_plot(f"{figdir}/{feature}_size_distribution")
