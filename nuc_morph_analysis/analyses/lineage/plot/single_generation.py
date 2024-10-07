import numpy as np
import matplotlib.pyplot as plt
from nuc_morph_analysis.analyses.lineage.get_features.confidence_interval import confidence_interval
from nuc_morph_analysis.lib.visualization.notebook_tools import (
    save_and_show_correlation_plot,
    save_and_show_plot,
)
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.analyses.lineage.get_features import between_pairs
from nuc_morph_analysis.analyses.volume_variation import plot_features
import scipy.stats as stats
import matplotlib
import seaborn as sb
import scipy.stats as spstats

matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"


def plot_corr(
    df,
    relationship_axis,
    feature,
    generations=1,
    control=False,
    col_threshold=None,
    values=False,
    figdir=None,
    unity_line=True,
):
    """
    Plot the coorelation between features of related cell a specified generation apart for several growth features.
    Plot the coorelation between features of unrelated cells that are born within 10 minutes of eachother
    Adding a distance threshold also controls for space (60 um)
    Adding a volume threshold controls for similar size at B

    Parameters
    ----------
    df: Dataframe
        The dataframe returned from lineage_pairs.get_related_pairs_dataset()

    relationship_axis: String
        'depth' (ie. mother daughter, grandmother daughter)
        'width' (ie. sister, cousin)

    generations: Int
        Int with value for what generation (depth/width) is desired for plotting.
        Default to the first generation mother / daughter and siblings.

    feature: String
        Feature or metric you want to plot (column name)

    control: Boolean
        Whether or not the dataframe is a control dataframe of unrelated pairs

    col_threshold: list of tuples
        [('distance',60), ('volume_at_b',65)]

    values: Boolean
        If False, will display scatter plot (Default)
        If True, will return the correlation and confidence intervals values

    figdir: String
        Relative path within nuc_morph_analysis/analyses folder. Must include '/figures/'. May
        include an initial prefix of the filename.

    unity_line: Boolean
        If True, will plot a unity line on the plot.

    Returns
    -------
    corr: Float
        Spearmann correlation value for feature

    percent: Array
        Array containing the lower and upper bounds of the 95% confidence intervals

    displays figure
    """

    df, column_1, color, marker, axis_1, axis_2 = get_relationship_information(
        df, relationship_axis, generations, control
    )
    column_2, column_3, x_label, y_label, lim = get_feature_information(feature, axis_1, axis_2)

    if feature == "volume_at_B":
        lim = [390, 710]
    if feature == "volume_at_C":
        lim = [750, 1410]

    if control is False:
        cols = [column_1, column_2, column_3]
        df = df[df[column_1] != 0]
        df = df[df[column_1] == generations]
        label = None
        control_fname = ""

    if control is True:
        cols = [column_2, column_3]
        color, x_label, y_label, label, df, control_fname = get_control_information(
            df, x_label, y_label, relationship_axis, col_threshold
        )

    df = df.dropna(subset=cols)

    nas = np.logical_or(np.isnan(df[column_3]), np.isnan(df[column_2]))
    df_pearson = df[~nas]

    pearson, p_pvalue = spstats.pearsonr(df_pearson[column_3], df_pearson[column_2])

    # corr = df[column_2].corr(df[column_3], method='spearman')
    corr = pearson
    _, _, _, percent = confidence_interval(
        df, n_resamples=500, column_2=column_2, column_3=column_3
    )
    percent = np.round(percent, 2)

    if values is True:
        return corr, percent
    else:
        plt.figure(figsize=(4, 4), dpi=300)
        plt.scatter(
            df[column_3],
            df[column_2],
            alpha=0.50,
            color=color,
            marker=marker,
            label=label,
            edgecolors="none",
        )

        if unity_line:
            plt.plot(lim, lim, color="black", linestyle="--", alpha=0.2, zorder=0)
            
        if p_pvalue < 0.001:
            pstring = "P<.001"
        else:
            pstring = str(np.round(p_pvalue, 3))
            pstring = f"P={pstring[1:]}"

        plt.title(
            f'r={"%.2f" % pearson} {pstring}, N={len(df)} ({percent[0]}, {percent[1] })'
        )
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        xticks = plt.xticks()[0]  # Get x-ticks
        plt.yticks(xticks)  # Set y-ticks to be the same as x-ticks
        plt.xticks(xticks)  # Set x-ticks to not be optimized
        plt.xlim(lim)
        plt.ylim(lim)
        if label is not None:
            plt.legend(fontsize=8, markerscale=0, handletextpad=-2, loc="upper left", frameon=False)
        plt.tight_layout()
        save_and_show_correlation_plot(
            figdir, relationship_axis, feature, generations, control_fname
        )


def get_relationship_information(df, relationship_axis, generations, control):
    """
    Get the relationship information for the plot.

    Parameters
    ----------
    df: Dataframe
        The dataframe returned from lineage_pairs.get_related_pairs_dataset()
    relationship_axis: String
        'depth' (ie. mother daughter, grandmother daughter)
        'width' (ie. sister, cousin)
    generations: Int
    control: Boolean

    Returns:
    -------
    df: Dataframe
        The dataframe for plotting
    column_1: String
        The column name for the relationship axis
    color: String
        hex color for the plot
    marker: String
        marker for the plot
    axis_1: String
        The label for the x-axis
    axis_2: String
        The label for the y-axis
    """
    relationship_dict = {
        "depth": {
            "column_1": "delta_depth",
            "color": "#0076F8",
            "marker": "^",
            "generations": {
                1: ("Mother", "Daughter"),
                2: ("Grandmother", "Daughter"),
                3: ("Great Grandmother", "Daughter"),
            },
        },
        "width": {
            "column_1": "cousiness",
            "color": "#FF3940",
            "marker": "o",
            "generations": {
                1: ("Sister 2", "Sister 1"),
                2: ("Cousin 2", "Cousin 1"),
                3: ("Second Cousin 2", "Second Cousin 1"),
            },
        },
    }

    column_1 = relationship_dict[relationship_axis]["column_1"]
    color = relationship_dict[relationship_axis]["color"]
    marker = relationship_dict[relationship_axis]["marker"]
    axis_1, axis_2 = relationship_dict[relationship_axis]["generations"][generations]

    if relationship_axis == "depth" and not control:
        df = df[df["same_branch"] == True]

    return df, column_1, color, marker, axis_1, axis_2


def get_feature_information(feature, axis_1, axis_2):
    """
    Get the column names, labels, and limits for the feature being plotted. Convert them to be compatible with the lineage dataframe.

    Parameters
    ----------
    feature: String
        Feature or metric you want to plot (column name)
    axis_1: String
        The label for the x-axis (ie. mother, sister, cousin)
    axis_2: String
        The label for the y-axis (ie. daughter, sister, cousin)

    Returns
    -------
    column_2: String
        The column name for the x-axis feature updated with tid
    column_3: String
        The column name for the y-axis feature updated with tid
    x_label: String
        The label for the x-axis
    y_label: String
        The label for the y-axis
    lim: Tuple
        The limits for the x and y axis
    """

    _, label, units, lim = get_plot_labels_for_metric(feature)
    column_2 = f"tid1_{feature}"
    column_3 = f"tid2_{feature}"
    x_label = f"{axis_1}\n{label} {units}"
    y_label = f"{axis_2}\n{label} {units}"
    return column_2, column_3, x_label, y_label, lim


def get_control_information(df, x_label, y_label, relationship_axis, col_threshold):
    """
    Customize the label based on what you are controlling for. Threshold the dataframe for distance.
    Set plot labels for the control plot.

    Parameters
    ----------
    df: Dataframe
        The dataframe for plotting
    relationship_axis: String
        'depth' (ie. mother daughter, grandmother daughter)
        'width' (ie. sister, cousin)
    x_label: String
    ylabel: String
    distance_threshold: Int
        Default is None, which results in no distance threshold
        Set in um
    volume_threshold: Int
        Default is None, which results in no volume threshold
        Set in µm³

    Returns
    -------
    color: String
        hex color for the control plot
    x_label: String
        The label for the x-axis of the control plot
    y_label: String
        The label for the y-axis of the control plot
    label: String
        The label for the control plot
    df: Dataframe
        The dataframe for plotting with threshold applied
    """
    color = "#808080"
    x_label = f"Control {x_label}"
    y_label = f"Control {y_label}"

    label_dict = {"width": "Born within 10 min", "depth": "Breakdown and formation within 60 min"}
    label = label_dict.get(relationship_axis, "")
    control_fname = ""

    if col_threshold is not None:
        for feature, threshold in col_threshold:
            df = df[df[feature] < threshold]
            _, name, units, _ = get_plot_labels_for_metric(feature)
            label += f"\n{name} < {threshold} {units}"
            control_fname += f"_{feature}_{threshold}"

    return color, x_label, y_label, label, df, control_fname


def plot_distance_between_sister_pairs(df, figdir=None):
    """
    Plot distribution of distances between sister pairs at transition (B)

    Parameters
    ----------
    df: Dataframe
        The dataframe returned from lineage_pairs.get_related_pairs_dataset()

    figdir: String
        Relative path within nuc_morph_analysis/analyses folder.

    Returns
    -------
    Saves and displays the figure
    """
    df, column_1, color, _, _, _ = get_relationship_information(
        df, relationship_axis="width", generations=1, control=False
    )
    cols = [column_1]
    df = df[df[column_1] != 0]
    df = df[df[column_1] == 1]
    df = df.dropna(subset=cols)
    df = between_pairs.get_distance(df, "B", "B")

    fig = plt.figure(figsize=(5, 4))
    plt.hist(df["distance"], bins=20, color=color)
    plt.title(f"Mean: {df['distance'].mean():.2f} µm, Std: {df['distance'].std():.2f} µm")
    plt.xlabel("Distance between sisters at transition (B) (µm)")
    plt.ylabel(f"Count (total N={len(df)})")
    plt.tight_layout()
    save_and_show_plot(figdir + "distance_between_sister_pairs", bbox_inches="tight")


def plot_transition_time_difference_between_sister_pairs(df, figdir=None):
    """
    Plot distribution of differences between sister pairs
    in transition time

    Parameters
    ----------
    df: Dataframe
        The dataframe returned from lineage_pairs.get_related_pairs_dataset()

    figdir: String
        Relative path within nuc_morph_analysis/analyses folder.

    Returns
    -------
    Saves and displays the figure
    """
    df, column_1, color, _, _, _ = get_relationship_information(
        df, relationship_axis="width", generations=1, control=False
    )
    cols = [column_1]
    df = df[df[column_1] != 0]
    df = df[df[column_1] == 1]
    df = df.dropna(subset=cols)
    df = between_pairs.add_difference_between(df)
    scale, _, _, _ = get_plot_labels_for_metric("time_at_B")
    feature = "difference_time_at_B"
    df[feature] = df[feature] * scale * 60

    fig = plt.figure(figsize=(5, 4))
    plt.hist(df[feature], bins=20, color=color)
    plt.title(f"Mean: {df[feature].mean():.2f} min, Std: {df[feature].std():.2f} min")
    plt.xlabel("Difference in transition time between sisters (min)")
    plt.ylabel(f"Count (total N={len(df)})")
    plt.tight_layout()
    save_and_show_plot(figdir + "transition_time_diff_between_sister_pairs", bbox_inches="tight")


def plot_volume_difference_between_sister_pairs(df, figdir=None):
    """
    Plot distribution of distances between sister pairs at transition (B)

    Parameters
    ----------
    df: Dataframe
        The dataframe returned from lineage_pairs.get_related_pairs_dataset()

    figdir: String
        Relative path within nuc_morph_analysis/analyses folder.

    Returns
    -------
    Saves and displays the figure
    """
    df, column_1, color, _, _, _ = get_relationship_information(
        df, relationship_axis="width", generations=1, control=False
    )
    cols = [column_1]
    df = df[df[column_1] != 0]
    df = df[df[column_1] == 1]
    df = df.dropna(subset=cols)
    df = between_pairs.get_distance(df, "B", "B")

    fig = plt.figure(figsize=(5, 4))
    plt.hist(df["difference_volume_at_B"], bins=20, color=color)
    plt.title(
        f"Mean: {df['difference_volume_at_B'].mean():.2f} µm³, Std: {df['difference_volume_at_B'].std():.2f} µm³"
    )
    plt.xlabel("Difference between sisters volume at transition (B) (µm³)")
    plt.ylabel(f"Count (total N={len(df)})")
    plt.tight_layout()
    save_and_show_plot(figdir + "volume_difference_at_B_between_sister_pairs", bbox_inches="tight")


def scatter_avg_sister_pairs(df, feature1, feature2, figdir):
    """
    Plot distribution of distances between sister pairs at transition (B)

    Parameters
    ----------
    df: Dataframe
        The dataframe returned from lineage_pairs.get_related_pairs_dataset()

    figdir: String
        Relative path within nuc_morph_analysis/analyses folder.

    Returns
    -------
    Saves and displays the figure
    """
    df, column_1, color, _, _, _ = get_relationship_information(
        df, relationship_axis="width", generations=1, control=False
    )
    cols = [column_1]
    df = df[df[column_1] != 0]
    df = df[df[column_1] == 1]
    df = df.dropna(subset=cols)

    df[f"avg_sister_{feature1}"] = (df[f"tid1_{feature1}"] + df[f"tid2_{feature1}"]) / 2
    df[f"avg_sister_{feature2}"] = (df[f"tid1_{feature2}"] + df[f"tid2_{feature2}"]) / 2
    plot_features.scatter_plot(
        df,
        colony="all_baseline",
        column_1=f"avg_sister_{feature1}",
        column_2=f"avg_sister_{feature2}",
        color_map="colony",
        figdir=figdir,
        color_column="tid1_colony_time_at_B",
    )


def get_kl_divergence(feature_1, feature_2):
    """
    Calculate the KL divergence for two sets of feature arrays. This requires the datasets
    to be the same size, so first we calculate the density histogram of each and then calculate the KL divergence.

    Parameters
    ----------
    feature_1: Series
        The first feature (ie df_all['volume_at_B'])
    feature_2: Series
        The second feature (ie df_sub['volume_at_B'])

    Returns
    -------
    kl12: Float
        KL divergence from feature_1 to feature_2
    kl21: Float
        KL divergence from feature_2 to feature_1
    average_kl: Float
        The average of kl12 and kl21
    """
    # Define the bins
    bins = np.linspace(
        min(feature_1.min(), feature_1.min()), max(feature_2.max(), feature_2.max()), num=50
    )
    # Calculate the histogram of each
    hist_1, _ = np.histogram(feature_1.values, bins=bins, density=True)
    hist_2, _ = np.histogram(feature_2.values, bins=bins, density=True)
    # Add a small value to avoid division by zero
    hist_1 += 1e-10
    hist_2 += 1e-10
    # Calculate the KL divergence
    kl12 = stats.entropy(hist_1, hist_2)
    kl21 = stats.entropy(hist_2, hist_1)
    average_kl = (kl12 + kl21) / 2

    return kl12, kl21, average_kl


def sum_feature_density(df_all, df, feature, figdir, add_kl_divergence=False):
    """
    Plot the feature density for a feature for the entire dataset and a subset of the dataset.

    Parameters
    ----------
    df_all: Dataframe
        The entire dataset track_level_feature_df
    df: Dataframe
        The related_pairs dataset
    feature: String
        Feature or metric you want to plot (column name)
    figdir: String
        Relative path within nuc_morph_analysis/analyses folder.
    histtype: String
        'step' or 'bar'
    add_kl_divergence: Boolean
        If True, will add the KL divergence to the plot
    plot_all: Boolean
        If True, will plot the entire dataset in black
    """
    df, column_1, color, _, _, _ = get_relationship_information(
        df, relationship_axis="width", generations=1, control=False
    )
    cols = [column_1]
    df = df[df[column_1] != 0]
    df = df[df[column_1] == 1]
    df = df.dropna(subset=cols)
    new_col = f"avg_sister_{feature}"
    df[new_col] = (df[f"tid1_{feature}"] + df[f"tid2_{feature}"]) / 2

    fig = plt.figure(figsize=(5, 5))

    def calc_stats(df, feature):
        mean = np.mean(df[feature])
        cov = np.std(df[feature]) / mean
        return mean, cov

    dataframes = [
        (df_all, feature, "black", "Small and Medium Datasets"),
        (df, new_col, "tab:blue", "Average sisters"),
    ]

    for df, feature, color, label_prefix in dataframes:
        mean, cov = calc_stats(df, feature)
        label = f"{label_prefix}, N={len(df)}, Mean={mean:.2f}, COV={cov:.2f}"
        ax = sb.kdeplot(df[feature], color=color, label=label)

    _, label, units, lim = get_plot_labels_for_metric(feature)
    plt.xlabel(f"{label} {units}")
    plt.ylabel("Density")
    plt.legend(bbox_to_anchor=(0.5, 1.1), loc="center")

    if add_kl_divergence:
        kl1, kl2, average_kl = get_kl_divergence(df_all[feature], df[new_col])
        print(f"KL divergence sym, {kl1:.2f}, {kl2:.2f}")
        plt.text(
            0.98,
            0.98,
            f"Avg KL divergence: {average_kl:.2f}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=plt.gca().transAxes,
        )

    save_and_show_plot(f"{figdir}{feature}_density_plot", figure=fig, bbox_inches="tight")


def difference_in_volume_at_B_vs_feature(df, feature, figdir):
    """
    Plot the difference in volume at B for all sister pairs compared to the difference in another of thier growth features.

    Parameters
    ----------
    df: Dataframe
        Related_pairs
    feature: Str
        Column name to take the difference of
    figdir:
        Relative path within nuc_morph_analysis/analyses folder to save the figure in.
    """

    df, column_1, color, _, _, _ = get_relationship_information(
        df, relationship_axis="width", generations=1, control=False
    )
    cols = [column_1]
    df = df[df[column_1] != 0]
    df = df[df[column_1] == 1]
    df = df.dropna(subset=cols)
    df[f"difference_{feature}"] = abs(df[f"tid1_{feature}"] - df[f"tid2_{feature}"])
    plot_features.scatter_plot(
        df,
        colony="all_baseline",
        column_1="difference_volume_at_B",
        column_2=f"difference_{feature}",
        color_map="#808080",
        figdir=figdir,
        fitting=True,
    )
