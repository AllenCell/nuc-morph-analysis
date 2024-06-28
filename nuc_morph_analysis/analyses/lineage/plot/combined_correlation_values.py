import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.analyses.lineage.plot import single_generation
import matplotlib
import numpy as np
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric

matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"


def plot(
    df_lineage,
    df_control_s,
    df_control_md,
    feature_list=["duration_BC_hr", "tscale_linearityfit_volume", "volume_at_B", "volume_at_C"],
    sister_col_threshold=None,
    mother_daughter_col_threshold=None,
    figdir=None,
    relationship_axis=["width", "depth"],
    legend=True,
):
    """
    This function plots all of the growth correlations for related pairs (ie mother-daughter and sister)
    and the control correlations (ie unrelated pairs) in a single figure.

    Parameters
    ----------
    df_lineage: Dataframe
        dataframe containing related pairs
        from nuc_morph_analysis/analyses/lineage/dataset/lineage_pairs_dataset.py
    df_control_s: Dataframe
        dataframe containing unrelated pairs controlling for sister pairs
    df_control_md: Dataframe
        dataframe containing unrelated pairs controlling for mother daughter pairs
    feature_list: List
        List of features to plot.
    sister_col_threshold: List
        List of tuples containing the column and threshold for the sister control dataframe
    mother_daughter_col_threshold: List
        List of tuples containing the column and threshold for the mother daughter control dataframe
    relationship_axis: List or str
        Relationship axis or axes to plot. Can be 'width', 'depth', or ['width', 'depth'].
    legend:
        Flag for whether to include legend

    Returns
    -------
    Figure
    """
    if len(relationship_axis) == 1:
        plt.figure(figsize=(len(feature_list) + 0.2, 5.25), dpi=300)
    elif len(relationship_axis) == 2:
        if legend:
            plt.figure(figsize=(len(feature_list) + 0.2, 6.25), dpi=300)
        else:
            plt.figure(figsize=(len(feature_list) + 0.2, 5), dpi=300)
    x_ticks_labels = [f"{get_plot_labels_for_metric(feature)[1]}" for feature in feature_list]
    x_ticks = np.arange(0, len(feature_list), 1)

    for axis in relationship_axis:
        for x_tick, feature in zip(x_ticks, feature_list):
            if axis == "width":
                corr, percent = single_generation.plot_corr(
                    df_lineage, axis, feature=feature, values=True
                )
                add_correlation_to_plot(corr, percent, x_tick, axis, len(relationship_axis))

                corr, percent = single_generation.plot_corr(
                    df_control_s,
                    axis,
                    feature=feature,
                    values=True,
                    control=True,
                    col_threshold=sister_col_threshold,
                )
                add_correlation_to_plot(
                    corr,
                    percent,
                    x_tick,
                    axis,
                    len(relationship_axis),
                    control=True,
                    sister_col_threshold=sister_col_threshold,
                )

            elif axis == "depth":
                corr, percent = single_generation.plot_corr(
                    df_lineage, axis, feature=feature, values=True
                )
                add_correlation_to_plot(corr, percent, x_tick, axis, len(relationship_axis))

                corr, percent = single_generation.plot_corr(
                    df_control_md,
                    axis,
                    feature=feature,
                    values=True,
                    control=True,
                    col_threshold=mother_daughter_col_threshold,
                )
                add_correlation_to_plot(
                    corr,
                    percent,
                    x_tick,
                    axis,
                    len(relationship_axis),
                    control=True,
                    mother_daughter_col_threshold=mother_daughter_col_threshold,
                )

    if len(relationship_axis) == 1:
        plt.ylim((0, 0.75))
    if len(relationship_axis) == 2:
        plt.ylim((-0.3, 0.75))
        plt.axhline(y=0, color="black", linestyle="--", alpha=0.2, zorder=0)

    plt.ylabel("Correlation", fontsize=14)
    plt.xlim((-0.5, len(feature_list) - 0.5))
    plt.yticks(fontsize=14)
    plt.xticks(
        x_ticks, x_ticks_labels, rotation=65, fontsize=10, ha="right", rotation_mode="anchor"
    )
    plt.tight_layout()

    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(
            by_label.values(),
            by_label.keys(),
            markerscale=1.1,
            labelspacing=0.8,
            handletextpad=0,
            bbox_to_anchor=(0.5, 1.15),
            loc="center",
            ncol=2,
            columnspacing=0.7,
            fontsize=8,
        )

    control_fname = ""
    if sister_col_threshold is not None:
        for feature, threshold in sister_col_threshold:
            control_fname += f"_s_{feature}_{threshold}"
    if mother_daughter_col_threshold is not None:
        for feature, threshold in mother_daughter_col_threshold:
            control_fname += f"_md_{feature}_{threshold}"
    save_and_show_plot(f"{figdir}/combined_correlation_plot{control_fname}", bbox_inches="tight")
    plt.close()


def add_correlation_to_plot(
    corr,
    percent,
    x_tick,
    relationship_axis,
    num_axes,
    control=False,
    sister_col_threshold=None,
    mother_daughter_col_threshold=None,
):
    """
    Sets the color, marker and legend for each relationship plotted.
    Adds the correlation and error bars using the confidence intervals.

    Paramaters
    ----------
    corr: Float
        correlation value
    percent: List
        confidence interval
    x_tick: Int
        x axis location
    relationship_axis: String
        'depth' or 'width'
    control: Boolean
        Whether or not it is the related pairs or control dataframe

    Returns
    -------
    Datapoint on plot
    """

    if relationship_axis == "depth":
        color = "#0076F8"
        shift = 0.06
        label = "Mother-Daughter"
        facecolor = color
        marker = "^"

    if relationship_axis == "width":
        color = "#FF3940"
        shift = -0.06 if num_axes > 1 else 0
        label = "Sister1-Sister2"
        facecolor = color
        marker = "o"

    if control is True:
        if relationship_axis == "depth":
            label = "Control Mother-Daughter:\nBreakdown within 60 min"
            marker = "^"
            shift = 0.06 if num_axes > 1 else 0
            color = "#808080"
            facecolor = "#808080"

            if mother_daughter_col_threshold is not None:
                for feature, threshold in mother_daughter_col_threshold:
                    _, name, units, _ = get_plot_labels_for_metric(feature)
                    label += f"\n{name} < {threshold} {units}"

        if relationship_axis == "width":
            color = "#808080"
            label = "Control Sister1-Sister2:\nBorn within 10 min"
            marker = "o"
            shift = -0.06
            color = "#808080"
            facecolor = "#808080"

            if sister_col_threshold is not None:
                for feature, threshold in sister_col_threshold:
                    _, name, units, _ = get_plot_labels_for_metric(feature)
                    label += f"\n{name} < {threshold} {units}"

    ytop = percent[1] - corr
    ybot = corr - percent[0]
    errorbars = [[ytop], [ybot]]
    x = x_tick + shift

    plt.errorbar(x, corr, yerr=errorbars, color=color, zorder=2, capsize=3.1, lw=2.5)
    plt.scatter(
        x,
        corr,
        color=color,
        marker=marker,
        s=100,
        facecolors=facecolor,
        label=label,
        zorder=4,
        edgecolors="none",
    )
