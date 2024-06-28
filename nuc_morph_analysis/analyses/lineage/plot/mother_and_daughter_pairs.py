from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.analyses.volume_variation.plot_features import confidence_interval
from nuc_morph_analysis.analyses.lineage.plot.single_generation import get_kl_divergence
from scipy import stats as spstats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sb
from scipy import stats as spstats
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"


def get_mother_and_daughter_pairs(df_full, pixel_size):
    """
    Get mother and daughter pairs from the full dataset. Make a dataframe with key features for each.
    Find where the mother's volume at C is equal to the sum of the daughters' volume at B. Volume columns
    are already in microns^3. Time columns are in frames and need to be converted to minutes downstream.

    Parameters
    ----------
    df_full : pd.DataFrame
        The full dataset with all timepoints and tracks.
    pixel_size : float
        The pixel size in microns.

    Returns
    -------
    df_mother_and_daughter_pairs : pd.DataFrame
    """

    df_mother_and_daughter_pairs = pd.DataFrame(
        columns=[
            "pid",
            "tid1",
            "tid2",
            "mothers_volume_at_C",
            "sum_sisters_volume_at_B",
            "transition1",
            "transition2",
            "crossing_index",
        ]
    )
    grouped = df_full[df_full["parent_id"].isin(df_full["track_id"])].groupby("parent_id")
    valid_pids = grouped.filter(lambda x: x["track_id"].nunique() > 1 and x.name not in [-1])[
        "parent_id"
    ].unique()

    for pid in valid_pids:
        dft = grouped.get_group(pid)
        tid1, tid2 = dft["track_id"].unique()[:2]

        dfp = df_full[df_full["track_id"] == pid]
        dft1 = df_full[df_full["track_id"] == tid1]
        dft2 = df_full[df_full["track_id"] == tid2]

        x = dfp.index_sequence.values
        y = dfp.volume.values
        x1 = dft1.index_sequence.values
        y1 = dft1.volume.values
        x2 = dft2.index_sequence.values
        y2 = dft2.volume.values
        common_x = np.intersect1d(x1, x2)
        y1_common = y1[np.isin(x1, common_x)]
        y2_common = y2[np.isin(x2, common_x)]
        y_sum = y1_common + y2_common

        mothers_volume_at_C = dfp.volume_at_C.unique()[0]

        transition1 = dft1.frame_transition.unique()[0]
        transition2 = dft2.frame_transition.unique()[0]
        sum_sisters_volume_at_B = dft1.volume_at_B.unique()[0] + dft2.volume_at_B.unique()[0]

        crossing_index = np.where(np.diff(np.sign(mothers_volume_at_C - y_sum * pixel_size**3)))[0]
        crossing = common_x[crossing_index[0]] if crossing_index.size > 0 else np.nan

        df_mother_and_daughter_pairs = pd.concat(
            [
                df_mother_and_daughter_pairs,
                pd.DataFrame(
                    {
                        "pid": [pid],
                        "tid1": [tid1],
                        "tid2": [tid2],
                        "mothers_volume_at_C": [mothers_volume_at_C],
                        "sum_sisters_volume_at_B": [sum_sisters_volume_at_B],
                        "transition1": [transition1],
                        "transition2": [transition2],
                        "crossing": [crossing],
                        "difference_in_timing_avg": [crossing - ((transition1 + transition2) / 2)],
                        "difference_in_timing1": [crossing - transition1],
                        "difference_in_timing2": [crossing - transition2],
                        "half_mothers_volume_at_C": [mothers_volume_at_C / 2],
                        "tid1_volume_at_B": [dft1.volume_at_B.unique()[0]],
                        "tid2_volume_at_B": [dft2.volume_at_B.unique()[0]],
                        "difference_half_vol_at_C_and_tid1_B": [
                            mothers_volume_at_C / 2 - dft1.volume_at_B.unique()[0]
                        ],
                        "difference_half_vol_at_C_and_tid2_B": [
                            mothers_volume_at_C / 2 - dft2.volume_at_B.unique()[0]
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )

    return df_mother_and_daughter_pairs


def plot_mother_and_daughter_trajectories(
    df_full, df_mother_and_daughter_pairs, interval, figdir, add_track_info=False
):
    """
    Plots trajectories for the mother, daughters, and the sum of the daughters' volumes, as well as
    key points of interest for each mother-daughter pair.

    Parameters
    ----------
    df_full : pd.DataFrame
        The full dataset with all timepoints and tracks.
    df_mother_and_daughter_pairs : pd.DataFrame
        The mother-daughter pairs dataframe.
    interval : float
        The time interval between frames in minutes.
    figdir : str
        The directory to save the figures.
    """
    for _, row in df_mother_and_daughter_pairs.iterrows():
        df_mother = df_full[df_full["track_id"] == row["pid"]]
        df_sister1 = df_full[df_full["track_id"] == row["tid1"]]
        df_sister2 = df_full[df_full["track_id"] == row["tid2"]]

        scale, label, units, _ = get_plot_labels_for_metric("volume")

        x = df_mother.index_sequence.values
        y = df_mother.volume.values
        x1 = df_sister1.index_sequence.values
        y1 = df_sister1.volume.values
        x2 = df_sister2.index_sequence.values
        y2 = df_sister2.volume.values
        common_x = np.intersect1d(x1, x2)
        y1_common = y1[np.isin(x1, common_x)]
        y2_common = y2[np.isin(x2, common_x)]
        y_sum = y1_common + y2_common

        fig, ax = plt.subplots(figsize=(8, 5))

        if add_track_info:
            plt.plot(x * interval, y * scale, label=f"Mother: {row['pid']}")
            plt.plot(x1 * interval, y1 * scale, label=f"Daughter 1: {row['tid1']}", alpha=0.7)
            plt.plot(x2 * interval, y2 * scale, label=f"Daughter 2: {row['tid2']}", alpha=0.7)

        else:
            plt.plot(x * interval, y * scale, label=f"Mother")
            plt.plot(x1 * interval, y1 * scale, label=f"Daughter 1", alpha=0.7)
            plt.plot(x2 * interval, y2 * scale, label=f"Daughter 2", alpha=0.7)

        plt.plot(common_x * interval, y_sum * scale, label=f"Sum of Daughters")
        plt.axhline(
            row["mothers_volume_at_C"],
            color="tab:blue",
            linestyle="--",
            label=f"Mother Volume at C",
            alpha=0.5,
        )
        plt.axvline(
            row["transition1"] * interval,
            color="tab:orange",
            linestyle="--",
            label=f"Daughter 1 Time at B",
            alpha=0.5,
        )
        plt.axvline(
            row["transition2"] * interval,
            color="tab:green",
            linestyle="--",
            label=f"Daughter 2 Time at B",
            alpha=0.5,
        )
        plt.scatter(
            row["crossing"] * interval,
            row["mothers_volume_at_C"],
            color="black",
            label="Mother Volume at C = Sum of Daughter Volumes",
            s=50,
            zorder=10,
            edgecolors="none",
        )

        plt.xlabel("Movie Time (min)", fontsize=14)
        plt.ylabel(f"{label} {units}", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
        plt.legend(loc="upper left", ncol=2, fontsize=10, columnspacing=0.7)
        plt.tight_layout()
        save_and_show_plot(
            f"{figdir}MotherDaughterTrajectories_PID{row['pid']}_TID{row['tid1']}TID_{row['tid2']}"
        )


def histogram_difference_in_timing(
    df_mother_and_daughter_pairs, interval, figdir, difference_col="difference_in_timing_avg"
):
    """
    Plot a histogram of the difference in timing between the mother's volume at C and each the daughters' volumes at B.

    Parameters
    ----------
    df_mother_and_daughter_pairs : pd.DataFrame
        The mother-daughter pairs dataframe.
    figdir : str
        The directory to save the figures.
    difference_col : str
        The column in the dataframe to plot the histogram for.
    """
    # negative means transition was after crossing, positive means transition was before crossing
    fig = plt.figure(figsize=(5, 4))
    md_1 = df_mother_and_daughter_pairs["difference_in_timing1"] * interval
    md_2 = df_mother_and_daughter_pairs["difference_in_timing2"] * interval
    md_all = np.concatenate([md_1, md_2])
    plt.hist(md_all, bins=20, color="#808080")
    plt.xlabel(
        "Difference in Time Mothers Volume at C\nand Daughters' Time at B (min)", fontsize=14
    )
    plt.ylabel(f"Count", fontsize=14)
    plt.title(f"N={len(md_all)}, Mean={np.nanmean(md_all):.2f}")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    save_and_show_plot(f"{figdir}mother_daughter_difference_in_timing_histogram", figure=fig)


def histogram_difference_in_volume(df_mother_and_daughter_pairs, figdir):
    """
    Plot a histogram of the difference in volume between mothers volume at C and the sum of the daughters' volumes at B.

    Parameters
    ----------
    df_mother_and_daughter_pairs : pd.DataFrame
        The mother-daughter pairs dataframe.
    figdir : str
    """
    fig = plt.figure(figsize=(5, 4))
    x = (
        df_mother_and_daughter_pairs["mothers_volume_at_C"]
        - df_mother_and_daughter_pairs["sum_sisters_volume_at_B"]
    )
    plt.hist(x, bins=20, color="#808080")
    _, _, units, _ = get_plot_labels_for_metric("volume")
    plt.xlabel(
        f"Difference in Mother's Volume at C\nand Sum of Daughters' Volumes at B {units}",
        fontsize=14,
    )
    plt.ylabel(f"Count", fontsize=14)
    plt.title(f"N={len(x)}, Mean={np.mean(x.values):.2f}, St Dev={np.std(x.values):.2f}")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    save_and_show_plot(f"{figdir}mother_daughter_difference_in_volume_histogram", figure=fig)


def histogram_percent_of_volume(df_mother_and_daughter_pairs, figdir):
    """
    Plot a histogram of the difference in volume between mothers volume at C and the sum of the daughters' volumes at B.

    Parameters
    ----------
    df_mother_and_daughter_pairs : pd.DataFrame
        The mother-daughter pairs dataframe.
    figdir : str
    """
    fig = plt.figure(figsize=(5, 4))
    mother = df_mother_and_daughter_pairs["mothers_volume_at_C"]
    daughters = df_mother_and_daughter_pairs["sum_sisters_volume_at_B"]
    x = (daughters / mother) * 100
    plt.hist(x, bins=20, color="#808080")
    _, _, units, _ = get_plot_labels_for_metric("volume")
    plt.xlabel(
        f"Sisters' combined starting volume\n as a percent of the mothers ending volume {units}",
        fontsize=14,
    )
    plt.ylabel(f"Count", fontsize=14)
    plt.title(f"N={len(x)}, Mean={np.mean(x.values):.2f}%, StDev={np.std(x.values):.2f}")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    save_and_show_plot(f"{figdir}mother_daughter_percent_difference", figure=fig)


def histogram_difference_half_mothers_volume_at_C_and_daughter_at_B(
    df_mother_and_daughter_pairs, figdir
):
    """
    Plot histogram of the difference in volume between half the mother's volume at C and the daughters volume at B.

    Paramaters
    ----------
    df_mother_and_daughter_pairs : pd.DataFrame
        The mother-daughter pairs dataframe.
    figdir : str
        The directory to save the figures.
    """
    md_1 = df_mother_and_daughter_pairs["difference_half_vol_at_C_and_tid1_B"]
    md_2 = df_mother_and_daughter_pairs["difference_half_vol_at_C_and_tid2_B"]
    md_all = np.concatenate([md_1, md_2])
    fig = plt.figure(figsize=(5, 4))
    plt.hist(abs(md_all), bins=20)
    plt.title(f"Mean :{np.mean(md_all):.2f}, StDev: {np.std(md_all):.2f}")
    plt.xlabel(
        "Difference between 1/2 Mother's ending volume\nand combined daughters'starting volume (µm\u00B3)",
        fontsize=14,
    )
    plt.ylabel(f"Count (Total N={len(df_mother_and_daughter_pairs)*2})", fontsize=14)
    plt.tight_layout()
    save_and_show_plot(f"{figdir}half_mothers_volume_at_c_and_daughter_at_b_histogram", figure=fig)


def scatterplot_crossing_vs_transition(
    df_mother_and_daughter_pairs, interval, figdir, unity_line=True
):
    """
    Plot a scatterplot of the time when the mother's volume at C is equal to the sum of the daughters' volumes at B
    versus the time of transition for the daughters.

    Parameters
    ----------
    df_mother_and_daughter_pairs : pd.DataFrame
        The mother-daughter pairs dataframe.
    interval : float
        The time interval between frames in minutes.
    figdir : str
        The directory to save the figures.
    """
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(
        df_mother_and_daughter_pairs["crossing"] * interval,
        df_mother_and_daughter_pairs["transition1"] * interval,
        label="Mother, Daughter 1",
        color="tab:orange",
        alpha=0.4,
        edgecolors="none",
    )
    plt.scatter(
        df_mother_and_daughter_pairs["crossing"] * interval,
        df_mother_and_daughter_pairs["transition2"] * interval,
        label="Mother, Daughter 2",
        color="tab:green",
        alpha=0.4,
        edgecolors="none",
    )

    plt.xlim(750, 2050)
    plt.ylim(750, 2050)

    if unity_line:
        plt.plot(
            [750, 2050], [750, 2050], color="black", linestyle="--", alpha=0.75, label="Unity Line"
        )

    plt.title(f"N = {len(df_mother_and_daughter_pairs)*2}")
    plt.xlabel("Time when Mother Volume at C =\nSum of Daughter Volumes (min)", fontsize=14)
    plt.ylabel("Daughter Time at B (min)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.tight_layout()
    save_and_show_plot(f"{figdir}mother_daughter_crossing_vs_transition_scatterplot", figure=fig)


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
    label = f"y = {fitted_line.slope:.2f}x + {fitted_line.intercept:.2f}"
    return (xplot, fitted_y), label, fitted_line.slope


def scatterplot_sum_sisters_volume_at_C_vs_mother_vol_at_C(
    df, figdir, reference_line=False, fit_line=False
):
    """
    Plot the sum of the sisters' volumes at C versus the mother's volume at C. Reference lines come from the function
    plot_mother_at_C_variation() below that calculates the spike in volume at C for the mother.

    Parameters
    ----------
    df: pd.DataFrame
        The mother-daughter pairs dataframe.
    """
    fig = plt.figure(figsize=(5, 5))
    x = df["mothers_volume_at_C"]
    y = df["sum_sisters_volume_at_B"]
    plt.scatter(x, y, color="black", alpha=0.4, edgecolors="none")

    nas = np.logical_or(
        np.isnan(df["mothers_volume_at_C"]), np.isnan(df["sum_sisters_volume_at_B"])
    )
    df_pearson = df[~nas]

    pearson, p_pvalue = spstats.pearsonr(
        df_pearson["mothers_volume_at_C"], df_pearson["sum_sisters_volume_at_B"]
    )
    (x_fit, y_fit), label, _ = theilslopes_fitting(x, y, xplot=None)

    fitting_title = ""
    if fit_line:
        plt.plot(x_fit, y_fit, label="Fit line", color="#15537d", linewidth=1, alpha=0.9)
        mean_slope, _, _, percent_slope = confidence_interval(
            df=df,
            n_resamples=500,
            column_1="mothers_volume_at_C",
            column_2="sum_sisters_volume_at_B",
            corr=False,
        )
        fitting_title = (
            f"\nTheils slope={mean_slope:.2f}, CI ({percent_slope[0]:.2f}, {percent_slope[1]:.2f})"
        )

    plt.plot(
        [850, 1400],
        [850, 1400],
        color="black",
        linestyle="--",
        alpha=0.75,
        label="Unity Line",
        zorder=0,
    )
    if reference_line:
        plt.plot(
            [850 + (850 * 0.04), 1400],
            [850, 1400 - (1400 * 0.04)],
            color="tab:red",
            linestyle="--",
            alpha=0.75,
            label="96% Mother's Volume",
            zorder=0,
        )

    _, _, units, _ = get_plot_labels_for_metric("volume")
    # plt.title(f"N = {len(df)}\nCorr={mean_corr:.2f}, CI ({percent[0]:.2f}, {percent[1]:.2f}){fitting_title}", fontsize=14)
    plt.title(f"N = {len(df)}\nR={pearson:.2f}, r={p_pvalue:.4f}", fontsize=14)
    plt.xlabel(f"Mother's Ending Volume {units}", fontsize=14)
    plt.ylabel(f"Sum of Sisters' Starting Volumes {units}", fontsize=14)
    plt.xlim(850, 1300)
    plt.ylim(850, 1300)
    plt.legend()
    plt.tight_layout()
    save_and_show_plot(
        f"{figdir}mother_vol_at_C_vs_sum_daughters_vol_B_scatterplot_withrefline{reference_line}",
        figure=fig,
        bbox_inches="tight",
    )


def difference_in_sister_transition(df_mother_and_daughter_pairs, interval, figdir, xlim=300):
    """
    Calculate the difference in transition timing between sisters.

    Parameters
    ----------
    df_mother_and_daughter_pairs : pd.DataFrame
        The mother-daughter pairs dataframe.

    Returns
    -------
    df_mother_and_daughter_pairs : pd.DataFrame
        The mother-daughter pairs dataframe with the difference in transition timing between sisters added.
    """
    fig = plt.figure(figsize=(5, 4))
    df_mother_and_daughter_pairs["difference_in_sister_transition"] = abs(
        df_mother_and_daughter_pairs["transition1"] * interval
        - df_mother_and_daughter_pairs["transition2"] * interval
    )
    plt.hist(df_mother_and_daughter_pairs["difference_in_sister_transition"], bins=100)
    plt.xlabel("Difference in sisters time at B (min)", fontsize=14)
    plt.ylabel(f"Count (Total N={len(df_mother_and_daughter_pairs)})", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(-5, xlim)
    plt.tight_layout()
    save_and_show_plot(f"{figdir}hist_difference_sister_time_at_B_{xlim}", figure=fig)


def difference_in_sister_transition_scatter(df_mother_and_daughter_pairs, interval, figdir):
    """
    Calculate the difference in transition timing between sisters.

    Parameters
    ----------
    df_mother_and_daughter_pairs : pd.DataFrame
        The mother-daughter pairs dataframe.

    Returns
    -------
    df_mother_and_daughter_pairs : pd.DataFrame
        The mother-daughter pairs dataframe with the difference in transition timing between sisters added.
    """
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(
        df_mother_and_daughter_pairs["transition1"] * interval,
        df_mother_and_daughter_pairs["transition2"] * interval,
    )
    plt.xlabel("Sister 1 Time at B (min)", fontsize=14)
    plt.ylabel("Sister 2 Time at B (min)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    save_and_show_plot(f"{figdir}scatter_difference_sister_time_at_B")


def diff_vol_diff_timing(df_mother_and_daughter_pairs, interval, figdir):
    """
    Calculate the difference in volume at B and the difference in timing for each mother-daughter pair.

    Parameters
    ----------
    df_mother_and_daughter_pairs : pd.DataFrame
        The mother-daughter pairs dataframe.

    Returns
    -------
    df_mother_and_daughter_pairs : pd.DataFrame
        The mother-daughter pairs dataframe with the difference in volume at B and the difference in timing added.
    """
    fig = plt.figure(figsize=(5, 5))
    df_mother_and_daughter_pairs["difference_in_volume_at_B"] = abs(
        df_mother_and_daughter_pairs["tid1_volume_at_B"]
        - df_mother_and_daughter_pairs["tid2_volume_at_B"]
    )
    df_mother_and_daughter_pairs["difference_in_timing"] = abs(
        df_mother_and_daughter_pairs["transition1"] * interval
        - df_mother_and_daughter_pairs["transition2"] * interval
    )
    plt.scatter(
        df_mother_and_daughter_pairs["difference_in_timing"],
        df_mother_and_daughter_pairs["difference_in_volume_at_B"],
    )
    plt.ylabel("Difference in sisters volume at B (min)", fontsize=14)
    plt.xlabel("Difference in sisters time at B (min)", fontsize=14)
    plt.tight_layout()
    save_and_show_plot(f"{figdir}scatter_difference_sister_time_at_B_vs_volume_at_B")


def plot_volume_at_C_variation(df_full, window, interval, figdir):
    """
    Mothers volume at C sometimes spikes up. This function shows a histogram of the difference in volume
    for the final window frames of the mother's trajectory and compares that to a window of the same size
    from 10 frames before breakdown in the trajectory.

    Parameters
    ----------
    df_full : pd.DataFrame
        The full dataset with all timepoints and tracks.
    df_mother_and_daughter_pairs : pd.DataFrame
        The mother-daughter pairs dataframe.
    interval : float
        The time interval between frames in minutes.
    figdir : str
        The directory to save the figures.
    """
    bins = np.arange(0, 100, 2)
    time_min = (window - 1) * interval

    spike_variation = []
    for _, dft in df_full.groupby("track_id"):
        scale, label, units, _ = get_plot_labels_for_metric("volume")
        y = dft.volume.values[-window:] * scale
        volume_difference = y.max() - y.min()
        spike_variation.append(volume_difference)
    plt.hist(
        spike_variation,
        bins=bins,
        alpha=0.5,
        label=f"Last {time_min} min, N={len(spike_variation)}\nMean={np.mean(spike_variation):.2f} {units}, Max={np.max(spike_variation):.2f} {units}",
    )
    plt.xlabel(f"Change in Volume {units}", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xlim(0, 100)
    plt.tight_layout()

    control_spike_variation = []
    for _, dft in df_full.groupby("track_id"):
        scale, label, units, _ = get_plot_labels_for_metric("volume")
        list_without_last_ten = dft.volume.values[:-4] * scale
        y = list_without_last_ten[-window:]
        volume_difference = y.max() - y.min()
        control_spike_variation.append(volume_difference)
    plt.hist(
        control_spike_variation,
        bins=bins,
        color="tab:orange",
        label=f"Control {time_min} min, N={len(control_spike_variation)}\nMean={np.mean(control_spike_variation):.2f} {units}, Max={np.max(control_spike_variation):.2f} {units}",
        alpha=0.5,
    )
    plt.ylabel("Count", fontsize=14)
    plt.xlim(0, 100)
    plt.legend()
    plt.tight_layout()
    save_and_show_plot(f"{figdir}volume_at_C_variation_histogram_{time_min}_minute_window")

    difference = [a - b for a, b in zip(spike_variation, control_spike_variation)]
    plt.hist(
        difference,
        bins=bins,
        alpha=0.5,
        color="tab:green",
        label=f"Difference in Volume {units}\nMean={np.mean(difference):.2f} {units}, Max={np.max(difference):.2f} {units}",
    )
    plt.xlabel(f"Magnitude increase {time_min} {units}", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xlim(0, 100)
    plt.legend()
    plt.tight_layout()
    save_and_show_plot(
        f"{figdir}difference_volume_at_C_variation_histogram_{time_min}_minute_window"
    )


def two_feature_density(df, feature1, feature2, figdir, add_kl_divergence=False):
    """
    Plot the feature density for a feature for the entire dataset and a subset of the dataset.

    Parameters
    ----------
    df: Dataframe
        The df_mother_and_daughter_pairs dataset
    feature1: String
        Feature or metric you want to plot (column name)
    feature2: String
        Feature or metric you want to plot (column name)
    figdir: String
        Relative path within nuc_morph_analysis/analyses folder.
    histtype: String
        'step' or 'bar'
    add_kl_divergence: Boolean
        If True, will add the KL divergence to the plot
    """
    data1 = df[feature1]
    data2 = df[feature2]

    fig = plt.figure(figsize=(5, 5))

    _, label, units, lim = get_plot_labels_for_metric(feature1)
    label1 = f"{label} {units}\nN={len(data1)}, Mean={np.mean(data1):.2f},  CV={(np.std(data1) / np.mean(data1)):.2f}"
    ax = sb.kdeplot(data1, color="black", label=label1)

    _, label, units, lim = get_plot_labels_for_metric(feature2)
    label2 = f"{label} {units}\nN={len(data2)}, Mean={np.mean(data2):.2f}, CV={(np.std(data2) / np.mean(data2)):.2f}"
    ax = sb.kdeplot(data2, color="tab:blue", label=label2)

    _, label, units, lim = get_plot_labels_for_metric("volume")
    plt.xlabel(f"{label} {units}")
    plt.ylabel("Density")
    plt.legend(bbox_to_anchor=(0.5, 1.15), loc="center")

    if add_kl_divergence:
        kl1, kl2, average_kl = get_kl_divergence(data1, data2)
        print(f"KL divergence sym, {kl1:.2f}, {kl2:.2f}")
    plt.tight_layout()
    save_and_show_plot(f"{figdir}density_{feature1}_{feature2}", figure=fig)
