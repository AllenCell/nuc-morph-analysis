import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
from nuc_morph_analysis.analyses.inhibitors import dataset_info
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from matplotlib.ticker import MaxNLocator
from scipy import stats
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.analyses.inhibitors.match_and_filter import DEATH_THRESHOLD

GROUP_AND_COLOR_LIST = [["control", "Greys"], ["perturb", "RdPu"]]

GROUP_AND_COLOR_DICT = {
    "control": matplotlib.colormaps["Dark2"].colors[7],
    "perturb": matplotlib.colormaps["Dark2"].colors[3],
}
DYING_COLOR = matplotlib.colormaps["Dark2"].colors[6]


dpi = 300

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.size"] = 8


def set_line_color_cycle(ax, N, cmap):
    N = np.min([50, N])
    colormap = eval(f"plt.cm.{cmap}(np.linspace(0.3,1,N))")
    ax = plt.gca()
    ax.set_prop_cycle(color=colormap)
    return ax


def determine_ylimit(ax, col):
    if col in ["volume"]:
        ax.set_yticks(np.arange(400, 1600, 300))
        ax.set_ylim([400, 1350])

    if col in ["volume_sub"]:
        ax.set_yticks(np.arange(-200, 300, 100))
        ax.set_ylim([-200, 280])


def get_ttest_pvalue_vector(dft1, dft2):
    pval_list = []
    time_index1 = dft1.index
    time_index2 = dft2.index
    time_index = time_index1.intersection(time_index2)
    for time in time_index:
        x1 = dft1.loc[time, :].dropna().values
        x2 = dft2.loc[time, :].dropna().values
        tstat, pval = stats.ttest_ind(x1, x2, equal_var=False)
        pval_list.append(pval)
    return time_index, np.array(pval_list)


def plot_regions_of_significance(ax, time_vec, pvalue_vec, stat_alpha):
    # now identify all regions of the pvalue_vec that are less than 0.05
    regions = []
    in_region = False
    for i, pval in enumerate(pvalue_vec):
        if pval < stat_alpha:
            if not in_region:
                start = i
                in_region = True
        else:
            if in_region:
                end = i
                regions.append((start, end))
                in_region = False
    if in_region:
        regions.append((start, len(pvalue_vec) - 1))
    for start, end in regions:
        ax.axvspan(time_vec[start], time_vec[end], ymin=0.02, ymax=0.05, color="gray", alpha=0.3)


def simple_plot_individual_trajectories(dff, col, dicti, pairs, matched=False):

    savedir = Path(__file__).parent / "figures" / "late_growth" / "individual tracks" / pairs

    for gci, (condition, cmap) in enumerate(GROUP_AND_COLOR_LIST):
        fig, ax = plt.subplots(1, 1, figsize=(2.5, 2))
        print(condition)
        dfg = dff[dff["condition"] == condition]
        dfgt = dfg.pivot(index="time_minutes_since_drug", columns="track_id", values=col)
        ax = set_line_color_cycle(ax, dfgt.shape[1], cmap)
        scale, label, unit, _ = get_plot_labels_for_metric(col)
        y = dfgt.values * scale
        x = np.tile(dfgt.index.values, (y.shape[1], 1)).T
        ax.plot(x, y, alpha=1, linewidth=0.5)
        ax.set_ylabel(f"{label} {unit}")

        ax.axvline(0, ymax=0.92, color="k", linestyle="--", zorder=1, linewidth=0.45)
        perturb_details = dataset_info.get_drug_perturbation_details_from_colony_name(
            dicti["perturb"]
        )
        condition_details = dataset_info.get_drug_perturbation_details_from_colony_name(
            dicti[condition]
        )
        determine_x_axis_details(ax, perturb_details)
        determine_ylimit(ax, col)

        condition_str = condition_details["drugs_string"]
        label_str = f"{condition_str}, N={dfgt.shape[1]}"
        color = GROUP_AND_COLOR_DICT[condition]
        plt.text(
            0.02, 0.98, f"{label_str}", color=color, ha="left", va="top", transform=ax.transAxes
        )

        titlestr = get_title_str(pairs)
        plt.title(titlestr)
        savepath = f"{savedir}{os.sep}-{condition}-{col}-{cmap}-{str(matched)}.pdf"

        save_and_show_plot(
            savepath,
            file_extension=".pdf",
            figure=fig,
            transparent=True,
            quiet=False,
            massive_output=False,
            keep_open=True,
        )
        plt.show()
        plt.close()


def plot_populations_and_stat_test(dff, col, dicti, pairs, stat_alpha=0.01, matched=False):

    savedir = Path(__file__).parent / "figures" / "late_growth" / "population tracks" / pairs

    scale, label, unit, _ = get_plot_labels_for_metric(col)

    dfg1 = dff.loc[dff["condition"] == "control", :]
    dft1 = dfg1.pivot_table(index="time_minutes_since_drug", columns="track_id", values=col)

    dfg2 = dff.loc[dff["condition"] == "perturb", :]
    dft2 = dfg2.pivot_table(index="time_minutes_since_drug", columns="track_id", values=col)

    # time_vec,pvalue_vec = get_ks_pvalue_vector(dft1,dft2)
    time_vec, pvalue_vec = get_ttest_pvalue_vector(dft1, dft2)

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2))
    for gci, (condition, dft) in enumerate([("control", dft1), ("perturb", dft2)]):
        mean = dft.mean(axis=1)
        low = dft.quantile(0.05, axis=1)
        high = dft.quantile(0.95, axis=1)

        color = GROUP_AND_COLOR_DICT[condition]

        plt.plot(mean.index, mean.values * scale, color=color, linewidth=2, zorder=1000 + gci)
        plt.fill_between(
            mean.index,
            low * scale,
            high * scale,
            alpha=0.3,
            color=color,
            zorder=50 + gci,
            edgecolor="None",
        )
        plot_regions_of_significance(ax, time_vec, pvalue_vec, stat_alpha)
        ax.set_xlabel("Time since inhibitior addition (min)")
        ax.set_ylabel(f"{label} {unit}")

        ax.axvline(0, ymax=0.85, color="k", linestyle="--", zorder=1, linewidth=0.45)
        perturb_details = dataset_info.get_drug_perturbation_details_from_colony_name(
            dicti["perturb"]
        )
        condition_details = dataset_info.get_drug_perturbation_details_from_colony_name(
            dicti[condition]
        )
        determine_x_axis_details(ax, perturb_details)
        determine_ylimit(ax, col)

        condition_str = condition_details["drugs_string"]
        label_str = f"{condition_str}, N={dft.shape[1]}"
        plt.text(
            0.02,
            0.91 + gci * 0.07,
            f"{label_str}",
            color=color,
            ha="left",
            va="top",
            transform=ax.transAxes,
        )

    titlestr = get_title_str(pairs)
    plt.title(titlestr)
    savepath = f"{savedir}{os.sep}-{condition}-{col}-{color}-{str(matched)}.pdf"

    save_and_show_plot(
        savepath,
        file_extension=".pdf",
        figure=fig,
        transparent=True,
        quiet=False,
        massive_output=False,
        keep_open=True,
    )
    plt.show()
    plt.close()


def get_title_str(pairs):
    titlestr = dataset_info.drug_names_for_titles(pairs)
    return titlestr


def determine_x_axis_details(ax, perturb_details):
    death_threshold_x = get_death_threshold_x(perturb_details)

    x_vec = (
        perturb_details["num_dying_x"] - perturb_details["drug_added_frame"]
    ) * perturb_details["time_interval_minutes"]
    x_label = "Time since drug addition (min)"
    y_vec = perturb_details["num_dying_y"]
    xlimits = [x_vec.min(), death_threshold_x]
    ax.set_xlim(xlimits)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=3, steps=[1.2, 6]))
    ax.margins(x=0.1, y=0.3)
    ax.set_xlabel(x_label)


def get_death_threshold_x(perturb_details):
    x_vec = (
        perturb_details["num_dying_x"] - perturb_details["drug_added_frame"]
    ) * perturb_details["time_interval_minutes"]
    y_vec = perturb_details["num_dying_y"]
    x_vec_i = np.arange(
        x_vec.min(),
        x_vec.max() + perturb_details["time_interval_minutes"],
        perturb_details["time_interval_minutes"],
    )
    y_vec_i = np.interp(x_vec_i, x_vec, y_vec)
    if np.sum(y_vec_i > DEATH_THRESHOLD) == 0:
        return x_vec[-1]
    else:
        ydyingthreshidx = np.where(y_vec_i > DEATH_THRESHOLD)[0][0]
        death_threshold_x = x_vec_i[ydyingthreshidx]
        return death_threshold_x


def expansion_plot_individual(df, dfsub1, dfsub2, pairs, chosen_condition, details):

    scale, label, unit, _ = get_plot_labels_for_metric("volume")
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2), sharex=False, sharey=False)
    for tid in df.track_id.unique():
        dfsubt = df[df.track_id == tid]
        plt.plot(
            dfsubt.time_minutes_since_drug,
            dfsubt.volume * scale,
            color="tab:grey",
            alpha=0.2,
            linewidth=0.5,
        )

    for tid in dfsub1.track_id.unique():
        dfsubt = dfsub1[dfsub1.track_id == tid]
        plt.plot(
            dfsubt.time_minutes_since_drug,
            dfsubt.volume * scale,
            # color = 'GROUP_AND_COLOR_DICT['control']',
            color="k",  # use black instead so the color shows up clearly
            alpha=0.7,
            linewidth=0.5,
        )

    for tid in dfsub2.track_id.unique():
        dfsubt = dfsub2[dfsub2.track_id == tid]
        plt.plot(
            dfsubt.time_minutes_since_drug,
            dfsubt.volume * scale,
            color=GROUP_AND_COLOR_DICT["perturb"],
            alpha=0.7,
            linewidth=0.5,
        )

    plt.axvline(0, color="k", linestyle="--")
    # plt.xlim(None,110)
    determine_x_axis_details(ax, details)
    plt.ylim([0, 2000])
    plt.ylabel("Volume (μm\u00B3)")
    plt.xlabel("Time since inhibitor addition (min)")
    plt.title("Identifying tracks before and after inhibitor addition")

    savedir = Path(__file__).parent / "figures" / "expansion" / pairs
    savepath = (
        f"{savedir}{os.sep}{pairs}before_after_inhibitor_track_selection{chosen_condition}.pdf"
    )

    save_and_show_plot(
        savepath,
        file_extension=".pdf",
        figure=fig,
        transparent=True,
        quiet=False,
        massive_output=False,
        keep_open=True,
    )
    plt.show()
    plt.close()


def expansion_plot_population_avg(dfsub1, dfsub2, savedir, pairs, chosen_condition, details):
    scale, label, unit, _ = get_plot_labels_for_metric("volume")
    dfc = pd.concat([dfsub1, dfsub2], names=["condish"], keys=["control", "perturb"])
    dfc.reset_index(inplace=True)
    # now create a dataframe where each row is a timepoint and each column is a track_id
    condition_str = details["drugs_string"]
    condition_list = [f"Before {condition_str}", f"After {condition_str}"]
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2), sharex=False, sharey=False)

    for ci, condition in enumerate(["control", "perturb"]):
        dfcsub = dfc[dfc.condish == condition]
        dfp = dfcsub.pivot(index="time_since_frame_formation", columns="track_id", values="volume")

        # determine mean and 5th and 95th percentiles
        dfp["mean"] = dfp.mean(axis=1)
        dfp["5th"] = dfp.quantile(0.05, axis=1)
        dfp["95th"] = dfp.quantile(0.95, axis=1)
        dfp["n"] = dfp.isna().sum(axis=1)

        # plot the mean and 5th and 95th percentiles
        ax.plot(
            dfp.index,
            dfp["mean"] * scale,
            color=GROUP_AND_COLOR_DICT[condition],
            linewidth=1,
            zorder=ci * 100,
        )
        ax.fill_between(
            dfp.index,
            dfp["5th"] * scale,
            dfp["95th"] * scale,
            color=GROUP_AND_COLOR_DICT[condition],
            alpha=0.5,
            zorder=ci * 100,
            edgecolor="None",
        )
        ax.set_xlim(-20, 90)
        ax.set_ylim(200, 800)
        ax.set_xlabel("Time since frame formation (min)")
        ax.set_ylabel("Volume (μm\u00B3)")
        ax.text(
            0.02,
            0.91 + ci * 0.07,
            f"{condition_list[ci]}, N = {dfp.shape[1]}",
            color=GROUP_AND_COLOR_DICT[condition],
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
    savedir = Path(__file__).parent / "figures" / "expansion" / pairs
    savepath = f"{savedir}{os.sep}{pairs}before_after_inbibitor{chosen_condition}.pdf"

    save_and_show_plot(
        savepath,
        file_extension=".pdf",
        figure=fig,
        transparent=True,
        quiet=False,
        massive_output=False,
        keep_open=True,
    )
    plt.show()
    plt.close()


def bootstrap_ratio_and_difference(dfsub1, dfsub2, pairs, chosen_condition, time_window=50):
    scale, label, unit, _ = get_plot_labels_for_metric("volume")
    iterations = 100
    dft1 = (
        dfsub1.pivot(index="track_id", columns="time_since_frame_formation", values="volume")
        * scale
    )
    dft2 = (
        dfsub2.pivot(index="track_id", columns="time_since_frame_formation", values="volume")
        * scale
    )
    ratio_list = []
    diff_list = []
    true_ratio = dft2.mean(axis=0) / dft1.mean(axis=0)
    true_diff = dft2.mean(axis=0) - dft1.mean(axis=0)
    for i in range(iterations):
        dft1b = dft1.sample(frac=1, replace=True)  # resample new track_ids
        dft2b = dft2.sample(frac=1, replace=True)  # resample new track_ids
        mean1 = dft1b.mean(axis=0)
        mean2 = dft2b.mean(axis=0)
        ratio_list.append(mean2 / mean1)
        diff_list.append(mean2 - mean1)
    ratio_df = pd.DataFrame(ratio_list)
    diff_df = pd.DataFrame(diff_list)
    ratio_lo = ratio_df.quantile(0.05)
    ratio_hi = ratio_df.quantile(0.95)
    diff_lo = diff_df.quantile(0.05)
    diff_hi = diff_df.quantile(0.95)

    color = GROUP_AND_COLOR_DICT["perturb"]
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2))
    ax.plot(true_ratio, color=color)

    avg_ratio = np.round(np.mean(true_ratio[time_window]), 2)
    plus_minus_ratio = f"[{np.round(np.mean(ratio_lo[time_window]),2)},{np.round(np.mean(ratio_hi[time_window]),2)}]"
    ax.text(
        0.02,
        0.98,
        f"Avg ratio {str(time_window)} minutes  =\n{avg_ratio} {plus_minus_ratio}",
        color="k",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    ax.fill_between(true_ratio.index, ratio_lo, ratio_hi, alpha=0.5, color=color, edgecolor="None")
    ax.set_title("Volume ratio \nPerturb/Control")
    ax.set_xlabel("Time since formation (min)")
    ax.set_ylabel("Volume ratio\n[5%-95% CI]")
    ax.set_ylim([0.7, 1.3])
    ax.set_xlim(-20, 90)
    ax.axhline(1, linestyle="--", color="k", linewidth=0.5)

    savedir = Path(__file__).parent / "figures" / "expansion" / pairs
    savepath = f"{savedir}{os.sep}{pairs}before_after_inbibitor_RATIO-{chosen_condition}.pdf"

    save_and_show_plot(
        savepath,
        file_extension=".pdf",
        figure=fig,
        transparent=True,
        quiet=False,
        massive_output=False,
        keep_open=True,
    )
    plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2))
    ax.plot(true_diff, color=color)

    avg_diff = np.round(np.mean(true_diff[time_window]), 1)
    plus_minus_diff = (
        f"[{np.round(np.mean(diff_lo[time_window]),1)},{np.round(np.mean(diff_hi[time_window]),1)}]"
    )

    ax.text(
        0.02,
        0.98,
        f"Avg. diff. {str(time_window)} minutes  =\n{avg_diff} {plus_minus_diff}",
        color="k",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )
    ax.fill_between(true_diff.index, diff_lo, diff_hi, alpha=0.5, color=color, edgecolor="None")
    ax.set_title(f"Volume difference {unit}\n Perturb-Control")
    ax.set_xlabel("Time since formation (min)")
    ax.set_ylabel(f"Volume difference {unit}\n[5%-95% CI]")
    ax.axhline(1, linestyle="--", color="k", linewidth=0.5)
    ax.set_ylim([-150, 150])
    ax.set_xlim(-20, 90)

    savedir = Path(__file__).parent / "figures" / "expansion" / pairs
    savepath = f"{savedir}{os.sep}{pairs}before_after_inbibitor_DIFF-{chosen_condition}.pdf"

    save_and_show_plot(
        savepath,
        file_extension=".pdf",
        figure=fig,
        transparent=True,
        quiet=False,
        massive_output=False,
        keep_open=True,
    )
    plt.show()
    plt.close()
