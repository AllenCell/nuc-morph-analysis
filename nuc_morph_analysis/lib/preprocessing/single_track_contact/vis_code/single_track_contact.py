import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from nuc_morph_analysis.lib.preprocessing.single_track_contact.vis_code import image_helper
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric

from nuc_morph_analysis.lib.preprocessing.system_info import (
    PIXEL_SIZE_YX_100x,
    PIXEL_SIZE_Z_100x,
    PIXEL_SIZE_YX_20x,
    PIXEL_SIZE_Z_20x,
)


def get_slicing_method_seg_or_raw(seg_or_raw):
    """
    This function returns the slicing method for the yx view based on whether the image is a segmentation or raw image.

    Parameters
    ----------
    seg_or_raw : str
        The type of image, either 'seg' or 'raw'.

    Returns
    -------
    method : str
        The method to be used for getting a slice for the yx view.
    """
    if seg_or_raw == "seg":
        max_sum_slice = ("sum", "mid", "mid")
    else:
        max_sum_slice = ("max", "mid", "mid")
    return max_sum_slice


def process_image_into_slices_and_stack_seg_or_raw(img, seg_or_raw="seg"):
    """
    This function takes in an image and returns the yx,zx, and zy slice views stacked together.

    Parameters
    ----------
    img : np.array
        The image to be sliced.
    seg_or_raw : str
        The type of image, either 'seg' or 'raw'. Important for determining interpolation when upsmapling or downsampling

    Returns
    -------
    stacked_slices : np.array
        The stacked slices in uint8
    """
    max_sum_slice = get_slicing_method_seg_or_raw(seg_or_raw)
    if seg_or_raw == "seg":
        scale_factors = [
            (1, 1),
            (PIXEL_SIZE_Z_100x / PIXEL_SIZE_YX_100x, 1),
            (PIXEL_SIZE_Z_100x / PIXEL_SIZE_YX_100x, 1),
        ]
        order = 0
    else:
        scale_factors = [
            (1, 1),
            (PIXEL_SIZE_Z_20x / PIXEL_SIZE_YX_20x, 1),
            (PIXEL_SIZE_Z_20x / PIXEL_SIZE_YX_20x, 1),
        ]
        order = 3

    slices = image_helper.process_image_into_slices(
        img, max_sum_slice, order=order, scale_factors=scale_factors
    )
    stacked_slices = image_helper.stack_the_slices_zyx(slices, width=0, gap_color=0)

    return stacked_slices


def create_df_feats(columns, dataset):
    """
    create a dataframe that carries the label names, units and scaling factors
    """
    dflist = []
    for metric_name in columns:
        out = get_plot_labels_for_metric(metric_name, dataset=dataset)
        feats = {"name": metric_name, "scale_factor": out[0], "label": out[1], "units": out[2]}
        dflist.append(pd.DataFrame(data=feats.values(), index=feats.keys()).T)
    df_feats = pd.concat(dflist)
    df_feats.set_index("name", inplace=True)
    return df_feats


def plot_tracks(dfnt1, dfnt3, i, timevar, feat_col, ax, fontsize=8, fontsize_tid=8):
    x1 = dfnt1[timevar].values
    y1 = dfnt1[feat_col].values
    ax.plot(x1, y1, label="old", linewidth=1, zorder=1000, color="tab:blue")

    x3 = dfnt3[timevar].values
    y3 = dfnt3[feat_col].values

    ax.plot(x3, y3, label="new", linewidth=1, zorder=2000, color="tab:orange")

    ax.plot(x3[i], y3[i], "o", color="tab:orange", label="current", markersize=4)
    # now find the points where dfnt3.min_dist is greater than np.sqrt(2)*10 pixels
    # and plot them as red dots
    min_dist = dfnt3["min_dist"].astype(float)
    min_dist_log = min_dist > np.sqrt(2) * 10
    if np.sum(min_dist_log) > 0:
        xmd = dfnt3[timevar].values[min_dist_log]
        ymd = dfnt3[feat_col].values[min_dist_log]
        ax.plot(xmd, ymd, "ko", label="min_dist > 10*sqrt(2)", markersize=2, alpha=1, zorder=0)

    # now add red dots where there are nans
    nan_log = np.isnan(y3)
    if np.sum(nan_log) > 0:
        xn = dfnt3[timevar].values[nan_log]
        yn = dfnt3[feat_col].values[nan_log]
        ax.plot(xn, yn, "ro", label="nan", markersize=2, alpha=1, zorder=200)

    if timevar == "time_hours":
        xticks = np.arange(0, 60 + 15, 15)
    elif timevar == "time_hours_from_0":
        xticks = np.arange(0, 24 + 8, 8)

    ax.set_xticks(xticks, fontsize=fontsize)

    if feat_col == "volume_um":
        yticks = np.arange(0, 1400 + 200, 200)
        ytickstr = [f"{y:.0f}" if i % 2 == 0 else "" for i, y in enumerate(yticks)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytickstr, fontsize=fontsize)
        ax.set_ylim([200, 1400])
        ylabel = "volume\n(um^3)"
    elif feat_col == "height_um":
        yticks = np.arange(0, 15 + 3, 3)
        ytickstr = [f"{y:.0f}" if i % 2 == 0 else "" for i, y in enumerate(yticks)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytickstr, fontsize=fontsize)
        ax.set_ylim([3, 10])
        ylabel = "height\n(um)"

    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel("hours", fontsize=fontsize)


def get_yticks(column, ticks_or_lims="ticks"):
    ytick_dict = {
        "volume": (0, 1500, 300),
        "mesh_vol": (0, 1500, 300),
        "mesh_sa": (0, 1000, 200),
        "SA_vol_ratio": (0.5, 1.2, 0.1),
        "height": (0, 12, 3),
        "height_percentile": (0, 12, 3),
        "xz_aspect": (0, 7, 1),
        "yz_aspect": (0, 1.5, 0.25),
        "xy_aspect": (0, 3, 0.5),
        "colony_depth": (0, 8, 2),
        "neigh_distance": (0, 80, 20),
        "density": (0, 0.025, 0.005),
        "dxdt_48_volume": (-50, 150, 50),
        "dxdt_24_volume": (-50, 150, 50),
        "dxdt_12_volume": (-50, 150, 50),
    }
    # now apply np.arange to the ytick_dict.values
    for k, v in ytick_dict.items():
        ytick_dict[k] = np.arange(v[0], v[1] + v[2], v[2])
        # ytick_dict[k] = np.linspace(v[0],v[1],int((v[1]-v[0])/v[2])+2)
    if ticks_or_lims == "ticks":
        return ytick_dict[column] if column in list(ytick_dict.keys()) else None
    elif ticks_or_lims == "lims":
        return (
            [ytick_dict[column][0], ytick_dict[column][-1]]
            if column in list(ytick_dict.keys())
            else None
        )


def format_number(n):
    if n == int(n):
        return "{:.0f}".format(n)
    else:
        return "{:.2f}".format(n)


def make_plot(
    ax_dict,
    ax_column_pairs,
    t,
    dffs,
    df_feats,
    frame_formation,
    frame_transition,
    frame_breakdown,
    frame_buffer,
    update=False,
):
    # determine time vector
    color_dict = {
        "volume": "k",
        "mesh_vol": "tab:olive",
        "mesh_sa": "k",
        "SA_vol_ratio": "k",
        "height": "k",
        "height_percentile": "tab:blue",
        "xy_aspect": "tab:purple",
        "xz_aspect": "tab:orange",
        "zy_aspect": "tab:olive",
        "colony_depth": "k",
        "neigh_distance": "k",
        "density": "k",
        "dxdt_48_volume": "tab:blue",
        "dxdt_24_volume": "tab:orange",
        "dxdt_12_volume": "tab:purple",
        "f": "tab:blue",
        "t": "tab:orange",
        "b": "tab:purple",
        "now": "tab:green",
    }
    time_in_frames = dffs.index.values
    time_in_frames_subbed = time_in_frames - time_in_frames[0]
    time_in_hours_subbed = time_in_frames_subbed * 5 / 60
    x = time_in_hours_subbed

    object_list = []
    for ax_name, column_list in ax_column_pairs:

        ax = ax_dict[ax_name]
        column = column_list[0]
        if update == False:  # make the trajectory plot
            for ci, col in enumerate(column_list):
                y = dffs[col].values * df_feats.loc[col, "scale_factor"]
                # ax.plot(x,y,label=col, color = color_dict[col],linewidth=1)
                # plot everything super thin and translucent
                ax.plot(
                    x,
                    y,
                    label=col,
                    color="tab:grey",
                    linewidth=0.5,
                    linestyle="-",
                    marker="None",
                    alpha=1,
                )

                # now plot the data after removing ONLY edge cell points
                log = dffs["fov_edge"] == False
                ax.plot(
                    x[log],
                    y[log],
                    label=col,
                    color="r",
                    linewidth=0.5,
                    marker="None",
                    markeredgecolor="r",
                    markerfacecolor="r",
                    markersize=0.5,
                )

                # now plot the data after removing outliers and edge cell points
                log = (dffs["is_outlier"] == False) & (dffs["fov_edge"] == False)
                ax.plot(
                    x[log],
                    y[log],
                    label=col,
                    color=color_dict[col] if col in list(color_dict.keys()) else "tab:pink",
                    linewidth=1,
                    marker=".",
                    markeredgecolor=(
                        color_dict[col] if col in list(color_dict.keys()) else "tab:pink"
                    ),
                    markerfacecolor=(
                        color_dict[col] if col in list(color_dict.keys()) else "tab:pink"
                    ),
                    markersize=0.5,
                )

        if t not in time_in_frames:
            p = ax.text(0.0, 0.0, f"no data at t={t}", transform=ax.transAxes)
            object_list.append(p)

        # now plot each of these key points
        time_point_value_list = [frame_formation, frame_transition, frame_breakdown, t]
        time_point_name_list = ["f", "t", "b", "now"]
        # do not update ff,ft,fb after the first plot has been made
        if update == True:
            time_point_value_list = [t]
            time_point_name_list = ["now"]

        for frame_point, frame_label in zip(time_point_value_list, time_point_name_list):
            if frame_point in time_in_frames:
                yval = dffs.loc[frame_point, column] * df_feats.loc[column, "scale_factor"]
                xval = (frame_point - time_in_frames[0]) * 5 / 60
                p = ax.plot(
                    xval,
                    yval,
                    "o",
                    label=frame_label,
                    color=(
                        color_dict[frame_label]
                        if frame_label in list(color_dict.keys())
                        else "tab:pink"
                    ),
                    markersize=3,
                )
                if frame_label == "now":  # only remove "now" points
                    object_list.append(p)
            else:
                xval = (frame_point - time_in_frames[0]) * 5 / 60
                p = ax.axvline(
                    x=xval,
                    color=(
                        color_dict[frame_label]
                        if frame_label in list(color_dict.keys())
                        else "tab:pink"
                    ),
                    linestyle="--",
                    label=frame_label,
                    alpha=0.5,
                    markersize=3,
                )
                object_list.append(p)
        # only define the axes boundaries, etc on the first instance (when update =false)
        if update == False:
            # create x ticks every 3 hrs
            max_time_hrs = np.max([24, time_in_hours_subbed[-1]])
            max_time_frames = max_time_hrs * 60 / 5
            # define a vector of frames going to the max time that starts at the first time point
            frame_lookup = np.arange(0, max_time_frames, 1).astype("int32") + time_in_frames[0]
            # convert to hours
            hours_lookup = (frame_lookup - time_in_frames[0]) * 5 / 60
            # create x ticks every 3 hrs
            xticks_hours = np.arange(0, max_time_hrs + 4, 4).astype("uint16")
            xticks_frames = [
                frame_lookup[np.argmin(np.abs(hours_lookup - x))] for x in xticks_hours
            ]

            fontsize = 7
            ax.set_xticks(xticks_hours)
            xtick_labels = [f"{x}\n{y}" for x, y in zip(xticks_hours, xticks_frames)]
            ax.set_xticklabels(xtick_labels, fontsize=fontsize)
            ax.set_xlim(0 - frame_buffer * 5 / 60, max_time_hrs + frame_buffer * 5 / 60)
            ax.set_ylabel(
                f'{df_feats.loc[column,"label"]} {df_feats.loc[column,"units"]}', fontsize=fontsize
            )
            ax.set_xlabel("Time (hours)\nFrame", fontsize=fontsize)
            ax.set_yticks(get_yticks(column, ticks_or_lims="ticks"))
            ax.set_yticklabels(
                [format_number(x) for x in get_yticks(column, ticks_or_lims="ticks")],
                fontsize=fontsize,
            )

            ax.set_ylim(get_yticks(column, ticks_or_lims="lims"))

        # only define the legend on the first instance (when update =false)
        if update == False:
            # define the lines
            line_list = [
                mlines.Line2D([], [], color=color_dict[col], linewidth=2, linestyle="-", label=col)
                for col in column_list
            ]
            line_list.append(
                mlines.Line2D(
                    [], [], color="tab:grey", linewidth=2, linestyle="-", label="fov_edge==True"
                )
            )
            line_list.append(
                mlines.Line2D(
                    [], [], color="r", linewidth=2, linestyle="-", label="is_outlier==True"
                )
            )

            if ax_name in ["vol_ax"]:
                line_name_list = column_list + ["fov_edge==True"] + ["is_outlier==True"]
            else:
                line_name_list = column_list

            points_list = [
                mlines.Line2D(
                    [],
                    [],
                    color=color_dict[point],
                    markersize=5,
                    marker="o",
                    linestyle="None",
                    label=point,
                )
                for point in time_point_name_list
            ]
            name_list = time_point_name_list

            if ax_name in ["vol_ax"]:

                leg2 = ax.legend(
                    points_list,
                    name_list,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.2),
                    fontsize=7,
                    ncol=len(points_list),
                    facecolor="None",
                )
                leg2.get_frame().set_alpha(0.5)
                leg2.get_frame().set_edgecolor("none")
                ax.add_artist(leg2)

            leg = ax.legend(
                line_list,
                line_name_list,
                loc="upper left",
                fontsize=7,
                ncol=1,
                facecolor="None",
                columnspacing=0.1,
                labelspacing=0.1,
            )
            leg.get_frame().set_alpha(0.5)
            leg.get_frame().set_edgecolor("none")

    return ax_dict, object_list


def remove_previously_drawn_objects(object_list):
    # clear the matplotlib items in the object list
    new_list = list(object_list)
    for obj in new_list:
        if str(type(obj)) == "<class 'list'>":
            for o in obj:
                o.remove()
        else:
            obj.remove()
    object_list = []
