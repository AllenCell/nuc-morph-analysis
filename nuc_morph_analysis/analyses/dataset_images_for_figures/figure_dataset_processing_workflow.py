# %%
# get the info for the datasets
import os

import matplotlib.pyplot as plt
import matplotlib

import numpy as np
from aicsimageio.writers import two_d_writer
from matplotlib.collections import PatchCollection  # type: ignore
from nuc_morph_analysis.lib.preprocessing.single_track_contact.export_code import export_helper
from nuc_morph_analysis.lib.preprocessing.single_track_contact.vis_code import image_helper

from nuc_morph_analysis.analyses.dataset_images_for_figures import figure_helper
from skimage import exposure
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.lib.visualization.example_tracks import EXAMPLE_TRACKS
from nuc_morph_analysis.lib.preprocessing.system_info import (
    RESCALE_FACTOR_100x_to_20X,
    PIXEL_SIZE_Z_100x,
    PIXEL_SIZE_YX_100x,
)
from nuc_morph_analysis.lib.preprocessing.single_track_contact.vis_code import image_helper

# create the saving directory if it doesn't exist
figure = "dataset"
panel = "processing_workflow"
savedir, fig_panel_str = figure_helper.get_save_dir_and_fig_panel_str(figure, panel)

# define dataset from which to collect images
# collect information for the dataset
colony = "medium"
timepoint_frame = 48  # choose the timepoints to load (30 minutes) from medium
# define which track_id to box and crop
track_id = EXAMPLE_TRACKS["figure_dataset_processing_workflow"]

# define how many frames to use to create the tracking tails
tail_length = 10
cell_color = np.asarray([0, 255, 255])

# show segmentations as sum projections of contours to emphasize 3D ness
use_cv_contours_for_3d = False

df = load_dataset_with_features(colony, load_local=True)
df["centroid_y_inv"] = 3120 - df["centroid_y"].values  # seg.shape[1]-cy
df["centroid_z_inv"] = 109 - df["centroid_z"].values  # seg.shape[0]-cz
df["centroid_z_inv_adjust"] = (
    df["centroid_z_inv"] * PIXEL_SIZE_Z_100x / PIXEL_SIZE_YX_100x
)  # convert from 0.108 to 0.29

# get info for this track at the specified timepoint
df_at_t = df[(df.index_sequence == timepoint_frame)]
df_at_t_for_track = df_at_t[(df_at_t.track_id == track_id)]

cell_label_num = df_at_t_for_track["label_img"].values[0]

# very important for color consistency define max value for the colormap based on the max value of the label_img for the whole FOV.
label_img_max = df_at_t["label_img"].astype(int).max()

# load the iamges
seg = export_helper.load_seg_fov_image(df_at_t)
egfp = export_helper.load_raw_fov_image(colony, timepoint_frame, channel="egfp")

# get [yx,zx,zy] slice lists for images
seg_slices = figure_helper.process_channels_into_slices(seg, "seg")
egfp_slices = figure_helper.process_channels_into_slices(egfp, "egfp")

# now define the ROI for the cell of interest by retrieving its crop ROI from the dataframe
crop_widths = np.asarray([200, 200])  # widths in Y and X; crop 100 pixels around the centroid
crop_widths_egfp = np.round(crop_widths * RESCALE_FACTOR_100x_to_20X, 0).astype("uint16")
cx, cy = df_at_t_for_track["centroid_x"].values[0], df_at_t_for_track["centroid_y"].values[0]

# get a crop from segmenation image
seg_crop = image_helper.get_single_crop(seg, cy, cx, crop_widths, 1, "uint16")
seg_crop[seg_crop != cell_label_num] = 0  # only keep the cell of interest
seg_crop_slices = figure_helper.process_channels_into_slices(seg_crop, "seg")

# get crop from egfp image
egfp_crop = image_helper.get_single_crop(
    egfp, cy, cx, crop_widths_egfp, RESCALE_FACTOR_100x_to_20X, "uint16"
)
egfp_crop_slices = figure_helper.process_channels_into_slices(egfp_crop, "egfp")

# rescale fluorescence intensities
egfp_slices_rs = figure_helper.rescale_intensities(egfp_slices, "egfp")
egfp_crop_slices_rs = figure_helper.rescale_intensities(egfp_crop_slices, "egfp")


seg_slices_label = [
    figure_helper.convert_2d_label_img_to_rgb(
        label_img=x, N=label_img_max, cell_id=int(cell_label_num), cell_color=cell_color
    )
    for x in seg_slices
]
seg_crop_slices_label = [
    figure_helper.convert_2d_label_img_to_rgb(
        label_img=x, N=label_img_max, cell_id=int(cell_label_num), cell_color=cell_color
    )
    for x in seg_crop_slices
]

list_of_contour_and_color_lists = figure_helper.get_contour_and_color_lists(
    seg_slices, label_img_max, cell_label_num, cell_color
)
list_of_contour_and_color_lists_crop = figure_helper.get_contour_and_color_lists(
    seg_crop_slices, label_img_max, cell_label_num, cell_color
)

adjust_cy = df_at_t_for_track["centroid_y_inv"].values[0]
crop_roi = image_helper.get_single_crop(
    seg, adjust_cy, cx, crop_widths, 1, "uint16", return_roi_array=True
)
fov_roi = np.asarray([0, seg.shape[0], 0, seg.shape[1], 0, seg.shape[2]])

# %%
list_of_rectangles_egfp = figure_helper.get_list_of_rectangles(crop_roi, mag="20x")
list_of_rectangles_seg = figure_helper.get_list_of_rectangles(crop_roi, mag="100x")

list_of_tail_lines_egfp = figure_helper.get_list_of_tail_lines(
    df,
    timepoint_frame,
    tail_length,
    seg_slices,
    "20x",
    fov_roi,
    2,
    label_img_max,
    cell_label_num,
    cell_color,
)
list_of_tail_lines_seg = figure_helper.get_list_of_tail_lines(
    df,
    timepoint_frame,
    tail_length,
    seg_slices,
    "100x",
    fov_roi,
    2,
    label_img_max,
    cell_label_num,
    cell_color,
)
list_of_tail_lines_egfp_crop = figure_helper.get_list_of_tail_lines(
    df,
    timepoint_frame,
    tail_length,
    seg_crop_slices,
    "20x",
    crop_roi,
    2,
    label_img_max,
    cell_label_num,
    cell_color,
)
list_of_tail_lines_seg_crop = figure_helper.get_list_of_tail_lines(
    df,
    timepoint_frame,
    tail_length,
    seg_crop_slices,
    "100x",
    crop_roi,
    2,
    label_img_max,
    cell_label_num,
    cell_color,
)

view_dict = {}
for si, yx_zx_zy_str in enumerate(["yx", "zx"]):
    view_dict[f"crop-{yx_zx_zy_str}"] = {
        "raw": egfp_crop_slices_rs[si],
        "label_cell": seg_crop_slices_label[si],
        "roi_patch-raw": list_of_rectangles_egfp[si],
        "roi_patch-seg": list_of_rectangles_seg[si],
        "contour_patches": list_of_contour_and_color_lists_crop[si],
        "line_dicts_list": list_of_tail_lines_egfp_crop[si],
    }
    view_dict[f"fullFOV-{yx_zx_zy_str}"] = {
        "raw": egfp_slices_rs[si],
        "label_cell": seg_slices_label[si],
        "roi_patch-raw": list_of_rectangles_egfp[si],
        "roi_patch-seg": list_of_rectangles_seg[si],
        "contour_patches": list_of_contour_and_color_lists[si],
        "line_dicts_list": list_of_tail_lines_egfp[si],
    }


# %%
# now there are three top panels with upper and lower panels (FOV)
# and three bottom panels with upper and lower panels (crops)
# from left to right it goes RAW, SEG, CONTOURS+tails
# now create a figure of 8.5 x 11 inches


# determine line width for tails and contours and ROI drawn with matplotlib
tail_linewidth = 0.5
contour_linewidth = 0.5
ROI_linewidth = 2
dpi = 300
fontsize = 8
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "Arial"
# now create a figure of 8.5 x 11 inches
# with the 4 images

# define names of colonies
name_lookup = {"small": "Small", "medium": "Medium", "large": "Large"}

# intialize the figure
# Create a figure with the desired total width
fig_width = 8.5  # Total width of the figure in inches
fig_height = 11  # Total height of the figure in inches

# Define the width of the axes in inches and convert to figure units
ax_width_inch = 2.5  # Width of the axes in inches

# define the x and y coordinates
ax_x_inch = 0.1  # x position of the axes in inches
ax_y_inch = 0.1  # y position of the axes in inches
ax_gap_inch = 0.1  # gap between axes in inches

ax_y_gap_inch = 0.1  # gap between top middle and bottom axes in inches
ax_y_small_gap_inch = 0.02  # gap between xy and zx axes in inches


class AxesCreator:
    def __init__(self, fig, img, zcrop_percentages):
        self.fig = fig
        self.img = img
        self.zcrop_percentages = zcrop_percentages
        fig_width, fig_height = fig.get_size_inches()
        self.fig_width = fig_width
        self.fig_height = fig_height

    def define_axis_size(self, ax_width_inch, ax_x_inch, ax_y_inch, ax_gap_inch):
        # determine axis height based on image aspect ratio
        if self.zcrop_percentages is None:
            ax_height_inch = ax_width_inch * (
                self.img.shape[0] / self.img.shape[1]
            )  # Height of the axes in inches
        else:
            self_height = self.img.shape[0] * (
                self.zcrop_percentages[1] - self.zcrop_percentages[0]
            )
            self_width = self.img.shape[1]
            ax_height_inch = ax_width_inch * (
                self_height / self_width
            )  # Height of the axes in inches

        # convert to figure units
        self.ax_width = ax_width_inch / self.fig_width  # Width of the axes in figure units
        self.ax_height = ax_height_inch / self.fig_height  # Height of the axes in figure units
        self.ax_x = ax_x_inch / self.fig_width  # x position of the axes in figure units
        self.ax_y = ax_y_inch / self.fig_height  # y position of the axes in figure units
        self.ax_gap = ax_gap_inch / self.fig_width  # gap between axes in figure units

    def add_axes(
        self,
        ax_x=None,
        ax_y=None,
        ax_width=None,
        ax_height=None,
    ):
        if ax_x is None:
            ax_x = self.ax_x
        if ax_y is None:
            ax_y = self.ax_y
        if ax_width is None:
            ax_width = self.ax_width
        if ax_height is None:
            ax_height = self.ax_height

        ax = self.fig.add_axes([ax_x, ax_y, ax_width, ax_height])
        ax.axis("off")
        return ax


zx_crop = [0.3, 0.7]  # percentage of the image to crop in the zx view (lower, higher)
# zx_crop = [0,1]
# for img_str in ['BOTTOM_LEFT_IMG','BOTTOM_CENTER_IMG','BOTTOM_RIGHT_IMG']:
#     pass
# Define the list of parameters
params = [
    ("crop-zx", "raw", "", "BOTTOM_LEFT_IMG_lower"),
    ("crop-zx", "seg", "", "BOTTOM_CENTER_IMG_lower"),
    ("crop-zx", "tails", "", "BOTTOM_RIGHT_IMG_lower"),
    ("crop-yx", "raw", "", "BOTTOM_LEFT_IMG"),
    ("crop-yx", "seg", "", "BOTTOM_CENTER_IMG"),
    ("crop-yx", "tails", "", "BOTTOM_RIGHT_IMG"),
    ("fullFOV-zx", "raw", "", "TOP_LEFT_IMG_lower"),
    ("fullFOV-zx", "seg", "", "TOP_CENTER_IMG_lower"),
    ("fullFOV-zx", "tails", "", "TOP_RIGHT_IMG_lower"),
    ("fullFOV-yx", "raw", "Fluorescence\n20x/0.8 NA", "TOP_LEFT_IMG"),
    ("fullFOV-yx", "seg", "Segmentation\n100x/1.25 NA", "TOP_CENTER_IMG"),
    ("fullFOV-yx", "tails", "Cell\ntracking", "TOP_RIGHT_IMG"),
]

# Create the figure
fig = plt.figure(figsize=(fig_width, fig_height))
# Loop over the parameters
ax_dict = {}
for i, (view_dict_str, overlays, titlestr, img_name) in enumerate(params):

    panel_dict = view_dict[f"{view_dict_str}"]
    if "raw" in overlays:
        img = panel_dict["raw"]
    elif "seg" in overlays:
        img = panel_dict["label_cell"]
    elif "tails" in overlays:
        img = panel_dict["raw"]

    if ("lower" in img_name) & ("crop" in view_dict_str):
        ac = AxesCreator(fig, img, zcrop_percentages=zx_crop)
    else:
        ac = AxesCreator(fig, img, zcrop_percentages=None)

    if ("BOTTOM" in img_name) and ("lower" in img_name):
        ax_y_inch_input = ax_y_inch
    elif ("BOTTOM" in img_name) and ("lower" not in img_name):
        ax_y_inch_input = (
            ax_dict["BOTTOM_LEFT_IMG_lower"].get_position().y1 * fig_height
        ) + ax_y_small_gap_inch

    elif ("TOP" in img_name) and ("lower" in img_name):
        ax_y_inch_input = (
            ax_dict["BOTTOM_LEFT_IMG"].get_position().y1 * fig_height
        ) + ax_y_gap_inch
    elif ("TOP" in img_name) and ("lower" not in img_name):
        ax_y_inch_input = (
            ax_dict["TOP_LEFT_IMG_lower"].get_position().y1 * fig_height
        ) + ax_y_small_gap_inch

    # (ax_bot_left.get_position().y1 * fig_height) + ax_y_gap_inch
    if "LEFT" in img_name:
        ax_x_factor = 0
    elif "CENTER" in img_name:
        ax_x_factor = 1
    elif "RIGHT" in img_name:
        ax_x_factor = 2

    ac.define_axis_size(
        ax_width_inch=ax_width_inch,
        ax_x_inch=ax_x_inch,
        ax_y_inch=ax_y_inch_input,
        ax_gap_inch=ax_gap_inch,
    )
    ax_x = ac.ax_x + (ax_x_factor * (ac.ax_width + ac.ax_gap))
    ax = ac.add_axes(ax_x=ax_x)
    ax_dict[img_name] = ax
    ax.imshow(
        img,
        cmap="gray",
        interpolation="nearest",
        zorder=1,
        aspect="auto",
    )

    if ("lower" in img_name) & ("crop" in view_dict_str):  # crop the image in Z using ylim
        img_height = img.shape[0]
        ylimits = [
            img_height * (1 - zx_crop[0]),
            img_height * (1 - zx_crop[1]),
        ]  # need to keep them reversed for image to display with Z=0 on bottom!
        print("cropping in Z")
        ax.set_ylim(ylimits)

    # add timepoint to the top of the image
    if "lower" not in img_name:
        ax.set_title(f"{titlestr}", fontsize=fontsize)

    os.makedirs(savedir, exist_ok=True)
    save_path = os.path.join(savedir, fig_panel_str + img_name + ".png")
    two_d_writer.TwoDWriter.save(img, uri=save_path)

    # draw the ROI
    if ("fullFOV" in view_dict_str) & ("lower" not in img_name):
        if "seg" in overlays:
            rect_list = panel_dict["roi_patch-seg"]
        else:
            rect_list = panel_dict["roi_patch-raw"]
        pc_roi_rect = PatchCollection(
            rect_list,
            edgecolor=[1, 1, 0],
            alpha=0.5,
            # linewidth=ROI_linewidth,
            linewidths=figure_helper.determine_linewidth_from_desired_microns(
                fig, ax, ROI_linewidth
            ),
            facecolor="None",
            zorder=1000,
        )
        ax.add_collection(pc_roi_rect)

    if "tails" in overlays:
        # draw the contours
        contour_and_color_list = panel_dict["contour_patches"]
        for contour_and_color in contour_and_color_list:
            contour = contour_and_color[0]

            # contour.set_linewidth(contour_linewidth)
            if "crop" in view_dict_str:
                linestyle = (0, (5, 5))  # (offset,  (point length,  point gap))
                tail_linewidth_adj = figure_helper.determine_linewidth_from_desired_microns(
                    fig, ax, tail_linewidth
                )
                contour_linewidth_adj = (
                    0.5
                    * figure_helper.determine_linewidth_from_desired_microns(
                        fig, ax, contour_linewidth
                    )
                )
            else:
                linestyle = (0, (5, 0))  # '-'
                contour_linewidth_adj = figure_helper.determine_linewidth_from_desired_microns(
                    fig, ax, contour_linewidth
                )
                tail_linewidth_adj = figure_helper.determine_linewidth_from_desired_microns(
                    fig, ax, tail_linewidth
                )

            pc_contours = PatchCollection(
                [contour],
                edgecolor=contour_and_color[1],
                linewidths=contour_linewidth_adj,
                facecolor="None",
                zorder=1000,
                linestyle=linestyle,
            )
            ax.add_collection(pc_contours)

        # draw the tails
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        for line_dicts in panel_dict["line_dicts_list"]:
            for line_dict in line_dicts:
                x1 = line_dict["x"][0]
                x2 = line_dict["x"][1]
                y1 = line_dict["y"][0]
                y2 = line_dict["y"][1]
                tailcolor = line_dict["color"]

                # ax.plot([x1, x2], [y1, y2], color=tailcolor, linewidth=tail_linewidth_adj, zorder=1000)
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=tailcolor,
                    linewidth=tail_linewidth_adj,
                    marker="o",
                    markersize=0,
                    zorder=1000,
                )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


for ax_name, axx in ax_dict.items():

    if "lower" in ax_name:
        add_text = False
        view = "ZX"
    else:
        add_text = True
        view = "XY"

    if "CENTER" in ax_name:
        is_seg = True  # this will make sure the correct pixel sizes are retrieved
    else:
        is_seg = False

    if "TOP" in ax_name:
        scalebarum = 20
    else:
        scalebarum = 5

    figure_helper.draw_scale_bar_matplotlib(
        axx,
        colony,
        scalebarum=scalebarum,
        add_text=add_text,
        is_seg=is_seg,
        loc="right",
        fontsize=fontsize,
    )
    # if ("TOP" in ax_name) & ("lower" in ax_name):
    #    figure_helper. add_view_arrows(axx, view=view, fontsize=5, shrink_factor=0.75, start_coord=(8, 2))
    # else:
    #     figure_helper.add_view_arrows(axx, view=view)

    # axx.set_title(titlestr)

# now save the figure as a pdf
savedir, fig_panel_str = figure_helper.get_save_dir_and_fig_panel_str(figure, panel)
savepath1 = os.path.join(savedir, fig_panel_str + ".pdf")
savepath2 = os.path.join(
    savedir, fig_panel_str + f"_{track_id}.pdf"
)  # save one version with track_id information in the name
for savepath in [savepath1, savepath2]:
    for suffix in [".pdf", ".png"]:
        print(os.path.abspath(savepath))
        fig.savefig(
            savepath.replace(".pdf", suffix),
            bbox_inches="tight",
            dpi=dpi,
            transparent=True,
        )
        os.chmod(savepath.replace(".pdf", suffix), 0o777)


# %%
