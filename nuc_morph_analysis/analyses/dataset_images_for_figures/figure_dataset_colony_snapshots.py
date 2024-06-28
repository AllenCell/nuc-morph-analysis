# %%
# get the info for the datasets
import os
import matplotlib.pyplot as plt
import numpy as np
from aicsimageio.writers import two_d_writer
from nuc_morph_analysis.analyses.dataset_images_for_figures import figure_helper

from nuc_morph_analysis.lib.preprocessing.load_data import (
    get_dataset_info_by_name,
    get_dataset_original_file_reader,
)
from nuc_morph_analysis.lib.preprocessing.single_track_contact.export_code import export_helper
import matplotlib


figure = "dataset"
panel = "colony_snapshots"
savedir, fig_panel_str = figure_helper.get_save_dir_and_fig_panel_str(figure, panel)

# define datasets from which to collect images
colonies = ["small", "medium", "large"]

# load all the images needed for figure
# choose the timepoints to load (beginning middle and end)
timepoints_hrs_list = np.asarray([0, 24, 47 + (25 / 60)])  # hrs

# collect beginning middle and end frames from each movie
panel_dict = {}
for colony in colonies:
    timepoints_frames_list = (
        timepoints_hrs_list * 60 / get_dataset_info_by_name(colony)["time_interval"]
    )  # days x (hrs x min /  min)
    timepoints_frames_list = timepoints_frames_list.astype("uint16")
    # moving here in case this loads from Zarr instead of CZI, although it may be slower
    reader = get_dataset_original_file_reader(colony)
    for timepoint_frame in timepoints_frames_list:
        egfp = export_helper.load_raw_fov_image(colony, timepoint_frame, reader, channel="egfp")
        egfp_slices = figure_helper.process_channels_into_slices(egfp, "egfp")
        egfp_slices_rs = figure_helper.rescale_intensities(egfp_slices, "egfp")

        panel_dict[(colony, timepoint_frame, "egfp", "yx")] = egfp_slices_rs[0]
        panel_dict[(colony, timepoint_frame, "egfp", "zx")] = egfp_slices_rs[1]
print("images loaded")

# %%
# define names of colonies
fontsize = 8
# intialize the figure
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "Arial"
# Create a figure with the desired total width
new_fig_width = 3.6
scale_widths = new_fig_width / 8.5
fig_width = 8.5 * scale_widths  # Total width of the figure in inches
fig_height = 11  # Total height of the figure in inches

# Define the width of the axes in inches and convert to figure units
ax_width_inch = 2.5 * scale_widths  # Width of the axes in inches

# define the x and y coordinates
ax_x_inch = 0.1  # x position of the axes in inches
ax_y_inch = 0.1  # y position of the axes in inches
ax_gap_inch = 0.1  # gap between axes in inches

ax_y_gap_inch = 0.1  # gap between top middle and bottom axes in inches
ax_y_small_gap_inch = 0.02  # gap between xy and zx axes in inches


class AxesCreator:
    def __init__(self, fig, img):
        self.fig = fig
        self.img = img
        fig_width, fig_height = fig.get_size_inches()
        self.fig_width = fig_width
        self.fig_height = fig_height

    def define_axis_size(self, ax_width_inch, ax_x_inch, ax_y_inch, ax_gap_inch):
        # determine axis height based on image aspect ratio
        ax_height_inch = ax_width_inch * (
            self.img.shape[0] / self.img.shape[1]
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


# Define the list of parameters
params = [
    ("large", timepoints_frames_list[0], "egfp", "zx", "BOTTOM_LEFT_IMG_lower"),
    ("large", timepoints_frames_list[1], "egfp", "zx", "BOTTOM_CENTER_IMG_lower"),
    ("large", timepoints_frames_list[2], "egfp", "zx", "BOTTOM_RIGHT_IMG_lower"),
    ("large", timepoints_frames_list[0], "egfp", "yx", "BOTTOM_LEFT_IMG"),
    ("large", timepoints_frames_list[1], "egfp", "yx", "BOTTOM_CENTER_IMG"),
    ("large", timepoints_frames_list[2], "egfp", "yx", "BOTTOM_RIGHT_IMG"),
    ("medium", timepoints_frames_list[0], "egfp", "zx", "MIDDLE_LEFT_IMG_lower"),
    ("medium", timepoints_frames_list[1], "egfp", "zx", "MIDDLE_CENTER_IMG_lower"),
    ("medium", timepoints_frames_list[2], "egfp", "zx", "MIDDLE_RIGHT_IMG_lower"),
    ("medium", timepoints_frames_list[0], "egfp", "yx", "MIDDLE_LEFT_IMG"),
    ("medium", timepoints_frames_list[1], "egfp", "yx", "MIDDLE_CENTER_IMG"),
    ("medium", timepoints_frames_list[2], "egfp", "yx", "MIDDLE_RIGHT_IMG"),
    ("small", timepoints_frames_list[0], "egfp", "zx", "TOP_LEFT_IMG_lower"),
    ("small", timepoints_frames_list[1], "egfp", "zx", "TOP_CENTER_IMG_lower"),
    ("small", timepoints_frames_list[2], "egfp", "zx", "TOP_RIGHT_IMG_lower"),
    ("small", timepoints_frames_list[0], "egfp", "yx", "TOP_LEFT_IMG"),
    ("small", timepoints_frames_list[1], "egfp", "yx", "TOP_CENTER_IMG"),
    ("small", timepoints_frames_list[2], "egfp", "yx", "TOP_RIGHT_IMG"),
]

# Create the figure
fig = plt.figure(figsize=(fig_width, fig_height))
# Loop over the parameters
ax_dict = {}
for i, (colony, num, color, plane, img_name) in enumerate(params):
    img = panel_dict[(colony, num, color, plane)]
    ac = AxesCreator(fig, img)

    if ("BOTTOM" in img_name) and ("lower" in img_name):
        ax_y_inch_input = ax_y_inch
    elif ("BOTTOM" in img_name) and ("lower" not in img_name):
        ax_y_inch_input = (
            ax_dict["BOTTOM_LEFT_IMG_lower"].get_position().y1 * fig_height
        ) + ax_y_small_gap_inch
    elif ("MIDDLE" in img_name) and ("lower" in img_name):
        ax_y_inch_input = (
            ax_dict["BOTTOM_LEFT_IMG"].get_position().y1 * fig_height
        ) + ax_y_gap_inch
    elif ("MIDDLE" in img_name) and ("lower" not in img_name):
        ax_y_inch_input = (
            ax_dict["MIDDLE_LEFT_IMG_lower"].get_position().y1 * fig_height
        ) + ax_y_small_gap_inch
    elif ("TOP" in img_name) and ("lower" in img_name):
        ax_y_inch_input = (
            ax_dict["MIDDLE_LEFT_IMG"].get_position().y1 * fig_height
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
    ax.imshow(img, cmap="gray", interpolation="nearest")

    # add the name of the colony to the left of the image
    if ("LEFT" in img_name) & ("lower" not in img_name):
        ax.set_ylabel(colony.capitalize(), fontsize=fontsize)
        ax.axis("on")
        ax.set_yticks([])
        # only keep y axis
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel("")
        # remove all spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    # add timepoint to the top of the image
    if ("TOP" in img_name) & ("lower" not in img_name):
        interval = get_dataset_info_by_name(colony)["time_interval"]
        # ax.set_title(f"{int(num*5/60)} hr", fontsize=fontsize)
        if (num * interval / 60) % 1 == 0:
            hr = int(num * interval / 60)
            timestr = f"{hr} hr"
        else:
            hr = int(np.floor(num * interval / 60))
            min = int((num * interval) % 60)
            timestr = f"{hr} hr {min} min"

        ax.set_title(f"{timestr}", fontsize=fontsize)
    save_path = os.path.join(savedir, fig_panel_str + img_name + ".png")
    two_d_writer.TwoDWriter.save(img, uri=save_path)

# # add matplotlib based to one image
figure_helper.draw_scale_bar_matplotlib(
    ax_dict["BOTTOM_LEFT_IMG"], colony, scalebarum=20, add_text=True, loc="right"
)

# now save the figure as a pdf
savedir, fig_panel_str = figure_helper.get_save_dir_and_fig_panel_str(figure, panel)
# savepath = os.path.join(savedir, fig_panel_str + ".pdf")
savepath = savedir / f"{fig_panel_str}.pdf"
for suffix in [".pdf", ".png"]:
    print(os.path.abspath(savepath))
    fig.savefig(
        str(savepath).replace(".pdf", suffix),
        bbox_inches="tight",
        dpi=600,
        transparent=True,
    )
