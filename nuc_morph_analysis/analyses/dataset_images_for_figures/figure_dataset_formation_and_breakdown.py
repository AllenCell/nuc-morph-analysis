# %%
# create the saving directory if it doesn"'t exist
import os
import matplotlib.pyplot as plt
from aicsimageio.writers import two_d_writer
from nuc_morph_analysis.analyses.dataset_images_for_figures import figure_helper
from matplotlib.collections import PatchCollection  # type: ignore
import pandas as pd

from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
from nuc_morph_analysis.lib.visualization.example_tracks import EXAMPLE_TRACKS
import matplotlib

figure = "dataset"
panel = "formation_and_breakdown"
savedir, fig_panel_str = figure_helper.get_save_dir_and_fig_panel_str(figure, panel)

# %%
# get the tracking dataframe

# define dataset from which to collect images
# collect information for the dataset
colony = "medium"

#  load the tracking CSV for medium from FMS
df = global_dataset_filtering.load_dataset_with_features(colony)
df_fmb = figure_helper.assemble_formation_middle_breakdown_dataframe(df)
# load the images for each timepoint
# %%
seg_img_list, raw_img_list = figure_helper.load_images_for_formation_middle_breakdown(
    df_fmb, df, colony
)
# %%
df_fmb = figure_helper.process_images_and_add_to_dataframe(df_fmb, df, seg_img_list, raw_img_list)
# %%
color_dict = {
    "formation": "tab:orange",
    "breakdown": "tab:blue",
    "middle": "tab:green",
}
# define the order of the classes through which to iterate
class_list = ["formation", "middle", "breakdown"]
# define the list of the views of the images to save
projection_list = ["yx", "zx"]
scalebarum = 10

# %%
# now there are three top panels with upper and lower panels (formation)
# and three middle panels with upper and lower panels (middle)
# and three bottom panels with upper and lower panels (breakdown)
# from left to right it goes 0,1,2,3,4 for all
# now create a figure of 8.5 x 11 inches

# determine line width for tails and contours and ROI drawn with matplotlib
contour_linewidth = 0.3
axes_outline_width = 2
dpi = 400
fontsize = 8

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "Arial"
# intialize the figure
# create a figure with the desired total width
fig_width = 8.5  # total width of the figure in inches
fig_height = 11  # total height of the figure in inches

# define the width of the axes in inches and convert to figure units
ax_width_inch = 1.5  # width of the axes in inches

# define the x and y coordinates
ax_x_inch = 0.1  # x position of the axes in inches
ax_y_inch = 0.1  # y position of the axes in inches
ax_gap_inch = 0.03  # gap between axes in inches

ax_y_gap_inch = 0.4  # gap between top middle and bottom axes in inches
ax_y_small_gap_inch = 0.02  # gap between yx and zx axes in inches


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


# for img_str in ['BOTTOM_LEFT_IMG','BOTTOM_CENTER_IMG','BOTTOM_RIGHT_IMG']:
#     pass
# Define the list of parameters
import itertools

ti_list = [0, 1, 2, 3, 4]
class_list = ["breakdown", "middle", "formation"]
view_list = ["zx", "yx"]
params = list(itertools.product(class_list, view_list, ti_list))
params


zx_crop = [0.2, 0.7]  # percentage of the image to crop in the zx view (lower, higher)
# Create the figure
fig = plt.figure(figsize=(fig_width, fig_height))
# Loop over the parameters
ax_dict = {}
dfb = df_fmb.reset_index().set_index(["class", "t"])
for i, (class_name, view, ti) in enumerate(params):

    img_name = f"{class_name}_{view}_{ti}"
    img = dfb.loc[(class_name, ti), f"raw_{view}"][0]
    seg_contours = dfb.loc[(class_name, ti), f"seg_{view}_contours"][0]

    if "zx" in img_name:
        ac = AxesCreator(fig, img, zcrop_percentages=zx_crop)
    else:
        ac = AxesCreator(fig, img, zcrop_percentages=None)

    if ("breakdown" in img_name) and ("zx" in img_name):
        ax_y_inch_input = ax_y_inch
    elif ("breakdown" in img_name) and ("zx" not in img_name):
        ax_y_inch_input = (
            ax_dict["breakdown_zx_0"].get_position().y1 * fig_height
        ) + ax_y_small_gap_inch

    elif ("middle" in img_name) and ("zx" in img_name):
        ax_y_inch_input = (ax_dict["breakdown_yx_0"].get_position().y1 * fig_height) + ax_y_gap_inch
    elif ("middle" in img_name) and ("zx" not in img_name):
        ax_y_inch_input = (
            ax_dict["middle_zx_0"].get_position().y1 * fig_height
        ) + ax_y_small_gap_inch

    elif ("formation" in img_name) and ("zx" in img_name):
        ax_y_inch_input = (ax_dict["middle_yx_0"].get_position().y1 * fig_height) + ax_y_gap_inch
    elif ("formation" in img_name) and ("zx" not in img_name):
        ax_y_inch_input = (
            ax_dict["formation_zx_0"].get_position().y1 * fig_height
        ) + ax_y_small_gap_inch

    # (ax_bot_left.get_position().y1 * fig_height) + ax_y_gap_inch
    ax_x_factor = ti
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

    if "zx" in img_name:  # crop the image in Z using ylim
        img_height = img.shape[0]
        ylimits = [
            img_height * (1 - zx_crop[0]),
            img_height * (1 - zx_crop[1]),
        ]  # need to keep them reversed for image to display with Z=0 on bottom!
        ax.set_ylim(ylimits)
        print("cropping in Z")

    # add the name of the colony to the left of the image
    if ti == 0:
        ylabel = "top view"
        if "zx" in img_name:
            ylabel = "side view"
        ax.set_ylabel(ylabel, rotation=90, fontsize=fontsize)
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

    if pd.isna(dfb.loc[(class_name, ti), f"title_str"]) == False:
        ax.set_title(
            dfb.loc[(class_name, ti), f"title_str"],
            fontsize=fontsize + 1,
            color=color_dict[class_name],
            fontweight="bold",
        )

    if ti == 2:
        # if "zx" not in img_name:
        #     ax.set_title(
        #         f"{class_name}",
        #         fontsize=fontsize + 1,
        #         color=color_dict[class_name],
        #         fontweight="bold",
        #     )
        ax.axis("on")
        ax.set_yticks([])
        # only keep y axis
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel("")

        color = color_dict[class_name]
        # edit the spines
        ax.tick_params(color=color, labelcolor=color)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(axes_outline_width)

    # add timepoint to the top of the image
    if "zx" not in img_name:
        timpoint = int(dfb.loc[(class_name, ti), "index_sequence"] * 5)
        ax.text(
            0.02,
            0.98,
            f"{timpoint} min",
            horizontalalignment="left",
            fontsize=fontsize,
            color="w",
            verticalalignment="top",
            transform=ax.transAxes,
        )

    save_path = os.path.join(savedir, fig_panel_str + img_name + ".png")
    two_d_writer.TwoDWriter.save(img, uri=save_path)

    # draw the contours
    contour_and_color_list = dfb.loc[(class_name, ti), f"seg_{view}_contours"][0]
    if (contour_and_color_list is not None) & (contour_and_color_list != "None"):
        for contour_and_color in contour_and_color_list:
            contour = contour_and_color[0]
            # contour.set_linewidth(contour_linewidth)
            linestyle = (0, (5, 5))  # 5 point length, 10 point gap

            pc_contours = PatchCollection(
                [contour],
                edgecolor=contour_and_color[1],
                linewidths=figure_helper.determine_linewidth_from_desired_microns(
                    fig, ax, contour_linewidth
                ),
                facecolor="None",
                zorder=1000,
                linestyle=linestyle,
            )
            ax.add_collection(pc_contours)

    # # add matplotlib based scalebars to each image

    for ax_name, ax in ax_dict.items():

        if "zx" in ax_name:
            add_text = False
            view = "ZX"
        else:
            add_text = True
            view = "YX"
        figure_helper.draw_scale_bar_matplotlib(
            ax, colony, scalebarum=5, add_text=add_text, fontsize=fontsize, loc="right"
        )
        figure_helper.add_view_arrows(ax, view=view)


# now save the figure as a pdf
track_id = EXAMPLE_TRACKS["figure_dataset_formation_and_breakdown"]
savedir, fig_panel_str = figure_helper.get_save_dir_and_fig_panel_str(figure, panel)
savepath1 = os.path.join(savedir, fig_panel_str + ".pdf")
savepath2 = os.path.join(
    savedir, fig_panel_str + f"_track={track_id}.pdf"
)  # save with track_id info in the name
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
