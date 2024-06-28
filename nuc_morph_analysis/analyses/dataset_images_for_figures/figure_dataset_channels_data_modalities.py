# %%
# get the info for the colonys
import matplotlib.pyplot as plt
from aicsimageio.writers import two_d_writer
from nuc_morph_analysis.analyses.dataset_images_for_figures import figure_helper
from nuc_morph_analysis.lib.preprocessing.load_data import get_dataset_original_file_reader
from nuc_morph_analysis.lib.preprocessing.single_track_contact.export_code import export_helper
import matplotlib


# define the directory where the images will be saved
figure = "dataset"
panel = "channels_data_modalities"
savedir, fig_panel_str = figure_helper.get_save_dir_and_fig_panel_str(figure, panel)

# define colony from which to collect images, and which timepoint
colony = "medium"
timepoint_frame = 48

reader = get_dataset_original_file_reader(colony)
bright = export_helper.load_raw_fov_image(colony, timepoint_frame, reader, channel="bright")
egfp = export_helper.load_raw_fov_image(colony, timepoint_frame, reader, channel="egfp")

bright_slices = figure_helper.process_channels_into_slices(bright, "bright")
egfp_slices = figure_helper.process_channels_into_slices(egfp, "egfp")

bright_slices_rs = figure_helper.rescale_intensities(bright_slices, "bright")
egfp_slices_rs = figure_helper.rescale_intensities(egfp_slices, "egfp")

print("images loaded")
# %%
panel_dict = {}
panel_dict[("medium", timepoint_frame, "bright", "yx")] = bright_slices_rs[0]
panel_dict[("medium", timepoint_frame, "bright", "zx")] = bright_slices_rs[1]
panel_dict[("medium", timepoint_frame, "egfp", "yx")] = egfp_slices_rs[0]
panel_dict[("medium", timepoint_frame, "egfp", "zx")] = egfp_slices_rs[1]

# now create a figure of 8.5 x 11 inches
# with the 4 images

# intialize the figure
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "Arial"
# Create a figure with the desired total width
new_fig_width = 4.4
scale_widths = new_fig_width / 8.5
fig_width = 8.5 * scale_widths  # Total width of the figure in inches
fig_height = 11  # Total height of the figure in inches

# Define the width of the axes in inches and convert to figure units
ax_width_inch = 4 * scale_widths  # Width of the axes in inches

# define the x and y coordinates
ax_x_inch = 0.1  # x position of the axes in inches
ax_y_inch = 0.1  # y position of the axes in inches
ax_gap_inch = 0.1  # gap between axes in inches

ax_y_gap_inch = 0.05  # gap between axes in inches


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


fig = plt.figure(figsize=(fig_width, fig_height))
img = panel_dict[("medium", timepoint_frame, "bright", "zx")]
ac_bot = AxesCreator(fig, img)
ac_bot.define_axis_size(
    ax_width_inch=ax_width_inch,
    ax_x_inch=ax_x_inch,
    ax_y_inch=ax_y_inch,
    ax_gap_inch=ax_gap_inch,
)
ax_bot_left = ac_bot.add_axes()
ax_bot_left.imshow(img, cmap="gray", interpolation="nearest")
# now save out the image
save_path = savedir / f"{fig_panel_str}BOTTOM_LEFT_IMG.png"
two_d_writer.TwoDWriter.save(img, uri=save_path)


img = panel_dict[("medium", timepoint_frame, "egfp", "zx")]
ac_bot.define_axis_size(
    ax_width_inch=ax_width_inch,
    ax_x_inch=ax_x_inch,
    ax_y_inch=ax_y_inch,
    ax_gap_inch=ax_gap_inch,
)
ax_bot_right = ac_bot.add_axes(ax_x=ac_bot.ax_x + ac_bot.ax_width + ac_bot.ax_gap)
ax_bot_right.imshow(img, cmap="gray", interpolation="nearest")
save_path = savedir / f"{fig_panel_str}BOTTOM_RIGHT_IMG.png"
two_d_writer.TwoDWriter.save(img, uri=save_path)

bot_position = ax_bot_left.get_position()
ax_y_inch2 = (bot_position.y1 * fig_height) + ax_y_gap_inch
img = panel_dict[("medium", timepoint_frame, "bright", "yx")]
ac_top = AxesCreator(fig, img)
ac_top.define_axis_size(
    ax_width_inch=ax_width_inch,
    ax_x_inch=ax_x_inch,
    ax_y_inch=ax_y_inch2,
    ax_gap_inch=ax_gap_inch,
)
ax_top_left = ac_top.add_axes()
ax_top_left.imshow(img, cmap="gray", interpolation="nearest")
save_path = savedir / f"{fig_panel_str}TOP_LEFT_IMG.png"
two_d_writer.TwoDWriter.save(img, uri=save_path)

img = panel_dict[("medium", timepoint_frame, "egfp", "yx")]
ac_top.define_axis_size(
    ax_width_inch=ax_width_inch,
    ax_x_inch=ax_x_inch,
    ax_y_inch=ax_y_inch2,
    ax_gap_inch=ax_gap_inch,
)
ax_top_right = ac_top.add_axes(ax_x=ac_top.ax_x + ac_top.ax_width + ac_top.ax_gap)
ax_top_right.imshow(img, cmap="gray", interpolation="nearest")
save_path = savedir / f"{fig_panel_str}TOP_RIGHT_IMG.png"
two_d_writer.TwoDWriter.save(img, uri=save_path)

figure_helper.draw_scale_bar_matplotlib(
    ax_top_left, colony, scalebarum=20, add_text=True, loc="right"
)

ax_top_left.set_title("Transmitted light (bright-field)", fontsize=8)
ax_top_right.set_title("EGFP-tagged lamin B1 fluorescence", fontsize=8)

# now save the figure as a pdf
savedir, fig_panel_str = figure_helper.get_save_dir_and_fig_panel_str(figure, panel)
save_path = savedir / f"{fig_panel_str}.pdf"
for suffix in [".pdf", ".png"]:
    print(save_path)
    fig.savefig(
        str(save_path).replace(".pdf", suffix),
        bbox_inches="tight",
        dpi=600,
        transparent=True,
    )
# %%
