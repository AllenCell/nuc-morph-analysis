# %%
# create the saving directory if it doesn"'t exist
import os
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection  # type: ignore
import pandas as pd

from nuc_morph_analysis.analyses.dataset_images_for_figures import figure_helper
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
from nuc_morph_analysis.lib.visualization.example_tracks import EXAMPLE_TRACKS

from nuc_morph_analysis.analyses.dataset_images_for_figures.figure_helper import AxesCreator

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "Arial"

load_local = False
make_frames = True

# set figure, panel to get save directory info
figure = "dataset"
panel = "formation_and_breakdown"
savedir, fig_panel_str = figure_helper.get_save_dir_and_fig_panel_str(figure, panel)

# set colony to medium
colony = "medium"

if make_frames:
    if load_local:
        if not os.path.exists(savedir/ "df_fmb.pkl"):
            print("no saved df_fmb.pkl, processing images and saving")
            #  load the tracking CSV for medium from FMS
            # define dataset from which to collect images
            # collect information for the dataset
            df = global_dataset_filtering.load_dataset_with_features()
            df = df[df["colony"] == colony]
            df_fb = figure_helper.assemble_formation_breakdown_movie_dataframe(df)
            # load the images for each timepoint and add them to the dataframe
            seg_img_list, raw_img_list = figure_helper.load_images_for_formation_middle_breakdown(
                df_fb, df, colony
            )
            df_fb = figure_helper.process_images_and_add_to_dataframe(df_fb, df, seg_img_list, raw_img_list)
            df_fb.to_pickle(os.path.join(savedir, "df_fmb.pkl"))
            print("saved df_fmb.pkl")
        else:
            print("reading from saved df_fmb.pkl")
            df_fb = pd.read_pickle(os.path.join(savedir, "df_fmb.pkl"))

    # %%
    # determine line width for tails and contours and ROI drawn with matplotlib
    contour_linewidth = 0.3
    axes_outline_width = 2
    dpi = 400
    fontsize = 8

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

    # Create the figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    zx_crop = [0.2, 0.7]  # percentage of the image to crop in the zx view (lower, higher)

    # %%
    # Loop over frames and views (ZX, ZY) to create the movie
    ax_dict = {}
    dfb = df_fb.reset_index().set_index("t")
    for i in range(dfb.shape[0]):
        for v, view in enumerate(["zx", "yx"]):

            img_name = f"{view}_{i}"
            img = dfb.loc[i, f"raw_{view}"][0]
            seg_contours = dfb.loc[i, f"seg_{view}_contours"][0]

            if "zx" in img_name:
                ac = AxesCreator(fig, img, zcrop_percentages=zx_crop)
            else:
                ac = AxesCreator(fig, img, zcrop_percentages=None)

            if "zx" in img_name:
                ax_y_inch_input = ax_y_inch
            else:
                ax_y_inch_input = (
                    ax_dict["zx_0"].get_position().y1 * fig_height
                ) + ax_y_small_gap_inch

            ac.define_axis_size(
                ax_width_inch=ax_width_inch,
                ax_x_inch=ax_x_inch,
                ax_y_inch=ax_y_inch_input,
                ax_gap_inch=ax_gap_inch,
            )
            ax_x = ac.ax_x
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

            # draw the segmentation contours as dashed yellow lines
            contour_and_color_list = dfb.loc[i, f"seg_{view}_contours"][0]
            if (contour_and_color_list is not None) & (contour_and_color_list != "None"):
                for contour_and_color in contour_and_color_list:
                    contour = contour_and_color[0]
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

        for ax_name, ax in ax_dict.items():
            # add scale bar, timestamp and view arrows as appropriate
            if "zx" in ax_name:
                view = "ZX"
            else:
                # for just the YX view, add a scale bar and timestamp
                view = "YX"
                figure_helper.draw_scale_bar_matplotlib(
                    ax, colony, scalebarum=5, add_text=True, fontsize=fontsize, loc="right"
                )
                frame = dfb.loc[i, "index_sequence"]
                figure_helper.draw_timestamp(ax, frame, fontsize=fontsize)
            # add arrows that orient the viewer
            figure_helper.add_view_arrows(ax, view=view)


        # now save the figure as a png
        track_id = EXAMPLE_TRACKS["figure_dataset_formation_and_breakdown"]
        savepath = os.path.join(savedir, "movie", f"frame_{i}.png")
        fig.savefig(
            savepath,
            bbox_inches="tight",
            dpi=dpi,
            transparent=True,
        )
        os.chmod(savepath, 0o777)

print ("Making mp4")
# get list of paths to all movie frames, make sure they are
# sorted in timepoint order then save to an mp4 movie
frame_list = os.listdir(savedir / "movie")
sorted_frames = sorted(frame_list, key=lambda x: int(x.split("_")[1].split(".")[0]))
figure_helper.make_mp4(savedir, "formation_breakdown_example_movie", sorted_frames)
