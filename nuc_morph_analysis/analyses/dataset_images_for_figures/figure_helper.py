from pathlib import Path
import numpy as np
import pandas as pd
from skimage import measure
import skimage.exposure as skex
from matplotlib.lines import Line2D
from nuc_morph_analysis.lib.preprocessing.system_info import (
    PIXEL_SIZE_YX_100x,
    PIXEL_SIZE_YX_20x,
    RESCALE_FACTOR_100x_to_20X,
)
from nuc_morph_analysis.lib.preprocessing.single_track_contact.vis_code import image_helper
from matplotlib.patches import Rectangle
from nuc_morph_analysis.lib.visualization.example_tracks import EXAMPLE_TRACKS
from nuc_morph_analysis.lib.preprocessing.load_data import get_dataset_original_file_reader
from nuc_morph_analysis.lib.preprocessing.single_track_contact.export_code import export_helper
from tqdm import tqdm
from matplotlib.patches import Polygon
from skimage import measure

INTENSITIES_DICT = {
    "egfp_max": (110, 140),
    "egfp_mid": (100, 135),
    "bright_mid": (23000, 35000),
    "bright_max": (23000, 35000),
}

MAX_SUM_SLICE_DICT = {
    "egfp": ("max", "mid", "mid"),
    "bright": ("mid", "mid", "mid"),
    "seg": ("max", "mid", "mid"),
}


def process_channels_into_slices(img, channel, dtype="uint16"):
    """
    Process the image into slices [yx,zx,zy] that are isotropic in pixel size

    Parameters
    ----------
    img : np.array
        The image to process
    channel : str
        The channel of the image (either "bright" or "egfp" or "seg")

    Returns
    -------
    slices : list
        The slices of the image in following order [yx,zx,zy]
    """

    max_sum_slice = MAX_SUM_SLICE_DICT[channel]
    order = 1 if channel == "egfp" else 0
    scale_factors = image_helper.get_scale_factors_for_resizing("20x")
    if channel == "seg":
        scale_factors = image_helper.get_scale_factors_for_resizing("100x")
    slices = image_helper.process_image_into_slices(
        img, max_sum_slice=max_sum_slice, order=0, scale_factors=scale_factors, dtype=dtype
    )
    return slices


def rescale_intensities(slices, bright_or_egfp, dtype="uint8"):
    """
    Rescale the intensities of the slices

    Parameters
    ----------
    slices : list
        The slices of the image in following order [yx,zx,zy]
    bright_or_egfp : str
        The channel of the image (either "bright" or "egfp")

    Returns
    -------
    slices_rs : list
        The rescaled slices
    """
    max_sum_slice = MAX_SUM_SLICE_DICT[bright_or_egfp]
    slices_rs = []
    for si, img_slice in enumerate(slices):
        intensities = INTENSITIES_DICT[f"{bright_or_egfp}_{max_sum_slice[si]}"]
        slices_rs.append(
            skex.rescale_intensity(image=img_slice, in_range=intensities, out_range=dtype).astype(
                dtype
            )
        )
    return slices_rs


def get_save_dir_and_fig_panel_str(figure, panel):
    fig_panel_str = f"f-{figure}-{panel}"
    savedir = Path(__file__).parent / f"Figure-{figure}/{fig_panel_str}"

    # now create the saving directory if it doesn't exist
    if not savedir.exists():
        savedir.mkdir(parents=True)
    return savedir, fig_panel_str


def return_glasbey_on_dark(N=255, cell_id=None, cell_color=None):
    """
    Publication
    The Glasbey LUT is based on the publication:

    Glasbey, C., van der Heijden, G., Toh, V. F. K., & Gray, A. (2007). Colour displays for categorical images. Color Research &amp; Application, 32(4), 304–309. doi:10.1002/col.20327

    arguments:
    N: int, the number of colors to return
    cell_id: int, the cell id to change the color of
    cell_color: array, the RGB color to change the cell to
    """
    from matplotlib.colors import LinearSegmentedColormap
    from nuc_morph_analysis.analyses.dataset_images_for_figures.glasbey_on_dark import (
        glasbey_on_dark_func,
    )

    glasbey_on_dark254 = glasbey_on_dark_func()
    # loop through glasbey_on_dark
    glasbey_on_dark = []
    glasbey_on_dark.append(
        [
            0,
            0,
            0,
        ]
    )
    for i in range(N):  # range is 0 to N-1
        glasbey_on_dark.append(glasbey_on_dark254[i % len(glasbey_on_dark254)])
    # full array is N+1 colors long so that the background color is not used, so that background = 0
    rgb_array0_255 = np.asarray(glasbey_on_dark)

    if cell_id is not None:
        rgb_array0_255[cell_id] = np.asarray(cell_color)

    cmap_name = "glasbey_on_dark"
    rgb_array0_1 = [tuple(np.asarray(x) / 255) for x in glasbey_on_dark]
    cmap = LinearSegmentedColormap.from_list(cmap_name, rgb_array0_1, N=N + 1)

    return np.asarray(rgb_array0_255), cmap, rgb_array0_255


def convert_2d_label_img_to_rgb(label_img, N=None, cell_id=None, cell_color=None):
    rgb_array0_255, _, _ = return_glasbey_on_dark(N=N, cell_id=cell_id, cell_color=cell_color)

    # try this instead
    return np.take(rgb_array0_255, label_img, axis=0).astype("uint8")


def get_contour_and_color_lists(seg_slices, label_img_max, cell_label_num, cell_color):
    list_of_contour_and_color_lists = []
    for si in range(len(seg_slices)):
        label_img = seg_slices[si].copy()
        # identify the contours from the label image and save them as matplotlib Polygons
        # important to recompute a scaling factor because zx views have been resized in Z, so original scaling factor does not apply in Z.
        contour_factor = [1 / RESCALE_FACTOR_100x_to_20X, 1 / RESCALE_FACTOR_100x_to_20X]
        # # extract the contours
        contour_and_color_list = get_matplotlib_contours_on_image(
            label_img,
            label_img_max,
            contour_factor,  # scale factor for y (same as x)
            cell_id=int(cell_label_num),
            cell_color=cell_color,
        )
        list_of_contour_and_color_lists.append(contour_and_color_list)
    return list_of_contour_and_color_lists


def get_list_of_rectangles(crop_roi, mag):
    """
    convert an roi for ZYX into a list of matplotlib rectangles for the
    YX view, ZX view, and ZY view

    Parameters
    ----------
    crop_roi : list
        The cropping ROI in ZYX format
    mag : str
        The magnification of the image

    Returns
    -------
    list_of_rectangles : list
        A list of matplotlib rectangles for the YX, ZX, and ZY views
    """
    list_of_rectangles = []
    for si in range(3):
        roi_for_view = [
            crop_roi[xi] for xi, x in enumerate(crop_roi) if xi not in [si * 2, (si * 2) + 1]
        ]
        if mag == "20x":
            scaling_array = [RESCALE_FACTOR_100x_to_20X, RESCALE_FACTOR_100x_to_20X]
        elif mag == "100x":
            scaling_array = [1, 1]
        roi_factor = np.asarray(
            [scaling_array[0], scaling_array[0], scaling_array[1], scaling_array[1]]
        )
        # convert track_id cropping ROI for plotting by dividing by the factor so it matches the image size
        roi = np.round(roi_for_view * roi_factor, 0).astype("uint16")
        # convert to matplotlib compatible X and Y coordinates
        y1, y2, x1, x2 = roi
        # define rectangle around cell of interest
        rect_list = [
            Rectangle((x1, y1), width=x2 - x1, height=y2 - y1, edgecolor=[1, 1, 0], alpha=0.5)
        ]
        list_of_rectangles.append(rect_list)
    return list_of_rectangles


def get_matplotlib_contours_on_image(
    label_img, N=None, scale_factor=np.asarray([1, 1]), cell_id=None, cell_color=None
):
    """
    take an 2d label image as an input and draw contours around the non-zero pixels in the same colors as the labels

    Parameters
    ----------
    label_img : np.array
        The 2D label image
    N : int
        The number of colors to return in the colormap
    scale_factor : np.array
        The scaling factor for the contours
    cell_id : int
        The cell id to change the color of
    cell_color : np.array
        The RGB color to change the cell to

    Returns
    -------
    patch_and_color_list : list
        A list of matplotlib Polygons and their colors
    """
    # use transposed image to get correct contour coordinates for matplotlib
    label_img_t = label_img.T

    rgb_array0_255, _, _ = return_glasbey_on_dark(N=N, cell_id=cell_id, cell_color=cell_color)

    patch_and_color_list = []
    unique_labels_in_image = [x for x in np.unique(label_img) if x > 0]
    for label in unique_labels_in_image:  # start from 1 to ignore background
        mask = label_img_t == label  # create a binary mask for the current label
        contours = measure.find_contours(mask, 0.5)  # find contours

        # contours = [np.swapaxes(x.T,1,0).T for x in contours]
        for contour in contours:
            color = rgb_array0_255[label] / 255
            poly = Polygon(
                contour
                / scale_factor[::-1],  # contours are in XZ order and scale_factor is in ZX order
                edgecolor=color,
            )
            patch_and_color_list.append((poly, color))

    return patch_and_color_list


def draw_scale_bar_matplotlib(
    ax, name, scalebarum=20, add_text=True, fontsize=8, is_seg=False, loc="left"
):
    """
    draw a scale bar on the axis for an image

    Parameters
    ----------
    ax : matplotlib axis
        The axis to draw the scale bar on
    name : str
        NOT USED
    scalebarum : int
        The length of the scale bar in microns
    add_text : bool
        Whether to add print the scale bar length and units as text above scale bar
    fontsize : int
        The fontsize of the text
    is_seg : bool
        Whether the image is a segmentation image (100x) or not (20x)
    loc : str
        The location of the scale bar on the image (either "left" or "right")

    Returns
    -------
    None
    """
    x_min_inches = 0.12  # inches from the left (or right) side of the axes
    y_min_inches = x_min_inches  # inches from the bottom of the axes

    # the width is a set number of pixels based on the pixel to um conversion
    pixel_size = PIXEL_SIZE_YX_100x if is_seg else PIXEL_SIZE_YX_20x
    width_pixels = scalebarum / pixel_size

    axis_width_inches = (
        ax.get_position().width * ax.get_figure().get_figwidth()
    )  # convert axes width to inches
    xlim_width = np.abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    if loc == "left":
        # define the X coordinates as fixed distance from left side of image
        x = np.min(ax.get_xlim()) + xlim_width * (x_min_inches / axis_width_inches)
        x2 = x + width_pixels
        xstr = x
    else:
        # define the X coordinates as fixed distance from right side of image
        x = np.max(ax.get_xlim()) - xlim_width * (x_min_inches / axis_width_inches)
        x2 = x - width_pixels
        xstr = x

    # define the Y coordinates as 5% of the axes height
    ylim_width = np.abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    axis_height_inches = (
        ax.get_position().height * ax.get_figure().get_figheight()
    )  # convert axes height to inches
    y = np.max(ax.get_ylim()) - ylim_width * (
        y_min_inches / axis_height_inches
    )  # NOTE: the max of ylim is the bottom of an image in matplotlib

    # the height is 5% of the axes height

    ax.plot([x, x2], [y, y], color="w", linewidth=2)
    # y2 needs to define a location above the linewidth
    if add_text:
        unitsstr = r"μm"
        ax.text(
            xstr,
            y,
            f"{scalebarum} {unitsstr}",
            color="w",
            fontsize=fontsize,
            ha=loc,
            va="bottom",
        )


def get_list_of_tail_lines(
    df,
    timepoint_frame,
    tail_length,
    seg_slices,
    mag,
    cropping_roi_seg,
    tail_linewidth=2,
    N=255,
    cell_id=None,
    cell_color=None,
):
    """
    for a tracking dataframe, and a given 2d image with labels,
    create matplotlib line objects for all of the track coordinates from timepoint_frame-tail_length to timepoint_frame

    Parameters
    ----------
    df : pd.DataFrame
        The tracking dataframe
    timepoint_frame : int
        The timepoint frame to plot the tails for
    tail_length : int
        The number of frames to plot the tails for
    seg_slices : list
        The slices of the label_imags (segmentation image) in [yx,zx,zy] format
    mag : str
        The magnification of the image, either "20x" or "100x"
    cropping_roi_seg : list
        The cropping ROI in ZYX format for the 100x segmentation image
    tail_linewidth : int
        The linewidth of the tails
    N : int
        The number of colors to return in the colormap
    cell_id : int
        The cell id to change the color of
    cell_color : np.array
        The RGB color to change the cell to

    Returns
    -------
    list_of_tails : list
        A list of matplotlib line objects for the tails
        list is tails for [yx,zx,zy] views

    """
    tail_frames = np.arange(timepoint_frame - tail_length, timepoint_frame + 1, 1).astype("uint16")
    # subset dataframe to specific frames
    dff = df[df["index_sequence"].isin(tail_frames)]
    df_at_t = dff[dff["index_sequence"] == timepoint_frame]
    list_of_tails = []
    for si in range(len(seg_slices)):
        label_img = seg_slices[si].copy()

        if si == 0:  # YX
            adjust_x = cropping_roi_seg[4]
            adjust_y = cropping_roi_seg[2]
            xcol = "centroid_x"
            ycol = "centroid_y_inv"
        elif si == 1:  # ZX
            adjust_x = cropping_roi_seg[4]
            adjust_y = cropping_roi_seg[0]
            xcol = "centroid_x"
            ycol = "centroid_z_inv_adjust"
        elif si == 2:  # not sure if this will work for ZY slices
            adjust_x = cropping_roi_seg[2]
            adjust_y = cropping_roi_seg[0]
            xcol = "centroid_y_inv"
            ycol = "centroid_z_inv_adjust"

        if mag == "20x":
            contour_factor = [RESCALE_FACTOR_100x_to_20X, RESCALE_FACTOR_100x_to_20X]
        elif mag == "100x":
            contour_factor = [1, 1]

        # determine track_ids to plot based on tracks present in current view of the label_img (max proj, xy or xz)
        label_nums_in_label_img = [x for x in np.unique(label_img) if x > 0]

        # determine what nuclei are present in the image at the current timepoint
        df_at_t_labels = df_at_t[df_at_t["label_img"].isin(label_nums_in_label_img)]
        track_id_list = df_at_t_labels.track_id.unique()

        # now filter the tracking dataframe to those nuclei
        dfft = dff[dff.track_id.isin(track_id_list)]
        dfft.set_index("index_sequence", inplace=True)

        # now iterate through all tracks on the current frame and draw their tails for all frames in frames
        # define the colormap for the tracks, N = highest label_value in full FOV label_img (NOT current view) so it is consistent across all views
        rgb0_255, _, _ = return_glasbey_on_dark(
            N=N,
            cell_id=cell_id,
            cell_color=cell_color,
        )
        line_dict_list = []
        for tail_track_id in track_id_list:
            dfsub = dfft[
                dfft.track_id == tail_track_id
            ]  # subset the dataframe to only include the current track_id

            x = (
                dfsub[xcol] - adjust_x
            )  # retrieve the x coordinates for the centroid and subtract the cropping ROI coordinates so that the coordinates are relative to the cropped image
            y = dfsub[ycol] - adjust_y

            # determine label_img value for track at t=timepoint_frame
            label_for_track = dfsub.label_img.values[-1].astype("uint16")

            x_adj = x * contour_factor[0]
            y_adj = y * contour_factor[1]
            line_dicts = []
            for ti in range(len(x) - 1):
                x1, x2 = x_adj.iloc[ti], x_adj.iloc[ti + 1]
                y1, y2 = y_adj.iloc[ti], y_adj.iloc[ti + 1]
                color0 = np.asarray(rgb0_255[label_for_track]) / (255)
                color = list(color0) + [1]
                tt = x_adj.index.values[ti]

                color = list(np.asarray(rgb0_255[label_for_track]) / (255))

                # make alpha decrease as the track goes back in time
                alpha = (tail_length - (timepoint_frame - tt)) / tail_length
                color.append(alpha)  # make color len=4 for RGBA

                # Create a Line2D object and add it to the list
                line = Line2D(
                    [x1, x2],
                    [y1, y2],
                    color=color,
                    linewidth=tail_linewidth,
                    marker="o",
                    markersize=0,
                )

                line_dict = {
                    "x": [x1, x2],
                    "y": [y1, y2],
                    "color": color,
                }
                line_dicts.append(line_dict)
            line_dict_list.append(line_dicts)
        list_of_tails.append(line_dict_list)
    return list_of_tails


def determine_linewidth_from_desired_microns(
    fig,
    ax,
    microns,
):
    """
    adjust a linewidth based on size of image line is plotted over

    Parameters:
    - pixel_width: Desired linewidth in pixels.
    - dpi: DPI of the figure.
    - fig_size_inches: Size of the figure in inches.

    Returns:
    - Linewidth in points.
    """
    # Calculate the total number of pixels in the figure
    fig_width_inches = fig.get_figwidth()
    ax_width_figure = ax.get_position().width
    ax_width_inches = ax_width_figure * fig_width_inches
    ax_width_pixels = np.ptp(ax.get_xlim())  # Range of x-axis in pixels

    # TODO: change this from being hardcoded
    pixel_size = PIXEL_SIZE_YX_20x
    ax_width_microns = ax_width_pixels * pixel_size
    ax_width_microns_scaled = ax_width_microns * ax_width_inches
    adjusted_linewidth = microns / ax_width_microns_scaled
    # print("ax_width_microns_scaled",ax_width_microns_scaled)
    # print("fig_dpi",fig.get_dpi())

    # linewidth is specified in points
    # 1 point is 1/72 of an inch
    # the desired size in microns is based on microns per inch of the image
    image_microns_per_inch = ax_width_microns / ax_width_inches

    linewidth_inches = microns / image_microns_per_inch
    linewidth_points = linewidth_inches * 72  # 72 points = 1 inch

    return linewidth_points


def add_view_arrows(ax, view="YX", fontsize=7, shrink_factor=1, start_coord=(8, 4)):
    """
    Add arrows to the image to indicate the view

    Parameters
    ----------
    ax : matplotlib axis
        The axis to add the arrows to
    view : str
        The view of the image (either "YX" or "ZX")
    fontsize : int
        The fontsize of the text
    shrink_factor : int
        The shrink factor for the arrows
    start_coord : tuple
        The starting coordinates for the arrows

    Returns
    -------
    None
    """

    # start_coord = np.asarray((8,4)) *  # x, y
    xy1 = np.asarray((10, 0)) * shrink_factor
    xy2 = np.asarray((0, 10)) * shrink_factor
    arrowstyle = "-|>,head_length=0.2,head_width=0.1"
    arrow_linewidth = 1
    # horizontal
    ax.annotate(
        "",
        xy=xy1 + start_coord,
        xycoords="axes points",
        xytext=start_coord,
        arrowprops=dict(
            arrowstyle=arrowstyle, fc="white", ec="white", shrinkA=0, lw=arrow_linewidth
        ),
    )
    ax.annotate(
        f"{view[1]}",
        xy=(0, 0),
        xycoords="axes points",
        xytext=xy1 + start_coord,
        color="white",
        va="center",
        ha="center",
        fontsize=fontsize,
    )

    # vertical
    ax.annotate(
        "",
        xy=xy2 + start_coord,
        xycoords="axes points",
        xytext=start_coord,
        arrowprops=dict(
            arrowstyle=arrowstyle, fc="white", ec="white", shrinkA=0, lw=arrow_linewidth
        ),
    )
    ax.annotate(
        f"{view[0]}",
        xy=(0, 0),
        xycoords="axes points",
        xytext=xy2 + start_coord,
        color="white",
        va="center",
        ha="center",
        fontsize=fontsize,
    )


def assemble_formation_middle_breakdown_dataframe(df):
    """
    assemble the dataframe for formation middle and breakdown timepoints for a specific track
    this dataframe specifies the coordinates of that track at the formation, middle, and breakdown timepoints
    it also records the time of those key points

    Parameters
    ----------
    df : pd.DataFrame
        The tracking dataframe

    Returns
    -------
    df_fmb : pd.DataFrame
        The dataframe for the formation, middle, and breakdown timepoints
    """
    csv_dir = Path(__file__).parent / "csvs"
    dff = pd.read_csv(csv_dir / "F1P4_track_id_83322_formation_centroids.csv")
    dfb = pd.read_csv(csv_dir / "F1P4_track_id_83322_breakdown_centroids.csv")
    dfman = pd.concat([dff, dfb])
    dfman.set_index("index_sequence", inplace=True)
    # %%
    # now use the tracking dataframe to retrieve the tracking centroid for the middle of the track
    # combine the information from the CSVs and the tracking dataframe into a small dataframe specific to the track of interest
    track_id = EXAMPLE_TRACKS["figure_dataset_formation_and_breakdown"]
    dfi = df[df["track_id"] == track_id].set_index("index_sequence")

    ws = 2  # window_size
    predicted_formation = dfi["predicted_formation"].values[0]
    formation_timepoints = np.arange(
        predicted_formation - ws, predicted_formation + ws + 1, 1
    ).astype(int)
    predicted_breakdown = dfi["predicted_breakdown"].values[0]
    breakdown_timepoints = np.arange(
        predicted_breakdown - ws, predicted_breakdown + ws + 1, 1
    ).astype(int)
    middle_of_track = np.uint16((predicted_formation + predicted_breakdown) / 2)
    middle_of_track_timepoints = np.arange(
        middle_of_track - ws, middle_of_track + ws + 1, 1
    ).astype(int)

    timepoints = np.concatenate(
        [formation_timepoints, middle_of_track_timepoints, breakdown_timepoints]
    )
    dflist = []
    for class_name, timepoints in zip(
        ["formation", "middle", "breakdown"],
        [formation_timepoints, middle_of_track_timepoints, breakdown_timepoints],
    ):
        for ti, timepoint in enumerate(timepoints):
            # use tracking centroid if segmentation present
            if timepoint in dfi.index:
                y = dfi.loc[timepoint, "centroid_y"]
                x = dfi.loc[timepoint, "centroid_x"]
            else:
                # otherwise use manual tracking centroid
                x = dfman.loc[timepoint, "x"]
                y = dfman.loc[timepoint, "y"]

            feats = {}
            feats["index_sequence"] = np.uint16(timepoint)
            feats["t"] = np.uint16(ti)
            feats["x"] = np.uint16(x)
            feats["y"] = np.uint16(y)
            feats["track_id"] = np.uint16(track_id)
            feats["class"] = class_name
            dflist.append(pd.DataFrame(data=feats.values(), index=feats.keys()).T)
    df_fmb = pd.concat(dflist)
    df_fmb.set_index("index_sequence", inplace=True)

    df_fmb = df_fmb.groupby("index_sequence").agg("first")
    df_fmb["ival"] = [x for x in range(df_fmb.shape[0])]
    df_fmb["x"] = df_fmb["x"].astype("uint16")
    df_fmb["y"] = df_fmb["y"].astype("uint16")

    # now add title annotations
    for column_name, column_label in (
        ["predicted_formation", "predicted formation"],
        ["predicted_breakdown", "predicted breakdown"],
        ["Ff", "formation"],
        ["Fb", "breakdown"],
    ):
        specified_t = dfi[column_name].values[0]
        df_fmb.loc[specified_t, "title_str"] = column_label
    df_fmb.loc[middle_of_track, "title_str"] = "middle"
    return df_fmb


def load_images_for_formation_middle_breakdown(df_fmb, df, colony):
    """
    load FOV images for all of the timepoints, and crop them around the centroids of the tracked cell
    return the segmentation crops and raw fluorescence crops

    Parameters
    ----------
    df_fmb : pd.DataFrame
        The dataframe for the formation, middle, and breakdown timepoints
    df : pd.DataFrame
        The tracking dataframe
    colony : str
        The colony to load the images from

    Returns
    -------
    seg_img_list : list
        The list of segmentation images [t0,t1,...,tn]
    raw_img_list : list
        The list of raw fluorescence images [t0,t1,...,tn]
    """
    # initialize the image reader for the CZI file containing movies from the three colonies
    raw_reader = get_dataset_original_file_reader(colony)

    # create the dataframe subset for retrieving the segmentation image path
    seg_img_list = []
    raw_img_list = []

    # define the dimensions of the crop
    crop_sizes_20x = [160, 160]
    crop_sizes_100x = [400, 400]

    # get the timepoints to get images from
    t_list = df_fmb.index.values
    for timepoint in tqdm(t_list):

        raw3dfov = export_helper.load_raw_fov_image(colony, timepoint, raw_reader, channel="egfp")
        dft = df[df.index_sequence == timepoint]
        seg3dfov = export_helper.load_seg_fov_image(dft)

        # determine the centroid of the cell
        xc, yc = df_fmb.loc[timepoint, "x"], df_fmb.loc[timepoint, "y"]

        seg_crop = image_helper.get_single_crop(seg3dfov, yc, xc, crop_sizes_100x, 1, "uint16")
        raw_crop = image_helper.get_single_crop(
            raw3dfov, yc, xc, crop_sizes_20x, RESCALE_FACTOR_100x_to_20X, "uint16"
        )

        # append the crops to lists
        seg_img_list.append(seg_crop)
        raw_img_list.append(raw_crop)

        # delete the large images to avoid memory overload (not as critical now that we don't load full 32 bit 3D 100x TF images)
        del seg3dfov
        del raw3dfov
    return seg_img_list, raw_img_list


def process_images_and_add_to_dataframe(df_fmb, df, seg_img_list, raw_img_list):
    """
    now convert the images in seg_img_list and raw_img_list into slice views with proper intensity rescaling
    also gather contours for segmentations

    store all into dataframe

    Parameters
    ----------
    df_fmb : pd.DataFrame
        The dataframe for the formation, middle, and breakdown timepoints
    df : pd.DataFrame
        The tracking dataframe
    seg_img_list : list
        The list of segmentation images [t0,t1,...,tn]
    raw_img_list : list
        The list of raw fluorescence images [t0,t1,...,tn]

    Returns
    -------
    df_fmb : pd.DataFrame
        The dataframe for the formation, middle, and breakdown timepoints with the images and contours added
    """
    # now add new columns to dffbi to store the 2D images and their segmentation contours
    df_fmb["raw_yx"] = None
    df_fmb["seg_yx"] = None
    df_fmb["raw_zx"] = None
    df_fmb["seg_zx"] = None
    df_fmb["raw_yx"] = None
    df_fmb["seg_yx"] = None
    df_fmb["seg_yx_contours"] = [["None"]] * df_fmb.shape[0]
    df_fmb["seg_zx_contours"] = [["None"]] * df_fmb.shape[0]
    df_fmb["seg_yx_contours"] = [["None"]] * df_fmb.shape[0]

    df_fmb["seg_yx_contours"] = df_fmb["seg_yx_contours"].astype(object)
    df_fmb["seg_zx_contours"] = df_fmb["seg_zx_contours"].astype(object)
    df_fmb["seg_yx_contours"] = df_fmb["seg_yx_contours"].astype(object)

    track_id = EXAMPLE_TRACKS["figure_dataset_formation_and_breakdown"]
    dfi = df[df["track_id"] == track_id].set_index("index_sequence")
    for ti in tqdm(range(df_fmb.shape[0])):
        # retrieve the raw and seg images
        raw_3d_image = raw_img_list[ti].copy()
        seg_3d_image = seg_img_list[ti].copy()

        # now make sure the label_img nucleus is the only one present in the seg_3d_image
        timepoint = df_fmb.index[ti]
        if timepoint in dfi.index:
            label_img_val = int(dfi.loc[timepoint, "label_img"])
        else:
            label_img_val = 0
        seg_3d_image[seg_3d_image != label_img_val] = 0
        seg_3d_image[seg_3d_image > 0] = 255

        raw_slices = process_channels_into_slices(raw_3d_image, "egfp", "uint16")
        seg_slices = process_channels_into_slices(seg_3d_image, "seg", "uint8")

        raw_slices_rs = rescale_intensities(raw_slices, "egfp", "uint8")

        # now add items to the dataframe
        for i, yx_zx_zy in enumerate(["yx", "zx"]):
            df_fmb.loc[timepoint, f"raw_{yx_zx_zy}"] = [raw_slices_rs[i]]
            df_fmb.loc[timepoint, f"seg_{yx_zx_zy}"] = [seg_slices[i]]

            # now retrieve the contours
            if (
                label_img_val > 0
            ):  # only draw the contour if the cell is present in the segmentation image
                # identify the contours from the label image and save them as matplotlib Polygons
                contour_factor = [1 / RESCALE_FACTOR_100x_to_20X, 1 / RESCALE_FACTOR_100x_to_20X]
                contour_and_color_list = get_matplotlib_contours_on_image(
                    seg_slices[i],
                    255,
                    contour_factor,  # scale factor for y (same as x)
                    cell_id=255,
                    cell_color=[255, 255, 0],
                )
            else:
                contour_and_color_list = [None]

            df_fmb.loc[timepoint, f"seg_{yx_zx_zy}_contours"] = [contour_and_color_list]

    return df_fmb
