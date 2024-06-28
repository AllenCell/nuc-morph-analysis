from pathlib import Path
from matplotlib.collections import PatchCollection  # type: ignore
from matplotlib.patches import Rectangle
import numpy as np
import skimage.exposure as skex
from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer, two_d_writer
import matplotlib.gridspec as gridspec  # type: ignore
import matplotlib.pyplot as plt

from nuc_morph_analysis.lib.preprocessing.load_data import get_channel_index, get_seg_fov_for_row
from nuc_morph_analysis.lib.preprocessing.single_track_contact.vis_code import image_helper
from nuc_morph_analysis.lib.preprocessing.single_track_contact.vis_code import single_track_contact

from nuc_morph_analysis.lib.preprocessing.load_data import (
    get_channel_index,
    get_dataset_original_file_reader,
)

from nuc_morph_analysis.lib.preprocessing.system_info import (
    PIXEL_SIZE_YX_20x,
    PIXEL_SIZE_Z_20x,
    PIXEL_SIZE_YX_100x,
    PIXEL_SIZE_Z_100x,
    RESCALE_FACTOR_100x_to_20X,
)


INTENSITIES = (100, 100 + 255)
DTYPE = "uint8"
STATIC_CROP_SIZES_YX = (400, 400)
STATIC_CROP_SIZES_20X_YX = (160, 160)  # = 400/2.5


AX_COLUMN_PAIRS = [
    ("vol_ax", ["volume"]),
    ("sa_ax", ["mesh_sa"]),
    ("volume_sa_ratio_ax", ["SA_vol_ratio"]),
    ("height_ax", ["height"]),
    ("aspect_ax", ["xz_aspect", "xy_aspect", "zy_aspect"]),
    ("depth_ax", ["colony_depth"]),
    ("neigh_ax", ["dxdt_12_volume", "dxdt_24_volume", "dxdt_48_volume"]),
    ("density_ax", ["density"]),
]


def get_single_crop_save_dir(seg_or_raw, dataset, create=False):
    """
    Get the save directory for the exported images

    Parameters
    ----------
    seg_or_raw : str
        Whether to export the raw or seg channel
    dataset : str
        Name of the dataset
    create : bool, optional
        Whether to create the directory if it does not exist, by default False

    Returns
    -------
    Path
        Path to the save directory
    """

    # NOTE WELL: saving thousands of files within the nuc_morph_analysis repo is not recommended
    # this will slow down vscode and git, and is not recommended for large datasets
    # instead, save to a local storage directory just outside the repo
    if seg_or_raw == "seg":
        save_dir = Path(__file__).parents[6] / "local_storage" / "single_nuclei_100XSEG" / dataset
    else:
        save_dir = Path(__file__).parents[6] / "local_storage" / "single_nuclei_20XEGFP" / dataset
    if (create == True) and (not save_dir.exists()):
        save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def get_fov_save_dir(dataset, create=False):
    """
    Get the save directory for the exported images

    Parameters
    ----------
    dataset : str
        Name of the dataset
    create : bool, optional
        Whether to create the directory if it does not exist, by default False

    Returns
    -------
    Path
        Path to the save directory
    """

    # NOTE WELL: saving thousands of files within the nuc_morph_analysis repo is not recommended
    # this will slow down vscode and git, and is not recommended for large datasets
    # instead, save to a local storage directory just outside the repo
    save_dir = Path(__file__).parents[6] / "local_storage" / "fov_zMIPs_20XEGFP" / dataset
    if (create == True) and (not save_dir.exists()):
        save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def get_single_track_contact_save_dir(dataset, create=False, single_frame=False):
    """
    Get the save directory for the exported images

    Parameters
    ----------
    dataset : str
        Name of the dataset
    create : bool, optional
        Whether to create the directory if it does not exist, by default False

    Returns
    -------
    Path
        Path to the save directory
    """

    # NOTE WELL: saving thousands of files within the nuc_morph_analysis repo is not recommended
    # this will slow down vscode and git, and is not recommended for large datasets
    # instead, save to a local storage directory just outside the repo
    dirname = (
        "single_track_contact_sheets"
        if single_frame == False
        else "single_track_contact_sheets_1frame"
    )
    save_dir = Path(__file__).parents[6] / "local_storage" / dirname / dataset
    if (create == True) and (not save_dir.exists()):
        save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def load_raw_fov_image(dataset, t, reader=None, channel="egfp"):
    """
    Load 3d image for a single timepoint in the 20x raw data

    Parameters
    ----------
    dataset : str
        Name of the dataset
    t : int
        Timepoint to retrieve
    reader : AICSImageIO reader object
        Reader object for the dataset
    channel : str
        Channel to retrieve

    Returns
    -------
    numpy array
        3d image for the timepoint
    """
    if reader is None:
        reader = get_dataset_original_file_reader(dataset)
    channel_index = get_channel_index(dataset, channel)
    img_out = reader.get_image_dask_data("ZYX", T=t, C=channel_index)
    return img_out.compute()


def load_seg_fov_image(dft):
    """
    Load 3d image for a single timepoint in the 100x seg channel
    rescale to 8 bit

    Parameters
    ----------
    dft : pandas dataframe
        Dataframe with the tracking info, already filtered to 'index_sequence' == t, and 'dataset' == dataset
    Returns
    -------
    numpy array
        3d image for the timepoint, 16 bit label image
    """
    img_out = get_seg_fov_for_row(dft.iloc[0])
    return img_out.compute()


def define_single_nucleus_save_path(dataset, t, track_id, cell_id, save_dir):
    save_path = (
        save_dir / f"{dataset}_t{str(t).zfill(3)}_track{str(track_id)}_cell_{cell_id}.ome.tiff"
    )
    return save_path


def define_save_paths(dataset, t, save_dir):
    """
    Define the save paths for the MIPs

    Parameters
    ----------
    dataset : str
        Name of the dataset
    t : int
        Timepoint to export
    save_dir : Path
        Path to save the output image

    Returns
    -------
    Path
        Path to save the full size MIP, and the downsampled MIP
    """
    save_path = save_dir / f"{dataset}_t{str(t).zfill(3)}_zmax.tiff"
    save_path2 = save_dir / f"{dataset}_t{str(t).zfill(3)}_zmax_downsampled2.tiff"
    return save_path, save_path2


def define_track_save_paths(dataset, tid, save_dir):
    """
    Define the save paths for the MIPs

    Parameters
    ----------
    dataset : str
        Name of the dataset
    tid : int
        Track ID
    save_dir : Path
        Path to save the output image

    Returns
    -------
    Path
        Path to save the single_track_contact_tiff_stackl
    """
    save_path = save_dir / f"{dataset}_{tid}_contact.tiff"
    return save_path


def create_the_figure():
    # create a figure
    fig = plt.figure(figsize=(16, 9))

    # create a gridspec object
    gs = gridspec.GridSpec(19, 33, figure=fig)
    ax_dict = {}
    # create the large subplot
    ax_dict["fov_ax"] = fig.add_subplot(gs[0:9, 1:14])

    # create the smaller subplots
    ax_dict["raw_ax"] = fig.add_subplot(gs[0:9, 15:24])
    ax_dict["seg_ax"] = fig.add_subplot(gs[0:9, 24:34])
    # ax_dict["raw_seg_ax"] = fig.add_subplot(gs[0:9, 15:34])

    ax_dict["vol_ax"] = fig.add_subplot(gs[10:14, 1:7])
    ax_dict["sa_ax"] = fig.add_subplot(gs[10:14, 9:15])
    ax_dict["volume_sa_ratio_ax"] = fig.add_subplot(gs[10:14, 17:23])
    ax_dict["height_ax"] = fig.add_subplot(gs[10:14, 25:31])

    ax_dict["aspect_ax"] = fig.add_subplot(gs[15:19, 1:7])
    ax_dict["neigh_ax"] = fig.add_subplot(gs[15:19, 9:15])
    ax_dict["density_ax"] = fig.add_subplot(gs[15:19, 17:23])
    ax_dict["depth_ax"] = fig.add_subplot(gs[15:19, 25:31])

    return fig, ax_dict


def load_exported_fov_imgs(dataset, time_list, small_fov=True):
    if small_fov:
        downsample = 1
    fov_dir = get_fov_save_dir(dataset, create=False)
    fov_files = [
        fov_dir / define_save_paths(dataset, t, fov_dir)[downsample].name for t in time_list
    ]
    fov_images = np.stack([AICSImage(f).get_image_data("YX").astype(DTYPE) for f in fov_files])

    # adjust the contrast for each of the images
    fov_images = skex.rescale_intensity(fov_images, in_range=(0, 40), out_range=(0, 255)).astype(
        DTYPE
    )

    return fov_images


def load_exported_crop_image(dft, t, dataset, img_dir, crop_sizes):

    if t in dft["index_sequence"].values:
        row = dft[dft["index_sequence"] == t].iloc[0]
        img_file = define_single_nucleus_save_path(
            dataset, row.index_sequence, row.track_id, row.CellId, img_dir
        )
        # load the image and return it
        return AICSImage(img_file).get_image_data("ZYX").astype(DTYPE)
    else:
        # we are uncertain how many Z were present, so just return a stack of zeros
        return np.zeros([1] + list(crop_sizes), dtype=DTYPE)


def process_while_loading_exported_crop_images(dataset, dft, time_list, seg_or_raw, num_workers=1):
    """
    Load and process the exported crop images for a single track

    Parameters
    ----------
    dataset : str
        Name of the dataset
    dft : pandas dataframe
        Dataframe with the tracking info for the current track
    time_list : list
        List of timepoints to export
    seg_or_raw : str
        Whether to export the raw or seg channel
    num_workers : int, optional
        Number of workers to use for parallel processing, by default 1

    Returns
    -------
    numpy array
        3d image (TYX) of the exported crop images
    """
    crop_sizes = STATIC_CROP_SIZES_YX if seg_or_raw == "seg" else STATIC_CROP_SIZES_20X_YX

    img_dir = get_single_crop_save_dir(seg_or_raw, dataset, create=False)

    stack_list = []
    for t in time_list:
        img = load_exported_crop_image(dft, t, dataset, img_dir, crop_sizes)
        if seg_or_raw == "seg":
            img[img == 1] = 50  # remove all neighbor nuclei
            # make all segmentation images 8 bit with 255 max
            img[img == 2] = 255
        stacked_slices = single_track_contact.process_image_into_slices_and_stack_seg_or_raw(
            img, seg_or_raw
        )
        stack_list.append(stacked_slices)

    stack = image_helper.stack_uneven_middle(stack_list, fill_value=0, dtype=DTYPE)
    # now apply contrast adjustment
    if seg_or_raw == "seg":
        stack = skex.rescale_intensity(stack, in_range=(0, 255), out_range=(0, 255)).astype(DTYPE)
    else:
        stack = skex.rescale_intensity(stack, in_range=(0, 30), out_range=(0, 255)).astype(DTYPE)

    return stack


def process_track(
    dft,
    fov_images,
    dffov,
    dataset,
    save_dir,
    num_workers,
    frame_buffer=5,
    dpi=80,
    single_frame=False,
    small_fov=True,
):
    """
    Process a single track to create a contact sheet movie

    Parameters
    ----------
    dft : pandas dataframe
        Dataframe with the tracking info for the current track
    fov_images : numpy array
        3d image (TYX) of the FOV for all timepoints
    dataset : str
        Name of the dataset
    save_dir : Path
        Path to save the output image directory
    num_workers : int
        Number of workers to use for parallel processing
    overwrite : bool
        Overwrite existing files
    frame_buffer : int, optional
        Number of frames to pad the movie on either side, by default 5
    dpi : int, optional
        DPI for the output image, by default 150
    """
    movie_path = define_track_save_paths(dataset, dft["track_id"].values[0], save_dir)

    track_id = dft["track_id"].values[0]
    min_time = dft.index_sequence.min()
    max_time = dft.index_sequence.max()

    # now for each track find its first time point and its last time point
    # number of frames to pad on beginning and end of tracking

    # frame at which to start the movie
    start_frame = np.max([min_time - frame_buffer, 0])
    # frame at which to end the movie
    end_frame = np.min([max_time + frame_buffer, dffov.index_sequence.max() - 1])

    time_list = np.arange(start_frame, end_frame + 1, 1)

    # determine whole track features
    frame_formation = dft["Ff"].dropna().values[0]
    frame_transition = dft["frame_transition"].dropna().values[0]
    frame_breakdown = dft["Fb"].dropna().values[0]

    fig, ax_dict = create_the_figure()
    fig.set_dpi(dpi)

    ax_column_pairs = AX_COLUMN_PAIRS

    fig_list = []
    if single_frame:
        time_list = [time_list[int(len(time_list) // 2)]]

    # now load the crop images
    workers = np.min([num_workers, len(time_list)])

    # time how long it takes to load the images
    stack_yx_zx_yz_seg = process_while_loading_exported_crop_images(
        dataset, dft, time_list, "seg", workers
    )
    stack_yx_zx_yz_raw = process_while_loading_exported_crop_images(
        dataset, dft, time_list, "raw", workers
    )

    # now make sure you have the plot axes defined by df_feats
    columns = []
    for ax in ax_column_pairs:
        columns.extend(ax[1])

    df_feats = single_track_contact.create_df_feats(columns, dataset)
    dffs = dft.set_index("index_sequence")

    # now create the figures per timepoint
    for ti in range(len(time_list)):
        t = time_list[ti]

        # now create the figure with the following subplots
        if ti == 0:
            object_list = []
            update = False
        else:
            update = True

        if update == True:
            single_track_contact.remove_previously_drawn_objects(object_list)

        _, object_list = single_track_contact.make_plot(
            ax_dict,
            ax_column_pairs,
            t,
            dffs,
            df_feats,
            frame_formation,
            frame_transition,
            frame_breakdown,
            frame_buffer,
            update=update,
        )
        # fov_images[t] is used instead of [ti] because fov_images has an image for each timpepoint
        out0 = ax_dict["fov_ax"].imshow(fov_images[t], cmap="gray")
        out1 = ax_dict["raw_ax"].imshow(
            stack_yx_zx_yz_raw[ti], cmap="gray", interpolation="nearest"
        )
        out2 = ax_dict["seg_ax"].imshow(
            stack_yx_zx_yz_seg[ti], cmap="gray", interpolation="nearest"
        )

        object_list.extend([out0, out1, out2])
        ax_dict["fov_ax"].set_title(
            f"{dataset} {track_id}\n frame={str(t).zfill(3)} \n time={str(np.round(t*5/60,2)).zfill(3)}"
        )
        ax_dict["raw_ax"].set_title(
            f"Raw {str(single_track_contact.get_slicing_method_seg_or_raw('raw'))}"
        )
        ax_dict["seg_ax"].set_title(
            f"Raw {str(single_track_contact.get_slicing_method_seg_or_raw('seg'))}"
        )
        if t in dffs.index.values:
            if (
                dffs.loc[t, "is_outlier"] == True
            ):  # sometimes is_outlier might not be present if it is an edge cell
                to = ax_dict["seg_ax"].text(
                    0.02,
                    0.98,
                    '"is_outlier==True"',
                    color="red",
                    fontsize=12,
                    transform=ax_dict["seg_ax"].transAxes,
                    ha="left",
                    va="top",
                )
                object_list.append(to)
            if (
                dffs.loc[t, "fov_edge"] == True
            ):  # sometimes is_outlier might not be present if it is an edge cell
                to = ax_dict["seg_ax"].text(
                    0.02,
                    0.02,
                    '"fov_edge==True"',
                    color="red",
                    fontsize=12,
                    transform=ax_dict["seg_ax"].transAxes,
                    ha="left",
                    va="bottom",
                )
                object_list.append(to)
        for ax_name in ["fov_ax", "raw_ax", "seg_ax"]:
            ax_dict[ax_name].axis("off")

        # add an roi rectangle to the fov_ax
        # get the roi
        if t in dffs.index.values:
            # only keep y and x portions of ROI
            roi_too_large = eval(dffs.loc[t, "roi"])[2::]
            # resize the ROI based on the 0.108 to 0.271 scaling
            roi_scale = RESCALE_FACTOR_100x_to_20X / 2 if small_fov else RESCALE_FACTOR_100x_to_20X
            roi = np.uint16(np.asarray(roi_too_large) * roi_scale)

            # now define the polygon, converting to matplotlib compatible X and Y coordinates

            y1, y2, x1, x2 = roi[0], roi[1], roi[2], roi[3]
            rect_list = [
                Rectangle((x1, y1), width=x2 - x1, height=y2 - y1, edgecolor=[1, 1, 0], alpha=0.5)
            ]

            # add the roi to the fov_ax
            pc_roi_rect = PatchCollection(
                rect_list,
                edgecolor=[1, 1, 0],
                alpha=0.5,
                # linewidth=ROI_linewidth,
                linewidths=2,
                facecolor="None",
                zorder=1000,
            )

            out3 = ax_dict["fov_ax"].add_collection(pc_roi_rect)
            object_list.append(out3)

        # fig.tight_layout()
        fig.canvas.draw()

        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        fig_list.append(image_from_plot)

    if single_frame:
        fig_stack = fig_list[0]
        two_d_writer.TwoDWriter.save(
            fig_stack,
            str(movie_path).replace(".tiff", ".png"),
        )
    else:
        fig_stack = np.stack(fig_list).astype("uint8")
        del fig_list
        # also save a tif for viewing in ImageJ
        ome_tiff_writer.OmeTiffWriter.save(
            fig_stack,
            str(movie_path).replace(".tiff", ".tif"),
            dim_order="TYXS",
        )
