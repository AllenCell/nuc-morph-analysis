from aicsimageio import AICSImage
from aicsimageio.writers import two_d_writer
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from skimage import exposure
import cv2
from matplotlib.collections import PatchCollection
from nuc_morph_analysis.lib.preprocessing import all_datasets
from nuc_morph_analysis.lib.preprocessing.system_info import (
    PIXEL_SIZE_Z_100x,
    PIXEL_SIZE_YX_100x,
    PIXEL_SIZE_Z_100x,
)
from nuc_morph_analysis.analyses.segmentation_model_validation.align_helper import (
    apply_shift,
    determine_shift,
)
from nuc_morph_analysis.lib.preprocessing.single_track_contact.vis_code.image_helper import (
    process_image_into_slices,
    stack_the_slices_zyx,
    get_single_crop,
    resize_low_res_image,
)

from nuc_morph_analysis.analyses.dataset_images_for_figures.figure_helper import (
    convert_2d_label_img_to_rgb,
)

CONTOUR_COLOR_DICT = {"pr_seg": (255, 0, 255), "gt_seg": (0, 255, 0), "we_seg": (0, 255, 0)}
INTENSITY_SCALING_PARAMS = {
    "lr": (100, 140),
    "hr": (400, 1200),
    "gt_seg": (0, 1),
    "pr_seg": (0, 1),
    "we_seg": (0, 1),
}

SEG_KEYS_TO_OVERLAY_LIST = [
    ["none"],
    ["pr_seg"],
    ["gt_seg"],
    ["gt_seg", "pr_seg"],
    ["we_seg"],
    ["we_seg", "pr_seg"],
]


def fine_tuning_path(subfolder: str, image_name: str) -> str:
    base_dir = all_datasets.segmentation_model_validation_URLs["base_dir_for_images"]
    return f"{base_dir}/{subfolder}/{image_name}"


def add_paths_to_manifest(df):
    """
    add s3 paths/urls to the manifest
    """
    df["generate_objects/lr"] = df["TestImageName"].apply(
        lambda x: fine_tuning_path("low_res_20x", x)
    )
    df["generate_objects/hr"] = df["TestImageName"].apply(
        lambda x: fine_tuning_path("high_res_100x", x)
    )
    df["run_watershed/real_lamin"] = df["TestImageName"].apply(
        lambda x: fine_tuning_path("watershed_segmentation_100x", x)
    )
    df["run_cytodl/nucseg"] = df["TestImageName"].apply(
        lambda x: fine_tuning_path("model_segmentation_100x", x)
    )
    df["run_watershed/with_edge"] = df["TestImageName"].apply(
        lambda x: fine_tuning_path("model_segmentation_100x", x)
    )

    return df


def define_FOVid(df):
    """
    define an FOVId column
    """
    df["id"] = df["TestImageName"]
    return df


def get_dictonary_of_images_from_manifest(row):
    """
    Get images from a row of a manifest file and place into dictionary
    #lr = low_res, 20x
    #hr = high_res, 100x
    #gt_seg = ground truth segmentation
    #pr_seg = predicted segmentation
    #we_seg = ground truth segmentation without edge nuclei removed from image

    Parameters
    ----------
    row : pd.Series
        A row of a manifest file

    Returns
    -------
    img_dict : dict
        A dictionary containing the images
    """

    lr_img_path = row["generate_objects/lr"]
    hr_img_path = row["generate_objects/hr"]
    gt_seg_path = row["run_watershed/real_lamin"]
    pr_seg_path = row["run_cytodl/nucseg"]
    gt_with_edge_seg_path = row["run_watershed/with_edge"]

    img_dict = {}
    for img_name, img_path in [
        ("lr", lr_img_path),
        ("hr", hr_img_path),
        ("gt_seg", gt_seg_path),
        ("pr_seg", pr_seg_path),
        ("we_seg", gt_with_edge_seg_path),
    ]:
        img = AICSImage(img_path).get_image_data("ZYX")
        img_dict[img_name] = img
    return img_dict


def pad_images_in_dictionary_to_same_size(img_dict, verbose=False):
    """
    Pad all images in a dictionary to the same size so that they can be overlayed

    Parameters
    ----------
    img_dict : dict
        A dictionary containing the images

    Returns
    -------
    img_dict : dict
        A dictionary containing the padded images
    verbose: bool
        If True, print out the padding information
    """
    # now you want to make sure all the images are the same size
    # first determine the maximum dimensions in Z Y and X for the images

    max_dims = [img_dict[img_name].shape for img_name in img_dict]
    max_dims = np.array(max_dims).max(axis=0)

    # now pad the images to the same size
    for img_name in img_dict:
        img = img_dict[img_name]
        pad_dims = [(0, max_dims[i] - img.shape[i]) for i in range(3)]
        if pad_dims != [(0, 0), (0, 0), (0, 0)]:
            if verbose:
                print(f"NOTE: Padding {img_name} by {pad_dims}")
        img_dict[img_name] = np.pad(img, pad_dims, mode="constant", constant_values=np.min(img))
    return img_dict


def resize_low_res_images_in_dict(img_dict):
    """
    Resize the low resolution image to match the size of the high resolution image

    Parameters
    ----------
    img_dict : dict
        A dictionary containing the images

    Returns
    -------
    img_dict : dict
        A dictionary containing the images
    """
    # resize the iamge
    img_dict["lr"] = resize_low_res_image(img_dict["lr"])
    # check the size
    if tuple(img_dict["lr"].shape) != img_dict["pr_seg"].shape:
        print(
            "The new shape does not match the high resolution image shape...BUT padding will fix this later"
        )
    return img_dict


def align_lr_images_to_hr_images(img_dict):
    """
    Align the low resolution images to the high resolution images
    low_res_images are lr and pr_seg
    high_res_images are hr, gt_seg, and we_seg
    alignment is simple translation in z,y,x
    The alignment is computed by comparing gt_seg as a proxy for 100x and pr_seg as a proxy for 20x.

    Parameters
    ----------
    img_dict : dict
        A dictionary containing the images

    Returns
    -------
    aligned_dict : dict
        A dictionary containing the images after alignment
    """
    # now align the lr image to the hr image
    shift_estimate = determine_shift(
        img_dict["gt_seg"].copy(), img_dict["pr_seg"].copy(), restrict_to_overlapping_objects=True
    )
    aligned_dict = apply_alignment_to_lr_and_pr_seg(img_dict, shift_estimate)
    return aligned_dict


def apply_alignment_to_lr_and_pr_seg(img_dict, shift_estimate):
    """
    Apply the alignment to the lr and pr_seg images

    Parameters
    ----------
    img_dict : dict
        A dictionary containing the images
    shift_estimate : tuple
        The shift estimate

    Returns
    -------
    aligned_dict : dict
        A dictionary containing the aligned images
    """
    aligned_dict = {}
    keys_to_align = ["lr", "pr_seg"]  # images to align
    for key, img in img_dict.items():
        order = 0 if "seg" in key else 1
        if key in keys_to_align:
            aligned_dict[key] = apply_shift(img, shift_estimate, order=order)
        else:
            aligned_dict[key] = img.copy()
    return aligned_dict


def apply_intensity_rescaling_to_raw_images_in_dict(aligned_dict):
    """
    Apply intensity rescaling to the images

    Parameters
    ----------
    aligned_dict : dict
        A dictionary containing the images

    Returns
    -------
    aligned_dict : dict
        A dictionary containing the rescaled images
    """

    for img_name in ["lr", "hr"]:
        in_range = INTENSITY_SCALING_PARAMS[img_name]
        aligned_dict[img_name] = exposure.rescale_intensity(
            aligned_dict[img_name], in_range=in_range, out_range="uint8"
        ).astype("uint8")
    return aligned_dict


def convert_values_from_lists_of_slices_into_single_slices(slice_dict):
    """
    convert a dictionary of lists of image slices into a dictionary of images with the slices named

    Parameters
    ----------
    slice_dict : dict
        A dictionary containing the lists of image slices

    Returns
    -------
    img_dict : dict
        A dictionary containing the named images
    """
    slice_order = ["yx", "zx", "zy"]
    img_dict = {}
    for slice_type in slice_dict.keys():
        for i, img_slice in enumerate(slice_dict[slice_type]):
            img_dict[f"{slice_type}_{slice_order[i]}"] = img_slice
    return img_dict


def determine_contours_for_image_dict(slice_dict):
    """
    Determine the contours for all segmentation images in a dictionary

    Parameters
    ----------
    slice_dict : dict
        A dictionary containing the segmentation images
    """
    contours_dict = {}
    for slice_type in slice_dict.keys():
        if "seg" in slice_type:
            img = slice_dict[slice_type]
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_dict[slice_type] = contours
    return contours_dict


def draw_contours_on_all_images_in_dict(img_dict, contours_dict, thickness=1):
    """
    Function to draw contours on images

    Parameters
    ----------
    img_dict : dict
        dictionary of images to draw contours on
    contours_dict : dict
        dictionary of contours to draw on images
    thickness : int
        thickness of the contour lines

    Returns
    -------
    contours_drawn_dict : dict
        dictionary of images with contours drawn on them
    """
    lr_hr_list = ["lr", "hr"]
    # this is the combination of contours to draw on the image

    # determine in individual slices present or not
    if np.sum(["_yx" in x for x in img_dict.keys()]) > 0:
        suffix_list = ["_yx", "_zx", "_zy"]
    else:
        suffix_list = [""]

    contours_drawn_dict = {}
    # iterate through all images and draw contours
    for suffix in suffix_list:
        for lr_hr in lr_hr_list:
            img = img_dict[f"{lr_hr}{suffix}"].copy()
            for seg_keys in SEG_KEYS_TO_OVERLAY_LIST:
                rgb = np.uint8(np.stack([img] * 3, axis=-1).copy())
                for seg_key in seg_keys:
                    if seg_key == "none":
                        img_contour = rgb
                    else:
                        img_contour = cv2.drawContours(
                            rgb,
                            contours_dict[f"{seg_key}{suffix}"],
                            -1,
                            CONTOUR_COLOR_DICT[seg_key],
                            thickness,
                        )
                contours_drawn_dict[f"{lr_hr}{suffix}_{'-'.join(seg_keys)}"] = img_contour
    return contours_drawn_dict


def determine_label_values_at_zyx_in_all_seg_images(zyx, img_dict):
    """
    determine the label values of nuclei in all segmentation images
    for a given ZYX location

    Parameters
    ----------
    zyx : list
        a list of z,y,x coordinates corresponding to cell centroid
    img_dict : dict
        a dictionary of images
        note this should be images BEFORE alignment (so zyx centroid is still valid)

    Returns
    -------
    label_dict : dict
        a dictionary containing the labels of the gt_seg, pr_seg, and we_seg images
    """
    label_dict = {}
    # determine the label of the gt_seg image at the zyx location
    label_gt_seg = img_dict["gt_seg"][zyx[0], zyx[1], zyx[2]]
    gt_seg_mask = img_dict["gt_seg"] == label_gt_seg
    label_dict["gt_seg"] = label_gt_seg

    for key in ["pr_seg", "we_seg"]:
        # label_pr_seg is the most common value within the mask
        pixels_in_mask = [x for x in img_dict[key][gt_seg_mask] if x != 0]
        if len(pixels_in_mask) == 0:
            print("no match found for label", label_gt_seg, " in ", key)
            label_dict[key] = None
        else:
            label_in_seg = np.bincount(pixels_in_mask).argmax()
            label_dict[key] = label_in_seg

    return label_dict


def keep_label_pixels_only_for_seg_images_in_dict(img_dict_input, label_dict):
    """
    Set all pixels not equal to the label number to zero in the images
    This way the segmentation for the nucleus of interest is the only one shown

    Parameters
    ----------
    img_dict_input : dict
        A dictionary containing the images
    label_dict : dict
        A dictionary containing the label numbers for each segmentation image

    Returns
    -------
    img_dict : dict
        A copy of the original input dictionary that now contains the images with only the nucleus of interest segmented
    """
    # copy so the original images are not modified
    img_dict = img_dict_input.copy()
    for key in label_dict.keys():
        img_dict[key] = np.uint8(img_dict[key] == label_dict[key]) * 255
    return img_dict


def crop_all_images_in_dictionary(aligned_dict, zyx, crop_pads_yx=[300, 300], dtype="uint8"):
    """
    Crop all images in the dictionary based on ZYX and crop pads

    Parameters
    ----------
    aligned_dict : dict
        A dictionary containing the aligned images
    zyx : list
        A list of z,y,x coordinates corresponding to cell centroid
    crop_pads_yx : list
        A list of values defining the crop sizes in YX
        Z is not cropped

    Returns
    -------
    crop_dict : dict
        A dictionary containing the cropped images
    """
    crop_dict = {}
    for key in aligned_dict.keys():
        img_out = aligned_dict[key]
        y0, x0 = zyx[1], zyx[2]
        crop_img = get_single_crop(img_out, y0, x0, crop_pads_yx, 1, "uint8")
        crop_dict[f"{key}"] = crop_img
    return crop_dict


def process_image_dict_into_slices(img_dict, dtype="uint8"):
    """
    Process the image dictionary into slices (UNSTACKED)

    Parameters
    ----------
    img_dict : dict
        A dictionary containing the images
        the segmented images should be semantic segmentations with foreground = 255 and background =0
    dtype : str
        The data type of the output image.

    Returns
    -------
    slice_dict : dict
        A dictionary containing the lists of img slices, list[0] is yx, list[1] is zx, list[2] is zy
    """

    # determine yx slice at which segmentation area is max
    Z = np.argmax(np.sum(np.float32(img_dict["gt_seg"]), axis=(1, 2)))
    max_sum_slice = (Z, "mid", "mid")
    slice_dict = {}
    for key in img_dict.keys():
        img = img_dict[key]
        order = 0 if "seg" in key else 1
        scale_factors = [
            (PIXEL_SIZE_YX_100x / PIXEL_SIZE_YX_100x, 1),  # yx; will = (1,1)
            (PIXEL_SIZE_Z_100x / PIXEL_SIZE_YX_100x, 1),  # zx; will ~= (0.29/0.108,1)
            (PIXEL_SIZE_Z_100x / PIXEL_SIZE_YX_100x, 1),  # zy; will ~= (0.29/0.108,1)
        ]
        slices = process_image_into_slices(
            img, max_sum_slice=max_sum_slice, order=order, scale_factors=scale_factors, dtype=dtype
        )
        slice_dict[key] = slices
    return slice_dict


def stack_slices_within_image_dict_of_slices(slice_dict, gap_color=255, width=1, isrgb=False):
    """
    Convert the lists of image slices within slice_dict into single images by stacking the image slices

    Parameters
    ----------
    slice_dict : dict
        A dictionary containing the slices, list[0] is yx, list[1] is zx, list[2] is zy
    gap_color : int
        The color of the gap between slices.
    width : int
        The width of the gap between slices.
    isrgb : bool
        Whether the images are RGB or not, default is False

    Returns
    -------
    stacked_slices_dict : dict
        A dictionary containing the stacked_slices, np.array
    """
    stacked_slices_dict = {}
    for key in slice_dict.keys():
        stacked_slices_dict[key] = stack_the_slices_zyx(
            slice_dict[key], width=width, gap_color=gap_color, isrgb=isrgb
        )

    return stacked_slices_dict


def save_out_images_in_dict(stack_dict, FOVid, cell_id, figdir):
    item_id = FOVid if cell_id == "None" else cell_id

    savedir = figdir / f"{FOVid}"
    if not savedir.exists():
        savedir.mkdir(exist_ok=True)

    for key in stack_dict.keys():
        img = stack_dict[key].copy()
        if "seg" in key:
            img[img > 0] = 255
        savename = savedir / f"{item_id}_{key}.png"
        two_d_writer.TwoDWriter.save(img, savename, dim_order="YX")


def save_out_images_in_dict_with_overlay(stack_dict, FOVid, cell_id, figdir):
    item_id = FOVid if cell_id == "None" else cell_id

    savedir = figdir / f"{FOVid}_with_overlay"
    if not savedir.exists():
        savedir.mkdir(exist_ok=True)

    for key in stack_dict.keys():
        savename = savedir / f"{item_id}_{key}.png"
        two_d_writer.TwoDWriter.save(stack_dict[key], savename, dim_order="YXS")


def save_out_images_in_dict_for_quick_scroll_review(stack_dict, FOVid, cell_id, figdir):
    item_id2 = "" if cell_id == "None" else f"-{cell_id}"

    savedir = figdir / "quick_scroll"
    if not savedir.exists():
        savedir.mkdir(exist_ok=True)

    key = [x for x in stack_dict.keys() if "hr_gt_seg-pr_seg" in x][0]
    savename = savedir / f"FOV={FOVid}cell={item_id2}_{key}.png"
    two_d_writer.TwoDWriter.save(stack_dict[key], savename, dim_order="YXS")
    print(f"saved images here: {savename}")


def draw_matplotlib_contours_on_image(ax, contour, edge_color, linewidths=1, linestyle="--"):
    # contours are from cv2 (M x 1 x 2)
    # Polygon wants (M x 2)
    edge_color = edge_color if np.max(edge_color) <= 1 else [x / 255 for x in edge_color]
    poly = Polygon(
        contour.take(0, 1),  # take index=0 from axis=1
        fill=None,
        edgecolor=edge_color,
        facecolor=None,
    )

    pc_contours = PatchCollection(
        [poly],
        edgecolor=edge_color,
        linewidths=linewidths,
        facecolor="None",
        zorder=1000,
        linestyle=linestyle,
    )
    ax.add_collection(pc_contours)


def save_out_images_in_dict_with_overlay_as_PDF(image_dict, contours_dict, FOVid, cell_id, figdir):
    item_id = FOVid if cell_id == "None" else cell_id

    savedir = figdir / f"{FOVid}_with_overlayPDFs"
    if not savedir.exists():
        savedir.mkdir(exist_ok=True)

    image_keys = ["lr", "hr"]
    for key in image_keys:
        img = image_dict[key].copy()
        image_aspect_ratio = img.shape[1] / img.shape[0]

        for seg_keys in SEG_KEYS_TO_OVERLAY_LIST:
            fig, ax = plt.subplots(1, 1, figsize=(4 * image_aspect_ratio, 4))
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.axis("off")
            for seg_key in seg_keys:
                if seg_key == "none":
                    continue

                for contour in contours_dict[f"{seg_key}"]:
                    draw_matplotlib_contours_on_image(
                        ax, contour, CONTOUR_COLOR_DICT[seg_key], linewidths=1, linestyle="--"
                    )
            savename = savedir / f"{item_id}_{key}_{'-'.join(seg_keys)}.pdf"
            fig.savefig(savename, bbox_inches="tight", dpi=300)
            plt.close(fig)


def save_out_seg_images_as_label_image(img_dict, figdir, FOVid, cell_id):
    item_id = FOVid if cell_id == "None" else cell_id

    savedir = figdir / f"{FOVid}_instance_seg_images"
    if not savedir.exists():
        savedir.mkdir(exist_ok=True)

    keylist = [x for x in img_dict.keys() if "seg" in x]
    for key in keylist:
        img_rgb = img_dict[key].copy()
        savename = savedir / f"{item_id}_{key}.png"
        two_d_writer.TwoDWriter.save(img_rgb, savename, dim_order="YXS")


def simple_convert_slice_dict_label_seg_images_to_rgb(slice_dict):
    keylist = [x for x in slice_dict.keys() if "seg" in x]
    new_dict = {}
    for key in keylist:
        new_dict[key] = [convert_2d_label_img_to_rgb(x, N=255) for x in slice_dict[key]]
    return new_dict


def process_and_save_images(aligned_dict_rs, figdir, FOVid, cell_id="None"):
    """
    Main function to process the aligned_dict_rs images into slices and save them out

    Parameters
    ----------
    aligned_dict_rs : dict
        dictionary of images to process
    figdir : Path
        directory to save out images
    FOVid : pd.Series
        id of the FOV
    cell_id : str
        id of the cell, if applicable

    Returns
    -------
    contour_overlay_images_stacked : dict
        dictionary of images with contours overlayed
    seg_images : dict
        dictionary of 3d segmentation images
    raw_images : dict
        dictionary of raw images
    contour_overlay_images_indiv_slices : dict
        dictionary of images with contours overlayed for individual slices
    """
    slice_dict = process_image_dict_into_slices(aligned_dict_rs)
    slice_dict_indiv_slices = convert_values_from_lists_of_slices_into_single_slices(slice_dict)
    slice_dict_stacked = stack_slices_within_image_dict_of_slices(
        slice_dict, gap_color=255, width=1
    )
    slice_dict_stacked0 = stack_slices_within_image_dict_of_slices(slice_dict, gap_color=0, width=1)
    contours_dict_indiv_slices = determine_contours_for_image_dict(slice_dict_indiv_slices)
    contours_dict_stacked = determine_contours_for_image_dict(slice_dict_stacked0)

    # now draw the contours on the images
    contour_overlay_images_indiv_slices = draw_contours_on_all_images_in_dict(
        slice_dict_indiv_slices, contours_dict_indiv_slices
    )
    contour_overlay_images_stacked = draw_contours_on_all_images_in_dict(
        slice_dict_stacked, contours_dict_stacked
    )

    save_out_images_in_dict(slice_dict_stacked, FOVid, cell_id, figdir)
    save_out_images_in_dict(slice_dict_indiv_slices, FOVid, cell_id, figdir)
    save_out_images_in_dict_with_overlay(contour_overlay_images_stacked, FOVid, cell_id, figdir)
    save_out_images_in_dict_for_quick_scroll_review(
        contour_overlay_images_stacked, FOVid, cell_id, figdir
    )
    save_out_images_in_dict_with_overlay_as_PDF(
        slice_dict_stacked, contours_dict_stacked, FOVid, cell_id, figdir
    )

    label_dict = simple_convert_slice_dict_label_seg_images_to_rgb(slice_dict)
    label_dict_stacked = stack_slices_within_image_dict_of_slices(
        label_dict, gap_color=255, width=1, isrgb=True
    )
    save_out_seg_images_as_label_image(label_dict_stacked, figdir, FOVid, cell_id)

    return contour_overlay_images_stacked, contour_overlay_images_indiv_slices
