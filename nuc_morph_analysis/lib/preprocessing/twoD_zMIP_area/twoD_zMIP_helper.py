import cv2
import dask.array as da
import skimage.exposure as skex
from scipy.ndimage import binary_fill_holes, gaussian_filter
import numpy as np
from skimage.morphology import remove_small_objects
from scipy.signal import find_peaks

from nuc_morph_analysis.lib.preprocessing.load_data import get_seg_fov_for_row


def find_focused_brightfield_slices(fov_bri_data, collect_intermediates=False):
    std_profile = fov_bri_data.std(axis=(1, 2))
    mean_profile = fov_bri_data.mean(axis=(1, 2))
    profile = std_profile / mean_profile
    feats = {"profile": profile} if collect_intermediates else {None: None}

    # use peak_finder find the trough in the profile
    peaks, _ = find_peaks(-1 * profile, distance=20)

    # if no peak is found
    if len(peaks) == 0:
        feats.update(
            {"brightfield_in_focus_slice": np.nan} if collect_intermediates else {None: None}
        )
        feats.update({"first_peak_slice": 0} if collect_intermediates else {None: None})
        feats.update(
            {"second_peak_slice": fov_bri_data.shape[0] - 1}
            if collect_intermediates
            else {None: None}
        )
        feats.update({"maximum_contrast_slice": np.nan} if collect_intermediates else {None: None})
        feats.update(
            {"brightfield_best_contrast_slice": np.nan} if collect_intermediates else {None: None}
        )
        brightfield_best_contrast_slice = np.nan
        brightfield_in_focus_slice = np.nan
    else:

        brightfield_in_focus_slice = peaks[np.argmin(profile[peaks])]
        feats.update(
            {"brightfield_in_focus_slice": brightfield_in_focus_slice}
            if collect_intermediates
            else {None: None}
        )

        # find the first peak (the peak in the profile before the trough)
        first_peak_slice = np.argmax(profile[:brightfield_in_focus_slice])
        feats.update(
            {"first_peak_slice": first_peak_slice} if collect_intermediates else {None: None}
        )

        # now find the second peak (the peak in the profile after the trough)
        max_after_trough = np.argmax(profile[brightfield_in_focus_slice:])
        second_peak_slice = max_after_trough + brightfield_in_focus_slice
        maximum_contrast_slice = second_peak_slice
        feats.update(
            {"second_peak_slice": second_peak_slice} if collect_intermediates else {None: None}
        )
        feats.update(
            {"maximum_contrast_slice": maximum_contrast_slice}
            if collect_intermediates
            else {None: None}
        )

        brightfield_best_contrast_slice = np.mean(
            [brightfield_in_focus_slice, second_peak_slice]
        ).astype(int)
        feats.update(
            {"brightfield_best_contrast_slice": brightfield_best_contrast_slice}
            if collect_intermediates
            else {None: None}
        )

    if collect_intermediates:
        return brightfield_best_contrast_slice, brightfield_in_focus_slice, feats
    else:
        return brightfield_best_contrast_slice, brightfield_in_focus_slice


def draw_contours_on_infocus_slice(brightfield_in_focus_slice, seg_slice):
    contours, hierarchy = cv2.findContours(
        seg_slice.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    im_lo, im_hi = np.percentile(brightfield_in_focus_slice, (10, 95))
    in_focus_rs = skex.rescale_intensity(
        brightfield_in_focus_slice, in_range=(im_lo, im_hi), out_range=(0, 255)
    ).astype("uint8")
    in_focus_rgb = np.stack([in_focus_rs] * 3, axis=-1)
    contour_img = cv2.drawContours(in_focus_rgb.copy(), contours, -1, (255, 255, 0), 2)
    return contour_img


def segment_colony_brightfield(bri, thresh=0.025, ksize=5, collect_intermediates=False):
    # find the first_peak slice and the second_peak slice (which are the slices with highest contrast)
    _, _, slice_feats = find_focused_brightfield_slices(bri, collect_intermediates=True)

    # only analyze the bright_field image between the frames of highest contrast the first peak to the second peak
    bri = bri[slice_feats["first_peak_slice"] : slice_feats["second_peak_slice"]]

    brightfield_img_mean = np.mean(bri)
    # retrieve the standard deviation in the z plane and in focus slice
    sip = np.std(bri, axis=0) / brightfield_img_mean
    intermediates = {"sip": sip} if collect_intermediates else {None: None}

    # Sobel edge detection
    if isinstance(sip, da.Array):
        sip = sip.compute()
    edges = np.abs(cv2.Sobel(sip, cv2.CV_64F, 1, 1, ksize=ksize))
    # Thresholding
    processed_img = edges > thresh
    intermediates.update(
        {"edges": edges, "thresholded": processed_img.copy()}
        if collect_intermediates
        else {None: None}
    )
    # remove small objects
    processed_img = remove_small_objects(processed_img, min_size=10000)
    intermediates.update(
        {"remove_small_objects": processed_img} if collect_intermediates else {None: None}
    )

    # apply hole filling
    processed_img = binary_fill_holes(processed_img)
    intermediates.update({"hole_fill1": processed_img} if collect_intermediates else {None: None})

    # Apply a Gaussian blur to the binary mask and threshold
    processed_img = gaussian_filter(processed_img.astype(float), sigma=3) > 0.5
    intermediates.update({"gaussian": processed_img} if collect_intermediates else {None: None})

    # fill the remaining holes
    processed_img = binary_fill_holes(processed_img)
    intermediates.update(
        {"hole_fill2=final": processed_img} if collect_intermediates else {None: None}
    )

    bad_segmentation_flag = False

    feats = {
        "processed_img": np.uint8(processed_img) * 255,
        "colony_area": np.sum(processed_img),
        "brightfield_img_mean": brightfield_img_mean,
    }
    # if an in focus brighfield slice could not be found, do not trust the segmentation
    if slice_feats["brightfield_best_contrast_slice"] == np.nan:
        feats.update({"colony_area": np.nan})

    if collect_intermediates:
        return feats, intermediates
    else:
        return feats


def get_best_contrast_brightfield_img_slices(bri, collect_intermediates=False):
    if collect_intermediates:
        brightfield_best_contrast_slice, brightfield_in_focus_slice, intermediates = (
            find_focused_brightfield_slices(bri, collect_intermediates=collect_intermediates)
        )
        intermediates.update(
            {"in_focus_img": bri[brightfield_in_focus_slice]}
            if brightfield_in_focus_slice is not np.nan
            else {"in_focus_img": np.zeros_like(bri[0])}
        )
        intermediates.update(
            {"best_contrast_img": bri[brightfield_best_contrast_slice]}
            if brightfield_best_contrast_slice is not np.nan
            else {"best_contrast_img": np.zeros_like(bri[0])}
        )
    else:
        brightfield_best_contrast_slice, brightfield_in_focus_slice = (
            find_focused_brightfield_slices(bri, collect_intermediates=collect_intermediates)
        )

    best_contrast_img = (
        bri[brightfield_best_contrast_slice]
        if brightfield_best_contrast_slice is not np.nan
        else np.zeros_like(bri[0])
    )
    im_lo, im_hi = np.percentile(best_contrast_img, (10, 95))
    best_contrast_img = skex.rescale_intensity(
        best_contrast_img, in_range=(im_lo, im_hi), out_range=(0, 255)
    ).astype("uint8")
    feats = {
        "brightfield_best_contrast_slice": brightfield_best_contrast_slice,
        "brightfield_in_focus_slice": brightfield_in_focus_slice,
        "best_contrast_img": best_contrast_img,
    }
    if collect_intermediates:
        return feats, intermediates
    else:
        return feats
