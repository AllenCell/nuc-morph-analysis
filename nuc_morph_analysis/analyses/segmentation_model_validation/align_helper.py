import skimage.exposure as skex
import cv2 as cv
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift


def filter_out_nonoverlapping_objects(reference_img, moving_img):
    """
    Filter out objects that are not present in BOTH the reference_img and the moving_img

    Parameters
    ----------
    reference_img : np.array
        The reference image (instance segmentation label image)
    moving_img : np.array
        The moving image (instance segmentation label image)

    Returns
    -------
    moving_img : np.array
        The moving image with objects NOT present in the reference image set to 0
    reference_img : np.array
        The reference image with objects NOT present in the moving image set to 0
    """
    # now only keep pixels in the moving image that overlap with pixels in the reference image
    ref_max = reference_img.max(axis=0) > 0
    mov_max = moving_img.max(axis=0)
    mov_max[~ref_max] = 0
    # now identify the unique pixels left in moving_img
    unique_vals_mov = np.unique(mov_max)
    # now set everywhere in moving_img that is not in unique_vals to 0
    moving_img[~np.isin(moving_img, unique_vals_mov)] = 0

    # repeat with ref_max and ref_img
    mov_max = moving_img.max(axis=0) > 0
    ref_max = reference_img.max(axis=0)
    ref_max[~mov_max] = 0
    unique_vals_ref = np.unique(ref_max)
    reference_img[~np.isin(reference_img, unique_vals_ref)] = 0

    return moving_img, reference_img


def determine_shift(reference_img, moving_img, restrict_to_overlapping_objects=True):
    """
    Determine the shift between two images using phase cross correlation

    Parameters
    ----------
    reference_img : np.array
        The reference image (binarized, NOT label image)
    moving_img : np.array
        The moving image (binarized, NOT label image)
    restrict_to_overlapping_objects : bool
        If True, only consider objects that are present in both images

    Returns
    -------
    shift_estimate : np.array
        The shift necessary to move the moving image to align with the reference image
    """
    if restrict_to_overlapping_objects:
        moving_img, reference_img = filter_out_nonoverlapping_objects(
            reference_img, moving_img.copy()
        )

    shift_estimate, _, _ = phase_cross_correlation(reference_img > 0, moving_img > 0)

    return shift_estimate * -1


def apply_shift(moving_img, shift_estimate, order=0):
    """
    Shift the moving image by the shift estimate

    Parameters
    ----------
    moving_img : np.array
        The moving image (label image)
    shift_estimate : np.array
        The shift estimate
    order : int
        The order of interpolation to use based on scipy.ndimage.shift function

    Returns
    -------
    registered_img : np.array
        The registered image
    """

    registered_img = shift(moving_img, -np.round(shift_estimate), order=order).astype("int32")
    return registered_img
