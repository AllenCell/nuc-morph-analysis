from typing import Collection
import numpy as np
from skimage import transform
from nuc_morph_analysis.lib.preprocessing.system_info import (
    PIXEL_SIZE_Z_20x,
    PIXEL_SIZE_YX_20x,
    PIXEL_SIZE_Z_100x,
    PIXEL_SIZE_YX_100x,
    RESCALE_FACTOR_100x_to_20X,
)


def stack_uneven_middle(arrays: Collection[np.ndarray], fill_value=0.0, dtype=None) -> np.ndarray:
    """
    Fits arrays into a single numpy array, even if they are
    different sizes. `fill_value` is the default value.
    Args:
            arrays: list of np arrays of various sizes
                (must be same rank, but not necessarily same size)
            fill_value (float, optional):

    Returns:
            np.ndarray
    """
    # Get the max size in each dimension
    size = tuple(max(s) for s in zip(*[a.shape for a in arrays]))
    # Make all the arrays the same size
    padded_arrays = [_pad_to(a, size, fill_value) for a in arrays]
    result = np.stack(padded_arrays)
    if dtype is not None:
        result = result.astype(dtype)
    assert result.shape == (len(arrays),) + size
    return result


def _pad_to(array, size, fill_value):
    """
    Pad an array to a target size, centered. Array.shape must be smaller than `size`.
    """
    padding = [(diff // 2, diff - diff // 2) for diff in np.subtract(size, array.shape)]
    result = np.pad(array, padding, constant_values=fill_value)
    assert result.shape == size
    return result


def get_yx_zx_zy_slices(img, max_sum_slice=("sum", "max", 0), dtype="uint8"):
    """
    This function takes in an image and returns the yx,zx, and zy slice views.

    Parameters
    ----------
    img : np.array
        The image to be sliced.
    seg_or_raw : str
        The type of image, either 'seg' or 'raw'. Important for determining interpolation when upsmapling or downsampling
    max_sum_slice : tuple
        A tuple containing the method to determine the slice to be used for the yx view.
        The first, second, and third elements are the method to be used for getting a slice for the yx, zx, and zy views respectively.
        "sum" will return a sum intensity projection
        "mid" will return the middle slice
        "max" will return a max intensity projection
        int will return the slice at that index.
    dtype : str
        The data type of the output image.

    Returns
    -------
    slice_list : list
        A list containing the yx, zx, and zy slices respectively.
    """

    slice_list = []
    for si, method in enumerate(max_sum_slice):
        if method == "sum":
            sum = np.float32(img).sum(axis=si)
            if sum.max() != 0:
                # slice_list.append(np.uint8((sum/sum.max())*255))
                sum_array = sum / sum.max() * np.iinfo(dtype).max
                slice_list.append(sum_array.astype(dtype))
            else:
                # Handle the case where sum.max() is 0
                slice_list.append(sum.astype(dtype))
                pass
        elif method == "max":
            slice_list.append(img.max(axis=si).astype(dtype))
        elif method == "mid":
            # take the middle slice
            slice_list.append(np.take(img, img.shape[si] // 2, axis=si).astype(dtype))
        else:
            # take a specified integer slice
            slice_list.append(np.take(img, method, axis=si).astype(dtype))
    return slice_list


def flip_slices(slice_list, slices_to_flip, dtype="uint8"):
    """
    This function takes in a list of slices and flips them in the z direction.

    Parameters
    ----------
    slice_list : list
        A list containing the yx, zx, and zy slices respectively.
    slices_to_flip : list
        A list containing the indexes of the slices to flip.
    dtype : str
        The data type of the output image.

    Returns
    -------
    slice_list : list
        A list containing the yx, zx, and zy slices respectively, flipped in the z direction.
    """
    for si in slices_to_flip:
        slice_list[si] = np.flipud(slice_list[si]).astype(dtype)
    return slice_list


def get_scale_factors_for_resizing(input_mag="100x"):
    """
    This returns the scale factors for resizing slices of an image.

    Parameters
    ----------
    input_mag : str
        The magnification of the input image. Either '100x' or '20x'

    Returns
    -------
    scale_factors : list
        A list of tuples containing the scale factors for rescaling the yx, zx, and zy slices respectively.
    """
    if input_mag == "100x":
        scale_factors = scale_factors = [
            (1, 1),
            (PIXEL_SIZE_Z_100x / PIXEL_SIZE_YX_100x, 1),
            (PIXEL_SIZE_Z_100x / PIXEL_SIZE_YX_100x, 1),
        ]

    elif input_mag == "20x":
        scale_factors = [
            (1, 1),
            (PIXEL_SIZE_Z_20x / PIXEL_SIZE_YX_20x, 1),
            (PIXEL_SIZE_Z_20x / PIXEL_SIZE_YX_20x, 1),
        ]
    else:
        raise ValueError("input_mag must be either '100x' or '20x'")
    return scale_factors


def resize_slices(slice_list, slices_to_resize, order, scale_factors=None, dtype="uint8"):
    """
    This function takes in a list of slices and resizes them.

    Parameters
    ----------
    slice_list : list
        A list containing the yx, zx, and zy slices respectively.
    slices_to_resize : list
        A list containing the slices to resize.
    order : int
        The order of interpolation to be used.
    scale_factors: list of tuples
        [yx_rescale factors, zx_rescale factors, zy_rescale factors]
        The ratios by which to rescale the each of the three slices
    dtype : str
        The data type of the output image.

    Returns
    -------
    slice_list : list
        A list containing the yx, zx, and zy slices respectively, resized.
    """
    if scale_factors is None:
        scale_factors = [(1, 1), (1, 1), (1, 1)]
    for si in slices_to_resize:
        slice_list[si] = transform.rescale(
            slice_list[si], scale_factors[si], preserve_range=True, anti_aliasing=False, order=order
        ).astype(dtype)
    return slice_list


def process_image_into_slices(img, max_sum_slice, order, scale_factors=None, dtype="uint8"):
    """
    This function takes in an image and returns the yx, zx, and zy slice views in a list

    Parameters
    ----------
    img : np.array
        The image to be sliced.
    max_sum_slice : tuple
        A tuple containing the method to determine the slice to be used for the yx view.
        The first, second, and third elements are the method to be used for getting a slice for the yx, zx, and zy views respectively.
        "sum" will return a sum intensity projection
        "mid" will return the middle slice
        "max" will return a max intensity projection
        int will return the slice at that index.
    order : int
        The order of interpolation to be used.
    scale_factors: list of tuples
        [yx_rescale factors, zx_rescale factors, zy_rescale factors]
        The ratios by which to rescale the each of the three slices
    dtype : str
        The data type of the output image.

    Returns
    -------
    stacked_slices : np.array
        The list of slices [yx,zx,zy]
    """
    # first flip the y direction
    img = np.flip(img, axis=1)
    slices = get_yx_zx_zy_slices(img, max_sum_slice=max_sum_slice, dtype=dtype)
    # now flip the other slices
    slices = flip_slices(slices, [1, 2], dtype=dtype)
    slices = resize_slices(slices, [1, 2], order, scale_factors, dtype=dtype)
    return slices


def stack_the_slices_zyx(slice_list, width=1, gap_color=255, isrgb=False):
    """
    This function takes in a list of slices and stacks them together.

    Parameters
    ----------
    slice_list : list
        A list containing the yx, zx, and zy slices respectively.
    width : int
        The width of the gap between slices.
    gap_color : int
        The color of the gap between slices.
    isrgb : bool
        Whether the slices are RGB or not.

    Returns
    -------
    yx_zx_zy : np.array
        The stacked slices.
    """

    yx, zx, zy = slice_list

    def create_gap(size, channels=1):
        if channels > 1:
            return np.ones([width, size, channels], dtype="uint8") * gap_color
        else:
            return np.ones([width, size], dtype="uint8") * gap_color

    def create_blank(size, channels=1, gap_color=255):
        if channels > 1:
            return np.ones([size, size, channels], dtype="uint8") * gap_color
        else:
            return np.ones([size, size], dtype="uint8") * gap_color

    def concatenate_slices(yx, zx, zy, channels=1):
        if width > 0:
            yx_zx = np.concatenate([yx, create_gap(zx.shape[1], channels), zx], axis=0).astype(
                "uint8"
            )
            zy_blank = np.concatenate(
                [
                    np.swapaxes(zy, 1, 0),
                    create_gap(zy.shape[0], channels),
                    create_blank(zx.shape[0], channels, gap_color),
                ],
                axis=0,
            ).astype("uint8")
            yx_zx_zy = np.concatenate(
                [yx_zx, np.swapaxes(create_gap(yx_zx.shape[0], channels), 1, 0), zy_blank], axis=1
            ).astype("uint8")
        else:
            yx_zx = np.concatenate([yx, zx], axis=0).astype("uint8")
            zy_blank = np.concatenate(
                [np.swapaxes(zy, 1, 0), create_blank(zx.shape[0], channels, gap_color)], axis=0
            ).astype("uint8")
            yx_zx_zy = np.concatenate([yx_zx, zy_blank], axis=1).astype("uint8")

        return yx_zx_zy

    if isrgb:
        channels = 3
    else:
        channels = 1

    yx_zx_zy = concatenate_slices(yx, zx, zy, channels)

    return yx_zx_zy


def get_single_crop(img_out, y0, x0, crop_sizes, scale_factor, first_dtype, return_roi_array=False):
    """
    Get a single crop image from the FOV image (img_out)

    Parameters
    ----------
    img_out : numpy array
        3d image of FOV for the timepoint
    y0 : int
        Y coordinate for the crop center
    x0 : int
        X coordinate for the crop center
    crop_sizes : tuple
        Size of the crop in YX dimensions
    scale_factor : float
        Scale factor to apply to the crop sizes and the YX coordinates
    first_dtype : str
        The data type of the output image
    return_roi_array : bool
        Whether to return the roi array instead of the image

    Returns
    -------
    crop_img : numpy array
        The cropped image

    """
    y = int(y0 * scale_factor)
    x = int(x0 * scale_factor)
    # initialize the crop image as zeros
    crop_img = np.zeros((img_out.shape[0], crop_sizes[0], crop_sizes[1]), dtype=first_dtype)
    y1 = np.max([int(y - crop_sizes[0] / 2), 0])
    y2 = np.min([int(y + crop_sizes[0] / 2), img_out.shape[1]])
    x1 = np.max([int(x - crop_sizes[1] / 2), 0])
    x2 = np.min([int(x + crop_sizes[1] / 2), img_out.shape[2]])
    crop_exp = np.index_exp[:, y1:y2, x1:x2]
    crop_img[:, : y2 - y1, : x2 - x1] = img_out[crop_exp]
    if return_roi_array:
        return np.asarray([0, img_out.shape[0], y1, y2, x1, x2])
    return crop_img


def determine_shape_of_20x_upsampled_to_100x(img):
    """
    Determine the shape of the 20x image upsampled to 100x

    Parameters
    ----------
    img: np.array
        The 20x image

    Returns
    -------
    new_shape : tuple
        The shape of the upsampled 20x image
    """
    new_shape = np.asarray(img.shape)
    # apply rescaling in Z
    new_shape[0] = new_shape[0] * (PIXEL_SIZE_Z_20x / PIXEL_SIZE_Z_100x)
    # apply rescaling in Y and X
    new_shape[1] = new_shape[1] * (1 / RESCALE_FACTOR_100x_to_20X)
    new_shape[2] = new_shape[2] * (1 / RESCALE_FACTOR_100x_to_20X)
    new_shape = np.round(new_shape).astype("int32")  # round to an integer size
    return new_shape


def resize_low_res_image(img):
    """
    Resize the low resolution image to match the size of the high resolution image

    Parameters
    ----------
    img : np.array
        The low resolution image

    Returns
    -------
    img : np.array
        The resized low resolution image
    """
    new_shape = determine_shape_of_20x_upsampled_to_100x(img)
    # resize the 20x to match the predicted 100x size
    img = transform.resize(
        img.copy(), new_shape, preserve_range=True, anti_aliasing=False, order=0
    ).astype(img.dtype)
    return img
