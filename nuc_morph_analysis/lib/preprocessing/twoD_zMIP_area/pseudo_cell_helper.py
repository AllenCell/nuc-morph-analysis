#%%
import numpy as np
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops_table
import pandas as pd
from nuc_morph_analysis.lib.preprocessing.system_info import PIXEL_SIZE_YX_100x

#%% define key functions
def get_pseudo_cell_boundaries_from_labeled_nucleus_image(labeled_nucleus_image, return_nucleus=False, return_img_dict=False):
    """
    function for returning the pseudo cell boundaries at all timepoints
    the psuedo cell boundary is a watershed segmentation of the max projection of the segmentation image
    this is similar to a Voronoi tesselation output except it is determined using the nucleus boundary
    rather than just the centroid, therefore giving more accurate results

    Parameters
    ----------
    labeled_nucleus_image : np.array
        the labeled nucleus image (3D) where each nucleus has a unique label
    return_nucleus : bool
        whether to return the nucleus imag (2D) as well 
    return_img_dict : bool
        whether to return the dictionary of images for validation
    
    Returns
    -------
    pseudo_cells_img : np.array
        the labeled pseudo cell image (2D) where each cell has a unique label
    nucleus_img : np.array
        the labeled nucleus image (2D) where each nucleus has a unique label
    img_dict : dict
        the dictionary of images for validation
    """
    # convert to max projection
    mip_of_labeled_image = np.max(labeled_nucleus_image, axis=0)

    # convert the image to binary
    binarized_mip = mip_of_labeled_image>0

    # compute distance transform (on the inverse of the image)
    distance = distance_transform_edt(~binarized_mip)

    # run watershed
    watershed_img = watershed(distance,markers=mip_of_labeled_image)

    # now map the labels in `labels` to the original image (mip2)
    # first, create a mask of the labels
    pseudo_cells_img = np.zeros_like(mip_of_labeled_image)
    for i in range(1, watershed_img.max()):
        mask = watershed_img == i
        pixels_in_cell = mip_of_labeled_image[mask]
        pixels_in_cell = pixels_in_cell[pixels_in_cell>0] # dont keep 0

        if len(pixels_in_cell) == 0:
            continue

        # if pixels in cell touches edge of image then remove
        if (np.any(mask[0,:]) or np.any(mask[-1,:]) or np.any(mask[:,0]) or np.any(mask[:,-1])):
            continue

        # find the most common pixel value
        most_common_pixel_value = np.bincount(pixels_in_cell).argmax()
        pseudo_cells_img[mask] = most_common_pixel_value


    if return_nucleus:
        return pseudo_cells_img, mip_of_labeled_image
    elif return_img_dict:
        img_dict = {
            'mip_of_labeled_image':(mip_of_labeled_image,"MIP of input segmentation"),
            'binarized_mip':(binarized_mip, "Binarization of MIP"),
            'distance':(distance, "Distance Transform"),
            'watershed_img':(watershed_img, "Watershed of Distance Transform"),
            'pseudo_cells_img':(pseudo_cells_img, "Pseudo Cell Segmentation"),
        }
        return pseudo_cells_img, mip_of_labeled_image, img_dict
    else:
        return pseudo_cells_img

def extract_2d_features(label_image):
    
    """
    extract 2d shape features from the label image

    Parameters
    ----------
    label_image : np.array
        the labeled image (2D) where each object has a unique label

    Returns
    -------
    df : pd.DataFrame
        the dataframe containing the features
    """

    prop_list = regionprops_table(label_image,
                                properties=('label',
                                            'area',
                                            'bbox',
                                            'centroid',
                                            'convex_area',
                                            'eccentricity',
                                            'equivalent_diameter',
                                            'extent',
                                            'filled_area',
                                            'major_axis_length',
                                            'minor_axis_length',
                                            'orientation',
                                            'perimeter',
                                            'solidity',
                                            ),
                                    )
    df =  pd.DataFrame(prop_list)
    return df

def add_metadata_to_df(df, colony, timepoint, resolution_level=0):
    """
    add metadata to the dataframe that is important for merging with the tracking manifest(s)

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe containing the features
    colony : str
        the colony name
    timepoint : int
        the timepoint frame
    resolution_level : int
        the resolution level to load from the OME-ZARR (0 is full, 1 is 2.5x downsampled...equivalent to 20x image size)

    Returns
    -------
    df : pd.DataFrame
        the dataframe containing the features with metadata added
    """
    
    df['resolution_level'] = resolution_level
    df['index_sequence'] = timepoint
    df['colony'] = colony

    # rename all columns to have prefix "2d_"
    df.columns = ['2d_'+col if col not in ['label','index_sequence','colony'] else col for col in df.columns]
    df['label_img'] = df['label']
    # drop rows where label_img is 0
    df = df[df['label_img']!=0]
    return df

def add_pixel_size_info(df, resolution_level, reader=None):
    """
    add pixel size to the dataframe after adjusting the size for the resolution level
    Parameters
    ----------
    df : pd.DataFrame
        the dataframe containing the features
    resolution_level : int
        the resolution level to load from the OME-ZARR (0 is full, 1 is 2.5x downsampled...equivalent to 20x image size)
    reader : zarr reader from bioio
        the reader for the OME-ZARR file image

    """
    # determine pixel size ratio since OME-ZARR does not store pixel size
    # correctly for different resolution levels yet
    
    if resolution_level>0:
        reader.set_resolution_level(0)
        shape0 = reader.shape[-1]
        reader.set_resolution_level(resolution_level)
        shape1 = reader.shape[-1]
        pixel_ratio = shape0/shape1
    else:
        pixel_ratio = 1
    df['pixel_size'] = PIXEL_SIZE_YX_100x*pixel_ratio
    return df

def merge_2d_features(pseudo_cell_features_df, nucleus_features_df):
    """
    merge the 2D features from the pseudo cell and nucleus images

    Parameters
    ----------
    pseudo_cell_features_df : pd.DataFrame
        the dataframe containing the 2D features from the pseudo cell image
    nucleus_features_df : pd.DataFrame
        the dataframe containing the 2D features from the nucleus image

    Returns
    -------
    df_2d : pd.DataFrame
        the merged dataframe containing the 2D features from the pseudo cell and nucleus images
    """
    df_2d = pd.merge(pseudo_cell_features_df, nucleus_features_df, on=['colony','label_img','index_sequence'], suffixes=('_pseudo_cell', '_nucleus'))
    return df_2d

def define_density_features(df_2d):
    """
    define density features in the 2D dataframe
    the first density feature (2d_area_nuc_cell_ratio) is defined as the area of a nucleus divided by the area of the (pseudo) cell
    the second density feature (inv_cyto_density) is defined as the inverse of the cytoplasmic area
        a sub feature (2d_area_cyto) is also defined as the area of the (pseudo) cell minus the area of the nucleus

    Parameters
    ----------
    df_2d : pd.DataFrame
        the 2D dataframe containing the features from 2d labeled nucleus image and 2d watershed-based pseudo cell image

    Returns
    -------
    df_2d : pd.DataFrame
        the 2D dataframe with the density features added (2d_area_nuc_cell_ratio, inv_cyto_density, 2d_area_cyto)
    """
    df_2d['2d_area_nuc_cell_ratio'] = df_2d['2d_area_nucleus'] / df_2d['2d_area_pseudo_cell'] # unitless
    df_2d['2d_area_cyto'] = df_2d['2d_area_pseudo_cell'] - df_2d['2d_area_nucleus'] # units of pixel_area
    df_2d['inv_cyto_density'] = 1/df_2d['2d_area_cyto'] # units of 1/pixel_area
    return df_2d

def choose_columns(df_2d):
    """
    choose the columns to keep in the 2D dataframe

    Parameters
    ----------
    df_2d : pd.DataFrame
        the 2D dataframe containing the features from 2d labeled nucleus image and 2d watershed-based pseudo cell image

    Returns
    -------
    df_2d : pd.DataFrame
        the 2D dataframe with the columns chosen
    """
    merge_cols = ['label_img','index_sequence','colony']
    feature_cols = ['2d_area_pseudo_cell','2d_area_nucleus','2d_area_nuc_cell_ratio',
                    '2d_area_cyto','inv_cyto_density',
                    '2d_resolution_level_nucleus','2d_resolution_level_pseudo_cell']
    columns_to_keep = merge_cols + feature_cols
    df_2d = df_2d[columns_to_keep]
    return df_2d

def get_pseudo_cell_boundaries(labeled_nucleus_image, colony='test', timepoint=0, reader=None, resolution_level=0, return_img_dict=False):
    """
    determine pseudo cell boundaries for each nuclues in a labeled nucleus image and extract 2d_features
    the pseudo cell boundary is determined by a watershed segmentation of the max projection of the labeled nucleus image
    the features are the 2d areas (or more complicated 2d shape features) from the pseudo cell and nucleus images


    Parameters
    ----------
    labeled_nucleus_image : np.array
        the labeled nucleus image (3D) where each nucleus has a unique label
    colony : str
        the colony name, default is 'test'
    timepoint : int
        the timepoint frame, default is 0
    reader : zarr reader from bioio
        the reader for the OME-ZARR file image, default is None
    resolution_level : int
        the resolution level to load from the OME-ZARR (0 is full, 1 is 2.5x downsampled...equivalent to 20x image size)
    return_img_dict : bool
        whether to return the dictionary of images for validation
    
    Returns
    -------
    df_2d : pd.DataFrame
        the dataframe containing the 2D features from the pseudo cell and nucleus images
        the key columns are for merging are label_img,index_sequence and colony
    img_dict : dict
        the dictionary of images for validation
    """
    # require image to be integer type
    assert labeled_nucleus_image.dtype in [np.uint8, np.uint16, np.int32, np.int64]

    if return_img_dict:
        pseudo_cell_image,nucleus_image,img_dict = get_pseudo_cell_boundaries_from_labeled_nucleus_image(labeled_nucleus_image,return_img_dict=True)
    else:
        pseudo_cell_image,nucleus_image = get_pseudo_cell_boundaries_from_labeled_nucleus_image(labeled_nucleus_image,return_nucleus=True)

    # extract features from 2D label image
    pseudo_cell_features_df = extract_2d_features(pseudo_cell_image)
    nucleus_features_df = extract_2d_features(nucleus_image)

    # add timepoint, colony, pixel_size, label_img to the dataframes
    pseudo_cell_features_df = add_metadata_to_df(pseudo_cell_features_df, colony, timepoint, resolution_level)
    nucleus_features_df = add_metadata_to_df(nucleus_features_df, colony, timepoint, resolution_level)
    
    # add pixel size info to the dataframes
    pseudo_cell_features_df = add_pixel_size_info(pseudo_cell_features_df, resolution_level, reader)
    nucleus_features_df = add_pixel_size_info(nucleus_features_df, resolution_level, reader)

    # now merge the two dataframes
    df_2d = merge_2d_features(pseudo_cell_features_df, nucleus_features_df)
   
    # define the density measure (2d_area_nuc_cell_ratio)
    df_2d = define_density_features(df_2d)

    # df_2d = choose_columns(df_2d)

    if return_img_dict:
        return df_2d, img_dict
    else:
        return df_2d