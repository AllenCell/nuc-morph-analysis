#%%
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
from bioio.writers import OmeTiffWriter
from pathlib import Path
from skimage.measure import regionprops_table
import pandas as pd
from nuc_morph_analysis.lib.preprocessing.system_info import PIXEL_SIZE_YX_100x
from nuc_morph_analysis.lib.preprocessing import load_data

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
            'mip_of_labeled_image':mip_of_labeled_image,
            'binarized_mip':binarized_mip,
            'distance':distance,
            'watershed_img':watershed_img,
            'pseudo_cells_img':pseudo_cells_img,
        }
        return pseudo_cells_img, mip_of_labeled_image, img_dict
    else:
        return pseudo_cells_img

def save_pseudo_cell_image(pseudo_cell_image, output_directory, colony, timepoint_frame, resolution_level=0,):
    """
    function for saving the pseudo cell image to the output directory

    Parameters
    ----------
    pseudo_cell_image : np.array
        the labeled pseudo cell image (2D) where each cell has a unique label corresponding to the label in the nucleus image
    output_directory : str
        the output directory to save the pseudo cell image
    colony : str
        the colony name
    timepoint_frame : int
        the timepoint frame to save the pseudo cell image
    resolution_level : int
        the resolution level (OME-ZARR) to save the pseudo cell image 

    Returns
    -------
    None
    """
    pseudo_cell_image = pseudo_cell_image.astype(np.uint16)
    save_dir = Path(output_directory) / colony 
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_name = f"{colony}_pseudo_cell_image_T{str(timepoint_frame).zfill(3)}_res{resolution_level}.ome.tif"
    save_path = Path(save_dir) / save_name
    OmeTiffWriter.save(pseudo_cell_image, save_path, dim_order="YX")


def extract_2d_features(label_image, colony='',timepoint=0, reader=None, resolution_level=0):
    
    """
    extract 2d shape features from the label image

    Parameters
    ----------
    label_image : np.array
        the labeled image (2D) where each object has a unique label
    colony : str
        the colony name
    timepoint : int
        the timepoint frame
    reader : zarr reader
        the reader, necessary for determining pixel size
    resolution_level : int
        the resolution level to load from the OME-ZARR (0 is full, 1 is 2.5x downsampled...equivalent to 20x image size)

    Returns
    -------
    df : pd.DataFrame
        the dataframe containing the features
    """

    # determine pixel size ratio since OME-ZARR does not store pixel size
    # correctly for different resolution levels yet

    reader.set_resolution_level(0)
    shape0 = reader.shape[-1]
    reader.set_resolution_level(resolution_level)
    shape1 = reader.shape[-1]
    pixel_ratio = shape0/shape1


    prop_list = regionprops_table(label_image,
                                properties=('label',
                                            'area',
                                            # 'bbox',
                                            # 'centroid',
                                            # 'convex_area',
                                            # 'eccentricity',
                                            # 'equivalent_diameter',
                                            # 'extent',
                                            # 'filled_area',
                                            # 'major_axis_length',
                                            # 'minor_axis_length',
                                            # 'orientation',
                                            # 'perimeter',
                                            # 'solidity',
                                            ),
                                    )
    df =  pd.DataFrame(prop_list)
    df['resolution_level'] = resolution_level
    df['index_sequence'] = timepoint
    df['colony'] = colony
    df['pixel_size'] = PIXEL_SIZE_YX_100x*pixel_ratio

    # rename all columns to have prefix "2d_"
    df.columns = ['2d_'+col if col not in ['label','index_sequence','colony'] else col for col in df.columns]
    df['label_img'] = df['label']
    # drop rows where label_img is 0
    df = df[df['label_img']!=0]

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

def define_density_feature(df_2d):
    """
    define the density feature in the 2D dataframe
    the density feature is defined as the area of a nucleus divided by the area of the (pseudo) cell

    Parameters
    ----------
    df_2d : pd.DataFrame
        the 2D dataframe containing the features from 2d labeled nucleus image and 2d watershed-based pseudo cell image

    Returns
    -------
    df_2d : pd.DataFrame
        the 2D dataframe with the density feature added (2d_area_nuc_cell_ratio)
    """
    df_2d['2d_area_nuc_cell_ratio'] = df_2d['2d_area_nucleus'] / df_2d['2d_area_pseudo_cell']
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
    feature_cols = ['2d_area_pseudo_cell','2d_area_nucleus','2d_area_nuc_cell_ratio','2d_resolution_level_nucleus','2d_resolution_level_pseudo_cell']
    columns_to_keep = merge_cols + feature_cols
    df_2d = df_2d[columns_to_keep]
    return df_2d

def get_pseudo_cell_boundaries(colony, timepoint, reader, resolution_level,return_img_dict=False):
    """
    determine pseudo cell boundaries for each nuclues in a labeled nucleus image and extract 2d_features
    the pseudo cell boundary is determined by a watershed segmentation of the max projection of the labeled nucleus image
    the features are the 2d areas (or more complicated 2d shape features) from the pseudo cell and nucleus images


    Parameters
    ----------
    colony : str
        the colony name
    timepoint : int
        the timepoint frame
    reader : zarr reader from bioio
        the reader for the OME-ZARR file image
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
    labeled_nucleus_image = reader.get_image_dask_data("ZYX", T=timepoint, C=0).compute()
    if return_img_dict:
        pseudo_cell_image,nucleus_image,img_dict = get_pseudo_cell_boundaries_from_labeled_nucleus_image(labeled_nucleus_image,return_img_dict=True)
    else:
        pseudo_cell_image,nucleus_image = get_pseudo_cell_boundaries_from_labeled_nucleus_image(labeled_nucleus_image,return_nucleus=True)

    # extract features from 2D label image
    pseudo_cell_features_df = extract_2d_features(pseudo_cell_image, colony=colony, timepoint=timepoint, reader=reader, resolution_level=resolution_level)
    nucleus_features_df = extract_2d_features(nucleus_image, colony=colony, timepoint=timepoint, reader=reader, resolution_level=resolution_level)

    # now merge the two dataframes
    df_2d = merge_2d_features(pseudo_cell_features_df, nucleus_features_df)
   
    # define the density measure (nuc_area_per_cell)
    df_2d = define_density_feature(df_2d)

    df_2d = choose_columns(df_2d)


    if return_img_dict:
        return df_2d, img_dict
    else:
        return df_2d