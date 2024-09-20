#%%
import numpy as np
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops_table
import pandas as pd
from pathlib import Path
from skimage.morphology import binary_erosion

"""
columns added are:
# for the nucleus and pseudocell segmentation images the following properties are calculated

# most features come from skimage.measure.regionprops_table
# for details about these features see scikit-image documentation
properties = [
'label',
'area',
'bbox', # becomes bbox-1, bbox-2, bbox-3
'centroid', # becomes centroid-1, centroid-2
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
]

# for the nucleus all features will take the form of 2d_{feature}_nucleus, such as 2d_area_nucleus
# for the pseudo cell all features will take the form of 2d_{feature}_pseudo_cell, such as 2d_area_pseudo_cell

# some specific new features are computed from these
[
'2d_area_nuc_cell_ratio', # ratio of nucleus area to pseudo cell area
'2d_area_cyto', # cytoplasmic area (pseudo cell area - nucleus area)
'inv_cyto_density', # inverse of cytoplasmic area (1/cytoplasmic area)
]

# some features come from measuring the true area of each object without using skimage.measure.regionprops_table
[
'label_true', 
'area_true', # becomes 2d_area_true_nucleus or 2d_area_true_pseudo_cell
'total_area_true',
]

# to get at proximity of a nucleus to its neighbors
# and some intensity features are computed
# these use a label image of the nucleus periphery (called the nucleus edge image)
# and as an intensity image they use the distance transform of the pseudo cell segmentation image (called the cyto_distance image)
# the minimum intensity of the cyto_distance image in the nucleus edge image is the minimum distance of the nucleus to its neighbors
[
'2d_intensity_max_edge',
'2d_intensity_mean_edge',
'2d_intensity_min_edge', 
]
"""

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
        pixels_in_cell = mip_of_labeled_image[mask].copy()
        pixels_in_cell = pixels_in_cell[pixels_in_cell>0] # dont keep 0

        if len(pixels_in_cell) == 0:
            continue

        # if pixels in cell touches edge of image then remove
        if (np.any(mask[0,:]) or np.any(mask[-1,:]) or np.any(mask[:,0]) or np.any(mask[:,-1])):
            continue

        # find the most common pixel value
        most_common_pixel_value = np.bincount(pixels_in_cell).argmax()
        pseudo_cells_img[mask] = most_common_pixel_value

    # to measure how close nuclei are to neighbors find the nucleus edge
    nucleus_eroded = binary_erosion(binarized_mip) # erode the nuclei
    nucleus_edges = np.logical_xor(binarized_mip,nucleus_eroded) # find the edges using XOR
    
    nucleus_edge_img = np.zeros_like(mip_of_labeled_image) # initialize image
    nucleus_edge_img[nucleus_edges] = mip_of_labeled_image[nucleus_edges] # transfer labels

    # create a distance transform that computes distance from pseudo_cell edge
    cell_shed = watershed(distance,markers=mip_of_labeled_image,watershed_line=True)
    cell_shed[pseudo_cells_img==0] = 0 # remove background pixels
    cyto_distance = distance_transform_edt(cell_shed>0)

    # now return cyto_distance as an "intensity image" for the nucleus_edge_img
    # then in skimage.measure.regionprops_table, use intensity_image=cyto_distance and labels=nucleus_edge_img
    # then you can get min_intensity, max_intensity, mean_intensity, etc. for each cell which will tell you
    # if the nucleus is touching a neighbor or not (e.g. if min_intensity is 0 then the nucleus is touching a neighbor)

    basic_dict = {
        "nucleus_img":mip_of_labeled_image,
        "pseudo_cells_img":pseudo_cells_img,
        "nucleus_edge_img":nucleus_edge_img,
        "cyto_distance":cyto_distance,
    }

    if return_img_dict:
        img_dict = {
            'mip_of_labeled_image':(mip_of_labeled_image,"MIP of input segmentation"),
            'binarized_mip':(binarized_mip, "Binarization of MIP"),
            'distance':(distance, "Distance Transform"),
            'watershed_img':(watershed_img, "Watershed of Distance Transform"),
            'pseudo_cells_img':(pseudo_cells_img, "Pseudo Cell Segmentation"),
            'cell_shed_bin': (cell_shed>0, "Binary Watershed of Distance Transform with Watershed Line"),
            'cyto_distance':(cyto_distance, "Distance Transform restricted to Cytoplasmic Segmentation"),
            'nucleus_edge_img':(nucleus_edge_img, "Nucleus Edge"),

        }
        return basic_dict, img_dict
    else:
        return basic_dict

def compute_true_area(label_img):
    """
    compute true area (in pixels) of each object
    (important as a sanity check for errors that have occured in parallel processing 
    and extracting features using skimage.measure.regionprops_table)

    Parameters
    ----------
    label_img : np.array
        the labeled image (2D) where each object has a unique label

    Returns
    -------
    df : pd.DataFrame
        the dataframe containing the true area of each object
    """
    total_area = np.sum(label_img>0)
    dflist =[]
    for li in np.unique(label_img):
        if li == 0:
            continue
        area = np.sum(label_img==li)
        feats = {'label_true':li,'area_true':area}
        dflist.append(pd.DataFrame(data=feats.values(),index=feats.keys()).T)
    df = pd.concat(dflist)
    df['total_area_true'] = [total_area]*len(df)
    df['area_true'] = df['area_true'].astype(int)
    df['total_area_true'] = df['total_area_true'].astype(int)
    df['label_true'] = df['label_true'].astype(int)
    return df

def extract_2d_features(label_image, intensity_image=None, suffix='_nucleus', prefix='2d_'):
    
    """
    extract 2d shape features from the label image

    Parameters
    ----------
    label_image : np.array
        the labeled image (2D) where each object has a unique label
    intensity_image : np.array
        the intensity image (2D) where each pixel has an intensity value

    Returns
    -------
    df : pd.DataFrame
        the dataframe containing the features
    """
    assert label_image.ndim == 2
    assert intensity_image is None or intensity_image.ndim == 2

    properties = [
        'label',
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
    ]
    if intensity_image is not None:
        properties = ['label','intensity_max','intensity_mean','intensity_min']
    prop_list = regionprops_table(label_image,
                                    intensity_image=intensity_image,
                                    properties=properties
                                    )
    
    df =  pd.DataFrame(prop_list)
    dft = compute_true_area(label_image)
    df = pd.merge(dft,df,left_on='label_true',right_on='label',how='left')
    df['img_shape'] = [str(label_image.shape)]*len(df) # add as a sanity check that the image size does not change when reading ZARRs
    df['label_img'] = df['label']
    # add prefix and suffix to all columns except label_img
    df.columns = [prefix + col + suffix if col not in ['label_img'] else col for col in df.columns]
    
    # drop rows where label_img is 0
    df = df[df['label_img']!=0]
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
    return df

def merge_2d_features(dfleft, dfright, suffixes=('_dup1','_dup2')):
    """
    merge the 2D features from the pseudo cell and nucleus images

    Parameters
    ----------
    dfleft : pd.DataFrame
        the dataframe to use as the left dataframe
    nucleus_features_df : pd.DataFrame
        the dataframe to use as the right dataframe

    Returns
    -------
    df_2d : pd.DataFrame
        the merged dataframe containing the 2D features from the pseudo cell and nucleus images
        (or cyto image)
    """
    # perform a left merge because we only want to match pseudo cells to true nuclei
    df_2d = pd.merge(dfleft, dfright, on=['colony','index_sequence','label_img'], suffixes=suffixes,how='left')
    # now perform a check that df_2d has the same number of rows as nucleus_features_df
    assert len(df_2d) == len(dfleft)
    return df_2d

def define_density_features(df_2d):
    """
    define density features in the 2D dataframe
    the first density feature (2d_area_nuc_cell_ratio) is defined as the area of a nucleus divided by the area of the (pseudo) cell
    the second density feature (inv_cyto_density) is defined as the inverse of the cytoplasmic area
        a sub feature is the cytoplasmic area (2d_area_cyto) which is defined as the difference between the pseudo cell area and the nucleus area

    Parameters
    ----------
    df_2d : pd.DataFrame
        the 2D dataframe containing the features from 2d labeled nucleus image and 2d watershed-based pseudo cell image

    Returns
    -------
    df_2d : pd.DataFrame
        the 2D dataframe with the density features added (2d_area_nuc_cell_ratio, inv_cyto_density)
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

def get_pseudo_cell_boundaries(labeled_nucleus_image, colony='test', timepoint=0, resolution_level=0, return_img_dict=False, return_2d_mip_only=False):
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
        basic_dict,img_dict = get_pseudo_cell_boundaries_from_labeled_nucleus_image(labeled_nucleus_image,return_img_dict=True)
    else:
        basic_dict = get_pseudo_cell_boundaries_from_labeled_nucleus_image(labeled_nucleus_image,return_nucleus=True)

    pseudo_cell_image = basic_dict['pseudo_cells_img']
    nucleus_image = basic_dict['nucleus_img']
    nucleus_edge_image = basic_dict['nucleus_edge_img']
    cyto_distance = basic_dict['cyto_distance']

    # extract features from 2D label image
    pseudo_cell_features_df = extract_2d_features(pseudo_cell_image,suffix='_pseudo_cell')
    nucleus_features_df = extract_2d_features(nucleus_image,suffix='_nucleus')
    edge_features_df = extract_2d_features(nucleus_edge_image,cyto_distance,suffix='_edge',prefix='2d_')

    # check that 'area' and 'area_true' are the same
    log1 = np.allclose(pseudo_cell_features_df['2d_area_pseudo_cell'],pseudo_cell_features_df['2d_area_true_pseudo_cell'])
    log2 = np.allclose(nucleus_features_df['2d_area_nucleus'],nucleus_features_df['2d_area_true_nucleus'])
    if (not log1) or (not log2):
        # save the images for debugging
        print('MISMATCH')
        savepath = Path(__file__).parent / 'debug' / '2d_area_mismatch' / f'imgs_{colony}_{timepoint}_{resolution_level}.npz'
        np.savez(savepath, pseudo_cell_image=pseudo_cell_image, nucleus_image=nucleus_image)
        print('saving here:',savepath)

    # add timepoint, colony, pixel_size, label_img to the dataframes
    pseudo_cell_features_df = add_metadata_to_df(pseudo_cell_features_df, colony, timepoint, resolution_level)
    nucleus_features_df = add_metadata_to_df(nucleus_features_df, colony, timepoint, resolution_level)
    edge_features_df = add_metadata_to_df(edge_features_df, colony, timepoint, resolution_level)

    # now merge the two dataframes
    df_2d = merge_2d_features(nucleus_features_df, pseudo_cell_features_df)
    df_2d = merge_2d_features(df_2d, edge_features_df)
   
    # define the density measure (2d_area_nuc_cell_ratio)
    df_2d = define_density_features(df_2d)

    # df_2d = choose_columns(df_2d)

    if return_img_dict:
        return df_2d, img_dict
    elif return_2d_mip_only:
        return df_2d, labeled_nucleus_image
    else:
        return df_2d