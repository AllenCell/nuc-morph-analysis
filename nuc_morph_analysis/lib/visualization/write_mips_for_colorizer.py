#%%
import os
import numpy as np
import skimage.exposure as skex
from skimage.transform import resize
from aicsimageio.writers import two_d_writer
from nuc_morph_analysis.lib.preprocessing.single_track_contact.export_code import export_helper
from nuc_morph_analysis.lib.preprocessing import load_data

# %%
def save_colony_backdrop_mips(colony, figdir="./data/tfe_backdrop/", dtype ="uint8", downsample_factor=2):
    """
    Save maximum intensity projection (MIP) images for a given colony.

    This function reads the raw field of view images for the specified colony. The images are
    processed to flip them to match the orientation of the segmentation images on TFE, intensity 
    rescaled, downsampled and max intensity projections in z are saved as png files to a directory.

    Parameters:
    -----------
    colony (str): The name of the colony to process.
    figdir (str): The directory where the MIP images will be saved. Default is "data/tfe_backdrop/".
    dtype (str): The data type to use for the images. Default is "uint8".
    downsample_factor (int): The factor by which to downsample the images. Default is 2.

    Returns:
    --------
    None
    """
    reader = load_data.get_dataset_original_file_reader(colony)
    for timepoint_frame in range(0, reader.dims.T, 1):
        egfp = export_helper.load_raw_fov_image(colony, timepoint_frame, reader, channel="egfp")
        egfp_flip = np.flip(egfp, axis=1) 
        egfp_mip = egfp_flip.max(axis=0).astype(dtype)
        egfp_mip_rescale = skex.rescale_intensity(image=egfp_mip, in_range=(110, 140), out_range=dtype).astype(dtype)
        
        downsample_shape = (egfp_mip_rescale.shape[0] // downsample_factor, egfp_mip_rescale.shape[1] // downsample_factor)
        egfp_mip_downsampled = resize(egfp_mip_rescale, downsample_shape, anti_aliasing=True, preserve_range=True).astype(dtype)
              
        save_path = os.path.join(figdir, colony)
        os.makedirs(save_path, exist_ok=True)
        save_image = os.path.join(save_path, f'{timepoint_frame}.png')
        two_d_writer.TwoDWriter.save(egfp_mip_downsampled, uri=save_image)