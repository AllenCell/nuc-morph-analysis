#%%
import os
import numpy as np
import skimage.exposure as skex
from skimage.transform import resize
from aicsimageio.writers import two_d_writer
from nuc_morph_analysis.lib.preprocessing.single_track_contact.export_code import export_helper
from nuc_morph_analysis.lib.preprocessing import load_data

# %%
def save_colony_backdrop_mips(colony, figdir, dtype ="uint8", downsample_factor=2, zarr_res_level=0):
    """
    Save maximum intensity projection (MIP) images for a given colony.

    This function reads the raw field of view images for the specified colony. The images are
    processed to flip them to match the orientation of the segmentation images on TFE, intensity 
    rescaled, downsampled and max intensity projections in z are saved as png files to a directory.
    
    Two downsampling methods are available. 
    
    1. Load directly a precomputed lower resolution zarr. Options are 0, 1, 2, 3, 4. 
    2. Resize the image using skimage and downsample the image using a custom scaling factor. 
    
    The default behavior is to resize the image using skimage. This is the same method used to generate
    MIPs in Figure 1. 

    Parameters:
    -----------
    colony (str): The name of the colony to process.
    figdir (str): The directory where the MIP images will be saved.
    dtype (str): The data type to use for the images. Default is "uint8".
    downsample_factor (int): The factor by which to downsample the images. Default is 2. 
    zarr_res_level (int): The resolution level to use when reading the zarr files. Default is 0. 

    Returns:
    --------
    None
    """
    print(f"Processing {colony} colony backdrops")
    reader = load_data.get_dataset_original_file_reader(colony)
    reader.set_resolution_level(zarr_res_level)
        
    for timepoint_frame in range(reader.dims.T):
        egfp = export_helper.load_raw_fov_image(colony, timepoint_frame, reader, channel="egfp")
        egfp_flip = np.flip(egfp, axis=1)
        egfp_mip = egfp_flip.max(axis=0).astype(dtype)
        
        if downsample_factor is not None:
            downsample_shape = (egfp_mip.shape[0] // downsample_factor, egfp_mip.shape[1] // downsample_factor)
            egfp_mip = resize(egfp_mip, downsample_shape, anti_aliasing=False, preserve_range=True, order=1).astype(dtype)
        
        egfp_mip = skex.rescale_intensity(image=egfp_mip, in_range=(110, 140), out_range=dtype).astype(dtype)
        
        os.makedirs(figdir, exist_ok=True)
        save_image = os.path.join(figdir, f'{timepoint_frame}.png')
        two_d_writer.TwoDWriter.save(egfp_mip, uri=save_image)