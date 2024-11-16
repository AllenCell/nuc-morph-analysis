#%%
import os
import numpy as np
import skimage.exposure as skex
from aicsimageio.writers import two_d_writer
from nuc_morph_analysis.lib.preprocessing.single_track_contact.export_code import export_helper
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, load_data

# %%
def save_colony_backdrop_mips(colony, figdir="data/tfe_backdrop/",dtype ="uint8"):
    """
    Save maximum intensity projection (MIP) images for a given colony.

    This function reads the raw field of view images for the specified colony. The images are
    processed to flip them to match the orientation of the segmentation images on TFE, intensity 
    rescaled to make visualization easier and max intensity projections in z are saved as TIFF files
    to a directory at the top level of the repository.
    
    Images are processed using the same method as Figure 1. 

    Parameters:
    -----------
    colony (str): The name of the colony to process.
    figdir (str): The directory where the MIP images will be saved. Default is "data/tfe_backdrop/".
    dtype (str): The data type to use for the images. Default is "uint8".

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

        repo_top_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        save_path = os.path.join(repo_top_dir, figdir, colony)
        os.makedirs(save_path, exist_ok=True)
        save_image = os.path.join(save_path, f'{timepoint_frame}.tiff')
        
        two_d_writer.TwoDWriter.save(egfp_mip_rescale, uri=save_image)

def add_backdrop_path_to_dataframe(df, figdir):
    df['tfe_backdrop_path'] = figdir + df['colony'] + '/' + df['index_sequence'].astype(str) + '.tiff'    
    return df

# %% slow takes ~1hr because multiprocessing doesnt work with the  s3
for colony in ['small', 'medium', 'large']:
    save_colony_backdrop_mips(colony)

# %%
df = global_dataset_filtering.load_dataset_with_features()
df2 = add_backdrop_path_to_dataframe(df, figdir="data/tfe_backdrop/")
# %%

