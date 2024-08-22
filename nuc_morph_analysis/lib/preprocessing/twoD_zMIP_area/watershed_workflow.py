from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import pseudo_cell_helper
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
from time import time

def process_timepoint(args):
    timepoint, colony, reader, resolution_level = args
    df_2d = pseudo_cell_helper.get_pseudo_cell_boundaries(colony, timepoint, reader, resolution_level)
    return df_2d

def get_pseudo_cell_boundaries_for_movie(colony, output_directory, resolution_level=0, parallel=False):
    """
    function for returning the pseudo cell boundaries at all timepoints
    the psuedo cell boundary is a watershed segmentation of the max projection of the segmentation image
    this is similar to a Voronoi tesselation output except it is determined using the nucleus boundary
    rather than just the centroid, therefore giving more accurate results

    Parameters
    ----------
    colony : str
        the colony name
    output_directory : str
        the output directory to save the pseudo cell images
    resolution_level : int
        the resolution level to load from the OME-ZARR (0 is full, 1 is 2.5x downsampled...equivalent to 20x image size)
    """
    # load the segmentation iamge
    reader = load_data.get_dataset_segmentation_file_reader(colony)
    if resolution_level>0:
        reader.set_resolution_level(resolution_level)

    if parallel==False:
        dflist = []
        for timepoint in tqdm(range(reader.dims.T), desc="Processing timepoints"):
            df_2d = process_timepoint([timepoint, colony, reader, resolution_level])
            dflist.append(df_2d)
    else:
        with Pool(cpu_count()) as p:
            dflist = list(tqdm(p.imap_unordered(process_timepoint, [(timepoint, colony, reader, resolution_level) for timepoint in range(reader.dims.T)]), total=reader.dims.T, desc="Processing timepoints"))
        
    # concatenate the dataframe
    df = pd.concat(dflist)

    # save the dataframe as a parquet
    df.to_parquet(output_directory / f"{colony}_pseudo_cell_boundaries.parquet")
    print('saved pseudo cell boundaries to:', output_directory / f"{colony}_pseudo_cell_boundaries.parquet")

if __name__ == "__main__":
    t1 = time()
    for colony in load_data.get_dataset_names(dataset='all_baseline'):
        print(colony)
        output_directory = Path(__file__).parents[6] / "local_storage" / "pseudo_cell_boundaries"
        resolution_level = 1 # default is to run analysis using the 2.5x downsampled images for speed
        get_pseudo_cell_boundaries_for_movie(colony, output_directory, resolution_level)
    t2 = time()
    print('total time:', t2-t1)