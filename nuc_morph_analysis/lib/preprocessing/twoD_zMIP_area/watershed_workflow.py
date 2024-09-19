from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import pseudo_cell_helper
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from multiprocessing import get_context, cpu_count

def get_image_and_run(colony,timepoint,resolution_level,return_img_dict=False):
    reader = load_data.get_dataset_segmentation_file_reader(colony)
    if resolution_level>0:
        reader.set_resolution_level(resolution_level)
    img = reader.get_image_dask_data("ZYX", T=timepoint, C=0).compute()
    del reader # close the reader so it doesn't have pickling issues in multiprocessing
    return pseudo_cell_helper.get_pseudo_cell_boundaries(img,colony, timepoint, resolution_level, return_img_dict=return_img_dict)

def process_timepoint(args):
    timepoint, colony, resolution_level = args
    return get_image_and_run(colony,timepoint,resolution_level)

def get_pseudo_cell_boundaries_for_movie(colony, resolution_level=1, output_directory=None, parallel=False, save_df=False, testing=False):
    """
    function for returning the pseudo cell boundaries at all timepoints
    the psuedo cell boundary is a watershed segmentation of the max projection of the segmentation image
    this is similar to a Voronoi tesselation output except it is determined using the nucleus boundary
    rather than just the centroid, therefore giving more accurate results

    Parameters
    ----------
    colony : str
        the colony name
    resolution_level : int
        the resolution level to load from the OME-ZARR, default is 1
        (0 is full, 1 is 2.5x downsampled...equivalent to 20x image size)
     output_directory : str
        the output directory to save the pseudo cell images
    parallel : bool
        whether to run the analysis in parallel or not
        NOTE: currently does not work when reading S3 files
    save_df : bool
        whether to save the pseudo cell boundaries as a csv, default is False
    testing : bool
        whether to run the function in testing mode, default is False
        in testing mode only 20 frames are run
    """
    # load the segmentation iamge
    reader = load_data.get_dataset_segmentation_file_reader(colony)
    args = [(timepoint, colony, resolution_level) for timepoint in range(reader.dims.T)]
    del reader # close the reader so it doesn't have pickling issues in multiprocessing

    nworkers = cpu_count()
    if testing:
        args = args[0:nworkers*2]

    if parallel==False:
        dflist = []
        for ai in tqdm(range(len(args)), desc="Processing timepoints"):
            df_2d = process_timepoint(args[ai])
            dflist.append(df_2d)
    else:
        step = nworkers*2
        arg_set = [args[ai:ai+step] for ai in range(0,len(args),step)]
        print("number of sets to process:",len(arg_set))
        print("step size:",step)
        dflist_list = []
        # for arg_subset in tqdm(arg_set, desc="Processing timepoints"):
        ai = 0
        while ai < len(arg_set):
            # code is set up this way to retry if encountrering aiohttp.client_exceptions.ServerDisconnectedError: Server disconnected
            try:
                arg_subset = arg_set[ai]
                with get_context('spawn').Pool(nworkers) as p:
                    dflist = list(tqdm(p.imap_unordered(process_timepoint, arg_subset), initial= ai*step, total=len(args), desc="Processing timepoints",leave=False))
                    dflist_list.extend(dflist)
                ai += 1
            
            # catch exceptions and try again (kill if keyboard interrupt)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    print("Keyboard interrupt detected, killing pool")
                    p.terminate()
                    p.join()
                    break
                else:
                    p.terminate()
                    p.join()
                print("Exception detected, trying again")

    # concatenate the dataframe
    df = pd.concat(dflist_list)

    if save_df:
        # save the dataframe as a parquet
        df.to_parquet(output_directory / f"{colony}_pseudo_cell_boundaries.parquet")
        print('saved pseudo cell boundaries to:', output_directory / f"{colony}_pseudo_cell_boundaries.parquet")

    return df

if __name__ == "__main__":
    for colony in load_data.get_dataset_names(dataset='all_baseline'):
        print(colony)
        output_directory = Path(__file__).parents[6] / "local_storage" / "pseudo_cell_boundaries"
        resolution_level = 1 # default is to run analysis using the 2.5x downsampled images for speed
        df = get_pseudo_cell_boundaries_for_movie(colony,
                                              resolution_level,
                                                output_directory,
                                                parallel=True,
                                                save_df=True,
                                                testing=False)