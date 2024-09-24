#%%
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.lib.visualization.plotting_tools import plot_colorized_img_with_labels

# TEMP: loading local for testing and speed

# for testing only use a subset of timepoints

def run_example(df:pd.DataFrame, colony:str = 'medium', timepoint:int = 57, resolution_level:int =1):
    """
    this code will plot the number of mitotic neighbors for cells at a specific timepoint

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe containing the data
    colony : str
        the colony to analyze
    timepoint : int
        the timepoint to analyze
    resolution_level : int
        the resolution level to use for the images (OME-ZARR)

    Returns
    -------
    None

    Outputs
    -------
    Saves a figure to the figures/example_timepoint_numbers_of_mitotic directory
    """
    # set figure directory
    figdir = Path(__file__).parent / "figures" / "example_timepoint_numbers_of_mitotic"
    os.makedirs(figdir,exist_ok=True)

    # color by number of mitotic neighbors
    # now plot the image with the mitotic neighbors
    reader = load_data.get_dataset_segmentation_file_reader(colony)
    if resolution_level>0:
        reader.set_resolution_level(resolution_level)

    lazy_img = reader.get_image_dask_data("ZYX",T=timepoint)
    img= lazy_img.compute()
    dfm = df.loc[df['colony']==colony].copy()
    dft = dfm[dfm['index_sequence']==timepoint]
    colormap_dict = {}
    cmap1 = plt.get_cmap('Dark2_r')
    col = 'number_of_frame_of_breakdown_neighbors'
    colormap_dict.update({f'{col}_{i}':(col,i,i+1,cmap1.colors[i],f"{i} mitotic neighbors") for i in range(dft[col].max()+1)}) # type: ignore
    colormap_dict.update({'frame_of_breakdown':('frame_of_breakdown',True,8,(1,0,0),f"breakdown event")})
    fig,ax = plt.subplots(figsize=(5,5),layout='constrained')
    _ = plot_colorized_img_with_labels(ax,img,dft,colormap_dict)
    savename = figdir / f'{colony}-{timepoint}-{col}_number_of_neighbors.png'
    savepath = figdir / savename
    save_and_show_plot(savepath.as_posix(),
                        file_extension='.png',
                        figure=fig,
                        transparent=False,
    )
    plt.show()

if __name__ == "__main__":
    df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline')
    run_example(df)