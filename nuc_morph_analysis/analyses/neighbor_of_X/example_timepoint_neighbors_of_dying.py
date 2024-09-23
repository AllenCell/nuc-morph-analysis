#%%
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.lib.visualization.plotting_tools import plot_colorized_img_with_labels

colony = 'medium'
def run_example(df:pd.DataFrame, colony:str = 'medium', timepoint = None, resolution_level:int =1):
    """
    create an image showing all neighbors of dying cells at a given timepoint for a specific colony
    use default track_id = 87124 to find the timepoint

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe containing the data
    colony : str
        the colony to analyze
    timepoint : int or None
        the timepoint to analyze
    resolution_level : int
        the resolution level to use for the images (OME-ZARR)

    Returns
    -------
    None

    Outputs
    -------
    Saves a figure to the figures/example_timepoint_neighbors_of_dying directory
    """
    dfc = df.loc[df['colony']==colony].copy()
    if timepoint is None:
        track_id = 87124
        timepoint = int(dfc.loc[dfc['track_id']==track_id,'identified_death'].values[0])

    dfm = dfc.loc[(dfc['index_sequence']==timepoint)].copy()
    #%%
    # set figure directory
    figdir = Path(__file__).parent / "figures" / "example_timepoint_neighbors_of_dying"
    os.makedirs(figdir,exist_ok=True)

    # load the segmentation image
    reader = load_data.get_dataset_segmentation_file_reader(colony)
    if resolution_level>0:
        reader.set_resolution_level(resolution_level)

    lazy_img = reader.get_image_dask_data("ZYX",T=timepoint)
    img= lazy_img.compute()

    dft = dfm[dfm['index_sequence']==timepoint]

    column_list = ['has_dying_neighbor']
    # now plot the image with the mitotic neighbors
    for col in column_list:
        dft[f'{col}2'] = dft[f'{col}'] +1 

        colormap_dict = {}
        colormap_dict.update({f"{col}_empty":(col,False,1,(0.4,0.4,0.4),f"")})
        colormap_dict.update({f"{col}":(col,True,2,(1,1,0),f"{col}")})
        colormap_dict.update({f"death":('frame_of_death',True,3,(1,0,1),f"death event")})

        fig,ax = plt.subplots(figsize=(5,5),layout='constrained')
        _ = plot_colorized_img_with_labels(ax,img,dft.copy(),colormap_dict)

        plt.title(f'neighbors of dying cells\n{col}\nt={timepoint}')
        plt.axis('off')

        savename = figdir / f'{colony}-{timepoint}-{col}_neighbors.png'
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