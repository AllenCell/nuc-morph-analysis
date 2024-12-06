#%%
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.lib.visualization.plotting_tools import plot_colorized_img_with_labels


def run_example(df:pd.DataFrame, colony:str = 'medium', timepoint:int = 48, frames:int = 10, resolution_level:int =1):
    """
    make a figure to display an image highlighting the the neighbors of mitotic cells at a specific timepoint

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe containing the data
    colony : str
        the colony to analyze
    timepoint : int
        the timepoint to analyze
    frames : int
        the number of frames to include before and after the timepoint
    resolution_level : int
        the resolution level to use for the images (OME-ZARR)

    Returns
    -------
    None

    Outputs
    -------
    Saves a figure to the figures/example_timepoint_neighbors_of_mitotic directory
    """
    dfc = df.loc[df['colony']==colony].copy()
    dfm = dfc.loc[(df['index_sequence']>=timepoint - frames) | (df['index_sequence']<=timepoint + frames)].copy()

    #%%
    # set figure directory
    figdir = Path(__file__).parent / "figures" / "example_timepoint_neighbors_of_mitotic"
    os.makedirs(figdir,exist_ok=True)

    # now load segmentation image at timepoint
    reader = load_data.get_dataset_segmentation_file_reader(colony)
    if resolution_level>0:
        reader.set_resolution_level(resolution_level)

    lazy_img = reader.get_image_dask_data("ZYX",T=timepoint)
    img= lazy_img.compute()

    dft = dfm[dfm['index_sequence']==timepoint]
    column_list = ['has_mitotic_neighbor_breakdown','has_mitotic_neighbor_formation','has_mitotic_neighbor','has_mitotic_neighbor_formation_backward_dilated','has_mitotic_neighbor_breakdown_forward_dilated','has_mitotic_neighbor_dilated','exiting_mitosis']
    # now plot the image with the mitotic neighbors
    for col in column_list:
        # colored_img = colorize_image(img.max(axis=0),dft,feature=f'{col}2')
        colormap_dict = {}
        colormap_dict.update({f"{col}_empty":(col,False,1,(0.4,0.4,0.4),f"")})
        colormap_dict.update({f"{col}":(col,True,2,(1,1,0),f"{col}")})
        colormap_dict.update({f"breakdown":('frame_of_breakdown',True,3,(1,0,1),f"breakdown event")})
        colormap_dict.update({f"formation":('frame_of_formation',True,4,(0,1,1),f"formation event")})


        fig,ax = plt.subplots(figsize=(5,5),layout='constrained')
        _ = plot_colorized_img_with_labels(ax,img,dft.copy(),colormap_dict)
        
        plt.title(f'neighbors of mitotic cells\n{col}')
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
