#%%
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
import numpy as np
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.lib.preprocessing import labeling_neighbors_helper
from nuc_morph_analysis.lib.visualization.plotting_tools import plot_colorized_img_with_labels


def run_example(df:pd.DataFrame, colony:str = 'medium', track_id:int =87124, frames_before:int=3, frames_after:int=3, resolution_level:int=1):
    """
    this function will plot a montage of images to show the neighbors of a dying track at the timepoint of death of a specific track_id

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe containing the data
    colony : str
        the colony to analyze
    track_id : int
        the track_id of the cell that will die
    frames_before : int
        the number of frames before the timepoint of death
    frames_after : int
        the number of frames after the timepoint of death
    resolution_level : int
        the resolution level to use for the images (OME-ZARR)
    
    Returns
    -------
    None

    Outputs
    -------
    Saves a figure to the figures/neighbors_of_dying_track directory
    """
    timepoint = int(df.loc[df['track_id']==track_id,'predicted_breakdown'].values[0])
    time_list = np.arange(timepoint-frames_before,timepoint+frames_after+1,dtype='uint16')

    dfc = df.loc[df['colony']==colony].copy()
    dfm = dfc.loc[(dfc['index_sequence'].isin(time_list))].copy()

    # recompute the features for this timepoint
    dfm.drop(columns=['has_dying_neighbor','has_dying_neighbor_forward_dilated','frame_of_death'],inplace=True)
    dfm = labeling_neighbors_helper.label_nuclei_that_neighbor_current_death_event(dfm)

    # set figure directory
    figdir = Path(__file__).parent / "figures" / "neighbors_of_dying_track"
    print(figdir)
    os.makedirs(figdir,exist_ok=True)

    # load the segmentation image
    reader = load_data.get_dataset_segmentation_file_reader(colony)
    if resolution_level>0:
        reader.set_resolution_level(resolution_level)

    dft = dfc[dfc['index_sequence']==timepoint]
    y,x = dft.loc[dft['track_id']==track_id,['centroid_y','centroid_x']].values[0]
    w=500
    if resolution_level==1:
        y = np.round(y/2.5).astype(int)
        x = np.round(x/2.5).astype(int)
        w = np.round(w/2.5).astype(int)

    crop_exp = np.index_exp[:,y-w:y+w,x-w:x+w]
    nrows = 2
    ncols = np.ceil(len(time_list)/nrows).astype(int)
    fig,ax = plt.subplots(nrows,ncols,figsize=(ncols*2.5,nrows*2.5), 
                        #   layout='constrained',
                        )
    axx = np.asarray([ax]).flatten()
    for ti,t in enumerate(time_list):
        current_ax = axx[ti]
        assert type(current_ax) is plt.Axes
        lazy_img = reader.get_image_dask_data("ZYX",T=t)
        img= lazy_img.compute()[crop_exp]

        dft = dfm[dfm['index_sequence']==t].copy()

        colormap_dict = {} #type:ignore
        cmap1 = plt.get_cmap('Dark2_r')
        colormap_dict.update({'nothing':('has_dying_neighbor',False,1,(0.4,0.4,0.4),f"")})
        colormap_dict.update({'track':('track_id',track_id,2,(0.5,0.0,0.0),f"cell that will die")}) #type:ignore
        colormap_dict.update({'frame_of_death':('frame_of_death',True,8,(1.0,0.0,0.5),f"moment of death")})
        colormap_dict.update({'neighbors_dying_cell_forward':('has_dying_neighbor_forward_dilated',True,3,(1,1,0),f"has dying neighbor (forward)")})
        colormap_dict.update({'has_dying_neighbor':('has_dying_neighbor',True,4,(0,1,0),f"has dying neighbor")})

        show_legend=True if ti==len(time_list)-1 else False
        current_ax = plot_colorized_img_with_labels(current_ax,img,dft,colormap_dict,show_legend=show_legend)
        current_ax.set_title(f't={t}')

    # set axis.off for all axes
    for curr_ax in axx:
        assert type(curr_ax) is plt.Axes
        curr_ax.axis('off')

    savename = f"{colony}-{track_id}-t={str(time_list)}"
    savepath = str(figdir / savename)
    save_and_show_plot(savepath,file_extension='.png',figure=fig,
                    transparent=False)
    plt.show()

if __name__ == "__main__":
    df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline')
    run_example(df)