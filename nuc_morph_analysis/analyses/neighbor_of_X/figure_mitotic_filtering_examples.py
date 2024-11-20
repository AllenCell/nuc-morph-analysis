#%%
#SuppFigS4 panel D, this code takes ~6 min to run
from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import watershed_workflow, pseudo_cell_helper, pseudo_cell_testing_helper
from pathlib import Path
import pandas as pd
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
from nuc_morph_analysis.lib.preprocessing import filter_data, load_data
from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from nuc_morph_analysis.lib.visualization.plotting_tools import colorize_image, get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot

from nuc_morph_analysis.analyses.dataset_images_for_figures.figure_helper import return_glasbey_on_dark


from nuc_morph_analysis.analyses.density.visually_validate_watershed_psuedo_cell_seg_workflow import get_contours_from_pair_of_2d_seg_image, draw_contours_on_image
from nuc_morph_analysis.analyses.dataset_images_for_figures.figure_helper import INTENSITIES_DICT
from tqdm import tqdm


import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 7

def determine_crop_size_and_location_from_track(track,df,crop_w=250,crop_h=250,RESOLUTION_LEVEL=1):
    """
    determine the crop size and location based on the track_id
    """
    crop_w = 250
    crop_h = 250
    track_df = df[(df['track_id']==track)]
    track_x = track_df['centroid_x'].values[0]
    track_y = track_df['centroid_y'].values[0]
    if RESOLUTION_LEVEL ==1:
        track_x = np.uint16(track_x//2.5)
        track_y = np.uint16(track_y//2.5)

    track_x = np.uint16(track_x - crop_w//2)
    track_y = np.uint16(track_y - crop_h//2)
    return track_x,track_y,crop_w,crop_h

def determine_colormaps(img,key,crop_exp):
    # determine colormaps and vmin,vmax
    if img.dtype == 'bool':
        cmap = cm.get_cmap('Greys_r')
        vmin = 0
        vmax = 1
    elif img.dtype == 'uint16':
        _, cmap, _ = return_glasbey_on_dark(N=img.max()+1,from_list=True)
        vmax = cmap.N-1
        vmin = 0
    else:
        cmap = cm.get_cmap('Greys_r')
        vmin=0
        vmax=np.max(img[crop_exp])

    if key=='raw_image':
        cmap = cm.get_cmap('Greys_r')
        vmin = INTENSITIES_DICT['egfp_max'][0]
        vmax= INTENSITIES_DICT['egfp_max'][1]
    return cmap, vmin, vmax

def run_validation_and_plot(track_id=87135,RESOLUTION_LEVEL=1,frames_before=1,frames_after=7,w=300,plot_everything=False):
    """
    run an image through the watershed based pseudo cell segmentation and examine the outputs
    optionally, run a test image through the same pipeline

    Parameters
    ----------
    track_id : int, optional
        The track to analyze, by default 87135
    RESOLUTION_LEVEL : int, optional
        The resolution level to analyze, by default 1
    plot_everything : bool, optional
        Whether to plot a large set of extra images colored by features and with contours, by default True
    testing : bool, optional
        Whether to run a test image through the pipeline, by default False

    Returns
    -------
    pd.DataFrame
        The dataframe containing the pseudo cell segmentation results
        if plot_everything is False
        if plot_everything is True, returns the full dataframe (after merging with the tracking dataset)

    """


   # load the tracking dataframe and apply appropriate filters
    df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline')
    # df = filter_data.all_timepoints_minimal_filtering(df)
    
    dftrack = df.loc[df['track_id']==track_id].copy()
    timepoint = int(dftrack['predicted_breakdown'].values[0])
    time_list = np.arange(timepoint-frames_before,timepoint+frames_after+1,dtype='uint16')
    
    colony = dftrack['colony'].values[0]
    dfc = df.loc[df['colony']==colony].copy()
    dfm = dfc.loc[(dfc['index_sequence'].isin(time_list))].copy()

    # set figure directory
    figdir = Path(__file__).parent / "figures" / "SuppFigS4_mitotic_filtering"
    figdir.mkdir(exist_ok=True,parents=True)

    # artificially set all nuclei to have predicted_breakdown and predicted_formation to -1
    dft = dfm[dfm['index_sequence']==timepoint]
    x1,y1,w,h = determine_crop_size_and_location_from_track(track_id,dft,crop_w=w,crop_h=w,RESOLUTION_LEVEL=1)

    nrows = 2
    ncols = len(time_list)
    assert ncols>1
    fig,axr = plt.subplots(nrows,ncols,figsize=(ncols*1.5,nrows*1.5))
    # remove x and y ticks
    [ax.axis('off') for ax in axr.flatten()] 

    for ti,tval in enumerate(tqdm(time_list)):
        print(ti)
        # perform watershed based pseudo cell segmentation
        _, img_dict = watershed_workflow.get_image_and_run(colony, tval, RESOLUTION_LEVEL, return_img_dict=True)

        # load the raw image and add to image dict
        raw_img_reader = load_data.get_dataset_original_file_reader(colony)
        raw_img = raw_img_reader.get_image_dask_data("ZYX", T=tval, C=0).max(axis=0).compute()    
        img_dict['raw_image'] = (raw_img,'mEGFP-tagged lamin B1')

        # now get the subset of the dataframe for the colony and timepoint
        dfsub = dfm[dfm['colony']==colony]
        dft = dfsub[dfsub['index_sequence']==tval]

        # now set the crop for the image
        crop_exp = np.index_exp[y1:y1+h,x1:x1+w]

        for rowi, key in enumerate(['colorize','raw_image']):
            ax = axr[rowi,ti] # top row axis
            assert type(ax) is plt.Axes
            if key in ['overlay','colorize']:
                img = np.zeros_like(img_dict['mip_of_labeled_image'][0])
                cell_img = img_dict['pseudo_cells_img'][0].copy()
                cell_img[img_dict['cell_shed_bin'][0]==0] = 0 # remove the edges
                contour_list = get_contours_from_pair_of_2d_seg_image(img_dict['mip_of_labeled_image'][0],cell_img,)
                label = f"{(tval//12)}:{str((tval%12)*5).zfill(2)} hr:min"
            else:
                img = img_dict[key][0]
                label = ""

            imgcmap, vmin, vmax = determine_colormaps(img,key,crop_exp)
            ax.imshow(img,
                        interpolation='nearest',
                        cmap = imgcmap,
                        vmin = vmin,
                        vmax = vmax,
                        origin='lower',
            )

            # adjust axes
            ax.set_title(label)
            ax.axis('off')
            ax.set_xlim([x1,x1+w])
            ax.set_ylim([y1,y1+h])

            if key in ['overlay','colorize']:
                labels_in_img = np.unique(img_dict['pseudo_cells_img'][0][crop_exp])
                dftsub = dft[dft['label_img'].isin(labels_in_img)].copy()


                colormap_dict = {} #type:ignore
                colormap_dict.update({'nothing':('frame_of_breakdown',False,1,(0.4,0.4,0.4),f"")}) 
                colormap_dict.update({'has_mitotic_neighbor_breakdown_dilated':('has_mitotic_neighbor_breakdown_forward_dilated',True,3,(0.8,0,0),f"has mitotic neighbor (forward)")})
                colormap_dict.update({'has_mitotic_neighbor_breakdown':('has_mitotic_neighbor_breakdown',True,4,(1.0,0.0,1.0),f"has mitotic neighbor")})
                colormap_dict.update({'track':('track_id',track_id,2,(0.0,1.0,0.0),f"cell that will divide")}) #type:ignore
                colormap_dict.update({'frame_of_breakdown':('frame_of_breakdown',True,8,(1.0,1.0,0.0),f"breakdown event")})
                # now update colors in contour_list based on colormap_dict
                # contour_list.append((label_img,nuc_contours,cell_contours,color))

                new_colors = np.zeros((np.max([x[2] for x in colormap_dict.values()])+1,3))
                for col in colormap_dict.keys():
                    new_colors[colormap_dict[col][2]] = colormap_dict[col][3]
                # attach alpha values
                new_colors = np.concatenate([new_colors,np.ones((new_colors.shape[0],1))],axis=1)
                newcmap = ListedColormap(new_colors)

                dftsub['color_col'] = 0
                for col in colormap_dict.keys():
                    dftsub.loc[dftsub[colormap_dict[col][0]]==colormap_dict[col][1], 'color_col'] = colormap_dict[col][2]

                # now draw the contours
                colorfeat = 'color_col'
                draw_contours_on_image(ax,contour_list,filled=True,colorize=False,dft=dftsub,linewidth=0.5,colorfeat=colorfeat,usedfcolors=True,cmap=newcmap)
    # now save the figure
    savedir = Path(__file__).parent / 'figures' / 'suppfigs4_mitotic_removal_figure_illustration'
    savedir.mkdir(exist_ok=True,parents=True)
        
    for ext in ['.png','.pdf']:
        savename = f'{colony}_{track_id}_{tval}_res{RESOLUTION_LEVEL}'
        savepath = savedir / savename
        print(f'Saved figure to {savepath}')
        save_and_show_plot(str(savepath),
                            file_extension=ext,
                            figure=fig,
                            transparent=True,
                            keep_open=True,
                            **{'dpi':600}
                        )

#%%
if __name__ == '__main__':
    dft1 = run_validation_and_plot()

