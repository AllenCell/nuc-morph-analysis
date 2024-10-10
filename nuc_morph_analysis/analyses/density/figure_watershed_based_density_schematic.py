#%%
from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import watershed_workflow, pseudo_cell_helper, pseudo_cell_testing_helper
from pathlib import Path
import pandas as pd
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
from nuc_morph_analysis.lib.preprocessing import filter_data, load_data

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from nuc_morph_analysis.lib.visualization.plotting_tools import colorize_image, get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot

from nuc_morph_analysis.analyses.dataset_images_for_figures.figure_helper import return_glasbey_on_dark


from nuc_morph_analysis.analyses.density.watershed_validate import get_contours_from_pair_of_2d_seg_image, draw_contours_on_image
from nuc_morph_analysis.analyses.dataset_images_for_figures.figure_helper import INTENSITIES_DICT

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


def run_validation_and_plot(TIMEPOINT=48,track=84103,colony='medium',RESOLUTION_LEVEL=1,plot_everything=False, testing=False):
    """
    run an image through the watershed based pseudo cell segmentation and examine the outputs
    optionally, run a test image through the same pipeline

    Parameters
    ----------
    TIMEPOINT : int, optional
        The timepoint to analyze, by default 48
    track : int, optional
        The track to analyze, by default 84103
    colony : str, optional
        The colony to analyze, by default 'medium'
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

    # perform watershed based pseudo cell segmentation
    if testing: # on test data
        labeled_nucleus_image = pseudo_cell_testing_helper.make_nucleus_image_array()
        df_2d, img_dict = pseudo_cell_helper.get_pseudo_cell_boundaries(labeled_nucleus_image, return_img_dict=True)
        colony = 'test'
    else: # or on real data
        df_2d, img_dict = watershed_workflow.get_image_and_run(colony, TIMEPOINT, RESOLUTION_LEVEL, return_img_dict=True)

    # load the raw image and add to image dict
    raw_img_reader = load_data.get_dataset_original_file_reader(colony)
    raw_img = raw_img_reader.get_image_dask_data("ZYX", T=TIMEPOINT, C=0).max(axis=0).compute()    
    img_dict['raw_image'] = (raw_img,'mEGFP-tagged lamin B1')

    # load the tracking dataframe and apply appropriate filters
    df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline')
    df = filter_data.all_timepoints_minimal_filtering(df)
    dfm = df.copy()

    # now get the subset of the dataframe for the colony and timepoint
    dfsub = dfm[dfm['colony']==colony]
    dft = dfsub[dfsub['index_sequence']==TIMEPOINT]

    track_x,track_y,crop_w,crop_h = determine_crop_size_and_location_from_track(track,dft,crop_w=250,crop_h=250,RESOLUTION_LEVEL=1)

    mip = img_dict['mip_of_labeled_image'][0] #use mip to determine full size crop

    # now iterate through crop sizes
    for full_crop, sizes in [('crop',(track_x,track_y,crop_w,crop_h)),('full',(0,0,mip.shape[1],mip.shape[0]))]:
        x1,y1,w,h = sizes
        crop_exp = np.index_exp[y1:y1+h,x1:x1+w]
        key_list = ['raw_image','mip_of_labeled_image','binarized_mip','distance','pseudo_cells_img','overlay','colorize']
        
        nrows = 1
        ncols = len(key_list)
        assert ncols>1
        fig,axr = plt.subplots(nrows,ncols,figsize=(ncols*1.5,nrows*1.5))
        axx = np.asarray([axr]).flatten()

        for i, key in enumerate(key_list):
            ax = axx[i]
            assert type(ax) is plt.Axes
            
            if key in ['overlay','colorize']:
                img = np.zeros_like(img_dict['mip_of_labeled_image'][0])
                cell_img = img_dict['pseudo_cells_img'][0].copy()
                cell_img[img_dict['cell_shed_bin'][0]==0] = 0 # remove the edges
                contour_list = get_contours_from_pair_of_2d_seg_image(img_dict['mip_of_labeled_image'][0],cell_img,)
                label = 'Cell and Nucleus Boundaries' if key=='overlay' else 'Colored by Nucleus\nto Cell Area Ratio'
            else:
                img = img_dict[key][0]
                label = img_dict[key][1]
            
            imgcmap, vmin, vmax = determine_colormaps(img,key,crop_exp)
            
            mappable = ax.imshow(img,
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


            position = ax.get_position().bounds
            if key == 'overlay':
                draw_contours_on_image(ax,contour_list,filled=True,linewidth=0.5)
            if key == 'colorize':
                labels_in_img = np.unique(img_dict['pseudo_cells_img'][0][crop_exp])
                dftsub = dft[dft['label_img'].isin(labels_in_img)]
                colorfeat='2d_area_nuc_cell_ratio' #feature
                cmapstr = 'cool' # colormap
                color_values = dftsub[colorfeat].values
                
                draw_contours_on_image(ax,contour_list,filled=True,colorize=True,dft=dftsub,linewidth=0.5,colorfeat=colorfeat,cmapstr=cmapstr)

                colorbarmin = np.nanmin(color_values)
                colorbarmax = np.nanmax(color_values)
                colorbar = plt.cm.ScalarMappable(cmap=cmapstr,norm=plt.Normalize(vmin=colorbarmin,vmax=colorbarmax))
                colorbar.set_array([])
                colorbar.set_clim(colorbarmin,colorbarmax)
                cbar = plt.colorbar(colorbar,ax=ax,location='bottom')
                _,label,unit,_ = get_plot_labels_for_metric(colorfeat)
                cbar.set_label(f"{label} {unit}")
                cbar.set_ticks([colorbarmin,colorbarmax])
                cbar.set_ticklabels([f'{colorbarmin:.2f}',f'{colorbarmax:.2f}'])
                
            if key in ['distance']:

                distance_scale = 0.108 # pixel size
                if RESOLUTION_LEVEL == 1:
                    distance_scale *= 2.5
                colorbar = plt.cm.ScalarMappable(cmap=imgcmap,norm=plt.Normalize(vmin=vmin,vmax=vmax))
                cbar = plt.colorbar(colorbar,ax=ax,location='bottom')
                cbar.set_ticks([vmin,vmax])
                cbar.set_ticklabels([f'{vmin*distance_scale:.2f}',f'{vmax*distance_scale:.2f}'])
                cbar.set_label('Distance Transform (µm)')
            
            if key in ['distance','colorize']:
                # adjust the axis back to its original position
                ax.set_position(position)
                # adjust position of colorbar
                cbar_position = cbar.ax.get_position()
                # cbar.ax.set_position([cbar_position.x0+0.05,cbar_position.y0,cbar_position.width,cbar_position.height]) # for left location
                cbar.ax.set_position([cbar_position.x0,cbar_position.y0-0.15,cbar_position.width,cbar_position.height]) # for bottom location

        # now save the figure
        savedir = Path(__file__).parent / 'figures' / 'watershed_figure_illustration'
        savedir.mkdir(exist_ok=True,parents=True)
        
        for ext in ['.png','.pdf']:
            savename = f'{colony}_{track}_{TIMEPOINT}_{full_crop}_res{RESOLUTION_LEVEL}_{cmapstr}'
            savepath = savedir / savename
            print(f'Saved figure to {savepath}')
            save_and_show_plot(str(savepath),
                               file_extension=ext,
                               figure=fig,
                               transparent=True,
                               keep_open=True,
                               **{'dpi':600}
                            )
    
if __name__ == '__main__':
    # set the details
    # dft0 = run_validation_and_plot(120,track=84619,colony='medium',RESOLUTION_LEVEL=1,plot_everything=True)
    # dft1 = run_validation_and_plot(0,track=87172,colony='medium',RESOLUTION_LEVEL=1,plot_everything=False)
    # dft1 = run_validation_and_plot(48,track=87099,colony='medium',RESOLUTION_LEVEL=1,plot_everything=False)
    dft1 = run_validation_and_plot(88,track=81463,colony='medium',RESOLUTION_LEVEL=1,plot_everything=False)

