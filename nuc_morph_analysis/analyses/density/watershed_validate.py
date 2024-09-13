#%%
from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import pseudo_cell_helper, watershed_workflow
from pathlib import Path
import pandas as pd
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
from nuc_morph_analysis.lib.preprocessing import filter_data

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from nuc_morph_analysis.analyses.dataset_images_for_figures.figure_helper import return_glasbey_on_dark
from nuc_morph_analysis.lib.preprocessing import labeling_neighbors_helper
from skimage.measure import find_contours


def colorize_image(mip, dft, feature='2d_area_nuc_cell_ratio'):
    """
    Function create an image where the segmentation image objects (e.g. nuclei)
    are colored by the a given feature from the dataframe of that timepoint 

    Parameters
    ----------
    mip : np.array
        the max intensity projection of the labeled image
    dft : pd.DataFrame
        the dataframe of the timepoint
    feature : str
        the feature to color the image by
    """
    
    # now recolor the image by matching the pixel values in image to label_img in dft
    recolored_img = np.zeros_like(mip).astype('float32')
    recolored_img[mip>0]=np.nan
    for _,row in dft.iterrows():
        recolored_img[mip==row['label_img']] = row[feature]
    return recolored_img

def get_contours_from_pair_of_2d_seg_image(nuc_mip,cell_mip):
        contour_list = [] # (label, nucleus_contour, cell_contour, color)
        rgb_array0_255, _, _ = return_glasbey_on_dark()
        for label_img in range(0,nuc_mip.max()+1):
            color = np.float64(rgb_array0_255[label_img % len(rgb_array0_255)])

            # get the nucleus boundary
            nucleus_boundary = nuc_mip == label_img
            # get contours
            nuc_contours = find_contours(nucleus_boundary, 0.5)
            
            # get the cell boundary
            cell = cell_mip == label_img
            # get contours
            cell_contours = find_contours(cell, 0.5)

            contour_list.append((label_img,nuc_contours,cell_contours,color))

        return contour_list

def draw_contours_on_image(axlist,contour_list):
    # draw contours
    for label, nuc_contours, cell_contours, color in contour_list:
        for contour in nuc_contours:
            axlist.plot(contour[:, 1], contour[:, 0], linewidth=1, color=color/255)
        for contour in cell_contours:
            axlist.plot(contour[:, 1], contour[:, 0], linewidth=1, color=color/255)
    return axlist

def plot_colorized_image_with_contours(img_dict,dft,feature,cmapstr,colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=True):
    # define the input images
    mip = img_dict['mip_of_labeled_image'][0]
    for full_crop, sizes in [('crop',(500,200,500,500)),('full',(0,0,mip.shape[1],mip.shape[0]))]:
        x1,y1,w,h = sizes
        crop_exp = np.index_exp[y1:y1+h,x1:x1+w]
        nuc_mip = img_dict['mip_of_labeled_image'][0]
        cell_mip = img_dict['pseudo_cells_img'][0]


        # create the colorized image image
        if feature == 'zeros':
            cimg = np.zeros_like(nuc_mip)
        else:
            cimg = colorize_image(nuc_mip, dft, feature=feature)
            
        # define the colormap for the image
        cmap = cm.get_cmap(cmapstr)
        if categorical:
            cimg = np.round(cimg).astype('uint16')

        # rgb = np.take(np.uint16(cmaparr*255),cimg.astype('uint16'),axis=0)
    
        # create the figure
        fig, axlist = plt.subplots(1, 1, figsize=(6, 4))
# 

        vmin,vmax = (0,cmap.N) if categorical else (None,None)
        mappable = axlist.imshow(cimg, interpolation='nearest',cmap=cmap,
                                    vmin=vmin,vmax=vmax,
                                    )
        cbar = plt.colorbar(mappable,ax=axlist,label=feature)
        if categorical:
            cbar.set_ticks(np.arange(0.5,cmap.N+0.5,1),labels=np.arange(0,cmap.N,1))
            
        if draw_contours:
            # create the contours
            contour_list = get_contours_from_pair_of_2d_seg_image(nuc_mip,cell_mip)
            draw_contours_on_image(axlist,contour_list)
        # remove the axis
        plt.axis('off')
        plt.xlim([x1,x1+w])
        plt.ylim([y1,y1+h])
        titlestr = f"{feature}" if not draw_contours else f"{feature} with contours"
        plt.title(f'{titlestr}')

        # now save the figure
        savedir = Path(__file__).parent / 'figures' / 'pseudo_cell_validation_extras'
        savedir.mkdir(exist_ok=True,parents=True)
        savename = f'{colony}_{TIMEPOINT}_{full_crop}_res{RESOLUTION_LEVEL}_{feature}_{draw_contours}.png'
        savepath = savedir / savename
        plt.savefig(savepath,
                    dpi=300,
                    bbox_inches='tight')

def make_validation_plot(TIMEPOINT=48,colony='medium',RESOLUTION_LEVEL=1,plot_everything=True):
    # load the segmentation image
    reader = load_data.get_dataset_segmentation_file_reader(colony)
    if RESOLUTION_LEVEL>0:
        reader.set_resolution_level(RESOLUTION_LEVEL)

    # perform watershed based pseudo cell segmentation
    df_2d, img_dict = watershed_workflow.get_image_and_run(colony, TIMEPOINT, reader, RESOLUTION_LEVEL, return_img_dict=True)

    # now load the tracking dataset and merge with the pseudo cell dataframe
    # first load the dataset and merge
    df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline',load_local=True)
    df = labeling_neighbors_helper.label_nuclei_that_neighbor_current_mitotic_event(df)
    df = labeling_neighbors_helper.label_nuclei_that_neighbor_current_death_event(df)
    df = filter_data.all_timepoints_minimal_filtering(df)
    dfm = pd.merge(df, df_2d, on=['label_img','index_sequence'], suffixes=('', '_pc'),how='left')

    # now get the subset of the dataframe for the colony and timepoint
    dfsub = dfm[dfm['colony']==colony]
    dft = dfsub[dfsub['index_sequence']==TIMEPOINT]

    # now display all of the intermediates of the
    # watershed based pseudo cell segmentation
    mip = img_dict['mip_of_labeled_image'][0]
    for full_crop, sizes in [('crop',(500,200,500,500)),('full',(0,0,mip.shape[1],mip.shape[0]))]:
        x1,y1,w,h = sizes
        crop_exp = np.index_exp[y1:y1+h,x1:x1+w]
        nrows = 1
        ncols = len(img_dict)
        fig,axr = plt.subplots(nrows,ncols,figsize=(ncols*4,nrows*4))
        axx = np.asarray([axr]).flatten()
        for i,key in enumerate(img_dict.keys()):
            ax = axx[i]
            assert type(ax) is plt.Axes
            img = img_dict[key][0]
            label = img_dict[key][1]
            ax.imshow(img[crop_exp],
                        interpolation='nearest',)
            ax.set_title(label)

        # record the size of the last axis
        right_ax = axx[-1]
        assert type(right_ax) is plt.Axes
        position = right_ax.get_position().bounds

        # now save the figure
        savedir = Path(__file__).parent / 'figures' / 'pseudo_cell_validation'
        savedir.mkdir(exist_ok=True,parents=True)
        savename = f'{colony}_{TIMEPOINT}_{full_crop}_res{RESOLUTION_LEVEL}.png'   
        savepath = savedir / savename
        plt.savefig(savepath,
                    dpi=300,
                    bbox_inches='tight')
        print(f'Saved figure to {savepath}')
        plt.show()

    # now create a plot drawing the boundaries of the nuclei and cells
    # overlayed on the image colored with the 2d_area_nuc_cell_ratio
    
    dft['2d_area_cyto'] = dft['2d_area_pseudo_cell'] - dft['2d_area_nucleus']
    dft = filter_data.apply_density_related_filters(dft)

    if plot_everything:
        plot_colorized_image_with_contours(img_dict,dft,'colony_depth','tab10',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=True,draw_contours=False)
        plot_colorized_image_with_contours(img_dict,dft,'colony_depth','tab10',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=True,draw_contours=True)
        plot_colorized_image_with_contours(img_dict,dft,'2d_area_nuc_cell_ratio','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=False)
        plot_colorized_image_with_contours(img_dict,dft,'2d_area_nuc_cell_ratio','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=True)
        plot_colorized_image_with_contours(img_dict,dft,'2d_area_pseudo_cell','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=False)
        plot_colorized_image_with_contours(img_dict,dft,'2d_area_pseudo_cell','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=True)
        plot_colorized_image_with_contours(img_dict,dft,'2d_area_cyto','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=False)
        plot_colorized_image_with_contours(img_dict,dft,'2d_area_cyto','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=True)    
        plot_colorized_image_with_contours(img_dict,dft,'density','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=False)
        plot_colorized_image_with_contours(img_dict,dft,'density','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=True)
        plot_colorized_image_with_contours(img_dict,dft,'zeros','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=True)

if __name__ == '__main__':
    # set the details
    make_validation_plot(plot_everything=False)