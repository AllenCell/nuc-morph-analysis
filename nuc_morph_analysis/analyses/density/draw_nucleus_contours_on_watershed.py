#%%
from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import watershed_workflow, pseudo_cell_helper, pseudo_cell_testing_helper
from pathlib import Path
import pandas as pd
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
from nuc_morph_analysis.lib.preprocessing import filter_data

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from nuc_morph_analysis.analyses.dataset_images_for_figures.figure_helper import return_glasbey_on_dark
from skimage.measure import find_contours
from nuc_morph_analysis.lib.visualization.plotting_tools import colorize_image
from nuc_morph_analysis.analyses.density.watershed_validate import get_contours_from_pair_of_2d_seg_image, draw_contours_on_image

        
def run_validation_and_plot(TIMEPOINT=48,colony='medium',RESOLUTION_LEVEL=1,plot_everything=False, testing=False):
    """
    run an image through the watershed based pseudo cell segmentation and examine the outputs
    optionally, run a test image through the same pipeline

    Parameters
    ----------
    TIMEPOINT : int, optional
        The timepoint to analyze, by default 48
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

    # now display all of the intermediates of the
    # watershed based pseudo cell segmentation
        # 'cyto_distance':(cyto_distance, "Distance Transform restricted\nto Cytoplasmic Segmentation"),
            # 'nucleus_edge_img':(nucleus_edge_img, "Nucleus Edge"),
    cmapstr = 'viridis'
    nuc_mip = img_dict['mip_of_labeled_image'][0]
    cell_mip = img_dict['pseudo_cells_img'][0]

    nuc_edge = img_dict['nucleus_edge_img'][0]
    cytp_dist = img_dict['cyto_distance'][0]
    for full_crop, sizes in [('crop',(500,200,500,500)),('full',(0,0,nuc_edge.shape[1],nuc_edge.shape[0]))]:
        x1,y1,w,h = sizes
        cimg = cytp_dist
            
        # define the colormap for the image
        cmap = cm.get_cmap(cmapstr)

        fig, axlist = plt.subplots(1, 1, figsize=(6, 4))

        vmin,vmax = (None,None)
        mappable = axlist.imshow(cimg, interpolation='nearest',cmap=cmap,
                                    vmin=vmin,vmax=vmax,
                                    )
        cbar = plt.colorbar(mappable,ax=axlist,label='cyto_distance')
            
        # create the contours
        contour_list = get_contours_from_pair_of_2d_seg_image(nuc_mip,cell_mip)
        # remove cell contours
        contour_list2 = []
        for contour in contour_list:
            contour_item = list(contour)
            contour_item[2] = [] # empty list
            contour_list2.append(contour_item)

        draw_contours_on_image(axlist,contour_list2,new_color=np.asarray((200,0,200))/255)
        # remove the axis
        plt.axis('off')
        plt.xlim([x1,x1+w])
        plt.ylim([y1,y1+h])
        titlestr = f"edges of nuclei overlaid on cytoplasmic distance transform\n{colony}_{TIMEPOINT}_{full_crop}_res{RESOLUTION_LEVEL}"
        plt.title(f'{titlestr}')

        # now save the figure
        savedir = Path(__file__).parent / 'figures' / 'validating nucleus edge'
        savedir.mkdir(exist_ok=True,parents=True)
        savename = f'{colony}_{TIMEPOINT}_{full_crop}_res{RESOLUTION_LEVEL}.png'
        savepath = savedir / savename
        plt.savefig(savepath,
                    dpi=300,
                    bbox_inches='tight')

if __name__ == '__main__':
    # set the details
    dft_test = run_validation_and_plot(testing=True)
    dft0 = run_validation_and_plot(plot_everything=True)