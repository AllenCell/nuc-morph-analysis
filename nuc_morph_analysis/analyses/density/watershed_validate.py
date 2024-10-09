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

def get_contours_from_pair_of_2d_seg_image(nuc_mip,cell_mip,dft=None):
        contour_list = [] # (label, nucleus_contour, cell_contour, color)
        rgb_array0_255, cmapper, _ = return_glasbey_on_dark()
        for label_img in range(1,nuc_mip.max()+1):
            if dft is not None:
                #ask if the label_img is in the dataframe
                if label_img not in dft['label_img'].values:
                    continue
            # color = np.float64(rgb_array0_255[label_img % len(rgb_array0_255)])
            color = np.float64(cmapper(label_img))

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

def draw_contours_on_image(axlist,contour_list,new_color=None,filled=False,colorize=False,dft=None,linewidth=1,colorfeat='2d_area_nuc_cell_ratio',cmapstr='PiYG'):
    # draw contours
    if dft is not None:
        minval = dft[colorfeat].min()
        maxval = dft[colorfeat].max()
    for label, nuc_contours, cell_contours, color in contour_list:
        if new_color is not None:
            color = new_color
        
        
        if colorize:
            if label not in dft['label_img'].values:
                continue
            value = dft[dft['label_img']==label][colorfeat].values[0]
            #rescale between 0 and 255
            new_value = (value - minval) / (maxval - minval)
            cmap = cm.get_cmap(cmapstr)
            color = np.float64(cmap(new_value))
        for contour in nuc_contours:
            axlist.plot(contour[:, 1], contour[:, 0], linewidth=linewidth, color=color)
            if filled: # now draw as filled
                axlist.fill(contour[:, 1], contour[:, 0], color=color, alpha=0.5)
        for contour in cell_contours:
            axlist.plot(contour[:, 1], contour[:, 0], linewidth=linewidth, color=color)
            if filled: # now draw as filled
                axlist.fill(contour[:, 1], contour[:, 0], color=color, alpha=0.5)

    return axlist

def plot_colorized_image_with_contours(img_dict,dft,feature,cmapstr,colony='test',TIMEPOINT=None,RESOLUTION_LEVEL=None,categorical=False,draw_contours=True):
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
            if categorical:
                if dft[feature].dtype == 'bool':
                    dft = dft.copy()
                    dft[feature] = dft[feature]+1
            cimg = colorize_image(nuc_mip, dft, feature=feature)
            
        # define the colormap for the image
        cmap = cm.get_cmap(cmapstr)
        if categorical:
            cimg = np.round(cimg).astype('uint16')

        # rgb = np.take(np.uint16(cmaparr*255),cimg.astype('uint16'),axis=0)
    
        # create the figure
        fig, axlist = plt.subplots(1, 1, figsize=(6, 4))

        vmin,vmax = (0,cmap.N) if categorical else (None,None)
        mappable = axlist.imshow(cimg, interpolation='nearest',cmap=cmap,
                                    vmin=vmin,vmax=vmax,
                                    )
        cbar = plt.colorbar(mappable,ax=axlist,label=feature)
        if categorical:
            cbar.set_ticks(np.arange(0.5,cmap.N+0.5,1),labels=np.arange(0,cmap.N,1))
            
        if draw_contours:
            # create the contours
            contour_list = get_contours_from_pair_of_2d_seg_image(nuc_mip,cell_mip,dft)
            draw_contours_on_image(axlist,contour_list,new_color=np.asarray((200,0,200))/255)
        # remove the axis
        plt.axis('off')
        plt.xlim([x1,x1+w])
        plt.ylim([y1,y1+h])
        titlestr = f"{feature}" if not draw_contours else f"{feature} with contours"
        plt.title(f'{titlestr}')

        # now save the figure
        savedir = Path(__file__).parent / 'figures' / 'watershed_validate_extras'
        savedir.mkdir(exist_ok=True,parents=True)
        savename = f'{colony}_{TIMEPOINT}_{full_crop}_res{RESOLUTION_LEVEL}_{feature}_{draw_contours}.png'
        savepath = savedir / savename
        plt.savefig(savepath,
                    dpi=300,
                    bbox_inches='tight')

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
    mip = img_dict['mip_of_labeled_image'][0]
    for full_crop, sizes in [('crop',(500,200,500,500)),('full',(0,0,mip.shape[1],mip.shape[0]))]:
        x1,y1,w,h = sizes
        crop_exp = np.index_exp[y1:y1+h,x1:x1+w]
        nrows = 2
        ncols = np.ceil(len(img_dict)//2).astype(int)
        assert ncols>1
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
        savedir = Path(__file__).parent / 'figures' / 'watershed_validate'
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
    
    if testing:
        columns = ['2d_area_nucleus','2d_area_pseudo_cell','2d_area_nuc_cell_ratio','zeros']
        for col in columns:
            plot_colorized_image_with_contours(img_dict,df_2d,col,'viridis',categorical=False,draw_contours=True)



    if plot_everything:
        # now load the tracking dataset and merge with the pseudo cell dataframe
        # first load the dataset and merge
        df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline')
        df = filter_data.all_timepoints_minimal_filtering(df)
        dfm = pd.merge(df, df_2d, on=['colony','index_sequence','label_img'], suffixes=('', '_pc'),how='left')

        # now get the subset of the dataframe for the colony and timepoint
        dfsub = dfm[dfm['colony']==colony]
        dft0 = dfsub[dfsub['index_sequence']==TIMEPOINT]
        dft = filter_data.apply_density_related_filters(dft0)


        plot_colorized_image_with_contours(img_dict,dft,'colony_depth','tab10',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=True,draw_contours=False)
        plot_colorized_image_with_contours(img_dict,dft,'colony_depth','tab10',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=True,draw_contours=True)
        plot_colorized_image_with_contours(img_dict,dft,'has_mitotic_neighbor_dilated','tab10',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=True,draw_contours=False)
        plot_colorized_image_with_contours(img_dict,dft,'has_mitotic_neighbor_dilated','tab10',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=True,draw_contours=True)        
        plot_colorized_image_with_contours(img_dict,dft,'2d_area_nuc_cell_ratio','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=False)
        plot_colorized_image_with_contours(img_dict,dft,'2d_area_nuc_cell_ratio','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=True)
        plot_colorized_image_with_contours(img_dict,dft,'2d_area_pseudo_cell','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=False)
        plot_colorized_image_with_contours(img_dict,dft,'2d_area_pseudo_cell','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=True)
        plot_colorized_image_with_contours(img_dict,dft,'2d_intensity_min_edge','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=False)
        plot_colorized_image_with_contours(img_dict,dft,'2d_intensity_min_edge','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=True)    
        plot_colorized_image_with_contours(img_dict,dft,'density','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=False)
        plot_colorized_image_with_contours(img_dict,dft,'density','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=True)
        plot_colorized_image_with_contours(img_dict,dft,'zeros','viridis',colony,TIMEPOINT,RESOLUTION_LEVEL,categorical=False,draw_contours=True)
        return dft0
    else:
        return df_2d
    
if __name__ == '__main__':
    # set the details
    # dft_test = run_validation_and_plot(testing=True)
    # dft0 = run_validation_and_plot(plot_everything=True)
    # dft0 = run_validation_and_plot(247,colony='small',RESOLUTION_LEVEL=1,plot_everything=True)
    dft0 = run_validation_and_plot(110,colony='medium',RESOLUTION_LEVEL=1,plot_everything=True)