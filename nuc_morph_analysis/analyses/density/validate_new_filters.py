#%%
from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import watershed_workflow, pseudo_cell_helper, pseudo_cell_testing_helper
from pathlib import Path
import pandas as pd
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
from nuc_morph_analysis.lib.preprocessing import filter_data

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from nuc_morph_analysis.analyses.dataset_images_for_figures.figure_helper import return_glasbey_on_dark
from skimage.measure import find_contours
from nuc_morph_analysis.lib.visualization.plotting_tools import colorize_image
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric

RESOLUTION_LEVEL=1
HV = 3 # color of highlight for filter

def get_contours_from_pair_of_2d_seg_image(nuc_mip,cell_mip,dft=None):
        contour_list = [] # (label, nucleus_contour, cell_contour, color)
        rgb_array0_255, _, _ = return_glasbey_on_dark()
        for label_img in range(0,nuc_mip.max()+1):
            if dft is not None:
                #ask if the label_img is in the dataframe
                if label_img not in dft['label_img'].values:
                    continue
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

def draw_contours_on_image(axlist,contour_list,new_color=None):
    # draw contours
    for label, nuc_contours, cell_contours, color in contour_list:
        if new_color is not None:
            color = new_color
        for contour in nuc_contours:
            axlist.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color=color/255)
        for contour in cell_contours:
            axlist.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color=color/255)
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
        cmap = cm[cmapstr]
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
            draw_contours_on_image(axlist,contour_list,new_color=np.asarray((200,0,200)))
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


def examine_image_after_filtering(colony,TIMEPOINT,dft,savedir):
    # perform watershed based pseudo cell segmentation
    df_2d, img_dict = watershed_workflow.get_image_and_run(colony, TIMEPOINT, RESOLUTION_LEVEL, return_img_dict=True)

    cmapstr = 'tab10'
    nuc_mip = img_dict['mip_of_labeled_image'][0]
    cell_mip = img_dict['pseudo_cells_img'][0]
    cimg = nuc_mip
    # define the colormap for the image
    cmap = cm[cmapstr]

    feature = 'highlight'
    categorical = True

    # create the colorized image image
    feat = feature
    if feature == 'zeros':
        cimg = np.zeros_like(nuc_mip)
    else:
        cimg = colorize_image(nuc_mip, dft, feature=feat)
        
    if categorical:
        cimg = np.round(cimg).astype('uint16')

    # rgb = np.take(np.uint16(cmaparr*255),cimg.astype('uint16'),axis=0)

    # create the figure
    fig, axlist = plt.subplots(1, 1, figsize=(8, 6))

    vmin,vmax = (0,cmap.N) if categorical else (None,None)
    mappable = axlist.imshow(cimg, interpolation='nearest',cmap=cmap,
                                vmin=vmin,vmax=vmax,
                                )
    cbar = plt.colorbar(mappable,ax=axlist,label=feature)
    if categorical:
        cbar.set_ticks(np.arange(0.5,cmap.N+0.5,1),labels=np.arange(0,cmap.N,1))
        
    # create the contours
    contour_list = get_contours_from_pair_of_2d_seg_image(nuc_mip,cell_mip)
    # remove cell contours
    contour_list2 = []
    for contour in contour_list:
        contour_item = list(contour)
        contour_item[1] = [] # empty list
        contour_list2.append(contour_item)

    draw_contours_on_image(axlist,contour_list2,new_color=np.asarray((200,0,200)))

    # print text where cells are highlighted
    dfth = dft[dft['highlight']==HV]
    for label in dfth['label_img'].values:
        x = dfth[dfth['label_img']==label]['centroid_x'].values[0] /2.5
        y = dfth[dfth['label_img']==label]['centroid_y'].values[0] /2.5
        track_id = dfth[dfth['label_img']==label]['track_id'].values[0]
        # axlist.text(0,0,f"{label}",color='black',fontsize=6)
        axlist.text(x,y,f"{track_id}",color='black',fontsize=8)
    # remove the axis
    plt.axis('off')
    titlestr = f"edges of nuclei overlaid on cytoplasmic distance transform\n{colony}_{TIMEPOINT}_res{RESOLUTION_LEVEL}"
    plt.title(f'{titlestr}')


    savename = f'{colony}_{TIMEPOINT}_res{RESOLUTION_LEVEL}.png'
    savepath = savedir / savename
    plt.savefig(savepath,
                dpi=300,
                bbox_inches='tight')
        

def main():

    #%%
    # combine the dataframes
    df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline',load_local=True)


    #%%
    # now apply filters
    # 2d_perimeter_nuc_cell_ratio < 0.4 #OR logic
    # 2d_perimeter_pseudo_cell > 500  #OR logic

    # Nuclear area to (pseudo)cell area ratio < 0.2 # OR logic

    # colony_depth <= 3 # just try to get edge cells # AND logic
    #      # or has a neighbor with colony depth ==1 # AND logic

    log1 = df['2d_perimeter_nuc_cell_ratio'] < 0.4
    # log2 = df['2d_perimeter_pseudo_cell'] > (500 / get_plot_labels_for_metric('2d_perimeter_pseudo_cell')[0])
    log2 = df['2d_perimeter_pseudo_cell'] > 500

    log3 = df['2d_area_nuc_cell_ratio'] < 0.2
    log4 = df['colony_depth'] <= 3

    log = (log1 | log2 | log3) & log4
    # log = (log4)
    df_highlight = df[log]
    df['highlight'] = 1 # everything is 1
    df.loc[df['2d_area_nuc_cell_ratio'].isna(),'highlight'] = 2 # NaN values
    df.loc[df['colony_depth']==1,'highlight'] = 4 # colony depth == 1
    df.loc[df_highlight.index,'highlight'] = HV # filter values


    # set all NaN values to 1
    # dft['highlight'] = dft['highlight'].astype('bool')
    print(f"Number of cells that meet the criteria: {df_highlight.shape[0]}")

    #%%

    dfh = df[df['highlight']==HV]
    colony_list = dfh['colony'].unique()
    for colony in colony_list:
        time_list = dfh[dfh['colony']==colony]['index_sequence'].unique()
        print(f"Colony: {colony}")
        print(f"Timepoints: {time_list}")
    #%%
    # now save the figure
    savedir = Path(__file__).parent / 'figures' / 'validating_new_filters'
    savedir.mkdir(exist_ok=True,parents=True)

    args_list =[]
    for colony in colony_list:
        time_list = dfh[dfh['colony']==colony]['index_sequence'].unique()
        for timepoint in time_list:
            TIMEPOINT = timepoint
            args_list.append((colony,TIMEPOINT,df[(df['colony']==colony) & (df['index_sequence']==TIMEPOINT)].copy(),savedir))
            

    # run in parallel
    from multiprocessing import Pool, cpu_count
    n_cores = cpu_count()
    print('running in parallel')
    with Pool(n_cores) as p:
        p.starmap(examine_image_after_filtering,args_list)

if __name__ == '__main__':
    main()

