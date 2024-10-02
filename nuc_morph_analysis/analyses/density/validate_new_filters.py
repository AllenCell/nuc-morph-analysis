#%%
from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import watershed_workflow
from pathlib import Path
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering

import numpy as np
import matplotlib.pyplot as plt
from nuc_morph_analysis.analyses.dataset_images_for_figures.figure_helper import return_glasbey_on_dark
from skimage.measure import find_contours
from nuc_morph_analysis.lib.visualization.plotting_tools import plot_colorized_img_with_labels
from multiprocessing import Pool, cpu_count

RESOLUTION_LEVEL=1

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
    for label, nuc_contours, cell_contours, color, width in contour_list:
        if new_color is not None:
            color = new_color
        for contour in nuc_contours:
            axlist.plot(contour[:, 1], contour[:, 0], linewidth=width, color=color/255)
        for contour in cell_contours:
            axlist.plot(contour[:, 1], contour[:, 0], linewidth=width, color=color/255)
    return axlist

def examine_image_after_filtering(colony,TIMEPOINT,dft,savedir):
    # perform watershed based pseudo cell segmentation
    df_2d, img_dict = watershed_workflow.get_image_and_run(colony, TIMEPOINT, RESOLUTION_LEVEL, return_img_dict=True)

    nuc_mip = img_dict['mip_of_labeled_image'][0]
    cell_mip = img_dict['pseudo_cells_img'][0]

    # create the figure
    fig, axlist = plt.subplots(1, 1, figsize=(8, 6))

    # define the colorscheme to be used
    colormap_dict = {}
    dft['all_cells']=True
    colormap_dict.update({f"1":('all_cells',True,1,(0.8,0.8,0.8),f"all cells")})
    colormap_dict.update({f"2":('bad_pseudo_cells_segmentation',True,3,(0.4,0.4,0.4),f"a priori bad seg")})
    colormap_dict.update({f"3":('colony_depth',1,4,(0,0.4,0.8),f"edge nucleus")})

    # reshape the image to be 3D for colorization
    nuc_mip_in = np.reshape(nuc_mip,(1,nuc_mip.shape[0],nuc_mip.shape[1]))
    axlist = plot_colorized_img_with_labels(axlist,nuc_mip_in,dft.copy(),colormap_dict)
        
    # create the contours
    contour_list = get_contours_from_pair_of_2d_seg_image(nuc_mip,cell_mip)
    # remove cell contours
    contour_list2 = []
    for contour in contour_list:
        contour_item = list(contour)
        
        contour_item[3] =  np.asarray((150,0,150))
        contour_item.append(0.5) # add width
        dfsub = dft[dft['label_img']==contour_item[0]]
        if len(dfsub) >0:
            if dfsub['uncaught_pseudo_cell_artifact'].values[0]:
                contour_item[3] = np.asarray((255,255,0))
                contour_item[4] = 1.5
            else:
                contour_item[1] = [] # empty list
        else:
            contour_item[1] = []

        contour_list2.append(contour_item)

    draw_contours_on_image(axlist,contour_list2)

    # print text where cells are highlighted
    dfth = dft[dft['uncaught_pseudo_cell_artifact']==True]
    for label in dfth['label_img'].values:
        x = dfth[dfth['label_img']==label]['centroid_x'].values[0] /2.5
        y = dfth[dfth['label_img']==label]['centroid_y'].values[0] /2.5
        track_id = dfth[dfth['label_img']==label]['track_id'].values[0]
        # axlist.text(0,0,f"{label}",color='black',fontsize=6)
        axlist.text(x,y,f"{track_id}",color='tab:red',fontsize=8)
    # remove the axis
    plt.axis('off')
    titlestr = f"newly filtered cells highlighted in yellow\n{colony}_{TIMEPOINT}_res{RESOLUTION_LEVEL}"
    plt.title(f'{titlestr}')


    savename = f'{colony}_{TIMEPOINT}_res{RESOLUTION_LEVEL}.png'
    savepath = savedir / savename
    plt.savefig(savepath,
                dpi=300,
                bbox_inches='tight')
    plt.show()
        

def main():
    # combine the dataframes
    df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline',load_local=True)
    dfh = df[df['uncaught_pseudo_cell_artifact']==True]
    print(f"Number of cells that meet the criteria: {dfh.shape[0]}")

    colony_list = dfh['colony'].unique()
    for colony in colony_list:
        time_list = dfh[dfh['colony']==colony]['index_sequence'].unique()
        print(f"Colony: {colony}")
        print(f"Timepoints: {time_list}")
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
    parallel = True
    if parallel:
        print('running in parallel')
        n_cores = cpu_count()
        with Pool(n_cores) as p:
            p.starmap(examine_image_after_filtering,args_list)
    else:
        for ai,args in enumerate(args_list):
            print(ai)
            examine_image_after_filtering(*args)

if __name__ == '__main__':
    main()

