#%%
from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import pseudo_cell_helper
from pathlib import Path
import pandas as pd
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from nuc_morph_analysis.analyses.dataset_images_for_figures.figure_helper import return_glasbey_on_dark
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


# set the details
TIMEPOINT = 48
colony = 'medium'
RESOLUTION_LEVEL = 1
# load the segmentation image
reader = load_data.get_dataset_segmentation_file_reader(colony)
if RESOLUTION_LEVEL>0:
    reader.set_resolution_level(RESOLUTION_LEVEL)

# perform watershed based pseudo cell segmentation
df_2d, img_dict = pseudo_cell_helper.get_pseudo_cell_boundaries(colony, TIMEPOINT, reader, RESOLUTION_LEVEL, return_img_dict=True)
#%%
# now color the nucleus segmentation by the 2d_area_nuc_cell_ratio feature
# first load the dataset and merge
df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline',load_local=True)
dfm = pd.merge(df, df_2d, on=['label_img','index_sequence'], suffixes=('', '_pc'))

# now get the subset of the dataframe for the colony and timepoint
dfsub = dfm[dfm['colony']==colony]
dft = dfsub[dfsub['index_sequence']==TIMEPOINT]

# now create an image where the nuclei are colored by the 2d_area_nuc_cell_ratio feature 
recolored_img = colorize_image(img_dict['mip_of_labeled_image'], dft, feature='2d_area_nuc_cell_ratio')

# add the recolored image to the img_dict
img_dict['recolored_img'] = recolored_img

#%%
# now display all of the intermediates of the
# watershed based pseudo cell segmentation
x1,y1 = 500,200
w,h = 500,500
for full_crop, sizes in [('crop',(500,200,500,500)),('full',(0,0,recolored_img.shape[1],recolored_img.shape[0]))]:
    x1,y1,w,h = sizes
    crop_exp = np.index_exp[y1:y1+h,x1:x1+w]
    nrows = 2
    ncols = np.ceil(len(img_dict)/2).astype(int)
    fig,axr = plt.subplots(nrows,ncols,figsize=(ncols*4,nrows*4))
    axx = np.asarray([axr]).flatten()
    for i,key in enumerate(img_dict.keys()):
        ax = axx[i]
        assert type(ax) is plt.Axes
        ax.imshow(img_dict[key][crop_exp],
                    interpolation='nearest',)
        ax.set_title(key)

    # record the size of the last axis
    right_ax = axx[-1]
    assert type(right_ax) is plt.Axes
    position = right_ax.get_position().bounds

    # add colorbars to the last axes
    cbar = plt.colorbar(right_ax.imshow(recolored_img[crop_exp],interpolation='nearest'))
    cbar.set_label('Nucleus area/Cell area')
    # resize the last axis to be its original size before the colorbar was present
    right_ax.set_position(position)
    # move the colorbar to the right of the last axis
    cbar.ax.set_position((position[0]+position[2]+0.05,position[1],0.05,position[3]))

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
#%%
# now explore what colony depth levels can be trusted to have good areas
x1,y1 = 500,200
w,h = 500,500
# color an image by colony depth
recolored_img_depth = colorize_image(img_dict['mip_of_labeled_image'], dft, feature='colony_depth')
img_dict['recolored_img_depth'] = recolored_img_depth
for full_crop, sizes in [('crop',(500,200,500,500)),('full',(0,0,recolored_img.shape[1],recolored_img.shape[0]))]:
    x1,y1,w,h = sizes
    crop_exp = np.index_exp[y1:y1+h,x1:x1+w]
    keys = ['recolored_img','pseudo_cells_img','recolored_img_depth']
    nrows = np.ceil(len(keys)).astype(int)
    ncols = 1
    fig,axlist = plt.subplots(nrows,ncols,figsize=(ncols*4,nrows*4))
    axlist = np.asarray([axlist]).flatten()    
    for i,key in enumerate(keys):
        if 'depth' in key:
            # use a categorical colormap
            cmapstr = 'tab20'
        else:
            cmapstr = 'viridis'

        img = img_dict[key][crop_exp]
        mappable = axlist[i].imshow(img,
                    interpolation='nearest',
                    cmap = cmapstr)
        axlist[i].set_title(key)

        # record the size of the last axis
        position = axlist[i].get_position().bounds

        # add colorbars to the last axes
        cbar = plt.colorbar(mappable,ax=axlist[i])
        cbar.set_label(key)
    # resize the last axis to be its original size before the colorbar was present
        axlist[i].set_position(position)
    # move the colorbar to the right of the last axis
        # cbar.ax.set_position(tuple([position[0]+position[2]+0.05,position[1],0.05,position[3]])))
        cbar.ax.set_position((position[0]+position[2]+0.05,position[1],0.05,position[3]))

    # now save the figure
    savedir = Path(__file__).parent / 'figures' / 'pseudo_cell_validation'
    savedir.mkdir(exist_ok=True,parents=True)
    savename = f'{colony}_{TIMEPOINT}_{full_crop}_colony_depth_res{RESOLUTION_LEVEL}.png'   
    savepath = savedir / savename
    plt.savefig(savepath,
                dpi=300,
                bbox_inches='tight')
    print(f'Saved figure to {savepath}')
    plt.show()

#%%
# now create a plot drawing the boundaries of the nuclei and cells
# overlayed on the image colored with the 2d_area_nuc_cell_ratio
rgb_array0_255, _, _ = return_glasbey_on_dark()

fig, axlist = plt.subplots(1, 1, figsize=(6, 4))
nuc_mip = img_dict['mip_of_labeled_image']
cell_mip = img_dict['pseudo_cells_img']
new_img = np.zeros(list(nuc_mip.shape) + [3]).astype('uint8')

for index, row in dft.iterrows():
    color = np.float64(rgb_array0_255[row['label_img'] % len(rgb_array0_255)])

    # now draw the density image colored in viridis colormap
    pixels = cell_mip == row['label_img']
    value = row['2d_area_nuc_cell_ratio']
    value = float(value)
    
    cmap = cm.get_cmap(cmapstr)
    color_for_value = np.array(cmap(value))
    color_for_value = np.array(color_for_value) * 255
    new_img[pixels] = color_for_value[:3].astype('uint8')

    # get the nucleus boundary
    nucleus_boundary = nuc_mip == row['label_img']
    # get contours
    contours = find_contours(nucleus_boundary, 0.5)
    # draw contours
    for contour in contours:
        axlist.plot(contour[:, 1], contour[:, 0], linewidth=1, color=color/255)

    # get the cell boundary
    cell = cell_mip == row['label_img']
    # get contours
    contours = find_contours(cell, 0.5)
    # draw contours
    for contour in contours:
        axlist.plot(contour[:, 1], contour[:, 0], linewidth=1, color=color/255)

plt.imshow(img_dict['recolored_img'],
           interpolation='nearest',)
# remove the axis
plt.axis('off')
plt.colorbar()

# now save the figure
savedir = Path(__file__).parent / 'figures' / 'pseudo_cell_validation'
savedir.mkdir(exist_ok=True,parents=True)
savename = f'{colony}_{TIMEPOINT}_res{RESOLUTION_LEVEL}_boundaries.png'   
savepath = savedir / savename
plt.savefig(savepath,
            dpi=300,
            bbox_inches='tight')
print(f'Saved figure to {savepath}')
plt.show()


