#%%
# from nuc_morph_analysis.lib.visualization.reference_points import COLONY_COLORS, COLONY_LABELS
# from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
from pathlib import Path
import os
import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.lib.preprocessing import labeling_neighbors_helper
from nuc_morph_analysis.lib.visualization.plotting_tools import colorize_image, plot_colorized_img_with_labels

# TEMP: loading local for testing and speed
df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline',load_local=True)

# for testing only use a subset of timepoints
TIMEPOINT = 48
RESOLUTION_LEVEL = 1
CMAP = 'Dark2_r'

frames = 10
colony = 'medium'
dfc = df.loc[df['colony']==colony].copy()
dfm = dfc.loc[(df['index_sequence']>=TIMEPOINT - frames) | (df['index_sequence']<=TIMEPOINT + frames)].copy()
dfm = labeling_neighbors_helper.label_nuclei_that_neighbor_current_mitotic_event(dfm,)

#%%
# set figure directory
figdir = Path(__file__).parent / "figures" / "example_timepoint_neighbors_of_mitotic"
os.makedirs(figdir,exist_ok=True)


# now load image at timepoint
# load the segmentation image
reader = load_data.get_dataset_segmentation_file_reader(colony)
if RESOLUTION_LEVEL>0:
    reader.set_resolution_level(RESOLUTION_LEVEL)

lazy_img = reader.get_image_dask_data("ZYX",T=TIMEPOINT)
img= lazy_img.compute()

dft = dfm[dfm['index_sequence']==TIMEPOINT]

column_list = ['has_mitotic_neighbor_formation_backward_dilated','has_mitotic_neighbor_breakdown_forward_dilated','has_mitotic_neighbor_dilated','exiting_mitosis']
# now plot the image with the mitotic neighbors
for col in column_list:
    dft[f'{col}2'] = dft[f'{col}'] +1 
    colored_img = colorize_image(img.max(axis=0),dft,feature=f'{col}2')
    fig,ax = plt.subplots(figsize=(3,3))
    plt.imshow(colored_img,
               cmap = CMAP,
               vmin=0,
               vmax=4,
               interpolation='nearest')
    
    plt.title(f'neighbors of mitotic cells\n{col}')
    plt.axis('off')

    savename = figdir / f'{colony}-{TIMEPOINT}-{col}-{CMAP}_neighbors.png'
    savepath = figdir / savename
    save_and_show_plot(savepath.as_posix(),
                       file_extension='.png',
                       figure=fig,
    )
    plt.show()


#%%
# and color by number of mitotic neighbors
# now plot the image with the mitotic neighbors
df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline',load_local=True)
dfm = labeling_neighbors_helper.label_nuclei_that_neighbor_current_mitotic_event(df)
#%%
t = 57
lazy_img = reader.get_image_dask_data("ZYX",T=t)
img= lazy_img.compute()
dft = dfm[dfm['index_sequence']==t]
colormap_dict = {}
cmap1 = plt.get_cmap('Dark2_r')
col = 'number_of_mitotic_frame_of_breakdown_neighbors'
colormap_dict.update({f'{col}_{i}':(col,i,i+1,cmap1.colors[i],f"{i} mitotic neighbors") for i in range(dfm[col].max()+1)}) # type: ignore
colormap_dict.update({'frame_of_breakdown':('frame_of_breakdown',True,8,(1,0,0),f"breakdown event")})
fig,ax = plt.subplots(figsize=(5,5))
_ = plot_colorized_img_with_labels(ax,img,dft,colormap_dict)

