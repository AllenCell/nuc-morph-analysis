#%%
# from nuc_morph_analysis.lib.visualization.reference_points import COLONY_COLORS, COLONY_LABELS
# from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.lib.preprocessing import labeling_neighbors_helper
from nuc_morph_analysis.lib.visualization.plotting_tools import plot_colorized_img_with_labels


# TEMP: loading local for testing and speed
df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline',load_local=True)

# for testing only use a subset of timepoints
CMAP = 'Dark2_r'
track_id = 87172

# TIMEPOINT = 48
TIMEPOINT = int(df.loc[df['track_id']==track_id,'predicted_breakdown'].values[0])
frames_before = 3
frames_after = 12

time_list = np.arange(TIMEPOINT-frames_before,TIMEPOINT+frames_after+1,dtype='uint16')
RESOLUTION_LEVEL = 1

colony = 'medium'
dfc = df.loc[df['colony']==colony].copy()
dfm = dfc.loc[(dfc['index_sequence'].isin(time_list))].copy()

# artificially set all nuclei to have predicted_breakdown and predicted_formation to -1
# except for track_id = 87172
dfm.loc[dfm['track_id']!=track_id,'predicted_breakdown'] = -1
dfm.loc[:,'predicted_formation'] = -1
# recompute the features
dfm.drop(columns=['has_mitotic_neighbor_breakdown','has_mitotic_neighbor_formation','has_mitotic_neighbor','has_mitotic_neighbor_formation_backward_dilated','has_mitotic_neighbor_breakdown_forward_dilated','has_mitotic_neighbor_dilated','exiting_mitosis'],inplace=True)
dfm = labeling_neighbors_helper.label_nuclei_that_neighbor_current_mitotic_event(dfm)

#%%
# set figure directory
RESOLUTION_LEVEL = 1
figdir = Path(__file__).parent / "figures" / "neighbors_of_mitotic_track"
os.makedirs(figdir,exist_ok=True)

# now load image at timepoint
# load the segmentation image
reader = load_data.get_dataset_segmentation_file_reader(colony)
if RESOLUTION_LEVEL>0:
    reader.set_resolution_level(RESOLUTION_LEVEL)

crop_exp = np.index_exp[:,700:1100,650:1050]
nrows = 2
ncols = np.ceil(len(time_list)/nrows).astype(int)
fig,ax = plt.subplots(nrows,ncols,figsize=(ncols*2.5,nrows*2.5))
axx = np.asarray([ax]).flatten()
for ti,t in enumerate(time_list):
    current_ax = axx[ti]
    assert type(current_ax) is plt.Axes
    lazy_img = reader.get_image_dask_data("ZYX",T=t)
    img= lazy_img.compute()[crop_exp]

    dft = dfm[dfm['index_sequence']==t]
   
    colormap_dict = {} #type:ignore
    cmap1 = plt.get_cmap('Dark2_r')
    colormap_dict.update({'nothing':('frame_of_breakdown',False,1,(0.4,0.4,0.4),f"")}) 
    colormap_dict.update({'track':('track_id',track_id,2,(0.5,0.0,0.0),f"cell that will divide")}) #type:ignore
    colormap_dict.update({'frame_of_breakdown':('frame_of_breakdown',True,8,(1.0,0.0,1.0),f"breakdown event")})
    colormap_dict.update({'has_mitotic_neighbor_breakdown_dilated':('has_mitotic_neighbor_breakdown_forward_dilated',True,3,(1,1,0),f"has mitotic neighbor (forward)")})
    colormap_dict.update({'has_mitotic_neighbor_breakdown':('has_mitotic_neighbor_breakdown',True,4,(0,1,0),f"has mitotic neighbor")})


    show_legend=True if ti==len(time_list)-1 else False
    current_ax = plot_colorized_img_with_labels(current_ax,img,dft,colormap_dict,show_legend=show_legend)
    current_ax.set_title(f't={t}')

# set axis.off for all axes
for curr_ax in axx:
    assert type(curr_ax) is plt.Axes
    curr_ax.axis('off')
    

savename = f"{colony}-{track_id}-{CMAP}t={str(time_list)}"
savepath = str(figdir / savename)
plt.tight_layout()
save_and_show_plot(savepath,file_extension='.png',figure=fig,
                   transparent=False)
plt.show()