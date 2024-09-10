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
from nuc_morph_analysis.analyses.mitotic_neighbors import labeling_neighbors_helper
from nuc_morph_analysis.analyses.density.watershed_validate import colorize_image


# TEMP: loading local for testing and speed
df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline',load_local=True)
df = labeling_neighbors_helper.identify_frames_of_death(df)
#%%
# for testing only use a subset of timepoints
CMAP = 'Dark2_r'
track_id = 87124

# TIMEPOINT = 48
TIMEPOINT = int(df.loc[df['track_id']==track_id,'identified_death'].values[0])
frames_before = 3
frames_after = 3

time_list = np.arange(TIMEPOINT-frames_before,TIMEPOINT+frames_after+1,dtype='uint16')
RESOLUTION_LEVEL = 1

colony = 'medium'
dfc = df.loc[df['colony']==colony].copy()
dfm = dfc.loc[(dfc['index_sequence'].isin(time_list))].copy()

dfm = labeling_neighbors_helper.label_nuclei_that_neighbor_current_death_event(dfm)

#%%
# set figure directory
RESOLUTION_LEVEL = 1
figdir = Path(__file__).parent / "figures" / "neighbors_of_dying_track"
print(figdir)
os.makedirs(figdir,exist_ok=True)
# now load image at timepoint
# load the segmentation image
reader = load_data.get_dataset_segmentation_file_reader(colony)
if RESOLUTION_LEVEL>0:
    reader.set_resolution_level(RESOLUTION_LEVEL)

TIMEPOINT
dft = dfc[dfc['index_sequence']==TIMEPOINT]
y,x = dft.loc[dft['track_id']==track_id,['centroid_y','centroid_x']].values[0]
w=500
if RESOLUTION_LEVEL==1:
    y = np.round(y/2.5).astype(int)
    x = np.round(x/2.5).astype(int)
    w = np.round(w/2.5).astype(int)

crop_exp = np.index_exp[:,y-w:y+w,x-w:x+w]
nrows = 2
ncols = np.ceil(len(time_list)/nrows).astype(int)
fig,ax = plt.subplots(nrows,ncols,figsize=(ncols*2.5,nrows*2.5))
axx = np.asarray([ax]).flatten()
for ti,t in enumerate(time_list):
    current_ax = axx[ti]
    assert type(current_ax) is plt.Axes
    lazy_img = reader.get_image_dask_data("ZYX",T=t)
    img= lazy_img.compute()[crop_exp]


    col = 'has_dying_neighbor'
    # now plot the image with the mitotic neighbors
    dft = dfm[dfm['index_sequence']==t].copy()
    dft[f'{col}2'] = dft[f'{col}'] +1 
    dft.loc[dft['track_id']==track_id,f'{col}2'] = 3
    dft.loc[dft['identified_death']==t,f'{col}2'] = 4
    colored_img = np.uint8(colorize_image(img.max(axis=0),dft,feature=f'{col}2'))
    current_ax.imshow(colored_img,
                vmin=0,
                vmax=8,
                cmap=CMAP,
                interpolation='nearest')

    current_ax.set_title(f't={t}')

# set axis.off for all axes
for curr_ax in axx:
    assert type(curr_ax) is plt.Axes
    curr_ax.axis('off')


savename = f"{colony}-{track_id}-{CMAP}t={str(time_list)}"
savepath = str(figdir / savename)
save_and_show_plot(savepath,file_extension='.png',figure=fig)
plt.show()