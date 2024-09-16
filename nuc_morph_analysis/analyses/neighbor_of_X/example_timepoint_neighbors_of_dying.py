#%%
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
from pathlib import Path
import os
import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.lib.visualization.plotting_tools import plot_colorized_img_with_labels

# TEMP: loading local for testing and speed
df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline',load_local=True)
#%%
# for testing only use a subset of timepoints
track_id = 87124
TIMEPOINT = int(df.loc[df['track_id']==track_id,'identified_death'].values[0])
RESOLUTION_LEVEL = 1

colony = 'medium'
dfc = df.loc[df['colony']==colony].copy()
dfm = dfc.loc[(dfc['index_sequence']==TIMEPOINT)].copy()
#%%
# set figure directory
figdir = Path(__file__).parent / "figures" / "example_timepoint_neighbors_of_dying"
os.makedirs(figdir,exist_ok=True)

# load the segmentation image
reader = load_data.get_dataset_segmentation_file_reader(colony)
if RESOLUTION_LEVEL>0:
    reader.set_resolution_level(RESOLUTION_LEVEL)

lazy_img = reader.get_image_dask_data("ZYX",T=TIMEPOINT)
img= lazy_img.compute()

dft = dfm[dfm['index_sequence']==TIMEPOINT]

column_list = ['has_dying_neighbor']
# now plot the image with the mitotic neighbors
for col in column_list:
    dft[f'{col}2'] = dft[f'{col}'] +1 

    colormap_dict = {}
    colormap_dict.update({f"{col}_empty":(col,False,1,(0.4,0.4,0.4),f"")})
    colormap_dict.update({f"{col}":(col,True,2,(1,1,0),f"{col}")})
    colormap_dict.update({f"death":('frame_of_death',True,3,(1,0,1),f"death event")})

    fig,ax = plt.subplots(figsize=(5,5),layout='constrained')
    _ = plot_colorized_img_with_labels(ax,img,dft.copy(),colormap_dict)

    plt.title(f'neighbors of dying cells\n{col}\nt={TIMEPOINT}')
    plt.axis('off')

    savename = figdir / f'{colony}-{TIMEPOINT}-{col}_neighbors.png'
    savepath = figdir / savename
    save_and_show_plot(savepath.as_posix(),
                       file_extension='.png',
                       figure=fig,
                       transparent=False,
    )
    plt.show()