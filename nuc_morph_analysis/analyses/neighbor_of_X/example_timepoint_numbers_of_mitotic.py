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

# for testing only use a subset of timepoints
RESOLUTION_LEVEL = 1

# set figure directory
figdir = Path(__file__).parent / "figures" / "example_timepoint_numbers_of_mitotic"
os.makedirs(figdir,exist_ok=True)

#%%

# color by number of mitotic neighbors
# now plot the image with the mitotic neighbors
colony = 'medium'
TIMEPOINT = 57
reader = load_data.get_dataset_segmentation_file_reader(colony)
lazy_img = reader.get_image_dask_data("ZYX",T=TIMEPOINT)
img= lazy_img.compute()
dfm = df.loc[df['colony']==colony].copy()
dft = dfm[dfm['index_sequence']==TIMEPOINT]
colormap_dict = {}
cmap1 = plt.get_cmap('Dark2_r')
col = 'number_of_frame_of_breakdown_neighbors'
colormap_dict.update({f'{col}_{i}':(col,i,i+1,cmap1.colors[i],f"{i} mitotic neighbors") for i in range(dft[col].max()+1)}) # type: ignore
colormap_dict.update({'frame_of_breakdown':('frame_of_breakdown',True,8,(1,0,0),f"breakdown event")})
fig,ax = plt.subplots(figsize=(5,5),layout='constrained')
_ = plot_colorized_img_with_labels(ax,img,dft,colormap_dict)
savename = figdir / f'{colony}-{TIMEPOINT}-{col}_number_of_neighbors.png'
savepath = figdir / savename
save_and_show_plot(savepath.as_posix(),
                    file_extension='.png',
                    figure=fig,
                    transparent=False,
)
plt.show()