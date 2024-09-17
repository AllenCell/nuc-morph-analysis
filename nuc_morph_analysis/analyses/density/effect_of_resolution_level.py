#%%
from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import watershed_workflow
from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import metrics
import os
import numpy as np
#%%
colony = 'medium'
timepoint = 48
reader = load_data.get_dataset_segmentation_file_reader(colony)

# now perform watershed_workflow for resolution_level 0 and 1
df_2d_list=[]
for resolution_level in [0,1]:
    df_2d = watershed_workflow.get_image_and_run(colony, timepoint, reader, resolution_level, return_img_dict=False)
    df_2d_list.append(df_2d)

#%%
# examine correlation between nucleus_area for all dataframes
dfm = pd.merge(df_2d_list[0], df_2d_list[1],
                on='label_img',
                  suffixes=('_0', '_1'))

# define figure directory
fig_dir = Path(__file__).parent / 'figures' / 'effect_of_resolution_level'
os.makedirs(str(fig_dir), exist_ok=True)

# now plot correlation
feature_list = ['2d_area_nucleus','2d_area_pseudo_cell','2d_area_cyto','inv_cyto_density','2d_area_nuc_cell_ratio']
for feature in feature_list:
    fig,ax = plt.subplots(figsize=(2.5,2.5),layout = 'constrained')
    
    if 'ratio' in feature: # ratio features
        xunits = 1
        yunits = 1
        unitstr = ''
    elif 'inv_cyto' in feature: # density features
        xunits = 1/((0.108)**2)
        yunits = 1/((0.108*2.5)**2)
        unitstr = '1/um^2'
    else: # area features
        xunits = (0.108)**2
        yunits = (0.108*2.5)**2
        unitstr = 'um^2'


    x = dfm[feature+'_0'].values.reshape(-1,1) * xunits
    y = dfm[feature+'_1'].values.reshape(-1,1) * yunits

    # compute the correlation
    r2 = metrics.r2_score(x, y)

    ax.scatter(x, y)
    ax.set_xlabel(f"{feature} ({unitstr})"+'\n at resolution level 0')
    ax.set_ylabel(f"{feature} ({unitstr})"+'\n at resolution level 1\n(2.5x downsampled)')
    ax.set_title(feature)

    # add the unity line
    ax.axline((0,0),(1,1),color='black',linestyle='--')
    ax.set_xlim([np.min([x.min(),y.min()]),np.max([x.max(),y.max()])])
    ax.set_ylim([np.min([x.min(),y.min()]),np.max([x.max(),y.max()])])

    ax.text(0.05,0.95,f'R^2 = {r2:.2f}',transform=ax.transAxes,
            ha='left',va='top')
    ax.set_aspect('equal')

    save_path = fig_dir / f'{feature}_scatterplot.png'
    save_and_show_plot(str(save_path),'.png',
                       fig,transparent=False)
    plt.show()

