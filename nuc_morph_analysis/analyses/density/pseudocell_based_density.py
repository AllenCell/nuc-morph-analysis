#%%
from nuc_morph_analysis.lib.visualization.reference_points import COLONY_COLORS, COLONY_LABELS
import matplotlib.pyplot as plt
import numpy as np

#%%
# TEMP: loading local for testing and speed
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
dfm = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline',load_local=True)

#%% 
# important set all edge cells to have a 2d_area_nuc_cell_ratio of nan after merging into the main dataframe
dfm.loc[dfm['colony_depth']==1,'2d_area_nuc_cell_ratio'] = np.nan
dfm.loc[dfm['colony_depth']==1,'2d_area_pseudo_cell'] = np.nan
dfm.loc[dfm['colony_depth']==1,'2d_area_nucleus'] = np.nan

#%%
fig,ax = plt.subplots(figsize=(4,3))
for colony in ['small','medium','large']:
    
    dfsub = dfm[dfm['colony']==colony]
    dfsub.dropna(subset=['2d_area_nuc_cell_ratio'],inplace=True)

    # create a pivot of the dataframe to get a 2d array of track_id x timepoint with each value being the density
    pivot = dfsub.pivot(index='colony_time', columns='track_id', values='2d_area_nuc_cell_ratio')
    pivot.head()

    mean = pivot.median(axis=1)
    lower = pivot.quantile(0.25,axis=1)
    upper = pivot.quantile(0.75,axis=1)
    ax.plot(mean.index,mean, label=colony)
    ax.fill_between(mean.index, lower, upper, alpha=0.5)
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Nucleus area/Cell area')
plt.show()


# NOTE to self: more plots are available in the initial_watershed_based_density_etc branch