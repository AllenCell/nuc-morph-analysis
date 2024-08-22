#%%
from nuc_morph_analysis.lib.preprocessing import load_data
from pathlib import Path
import pandas as pd
from nuc_morph_analysis.lib.visualization.reference_points import COLONY_COLORS, COLONY_LABELS
import matplotlib.pyplot as plt
import numpy as np

#%%
# TEMP: this block of code is temporary until the feature is added to generate_main_manifest.py
colonies = load_data.get_dataset_names(dataset='all_baseline')
output_directory = Path(__file__).parents[6] / "local_storage" / "pseudo_cell_boundaries"
resolution_level = 1

pqdir = '/allen/aics/assay-dev/users/Frick/PythonProjects/repos/local_storage/pseudo_cell_boundaries/'
# suffixes=('_pseudo_cell', '_nucleus')
# large_pseudo_cell_boundaries.parquet
dflist=[]
for colony in colonies:
    print(colony)
    pqfilepath = pqdir + colony + '_pseudo_cell_boundaries.parquet'
    dfsub = pd.read_parquet(pqfilepath)
    dflist.append(dfsub)
dfp = pd.concat(dflist)
print(dfp.shape)
dfp.head()

#%%
# TEMP: loading local for testing and speed
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline',load_local=True) 
#%%
if '2d_colony_nucleus' in dfp.columns:
    dfp['colony'] = dfp['2d_colony_nucleus']
dfm = pd.merge(df, dfp, on=['colony','index_sequence','label_img'], suffixes=('', '_pc'))
dfm.head()

#%% 
# important set all edge cells to have a nuc_area_per_cell of nan after merging into the main dataframe
dfm.loc[dfm['colony_depth']==1,'nuc_area_per_cell'] = np.nan
dfm.loc[dfm['colony_depth']==1,'2d_area_pseudo_cell'] = np.nan
dfm.loc[dfm['colony_depth']==1,'2d_area_nucleus'] = np.nan



#%%
fig,ax = plt.subplots(figsize=(4,3))

for colony in ['medium','large']:
    
    dfsub = dfm[dfm['colony']==colony]
    dfsub.dropna(subset=['nuc_area_per_cell'],inplace=True)

    # remove edge cells
    # dfsub = dfsub.loc[dfsub['colony_depth']>2]
    # create a pivot of the dataframe to get a 2d array of track_id x timepoint with each value being the density
    pivot = dfsub.pivot(index='colony_time', columns='track_id', values='nuc_area_per_cell')
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