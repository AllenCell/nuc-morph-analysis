#%%
from pathlib import Path
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, load_data, filter_data
from nuc_morph_analysis.lib.preprocessing import add_times
import matplotlib
import matplotlib.pyplot as plt
from nuc_morph_analysis.analyses.volume.plot_help import group_and_extract
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression

# %%
# set up plot parameters and figure saving directory
matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({"font.size": 20})
plt.rcParams["figure.figsize"] = [5, 4]
figdir = f"volume/figures/local_growth/"

# %%
# load data
df = global_dataset_filtering.load_dataset_with_features()
df_full = filter_data.all_timepoints_full_tracks(df)
df_full = add_times.digitize_time_column(df_full,0,1,step_size=0.02,time_col='normalized_time',new_col='dig_time')

#%%
# now collect the average transient growth rates for different cell cycle windows
cell_cycle_width = 0.2
cell_cycle_centers = [0.3,0.5,0.7]
cell_cycle_bins = [(cc-cell_cycle_width/2,cc+cell_cycle_width/2) for cc in cell_cycle_centers]

plot_type = 'mean'
colony_list = ['small','medium','large']
# plot the mean of the value (ycol) binned by xcol for given cell cycle windows
ycol = 'dxdt_48_volume_dips_removed_um'
xcol1 = 'index_sequence'
dflist =[]
for colony in colony_list:

    dfc = df_full[df_full['colony']==colony]
    for ri,cell_cycle_bin in enumerate(cell_cycle_bins):
        dfcc = dfc[(dfc['dig_time'] >= cell_cycle_bin[0]) & (dfc['dig_time'] <= cell_cycle_bin[1])]
        
        dfnew = group_and_extract(dfcc,xcol1,ycol)

        
        dfnew['cell_cycle'] = ri
        dfnew['colony'] = colony
        dfnew['bin'] = f"{cell_cycle_bin[0]:.1f}-{cell_cycle_bin[1]:.1f}"
        dflist.append(dfnew)

dfall = pd.concat(dflist)
#%%
# now ask how correlated the avg transient growth rates are between different cell cycle windows
for colony, dfallc in dfall.groupby('colony'):
    cell_cycle_combos = list(combinations(dfall['cell_cycle'].unique(),2))

    for c1,c2 in cell_cycle_combos:
        df1 = dfallc[dfallc['cell_cycle']==c1]
        df2 = dfallc[dfallc['cell_cycle']==c2]
        dfcat = pd.merge(df1,df2,on='index_sequence',suffixes=('_1','_2'),how='inner')
        dfcat.dropna(subset=[f'nanmean_1',f'nanmean_2'],inplace=True)

        x = dfcat[f'nanmean_1']
        y = dfcat[f'nanmean_2']

        reg = LinearRegression().fit(x.values.reshape(-1,1),y)
        r2 = reg.score(x.values.reshape(-1,1),y)
        print(f"{colony} {c1}-{c2} r2={r2:.2f}")

    
