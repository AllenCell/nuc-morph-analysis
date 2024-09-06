#%%
from nuc_morph_analysis.lib.visualization.reference_points import COLONY_COLORS, COLONY_LABELS
import matplotlib.pyplot as plt
import numpy as np
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
from nuc_morph_analysis.lib.preprocessing import filter_data
from pathlib import Path
#%%
# set figure directory
figdir = Path(__file__).parent / "figures"
figdir.mkdir(exist_ok=True)

# TEMP: loading local for testing and speed
dfm = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline',load_local=True)
dfm = filter_data.all_timepoints_minimal_filtering(dfm)
#%% 
# important set all edge cells to have a 2d_area_nuc_cell_ratio of nan after merging into the main dataframe
dfm.loc[dfm['colony_depth']==1,'2d_area_nuc_cell_ratio'] = np.nan
dfm.loc[dfm['colony_depth']==1,'2d_area_pseudo_cell'] = np.nan
dfm.loc[dfm['colony_depth']==1,'2d_area_nucleus'] = np.nan

#%%
fig,ax = plt.subplots(figsize=(4,3))
x_col = "colony_time"
y_col = '2d_area_nuc_cell_ratio'
for colony in ['small','medium','large']:
    
    dfsub = dfm[dfm['colony']==colony].copy()
    dfsub.dropna(subset=[y_col],inplace=True)

    # create a pivot of the dataframe to get a 2d array of track_id x timepoint with each value being the density
    pivot = dfsub.pivot(index=x_col, columns='track_id', values=y_col)
    pivot.head()

    mean = pivot.median(axis=1)
    lower = pivot.quantile(0.25,axis=1)
    upper = pivot.quantile(0.75,axis=1)
    
    xscale_factor, xlabel, xunit, xlimit = get_plot_labels_for_metric(x_col)
    x = mean.index * xscale_factor
    yscale_factor, ylabel, yunit, ylimit = get_plot_labels_for_metric(y_col)
    y = mean.values * yscale_factor
    yl = lower.values * yscale_factor
    yu = upper.values * yscale_factor
    
    ax.plot(x, y, label=colony)
    ax.fill_between(x, yl, yu, alpha=0.5)
    ax.set_xlabel(f"{xlabel} {xunit}")
    ax.set_ylabel(f"{ylabel} {yunit}")


ax.legend(loc="upper right", handletextpad=0.7, frameon=False)
plt.tight_layout()
for ext in ['.png','.pdf']:
    save_and_show_plot(
        f"{figdir}/2d_area_nuc_cell_ratio",
        file_extension=ext,
        dpi=300,
        transparent=True,
    )
plt.show()


# NOTE to self: more plots are available in the initial_watershed_based_density_etc branch