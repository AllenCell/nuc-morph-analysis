# %%
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.lib.preprocessing import filter_data 
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
import numpy as np
import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.preprocessing import add_times
from nuc_morph_analysis.lib.preprocessing import compute_change_over_time
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot 
from pathlib import Path
from nuc_morph_analysis.lib.visualization import plotting_tools

# get the set1 colormap
from matplotlib.cm import get_cmap
# cmap = get_cmap('Set1')
# set1[0,1,7]
# CYCLE_COLOR_DICT = {0:cmap.colors[0],1:cmap.colors[1],2:cmap.colors[7]}

# tab20b[5,9,13
cmap = get_cmap(name='plasma')
CYCLE_COLOR_DICT = {0:cmap.colors[5],1:cmap.colors[90],2:cmap.colors[220]}

# cmap = get_cmap('tab20b')
# CYCLE_COLOR_DICT = {0:cmap.colors[8],1:cmap.colors[16],2:cmap.colors[17]}


#%%
# load the data
df = load_dataset_with_features('all_baseline',load_local=True)
df2 = compute_change_over_time.run_script(df.set_index('CellId'), dxdt_feature_list = ['volume'], bin_interval_list=[12,24,48]) 
            
# filter to only full tracks (full_)
#%%
df_full = filter_data.add_analyzed_dataset_columns(df2)
df_full = df_full[df_full["full_interphase_dataset"] == True]
df_full = add_times.digitize_time_column(df_full,0,1,step_size=0.02,time_col='normalized_time',new_col='dig_time')

#%%
def group_and_extract(dfcc,xcol,ycol):
    grouper = dfcc.groupby(xcol)
    dfg = grouper[ycol].agg([np.nanmean,np.nanstd,'count',lambda x: np.nanpercentile(x,5),lambda x: np.nanpercentile(x,95)])
    dfg.rename(columns={'<lambda_0>':'5th','<lambda_1>':'95th'},inplace=True)
    return dfg

def plot_dfg(dfg,xcol1,ycol,labelstr,curr_ax,plot_type='mean',colorby=None):
    # remove rows with less than 10 counts
    dfg= dfg[dfg['count'] >= 10]

    xscale,xlabel,xunit,_ = get_plot_labels_for_metric(xcol1)
    yscale,ylabel,yunit,_ = get_plot_labels_for_metric(ycol)

    x = dfg.index * xscale
    y = dfg['nanmean'].values * yscale
    ylo = dfg['5th'].values * yscale
    yhi = dfg['95th'].values * yscale

    color = None
    if colorby is not None:
        if colorby == 'colony':
            color =plotting_tools.COLONY_COLORS[dfg['colony'].values[0]]
        if colorby == 'cellcycle':
            color = CYCLE_COLOR_DICT[dfg['cell_cycle'].values[0]]

    assert type(curr_ax) == plt.Axes

    if (plot_type=='STD') | (plot_type=='CV'):
        mean = dfg['nanmean'].values * yscale
        std = dfg['nanstd'].values * yscale

        all_values = dfcc[ycol].values * yscale
        if plot_type=='STD':
            y = std
            y_all = np.nanstd(all_values)
            titlestr = f"Stdev of {ylabel} {yunit}"

        elif plot_type=='CV':
            y = std/mean
            y_all = np.nanstd(all_values)/np.nanmean(all_values)
            titlestr = f"CV of {ylabel}"

        curr_ax.plot(x,y,label=labelstr, color = color)
        
        curr_ax.axhline(y_all, linestyle='--',color='k',label='across all')
        curr_ax.set_ylabel(titlestr)

    elif plot_type=='mean':
        curr_ax.plot(x,y,label=labelstr, color = color)
        curr_ax.fill_between(x,ylo,yhi,alpha=0.2, color = color, edgecolor='none')
        curr_ax.set_ylabel(f"{ylabel} {yunit}\n(90% interpercentile range)")

    elif plot_type=='count':
        curr_ax.plot(x,dfg['count'].values,label=labelstr, color = color)
        curr_ax.set_ylabel(f"Counts")


    curr_ax.set_xlabel(f"{xlabel} {xunit}")
    if xcol1 == 'index_sequence':
        curr_ax.set_xlim(0,48)
    elif xcol1 == 'dig_time':
        curr_ax.set_xlim(0,1)

    if (ycol == 'dxdt_48_volume') & (plot_type == 'mean'):
        curr_ax.set_ylim(-20,75)
    elif (ycol == 'volume') & (plot_type == 'mean'):
        curr_ax.set_ylim(400,1200)

    return curr_ax

#%%

#%%
# set global font sizes to be 8
fs = 16
plt.rcParams.update({'font.size': fs})
plt.rcParams.update({'axes.titlesize': fs})
plt.rcParams.update({'axes.labelsize': fs})
plt.rcParams.update({'xtick.labelsize': fs})
plt.rcParams.update({'ytick.labelsize': fs})
plt.rcParams.update({'legend.fontsize': fs})
#%%
colony_list = ['small','medium','large']

nrows = 1
ncols = len(colony_list)
figdir = Path(__file__).parent / 'figures' / 'figure_5'

for ycol in ['dxdt_48_volume','dxdt_24_volume','dxdt_12_volume','volume','height']:
    for xcol1 in ['index_sequence','dig_time','colony_depth']:
        for plot_type in ['mean','STD','CV','count']:

            fig,ax = plt.subplots(nrows,ncols,figsize=(ncols*3,nrows*2),constrained_layout=True,
                            sharey=True)
            assert type(ax) == np.ndarray # for mypy
            for ci,colony in enumerate(colony_list):
                dfc = df_full[df_full['colony']==colony]
                dfg = group_and_extract(dfc,xcol1,ycol)
                curr_ax = ax[ci]
                dfg['colony'] = colony
                curr_ax = plot_dfg(dfg,xcol1,ycol,"",curr_ax,plot_type=plot_type,colorby='colony')
            curr_ax.legend(loc='center left',bbox_to_anchor=(1.05,0.5))
            for ext in ['.png','.pdf']:
                savepath = figdir / f"ALL_{ycol}_{xcol1}_{plot_type}{ext}"
                save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)
            plt.show()

#%%
figdir = Path(__file__).parent / 'figures' / 'figure_5_cell_cycle_bins'
figdir.mkdir(exist_ok=True,parents=True)

cell_cycle_width = 0.2
cell_cycle_centers = [0.3,0.5,0.7]
cell_cycle_bins = [(cc-cell_cycle_width/2,cc+cell_cycle_width/2) for cc in cell_cycle_centers]


nrows = 1
ncols = len(colony_list)
# plot the mean of the value (ycol) binned by xcol for given cell cycle windows
for xcol1 in ['index_sequence','dig_time','colony_depth']:
    for ycol in ['dxdt_48_volume','dxdt_24_volume','dxdt_12_volume','volume','height']:
        for plot_type in ['mean','count']:

            sharey = False if plot_type == 'count' else True
            fig,ax = plt.subplots(nrows,ncols,figsize=(ncols*3,nrows*2),constrained_layout=True,
                            sharey=sharey)
            assert type(ax) == np.ndarray # for mypy

            for ci,colony in enumerate(colony_list):
                dfc = df_full[df_full['colony']==colony]

                for ri,cell_cycle_bin in enumerate(cell_cycle_bins):
                    dfcc = dfc[(dfc['dig_time'] >= cell_cycle_bin[0]) & (dfc['dig_time'] <= cell_cycle_bin[1])]
                    
                    # top column
                    dfg = group_and_extract(dfcc,xcol1,ycol)
                    dfg['cell_cycle'] = ri
                    curr_ax = ax[ci]
                    labelstr = f"{cell_cycle_bin[0]:.2f} to {cell_cycle_bin[1]:.2f}"
                    curr_ax = plot_dfg(dfg,xcol1,ycol,labelstr,curr_ax,plot_type=plot_type,colorby='cellcycle')

            curr_ax.legend(loc='center left',bbox_to_anchor=(1.05,0.5),title='Cell cycle window')

            # savepath = figdir / f"cell_cycle_bins_{ycol}_{xcol1}_{plot_type}.png"
            for ext in ['.png','.pdf']:
                savepath = figdir / f"cell_cycle_bins_{ycol}_{xcol1}_{plot_type}{ext}"
                save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)
            plt.show()
