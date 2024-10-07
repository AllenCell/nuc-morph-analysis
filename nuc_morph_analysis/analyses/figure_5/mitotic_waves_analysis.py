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

#%%
# load the data
df = load_dataset_with_features('all_baseline',load_local=True)
# get volume dynamics
df = compute_change_over_time.run_script(df, dxdt_feature_list = ['volume'], bin_interval_list=[12,24,48]) 
df = compute_change_over_time.run_script(df, dxdt_feature_list = ['distance_from_centroid','height',], bin_interval_list=[12,24,48]) 

#%%
df = add_times.digitize_time_column(df,0,578,step_size=12,time_col ='index_sequence',new_col ='index_sequence_1hr')
df = add_times.digitize_time_column(df,0,578,step_size=6,time_col ='index_sequence',new_col ='index_sequence_30min')


#%%
def group_and_extract(dfcc,xcol,ycol):
    grouper = dfcc.groupby(xcol)
    dfg = grouper[ycol].agg([np.nanmean,np.nanstd,'sum','count',lambda x: np.nanpercentile(x,5),lambda x: np.nanpercentile(x,95)])
    dfg.rename(columns={'<lambda_0>':'5th','<lambda_1>':'95th'},inplace=True)
    return dfg

def plot_dfg(dfcc,xcol,ycol,labelstr,curr_ax,plot_type='mean',colorby=None):
    # remove rows with less than 10 counts
    dfg = group_and_extract(dfcc,xcol,ycol)
    dfg= dfg[dfg['count'] >= 10]

    xscale,xlabel,xunit,_ = get_plot_labels_for_metric(xcol)
    yscale,ylabel,yunit,_ = get_plot_labels_for_metric(ycol)

    x = dfg.index * xscale
    y = dfg['nanmean'].values * yscale
    ylo = dfg['5th'].values * yscale
    yhi = dfg['95th'].values * yscale

    color = None
    if colorby is not None:
        if colorby == 'colony':
            color =plotting_tools.COLONY_COLORS[dfcc['colony'].values[0]]
        else:
            color = colorby

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

        curr_ax.plot(x,y,label=labelstr, color = color, linewidth=1)
        
        curr_ax.axhline(y_all, linestyle='--',color='k',label='across all')
        curr_ax.set_ylabel(titlestr)

    elif plot_type=='mean':
        curr_ax.plot(x,y,label=labelstr, color = color, linewidth=1)
        curr_ax.fill_between(x,ylo,yhi,alpha=0.2, color = color, edgecolor='none')
        curr_ax.set_ylabel(f"{ylabel} {yunit}\n(90% interpercentile range)")

    elif plot_type=='count':
        curr_ax.plot(x,dfg['count'].values,label=labelstr, color = color, linewidth=1)
        curr_ax.set_ylabel(f"Counts")
    elif plot_type=='sum':

        curr_ax.plot(x,dfg['sum'].values * yscale,label=labelstr, color = color, linewidth=1)
        curr_ax.set_ylabel(f"Total {ylabel} {yunit}")

    elif plot_type=='sum_norm':
            
        curr_ax.plot(x,dfg['sum'].values / dfg['count'].values,label=labelstr, color = color, linewidth=1)
        curr_ax.set_ylabel(f"Total {ylabel} {yunit} (normalized)")

    curr_ax.set_xlabel(f"{xlabel} {xunit}")
    if xcol == 'index_sequence':
        curr_ax.set_xlim(0,48)
        curr_ax.set_xticks(np.arange(0,48+12,12))
    elif xcol == 'dig_time':
        curr_ax.set_xlim(0,1)

    if (ycol == 'dxdt_48_volume') & (plot_type == 'mean'):
        curr_ax.set_yticks(np.arange(-20,80,20))
        curr_ax.set_ylim(-5,70)


    elif (ycol == 'volume') & (plot_type == 'mean'):
        curr_ax.set_ylim(400,1200)

    return curr_ax

#%%
xcolbase = 'index_sequence'

ycol_list = [
    ('frame_of_breakdown','sum_norm',None),
    ('frame_of_formation','sum_norm',None),
    ('dxdt_24_volume','mean',None),
    ('dxdt_24_distance_from_centroid','mean',3),
    ('dxdt_24_height','mean',None),

    ]
colony_list = ['small','medium','large']
ncols = len(colony_list)
nrows = len(ycol_list)

fig,axlist = plt.subplots(nrows,ncols,figsize=(5*ncols,3*nrows),
                          sharey=False,
                          constrained_layout=True)
for yi,(ycol,plot_type,depth) in enumerate(ycol_list):
    xcol = f'{xcolbase}_1hr' if 'frame' in ycol else xcolbase

    if depth is not None:
        dfin = df[df['colony_depth']==depth]
    else:
        dfin = df

    for ci,colony in enumerate(colony_list):
        dfc = dfin[dfin['colony']==colony].copy()
        dfc[ycol] = dfc[ycol].astype(float)
        row = yi
        col = ci
        curr_ax = axlist[row,col]

        assert type(curr_ax) == plt.Axes
        labelstr = ycol if depth is None else f"{ycol} (depth {depth})"
        curr_ax = plot_dfg(dfc,xcol,ycol,labelstr,curr_ax,plot_type=plot_type,colorby='colony')
plt.show()

#%%
from scipy.signal import find_peaks

# find the peaks in the frame_of_formation 

xcolbase = 'index_sequence'
ycol = 'frame_of_formation'

for ci,colony in enumerate(colony_list):
    dfc = df[df['colony']==colony].copy()
    print(f"colony: {colony}")
    dfc[ycol] = dfc[ycol].astype(float)

    xcol = f'{xcolbase}_1hr' if 'frame' in ycol else xcolbase

    dfg = group_and_extract(dfc,xcol,ycol)

    x = dfg.index
    y = dfg['sum'].values / dfg['count'].values
    peaks, _ = find_peaks(y,width = 1, prominence = 0.02)
    plt.plot(x,y)
    plt.plot(x[peaks], y[peaks], "x")
    plt.show()

    # determine peak times
    peak_times = x[peaks]
    # now plot the behavior of the features in ycol_list around the peak times
    ncols = len(peak_times)
    nrows = len(ycol_list)
    fig,axlist = plt.subplots(nrows,ncols,figsize=(5*ncols,3*nrows),
                                sharey=False,
                                constrained_layout=True)
    for pi,peak_time in enumerate(peak_times):
        print(f"peak {pi}: {peak_time:.2f} hours")
        for yi,(ycol,plot_type,depth) in enumerate(ycol_list):
            if depth is not None:
                dfin = dfc[dfc['colony_depth']==depth]
            else:
                dfin = dfc

            dfin[ycol] = dfin[ycol].astype(float)
            row = yi
            col = pi
            curr_ax = axlist[row,col]

            assert type(curr_ax) == plt.Axes
            labelstr = ycol if depth is None else f"{ycol} (depth {depth})"
            curr_ax = plot_dfg(dfin,xcol,ycol,labelstr,curr_ax,plot_type=plot_type,colorby='colony')
            curr_ax.set_xlim(peak_time-100,peak_time+100)
            curr_ax.axvline(peak_time,linestyle='--',color='k')
    plt.show()




