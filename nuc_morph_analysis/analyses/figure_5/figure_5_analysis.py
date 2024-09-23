# %%
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.lib.preprocessing import filter_data 
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
import numpy as np
import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.preprocessing import add_times
#%%
df = load_dataset_with_features('all_baseline',load_local=True)
df.shape
#%%
df_full = filter_data.all_timepoints_full_tracks(df)
df_full.shape   
#%%
# now validate behavior of digitize_time_column
add_times.validate_dig_time_with_plot()

#%%
df_full = add_times.digitize_time_column(df_full,0,1,step_size=0.02,time_col='normalized_time',new_col='dig_time')

# now group into cell cycle bins
# create a pivot table
df_pivot = df_full.pivot_table(index='track_id',columns='dig_time',values='volume',aggfunc='mean')
df_pivot.shape

# 
xcol = 'index_sequence'
ycol = 'volume'

cell_cycle_width = 0.2
cell_cycle_centers = [0.3,0.5,0.7]
cell_cycle_bins = [(cc-cell_cycle_width/2,cc+cell_cycle_width/2) for cc in cell_cycle_centers]

colony_list = ['small','medium','large']

# top row of plots will be ycol = dxdt_48_volume, xcol = index_sequence
# bottom row of plots will be ycol = dxdt_48_volume, xcol = dig_time
# each column will be a different colony

nrows = 2
ncols = len(colony_list)

xcol1 = 'index_sequence'
xcol2 = 'dig_time'
groupbycol = 'label_img'
for ycol in ['dxdt_48_volume','volume']:

    fig,ax = plt.subplots(nrows,ncols,figsize=(ncols*4,nrows*2.5),constrained_layout=True,
                      sharey=True)
    assert type(ax) == np.ndarray # for mypy

    for ci,colony in enumerate(colony_list):
        dfc = df_full[df_full['colony']==colony]

        for ri,cell_cycle_bin in enumerate(cell_cycle_bins):
            dfcc = dfc[(dfc['dig_time'] >= cell_cycle_bin[0]) & (dfc['dig_time'] <= cell_cycle_bin[1])]
            
            # top column
            pivot = dfcc.pivot_table(index=xcol1,columns=groupbycol,values=ycol,aggfunc='mean')
            # remove rows with less than 10 counts
            count = pivot.count(axis=1)
            pivot = pivot[count >= 10]

            xscale,xlabel,xunit,_ = get_plot_labels_for_metric(xcol1)
            yscale,ylabel,yunit,_ = get_plot_labels_for_metric(ycol)

            x = pivot.index * xscale
            y = pivot.mean(axis=1)
            ylo = pivot.quantile(0.05,axis=1)
            yhi = pivot.quantile(0.95,axis=1)

            curr_ax = ax[0,ci]
            assert type(curr_ax) == plt.Axes

            curr_ax.plot(x,y,label=f"{cell_cycle_bin[0]:.2f} to {cell_cycle_bin[1]:.2f}")
            curr_ax.fill_between(x,ylo,yhi,alpha=0.2)
            curr_ax.set_xlabel(f"{xlabel} {xunit}")
            curr_ax.set_ylabel(f"{ylabel} {yunit}\n(90% interpercentile range)")

            # bottom column
            pivot = dfcc.pivot_table(index=xcol2,columns=groupbycol,values=ycol,aggfunc='mean')
            # remove rows with less than 10 counts
            count = pivot.count(axis=1)
            pivot = pivot[count >= 10]

            xscale,xlabel,xunit,_ = get_plot_labels_for_metric(xcol2)
            yscale,ylabel,yunit,_ = get_plot_labels_for_metric(ycol)

            x = pivot.index * xscale
            y = pivot.mean(axis=1)
            ylo = pivot.quantile(0.05,axis=1)
            yhi = pivot.quantile(0.95,axis=1)

            curr_ax = ax[1,ci]
            assert type(curr_ax) == plt.Axes

            curr_ax.plot(x,y,label=f"{cell_cycle_bin[0]:.2f} to {cell_cycle_bin[1]:.2f}")
            curr_ax.fill_between(x,ylo,yhi,alpha=0.2)
            curr_ax.set_xlabel(f"{xlabel} {xunit}")
            curr_ax.set_ylabel(f"{ylabel} {yunit}\n(90% interpercentile range)")



#%%
xcol1 = 'index_sequence'
xcol2 = 'dig_time'
groupbycol = 'label_img'
for ycol in ['dxdt_48_volume','volume']:

    fig,ax = plt.subplots(nrows,ncols,figsize=(ncols*4,nrows*2.5),constrained_layout=True,
                      sharey=True)
    assert type(ax) == np.ndarray # for mypy

    for ci,colony in enumerate(colony_list):
        dfc = df_full[df_full['colony']==colony]

        
        # top column
        pivot = dfc.pivot_table(index=xcol1,columns=groupbycol,values=ycol,aggfunc='mean')
        # remove rows with less than 10 counts
        count = pivot.count(axis=1)
        pivot = pivot[count >= 10]

        xscale,xlabel,xunit,_ = get_plot_labels_for_metric(xcol1)
        yscale,ylabel,yunit,_ = get_plot_labels_for_metric(ycol)



        x = pivot.index * xscale
        y = pivot.std(axis=1)/pivot.mean(axis=1)

        curr_ax = ax[0,ci]
        assert type(curr_ax) == plt.Axes

        all_values = pivot.values.flatten()
        cv_all = np.nanstd(all_values)/np.nanmean(all_values)

        curr_ax.plot(x,y)
        curr_ax.axhline(cv_all, linestyle='--',color='k',label='across all timepoints')

        curr_ax.set_xlabel(f"{xlabel} {xunit}")
        curr_ax.set_ylabel(f"CV of {ylabel}")

        # bottom column
        pivot = dfc.pivot_table(index=xcol2,columns=groupbycol,values=ycol,aggfunc='mean')
        # remove rows with less than 10 counts
        count = pivot.count(axis=1)
        pivot = pivot[count >= 10]

        xscale,xlabel,xunit,_ = get_plot_labels_for_metric(xcol2)
        yscale,ylabel,yunit,_ = get_plot_labels_for_metric(ycol)

        x = pivot.index * xscale
        y = pivot.std(axis=1)/pivot.mean(axis=1)
        all_values = pivot.values.flatten()
        cv_all = np.nanstd(all_values)/np.nanmean(all_values)


        curr_ax = ax[1,ci]
        assert type(curr_ax) == plt.Axes

        curr_ax.plot(x,y)
        curr_ax.axhline(cv_all, linestyle='--',color='k',label='all cell cycles')
        curr_ax.set_xlabel(f"{xlabel} {xunit}")
        curr_ax.set_ylabel(f"CV of {ylabel}")



