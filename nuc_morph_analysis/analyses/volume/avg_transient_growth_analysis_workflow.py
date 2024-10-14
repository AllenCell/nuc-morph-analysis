# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.lib.preprocessing import filter_data, add_times
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot 
from nuc_morph_analysis.lib.visualization import plotting_tools
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric

# define cell cycle colors
cmap = plt.get_cmap(name='plasma')
# CYCLE_COLOR_DICT = {0:cmap.colors[5],1:cmap.colors[90],2:cmap.colors[220], 3:(0,0,1), 4:(1,0,0)}
CYCLE_COLOR_DICT = {0:cmap(5),1:cmap(90),2:cmap(220), 3:(0,0,1), 4:(1,0,0)}

#%%
# load the data
df = load_dataset_with_features('all_baseline',load_local=True)
df = filter_data.all_timepoints_minimal_filtering(df) # apply minimal filterting
df_full = filter_data.all_timepoints_full_tracks(df) # filter to only full tracks

# add digitzed normalized time column for cell cycle subest filtering
df_full = add_times.digitize_time_column(df_full,0,1,step_size=0.02,time_col='normalized_time',new_col='dig_time')

#%%
def group_and_extract(dfcc,xcol,ycol):
    """
    function for grouping and extracting the mean and 90% interpercentile range
    of the value (ycol) binned by xcol from dfcc

    Parameters
    ----------
    dfcc : pd.DataFrame
        dataframe to group
    xcol : str
        column to bin by (e.g. 'index_sequence')
    ycol : str
        column to extract (e.g. 'dxdt_48_volume')

    Returns
    -------
    dfg : pd.DataFrame
        dataframe after grouping. 
        contains columns 'nanmean','nanstd','count','5th','95th'
    """
    grouper = dfcc.groupby(xcol)
    dfg = grouper[ycol].agg([np.nanmean,np.nanstd,'count',lambda x: np.nanpercentile(x,5),lambda x: np.nanpercentile(x,95)])
    dfg.rename(columns={'<lambda_0>':'5th','<lambda_1>':'95th'},inplace=True)
    return dfg

def plot_dfg(dfcc,xcol,ycol,labelstr,curr_ax,plot_type='mean',colorby=None):
    """
    plot the mean of the value (ycol) binned by xcol from dfcc
    along with the 90% interpercentile range


    Parameters
    ----------
    dfcc : pd.DataFrame
        dataframe to plot
    xcol : str
        column to bin by
    ycol : str
        column to plot
    labelstr : str
        label for legend
    curr_ax : plt.Axes
        axis to plot on
    plot_type : str, optional
        'mean' or 'count'. The default is 'mean'.
    colorby : str, optional
        color to plot. The default is None.
        can be 'colony','cellcycle', or a color

    Returns
    -------
    curr_ax : plt.Axes
        axis with plot
    """
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
        elif colorby == 'cellcycle':
            color = CYCLE_COLOR_DICT[dfcc['cell_cycle'].values[0]]
        else:
            color = colorby

    assert type(curr_ax) == plt.Axes

    if plot_type=='mean':
        curr_ax.plot(x,y,label=labelstr, color = color, linewidth=0.5)
        curr_ax.fill_between(x,ylo,yhi,alpha=0.2, color = color, edgecolor='none')

    elif plot_type=='count':
        curr_ax.plot(x,dfg['count'].values,label=labelstr, linewidth=0.5, color = color)
    
    # adjust x axis details
    curr_ax.set_xlabel(f"{xlabel} {xunit}")
    if xcol == 'index_sequence':
        curr_ax.set_xlim(0,48)
        curr_ax.set_xticks(np.arange(0,48,12))
    elif xcol == 'dig_time':
        curr_ax.set_xlim(0,1)

    # adjust y axis details
    if plot_type == 'mean':
        curr_ax.set_ylabel(f"{ylabel} {yunit}\n(90% interpercentile range)")
    elif plot_type == 'count':
        curr_ax.set_ylabel(f"Counts")

    if (ycol == 'dxdt_48_volume') & (plot_type == 'mean'):
        curr_ax.set_yticks(np.arange(-20,80,20))
        curr_ax.set_ylim(-5,70)
    elif (ycol == 'volume') & (plot_type == 'mean'):
        curr_ax.set_ylim(400,1200)

    return curr_ax

def adjust_axis_positions(fig,ax,curr_pos=None,width=1,height=0.7,space=0.075):
    """
    adjust the position of the axes in this code

    Parameters
    ----------
    fig : plt.Figure
    ax : list of plt.Axes
    curr_pos : list, optional
        [x,y,width,height] in figure coordinates. The default is None.
    width : float, optional
        width of the axis in inches. The default is 1.
    height : float, optional
        height of the axis in inches. The default is 0.7.
    space : float, optional
        space between axes in inches. The default is 0.075.

    Returns
    -------
    fig : plt.Figure
    ax : list of plt.Axes
    """
    # now adjust axis positions
    for ci,cax in enumerate(ax):
        # make the axis = 1.0" wide x 0.7" tall
        if curr_pos is None:
            curr_pos = [1,1,width,height]
        else:
            curr_pos = [curr_pos[0] +  width + space ,curr_pos[1],width,height]
        # now adjust curr_pos to be in figure coordinates
        curr_pos_fig = [curr_pos[0]/fw,curr_pos[1]/fh,curr_pos[2]/fw,curr_pos[3]/fh]
        cax.set_position(curr_pos_fig)
        
        if ci>0:
            # remove ytick labels
            cax.set_yticklabels([])
            cax.set_ylabel('')
    return fig,ax

#%% update plotting parameters
# set global font sizes to be 8
fs = 8
plt.rcParams.update({'font.size': fs})
plt.rcParams.update({'axes.titlesize': fs})
plt.rcParams.update({'axes.labelsize': fs})
plt.rcParams.update({'xtick.labelsize': fs})
plt.rcParams.update({'ytick.labelsize': fs})
plt.rcParams.update({'legend.fontsize': fs})
# set axis linewidth
plt.rcParams.update({'axes.linewidth': 0.5})

# remove top and right axis lines
plt.rcParams.update({'axes.spines.top': False})
plt.rcParams.update({'axes.spines.right': False})

fx=1.5
fy=1
fw = 6.5
fh = 2.5

#%% make plot of medium colony only for panel F (left)
colony_list = ['medium']

nrows = 1
ncols = len(colony_list)
figdir = Path(__file__).parent / 'figures' / 'fig5_all_nuclei'

ycol = 'dxdt_48_volume'
xcol1 = 'index_sequence'
plot_type = 'mean'

fig,ax = plt.subplots(nrows,ncols,figsize=(fw,fh))
ax = np.asarray([ax]) if type(ax) != np.ndarray else ax # for mypy
assert type(ax) == np.ndarray # for mypy

for ci,colony in enumerate(colony_list):
    dfc = df[df['colony']==colony]
    curr_ax = ax[ci]
    curr_ax = plot_dfg(dfc,xcol1,ycol,"",curr_ax,plot_type=plot_type,colorby='colony')

fig,ax = adjust_axis_positions(fig,ax,curr_pos=None,width=1,height=0.7,space=0.075)

for ext in ['.png','.pdf']:
    savepath = figdir / f"ALL_{ycol}_{xcol1}_{plot_type}{ext}"
    save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)
plt.show()

#%% now make plot with cell cycle bins overlayed on medium colony only (panel F right)
figdir = Path(__file__).parent / 'figures' / 'fig5_fulltracks_cell_cycle_bins_by_colony'
figdir.mkdir(exist_ok=True,parents=True)

cell_cycle_width = 0.2
cell_cycle_centers = [0.3,0.5,0.7]
cell_cycle_bins = [(cc-cell_cycle_width/2,cc+cell_cycle_width/2) for cc in cell_cycle_centers]

# plot the mean of the value (ycol) binned by xcol for given cell cycle windows
xcol1 = 'index_sequence'
ycol = 'dxdt_48_volume'
plot_type = 'mean'
colony_list = ['small','medium','large']
for colony in colony_list:

    nrows = 1
    ncols = len(cell_cycle_bins)

    fig,ax = plt.subplots(nrows,ncols,figsize=(fw,fh))
    assert type(ax) == np.ndarray # for mypy

    dfc = df_full[df_full['colony']==colony]

    for ri,cell_cycle_bin in enumerate(cell_cycle_bins):
        curr_ax = ax[ri]
        curr_ax = plot_dfg(dfc,xcol1,ycol,"all",curr_ax,plot_type=plot_type,colorby='colony')
        dfcc = dfc[(dfc['dig_time'] >= cell_cycle_bin[0]) & (dfc['dig_time'] <= cell_cycle_bin[1])]
        
        # top column
        dfcc['cell_cycle'] = ri
        labelstr = f"{cell_cycle_bin[0]:.2f} to {cell_cycle_bin[1]:.2f}"
        curr_ax = plot_dfg(dfcc,xcol1,ycol,"cell cycle subset",curr_ax,plot_type=plot_type,colorby='k')
        curr_ax.set_title(labelstr,color='k')
    curr_ax.legend(loc='center left',bbox_to_anchor=(1.05,0.5),title='Cell cycle window')

    fig,ax = adjust_axis_positions(fig,ax,curr_pos=None,width=1,height=0.7,space=0.075)
    
    plt.suptitle(f"{colony}")
    # savepath = figdir / f"cell_cycle_bins_{ycol}_{xcol1}_{plot_type}.png"
    for ext in ['.png','.pdf']:
        savepath = figdir / f"cell_cycle_bins_for_only_{colony}_{ycol}_{xcol1}_{plot_type}{ext}"
        save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)
    plt.show()


#%%
# try a version that overalys all cell cycle bins together (panel F alt)
figdir = Path(__file__).parent / 'figures' / 'fig5_fulltracks_cell_cycle_bins_ONLY'
figdir.mkdir(exist_ok=True,parents=True)

cell_cycle_width = 0.2
cell_cycle_centers = [0.3,0.5,0.7]
cell_cycle_bins = [(cc-cell_cycle_width/2,cc+cell_cycle_width/2) for cc in cell_cycle_centers]

# plot the mean of the value (ycol) binned by xcol for given cell cycle windows
xcol1 = 'index_sequence'
ycol = 'dxdt_48_volume'
plot_type = 'mean'
for colony in colony_list:

    nrows = 1
    ncols = 1

    fig,ax = plt.subplots(nrows,ncols,figsize=(fw,fh))
    if type(ax) != np.ndarray:
        ax = np.asarray([ax])
    assert type(ax) == np.ndarray # for mypy

    dfc = df_full[df_full['colony']==colony]

    for ri,cell_cycle_bin in enumerate(cell_cycle_bins):
        curr_ax = ax[0]
        dfcc = dfc[(dfc['dig_time'] >= cell_cycle_bin[0]) & (dfc['dig_time'] <= cell_cycle_bin[1])]
        
        # top column
        dfcc['cell_cycle'] = ri
        labelstr = f"{cell_cycle_bin[0]:.2f} to {cell_cycle_bin[1]:.2f}"
        curr_ax = plot_dfg(dfcc,xcol1,ycol,labelstr,curr_ax,plot_type=plot_type,colorby=CYCLE_COLOR_DICT[ri])
    curr_ax.legend(loc='center left',bbox_to_anchor=(1.05,0.5),title='Cell cycle window')

    fig,ax = adjust_axis_positions(fig,ax,curr_pos=None,width=1,height=0.7,space=0.075)
    
    plt.suptitle(f"{colony}")
    # savepath = figdir / f"cell_cycle_bins_{ycol}_{xcol1}_{plot_type}.png"
    for ext in ['.png','.pdf']:
        savepath = figdir / f"cell_cycle_bins_for_only_{colony}_{ycol}_{xcol1}_{plot_type}{ext}"
        save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)
    plt.show()

#%%
# create supplemental plot showing how dxdt_48_volume changes over dig_time
figdir = Path(__file__).parent / 'figures' / 'fig5_dig_time_supp'
figdir.mkdir(exist_ok=True,parents=True)

cell_cycle_width = 0.2
cell_cycle_centers = [0.3,0.5,0.7]
cell_cycle_bins = [(cc-cell_cycle_width/2,cc+cell_cycle_width/2) for cc in cell_cycle_centers]
cell_cycle_bins =  cell_cycle_bins
# plot the mean of the value (ycol) binned by xcol for given cell cycle windows
colony='medium'

xcol1 = 'dig_time'
ycol = 'dxdt_48_volume'
plot_type = 'mean'

nrows = 1
ncols = 1

for colony in ['small','medium','large']:
    fig,ax = plt.subplots(nrows,ncols,figsize=(fw,fh))
    if type(ax) != np.ndarray:
        ax = np.asarray([ax])
    assert type(ax) == np.ndarray # for mypy

    dfc = df_full[df_full['colony']==colony]

    for ri,cell_cycle_bin in enumerate(cell_cycle_bins):
        curr_ax = ax[0]
        dfcc = dfc[(dfc['dig_time'] >= cell_cycle_bin[0]) & (dfc['dig_time'] <= cell_cycle_bin[1])]
        
        # top column
        dfcc['cell_cycle'] = ri
        labelstr = f"{cell_cycle_bin[0]:.2f} to {cell_cycle_bin[1]:.2f}"
        curr_ax = plot_dfg(dfcc,xcol1,ycol,labelstr,curr_ax,plot_type=plot_type,colorby=CYCLE_COLOR_DICT[ri])
    curr_ax.legend(loc='center left',bbox_to_anchor=(1.05,0.5),title='Cell cycle window')

    fig,ax = adjust_axis_positions(fig,ax,curr_pos=None,width=1,height=0.7,space=0.075)

    plt.suptitle(f"{colony}")
    # savepath = figdir / f"cell_cycle_bins_{ycol}_{xcol1}_{plot_type}.png"
    for ext in ['.png','.pdf']:
        savepath = figdir / f"cell_cycle_bins_for_only_{colony}_{ycol}_{xcol1}_{plot_type}{ext}"
        save_and_show_plot(str(savepath),ext,fig,transparent=False,keep_open=True)
    plt.show()