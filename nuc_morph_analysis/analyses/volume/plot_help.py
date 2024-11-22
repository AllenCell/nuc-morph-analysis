import matplotlib.pyplot as plt
import numpy as np
from nuc_morph_analysis.lib.visualization import plotting_tools
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric

# define cell cycle colors
cmap = plt.get_cmap(name='plasma')
# CYCLE_COLOR_DICT = {0:cmap.colors[5],1:cmap.colors[90],2:cmap.colors[220], 3:(0,0,1), 4:(1,0,0)}
CYCLE_COLOR_DICT = {0:cmap(5),1:cmap(90),2:cmap(220), 3:(0,0,1), 4:(1,0,0)}

def adjust_axis_positions(fig,ax,curr_pos=None,width=1,height=0.7,space=0.075,keep_labels=False, horizontal=True):
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
    fw,fh = fig.get_size_inches()
    for ci,cax in enumerate(ax):
        # make the axis = 1.0" wide x 0.7" tall
        print(width,height,fw,fh)
        if curr_pos is None:
            curr_pos = [1,1 +  height,width,height]
        else:
            if horizontal:
                curr_pos = [curr_pos[0] +  width + space ,curr_pos[1],width,height]
            else:
                curr_pos = [curr_pos[0],curr_pos[1] - height - space, width, height]
        # now adjust curr_pos to be in figure coordinates
        curr_pos_fig = [curr_pos[0]/fw,curr_pos[1]/fh,curr_pos[2]/fw,curr_pos[3]/fh]
        cax.set_position(curr_pos_fig)

        if horizontal:
            if (ci>0) & (not keep_labels):
                    # remove ytick labels
                    cax.set_yticklabels([])
                    cax.set_ylabel('')
        else: 
            if (ci < len(ax)-1) & (not keep_labels):
                # remove xtick labels
                cax.set_xticklabels([])
                cax.set_xlabel('')
            if (ci>0) & (not keep_labels):
                cax.set_title('')
    return fig,ax

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

def plot_dfg(dfcc,xcol,ycol,labelstr,curr_ax,plot_type='mean',colorby=None,required_N=10):
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
    required_N : int, optional
        required number of valid datapoitns for averaging at a given timepoint to be included

    Returns
    -------
    curr_ax : plt.Axes
        axis with plot
    """
    # remove rows with less than 10 counts
    dfg = group_and_extract(dfcc,xcol,ycol)
    dfgindex = dfg['count']<required_N
    print(f" timepoints with less than {required_N} counts: {dfg[dfgindex].index.values}")
    dfg= dfg[dfg['count'] >= required_N]
    print(labelstr,dfg['count'].min(),dfg['count'].max(),dfg['count'].mean(),dfg['count'].sum(), "t=",dfg.shape[0])
    

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
    curr_ax.set_xlabel(f"Movie time {xunit}")
    if xcol == 'index_sequence':
        curr_ax.set_xticks(np.arange(0,60,10))
        curr_ax.set_xlim(0,48)
    elif xcol == 'dig_time':
        curr_ax.set_xticks(np.arange(0,1.2,0.2))
        curr_ax.set_xlim(0,1)

    # adjust y axis details
    if plot_type == 'mean':
        curr_ax.set_ylabel(f"(90% interpercentile range)\nAvg. nuclear transient\ngrowth rate {yunit}")
    elif plot_type == 'count':
        curr_ax.set_ylabel(f"Counts")

    if (ycol in ['dxdt_48_volume','dxdt_48_volume_dips_removed_um_unfilled']) & (plot_type == 'mean'):
        curr_ax.set_yticks(np.arange(-20,80,20))
        curr_ax.set_ylim(-5,70)
    if ('per_V' in ycol) & (plot_type == 'mean'):
        curr_ax.set_yticks(np.arange(-0,0.1,0.02))
        curr_ax.set_ylim(0.012,0.07)
    elif (ycol == 'volume') & (plot_type == 'mean'):
        curr_ax.set_ylim(400,1200)

    return curr_ax

def update_plotting_params(fs=7,fw=6.5,fh=8):
    plt.rcParams.update({'font.size': fs})
    plt.rcParams.update({'axes.titlesize': fs})
    plt.rcParams.update({'axes.labelsize': fs})
    plt.rcParams.update({'xtick.labelsize': fs})
    plt.rcParams.update({'ytick.labelsize': fs})
    plt.rcParams.update({'legend.fontsize': fs})
    # set axis linewidth
    plt.rcParams.update({'axes.linewidth': 0.5})
    plt.rcParams.update({'lines.linewidth': 0.7})

    # remove top and right axis lines
    plt.rcParams.update({'axes.spines.top': False})
    plt.rcParams.update({'axes.spines.right': False})

    # reduce the tick length
    plt.rcParams.update({'xtick.major.size': 1.5})
    plt.rcParams.update({'ytick.major.size': 1.5})
    return fs,fw,fh


#%%
# plot 1: neighbors volume over time
def plot_neighbors_volume_over_time(dfcolony,track_id_list,xcol='index_sequence',ycol='volume',fw=6.5,fh=8):
    fig,ax = plt.subplots(1,1,figsize=(fw,fh))
    axlist = np.asarray([ax]) if type(ax) != np.ndarray else ax
    assert type(axlist) == np.ndarray # for mypy

    # plot volume over time for all tracks
    xscale, _, _, _ = get_plot_labels_for_metric(xcol)
    yscale, ylabel, yunit, _ = get_plot_labels_for_metric(ycol)
    for ti, track_id in enumerate(track_id_list):
        dftrack = dfcolony[dfcolony["track_id"] == track_id]
        x = dftrack[xcol].values * xscale
        y = dftrack[ycol].values * yscale
        ax.plot(x, y, label=f"{track_id}",zorder=-100,linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig,axlist = adjust_axis_positions(fig,axlist,curr_pos=None,width=1,height=1,space=0.075)

    # set x axis limits and ticks based on all neighbor tracks
    dfall = dfcolony[dfcolony['track_id'].isin(track_id_list)]
    dfall.dropna(subset=[xcol],inplace=True)
    xmin,xmax = dfall[xcol].min() * xscale, dfall[xcol].max() * xscale

    # adjust X and Y axis limits and ticks
    ax.set_xticks(np.arange(0,48+12,6))
    ax.set_xlim(xmin-1,xmax+1)
    ax.set_yticks(np.arange(0,2000,200))
    ax.set_ylim(0,1400)

    # add title and labels
    ax.set_title('immediate neighbors of main track')
    ax.set_xlabel('Movie time (hr)')
    ax.set_ylabel(f"{ylabel} {yunit}")

    return fig,axlist


def plot_tracks_aligned_at_volume_drop_onset(dfcolony,track_id_list,MAIN_TRACK_ID,timepoint,xcol='index_sequence',ycol='volume',interval=5,shift=5,fw=6.5,fh=8):
    # interval = 5 #minutes per frame
    # shift = 5 #frames
    fig,ax = plt.subplots(1,1,figsize=(fw,fh))
    axlist = np.asarray([ax]) if type(ax) != np.ndarray else ax

    yscale, _, yunit, _ = get_plot_labels_for_metric(ycol)
    # now plot the neighbors in color
    for ti, track_id in enumerate(track_id_list):
        dftrack = dfcolony[dfcolony["track_id"] == track_id]
        x0 = dftrack[xcol].values
        y0 = dftrack[ycol].values

        # find the index of the timepoint
        idx = np.where(x0 == (timepoint - shift))[0][0]
        x_at_drop_onset = x0[idx]
        y_at_drop_onset = y0[idx]

        x = (x0 - x_at_drop_onset )* interval
        y = (y0 - y_at_drop_onset) * yscale

        ax.plot(x, y, label=f"{track_id}",zorder=-100,marker='.',markersize=1)

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig,axlist = adjust_axis_positions(fig,axlist,curr_pos=None,width=1,height=1,space=0.075)

    ax.set_xticks(np.arange(-75,100,25))
    ax.set_xlim(-25,85)
    ax.set_yticks(np.arange(-300,400,100))
    ax.set_ylim(-300,150)

    ax.set_xlabel('Time relative to\ndrop onset (min)')
    ax.set_ylabel(f"Change in volume relative\nvolume at drop onset {yunit}")

    return fig,axlist


def _set_labels(ax,xcol,ycol1):
    """
    used in fit_volume_smooths_out_punching.py
    """
    # get labels
    xscale,xlabel,xunit,_ = get_plot_labels_for_metric(xcol)
    yscale1,ylabel1,yunit1,_ = get_plot_labels_for_metric(ycol1)
    ax.set_xlabel(f"{xlabel} {xunit}")
    ax.set_ylabel(f"{ylabel1} {yunit1}")
    if 'dxdt' in ycol1:
        ax.set_ylabel(f"Nuclear transient\ngrowth rate (μm\u00B3)")

    # adjust y-axis
    if ycol1 == 'volume':
        ax.set_yticks(np.arange(0,1400,200))
        ax.set_ylim(300,1200)
    elif ycol1 == 'dxdt_48_volume':
        ax.set_yticks(np.arange(-20,100,20))
        ax.set_ylim(-5,80)

    # adjust x-axis
    ax.set_xlabel(f"Synchonrized nuclear\ngrowth time (hr)")
    return ax

def _plot_lines(df_track,xcol,ycol1,ycol2,ax):
    """
    used in fit_volume_smooths_out_punching.py
    """
    xscale,xlabel,xunit,_ = get_plot_labels_for_metric(xcol)
    yscale1,ylabel1,yunit1,_ = get_plot_labels_for_metric(ycol1)
    yscale2,ylabel2,yunit2,_ = get_plot_labels_for_metric(ycol2)

    x = df_track[xcol].astype('float').values 
    transition = df_track["frame_transition"].astype('float').min()
    x -= transition 
    x *= xscale

    y1 = df_track[ycol1].values * yscale1
    y2 = df_track[ycol2].values * yscale2

    ax.plot(x,y1,'k-',label=f"{ylabel1}")
    ax.plot(x,y2,'r-',label=f"{ylabel2}")

    ax = _set_labels(ax,xcol,ycol1)

    return ax



def plot_track_with_fit_line(df_track,ax,xcol='index_sequence',ycol1='volume',ycol2='volume_dips_removed_um_unfilled'):
    """
    used in fit_volume_smooths_out_punching.py
    """

    ax = _plot_lines(df_track,xcol,ycol1,ycol2,ax)
    return ax


#%% figure plotting volume trajectory
# create figure and axes
def plot_track_with_volume_dip(ax,df0,main_track_id,xcol='index_sequence',ycol='volume',start_at_0=True,add_time_point_lines=False,timepoint=None):
    """
    used for fit_volume_smooths_out_punching.py to show the effect of volume dip on a track volume and its dxdt_48_volume
    """
    dftrack = df0[df0["track_id"] == main_track_id]
    xscale, xlabel, xunit, _ = get_plot_labels_for_metric(xcol)
    yscale, ylabel, yunit, _ = get_plot_labels_for_metric(ycol)

    x0 = dftrack[xcol].values 
    transition = dftrack["frame_transition"].astype('float').min()
    if start_at_0:
        x = (x0 - transition)*xscale
    else:
        x = (x0)*xscale

    y = dftrack[ycol].values * yscale
    ax.plot(x, y, 'k-', label=f"track {main_track_id}",linewidth=0.7)
    if add_time_point_lines:
        tx1l_frame = timepoint-4*12
        tx1r_frame, tx2l_frame = timepoint,timepoint
        tx2r_frame = timepoint+4*12

        tx1_frame = np.asarray([tx1l_frame,tx1r_frame])
        tx2_frame = np.asarray([tx2l_frame,tx2r_frame])

        if start_at_0:
            tx1 = (tx1_frame - transition) * xscale
            tx2 = (tx2_frame - transition) * xscale
        else:
            tx1 = tx1_frame * xscale
            tx2 = tx2_frame * xscale
        ty1 = dftrack[dftrack['index_sequence'].isin(tx1_frame)][ycol].values * yscale
        ty2 = dftrack[dftrack['index_sequence'].isin(tx2_frame)][ycol].values * yscale
        ax.plot(tx1,ty1,'c-',linewidth=0.7)
        ax.plot(tx2,ty2,'m-',linewidth=0.7)
        ax.scatter(np.mean(tx1),np.mean(ty1),c='c',s=5)
        ax.scatter(np.mean(tx2),np.mean(ty2),c='m',s=5)



    ax.set_xlabel(f"Synchonrized nuclear\ngrowth time (hr)")
    ax.set_ylabel(f"{ylabel} {yunit}")
    ax.set_title(f"track {main_track_id}")
    ax.set_xticks(np.arange(0,20,4))
    ax.set_xlim(-2,np.max(x))

    ax = _set_labels(ax,xcol,ycol)


    return ax




def plot_dip_detection_validation(dftrack,peak_str = 'dips'):
    # axis 0 is interpolated raw_volume, sg_smoothed volume and power law fit volume (steps 0 and step1)
    # axis 1 is smoothed volume detrended by power law fit (and detected peaks)
    #     and it would be cool to draw the peak magnitude as vertical line on the plot
    # axis 2 is the raw volume with the peak annotated (and with the peak magnitude as vertical line)
    # axis 3 is the raw volume with the peak removed
    track_id = dftrack['track_id'].values[0]
    transition = dftrack['frame_transition']
    dftrack['transition_time'] = dftrack['index_sequence'].copy() - transition
    dftrack.set_index('index_sequence',inplace=True)
    ncols = 2
    nrows = 1
    fig,ax = plt.subplots(nrows,ncols,figsize=(6.5,8))
    ax = np.asarray([ax]) if type(ax) != np.ndarray else ax
    assert type(ax) == np.ndarray

    # ax0
    # plot raw volume on ax0 and ax2
    curr_ax = ax[0]
    xscale, xlabel, xunit, _ = get_plot_labels_for_metric("index_sequence")
    yscale, ylabel, yunit, _ = get_plot_labels_for_metric("volume")
    x = dftrack.transition_time.values * xscale
    y = dftrack['volume'].values * yscale
    curr_ax.plot(x,y,'k',linewidth=1,label='raw volume')
    curr_ax = ax[0]
    y = dftrack['volume_smooth'].values 
    curr_ax.plot(x,y,'g',linewidth=0.7, label='smoothed')

    ycol = 'fit_volume_interpolated'
    yscale = 1
    y = dftrack[ycol].values * yscale
    curr_ax.plot(x,y,label='power law fit',color='m',linewidth=0.7,linestyle='-')
    curr_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
                   handlelength=1,markerscale=1,frameon=False)
    curr_ax.set_ylabel(f"{ylabel} {yunit}")
    curr_ax.set_title('step1:\nsmooth volume trajectory\nand detrend with power law fit')
    curr_ax.set_yticks([400,600,800,1000,1200])
    curr_ax.set_ylim(400,1200)

    # ax1
    curr_ax = ax[1]
    ycol = f'volume_smooth_detrended'
    y = dftrack[ycol].values
    curr_ax.plot(x,y,color='g',linewidth=0.7,label='smoothed & detrended',zorder=-100)
    curr_ax.set_ylabel(f"Detrended volume {yunit}")
    curr_ax.set_title('step2:\nfind (inverse) peaks in\ndetrended volume trajectory')
    peaks = dftrack[f'volume_{peak_str}_peak_mask_at_center'] # boolean array
    peaks = peaks[peaks>0].index
    xpeak  = dftrack.loc[peaks,'transition_time'] * xscale
    ypeak = dftrack.loc[peaks,ycol]
    curr_ax.scatter(xpeak,ypeak,color='m',marker='o',s=2,label='peak')

    mask = dftrack[f'volume_{peak_str}_peak_mask_at_region']
    xpeak_mask = dftrack.loc[mask,'transition_time'] * xscale
    ypeak_mask = dftrack.loc[mask,ycol] * yscale

    curr_ax.plot(xpeak_mask,ypeak_mask,color='m',linestyle=':',linewidth=0.5,
                    zorder=1000,label='peak region')


    curr_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                   handlelength=1,markerscale=1,frameon=False)
    curr_ax.set_yticks([-100,-50,0,50,100])
    curr_ax.set_ylim(-150,150)


 

    # set xlimit of all axes
    xlimmax = np.max([axx.get_xlim()[1] for axx in ax])
    for curr_ax in ax:
        curr_ax.set_xticks(np.arange(0,20,4))
        curr_ax.set_xlim(-2,xlimmax)
        curr_ax.set_xlabel(f"Synchronized nuclear growth time (hr)")

    # plt.suptitle(f"track {track_id}")
    # fig,ax = adjust_axis_positions(fig,ax,curr_pos=None,width=2,height=2,space=0.6,keep_labels=True)
    return fig,ax



def plot_dxdt_over_time_by_cell_cycle(dfc,ycol,xcol1='index_sequence',plot_type='mean',cell_cycle_width=0.2,cell_cycle_centers=[0.3,0.5,0.7],bin_labels=['Early','Mid','Late'],fw=6.5,fh=8):
    """
    plot dxdt over time

    Parameters
    ----------
    dfc : pd.DataFrame
        dataframe to plot, must be for single colony
    ycol : str
        column to plot
    xcol1 : str, optional
        column to bin by. The default is 'index_sequence'.
    plot_type : str, optional
        'mean' or 'count'. The default is 'mean'.
    cell_cycle_width : float, optional
        width of the cell cycle bin. The default is 0.2.
    cell_cycle_centers : list, optional
        centers of the cell cycle bins. The default is [0.3,0.5,0.7].
    bin_labels : list, optional
        labels for the bins. The default is ['early','mid','late'].
    fw : float, optional
        width of the figure. The default is 6.5.
    fh : float, optional
        height of the figure. The default is 8.

    Returns
    -------
    fig : plt.Figure
    ax : list of plt.Axes
    """
    cell_cycle_bins = [(cc-cell_cycle_width/2,cc+cell_cycle_width/2) for cc in cell_cycle_centers]
    
    nrows = 1
    ncols = len(cell_cycle_bins)+1

    fig,ax = plt.subplots(nrows,ncols,figsize=(fw,fh))
    assert type(ax) == np.ndarray # for mypy

    
    colony = dfc['colony'].values[0] 

    # first plot the whole colony by itself on an axis
    curr_ax = ax[0]
    curr_ax = plot_dfg(dfc,xcol1,ycol,f"{colony.capitalize()}",curr_ax,plot_type=plot_type,colorby='colony')

    curr_ax.legend(loc='lower left',
            fontsize=6,frameon=False,
            markerscale=1,handlelength=1,
            labelspacing=0,
            bbox_to_anchor=(-0.05,-0.1),

            )

    for ri,cell_cycle_bin in enumerate(cell_cycle_bins):
        curr_ax = ax[ri+1]
        curr_ax = plot_dfg(dfc,xcol1,ycol,"",curr_ax,plot_type=plot_type,colorby='colony')
        dfcc = dfc[(dfc['dig_time'] >= cell_cycle_bin[0]) & (dfc['dig_time'] <= cell_cycle_bin[1])]
        
        # top column
        dfcc['cell_cycle'] = ri
        curr_ax = plot_dfg(dfcc,xcol1,ycol,bin_labels[ri],curr_ax,plot_type=plot_type,colorby='k')
        curr_ax.legend(loc='lower left',
                    fontsize=6,frameon=False,
                    markerscale=1,handlelength=0.5,
                    labelspacing=0,
                    bbox_to_anchor=(-0.05,-0.1),
                    )
    return fig,ax