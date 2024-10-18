# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.lib.preprocessing import filter_data, add_times, compute_change_over_time
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot 
from nuc_morph_analysis.lib.visualization import plotting_tools
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric

#%%
# load the data
df = load_dataset_with_features('all_baseline',load_local=True)
df = filter_data.all_timepoints_minimal_filtering(df) # apply minimal filterting
df = compute_change_over_time.run_script(df, dxdt_feature_list = ['volume','fit_volume'], bin_interval_list=[12,24,48]) 


df['dxdt_48_volume_per_V'] = df['dxdt_48_volume'] / df['volume'] # normalize by volume
df['dxdt_48_fit_volume_per_V'] = df['dxdt_48_fit_volume'] / df['volume'] # normalize by volume
df_full = filter_data.all_timepoints_full_tracks(df) # filter to only full tracks


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


#%%
# is there a relationship between dv/dt and volume?

# import sklearn package to fit a linear regression model
from sklearn.linear_model import LinearRegression
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


#%% all time
colony_list = ['small','medium','large','all']
xcol = 'volume'
ycol = 'dxdt_48_volume'
ycol_list = ['dxdt_48_volume','dxdt_48_fit_volume','dxdt_48_volume_per_V','dxdt_48_fit_volume_per_V']
for ycol in ycol_list:
    for colony in colony_list:
        df_colony = df[df.colony == colony] if colony != 'all' else df
        df_colony = df_colony.dropna(subset=[xcol,ycol])


        fig,axlist = plt.subplots(1,1,figsize=(fw,fh))
        axlist = np.asarray([axlist]) if type(axlist) != np.ndarray else axlist
        ax = axlist[0]
        color = plotting_tools.COLONY_COLORS[colony] if colony != 'all' else 'tab:grey'
        xscale,xlabel,xunit,_ = get_plot_labels_for_metric(xcol)
        yscale,ylabel,yunit,_ = get_plot_labels_for_metric(ycol)

        yscale = (0.108**3 / (1/12)) if ycol == 'dxdt_48_volume' else xscale
        yscale = (1 / (1/12)) if ycol == 'dxdt_48_fit_volume' in ycol else yscale

        yscale = (1 / (1/12)) if 'dxdt_48_volume_per_V' in ycol else yscale
        yscale = ((1/0.108**3) / (1/12)) if 'dxdt_48_fit_volume_per_V' in ycol else yscale

        
        x = df_colony[xcol].values * xscale
        y = df_colony[ycol].values * yscale
        reg = LinearRegression().fit(x.reshape(-1,1),y)
        y_pred = reg.predict(x.reshape(-1,1))
        alpha = np.min([1,np.max([1/(np.sqrt(len(x))/10),0.01])])
        ax.scatter(x,y,s=1,c=color,alpha=alpha)
        ax.plot(x,y_pred,'k--',linewidth=0.75)
        ax.set_xlabel(f"{xlabel} {xunit}")
        ax.set_ylabel(f"{ylabel} {yunit}")

        # resize the axis to be 1" wide x 1" tall
        fig,axlist = adjust_axis_positions(fig,axlist,curr_pos=None,width=1,height=1,space=0.075)

        #print the R^2 value and y=mx+b values on plot
        r2 = reg.score(x.reshape(-1,1),y)
        m = reg.coef_[0]
        b = reg.intercept_
        ax.text(0.05,0.95,f"R^2={r2:.2f}",transform=ax.transAxes,
                ha='left',va='top',fontsize=fs)
        ax.text(0.05,0.85,f"y={m:.2f}x+{b:.2f}",transform=ax.transAxes,
                ha='left',va='top',fontsize=fs)


#%% all time
colony_list = ['small','medium','large','all']
xcol = 'volume'
ycol = 'dxdt_48_volume'
ycol_list = ['dxdt_48_volume','dxdt_48_fit_volume','dxdt_48_volume_per_V','dxdt_48_fit_volume_per_V']
for ycol in ycol_list:
    for colony in colony_list:
        df_colony = df[df.colony == colony] if colony != 'all' else df
        df_colony = df_colony.dropna(subset=[xcol,ycol])
        df_colony = df_colony[df_colony.index_sequence == 250]

        
        fig,axlist = plt.subplots(1,1,figsize=(fw,fh))
        axlist = np.asarray([axlist]) if type(axlist) != np.ndarray else axlist
        ax = axlist[0]
        color = plotting_tools.COLONY_COLORS[colony] if colony != 'all' else 'tab:grey'
        xscale,xlabel,xunit,_ = get_plot_labels_for_metric(xcol)
        yscale,ylabel,yunit,_ = get_plot_labels_for_metric(ycol)

        yscale = (0.108**3 / (1/12)) if ycol == 'dxdt_48_volume' else xscale
        yscale = (1 / (1/12)) if ycol == 'dxdt_48_fit_volume' in ycol else yscale

        yscale = (1 / (1/12)) if 'dxdt_48_volume_per_V' in ycol else yscale
        yscale = ((1/0.108**3) / (1/12)) if 'dxdt_48_fit_volume_per_V' in ycol else yscale

        
        x = df_colony[xcol].values * xscale
        y = df_colony[ycol].values * yscale
        reg = LinearRegression().fit(x.reshape(-1,1),y)
        y_pred = reg.predict(x.reshape(-1,1))
        alpha = np.min([1,np.max([1/(np.sqrt(len(x))/10),0.01])])
        ax.scatter(x,y,s=1,c=color,alpha=alpha)
        ax.plot(x,y_pred,'k--',linewidth=0.75)
        ax.set_xlabel(f"{xlabel} {xunit}")
        ax.set_ylabel(f"{ylabel} {yunit}")

        # resize the axis to be 1" wide x 1" tall
        fig,axlist = adjust_axis_positions(fig,axlist,curr_pos=None,width=1,height=1,space=0.075)

        #print the R^2 value and y=mx+b values on plot
        r2 = reg.score(x.reshape(-1,1),y)
        m = reg.coef_[0]
        b = reg.intercept_
        ax.text(0.05,0.95,f"R^2={r2:.2f}",transform=ax.transAxes,
                ha='left',va='top',fontsize=fs)
        ax.text(0.05,0.85,f"y={m:.2f}x+{b:.2f}",transform=ax.transAxes,
                ha='left',va='top',fontsize=fs)
