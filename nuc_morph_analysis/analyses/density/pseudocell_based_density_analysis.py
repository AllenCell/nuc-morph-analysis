#%%
from nuc_morph_analysis.lib.visualization.reference_points import COLONY_COLORS, COLONY_LABELS
import matplotlib.pyplot as plt
import numpy as np
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
from nuc_morph_analysis.lib.preprocessing import filter_data
from pathlib import Path
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import MaxNLocator

#%%
# set figure directory
figdir = Path(__file__).parent / "figures" / "psuedocell_based_density_analysis"
figdir.mkdir(exist_ok=True)

# TEMP: loading local for testing and speed
df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline')
#%% now apply the filtering
# apply minimal filtering to ensure only good segmentations are present
dfm = filter_data.all_timepoints_minimal_filtering(df)

#%% # plot density over time for each colony  along colony time 
import matplotlib
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams.update({'axes.titlesize': 8})
matplotlib.rcParams.update({'axes.labelsize': 8})
matplotlib.rcParams.update({'xtick.labelsize': 8})
matplotlib.rcParams.update({'ytick.labelsize': 8})
matplotlib.rcParams.update({'legend.fontsize': 7})
matplotlib.rcParams.update({'legend.title_fontsize': 8})
matplotlib.rcParams.update({'figure.titlesize': 8})
# make axis linewidth thinner
matplotlib.rcParams.update({'axes.linewidth': 0.5})


x_col = "colony_time"
column_val = 'label_img'
# feature_list = ['2d_area_nuc_cell_ratio','density','2d_area_nucleus','2d_area_pseudo_cell',
#                 '2d_area_cyto','inv_cyto_density','2d_intensity_min_edge','2d_intensity_max_edge','2d_intensity_mean_edge'
#                 ]
feature_list = ['2d_area_nuc_cell_ratio']

figure_size_inches = np.asarray((6, 4))
for y_col in feature_list:
    fig,ax = plt.subplots(figsize=figure_size_inches)

    for colony in ['small','medium','large']:
        
        dfsub = dfm[dfm['colony']==colony].copy()
        dfsub.dropna(subset=[y_col],inplace=True)

        # create a pivot of the dataframe to get a 2d array of track_id x timepoint with each value being the density
        pivot = dfsub.pivot(index=x_col, columns=column_val, values=y_col)
        pivot.head()

        mean = pivot.median(axis=1)
        lower = pivot.quantile(0.05,axis=1)
        upper = pivot.quantile(0.95,axis=1)
        
        xscale_factor, xlabel, xunit, xlimit = get_plot_labels_for_metric(x_col)
        x = mean.index * xscale_factor
        yscale_factor, ylabel, yunit, ylimit = get_plot_labels_for_metric(y_col)
        y = mean.values * yscale_factor
        yl = lower.values * yscale_factor
        yu = upper.values * yscale_factor
        
        ax.plot(x, y, label=COLONY_LABELS[colony], color=COLONY_COLORS[colony],
                    linewidth=0.5)
        ax.fill_between(x, yl, yu, alpha=0.2, color=COLONY_COLORS[colony],
                        edgecolor='none')
        ax.set_xlabel(f"{xlabel} {xunit}")
        ax.set_ylabel(f"{ylabel} {yunit}\n(90% interpercentile range)")
        ax.yaxis.set_major_locator(MaxNLocator(4,steps=[1]))
        ax.set_xticks(np.arange(0,80,10))
        ax.set_xlim([-2,(dfm[x_col].max()*xscale_factor) + 2])

        # make the axis 1.5" x 1" in size (units are inches)
        
        desired_axis_size_inches = np.asarray((1.5,1))
        axis_size_figure_units = desired_axis_size_inches / figure_size_inches
        ax.set_position([0.1,0.1,*axis_size_figure_units])
        


    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5),
               handletextpad=0.5, frameon=False,
               handlelength=0.5, labelspacing=0.5)

    for ext in ['.png','.pdf']:
        save_and_show_plot(
            f"{figdir}/{y_col}_vs_{x_col}_by_colony",
            file_extension=ext,
            dpi=300,
            transparent=False,
            keep_open=True
        )
    plt.show()

#%%
# plot density as a function of nucleus size (and compare to old density metric)
colony='medium'
x_col = '2d_area_nucleus'
for yi,y_col in enumerate(['2d_area_nuc_cell_ratio','density','inv_cyto_density']):

    dfsub = dfm[dfm['colony']==colony].copy()
    dfsub.dropna(subset=[y_col],inplace=True)
    dfsub.dropna(subset=[x_col],inplace=True)

    xscale_factor, xlabel, xunit, xlimit = get_plot_labels_for_metric(x_col)
    yscale_factor, ylabel, yunit, ylimit = get_plot_labels_for_metric(y_col)
    x = dfsub[x_col].values * xscale_factor
    y = dfsub[y_col].values * yscale_factor

    fig,ax = plt.subplots(figsize=(4,3))
    alpha = np.min([1,10/np.sqrt(len(x))])
    ax.scatter(x,y, s=1, alpha=alpha, label=COLONY_LABELS[colony], color=COLONY_COLORS[colony])
    ax.set_xlabel(f"{xlabel} {xunit}")
    ax.set_ylabel(f"{ylabel} {yunit}")


    # add best fit line
    reg = LinearRegression().fit(x.reshape(-1,1), y)
    y_pred = reg.predict(x.reshape(-1,1))
    ax.plot(x, y_pred, color='k', lw=2)
    ax.text(0.05,0.95,f"R2={reg.score(x.reshape(-1,1),y):.2f}",transform=ax.transAxes,ha='left',va='top')

    plt.title(f"{y_col}\nvs{x_col}\nfor {colony}")
    for ext in ['.png','.pdf']:
        save_and_show_plot(
            f"{figdir}/{y_col}_vs_{x_col}_by_colony",
            file_extension=ext,
            dpi=300,
            transparent=True,
        )
    plt.show()

