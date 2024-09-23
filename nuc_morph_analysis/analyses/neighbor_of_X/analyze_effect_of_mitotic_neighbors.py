#%%
# this script attempts to correlate the number of mitotic 
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering,filter_data
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#%%
df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline',load_local=True)
#%%
dfc = df[df['colony'] == 'medium']
dft = filter_data.track_level_features(dfc)

#%%
# ask how correlated number of mitotic neighbors is with other features
mitotic_event_features = [
    'number_of_frame_of_breakdown_neighbors',
    'has_mitotic_neighbor_breakdown',
    'has_dying_neighbor',
    'number_of_frame_of_death_neighbors'
]

# set figure directory
resolution_level = 1
figdir = Path(__file__).parent / "figures" / "analysis_of_neighbor_events_with_features"
os.makedirs(figdir,exist_ok=True)


ycol_list = ['volume_fold_change_BC','duration_BC']
for ycol in ycol_list:
    nrows = 1
    ncols = len(mitotic_event_features)
    fig,ax = plt.subplots(nrows,ncols,figsize=(ncols*3,nrows*3), layout='constrained')
    assert type(ax) == np.ndarray # for mypy
    for fi,feature in enumerate(mitotic_event_features):
        xcol = f"sum_{feature}"
        xscale,xlabel,xunit,_ = get_plot_labels_for_metric(xcol)
        yscale,ylabel,yunit,_ = get_plot_labels_for_metric(ycol)
        
        x = dft[xcol] * xscale
        y = dft[ycol]

        # fit a linear regression
        model = LinearRegression()
        model.fit(x.values.reshape(-1,1),y)
        y_pred = model.predict(x.values.reshape(-1,1))
        r2 = model.score(x.values.reshape(-1,1),y)

        curr_ax = ax[fi]
        assert type(curr_ax) == plt.Axes # for mypy
        curr_ax.scatter(x,y)
        curr_ax.set_xlabel(f'{xlabel} {xunit}')
        curr_ax.set_ylabel(f'{ylabel} {yunit}')
        curr_ax.set_title(f'{feature}\nvs\n{ycol}')
        curr_ax.plot(x,y_pred,'--')
        curr_ax.text(0.05,0.95,f'R^2 = {r2:.2f}',transform=curr_ax.transAxes,
                             va='top',ha='left',fontsize=8)
    savename = figdir / f'{ycol}_vs_neighbor_event_features.png'
    save_and_show_plot(str(savename),file_extension='.png',figure=fig,transparent=False)
    plt.show()

