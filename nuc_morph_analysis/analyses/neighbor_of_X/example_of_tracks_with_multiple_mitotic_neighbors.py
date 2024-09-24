#%%
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric

from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, filter_data
from pathlib import Path
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%%
def run_example(df:pd.DataFrame, colony:str = 'medium', column:str = 'has_mitotic_neighbor_breakdown'):
    """
    this code will plot example tracks with the highest, median, and lowest sum of mitotic events

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe containing the data
    colony : str
        the colony to analyze
    column : str
        the column to analyze (e.g. 'has_mitotic_neighbor_breakdown')

    Returns
    -------
    None

    Outputs
    -------
    Saves a figure to the figures/example_sum_of_mitotic_evnts
    """
    
    figdir = Path(__file__).parent / "figures" / "example_sum_of_mitotic_evnts"
    os.makedirs(figdir,exist_ok=True)

    dfc = df.loc[df['colony']==colony].copy()
    dff = filter_data.get_dataframe_of_full_tracks(dfc)
    sum_column = f"sum_{column}"
    dfmax = dff[dff[f"{sum_column}"]==dff[f"{sum_column}"].max()]
    dfmedian = dff[dff[f"{sum_column}"]==dff[f"{sum_column}"].median()]
    dfmin = dff[dff[f"{sum_column}"]==dff[f"{sum_column}"].min()]

    track_id1 = dfmax.track_id.values[0]
    track_id2 = dfmedian.track_id.values[0]
    track_id3 = dfmin.track_id.values[0]

    fig,ax = plt.subplots(nrows=2, ncols=3, figsize=(12,8),constrained_layout=True)
    assert type(ax) == np.ndarray # for mypy
    for ti,track_id in enumerate([track_id1,track_id2,track_id3]):

        dft = dff[dff['track_id']==track_id]
        xscale,xlabel,xunit,_ = get_plot_labels_for_metric('index_sequence')
        x = dft['index_sequence'].values
        y = dft[column].values
        ax1 = ax[0,ti]
        assert type(ax1) == plt.Axes # for mypy
        ax1.plot(x,y)
        titlestr = f"track_id={track_id}\n{sum_column}={dft[sum_column].values[0]}"
        ax1.set_title(titlestr)
        ax1.set_xlabel(f"{xlabel} {xunit}")
        ax1.set_ylabel(column)

        yscale,ylabel,yunit,_ = get_plot_labels_for_metric('volume')
        x2 = dft['index_sequence'].values * xscale
        y2 = dft['volume'].values *yscale
        ax2 = ax[1,ti]
        assert type(ax2) == plt.Axes # for mypy
        ax2.plot(x2,y2)
        ax2.set_title(f"{track_id} volume")
        ax2.set_xlabel(f"{xlabel} {xunit}")
        ax2.set_ylabel(f"{ylabel} {yunit}")
    savename = f"{colony}-sum_of_{column}_example_tracks.png"
    savepath = str(figdir / savename)
    save_and_show_plot(savepath,file_extension='.png',figure=fig,
                    transparent=False)
    plt.show()

if __name__ == "__main__":
    df = global_dataset_filtering.load_dataset_with_features(dataset='all_baseline')
    run_example(df)