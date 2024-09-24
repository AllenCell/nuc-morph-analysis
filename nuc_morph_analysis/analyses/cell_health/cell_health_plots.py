from nuc_morph_analysis.lib.visualization.reference_points import COLONY_COLORS
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# set up plot parameters and figure saving directory
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 18
        
def plot_event_histogram(df, event_type, figdir):
    '''
    Plots event counts, number of cells, and percent events per hour for each colony.
    
    Parameters
    ----------
    df : DataFrame
        Unfiltered dataset with features   
    event_type : str
        Type of event to plot ('cell_death' or 'cell_division')
    figdir : str
        Directory to save the figure
    
    Results
    -------
    Plots are saved in the figdir
    '''
    for colony, df_colony in df.groupby('colony'):
        index_sequence_list = []
        event_count = []
        num_cells = []
        
        if event_type == 'cell_death':
            df_event = df_colony.loc[df_colony.groupby('track_id')['index_sequence'].idxmax()]
            event_label = 'death'
            lim1 = (0, 27)
            lim2 = (0, 2.5)
            
            for hour_bin, dft in df_colony.groupby(df_colony['index_sequence'] // 12):
                df_sub_event = df_event.loc[df_event['index_sequence'] // 12 == hour_bin]
                index_sequence_list.append(hour_bin)
                event_count.append((df_sub_event['termination'] == 2).sum())
                num_cells.append(dft.track_id.nunique())
            
        if event_type == 'cell_division':
            df_event = df_colony[df_colony['index_sequence']==df_colony['predicted_breakdown']]
            df_event = df_event[df_event['termination']!=2]
            event_label = 'division'
            lim1 = (0, 51)
            lim2 = (0, 8.5)
            
            df_divide = df_colony[df_colony['index_sequence']==df_colony['predicted_breakdown']]
            df_divide = df_divide[df_divide['termination']!=2]
            for hour_bin, dft in df_colony.groupby(df_colony['index_sequence'] // 12):
                df_sub = df_divide.loc[df_divide['index_sequence'] // 12 == hour_bin]
                index_sequence_list.append(hour_bin)
                event_count.append(df_sub.track_id.nunique())
                num_cells.append(dft.track_id.nunique())
            
        percent_event = (np.array(event_count) / np.array(num_cells)) * 100
        
        fig, ax = plt.subplots(1, 3, figsize=(15,5))
        plt.subplots_adjust(wspace=0.3)
        ax[0].bar(np.array(index_sequence_list), event_count, label=colony.capitalize(),
                color=COLONY_COLORS[colony], alpha=.75)
        ax[0].legend(loc='upper left',  frameon=False)
        ax[0].set_ylabel(f'Count of cell {event_label} events (N={sum(event_count)})')
        ax[0].set_xlabel('Time (hr)')
        ax[0].set_ylim(lim1)
        ax[0].tick_params()
        
        ax[1].bar(np.array(index_sequence_list), num_cells, label=colony,
        color=COLONY_COLORS[colony], alpha=.75)
        ax[1].set_ylabel('Count of cells in FOV')
        ax[1].set_xlabel('Time (hr)')
        ax[1].set_ylim(0,1200)
        ax[1].tick_params()
        
        ax[2].bar(np.array(index_sequence_list), percent_event, label=colony, 
                  color=COLONY_COLORS[colony], alpha=.75)
        ax[2].set_ylim(lim2)
        ax[2].tick_params()
        
        ax[2].set_ylabel(f'Occurence of {event_label} normalized\nby number of cells in FOV (%)')  
        ax[2].set_xlabel('Time (hr)') 
        plt.tight_layout()
        
        save_and_show_plot(f'{figdir}/{event_label}_histogram_{colony}')
    

def subset_main_dataset(df, df_full, index_sequence_threshold=480):
    '''
    Timepoints after 40 hours into the timelapse have the greatest occurrence of cell death. 
    To test the effect of this data on our results, we used this function to omit the 
    final timepoints of the movie and re-run analysis workflows. We saw no significant differences.
    
    Parameters
    ----------
    df: pandas.DataFrame
        The main dataframe to be filtered.
    df_full: pandas.DataFrame
        The full dataframe used to identify tracks to drop.

    Returns
    ------
    df_sub: pandas.DataFrame
        Filtered dataframe with timepoints less than 40 hours.
    '''
    # remove full tracks from analysis dataset that go to the end of the movie
    df_full_to_drop = df_full[df_full['index_sequence']>index_sequence_threshold]
    tracks_to_drop = df_full_to_drop.track_id.unique()

    df_copy = df.copy()
    # set 'is_full_track' to False for tracks in the tracks_to_drop list
    df_copy.loc[df['track_id'].isin(tracks_to_drop), 'is_full_track'] = False

    # remove timepoints after 40 hours
    df_sub = df_copy[df_copy['index_sequence']<=index_sequence_threshold]
    return df_sub