from nuc_morph_analysis.lib.visualization.reference_points import COLONY_COLORS
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

FONTSIZE=12
        
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
            
            df_last = df_colony.loc[df_colony.groupby('track_id')['index_sequence'].idxmax()]
            for hour_bin, dft in df_colony.groupby(df_colony['index_sequence'] // 12):
                df_sub_last = df_last.loc[df_last['index_sequence'] // 12 == hour_bin]
                index_sequence_list.append(hour_bin)
                event_count.append((df_sub_last['termination'] == 2).sum())
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
        
        fig, ax = plt.subplots(1, 3, figsize=(17,5))
        ax[0].bar(np.array(index_sequence_list), event_count, label=colony,
                color=COLONY_COLORS[colony], alpha=.75)
        ax[0].legend(fontsize=FONTSIZE, loc='upper left')
        ax[0].set_ylabel(f'Count of cell {event_label} events (Total N={sum(event_count)})', fontsize=FONTSIZE)
        ax[0].set_xlabel('Time (hr)', fontsize=FONTSIZE)
        ax[0].set_ylim(lim1)
        ax[0].tick_params(labelsize=FONTSIZE)
        
        ax[1].bar(np.array(index_sequence_list), num_cells, label=colony,
        color=COLONY_COLORS[colony], alpha=.75)
        ax[1].set_ylabel('Number of cells in FOV', fontsize=FONTSIZE)
        ax[1].set_xlabel('Time (hr)', fontsize=FONTSIZE)
        ax[1].set_ylim(0,1200)
        ax[1].tick_params(labelsize=FONTSIZE)
        
        ax[2].bar(np.array(index_sequence_list), percent_event, label=colony, 
                  color=COLONY_COLORS[colony], alpha=.75)
        ax[2].set_ylim(lim2)
        ax[2].tick_params(labelsize=FONTSIZE)
        
        ax[2].set_ylabel(f'Occurence of {event_label}\nnormalized by number of cells in FOV (%)', fontsize=FONTSIZE)  
        ax[2].set_xlabel('Time (hr)', fontsize=FONTSIZE) 
        
        save_and_show_plot(f'{figdir}/{event_label}_histogram_{colony}')
        
def cell_death_colony_time(df, figdir):
    plt.figure(figsize=(5,5))
    for colony, df_colony in df.groupby('colony'):
        index_sequence_list = []
        apoptosis_count = []
        num_cells = []
        
        df_last = df_colony.loc[df_colony.groupby('track_id')['colony_time'].idxmax()]
        for hour_bin, dft in df_colony.groupby(df_colony['colony_time'] // 12):
            df_sub_last = df_last.loc[df_last['colony_time'] // 12 == hour_bin]
            index_sequence_list.append(hour_bin)
            apoptosis_count.append((df_sub_last['termination'] == 2).sum())
            num_cells.append(dft.track_id.nunique())
            
        percent_cell_death = (np.array(apoptosis_count) / np.array(num_cells)) * 100

        plt.bar(np.array(index_sequence_list), percent_cell_death, label=colony, 
                color=COLONY_COLORS[colony], alpha=.55)
        
    plt.legend(fontsize=FONTSIZE)
    plt.xlabel('Aligned colony time (hr)', fontsize=FONTSIZE)
    plt.ylabel('Percent of cell that die per hour', 
                fontsize=FONTSIZE)  
    plt.ylim(0,2.5)
    save_and_show_plot(f'{figdir}/cell_death_aligned_colony_time')
    
def cell_death_feeding_controls(df_list, dataset_list, interval):
    for df, colony in zip(df_list, dataset_list):
        plt.figure(figsize=(5,5))
        plt.hist(df.Slice * interval/60, bins=range(0, 48 + 1, 1), alpha=.75, 
                label=colony, color=COLONY_COLORS[colony])
        plt.xlabel('Time (hrs)', fontsize=FONTSIZE)
        plt.ylabel(f'Count of cell death events, (Total N={len(df)})', fontsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE, loc='upper left')
        plt.ylim(0,27)
        plt.show()
        
