# %%
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, load_data, filter_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nuc_morph_analysis.lib.visualization.reference_points import COLONY_COLORS
# %%
df = global_dataset_filtering.load_dataset_with_features(remove_growth_outliers=False)

#%%
interval = load_data.get_dataset_time_interval_in_min("all_baseline")
# %%
for colony, df_colony in df.groupby('colony'):
    x_values = []
    for tid, dft in df_colony.groupby('track_id'):
        if dft['termination'].max() == 2:
            x_values.append(dft['index_sequence'].max() * interval/60)
    bins = range(0, 48 + 1, 1)
    plt.hist(x_values, bins=bins, 
             label=f'{colony}, N = {len(x_values)}', 
             color=COLONY_COLORS[colony], alpha=.5)
    plt.legend()
    plt.xlabel('Time (hrs)')
    plt.ylabel('Count of cell death events')
    plt.ylim(0,30)
    plt.show()
#%%
for colony, df_colony in df.groupby('colony'):
    x_values = []
    for tid, dft in df_colony.groupby('track_id'):
        if dft['termination'].max() == 2:
            x_values.append(dft['colony_time'].max() * interval/60)
    bins = range(0, 72 + 1, 1)
    plt.hist(x_values, bins=bins, 
            label=f'{colony}, N = {len(x_values)}', 
            color=COLONY_COLORS[colony], alpha=.5)
plt.legend()
plt.xlabel('Aligned colony time (hrs)')
plt.ylabel('Count of cell death events')
plt.ylim(0,30)
plt.show()

# %%
for colony, df_colony in df.groupby('colony'):
    frame = []
    
    max_index_sequence_rows = df_colony.loc[df_colony.groupby('track_id')['index_sequence'].idxmax()]
    for frame, dft in max_index_sequence_rows.groupby('index_sequence'):
        apoptosis_count = (dft['termination'] == 2).sum()
        plt.scatter(frame * interval, apoptosis_count)