# %%
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
from nuc_morph_analysis.analyses.cell_health import cell_health_plots

# %%
df = global_dataset_filtering.load_dataset_with_features()

# %% 
cell_health_plots.plot_event_histogram(df, 'cell_death', 'cell_health/figures/')
    
# %%
