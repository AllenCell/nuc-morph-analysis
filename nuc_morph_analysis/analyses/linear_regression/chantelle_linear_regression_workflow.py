#%%
import warnings
import numpy as np
import pandas as pd
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, filter_data
from nuc_morph_analysis.analyses.linear_regression.linear_regression_workflow import fit_linear_regression
from nuc_morph_analysis.analyses.linear_regression.analysis_plots import (run_regression_workflow, plot_feature_correlations,
                                                                            plot_heatmap)
from nuc_morph_analysis.analyses.linear_regression.select_features import (get_feature_list, 
                                                                           TARGET_SETTINGS)
from nuc_morph_analysis.analyses.linear_regression.linear_regression_workflow_greedy_removal import main
pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action="ignore", category=FutureWarning)

#%%
df_all = global_dataset_filtering.load_dataset_with_features()
df_full = filter_data.all_timepoints_full_tracks(df_all)
df_track_level_features = filter_data.track_level_features(df_full)

#%%
FIGDIR='linear_regression/figures/'
TARGETS = ['duration_BC', 'delta_volume_BC']
CONFIG = {
    'all_features': ['start_intrinsic', 'lifetime_intrinsic', 'start_extrinsic', 'lifetime_extrinsic'],
    'start_intrinsic': ['start_intrinsic'],
    'lifetime_intrinsic': ['lifetime_intrinsic'],
    'start_extrinsic': ['start_extrinsic'],
    'lifetime_extrinsic': ['lifetime_extrinsic'],
    'intrinsic': ['start_intrinsic', 'lifetime_intrinsic'],
    'extrinsic': ['start_extrinsic', 'lifetime_extrinsic'],
}

#%%
plot_feature_correlations(df_track_level_features, get_feature_list(CONFIG['all_features'], None), FIGDIR)

#%% preprocess dataframe to ensure same N for all analysis
dropna_cols = get_feature_list(CONFIG['all_features'], None)
data = df_track_level_features.dropna(subset=dropna_cols)
print(f"Number of tracks: {len(data)}")

#%% Create maxtrix of r squared values
df = run_regression_workflow(TARGETS, CONFIG, data, FIGDIR, alpha=0)
plot_heatmap(df, FIGDIR)

#%% Create movie of increasing alpha
for target in ['duration_BC', 'delta_volume_BC']:
    fit_linear_regression(
        data, 
        cols=get_feature_list(CONFIG['all_features'], target), 
        target=target, 
        alpha=np.arange(0, 15, 0.2, dtype=float),
        tol=TARGET_SETTINGS[target]['tolerance'], 
        save_path=FIGDIR, 
        save=True
    )
    
# %% Greedy removal
for target in TARGETS:
    main(cols=get_feature_list(CONFIG['all_features'], target),
        target=target,
        alpha_range=np.arange(0.5, 15, 0.2, dtype=float),
        tolerance=TARGET_SETTINGS[target]['tolerance'],
        save_path=FIGDIR,
        max_iterations=100,
        preloaded_dataframe=data)