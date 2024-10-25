#%%
import warnings
import numpy as np
import pandas as pd
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, filter_data
from nuc_morph_analysis.analyses.linear_regression.linear_regression_workflow import fit_linear_regression
from nuc_morph_analysis.analyses.linear_regression.analysis_plots import (run_regression_workflow,
                                                                          plot_feature_cluster_correlations, 
                                                                          plot_heatmap, 
                                                                          plot_feature_contribution)
from nuc_morph_analysis.analyses.linear_regression.select_features import (get_feature_list, 
                                                                           TARGET_SETTINGS)

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action="ignore", category=FutureWarning)

#%%
df_all = global_dataset_filtering.load_dataset_with_features()
df_full = filter_data.all_timepoints_full_tracks(df_all)
df_track_level_features = filter_data.track_level_features(df_full)

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
EXTENDED_WORKFLOW = False


#%% Preprocess dataframe to ensure same N for all analysis (lineage features are being used)
dropna_cols = get_feature_list(CONFIG['all_features'], None)
data = df_track_level_features.dropna(subset=dropna_cols)
print(f"Number of tracks: {len(data)}")

#%% Create maxtrix of r squared values
df = run_regression_workflow(TARGETS, CONFIG, data, FIGDIR, alpha=0)
plot_heatmap(df, FIGDIR, 'YlOrRd')

#%% Plot feature importance
for target in ['duration_BC', 'delta_volume_BC']: 
    df_alpha, df_test, df_coeff = fit_linear_regression(data, 
                                                        cols=get_feature_list(CONFIG['all_features'], target), 
                                                        target=target, alpha=[TARGET_SETTINGS[target]['max_alpha']],
                                                        tol=TARGET_SETTINGS[target]['tolerance'], save_path=FIGDIR, save=False)
        
    plot_feature_contribution(df_alpha, df_test, df_coeff, target, TARGET_SETTINGS[target]['fig_height'], FIGDIR)

#%% Plot feature correlations
plot_feature_cluster_correlations(df_track_level_features, get_feature_list(CONFIG['all_features'], None), FIGDIR)

#%% Create movie of increasing alpha until tolerance of 0.05 is reached
if EXTENDED_WORKFLOW:
    for target in ['duration_BC', 'delta_volume_BC']:
        fit_linear_regression(
            data, 
            cols=get_feature_list(CONFIG['all_features'], target), 
            target=target, 
            alpha=np.arange(0, 15, 0.2, dtype=float),
            tol=TARGET_SETTINGS[target]['tolerance'], 
            save_path=FIGDIR, 
            save=True)