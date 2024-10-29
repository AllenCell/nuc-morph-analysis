#%%
import numpy as np
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, filter_data
from nuc_morph_analysis.analyses.linear_regression.linear_regression import fit_linear_regression
from nuc_morph_analysis.analyses.linear_regression.select_features import get_feature_list
from nuc_morph_analysis.analyses.linear_regression.analysis_plots import (run_regression_workflow,
                                                                          plot_feature_cluster_correlations, 
                                                                          plot_heatmap, 
                                                                          plot_feature_contribution)

#%%
df_all = global_dataset_filtering.load_dataset_with_features()
df_full = filter_data.all_timepoints_full_tracks(df_all)
df_track_level_features = filter_data.track_level_features(df_full)

#%%
CONFIG = {
    'all_features': ['start_intrinsic', 'lifetime_intrinsic', 'start_extrinsic', 'lifetime_extrinsic'],
    'start_intrinsic': ['start_intrinsic'],
    'lifetime_intrinsic': ['lifetime_intrinsic'],
    'start_extrinsic': ['start_extrinsic'],
    'lifetime_extrinsic': ['lifetime_extrinsic'],
    'intrinsic': ['start_intrinsic', 'lifetime_intrinsic'],
    'extrinsic': ['start_extrinsic', 'lifetime_extrinsic'],
}
FIGDIR='linear_regression/figures/'
TARGETS = ['duration_BC', 'delta_volume_BC']

#%% Preprocess dataframe to ensure same N for all analysis (full tracks with lineage features)
dropna_cols = get_feature_list(CONFIG['all_features'], None)
data = df_track_level_features.dropna(subset=dropna_cols)
print(f"Number of tracks: {len(data)}")

#%% Create maxtrix of r squared values
df = run_regression_workflow(TARGETS, CONFIG, data, alpha=0)
plot_heatmap(df, FIGDIR, 'YlOrRd')

#%%
RECOMPUTE_ALPHA = True
TOLERANCE = 0.05
computed_alpha = {'duration_BC': 1.4,
                  'delta_volume_BC': 10.2}

#%% Recompute maximum alpha when tolerance is reached
if RECOMPUTE_ALPHA:
    for target in ['duration_BC', 'delta_volume_BC']:
        all_coef_alpha, all_test_sc, all_perms = fit_linear_regression(
            data, 
            cols=get_feature_list(CONFIG['all_features'], target), 
            target=target, 
            alpha=np.arange(0, 15, 0.2, dtype=float),
            tol=TOLERANCE, 
            save_path="figures/", 
            save=True)

        max_alpha = all_coef_alpha.alpha.max()
        computed_alpha[target] = max_alpha
        print(f"{target}: {max_alpha}")

#%% Plot feature importance for max alpha when tolerance is reached
for target, fig_height in zip(['duration_BC', 'delta_volume_BC'], [6, 2]): 
    df_alpha, df_test, df_coeff = fit_linear_regression(data, 
                                                        cols=get_feature_list(CONFIG['all_features'], target), 
                                                        target=target, alpha=[computed_alpha[target]],
                                                        tol=TOLERANCE, save_path=FIGDIR, save=False)
    plot_feature_contribution(df_alpha, df_test, df_coeff, target, computed_alpha[target], fig_height, FIGDIR)

#%% Plot feature correlations using all full tracks
plot_feature_cluster_correlations(df_track_level_features, get_feature_list(CONFIG['all_features'], None), FIGDIR)
#%%
