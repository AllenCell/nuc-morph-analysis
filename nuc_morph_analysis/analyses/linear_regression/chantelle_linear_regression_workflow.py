#%%
import warnings
import numpy as np
import pandas as pd
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, filter_data
from nuc_morph_analysis.analyses.linear_regression.linear_regression_workflow import fit_linear_regression
from nuc_morph_analysis.analyses.linear_regression.select_features import (get_feature_list,
                                                                           plot_feature_correlations, 
                                                                           TARGET_SETTINGS)

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action="ignore", category=FutureWarning)

#%%
df_all = global_dataset_filtering.load_dataset_with_features()
df_full = filter_data.all_timepoints_full_tracks(df_all)
df_track_level_features = filter_data.track_level_features(df_full)

#%% 
feature_list = get_feature_list(['features', 'lineage_feats'], None)
plot_feature_correlations(df_track_level_features, feature_list, "linear_regression/figures")

#%%
for target in ['volume_at_C', 'delta_volume_BC', 'duration_BC']:
    fit_linear_regression(
        df_track_level_features, 
        cols=get_feature_list(['features'], target), 
        target=target, 
        alpha=np.arange(0, 15, 0.1, dtype=float),
        tol=TARGET_SETTINGS[target]['tolerance'], 
        save_path="./figures/feats/", 
        save=True
    )
    print(f"Finished {target}")
#%%
for target in ['volume_at_C', 'delta_volume_BC', 'duration_BC']:
    fit_linear_regression(
        df_track_level_features, 
        cols=get_feature_list(['features', 'lineage_feats'], target), 
        target=target, 
        alpha=np.arange(0, 15, 0.1, dtype=float),
        tol=TARGET_SETTINGS[target]['tolerance'], 
        save_path="./figures/feats_plus_lineage/", 
        save=True
    )
    print(f"Finished {target} with lineage")

# %%
