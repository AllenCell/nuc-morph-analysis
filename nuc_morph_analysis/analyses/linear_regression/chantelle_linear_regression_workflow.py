#%%
import warnings
import numpy as np
import pandas as pd
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, filter_data
from nuc_morph_analysis.analyses.linear_regression.linear_regression_workflow import fit_linear_regression
from nuc_morph_analysis.analyses.linear_regression.select_features import (get_feature_list,
                                                                           plot_feature_correlations, 
                                                                           TARGET_SETTINGS)

import seaborn as sns
import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action="ignore", category=FutureWarning)

#%%
df_all = global_dataset_filtering.load_dataset_with_features()
df_full = filter_data.all_timepoints_full_tracks(df_all)
df_track_level_features = filter_data.track_level_features(df_full)

#%% 
feature_list = get_feature_list(['features', 'lineage_feats'], None)
plot_feature_correlations(df_track_level_features, feature_list, "linear_regression/figures")


# %%
config = {
    'duration_BC': [
        {'name': 'all_features', 'features': ['start_intrinsic', 'lifetime_intrinsic', 'start_extrinsic', 'lifetime_extrinsic']},
        {'name': 'start_intrinsic', 'features': ['start_intrinsic']},
        {'name': 'lifetime_intrinsic', 'features': ['lifetime_intrinsic']},
        {'name': 'start_extrinsic', 'features': ['start_extrinsic']},
        {'name': 'lifetime_extrinsic', 'features': ['lifetime_extrinsic']}
    ],
    'delta_volume_BC': [
        {'name': 'all_features', 'features': ['start_intrinsic', 'lifetime_intrinsic', 'start_extrinsic', 'lifetime_extrinsic']},
        {'name': 'start_intrinsic', 'features': ['start_intrinsic']},
        {'name': 'lifetime_intrinsic', 'features': ['lifetime_intrinsic']},
        {'name': 'start_extrinsic', 'features': ['start_extrinsic']},
        {'name': 'lifetime_extrinsic', 'features': ['lifetime_extrinsic']}
    ]
}
df = pd.DataFrame(columns=['target', 'r_squared', 'stdev', 'feature_group', 'alpha', 'feats_used'])

def run_regression(target, features, name,):
    _, all_test_sc, _ = fit_linear_regression(
        df_track_level_features, 
        cols=get_feature_list(features, target), 
        target=target, 
        alpha=[0],
        tol=0.4, 
        save_path=f"./figures/r_squared_matrix/{name}/", 
        save=False,
        multiple_predictions=False
    )
    print(f"Target {target}, Alpha: 0. Feature group: {name}")
    
    r_squared = round(all_test_sc["Test r$^2$"].mean(), 3)
    std = round(all_test_sc["Test r$^2$"].std(), 3)
    
    return {'target': target, 'feature_group': name, 'r_squared': r_squared, 'stdev': std, 'alpha': 0, 'feats_used': get_feature_list(features, target)}

for target, configs in config.items():
    for config in configs:
        result = run_regression(target, **config)
        df = df.append(result, ignore_index=True)
        df.to_csv(f"./figures/r_squared_matrix/r_squared_results.csv")
#%%
print(df.iloc[:, :5])

#%%
for target in ['duration_BC', 'delta_volume_BC',]:
    fit_linear_regression(
        df_track_level_features, 
        cols=get_feature_list(['start_intrinsic', 'lifetime_intrinsic', 'start_extrinsic', 'lifetime_extrinsic'], target), 
        target=target, 
        alpha=np.arange(0, 15, 0.1, dtype=float),
        tol=TARGET_SETTINGS[target]['tolerance'], 
        save_path="./figures/test_tolerance/", 
        save=True
    )
    print(f"Finished {target} with lineage")

#%%
# Split the 'feature_group' column into two
df[['start_lifetime', 'intrinsic_extrinsic']] = df['feature_group'].str.split('_', expand=True)
def replace_values(val):
    if val in ['all', 'features']:
        return 'all_features'
    else:
        return val
df[['start_lifetime', 'intrinsic_extrinsic']] = df[['start_lifetime', 'intrinsic_extrinsic']].applymap(replace_values)

# Then create the pivot table and heatmap as before
for target, df_target in df.groupby('target'):
    pivot_df = df_target.pivot(index='start_lifetime', columns='intrinsic_extrinsic', values='r_squared')

    fig, ax = plt.subplots(figsize=(10, 8))  # Create a figure and a set of subplots

    sns.heatmap(pivot_df, annot=True, cmap='coolwarm', ax=ax, vmin=0, vmax=0.5)
    
    ax.set_xticklabels(['','Extrinsic','Intrinsic'])
    ax.xaxis.tick_top()
    ax.set_yticklabels(['', 'Average over growth', 'Start of growth'], rotation=0)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='both', length=0)
    title = ax.set_title(f'Target: {get_plot_labels_for_metric(target)[1]}', loc='left')
    title.set_position([-0.3,1])
    plt.show()



# %%
