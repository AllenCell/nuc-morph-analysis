import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.analyses.linear_regression.linear_regression_workflow import fit_linear_regression
from nuc_morph_analysis.analyses.linear_regression.select_features import (get_feature_list)


def plot_feature_correlations(df_track_level_features, feature_list, figdir):
    """
    Plot heatmap of feature correlations.   
    
    Parameters
    ----------
    df_track_level_features : pd.DataFrame
        DataFrame containing track level features
    feature_list : list
        List of features to include in the heatmap
        Output from get_feature_list
    figdir : str
        Directory to save the figure

    Returns
    -------
    Figure
    """
    data = df_track_level_features[feature_list]

    plt.rc('font', size=22)
    plt.figure(figsize=(28, 25))
    sns.heatmap(data.corr(), annot=True, fmt=".1f", cmap='BrBG', vmin=-1, vmax=1, cbar_kws={"shrink": 0.5, "pad": 0.02})

    column_names = [get_plot_labels_for_metric(col)[1] for col in data.columns]
    plt.xticks([x + 0.5 for x in range(len(column_names))], column_names)
    plt.yticks([y + 0.5 for y in range(len(column_names))], column_names)
    plt.tight_layout()
    
    save_and_show_plot(f'{figdir}/feature_correlation_heatmap')

    
def run_regression(df_track_level_features, target, features, name, alpha, figdir):
        _, all_test_sc, _ = fit_linear_regression(
            df_track_level_features, 
            cols=get_feature_list(features, target), 
            target=target, 
            alpha=alpha,
            tol=0.04, 
            save_path=figdir,
            save=False,
            multiple_predictions=False
        )
        print(f"Target {target}, Alpha: {alpha}. Feature group: {name}")
        r_squared = round(all_test_sc["Test r$^2$"].mean(), 3)
        std = round(all_test_sc["Test r$^2$"].std(), 3)    
        return {'target': target, 'feature_group': name, 'r_squared': r_squared, 'stdev': std, 'alpha': 0, 'feats_used': get_feature_list(features, target)}
    
def run_regression_workflow(targets, feature_configs, df_track_level_features, figdir, alpha):
    df = pd.DataFrame(columns=['target', 'r_squared', 'stdev', 'feature_group', 'alpha', 'feats_used'])

    for target in targets:
        for name, features in feature_configs.items():
            result = run_regression(df_track_level_features, target, features, name, [alpha], figdir)
            df = df.append(result, ignore_index=True)
            df.to_csv(f"{figdir}r_squared_results.csv")

    df['num_feats_used'] = df['feats_used'].apply(lambda x: len(x))
    
    return df

    
def plot_heatmap(df, figdir):
    """
    Plot heatmap of r_squared values for different feature groups.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing r_squared values for different feature groups
    figdir : str
        Directory to save the figure
    
    Returns
    -------
    Figure
    """
    # Split the 'feature_group' column into two
    df[['start_lifetime', 'intrinsic_extrinsic']] = df['feature_group'].str.split('_', expand=True)
    def replace_values(val):
        if val in ['all', 'features']:
            return 'all_features'
        else:
            return val
    df[['start_lifetime', 'intrinsic_extrinsic']] = df[['start_lifetime', 'intrinsic_extrinsic']].applymap(replace_values)
    for index, row in df.iterrows():
        if row['start_lifetime'] == 'intrinsic':
            df.at[index, 'intrinsic_extrinsic'] = 'intrinsic'
            df.at[index, 'start_lifetime'] = 'both'
        elif row['start_lifetime'] == 'extrinsic':
            df.at[index, 'intrinsic_extrinsic'] = 'extrinsic'
            df.at[index, 'start_lifetime'] = 'both'

    for target, df_target in df.groupby('target'):
        pivot_df = df_target.pivot(index='start_lifetime', columns='intrinsic_extrinsic', values='r_squared')
        pivot_df_std = df_target.pivot(index='start_lifetime', columns='intrinsic_extrinsic', values='stdev')

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(pivot_df, annot=False, cmap='coolwarm', ax=ax, vmin=0, vmax=0.5)
        
        for text_x in range(pivot_df.shape[0]):
            for text_y in range(pivot_df.shape[1]):
                value = pivot_df.iloc[text_x, text_y]
                std_dev = pivot_df_std.iloc[text_x, text_y]
                if not np.isnan(value):
                    ax.text(text_y+0.5, text_x+0.5, f'{value:.2f} ± {std_dev:.2f}', 
                            horizontalalignment='center', 
                            verticalalignment='center')
        
        ax.set_xticklabels(['','Extrinsic','Intrinsic'])
        ax.xaxis.tick_top()
        ax.set_yticklabels(['', 'Both', 'Lifetime', 'Start of growth'], rotation=0)

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='both', length=0)
        title = ax.set_title(f'Target: {get_plot_labels_for_metric(target)[1]}', loc='left')
        title.set_position([-0.1,1])
        save_and_show_plot(f'{figdir}{target}_prediction_r_squared_matrix_alpha_{df.alpha[0]}')


