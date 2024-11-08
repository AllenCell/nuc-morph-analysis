import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.analyses.linear_regression.linear_regression import fit_linear_regression
from nuc_morph_analysis.analyses.linear_regression.select_features import (get_feature_list)

def plot_feature_cluster_correlations(df_track_level_features, feature_list, figdir):
    """
    Plot clustermap of feature correlations.   
    
    Parameters
    ----------
    df_track_level_features : pd.DataFrame
        DataFrame containing track level features
    feature_list : list
        List of features to include in the clustermap
        Output from get_feature_list
    figdir : str
        Directory to save the figure

    Returns
    -------
    Figure
    """
    data = df_track_level_features[feature_list]
    
    cluster_grid = sns.clustermap(data.corr(), vmin=-1, vmax=1, cmap='BrBG', #'vlag' red to blue
                                cbar_pos=(0.4, 0.9, 0.3, 0.02), 
                                cbar_kws={"orientation": "horizontal"},
                                annot=True, fmt=".1f", annot_kws={"size": 12},
                                figsize=(18, 18))

        
    # Hide the dendrograms
    cluster_grid.ax_row_dendrogram.set_visible(False)
    cluster_grid.ax_col_dendrogram.set_visible(False)
    
    # Get the reordered labels using the dendrogram information
    reordered_column_index = cluster_grid.dendrogram_col.reordered_ind
    reordered_labels = [get_plot_labels_for_metric(data.columns[i])[1] for i in reordered_column_index]
    
    # add a number to the end of each label
    reordered_labels = [f'{label} ({i+1})' for i, label in enumerate(reordered_labels)]
    
    # for the bottom label just use numbers 
    numbered_labels = [f'{i+1}' for i in range(len(reordered_labels))]

    # Ensure the number of labels matches the number of ticks
    cluster_grid.ax_heatmap.set_xticks([x + 0.5 for x in range(len(numbered_labels))])
    cluster_grid.ax_heatmap.set_xticklabels(numbered_labels, rotation=0)
    cluster_grid.ax_heatmap.set_yticks([y + 0.5 for y in range(len(reordered_labels))])
    cluster_grid.ax_heatmap.set_yticklabels(reordered_labels, rotation=0)
    
    # Adjust the padding between the labels and the heatmap
    cluster_grid.ax_heatmap.tick_params(axis='x', labelsize=12, width=0.7)
    cluster_grid.ax_heatmap.tick_params(axis='y', labelsize=12, width=0.7,  labelright=False, labelleft=True, left=True, right=False)

    save_and_show_plot(f'{figdir}/feature_correlation_clustermap', figure=cluster_grid.fig, bbox_inches='tight', dpi=300)


def run_regression(df_track_level_features, target, features, name, alpha, figdir="figures/"):
    """
    Run linear regression on the given dataset and return the results.

    Parameters:
    ----------
    df_track_level_features (pd.DataFrame): DataFrame containing the track level features.
    target (str): The target variable for regression.
    features (list): List of features to be used for regression.
    name (str): Name of the feature group.
    alpha (list): List of alpha values for regularization.
    figdir (str): Directory path to save the figures.

    Returns:
    --------
    dict: A dictionary containing the target, feature group name, mean R-squared value, 
          standard deviation of R-squared values, alpha value, and the features used.
    """
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
    r_squared = all_test_sc["Test r$^2$"].mean()
    std = all_test_sc["Test r$^2$"].std()
    return {'target': target, 'feature_group': name, 'r_squared': r_squared, 'stdev': std, 'alpha': 0, 'feats_used': get_feature_list(features, target)}
    
def run_regression_workflow(targets, feature_configs, df_track_level_features, alpha, figdir="figures/"):
    """
    Run the regression workflow for multiple targets and feature configurations.

    Parameters:
    ----------
    targets (list): List of target variables for regression.
    feature_configs (dict): Dictionary where keys are feature group names and values are lists of features.
    df_track_level_features (pd.DataFrame): DataFrame containing the track level features.
    figdir (str): Directory path to save the figures and results.
    alpha (float): Alpha value for regularization.

    Returns:
    --------
    pd.DataFrame: DataFrame containing the results of the regression workflow, including target, 
                  R-squared value, standard deviation, feature group, alpha value, and features used.
    """
    df = pd.DataFrame(columns=['target', 'r_squared', 'stdev', 'feature_group', 'alpha', 'feats_used'])

    for target in targets:
        for name, features in feature_configs.items():
            result = run_regression(df_track_level_features, target, features, name, [alpha], figdir)
            df = df.append(result, ignore_index=True)
            df.to_csv(f"{figdir}r_squared_results.csv")

    df['num_feats_used'] = df['feats_used'].apply(lambda x: len(x))
    
    return df

    
def plot_heatmap(df, figdir, cmap='coolwarm'):
    """
    Plot heatmap of r_squared values for different feature groups.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing r_squared values for different feature groups
    figdir : str
        Directory to save the figure
    cmap: str 
        linear colormap
    
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

        sns.heatmap(pivot_df, annot=False, cmap=cmap, ax=ax, vmin=0, vmax=0.5)
        
        first_element = True
        for text_x in range(pivot_df.shape[0]):
            for text_y in range(pivot_df.shape[1]):
                value = pivot_df.iloc[text_x, text_y]
                std_dev = pivot_df_std.iloc[text_x, text_y]
                if not np.isnan(value):
                    color = 'white' if first_element else 'black'
                    ax.text(text_y+0.5, text_x+0.5, f'{value:.2f} ± {std_dev:.2f}', 
                            horizontalalignment='center', 
                            verticalalignment='center',
                            color=color, fontsize=16)
                    first_element = False
        
        ax.set_xticklabels(['','Extrinsic','Intrinsic'])
        ax.xaxis.tick_top()
        ax.set_yticklabels(['', 'Both', 'Lifetime', 'Start of growth'], rotation=0)

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='both', length=0)
        title = ax.set_title(f'Target: {get_plot_labels_for_metric(target)[1]}', loc='left')
        title.set_position([-0.1,1])
        save_and_show_plot(f'{figdir}{target}_prediction_r_squared_matrix_alpha_{df.alpha[0]}',  bbox_inches='tight')


def plot_feature_contribution(coef_alpha, test_sc, perms, target, alpha, fig_height, figdir):
    """
    For a given target, plot feature importance for each feature in the linear model at a specified alpha.
    Features that touch 0 are considered not important and are excluded from the plot. 
    
    Paramaters
    -------
    coef_alpha: pd.DataFrame
        DataFrame containing the coefficient importance for each feature
    test_sc: pd.DataFrame
        DataFrame containing the test r2 scores
    perms: pd.DataFrame
        DataFrame containing the permutation test results
    target: str
        Prediction feature
    alpha: int
        Regularization parameter
    fig_height: int
        Height of the figure based on number of important features
    save_path: str
        Path to save the plot
        
    Returns
    -------
    Figure
    """
    p_value = round(perms["p_value"].item(), 3)
    test_r2_mean = round(test_sc["Test r$^2$"].mean(), 2)
    test_r2_std = round(test_sc["Test r$^2$"].std() / 2, 2)
    
    for col, df_col in coef_alpha.groupby("Column"):
        lower_bound = df_col["Coefficient Importance"].mean() - df_col["Coefficient Importance"].std()
        upper_bound = df_col["Coefficient Importance"].mean() + df_col["Coefficient Importance"].std()
        if lower_bound < 0 and upper_bound > 0 or df_col["Coefficient Importance"].mean() == 0:
            coef_alpha = coef_alpha[coef_alpha["Column"] != col] # if coeff importance is 0, dont plot feature
    
    coef_alpha['Magnitude coefficient importance'] = abs(coef_alpha['Coefficient Importance'])
    coef_alpha['Sign'] = coef_alpha['Coefficient Importance'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
    
    coef_alpha['Mean Magnitude'] = coef_alpha.groupby('Column')['Magnitude coefficient importance'].transform('mean')
    coef_alpha = coef_alpha.sort_values('Mean Magnitude', ascending=False).drop(columns=['Mean Magnitude'])
    
    plt.figure(figsize=(4,fig_height*.5))
    ax = sns.barplot(
        data=coef_alpha,
        y="Column",
        x="Magnitude coefficient importance",
        hue="Sign",
        palette={'Positive': '#156082', 'Negative': 'grey'},
        errorbar="sd",
        width=0.7, 
        native_scale=True)
    
    for patch in ax.patches:
            patch.set_edgecolor('black') 
            patch.set_linewidth(1.5)
            current_height = patch.get_height()
            desired_height = 0.7
            patch.set_height(desired_height)
            patch.set_y(patch.get_y() + (current_height - desired_height) * 0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.ylabel("")
    plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    if target == 'delta_volume_BC':
        ax.get_legend().remove()
    plt.title(f"Target: {get_plot_labels_for_metric(target)[1]}, alpha={alpha}, test r\u00B2={test_r2_mean}±{test_r2_std}, P={p_value}")
    label_list = [get_plot_labels_for_metric(col)[1] for col in coef_alpha["Column"].unique()]
    
    plt.yticks(ticks=range(len(label_list)), labels=label_list)
    save_and_show_plot(f'{figdir}/coefficients_{target}_alpha_{alpha}', dpi=300, bbox_inches='tight')