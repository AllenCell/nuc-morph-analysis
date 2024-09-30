import seaborn as sns
import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot

FEATURE_GROUPS = {
    'features': [
        'volume_at_B', #intrinic at start of growth
        'height_at_B',
        'SA_vol_ratio_at_B',
        'SA_at_B',
        'xy_aspect_at_B',
        'time_at_B', 
        'colony_time_at_B',
        'density_at_B',

        'neighbor_avg_lrm_volume_90um_at_B', # extrinsic at start of growth
        'neighbor_avg_lrm_height_90um_at_B',
        'neighbor_avg_lrm_density_90um_at_B',
        'neighbor_avg_lrm_xy_aspect_90um_at_B',
        'neighbor_avg_lrm_mesh_sa_90um_at_B',
        'early_transient_gr_90um',
        
        'duration_BC', # intrinsic lifetime
        'volume_at_C',
        'delta_volume_BC',
        'late_growth_rate_by_endpoints',
        'tscale_linearityfit_volume',
        
        'normalized_sum_has_mitotic_neighbor', # extrinsic lifetime
        'normalized_sum_has_dying_neighbor',
        'mean_neighbor_avg_lrm_volume_90um', 
        'mean_neighbor_avg_lrm_height_90um',
        'mean_neighbor_avg_lrm_density_90um',
        'mean_neighbor_avg_lrm_xy_aspect_90um',
        'mean_neighbor_avg_lrm_mesh_sa_90um',
        'mean_neighbor_avg_dxdt_48_volume_90um',
        ],
    
    'lineage_feats': [
        'sisters_volume_at_B',
        'sisters_duration_BC',
    ],
}

TARGET_CONTAINTING_FEATS = {
    'duration_BC': [
        'duration_BC',
        'late_growth_rate_by_endpoints', 
        ],
    'volume_at_C': [
        'volume_at_C',
        'delta_volume_BC',
        'late_growth_rate_by_endpoints',
        ],
    'delta_volume_BC': [
        'volume_at_C',
        'delta_volume_BC',
        'late_growth_rate_by_endpoints',
    ]
}

TARGET_SETTINGS = {
    'duration_BC': {
        'tolerance': 0.08,
    },
    'volume_at_C': {
        'tolerance': 0.05,
    },
    'delta_volume_BC': {
        'tolerance': 0.08,
    }
}

def get_feature_list(feature_group_list, target):
    """
    Get feature list to include in linear model. 
    Gets full features list and excludes ones that contain target information. 
    
    Parameters
    ----------
    feature_group_list: list
        List of feature groups to include in the feature list
    target: str
        Target variable to predict
        
    Returns
    -------
    features: list
        List of features to include in the linear model
    """
    features = []
    for group in feature_group_list:
        features = features + FEATURE_GROUPS[group]
    
    if target is not None: 
        features = [feature for feature in features if feature not in TARGET_CONTAINTING_FEATS[target]]
    
    return features

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
    plt.figure(figsize=(27, 24))
    sns.heatmap(data.corr(), annot=True, fmt=".1f", cmap='BrBG', vmin=-1, vmax=1, cbar_kws={"shrink": 0.5, "pad": 0.02})

    column_names = [get_plot_labels_for_metric(col)[1] for col in data.columns]
    plt.xticks([x + 0.5 for x in range(len(column_names))], column_names)
    plt.yticks([y + 0.5 for y in range(len(column_names))], column_names)
    plt.tight_layout()
    
    save_and_show_plot(f'{figdir}/feature_correlation_heatmap')