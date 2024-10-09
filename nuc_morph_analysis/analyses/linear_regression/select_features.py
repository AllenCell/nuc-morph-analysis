FEATURE_GROUPS = {
    'start_intrinsic': [ #intrinic at start of growth
        'volume_at_B', 
        'height_at_B',
        'SA_vol_ratio_at_B',
        'SA_at_B',
        'xy_aspect_at_B',
        'sisters_volume_at_B',
    ],
    
    'lifetime_intrinsic': [ # intrinsic lifetime
        'duration_BC', 
        'volume_at_C',
        'delta_volume_BC',
        'late_growth_rate_by_endpoints',
        'tscale_linearityfit_volume',
        'sisters_duration_BC',
        'sisters_delta_volume_BC',
    ],
    
    'start_extrinsic': [ # extrinsic at start of growth
        'time_at_B', 
        'colony_time_at_B',
        'neighbor_avg_lrm_volume_90um_at_B', 
        'neighbor_avg_lrm_height_90um_at_B',
        'neighbor_avg_lrm_xy_aspect_90um_at_B',
        'neighbor_avg_lrm_mesh_sa_90um_at_B',
        'early_transient_gr_90um',   
    ],
    
    'lifetime_extrinsic': [ # extrinsic lifetime
        'normalized_sum_has_mitotic_neighbor',
        'normalized_sum_has_dying_neighbor',
        'mean_neighbor_avg_lrm_volume_90um', 
        'mean_neighbor_avg_lrm_height_90um',
        'mean_neighbor_avg_lrm_xy_aspect_90um',
        'mean_neighbor_avg_lrm_mesh_sa_90um',
        'mean_neighbor_avg_dxdt_48_volume_90um',
        'mean_neighbor_avg_lrm_2d_area_nuc_cell_ratio_90um',
        ],
}

TARGET_CONTAINTING_FEATS = {
    'duration_BC': [
        'duration_BC',
        'late_growth_rate_by_endpoints',
        ],
    'delta_volume_BC': [
        'volume_at_C',
        'delta_volume_BC',
        'late_growth_rate_by_endpoints',
        ],
    None: [
        '',
    ]
}

TARGET_SETTINGS = {
    'duration_BC': {
        'tolerance': 0.04,
        'max_alpha': 0.7,
    },
    'delta_volume_BC': {
        'tolerance': 0.04,
        'max_alpha': 12.3,
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

