
#%%
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
        'early_transient_gr_whole_colony',
        
        'sum_has_mitotic_neighbor', # extrinsic lifetime
        'sum_has_dying_neighbor',
        'mean_neighbor_avg_lrm_volume_90um', 
        'mean_neighbor_avg_lrm_height_90um',
        'mean_neighbor_avg_lrm_density_90um',
        'mean_neighbor_avg_lrm_xy_aspect_90um',
        'mean_neighbor_avg_lrm_mesh_sa_90um',
        'mean_neighbor_avg_dxdt_48_volume_whole_colony',
        ],
    
    'lineage_feats': [
        'sisters_volume_at_B',
        'sisters_duration_BC',
    ],
}

TARGET_CONTAINTING_FEATS = {
    'duration_BC': [
        ''
        ],
    'volume_at_C': [
        ''
        ],
    'delta_volume_BC': [
        ''
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
    features = []
    for group in feature_group_list:
        features = features + FEATURE_GROUPS[group]
    
    features = [feature for feature in features if feature not in TARGET_CONTAINTING_FEATS[target]]
    
    return features