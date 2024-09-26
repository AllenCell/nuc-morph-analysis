from nuc_morph_analysis.lib.preprocessing.load_data import (
    get_dataset_pixel_size,
    get_dataset_time_interval_in_min,
)

from nuc_morph_analysis.lib.preprocessing.compute_change_over_time import (
    BIN_INTERVAL_LIST,
    DXDT_FEATURE_LIST,
    DXDT_PREFIX,
)
from nuc_morph_analysis.lib.preprocessing.add_neighborhood_avg_features import (
    LOCAL_RADIUS_STR_LIST,
    NEIGHBOR_FEATURE_LIST,
    NEIGHBOR_PREFIX,
)
from nuc_morph_analysis.lib.preprocessing.system_info import PIXEL_SIZE_YX_20x


def get_one_to_one_dict(many_to_one_dict):
    """
    Creates a one-to-one dictionary from a many-to-one mapping

    Parameters
    ----------
    many_to_one_dict: dict
        dictionary with tuples as keys indicating many-to-one mappings

    Returns
    ----------
    scale_factor_dict: dict
        dictionary with one-to-one mappings
    """
    one_to_one_dict = {}

    for keys, value in many_to_one_dict.items():
        if isinstance(keys, tuple):
            for key in keys:
                one_to_one_dict[key] = value
        else:
            one_to_one_dict[keys] = value

    return one_to_one_dict


def get_scale_factor_table(dataset="all_baseline"):
    # scaling factors for quantities
    pix_size = get_dataset_pixel_size(dataset)
    time_interval_minutes = get_dataset_time_interval_in_min(dataset)
    dict1 = {
        (
            "height",
            "width",
            "length",
            "distance_from_centroid",
            "max_distance_from_centroid",
        ): pix_size,
        ("mesh_sa"): pix_size**2,
        ("volume", "volume_sub"): pix_size**3,
        ("density", "avg_density", "avg_early_density", "avg_late_density"): 1 / pix_size**2,
        (
            "colony_time",
            "sync_time_Ff",
            "duration_BC",
            "index_sequence",
        ): time_interval_minutes
        / 60,
        ("duration_AB"): time_interval_minutes,
        ("late_growth_rate_by_endpoints"): 60 / time_interval_minutes,
        ("duration_BC"): time_interval_minutes / 60,
        ("late_duration"): 1 / 60,
        ("SA_vol_ratio"): 1 / pix_size,
        ("colony_area"): PIXEL_SIZE_YX_20x**2,
        ("nucleus_colony_area_ratio"): pix_size**2 / PIXEL_SIZE_YX_20x**2,
        ("seg_twoD_zMIP_area"): pix_size**2,
    }

    # add non dxdt columns and other non-traditional columns
    temp_dict = get_one_to_one_dict(dict1)
    hours_per_frame = time_interval_minutes / 60
    for feature in DXDT_FEATURE_LIST:
        dict1.update(
            {
                f"{DXDT_PREFIX}{bin_interval}_{feature}": temp_dict[feature] / (hours_per_frame)
                for bin_interval in BIN_INTERVAL_LIST
            }
        )
    dict1.update(
        {
            f"{DXDT_PREFIX}{bin_interval}_volume_per_V": 1 / (hours_per_frame)
            for bin_interval in BIN_INTERVAL_LIST
        }
    )

    for local_radius_str in LOCAL_RADIUS_STR_LIST:
        for feature in NEIGHBOR_FEATURE_LIST:
            dict1.update({f"{NEIGHBOR_PREFIX}{feature}_{local_radius_str}": temp_dict[feature]})
            dict1.update(
                {
                    f"{NEIGHBOR_PREFIX}{DXDT_PREFIX}{bin_interval}_{feature}_{local_radius_str}": temp_dict[
                        feature
                    ]
                    / (hours_per_frame)
                    for bin_interval in BIN_INTERVAL_LIST
                }
            )
        # add {DXDT_PREFIX}{bin_interval}_volume_per_V
        dict1.update(
            {
                f"{NEIGHBOR_PREFIX}{DXDT_PREFIX}{bin_interval}_volume_per_V_{local_radius_str}": 1
                / (hours_per_frame)
                for bin_interval in BIN_INTERVAL_LIST
            }
        )

    # add name for coordinated_local_growth_fig.py
    for bin_interval in BIN_INTERVAL_LIST:
        for local_radius_str in LOCAL_RADIUS_STR_LIST:
            dict1.update({f"dxdt_t2-dxdt_t1": 1})
            dict1.update({f"dvdt_t2-dvdt_t1_neighbors_{bin_interval}_{local_radius_str}": 1})
            dict1.update({f"dvdt_t2-dvdt_t1_self_{bin_interval}_{local_radius_str}": 1})

    return dict1


# labels for plot axes
LABEL_TABLE = {
    # Times and durations
    ("sync_time_Ff", "index_sequence"): "Time",
    "normalized_time": "Normalized Interphase Time",
    "colony_time": "Aligned Colony Time",
    ("Ff", "frame_formation"): "Formation",
    ("frame_inflection", "frame_transition"): "Transtion",
    ("Fb", "frame_breakdown"): "Breakdown",
    "time_at_A": "Time at Formation",
    "time_at_B": "Starting movie time",
    "time_at_C": "Movie Time at Breakdown",
    "colony_time_at_A": "Aligned Colony Time at Formation",
    "colony_time_at_B": "Starting Aligned Colony Time",
    "colony_time_at_C": "Ending Aligned Colony Time",
    "duration_AB": "Rapid Expansion Duration",
    ("duration_BC", "duration_BC_hr"): "Growth Duration",
    # Volume
    "volume": "Volume",
    "volume_at_A": "Volume at Formation",
    "volume_at_B": "Starting volume",
    "volume_at_C": "Ending Volume",
    "Volume_C": "Volume at Breakdown",
    "volume_fold_change_BC": "Volume Fold-Change",
    "volume_fold_change_fromB": "Volume Fold-Change relative to Starting Volume",
    "delta_volume_BC": "Added Volume",
    "difference_volume_at_B": "Difference in Starting Volume",
    "difference_half_vol_at_C_and_B": "1/2 Mother Ending Volume - Daughter Starting Volume",
    "avg_sister_volume_fold_change_BC": "Sisters Average Volume Fold-Change",
    "avg_sister_volume_at_B": "Sisters Average Starting Volume",
    "volume_sub": "Change in volume",
    # Growth rates
    "exp_growth_coeff_BC": "Exponetial Growth Coeff. B to C",
    "linear_growth_rate_BC": "Late Growth Rate",
    "tscale_linearityfit_volume": "Fitted Time Scaling Factor (\u03B1)",
    "RMSE_linearityfit_volume": "Root Mean Squared Error",
    "late_growth_rate_by_endpoints": "Growth Rate",
    "dxdt_t2-dxdt_t1": "Late average transient growth rate - early average transient growth rate",
    # Height
    "height": "Height",
    "avg_height": "Average Height",
    "height_fold_change_BC": "Growth Height Fold-Change",
    "height_at_B": "Starting height",
    "height_at_C": "Ending Height",
    # Surface Area
    "mesh_sa": "Surface Area",
    "SA_at_A": "Surface Area at A",
    "SA_at_B": "Starting Surface Area",
    "SA_at_C": "Ending Surface Area",
    "delta_SA_BC": "\u0394Surface Area B to C",
    "SA_fold_change_BC": "Surface Area Fold-Change B to C",
    "SA_fold_change_fromB": "Surface Area Fold-Change",
    "SA_vol_ratio": "SA/Volume",
    "tscale_linearityfit_SA": "Fitted Surface Area Time Scaling (\u03B1)",
    "RMSE_linearityfit_SA": "Root Mean Squared Error",
    # Dimensions beyond height
    "length": "XY short axis width",
    "width": "XY long axis length",
    # Aspect Ratio
    "xy_aspect": "XY Aspect Ratio",
    "xz_aspect": "XZ Aspect Ratio",
    "zy_aspect": "YZ Aspect Ratio",
    "xz_aspect_fold_change_BC": "XZ Aspect Ratio Fold-Change B to C",
    "avg_xz_aspect_ratio": "Average XZ Aspect Ratio",
    "xz_aspect_at_B": "Starting XZ aspect ratio",
    # Colony Position
    "distance": "Distance",
    "distance_from_centroid": "Distance From Centroid",
    "normalized_distance_from_centroid": "Normalized Distance From Centroid",
    "max_distance_from_centroid": "Max Distance From Centroid",
    "colony_depth": "Colony Depth",
    "normalized_colony_depth": "Normalized Colony Depth",
    "max_colony_depth": "Max Colony Depth",
    "avg_colony_depth": "Average Colony Depth",
    # Density
    "colony_non_circularity": "Colony Non-circularity",
    "colony_non_circularity_scaled": "Scaled Colony Non-circularity",
    "avg_early_density": "Early Density",
    "avg_late_density": "Late Density",
    "density": "Density",
    "avg_density": "Average Density",
    # Lineage
    "parent_id": "Parent ID",
    "family_id": "Family ID",
    "sisters_volume_at_B": "Sisters starting volume",
    "sisters_duration_BC": "Sisters growth duration",
    # Flags
    "is_outlier": "Outlier Flag",
    "is_tp_outlier": "Single Timepoint Outlier",
    "is_outlier_track": "Outlier Track Flag",
    "is_growth_outlier": "Growth Feature Outlier Flag",
    "fov_edge": "FOV-edge Flag",
    "termination": "Track Termination",
    "is_full_track": "Full Interphase Track Flag",
    # colony segmentations
    "colony_area": "area of colony (brightfield)",
    "nucleus_colony_area_ratio": "ratio of nuclear area to colony area",
    "seg_twoD_zMIP_area": "total projected nuclear area",
    # LRM feats
    "height_at_B": "Starting height",
    "density_at_B": "Starting density",
    "xy_aspect_at_B": "Starting XY aspect ratio",
    "SA_vol_ratio_at_B": "Starting SA/Volume ratio",
    "early_transient_gr_whole_colony": "~Starting transient growth rate of whole colony",
    "mean_volume": "Mean volume",
    "mean_height": "Mean height",
    "mean_density": "Mean density",
    "mean_mesh_sa": "Mean surface area",
    "mean_xy_aspect": "Mean XY aspect ratio",
    "mean_SA_vol_ratio": "Mean SA/Volume ratio",
    "mean_neighbor_avg_dxdt_48_volume_whole_colony": "Mean transient growth rate of whole colony",
    "std_volume": "Stdev. volume",
    "std_height": "Stdev. height",
    "std_density": "Stdev. density",
    "std_mesh_sa": "Stdev. surface area",
    "std_xy_aspect": "Stdev. XY aspect ratio",
    "std_SA_vol_ratio": "Stdev. SA/Volume ratio",
    "std_neighbor_avg_dxdt_48_volume_whole_colony": "Stdev. transient growth rate of whole colony",
    'neighbor_avg_lrm_volume_90um_at_B': "Starting avg. volume in 90um radius", 
    'neighbor_avg_lrm_height_90um_at_B': "Starting avg. height in 90um radius",
    'neighbor_avg_lrm_density_90um_at_B': "Starting avg. density in 90um radius",
    'neighbor_avg_lrm_xy_aspect_90um_at_B': "Starting avg. XY aspect ratio in 90um radius",
    'neighbor_avg_lrm_mesh_sa_90um_at_B': "Starting avg. surface area in 90um radius",
    
    'mean_neighbor_avg_lrm_volume_90um': " Avg. mean volume in 90um radius", 
    'mean_neighbor_avg_lrm_height_90um': "Avg. mean height in 90um radius",
    'mean_neighbor_avg_lrm_density_90um': "Avg. mean density in 90um radius",
    'mean_neighbor_avg_lrm_xy_aspect_90um': "Avg. mean XY aspect ratio in 90um radius",
    'mean_neighbor_avg_lrm_mesh_sa_90um': "Avg. mean surface area in 90um radius",
    
    # mitotic and apoptotic neighbor columns
    "number_of_frame_of_breakdown_neighbors": "# of neighboring cells undergoing breakdown",
    "number_of_frame_of_formation_neighbors": "# of neighboring cells undergoing formation",
    "number_of_frame_of_death_neighbors": "# of neighboring cells undergoing death",
    "sum_has_mitotic_neighbor": "Sum of mitotic neighbors",
    "sum_has_dying_neighbor": "Sum of dying neighbors",  
}



def convert_to_hr(bin_interval, dataset="all_baseline"):
    """
    convert bin interval to hrs
    """
    time_interval_minutes = get_dataset_time_interval_in_min(dataset)
    out = bin_interval * time_interval_minutes / 60
    # if out is integer, then have no decimal place
    if out.is_integer():
        return str(int(out))
    return f"{out:.1f}"


temp_dict = get_one_to_one_dict(LABEL_TABLE)
for bin_interval in BIN_INTERVAL_LIST:

    LABEL_TABLE.update(
        {
            f"dxdt_{bin_interval}_volume_per_V": f"\u0394V/(\u0394T={convert_to_hr(bin_interval)}hr)/V"
        }
    )
    for feature in DXDT_FEATURE_LIST:
        LABEL_TABLE.update(
            {
                f"dxdt_{bin_interval}_{feature}": f"\u0394{temp_dict[feature]}/(\u0394T={convert_to_hr(bin_interval)}hr)"
            }
        )

# now add the neighborhood columns
for local_radius_str in LOCAL_RADIUS_STR_LIST:
    for feature in NEIGHBOR_FEATURE_LIST:
        LABEL_TABLE.update(
            {
                f"{NEIGHBOR_PREFIX}{feature}_{local_radius_str}": "Avg "
                + temp_dict[feature]
                + " of neighbors\nin "
                + local_radius_str.replace("_", " ").replace("um", " μm")
            }
        )
        LABEL_TABLE.update(
            {
                f"{NEIGHBOR_PREFIX}{DXDT_PREFIX}{bin_interval}_{feature}_{local_radius_str}": "Avg \u0394"
                + temp_dict[feature]
                + "/"
                + "(\u0394T="
                + convert_to_hr(bin_interval)
                + "hr) of neighbors\nin "
                + local_radius_str.replace("_", " ").replace("um", " μm")
                for bin_interval in BIN_INTERVAL_LIST
            }
        )
    # add {DXDT_PREFIX}{bin_interval}_volume_per_V
    LABEL_TABLE.update(
        {
            f"{NEIGHBOR_PREFIX}{DXDT_PREFIX}{bin_interval}_volume_per_V_{local_radius_str}": "Avg \u0394V/(\u0394T="
            + convert_to_hr(bin_interval)
            + "hr)/V of neighbors\nin "
            + local_radius_str.replace("_", " ").replace("um", " μm")
            for bin_interval in BIN_INTERVAL_LIST
        }
    )

# add name for coordinated_local_growth_fig.py
for bin_interval in BIN_INTERVAL_LIST:
    for local_radius_str in LOCAL_RADIUS_STR_LIST:
        LABEL_TABLE.update(
            {
                f"dvdt_t2-dvdt_t1_neighbors_{bin_interval}_{local_radius_str}": "Avg \u0394V/\u0394T (t2) - Avg \u0394V/\u0394T (t1)\nof neighbors"
            }
        )
        LABEL_TABLE.update(
            {
                f"dvdt_t2-dvdt_t1_self_{bin_interval}_{local_radius_str}": "Avg \u0394V/\u0394T (t2) - Avg \u0394V/\u0394T (t1)"
            }
        )


COLORIZER_LABEL_TABLE = {
    "index_sequence": "Time",
    "sync_time_Ff": "Synchronized time",
    "normalized_time": "Normalized interphase time",
    "colony_time": "Aligned colony time",
    "time_at_A": "Time at start of expansion",
    "time_at_B": "Time at start of growth",
    "time_at_C": "Time at end of growth",
    "colony_time_at_B": "Aligned colony time at start of growth",
    "duration_AB": "Expansion duration",
    "duration_AC": "Interphase duration",
    ("duration_BC", "duration_BC_hr"): "Growth duration",
    "height": "Height",
    "volume": "Volume",
    "volume_at_A": "Volume at start of expansion",
    "volume_at_B": "Volume at start of growth",
    "volume_at_C": "Volume at end of growth",
    "Volume_C": "Volume at end of growth",
    "volume_fold_change_BC": "Growth volume fold-change",
    "volume_fold_change_fromB": "Volume fold-change relative to starting volume",
    "delta_volume_BC": "Added volume during growth",
    "tscale_linearityfit_volume": "Fitted time scaling factor \u03B1",
    "RMSE_linearityfit_volume": "Fitted volume RMSE",
    "growth_rate_AB": "Expansion rate",
    "late_growth_rate_by_endpoints": "Growth rate",
    "dxdt_48_volume": "Transient growth rate",
    "neighbor_avg_dxdt_48_volume_whole_colony": "Colony-averaged transient growth rate",
    "neighbor_avg_dxdt_48_volume_90um": "Neighborhood-averaged transient growth rate",
    "mesh_sa": "Surface area",
    "SA_at_B": "Surface area at start of growth",
    "SA_at_C": "Surface area at end of growth",
    "delta_SA_BC": " Added surface area during growth",
    "SA_fold_change_BC": "Surface area fold-change during growth",
    "SA_fold_change_fromB": "Surface area fold-change relative to starting surface area",
    "SA_vol_ratio": "Surface area to volume ratio",
    "tscale_linearityfit_SA": "Fitted surface area time scaling \u03B1",
    "RMSE_linearityfit_SA": "Fitted surface area RMSE",
    "xy_aspect": "XY aspect ratio",
    "xz_aspect": "XZ aspect ratio",
    "zy_aspect": "YZ aspect ratio",
    "distance_from_centroid": "Distance from colony center",
    "normalized_colony_depth": "Normalized distance from colony center",
    "density": "Density",
    "family_id": "Family ID",
    "is_growth_outlier": "Growth outlier filter",
    "termination": "Trajectory termination annotation",
    "baseline_colonies_dataset": "Baseline colonies dataset filter",
    "full_interphase_dataset": "Full-interphase dataset filter",
    "lineage_annotated_dataset": "Lineage-annotated dataset filter",
    
    # mitotic and apoptotic neighbor columns
    "number_of_frame_of_breakdown_neighbors": "# of neighboring cells undergoing breakdown",
    "number_of_frame_of_formation_neighbors": "# of neighboring cells undergoing formation",
    "number_of_frame_of_death_neighbors": "# of neighboring cells undergoing death",
}

# units for quantities
UNIT_TABLE = {
    # Spatial
    (
        "width",
        "length",
        "distance_from_centroid",
        "max_distance_from_centroid",
        "height",
        "height_at_B",
        "height_at_C",
        "avg_height",
        "distance",
    ): "(μm)",
    (
        "RMSE_linearityfit_SA",
        "mesh_sa",
        "SA_at_B",
        "SA_at_C",
        "delta_SA_BC",
        "difference_SA_at_B",
        "colony_area",
        "seg_twoD_zMIP_area",
    ): "(μm²)",
    (
        "volume",
        "delta_volume_BC",
        "Volume_C",
        "RMSE_linearityfit_volume",
        "volume_at_A",
        "volume_at_B",
        "volume_at_C",
        "difference_volume_at_B",
        "difference_half_vol_at_C_and_B" "avg_sister_volume_at_B",
        "volume_sub",
    ): "(μm\u00B3)",
    "SA_vol_ratio": "(μm⁻¹)",
    (
        "density",
        "avg_early_density",
        "avg_late_density",
        "avg_density",
    ): "(μm⁻²)",
    # Temporal
    (
        "colony_time",
        "sync_time_Ff",
        "duration_AB",
        "duration_AC",
        "duration_BC",
        "duration_BC_hr",
        "index_sequence",
        "colony_time_at_B",
        "colony_time_at_C",
        "time_at_A",
        "time_at_B",
        "time_at_C",
        "time_bin_avg",
    ): "(hr)",
    "duration_AB": "(min)",
    # Rates
    (
        "growth_rate_AB",
        "linear_growth_rate_BC",
        "late_growth_rate_by_endpoints",
    ): "(μm\u00B3/hr)",
    "exp_growth_coeff_BC": "(hr⁻¹)",
}

# now add the dxdt columns
temp_dict = get_one_to_one_dict(UNIT_TABLE)
for bin_interval in BIN_INTERVAL_LIST:
    UNIT_TABLE.update({f"dxdt_{bin_interval}_volume_per_V": "(hr⁻¹})"})
    for feature in DXDT_FEATURE_LIST:
        UNIT_TABLE.update({f"dxdt_{bin_interval}_{feature}": f"({temp_dict[feature][1:-1]}/hr)"})

# now add the neighborhood columns
for local_radius_str in LOCAL_RADIUS_STR_LIST:
    for feature in NEIGHBOR_FEATURE_LIST:
        UNIT_TABLE.update({f"{NEIGHBOR_PREFIX}{feature}_{local_radius_str}": temp_dict[feature]})
        UNIT_TABLE.update(
            {
                f"{NEIGHBOR_PREFIX}{DXDT_PREFIX}{bin_interval}_{feature}_{local_radius_str}": f"({temp_dict[feature][1:-1]}/hr)"
                for bin_interval in BIN_INTERVAL_LIST
            }
        )
    # add {DXDT_PREFIX}{bin_interval}_volume_per_V
    UNIT_TABLE.update(
        {
            f"{NEIGHBOR_PREFIX}{DXDT_PREFIX}{bin_interval}_volume_per_V_{local_radius_str}": "(hr⁻¹)"
            for bin_interval in BIN_INTERVAL_LIST
        }
    )

# add name for coordinated_local_growth_fig.py
for bin_interval in BIN_INTERVAL_LIST:
    for local_radius_str in LOCAL_RADIUS_STR_LIST:
        UNIT_TABLE.update({"dxdt_t2-dxdt": "(μm\u00B3/hr)"})
        UNIT_TABLE.update(
            {f"dvdt_t2-dvdt_t1_self_{bin_interval}_{local_radius_str}": "(μm\u00B3/hr)"}
        )
        UNIT_TABLE.update(
            {f"dvdt_t2-dvdt_t1_neighbors_{bin_interval}_{local_radius_str}": "(μm\u00B3/hr)"}
        )
        UNIT_TABLE.update(
            {f"dvdt_t2-dvdt_t1_self_{bin_interval}_{local_radius_str}": "(μm\u00B3/hr)"}
        )


# limits when growth outliers are filtered out
LIMIT_TABLE = {
    # Times and durations
    "colony_time_at_A": (-1, 60),
    "colony_time_at_B": (-1, 60),
    "colony_time_at_C": (-1, 70),
    "sync_time_Ff": (-0.5, 24),
    "normalized_time": (-0.01, 1),
    "duration_AB": (10, 100),
    "duration_BC": (8, 24),
    "duration_BC_hr": (8, 24),
    "time_at_A": (0, 35),
    "time_at_B": (0, 37),
    "time_at_C": (0, 50),
    # Volume
    "volume": (200, 1400),
    "volume_at_A": (200, 500),
    "volume_at_B": (300, 750),
    "volume_at_C": (650, 1400),
    "volume_fold_change_BC": (1.2, 2.8),
    "volume_fold_change_fromB": (0.3, 2.8),
    "delta_volume_BC": (200, 900),
    "volume_sub": (-250, 250),
    # Growth rates
    "growth_rate_AB": (0, 800),
    "linear_growth_rate_BC": (5, 60),
    "late_growth_rate_by_endpoints": (15, 60),
    "alpha": (0, 0.1),
    "tscale_linearityfit_volume": (0.25, 2.75),
    "RMSE_linearityfit_volume": (10, 50),
    "exp_growth_coeff_BC": (0, 0.1),
    "ln(Vb)": (5.5, 6.5),
    "Ln(Vc/Vb)": (0, 1.4),
    # Height
    "height_at_B": (3.5, 10.5),
    "height_at_C": (3.5, 10.5),
    "height_fold_change_BC": (0.5, 2.75),
    "avg_height": (4, 9),
    # Surface Area
    "mesh_sa": (150, 1400),
    "SA_fold_change_BC": (0.75, 3.25),
    "SA_fold_change_fromB": (0.3, 4),
    "SA_at_B": (300, 525),
    "SA_at_C": (450, 1200),
    "delta_SA_BC": (0, 1000),
    "SA_vol_ratio": (0.5, 1.25),
    "tscale_linearityfit_SA": (0, 2.5),
    "RMSE_linearityfit_SA": (10, 50),
    # Aspect Ratio
    "avg_xz_aspect": (1, 5.5),
    "xz_aspect_at_B": (1, 5.5),
    "xz_aspect_fold_change_BC": (0, 3.25),
    # Colony Position
    "avg_colony_depth": (0, 9),
    # Density
    "avg_density": (6.1e-4, 4.6e-3),
}

# limits when growth outliers are left in the dataset
LIMIT_TABLE_WITH_GROWTH_OUTLIERS = {
    # Times and durations
    "colony_time_at_A": (-5, 60),
    "colony_time_at_B": (-5, 60),
    "colony_time_at_C": (7, 72),
    "sync_time_Ff": (-1, 35),
    "normalized_time": (-0.025, 1),
    "duration_AB": (10, 100),
    "duration_BC": (8, 38),
    "duration_BC_hr": (8, 24),
    "time_at_A": (-5, 35),
    "time_at_B": (0, 37),
    "time_at_C": (0, 50),
    # Volume
    "volume": (200, 2000),
    "volume_at_A": (200, 500),
    "volume_at_B": (300, 750),
    "volume_at_C": (450, 1900),
    "volume_fold_change_BC": (1.2, 2.8),
    "volume_fold_change_fromB": (0.3, 4),
    "delta_volume_BC": (0, 1300),
    # Growth rates
    "growth_rate_AB": (0, 800),
    "linear_growth_rate_BC": (5, 60),
    "late_growth_rate_by_endpoints": (10, 60),
    "alpha": (0, 0.1),
    "tscale_linearityfit_volume": (0, 2.75),
    "RMSE_linearityfit_volume": (10, 50),
    "exp_growth_coeff_BC": (0, 0.1),
    "ln(Vb)": (5.5, 6.5),
    "Ln(Vc/Vb)": (0, 1.4),
    # Height
    "height_at_B": (3.5, 10.5),
    "height_at_C": (3.5, 10.5),
    "height_fold_change_BC": (0.5, 2.75),
    "avg_height": (4, 9),
    # Surface Area
    "mesh_sa": (150, 1400),
    "SA_fold_change_BC": (0.75, 3.25),
    "SA_fold_change_fromB": (0.3, 4),
    "SA_at_B": (300, 525),
    "SA_at_C": (450, 1200),
    "delta_SA_BC": (0, 1000),
    "SA_vol_ratio": (0.5, 1.25),
    "tscale_linearityfit_SA": (0, 2.5),
    "RMSE_linearityfit_SA": (10, 50),
    # Aspect Ratio
    "avg_xz_aspect": (1, 5.5),
    "xz_aspect_at_B": (1, 5.5),
    "xz_aspect_fold_change_BC": (0, 3.25),
    # Colony Position
    "avg_colony_depth": (0, 9),
    # Density
    "avg_density": (4e-6, 6e-5),
    # colony area
    "colony_area": (0, 200000),  # max area of 20 FOV i 170,000 µm^2
    "nucleus_colony_area_ratio": (0.3, 0.55),
    "seg_twoD_zMIP_area": (0, 200000),  # max area of 20 FOV i 170,000 µm^2
}


# limits for feature averages
AVG_LIMIT_TABLE = {
    # Times and durations
    "volume": (400, 1200),
    "exiting_mitosis": (0, 0.4),
    "nucleus_colony_area_ratio": (2, 3),
    "density": (0.0010, 0.0045),
    "dxdt_48_volume": (10, 80),
    "dxdt_24_volume": (10, 80),
    "tscale_linearityfit_volume": (0.2, 3),
    "linear_growth_rate_BC": (10, 70),
    "height": (3.5, 11.75),
    "dxdt_24_height": (-0.5, 0.5),
    "xz_aspect": (1, 5),
    "xy_aspect": (1, 2),
    "SA_vol_ratio": (0.5, 1.25),
    "duration_BC": (8, 30),
    "volume_at_B": (300, 900),
    "volume_at_C": (600, 2000),
    "delta_volume_BC": (300, 1400),
    "volume_fold_change_BC": (1.3, 3.7),
    "SA_at_B": (300, 800),
    "SA_at_C": (500, 1400),
    "delta_SA_BC": (100, 800),
    "SA_fold_change_BC": (1.2, 3),
    "normalized_time": (0, 1),
    "entering_apoptosis": (0, 0.05),
    "mesh_sa": (100, 1200),
    "colony_area": (0, 200000),  # max area of 20 FOV i 170,000 µm^2
    "nucleus_colony_area_ratio": (0.3, 0.55),
    "seg_twoD_zMIP_area": (0, 200000),  # max area of 20 FOV i 170,000 µm^2
}
