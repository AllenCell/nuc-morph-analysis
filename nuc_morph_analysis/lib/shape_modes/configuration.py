from pathlib import Path
from copy import deepcopy
from typing import Dict

nucmorph_all_tracks: Dict[str, Dict] = {
    "project": {
        "local_staging": Path(__file__).parents[3]
        / "nuc_morph_analysis/analyses/shape/figures/all_tracks_shape_modes/shape_analysis/shape_space",
        "overwrite": True,
    },
    "data": {"nucleus": {"alias": "NUC", "channel": "dna_segmentation", "color": "#3AADA7"}},
    "features": {
        "aliases": ["NUC"],
        # SHE - Spherical Harmonics Expansion
        "SHE": {
            "alignment": {"align": True, "reference": "nucleus", "unique": False},
            "aliases": ["NUC"],
            # Size of Gaussian kernal used to smooth the
            # images before SHE coefficients calculation
            "sigma": 2,
            # Number of SHE coefficients used to describe cell
            # and nuclear shape
            "lmax": 16,
        },
    },
    "preprocessing": {
        "remove_mitotics": True,
        "remove_outliers": True,
        "filtering": {"csv": "", "filter": False, "specs": {}},
    },
    "shapespace": {
        # Specify the a set of aliases here
        "aliases": ["NUC"],
        # Sort shape modes by volume of
        "sorter": "NUC",
        # Percentage of exteme points to be removed
        "removal_pct": 0.25,
        # Number of principal components to be calculated
        "number_of_shape_modes": 8,
        "map_points": [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        "plot": {
            "frame": False,
            "swapxy_on_zproj": False,
            # limits of x and y axies in the animated GIFs
            "limits": [-150, 150, -80, 80],
        },
    },
    "parameterization": {
        "inner": "NUC",
        "outer": "MEM",
        "parameterize": ["RAWSTR", "STR"],
        "number_of_interpolating_points": 32,
    },
    "aggregation": {"type": ["avg"]},
    "structures": {
        "lamin": [
            "nuclear envelope",
            "#084AE7",
            "{'raw': (475,1700), 'seg': (0,30), 'avgseg': (0,60)}",
        ]
    },
    "distribute": {},
}

nucmorph_full_tracks: Dict[str, Dict] = deepcopy(nucmorph_all_tracks)
nucmorph_full_tracks["project"]["local_staging"] = (
    Path(__file__).parents[3]
    / "nuc_morph_analysis/analyses/shape/figures/full_tracks_shape_modes/shape_analysis/shape_space"
)

nucmorph_fixed_control: Dict[str, Dict] = deepcopy(nucmorph_all_tracks)
nucmorph_fixed_control["project"]["local_staging"] = (
    Path(__file__).parents[3]
    / "nuc_morph_analysis/analyses/error_morflowgenesis/figures/shape_analysis/shape_space"
)
