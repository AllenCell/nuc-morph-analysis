from yarl import URL

from nuc_morph_analysis.lib.preprocessing.system_info import PIXEL_SIZE_YX_100x


S3_PREFIX = URL(
    "https://allencell.s3.amazonaws.com/aics/nuc-morph-dataset/hipsc_fov_nuclei_timelapse_dataset/hipsc_fov_nuclei_timelapse_data_used_for_analysis"
)
BASELINE_COLONY_DIR = S3_PREFIX / "baseline_colonies_fov_timelapse_dataset"
FIXED_CONTROL_DIR = S3_PREFIX / "fixed_control_fov_timelapse_dataset"
FEEDING_CONTROL_DIR = S3_PREFIX / "feeding_control_fov_timelapse_dataset"
APHIDICOLIN_DIR = S3_PREFIX / "dna_replication_inhibitor_fov_timelapse_dataset"
IMPORTAZOLE_DIR = S3_PREFIX / "nuclear_import_inhibitor_fov_timelapse_dataset"

datasets = [
    {
        "name": "all_baseline",  # this is the common info for all baseline datasets
        # FMS ID for 2024-06-25_main_manifest.parquet generated from morflowgenesis v0.3.0
        # with generate_main_manifest.py at commit 6e9eb0962343113ab3999ce6b59d8331ddab9a45
        "fmsid": "97b8765af33a4b4ab4da39afc995324f",  # morflowgenesis v0.3.0
        "pixel_size": PIXEL_SIZE_YX_100x,
        "time_interval": 5,  # min
        "length_threshold": 10.0,  # hours,
        "experiment": "baseline",
        "overview": None,
        "egfp_channel_index": 0,
        "brightfield_channel_index": 1,
    },
    {
        "name": "all_feeding_control",  # this is the common info for all "feeding_control" datasets
        # FMS ID for 2024-06-16_feeding_control_main_manifest.parquet generated from morflowgenesis v0.3.0
        # with generate_perturbation_manifest.py at commit ebe76b5e84c9ca24617e4d04aed8acc1c2c3bb62
        "fmsid": "8ecd9b04329b490baec500859e276fbe",  # morflowgenesis v 0.3.0
        "pixel_size": PIXEL_SIZE_YX_100x,
        "time_interval": 5,  # min
        "length_threshold": 10.0,  # hours,
        "experiment": "feeding_control",
        "overview": None,
        "egfp_channel_index": 1,
        "brightfield_channel_index": 0,
    },
    {
        "name": "all_drug_perturbation",  # this is the common info for all "drug_perturbation" datasets
        # FMS ID for 2024-06-24_drug_perturbation_main_manifest.parquet generated from morflowgenesis v0.3.0
        # with generate_perturbation_manifest.py at commit 725ed45a6413391b9927610649e6209c04bcae9f
        "fmsid": "19e1125fd9c4413e8babe2e9de8d9b87",  # morflowgenesis v 0.3.0
        "pixel_size": PIXEL_SIZE_YX_100x,
        "time_interval": 5,  # min
        "length_threshold": 10.0,  # hours,
        "experiment": "drug_perturbation",
        "overview": None,
        "egfp_channel_index": 0,
        "brightfield_channel_index": 1,
    },
    {
        "name": "small",
        "pixel_size": PIXEL_SIZE_YX_100x,
        "original_fmsid": "7191a69c6d8f4f37b7a43cc962c72935",
        "s3_path_raw": BASELINE_COLONY_DIR / "20200323_09_small/raw.ome.zarr",
        "s3_path_seg": BASELINE_COLONY_DIR / "20200323_09_small/seg.ome.zarr",
        "scene": 8,
        "time_interval": 5,  # min
        "length_threshold": 10.0,  # hours,
        "experiment": "baseline",
        "overview": None,
        "egfp_channel_index": 0,
        "brightfield_channel_index": 1,
    },
    {
        "name": "medium",
        "pixel_size": PIXEL_SIZE_YX_100x,
        "original_fmsid": "7191a69c6d8f4f37b7a43cc962c72935",
        "s3_path_raw": BASELINE_COLONY_DIR / "20200323_06_medium/raw.ome.zarr",
        "s3_path_seg": BASELINE_COLONY_DIR / "20200323_06_medium/seg.ome.zarr",
        "scene": 5,
        "time_interval": 5,  # min
        "length_threshold": 10.0,  # hours,
        "experiment": "baseline",
        "overview": None,
        "egfp_channel_index": 0,
        "brightfield_channel_index": 1,
    },
    {
        "name": "large",
        "pixel_size": PIXEL_SIZE_YX_100x,
        "original_fmsid": "7191a69c6d8f4f37b7a43cc962c72935",
        "s3_path_raw": BASELINE_COLONY_DIR / "20200323_05_large/raw.ome.zarr",
        "s3_path_seg": BASELINE_COLONY_DIR / "20200323_05_large/seg.ome.zarr",
        "scene": 4,
        "time_interval": 5,  # min
        "length_threshold": 10.0,  # hours,
        "experiment": "baseline",
        "overview": None,
        "egfp_channel_index": 0,
        "brightfield_channel_index": 1,
    },
    {
        "name": "fixed_control",
        "fmsid": "043aa487a4e24d69ba509669720e7afc",  # original w/o SHE coeff: "2456f5bebdac4a49a9a623cdf2c28bce"
        "pixel_size": PIXEL_SIZE_YX_100x,
        "original_fmsid": "c025f50129a94e9ea5590022b7bca029",
        "s3_path_raw": FIXED_CONTROL_DIR / "20220901_01/raw.ome.zarr",
        "s3_path_seg": FIXED_CONTROL_DIR / "20220901_01/seg.ome.zarr",
        "time_interval": 5,  # each frame is simulating how far cells move in a 5 minute interval
        "length_threshold": 0.0,  # hours,
        "experiment": "fixed_control",
        "scene": 0,
        "egfp_channel_index": 1,
        "brightfield_channel_index": 0,
    },
    {
        "name": "feeding_control_baseline",
        "pixel_size": PIXEL_SIZE_YX_100x,
        "original_fmsid": "491edc8400d0466ab48f01ee9498494d",
        "s3_path_raw": FEEDING_CONTROL_DIR / "20230720_01_control/raw.ome.zarr",
        "s3_path_seg": FEEDING_CONTROL_DIR / "20230720_01_control/seg.ome.zarr",
        "scene": 0,
        "time_interval": 5,
        "length_threshold": 10.0,  # hours,
        "experiment": "feeding_control",
        "overview": None,
        "egfp_channel_index": 1,
        "brightfield_channel_index": 0,
    },
    {
        "name": "feeding_control_starved",
        "pixel_size": PIXEL_SIZE_YX_100x,
        # 491edc8400d0466ab48f01ee9498494d is the July 20, 2023 CZI
        "original_fmsid": "491edc8400d0466ab48f01ee9498494d",
        # Scene number in the folder path (number after the date) is 1-indexed
        "s3_path_raw": FEEDING_CONTROL_DIR / "20230720_04_pre-starved/raw.ome.zarr",
        "s3_path_seg": FEEDING_CONTROL_DIR / "20230720_04_pre-starved/seg.ome.zarr",
        "scene": 3,  # zero-indexed
        "time_interval": 5,
        "length_threshold": 10.0,  # hours,
        "experiment": "feeding_control",
        "overview": None,
        "egfp_channel_index": 1,
        "brightfield_channel_index": 0,
    },
    {
        "name": "feeding_control_refeed",
        "pixel_size": PIXEL_SIZE_YX_100x,
        "original_fmsid": "491edc8400d0466ab48f01ee9498494d",
        "s3_path_raw": FEEDING_CONTROL_DIR / "20230720_07_re-fed/raw.ome.zarr",
        "s3_path_seg": FEEDING_CONTROL_DIR / "20230720_07_re-fed/seg.ome.zarr",
        "scene": 6,
        "time_interval": 5,
        "length_threshold": 10.0,  # hours,
        "experiment": "feeding_control",
        "overview": None,
        "egfp_channel_index": 1,
        "brightfield_channel_index": 0,
    },
    {
        # aphidicolin control
        "name": "drug_perturbation_1_scene0",
        "pixel_size": PIXEL_SIZE_YX_100x,
        "original_fmsid": "5d1d2b2d6e0b40b4968e92f5113ec498",
        "s3_path_raw": APHIDICOLIN_DIR / "20230424_01_control/raw.ome.zarr",
        "s3_path_seg": APHIDICOLIN_DIR / "20230424_01_control/seg.ome.zarr",
        "scene": 0,
        "time_interval": 5,
        "length_threshold": 10.0,  # hours,
        "experiment": "drug_perturbation",
        "overview": None,
        "egfp_channel_index": 1,
        "brightfield_channel_index": 0,
    },
    {
        # aphidicolin control April 24
        "name": "drug_perturbation_1_scene2",
        "pixel_size": PIXEL_SIZE_YX_100x,
        "original_fmsid": "5d1d2b2d6e0b40b4968e92f5113ec498",
        "s3_path_raw": APHIDICOLIN_DIR / "20230424_03_control/raw.ome.zarr",
        "s3_path_seg": APHIDICOLIN_DIR / "20230424_03_control/seg.ome.zarr",
        "scene": 2,
        "time_interval": 5,
        "length_threshold": 10.0,  # hours,
        "experiment": "drug_perturbation",
        "overview": None,
        "egfp_channel_index": 1,
        "brightfield_channel_index": 0,
    },
    {
        # perturbation 1 is always April 24, 2023
        # aphidicolin treatment 20230424
        "name": "drug_perturbation_1_scene4",
        "pixel_size": PIXEL_SIZE_YX_100x,
        "original_fmsid": "5d1d2b2d6e0b40b4968e92f5113ec498",
        "s3_path_raw": APHIDICOLIN_DIR / "20230424_05_aphidicolin/raw.ome.zarr",
        "s3_path_seg": APHIDICOLIN_DIR / "20230424_05_aphidicolin/seg.ome.zarr",
        "scene": 4,
        "time_interval": 5,
        "length_threshold": 10.0,  # hours,
        "experiment": "drug_perturbation",
        "overview": None,
        "egfp_channel_index": 1,
        "brightfield_channel_index": 0,
    },
    {
        # control for April 17, 2023 puromycin and importazole
        "name": "drug_perturbation_2_scene0",
        "pixel_size": PIXEL_SIZE_YX_100x,
        "original_fmsid": "f648379a70b049dcb5cbfc1893411a97",
        "s3_path_raw": IMPORTAZOLE_DIR / "20230417_01_control/raw.ome.zarr",
        "s3_path_seg": IMPORTAZOLE_DIR / "20230417_01_control/seg.ome.zarr",
        "scene": 0,
        "time_interval": 5,  # each frame is simulating how far cells move in a 5 minute interval
        "length_threshold": 10.0,  # hours,
        "experiment": "drug_perturbation",
        "overview": None,
        "egfp_channel_index": 1,
        "brightfield_channel_index": 0,
    },
    {
        # treatment importazole April 17, 2023
        "name": "drug_perturbation_2_scene6",
        "pixel_size": PIXEL_SIZE_YX_100x,
        "original_fmsid": "f648379a70b049dcb5cbfc1893411a97",
        "s3_path_raw": IMPORTAZOLE_DIR / "20230417_07_importazole/raw.ome.zarr",
        "s3_path_seg": IMPORTAZOLE_DIR / "20230417_07_importazole/seg.ome.zarr",
        "scene": 6,
        "time_interval": 5,  # each frame is simulating how far cells move in a 5 minute interval
        "length_threshold": 10.0,  # hours,
        "experiment": "drug_perturbation",
        "overview": None,
        "egfp_channel_index": 1,
        "brightfield_channel_index": 0,
    },
    {
        # aphidicolin April 11 control
        "name": "drug_perturbation_4_scene2",
        "pixel_size": PIXEL_SIZE_YX_100x,
        "original_fmsid": "1d284940dc8f4b3d8360f85b5b9beee7",
        "s3_path_raw": APHIDICOLIN_DIR / "20220411_03_control/raw.ome.zarr",
        "s3_path_seg": APHIDICOLIN_DIR / "20220411_03_control/seg.ome.zarr",
        "scene": 2,
        "time_interval": 5,
        "length_threshold": 10.0,  # hours,
        "experiment": "drug_perturbation",
        "overview": None,
        "egfp_channel_index": 0,
        "brightfield_channel_index": 1,
    },
    {
        # aphidicolin April 11 treatment
        "name": "drug_perturbation_4_scene4",
        "pixel_size": PIXEL_SIZE_YX_100x,
        "original_fmsid": "1d284940dc8f4b3d8360f85b5b9beee7",
        "s3_path_raw": APHIDICOLIN_DIR / "20220411_05_aphidicolin/raw.ome.zarr",
        "s3_path_seg": APHIDICOLIN_DIR / "20220411_05_aphidicolin/seg.ome.zarr",
        "scene": 4,
        "time_interval": 5,
        "length_threshold": 10.0,  # hours,
        "experiment": "drug_perturbation",
        "overview": None,
        "egfp_channel_index": 0,
        "brightfield_channel_index": 1,
    },
]

# Used to produce the main datasets above
manual_curation_manifests = {
    "lineage_annotations": {
        "small": {
            # 2024-06-20_baby_bear_track_matched_annotations.csv
            "fmsid": "d2975553ae65488f869737d243fdb7af",
        },
        "medium": {
            # 2024-06-20_goldilocks_track_matched_annotations.csv
            "fmsid": "691f7f4f39e84160b41b27c417b3748a",
        },
    },
    "apoptosis_annotations": {
        "large": {
            # 2024-06-20_mama_bear_termination_annotations.csv
            "fmsid": "d696526047644acba76c543ef4c80040",
        }
    },
}

# Generated by morflowgenesis v0.3.0
morflowgenesis_outputs = {
    "small": {
        "fmsid": "163a11d0608f4f3894ca3cd1d184414e",
        "scene": 8,
    },
    "medium": {
        "fmsid": "7656aa639dc74fff96a94b618baefa88",
        "scene": 5,
    },
    "large": {
        "fmsid": "f1cd37596ac34dd28498c7ba2bff95dd",
        "scene": 4,
    },
    "feeding_control_baseline": {
        "fmsid": "4abbabc7df7240b382baa18461eb9afb",
        "scene": 0,
    },
    "feeding_control_starved": {
        "fmsid": "12931ef3323e448482de4815fb764df6",
        "scene": 3,
    },
    "feeding_control_refeed": {
        "fmsid": "98fbd00e7b264057a136e48a51bdc704",
        "scene": 6,
    },
    "fixed_control": {
        "fmsid": "cf294926af2642eda263a6f0686c65c7",
    },
    "drug_perturbation_1_scene0": {
        "fmsid": "ed6b6fe251fd44fd8d955173f00daf62",
        "scene": 0,
    },
    "drug_perturbation_1_scene2": {
        "fmsid": "2be9f89ce9b7433a8484a3c0c6a194c3",
        "scene": 2,
    },
    "drug_perturbation_1_scene4": {
        "fmsid": "88e1cc42a3ac4c15a4242d73351cf574",
        "scene": 4,
    },
    "drug_perturbation_2_scene0": {
        "fmsid": "9de181f9f3b3421596a6c5156e9bce41",
        "scene": 0,
    },
    "drug_perturbation_2_scene6": {
        "fmsid": "b35bbd3b9c954098b66121a15f89e76a",
        "scene": 6,
    },
    "drug_perturbation_4_scene2": {
        "fmsid": "4dbbd46e751f4e9d85a1fc192ae0b30c",
        "scene": 2,
    },
    "drug_perturbation_4_scene4": {
        "fmsid": "dccc85553b014235a55284ee78b925b3",
        "scene": 4,
    },
}

# This will need to be updated when Quilt is updated to move this file from additional_data_for_fine_tuning to data_for_fine_tuning.
segmentation_model_validation_URLs = {
    "single_cell_features": "https://allencell.s3.amazonaws.com/aics/nuc_morph_data/additional_data_for_fine_tuning/LaminB1/single_nucleus_features.csv",
    "base_dir_for_images": "https://allencell.s3.amazonaws.com/aics/nuc_morph_data/additional_data_for_fine_tuning/LaminB1",
}
