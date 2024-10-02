import functools
import os
import time
import dask.array as da
import pandas as pd
import platform
from bioio import BioImage

from nuc_morph_analysis.lib.preprocessing.all_datasets import (
    datasets,
    manual_curation_manifests,
)

try:
    # aicsfiles is an optional dependency for users on the AICS intranet
    from aicsfiles import FileManagementSystem
except ImportError:
    FileManagementSystem = None

BASELINE_COLONY_NAMES = {
    "baby_bear": "small",
    "goldilocks": "medium",
    "mama_bear": "large",
}


def get_valid_path(record) -> str:
    """
    Converts a FMS path to one that can be read cross-platform
    """
    recordpath = record.path
    if platform.system() == "Windows":
        recordpath = "/" + recordpath
    return recordpath


def get_dataframe_by_info(info):
    """
    Read a CSV or parquet file from FMS into a pandas dataframe
    """
    # Get an S3 or local path
    if FileManagementSystem is not None and os.path.exists("/allen/aics"):
        fms = FileManagementSystem()
        fmsid = info["fmsid"]
        record = fms.get_file_by_id(fmsid)
        path = get_valid_path(record)
        print("NOTE: Reading data from FMS instead of S3 (quilt)")
    else:
        path = str(info["s3_path"])

    # Load dataframe by file format
    if path.endswith("csv"):
        df = pd.read_csv(path)
        # use height calculated from 1st to 99th percentile values
        # rather than the most extreme values
        df["height"] = df["height_percentile"]
        return df
    elif path.endswith("parquet"):
        df = pd.read_parquet(path)
        # use height calculated from 1st to 99th percentile values
        # rather than the most extreme values
        df["height"] = df["height_percentile"]
        return df
    else:
        raise ValueError(f"Unknown format {path.split('.')[-1]}")


def load_morflowgenesis_dataframe(dataset):
    info = datasets[dataset]["morflowgenesis_output"]
    single_colony_df = get_dataframe_by_info(info)
    single_colony_df.set_index("CellId")
    return single_colony_df


def load_lineage_annotations(dataset: str) -> pd.DataFrame:
    # Currently only the small and medium datasets have lineage annotations
    assert dataset == "small" or dataset == "medium"
    info = manual_curation_manifests["lineage_annotations"][dataset]
    return get_dataframe_by_info(info)


def load_apoptosis_annotations(dataset: str) -> pd.DataFrame:
    assert dataset == "large"  # Currently only the large dataset has apoptosis annotations
    info = manual_curation_manifests["apoptosis_annotations"][dataset]
    return get_dataframe_by_info(info)


def get_length_threshold_in_tps(name):
    """
    Converts the length threshold in the dict above from hours
    to number of timepoints.
    """
    info = get_dataset_info_by_name(name)
    THRESH = info["length_threshold"]
    dt = get_dataset_time_interval_in_min(name)
    return int(THRESH * 60 / dt)


def get_channel_index(name, channel):
    """
    Parameters
    ----------
    name: string
        Name of dataset
    channel: string
        "egfp" or "bright"
    """
    info = get_dataset_info_by_name(name)
    if channel == "egfp":
        return info["egfp_channel_index"]
    elif channel == "bright":
        return info["brightfield_channel_index"]


def get_available_datasets(experiments=["baseline"]):
    """
    Returns list with names of availabe datasets

    Parameters
    ----------
    experiments: list
        List of experiment names.

    Returns
    -------
    names: list
        List with names of availabe datasets
    """
    names = [
        name
        for name, info in datasets.items()
        if (
            (name not in ["all_baseline", "all_feeding_control", "all_drug_perturbation"])
            and (info["experiment"] in experiments)
        )
    ]
    return names


def get_dataset_experiment_group_by_name(name):
    """
    Find the experiment group of a given dataset by its name.

    Parameters
    ----------
    name: String
        Dataset name
    Returns
    -------
    ds: dict
        Ditionary with dataset info
    """
    return get_dataset_info_by_name(name)["experiment"]


def get_dataset_info_by_name(name):
    """
    Find the info of a given dataset by its name.

    Parameters
    ----------
    name: String
        Dataset name
    Returns
    -------
    ds: dict
        Ditionary with dataset info
    """
    if name not in datasets:
        raise ValueError(f"Dataset {name} not available.")
    return datasets[name]


def get_dataset_original_file_reader(dataset: str, quiet=True) -> BioImage:
    """
    Return BioImage reader for the original data file of a dataset.

    Parameters
    ----------
    dataset: string
        Name of dataset

    Returns
    -------
    reader: BioImage

    TIP: use
        raw = reader.get_image_dask_data("ZYX", T=0, C=0)
    to get channel 1 of timepoint 0. C=1 corresponds to
    bright field and C=0 is Lamin.
    """
    info = get_dataset_info_by_name(dataset)
    # Read the ZARR from S3
    s3_path_raw = str(info["s3_path_raw"])
    start = time.time()
    reader = BioImage(s3_path_raw)
    if not quiet:
        print(f"Connected to S3 file in {time.time()-start:.1f}s")
    return reader


def get_dataset_segmentation_file_reader(dataset: str) -> BioImage:
    info = get_dataset_info_by_name(dataset)
    # Read the ZARR from S3
    s3_path_seg = str(info["s3_path_seg"])
    return BioImage(s3_path_seg)


def load_dataset(dataset):
    """
    This function loads a dataset using its FMS ID. Some datasets have
    known keynames such as "goldilocks" - in this case, a key name
    can be used in place of the FMS ID.
    Alternatively, a path to a local file can be provided to load
    a dataset directly.

    Parameters
    ----------
    dataset: String
        String containing an FMS ID or dataset keyname
        Give the FMS ID of the dataset to be loaded. In the case of
        the goldilocks, mama bear and baby bear datasets, the
        dataset key name ("goldilocks", "mama_bear" or "baby_bear")
        may be passed instead to get the dataset for that colony
        with features and timepoint classifier columns added.

    Returns
    -------
    df: Dataframe
        Returns the dataframe for this dataset
    """

    print(f"Loading dataset {dataset}...")

    info = get_dataset_info_by_name(dataset)

    # load dataset from fms
    if dataset in ["baby_bear", "goldilocks", "mama_bear"]:
        df_all = load_all_datasets()
        df = df_all[df_all.colony == BASELINE_COLONY_NAMES[dataset]]
    if dataset in ["feeding_control_baseline", "feeding_control_starved", "feeding_control_refeed"]:
        df_all = load_all_datasets("all_feeding_control")
        df = df_all[df_all.colony == dataset]
    else:
        df = get_dataframe_by_info(info).set_index("CellId")
        df["dataset"] = dataset
    

    # use height calculated from 1st to 99th percentile values
    # rather than the most extreme values
    df["height"] = df["height_percentile"]

    # set column types and drop Nan rows
    cols = [
        ("fov_edge", bool),
        ("track_id", int),
        ("index_sequence", int),
    ]
    df = df.dropna(subset=[c for c, _ in cols])
    for col, coltype in cols:
        df[col] = df[col].astype(coltype)

    df = df.sort_values(by=["track_id", "index_sequence"])

    print(f"Dataset loaded.")
    print(f"{df.shape[0]} single-timepoint nuclei in dataset.")
    print(f"{df.track_id.nunique()} nuclear tracks in dataset.")

    return df


def get_dataset_pixel_size(dataset):
    """
    Get pixel size in microns for a dataset

    Parameters
    ----------
    dataset: string
        Name of dataset

    Returns
    -------
    float
        Pixel size in microns for this dataset
    """

    info = get_dataset_info_by_name(dataset)

    return info["pixel_size"]


def get_dataset_time_interval_in_min(dataset):
    """
    Get time interval in minutes for a dataset

    Parameters
    ----------
    dataset: string
        Name of dataset

    Returns
    -------
    float
        Time interval in minutes for this dataset
    """

    info = get_dataset_info_by_name(dataset)

    return info["time_interval"]


@functools.lru_cache(maxsize=1, typed=False)
def load_all_datasets(all_dataset="all_baseline"):
    """
    Get dataframe with all three main colonies and all pre-processed columns.
    This function uses automatic caching, so that repeated runs in the same script or interactive
    python kernel will immediately return the previously-loaded data.

    The returned dataframe has ~600k rows and takes up about 5GB as a CSV.

    Returns
    -------
    pandas.DataFrame
        One row per timepoint per cell.
        See docs/feature_documentation.md for a description of all the columns.
    """
    info = get_dataset_info_by_name(all_dataset)
    start = time.time()
    df = get_dataframe_by_info(info)
    print(f"Read dataset {all_dataset} from parquet file ({time.time()-start:.1f}s)")

    # Add deprecated columns
    old_dataset_names = {colony: dataset for dataset, colony in BASELINE_COLONY_NAMES.items()}
    df["dataset"] = df.colony.map(old_dataset_names)

    # use height calculated from 1st to 99th percentile values
    # rather than the most extreme values
    df["height"] = df["height_percentile"]

    return df.set_index("CellId")


def get_raw_fov_at_timepoint(dataset, index_sequence, channel_name="bright"):
    """
    Parameters
    ----------
    dataset: string
        Name of dataset
    index_sequence: int
        Timepoint to be loaded
    channel_name: string
        Name of the channel to be loaded. Options are "bright" or "egfp"

    Returns
    -------
    dask.array
        Pixel data for the raw FOV at the selected timepoint and channel.
    """
    reader = get_dataset_original_file_reader(dataset)
    channel = get_channel_index(dataset, channel_name)
    assert channel is not None, f"Channel not found for dataset {dataset}"
    return reader.get_image_dask_data("ZYX", T=index_sequence, C=channel)


def get_seg_fov_for_dataset_at_frame(dataset: str, timepoint: int, resolution_level=0) -> da.Array:
    reader = get_dataset_segmentation_file_reader(dataset)
    if resolution_level > 0:
        reader.set_resolution_level(resolution_level)
    # Channel is always 0 because only one channel in segmentation
    return reader.get_image_dask_data("ZYX", T=timepoint, C=0)


def get_seg_fov_for_row(row: pd.Series) -> da.Array:
    return get_seg_fov_for_dataset_at_frame(row.colony, row.index_sequence)


def get_dataset_names(dataset=None):
    """
    Returns list with names of specified individual datasets

    Parameters
    ----------
    dataset: str
        Name of dataset or "all_baseline" or "all_drug" or "all_feeding_control"

    Returns
    -------
    names: list
        List with names of specified individual datasets
    """
    if (dataset == None) or (dataset == "all"):
        dataset_names = get_all_individual_dataset_names()
    elif dataset == "all_baseline":
        dataset_names = get_all_individual_dataset_names(experiment_filter="baseline")
    elif dataset == "all_drug":
        dataset_names = get_all_individual_dataset_names(experiment_filter="drug")
    elif dataset == "all_feeding_control":
        dataset_names = get_all_individual_dataset_names(experiment_filter="feeding")
    else:
        dataset_names = [dataset]
    return dataset_names


def get_all_individual_dataset_names(experiment_filter=None):
    """
    Returns list with names of all individual datasets

    Parameters
    ----------
    experiment_filter: str
        Filter for experiment name.
        example: "baseline" or "feeding" or "drug"

    Returns
    -------
    names: list
        List with names of all individual datasets
    """
    IGNORED_NAMES = [
        # remove names with "all"
        "all_baseline",
        "all_feeding_control",
        "all_drug_perturbation",
        # remove unused or obselete names
        "baby_bear",
        "goldilocks",
        "mama_bear",
    ]

    names = [name for name in datasets.keys() if name not in IGNORED_NAMES]

    if experiment_filter is not None:
        names = [name for name in names if experiment_filter in datasets[name]["experiment"]]

    return names
