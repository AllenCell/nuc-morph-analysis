import functools
import time
import dask.array as da
import pandas as pd
import platform
from bioio import BioImage

from nuc_morph_analysis.lib.preprocessing.all_datasets import (
    datasets,
    manual_curation_manifests,
    morflowgenesis_outputs,
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


def get_valid_path(record):
    """
    Converts a FMS path to one that can be read cross-platform
    """
    recordpath = record.path
    if platform.system() == "Windows":
        recordpath = "/" + recordpath
    return recordpath


def get_dataframe_by_id(fmsid, format="csv"):
    """
    Read a CSV or parquet file from FMS into a pandas dataframe
    """
    fms = FileManagementSystem()
    record = fms.get_file_by_id(fmsid)
    recordpath = get_valid_path(record)
    if format == "csv":
        return pd.read_csv(recordpath)
    elif format == "parquet":
        return pd.read_parquet(recordpath)
    else:
        raise ValueError(f"Unknown format {format}")


def load_morflowgenesis_dataframe(dataset):
    fmsid = morflowgenesis_outputs[dataset]["fmsid"]
    single_colony_df = get_dataframe_by_id(fmsid)
    single_colony_df.set_index("CellId")
    return single_colony_df


def load_lineage_annotations(dataset: str) -> pd.DataFrame:
    # Currently only the small and medium datasets have lineage annotations
    assert dataset == "small" or dataset == "medium"
    fmsid = manual_curation_manifests["lineage_annotations"][dataset]["fmsid"]
    return get_dataframe_by_id(fmsid)


def load_apoptosis_annotations(dataset: str) -> pd.DataFrame:
    assert dataset == "large"  # Currently only the large dataset has apoptosis annotations
    fmsid = manual_curation_manifests["apoptosis_annotations"][dataset]["fmsid"]
    return get_dataframe_by_id(fmsid)


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
        ds["name"]
        for ds in datasets
        if (
            (ds["name"] not in ["all_baseline", "all_feeding_control", "all_drug_perturbation"])
            and (ds["experiment"] in experiments)
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

    for ds in datasets:
        if name == ds["name"]:
            return ds["experiment"]
    raise ValueError(f"Dataset {name} not available.")


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

    for ds in datasets:
        if name == ds["name"]:
            return ds
    raise ValueError(f"Dataset {name} not available.")


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


def load_dataset(dataset, datadir=None, nrows=None):
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

    datadir: String
        String giving path to data directory for this dataset

    nrows: int
        Only read first nrows of the dataset. Useful for debug since
        loading the datasets takes some time. Leave it as None to read
        all rows.

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
        fms = FileManagementSystem()
        if datadir is not None:
            record = fms.retrieve_file(info["fmsid"], datadir)[1]
        else:
            record = fms.get_file_by_id(info["fmsid"])
        df = pd.read_csv(record.path, nrows=nrows).set_index("CellId")
        df["dataset"] = dataset

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
        See docs/manifest-columns.md for a description of all the columns.
    """
    fmsid = get_dataset_info_by_name(all_dataset)["fmsid"]
    start = time.time()
    df = get_dataframe_by_id(fmsid, format="parquet")
    print(f"Read dataset with FMS ID {fmsid} from parquet file ({time.time()-start:.1f}s)")

    # Add deprecated columns
    old_dataset_names = {colony: dataset for dataset, colony in BASELINE_COLONY_NAMES.items()}
    df["dataset"] = df.colony.map(old_dataset_names)

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


def get_seg_fov_for_dataset_at_frame(dataset: str, timepoint: int) -> da.Array:
    reader = get_dataset_segmentation_file_reader(dataset)
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
    names = [ds["name"] for ds in datasets]

    # remove names with "all)"
    names = [
        name
        for name in names
        if name not in ["all_baseline", "all_feeding_control", "all_drug_perturbation"]
    ]

    # now remove unused or obselete names
    names = [name for name in names if name not in ["baby_bear", "goldilocks", "mama_bear"]]

    if experiment_filter is not None:
        names = [
            ds["name"]
            for ds in datasets
            if (experiment_filter in ds["experiment"]) & (ds["name"] in names)
        ]

    return names
