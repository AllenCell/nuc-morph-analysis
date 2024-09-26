"""
NOTE: PLEASE READ INSTRUCTIONS AND NOTES ON VERSION CONTROL AT
`write_data_for_colorizer_README.md` BEFORE RUNNING THIS SCRIPT.
"""

from dataclasses import dataclass
import multiprocessing
from typing import List, Sequence, Tuple, Optional
import argparse
import logging
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
import time
from pathlib import Path

from nuc_morph_analysis.lib.preprocessing.load_data import (
    get_dataset_pixel_size,
    get_seg_fov_for_dataset_at_frame,
    get_seg_fov_for_row,
)
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import (
    load_dataset_with_features,
)
from nuc_morph_analysis.lib.visualization.plotting_tools import (
    get_plot_labels_for_metric,
)
from nuc_morph_analysis.lib.visualization.glossary import (
    GLOSSARY,
    )

from colorizer_data.writer import ColorizerDatasetWriter
from colorizer_data.writer import (
    ColorizerMetadata,
    FeatureInfo,
    FeatureType,
)
from colorizer_data.utils import (
    INITIAL_INDEX_COLUMN,
    configureLogging,
    generate_frame_paths,
    get_total_objects,
    scale_image,
    remap_segmented_image,
    update_bounding_box_data,
)


@dataclass
class NucMorphFeatureSpec:
    column_name: str
    type: FeatureType = FeatureType.CONTINUOUS
    categories: Optional[List[str]] = None


# DATASET SPEC: See DATA_FORMAT.md for more details on the dataset format!
# You can find the most updated version on GitHub here:
# https://github.com/allen-cell-animated/colorizer-data/blob/main/documentation/DATA_FORMAT.md

# NUCMORPH DATA REFERENCE:
# colony    string	In FMS manifest	Name of which dataset this row of data belongs to (small, medium, or large)
# track_id	int	In FMS manifest	ID for a single nucleus in all frames for which it exists (single value per nucleus, consistent across multiple frames)
# CellID	hash	In FMS manifest	ID for a single instance/frame of a nucleus (every nucleus has a different value in every frame)
# index_sequence	int	In FMS manifest	frame number associated with the nucleus data in a given row, relative to the start of the movie
# colony_time	int	Needs calculated and added	Frame number staggered by a given amount per dataset, so that the frame numbers in all datasets are temporally algined relative to one another rather than all starting at 0
# is_outlier	boolean	In FMS manifest	True if this nucleus in this frame is flagged as an outlier (a single nucleus may be an outlier in some frames but not others)
# edge_cell	boolean	In FMS manifest	True if this nucleus touches the edge of the FOV
# volume	float	In FMS manifest	Volume of a single nucleus in pixels in a given frame
# height	float	In FMS manifest	Height (in the z-direction) of the a single nucleus in pixels in a given frame
# NUC_PC1	float	Needs calculated and added	Value for shape mode 1 for a single nucleus in a given frame
# NUC_PC2	float	Needs calculated and added	Value for shape mode 2 for a single nucleus in a given frame
# NUC_PC3	float	Needs calculated and added	Value for shape mode 3 for a single nucleus in a given frame
# NUC_PC4	float	Needs calculated and added	Value for shape mode 4 for a single nucleus in a given frame
# NUC_PC5	float	Needs calculated and added	Value for shape mode 5 for a single nucleus in a given frame
# NUC_PC6	float	Needs calculated and added	Value for shape mode 6 for a single nucleus in a given frame
# NUC_PC7	float	Needs calculated and added	Value for shape mode 7 for a single nucleus in a given frame
# NUC_PC8	float	Needs calculated and added	Value for shape mode 8 for a single nucleus in a given frame


OBJECT_ID_COLUMN = "label_img"
"""Column of object IDs (or unique row number)."""
TRACK_ID_COLUMN = "track_id"
"""Column of track ID for each object."""
TIMES_COLUMN = "index_sequence"
"""Column of the name of the colony/dataset."""
COLONY_COLUMN = "colony"
"""Column of path to the segmented image data or z stack for the frame."""
CENTROIDS_X_COLUMN = "centroid_x"
"""Column of X centroid coordinates, in pixels of original image data."""
CENTROIDS_Y_COLUMN = "centroid_y"
"""Column of Y centroid coordinates, in pixels of original image data."""
OUTLIERS_COLUMN = "is_outlier"
"""Column of outlier status for each object. (true/false)"""

"""Columns of feature data to include in the dataset. Each column will be its own feature file."""

FEATURE_COLUMNS = {
    "baseline_colonies_dataset": [
        NucMorphFeatureSpec("volume"),
        NucMorphFeatureSpec("height"),
        NucMorphFeatureSpec("xy_aspect"),
        NucMorphFeatureSpec("dxdt_48_volume"),
        NucMorphFeatureSpec("density"),
        NucMorphFeatureSpec("normalized_colony_depth"),
        NucMorphFeatureSpec(
            "termination", FeatureType.CATEGORICAL, ["Division", "Leaves FOV", "Apoptosis"]
        ),
        NucMorphFeatureSpec("distance_from_centroid"),
        NucMorphFeatureSpec("neighbor_avg_dxdt_48_volume_whole_colony"),
        NucMorphFeatureSpec("neighbor_avg_dxdt_48_volume_90um"),
        NucMorphFeatureSpec("zy_aspect"),
        NucMorphFeatureSpec("xz_aspect"),
    ],
    "full_interphase_dataset": [
        NucMorphFeatureSpec("volume"),
        NucMorphFeatureSpec("height"),
        NucMorphFeatureSpec("xy_aspect"),
        NucMorphFeatureSpec("volume_at_B"),
        NucMorphFeatureSpec("volume_at_C"),
        NucMorphFeatureSpec("volume_fold_change_BC"),
        NucMorphFeatureSpec("delta_volume_BC"),
        NucMorphFeatureSpec("duration_BC"),
        NucMorphFeatureSpec("late_growth_rate_by_endpoints"),
        NucMorphFeatureSpec("tscale_linearityfit_volume"),
        NucMorphFeatureSpec("dxdt_48_volume"),
        NucMorphFeatureSpec("density"),
        NucMorphFeatureSpec("normalized_time"),
        NucMorphFeatureSpec("sync_time_Ff"),
        NucMorphFeatureSpec("time_at_B"),
        NucMorphFeatureSpec("colony_time_at_B"),
        NucMorphFeatureSpec("normalized_colony_depth"),
        NucMorphFeatureSpec(
            "termination", FeatureType.CATEGORICAL, ["Division", "Leaves FOV", "Apoptosis"]
        ),
        NucMorphFeatureSpec("volume_at_A"),
        NucMorphFeatureSpec("time_at_A"),
        NucMorphFeatureSpec("time_at_C"),
        NucMorphFeatureSpec("duration_AB"),
        NucMorphFeatureSpec("duration_AC"),
        NucMorphFeatureSpec("growth_rate_AB"),
        NucMorphFeatureSpec("volume_fold_change_fromB"),
        NucMorphFeatureSpec("distance_from_centroid"),
        NucMorphFeatureSpec("neighbor_avg_dxdt_48_volume_whole_colony"),
        NucMorphFeatureSpec("neighbor_avg_dxdt_48_volume_90um"),
        NucMorphFeatureSpec("zy_aspect"),
        NucMorphFeatureSpec("xz_aspect"),
    ],
    "lineage_annotated_dataset": [
        NucMorphFeatureSpec("volume"),
        NucMorphFeatureSpec("height"),
        NucMorphFeatureSpec("xy_aspect"),
        NucMorphFeatureSpec("family_id", FeatureType.DISCRETE),
        NucMorphFeatureSpec("volume_at_B"),
        NucMorphFeatureSpec("volume_at_C"),
        NucMorphFeatureSpec("volume_fold_change_BC"),
        NucMorphFeatureSpec("delta_volume_BC"),
        NucMorphFeatureSpec("duration_BC"),
        NucMorphFeatureSpec("late_growth_rate_by_endpoints"),
        NucMorphFeatureSpec("tscale_linearityfit_volume"),
        NucMorphFeatureSpec("dxdt_48_volume"),
        NucMorphFeatureSpec("density"),
        NucMorphFeatureSpec("normalized_time"),
        NucMorphFeatureSpec("sync_time_Ff"),
        NucMorphFeatureSpec("time_at_B"),
        NucMorphFeatureSpec("colony_time_at_B"),
        NucMorphFeatureSpec("normalized_colony_depth"),
        NucMorphFeatureSpec(
            "termination", FeatureType.CATEGORICAL, ["Division", "Leaves FOV", "Apoptosis"]
        ),
        NucMorphFeatureSpec("volume_at_A"),
        NucMorphFeatureSpec("time_at_A"),
        NucMorphFeatureSpec("time_at_C"),
        NucMorphFeatureSpec("duration_AB"),
        NucMorphFeatureSpec("duration_AC"),
        NucMorphFeatureSpec("growth_rate_AB"),
        NucMorphFeatureSpec("volume_fold_change_fromB"),
        NucMorphFeatureSpec("distance_from_centroid"),
        NucMorphFeatureSpec("neighbor_avg_dxdt_48_volume_whole_colony"),
        NucMorphFeatureSpec("neighbor_avg_dxdt_48_volume_90um"),
        NucMorphFeatureSpec("zy_aspect"),
        NucMorphFeatureSpec("xz_aspect"),
    ],
    "exploratory_dataset": [
        NucMorphFeatureSpec("volume"),
        NucMorphFeatureSpec("height"),
        NucMorphFeatureSpec("xy_aspect"),
        NucMorphFeatureSpec("family_id", FeatureType.DISCRETE),
        NucMorphFeatureSpec("volume_at_B"),
        NucMorphFeatureSpec("volume_at_C"),
        NucMorphFeatureSpec("volume_fold_change_BC"),
        NucMorphFeatureSpec("delta_volume_BC"),
        NucMorphFeatureSpec("duration_BC"),
        NucMorphFeatureSpec("late_growth_rate_by_endpoints"),
        NucMorphFeatureSpec("tscale_linearityfit_volume"),
        NucMorphFeatureSpec("dxdt_48_volume"),
        NucMorphFeatureSpec("density"),
        NucMorphFeatureSpec("normalized_time"),
        NucMorphFeatureSpec("sync_time_Ff"),
        NucMorphFeatureSpec("time_at_B"),
        NucMorphFeatureSpec("colony_time_at_B"),
        NucMorphFeatureSpec("normalized_colony_depth"),
        NucMorphFeatureSpec(
            "termination", FeatureType.CATEGORICAL, ["Division", "Leaves FOV", "Apoptosis"]
        ),
        NucMorphFeatureSpec("is_growth_outlier", FeatureType.CATEGORICAL, ["False", "True"]),
        NucMorphFeatureSpec(
            "baseline_colonies_dataset", FeatureType.CATEGORICAL, ["False", "True"]
        ),
        NucMorphFeatureSpec("full_interphase_dataset", FeatureType.CATEGORICAL, ["False", "True"]),
        NucMorphFeatureSpec(
            "lineage_annotated_dataset", FeatureType.CATEGORICAL, ["False", "True"]
        ),
        NucMorphFeatureSpec("volume_at_A"),
        NucMorphFeatureSpec("time_at_A"),
        NucMorphFeatureSpec("time_at_C"),
        NucMorphFeatureSpec("duration_AB"),
        NucMorphFeatureSpec("duration_AC"),
        NucMorphFeatureSpec("growth_rate_AB"),
        NucMorphFeatureSpec("volume_fold_change_fromB"),
        NucMorphFeatureSpec("distance_from_centroid"),
        NucMorphFeatureSpec("neighbor_avg_dxdt_48_volume_whole_colony"),
        NucMorphFeatureSpec("neighbor_avg_dxdt_48_volume_90um"),
        NucMorphFeatureSpec("zy_aspect"),
        NucMorphFeatureSpec("xz_aspect"),
        NucMorphFeatureSpec("mesh_sa"),
        NucMorphFeatureSpec("SA_at_B"),
        NucMorphFeatureSpec("SA_at_C"),
        NucMorphFeatureSpec("SA_fold_change_BC"),
        NucMorphFeatureSpec("SA_fold_change_fromB"),
        NucMorphFeatureSpec("delta_SA_BC"),
        NucMorphFeatureSpec("SA_vol_ratio"),

        # mitotic and apoptotic neighbor columns
        NucMorphFeatureSpec(column_name="frame_of_breakdown", type=FeatureType.CATEGORICAL, categories=["False", "True"]),
        NucMorphFeatureSpec(column_name="frame_of_formation", type=FeatureType.CATEGORICAL, categories=["False", "True"]),
        NucMorphFeatureSpec(column_name="number_of_frame_of_breakdown_neighbors"),
        NucMorphFeatureSpec(column_name="number_of_frame_of_formation_neighbors"),
        NucMorphFeatureSpec(column_name="has_mitotic_neighbor_breakdown", type=FeatureType.CATEGORICAL, categories=["False", "True"]),
        NucMorphFeatureSpec(column_name="has_mitotic_neighbor_formation", type=FeatureType.CATEGORICAL, categories=["False", "True"]),
        NucMorphFeatureSpec(column_name="has_mitotic_neighbor_breakdown_forward_dilated", type=FeatureType.CATEGORICAL, categories=["False", "True"]),
        NucMorphFeatureSpec(column_name="has_mitotic_neighbor_formation_backward_dilated", type=FeatureType.CATEGORICAL, categories=["False", "True"]),
        NucMorphFeatureSpec(column_name="has_mitotic_neighbor", type=FeatureType.CATEGORICAL, categories=["False", "True"]),
        NucMorphFeatureSpec(column_name="has_mitotic_neighbor_dilated", type=FeatureType.CATEGORICAL, categories=["False", "True"]),
        NucMorphFeatureSpec(column_name="frame_of_death", type=FeatureType.CATEGORICAL, categories=["False", "True"]),
        NucMorphFeatureSpec(column_name="has_dying_neighbor", type=FeatureType.CATEGORICAL, categories=["False", "True"]),
        NucMorphFeatureSpec(column_name="has_dying_neighbor_forward_dilated", type=FeatureType.CATEGORICAL, categories=["False", "True"]),
        NucMorphFeatureSpec(column_name="number_of_frame_of_death_neighbors"),
        NucMorphFeatureSpec(column_name="sum_has_mitotic_neighbor_breakdown"), # per track feature
        NucMorphFeatureSpec(column_name="sum_has_mitotic_neighbor_formation"),# per track feature
        NucMorphFeatureSpec(column_name="sum_has_mitotic_neighbor"),# per track feature
        NucMorphFeatureSpec(column_name="sum_has_dying_neighbor"),# per track feature
        NucMorphFeatureSpec(column_name="sum_number_of_frame_of_breakdown_neighbors"),# per track feature
        NucMorphFeatureSpec(column_name="number_of_frame_of_death_neighbors"),# per track feature


    ],
}


def make_frame(
    grouped_frames: DataFrameGroupBy,
    group_name: int,
    frame: pd.DataFrame,
    scale: float,
    bounds_arr: Sequence[int],
    writer: ColorizerDatasetWriter,
):
    start_time = time.time()

    # Get the path to the segmented zstack image frame from the first row (should be the same for
    # all rows in this group, since they are all on the same frame).
    row = frame.iloc[0]
    frame_number = row[TIMES_COLUMN]
    # Flatten the z-stack to a 2D image.
    zstack = get_seg_fov_for_dataset_at_frame(row[COLONY_COLUMN], frame_number)
    seg2d = zstack.max(axis=0)
    # Flip vertically to match our figure orientation
    seg2d = np.flipud(seg2d)

    # Scale the image and format as integers.
    seg2d = scale_image(seg2d.compute(), scale)
    seg2d = seg2d.astype(np.uint32)

    # Remap the frame image so the IDs are unique across the whole dataset.
    seg_remapped, lut = remap_segmented_image(
        seg2d,
        frame,
        OBJECT_ID_COLUMN,
    )

    writer.write_image(seg_remapped, frame_number)
    update_bounding_box_data(bounds_arr, seg_remapped)

    time_elapsed = time.time() - start_time
    logging.info("Frame {} finished in {:5.2f} seconds.".format(int(frame_number), time_elapsed))


def make_all_frames(
    grouped_frames: DataFrameGroupBy,
    scale: float,
    writer: ColorizerDatasetWriter,
    parallel: bool,
):
    """
    Generate the images and bounding boxes for each time step in the dataset.
    """
    nframes = len(grouped_frames)
    total_objects = get_total_objects(grouped_frames)
    logging.info("Making {} frames...".format(nframes))

    with multiprocessing.Manager() as manager:
        bounds_arr = manager.Array("i", [0] * int(total_objects * 4))
        if parallel:
            with multiprocessing.Pool() as pool:
                pool.starmap(
                    make_frame,
                    [
                        (grouped_frames, group_name, frame, scale, bounds_arr, writer)
                        for group_name, frame in grouped_frames  # type: ignore
                    ],
                )
        else:
            # mypy seems to thing DataFrameGroupBy.__iter__ is undefined, but it is.
            for group_name, frame in grouped_frames:  # type: ignore [attr-defined]
                make_frame(grouped_frames, group_name, frame, scale, bounds_arr, writer)
        writer.write_data(bounds=np.array(bounds_arr, dtype=np.uint32))


def make_features(
    dataset: pd.DataFrame,
    features: List[NucMorphFeatureSpec],
    dataset_name: str,
    writer: ColorizerDatasetWriter,
):
    """
    Generate the outlier, track, time, centroid, and feature data files.
    """
    # Collect array data from the dataframe for writing.
    outliers = dataset[OUTLIERS_COLUMN].to_numpy()
    tracks = dataset[TRACK_ID_COLUMN].to_numpy()
    times = dataset[TIMES_COLUMN].to_numpy()
    centroids_x = dataset[CENTROIDS_X_COLUMN].to_numpy()
    centroids_y = dataset[CENTROIDS_Y_COLUMN].to_numpy()
    # flip y coordinates to match our figure orientation
    centroids_y = 3120 - centroids_y

    writer.write_data(
        tracks=tracks,
        times=times,
        centroids_x=centroids_x,
        centroids_y=centroids_y,
        outliers=outliers,
    )

    for feature in features:
        if feature.column_name not in dataset.columns:
            logging.warning(
                "Feature '{}' not found in dataset. Skipping...".format(feature.column_name)
            )
            continue

        (scale_factor, label, unit, _) = get_plot_labels_for_metric(
            feature.column_name,
            dataset=dataset_name,
            colorizer=True,
        )
        
        # Remove parentheses from unit names, if included.
        if len(unit) >= 2 and unit[0] == "(" and unit[-1] == ")":
            unit = unit[1:-1]

        data = dataset[feature.column_name]

        # Get data and scale to use actual units
        if scale_factor is not None:
            data = data * scale_factor
            
        description = GLOSSARY[feature.column_name]

        writer.write_feature(
            data,
            FeatureInfo(label=label, unit=unit, type=feature.type, categories=feature.categories, description=description),
            outliers=outliers,
        )


def get_dataset_dimensions(
    grouped_frames: DataFrameGroupBy, pixsize: float
) -> Tuple[float, float, str]:
    """Get the dimensions of the dataset from the first frame, in units.
    Returns (width, height, unit)."""
    row = grouped_frames.get_group(0).iloc[0]
    # Uses COLONY_COLUMN (colony)
    zstack = get_seg_fov_for_row(row)
    (_, Y, X) = zstack.shape
    return (X * pixsize, Y * pixsize, "µm")


def make_dataset(
    output_dir="./data/",
    dataset="all",
    filter="all",
    do_frames=True,
    scale=0.25,
    parallel=False,
):
    """Make a new dataset from the given data, and write the complete dataset
    files to the given output directory.
    """

    if args.filter == "all":
        filters = FEATURE_COLUMNS.keys()
    else:
        filters = [args.filter]

    # load the dataset once
    df_all = load_dataset_with_features("all_baseline", remove_growth_outliers=False)

    for filter in filters:
        output_dir_subset = Path(output_dir) / filter
        output_dir_subset.mkdir(parents=True, exist_ok=True)
        output_dir_subset = str(output_dir_subset)

        df_filter = df_all.copy()
        df_filter.loc[df_filter[filter] == False, "is_outlier"] = True

        if filter == "lineage_annotated_dataset":
            df_filter = df_filter[df_filter[COLONY_COLUMN] != "large"]

        if args.dataset == "all":
            datasets = df_filter.colony.unique()
        else:
            datasets = [args.dataset]

        for dataset in datasets:

            logging.info("Writing data for " + dataset + " colony " + filter)
            writer = ColorizerDatasetWriter(output_dir_subset, dataset, scale=scale)

            # filter to colony and subset
            pixsize = get_dataset_pixel_size(dataset)
            df = df_filter[df_filter[COLONY_COLUMN] == dataset]
            logging.info("Loaded colony '" + str(dataset) + "'.")
            full_dataset = df

            # set nuclei with ture fov_edge to have true is_outlier
            # otherwise edge nuclei will be colorized
            df.loc[df["fov_edge"], "is_outlier"] = True

            # Make a reduced dataframe grouped by time (frame number).
            columns = [TRACK_ID_COLUMN, TIMES_COLUMN, COLONY_COLUMN, OBJECT_ID_COLUMN]
            reduced_dataset = full_dataset[columns]
            reduced_dataset = reduced_dataset.reset_index(drop=True)
            reduced_dataset[INITIAL_INDEX_COLUMN] = reduced_dataset.index.values
            grouped_frames = reduced_dataset.groupby(TIMES_COLUMN)

            dims = get_dataset_dimensions(grouped_frames, pixsize)
            metadata = ColorizerMetadata(
                frame_width=dims[0],
                frame_height=dims[1],
                frame_units=dims[2],
                name="Nucmorph Dataset v3",
                description="",
                author="Allen Institute for Cell Science",
                dataset_version="v3.0.0",
                frame_duration_sec=300,
            )

            # Make the features, frame data, and manifest.
            nframes = len(grouped_frames)
            writer.set_frame_paths(generate_frame_paths(nframes))

            make_features(full_dataset, FEATURE_COLUMNS[filter], dataset, writer)
            if do_frames:
                make_all_frames(grouped_frames, scale, writer, parallel)
            writer.write_manifest(metadata=metadata)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    type=str,
    default="./data/",
    help="Parent directory to output to. Data will be written to a subdirectory named after the dataset parameter.",
)

parser.add_argument(
    "--dataset",
    type=str,
    default="all",
    help="Compatible named FMS dataset or FMS id to load. Options are 'small', 'medium', 'large', or 'all' which creates datasets from all colonies.",
)
parser.add_argument(
    "--noframes",
    action="store_true",
    help="If included, generates only the feature data, centroids, track data, and manifest, skipping the frame and bounding box generation.",
)
parser.add_argument(
    "--scale",
    type=float,
    default=0.25,
    help="Uniform scale factor that original image dimensions will be scaled by. 1 is original size, 0.5 is half-size in both X and Y.",
)
parser.add_argument(
    "--filter",
    type=str,
    default="all",
    help="Dataset to filter to and use features from. Options are 'baseline_colonies_dataset', 'full_interphase_dataset', 'lineage_annotated_dataset', 'exploratory_dataset' or 'all' which creates datasets with all filters. Default is 'all'",
)
parser.add_argument(
    "--parallel",
    action="store_true",
    help="Generate frames in parallel. Faster, but demands more memory. "
    "Has no effect with --noframes.",
)

args = parser.parse_args()


def main():
    configureLogging(args.output_dir)
    logging.info("Starting...")

    make_dataset(
        output_dir=args.output_dir,
        dataset=args.dataset,
        filter=args.filter,
        do_frames=not args.noframes,
        scale=args.scale,
        parallel=args.parallel,
    )


if __name__ == "__main__":
    main()
