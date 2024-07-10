from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, ConvexHull
import shapely

from nuc_morph_analysis.lib.preprocessing.colony_depth import calculate_depth
from nuc_morph_analysis.utilities.warn_slow import warn_slow


@warn_slow("6min")  # Usually takes ~3min
def add_colony_metrics(df: pd.DataFrame):
    """
    Reads in a dataframe or CSV and outputs the same dataset with added colony metrics columns.
    Colony metrics are calculated for all nuclei above the built in volume threshold (11,000
    isotropic voxels, about 14 cubic microns if the voxel width is 0.108um). Nuclei below the volume
    threshold are ignored in all colony metric calculations.

    Parameters:
    ----------
    df: pandas.DataFrame
        Input dataframe. Must have the following columns:
          CellId
          centroid_x
          centroid_y
          label_img
          volume

    Returns
    -------
    all_time: pandas.DataFrame
        Same rows as input df, with the following new columns:
          colony_depth: int. Outermost ring of cells is the FOV is depth 1, those adjacent to depth
                        1 are depth 2, etc.
          neighbors: string. List of neighboring Cell IDs
          neigh_distance: float. Unit: voxels. Mean distance to neighboring cells
          density: float. Unit: voxels. 1 / neigh_distance^2
    """
    # This function is only intended to run on data from one colony at a time
    if "dataset" in df.columns:
        if len(df.dataset.unique()) > 1:
            raise ValueError(
                "Cannot add colony metrics. Input dataframe must only have data from one colony."
            )

    df["colony_depth"] = np.nan
    df["neighbors"] = np.nan

    with Pool() as pool:
        groups = [df_timepoint for _, df_timepoint in df.groupby(by="index_sequence")]
        all_time = pool.map(_add_colony_metrics_one_tp, groups)

    return pd.concat(all_time, axis="rows")


def _add_colony_metrics_one_tp(df_timepoint: pd.DataFrame):
    depth_map, neighborhoods, neigh_dists, densities = _calc_colony_metrics(df_timepoint)
    for _, (lbl, depth) in enumerate(depth_map.items()):
        df_timepoint.loc[df_timepoint["label_img"] == lbl, "colony_depth"] = depth

    for _, (lbl, dist) in enumerate(neigh_dists.items()):
        df_timepoint.loc[df_timepoint["label_img"] == lbl, "neigh_distance"] = dist

    for _, (lbl, density) in enumerate(densities.items()):
        df_timepoint.loc[df_timepoint["label_img"] == lbl, "density"] = density

    for _, (lbl, neighbors) in enumerate(neighborhoods.items()):
        neighbor_ids = []
        for neighbor in neighbors:
            neighbor_ids.append(
                df_timepoint.loc[df_timepoint["label_img"] == neighbor, "CellId"].values[0]
            )

        df_timepoint.loc[df_timepoint["label_img"] == lbl, "neighbors"] = str(neighbor_ids)
    return df_timepoint


def _calc_colony_metrics(df_timepoint):
    # size filter
    df_timepoint = df_timepoint[df_timepoint["volume"].values > 11000]

    # Build the Voronoi diagram
    centroids_list = df_timepoint[["centroid_x", "centroid_y"]].to_numpy()
    voronoi = Voronoi(centroids_list)

    # Iterate over the cell-cell boundaries in the voronoi diagram get cell adjacency
    labels = df_timepoint["label_img"].values
    neighbors = _make_neighbor_map(voronoi, labels)

    centroids_by_label = {label: centroids_list[index] for index, label in enumerate(labels)}
    neigh_distance, density = _calculate_distance_density(labels, neighbors, centroids_by_label)

    depth1_labels = _get_depth1_labels(labels, centroids_list, voronoi)
    depth_map = calculate_depth(neighbors, depth1_labels)

    return depth_map, neighbors, neigh_distance, density


def _calculate_distance_density(labels, neighbors, centroids):
    density = {}
    neigh_distance = {}
    for lbl in labels:
        try:
            neighbor_labels = neighbors[lbl]
            # rest of your code
        except KeyError:
            # The missing lbl is a symptom of the problem. The count of
            # neighbors is confused by the cells stacked in the z dimension.
            print(f"KeyError: {lbl} not found in neighbors.  This may be caused by cell tracks overlapping in the xy plane")
            continue
        centroid = np.array(centroids[lbl])
        dists = []
        for neighbor in neighbor_labels:
            if neighbor != lbl:
                dist = np.sqrt(np.sum((centroid - np.array(centroids[neighbor])) ** 2, axis=0))
                dists.append(dist)
        density[lbl] = 1 / np.mean(dists) ** 2
        neigh_distance[lbl] = np.mean(dists)
    return neigh_distance, density


def _make_neighbor_map(voronoi, labels):
    neighbor_map = {}
    # Voronoi.ridge_points is "Indices of the points between which each Voronoi ridge lies."
    # In other words, each boundary of a cell in the voronoi diagram is represented by one
    # item in the ridge_points list. Each item in the list is an array of length 2 containing
    # the indices of the cell centroids on either side of the boundary.
    for [index1, index2] in voronoi.ridge_points:
        label1 = labels[index1]
        label2 = labels[index2]
        if label1 not in neighbor_map:
            neighbor_map[label1] = set()
        if label2 not in neighbor_map:
            neighbor_map[label2] = set()
        neighbor_map[label1].add(label2)
        neighbor_map[label2].add(label1)
    return neighbor_map


def _get_depth1_labels(labels, centroids, voronoi):
    """
    Nuclei at colony depth 1 are determined as follows.
      1. Compute the convex hull of the centroids of the nuclei.
      2. Compute a voronoi tesselation of the plane using the centroids of the nuclei.
      3. If a nucleus's voronoi cell intersects the boundary of the convex hull, that nucleus has
         depth 1.

    The relevant test for this function is test_voronoi.test_voronoi_depths

    Parameters
    ----------
    labels: list[T]
    centroids: list[list[int]]
        The indices of labels and centroids are paired: labels[i] corresponds to centroids[i] for
        all i
    voronoi: scipy.spatial.Voronoi

    Returns
    -------
    label_subset: list[T]
    """
    # Compute the boundary of convex hull
    hull = ConvexHull(centroids)
    hull_shape = shapely.LinearRing(hull.points[hull.vertices])

    # For each line segment ("ridge") in the voronoi diagram, determine if that ridge intersects the
    # boundary of the convex hull
    ridges = [shapely.LineString(ridge) for ridge in _voronoi_ridge_cartesian(voronoi)]
    intersecting_ridges = [hull_shape.intersects(ridge) for ridge in ridges]

    # If the boundary of the convex hull intersects a line segment, it must intersect the voronoi
    # cells on either side.
    # outer_centroid_indices is an integer array of indices into the centroids list
    # This line flattens and converts to a set to remove duplicates, as voronoi cells in general
    # have multiple intersecting ridges
    outer_centroid_indices = list(set(voronoi.ridge_points[intersecting_ridges].flat))

    # In addition, add all points that are vertices of the convex hull. It is possible for a
    # centroid on the convex hull not to be in outer_centroid_indices if the "ridges" that the
    # convex hull would intersect have one end at infinity. Such rays are not included in the list
    # of ridges.
    # This can only happen for infinite voronoi cells, which are indicated by an index of -1 in
    # Voronoi.regions (-1 in qhull refers to the "point at infinity").
    convex_hull_points_mask = [
        -1 in voronoi.regions[region_index] for region_index in voronoi.point_region
    ]

    # Remove duplicate labels from the two collections with set
    return list(set(labels[convex_hull_points_mask]) | set(labels[outer_centroid_indices]))


def _voronoi_vertex_indices_to_cartesian(voronoi, vertex_indices):
    return [voronoi.vertices[i] for i in vertex_indices]


def _voronoi_ridge_cartesian(voronoi):
    return [
        _voronoi_vertex_indices_to_cartesian(voronoi, ridge_vertex_indices)
        for ridge_vertex_indices in voronoi.ridge_vertices
    ]
