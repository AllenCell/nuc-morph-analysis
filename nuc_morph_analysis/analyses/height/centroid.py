import numpy as np


def get_centroid(df_timepoint):
    """
    Calculate the centroid of a given timepoint DataFrame.

    Parameters
    ----------
    df_timepoint : pandas.DataFrame
        DataFrame containing the centroid coordinates for each object at a specific timepoint.

    Returns
    -------
    frame_centroids : numpy.ndarray
        Array of centroid coordinates for each object in the timepoint DataFrame.
    track_centroid : numpy.ndarray
        Centroid coordinates of the closest object to the colony centroid.
    df_nuc : pandas.Series
        Series containing the centroid coordinates and other information of the
        closest object to the colony centroid.
    """
    frame_centroids = df_timepoint[["centroid_x", "centroid_y"]].values
    colony_centroid = np.mean(frame_centroids, axis=0)
    distance_from_origin = np.linalg.norm(frame_centroids - colony_centroid, axis=1)
    closest_to_origin = np.argmin(distance_from_origin)
    df_nuc = df_timepoint.iloc[closest_to_origin]
    track_centroid = df_nuc[["centroid_x", "centroid_y"]].values.astype(float)

    return frame_centroids, track_centroid, df_nuc


def get_neighbor_centroids(df_nuc, df_timepoint):
    """
    Get the centroids of neighboring cells.

    Parameters
    ----------
    df_nuc : DataFrame
        The DataFrame containing information about the nucleus.
    df_timepoint : DataFrame
        The DataFrame containing information about the timepoint.

    Returns
    -------
    neighbor_centroids : ndarray
        An array of shape (N, 2) containing the x and y coordinates of the
        centroids of neighboring cells.
    """
    neighbor_centroids_list = []
    neighbor_list = eval(df_nuc.neighbors)
    for neighbor_index in neighbor_list:
        df_neighbor = df_timepoint[df_timepoint.CellId == neighbor_index]
        centroid_neighbor = df_neighbor[["centroid_x", "centroid_y"]].values
        neighbor_centroids_list.append(centroid_neighbor)
    neighbor_centroids = np.concatenate(neighbor_centroids_list, axis=0, dtype=float)

    return neighbor_centroids
