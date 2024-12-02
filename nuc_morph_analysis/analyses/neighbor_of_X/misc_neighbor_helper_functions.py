import numpy as np
from scipy.spatial import distance_matrix

def find_immediate_neighbors(dfsearch,main_track_id,timepoint):
    colony = dfsearch[dfsearch["track_id"] == main_track_id]["colony"].values[0]
    dfcolony=dfsearch[dfsearch['colony']==colony]

    # get tracks by looking for neighbors
    dftrack = dfcolony[(dfcolony["track_id"] == main_track_id) & (dfcolony["index_sequence"] == timepoint)]
    cell_ids = dftrack["neighbors"].apply(lambda x:eval(x)).values[0]

    dftime = dfcolony[dfcolony["index_sequence"] == timepoint]
    immediate_neighbor_track_ids = dftime.loc[dftime.index.isin(cell_ids), "track_id"].values
    return immediate_neighbor_track_ids


def compute_distances_between_nuclei(dftime,main_track_id):
    # alternative workflow to get tracks by measuring distance to neighbors
    dftrack = dftime[(dftime["track_id"] == main_track_id)]
    centroids = dftrack[["centroid_x", "centroid_y"]].values
    centroids_time = dftime[["centroid_x", "centroid_y"]].values
    dist = distance_matrix(centroids, centroids_time)
    return dist


def get_ordered_list_of_nuclei_by_distance_to_track(dist,dftime):
    # now determine the ordered list of neighbors
    sorted_index = np.argsort(dist, axis=1).reshape(-1,)
    sorted_dist = np.sort(dist, axis=1).reshape(-1,)
    sorted_cell_ids = dftime.index.values[sorted_index]
    sorted_track_ids = dftime['track_id'].values[sorted_index]
    return sorted_dist,sorted_cell_ids,sorted_track_ids,sorted_index

def get_a_cells_neighbors_as_track_id_list(df0,main_track_id,TIMEPOINT,return_self=True):
    """
    identify the neighboring tracks of a cell at a given timepoint

    Parameters
    ----------
    df0 : pd.DataFrame
        dataframe with columns ['colony','track_id','index_sequence','label_img','frame_transition','neighbors']
    main_track_id : int
        track_id of the cell of interest
    timepoint : int
        timepoint at which to find neighbors
    return_self : bool
        whether to include the cell of interest in the list of neighbors

    Returns
    -------
    track_id_list : list
        list of track_ids that are neighbors of the cell of interest at the given
    """
    dftrack = df0[df0["track_id"] == main_track_id]
    colony = dftrack["colony"].values[0]
    dfcolony = df0[df0["colony"] == colony]
    df_after_transition_only = dfcolony[dfcolony['index_sequence'] > dfcolony['frame_transition']] # only include after growth

    # find all neighbors that are immediate neighbors (and have passed the transition point)
    immediate_neighbor_track_ids = find_immediate_neighbors(df_after_transition_only,main_track_id,TIMEPOINT)

    # combine the main track with the immediate neighbors into a list
    track_id_list = list(immediate_neighbor_track_ids)
    if return_self:
        track_id_list.append(main_track_id)
    return track_id_list
