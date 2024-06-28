import numpy as np

# Get tracks related to a particular track_id


def get_parent(df, track_id):
    """
    Gets the track_id of the parent

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe
    track_id: int
        The track_id of the cell

    Returns
    -------
    parent_id: List
        List containing the track_id of the parent cell
    """
    parent_id = df.loc[df.track_id == track_id].parent_id.unique()
    parent_id = [i for i in parent_id if i != -1]
    return parent_id


def get_offsprings(df, track_id):
    """
    Gets the track_id(s) of the child(ren)

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe
    track_id: Int
        The track_id of the cell

    Returns
    -------
    offsprings: List
        List containing the track_ids (int) of the children
    """
    offsprings = df.loc[df.parent_id == track_id].track_id.unique()
    return offsprings.tolist()


def get_relatives(df, track_id):
    """
    Gets the track_ids of the directly related generations. The parent and the children.

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe
    track_id: Int
        The track_id of the cell

    Returns
    -------
    track_ids: List
        List containing the track_ids of the immediate relatives
    """
    track_ids = get_parent(df, track_id) + get_offsprings(df, track_id)
    return track_ids


def get_ids_in_same_lineage_as(df, track_id):
    """
    Gets the track_ids all the related tracks.

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe
    track_id: Int
        The track_id of the cell

    Returns
    -------
    lineage: List
        List containing the track_ids in the lineage
    """
    lineage = [track_id]
    old, new = 0, 1
    while old != new:
        pool = []
        for track_id in lineage:
            pool += get_relatives(df, track_id)
        lineage = np.unique(lineage + pool).tolist()
        old = new
        new = len(lineage)
    return lineage


def get_sibling_pairs(df):
    """
    Gets all the sibling pairs

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe

    Returns
    -------
    sister_list: List
        List containing of np.arrays containing sibling pair track_ids
    """
    sister_list = []
    for p_id, dft in df.reset_index().groupby("parent_id"):
        if p_id != -1:
            t_ids = dft.track_id.unique()
            if len(t_ids) == 2:
                sister_list.append(t_ids)
    return sister_list
