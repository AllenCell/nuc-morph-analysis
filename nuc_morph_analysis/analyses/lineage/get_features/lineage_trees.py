import numpy as np
import bigtree
from nuc_morph_analysis.analyses.lineage.get_features.family_functions import (
    get_ids_in_same_lineage_as,
)


def get_roots(dfm):
    """
    This function finds all the roots to the family trees from the dataframe.
    A root cell is a track_id that has no parent but has children.

    Parameters
    ----------
    dfm: Dataframe
        Dataset manifest, after preproccesing

    Returns
    -------
    root_list: list
        List containing the track_ids for all the root cells.
    """

    root_list = []
    no_parent = dfm[dfm["parent_id"] == -1.0]
    noparent_list = no_parent["track_id"].unique()
    parent_ids = dfm["parent_id"].unique()
    for i in noparent_list:
        if i in parent_ids:
            root_list.append(i)
    return root_list


def get_descendants(dfm):
    """
    This function finds all the decendants. Ordering them by index sequence
    will allow us to populate the trees knowing that that parent of that cell
    will come earlier in the list than its decendant.

    Parameters
    ----------
    dfm: Dataframe
        Dataset manifest, after preproccesing

    Returns
    -------
    root_list: list
        List containing the track_ids for all the root cells.
    """

    descendants_list = []
    root_list = get_roots(dfm)
    for i in root_list:
        lineage = get_ids_in_same_lineage_as(dfm, track_id=i)
        for i in lineage:
            if i in root_list:
                continue
            else:
                in_seq = dfm.loc[dfm["track_id"] == i, "index_sequence"].min()
                descendants_list.append([in_seq, i])
    descendants_list.sort()
    return descendants_list


def get_trees(dfm):
    """
    This function gets all the roots and decendants for all the lineage trees in the dataframe
    and populates them into nodes. These are now in a format that is compatible with functions from
    bigtree. We could also choose to write our own functions to work with them in the future.

    Parameters
    ----------
    dfm: Dataframe
        Dataset manifest, after preproccesing

    Returns
    -------
    node_list: list
        List containing all the nodes for all the lineage trees.
    """

    root_list = get_roots(dfm)
    descendants_list = get_descendants(dfm)

    node_list = []
    for i in root_list:
        node_list.append(bigtree.Node(i))
    for j in descendants_list:
        # find parent of this id
        parentId = dfm.loc[dfm["track_id"] == j[1], "parent_id"].unique()
        parentNode = None
        for n in node_list:
            parentNode = bigtree.find_name(n, parentId)
            if parentNode != None:
                bigtree.Node(j[1], parent=parentNode)
                break
        if parentNode == None:
            print("MISSING:", "PID", parentId, "trackid", j)
    return node_list
