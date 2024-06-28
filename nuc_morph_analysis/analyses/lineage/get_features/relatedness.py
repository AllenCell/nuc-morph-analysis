import numpy as np
import bigtree
import os


def get_track_ids_from_path(path_name):
    """
    Get all the track_ids that are on located on the same branch
    that come before the track of interest.

    Paramaters
    -------
    path_name: Str
        n.pathname from a bigtree node

    Returns
    -------
    new_path: List
        list of track_ids up one a branch up to that track
    """
    new_path = path_name.split("/")[1:]
    new_path = [eval(v) for v in new_path]
    return new_path


def get_track_ids_for_family_tree(family_tree):
    """
    Get all the track_ids in a lineage tree

    Paramaters
    -------
    family_tree: bigtree graph
        node list of one tree

    Returns
    -------
    track_ids: List
        list containing all the track_ids in a tree
    """
    track_ids = []
    nodes_in_tree = list(bigtree.findall(family_tree, condition=lambda node: node.name))
    for n in nodes_in_tree:
        track_id = n.name
        track_ids.append(track_id)
    return track_ids


def get_paths_for_family_tree(family_tree):
    """
    Get all the path_names for each track in the tree

    Paramaters
    -------
    family_tree: bigtree graph
        node list of one tree

    Returns
    -------
    paths: List
        list containing all the paths in a tree
    """
    paths = []
    nodes_in_tree = list(bigtree.findall(family_tree, condition=lambda node: node.name))
    for n in nodes_in_tree:
        new_path = get_track_ids_from_path(path_name=n.path_name)
        paths.append(new_path)
    return paths


def calculate_cousiness(tid1, tid2, paths):
    """
    Metric for how far apart in width related nuclei are located

    Paramaters
    -------
    tid1: Int
        track_id 1
    tid2: Int
        track_id 2
    paths: List
        list of the paths for each nuclei in the lineage tree

    Returns
    -------
    np.min(cs): Int
        Coussiness metric, width
    abs_depth: Int
        The depth in the tree for which the coussiness metric takes place
    """
    cs = []
    for i, p1 in enumerate(paths):
        for j, p2 in enumerate(paths):
            if (j > i) and (len(p1) == len(p2)):
                if ((tid1 == p1[-1] or tid2 == p1[-1])) and ((tid1 == p2[-1]) or (tid2 == p2[-1])):
                    ftlist = [v1 == v2 for (v1, v2) in zip(p1, p2)][::-1]
                    c = np.where(ftlist)[0]
                    cs.append(np.min(c))
                    abs_depth = len(p1)
    return np.min(cs), abs_depth


def check_same_branch(tid1, tid2, path1, path2):
    """
    Depth only tells you the difference in generation on the tree. Now we need to check if those related cells
    are on the same branch. True means they are on the same branch and therfore can be used for
    mother daughter analysis.

    Paramaters
    -------
    tid1: Int
        track_id 1
    tid2: Int
        track_id 2
    path1: Str
        path containing the track_id of that track of interest and
        the ones that come brefore it (ie branch ending with that track)
    path2:
        path containing the track_id of that track of interest and
        the ones that come brefore it (ie branch ending with that track)

    Returns
    -------
    Boolean: True or False
        True means they are on the same branch
    """
    path1 = os.path.normpath(path1)
    path2 = os.path.normpath(path2)
    list1 = path1.split(os.sep)
    list1.pop(0)
    list2 = path2.split(os.sep)
    list2.pop(0)

    branch1 = [eval(i) for i in list1]
    branch2 = [eval(i) for i in list2]
    if tid1 in branch2 or tid2 in branch1:
        return True
    else:
        return False
