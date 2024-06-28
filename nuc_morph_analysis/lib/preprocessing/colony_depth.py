def calculate_depth(neighbors, depth1_labels):
    """
    Given the neighborhood (graph) structure of a collection of cells and a set of cells at depth 1,
    compute the "depth" of each cell, defined as follows for n >= 1: depth n+1 cells are those
    adjacent to depth n cells and no cells of lower depth. Alternative definition: the depth of a
    cell is the graph distance to the nearest depth 1 cell, plus 1.

    Parameters
    ----------
    neighbors: Dict[T, Set[T]]
      A dictionary mapping labels (of any type T) to sets of labels

    depth1_labels: Iterable[T]
      A collection of labels at depth 1

    Returns
    -------
    depth_map: Dict[T, int]
      A dictionary mapping labels to their depth. Only includes labels reachable from depth1_labels
    """
    # This is just an implementation of breadth first search

    # Initialize with the layer at depth 1
    layer = set(depth1_labels)
    depth = 1
    depth_map = {}

    while len(layer) > 0:
        # Save the depth values for this layer
        for lbl in layer:
            depth_map[lbl] = depth

        # Cells at depth n+1 are the neighbors of cells at depth n that don't have a depth yet
        previous = layer
        depth += 1
        candidates = set.union(*(neighbors[lbl] for lbl in previous))
        layer = {lbl for lbl in candidates if lbl not in depth_map}

    return depth_map
