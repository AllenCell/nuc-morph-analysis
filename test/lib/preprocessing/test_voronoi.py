import numpy as np
from nuc_morph_analysis.lib.preprocessing.add_colony_metrics import add_colony_metrics
import pandas as pd


def test_voronoi_synthetic_distance_density():
    """
    This test uses a set of cells laid out in the following pattern.
     a b c d
     e f g h
     i j k
     l m
    """
    # ARRANGE
    # fmt: off
    ids = ["a", "b", "c", "d",
           "e", "f", "g", "h",
           "i", "j", "k",
           "l", "m"]
    centroid_x = [ 10, 20, 30, 40,
                   10, 20, 30, 40,
                   10, 20, 30,
                   10, 20 ]
    centroid_y = [ 40, 40, 40, 40,
                   30, 30, 30, 30,
                   20, 20, 20,
                   10, 10 ]
    label_img = [ 1, 2, 3, 4,
                  5, 6, 7, 8,
                  9, 10, 11,
                  12, 13 ]
    # fmt: on

    vols = np.full(
        len(ids), 100000
    )  # Must be large enough to pass the filter that excludes small nuclei
    index_seqs = np.full(len(ids), 1)

    df = pd.DataFrame()
    df["index_sequence"] = index_seqs
    df["label_img"] = label_img
    df["CellId"] = ids
    df["volume"] = vols
    df["centroid_y"] = centroid_y
    df["centroid_x"] = centroid_x

    # ACT
    df_colony_metrics = add_colony_metrics(df)

    # ASSERT
    expected_neighbor_distances = np.full(
        len(ids), 10.0
    )  # Most cells only have orthogonal neighbors
    expected_neighbor_distances[7] = np.mean([10, 10, 10.0 * np.sqrt(2)])  # Cell h
    expected_neighbor_distances[10] = np.mean(
        [10, 10, 10.0 * np.sqrt(2), 10.0 * np.sqrt(2)]
    )  # Cell k
    expected_neighbor_distances[12] = np.mean([10, 10, 10.0 * np.sqrt(2)])  # Cell m
    assert np.allclose(df_colony_metrics.neigh_distance, expected_neighbor_distances)

    expected_densities = 1 / expected_neighbor_distances**2
    assert np.allclose(df_colony_metrics.density, expected_densities)


def test_voronoi_neighbors():
    """
    This test uses a set of cells laid out in the following pattern.
     a b c d
     e f g h
     i j k
     l m
    """
    # ARRANGE
    # fmt: off
    ids = ["a", "b", "c", "d",
           "e", "f", "g", "h",
           "i", "j", "k",
           "l", "m"]
    centroid_x = [ 10, 20, 30, 40,
                   10, 20, 30, 40,
                   10, 20, 30,
                   10, 20 ]
    centroid_y = [ 40, 40, 40, 40,
                   30, 30, 30, 30,
                   20, 20, 20,
                   10, 10 ]
    label_img = [ 1, 2, 3, 4,
                  5, 6, 7, 8,
                  9, 10, 11,
                  12, 13 ]
    # fmt: on

    vols = np.full(
        len(ids), 100000
    )  # Must be large enough to pass the filter that excludes small nuclei
    index_seqs = np.full(len(ids), 1)

    df = pd.DataFrame()
    df["index_sequence"] = index_seqs
    df["label_img"] = label_img
    df["CellId"] = ids
    df["volume"] = vols
    df["centroid_y"] = centroid_y
    df["centroid_x"] = centroid_x

    # ACT
    df_colony_metrics = add_colony_metrics(df)

    # ASSERT
    expected_neighbors = [
        {"b", "e"},  # 'a'
        {"a", "c", "f"},  # 'b'
        {"b", "d", "g"},  # 'c'
        {"c", "h"},  # 'd'
        {"a", "f", "i"},  # 'e'
        {"b", "e", "g", "j"},  # 'f'
        {"c", "f", "h", "k"},  # 'g'
        {"d", "g", "k"},  # 'h'
        {"e", "j", "l"},  # 'i'
        {"f", "i", "k", "m"},  # 'j'
        {"g", "h", "j", "m"},  # 'k'
        {"i", "m"},  # 'l'
        {"j", "k", "l"},  # 'm'
    ]
    actual_neighbors = [set(eval(neighbors)) for neighbors in df_colony_metrics.neighbors]
    assert actual_neighbors == expected_neighbors

def test_voronoi_neighbors_with_stacked_cells():
    """
    This test uses a set of cells laid out in the following pattern.
    e and x have identical (x,y) centroids but different z (unused)
     a b c
     d ex f
     g h i

     This test pins the current behavior; if a new way of handling
     cells with same centroids (different z) is developed, the test
     will need updating
    """
    # ARRANGE
    # fmt: off
    ids =        ["a", "b", "c", "d", "e", "f", "g", "h", "i", "x"]
    centroid_x = [ 10,  20,  30,  10,  20,  30,  10,  20,  30,  20]
    centroid_y = [ 10,  10,  10,  20,  20,  20,  30,  30,  30,  20]
    label_img =  [  1,   2,   3,   4,   5,   6,   7,   8,   9,  10]
    # fmt: on

    vols = np.full(
        len(ids), 100000
    )  # Must be large enough to pass the filter that excludes small nuclei
    index_seqs = np.full(len(ids), 1)

    df = pd.DataFrame()
    df["index_sequence"] = index_seqs
    df["label_img"] = label_img
    df["CellId"] = ids
    df["volume"] = vols
    df["centroid_y"] = centroid_y
    df["centroid_x"] = centroid_x

    # ACT
    df_colony_metrics = add_colony_metrics(df)

    # ASSERT
    # very fragile; order matters
    # the key part here is cell 'x' does NOT appear due to overlap with 'e'
    # 'x' is not a neighbor, and has no neighbors
    expected_neighbors = [
        ["b", "d"],        # 'a'
        ["a", "c", "e"],  # 'b'
        ["b", "f"],       # 'c'
        ["a", "e", "g"],  # 'd'
        ["h", "b", "d", "f"],  # 'e'
        ["i", "c", "e"],  # 'f'
        ["h", "d"],       # 'g'
        ["i", "e", "g"],  # 'h'
        ["h", "f"]  # 'i'
    ]
    actual_neighbors = []

    for neighbor in df_colony_metrics.neighbors:
        if isinstance(neighbor, str):
            temp = eval(neighbor)
            actual_neighbors.append(temp)
    assert len(actual_neighbors) == len(expected_neighbors)
    assert actual_neighbors == expected_neighbors

def test_voronoi_depths():
    """
    The purpose of this test is to check that the set of depth 1 nuclei includes not only nuclei on
    the boundary of the convex hull of the colony, but also nuclei which are "close enough" to the
    boundary. (A nucleus is close enough if in the voronoi diagram the region around the nucleus
    crosses the convex hull.)
    """
    # ARRANGE
    ids = list(range(0, 25))
    df = pd.DataFrame(
        {
            "index_sequence": np.full(len(ids), 1),
            "label_img": ids,
            "CellId": ids,
            "volume": np.full(
                len(ids), 100000
            ),  # Must be large enough to pass the filter that excludes small nuclei
            # fmt: off
        "centroid_x": [ 863.,  927.,  937., 1103., 1148., 1212., 1811., 2556., 2649.,
                       2858., 3196., 3292., 3435., 3475., 3530., 3542., 3631.,
                       # depth 2
                       2127., 2321., 2650., 2822., 2953., 3194.,
                       # depth 3
                       2410., 2453.],
        "centroid_y": [2541., 1531., 2345.,  613., 1178.,   43.,   46.,  486., 3107.,
                        138., 2584., 2307., 1161.,  767., 2208., 2030.,  995.,
                       # depth 2
                       1475., 1945., 1392., 2112., 1866., 1212.,
                       # depth 3
                       1553., 1682.]
            # fmt: on
        }
    )

    # To visualize this test data, uncomment the following block
    # from scipy.spatial import Voronoi, voronoi_plot_2d
    # import matplotlib.pyplot as plt
    # centroids_list = df[["centroid_x", "centroid_y"]].to_numpy()
    # voronoi = Voronoi(centroids_list)
    # voronoi_plot_2d(voronoi, show_vertices=False)
    # plt.savefig("test_voronoi_depths.png")
    # plt.show()

    # ACT
    df_colony_metrics = add_colony_metrics(df)

    # ASSERT
    expected_depths = np.full(len(ids), 1)
    expected_depths[17:23] = 2
    expected_depths[23:25] = 3
    assert np.allclose(expected_depths, df_colony_metrics.colony_depth.values)
