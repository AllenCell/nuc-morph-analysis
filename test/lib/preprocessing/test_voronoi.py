import numpy as np
from nuc_morph_analysis.lib.preprocessing.add_colony_metrics import add_colony_metrics
import pandas as pd


def test_voronoi_real_densities():
    """
    Load 2 frames of medium colony,
    execute voronoi calculation to get neighbors, distances and densities
    compare to gt density (from colony_metrics dataset in FMS)
    """

    # These are all neighbors and relevant features for 2 cells in medium colony
    # Cell 1 - 41f330ab2e8a5ad827ca53e66b632e3a3fff209c260614819960d9cc (a)
    # Cell 2 - 7ea0e4bd3587da25dde34c2479e0c7ac10787c4db34259b441664a8f (b)

    index_seqs = [355, 356, 355, 355, 355, 355, 355, 355, 356, 356, 356, 356, 356, 356]
    label_img = [
        95.0,
        178.0,
        190.0,
        201.0,
        189.0,
        194.0,
        98.0,
        188.0,
        170.0,
        188.0,
        171.0,
        179.0,
        182.0,
        85.0,
    ]
    ids = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n"]
    vols = [
        1069444.0,
        1090362.0,
        348357.0,
        329564.0,
        422921.0,
        591441.0,
        398974.0,
        438732.0,
        354709.0,
        324528.0,
        428091.0,
        611958.0,
        410333.0,
        489027.0,
    ]
    centroid_y = [
        1227,
        1226,
        1059,
        1410,
        1103,
        1265,
        1288,
        1050,
        1055,
        1395,
        1096,
        1257,
        1278,
        1040,
    ]
    centroid_x = [
        3197,
        3185,
        3192,
        3169,
        2948,
        2904,
        3483,
        3433,
        3195,
        3170,
        2945,
        2904,
        3491,
        3432,
    ]
    density_gt = [
        0.01584935,
        0.01582909,
        0.01949619,
        0.01791645,
        0.01520982,
        0.01631341,
        0.01520432,
        0.01723284,
        0.01947099,
        0.01637999,
        0.01529952,
        0.01653538,
        0.01463112,
        0.01745064,
    ]

    # Create dataframe with these features
    df = pd.DataFrame()
    df["index_sequence"] = index_seqs
    df["label_img"] = label_img
    df["CellId"] = ids
    df["volume"] = vols
    df["centroid_y"] = centroid_y
    df["centroid_x"] = centroid_x
    df["density_gt"] = density_gt

    # run colony metrics calculation
    df_colony_metrics = add_colony_metrics(df)
    # The new density value should be approximately the same as the old density value divided by 4,
    # squared. This is because the old density was computed as 1 / mean(neighbor distances), where
    # the neighbor distances were downsampled by a factor of 4. The new density does not downsample
    # and uses 1 / mean(neighbor distances)^2 so that the units are closer to what is expected
    # from a "density" metric
    df_colony_metrics["old_density"] = df_colony_metrics["density"].apply(lambda x: 4 * np.sqrt(x))
    assert np.allclose(df_colony_metrics["old_density"], df_colony_metrics.density_gt, rtol=0.2)


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
