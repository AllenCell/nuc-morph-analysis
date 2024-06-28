from nuc_morph_analysis.lib.preprocessing import colony_depth


def test_colony_depth_1d():
    # This test simulates a colony of cells laid out in the following pattern
    #   1 2 3 4 5 6 7 8
    #   ^           ^
    # Cells 1 and 7 are defined as "depth 1"
    neighbors = {
        1: {2},
        2: {1, 3},
        3: {2, 4},
        4: {3, 5},
        5: {4, 6},
        6: {5, 7},
        7: {6, 8},
        8: {7},
    }
    depths = colony_depth.calculate_depth(neighbors, [1, 7])
    assert depths == {1: 1, 2: 2, 3: 3, 4: 4, 5: 3, 6: 2, 7: 1, 8: 2}


def test_colony_depth_2d():
    # This test simulates a colony of cells laid out in the following pattern
    #       1
    #    2  3  4
    # 5  6  7  8  9
    #   10 11 12
    #      13
    neighbors = {
        1: {2, 3, 4},
        2: {3, 6},
        3: {1, 2, 4, 7},
        4: {3, 8},
        5: {2, 6, 10},
        6: {2, 5, 7, 10},
        7: {3, 6, 8, 11},
        8: {4, 7, 9, 12},
        9: {4, 8, 12},
        10: {6, 11},
        11: {7, 10, 12, 13},
        12: {8, 13},
        13: {10, 11, 12},
    }
    depths = colony_depth.calculate_depth(neighbors, [1, 2, 4, 5, 9, 10, 12, 13])
    assert depths == {
        1: 1,
        # Row 2
        2: 1,
        3: 2,
        4: 1,
        # Row 3
        5: 1,
        6: 2,
        7: 3,
        8: 2,
        9: 1,
        # Row 4
        10: 1,
        11: 2,
        12: 1,
        # Row 5
        13: 1,
    }
