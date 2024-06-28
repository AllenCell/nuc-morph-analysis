import numpy as np
from nuc_morph_analysis.lib.preprocessing.single_track_contact.vis_code import (
    image_helper,
)


def test_stack_uneven_middle_doesnt_crop():
    # ARRANGE
    arr1 = np.ones((2, 2))
    arr2 = np.ones((4, 1))

    # ACT
    result = image_helper.stack_uneven_middle([arr1, arr2])

    # ASSERT
    np.testing.assert_array_equal(
        result[0],
        [
            [0, 0],
            [1, 1],
            [1, 1],
            [0, 0],
        ],
    )
    np.testing.assert_array_equal(
        result[1],
        [
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
        ],
    )


def test_stack_uneven_middle_aligns_to_center():
    # ARRANGE
    arr1 = np.ones((5, 5))
    arr2 = np.ones((2, 2))

    # ACT
    result = image_helper.stack_uneven_middle([arr1, arr2])

    # ASSERT
    np.testing.assert_array_equal(result[0], arr1)
    # Since 2 is even and 5 is odd, the padding is not exactly centered
    np.testing.assert_array_equal(
        result[1],
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    )
