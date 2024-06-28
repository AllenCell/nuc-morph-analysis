import numpy as np
import pandas as pd
from nuc_morph_analysis.lib.preprocessing.track_matching.merge_formation_breakdown import (
    merge_formation_breakdown,
)


def test_merge_formation_breakdown_simple():
    # ARRANGE
    # Create a DataFrame with two tracks
    df = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2],
            "index_sequence": [1, 2, 1, 2],
            "predicted_formation": [100, -1, 200, -1],
            "predicted_breakdown": [-1, 110, -1, 210],
        }
    )

    # ACT
    result = merge_formation_breakdown(df)

    # ASSERT
    expected = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2],
            "index_sequence": [1, 2, 1, 2],
            "predicted_formation": [100, 100, 200, 200],
            "predicted_breakdown": [110, 110, 210, 210],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_merged_tracks_formation_in_middle():
    # ARRANGE
    df = pd.DataFrame(
        {
            "track_id": [1, 1, 1, 1],
            "index_sequence": [1, 2, 3, 4],
            "predicted_formation": [-1, 100, -1, -1],
            "predicted_breakdown": [-1, -1, 210, -1],
        }
    )

    # ACT
    result = merge_formation_breakdown(df)

    # ASSERT
    expected = pd.DataFrame(
        {
            "track_id": [1, 1, 1, 1],
            "index_sequence": [1, 2, 3, 4],
            "predicted_formation": [100, 100, 100, 100],
            "predicted_breakdown": [210, 210, 210, 210],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_merged_tracks_multiple_values():
    # ARRANGE
    df = pd.DataFrame(
        {
            "track_id": [1, 1, 1, 1, 1],
            "index_sequence": [1, 2, 3, 4, 5],
            # First formation value (100) is used, even if it's (somehow) larger
            "predicted_formation": [-1, 100, 99, -1, -1],
            # Last breakdown value (210) is used, even if it's (somehow) smaller
            "predicted_breakdown": [-1, -1, 211, 210, -1],
        }
    )

    # ACT
    result = merge_formation_breakdown(df)

    # ASSERT
    expected = pd.DataFrame(
        {
            "track_id": [1, 1, 1, 1, 1],
            "index_sequence": [1, 2, 3, 4, 5],
            "predicted_formation": [100, 100, 100, 100, 100],
            "predicted_breakdown": [210, 210, 210, 210, 210],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_merged_tracks_no_values():
    # ARRANGE
    df = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2],
            "index_sequence": [1, 2, 3, 4],
            "predicted_formation": [-1, -1, np.nan, np.nan],
            "predicted_breakdown": [-1, 30, 40, np.nan],
        }
    )

    # ACT
    result = merge_formation_breakdown(df)

    # ASSERT
    expected = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2],
            "index_sequence": [1, 2, 3, 4],
            "predicted_formation": [-1.0, -1.0, np.nan, np.nan],
            "predicted_breakdown": [30.0, 30.0, 40.0, 40.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)
