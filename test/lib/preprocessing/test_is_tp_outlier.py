# %%
import pandas as pd
from nuc_morph_analysis.lib.preprocessing import is_tp_outlier


def test_outlier_detection():
    # ARRANGE
    track1 = pd.DataFrame(
        {
            "track_id": [1 for _ in range(28)],
            "index_sequence": list(range(1, 29)),
            # fmt: off
        'volume': [100, 100, 100, 100, 100, 100, 100, 100, 100, 300, 100, 100, 100, 100,
                   100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            # fmt: on
        }
    )

    # ACT
    result = is_tp_outlier.outlier_detection(track1)

    # ASSERT
    expected = pd.Series(track1["volume"] == 300, name="is_tp_outlier")
    pd.testing.assert_series_equal(result.is_tp_outlier, expected)


def test_outlier_detection_three_tracks():
    """The tracks should be evaluated independently for outliers."""
    # ARRANGE
    df = pd.DataFrame(
        {
            "track_id": [1 for _ in range(28)],
            "index_sequence": list(range(1, 29)),
            "volume": [100] * 9 + [300] + [100] * 18,
        }
    )

    # ACT
    results = is_tp_outlier.outlier_detection(df)

    # ASSERT
    for track_id, result in results.groupby("track_id"):
        if track_id == 1:
            pd.testing.assert_series_equal(
                result.is_tp_outlier,
                pd.Series([False] * 9 + [True] + [False] * 18, name="is_tp_outlier"),
            )
        if track_id == 2:
            pd.testing.assert_series_equal(
                result.is_tp_outlier.reset_index(drop=True),
                pd.Series([False] * 9 + [True] + [False] * 18, name="is_tp_outlier"),
            )
        if track_id == 3:
            pd.testing.assert_series_equal(
                result.is_tp_outlier.reset_index(drop=True),
                pd.Series([False] * 9 + [True] + [False] * 18, name="is_tp_outlier"),
            )


# %%
test_outlier_detection()
# %%
test_outlier_detection_three_tracks()
# %%
