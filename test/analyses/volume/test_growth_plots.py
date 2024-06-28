import numpy as np
import pandas as pd
from nuc_morph_analysis.lib.preprocessing.add_times import get_nearest_frame


def test_get_nearest_frame():
    # Create a sample dataframe
    df_track = pd.DataFrame({"index_sequence": [1, 2, 3, 4, 5]})

    # Test when the calculated transition is exactly at a frame
    assert get_nearest_frame(df_track, 3) == 3

    # Test when the calculated transition is between two frames
    assert get_nearest_frame(df_track, 2.2) == 2
    assert get_nearest_frame(df_track, 2.5) == 2
    assert get_nearest_frame(df_track, 2.7) == 3

    # Test when the calculated transition is more than 2 frames away
    assert np.isnan(get_nearest_frame(df_track, 7.1))
