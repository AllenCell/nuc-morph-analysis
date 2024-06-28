import pandas as pd
import numpy as np
from nuc_morph_analysis.utilities.warn_slow import warn_slow


@warn_slow("2m")  # Benchmarks for this function are ~1min
def merge_formation_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    After merging tracks, newly merged tracks will have conflicting formation and breakdown values.
    Example:
        track_id | index_sequence | predicted_formation | predicted_breakdown
        ---------|----------------|---------------------|--------------------
        59       | 514            | 152                 | -1
        59       | 519            | -1                  | 561

    This function fixes the conflicting values by taking the first formation value and the last
    breakdown value for each track.

    Parameters
    ----------
    df: pandas.DataFrame
        The DataFrame to fix. Must have columns "track_id", "predicted_formation", and
        "predicted_breakdown".

    Returns
    -------
    pandas.DataFrame
        Copy of the input dataframe identical except for the predicted_formation and
        predicted_breakdown columns.
    """

    def formation_or_breakdown(col, first_or_last):
        index = 0 if first_or_last == "first" else -1

        def get_first_or_last_value_from_track(df_track):
            good_values = df_track[(df_track[col] != -1) & ~df_track[col].isna()]
            if len(good_values) > 0:
                return good_values.sort_values("index_sequence").iloc[index][col]
            else:
                return df_track.iloc[index][col]

        return (
            df[["index_sequence", "track_id", col]]
            .groupby(by="track_id")
            .apply(get_first_or_last_value_from_track)
        )

    formation = formation_or_breakdown("predicted_formation", "first")
    breakdown = formation_or_breakdown("predicted_breakdown", "last")

    # Use the first formation value and the last breakdown value for each track
    new_cols = pd.DataFrame({"predicted_formation": formation, "predicted_breakdown": breakdown})

    result = df.drop(columns=["predicted_formation", "predicted_breakdown"])
    # This merge is the slowest step of this function
    result = result.merge(new_cols, left_on="track_id", how="left", right_index=True)
    return result


@warn_slow(12)  # Benchmarks for this function are ~1-4s
def validate_formation_breakdown(df: pd.DataFrame, threshold=5):
    """
    Check the predicted_formation and predicted_breakdown columns for consistency.

    Parameters
    ----------
    df: pandas.DataFrame
        The DataFrame to validate. Must have columns "track_id", "predicted_formation", and
        "predicted_breakdown".
    threshold: int
        Threshold number of tracks with formation after breakdown to just drop rather than
        throwing an error and halting manifest writing
    """
    # Make sure the columns actually have values
    assert df.predicted_formation.notna().any(), "predicted_formation must not be all-null"
    assert df.predicted_breakdown.notna().any(), "predicted_breakdown must not be all-null"

    # Main assertion: no track has more than one formation or breakdown value
    grouped = df.groupby("track_id")
    formation_counts = grouped[["predicted_formation"]].nunique().predicted_formation
    assert len(formation_counts[formation_counts > 1]) == 0, "predicted_formation must be unique"
    breakdown_counts = grouped[["predicted_breakdown"]].nunique().predicted_breakdown
    assert len(breakdown_counts[breakdown_counts > 1]) == 0, "predicted_breakdown must be unique"

    # If a track has both formation and breakdown, formation should come first
    with_both = df[(df.predicted_formation >= 0) & (df.predicted_breakdown >= 0)]
    with_formation_after_breakdown = []
    for track, df_track in with_both.groupby("track_id"):
        if df_track["predicted_formation"].values[0] >= df_track["predicted_breakdown"].values[0]:
            with_formation_after_breakdown.append(track)
    if len(with_formation_after_breakdown) < threshold:
        with_both.loc[
            with_both["track_id"].isin(with_formation_after_breakdown), "predicted_formation"
        ] = np.nan
        with_both.loc[
            with_both["track_id"].isin(with_formation_after_breakdown), "predicted_breakdown"
        ] = np.nan
        print(f"{len(with_formation_after_breakdown)} tracks with formation after breakdown found.")
        print("Their formation and breakdown frames were reset to NaNs.")
    assert not (
        with_both.predicted_formation >= with_both.predicted_breakdown
    ).any(), (
        f"more than {threshold} tracks found with predicted_formation after predicted_breakdown"
    )
