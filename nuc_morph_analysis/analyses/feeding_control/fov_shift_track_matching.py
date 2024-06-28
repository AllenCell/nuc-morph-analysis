import numpy as np

# appropriate shift was found by visual manual alignment of centroids
FRAMES_TO_SHIFT = {
    "feeding_control_baseline": [(259, -35, -40)],
    "feeding_control_starved": [(259, -35, -40)],
    "feeding_control_refeed": [(259, -35, -40)],
}


def match_tracks(df, dataset, frames_to_shift, distance_threshold=50):
    """
    Find the tracks that match after applying the shift. Tracks can only be matched to the closest single other track.
    Tracks must be within a distance threshold to prevent from matching with tracks that appear on the FOV edges.

    Paramaters
    ----------
    df: dataframe
        Dataframe to match tracks
    dataset: string
        Name of the dataset
    frames_to_shift: tuple
        Tuple of frame, x_shift, y_shift. Can come from the present dictionary FRAMES_TO_SHIFT
    distance_threshold:
        Maximum distance between two tracks to be considered a match.

    Returns
    -------
    df_matched_tracks: dataframe
        Dataframe of matched tracks with the columns:
        track_id_1, centroid_x_1, centroid_y_1, track_id_2, centroid_x_2, centroid_y_2, distance
    """
    print(f"Finding matches for {dataset}")
    df_sub = df[["track_id", "centroid_x", "centroid_y", "index_sequence"]]
    frame, x_shift, y_shift = frames_to_shift
    df_timepoint1 = df_sub[
        (df_sub.index_sequence == frame)
        & df_sub.track_id.isin(df_sub[df_sub.index_sequence == frame].track_id)
    ]
    df_timepoint2 = df_sub[
        (df_sub.index_sequence == frame + 1)
        & df_sub.track_id.isin(df_sub[df_sub.index_sequence == frame + 1].track_id)
    ]

    # Shift the coordinates of df_timepoint2
    df_timepoint2_shifted = df_timepoint2.copy()
    df_timepoint2_shifted.loc[:, "centroid_x"] += x_shift
    df_timepoint2_shifted.loc[:, "centroid_y"] += y_shift

    # Add suffix to each timepoint dataframe
    df_timepoint1 = df_timepoint1.add_suffix("_1")
    df_timepoint2_shifted = df_timepoint2_shifted.add_suffix("_2")

    # Create a temporary key for merging
    df_timepoint1["key"] = 0
    df_timepoint2_shifted["key"] = 0
    # Create a Cartesian product of the two dataframes
    df_cartesian = df_timepoint1.merge(df_timepoint2_shifted, how="outer", on="key")
    df_cartesian.drop("key", axis=1, inplace=True)
    df_cartesian["distance"] = np.sqrt(
        (df_cartesian["centroid_x_1"] - df_cartesian["centroid_x_2"]) ** 2
        + (df_cartesian["centroid_y_1"] - df_cartesian["centroid_y_2"]) ** 2
    )

    df_distances = df_cartesian.loc[df_cartesian.groupby("track_id_1")["distance"].idxmin()]
    df_distances = df_distances.loc[df_distances.groupby("track_id_2")["distance"].idxmin()]
    df_matched_tracks = df_distances[df_distances["distance"] < distance_threshold]

    return df_matched_tracks


def update_track_ids(df, df_matched_tracks, frames_to_shift):
    """
    Update the track ids in the dataframe. Check to make sure the track doesnt continue before or after
    in order to make the match.

    Parameters
    ----------
    df: dataframe
        Dataframe to update the track ids
    df_matched_tracks: dataframe
        Dataframe of matched tracks with the columns:
        track_id_1, centroid_x_1, centroid_y_1, track_id_2, centroid_x_2, centroid_y_2, distance
    frames_to_shift: tuple
        Tuple of frame, x_shift, y_shift. Can come from the present dictionary FRAMES_TO_SHIFT

    Returns
    -------
    df_updated: dataframe
        Dataframe with updated track ids
    """
    frame, _, _ = frames_to_shift
    df_new_matches = df_matched_tracks.query("track_id_1 != track_id_2")
    df_updated = df.copy()
    df_updated["track_matched"] = False
    df_updated["track_match_issue"] = False
    for track_id_1, track_id_2 in zip(df_new_matches.track_id_1, df_new_matches.track_id_2):
        dft1 = df_updated[df_updated.track_id == track_id_1]
        dft2 = df_updated[df_updated.track_id == track_id_2]
        if dft1.index_sequence.max() <= frame and dft2.index_sequence.min() >= frame + 1:
            df_updated.loc[df_updated.track_id == track_id_2, "track_id"] = track_id_1
            df_updated.loc[df_updated.track_id == track_id_1, "track_matched"] = True
        else:
            df_updated.loc[df_updated.track_id == track_id_1, "track_match_issue"] = True
            df_updated.loc[df_updated.track_id == track_id_2, "track_match_issue"] = True
    return df_updated


def match_and_update_dataframe(df, dataset, FRAMES_TO_SHIFT):
    """
    Update the tracks for shifted frames.

    Paramaters
    ----------
    df: dataframe
        Dataframe to update the track ids
    dataset: string
        Name of dataset
    frames_to_shift: tuple
        Tuple of frame, x_shift, y_shift. Can come from the present dictionary FRAMES_TO_SHIFT

    Returns
    -------
    df: dataframe
        Dataframe with updated track_ids after shifting and track matching
    """
    for frames_to_shift in FRAMES_TO_SHIFT[dataset]:
        df_matched_tracks = match_tracks(df, dataset, frames_to_shift)
        df_updated = update_track_ids(df, df_matched_tracks, frames_to_shift)
        df = df_updated
    return df
