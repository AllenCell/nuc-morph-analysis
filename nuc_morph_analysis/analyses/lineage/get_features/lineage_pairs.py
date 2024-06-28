import pandas as pd
import bigtree
import numpy as np
import random
from nuc_morph_analysis.analyses.lineage.get_features import (
    between_pairs,
    convert_units,
    family_functions,
    relatedness,
    dataset_stats,
)
from nuc_morph_analysis.lib.preprocessing import load_data


def get_related_pairs_dataset(track_level_feature_df, family_trees):
    """
    Get related pairs dataset.

    Parameters
    ----------
    track_level_feature_df : pd.DataFrame
        The track level feature dataframe.
    family_trees : list
        The list of family trees.

    Returns
    -------
    related_pairs_df_with_features : pd.DataFrame
    """
    full_tracks = list(track_level_feature_df.track_id)
    related_pairs = []

    for i, family_tree in enumerate(family_trees):
        track_ids = relatedness.get_track_ids_for_family_tree(family_tree)
        for j, tid1 in enumerate(track_ids):
            for k, tid2 in enumerate(track_ids):
                if j > k:
                    if tid1 in full_tracks and tid2 in full_tracks:
                        n1 = bigtree.find_name(family_tree, tid1)
                        n2 = bigtree.find_name(family_tree, tid2)
                        if n1.depth == n2.depth:
                            paths = relatedness.get_paths_for_family_tree(family_tree)
                            cs = relatedness.calculate_cousiness(tid1, tid2, paths)
                            related_pairs.append(
                                {
                                    "track_id1": tid1,
                                    "track_id2": tid2,
                                    "lineage_id": i,
                                    "absolute_depth": n1.depth,
                                    "cousiness": cs[0],
                                    "delta_depth": 0,
                                    "same_branch": np.nan,
                                }
                            )
                        else:
                            check = relatedness.check_same_branch(
                                tid1, tid2, n1.path_name, n2.path_name
                            )
                            related_pairs.append(
                                {
                                    "track_id1": tid1,
                                    "track_id2": tid2,
                                    "lineage_id": i,
                                    "absolute_depth": np.nan,
                                    "cousiness": np.nan,
                                    "delta_depth": abs(n1.depth - n2.depth),
                                    "same_branch": check,
                                }
                            )

    related_pairs_df = pd.DataFrame(
        related_pairs,
        columns=[
            "track_id1",
            "track_id2",
            "lineage_id",
            "absolute_depth",
            "cousiness",
            "delta_depth",
            "same_branch",
        ],
    )

    related_pairs_df_with_features = get_features_for_pairs(
        related_pairs_df, track_level_feature_df
    )
    dataset_stats.print_number_of_sister_md_pairs(related_pairs_df_with_features)

    return related_pairs_df_with_features


def get_sister_control_dataframe(track_level_feature_df):
    """
    Get sister control dataframe.

    Parameters
    ----------
    track_level_feature_df : pd.DataFrame
        The track level feature dataframe.

    Returns
    -------
    unrelated_pairs_df_with_features : pd.DataFrame
    """
    full_tracks = list(track_level_feature_df.track_id)
    frame_born_dict = track_level_feature_df.set_index("track_id")["Ff"].to_dict()
    colony_dict = track_level_feature_df.set_index("track_id")["colony"].to_dict()
    unrelated_pairs = []

    random.seed(4)  # the manifest is sorted by length
    random.shuffle(full_tracks)  # track 2 would always be longer than track 1 if not randomized

    for i, tid1 in enumerate(full_tracks):
        for j, tid2 in enumerate(full_tracks):
            if i > j:
                if colony_dict[tid1] == colony_dict[tid2]:
                    if (
                        abs(frame_born_dict[tid1] - frame_born_dict[tid2]) < 3
                    ):  # 2*5min = 10 minutes
                        lineage = family_functions.get_ids_in_same_lineage_as(
                            track_level_feature_df, tid1
                        )
                        related = tid2 in lineage
                        if related is False:
                            unrelated_pairs.append(
                                {"track_id1": tid1, "track_id2": tid2, "control": "sister_control"}
                            )

    unrelated_pairs_df = pd.DataFrame(
        unrelated_pairs, columns=["track_id1", "track_id2", "control"]
    )

    unrelated_pairs_df_with_features = get_features_for_pairs(
        unrelated_pairs_df, track_level_feature_df
    )
    unrelated_pairs_df_with_features = between_pairs.get_distance(
        unrelated_pairs_df_with_features, "A", "A"
    )

    return unrelated_pairs_df_with_features


def get_mother_daughter_control_dataframe(track_level_feature_df):
    """
    Get mother daughter control dataframe.

    Parameters
    ----------
    track_level_feature_df : pd.DataFrame
        The track level feature dataframe.

    Returns
    -------
    unrelated_pairs_df_with_features : pd.DataFrame
    """
    full_tracks = list(track_level_feature_df.track_id)
    frame_breakdown_dict = track_level_feature_df.set_index("track_id")["Fb"].to_dict()
    frame_born_dict = track_level_feature_df.set_index("track_id")["Ff"].to_dict()
    colony_dict = track_level_feature_df.set_index("track_id")["colony"].to_dict()
    unrelated_pairs = []

    random.seed(4)  # the manifest is sorted by length
    random.shuffle(full_tracks)  # track 2 would always be longer than track 1 if not randomized

    for i, tid1 in enumerate(full_tracks):
        for j, tid2 in enumerate(full_tracks):
            if i != j:
                if colony_dict[tid1] == colony_dict[tid2]:
                    difference = frame_born_dict[tid1] - frame_breakdown_dict[tid2]
                    if difference < 12 and difference > 0:  # between 0 and 60 minutes apart
                        lineage = family_functions.get_ids_in_same_lineage_as(
                            track_level_feature_df, tid1
                        )
                        related = tid2 in lineage
                        if related is False:
                            unrelated_pairs.append(
                                {
                                    "track_id1": tid1,
                                    "track_id2": tid2,
                                    "control": "mother_daughter_control",
                                }
                            )

    unrelated_pairs_df_with_features = pd.DataFrame(
        unrelated_pairs, columns=["track_id1", "track_id2", "control"]
    )

    unrelated_pairs_df_with_features = get_features_for_pairs(
        unrelated_pairs_df_with_features, track_level_feature_df
    )
    unrelated_pairs_df_with_features = between_pairs.get_distance(
        unrelated_pairs_df_with_features, "C", "A"
    )
    unrelated_pairs_df_with_features = between_pairs.difference_half_vol_at_C_and_B(
        unrelated_pairs_df_with_features
    )

    return unrelated_pairs_df_with_features


def get_features_for_pairs(pairs_df, track_level_feature_df):
    """
    Get features for pairs by merging the preceding track level feature dataframe with the pairs dataframe.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        The pairs dataframe.
    track_level_feature_df : pd.DataFrame
        The track level feature dataframe.

    Returns
    -------
    pairs_with_features : pd.DataFrame
    """
    track_level_feature_df = track_level_feature_df.drop(
        track_level_feature_df.filter(like="NUC_").columns, axis=1
    )
    interval = load_data.get_dataset_time_interval_in_min("all_baseline")
    track_level_feature_df = convert_units.convert_duration_to_hours(
        track_level_feature_df, interval
    )
    track_level_feature_df = convert_units.convert_growth_rate_to_per_hour(
        track_level_feature_df, interval
    )

    track_level_feature_df_tid1 = track_level_feature_df[
        track_level_feature_df.track_id.isin(set(pairs_df.track_id1))
    ]
    track_level_feature_df_tid2 = track_level_feature_df[
        track_level_feature_df.track_id.isin(set(pairs_df.track_id2))
    ]

    track_level_feature_df_tid1 = track_level_feature_df_tid1.add_prefix("tid1_")
    track_level_feature_df_tid2 = track_level_feature_df_tid2.add_prefix("tid2_")

    pairs_with_features = pd.merge(
        pairs_df,
        track_level_feature_df_tid1,
        left_on="track_id1",
        right_on="tid1_track_id",
        how="left",
    )
    pairs_with_features = pd.merge(
        pairs_with_features,
        track_level_feature_df_tid2,
        left_on="track_id2",
        right_on="tid2_track_id",
        how="left",
    )

    pairs_with_features = between_pairs.add_difference_between(pairs_with_features)
    pairs_with_features = between_pairs.sum_between(pairs_with_features)

    return pairs_with_features
