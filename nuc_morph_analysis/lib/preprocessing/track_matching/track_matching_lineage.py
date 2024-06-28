import pandas as pd

from nuc_morph_analysis.lib.preprocessing.track_matching.merge_formation_breakdown import (
    merge_formation_breakdown,
)


def merge_lineage_annotations(annotations: pd.DataFrame, morflowgenesis: pd.DataFrame):
    # Add parent_id, termination, and merged_track_id column
    merged = morflowgenesis.merge(
        annotations, left_on="track_id", right_on="morflowgenesis_track_id", how="left"
    )
    # Replace track_id with merged_track_id, if present
    merged.loc[~merged.merged_morflowgenesis_track_id.isna(), "track_id"] = (
        merged.merged_morflowgenesis_track_id
    )
    # TODO drop these columns before saving the annotations
    merged = merged.drop(
        columns=[
            "morflowgenesis_track_id",
            "track_id_newly_annotated_2023_segmentations",
            "merged_morflowgenesis_track_id",
        ]
    )
    # Pick predicted_formation and predicted_breakdown for merged tracks
    return merge_formation_breakdown(merged)
