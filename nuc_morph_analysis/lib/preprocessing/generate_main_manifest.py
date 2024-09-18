"""
Combine morflowgenesis segmentations with lineage annotations to create the main manifest.

Run as an interactive script or with:
  pdm run python nuc_morph_analysis/lib/preprocessing/generate_main_manifest.py
"""

# %%
import logging
import numpy as np
import pandas as pd

from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.lib.preprocessing.track_matching import (
    track_matching_apoptosis,
    track_matching_lineage,
)
from nuc_morph_analysis.lib.preprocessing.add_colony_metrics import add_colony_metrics
from nuc_morph_analysis.lib.preprocessing.track_matching.merge_formation_breakdown import (
    validate_formation_breakdown,
)
from nuc_morph_analysis.utilities.warn_slow import warn_slow
from nuc_morph_analysis.lib.preprocessing.generate_manifest_helper import (
    write_result,
    update_id_by_colony,
)
from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import watershed_workflow



# %%
@warn_slow("60min")  # takes about 60 min (fomerly; In testing takes about 4-5 minutes)
def generate_manifest_one_colony(morflowgenesis_df, colony, manual_lineage_annotations=None):
    """
    Parameters
    ----------
    morflowgenesis_df: pandas.Dataframe
    colony: str
        "small", "medium", or "large"
    manual_lineage_annotations: pandas.Dataframe, optional
        Manual lineage annotations for the colony. Must have columns "track_id", "parent_id", and
        "termination". If not included, parent_id and termination columns will be filled with NaN.
    """
    # ---------------------------
    # STEP 1: Drop unused columns
    # ---------------------------
    morflowgenesis_df = morflowgenesis_df.drop(
        [
            "lineage_id",
            "dataset",
        ],
        axis="columns",
    )
    morflowgenesis_df = morflowgenesis_df.filter(regex=r"^(?!Unnamed)")

    # ----------------------
    # STEP 2: Rename columns
    # ----------------------
    morflowgenesis_df = morflowgenesis_df.rename(
        columns={
            "formation": "predicted_formation",
            "breakdown": "predicted_breakdown",
        }
    )

    # -------------------------------------------
    # STEP 3: Add parent_id & termination columns
    # -------------------------------------------
    if manual_lineage_annotations is not None:
        step3_df = track_matching_lineage.merge_lineage_annotations(
            manual_lineage_annotations, morflowgenesis_df
        )
    else:
        step3_df = morflowgenesis_df
        step3_df["parent_id"] = np.nan

    # ---------------------------------------------
    # STEP 4: Make track IDs unique
    # ---------------------------------------------
    step4_df = step3_df.copy()
    step4_df["colony"] = colony
    step4_df["track_id"] = step4_df.apply(
        lambda x: update_id_by_colony(x, x.track_id), axis="columns"
    )
    step4_df["parent_id"] = step4_df.apply(
        lambda x: update_id_by_colony(x, x.parent_id), axis="columns"
    )

    # --------------------------
    # STEP 5: Add colony metrics
    # --------------------------
    # density and other add_colony_metrics features
    logging.info("Calculating colony metrics")
    step5_df = step4_df.copy()
    step5_df = add_colony_metrics(step5_df)

    # --------------------------
    # STEP 6: calculate 2D object-based density
    # --------------------------
    logging.info("Calculating image-based density metrics")
    step6_df = step5_df.copy()
    density_df = watershed_workflow.get_pseudo_cell_boundaries_for_movie(colony,parallel=True)
    # now merge the density_df with the main dataframe
    step6_df = pd.merge(step6_df,
                            density_df,
                            on=['colony','index_sequence','label_img'],
                            suffixes=('', '__pc'),
                            how='left')
    # now remove columns with __pc suffix
    step6_df = step6_df[step6_df.columns.drop(list(step6_df.filter(regex='__pc')))]
    print(step6_df.shape)
    print("step5_df.shape",step5_df.shape)
    print("step6_df.shape",step6_df.shape)

    return step6_df


@warn_slow("90s")  # Usually takes 20-30s
def write_main_manifest(df, destdir=None, format="parquet"):
    """
    Parameters
    ----------
    df: pandas.DataFrame
    destdir: str or Path, optional
        Absolute path to write to. Defaults to nuc-morph-analysis/data
    format: str, optional
        "parquet" or "csv"
    """
    write_result(df, "main_manifest", destdir, format)

def run_workflow():
    # %%
    dataset = "large"
    morflowgenesis_df = load_data.load_morflowgenesis_dataframe(dataset)

    termination_df = load_data.load_apoptosis_annotations(dataset)
    df_with_apop_annotation = track_matching_apoptosis.merge_termination_annotations(
        morflowgenesis_df, termination_df
    )
    df = generate_manifest_one_colony(df_with_apop_annotation, dataset)

    # %%
    for dataset in ["small", "medium"]:
        morflowgenesis_df = load_data.load_morflowgenesis_dataframe(dataset)
        annotations_for_morflowgenesis = load_data.load_lineage_annotations(dataset)

        output = generate_manifest_one_colony(
            morflowgenesis_df, dataset, annotations_for_morflowgenesis
        )
        df = pd.concat([df, output], axis="rows")


    # %% Each track should have a single formation/breakdown value
    validate_formation_breakdown(df)

    # %%
    write_main_manifest(df)
    # %%

if __name__ == "__main__":
    run_workflow()