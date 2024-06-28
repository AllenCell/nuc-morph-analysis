# %%
"""
Combine morflowgenesis segmentations with time consuming features to create the main manifest.

Run as an interactive script or with:
  pdm run python nuc_morph_analysis/lib/preprocessing/generate_perturbation_manifest.py
"""
import logging
import numpy as np
import pandas as pd
from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.lib.preprocessing.add_colony_metrics import add_colony_metrics
from nuc_morph_analysis.lib.preprocessing.generate_manifest_helper import (
    write_result,
    update_id_by_colony,
)
from nuc_morph_analysis.analyses.feeding_control.fov_shift_track_matching import (
    match_and_update_dataframe,
    FRAMES_TO_SHIFT,
)


# %%
def generate_manifest_one_colony(morflowgenesis_df, dataset, experiments=None):
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
    if "formation" not in morflowgenesis_df.columns:
        logging.warning("No formation column found. Adding empty formation column.")
        morflowgenesis_df["formation"] = np.nan

    if "breakdown" not in morflowgenesis_df.columns:
        logging.warning("No breakdown column found. Adding empty breakdown column.")
        morflowgenesis_df["breakdown"] = np.nan

    morflowgenesis_df = morflowgenesis_df.rename(
        columns={
            "formation": "predicted_formation",
            "breakdown": "predicted_breakdown",
        }
    )

    # -------------------------------------------
    # STEP 3: Add empty feature columns
    # -------------------------------------------
    step3_df = morflowgenesis_df.copy()
    step3_df["parent_id"] = np.nan
    step3_df["termination"] = np.nan
    step3_df["colony_time"] = np.nan

    # ---------------------------------------------
    # STEP 4: Make track IDs unique
    # ---------------------------------------------
    step4_df = step3_df.copy()
    step4_df["colony"] = dataset
    step4_df["track_id"] = step4_df.apply(
        lambda x: update_id_by_colony(x, x.track_id), axis="columns"
    )
    step4_df["parent_id"] = step4_df.apply(
        lambda x: update_id_by_colony(x, x.parent_id), axis="columns"
    )

    # ---------------------------------------------
    # Bonus step: Fix broken tracks in feeding controls
    # ---------------------------------------------
    if experiments == "feeding_control":
        step5_df = step4_df.copy()
        step5_df = match_and_update_dataframe(step5_df, dataset, FRAMES_TO_SHIFT)
    else:
        step5_df = step4_df

    # --------------------------
    # STEP 5: Add colony metrics
    # --------------------------
    # density and other add_colony_metrics features
    logging.info("Calculating colony metrics")
    return add_colony_metrics(step5_df)


def get_combined_manifest(experiments):
    """
    experiments: Str
    "feeding control" or "drug perturbation"
    """
    datasets = load_data.get_available_datasets(experiments=experiments)

    dataframes = []
    for colony in datasets:
        df_single_dataset = generate_manifest_one_colony(
            load_data.load_morflowgenesis_dataframe(colony), colony, experiments
        )
        dataframes.append(df_single_dataset)

    return pd.concat(dataframes)


# %%
for experiments in ["feeding_control", "drug_perturbation"]:
    df = get_combined_manifest(experiments)
    write_result(df, f"{experiments}_main_manifest", format="parquet")
# %%
