# %%
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, filter_data
from datetime import datetime
from pathlib import Path


# %%
def check_columns(df1, df2):
    """
    Check if the columns in two dataframes are the same.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe
    df2 : pd.DataFrame
        Second dataframe
    """
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    if cols1 != cols2:
        raise ValueError(
            f"Columns in dataframes are not the same: {cols1.symmetric_difference(cols2)}"
        )


def save_dataset_for_quilt(df, dataset_name, destdir=None):
    """
    Save a dataset to a csv file in the data directory.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save
    dataset_name : str
        Name of the dataset
    destdir : str
        Destination

    Returns
    -------
    None
    """
    date = datetime.today().strftime("%Y-%m-%d")
    if destdir is None:
        destdir = Path(__file__).parent.parent.parent.parent
        datadir = destdir / "data"
        datadir.mkdir(exist_ok=True, parents=True)
    df.to_csv(f"{destdir}/data/{dataset_name}_{date}.csv", index=False)
    print(f"file saved to {destdir}/data/")


# %% Load baseline colonies
df_all_baseline = global_dataset_filtering.load_dataset_with_features()
print(*[col for col in df_all_baseline.columns if "NUC_sh" not in col], sep="\n")

# %% Filter baseline colonies
df_baseline = filter_data.all_timepoints_minimal_filtering(df_all_baseline)
df_full_baseline = filter_data.all_timepoints_full_tracks(df_baseline)
df_lineage = df_full_baseline[df_full_baseline["colony"].isin(["small", "medium"])]

# %% ignore baseline analysis dataset columns in comparison
df_all_baseline = global_dataset_filtering.remove_columns(
    df_all_baseline,
    [
        "exploratory_dataset",
        "baseline_colonies_dataset",
        "full_interphase_dataset",
        "lineage_annotated_dataset",
    ],
)

# %% Load feeding control
df_all_feeding_control = global_dataset_filtering.load_dataset_with_features("all_feeding_control")
df_all_feeding_control = global_dataset_filtering.remove_columns(
    df_all_feeding_control, ["track_match_issue", "track_matched"]
)
check_columns(df_all_baseline, df_all_feeding_control)
df_full_feeding_control = filter_data.all_timepoints_full_tracks(df_all_feeding_control)

# %% Load inhibitor dataset
df_all_inhibitor = global_dataset_filtering.load_dataset_with_features("all_drug_perturbation")
check_columns(df_all_baseline, df_all_inhibitor)
df_all_inhibitor = filter_data.all_timepoints_minimal_filtering(df_all_inhibitor)

aphidicolin_scenes = [
    "drug_perturbation_1_scene0",
    "drug_perturbation_1_scene2",
    "drug_perturbation_1_scene4",
    "drug_perturbation_4_scene2",
    "drug_perturbation_4_scene4",
]
df_aphidicolin = df_all_inhibitor[df_all_inhibitor["colony"].isin(aphidicolin_scenes)]

importazole_scenes = ["drug_perturbation_2_scene0", "drug_perturbation_2_scene6"]
df_importazole = df_all_inhibitor[df_all_inhibitor["colony"].isin(importazole_scenes)]

# %%
save_dataset_for_quilt(df_all_baseline, "baseline_colonies_unfiltered_feature_dataset")
save_dataset_for_quilt(df_baseline, "baseline_colonies_analysis_dataset")
save_dataset_for_quilt(df_full_baseline, "full-interphase_dataset")
save_dataset_for_quilt(df_lineage, "lineage-annotated_analysis_dataset")
save_dataset_for_quilt(df_full_feeding_control, "feeding_control_analysis_dataset")
save_dataset_for_quilt(df_aphidicolin, "dna_replication_inhibitor_analysis_dataset")
save_dataset_for_quilt(df_importazole, "nuclear_import_inhibitor_analysis_dataset")
# %%
