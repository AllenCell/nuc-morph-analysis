import pandas as pd
from nuc_morph_analysis.lib.preprocessing import (
    load_data,
    filter_data,
    is_tp_outlier,
    global_dataset_filtering,
)
from cvapipe_analysis.tools import controller, shapespace
from nuc_morph_analysis.lib.shape_modes import configuration


def get_all_dataframe_for_shape_space():
    """
    To test how the shape modes vary over time in our fixed control dataset, we first need to create the shape space
    for the overall dataset. This function will load the baseline colony datasets and performing the appropriate preprocessing
    to create the shape space.

    Returns
    -------
    df_filtered: DataFrame
        DataFrame containing the filtered baseline colony data filtering out the short tracks
    """
    all_colonies = global_dataset_filtering.load_dataset_with_features()
    shcoeff_columns = [col for col in all_colonies.columns if "NUC_shcoeffs_" in col]
    pca_df = all_colonies.dropna(subset=shcoeff_columns).copy()
    pca_df["NUC_position_depth"] = pca_df.height
    pca_df["NUC_shape_volume"] = pca_df.volume
    df_filtered = filter_data.all_timepoints_minimal_filtering(pca_df)
    return df_filtered


def get_fixed_dataframe_for_shape_space(tid_outliers=[]):
    """
    Filter fixed dataset to remove outliers, nuclei touching the edge of the fov, short tracks,
    and a specific track that is an outlier. Add columns required for calculating shape modes.

    Paramaters
    ----------
    tid_outliers: list
        List of track ids that are obviously different from the other nuclier tracks

    Returns
    -------
    df_fix_filtered: DataFrame
        DataFrame containing the filtered fixed control dataset
    """
    df_fix = load_data.load_morflowgenesis_dataframe("fixed_control")
    df_fix["is_tp_outlier"] = is_tp_outlier.outlier_detection(df_fix).is_tp_outlier
    df_fix = df_fix[~df_fix["track_id"].isin(tid_outliers)]
    df_fix["NUC_position_depth"] = df_fix.height
    df_fix["NUC_shape_volume"] = df_fix.volume
    df_fix_filtered = filter_data.filter_out_short(df_fix, length_threshold=19)
    df_fix_filtered = df_fix_filtered[~df_fix_filtered.is_tp_outlier & ~df_fix_filtered.fov_edge]
    return df_fix_filtered


def get_dataframe_with_shape_modes(df_filtered, df_fix, config=None):
    """
    To test how the shape modes vary over time in our fixed control dataset, we first need to create the shape space
    for the overall dataset. This function will create the shape space with the baseline colony datasets. Then, it will
    project the fixed control dataset onto the shape space. Finally, it will add the shape modes to the fixed control dataset.

    Parameters
    ----------
    df_filtered: DataFrame
        DataFrame containing the filtered baseline colony data filtering out the short tracks
    df_fix: DataFrame
        DataFrame containing the fixed control dataset
    config_path: str
        Path to the config file. If None, will use the default config file.

    Returns
    -------
    df_fix: DataFrame
        DataFrame containing the fixed control dataset with the shape modes added
    """
    if config is None:
        config = configuration.nucmorph_fixed_control

    control = controller.Controller(config)
    # device = io.LocalStagingIO(control)
    space = shapespace.ShapeSpace(control)
    space.execute(df_filtered)

    df_fix_pca = df_fix[space.features].values
    fix_shape_modes = space.pca.transform(df_fix_pca)
    df_fix_shape_modes = pd.DataFrame(
        fix_shape_modes, columns=space.shape_modes.columns, index=df_fix.index
    )
    for col in df_fix_shape_modes.columns:
        df_fix[col] = df_fix_shape_modes[col]

    return df_fix
