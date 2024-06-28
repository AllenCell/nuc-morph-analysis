def print_number_of_sister_md_pairs(df):
    """
    Calculate the number of related pair we have

    Parameters
    ----------
    df: Dataframe
        Lineage dataframe returned from nuc_morph_analysis/analyses/lineage/dataset/lineage_pairs_dataset.py

    Returns
    -------
    print statement with stats
    """
    df_mother_daughter_pairs = df[df["delta_depth"] == 1]
    df_mother_daughter_pairs = df_mother_daughter_pairs[
        df_mother_daughter_pairs["same_branch"] == True
    ]
    print(f"{len(df_mother_daughter_pairs)} mother daughter pairs ")

    df_sister_pairs = df[df["cousiness"] == 1]
    print(f"{len(df_sister_pairs)} sister pairs")

    df_g_mother_daughter_pairs = df[df["delta_depth"] == 2]
    df_g_mother_daughter_pairs = df_g_mother_daughter_pairs[
        df_g_mother_daughter_pairs["same_branch"] == True
    ]
    print(f"{len(df_g_mother_daughter_pairs)} grandmother daughter pairs ")

    df_cousin_pairs = df[df["cousiness"] == 2]
    print(f"{len(df_cousin_pairs)} cousin pairs")
