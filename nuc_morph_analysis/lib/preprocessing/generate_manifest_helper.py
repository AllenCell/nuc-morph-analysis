from datetime import datetime
from pathlib import Path
import os
import numpy as np


def write_result(df, title, destdir=None, format="csv"):
    """
    Parameters
    ----------
    df: pandas.DataFrame
    destdir: str or Path, optional
        Absolute path to write to. Defaults to nuc-morph-analysis/data
    format: str, optional
        "parquet" or "csv"
    """
    if destdir is None:
        destdir = Path(__file__).parent.parent.parent.parent / "data"
    datestr = datetime.today().strftime("%Y-%m-%d")
    os.makedirs(destdir, exist_ok=True)

    substrings = ["drug_perturbation", "feeding_control"]
    for substring in substrings:
        title = remove_subsequent_occurrences(title, substring)

    filename = f"{destdir}/{datestr}_{title}.{format}"

    if format == "parquet":
        df.to_parquet(filename, index=False)
    elif format == "csv":
        df.to_csv(filename, index=False)
    else:
        raise ValueError(f"Unknown format: {format}")
    print(f"Wrote result to {filename}")


def remove_subsequent_occurrences(s, sub):
    first_occurrence = s.find(sub)
    if first_occurrence == -1:
        return s
    return s[:first_occurrence] + s[first_occurrence:].replace(sub, "", s.count(sub) - 1)


def update_id_by_colony(row, id):
    """
    Update track_ids and parent_ids to be for each colony by using a prefix integer in front the
    parent_id and track_id. For example, in the small colony, the track IDs will have a 1 before
    them (ie. 41 turns into 141) while the medium colony will have a 2 (ie. 41 turns into 241).
    Track_id values should never be -1 or nan. Parent_id values of -1 or nan mean that these cells
    do not have a parent so we need to preserve that information.

    Parameters
    ----------
    row: Pandas series
        Row of the dataframe corresponding to the ids to be loaded.
    id: Int or Float
        x.track_id or x.parent_id

    Returns
    -------
    id: int
        Unique matching parent id for this colony.
    """
    if id == -1 or np.isnan(id):
        return id
    else:
        if row.colony == "small":
            prefix = 7
        elif row.colony == "medium":
            prefix = 8
        elif row.colony == "large":
            prefix = 9

        elif row.colony == "drug_perturbation_1_scene0":
            prefix = 10
        elif row.colony == "drug_perturbation_1_scene2":
            prefix = 11
        elif row.colony == "drug_perturbation_1_scene3":
            prefix = 12
        elif row.colony == "drug_perturbation_1_scene4":
            prefix = 13
        elif row.colony == "drug_perturbation_2_scene0":
            prefix = 14
        elif row.colony == "drug_perturbation_2_scene6":
            prefix = 15
        elif row.colony == "drug_perturbation_2_scene14":
            prefix = 16
        elif row.colony == "drug_perturbation_4_scene2":
            prefix = 17
        elif row.colony == "drug_perturbation_4_scene4":
            prefix = 18

        elif row.colony == "feeding_control_baseline":
            prefix = 20
        elif row.colony == "feeding_control_starved":
            prefix = 21
        elif row.colony == "feeding_control_refeed":
            prefix = 22

        # The parent_id column fails without the float conversion.
        return int(float(f"{prefix}{id}"))
