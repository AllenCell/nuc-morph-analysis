import numpy as np
from multiprocessing import Pool
import pandas as pd
import ast
from tqdm import tqdm


def get_inputs(df, feature_keys) -> tuple:
    return tuple([d.copy() for d in df[["neighbors"] + feature_keys].to_dict(orient="records")])


def compute_neigh(tup):
    """
    Compute density given a cell and its neighbors

    Parameters
    ----------
    tup: tuple
        A tuple that contains the following keys

        CellID
        neighbors
        height
        centroid_x
        centroid_y

    Returns
    -------
    A dataframe with the following information

    CellId
    Height
    Neigh ID
    Distance

    across all the neighboring Cell IDs

    """
    cell_id = tup["CellId"]
    track_id = tup["track_id"]
    index_sequence = tup["index_sequence"]
    neighbors = tup["neighbors"]
    height = tup["height"]
    growth_rate = tup["late_growth_rate_by_endpoints"]
    p1 = [tup["centroid_x"], tup["centroid_y"]]
    p1 = np.array(p1)

    try:
        neighbors = ast.literal_eval(neighbors)
        neighbors = [i for i in neighbors if i != cell_id]
    except:
        neighbors = None

    if neighbors is None:
        return 0, 0
    else:
        return_dict = {
            "CellId": [],
            "track_id": [],
            "index_sequence": [],
            "dist": [],
            "neigh_id": [],
            "height": [],
            "growth_rate": [],
        }

        this_kk1 = kk.loc[kk["CellId"].isin(neighbors)].sample(frac=1).reset_index(drop=True)

        if this_kk1.shape[0] == 0:
            return None
        else:
            p2 = np.array(this_kk1[["centroid_x", "centroid_y"]])

            ids = this_kk1["CellId"]

            for inds in range(p2.shape[0]):
                squared_dist = np.sum((p1 - p2[inds]) ** 2, axis=0)
                dist = np.sqrt(squared_dist)
                return_dict["CellId"].append(cell_id)
                return_dict["track_id"].append(track_id)
                return_dict["index_sequence"].append(index_sequence)
                return_dict["neigh_id"].append(ids[inds])
                return_dict["height"].append(height)
                return_dict["dist"].append(dist)
                return_dict["growth_rate"].append(growth_rate)

            return_dict = pd.DataFrame(return_dict)
            return return_dict


def compute_density(df, global_df, num_workers=1, old_unfiltered_method=False):
    """
    Main function to run
    Given a dataframe, compute density for each row
    args
    ----------
    df: dataframe of cell ids with location and height information
    global_df: dataframe of all neighboring cell ids with location and height information
    num_workers: number of workers for multiprocessing
    old_unfiltered_method: whether to use the old (unfiltered) method that does not remove bad pseudo cells
    (e.g. cells next to mitotic cells where there are missing segmentations)
    """
    feature_keys = [
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "CellId",
        "track_id",
        "index_sequence",
        "height",
        "late_growth_rate_by_endpoints",
    ]
    inputs = get_inputs(df, feature_keys)

    global kk
    kk = global_df  # has information about all neighbors

    with Pool(num_workers) as p:
        neigh_stats = tuple(
            tqdm(
                p.imap_unordered(
                    compute_neigh,
                    inputs,
                ),
                total=len(inputs),
                desc="compute_neigh",
            )
        )
    neigh_stats = [i for i in neigh_stats if i is not None]
    neigh_stats = pd.concat(neigh_stats, axis=0)
    neigh_stats = neigh_stats.groupby(["CellId"]).mean()


    print('old_unfiltered_method', old_unfiltered_method)
    if not old_unfiltered_method:
        # now remove CellIds that have bad pseudo cell segmentation (i.e. bad_pseudo_cells_segmentation)
        # 'uncaught_pseudo_cell_artifact','bad_pseudo_cells_segmentation'
        print("Removing bad pseudo cells")
        print('Before removing bad pseudo cells: ', neigh_stats.shape[0])
        cell_ids_to_remove = df[df['bad_pseudo_cells_segmentation'] == True]['CellId'].values
        neigh_stats = neigh_stats[~neigh_stats.index.isin(cell_ids_to_remove)]
        print('After removing bad pseudo cells: ', neigh_stats.shape[0])

    return neigh_stats
