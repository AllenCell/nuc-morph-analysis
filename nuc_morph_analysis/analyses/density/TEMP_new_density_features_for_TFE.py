#%%
from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import pseudo_cell_helper
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from nuc_morph_analysis.lib.visualization.reference_points import COLONY_COLORS, COLONY_LABELS
from nuc_morph_analysis.lib.preprocessing import compute_change_over_time

def add_new_features(df):
    # this is a temporary function that loads these features as a separate CSV
    df_master= df.copy()
    # load locally saved density features
    colonies = load_data.get_dataset_names(dataset='all_baseline')
    output_directory = Path(__file__).parents[6] / "local_storage" / "pseudo_cell_boundaries"
    resolution_level = 1

    pqdir = '/allen/aics/assay-dev/users/Frick/PythonProjects/repos/local_storage/pseudo_cell_boundaries/'
    dflist=[]
    for colony in colonies:
        print(colony)
        pqfilepath = pqdir + colony + '_pseudo_cell_boundaries.parquet'
        dfsub = pd.read_parquet(pqfilepath)
        dflist.append(dfsub)
    dfp = pd.concat(dflist)
    print(dfp.shape)
    dfp.head()


    #%% merge the dataframe
    import numpy as np
    if '2d_colony_nucleus' in dfp.columns:
        dfp['colony'] = dfp['2d_colony_nucleus']
    dfm = pd.merge(df, dfp, on=['colony','index_sequence','label_img'], suffixes=('', '_pc'), how='left')
    dfm.head()

    # check that no rows have been dropped
    print('running the check!')
    if df_master.shape[0] != dfm.shape[0]:
        raise Exception(
            f"The loaded manifest has {dfm.shape[0]} rows and your \
            final manifest has {df_master.shape[0]} rows.\
            Please revise code to leave manifest rows unchanged."
        )
    return dfm
