import pandas as pd


def merge_termination_annotations(df_morflowgenesis, df_annotations):
    return pd.merge(df_morflowgenesis, df_annotations, on="track_id", how="left")
