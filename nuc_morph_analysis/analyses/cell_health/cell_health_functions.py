import pandas as pd

def subset_main_dataset(df, df_full, index_sequence_threshold=480):
    '''
    Timepoints after 40 hours into the timelapse have the greatest occurrence of cell death. 
    To test the effect of this data on our results, we will can omit the final timepoints of the movie. 

    
    Parameters
    ----------
    df: pandas.DataFrame
        The main dataframe to be filtered.
    df_full: pandas.DataFrame
        The full dataframe used to identify tracks to drop.

    Returns
    ------
    df_sub: pandas.DataFrame
        Filtered dataframe with timepoints less than 40 hours.
    '''
    # remove full tracks from analysis dataset that go to the end of the movie
    df_full_to_drop = df_full[df_full['index_sequence']>index_sequence_threshold]
    tracks_to_drop = df_full_to_drop.track_id.unique()

    df_copy = df.copy()
    # set 'is_full_track' to False for tracks in the tracks_to_drop list
    df_copy.loc[df['track_id'].isin(tracks_to_drop), 'is_full_track'] = False

    # remove timepoints after 40 hours
    df_sub = df_copy[df_copy['index_sequence']<=index_sequence_threshold]
    return df_sub