import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

def correct_cellids_with_missing_neighbors(df):
    """
    Function to correct the `neighbors` column in the dataframe
    where the `neighbors` column is a string representation of a list
    but some of the entries are 'None' instead of '[]'

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe with the `neighbors` column

    Returns
    -------
    df : pd.DataFrame
        the dataframe with the corrected `neighbors` column
    """

    # one CellId is weird and has neighbors = None, so we need to change that
    df.loc[df['CellId']=='ed8124da9dfe45bc3b64d65aeb7446ff70db4255d41309bb5d1eb9b4','neighbors'] = '[]'
    return df

def mark_frames_of_formation_and_breakdown(df):
    """
    define new columns called `frame_of_breakdown` and `frame_of_formation`
    to store the frame of the mitotic event (i.e. breakdown or formation)

    frame_of_breakdown: boolean, True if the frame is the frame of breakdown
    frame_of_formation: boolean, False if the frame is the frame of formation

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe with the mitotic event predictions
        (also needs index_sequence and predicted_breakdown and predicted_formation columns)

    Returns
    -------
    df : pd.DataFrame
        the dataframe with the new columns
    """

    df['frame_of_breakdown'] = df['index_sequence']==df['predicted_breakdown']
    df['frame_of_formation'] = df['index_sequence']==df['predicted_formation']
    return df

def find_neighbors_of_cells(df,bool_col=None,new_col=None):
    """
    This function takes in a dataframe, optionally subsets it with a boolean column
    so that the dataframe only contains a specific set of cells of interest (e.g. mitotic cells or dying cells)
    and finds all the neighbors of these specific cells in the subset
    and marks those neighbors with a new boolean column

    Note: it is required that each entry of the `neighbors` column is a string representation of a list
    the function `correct_cellids_with_missing_neighbors` is be used to correct the `neighbors` column 
    """
    # initialize new columns to store the new boolean values
    if new_col is None:
        new_col = f'has_{bool_col}_neighbor'
    df[new_col] = False
    df[new_col] = df[new_col].astype(bool)

    df[f'number_of_mitotic_{bool_col}_neighbors'] = [0]*len(df)

    if bool_col is not None:
        dfmsub = df[df[bool_col]]
    else:
        dfmsub = df

    # convert the string representation of list of neighbors to a list
    dfmsub['neighbor_list'] = dfmsub['neighbors'].apply(lambda x: eval(x)) 


    list_of_neighbors = dfmsub['neighbor_list'].values
    if len(list_of_neighbors)>0:
        single_list = np.concatenate(dfmsub['neighbor_list'].values) # get a list of all the neighbor cellids
        df.loc[df['CellId'].isin(single_list),new_col] = True

        # now count the number of times each CellId appears in single_list
        # and store that in the `number_of_mitotic_{bool_col}_neighbors` column
        values,counts = np.unique(single_list,return_counts=True)
        df.set_index('CellId',inplace=True)
        df.loc[values,f'number_of_mitotic_{bool_col}_neighbors'] = counts
        df.reset_index(inplace=True)

    else:
        print('NOTE: No neighbors found for',bool_col)
        df.loc[:,new_col] = False
    return df

def mark_all_neighbors_of_mitotic_events(df):
    """
    finds all `predicted_breakdown` and `predicted_formation` events
    and marks all the neighbors of those events

    Note: it is required that each entry of the `neighbors` column is a string representation of a list
    the function `correct_cellids_with_missing_neighbors` is be used to correct the `neighbors` column 

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe with the mitotic event predictions
        (also needs neighbors column)

    Returns
    -------
    df : pd.DataFrame
        the dataframe with the new columns `has_mitotic_neighbor_breakdown` and `has_mitotic_neighbor_formation`
    """

    # subset the dataframe to only the mitotic events
    # df_mit = df[(df['frame_of_breakdown']) | (df['frame_of_formation'])]

    for subcol in ['breakdown','formation']:
        new_col = f'has_mitotic_neighbor_{subcol}'

        # now find all neighbors (CellIds) of the mitotic cells
        # and set the `has_mitotic_neighbor_{subcol}` column to True for those neighbors
        col = f'frame_of_{subcol}'
        df = find_neighbors_of_cells(df,col,new_col)
    return df

def expand_boolean_labels_in_time(df, feature, iterations=4, direction='forward'):
    """
    Function to expand boolean labels in time
    by dilating the boolean labels in time

    the workflow converts the boolean labels to a pivot table and assumes that the index is the timepoint
    and that the time values are sorted in ascending order (e.g. 0,1,2,3,4,5,6,7,8,9,10 from top to bottom)
    and the columns are the track_ids

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe with the boolean labels
    feature : str
        the name of the boolean label column
    iterations : int
        the number of iterations to dilate the boolean labels
    direction : str
        the direction to dilate the boolean labels
        'forward' or 'backward'
        this determines the connectivity matrix used for the dilation

    Returns
    -------
    df : pd.DataFrame
        the dataframe with the new column `{feature}_{direction}_dilated`
    """

    # now propogate these annotations
    # forward propogate breakdown events backwards in time
    # backward propogate formation events forwards in time

    # first find the tracks where `has_mitotic_neighbor_breakdown` (or formation) is true
    x_col = "index_sequence"
    column_val = 'track_id'

    pivot = df.pivot(index=x_col, columns=column_val, values=feature)

    dilated_matrix = perform_boolean_dilation_on_pivot(pivot, iterations=iterations, direction=direction)

    # now put the dilated matrix back into the dataframe
    dilated_matrix = pd.DataFrame(dilated_matrix, index=pivot.index, columns=pivot.columns)
    dilated_matrix = dilated_matrix.stack().reset_index()
    dilated_matrix.columns = [x_col,column_val,f"{feature}_{direction}_dilated"]
    df = pd.merge(df,dilated_matrix, on=[x_col,column_val],how='left',suffixes=('','_dilated'))
    return df

def perform_boolean_dilation_on_pivot(pivot, iterations, direction):
    """
    takes a pandas pivot table with boolean values and dilates the True values
      in the matrix up ('backward') or down ('forward')

    Parameters
    ----------
    pivot : pd.DataFrame
        the pivot table with boolean values
        NaN values are allowed and will be treated as False
    iterations : int
        the number of iterations to dilate the boolean labels
    direction : str
        the direction to dilate the boolean labels
        'forward' or 'backward'
        this determines the connectivity matrix used for the dilation

    Returns
    -------
    dilated_matrix : np.array
        the dilated matrix
    """
    # keep the Nans because those are points where tracking is non-valid
    isnan_matrix = pivot.isna()

    # now convert the pivot to a boolean matrix 
    bool_mat = pivot.copy()
    bool_mat[bool_mat.isna()] = False # make all the NaNs False so boolean operations work
    bool_mat = bool_mat.astype(bool)

    # now expand the True values upward in the matrix
    if direction == 'backward':
        connectivity_matrix = np.asarray(
            [[0,1,0],
            [0,1,0],
            [0,0,0]]
            )
    elif direction == 'forward':
        connectivity_matrix = np.asarray(
            [[0,0,0],
            [0,1,0],
            [0,1,0]]
            )

    # now expand the True values upward in the matrix
    dilated_matrix = binary_dilation(bool_mat.values,
                                    structure=connectivity_matrix,
                                    iterations=iterations,
                                    )

    # remove all the non-valid points
    dilated_matrix = np.logical_and(dilated_matrix,~(isnan_matrix.values))
    return dilated_matrix

def combine_formation_and_breakdown_labels(df):
    """
    define a new column called `has_mitotic_neighbor_dilated`
    that combines the `has_mitotic_neighbor_breakdown_dilated` and `has_mitotic_neighbor_formation_dilated` columns
    using the logical OR operation

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe with columns `has_mitotic_neighbor_breakdown_dilated` and `has_mitotic_neighbor_formation_dilated`
        (also needs index_sequence and track_id columns)
    
    Returns
    -------
    df : pd.DataFrame
        the dataframe with the new column `has_mitotic_neighbor_dilated` and `has_mitotic_neighbor`
    """
    df['has_mitotic_neighbor_dilated'] = df['has_mitotic_neighbor_breakdown_forward_dilated'] | df['has_mitotic_neighbor_formation_backward_dilated']
    df['has_mitotic_neighbor'] = df['has_mitotic_neighbor_breakdown'] | df['has_mitotic_neighbor_formation']
    
    return df

def label_nuclei_that_neighbor_current_mitotic_event(df,iterations=9):
    """
    Function to label the nuclei that neighbor the current mitotic event
    it looks for all formation and breakdown events, 
    then marks/labels the neighbors of those events at that timepoint
    and expands those labels to capture the other timepoints where the 
    mitotic cells lack a segmentation
    
    Parameters
    ----------
    df : pd.DataFrame
        the dataframe with the mitotic event predictions
        (also needs neighbors column)
    iterations : int
        the number of iterations to dilate the boolean labels
        
    Returns
    -------    
    df : pd.DataFrame
        the dataframe with the new columns `has_mitotic_neighbor_breakdown_forward_dilated`
        and `has_mitotic_neighbor_formation_backward_dilated` and
        the combined column `has_mitotic_neighbor_dilated`
    """
    
    df = correct_cellids_with_missing_neighbors(df)
    df = mark_frames_of_formation_and_breakdown(df)
    df = mark_all_neighbors_of_mitotic_events(df)
    df = expand_boolean_labels_in_time(df, 'has_mitotic_neighbor_breakdown', iterations, direction='forward')
    df = expand_boolean_labels_in_time(df, 'has_mitotic_neighbor_formation', iterations, direction='backward')
    df = combine_formation_and_breakdown_labels(df)
    return df


#%%
def identify_frames_of_death(df):
    """
    identify the frames of death for each track_id with a termination event = 2 (dying)
    the frame of death is the last frame where the track is present

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe with the track_id, index_sequence, and termination columns

    Returns
    -------
    df : pd.DataFrame
        the dataframe with the new column `identified_death`
    """
    dft = df.loc[df.termination == 2,:]
    dftg = dft.groupby('track_id').agg({'index_sequence':'max'}).reset_index()
    dftg['identified_death'] = np.uint16(dftg['index_sequence'].values)
    df = pd.merge(df,dftg[['track_id','identified_death']],on='track_id',how='left',suffixes=('','_death'))
    return df

def mark_frames_of_death(df):
    """
    define new column called `frame_of_death` 
    to store the frame of the death (apoptotic) event

    frame_of_death: boolean, True if the frame is the frame of death

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe with the death event predictions
        (also needs index_sequence and identified_death)
        note: identify_frames_of_death function should be used to identify the frames of death

    Returns
    -------
    df : pd.DataFrame
        the dataframe with the new columns
    """

    df['frame_of_death'] = df['index_sequence']==df['identified_death']
    return df

def label_nuclei_that_neighbor_current_death_event(df):
    df = correct_cellids_with_missing_neighbors(df)
    df = identify_frames_of_death(df)
    df = mark_frames_of_death(df)
    df = find_neighbors_of_cells(df,bool_col='frame_of_death',new_col='has_dying_neighbor')
    df = expand_boolean_labels_in_time(df, 'has_dying_neighbor', iterations=6, direction='forward')
    return df


