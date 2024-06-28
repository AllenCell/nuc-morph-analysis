import numpy as np


def calculate_mean_height(df, pixel_size):
    """
    Calculate the mean height for a given index_sequence (i.e. timepoint) and the standard deviation of the mean.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    pixel_size : float
        Pixel size in microns.

    Returns
    -------
    mean_height : list
        List of mean heights for each index_sequence.
    standard_dev_height : list
        List of standard deviations of the mean heights for each index_sequence.
    """
    mean_height = []
    standard_dev_height = []
    for _, df_frame in df.groupby("index_sequence"):
        height = df_frame.height * pixel_size
        mean_height.append(np.mean(height))
        standard_dev_height.append(np.std(height))
    return mean_height, standard_dev_height


def find_colony_time_alignment(df1, df2, pixel_size, interval, plot=False):
    """
    Aligns two colonies in time based by minimizing the difference between thier mean height trajectories.

    Parameters
    ----------
    df1 : Dataframe
        Dataframe containing the first colony.
    df2 : Dataframe
        Dataframe containing the second colony.
    pixel_size : float
        Pixel size in microns.

    Returns
    -------
    time_lag: Int
        The time delay between the two colonies in frames
    """
    mean_height_1, _ = calculate_mean_height(df1, pixel_size)
    mean_height_2, _ = calculate_mean_height(df2, pixel_size)
    height_list_1 = list(mean_height_1)
    height_list_2 = list(mean_height_2)
    iteration_counter = 0
    squared_difference_list = []
    while iteration_counter < df1.index_sequence.nunique():
        squared_difference = [
            (height1 - height2) ** 2 for (height1, height2) in zip(height_list_1, height_list_2)
        ]

        del height_list_1[0]
        del height_list_2[-1]
        squared_difference_list.append(np.mean(squared_difference))
        iteration_counter = iteration_counter + 1
    index_of_min_difference = np.where(squared_difference_list == np.min(squared_difference_list))
    time_lag = index_of_min_difference[0][0]
    return time_lag
