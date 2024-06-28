import numpy as np


def absolute_error_per_datapoint(values):
    """
    Calculate the absolute error for each datapoint in a list of values (i.e. volumes) for a given track.
    The median value is the assumed to be the true value as the dataset used is a fixed plate imaged multiple times,
    such that calculated features should not be changing.

    Parameters
    -----
    values: list of floats
        The values for a given feature for a given track. (i.e. volumes, heights, surface areas, shape modes)

    Returns
    -----
    absolute_error: list of floats
        The absolute error for each datapoint in the list of values.
    """
    median = np.median(values)
    absolute_error = abs(values - median)
    return absolute_error


def percent_error_per_datapoint(values):
    """
    Calculate the percent error for each datapoint in a list of values (i.e. volumes) for a given track.
    The median value is the assumed to be the true value as the dataset used is a fixed plate imaged multiple times,
    such that calculated features should not be changing.

    Parameters
    -----
    values: list of floats
        The values for a given feature for a given track. (i.e. volumes, heights, surface areas, shape modes)

    Returns
    -----
    percent_error: list of floats
        The percent error for each datapoint in the list of values.
    """
    median = np.median(values)
    absolute_error = abs(values - median)
    percent_error = abs(absolute_error / median) * 100
    return percent_error


def percent_error_population_variation(df, column_name, values):
    """
    Calculate the percent error taking into account the population variation for a given feature.
    The median value is the assumed to be the true value as the dataset used is a fixed plate imaged multiple times,
    such that calculated features should not be changing. This metric is particularly relavent for shapemodes
    where the absolute and percent error are huge due to the small value of the PC mode and the range it has on a given track.
    Seeing how that compares to all the range of possible PC values for that shape mode helps contextualize that error.

    Parameters
    -----
    values: list of floats
        The values for a given feature for a given track. (i.e. volumes, heights, surface areas, shape modes)
        Specifically relavent for shape modes

    Returns
    -----
    population_variation: float
        The population variation for the given feature for the given track.
    """
    median = np.median(values)
    absolute_error = abs(values - median)
    pop_var = df[column_name].quantile(0.95) - df[column_name].quantile(0.05)
    percent_error_pop_var = abs(absolute_error / pop_var) * 100
    return percent_error_pop_var


def add_error_to_df(
    df, pixel_size, error_type, features=["volume", "surface_area", "height", "shape_modes"]
):
    """
    Add error columns to a dataframe. The error is calculated as the absolute error or the percent error.
    The error is calculated for each datapoint in a track. The error is calculated for the volume, height,
    surface area, and shape modes of each track.

    Parameters
    -----
    df: Dataframe
        The fixed control dataframe.
    pixel_size: float
        The pixel size of the images returned by the load_data.get_dataset_pixel_size('fixed_control') function.
    error_type: string
        'absolute', 'percent', 'pop_var'
    features: list of strings
        The features to calculate the error for.
        The options are 'volume', 'surface_area', 'height', and 'shape_modes'.

    Returns
    -----
    df: Dataframe
        The fixed control dataframe with the error columns added.
    """

    for feature in features:
        if feature == "volume":
            column_names = ["volume"]
            scale = pixel_size * pixel_size * pixel_size
            labels = ["volume"]
        if feature == "mesh_volume":
            column_names = ["mesh_vol"]
            scale = pixel_size * pixel_size * pixel_size
            labels = ["mesh_volume"]
        if feature == "height":
            column_names = ["height"]
            scale = pixel_size
            labels = ["height"]
        if feature == "surface_area":
            column_names = ["mesh_sa"]
            scale = pixel_size * pixel_size
            labels = ["surface_area"]
        if feature == "shape_modes":
            column_names = [
                "NUC_PC1",
                "NUC_PC2",
                "NUC_PC3",
                "NUC_PC4",
                "NUC_PC5",
                "NUC_PC6",
                "NUC_PC7",
                "NUC_PC8",
            ]
            scale = 1
            labels = column_names

        for column_name, label in zip(column_names, labels):
            error_list = []
            for _, dft in df.groupby("track_id"):
                values = dft[column_name].to_numpy() * scale

                if error_type == "absolute":
                    error_values = absolute_error_per_datapoint(values)
                if error_type == "percent":
                    error_values = percent_error_per_datapoint(values)
                if error_type == "pop_var":
                    error_values = percent_error_population_variation(df, column_name, values)

                for value in error_values:
                    error_list.append(value)
            df[f"{error_type}_error_{label}"] = error_list
    return df


def get_error_value(df, error_type, feature, percentile="99%"):
    """
    The absolute error and the percent error for each feature is precalculated and added to the dataframe. Now we can use the
    distrubution of that error to determine the error represented for a given feature. The 99th percentile is used as the default.

    Parameters
    -----
    df: Dataframe
        The fixed control dataframe with error columns added
    error_type: string
        'absolute', 'percent', 'pop_var'
    feature: string
        The feature to get the error value for. The options are 'volume', 'surface_area', 'height',
        'NUC_PC1', 'NUC_PC2', 'NUC_PC3', 'NUC_PC4', 'NUC_PC5', 'NUC_PC6', 'NUC_PC7', 'NUC_PC8'.
    percentile: string
        The percentile to get the error value for.
        The options are '5%', '25%', '50%', '75%', '95%', and '99%'.

    Returns
    -----
    error_value: float
        The error value for the given feature.
    """
    column_name = [f"{error_type}_error_{feature}"]
    error_statistics = df[column_name].describe([0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    error_value = error_statistics.loc[percentile, column_name][0]
    return np.round(error_value, 2)


def print_error_values(df, percentile="99%"):
    """
    Generate error metric for all measured features.
    These values are hard coded in the get_error_value function.

    Parameters
    -----
    df: Dataframe
        The fixed control dataframe with error columns added
    percentile: string
        The percentile to get the error value for.
        The options are '5%', '25%', '50%', '75%', '95%', and '99%'.

    Returns
    -----
    Print statements of the error values for all measured features.
    """

    feature_list = [
        "volume",
        "height",
        "surface_area",
        "NUC_PC1",
        "NUC_PC2",
        "NUC_PC3",
        "NUC_PC4",
        "NUC_PC5",
        "NUC_PC6",
        "NUC_PC7",
        "NUC_PC8",
    ]

    print("Absolute error values")
    for feature in feature_list:
        error_value = get_error_value(
            df, error_type="absolute", feature=feature, percentile=percentile
        )
        print(f"{feature}: {error_value}")

    print("\nPercent error values ")
    for feature in feature_list:
        error_value = get_error_value(
            df, error_type="percent", feature=feature, percentile=percentile
        )
        print(f"{feature}: {error_value}")

    print("\nPercent error population variation values ")
    for feature in feature_list:
        error_value = get_error_value(
            df, error_type="pop_var", feature=feature, percentile=percentile
        )
        print(f"{feature}: {error_value}")
