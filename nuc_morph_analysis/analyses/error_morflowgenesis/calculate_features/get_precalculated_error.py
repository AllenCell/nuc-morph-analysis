error_values = [
    {
        "error_type": "absolute",
        "percentile": "99%",
        "volume": 25.82,
        "height": 0.32,
        "surface_area": 8.45,
        "NUC_PC1": 0.45,
        "NUC_PC2": 0.7,
        "NUC_PC3": 0.33,
        "NUC_PC4": 0.62,
        "NUC_PC5": 0.76,
        "NUC_PC6": 0.32,
        "NUC_PC7": 0.35,
        "NUC_PC8": 0.43,
    },
    {
        "error_type": "percent",
        "percentile": "99%",
        "volume": 2.84,
        "height": 4.76,
        "surface_area": 1.55,
        "NUC_PC1": 84.83,
        "NUC_PC2": 291.61,
        "NUC_PC3": 239.66,
        "NUC_PC4": 199.65,
        "NUC_PC5": 363.19,
        "NUC_PC6": 912.32,
        "NUC_PC7": 1239.55,
        "NUC_PC8": 1252.81,
    },
    {
        "error_type": "percent_of_population_variation",
        "percentile": "99%",
        "volume": 0.0,
        "height": 1.12,
        "surface_area": 0.03,
        "NUC_PC1": 2.6,
        "NUC_PC2": 6.48,
        "NUC_PC3": 4.97,
        "NUC_PC4": 5.45,
        "NUC_PC5": 6.22,
        "NUC_PC6": 13.14,
        "NUC_PC7": 12.0,
        "NUC_PC8": 13.39,
    },
]


def get_error_values_by_type(error_type):
    """
    Find the error values for a given error type.

    Parameters
    ----------
    name: String
        'absolute' or 'percent'
    Returns
    -------
    error: dict
        Ditionary with error info
    """
    for error in error_values:
        if error_type == error["error_type"]:
            return error
    raise ValueError(f"Error type {error_type} not found")


def get_precalculated_error_value(error_type, feature):
    """
    Get the precalculated error value for a given feature. The error is calculated as the absolute error or the percent error.

    Parameters
    -----
    error_type: string
        'absolute' or 'percent'
    feature: string
        The feature to get the error value for. The options are 'volume', 'surface_area', 'height',
        'NUC_PC1', 'NUC_PC2', 'NUC_PC3', 'NUC_PC4', 'NUC_PC5', 'NUC_PC6', 'NUC_PC7', 'NUC_PC8'.
    """
    error_values = get_error_values_by_type(error_type)
    return error_values[feature]
