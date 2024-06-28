import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot


def check_measurement_vs_error(df, pixel_size, error_type, measurement_type):
    """
    Check to see if the amount of error in our segmentations and therefore downstream measurements
    has a dependence on volume, height, or surface area.

    Parameters
    ----------
    df : pandas dataframe
        fixed_control dataframe with error columns added
    pixel_size : float
        pixel size returned from function load_data.get_dataset_pixel_size('fixed_control')
    error_type : str
        'percent' or 'absolute'
    measurement_type : str
        'volume', 'height', or 'sa'

    Returns
    -------
    scatter plot of median measurement vs maximum error per track
    """
    if measurement_type == "volume":
        median_measurements = df.groupby("track_id").volume.median() * pixel_size**3
        error_column = f"{error_type}_error_volume"
    elif measurement_type == "height":
        median_measurements = df.groupby("track_id").height.median() * pixel_size
        error_column = f"{error_type}_error_height"
    elif measurement_type == "surface_area":
        median_measurements = df.groupby("track_id").mesh_sa.median() * pixel_size**2
        error_column = f"{error_type}_error_surface_area"
    else:
        print("Invalid measurement type")
        return

    error_values = df.groupby("track_id")[error_column].median().to_numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(median_measurements, error_values, alpha=0.5, edgecolors="none")
    plt.xlabel(f"Median {measurement_type} per track (µm³)", fontsize=20)
    plt.ylabel(f"Median {error_type} error per track (µm³)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    save_and_show_plot(f"error_morflowgenesis/figures/check_measurement_vs_error_{error_column}")
