import matplotlib.pyplot as plt
import scipy
from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot

fs = 14


def full_track_with_error_bounds(df_full, colony, error_value, error_type, n=5, feature="volume"):
    """
    Add error bars to the full track plots for the baseline colony dataset.

    Parameters
    ----------
    df_full: Dataframe
        baseline colony manifest filtered to full tracks
    colony: string
        "medium"
    n: int
        How many tracks to plot
    feature: string
        'volume','height'

    Returns
    -------
    Plot of full tracks with error bars
    """

    pixel_size = load_data.get_dataset_pixel_size("all_baseline")
    interval = load_data.get_dataset_time_interval_in_min("all_baseline")

    df_full = df_full[df_full["colony"] == colony]

    tid_list = df_full.track_id.unique()
    tid_list = tid_list[:n]
    df_full = df_full[df_full["track_id"].isin(tid_list)]

    if feature == "volume":
        column_name = "volume"
        scale = pixel_size * pixel_size * pixel_size
        y_label = "Volume ($um ^3$)"
        ylim = 1800
    if feature == "height":
        column_name = "height"
        scale = pixel_size
        y_label = "Height ($um$)"
        ylim = 12

    for tid, dft in df_full.groupby("track_id"):
        x = dft.index_sequence.to_numpy() * interval
        y = dft[column_name].to_numpy() * scale
        rolling_mean = scipy.signal.medfilt(y, 7)
        plt.plot(x - x.min(), y, label=f"track_id: {tid}")
        if error_type == "absolute":
            plt.fill_between(
                x - x.min(),
                rolling_mean - error_value,
                rolling_mean + error_value,
                alpha=0.25,
                label=f"error: ±{error_value}",
                edgecolor="none",
            )
        if error_type == "percent":
            rolling_error = rolling_mean * error_value / 100
            plt.fill_between(
                x - x.min(),
                rolling_mean - rolling_error,
                rolling_mean + rolling_error,
                alpha=0.25,
                label=f"error: ±{error_value}",
                edgecolor="none",
            )
        plt.ylabel(y_label, fontsize=fs)
        plt.xlabel("Time (minutes)", fontsize=fs)
        plt.legend()
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.ylim(0, ylim)
        plt.tight_layout()
        save_and_show_plot(
            f"error_morflowgenesis/figures/full_track_with_error_bounds_{colony}_{feature}_{error_type}_{tid}"
        )
