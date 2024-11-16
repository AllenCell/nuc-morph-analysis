import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.optimize import curve_fit
from nuc_morph_analysis.lib.preprocessing.curve_fitting import powerfunc
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric


def add_early_growth_rate(df, interval, flag_dropna=False):
    """
    This function calculated the growth rate during the earlier growth
    period, taking place from the frame of formation to the frame of
    inflection. This is a single value for each track. It adds a column to
    the input dataframe with this value for all df entries corresponding
    to this track.

    Parameters
    ----------
    df: Dataframe
        Input dataset of full tracks

    pix_size: float
        Size of pixels in microns for this dataset

    interval: int
        Time interval between frames in minutes

    flag_dropna: bool
        Flag to drop nan growth rates

    Returns
    -------
    Dataframe
        Dataset with growth rate A-B in minutes per hour for each trajectory added
    """

    count = 0
    for tid, df_track in df.groupby("track_id"):
        diff = df_track["volume_at_B"].iloc[0] - df_track["volume_at_A"].iloc[0]
        duration = df_track["duration_AB"].iloc[0] * interval / 60
        gr = diff / duration

        df.loc[df.track_id == tid, "growth_rate_AB"] = gr
        if np.isnan(gr):
            count += 1

    if flag_dropna:
        df = df.dropna(subset="growth_rate_AB")
    if count > 0:
        print(f"Failed early growth rate count: {count}")
    return df


def fit_tracks_to_time_powerlaw(
    df,
    feature_col,
    interval,
    plot=False,
):
    """

    Parameters
    ----------
    df : DataFrame
        full tracks dataframe
    feature_col: str
        Name of column with feature to fit
    interval : float
        The time interval in minutes
    plot : bool, optional
        If True, a plot of the volume vs time for each track and its exponential fit is displayed. The default is False.

    Returns
    -------
    df : Dataframe
        The input dataframe with an additional columns with linearity fit parameters added
    """

    # get parameters based on fitting volume or SA
    if feature_col == "volume":
        short = "volume"
        atB_0 = 550
        rate_0 = 35
        tscale_0 = 1
    elif feature_col == "mesh_sa":
        short = "SA"
        atB_0 = 400
        rate_0 = 15
        tscale_0 = 1
    else:
        raise ValueError(
            "Function currently only designed to work \
                        with volume and mesh_sa.\
                        To fit antoher column, update this function."
        )
    trackdir = f"volume/figures/linearity/{short}/tracks/"
    yscale, ylabel, yunits, _ = get_plot_labels_for_metric(feature_col)

    features = [
        f"tscale_linearityfit_{short}",
        f"atB_linearityfit_{short}",
        f"rate_linearityfit_{short}",
        f"RMSE_linearityfit_{short}",
    ]
    for feature in features:
        df[feature] = np.nan

    fail_count = 0
    for track, df_track in df.groupby("track_id"):

        try:
            # trim tracks to transition to breakdown
            transition = df_track["frame_transition"].min()
            fb = df_track["Fb"].values.min()
            df_track = df_track.sort_values("index_sequence")
            df_track_trim = df_track[
                (df_track.index_sequence > transition) & (df_track.index_sequence <= fb)
            ]

            # get trimmed track times and volumes
            x = df_track_trim["index_sequence"].values * interval / 60
            x -= x[0]
            y = df_track_trim[feature_col].values * yscale

            # fit trimmed track to model with initial guesses
            popt, _ = curve_fit(powerfunc, x, y, [rate_0, atB_0, tscale_0])

            # get fits parameters, residuals and mse
            z = powerfunc(x, *popt)
            res = z - y
            rmse = np.sqrt(np.nanmean(res**2))
            rate = popt[0]
            atB = popt[1]
            tscale = popt[2]

            # plot real track and fitted model
            if plot is True:
                plt.plot(x, y, label=f"Track ID: {track}", c="grey", alpha=0.5)
                plt.plot(
                    x,
                    z,
                    c="red",
                    alpha=0.5,
                    label=f"tscale {np.round(tscale,2)}, atB {np.round(atB)}, "
                    f"r {np.round(rate)}, RMSE {np.round(rmse)}",
                )
                plt.legend()
                plt.ylabel(f"{ylabel} {yunits}")
                plt.xlabel("Time (hr)")
                plt.tight_layout()
                save_and_show_plot(f"{trackdir}/track{track}", quiet=True)
                plt.close()

            # add fit parameters and error to manifest
            df.loc[df_track.index, f"tscale_linearityfit_{short}"] = tscale
            df.loc[df_track.index, f"atB_linearityfit_{short}"] = atB
            df.loc[df_track.index, f"rate_linearityfit_{short}"] = rate
            df.loc[df_track.index, f"RMSE_linearityfit_{short}"] = rmse

        except Exception:
            fail_count += 1

    if fail_count > 0:
        print(f"Failed {short} linearity fit count: {fail_count}")

    return df


def plot_fit_parameter_distribution(
    df_all, figdir, feature, by_colony_flag=False, density_flag=True, nbins=20
):
    """
    This function plots the mean volume fold change for
    all tracks over normalized time, with the standard
    deviation shaded around it.

    Parameters
    ----------
    df_all: Dataframe
        Dataframe containing full-track data from all colony datasets
    figdir: path
        Path to directory to save all figures for this script
    by_colony_flag: bool
        Flag for whether to separate data out by colony
    density_flag: bool
        Flag for whether to use density instead of histogram
    nbins: int
        Number of bins for histogram
    """

    _, xlabel, xunits, _ = get_plot_labels_for_metric(feature)

    figlabel = f"{figdir}/{feature}_distribution"

    if by_colony_flag:
        colors = ["k", "#1B9E77", "#D95F02", "#7570B3"]
        colonies = ["all_baseline", "small", "medium", "large"]
        labels = ["All Colonies", "Small", "Medium", "Large"]
        figlabel += "_bycolony"
    else:
        colors = "k"
        colonies = ["all_baseline"]
        labels = ["All Colonies"]

    if density_flag:
        figlabel += "_density"
        ylabel = "Probability Density"
    else:
        figlabel += "_hist"
        ylabel = "Counts"

    _, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=300)

    for color, colony, label, ind in zip(colors, colonies, labels, [0, 2, 4, 6]):

        if by_colony_flag and colony != "all_baseline":
            df = df_all[df_all["colony"] == colony]
        else:
            df = df_all

        if density_flag:
            # get kde and find feature value giving max to add to legend
            sb.kdeplot(df[feature], color=color, ax=ax)
            x = ax.lines[ind].get_xdata()  # Get the x data of the distribution
            y = ax.lines[ind].get_ydata()  # Get the y data of the distribution
            maxid = np.argmax(y)  # The id of the peak (maximum of y data)
            label += f" ({np.round(x[maxid],2)})"
            # replot kde now with complete label
            sb.kdeplot(df[feature], color=color, label=label, ax=ax)

        else:
            plt.hist(df[feature], histtype="step", bins=nbins, color=color, label=label)

    if "tscale" in feature:
        plt.axvline(1, color="k", linestyle=":")
    if "RMSE" in feature:
        plt.axvline(5.49, color="k", linestyle=":", label=f"Error (5.49 {xunits[1:-1]})")

    plt.xlabel(f"{xlabel} {xunits}")
    plt.ylabel(ylabel)
    plt.legend(prop={"size": 12})
    plt.tight_layout()
    save_and_show_plot(figlabel)
    plt.close()


def add_late_growth_rate_by_endpoints(df):
    """
    This function adds the most basic version of a growth rate, given
    simply by the ratio of the volume change to the time interval from B to C

    Parameters
    ----------
    df : DataFrame
        full tracks dataframe

    Returns
    -------
    df : Dataframe
        The input dataframe with an additional column 'late_growth_rate_by_endpoints'
    """
    df["late_growth_rate_by_endpoints"] = df["delta_volume_BC"] / df["duration_BC"]
    return df
