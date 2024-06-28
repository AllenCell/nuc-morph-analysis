import matplotlib.pyplot as plt
import numpy as np
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.example_tracks import EXAMPLE_TRACKS
import seaborn as sb
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"

ALPHA_TRACKS = 0.035
TIME_BIN = 0.02


def plot_all_tracks_synchronized(
    df,
    figdir,
    feature="volume",
    time="sync_time_Ff",
    highlight_samples=True,
    label_samples=False,
    alpha=ALPHA_TRACKS,
):
    """
    This function plots all tracks in real time and real volume,
    synchronized to start at time t=0. It then creates a second version
    with two sample tracks with different growth rates and durations
    highlighted in two colors (green and orange)

    Parameters
    ----------
    df: Dataframe
        Dataframe containing full-track data from all colony datasets
    figdir: path
        Path to directory to save all figures for this script
    feature: string
        volume or surface area column name
    time: string
        Column name for times to use
    highlight_samples: bool
        Flag to highlight sample tracks
    label_samples: bool
        Flag to label sample tracks and give unique colors
    tid1: int
        Track ID for one track to highlight
    tid2: int
        Track ID for a second track to highlight
    alpha: float
        Transparency of lines for plotting tracks
    """

    figlabel = f"{figdir}/all_tracks_{feature}_{time}"

    yscale, ylabel, yunits, ylim = get_plot_labels_for_metric(feature)
    xscale, xlabel, xunits, xlim = get_plot_labels_for_metric(time)

    _, ax = plt.subplots(dpi=300)
    for track_id, df_track in df.groupby("track_id"):
        df_track = df_track.sort_values("index_sequence")
        mask = (df_track["index_sequence"] >= df_track["Ff"]) & (
            df_track["index_sequence"] <= df_track["Fb"]
        )
        df_track = df_track.loc[mask]
        x = df_track[time] * xscale
        y = df_track[feature].values * yscale
        ax.plot(x, y, c="#808080", alpha=alpha)
    plt.xlabel(f"{xlabel} {xunits}")
    plt.ylabel(f"{ylabel} {yunits}")
    plt.xlim(xlim)
    plt.ylim(ylim)

    if highlight_samples:
        track_ids = EXAMPLE_TRACKS["sample_full_trajectories"]
        figlabel += "_highlight_samples"
        if label_samples:
            figlabel += "_labeled"
        for track_id in track_ids:
            df_track = df.loc[df["track_id"] == track_id]
            mask = (df_track["index_sequence"] >= df_track["Ff"]) & (
                df_track["index_sequence"] <= df_track["Fb"]
            )
            df_track = df_track.loc[mask]
            x = df_track[time] * xscale
            y = df_track[feature].values * yscale
            if label_samples:
                ax.plot(x, y, lw=2, label=f"Track {track_id}")
            else:
                ax.plot(x, y, color="k", lw=2)
        plt.xlabel(f"{xlabel} {xunits}")
        plt.ylabel(f"{ylabel} {yunits}")
        plt.xlim(xlim)
        plt.ylim(ylim)

    if highlight_samples and label_samples:
        plt.legend()

    plt.tight_layout()
    save_and_show_plot(figlabel)


def plot_mean_track_with_ci(
    df_all, figdir, feature="volume", by_colony_flag=False, percentile_flag=True
):
    """
    This function plots the mean volume (in real units) for
    all tracks over normalized time, with the standard
    deviation shaded around it.

    Parameters
    ----------
    df_all: Dataframe
        Dataframe containing full-track data from all colony datasets
    figdir: path
        Path to directory to save all figures for this script
    feature: string
        volume or surface area column name
    by_colony_flag: bool
        Flag for whether to separate data out by colony
    percentile_flag: bool
        Flag for whether to plot standard deviation
    """

    def percentile(n):
        def percentile_(x):
            return x.quantile(n)

        percentile_.__name__ = "percentile_{:02.0f}".format(n * 100)
        return percentile_

    yscale, ylabel, yunits, ylim = get_plot_labels_for_metric(feature)
    _, xlabel, _, xlim = get_plot_labels_for_metric("normalized_time")

    figlabel = f"{figdir}/{feature}_mean_track"

    if by_colony_flag:
        colors = ["#1B9E77", "#D95F02", "#7570B3"]
        colonies = ["small", "medium", "large"]
        labels = ["Small Mean", "Medium Mean", "Large Mean"]
        labels2 = [
            "Small 5th to 95 Percentiles",
            "Medium 5th to 95 Percentiles",
            "Large 5th to 95 Percentiles",
        ]
        figlabel += "_bycolony"
    else:
        colors = "k"
        colonies = ["all_baseline"]
        labels = ["Mean"]
        labels2 = ["5th to 95 Percentiles"]

    if percentile_flag:
        figlabel += "_percentile"

    for color, dataset, label, label2 in zip(colors, colonies, labels, labels2):

        if by_colony_flag:
            df = df_all[df_all["colony"] == dataset]
        else:
            df = df_all

        # Remove tracks shorter than digitized time threshold
        df_agg = df.loc[df["normalized_time"] <= 1].dropna(subset=["normalized_time"])

        # Create binned/digitized time
        timedig_bins = np.arange(0, 1 + TIME_BIN, TIME_BIN)
        inds = np.digitize(df_agg["normalized_time"], timedig_bins)
        df_agg["dig_time"] = timedig_bins[inds - 1]
        df_agg_pop = (
            df_agg[[feature, "dig_time"]]
            .groupby(["dig_time"])
            .agg(["mean", "median", "std", percentile(0.05), percentile(0.95)])
        )

        # Plot coeff of variation with std and sem shaded in bottom right panel
        plt.plot(
            df_agg_pop.index,
            df_agg_pop[feature, "mean"] * yscale,
            "-",
            alpha=1,
            color=color,
            label=label,
        )
        if percentile_flag:
            ysup = df_agg_pop[feature, "percentile_05"] * yscale
            yinf = df_agg_pop[feature, "percentile_95"] * yscale

            plt.fill_between(
                df_agg_pop.index,
                y1=ysup,
                y2=yinf,
                alpha=0.2,
                color=color,
                label=label2,
                edgecolor="none",
            )

    if "fold" in feature:
        plt.axhline(1, color="k", linestyle=":")
        plt.axhline(2, color="k", linestyle=":")
        ylim = (0.5, 2.5)

    plt.xlabel(xlabel)
    plt.ylabel(f"Average {ylabel} {yunits}")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.tight_layout()
    save_and_show_plot(figlabel)


def bc_fold_change_distribution(
    df_all,
    figdir,
    feature="volume_fold_change_BC",
    by_colony_flag=False,
    density_flag=True,
    nbins=20,
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
    feature: string
        volume or surface area column name
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

    for color, dataset, label, ind in zip(colors, colonies, labels, [0, 2, 4, 6]):

        if by_colony_flag and dataset != "all_baseline":
            df = df_all[df_all["colony"] == dataset]
        else:
            df = df_all

        if density_flag:
            # get kde and find foldchange giving max to add to legend
            ax = sb.kdeplot(df[feature], color=color)
            x = ax.lines[ind].get_xdata()  # Get the x data of the distribution
            y = ax.lines[ind].get_ydata()  # Get the y data of the distribution
            maxid = np.argmax(y)  # The id of the peak (maximum of y data)
            label += f" ({np.round(x[maxid],2)})"
            # replot kde now with complete label
            ax = sb.kdeplot(df[feature], color=color, label=label)

        else:
            plt.hist(df[feature], histtype="step", bins=nbins, color=color, label=label)

    plt.axvline(2, color="k", linestyle=":")
    plt.xlabel(f"{xlabel} {xunits}")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    save_and_show_plot(figlabel)
