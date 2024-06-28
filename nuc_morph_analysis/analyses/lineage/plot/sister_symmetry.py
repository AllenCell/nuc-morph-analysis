import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.analyses.lineage.plot.single_generation import get_kl_divergence
import numpy as np
import scipy.stats as stats
import seaborn as sb


def visualize_volume_difference_for_division_symmetry_thresholds(df_pairs, figdir):
    """
    Visualize the volume difference at B for the sister pairs that come from symmetric, similar and asymmetric division events.

    Parameters
    ----------
    same_size_at_B: List
        List of volume differences at B for sister pairs that come from symmetric division events
    similar_size_at_B: List
        List of volume differences at B for sister pairs that come from similar division events
    different_size_at_B: List
        List of volume differences at B for sister pairs that come from asymmetric division events
    """
    df_symmetric = df_pairs[df_pairs.symmetric]
    df_similar = df_pairs[df_pairs.similar]
    df_aysmmetric = df_pairs[df_pairs.asymmetric]

    same_size_at_B = df_symmetric.difference_at_B.values
    similar_size_at_B = df_similar.difference_at_B.values
    different_size_at_B = df_aysmmetric.difference_at_B.values

    set_bins = np.arange(0, 140, 2)
    plt.hist(same_size_at_B, bins=set_bins, color="#7600bf")
    plt.hist(similar_size_at_B, bins=set_bins, color="grey", alpha=0.5)
    plt.hist(different_size_at_B, bins=set_bins, color="tab:green")
    plt.xlabel("Volume difference at B ($µm^3$)", fontsize=14)
    plt.ylabel(
        f"Number of pairs (N={len(same_size_at_B)+len(similar_size_at_B)+len(different_size_at_B)})",
        fontsize=14,
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(-5, 140)
    plt.legend(
        [
            f"Same size at B, N={len(same_size_at_B)}",
            f"Similar size at B, N={len(similar_size_at_B)}",
            f"Different size at B, N={len(different_size_at_B)}",
        ]
    )
    save_and_show_plot(f"{figdir}volume_difference_at_B")


def timing_of_asymmetric_and_symmetric_divisions(df_pairs, figdir):
    """
    Visualize the time at B for the sister pairs that come from symmetric and asymmetric division events.

    Parameters
    ----------
    same_size_at_B: List
        List of time at B for sister pairs that come from symmetric division events
    different_size_at_B: List
        List of time at B for sister pairs that come from asymmetric division events
    """
    df_symmetric = df_pairs[df_pairs.symmetric]
    df_aysmmetric = df_pairs[df_pairs.asymmetric]

    same_time_at_B = df_symmetric.time_at_B_min.values
    different_time_at_B = df_aysmmetric.time_at_B_min.values

    set_bins = np.arange(0, 2000, 25)
    plt.hist(same_time_at_B, bins=set_bins, alpha=0.5, color="#7600bf")
    plt.hist(different_time_at_B, bins=set_bins, alpha=0.5, color="tab:green")
    plt.xlabel("Time at B (min)", fontsize=14)
    plt.ylabel(f"Number of pairs (N={len(same_time_at_B)+len(different_time_at_B)})", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(
        [
            f"Same time at B, N={len(same_time_at_B)}",
            f"Different time at B, N={len(different_time_at_B)}",
        ]
    )
    save_and_show_plot(f"{figdir}time_at_B_for_sym_and_asym_dividers")


def plot_sister_trajectories(df_full, df_pairs, interval, pixel_size, figdir):
    """
    Plot the trajectories of the sister pairs that come from symmetric and asymmetric division events.

    Parameters
    ----------
    df_full: Dataframe
        The dataset dataframe with single track features
    sister_pairs: List
        List of sister pairs that come from symmetric and asymmetric division events
    """
    for index, row in df_pairs.iterrows():
        tid1 = row["tid1"]
        tid2 = row["tid2"]
        dft1 = df_full[df_full.track_id == tid1]
        dft2 = df_full[df_full.track_id == tid2]
        plt.plot(
            dft1.index_sequence * interval,
            dft1.volume * pixel_size**3,
            color="tab:purple",
            label=f"Small sister {tid1}",
        )
        plt.plot(
            dft2.index_sequence * interval,
            dft2.volume * pixel_size**3,
            color="tab:orange",
            label=f"Big sister {tid2}",
        )
        plt.axvline(
            x=dft1.time_at_B.values[0] * 60,
            color="tab:purple",
            linestyle="--",
            alpha=0.5,
            label="Time at B",
        )
        plt.axvline(
            x=dft2.time_at_B.values[0] * 60,
            color="tab:orange",
            linestyle="--",
            alpha=0.5,
            label="Time at B",
        )
        plt.xlabel("Time (min)")
        plt.ylabel("Volume ($µm^3$)")
        plt.legend()
        save_and_show_plot(f"{figdir}{tid1}_{tid2}")


def plot_mother_volume_distribution(track_level_feature_df, df_pairs, figdir):
    """
    Plot the distribution of the mother's volume at C for the sister pairs that come from symmetric and asymmetric division events.

    Parameters
    ----------
    track_level_feature_df: Dataframe
        The dataset dataframe with track level features
    symmetric_sister_pairs: List
        List of sister pairs that come from symmetric division events
    asymmetric_sister_pairs: List
        List of sister pairs that come from asymmetric division events
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    df_symmetric = df_pairs[df_pairs.symmetric]
    df_aysmmetric = df_pairs[df_pairs.asymmetric]

    symmetric_mother_df = track_level_feature_df[
        track_level_feature_df["track_id"].isin(df_symmetric.pid.values)
    ]
    asymmetric_mother_df = track_level_feature_df[
        track_level_feature_df["track_id"].isin(df_aysmmetric.pid.values)
    ]
    # label0 = f"Small and Medium dataset, N={len(track_level_feature_df)}, Mean={np.mean(track_level_feature_df['volume_at_C']):.2f},  COV={(np.std(track_level_feature_df['volume_at_C']) / np.mean(track_level_feature_df['volume_at_C'])):.2f}"
    label1 = f"Divides symmetrically, N={len(symmetric_mother_df)}, Mean={np.mean(symmetric_mother_df['volume_at_C']):.2f},  COV={(np.std(symmetric_mother_df['volume_at_C']) / np.mean(symmetric_mother_df['volume_at_C'])):.2f}"
    label2 = f"Divides asymmetrically, N={len(asymmetric_mother_df)}, Mean={np.mean(asymmetric_mother_df['volume_at_C']):.2f},  COV={(np.std(asymmetric_mother_df['volume_at_C']) / np.mean(asymmetric_mother_df['volume_at_C'])):.2f}"
    # ax = sb.kdeplot(track_level_feature_df['volume_at_C'], color='black', label=label0)
    ax = sb.kdeplot(symmetric_mother_df["volume_at_C"], color="#7600bf", label=label1)
    ax = sb.kdeplot(asymmetric_mother_df["volume_at_C"], color="tab:green", label=label2)

    plt.xlabel("Volume at C ($µm^3$)")
    plt.ylabel("Count")
    plt.legend(bbox_to_anchor=(0.5, 1.1), loc="center")
    plt.tight_layout()
    save_and_show_plot(f"{figdir}mother_volume_at_C")


def bootstrap_confidence_cov(df, feature, n_bootstraps=500):
    """
    Calculate the confidence interval for the coefficient of variation of a feature using bootstrapping.

    Parameters
    ----------
    df: Dataframe
        The dataset dataframe with track level features
    feature: String
        The feature to calculate the coefficient of variation for
    n_bootstraps: Int
        The number of bootstraps to perform
    """
    covs = []

    for i in range(n_bootstraps):
        np.random.seed(i)
        sample = df[feature].sample(frac=1, replace=True)
        cov = np.std(sample) / np.mean(sample)
        covs.append(cov)
    return np.percentile(covs, 5), np.percentile(covs, 95)


def feature_density(df_all, df_symmetric, df_asymmetric, feature, figdir, add_kl_divergence=False):
    """
    Plot feature density for symmetric and asymmetric dividers. Option to show that relative to the full datasets.

    Parameters
    ----------
    df_all: Dataframe
        The dataset dataframe with all track level features
    df_symmetric: Dataframe
        The dataset dataframe with track level features for symmetric dividers
    df_asymmetric: Dataframe
        The dataset dataframe with track level features for asymmetric dividers
    feature: String
        The feature to plot
    figdir: String
        The directory to save the figure
    add_kl_divergence: Boolean
        Whether to add the KL divergence to the plot
    """
    fig = plt.figure(figsize=(6, 6))

    def calc_stats(df, feature):
        mean = np.mean(df[feature])
        cov = np.std(df[feature]) / mean
        low, high = bootstrap_confidence_cov(df, feature)
        return mean, cov, low, high

    dataframes = [
        (df_all, "black", "Small and Medium Datasets"),
        (df_symmetric, "#7600bf", "Symmetric"),
        (df_asymmetric, "tab:green", "Asymmetric"),
    ]

    for df, color, label_prefix in dataframes:
        mean, cov, low, high = calc_stats(df, feature)
        label = f"{label_prefix}, N={len(df)}, Mean={mean:.2f}, COV={cov:.2f} CI=({low:.2f}, {high:.2f})"
        ax = sb.kdeplot(df[feature], color=color, label=label)
        _, label, units, lim = get_plot_labels_for_metric(feature)
        ax.set_xlabel(f"{label} {units}")
        ax.set_ylabel("Density")

    plt.legend(bbox_to_anchor=(0.5, 1.1), loc="center")

    if add_kl_divergence:
        kl12, kl21, average_kl = get_kl_divergence(df_symmetric[feature], df_asymmetric[feature])
        plt.text(
            0.98,
            0.98,
            f"KL divergence sym->asym: {kl12:.2f}\nKL divergence asym->sym: {kl21:.2f}\nAvg KL divergence: {average_kl:.2f}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=plt.gca().transAxes,
        )

    save_and_show_plot(f"{figdir}{feature}_density_plot", figure=fig, bbox_inches="tight")
