import ast
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, load_data, filter_data
from nuc_morph_analysis.lib.preprocessing.neighbor_analysis import compute_density
from nuc_morph_analysis.lib.preprocessing import add_times
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric

# make plot text editable in Illustrator
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"


def toymodel(nc_ratio=0.29, cvol=745/(0.108**3), cell_H_mod=2, num_workers=1, old_unfiltered_method=False):
    """
    main function to run

    args
    --------
    nc_ratio: ratio of nuclear to cytoplastic volumes - default is 0.3
    cvol: target volume of nuclei to fit toy model to (default is 745, mean volume of cells in the middle of the cell cycle)
    num_workers: how many workers to use for multiprocessing
    old_unfiltered_method: whether to use the old (unfiltered) method that does not remove bad pseudo cells
    """
    save_path = Path(__file__).parent / "figures"
    save_path.mkdir(parents=True, exist_ok=True)
    pix_size = load_data.get_dataset_pixel_size("all_baseline")
    data, cvol = get_data(cvol, save_path, num_workers, old_unfiltered_method)
    cvol_um = cvol * pix_size**3
    cvol_um_cell = cvol_um * 1 / nc_ratio
    # rescale features
    data["height"] = data["height"] * pix_size
    data["dist"] = data["dist"] * pix_size

    stats = get_toy_model(data, cvol_um_cell, cell_H_mod)

    plot_toy_model(data, stats, save_path, old_unfiltered_method)
    plot_growth_rate(data, save_path, old_unfiltered_method)


def get_toy_model(data, cvol_um_cell, cell_H_mod=0):
    """
    Fit toy model to set of nuclei that all have similar volumes
    do this by interpolating height values, and then for
    every height calculating
    1. radius of a cylinder as R = sqrt(V/(2 pi H))
    2. radius of a hexagon as R = sqrt(2V/3sqrt(3)H)
    Return this information as distance = 2*R via a dataframe

    arguments
    -----
    data: dataframe with real data
    cvol_um_cell: volume of a single cell
    cell_H_mod: height difference between top of nucleus and top of cell to use for toy model. default is 0
    """
    H = np.linspace(data["height"].min(), data["height"].max(), 100)
    # cell_H = H + 2
    cell_H = H + cell_H_mod
    # Volume of cylinder is pi r^2 H
    # distance = np.sqrt(cvol/(np.pi * H))*2 for cylinder
    dist_cylindrical = 2 * np.sqrt(cvol_um_cell / (np.pi * cell_H))  # assuming cylindrical toy model

    # Volume of hexagon is (3√3/2)s^2 × h
    # s = sqrt(2V/3sqrt(3)H)
    S = np.sqrt(cvol_um_cell * 2 / (cell_H * 3 * np.sqrt(3)))
    # normal to hexagon side is S *sqrt(3)/2
    # distance is 2 * S * sqrt(3)/2
    dist_hexagonal = 2 * S * np.sqrt(3) / 2

    stats = pd.DataFrame(
        {
            "height": H,
            "cell_height": cell_H,
            "dist_cylindrical": dist_cylindrical,
            "dist_hexagonal": dist_hexagonal,
            "cell_vol": cvol_um_cell,
            "cell_height_diff": cell_H_mod,
        }
    )
    return stats


def plot_toy_model(data, toy_stats, save_path=Path("./"), old_unfiltered_method=False):
    """
    Plot real data and toy model curves
    For the real data, plot a gaussian weighted moving average

    arguments
    -----
    data: dataframe with real data
    toy_stats: dataframe with toy model fits
    save_path: path to save pdf
    old_unfiltered_method: whether to use the old (unfiltered) method that does not remove bad pseudo cells
    """
    data = data.sort_values(by="height")
    x = data["height"]
    y = data["dist"]

    toy_stats = toy_stats.sort_values(by="height")
    x2 = toy_stats["height"]
    y2_cylinder = toy_stats["dist_cylindrical"]
    y2_hexagon = toy_stats["dist_hexagonal"]

    logging.getLogger("matplotlib.font_manager").disabled = True
    fig, axes = plt.subplots(1, 1, figsize=(7, 5))

    bins, average, std = weighted_moving_average(x, y, width=0.2, step_size=0.5)
    lower_bound = average - std
    upper_bound = average + std
    axes.scatter(x, y, color="#808080", edgecolors="none")
    axes.set_xlabel("Height (µm)")
    axes.set_ylabel("Mean distance to neighbors (µm)")
    axes.plot(bins, average, label="Average from Data", c="#808080")
    axes.fill_between(bins, lower_bound, upper_bound, alpha=0.3, color="#808080", edgecolor="none")
    axes.scatter(
        x2,
        y2_hexagon,
        label="Hexagonal Packing Model",
        c="#e6ab02",
        edgecolors="none",
    )
    axes.set_ylim(10, 40)
    axes.legend()
    suffix = "unfiltered.pdf" if old_unfiltered_method else ".pdf"
    savename = f"toymodel-{toy_stats['cell_height_diff'].values[0]}-{toy_stats['cell_vol'].values[0]}{suffix}"
    fig.savefig(save_path / savename, bbox_inches="tight")


def plot_growth_rate(data, save_path=Path("./"), old_unfiltered_method=False):
    """
    Plot growth rate vs crowding
    crowding is calculated as a mean distance to neighbors
    old_unfiltered_method: whether to use the old (unfiltered) method that does not remove bad pseudo cells
    """
    data = data.sort_values(by="growth_rate")
    y = data["growth_rate"]
    x = data["dist"]

    logging.getLogger("matplotlib.font_manager").disabled = True
    fig, axes = plt.subplots(1, 1, figsize=(7, 5))

    bins, average, std = weighted_moving_average(x, y, width=1, step_size=0.5)
    lower_bound = average - std
    upper_bound = average + std
    axes.plot(x, y, ".")
    axes.set_ylabel("Late growth rate")
    axes.set_xlabel("Mean distance to neighbors (µm)")
    axes.plot(bins, average, label=f"gaussian weighted moving average", c="tab:blue")
    axes.fill_between(bins, lower_bound, upper_bound, alpha=0.3, edgecolor="none")
    axes.legend()
    savename = "growthrate_vs_crowding_unfiltered.pdf" if old_unfiltered_method else "growthrate_vs_crowding.pdf"
    fig.savefig(save_path / savename, bbox_inches="tight")


def weighted_moving_average(x, y, step_size=0.05, width=1):
    """
    Utility function to compute a gaussian weighted moving average
    """
    bin_centers = np.arange(np.min(x), np.max(x) - 0.5 * step_size, step_size) + 0.5 * step_size
    bin_avg = np.zeros(len(bin_centers))
    bin_std = np.zeros(len(bin_centers))

    # We're going to weight with a Gaussian function
    def gaussian(x, amp=1, mean=0, sigma=1):
        return amp * np.exp(-((x - mean) ** 2) / (2 * sigma**2))

    for index in range(0, len(bin_centers)):
        bin_center = bin_centers[index]
        weights = gaussian(x, mean=bin_center, sigma=width)
        bin_avg[index] = np.average(y, weights=weights)
        variance = np.average((y - bin_avg[index]) ** 2, weights=weights)
        bin_std[index] = math.sqrt(variance)

    return (bin_centers, bin_avg, bin_std)


def get_data(cvol, save_path=Path("./"), num_workers=1, old_unfiltered_method=False):
    """
    Load all datasets, remove outliers and edge cells, fitler to tracks > 120
    frames, select all nuclei that are close to a set volume (cvol), and compute
    average distance to all neighbors for each of these nuclei

    args
    -------
    cvol: volume of nuclei to select. default = 0.5e6
    save_path: path to save pdf
    num_workers: how many workers to use for multiprocessing
    """
    
    df = global_dataset_filtering.load_dataset_with_features(remove_growth_outliers=False)
    df = filter_data.filter_all_outliers(df)
    # df = filter_data.filter_out_cells_entering_or_exiting_mitosis(df)

    df_full = filter_data.all_timepoints_full_tracks(df)
    df_ft = df_full[df_full["colony"].isin(["small", "medium", "large"])].reset_index()

    scale, _, _, _ = get_plot_labels_for_metric("volume")
    # plot chosen volume
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for track, df_track in df_ft.groupby("track_id"):
        ax.plot(df_track.index_sequence, df_track.volume * scale, alpha=0.5, lw=0.3)
    ax.axhline(y=cvol * scale)
    fig.savefig(save_path / "chosen_volume.pdf")

    # get similar volumes to chosen volume
    df_ft["delta_vol"] = (df_ft.volume - cvol).abs()
    df_cvol = df_ft.loc[df_ft.groupby("track_id").delta_vol.idxmin()]
    df_cvol = df_cvol.loc[df_cvol.delta_vol < 1e3]

    # subset to relevant neighbor IDs
    neighbor_ids = list(df_cvol["neighbors"].apply(lambda x: ast.literal_eval(x)).values)
    neighbor_ids = [
        item for sublist in neighbor_ids for item in sublist if item in df.index.values
    ]

    # Reset index because compute_density expects a CellId column
    df_all = df.loc[df.index.isin(neighbor_ids)].reset_index()

    # compute distance/density metric
    neigh_stats = compute_density(df_cvol, df_all, num_workers, old_unfiltered_method)
    neigh_stats = neigh_stats.reset_index()

    return neigh_stats, cvol

def determine_volume_at_middle_of_cell_cycle():

    df = global_dataset_filtering.load_dataset_with_features(remove_growth_outliers=True)
    df = filter_data.filter_all_outliers(df)
    df_full = filter_data.all_timepoints_full_tracks(df)
    df_full = add_times.digitize_time_column(df_full,minval=0,maxval=1,step_size=0.05,time_col='normalized_time',new_col='dig_time')
    grouper = df_full[['volume','dig_time']].groupby('dig_time')
    dfg = grouper.mean()
    scale,_,_,_ = get_plot_labels_for_metric('volume')
    mean_vol = dfg.loc[0.5,'volume'].mean()*scale
    print(mean_vol)
    plt.plot(dfg.index,dfg['volume']*scale)
    plt.show()


if __name__ == "__main__":
    for cell_H_mod in [0, 1.5, 2]:
        toymodel(cell_H_mod=cell_H_mod)
        toymodel(cell_H_mod=cell_H_mod,old_unfiltered_method=True)
