import concurrent
import os

import imageio.v3 as iio
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as fm
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from nuc_morph_analysis.lib.preprocessing.load_data import (
    get_dataset_pixel_size,
    get_seg_fov_for_row,
)
from nuc_morph_analysis.lib.visualization.plotting_tools import (
    get_plot_labels_for_metric,
)

from tqdm import tqdm
import multiprocessing
import shutil

mpl.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"
mpl.use("agg")


def make_gif(input_dir, output_dir, label, fps=10):
    """
    Create/save gif from frames of colony colored by a feature.

    Parameters
    ----------
    varname: String
        Name of feature to create gif of
    save_dir: String
        Name of the base directory to save figures to
    duration: float
        how fast (length of time) to play the frames
    """
    save_filepath = f"{output_dir}/{label}_{fps}fps.gif"
    num_frames = len(os.listdir(input_dir))
    if num_frames == 0:
        print(f"No compatible images found in {input_dir}")
        return
    with iio.imopen(save_filepath, "w", extension=".gif") as writer:
        for filename in sorted(os.listdir(input_dir)):
            if "png" in filename:
                writer.write(iio.imread(f"{input_dir}/{filename}"), duration=(1000 * 1 / fps))


def pad_to_even(frame):
    """
    Make sure a frame has even width/height by padding with zeros

    Parameters
    ----------
    frame: ndarray
        Input shape is (x,y,n)

    Returns
    -------
    new_frame: ndarray
      Output shape is (x,y,n), (x+1,y,n), (x,y+1,n), or (x+1,y+1,n)
    """
    bottom_padding = frame.shape[0] % 2 == 1
    right_padding = frame.shape[1] % 2 == 1
    return np.pad(frame, ((0, bottom_padding), (0, right_padding), (0, 0)))


def make_mp4(input_dir, output_dir, label, fps=10):
    """
    Create/save mp4 from frames of colony colored by a feature.

    Parameters
    ----------
    varname: String
        Name of feature to create mp4 of
    save_dir: String
        Name of the base directory to save figures to
    fps: float
        how fast (frames per second) to play the frames
    """
    save_filepath = f"{output_dir}/{label}_{fps}fps.mp4"
    num_frames = len(os.listdir(input_dir))
    if num_frames == 0:
        print(f"No compatible images found in {input_dir}")
        return
    with iio.imopen(save_filepath, "w", extension=".mp4", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=fps)
        for filename in sorted(os.listdir(input_dir)):
            if "png" in filename:
                # Many of our pngs are in a 4-channel RGBA format, but the mp4 writer needs
                # 3-channel RGB
                frame = iio.imread(f"{input_dir}/{filename}", mode="RGB")
                writer.write_frame(pad_to_even(frame))


def create_movie_filestructure(base_figdir, varnames, combined_colormap=False):
    """
    This function creates a series of directories and subdirectories
    for figures generated in this analysis to be save to, in a
    pre-determined organized way. It first checks to see if each
    subdirectory exists, then creates it if it does not.

    Structure of folder where images are saved:
    base_figdir/(combined_colormap)/var_name/frame_images/

    Structure of folder where stitched movies are saved:
    base_figdir/(combined_colormap)/var_name/movies/

    Parameters
    ----------
    varnames: List of strings
        List of names of features for which to create movies

    figdir: Path
        Path to figure directory for this analysis

    Returns
    -------
    dir: String
        Name of the base directory to save images and movies to.
    """

    if combined_colormap:
        base_figdir = base_figdir / "combined_colormap"
        base_figdir.mkdir(parents=True, exist_ok=True)

    for var in varnames:
        var_dir = base_figdir / var
        var_dir.mkdir(parents=True, exist_ok=True)

        for nested_dir in ["frame_images", "movies"]:
            new_dir = var_dir / nested_dir
            new_dir.mkdir(parents=True, exist_ok=True)

    return base_figdir


def get_colorized_img(df, varname, scale, filter_vals=True):
    """
    This function takes the max projection image for each frame and
    colorizes nuclei by spectrum of values for this feature

    Parameters
    ----------
    df: Dataframe
        One colony at one time point

    varname: String
        Name of feature to colorize frame by

    scale: float
        Multiplier to rescale values to real units

    filter_vals: bool
        Filter outliers and edge cells

    Returns
    -------
    img: image as array
        Array giving color values at each pixel of this frame
    """
    fov = get_seg_fov_for_row(df.iloc[0]).squeeze().max(axis=0)
    img = np.full_like(fov, np.nan, dtype=np.float32)
    for index, row in df.iterrows():
        label = int(row.label_img)
        if filter_vals and (
            np.isnan(df.at[index, varname])
            or df.at[index, "is_outlier"]
            # or df.at[index, "fov_edge"]
        ):
            df.at[index, varname] = -9999999
        img[fov == label] = df.at[index, varname] * scale
    return img


def process_this_tp(params):
    """
    Creates entire movie image for a single frame including colorbar
    (optional) and makes image from colorized array.

    Parameters
    ---------
    params: Dictionary
        Object consolidating all information about a frame.
        This includes the dataframe for just this FOV, the timepoint
        to be used for the elapsed time printout, the name of the
        variable to colorize by, where to sace the frames, and the
        appropriate colorbar for this feature.
    """

    df_fov = params["df_fov"]
    tp = params["tp"]
    varname = params["var"]
    colorbar_vals = params["colorbar_vals"]
    categorical = params["categorical"]
    savedir = params["savedir"]
    colorbar = params["colorbar"]
    pix_size = params["pix_size"]
    lut = params["lut"]
    cmap_name = params["cmap"]
    save_format = params.get("save_format", "png")

    scale, title, units, _ = get_plot_labels_for_metric(varname)

    img = get_colorized_img(df_fov, varname, scale)

    fig, ax = plt.subplots(1, 1, dpi=300)
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    colorbar_vals = colorbar_vals * scale
    vals_nonnan = colorbar_vals[~np.isnan(colorbar_vals)]

    if categorical:
        category_vals = np.unique(vals_nonnan)
        num_categories = len(category_vals)
        base_cmap = plt.get_cmap(cmap_name if cmap_name else "tab10")
        colors = [base_cmap(i) for i in range(num_categories)]
        cmap = ListedColormap(colors)
        vmin, vmax = np.nanmin(category_vals), np.nanmax(category_vals)
    else:
        cmap = mpl.cm.get_cmap(cmap_name if cmap_name else "cool")
        vmin, vmax = np.percentile(vals_nonnan, [1, 99])

    cmap.set_under("gainsboro")
    cmap.set_over("gainsboro")
    pos = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

    # set colorbar params, if using
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cax.axis("off")
    cbar = fig.colorbar(pos, ax=cax)
    cbar.set_label(f"{title} {units}")
    cbar.ax.tick_params(labelsize=14)

    if categorical:
        tick_locs = (
            (np.arange(num_categories) + 0.5) * (num_categories - 1) / num_categories
        ) + vmin
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels([lut[val] for val in category_vals])

    if not colorbar:
        cbar.remove()  # need to do it this way to get the right color scheme

    # remove whitespace around figure
    plt.subplots_adjust(top=1, bottom=0, right=0.85, left=0, hspace=0, wspace=0.1)

    # create timestamp
    t_min = (tp * 5) % 60
    t_hr = int(np.floor((tp * 5) / 60))
    ax.annotate(
        f"{t_hr} hr: {t_min} min",
        fontsize=14,
        xy=(0.05, 0.04),
        xycoords="figure fraction",
    )

    # create scalebar
    scalebar = AnchoredSizeBar(
        ax.transData,
        50 / pix_size,
        r"50 $\mu m$",
        "lower right",
        bbox_to_anchor=(0.95, -0.125),
        bbox_transform=ax.transAxes,
        color="black",
        pad=0,
        borderpad=0,
        frameon=False,
        size_vertical=30,
        label_top=False,
        fontproperties=fm.FontProperties(size=14),
    )
    ax.add_artist(scalebar)

    plt.savefig(
        savedir / varname / f"frame_images/colony_{tp:03d}.{save_format}",
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor="w",
        format=save_format,
        transparent=True,
    )
    fig.clf()
    plt.close(fig)


def is_categorical(varname):
    return varname in ["colony_position_group", "colony_depth", "family_id"]


def create_colorized_fov_movie(
    df,
    savedir,
    var,
    pix_size=get_dataset_pixel_size("all_baseline"),
    colorbar=True,
    categorical=None,
    df_all=None,
    parallel_flag=False,
    cmap=None,
    save_type="both",
    cleanup_images=False,
    save_format="png",
):
    """
    Cycles through all timepoints of a dataset in parallel
    to create the colorized frame images for all frames

    Parameters
    ----------
    df: Dataframe
        Full dataset
    savedir: Path
        Path to the base directory to save figures to.
    colorbar: bool
        whether to create colorbar
    parallel_flag: bool
        Flag of whether to run timepoints in parallel
    var: String
        Name of variable by which to colorize movie frames
    df_all: Dataframe
        Dataset from all colonies
    """
    mpl.use("agg")
    tps = sorted(df.index_sequence.unique())

    if categorical is None:
        categorical = is_categorical(var)

    print(f"Creating images for variable: {var}")

    if df_all is None:
        df_all = df.copy()
    else:
        df_all = df_all.copy()

    if categorical:
        if f"{var}_original" not in df_all.columns:
            df_all[f"{var}_original"] = df_all[var]
        else:
            df_all[var] = df_all[f"{var}_original"]
        category_labels = np.sort(df_all[f"{var}_original"].dropna().unique())
        lut = dict(zip(category_labels, range(1, len(category_labels) + 1)))
        inv_lut = {value: key for key, value in lut.items()}
        df_all[var] = df_all[var].map(lut)
        df[var] = df[var].map(lut)

    # run plotting
    params = [
        {
            "df_fov": df.loc[df.index_sequence == tp],
            "tp": tp,
            "var": var,
            "categorical": categorical,
            "colorbar_vals": df_all[var].values,
            "savedir": savedir,
            "lut": inv_lut if categorical else None,
            "colorbar": colorbar,
            "pix_size": pix_size,
            "cmap": cmap,
            "save_format": save_format,
        }
        for tp in tps
    ]

    if parallel_flag:
        num_processes = int(np.floor(0.8 * multiprocessing.cpu_count()))
        with concurrent.futures.ProcessPoolExecutor(num_processes) as executor:
            # tqdm(executor.map(process_this_tp, params), total=len(params))
            executor.map(process_this_tp, params)
    else:
        for param in tqdm(params):
            process_this_tp(param)

    input_dir = f"{savedir}/{var}/frame_images/"
    output_dir = f"{savedir}/{var}/movies/"
    if save_type == "gif" or save_type == "both":
        make_gif(input_dir=input_dir, output_dir=output_dir, label=var)
    if save_type == "mp4" or save_type == "both":
        make_mp4(input_dir=input_dir, output_dir=output_dir, label=var)

    if cleanup_images:
        print(f"Cleaning up images in {input_dir}")
        shutil.rmtree(input_dir)
