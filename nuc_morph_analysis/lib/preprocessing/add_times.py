from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize as spopt
from scipy import interpolate as spinterp
from nuc_morph_analysis.lib.preprocessing.curve_fitting import hyperbola, hyperbola_jacobian, line
from nuc_morph_analysis.utilities.viz_utilities import plot_tracks_with_tps


def create_traj_fig_filestructure(figdir):
    """
    This function creates a series of directories and subdirectories
    for figures generated in this analysis to be save to, in a
    pre-determined organized way. It first checks to see if each
    subdirectory exists, then creates it if it does not.

    Parameters
    ----------
    figdir: Path
        Path to figure directory for this analysis

    Returns
    -------
    dir: Path
        Path to the base directory to save figures to
    """

    # Cycle through layers of nested subdirectories and create all
    traj_dir = figdir / "trajectories"
    for nested_dir in ["norm_times", "inflection"]:
        new_dir = traj_dir / nested_dir
        new_dir.mkdir(parents=True, exist_ok=True)
        if nested_dir == "inflection":
            for subdir in ["debug", "tracks"]:
                sub_dir = new_dir / subdir
                sub_dir.mkdir(parents=True, exist_ok=True)

    return traj_dir


def get_trajectory_inflection_frame(
    x_input, y_input, track, xbuffer=40, display=False, figdir=None, example_track_figure=False
):
    """
    This function gives the inflection frame for a given nuclear trajectory.
    Parameters
    ----------
    x_input: list of ints
        Frame numbers
    y_input: list of floats
        Nuclear volumes
    xbuffer: int
        Number of frames to be used in the calculation. This needs to be
        adjusted if the movie has time interval other than 5min. Ideally
        something that captures around 3.3hrs after formation.
    display: bool or matplotlib axis
        If False no plot is shown. If True then shows a new plot for current
        fitting. If a matplotlib axis is shown the plot is shown in that axis.
    figdir: Path
        Path to save figures
    track: int
        track id
    example_track_figure:
        return ax, fig for supplemental figure on transition calculation

    Returns
    -------
    float
        The approximated inflection frame; this value is not rounded
        to the nearest integer, so it may be a time (in frames) which
        actually falls between two frames
    """

    low = [-1, -1, -1, -1, -1]
    high = [np.inf, np.inf, np.inf, np.inf, 500]

    x = x_input[:xbuffer]
    y = y_input[:xbuffer]

    # Rescale data by mean-substracting and dividing by standard deviation
    x_scaled = (x - x.mean()) / x.std()
    y_scaled = (y - y.mean()) / y.std()

    # Get cumulative frame-by-frame difference squared and normalize
    length = [0] + (np.diff(x_scaled) ** 2 + np.diff(y_scaled) ** 2).tolist()
    length = np.cumsum(length)
    length /= length[-1]

    # Interpolate values
    fx = spinterp.interpolate.interp1d(length, x_scaled)
    fy = spinterp.interpolate.interp1d(length, y_scaled)
    length_int = np.linspace(0, 1, 64)
    xinterp = fx(length_int)
    yinterp = fy(length_int)

    # Find best fit for inflection point
    try:
        # With the explicit Jacobian and increased maximum iterations, the curve fitting
        # is both performant and can find a good fit for every track.
        popt, pcov = spopt.curve_fit(
            hyperbola, xinterp, yinterp, bounds=(low, high), jac=hyperbola_jacobian, max_nfev=2000
        )
        a1, a2, b1, b2, alpha = popt[:]
        yfit = hyperbola(xinterp, a1, a2, b1, b2, alpha)
        if a1 != a2:
            xcross = (b2 - b1) / (a1 - a2)
            dist = (xinterp - xcross) ** 2 + (yfit - line(xcross, a1, b1)) ** 2
            xc = xinterp[dist.argmin()]
            frame = (xc * x.std()) + x.mean()
        else:
            return None

    except Exception as ex:
        print(ex)
        return None

    if display:
        if isinstance(display, bool):
            fig, ax = plt.subplots(1, 1)
        else:
            ax = display
        ax.plot(xinterp, yinterp, "-o")
        ax.plot(xinterp, line(xinterp, a1, b1), "--", color="red")
        ax.plot(xinterp, line(xinterp, a2, b2), "--", color="red")
        ax.plot(xinterp, yfit, "--", color="k", lw=2)
        ax.axvline(x=xc, color="black")
        ax.set_xlim(x_scaled.min(), x_scaled.max())
        ax.set_ylim(y_scaled.min(), y_scaled.max())
        ax.set_aspect("equal")
        if isinstance(display, bool):
            plt.show()
            if figdir is not None:
                plt.savefig(f"{figdir}/{track}.png")

    if example_track_figure:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6.5))
        ax.plot(xinterp, yinterp, "-o", alpha=0.75, label=f"Linear interpolation of track {track}")
        ax.plot(xinterp, yfit, "--", color="k", lw=1.5, label="Hyperbola fit")
        ax.plot(
            xinterp, line(xinterp, a1, b1), "--", lw=1, color="red", label="Hyperbola asymptotes"
        )
        ax.plot(xinterp, line(xinterp, a2, b2), "--", lw=1, color="red")
        ax.scatter(xcross, line(xcross, a1, b1), color="red", zorder=100, label="Hyperbola center")
        ax.plot(
            [xcross, xc],
            [line(xcross, a1, b1), yfit[dist.argmin()]],
            color="grey",
            lw=2,
            alpha=0.75,
            linestyle="-",
            label="Distance from hyperbola center to\nclosest point on the hyperbola that\nshares x values with interpolated data",
        )
        ax.axvline(
            xc,
            color="black",
            alpha=0.75,
            label=f"Calculated transition = {xc:.2f}\nRescaled calculated transition = {frame*5:.2f} min",
        )

        increase_lim = 0.25
        ax.set_xlim(x_scaled.min() - increase_lim, x_scaled.max() + increase_lim)
        ax.set_ylim(y_scaled.min() - increase_lim, y_scaled.max() + increase_lim)
        ax.set_xlabel("Z-score Normalized Time")
        ax.set_ylabel("Z-score Normalized Volume")
        ax.set_aspect("equal")
        plt.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", labelspacing=1.2, frameon=False)
        plt.tight_layout()
        return fig, ax

    return frame


def add_normalized_time_for_full_tracks(
    df,
    start_col="Ff",
    stop_col="Fb",
):
    """
    Adds normalized time to a dataframe containing full tracks only.
    With an option to filter tracks by length.

    Parameters
    ----------
    df: Pandas dataframe
        NucMorph dataframe as generated by the function
        filter_data.get_dataframe_of_full_tracks
    start_col: str
        Column indicating time value to start at. Is set to
        lamin formation frame by default
    stop_col: str
        Column indicating time value to stop at. Is set to
        lamin breakdown frame by default

    Returns
    -------
    df: Dataframe
        Dataframe with normalized time

    """
    # Compute normalize time and keep tracks longer than 40 tps
    for _, df_track in df.groupby("track_id"):
        f, b = df_track[[start_col, stop_col]].min().values
        df.loc[df_track.index, "normalized_time"] = (df_track.index_sequence - f) / (b - f)
    df.loc[~df.normalized_time.between(0.0, 1.0), "normalized_time"] = np.nan
    ntracks = df["track_id"].nunique()
    print(f"{ntracks} full tracks before track-level filtering")
    df = df.dropna(subset=["normalized_time"]).copy()
    ntracks = df["track_id"].nunique()
    print(f"{ntracks} full tracks in dataset after adding normalized time")
    return df


def add_synchronized_time(df, synch_column="index_sequence"):
    """
    Get synchronized time in units of frames.

    Parameters
    ----------
    df : DataFrame
        The dataframe
    synch_column : str
        Frame to synchronize the time to
        "index_sequence" will sync to the first frame
        or "Ff", "frame_transition", or "Fb" can be specified

    Returns
    -------
    df: DataFrame
        The sync_time in units of frames at specified frame in new column.
    """
    for _, df_track in df.groupby("track_id"):
        sync_time = df_track.index_sequence - df_track[synch_column].min()
        df.loc[df_track.index, f"sync_time_{synch_column}"] = sync_time
    return df


time_bin_edges = np.arange(-0.04, 1.04, 0.04)


def add_binned_normalized_time_for_full_tracks(df):
    """
    Adds binned normalized time to a dataframe containing full tracks only.
    bins are np.arange(-0.04, 1.04, 0.04)
    """
    # Compute normalize time and keep tracks longer than 40 tps
    for _, df_track in df.groupby("track_id"):
        f, b = df_track[["Ff", "Fb"]].min().values
        df_track["digitized_normalized_time"] = pd.cut(
            df_track["normalized_time"], bins=time_bin_edges
        )
        df_track["digitized_normalized_time"] = df_track["digitized_normalized_time"].astype(
            "category"
        )
        cat_columns = df_track.select_dtypes(["category"]).columns
        df_track[cat_columns] = df_track[cat_columns].apply(lambda x: x.cat.codes)
        df_track["digitized_normalized_time"] = df_track["digitized_normalized_time"].astype(int)
        df_track[["digitized_normalized_time"]] = df_track[["digitized_normalized_time"]].apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        df.loc[df_track.index, "normalized_time"] = (df_track.index_sequence - f) / (b - f)
        df.loc[df_track.index, "digitized_normalized_time"] = df_track.digitized_normalized_time
    df.loc[~df.normalized_time.between(0.0, 1.0), "normalized_time"] = np.nan
    df = df.dropna(subset=["normalized_time"]).copy()

    return df


def get_nearest_frame(df_track, calculated_transition):
    """
    Finds the frame closest to the calculated transition point.
    If the closest frame is more than 2 frames (10 minutes) away it returns np.nan

    Parameters
    ----------
    df_track: DataFrame
        DataFrame containing acquisitions from a track

    frame: float
        In units of frames, indicates a point in time in the track

    Returns:
    --------
    frame_transition: Int
        Frame closest to the calculated transition point
    """
    # Not all frames are guaranteed, so get the frame closest to key_point
    distance_from_key_point = df_track.index_sequence - calculated_transition
    closest_frame = df_track.iloc[np.abs(distance_from_key_point).argmin()]
    frame_transition = closest_frame.index_sequence
    if calculated_transition - frame_transition > 2:
        frame_transition = np.nan
    return frame_transition


def _get_frame_inflection(track, df_track, pix_size, frame_formation_col, display, debug_dir):
    """
    Helper function for add_transition_point.

    Parameters
    ----------
    See add_transition_point

    Returns
    -------
    track: int
        Same as input track, for downstream convenience
    frame_transition: int
        Estimated frame of inflection/transition
    """
    df_track = df_track.sort_values("index_sequence")
    xo = df_track[frame_formation_col].values[0]
    if isinstance(xo, str):
        xo = eval(xo)[0]
    if not np.isscalar(xo):
        xo = xo[0]
    x = df_track.index_sequence.values - xo
    y = df_track.volume.values * (pix_size**3)
    inf_pt_rel = get_trajectory_inflection_frame(x, y, track, display=display, figdir=debug_dir)

    if inf_pt_rel is None:
        frame_transition = np.nan
    else:
        calculated_transition = xo + inf_pt_rel
        frame_transition = get_nearest_frame(df_track, calculated_transition)
        if frame_transition == df_track.Ff.values[0]:
            frame_transition = np.nan
    return (track, frame_transition)


def merge_preserve_left_index(left, right, **kwargs):
    """
    Normally pandas.merge returns a dataframe with a new index. This version
    keeps the index of the left dataframe.
    This function probably belongs in a different file.

    Parameters
    ----------
    Identical to pandas.merge

    Return
    ------
    merged_with_new_index: pandas.DataFrame
        Identical to pandas.merge, but with the index matching the left dataframe
    """
    if left.index.name in right.columns:
        return ValueError(f"Index name {left.index.name} is already a column in right dataframe")
    if left.index.name in left.columns:
        return ValueError(f"Index name {left.index.name} is already a column in left dataframe")
    preserved_left = left.reset_index()
    merged_with_new_index = pd.merge(preserved_left, right, **kwargs)
    return merged_with_new_index.set_index(left.index.name)


def add_transition_point(
    df,
    pix_size,
    frame_formation_col="Ff",
    frame_breakdown_col="Fb",
    figdir=None,
    flag_dropna=True,
    display=False,
):
    """
    This is the master function for this script. It cycles through
    the individual nuclear tracks, finds the transition point for
    each one and adds a column with all the transition points
    to the dataset.

    Parameters
    ----------
    df: Dataframe
        Initial dataset
    pix_size: float
        size of pixels in microns for this dataset
    frame_formation_col: string
        Name of column with formation frames
    frame_breakdown_col: string
        Name of column with breakdown frames
    figdir: Path
        Path to directory to store figures
    flag_drop_na: bool
        Flag to drop tracks with no transition points

    Returns
    -------
    df: Dataframe
        Returns the initial dataset with a column added giving the
        transition point for each nuclear volume track, as a frame
        number.
    """
    if "frame_transition" in df.columns:
        raise ValueError("frame_transition column already exists in dataframe")

    if figdir is not None:
        trajdir = create_traj_fig_filestructure(figdir)
        inf_fig_dir = trajdir / "transition"
        debug_dir = inf_fig_dir / "debug"
    else:
        debug_dir = None

    # Get frame inflection for each track in a separate process.
    # Takes ~20s for the 3 main colonies with 8 cores.
    extra_args = (pix_size, frame_formation_col, display, debug_dir)
    with Pool() as pool:
        args = [(track, df_track, *extra_args) for track, df_track in df.groupby("track_id")]
        data = pool.starmap(_get_frame_inflection, args)
    df_frame_transition = pd.DataFrame(data, columns=["track_id", "frame_transition"])
    df = merge_preserve_left_index(df, df_frame_transition, on="track_id", how="left")

    if flag_dropna:
        df = df.dropna(subset="frame_transition")

    if figdir is not None:
        plot_tracks_with_tps(
            df,
            pix_size,
            frame_formation_col,
            "frame_transition",
            frame_breakdown_col,
            inf_fig_dir / "tracks",
        )

    ntracks = df["track_id"].nunique()
    print(f"{ntracks} full tracks in dataset after transition point added")

    return df


def determine_bin_centers(minval,maxval,number_of_bins=None,step_size=None):
    """
    determine the bin centers for digitizing time
    Note: user must specify either number_of_bins or step_size

    Parameters
    ----------
    minval : float
        minimum value of time
    maxval : float
        maximum value of time
    number_of_bins : int
        number of bins to use
    step_size : float
        step size to use

    Returns
    -------
    np.array
        bin centers
    """
    if number_of_bins is not None:
        bin_centers = np.linspace(minval,maxval,number_of_bins)
    elif step_size is not None:
        bin_centers = np.arange(minval,maxval+step_size,step_size)
    else:
        raise ValueError('Must specify either number_of_bins or step_size')
    return bin_centers

def digitize_time_array(time_array,bin_centers):
    """
    convert an array of time values to a digitized array of time values with a specified number of bins
    (or step size)
    Note: user must specify either number_of_bins or step_size

    Parameters
    ----------
    time_array : np.array
        array of time values
    number_of_bins : int
        number of bins to use
    step_size : float
        step size to use

    Returns
    -------
    np.array
        digitized time array
    """
    step_size = bin_centers[1] - bin_centers[0]
    bin_edges = np.append(bin_centers-step_size/2,bin_centers[-1]+step_size/2)
    dig_time_idxs = np.digitize(time_array,bin_edges,right=False)
    dig_time = bin_centers[dig_time_idxs-1]
    return dig_time

def digitize_time_column(df, minval, maxval, number_of_bins=None, step_size=None, time_col = 'normalized_time', new_col='dig_time'):
    """
    create a new column in the dataframe that digitizes input time column
    Note: user must specify either number_of_bins or step_size

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to use
    minval : float
        minimum value of time
    maxval : float
        maximum value of time
    number_of_bins : int
        number of bins to use
    step_size : float
        step size to use

    Returns
    -------
    pd.DataFrame
        dataframe with new column
    """

    time_array = df[time_col].values
    bin_centers = determine_bin_centers(minval,maxval,number_of_bins=number_of_bins,step_size=step_size)
    dig_time_array = digitize_time_array(time_array,bin_centers)
    df[new_col] = dig_time_array
    return df

def validate_dig_time_with_plot(time_array = np.linspace(0,1,1000), number_of_bins=6, old_method=False):
    """
    this visualizes how the input array is binned by plotting
    the input array (x-axis) vs the digitized array (y-axis)
    """

    if old_method:
        time_array = np.linspace(0,1,1000)
        TIME_BIN = 1/number_of_bins
        df_agg = pd.DataFrame({'normalized_time':time_array})
        timedig_bins = np.arange(0, 1 + TIME_BIN, TIME_BIN)
        inds = np.digitize(df_agg["normalized_time"], timedig_bins)
        df_agg["dig_time"] = timedig_bins[inds - 1]
        dig_time = df_agg['dig_time'].values
        extrastr = '\n(old method)'
    else:
        bin_centers = determine_bin_centers(0,1,number_of_bins=number_of_bins)
        dig_time = digitize_time_array(time_array,bin_centers)
        extrastr = ''
    fig, ax = plt.subplots(figsize=(3,3))
    plt.plot(time_array,dig_time)
    plt.xlabel('time')
    plt.ylabel('dig_time')
    plt.title(f'Validation of digitized time\n{number_of_bins} bins{extrastr}')
    return fig, ax