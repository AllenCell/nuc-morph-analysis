import pandas as pd
import numpy as np
from scipy.signal import medfilt
from scipy.spatial import distance_matrix
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.analyses.inhibitors import dataset_info

DEATH_THRESHOLD = 25


def identify_matched_tracks(dff, col="volume_init", close_thresh_um=50):
    scale, label, unit, _ = get_plot_labels_for_metric("volume")
    close_thresh = close_thresh_um / scale
    dff1 = dff[dff["condition"] == "control"].groupby("track_id").agg("first")
    dff2 = dff[dff["condition"] == "perturb"].groupby("track_id").agg("first")

    dff1v = dff1[[col]].dropna()
    dff2v = dff2[[col]].dropna()

    # retrieve volume at drug added frame
    volumes1 = dff1v[col].values
    volumes2 = dff2v[col].values  # perturb volumes

    # compute the pairwise distances between perturb volumes and control volumes
    dmat = distance_matrix(
        volumes2.reshape(-1, 1),
        volumes1.reshape(
            -1,
            1,
        ),
    )
    # determine the index of the minimum distances by sorting along axis=1. sdmat[pindex,0] = cindex (cindex of closest match to a given pindex)
    sdmat = np.argsort(dmat, axis=1)

    check_against, ckeep, pkeep = [], [], []
    for pi, ci in enumerate(sdmat[:, 0]):
        close = (
            np.abs(volumes1[ci] - volumes2[pi]) < close_thresh
        )  # only keep a match if it is within the determined threshold
        if (ci not in check_against) & (
            close
        ):  # if ci and pi combo has not been chosen yet and is close enough, then keep
            ckeep.append(ci)
            pkeep.append(pi)

        check_against.append(ci)

    dfpvout = dff2v.iloc[np.asarray(pkeep)]
    dfcvout = dff1v.iloc[np.asarray(ckeep)]
    p_ids = list(set(dfpvout.reset_index()["track_id"].tolist()))
    c_ids = list(set(dfcvout.reset_index()["track_id"].tolist()))

    # tlist = p_ids+c_ids
    # print(len(p_ids),len(c_ids))
    return p_ids, c_ids


def filter_to_long_enough_tracks(dfc):
    dfc1 = dfc.loc[dfc["time_minutes_since_drug"] <= -60, :]
    dfc2 = dfc.loc[dfc["time_minutes_since_drug"] >= 120, :]
    # now find the track_ids that are in both dfc1 and dfc2
    long_enough_track_ids = list(
        set(dfc1.track_id.unique()).intersection(set(dfc2.track_id.unique()))
    )
    # now filter the dataframe to only include the long enough track_ids
    dff = dfc.loc[dfc["track_id"].isin(long_enough_track_ids), :]
    return dff


def filter_at_death_threshold(df, dataset):
    # filter out tracks that have more than X% of their frames as dead cells
    xi = dataset["num_dying_x"]
    yi = dataset["num_dying_y"]
    xint = np.arange(xi[0], xi[-1] + 1)
    yint = np.interp(xint, xi, yi)
    x_idx = np.where(yint > DEATH_THRESHOLD)[0]
    if np.sum(yint > DEATH_THRESHOLD) == 0:
        xthresh = xint[-1]
    else:
        xthresh = xint[x_idx[0]]

    df = df[df["index_sequence"] < xthresh]
    print(f'removing all data after {xthresh} (drug added at {dataset["drug_added_frame"]})')
    return df


def compute_and_normalize_by_initial_volumes(
    df,
    drug_added_frame,
    frame_width=3,
    column_list=["volume", "volume_um", "mesh_sa", "SA_vol_ratio"],
):

    ################################################################################
    dfin = df.copy()

    # identify the a set of frames before drug addition to normalize to
    # these will be defined as the "initial volume" for each track,
    # meaning the volume immediately preceeding inhibitor addition

    frame_before_drug = drug_added_frame - 1
    pdslice = pd.IndexSlice[:, range(int(frame_before_drug - frame_width), int(frame_before_drug))]

    df.set_index(["track_id", "index_sequence"], inplace=True)
    grouper = df.loc[pdslice, column_list].groupby(["track_id"])

    # require all tracks to values at all frames in the window
    count = grouper.count()
    count = count[count[column_list[0]] == frame_width]
    dfsub = df.loc[pd.IndexSlice[count.index, :], :]

    df_daf = (
        dfsub.loc[pdslice, column_list].groupby(["track_id"]).agg("mean")
    )  # compute the mean volume for a track in the given window

    # after merge volume_init will now have the value for avg. volume for a range of frames around just before the drug added frame
    df = pd.merge(
        df.reset_index(),
        df_daf[column_list],  # old = dflsub
        on=["track_id"],
        suffixes=("", "_init"),
        how="outer",
    )

    # now that all track_ids have a "initial volume-1" (volume at t=16) to compare to, we can compute normalized trajectories
    for col in column_list:
        df[f"{col}_norm"] = df[col] / df[f"{col}_init"]
        df[f"{col}_sub"] = df[col] - df[f"{col}_init"]

    df.reset_index(inplace=True)
    return df


def acquire_dividing_cells_before_and_after_drug_addition(df, details, pairs, chosen_condition):
    dfsub = df[df.predicted_formation >= 0]  # require tracks to have formation frame
    # create a time since frame formation column

    # require tracks to have at least 45 minutes of data
    dfsubp = dfsub.pivot(index="track_id", columns="index_sequence", values="volume")
    long_enough_tracks = dfsubp.loc[
        dfsubp.count(axis=1) >= 45 / details["time_interval_minutes"]
    ].index
    dfsub = dfsub[dfsub.track_id.isin(long_enough_tracks)]

    # require tracks to have a starting volume less than start_thresh
    start_thresh = 450 / details["pixel_size"] ** 3
    dfg = dfsub.groupby("track_id").first()
    dfsub = dfsub[dfsub.track_id.isin(dfg[dfg.volume < start_thresh].index)]

    # now require that the tracks BEFORE drug addition have a predicted formation 90 minutes before drug addition
    time_before_thresh = 90 / details["time_interval_minutes"]
    before_log = dfsub.predicted_formation < (details["drug_added_frame"] - time_before_thresh)
    dfsub1 = dfsub[before_log]

    # now require that the tracks AFTER drug addition have a predicted formation 5 minutes after before drug addition AND before 60 minutes after drug addition
    time_after_thresh = 60 / details["time_interval_minutes"]
    after_log = dfsub.predicted_formation > (details["drug_added_frame"] + 1)
    before_log = dfsub.predicted_formation < (details["drug_added_frame"] + time_after_thresh)
    dfsub2 = dfsub[after_log & before_log]

    # now require all BEFORE drug tracks to end before drug addition
    dfsub1 = dfsub1[dfsub1["time_minutes_since_drug"] <= 0]

    # now require all tracks to end at least 2 hours after drug addition
    dfsub2 = dfsub2[dfsub2["time_minutes_since_drug"] <= 120]

    # exclude manually identified tracks
    exclude_dict = dataset_info.EXCLUDE_tracks_expansion[pairs]
    exclude_list = exclude_dict[chosen_condition]
    # drop track_ids in this list
    dfsub1 = dfsub1[~dfsub1.track_id.isin(exclude_list)]
    dfsub2 = dfsub2[~dfsub2.track_id.isin(exclude_list)]
    return dfsub1, dfsub2
