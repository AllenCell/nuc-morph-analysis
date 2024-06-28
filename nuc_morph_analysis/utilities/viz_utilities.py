import matplotlib.pyplot as plt
import numpy as np


def plot_tracks_with_tps(df, pix_size, ffc, fic, fbc, figdir=None):
    """
    Cycle through all tracks in dataset and plot volume track,
    mark frames of formation, inflection and breakdown in red,
    green and blue respectively and show or save resulting images

    Parameters
    ----------
    df: DataFrame
        Colony dataset
    pix_size: float
        Size of pixels in microns
    ffc: str
        Column name for frame of formation
    fic: str
        Column name for frame of inflexion
    fbc: str
        Column name for frame of breakdown
    figdir: Path
        Path to where figures for this dataset are saved
    """

    if figdir is not None:
        traj_dir = figdir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        inf_dir = traj_dir / "inflection"
        inf_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving inflection figs in {inf_dir}")

    count = 0
    count2 = 0
    for track, df_track in df.groupby("track_id"):
        df_track = df_track.sort_values("index_sequence")
        x = df_track.index_sequence.values
        y = df_track.volume.values * (pix_size**3)
        plt.clf()
        plt.plot(x, y, color="k")
        for col, color in zip([ffc, fic, fbc], ["r", "g", "b"]):
            frame = df_track[col].values[0]
            if isinstance(frame, list):
                frame = frame[0]
            if col == "frame_formation" and frame == df_track["index_sequence"].values[0]:
                count += 1
            elif col == "frame_formation" and frame < df_track["index_sequence"].values[0]:
                count2 += 1
            if len(y[np.where(x == round(frame))]) == 0:
                closest_ind = np.abs(x - frame).argmin()
                if x[closest_ind] - frame < 2:
                    yframe = y[closest_ind]
                else:
                    yframe = 0
            else:
                yframe = y[np.where(x == round(frame))][0]
            plt.scatter(frame, yframe, color=color)
        plt.title(track)
        plt.xlabel("Frames")
        plt.ylabel(r"Volume ($\mu m^3$)")
        plt.show()

        if figdir is not None:
            plt.savefig(f"{inf_dir}/{track}.png")

    print(f"Number tracks with ff=first: {count}")
    print(f"Number tracks with ff<first: {count2}")
